# SPDX-License-Identifier: Apache-2.0
"""0.3.2: the context of an Inkling-2 folder from the free GPU memory when --max-ctx is not given (yasakani/gpu.py ink2_context,
yasakani/inkling2_cli.py context_of / auto_context).  No GPU and no model: nvidia-smi and the model folder are stand-ins."""
import argparse
import io
import contextlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from yasakani import gpu as G
from yasakani import inkling2 as ink
from yasakani import inkling2_cli as C
from yasakani._native import YasakaniError

WEIGHTS, DRAFTER = 15052423172, 209688040              # the converted quality folder on Hugging Face (its yasakani.json)
CACHE = 65536                                          # ink.bytes_per_position of its config.json
# nvidia-smi's GPU memory taken by the 0.3.1 engine with that folder on the RTX 5090 (MiB, the larger of after the load and after
# one answer to a prompt of 3,923 tokens): (context, drafter, rollback positions) -> MiB
MEASURED = {(8192, True, 512): 17775, (16384, True, 512): 18328, (32768, True, 512): 19701, (8192, False, 512): 17421,
            (32768, False, 512): 19104, (32768, True, 0): 18908}
CONFIG = {"text_config": {"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128,
                          "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3}}}


def card(total, used, name="NVIDIA GeForce RTX 4090"):
    return G.Detected(gpus=[G.GpuInfo(index=0, name=name, uuid="GPU-a", compute_cap=(8, 9), total_mib=total, used_mib=used)],
                      candidates=[G.GpuInfo(index=0, name=name, uuid="GPU-a", compute_cap=(8, 9), total_mib=total, used_mib=used)],
                      device=0, numbered=True)


def size(n):
    return SimpleNamespace(stat=lambda: SimpleNamespace(st_size=n))


def pack(drafter=True, default=32768):
    return SimpleNamespace(default_ctx=default, max_ctx=1048576, weights=size(WEIGHTS), drafter=size(DRAFTER) if drafter else None,
                           root=Path("unused"))


class RuleTests(unittest.TestCase):
    def per(self, drafter=True):
        return G.ink2_bytes_per_position(CACHE, drafter)

    def test_the_cache_of_the_model(self):
        self.assertEqual(ink.bytes_per_position(CONFIG), CACHE)                     # 2.15 GB at 32,768 positions
        self.assertEqual(self.per(False), 77824)
        self.assertEqual(self.per(True), 77824 + 10240)

    def test_never_less_than_measured(self):
        """The rule asks at least for the memory the engine took in the measurement, for every measured case."""
        for (ctx, drafter, rollback), mib in MEASURED.items():
            files = WEIGHTS + (DRAFTER if drafter else 0)
            need = G.ink2_need_mib(ctx, files, self.per(drafter), rollback)
            self.assertGreaterEqual(need - G.MARGIN_MIB, mib, (ctx, drafter, rollback))
            self.assertLess(need - mib, 1500, (ctx, drafter, rollback))             # and not far more
            self.assertEqual(G.ink2_context(need, files, self.per(drafter), ctx, rollback), ctx)
            self.assertLess(G.ink2_context(need - 300, files, self.per(drafter), ctx, rollback), ctx)

    def test_default_shorter_and_none(self):
        f, per = WEIGHTS + DRAFTER, self.per()
        self.assertEqual(G.ink2_context(28914, f, per, 32768, 512), 32768)          # the RTX 5090 of the measurement
        self.assertEqual(G.ink2_context(60000, f, per, 32768, 512), 32768)          # never more than the folder's default
        n = G.ink2_context(19000, f, per, 32768, 512)
        self.assertTrue(G.INK2_MIN_CONTEXT <= n < 32768 and n % G.INK2_CONTEXT_STEP == 0, n)
        self.assertEqual(G.ink2_context(15000, f, per, 32768, 512), 0)              # a 16 GB card
        self.assertEqual(G.ink2_context(0, f, per, 32768, 512), 0)
        small = G.ink2_need_mib(4096, f, per, 512)
        self.assertEqual(G.ink2_context(small, f, per, 4096, 512), 4096)            # a folder with a short default of its own
        lower = [G.ink2_context(free, f, per, 32768, 512) for free in range(17000, 22000, 50)]
        self.assertEqual(lower, sorted(lower))                                      # more free memory never gives less context

    def test_rollback(self):
        self.assertEqual(G.ink2_rollback_mib(0), 0)                                  # the engine's own 64
        self.assertEqual(G.ink2_rollback_mib(64), 0)
        self.assertAlmostEqual(G.ink2_rollback_mib(512), 448 * 1.8)
        self.assertGreater(G.ink2_rollback_mib(1024), G.ink2_rollback_mib(512))
        self.assertEqual(ink.rollback_positions({}), 512)
        self.assertEqual(ink.rollback_positions({"YASAKANI_INK2_ROLLBACK": "0"}), 0)
        self.assertEqual(ink.rollback_positions({"YASAKANI_INK2_ROLLBACK": "1024"}), 1024)
        self.assertEqual(ink.rollback_positions({"YASAKANI_INK2_ROLLBACK": "many"}), 512)   # the load refuses it with a message


class ContextOfTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(ink, "read_config", lambda root: CONFIG)
        p.start()
        self.addCleanup(p.stop)
        e = mock.patch.dict(os.environ, {}, clear=False)
        e.start()
        self.addCleanup(e.stop)
        for k in ("YASAKANI_DRAFT", "YASAKANI_DRAFT_DEPTH", "YASAKANI_INK2_ROLLBACK"):
            os.environ.pop(k, None)
        os.environ["YASAKANI_INK2_FAST"] = "off"     # the standard path's rule (0.3.4: auto = fast on the RTX 4090 of card(); see
        self.lines = []                              # test_inkling2_fast_auto.py for the fast path)

    def auto(self, det, **kw):
        args = argparse.Namespace(max_ctx=None, device=0, draft=kw.pop("draft", None), draft_depth=None)
        return C.auto_context(args, kw.pop("pk", None) or pack(), det=det, say=self.lines.append)

    def test_fits(self):
        self.assertEqual(self.auto(card(32607, 3693, "NVIDIA GeForce RTX 5090")), 32768)
        self.assertEqual(len(self.lines), 1)
        self.assertIn("context: 32768 tokens (the folder's default; 28914 of 32607 MiB of GPU memory free)", self.lines[0])
        self.assertIn("--max-ctx sets it yourself", self.lines[0])

    def test_shorter_on_a_fuller_card(self):
        n = self.auto(card(24564, 5000))                                             # 19,564 MiB free
        self.assertTrue(G.INK2_MIN_CONTEXT <= n < 32768, n)
        self.assertEqual(n, G.ink2_context(19564, WEIGHTS + DRAFTER, G.ink2_bytes_per_position(CACHE, True), 32768, 512))
        self.assertIn("context: %d tokens, chosen from the free GPU memory (19564 of 24564 MiB free; the folder's default 32768 "
                      "does not fit)" % n, self.lines[0])

    def test_drafter_off_leaves_more_room(self):
        on = self.auto(card(24564, 5000))
        off = self.auto(card(24564, 5000), draft="off")
        self.assertGreater(off, on)
        with mock.patch.dict(os.environ, {"YASAKANI_DRAFT": "off"}):
            self.assertEqual(self.auto(card(24564, 5000)), off)
        self.assertEqual(self.auto(card(24564, 5000), pk=pack(drafter=False)), off)  # a folder without a drafter

    def test_rollback_of_the_environment_is_counted(self):
        base = self.auto(card(24564, 5000))
        with mock.patch.dict(os.environ, {"YASAKANI_INK2_ROLLBACK": "0"}):
            self.assertGreater(self.auto(card(24564, 5000)), base)

    def test_refusal_names_max_ctx(self):
        with self.assertRaises(YasakaniError) as cm:
            self.auto(card(16376, 600))
        msg = str(cm.exception)
        self.assertIn("does not fit in the free GPU memory (15776 of 16376 MiB free", msg)
        self.assertIn("--max-ctx", msg)
        self.assertIn("--draft off", msg)
        with self.assertRaises(YasakaniError) as cm:
            self.auto(card(16376, 600), draft="off")
        self.assertNotIn("--draft off", str(cm.exception))

    def test_without_nvidia_smi_the_default(self):
        self.assertEqual(self.auto(G.Detected()), 32768)
        self.assertIn("the free GPU memory could not be read", self.lines[0])
        g = card(24564, 1000)
        g.candidates[0] = G.GpuInfo(index=0, name="x", total_mib=None, used_mib=None)
        self.assertEqual(self.auto(g), 32768)

    def test_max_ctx_wins_and_asks_nothing(self):
        asked = []
        with mock.patch.object(G, "detect", lambda *a, **k: asked.append(a) or card(16376, 600)):
            args = argparse.Namespace(max_ctx=4096, device=0, draft=None, draft_depth=None)
            self.assertEqual(C.context_of(args, pack()), 4096)
            for bad in (100, 2 << 20):
                with self.assertRaises(YasakaniError):
                    C.context_of(argparse.Namespace(max_ctx=bad, device=0, draft=None), pack())
        self.assertEqual(asked, [])

    def test_chosen_once_per_command(self):
        """serve asks before it listens and again when it loads: nvidia-smi is asked and the line printed once."""
        asked = []
        args = argparse.Namespace(max_ctx=None, device=0, draft=None, draft_depth=None)
        out = io.StringIO()
        with mock.patch.object(G, "detect", lambda *a, **k: asked.append(a) or card(32607, 3693)), contextlib.redirect_stderr(out), \
                contextlib.redirect_stdout(out):
            self.assertEqual(C.context_of(args, pack()), 32768)
            self.assertEqual(C.context_of(args, pack()), 32768)
        self.assertEqual(len(asked), 1)
        self.assertEqual(out.getvalue().count("context: 32768 tokens"), 1)

    def test_device_is_passed_to_the_detection(self):
        seen = []
        args = argparse.Namespace(max_ctx=None, device=1, draft=None, draft_depth=None)
        with mock.patch.object(G, "detect", lambda device=0, **k: seen.append(device) or card(32607, 3693)):
            C.auto_context(args, pack(), say=lambda s: None)
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
