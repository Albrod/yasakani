# SPDX-License-Identifier: Apache-2.0
"""0.3.4: the fast path of Inkling-2 Mini on by default on RTX 30, 40 and 50 cards (inkling2.py FAST_AUTO_CAPS, fast_expected,
fast_choice, Ink2Engine; inkling2_cli.py open_stack, bench --json; report.py, the margin of the copy cycles).  No GPU and no model:
the engine library, nvidia-smi and the model folder are stand-ins.  The planning of the context (fast_expected) and the path the
engine runs (Ink2Engine) must agree for every compute capability and mode."""
import argparse
import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from yasakani import gpu as G
from yasakani import inkling2 as ink
from yasakani import inkling2_cli as C
from yasakani import report
from yasakani import widecheck as W
from yasakani._native import YasakaniError

from tests.test_inkling2_context import CONFIG, DRAFTER, WEIGHTS

CAPS = ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0), (12, 1), None, ink.CAP_UNKNOWN)
MODES = ("auto", "on", "off")


# ---- stand-ins of the engine library ----------------------------------------------------------------------------------------------
class FakeNative:
    def __init__(self):
        self.options, self.vocab = {}, 16

    def load(self, root, backend=None, max_positions=0, dll=None, weights=None, log=None, rollback=0, max_rows=0, chunk=0,
             kv_mxfp8=0, kv_hot=0):
        self.kw = dict(max_rows=max_rows, chunk=chunk, kv_mxfp8=kv_mxfp8, kv_hot=kv_hot)
        return self

    def set_option(self, name, value):
        self.options[name] = value

    def close(self):
        pass


def fake_lib(nvfp4=False, has_fast=True, cuda=True):
    lib = SimpleNamespace(caps=16 if has_fast else 0, ink2_build_info=lambda: b"inkling2 cuda" if cuda else b"inkling2 host")
    if has_fast:
        lib.ink2_fast_path = object()
    mod = SimpleNamespace(Ink2Native=FakeNative, nvfp4_prompt=lambda _lib: 1 if nvfp4 else 0)
    return lib, mod


def engine(mode, cap=None, nvfp4=False, has_fast=True, cuda=True, detect=None):
    """An Ink2Engine over the stand-in library; ``detect``: what nvidia-smi gives when the engine asks (or an exception)."""
    lib, mod = fake_lib(nvfp4, has_fast, cuda)
    pk = SimpleNamespace(root=Path("unused"), weights_format="hf-safetensors", weights=SimpleNamespace(name="model.safetensors"))
    asked = []

    def det(device=0, environ=None, *a, **k):
        asked.append(device)
        if isinstance(detect, Exception):
            raise detect
        return detect if detect is not None else G.Detected()

    with mock.patch.object(ink, "read_config", lambda root: CONFIG), mock.patch.object(ink, "find_dll", lambda d: Path("x.dll")), \
            mock.patch.object(ink, "find_binding", lambda d: Path("binding.py")), mock.patch.object(ink, "load_binding", lambda p: mod), \
            mock.patch.object(ink, "open_library", lambda m, d: lib), mock.patch.object(G, "detect", det), \
            mock.patch.dict(os.environ, {"YASAKANI_INK2_BACKEND": "", "YASAKANI_INK2_ROLLBACK": ""}):
        eng = ink.Ink2Engine(pk, 4096, fast=mode, compute_cap=cap)
    eng.asked = asked
    return eng


def one_gpu(cc, total=24564, used=1100, name="NVIDIA GeForce RTX 3090 Ti"):
    g = G.GpuInfo(index=0, name=name, uuid="GPU-a", compute_cap=cc, total_mib=total, used_mib=used)
    return G.Detected(gpus=[g], candidates=[g], device=0, numbered=True)


# ---- the rule ---------------------------------------------------------------------------------------------------------------------
class RuleTests(unittest.TestCase):
    def test_the_caps(self):
        self.assertEqual(ink.FAST_AUTO_CAPS, ((8, 6), (8, 9), (12, 0)))
        for cc in CAPS:
            self.assertEqual(ink.fast_expected("auto", cc), cc in ((8, 6), (8, 9), (12, 0)), cc)
            self.assertTrue(ink.fast_expected("on", cc), cc)
            self.assertFalse(ink.fast_expected("off", cc), cc)
        self.assertTrue(ink.fast_expected("auto", [8, 6]))                         # a list from JSON counts as well

    def test_engine_compute_cap(self):
        with mock.patch.object(G, "detect", lambda *a, **k: one_gpu((8, 6))):
            self.assertEqual(ink.engine_compute_cap(), (8, 6))
        with mock.patch.object(G, "detect", lambda *a, **k: G.Detected()):
            self.assertIsNone(ink.engine_compute_cap())
        with mock.patch.object(G, "detect", mock.Mock(side_effect=OSError("no nvidia-smi"))):
            self.assertIsNone(ink.engine_compute_cap())


class EngineTests(unittest.TestCase):
    """Ink2Engine with the compute capability of the caller, and without it (then it asks nvidia-smi itself)."""

    def test_planning_and_engine_agree(self):
        for cc in CAPS:
            for mode in MODES:
                nvfp4 = cc == (12, 0)                                               # where the real library reports NVFP4
                eng = engine(mode, cap=cc if cc is not None else ink.CAP_UNKNOWN, nvfp4=nvfp4)
                self.assertEqual(eng.fast, ink.fast_expected(mode, cc), (cc, mode))
                self.assertEqual(eng.fast_nvfp4, eng.fast and nvfp4, (cc, mode))
                self.assertEqual(eng.asked, [], (cc, mode))                         # given: nvidia-smi is not asked again
                self.assertEqual(eng.native.kw["kv_mxfp8"], 1 if eng.fast else 0, (cc, mode))
                self.assertEqual("nvfp4_prefill" in eng.native.options, eng.fast_nvfp4, (cc, mode))
                self.assertLessEqual(len(eng.fast_note), 140, eng.fast_note)

    def test_api_without_a_cap_resolves_as_the_command_line(self):
        for cc in ((7, 5), (8, 6), (8, 9), (9, 0), (12, 0), None):
            for mode in MODES:
                nvfp4 = cc == (12, 0)
                api = engine(mode, nvfp4=nvfp4, detect=one_gpu(cc))
                cmd = engine(mode, cap=cc if cc is not None else ink.CAP_UNKNOWN, nvfp4=nvfp4)
                self.assertEqual((api.fast, api.fast_nvfp4, api.fast_note), (cmd.fast, cmd.fast_nvfp4, cmd.fast_note), (cc, mode))
                self.assertEqual(api.asked, [0] if mode == "auto" and not nvfp4 else [], (cc, mode))   # device 0 of what CUDA sees

    def test_nvfp4_is_on_without_asking(self):
        eng = engine("auto", nvfp4=True, detect=OSError("must not be asked"))
        self.assertTrue(eng.fast and eng.fast_nvfp4)
        self.assertEqual(eng.asked, [])
        self.assertEqual(eng.fast_note, "fast path: on (8-bit key/value cache, NVFP4 prompt products; --fast off for the standard path)")

    def test_nvidia_smi_fails(self):
        eng = engine("auto", detect=RuntimeError("broken"))
        self.assertFalse(eng.fast)
        self.assertIn("compute capability is unknown", eng.fast_note)

    def test_old_library_and_cpu_build(self):
        eng = engine("auto", cap=(8, 6), has_fast=False)
        self.assertFalse(eng.fast)
        self.assertEqual(eng.fast_note, "fast path: off (this engine library has no fast path; the library of 0.3.3 or later has it)")
        with self.assertRaises(YasakaniError) as cm:
            engine("on", cap=(8, 6), has_fast=False)
        self.assertIn("0.3.3 or later (x.dll)", str(cm.exception))
        for mode in MODES:
            eng = engine(mode, cap=(12, 0), nvfp4=True, cuda=False)
            self.assertFalse(eng.fast or eng.fast_nvfp4, mode)
            self.assertTrue(eng.fast_note.startswith("fast path: off"), eng.fast_note)
            self.assertEqual(eng.asked, [])

    def test_start_up_lines(self):
        want = {
            ("auto", (8, 6)): "fast path: on (8-bit key/value cache; this GPU has no NVFP4 prompt products; --fast off for the standard path)",
            ("auto", (8, 9)): "fast path: on (8-bit key/value cache; this GPU has no NVFP4 prompt products; --fast off for the standard path)",
            ("on", (7, 5)): "fast path: on (8-bit key/value cache; this GPU has no NVFP4 prompt products; --fast off for the standard path)",
            ("auto", (7, 5)): "fast path: off (auto turns it on for RTX 30, 40 and 50 cards; --fast on turns it on here)",
            ("auto", (9, 0)): "fast path: off (auto turns it on for RTX 30, 40 and 50 cards; --fast on turns it on here)",
            ("auto", ink.CAP_UNKNOWN): "fast path: off (the GPU's compute capability is unknown; auto turns it on for RTX 30, 40 and 50 "
                                       "cards; --fast on turns it on)",
            ("off", (8, 6)): "fast path: off (--fast off or YASAKANI_INK2_FAST=off; auto turns it on for RTX 30, 40 and 50 cards)",
        }
        for (mode, cc), line in want.items():
            self.assertEqual(engine(mode, cap=cc).fast_note, line, (mode, cc))
            self.assertLessEqual(len(line), 140)
        for nv, cc in ((True, (12, 0)), (False, (8, 6)), (False, (7, 5)), (False, None)):
            for mode in MODES:
                for has_fast in (True, False):
                    for backend in ("cuda", "host"):
                        try:
                            _, _, note = ink.fast_choice(mode, cc, nv, has_fast, backend)
                        except YasakaniError:
                            self.assertEqual((mode, has_fast), ("on", False))
                            continue
                        self.assertLessEqual(len(note), 140, note)
                        self.assertTrue("--fast" in note or "library" in note or "backend" in note, note)


# ---- the command line ------------------------------------------------------------------------------------------------------------
class CommandLineTests(unittest.TestCase):
    def open_stack(self, cc, mode=None, max_ctx=None):
        """open_stack up to the engine: (the compute capability given to open_runner, the cache line, the context)."""
        seen, lines = {}, []
        pk = SimpleNamespace(default_ctx=32768, max_ctx=1048576, root=Path("unused"), model_id="m", weights_format="yskn",
                             weights=SimpleNamespace(stat=lambda: SimpleNamespace(st_size=WEIGHTS)),
                             drafter=SimpleNamespace(stat=lambda: SimpleNamespace(st_size=DRAFTER)))

        def runner(*a, **kw):
            seen.update(kw)
            raise KeyboardInterrupt

        args = argparse.Namespace(model_dir_path=Path("unused"), max_ctx=max_ctx, device=0, draft=None, draft_depth=None, fast=mode)
        with mock.patch.object(ink.Ink2Pack, "load", lambda p: pk), mock.patch.object(ink, "read_config", lambda root: CONFIG), \
                mock.patch.object(G, "detect", lambda *a, **k: one_gpu(cc)), mock.patch.object(ink, "open_runner", runner), \
                mock.patch.object(C, "_say", lines.append), mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": "", "YASAKANI_DRAFT": "",
                                                                                          "YASAKANI_INK2_ROLLBACK": ""}):
            with self.assertRaises(KeyboardInterrupt):
                C.open_stack(args)
        return seen, lines

    def test_the_engine_gets_the_planned_cap(self):
        for cc in ((7, 5), (8, 6), (8, 9), (9, 0), (12, 0), None):
            for mode in MODES:
                for max_ctx in (None, 131072):
                    seen, lines = self.open_stack(cc, mode, max_ctx)
                    got = seen["compute_cap"]
                    self.assertEqual(got, (cc if cc is not None else ink.CAP_UNKNOWN) if mode == "auto" or max_ctx is None
                                     else ink.CAP_UNKNOWN, (cc, mode, max_ctx))
                    fast = ink.fast_expected(mode, cc)
                    kv = ink.bytes_per_position(CONFIG, mxfp8=fast) * (max_ctx or 32768)
                    self.assertIn("key/value cache %.2f GB" % (kv / 1e9), lines[-1], (cc, mode, max_ctx))

    def test_bench_settings(self):
        eng = SimpleNamespace(fast_mode="auto", fast=True, fast_nvfp4=False, compute_cap=(8, 6), copying=True, copy_mode="auto",
                              copy_min=12)
        self.assertEqual(C.bench_settings(eng), dict(fast=dict(mode="auto", on=True, nvfp4_prompt=False, compute_cap="8.6"),
                                                     copy=dict(mode="auto", on=True, min_match=12)))
        eng = SimpleNamespace(fast_mode="off", fast=False, fast_nvfp4=False, compute_cap=ink.CAP_UNKNOWN, copying=False,
                              copy_mode="off", copy_min=None)
        s = C.bench_settings(eng)
        self.assertEqual(s, dict(fast=dict(mode="off", on=False, nvfp4_prompt=False, compute_cap=None),
                                 copy=dict(mode="off", on=False, min_match=None)))
        json.dumps(s)

    def test_bench_json_carries_them(self):
        eng = engine("auto", cap=(8, 9))
        out = io.StringIO()
        args = argparse.Namespace(n=4, prompts_file=None, json=True, think=False, runs=1)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=4, gen_ms=100.0, prompt_ms=10.0, spec=None)
        stack = (SimpleNamespace(model_id="m"), eng, None, SimpleNamespace(run=lambda p, emit: usage))
        eng.reset = lambda: None
        with mock.patch.object(C, "open_stack", lambda a, warm=True: stack), mock.patch.object(C, "refuse_unsupported", lambda a: None), \
                mock.patch("yasakani.cli._gpu_lines", lambda: ["gpu"]), mock.patch("yasakani.cli._load_prompts", lambda f: [("t", "x")]), \
                mock.patch.object(C, "_say", lambda *a: None), contextlib.redirect_stdout(out):
            self.assertEqual(C.cmd_bench(args), 0)
        rep = json.loads(out.getvalue())
        self.assertEqual(rep["fast"], dict(mode="auto", on=True, nvfp4_prompt=False, compute_cap="8.9"))
        self.assertEqual(rep["copy"], dict(mode="off", on=False, min_match=None))


# ---- a 24 GB card by the package's rule --------------------------------------------------------------------------------------------
class Card24Tests(unittest.TestCase):
    """24,564 MiB with 1,100 MiB in use (23,464 MiB free), the quality folder with its converted drafter, rollback 512.  The issue
    of an RTX 3090 Ti: a prompt of 131,072 tokens was read with the fast path on such a card."""
    FREE = 24564 - 1100

    def per(self, fast):
        return G.ink2_bytes_per_position(ink.bytes_per_position(CONFIG, mxfp8=fast), True, fast=fast)

    def test_numbers(self):
        files = WEIGHTS + DRAFTER
        got = {}
        for fast in (False, True):
            got[fast] = (G.ink2_context(self.FREE, files, self.per(fast), 32768, 512, fast=fast),     # without --max-ctx
                         G.ink2_context(self.FREE, files, self.per(fast), 1 << 20, 512, fast=fast),   # the longest that fits
                         G.ink2_need_mib(131072, files, self.per(fast), 512, fast=fast))              # what --max-ctx 131072 needs
        self.assertEqual(got[False], (32768, 65536, 28682))
        self.assertEqual(got[True], (32768, 200704, 21566))
        self.assertGreater(got[False][2], self.FREE)                                 # standard path: 131,072 does not fit
        self.assertLessEqual(got[True][2], self.FREE)                                # fast path: it fits, as reported

    def test_auto_context_on_an_rtx_30_card(self):
        lines = []
        with mock.patch.object(ink, "read_config", lambda root: CONFIG), mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": "",
                                                                                                     "YASAKANI_INK2_ROLLBACK": ""}):
            for mode, want in (("auto", True), ("off", False)):
                args = argparse.Namespace(max_ctx=None, device=0, draft=None, draft_depth=None, fast=mode)
                pk = SimpleNamespace(default_ctx=1 << 20, max_ctx=1 << 20, root=Path("unused"),
                                     weights=SimpleNamespace(stat=lambda: SimpleNamespace(st_size=WEIGHTS)),
                                     drafter=SimpleNamespace(stat=lambda: SimpleNamespace(st_size=DRAFTER)))
                n = C.auto_context(args, pk, det=one_gpu((8, 6)), say=lines.append)
                self.assertEqual(n, 200704 if want else 65536, mode)
                self.assertEqual(args.ink2_compute_cap, (8, 6))


# ---- yasakani report: the margin of the card ---------------------------------------------------------------------------------------
class ReportMarginTests(unittest.TestCase):
    def table(self, native):
        timing = W.Timing(rounds=[{16: (1.0, 1.0), 32: (1.5, 1.5), 48: (2.0, 2.0), 64: (2.5, 2.5)}] * 3, sampled={}, redone=False)
        _, widths = W.decide(timing, native=native)
        g = {k: None for k in ("total_mib", "sms", "code", "driver", "driver_model", "power_limit_w", "max_sm_mhz", "max_mem_mhz",
                               "display_attached", "display_active", "pstate", "temperature_c", "sm_mhz", "slowdown_reasons",
                               "free_mib_now")}
        g.update(name="g", cc="12.0" if native else "8.6")
        m = dict(package="yasakani x", os="o", python="3", library=dict(file="yskn.dll", sha256="0" * 16, build_info="b"), gpu=g,
                 model=dict(id="m", weights="w", context=1),
                 wide=dict(setting="on", status="-", mask=0, widths="-", reason="", cache="-", check_seconds=0.0, free_after_mib=None))
        rep = dict(machine=m, check=dict(widths=[W.asdict(w) for w in widths], timing=dict(rounds=[], redone=False)))
        return [ln for ln in report.render(rep).split("\n") if ln.startswith("**Copy cycles**")]

    def test_the_margin_of_the_card(self):
        self.assertIn("limit = 0.9 × rows / 16", self.table(True)[0])
        self.assertIn("limit = 0.8 × rows / 16", self.table(False)[0])
        self.assertEqual(report._margin_text([]), "?")


if __name__ == "__main__":
    unittest.main()
