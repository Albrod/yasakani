# SPDX-License-Identifier: Apache-2.0
"""0.3.3: the fast path of Inkling-2 Mini (--fast auto | on | off, $YASAKANI_INK2_FAST): the choice, the context planning with the
8-bit cache, the command line, the drafter settings it brings, and copy decoding next to it.  No GPU: the engine parts run on the
CPU build of the tests only when YASAKANI_TEST_INK2_DLL and YASAKANI_TEST_INK2_SOURCE are set (see test_inkling2.py)."""
import os
import unittest
from types import SimpleNamespace

from yasakani import _ink2_copyedit as ce
from yasakani import _ink2_draft_gov as gov
from yasakani import cli
from yasakani import gpu as G
from yasakani import inkling2 as ink
from yasakani._native import YasakaniError

from tests.test_inkling2_context import CONFIG, DRAFTER, WEIGHTS

# GPU memory taken by the engine on the RTX 5090 in one measurement (MiB above the use before the start; the converted quality folder
# with its drafter, rollback 512): context -> (0.3.2 path, fast path).  The rule must not save more than the fast path did.
MEASURED = {33024: (20393, 18971), 131072: (29139, 21393)}


class ModeTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ink.fast_mode_of(None, {}), "auto")
        self.assertEqual(ink.fast_mode_of(None, {"YASAKANI_INK2_FAST": " OFF "}), "off")
        self.assertEqual(ink.fast_mode_of("on", {"YASAKANI_INK2_FAST": "off"}), "on")       # the option before the variable
        with self.assertRaises(YasakaniError):
            ink.fast_mode_of(None, {"YASAKANI_INK2_FAST": "max"})

    def test_auto_is_on_for_the_rtx_50_series_only(self):
        self.assertTrue(ink.fast_expected("auto", (12, 0)))
        for cc in ((8, 6), (8, 9), (12, 1), None):
            self.assertFalse(ink.fast_expected("auto", cc), cc)
        self.assertTrue(ink.fast_expected("on", (8, 6)))
        self.assertFalse(ink.fast_expected("off", (12, 0)))

    def test_the_options_of_the_fast_path(self):
        self.assertEqual(dict(ink.FAST_LOAD_OPTIONS), {"max_rows": 256, "chunk": 256, "kv_mxfp8": 1, "kv_hot": 128})
        opts = dict(ink.FAST_ENGINE_OPTIONS)
        self.assertEqual(opts["flash_attn"], 3)
        self.assertEqual(opts["nvfp4_prefill"], 4)
        self.assertEqual((opts["fast"], opts["splitk_attn"], opts["flash_min_rows"], opts["splitk_max_rows"]), (1, 1, 9, 8))

    def test_command_line(self):
        p = cli.build_parser()
        for cmd in ("chat", "serve", "bench"):
            a = p.parse_args([cmd, "--fast", "off", "--copy", "auto"])
            self.assertEqual((a.fast, a.copy), ("off", "auto"), cmd)
            self.assertIsNone(p.parse_args([cmd]).fast)


class PlanTests(unittest.TestCase):
    def test_the_8bit_cache(self):
        fp32 = ink.bytes_per_position(CONFIG)
        self.assertEqual(ink.bytes_per_position(CONFIG, mxfp8=True), fp32 // 4 + fp32 // 128)    # 16,896 B: codes + scales
        self.assertEqual(G.ink2_bytes_per_position(16896, True, fast=True), 16896 + G.INK2_DRAFT_BYTES_PER_POSITION)

    def test_saves_no_more_than_measured(self):
        files = WEIGHTS + DRAFTER
        slow = G.ink2_bytes_per_position(ink.bytes_per_position(CONFIG), True)
        fast = G.ink2_bytes_per_position(ink.bytes_per_position(CONFIG, mxfp8=True), True, fast=True)
        for ctx, (m_slow, m_fast) in MEASURED.items():
            saved = G.ink2_need_mib(ctx, files, slow, 512) - G.ink2_need_mib(ctx, files, fast, 512, fast=True)
            self.assertLessEqual(saved, m_slow - m_fast, ctx)
            self.assertGreater(saved, 0.9 * (m_slow - m_fast) - 600, ctx)              # and not far less

    def test_more_context_than_without(self):
        files = WEIGHTS + DRAFTER
        slow = G.ink2_context(30000, files, G.ink2_bytes_per_position(ink.bytes_per_position(CONFIG), True), 1 << 20, 512)
        fast = G.ink2_context(30000, files, G.ink2_bytes_per_position(ink.bytes_per_position(CONFIG, True), True, True), 1 << 20, 512,
                              fast=True)
        self.assertGreater(fast, 3 * slow)


class DrafterTests(unittest.TestCase):
    def fake(self, fast):
        calls = []
        nat = SimpleNamespace(draft_set=lambda name, v: calls.append((name, v)), position=lambda: 0)
        if fast:
            nat.draft_fast = True
        return nat, calls

    def test_fast_path_brings_split_k_and_the_hybrid_rule(self):
        nat, calls = self.fake(True)
        gov.after_load(nat)
        self.assertEqual(calls, [("attn_sk", 1)])
        self.assertEqual(gov.draft_gov(nat), "hyb")

    def test_without_it_the_drafter_of_032(self):
        nat, calls = self.fake(False)
        gov.after_load(nat)
        self.assertEqual(calls, [])
        self.assertFalse(hasattr(nat, "_cgov"))
        self.assertEqual(gov.draft_attn_sk(nat), 0)


class CopyTests(unittest.TestCase):
    def test_mode_and_rows(self):
        self.assertEqual(ce.mode_of(None, {}), "off")
        self.assertEqual(ce.mode_of(None, {"YASAKANI_INK2_COPY": "auto"}), "auto")
        with self.assertRaises(ValueError):
            ce.min_of(3, {})
        self.assertEqual(ce.max_rows_of(SimpleNamespace(flash_attn=3)), ce.FLASH_SAFE_ROWS)
        self.assertGreater(ce.max_rows_of(SimpleNamespace()), 1000)


class SelfcheckTests(unittest.TestCase):
    """The self-check compares the standard path unless --fast is given: its reference tokens are those of the standard path."""

    def test_the_standard_path_unless_fast_is_given(self):
        from yasakani import inkling2_cli as icli
        self.assertEqual(icli.selfcheck_fast(SimpleNamespace()), "off")
        self.assertEqual(icli.selfcheck_fast(SimpleNamespace(fast=None)), "off")
        for mode in ("auto", "on", "off"):
            self.assertEqual(icli.selfcheck_fast(SimpleNamespace(fast=mode)), mode)

@unittest.skipUnless(os.environ.get("YASAKANI_TEST_INK2_DLL") and os.environ.get("YASAKANI_TEST_INK2_SOURCE"),
                     "needs YASAKANI_TEST_INK2_DLL and YASAKANI_TEST_INK2_SOURCE")
class HostEngineTests(unittest.TestCase):
    """The CPU build: the fast path stays off (it runs on the GPU only), whatever is asked, and the engine is that of 0.3.2."""

    def test_off_on_the_cpu_build(self):
        from tests.test_inkling2 import tiny_folder
        pack = ink.Ink2Pack.load(tiny_folder())
        for mode in ("auto", "on", "off"):
            eng = ink.Ink2Engine(pack, 256, dll=os.environ["YASAKANI_TEST_INK2_DLL"], fast=mode)
            try:
                self.assertFalse(eng.fast, mode)
                self.assertFalse(eng.device_greedy, mode)
                self.assertTrue(eng.fast_note.startswith("fast path: off"), eng.fast_note)
            finally:
                eng.close()


if __name__ == "__main__":
    unittest.main()
