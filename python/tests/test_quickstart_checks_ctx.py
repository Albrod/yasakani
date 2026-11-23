# SPDX-License-Identifier: Apache-2.0
"""0.3.4, the quickstart: the message of --skip-checks (issue #18) and the launcher's copy of the Inkling-2 context rule with the fast
path (yasakani/gpu.py ink2_context, ink2_bytes_per_position; inkling2.py fast_expected and FAST_AUTO_CAPS: on for RTX 30, 40 and 50
cards since 0.3.4).

The launcher runs with the stand-ins of test_quickstart_web.LauncherTests (nothing is downloaded, no engine is started).
"""
import os
import unittest
from unittest import mock

from tests import test_quickstart_web as W
from yasakani import gpu as G
from yasakani import inkling2 as ink

QS = W.QS
CONFIG_CACHE = 65536                     # config.json's cache per position of the published folder (float32)


class LauncherRuleTests(unittest.TestCase):
    """estimate_ink2_context(free, fast) gives the package's context for the same free memory, on both paths."""

    def test_same_as_package_on_both_paths(self):
        files = sum(QS.INK2_FILES_BYTES)
        cfg = {"text_config": {}}
        self.assertEqual(ink.bytes_per_position(cfg), CONFIG_CACHE)              # the defaults are the published folder's numbers
        for fast in (False, True):
            per = G.ink2_bytes_per_position(ink.bytes_per_position(cfg, mxfp8=fast), True, fast=fast)
            self.assertEqual(per, QS.INK2_FAST_BYTES_PER_POSITION if fast else QS.INK2_BYTES_PER_POSITION)
            self.assertEqual(QS.INK2_OVERHEAD_MIB + (QS.INK2_FAST_OVERHEAD_MIB if fast else 0), G.ink2_overhead_mib(fast))
            for free in range(0, 40001, 11):
                want = G.ink2_context(free, files, per, 32768, 512, fast=fast)
                self.assertEqual(QS.estimate_ink2_context(free, fast=fast), want, (fast, free))
        self.assertEqual(QS.estimate_ink2_context(19500), QS.estimate_ink2_context(19500, fast=False))   # one argument: standard
        self.assertLess(QS.estimate_ink2_context(19500), QS.estimate_ink2_context(19500, fast=True))

    def test_fast_expected_as_package(self):
        self.assertEqual(QS.INK2_FAST_AUTO_CAPS, ink.FAST_AUTO_CAPS)
        for cc in ((7, 5), (8, 6), (8, 9), (9, 0), (12, 0), (12, 1), None):
            for env in ({}, {"YASAKANI_INK2_FAST": "on"}, {"YASAKANI_INK2_FAST": "off"}, {"YASAKANI_INK2_FAST": " AUTO "}):
                want = ink.fast_expected(ink.fast_mode_of(None, env), cc)
                self.assertEqual(QS.ink2_fast_expected(cc, env), want, (cc, env))


class LauncherNoteTests(W.LauncherTests):

    def test_skip_checks_says_they_were_not_run(self):
        """#18: with --skip-checks step 6 said "The checks were done before (skipped)." although they never ran."""
        h = self.home(checked=())
        self.assertEqual(self.run_main(h, "--action", "none", "--skip-checks"), 0)
        out = self.out.getvalue()
        self.assertIn("The checks are not run (--skip-checks): the output has not been checked.", out)
        self.assertNotIn("The checks were done before", out)
        self.out.truncate(0)
        self.out.seek(0)
        h2 = self.tmp / "h2"
        os.rename(h, h2)
        h = self.home(checked=("quality",))                                    # passed before: as it was
        self.assertEqual(self.run_main(h, "--action", "none"), 0)
        self.assertIn("The checks were done before (skipped).", self.out.getvalue())

    def test_inkling2_note_follows_the_fast_path(self):
        """The note gives the context of chat and serve: the fast path on RTX 30, 40 and 50 cards (0.3.4; the launcher takes no
        other cards, so the standard path's note is covered by test_fast_expected_as_package)."""
        used = 24564 - 19000
        for name, cc in (("NVIDIA GeForce RTX 5090 D", (12, 0)), ("NVIDIA GeForce RTX 4090", (8, 9)),
                         ("NVIDIA GeForce RTX 3090 Ti", (8, 6))):
            h = self.tmp / ("h_%d%d" % cc)
            card = dict(index=0, uuid="GPU-b", name=name, cc=cc, total=24564, used=used, driver="616.92")
            want = QS.estimate_ink2_context(19000, fast=True)
            self.assertTrue(ink.fast_expected("auto", cc), cc)
            self.assertTrue(8192 <= want < 32768, want)
            self.out.truncate(0)
            self.out.seek(0)
            with mock.patch.object(self, "tmp", h), mock.patch.object(QS, "query_gpus", lambda run=None, card=card: [dict(card)]), \
                    mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}):
                home = self.home(checked=("inkling2",), downloaded=("inkling2",), model="inkling2")
                self.assertEqual(self.run_main(home, "--action", "none"), 0)
            self.assertIn("(%s tokens with what is free now; at most 32,768)" % "{:,}".format(want), self.out.getvalue(), name)


def load_tests(loader, tests, pattern):
    """Only the tests of this file (LauncherNoteTests inherits those of LauncherTests, which test_quickstart_web runs)."""
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(LauncherRuleTests))
    for n in ("test_skip_checks_says_they_were_not_run", "test_inkling2_note_follows_the_fast_path"):
        suite.addTest(LauncherNoteTests(n))
    return suite


if __name__ == "__main__":
    unittest.main()
