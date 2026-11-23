# SPDX-License-Identifier: Apache-2.0
"""0.3.4: the rule of the wide copy passes (yasakani/widecheck.py: decide) with the copy-cycle ratios users reported.

RTX 50 (``native``) keeps a width unless every round is over NATIVE_MARGIN x W / 16 (0.9; 0.8 before).  On the RTX 5060 Ti of issue #31
the old limit closed all three widths, although with every width open the three edit tasks ran 1.23 to 1.24 times as fast with the same
output.  RTX 30 / 40 keep MARGIN = 0.8.  Each case: the greedy rounds and the sampled round as the report printed them, and the widths
the check opened before (0.3.3) and opens now.
"""
import unittest

from yasakani import widecheck as wc

# card, native (RTX 50), {width: (greedy rounds, sampled)}, widths opened by 0.3.3, widths opened now
REPORTED = [
    ("RTX 5060 Ti, #31", True, {32: ((1.79, 1.73, 1.78), 1.79), 48: ((2.55, 2.53, 2.54), 2.54), 64: ((3.22, 3.21, 3.21), 3.21)},
     [], [32, 48, 64]),
    ("RTX 5060 Ti, #24", True, {32: ((1.76, 1.75, 1.76), 1.76), 48: ((2.54, 2.45, 2.45), 2.45), 64: ((3.00, 3.00, 3.01), 3.00)},
     [64], [32, 48, 64]),
    ("RTX 5090 (tests and documents)", True, {32: ((1.39,) * 3, None), 48: ((1.94,) * 3, None), 64: ((2.27,) * 3, None)},
     [32, 48, 64], [32, 48, 64]),
    ("RTX 4090, #24", False, {32: ((1.19, 1.18, 1.24), 1.23), 48: ((1.65, 1.62, 1.61), 1.63), 64: ((1.81, 1.91, 1.93), 1.91)},
     [32, 48, 64], [32, 48, 64]),
    ("RTX 3090, #24", False, {32: ((1.48, 1.47, 1.47), 1.48), 48: ((2.03, 2.05, 2.03), 2.04), 64: ((2.44, 2.42, 2.45), 2.44)},
     [32, 48, 64], [32, 48, 64]),
    ("RTX 3090 Ti, #24", False, {32: ((1.48, 1.48, 1.48), 1.49), 48: ((2.02, 2.04, 2.03), 2.04), 64: ((2.44, 2.40, 2.41), 2.45)},
     [32, 48, 64], [32, 48, 64]),
    ("RTX 3090 Ti max-quality, #24", False,
     {32: ((1.35, 1.36, 1.34), 1.35), 48: ((1.82, 1.86, 1.83), 1.84), 64: ((2.18, 2.19, 2.18), 2.18)}, [32, 48, 64], [32, 48, 64]),
]


def timing_of(ratios, t16=40.0):
    t = wc.Timing()
    n = len(next(iter(ratios.values()))[0])
    t.rounds = [{w: (t16, t16 * r[0][k]) for w, r in ratios.items()} for k in range(n)]
    t.sampled = {w: (t16, t16 * r[1]) for w, r in ratios.items() if r[1] is not None}
    return t


class ReportedCardsTests(unittest.TestCase):

    def test_reported_cards(self):
        for card, native, ratios, before, now in REPORTED:
            t = timing_of(ratios)
            self.assertEqual(wc.widths_of(wc.decide(t, native, margin=wc.MARGIN)[0]), before, card)   # the rule of 0.3.3
            mask, ws = wc.decide(t, native)
            self.assertEqual(wc.widths_of(mask), now, card)
            self.assertFalse(any(w.unsure for w in ws), card)

    def test_limits(self):
        """RTX 50: 1.80 / 2.70 / 3.60; RTX 30 / 40 as before: 1.60 / 2.40 / 3.20."""
        t = timing_of(REPORTED[0][2])
        self.assertEqual([round(w.limit, 2) for w in wc.decide(t, True)[1]], [1.8, 2.7, 3.6])
        self.assertEqual([round(w.limit, 2) for w in wc.decide(t, False)[1]], [1.6, 2.4, 3.2])
        self.assertEqual(wc.decide(t, False)[0], 0)                  # the #31 ratios on an RTX 30 / 40 card: still closed

    def test_a_kept_width_still_pays(self):
        """At the limit a cycle of W rows gives W tokens with every row accepted: (W / 16) / (0.9 x W / 16) = 1.11 times 16 rows."""
        self.assertLess(wc.NATIVE_MARGIN, 1.0)
        for w in wc.WIDTHS:
            self.assertAlmostEqual((w / 16) / wc.limit(w, wc.NATIVE_MARGIN), 1 / 0.9)
        over = {w: ((wc.limit(w, wc.NATIVE_MARGIN) + 0.01,) * 3, wc.limit(w, wc.NATIVE_MARGIN) + 0.01) for w in wc.WIDTHS}
        self.assertEqual(wc.decide(timing_of(over), True)[0], 0)      # every round just over the new limit: closed
        even = {w: ((w / 16,) * 3, w / 16) for w in wc.WIDTHS}         # no faster than 16 rows even when all are accepted
        self.assertEqual(wc.decide(timing_of(even), True)[0], 0)

    def test_one_round_under_keeps_it(self):
        """As before on RTX 50: one round under the limit keeps the width."""
        r = {32: ((1.95, 1.79, 1.95), 1.95), 48: ((2.5,) * 3, 2.5), 64: ((3.0,) * 3, 3.0)}
        self.assertEqual(wc.widths_of(wc.decide(timing_of(r), True)[0]), [32, 48, 64])


if __name__ == "__main__":
    unittest.main()
