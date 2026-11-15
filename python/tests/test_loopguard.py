# SPDX-License-Identifier: Apache-2.0
"""The loop guard and the thinking budget cut at a position that depends only on the ids, not on how they arrive."""
import unittest

import numpy as np

from yasakani import loopguard
from yasakani.loopguard import DIGITS, RUN_ANSWER, RUN_REASONING, LoopWatch, loop_period

THINK_END = 50
DIGIT_IDS = np.arange(10, 20, dtype=np.int32)


def reference(ids, in_reasoning=True, base=0, budget=0, loops=True):
    """The rules applied one id at a time: (where, kind, ids kept, period) of the first hit, or None."""
    think_at = None if in_reasoning else -1
    dig = 0
    for m in range(base + 1, len(ids) + 1):
        t = int(ids[m - 1])
        if think_at is None:
            if t == THINK_END:
                think_at = m - 1
                continue
            dig = dig + 1 if t in DIGIT_IDS else 0
            p = loop_period(ids, base, m, RUN_REASONING)
            if loops and p:
                return ("reasoning", "repeat", m, p)
            if loops and dig >= DIGITS:
                return ("reasoning", "digits", m, 0)
            if budget and m - base >= budget:
                return ("reasoning", "budget", m, 0)
        else:
            p = loop_period(ids, max(think_at + 1, base), m, RUN_ANSWER)
            if loops and p:
                return ("answer", "repeat", m, p)
    return None


def watch(ids, chunks, in_reasoning=True, base=0, budget=0, loops=True):
    w = LoopWatch(THINK_END, DIGIT_IDS, in_reasoning, base=base, budget=budget, loops=loops)
    n = 0
    for c in chunks:
        n = min(len(ids), n + c)
        h = w.check(ids[:n])
        if h is not None:
            return (h.where, h.kind, h.at, h.period)
        if n == len(ids):
            break
    h = w.check(ids)
    return None if h is None else (h.where, h.kind, h.at, h.period)


def chunkings(n, rng):
    yield [1] * n
    for c in (2, 7, 16, 48, 64, 1000):
        yield [c] * (n // c + 2)
    yield list(rng.integers(1, 80, size=n))
    yield [n]


class LoopWatchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def noise(self, n):
        return self.rng.integers(100, 5000, size=n).astype(np.int32)

    def same_for_all_chunkings(self, ids, expect_kind, **kw):
        ref = reference(ids, **kw)
        if expect_kind is None:
            self.assertIsNone(ref)
        else:
            self.assertIsNotNone(ref)
            self.assertEqual(ref[1], expect_kind)
        for ch in chunkings(len(ids), self.rng):
            self.assertEqual(watch(ids, ch, **kw), ref, "chunks %s..." % ch[:5])
        return ref

    def test_repeat_in_the_thinking(self):
        unit = self.noise(5)
        ids = np.concatenate([self.noise(300), np.tile(unit, 120), self.noise(50)])
        ref = self.same_for_all_chunkings(ids, "repeat")
        self.assertEqual(ref[3], 5)
        self.assertLess(ref[2], 300 + 5 * 120)

    def test_repeat_that_stops_inside_one_poll(self):
        # exactly long enough to count, then broken: a check that only looks at the newest id would miss it with big chunks
        ids = np.concatenate([self.noise(40), np.tile(self.noise(3), (RUN_REASONING + 3) // 3 + 1), self.noise(200)])
        self.same_for_all_chunkings(ids, "repeat")

    def test_repeat_in_the_answer(self):
        ids = np.concatenate([self.noise(30), [THINK_END], self.noise(40), np.tile(self.noise(7), 200), self.noise(10)]).astype(np.int32)
        ref = self.same_for_all_chunkings(ids, "repeat")
        self.assertEqual(ref[0], "answer")

    def test_answer_from_the_start(self):
        ids = np.concatenate([self.noise(20), np.tile(self.noise(4), 300)]).astype(np.int32)
        ref = self.same_for_all_chunkings(ids, "repeat", in_reasoning=False)
        self.assertEqual(ref[0], "answer")

    def test_digits(self):
        ids = np.concatenate([self.noise(30), self.rng.integers(10, 20, size=DIGITS + 100), self.noise(10)]).astype(np.int32)
        ref = self.same_for_all_chunkings(ids, "digits")
        self.assertEqual(ref[2], 30 + DIGITS)

    def test_budget(self):
        ids = self.noise(900)
        ref = self.same_for_all_chunkings(ids, "budget", budget=500)
        self.assertEqual(ref[2], 500)

    def test_budget_with_the_thinking_ending_at_the_limit(self):
        for end in (497, 498, 499, 500, 501):
            ids = self.noise(900)
            ids[end] = THINK_END
            ref = self.same_for_all_chunkings(ids, "budget" if end >= 500 else None, budget=500)
            if ref:
                self.assertEqual(ref[2], 500)

    def test_budget_only_when_loops_are_off(self):
        ids = np.concatenate([self.noise(100), np.tile(self.noise(3), 400)]).astype(np.int32)
        ref = self.same_for_all_chunkings(ids, "budget", budget=700, loops=False)
        self.assertEqual(ref[2], 700)

    def test_base(self):
        ids = np.concatenate([np.tile(self.noise(2), 200), self.noise(400)]).astype(np.int32)
        self.same_for_all_chunkings(ids, "budget", base=400, budget=300)

    def test_nothing(self):
        self.same_for_all_chunkings(self.noise(1500), None)

    def test_random_mixtures(self):
        for trial in range(40):
            parts = []
            for _ in range(self.rng.integers(2, 6)):
                kind = self.rng.integers(0, 4)
                if kind == 0:
                    parts.append(self.noise(int(self.rng.integers(1, 300))))
                elif kind == 1:
                    parts.append(np.tile(self.noise(int(self.rng.integers(1, 17))), int(self.rng.integers(5, 90))))
                elif kind == 2:
                    parts.append(self.rng.integers(10, 20, size=int(self.rng.integers(1, 1200))).astype(np.int32))
                else:
                    parts.append(np.array([THINK_END], np.int32))
            ids = np.concatenate(parts).astype(np.int32)
            budget = int(self.rng.integers(0, 2)) * int(self.rng.integers(50, 1500))
            kw = dict(budget=budget, in_reasoning=bool(trial % 5), loops=bool(trial % 7))
            ref = reference(ids, **kw)
            for ch in chunkings(len(ids), self.rng):
                self.assertEqual(watch(ids, ch, **kw), ref, "trial %d" % trial)

    def test_hit_is_reported_once(self):
        w = LoopWatch(THINK_END, DIGIT_IDS, True, budget=10)
        ids = self.noise(40)
        self.assertIsNotNone(w.check(ids[:20]))
        self.assertIsNone(w.check(ids))
        self.assertEqual(w.hit.at, 10)
        self.assertIs(loopguard.LoopWatch, LoopWatch)


if __name__ == "__main__":
    unittest.main()
