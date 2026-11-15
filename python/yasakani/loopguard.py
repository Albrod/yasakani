# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Loop guard: stop a generation that has fallen into an endless repetition.

Models sometimes get stuck writing the same few ids forever (or thousands of digits in a row) while "thinking".  The guard watches the
ids as they are committed and reports a loop; the caller then cancels the generation and, inside the thinking, asks the model to answer
from what it has.  Requests without a loop are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PERIOD_MAX = 16            # longest repeating unit that is recognised (ids)
RUN_REASONING = 256        # an exact repetition this long inside the thinking is a loop
RUN_ANSWER = 1024          # ... inside the answer
DIGITS = 1000              # this many digit ids in a row inside the thinking is a loop

BRIDGE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n")
NOTICE = "\n\n[loop guard: the same sequence kept repeating, so the thinking was cut here and the answer follows]\n"
BUDGET_NOTICE = "\n\n[thinking budget: the thinking reached its limit, so it was cut here and the answer follows]\n"


def loop_period(ids: np.ndarray, start: int, n: int, run: int) -> int:
    """If the last ``run + p`` ids of ``ids[start:n]`` repeat with a period ``p <= PERIOD_MAX``, the smallest such p, else 0."""
    if n - start < run + PERIOD_MAX:
        return 0
    t = ids[n - run - PERIOD_MAX:n]
    tail = t[PERIOD_MAX:]
    for p in range(1, PERIOD_MAX + 1):
        if np.array_equal(tail, t[PERIOD_MAX - p:len(t) - p]):
            return p
    return 0


@dataclass
class LoopHit:
    where: str            # "reasoning" | "answer"
    kind: str             # "repeat" | "digits" | "budget"
    at: int               # ids to keep: the smallest count at which the rule holds
    period: int = 0
    run: int = 0


class LoopWatch:
    """Feed it the growing id array (``check(ids)``); it returns a ``LoopHit`` once, then stays quiet.

    Every position is looked at, however the ids arrive: ``LoopHit.at`` is the smallest number of ids at which a rule holds (repeat,
    then digits, then budget).  The caller keeps exactly ``at`` ids, so the text of a request does not depend on how many ids the
    engine commits per pass or on when the caller looks."""

    def __init__(self, think_end_id: int, digit_ids: np.ndarray, in_reasoning: bool, base: int = 0, budget: int = 0, loops: bool = True):
        self.think_end = int(think_end_id)
        self.digits = np.asarray(digit_ids, np.int32)
        self.base = int(base)                           # ids before this index belong to an earlier stage and are not looked at
        self.budget = int(budget)                       # > 0: the thinking may take this many ids at most (0 = no limit)
        self.loops = bool(loops)                        # False: only the budget is watched
        self.think_at: Optional[int] = None if in_reasoning else -1
        self.checked = self.base
        self.dig_run = 0
        self.runs = [0] * (PERIOD_MAX + 1)              # per period p: ids in a row with ids[i] == ids[i - p], up to the last checked id
        self.hit: Optional[LoopHit] = None

    def check(self, ids: np.ndarray) -> Optional[LoopHit]:
        n = len(ids)
        if self.hit is not None or n <= self.checked:
            return None
        ids = np.asarray(ids)
        lo, self.checked = self.checked, n
        if self.think_at is None:
            found = np.flatnonzero(ids[lo:n] == self.think_end)
            k = lo + int(found[0]) if len(found) else n          # ids[lo:k] are thinking
            self.hit = self._reasoning(ids, lo, k)
            if self.hit is not None or k == n:
                return self.hit
            self.think_at = k
            self.runs = [0] * (PERIOD_MAX + 1)
            lo = k + 1
        if self.loops:
            s = max(self.think_at + 1, self.base)
            m, p = self._repeat(ids, max(lo, s), n, s, RUN_ANSWER)
            if m:
                self.hit = LoopHit(where="answer", kind="repeat", at=m, period=p)
        return self.hit

    def _reasoning(self, ids: np.ndarray, lo: int, k: int) -> Optional[LoopHit]:
        found = []                                       # (ids kept, rule order, hit)
        if self.loops:
            m, p = self._repeat(ids, lo, k, self.base, RUN_REASONING)
            if m:
                found.append((m, 0, LoopHit(where="reasoning", kind="repeat", at=m, period=p)))
            m = self._digits(ids, lo, k)
            if m:
                found.append((m, 1, LoopHit(where="reasoning", kind="digits", at=m, run=DIGITS)))
        if self.budget > 0:
            m = self.base + self.budget
            if lo < m <= k:
                found.append((m, 2, LoopHit(where="reasoning", kind="budget", at=m)))
        return min(found, key=lambda f: (f[0], f[1]))[2] if found else None

    def _repeat(self, ids: np.ndarray, a0: int, b: int, s: int, run: int):
        """The smallest m in (a0, b] such that ``loop_period(ids, s, m, run)`` is nonzero, and that period; (0, 0) if there is none.
        ``self.runs`` carries the counts from the ids before a0 (same start s)."""
        best_m, best_p = 0, 0
        first = s + run + PERIOD_MAX - 1                 # loop_period needs m - s >= run + PERIOD_MAX, i.e. index m - 1 >= first
        for p in range(1, PERIOD_MAX + 1):
            a = max(a0, s + p)
            if a >= b:
                continue
            carry = self.runs[p] if a == a0 else 0
            idx = np.arange(b - a)
            same = ids[a:b] == ids[a - p:b - p]
            cnt = idx - np.maximum.accumulate(np.where(same, -1 - carry, idx))
            self.runs[p] = int(cnt[-1])
            ok = np.flatnonzero((cnt >= run) & (idx >= first - a))
            if len(ok) and (best_m == 0 or a + int(ok[0]) + 1 < best_m):
                best_m, best_p = a + int(ok[0]) + 1, p
        return best_m, best_p

    def _digits(self, ids: np.ndarray, lo: int, k: int) -> int:
        """The smallest m in (lo, k] at which the last DIGITS ids are all digits (0 if there is none)."""
        if k <= lo:
            return 0
        idx = np.arange(k - lo)
        dig = np.isin(ids[lo:k], self.digits)
        cnt = idx - np.maximum.accumulate(np.where(dig, -1 - self.dig_run, idx))
        self.dig_run = int(cnt[-1])
        ok = np.flatnonzero(cnt >= DIGITS)
        return lo + int(ok[0]) + 1 if len(ok) else 0
