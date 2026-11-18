# SPDX-License-Identifier: Apache-2.0
"""The scheduler (several requests at once on one engine) against the stub library: every request gets the ids, the finish and the
token counts it gets alone on the single path (the stub's stand-in model gives a stream the same ids whatever shares its passes), also
where the scheduler cuts the passes (stop ids inside the drafts, the length limit, the end of the context), when requests are
cancelled, move between the carrier and the slots, continue an earlier stage or turn, or carry images.  The finished stream an open
request keeps for its next stage (RESERVE_S) is never taken by another request, and a waiting next stage is read before new requests."""
import math
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import support
from yasakani import sched as S
from yasakani._native import YasakaniError
from yasakani.engine import CONTEXT_MARGIN, Sampling
from yasakani.runtime import EOS_IDS
from yasakani.sched import QueueFull, ScheduledEngine, Scheduler, Stage

EOS, EOS2 = EOS_IDS
GREEDY = Sampling(temperature=0.0)
MIXED = [GREEDY, Sampling(1.0, 0.95, 20, 0.0, 11, 3), Sampling(0.7, 0.8, 5, 0.05, 12, 2), Sampling(1.2, 1.0, 20, 0.0, 13, 3),
         Sampling(0.3, 0.9, 10, 0.1, 14, 3)]
HAVE_TOK = support.tokenizer_dir() is not None
KEEP_LONG = 600.0                       # RESERVE_S in the tests of kept streams: the next stages come in time also on a busy machine


def drain(t, timeout=60):
    """The result of a ticket submitted to the scheduler directly."""
    while True:
        kind, v = t.q.get(timeout=timeout)
        if kind == "done":
            return v
        if kind == "error":
            raise v


def wait_for(cond, timeout=20.0, what="a condition"):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for " + what)


class Job:
    def __init__(self, prompt, max_new, sampling, stops=EOS_IDS, after=None, cancel_at=None):
        self.prompt, self.max_new, self.sampling, self.stops = list(prompt), max_new, sampling, stops
        self.after = after                      # (job index, ids): start once that job has this many ids
        self.cancel_at = cancel_at              # cancel once this many ids are there
        self.result = self.error = None
        self.started = threading.Event()
        self.seen = 0
        self.progress = threading.Condition()


class SchedTests(unittest.TestCase):
    MAX_CTX = 4096

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.stub.clear_scripts()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=self.MAX_CTX)
        self.ref = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=self.MAX_CTX, name="ref")
        self.model(eos_every=25)
        self.scheds = []

    def tearDown(self):
        for s in self.scheds:
            s.close()
        self.stub.delay_us(0)
        self.e.close()
        self.ref.close()
        self.tmp.cleanup()

    # ---- helpers ----------------------------------------------------------------------------------------------------------------
    def model(self, eos_every=25, vocab=64):
        for e in (self.e, self.ref):
            self.stub.model(e, True, eos_every=eos_every, vocab=vocab)

    def sched(self, n, **kw):
        s = Scheduler(self.e, n, **kw)
        self.scheds.append(s)
        return s

    def single(self, prompt, max_new, s, stops=EOS_IDS):
        self.ref.reset()
        self.ref.set_sampling(s)
        return self.ref.generate(prompt, max_new, stop_ids=stops)

    def run_jobs(self, sched, jobs, timeout=60):
        """Each job on its own thread through a ScheduledEngine (as a server's request threads)."""
        def work(j):
            if j.after is not None:
                k, n = j.after
                with jobs[k].progress:
                    jobs[k].progress.wait_for(lambda: jobs[k].seen >= n or jobs[k].result is not None or jobs[k].error, timeout)
            se = ScheduledEngine(sched)
            se.set_sampling(j.sampling)
            j.started.set()

            def on_tokens(ids):
                with j.progress:
                    j.seen = len(ids)
                    j.progress.notify_all()
                if j.cancel_at is not None and len(ids) >= j.cancel_at:
                    se.cancel()
            try:
                j.result = se.generate(j.prompt, j.max_new, stop_ids=j.stops, on_tokens=on_tokens)
            except Exception as ex:                                       # noqa: BLE001
                j.error = ex
            finally:
                se.close()
                with j.progress:
                    j.progress.notify_all()
        ths = [threading.Thread(target=work, args=(j,), daemon=True) for j in jobs]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout)
            self.assertFalse(t.is_alive(), "a request did not finish")
        return jobs

    def check_same(self, jobs, what=""):
        for i, j in enumerate(jobs):
            self.assertIsNone(j.error, "job %d %s" % (i, what))
            ref = self.single(j.prompt, j.max_new, j.sampling, j.stops)
            r = j.result
            if j.cancel_at is not None:
                self.assertEqual(r.finish, "cancelled", "job %d %s" % (i, what))
                self.assertEqual(r.ids, ref.ids[:len(r.ids)], "job %d %s" % (i, what))
                continue
            msg = "job %d %s (max_new %d, %s)" % (i, what, j.max_new, j.sampling)
            self.assertEqual(r.ids, ref.ids, msg)
            self.assertEqual(r.finish, ref.finish, msg)
            self.assertEqual((r.prompt_tokens, r.completion_tokens), (ref.prompt_tokens, ref.completion_tokens), msg)

    def jobs(self, n, seed, max_new=(1, 2, 3, 17, 40, 64), samplings=MIXED, stops=EOS_IDS):
        rng = random.Random(seed)
        return [Job([rng.randrange(10, 5000) for _ in range(rng.randrange(3, 40))], rng.choice(max_new), rng.choice(samplings), stops)
                for _ in range(n)]

    @staticmethod
    def rows(trace):
        return [t for t in trace if t[0] == "rows"]

    # ---- the same ids as alone --------------------------------------------------------------------------------------------------
    def test_streams_1_to_8_and_more_than_rows(self):
        self.stub.delay_us(150)
        for n, draft_mode in ((1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 1), (7, 0), (8, 0), (18, 1)):
            with self.subTest(slots=n, draft_mode=draft_mode):
                self.stub.draft_mode(self.e, draft_mode)
                trace = []
                s = self.sched(n, trace=trace)
                jobs = self.run_jobs(s, self.jobs(max(12, n + 4), seed=40 + n))
                self.check_same(jobs, "slots %d" % n)
                passes = self.rows(trace)
                if n > 1:
                    self.assertGreater(len(passes), 0)
                    self.assertGreater(s.perf["streams"] / max(1, s.perf["passes"]), 1.0)   # passes did carry several streams
                self.assertTrue(all(len(r[2]) <= 8 for r in passes))
                s.close()
                self.assertEqual(s.status()["state"], "stopped")

    def test_stop_ids_inside_the_drafts(self):
        """Stop ids (both) as the first draft, a middle draft, the last row's draft and the answer after the last row: the stream ends
        there with the stop id, which is not committed (the slot's position stays at the ids before it)."""
        self.model(eos_every=4)
        self.stub.draft_mode(self.e, 1)                                    # all drafts right: stops come inside them
        self.stub.delay_us(100)
        seen = set()
        want = {(w, stop) for w in ("first", "middle", "last", "answer") for stop in EOS_IDS}
        for rnd in range(24):
            if want <= seen:
                break
            trace = []
            s = self.sched(4, trace=trace, back_to_one=False)
            jobs = self.run_jobs(s, self.jobs(16, seed=300 + rnd, max_new=(40, 64)))
            self.check_same(jobs, "round %d" % rnd)
            for _, tid, toks, am, a, new, share, left in self.rows(trace):
                if new[-1] in EOS_IDS:
                    k, n = len(new), len(toks)
                    where = ("answer" if k == n else "first" if k == 1 else "last" if k == n - 1 else "middle") \
                        if (k == n or toks[k] == new[-1]) else "answer"
                    if n > 2 or where == "answer":
                        seen.add((where, new[-1]))
            s.close()
        self.assertEqual(want - seen, set())

    def test_ignore_eos(self):
        self.model(eos_every=3)
        self.stub.draft_mode(self.e, 1)
        s = self.sched(3)
        jobs = self.run_jobs(s, self.jobs(8, seed=5, max_new=(30, 47), stops=None))
        self.check_same(jobs)
        for j in jobs:
            self.assertEqual(len(j.result.ids), j.max_new)
            self.assertEqual(j.result.finish, "length")
            self.assertTrue(set(j.result.ids) & set(EOS_IDS))              # stop ids came and did not end the stream

    def test_stop_id_forms(self):
        """Every form of ``stop_ids`` ends a stream where the single path ends it: None (no stop id), an empty sequence (the engine's
        own two, the model's end ids), one id, two, and three or more (the engine takes the first two; the others are not stop ids).
        Requests at once on 2 slots, with and without drafts."""
        self.model(eos_every=6, vocab=8)                                   # tokens 1000..1007 besides the stop ids
        self.stub.delay_us(100)
        often = 1003
        forms = (None, (), (EOS,), EOS_IDS, (EOS2, EOS), EOS_IDS + (often,), (often,), (often, 1005, EOS))
        for draft_mode in (0, 1):
            self.stub.draft_mode(self.e, draft_mode)
            for k, stops in enumerate(forms):
                with self.subTest(stops=stops, draft_mode=draft_mode):
                    s = self.sched(2)
                    jobs = self.run_jobs(s, self.jobs(6, seed=900 + 10 * draft_mode + k, max_new=(48,), stops=stops))
                    self.check_same(jobs, "stops %s" % (stops,))
                    fin = [j.result.finish for j in jobs]
                    if stops is None:
                        self.assertEqual(set(fin), {"length"})
                    else:
                        self.assertIn("stop", fin)                         # the case is not empty
                    if stops and len(stops) > 2:                           # the third id is not a stop id: it came and went on
                        self.assertTrue(any(stops[2] in j.result.ids[:-1] for j in jobs))
                    s.close()

    def test_length_limits(self):
        """max_new = 1 and 2; a stop id as the last id allowed, one before it, and the limit right before a stop; the limit inside what
        the drafter would have drafted (the stream then gets only the rows it may still fill)."""
        self.model(eos_every=12)
        self.stub.draft_mode(self.e, 1)
        rng = random.Random(77)
        jobs = []
        while len(jobs) < 24:
            p, smp = [rng.randrange(10, 5000) for _ in range(rng.randrange(3, 30))], rng.choice(MIXED)
            ids = self.single(p, 200, smp).ids
            k = len(ids)                                                    # the stop id is the k-th id
            if ids[-1] not in EOS_IDS or k < 4:
                continue
            for m in (k, k + 1, k - 1, 1, 2):
                jobs.append(Job(p, m, smp))
        self.stub.delay_us(100)
        trace = []
        s = self.sched(5, trace=trace, back_to_one=False)
        self.run_jobs(s, jobs)
        self.check_same(jobs)
        fin = {(j.max_new, j.result.finish, len(j.result.ids)) for j in jobs}
        self.assertTrue(any(f == "stop" and n == m for m, f, n in fin))           # the stop id as the last id allowed
        self.assertTrue(any(f == "stop" and n == m - 1 for m, f, n in fin))       # one before the limit
        self.assertTrue(any(f == "length" and n == m and m > 2 for m, f, n in fin))
        cut = [r for r in self.rows(trace) if len(r[2]) == r[7] < r[6]]          # rows cut by the limit, not by the share
        self.assertTrue(cut)
        self.assertTrue(any(len(r[5]) == r[7] for r in cut))                     # ... and the stream ended right at the limit

    def test_context_margin(self):
        """Prompts near the end of the context: plan() gives fewer ids than asked, as on the single path."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 1)
        jobs = []
        for room in (1, 2, 3, 5, 9, 17):
            n = self.MAX_CTX - CONTEXT_MARGIN - room
            jobs.append(Job([(i * 7) % 4000 + 1 for i in range(n)], 100, MIXED[room % 5]))
        jobs.append(Job([3, 4, 5], 50, GREEDY))
        self.stub.delay_us(100)
        s = self.sched(3)
        self.run_jobs(s, jobs)
        self.check_same(jobs)
        for j, room in zip(jobs, (1, 2, 3, 5, 9, 17)):
            self.assertEqual(len(j.result.ids), room + 1)
        se = ScheduledEngine(s)                                              # no room at all: refused as on the single path
        with self.assertRaises(ValueError):
            se.generate([1] * (self.MAX_CTX - CONTEXT_MARGIN), 10)
        with self.assertRaises(ValueError):
            self.single([1] * (self.MAX_CTX - CONTEXT_MARGIN), 10, GREEDY)

    # ---- moving between the carrier and the slots ---------------------------------------------------------------------------------
    def test_alone_then_slot_then_back(self):
        """A request runs on the carrier alone; a second one comes: the first moves into a slot; the second ends: the first goes back
        to the carrier alone.  Both get their ids."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(300)
        trace = []
        s = self.sched(4, trace=trace)
        jobs = [Job([5, 6, 7, 8], 400, MIXED[1]), Job([9, 10], 30, MIXED[2], after=(0, 40)), Job([11, 12, 13], 25, GREEDY, after=(0, 200))]
        self.run_jobs(s, jobs)
        self.check_same(jobs)
        st = jobs[0].result.stats
        self.assertGreaterEqual(st["moves"], 2)
        self.assertGreaterEqual(st["back_to_one"], 2)
        self.assertGreater(st["ms_passes"], 0)
        kinds = [t[0] for t in trace]
        self.assertLess(kinds.index("move"), kinds.index("back"))
        self.assertEqual(jobs[0].result.snap_mode, 0)
        self.assertEqual(s.status()["slots_busy"], 0)

    def test_cancel(self):
        """Cancelled in a slot and on the carrier alone: the stream ends at a pass boundary with the ids so far; the others are not
        touched.  (Cancelled while waiting: test_queue_full.)"""
        self.model(eos_every=0)
        self.stub.delay_us(400)
        s = self.sched(2)
        jobs = [Job([1, 2, 3], 300, GREEDY, cancel_at=20), Job([4, 5], 120, MIXED[1], after=(0, 3)),
                Job([6, 7, 8], 60, MIXED[3], after=(0, 3))]
        self.run_jobs(s, jobs)
        self.check_same(jobs)
        self.assertEqual(jobs[0].result.finish, "cancelled")
        self.assertGreaterEqual(len(jobs[0].result.ids), 20)
        self.assertGreater(jobs[0].result.stats["ms_passes"], 0)            # it was in a slot
        solo = Job([3, 1, 4, 1, 5], 500, MIXED[2], cancel_at=7)             # alone on the carrier
        self.run_jobs(s, [solo])
        self.check_same([solo])
        self.assertEqual(solo.result.stats["ms_passes"], 0)
        self.assertLess(len(solo.result.ids), 500)

    def test_stages_of_one_request(self):
        """The next stage of a request (thinking budget: the same prompt and the ids so far; loop cut: a shorter head and a bridge) is
        read on the carrier after the request's stream comes back from its slot: the same reading mode as alone."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(200)
        s = self.sched(3)
        p, smp = [21, 22, 23, 24], MIXED[3]
        for others in (False, True):
            for kind in ("budget", "loop"):
                with self.subTest(others=others, kind=kind):
                    bg = [Job([40 + i, 41, 42], 150, MIXED[i]) for i in range(2)] if others else []
                    out = {}

                    def staged():
                        if bg:
                            with bg[0].progress:
                                bg[0].progress.wait_for(lambda: bg[0].seen >= 5, 30)
                        with ScheduledEngine(s) as se:
                            se.set_sampling(smp)
                            r1 = se.generate(p, 12)
                            nxt = p + r1.ids if kind == "budget" else p + r1.ids[:9] + [7001, 7002]
                            out["r"] = (r1, se.generate(nxt, 30), nxt)
                    th = threading.Thread(target=staged, daemon=True)
                    th.start()
                    self.run_jobs(s, bg)
                    th.join(30)
                    r1, r2, nxt = out["r"]
                    self.ref.reset()
                    self.ref.set_sampling(smp)
                    a = self.ref.generate(p, 12)
                    b = self.ref.generate(nxt, 30)                         # the second call of the single path
                    self.assertEqual((r1.ids, r2.ids), (a.ids, b.ids))
                    self.assertEqual((r2.snap_mode, r2.cached_tokens), (b.snap_mode, b.cached_tokens))
                    self.assertEqual(r2.snap_mode, 1 if kind == "budget" else 0)
                    self.check_same(bg)

    def test_next_turn_continues_its_slot(self):
        """A conversation's next turn continues the KV its first turn left (in a slot or on the carrier): read as a continuation, as
        when the conversation is served alone.  Without reuse it is read from the start; a new conversation never continues another."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        for reuse in (True, False):
            with self.subTest(reuse=reuse):
                trace = []
                s = self.sched(3, reuse=reuse, trace=trace)
                p = [31, 32, 33, 34, 35]
                t1 = self.run_jobs(s, [Job(p, 20, GREEDY)])[0].result               # alone: the carrier keeps it
                other = self.run_jobs(s, [Job([77, 78, 79], 15, MIXED[1])])[0]       # another conversation: the first is parked
                self.assertEqual(other.result.snap_mode, 0)
                turn2 = p + t1.ids + [901, 902]
                j2 = self.run_jobs(s, [Job(turn2, 25, GREEDY)])[0]
                self.check_same([j2])
                if reuse:
                    self.assertEqual((j2.result.snap_mode, j2.result.cached_tokens), (1, len(p) + len(t1.ids) - 1))
                    self.assertIn("park", [t[0] for t in trace])
                else:
                    self.assertEqual(j2.result.snap_mode, 0)
                    self.assertNotIn("park", [t[0] for t in trace])
                # the turn that follows at once is read on the carrier itself
                turn3 = turn2 + j2.result.ids + [903]
                j3 = self.run_jobs(s, [Job(turn3, 10, GREEDY)])[0]
                self.check_same([j3])
                self.assertEqual(j3.result.snap_mode, 1)
                s.close()

    def test_a_running_request_keeps_its_stream(self):
        """Between two stages of a request, another request whose prompt happens to continue its stream does not take it: that one is
        read from the start, and the request's next stage still continues its own stream."""
        self.model(eos_every=0)
        s = self.sched(2)
        p = [61, 62, 63]
        with ScheduledEngine(s) as se:
            se.set_sampling(GREEDY)
            r1 = se.generate(p, 8)                                         # alone: the carrier holds it until the next stage
            intruder = self.run_jobs(s, [Job(p + r1.ids + [5], 6, GREEDY)])[0]
            r2 = se.generate(p + r1.ids, 10)
        self.check_same([intruder])
        self.assertEqual(intruder.result.snap_mode, 0)
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (1, len(p) + len(r1.ids) - 1))
        self.assertEqual(r1.ids + r2.ids, self.single(p, 18, GREEDY).ids)

    def test_slot_reuse_order(self):
        """A turn that continues a conversation kept in a slot takes that slot; a request that needs a slot when every slot holds a
        finished conversation takes the one used longest ago."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 1)
        self.stub.delay_us(200)
        trace = []
        s = self.sched(2, trace=trace, back_to_one=False)
        pa, pb, pc, pd = [50, 51, 52], [60, 61], [80, 81], [90, 91]
        with s._cond:                                                      # both in one round: both are read into slots
            ta, tb = s.submit(Stage(pa, 40, EOS_IDS, GREEDY)), s.submit(Stage(pb, 60, EOS_IDS, MIXED[1]))
        ra, rb = drain(ta), drain(tb)
        sa, sb = int(ra.stats["slot"]), int(rb.stats["slot"])
        self.assertEqual(sorted((sa, sb)), [0, 1])
        wait_for(lambda: s.status()["slots_idle"] == 2, what="both slots to hold their streams")
        a2 = Job(pa + ra.ids + [905], 10, GREEDY)                          # alone: a's slot comes back to the carrier
        self.run_jobs(s, [a2])
        self.assertEqual(a2.result.snap_mode, 1)
        self.assertEqual([t[2] for t in trace if t[0] == "read"][-1], sa)
        n0 = len(trace)
        self.stub.delay_us(1000)                                           # c runs alone for a second or more ...
        tc = s.submit(Stage(pc, 3000, EOS_IDS, MIXED[2]))                  # (a2 is parked into a's slot first)
        self.assertEqual(tc.q.get(timeout=30)[0], "ids")
        td = s.submit(Stage(pd, 30, EOS_IDS, MIXED[3]))                    # ... when d comes: c moves into b's slot (used longer ago)
        rd = drain(td)
        tc.cancel()
        rc = drain(tc)
        self.stub.delay_us(0)
        self.assertIn(("park", sa), trace[n0:])
        self.assertEqual([t[2] for t in trace[n0:] if t[0] == "move"], [sb])
        self.check_same([a2])
        for p, n, smp, r in ((pa, 40, GREEDY, ra), (pb, 60, MIXED[1], rb), (pd, 30, MIXED[3], rd)):
            ref = self.single(p, n, smp)
            self.assertEqual((r.ids, r.finish), (ref.ids, ref.finish))
        self.assertEqual(rc.finish, "cancelled")
        self.assertEqual(rc.ids, self.single(pc, 3000, MIXED[2]).ids[:len(rc.ids)])

    # ---- the stream an open request keeps for its next stage ------------------------------------------------------------------------
    def first_stages_together(self, s, reqs):
        """The first stages of requests [(ScheduledEngine, prompt, max_new)] reach the scheduler in one round, so every one is read
        into a slot (none runs alone); each request's thread then takes its own end.  Returns their results."""
        with s._cond:
            tickets = [se._submit(Stage(list(p), n, EOS_IDS, se.sampling, se)) for se, p, n in reqs]
        out = [None] * len(reqs)

        def take(i):
            se, p, n = reqs[i]
            out[i] = se._wait(tickets[i], None, se.plan(len(p), n) + 1)
        ths = [threading.Thread(target=take, args=(i,), daemon=True) for i in range(len(reqs))]
        for t in ths:
            t.start()
        for t in ths:
            t.join(60)
        self.assertTrue(all(r is not None for r in out), "a first stage did not finish")
        return out

    def two_stage_reference(self, p, n1, n2, smp, cut=None):
        """Both stages of a request on the single path: the second continues the first (or, with ``cut``, a loop cut: the head up to
        ``cut`` and a bridge)."""
        self.ref.reset()
        self.ref.set_sampling(smp)
        a = self.ref.generate(p, n1)
        b = self.ref.generate(p + a.ids if cut is None else p + a.ids[:cut] + [7001, 7002], n2)
        return a, b

    def test_kept_streams_are_not_taken(self):
        """Two requests end their first stages in the two slots and keep them for their next stages.  Two new requests come at once
        meanwhile, while nothing runs (before the fix the first one ran alone on the carrier and then moved into a kept slot, or went
        straight into one): they wait.  Both next stages continue their streams in their slots, read as on the single path (snap_mode
        1 and its cached count), before the new requests; every request gets its ids."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(300)
        with mock.patch.object(S, "RESERVE_S", KEEP_LONG):
            for trial in range(3):
                with self.subTest(trial=trial):
                    trace = []
                    s = self.sched(2, back_to_one=False, trace=trace)
                    reqs = [([11, 12, 13, 14 + trial], 30, 10, GREEDY), ([21, 22, 23 + trial], 24, 10, MIXED[1])]
                    ses = [ScheduledEngine(s) for _ in reqs]
                    for se, (_, _, _, smp) in zip(ses, reqs):
                        se.set_sampling(smp)
                    r1 = self.first_stages_together(s, [(se, p, n1) for se, (p, n1, _, _) in zip(ses, reqs)])
                    self.assertEqual([t[4] for t in trace if t[0] == "read"], [False, False])     # both were read into slots
                    n0 = len(trace)
                    news = [([31, 32, 33 + trial], MIXED[2]), ([41, 42, 43 + trial], MIXED[3])]
                    with s._cond:
                        tn = [s.submit(Stage(p, 20, EOS_IDS, smp)) for p, smp in news]
                    time.sleep(0.2)
                    self.assertEqual(s.status()["queued"], 2)                                       # they wait
                    self.assertEqual([t for t in trace[n0:] if t[0] in ("read", "move", "park")], [])
                    out = [None, None]

                    def second(i):
                        p, n1, n2, _ = reqs[i]
                        try:
                            out[i] = ses[i].generate(p + r1[i].ids, n2)
                        finally:
                            ses[i].close()
                    ths = [threading.Thread(target=second, args=(i,), daemon=True) for i in range(2)]
                    for t in ths:
                        t.start()
                    for t in ths:
                        t.join(60)
                    rn = [drain(t) for t in tn]
                    for i, (p, n1, n2, smp) in enumerate(reqs):
                        a, b = self.two_stage_reference(p, n1, n2, smp)
                        self.assertEqual((r1[i].ids, out[i].ids), (a.ids, b.ids))
                        self.assertEqual((out[i].snap_mode, out[i].cached_tokens), (b.snap_mode, b.cached_tokens))
                        self.assertEqual((b.snap_mode, b.cached_tokens), (1, len(p) + n1 - 1))
                    for (p, smp), r in zip(news, rn):
                        ref = self.single(p, 20, smp)
                        self.assertEqual((r.ids, r.finish, r.snap_mode), (ref.ids, ref.finish, 0))
                    reads = [t for t in trace[n0:] if t[0] == "read"]
                    new_ids = {t.id for t in tn}
                    self.assertEqual(len(reads), 4)
                    self.assertNotIn(reads[0][1], new_ids)                              # a next stage came first ...
                    self.assertEqual(sorted(t[2] for t in reads if t[1] not in new_ids), [0, 1])   # ... each from its slot
                    self.assertEqual([t[2] for t in reads if t[1] in new_ids], [-1, -1])
                    s.close()

    def test_kept_streams_under_load(self):
        """The same with requests that come at random times: on 2 to 4 slots, requests with two stages (the second continues the
        first, or cuts it as a loop cut does; the client takes 0 to 20 ms in between) and single requests.  Every next stage is read as
        on the single path (snap_mode and cached count), and every request gets the ids it gets alone."""
        self.model(eos_every=25)
        rng = random.Random(4711)
        with mock.patch.object(S, "RESERVE_S", KEEP_LONG):
            for rnd in range(8):
                n = rng.choice((2, 3, 4))
                draft_mode = rnd % 3
                self.stub.draft_mode(self.e, draft_mode)
                self.stub.delay_us(rng.choice((100, 300)))
                trace = []
                s = self.sched(n, back_to_one=bool(rnd % 2), trace=trace)
                plans = []
                for k in range(rng.randrange(n + 1, 2 * n + 3)):
                    p = [rng.randrange(10, 5000) for _ in range(rng.randrange(3, 20))]
                    n1, n2 = rng.choice((1, 2, 9, 30, 60)), rng.choice((1, 5, 30))
                    cut = rng.choice((None, None, 0, 3))
                    plans.append(dict(kind="two", p=p, n1=n1, n2=n2, cut=cut, smp=rng.choice(MIXED), start=rng.uniform(0, 0.03),
                                      gap=rng.uniform(0, 0.02)))
                for k in range(rng.randrange(2, 9)):
                    p = [rng.randrange(10, 5000) for _ in range(rng.randrange(3, 20))]
                    plans.append(dict(kind="one", p=p, n1=rng.choice((1, 3, 20, 50)), smp=rng.choice(MIXED), start=rng.uniform(0, 0.05)))
                res = [None] * len(plans)

                def work(i):
                    q = plans[i]
                    time.sleep(q["start"])
                    try:
                        with ScheduledEngine(s) as se:
                            se.set_sampling(q["smp"])
                            r1 = se.generate(q["p"], q["n1"])
                            if q["kind"] == "one":
                                res[i] = (r1,)
                                return
                            time.sleep(q["gap"])
                            cut = q["cut"]
                            nxt = q["p"] + r1.ids if cut is None else q["p"] + r1.ids[:min(cut, len(r1.ids))] + [7001, 7002]
                            res[i] = (r1, se.generate(nxt, q["n2"]))
                    except Exception as ex:                                       # noqa: BLE001
                        res[i] = ex
                ths = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(plans))]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join(120)
                    self.assertFalse(t.is_alive(), "a request did not finish")
                s.close()
                self.stub.delay_us(0)
                for i, q in enumerate(plans):
                    msg = "round %d, request %d (%s, %d slots, draft mode %d)" % (rnd, i, q["kind"], n, draft_mode)
                    self.assertNotIsInstance(res[i], Exception, msg)
                    if q["kind"] == "one":
                        ref = self.single(q["p"], q["n1"], q["smp"])
                        self.assertEqual((res[i][0].ids, res[i][0].finish), (ref.ids, ref.finish), msg)
                        continue
                    r1, r2 = res[i]
                    cut = None if q["cut"] is None else min(q["cut"], len(r1.ids))
                    a, b = self.two_stage_reference(q["p"], q["n1"], q["n2"], q["smp"], cut)
                    self.assertEqual((r1.ids, r1.finish, r2.ids, r2.finish), (a.ids, a.finish, b.ids, b.finish), msg)
                    self.assertEqual((r2.snap_mode, r2.cached_tokens), (b.snap_mode, b.cached_tokens), msg)

    def test_next_turns_of_two_alike_conversations_claim_the_older_stream_first(self):
        """Two conversations with the same tokens end their first turns in the two slots and close with keep_s (tool calls), the one in
        the second slot first.  The first next turn that comes claims that stream (its keep runs out first); the second next turn claims
        the other.  Both continue a stream (the ids and prompt-reuse mode of the single path)."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        trace = []
        s = self.sched(2, back_to_one=False, trace=trace)
        p = [11, 12, 13]
        ses = [ScheduledEngine(s) for _ in range(2)]
        for se in ses:
            se.set_sampling(GREEDY)
        r1 = self.first_stages_together(s, [(se, p, 10) for se in ses])
        slots = [int(r.stats["slot"]) for r in r1]
        self.assertEqual(sorted(slots), [0, 1])
        self.assertEqual(r1[0].ids, r1[1].ids)
        first = slots.index(1)                                                  # the conversation in the second slot closes first
        ses[first].close(keep_s=KEEP_LONG)
        time.sleep(0.01)
        ses[1 - first].close(keep_s=KEEP_LONG)
        p2 = p + r1[0].ids + [60, 61]
        with s._cond:                                                           # both claims before the scheduler reads either
            held = [s._slots[i].held for i in range(2)]
            t1 = s.submit(Stage(p2, 5, EOS_IDS, GREEDY))
            self.assertIs(held[1].claim, t1)
            t2 = s.submit(Stage(p2, 5, EOS_IDS, GREEDY))
            self.assertIs(held[0].claim, t2)
        ra, rb = drain(t1), drain(t2)
        self.ref.reset()
        self.ref.set_sampling(GREEDY)
        w1 = self.ref.generate(p, 10)
        w2 = self.ref.generate(p2, 5)
        self.assertEqual(r1[0].ids, w1.ids)
        for r in (ra, rb):
            self.assertEqual((r.ids, r.finish, r.snap_mode, r.cached_tokens), (w2.ids, w2.finish, 1, w2.cached_tokens))
        reads = {t[1]: t[2] for t in trace if t[0] == "read"}
        self.assertEqual((reads[t1.id], reads[t2.id]), (1, 0))

    def test_turn_keep_rules(self):
        """How long the stream of an answer with tool calls is kept for the next turn: about the time reading the conversation again
        takes (sched.read_seconds over READ_S), from KEEP_TURN_MIN_S to KEEP_TURN_MAX_S; or the keep_turn given (0: not kept).
        close(next_turn=True) takes it for the length of the request's last stream (prompt and ids); a claim keeps a stream at least
        that long."""
        self.assertEqual(S.read_seconds(0), S.READ_S[0][1])
        for n, sec in S.READ_S:
            self.assertAlmostEqual(S.read_seconds(n), sec)
        (n0, s0), (n1, s1) = S.READ_S[3], S.READ_S[4]
        self.assertAlmostEqual(S.read_seconds((n0 + n1) // 2), (s0 + s1) / 2)
        (n0, s0), (n1, s1) = S.READ_S[-2], S.READ_S[-1]
        self.assertAlmostEqual(S.read_seconds(2 * n1 - n0), 2 * s1 - s0)      # past the last point: the last slope
        self.assertTrue(all(S.read_seconds(n) < S.read_seconds(n + 1000) for n in range(0, 300000, 977)))
        s = self.sched(2)
        self.assertEqual(s.turn_keep_s(10), S.KEEP_TURN_MIN_S)
        n_mid = next(n for n in range(0, 300000, 256) if S.read_seconds(n) > 2 * S.KEEP_TURN_MIN_S)
        self.assertAlmostEqual(s.turn_keep_s(n_mid), S.read_seconds(n_mid))
        self.assertEqual(s.turn_keep_s(10 ** 7), S.KEEP_TURN_MAX_S)
        self.assertEqual([self.sched(2, keep_turn=k).turn_keep_s(n_mid) for k in (0, 4.5)], [0.0, 4.5])
        with self.assertRaises(ValueError):
            self.sched(2, keep_turn=-1)
        with self.assertRaises(ValueError):
            self.sched(2, keep_turn=S.KEEP_TURN_LIMIT_S + 1)
        se = ScheduledEngine(s)
        se.close(next_turn=True)                                               # nothing generated: nothing kept
        self.assertEqual(se._keep_s, 0.0)
        se = ScheduledEngine(s)
        se.set_sampling(GREEDY)
        r = se.generate([5, 6, 7], 12)
        with mock.patch.object(S, "READ_S", ((0, 0.0), (10, 2.5))):
            se.close(next_turn=True)
        self.assertAlmostEqual(se._keep_s, 2.5 * (3 + len(r.ids)) / 10)
        se = ScheduledEngine(s)
        se.close(keep_s=1.5, next_turn=True)                                   # given: as given
        self.assertEqual(se._keep_s, 1.5)

        class Owner:
            _open = False
        o = Owner()
        h = S._Held([1, 2, 3], 4, o, tid=7)
        o._taken_tid, o._t_taken, o._t_closed, o._keep_s = 7, h.t_end + 0.1, h.t_end + 0.2, S.RESERVE_MAX_S + 20.0
        h.claim = S.Ticket(S.Stage([1, 2, 3, 4, 5], 5), None)
        self.assertTrue(S._kept(h, o._t_closed + S.RESERVE_MAX_S + 10.0))     # a claim keeps it at least as long as keep_s
        self.assertFalse(S._kept(h, o._t_closed + S.RESERVE_MAX_S + 20.1))

    def test_a_long_conversation_is_kept_longer_for_its_next_turn(self):
        """Two conversations end their first turns in the two slots with tool calls (close(next_turn=True)); the read is made slow here,
        so the short one is kept KEEP_TURN_MIN_S and the long one about the time reading it again takes.  Two new requests wait: the
        first takes the short one's slot when its keep runs out; the second still waits when the long conversation's next turn comes
        after that, and the turn continues its stream (the ids and prompt-reuse mode of the single path)."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(1000)
        trace = []
        with mock.patch.object(S, "READ_S", ((0, 0.0), (1000, 1.0))), mock.patch.object(S, "KEEP_TURN_MIN_S", 0.3):
            s = self.sched(2, back_to_one=False, trace=trace)
            ps, pl = [11, 12, 13], [100 + (i * 7) % 3000 for i in range(1500)]
            se_s, se_l = ScheduledEngine(s), ScheduledEngine(s)
            for se in (se_s, se_l):
                se.set_sampling(GREEDY)
            r1 = self.first_stages_together(s, [(se_s, ps, 5), (se_l, pl, 5)])
            slot_s, slot_l = (int(r.stats["slot"]) for r in r1)
            t0 = time.monotonic()
            se_s.close(next_turn=True)
            se_l.close(next_turn=True)
            self.assertEqual(se_s._keep_s, 0.3)
            self.assertAlmostEqual(se_l._keep_s, (len(pl) + len(r1[1].ids)) / 1000.0)
            tb = [s.submit(Stage([90 + i, 91], 3000, EOS_IDS, MIXED[1 + i])) for i in range(2)]
            wait_for(lambda: any(t[0] == "read" and t[1] == tb[0].id for t in trace), what="the first new request to be read")
            self.assertGreater(time.monotonic() - t0, 0.25)                    # after the short keep
            time.sleep(max(0.0, 0.8 - (time.monotonic() - t0)))
            self.assertEqual(s.status()["queued"], 1)                           # the long conversation's slot is still kept
            p2 = pl + r1[1].ids + [60, 61]
            tl = s.submit(Stage(p2, 5, EOS_IDS, GREEDY))                        # its next turn, after the short keep
            rl = drain(tl)
            for t in tb:
                t.cancel()
            rb = [drain(t) for t in tb]
        self.stub.delay_us(0)
        self.ref.reset()
        self.ref.set_sampling(GREEDY)
        w1 = self.ref.generate(pl, 5)
        w2 = self.ref.generate(p2, 5)
        self.assertEqual((r1[1].ids, rl.ids, rl.finish), (w1.ids, w2.ids, w2.finish))
        self.assertEqual((rl.snap_mode, rl.cached_tokens), (1, w2.cached_tokens))
        self.assertEqual([t[2] for t in trace if t[0] == "read" and t[1] == tl.id], [slot_l])
        for i, r in enumerate(rb):
            self.assertEqual(r.ids, self.single([90 + i, 91], 3000, MIXED[1 + i]).ids[:len(r.ids)])

    # ---- a new request's wait for the places kept for next turns ----------------------------------------------------------------
    def next_turn_at_once(self, s, se_prev, p, max_new):
        """An agent's next turn sent at the moment its answer closes: the close (with next_turn) and the submit in one step under the
        scheduler's lock, so the scheduler never sees the kept place without its claim.  Returns (its ScheduledEngine, result)."""
        se = ScheduledEngine(s)
        se.set_sampling(GREEDY)
        with s._cond:
            se_prev.close(next_turn=True)
            t = se._submit(Stage(list(p), max_new, EOS_IDS, se.sampling, se))
        return se, se._wait(t, None, se.plan(len(p), max_new) + 1)

    def agents_and_a_new_request(self, turns, max_wait, gap=0.02, max_new=30, at_once=False, more=()):
        """Two agents on two slots: each sends ``turns`` turns of ``max_new`` ids that end with tool calls (close with next_turn) and
        its next turn ``gap`` after each answer (``at_once``: at the moment the answer closes), so their places are always kept,
        claimed or running.  A new request comes at 0.3 s (and more at the times ``more``).  Returns the new request's (result, submit
        and end time), the agents' calls [(prompt, result, end time)], when each agent was done, the scheduler, and the results of the
        other new requests [(result, submit time)]."""
        self.model(eos_every=0)
        self.stub.delay_us(1000)
        s = self.sched(2, max_wait=max_wait)
        t0 = time.perf_counter()
        calls, done, errors = [], {}, []

        def agent(i):
            p = [100 + i, 7, 8, 9]
            se = None
            try:
                for j in range(turns):
                    if at_once and se is not None:
                        se, r = self.next_turn_at_once(s, se, p, max_new)
                    else:
                        se = ScheduledEngine(s)
                        se.set_sampling(GREEDY)
                        r = se.generate(p, max_new)
                    calls.append((list(p), r, time.perf_counter() - t0))
                    if not at_once or j == turns - 1:
                        se.close(next_turn=j < turns - 1)
                        time.sleep(gap)
                    p = p + r.ids + [60 + j % 30, 61]
            except Exception as ex:                                     # noqa: BLE001
                errors.append(ex)
            done[i] = time.perf_counter() - t0
        extra = []

        def newcomer(k, at):
            time.sleep(max(0.0, at - (time.perf_counter() - t0)))
            se = ScheduledEngine(s)
            se.set_sampling(MIXED[2 + k % 3])
            t_s = time.perf_counter() - t0
            extra.append((k, se.generate([95 + k, 91, 92], 30), t_s))
            se.close()
        ths = [threading.Thread(target=agent, args=(i,), daemon=True) for i in range(2)]
        ths += [threading.Thread(target=newcomer, args=(k, at), daemon=True) for k, at in enumerate(more)]
        for t in ths:
            t.start()
        time.sleep(0.3)
        se = ScheduledEngine(s)
        se.set_sampling(MIXED[1])
        t_sub = time.perf_counter() - t0
        r_new = se.generate([90, 91, 92], 30)
        t_end = time.perf_counter() - t0
        se.close()
        for t in ths:
            t.join(120)
            self.assertFalse(t.is_alive(), "an agent did not finish")
        self.stub.delay_us(0)
        self.assertEqual(errors, [])
        self.assertEqual(r_new.ids, self.single([90, 91, 92], 30, MIXED[1]).ids)
        for k, r, _ in extra:
            self.assertEqual(r.ids, self.single([95 + k, 91, 92], 30, MIXED[2 + k % 3]).ids)
        for p, r, _ in calls:                                                  # every turn: the ids of the single path
            self.assertEqual((r.ids, r.finish), tuple(getattr(self.single(p, max_new, GREEDY), k) for k in ("ids", "finish")))
        for r in [r_new] + [r for _, r, _ in extra]:                           # the kept wait is a part of the wait
            self.assertLessEqual(r.stats["kept_wait_ms"], r.stats["queue_ms"] + 1.0)
        return (r_new, t_sub, t_end), calls, done, s, [(r, t) for _, r, t in sorted(extra, key=lambda x: x[0])]

    def test_a_new_request_waits_for_kept_places_at_most_max_wait(self):
        """The case of the review: two agents on two slots, 30 and 120 turns, 20 ms from each answer to the next turn.  Without a bound
        (max_wait 600 s, as before this change) a new request that came at 0.3 s waited until an agent was done (on the stub 705 ms
        for 30 turns, 3,711 ms for 120).  With max_wait 0.3 s it takes a place kept for a next turn once it has waited that long while
        only kept places stood in its way, long before the agents are done, whatever their number of turns; the next turn of that
        agent is then read from the start.  Auto: the bound is the time reading the kept conversation again would take and
        WAIT_FLOOR_S.  Every request gets the ids of the single path."""
        (r, t_sub, t_end), calls, done, s, _ = self.agents_and_a_new_request(30, 600.0)
        first_last = min(t for p, x, t in calls if len(p) == max(len(q) for q, _, _ in calls))   # an agent's last turn ended ...
        self.assertGreater(t_sub + r.stats["queue_ms"] / 1000.0, min(done.values()) - 0.1)       # ... before the new request was read
        self.assertEqual(s.perf["takeovers"], 0)
        self.assertGreater(r.stats["queue_ms"], 300.0)
        self.assertLess(first_last, t_end)
        for turns, max_wait in ((30, 0.3), (120, 0.3), (120, None)):
            with self.subTest(turns=turns, max_wait=max_wait):
                (r, t_sub, t_end), calls, done, s, _ = self.agents_and_a_new_request(turns, max_wait)
                ends_before = [len(p) + len(x.ids) for p, x, t in calls if t <= t_end]
                bound = max_wait if max_wait is not None else s.turn_wait_s(max(ends_before))
                self.assertLessEqual(r.stats["kept_wait_ms"], 1000.0 * (bound + 0.06))
                self.assertLess(r.stats["queue_ms"], 1000.0 * (3 * bound + 0.3))
                self.assertLess(t_end, min(done.values()))                    # read while the agents still ran
                self.assertGreaterEqual(s.perf["takeovers"], 1)
                lost = [x for p, x, t in calls if len(p) > 4 and x.snap_mode != 1]   # the turn whose place was taken: from the start
                self.assertGreaterEqual(len(lost), 1)

    def test_the_wait_bound_counts_next_turns_that_run_before_a_new_request(self):
        """The bound holds in time, not only in the scheduler's count, when the agents' turns are long.  Two agents on two slots send
        30 turns of 300 ids each (about 0.1 s on the stub), the next turn 20 ms after each answer; max_wait 0.3 s.  The places are
        kept only for those 20 ms of each turn, and the next turns that came after the new request run before it the rest of the time.
        Before the kept-wait clock counted those runs, the new request waited 1.16 s (its kept wait 0.30 s, the time of the keeps
        alone); now it is read about when its bound is up (the next turn running then ends first).  Also with next turns sent at the
        moment each answer closes (no time with a place kept and unclaimed): before, the new request waited until an agent was done;
        it must also come before the next turns that came after it once its bound is up.  A second new request that comes later
        counts its own wait only (its kept wait is never more than its wait).  Every request gets the ids of the single path."""
        self.e.close()                                                          # conversations of about 9,000 tokens
        self.ref.close()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=16384)
        self.ref = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=16384, name="ref")
        for at_once in (False, True):
            with self.subTest(at_once=at_once):
                (r, t_sub, t_end), calls, done, s, more = self.agents_and_a_new_request(30, 0.3, max_new=300, at_once=at_once,
                                                                                        more=(1.2,))
                self.assertLess(r.stats["queue_ms"], 1000.0 * (0.3 + 0.3))
                self.assertGreaterEqual(r.stats["kept_wait_ms"], 300.0)
                self.assertLess(t_end, min(done.values()))                    # read while the agents still ran
                (r2, t2), = more
                self.assertLess(t2 + r2.stats["queue_ms"] / 1000.0, min(done.values()))
                self.assertLess(r2.stats["queue_ms"], 1000.0 * (0.3 + 0.3))
                self.assertGreaterEqual(s.perf["takeovers"], 2)
                lost = [x for p, x, t in calls if len(p) > 4 and x.snap_mode != 1]
                self.assertGreaterEqual(len(lost), 2)                        # the turns whose places were taken: from the start

    def test_a_new_request_that_waited_does_not_drop_a_stream_an_open_request_keeps(self):
        """Slot b is kept for b's next turn, slot c is idle (c closed without tool calls), and the carrier holds r's stream: r ran
        alone and is open (its next stage comes 0.6 s later, within RESERVE_S).  A new request waits 0.2 s (max_wait) and is then
        read: it runs in slot c (in a slot, not alone on the carrier), and r's stream goes into b's place, given up for it (a
        takeover).  Before, the new request's plan counted b's place for r's stream but the move into a slot did not see it: r's
        stream was dropped (nothing counted), against the rule that the streams open requests keep for their next stages are never
        taken.  r's next stage continues its stream (snap_mode 1, the cached count of the single path)."""
        self.model(eos_every=0)
        trace = []
        with mock.patch.object(S, "RESERVE_S", 5.0):
            s = self.sched(2, back_to_one=False, keep_turn=30.0, max_wait=0.2, trace=trace)
            self.stub.delay_us(2000)
            ses = [ScheduledEngine(s) for _ in range(2)]
            for se in ses:
                se.set_sampling(GREEDY)
            self.first_stages_together(s, [(ses[0], [11, 12, 13], 40), (ses[1], [21, 22, 23], 40)])
            ses[0].close(next_turn=True)                                        # b: tool calls, its slot kept 30 s
            ses[1].close()                                                      # c: a plain answer, its slot idle and not kept
            self.stub.delay_us(0)
            se_r = ScheduledEngine(s)
            se_r.set_sampling(GREEDY)
            pr = [31, 32, 33]
            r1 = se_r.generate(pr, 5)                                           # alone on the carrier (c's slot is left for it)
            self.assertEqual(r1.stats["slot"], -1.0)                           # r ended there: the carrier holds its stream
            tn = s.submit(Stage([41, 42, 43], 5, EOS_IDS, MIXED[1]))
            time.sleep(0.6)
            r2 = se_r.generate(pr + r1.ids + [50, 51], 5)
            rn = drain(tn)
            se_r.close()
        self.ref.reset()
        self.ref.set_sampling(GREEDY)
        w1 = self.ref.generate(pr, 5)
        w2 = self.ref.generate(pr + w1.ids + [50, 51], 5)
        self.assertEqual((r1.ids, r2.ids), (w1.ids, w2.ids))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (1, w2.cached_tokens))
        self.assertEqual(rn.ids, self.single([41, 42, 43], 5, MIXED[1]).ids)
        reads = [t for t in trace if t[0] == "read" and t[1] == tn.id]
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0][4], False)                                   # in a slot: not alone on the carrier
        self.assertGreaterEqual(reads[0][5], 0)
        self.assertEqual(s.perf["takeovers"], 1)
        self.assertGreaterEqual(rn.stats["kept_wait_ms"], 200.0)

    def test_next_turns_that_need_each_others_places_do_not_wait_for_their_keeps(self):
        """Both slots are kept for next turns (a, b) and the carrier holds r's stream, kept for r's next turn (r's later stage ran
        alone: no slot was free).  (1) The next turns of a, b and r come at once and claim their streams: those of a and b each need
        a place for the carrier's stream first, that of r a slot to run in, so before this nothing ran until the claimed keeps ran
        out (RESERVE_MAX_S: 30 s, 3 s here).  Now r's next turn runs alone on the carrier as a later stage does, then the other two
        follow, all three continuing their streams.  (2) Only the next turns of a and b come: nothing runs, and r's stream (kept,
        unclaimed) stands in their way, so it is given up (a takeover) and both are read at once; r's next turn comes later and is
        read from the start.  Every request gets the ids of the single path."""
        self.model(eos_every=0)
        for case in ("all", "two"):
            with self.subTest(case=case), mock.patch.object(S, "RESERVE_MAX_S", 3.0):
                trace = []
                s = self.sched(2, back_to_one=False, keep_turn=2.5, max_wait=600.0, trace=trace)
                self.stub.delay_us(2000)
                pa, pb, pr = [11, 12, 13], [21, 22, 23], [31, 32, 33]
                ses = [ScheduledEngine(s) for _ in range(2)]
                for se in ses:
                    se.set_sampling(GREEDY)
                ra, rb = self.first_stages_together(s, [(ses[0], pa, 40), (ses[1], pb, 40)])
                for se in ses:
                    se.close(next_turn=True)                                     # both slots kept for the next turns
                self.stub.delay_us(0)
                se_r = ScheduledEngine(s)
                se_r.set_sampling(GREEDY)
                tr = s.submit(Stage(pr, 5, EOS_IDS, se_r.sampling, se_r, later=True))   # its stream is gone: it runs alone
                rr = drain(tr)
                self.assertEqual(rr.stats["slot"], -1.0)
                se_r._t_taken, se_r._taken_tid, se_r._last_n = time.perf_counter(), tr.id, len(pr) + len(rr.ids)
                se_r.close(next_turn=True)                                       # the carrier's stream kept for r's next turn
                nxt = {"a": pa + ra.ids + [60, 61], "b": pb + rb.ids + [60, 61], "r": pr + rr.ids + [60, 61]}
                t0 = time.perf_counter()
                with s._cond:
                    tk = {k: s.submit(Stage(p, 5, EOS_IDS, GREEDY)) for k, p in nxt.items() if case == "all" or k != "r"}
                out = {k: drain(t) for k, t in tk.items()}
                took = time.perf_counter() - t0
                if case == "two":
                    out["r"] = drain(s.submit(Stage(nxt["r"], 5, EOS_IDS, GREEDY)))
                self.assertLess(took, 1.5)
                for k, first, n1 in (("a", pa, 40), ("b", pb, 40), ("r", pr, 5)):
                    self.ref.reset()
                    self.ref.set_sampling(GREEDY)
                    self.ref.generate(first, n1)
                    w2 = self.ref.generate(nxt[k], 5)
                    self.assertEqual(out[k].ids, w2.ids, k)
                    if case == "all" or k != "r":
                        self.assertEqual((out[k].snap_mode, out[k].cached_tokens), (1, w2.cached_tokens), k)
                if case == "all":
                    self.assertEqual(s.perf["takeovers"], 0)
                    self.assertNotIn("unstick", [t[0] for t in trace])
                else:
                    self.assertEqual(out["r"].snap_mode, 0)
                    self.assertEqual(s.perf["takeovers"], 1)
                    self.assertIn("unstick", [t[0] for t in trace])
                s.close()

    def test_wait_rules(self):
        """turn_wait_s: max_wait, or the read of the kept conversation (at most KEEP_TURN_MAX_S) and WAIT_FLOOR_S.  A view for a new
        request that has waited ``age`` gives up a stream kept for a next turn (a closed request's) once ``age`` reaches its wait, and
        never a stream an open request keeps for its next stage."""
        s = self.sched(2)
        self.assertAlmostEqual(s.turn_wait_s(5000), S.WAIT_FLOOR_S + S.read_seconds(5000))
        self.assertEqual(s.turn_wait_s(10 ** 7), S.WAIT_FLOOR_S + S.KEEP_TURN_MAX_S)
        self.assertEqual(self.sched(2, max_wait=1.5).turn_wait_s(5000), 1.5)
        with self.assertRaises(ValueError):
            self.sched(2, max_wait=-0.1)

        class Owner:
            _open, _pending, _t_sub, _keep_s = False, False, 0.0, 600.0
        closed, opened = Owner(), Owner()
        opened._open = True
        now = time.perf_counter()
        for o in (closed, opened):
            o._taken_tid, o._t_taken, o._t_closed = 7, now - 0.1, now - 0.05
        hc, ho = S._Held(list(range(500)), 4, closed, tid=7), S._Held([1, 2], 3, opened, tid=7)
        v = S._View()
        self.assertEqual((v.kept(hc), v.kept(ho)), (True, True))
        w = v.aged(1.0, lambda h: 2.0)
        self.assertEqual((w.kept(hc), w.kept(ho), w.held_kept(hc)), (True, True, True))
        w = v.aged(2.0, lambda h: 2.0)
        self.assertEqual((w.kept(hc), w.kept(ho), w.held_kept(hc)), (False, True, True))
        self.assertEqual((v.kept(hc), w.now), (True, v.now))                  # the first view is unchanged, the clock the same
        self.assertFalse(v.aged(math.inf, lambda h: 1e9).kept(hc))

    def test_alone_does_not_move_into_a_kept_slot(self):
        """A request that runs alone on the carrier while both slots are kept (a later stage whose stream was given away: it may run
        alone when nothing else runs) does not move into a kept slot when another request comes: it goes on, and the new request
        waits.  When one request closes, its slot is no longer kept: the stream moves into it.  The other request's next stage then
        still continues its stream in its slot, read as alone."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(2000)                                                 # e runs for seconds
        with mock.patch.object(S, "RESERVE_S", KEEP_LONG):
            trace = []
            s = self.sched(2, back_to_one=False, trace=trace)
            reqs = [([11, 12, 13], 8, GREEDY), ([21, 22], 6, MIXED[1])]
            ses = [ScheduledEngine(s) for _ in reqs]
            for se, (_, _, smp) in zip(ses, reqs):
                se.set_sampling(smp)
            r1 = self.first_stages_together(s, [(se, p, n) for se, (p, n, _) in zip(ses, reqs)])
            slot_a, slot_b = [int(r.stats["slot"]) for r in r1]
            pe, pc = [71, 72, 73], [81, 82]
            te = s.submit(Stage(pe, 3000, EOS_IDS, GREEDY, later=True))
            self.assertEqual(te.q.get(timeout=30)[0], "ids")                    # e runs alone on the carrier
            tc = s.submit(Stage(pc, 20, EOS_IDS, MIXED[2]))
            time.sleep(0.3)
            self.assertEqual(s.status()["queued"], 1)                           # c waits
            self.assertEqual([t for t in trace if t[0] in ("move", "park")], [])  # e stays alone
            ses[0].close()                                                       # a's slot is no longer kept: e moves into it
            wait_for(lambda: [t[:3] for t in trace if t[0] == "move"] == [("move", te.id, slot_a)], what="e to move into a's slot")
            time.sleep(0.2)
            self.assertEqual(s.status()["queued"], 1)                           # b's slot is still kept: c still waits
            pb, nb, smp_b = reqs[1]
            r2 = ses[1].generate(pb + r1[1].ids, 10)                            # b's next stage, in its slot next to e
            ses[1].close()
            rc = drain(tc)
            te.cancel()
            re_ = drain(te)
            self.stub.delay_us(0)
            a, b = self.two_stage_reference(pb, nb, 10, smp_b)
            self.assertEqual((r1[1].ids, r2.ids), (a.ids, b.ids))
            self.assertEqual((r2.snap_mode, r2.cached_tokens), (b.snap_mode, b.cached_tokens))
            self.assertEqual(b.snap_mode, 1)
            self.assertEqual([t[2] for t in trace if t[0] == "read" and t[1] == tc.id], [-1])
            self.assertEqual(rc.ids, self.single(pc, 20, MIXED[2]).ids)
            self.assertIn(re_.finish, ("cancelled", "length"))
            self.assertEqual(re_.ids, self.single(pe, 3000, GREEDY).ids[:len(re_.ids)])
            self.assertGreaterEqual(re_.stats["moves"], 1)
            self.assertNotEqual(slot_a, slot_b)

    def test_a_waiting_later_stage_comes_before_new_requests(self):
        """A request's next stage whose stream was given away and a new request both wait for a place.  The place comes free between
        the scheduler's look at the next stage and its look at the new request (the window is widened here: right after the first
        look, the reservation of the place ends, or its request closes).  The next stage is still read first: one round decides with
        one reading of the clock and of the kept streams, and reads no new request while a later stage waits."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 2)                                         # one id a pass: x runs through the test
        self.stub.delay_us(1000)
        keep = 1.0
        for how in ("expires", "closes"):
            with self.subTest(how=how), mock.patch.object(S, "RESERVE_S", keep):
                trace = []
                s = self.sched(2, trace=trace)
                ses = {k: ScheduledEngine(s) for k in "xabd"}
                for se in ses.values():
                    se.set_sampling(GREEDY)
                x, a, b, d = (ses[k] for k in "xabd")
                out, ths = {}, {}

                def run(key, prompt, n):
                    try:
                        out[key] = ses[key].generate(prompt, n)
                    except Exception as ex:                                     # noqa: BLE001
                        out[key] = ex

                def start(key, prompt, n):
                    ths[key] = threading.Thread(target=run, args=(key, prompt, n), daemon=True)
                    ths[key].start()

                start("x", [1, 2, 3], 100000)
                wait_for(lambda: s.status()["slots_busy"] == 1, what="x to run")
                p = [40, 41, 42]
                r1 = a.generate(p, 5)                                           # in a slot next to x; a keeps it
                start("b", [50, 51], 100000)                                    # takes a's slot once a's reservation ends
                wait_for(lambda: s.status()["slots_busy"] == 2, timeout=keep + 20, what="b to take a's slot")
                start("d", [60, 61], 8)                                         # a new request: no place, it waits
                wait_for(lambda: s.status()["queued"] == 1 and d._ticket is not None, what="d to wait")
                start("a", p + r1.ids, 6)                                       # a's next stage: its stream is gone, it waits
                wait_for(lambda: s.status()["queued"] == 2 and a._ticket is not None, what="a's next stage to wait")
                a_id, d_id = a._ticket.id, d._ticket.id
                fired = []
                fits = Scheduler._fits

                def widened(sched, st, *rest):
                    ok = fits(sched, st, *rest)
                    if not ok and st.affinity is a and not b._pending and not fired:   # b has taken its end; b keeps its slot
                        fired.append(how)
                        if how == "expires":
                            until = b._t_taken + keep + 0.05
                            while time.perf_counter() < until:
                                time.sleep(0.005)
                        else:
                            b.close()
                    return ok
                with mock.patch.object(Scheduler, "_fits", widened):
                    b.cancel()                                                  # b's slot comes free (after its reservation)
                    ths["a"].join(30)
                self.assertEqual(fired, [how], "the window between the two looks was not reached")
                a.close()
                ths["d"].join(30)
                x.cancel()
                for key in "xb":
                    ths[key].join(30)
                for se in ses.values():
                    se.close()
                s.close()
                for key in "xabd":
                    self.assertNotIsInstance(out.get(key), Exception, key)
                reads = [t[1] for t in trace if t[0] == "read"]
                self.assertLess(reads.index(a_id), reads.index(d_id))
                w1, w2 = self.two_stage_reference(p, 5, 6, GREEDY)
                self.assertEqual((r1.ids, out["a"].ids, out["a"].finish), (w1.ids, w2.ids, w2.finish))
                self.assertEqual(out["d"].ids, self.single([60, 61], 8, GREEDY).ids)

    def test_a_later_stage_runs_alone_while_a_new_request_waits(self):
        """Every slot is kept and nothing runs; a new request and a request's next stage whose stream was given away come in one round.
        The new request finds no place and waits, but does not hold the next stage back: that one runs alone on the carrier, as
        before.  When the slots are given up, the new request is read."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        with mock.patch.object(S, "RESERVE_S", KEEP_LONG):
            trace = []
            s = self.sched(2, back_to_one=False, trace=trace)
            reqs = [([11, 12, 13], 8, GREEDY), ([21, 22], 6, MIXED[1])]
            ses = [ScheduledEngine(s) for _ in reqs]
            for se, (_, _, smp) in zip(ses, reqs):
                se.set_sampling(smp)
            self.first_stages_together(s, [(se, p, n) for se, (p, n, _) in zip(ses, reqs)])
            pn, pe = [31, 32, 33], [71, 72, 73]
            with s._cond:
                tn = s.submit(Stage(pn, 20, EOS_IDS, MIXED[2]))
                te = s.submit(Stage(pe, 30, EOS_IDS, GREEDY, later=True))
            re_ = drain(te)
            self.assertEqual(s.status()["queued"], 1)                           # the new request still waits
            for se in ses:
                se.close()
            rn = drain(tn)
            self.assertEqual((re_.ids, re_.finish), tuple(getattr(self.single(pe, 30, GREEDY), k) for k in ("ids", "finish")))
            self.assertEqual(rn.ids, self.single(pn, 20, MIXED[2]).ids)
            reads = [t for t in trace if t[0] == "read"]
            self.assertEqual([t[1] for t in reads][-2:], [te.id, tn.id])
            self.assertTrue(reads[-2][4])                                       # alone on the carrier

    def test_one_view_per_decision(self):
        """sched._View: one decision of the scheduler reads the clock once, and whether a request is open and a stream is kept once
        (at the first look); the request may change meanwhile."""
        class Owner:
            _open, _pending, _t_sub, _t_taken, _taken_tid = True, False, 0.0, 0.0, 5

        o = Owner()
        h = S._Held([1, 2], 3, o, tid=5)
        o._t_taken = time.perf_counter()
        v = S._View()
        self.assertEqual((v.is_open(o), v.kept(h)), (True, True))
        o._t_taken -= 2 * S.RESERVE_MAX_S                                        # its reservation ends ...
        o._open = False                                                          # ... and it closes
        self.assertEqual((v.is_open(o), v.kept(h)), (True, True))
        w = S._View()
        self.assertEqual((w.is_open(o), w.kept(h)), (False, False))
        self.assertGreaterEqual(w.now, v.now)
        self.assertEqual((v.is_open(None), v.kept(None)), (False, False))

    def test_kept_rules(self):
        """When a stream counts as kept (sched._kept): its request open; not taken yet by the request's thread (at most RESERVE_MAX_S
        after the end); RESERVE_S after the take; a next stage sent within that time keeps it until RESERVE_MAX_S after the take; a
        next stage sent later, a later stage that ended without it, or a take after RESERVE_MAX_S give it up for good."""
        class Owner:
            _open, _pending, _t_sub, _t_taken, _taken_tid = True, False, 0.0, 0.0, 0

        o = Owner()
        h = S._Held([1, 2], 3, o, tid=5)
        h.t_end = t = 1000.0
        R, M = S.RESERVE_S, S.RESERVE_MAX_S
        o._pending, o._taken_tid = True, 4                                       # the thread still waits on the stage
        self.assertTrue(S._kept(h, t + M - 0.1))
        self.assertFalse(S._kept(h, t + M + 0.1))
        o._pending, o._taken_tid, o._t_taken = False, 5, t + 0.5                 # taken
        self.assertTrue(S._kept(h, t + 0.5 + R - 0.1))
        self.assertFalse(S._kept(h, t + 0.5 + R + 0.1))
        o._pending, o._t_sub = True, t + 0.5 + R - 0.2                           # the next stage came in time and waits
        self.assertTrue(S._kept(h, t + 0.5 + M - 0.1))
        self.assertFalse(S._kept(h, t + 0.5 + M + 0.1))
        o._t_sub = t + 0.5 + R + 0.2                                             # ... came late
        self.assertFalse(S._kept(h, t + 0.5 + R + 0.3))
        o._pending, o._taken_tid, o._t_taken = False, 6, t + 0.5 + R + 0.4      # a later stage ended (cancelled while it waited)
        self.assertFalse(S._kept(h, t + 0.5 + R + 0.5))
        o._taken_tid, o._t_taken = 5, t + M + 1.0                               # taken after RESERVE_MAX_S
        self.assertFalse(S._kept(h, t + M + 1.5))
        o._t_taken = t + 0.5
        o._open = False
        self.assertFalse(S._kept(h, t + 0.6))
        self.assertFalse(S._kept(None, t))

    # ---- limits, failures, images -------------------------------------------------------------------------------------------------
    def test_queue_full(self):
        self.model(eos_every=0)
        self.stub.delay_us(2000)
        s = self.sched(2, max_queue=2)
        ts = [s.submit(Stage([1, 2, 3 + i], 400, EOS_IDS, GREEDY)) for i in range(2)]
        t0 = time.time()
        while s.status()["slots_busy"] < 2 and time.time() - t0 < 10:
            time.sleep(0.005)
        ts += [s.submit(Stage([4, 5, 6 + i], 10, EOS_IDS, GREEDY)) for i in range(2)]
        with self.assertRaises(QueueFull) as cm:
            s.submit(Stage([7, 8], 10, EOS_IDS, GREEDY))
        self.assertEqual(cm.exception.retry_after, 2)
        self.assertEqual(s.status()["queued"], 2)
        for t in ts:
            t.cancel()
        for t in ts:
            kind, r = t.q.get(timeout=30)
            while kind == "ids":
                kind, r = t.q.get(timeout=30)
            self.assertEqual((kind, r.finish), ("done", "cancelled"))

    def test_engine_failure_ends_everything(self):
        self.model(eos_every=0)
        self.stub.delay_us(300)
        s = self.sched(3)
        self.stub.ms_fail_in(self.e, 6)
        jobs = self.run_jobs(s, [Job([1, 2], 200, GREEDY), Job([3, 4], 200, MIXED[1], after=(0, 2)),
                                 Job([5, 6], 200, MIXED[2], after=(0, 2))])
        for j in jobs:
            self.assertIsInstance(j.error, YasakaniError)
            self.assertIn("engine failed", str(j.error))
        st = s.status()
        self.assertEqual(st["state"], "error")
        self.assertIn("injected failure", st["error"])
        with self.assertRaisesRegex(YasakaniError, "failed earlier"):
            s.submit(Stage([1], 5))

    def test_refused_settings(self):
        pack = support.make_pack(Path(self.tmp.name), "kv8")
        from unittest import mock
        import os
        from yasakani.engine import Engine
        from yasakani.runtime import RuntimeProfile
        with mock.patch.dict(os.environ):
            e = Engine(pack, max_ctx=4096, lib=self.stub.lib, profile=RuntimeProfile.default(pack, {"YSKN_TB_CAP": "0"}))
            try:
                with self.assertRaisesRegex(YasakaniError, "8 bits"):
                    Scheduler(e, 2)
            finally:
                e.close()

    def test_images_run_alone(self):
        """A request with image input has the carrier to itself (no pass runs for the others meanwhile), for all of its stages; then
        the image input is cleared and the others go on.  Everyone gets their ids."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        for e in (self.e, self.ref):
            self.stub.give_visual(e)
        self.stub.delay_us(300)
        trace = []
        s = self.sched(3, trace=trace)
        bg = [Job([70 + i, 71, 72], 150, MIXED[i]) for i in range(3)]
        n_prompt, n_img = 12, 4
        vis = dict(features=np.ones((n_img, 5120), np.uint16), slots=np.arange(2, 2 + n_img, dtype=np.int32),
                   positions=np.tile(np.arange(n_prompt, dtype=np.int32), (3, 1)), delta=0)
        p = list(range(100, 100 + n_prompt))
        out = {}

        def image_request():
            with bg[0].progress:
                bg[0].progress.wait_for(lambda: bg[0].seen >= 10, 30)
            with ScheduledEngine(s) as se:
                se.set_sampling(MIXED[1])
                with se.visual_input(**vis):
                    r1 = se.generate(p, 10)
                    r2 = se.generate(p + r1.ids, 20)
                out["r"] = (r1, r2)
        th = threading.Thread(target=image_request, daemon=True)
        th.start()
        self.run_jobs(s, bg)
        th.join(30)
        self.check_same(bg)
        r1, r2 = out["r"]
        self.ref.reset()
        self.ref.set_sampling(MIXED[1])
        with self.ref.visual_input(**vis):
            a = self.ref.generate(p, 10)
            b = self.ref.generate(p + a.ids, 20)
        self.assertEqual((r1.ids, r2.ids), (a.ids, b.ids))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (b.snap_mode, b.cached_tokens))
        self.assertEqual(self.stub.visual(self.e), (0, 2))                         # set once, cleared once
        kinds = [t[0] for t in trace]
        i0, i1 = kinds.index("visual"), kinds.index("visual_end")
        self.assertNotIn("rows", kinds[i0:i1])                                     # nobody else ran meanwhile
        self.assertIn("rows", kinds[i1:])

    def test_older_library_has_no_parallel(self):
        stub = support.Stub(no_ms=True)
        e = support.open_engine(Path(self.tmp.name), stub, name="old")
        try:
            with self.assertRaisesRegex(YasakaniError, "multi-stream"):
                Scheduler(e, 2)
        finally:
            e.close()

    # ---- the Python cost of a pass ----------------------------------------------------------------------------------------------
    def test_python_cost_per_pass(self):
        """What the scheduler itself spends per pass (the engine calls are not counted); the stub's passes take no time here."""
        self.model(eos_every=0)
        self.stub.draft_mode(self.e, 0)
        self.stub.delay_us(300)                                             # so that the requests overlap
        for n in (4, 8, 16):
            s = self.sched(n, back_to_one=False)
            jobs = [Job([100 + i, 2, 3], 300, MIXED[i % 5], after=(0, 2) if i else None) for i in range(n)]
            self.run_jobs(s, jobs)
            self.check_same(jobs)
            p = s.perf
            self.assertGreater(p["passes"], 50)
            py_us = 1e6 * (p["round_s"] - p["engine_s"]) / max(1, p["passes"])
            sys.stderr.write("\n  scheduler cost per pass, %2d streams: %.0f us Python in the round (%.0f us in the pass step; + %.0f us in"
                             " the stub), %d passes, %.1f streams per pass\n"
                             % (n, py_us, 1e6 * (p["pass_s"] - p["engine_s"]) / max(1, p["passes"]),
                                1e6 * p["engine_s"] / max(1, p["passes"]), p["passes"], p["streams"] / max(1, p["passes"])))
            self.assertLess(py_us, 5000)
            s.close()


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ChatOverSchedulerTests(unittest.TestCase):
    """ChatRunner on a ScheduledEngine gives what it gives on the engine itself (text, tool calls, usage), also with a thinking
    budget (a second stage) and a stop string (compared up to the stop)."""

    @classmethod
    def setUpClass(cls):
        from yasakani.textio import ChatFormatter
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.stub.clear_scripts()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        self.ref = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000, name="ref")
        for e in (self.e, self.ref):
            self.stub.model(e, True, eos_every=40, vocab=64)
        self.s = Scheduler(self.e, 3)

    def tearDown(self):
        self.s.close()
        self.e.close()
        self.ref.close()
        self.tmp.cleanup()

    def chat(self, engine, params):
        from yasakani.chat import ChatRunner
        ev = []
        u = ChatRunner(engine, self.fmt, seed0=5).run(params, lambda k, t: ev.append((k, t)))
        return {k: "".join(t for kk, t in ev if kk == k) for k in ("reasoning", "content")}, u

    def test_same_text_and_usage(self):
        from yasakani.chat import ChatParams
        cases = [dict(template_kwargs={"enable_thinking": False}, temperature=0.0, max_tokens=60),
                 dict(template_kwargs={"enable_thinking": True}, max_tokens=80, thinking_budget=12, seed=3),
                 dict(template_kwargs={"enable_thinking": False}, max_tokens=50, seed=9, ignore_eos=True)]
        for kw in cases:
            with self.subTest(**{k: v for k, v in kw.items() if k != "template_kwargs"}):
                msgs = [{"role": "user", "content": "say something"}]
                self.ref.reset()
                want, wu = self.chat(self.ref, ChatParams(messages=msgs, **kw))
                with ScheduledEngine(self.s) as se:
                    got, gu = self.chat(se, ChatParams(messages=msgs, **kw))
                self.assertEqual(got, want)
                for f in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "finish", "tool_calls", "snap_mode"):
                    self.assertEqual(getattr(gu, f), getattr(wu, f), f)
                self.assertEqual(bool(gu.loop_guard), bool(wu.loop_guard))
                if wu.loop_guard:
                    self.assertEqual({k: v for k, v in gu.loop_guard.items()}, wu.loop_guard)

    def test_stop_string(self):
        from yasakani.chat import ChatParams
        msgs = [{"role": "user", "content": "say something"}]
        kw = dict(template_kwargs={"enable_thinking": False}, temperature=0.0, max_tokens=60)
        self.ref.reset()
        full, _ = self.chat(self.ref, ChatParams(messages=msgs, **kw))
        text = full["content"]
        i = next(k for k in range(len(text) // 2, len(text) - 3) if text[k:k + 3].strip() == text[k:k + 3])
        stop = text[i:i + 3]
        self.ref.reset()
        want, wu = self.chat(self.ref, ChatParams(messages=msgs, stop=[stop], **kw))
        with ScheduledEngine(self.s) as se:
            got, gu = self.chat(se, ChatParams(messages=msgs, stop=[stop], **kw))
        self.assertEqual(got["content"], want["content"])                # up to the stop (the ids after it depend on timing)
        self.assertEqual((gu.finish, wu.finish), ("stop", "stop"))


if __name__ == "__main__":
    unittest.main()
