# SPDX-License-Identifier: Apache-2.0
"""The multi-stream entries of Engine (0.1.7) against the stub library: the stub's stand-in model gives every stream the ids it gives
alone, and refuses what the engine library refuses (see the notes at the top of tests/stub/yskn_stub.c)."""
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani._native import MS_SIGNATURES, YasakaniError
from yasakani.engine import CFG_FRONTIER, MS_ROWS, MS_TOP, Engine, Sampling, ms_accept
from yasakani.runtime import EOS_IDS, RuntimeProfile

EOS, EOS2 = EOS_IDS
GREEDY = Sampling(temperature=0.0)
MIXED = [GREEDY, Sampling(1.0, 0.95, 20, 0.0, 11, 3), Sampling(0.7, 0.8, 5, 0.05, 12, 2), Sampling(1.2, 1.0, 20, 0.0, 13, 3),
         Sampling(0.3, 0.9, 10, 0.1, 14, 3)]


class AcceptTests(unittest.TestCase):
    """ms_accept: the count the scheduler gives ms_commit (no library needed)."""

    def test_counts(self):
        f, a, b, c, x, y = 5, 10, 11, 12, 20, 21
        self.assertEqual(ms_accept([f, a, b, c], [a, b, x, y]), (2, [a, b, x]))    # the third draft is wrong: the pass's token after b
        self.assertEqual(ms_accept([f, a], [x, y]), (0, [x]))                        # nothing confirmed: the token after the frontier
        self.assertEqual(ms_accept([f, a, b], [a, b, c]), (2, [a, b, c]))            # all confirmed, plus the token after the last
        self.assertEqual(ms_accept([f], [x]), (0, [x]))                              # one row: no drafts

    def test_a_confirmed_stop_ends_the_stream(self):
        f, a, c, d = 5, 10, 12, 13
        self.assertEqual(ms_accept([f, a, EOS, c], [a, EOS, c, d], EOS_IDS), (1, [a, EOS]))   # not committed, the last id
        self.assertEqual(ms_accept([f, EOS2, c], [EOS2, c, d], EOS_IDS), (0, [EOS2]))
        self.assertEqual(ms_accept([f, a], [EOS, d], EOS_IDS), (0, [EOS]))                     # a stop after the frontier
        self.assertEqual(ms_accept([f, a, EOS, c], [a, EOS, c, d], ()), (3, [a, EOS, c, d]))  # ignore_eos: no cut

    def test_shapes(self):
        with self.assertRaises(ValueError):
            ms_accept([], [])
        with self.assertRaises(ValueError):
            ms_accept([1, 2], [3])


class OlderLibraryTests(unittest.TestCase):
    """A library without the multi-stream entries: everything of 0.1.x works; the new calls say why they cannot."""

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub(no_ms=True)

    def test_no_multi_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = support.open_engine(Path(tmp), self.stub)
            try:
                self.assertFalse(self.stub.lib.supports_parallel)
                self.assertFalse(e.supports_parallel)
                self.assertFalse(any(n in self.stub.lib.fn for n in MS_SIGNATURES))
                for call in (e.ms_info, lambda: e.ms_open(2), lambda: e.ms_load(0), e.live_drop, e.ms_flush,
                             lambda: e.ms_pass([0], [1]), lambda: e.ms_commit([], [])):
                    with self.assertRaisesRegex(YasakaniError, "multi-stream"):
                        call()
                with self.assertRaisesRegex(YasakaniError, "frontier"):
                    e.generate([5, 6], 4, frontier=True)
                self.stub.push([31, 32, EOS])
                self.assertEqual(e.generate([5, 6], 50).ids, [31, 32, EOS])
                self.assertEqual(len(self.stub.last_cfg()), 46)
            finally:
                e.close()


class MsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.e = support.open_engine(Path(self.tmp.name), self.stub)
        self.stub.model(self.e, True, eos_every=25)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()

    # ---- helpers ----------------------------------------------------------------------------------------------------------------
    def solo(self, prompt, max_new, s, stops=EOS_IDS):
        self.e.reset()
        self.e.set_sampling(s)
        return self.e.generate(prompt, max_new, stop_ids=stops if stops else None)

    def read(self, prompt, s, slot, stops=EOS_IDS):
        self.e.set_sampling(s)
        r = self.e.generate(prompt, 1, stop_ids=stops if stops else None)
        self.e.ms_load(slot)
        return list(r.ids)

    def run_ms(self, jobs, n_slots, *, drafts="dev", stops=EOS_IDS, oracle=None, rows=MS_ROWS):
        """jobs = [(prompt, max_new, sampling)], run in n_slots slots (a slot takes the next job when its stream ends).  drafts: "dev" =
        the drafter, "none" = one row per stream, "wrong" = the caller's drafts (all wrong), "oracle" = the caller's drafts from
        oracle[j] (the ids of the job alone).  Returns the ids per job and what the passes did."""
        e = self.e
        if e.ms_slots != n_slots:
            e.ms_open(n_slots)
        info = dict(passes=0, cut_in_drafts=0, accepted=0, joins_mid_run=0)
        ids = [None] * len(jobs)
        pending, free, live = list(range(len(jobs))), list(range(n_slots)), {}

        def done(j):
            return len(ids[j]) >= jobs[j][1] or (bool(stops) and ids[j][-1] in stops)

        while pending or live:
            for slot, j in list(live.items()):
                if done(j):
                    self.assertEqual(e.ms_slot_pos(slot), len(jobs[j][0]) + len(ids[j]) - 1)
                    del live[slot]
                    free.append(slot)
            while pending and free:
                j, slot = pending.pop(0), free.pop(0)
                ids[j] = self.read(jobs[j][0], jobs[j][2], slot, stops)
                info["joins_mid_run"] += 1 if live else 0
                if done(j):
                    self.assertEqual(e.ms_slot_pos(slot), len(jobs[j][0]))
                    free.append(slot)
                else:
                    live[slot] = j
            if not live:
                continue
            act = sorted(live)
            per = min(8, rows // len(act))
            extra = rows - per * len(act) if per < 8 else 0
            slots, toks, dev, groups = [], [], [], []
            for q, slot in enumerate(act):
                j = live[slot]
                want = per + (1 if q < extra else 0) - 1 if drafts != "none" else 0
                want = max(0, min(want, jobs[j][1] - len(ids[j]) - 1))
                if drafts == "oracle":
                    own = list(oracle[j][len(ids[j]):len(ids[j]) + want])
                    own += [0] * (want - len(own))
                else:
                    own = [0] * want
                groups.append((slot, j, len(slots), 1 + want))
                slots += [slot] * (1 + want)
                toks += [ids[j][-1]] + own
                dev += [1] + [0] * want
            r = e.ms_pass(slots, toks, dev if drafts == "dev" else None)
            info["passes"] += 1
            cs, ca = [], []
            for slot, j, r0, n in groups:
                a, new = ms_accept(r.toks[r0:r0 + n], r.argmax[r0:r0 + n], stops)
                if new[-1] in (stops or ()) and len(new) - 1 < n - 1 and r.toks[r0 + len(new)] == new[-1]:
                    info["cut_in_drafts"] += 1
                ids[j] += new
                info["accepted"] += a
                cs.append(slot)
                ca.append(a)
            e.ms_commit(cs, ca)
        return ids, info

    def jobs(self, n, seed, max_new=(1, 2, 3, 17, 40, 64), samplings=MIXED):
        rng = random.Random(seed)
        return [([rng.randrange(1, 5000) for _ in range(rng.randrange(3, 40))], rng.choice(max_new), rng.choice(samplings))
                for _ in range(n)]

    def check_equal(self, jobs, got, stops=EOS_IDS):
        for j, (p, k, s) in enumerate(jobs):
            ref = self.solo(p, k, s, stops)
            self.assertEqual(got[j], ref.ids, "job %d (max_new %d, %s)" % (j, k, s))
            self.assertEqual(ref.finish, "stop" if (stops and ref.ids[-1] in stops) else "length")

    # ---- setup ------------------------------------------------------------------------------------------------------------------
    def test_entries_and_info(self):
        self.assertTrue(self.e.supports_parallel)
        self.assertTrue(all(n in self.stub.lib.fn for n in MS_SIGNATURES))
        i = self.e.ms_info()
        self.assertEqual((i.unsupported, i.reason, i.kv_format, i.kv5_bits), (0, "", 5, 5))
        self.assertEqual((i.max_ctx, i.max_rows, i.max_slots, i.slots_made, i.slots_bound), (4096, 16, 64, 0, 0))
        self.assertTrue(i.drafter and i.draft_head)
        self.assertGreater(i.kv_bytes_per_position, 0)
        self.assertGreater(i.slot_bytes, 0)

    def test_8bit_kv_is_refused(self):
        pack = support.make_pack(Path(self.tmp.name), "kv8")
        with mock.patch.dict(os.environ):                          # the profile is applied to the process environment
            e = Engine(pack, max_ctx=4096, lib=self.stub.lib, profile=RuntimeProfile.default(pack, {"YSKN_TB_CAP": "0"}))
            try:
                i = e.ms_info()
                self.assertEqual((i.unsupported, i.kv_format, i.kv5_bits), (2, 8, 0))
                self.assertIn("8 bits", i.reason)
                with self.assertRaisesRegex(YasakaniError, "8 bits"):
                    e.ms_open(2)
                self.assertEqual(e.ms_slots, 0)
            finally:
                e.close()

    def test_open(self):
        i = self.e.ms_open(3)
        self.assertEqual((i.slots_made, i.slots_bound, self.e.ms_slots, self.e.ms_drafter), (3, 3, 3, True))
        i = self.e.ms_open(2, drafter=False)
        self.assertEqual((i.slots_made, i.slots_bound, self.e.ms_slots, self.e.ms_drafter), (3, 2, 2, False))
        for bad in (0, 65):
            with self.assertRaises(ValueError):
                self.e.ms_open(bad)

    def test_open_out_of_memory(self):
        self.e.ms_open(2)
        with mock.patch.dict(os.environ, {"STUB_MS_BIND_FAIL": "1"}):
            with self.assertRaisesRegex(YasakaniError, "out of memory"):
                self.e.ms_open(4)
        self.assertEqual(self.e.ms_slots, 0)
        with self.assertRaisesRegex(YasakaniError, "ms_open first"):
            self.e.ms_pass([0], [1])

    def test_calls_before_open(self):
        for call in (lambda: self.e.ms_load(0), lambda: self.e.ms_pass([0], [1]), lambda: self.e.ms_commit([], []),
                     lambda: self.e.ms_slot_pos(0), lambda: self.e.ms_set_sampling(0, GREEDY)):
            with self.assertRaisesRegex(YasakaniError, "ms_open first"):
                call()

    # ---- the same ids as alone --------------------------------------------------------------------------------------------------
    def test_streams_match_single_runs(self):
        for mode in (0, 1, 2):                                     # drafts right up to a place / all right / the first one wrong
            with self.subTest(draft_mode=mode):
                self.stub.draft_mode(self.e, mode)
                jobs = self.jobs(10, seed=100 + mode)
                got, info = self.run_ms(jobs, 4)
                self.check_equal(jobs, got)
                self.assertGreater(info["joins_mid_run"], 0)
                if mode == 1:
                    self.assertGreater(info["accepted"], 0)
                if mode == 2:
                    self.assertEqual(info["accepted"], 0)

    def test_caller_drafts_and_one_row(self):
        jobs = self.jobs(8, seed=7)
        oracle = [self.solo(p, k, s).ids for p, k, s in jobs]
        for drafts in ("none", "wrong", "oracle"):
            with self.subTest(drafts=drafts):
                self.e.ms_flush()
                got, info = self.run_ms(jobs, 3, drafts=drafts, oracle=oracle)
                self.check_equal(jobs, got)
                if drafts == "oracle":
                    self.assertGreater(info["accepted"], 0)
                else:
                    self.assertEqual(info["accepted"], 0)

    def test_stops_inside_drafts(self):
        self.stub.model(self.e, True, eos_every=5)
        self.stub.draft_mode(self.e, 1)
        jobs = self.jobs(16, seed=3, max_new=(40, 64))
        got, info = self.run_ms(jobs, 4)
        self.check_equal(jobs, got)
        self.assertGreater(info["cut_in_drafts"], 0)
        self.assertTrue(any(g[-1] == EOS for g in got) and any(g[-1] == EOS2 for g in got))

    def test_ignore_eos(self):
        self.stub.model(self.e, True, eos_every=4)
        self.stub.draft_mode(self.e, 1)
        jobs = self.jobs(6, seed=5, max_new=(30, 47))
        got, _ = self.run_ms(jobs, 3, stops=())
        self.check_equal(jobs, got, stops=())
        for (p, k, s), g in zip(jobs, got):
            self.assertEqual(len(g), k)
            self.assertTrue(set(g) & set(EOS_IDS))                 # stop ids were produced and did not end the streams

    def test_rows_of_other_widths(self):
        self.stub.draft_mode(self.e, 1)
        jobs = self.jobs(6, seed=9)
        for rows in (2, 5, 16):
            with self.subTest(rows=rows):
                self.e.ms_flush()
                got, _ = self.run_ms(jobs, 2, rows=rows)
                self.check_equal(jobs, got)

    # ---- refusals: nothing changes ----------------------------------------------------------------------------------------------
    def test_first_row_must_be_the_frontier(self):
        self.e.ms_open(2)
        ids = self.read([5, 6, 7], GREEDY, 0)
        self.assertEqual(self.stub.slot_front(self.e, 0), ids[-1])
        with self.assertRaisesRegex(YasakaniError, "emitted"):
            self.e.ms_pass([0], [ids[-1] + 1])
        with self.assertRaisesRegex(YasakaniError, "emitted"):
            self.e.ms_pass([0, 0], [ids[-1] + 1, 0], [1, 0])
        self.assertEqual(self.e.ms_slot_pos(0), 3)
        r = self.e.ms_pass([0], [ids[-1]])
        self.e.ms_commit([], [])
        self.assertEqual(self.e.ms_slot_pos(0), 4)
        self.assertEqual(self.stub.slot_front(self.e, 0), r.argmax[0])

    def test_empty_slot_and_rows_apart(self):
        self.e.ms_open(3)
        ids = self.read([5, 6, 7], GREEDY, 0)
        with self.assertRaisesRegex(YasakaniError, "holds no stream"):
            self.e.ms_pass([1], [9])
        ids1 = self.read([8, 9], GREEDY, 1)
        with self.assertRaisesRegex(YasakaniError, "next to each other"):
            self.e.ms_pass([0, 1, 0], [ids[-1], ids1[-1], 0])
        with self.assertRaisesRegex(YasakaniError, "no such stream slot"):
            self.e.ms_pass([3], [0])
        self.e.ms_pass([0, 0, 1], [ids[-1], 0, ids1[-1]])
        self.e.ms_commit([], [])

    def test_pass_shapes(self):
        self.e.ms_open(1)
        for slots, toks, dev in (([], [], None), ([0] * 17, [0] * 17, None), ([0, 0], [1], None), ([0], [1], [1, 0])):
            with self.assertRaises(ValueError):
                self.e.ms_pass(slots, toks, dev)

    def test_commit_checks(self):
        self.e.ms_open(2)
        self.stub.draft_mode(self.e, 2)                            # the first draft is wrong
        ids = self.read([5, 6, 7], GREEDY, 0)
        r = self.e.ms_pass([0] * 4, [ids[-1], 0, 0, 0], [1, 0, 0, 0])
        self.assertNotEqual(r.toks[1], r.argmax[0])
        for slots, acc, msg in (([0], [1], "did not confirm"), ([0], [4], "more accepted"), ([1], [0], "no row"),
                                ([0, 0], [0, 0], "listed twice"), ([0], [-1], "more accepted")):
            with self.assertRaisesRegex(YasakaniError, msg):
                self.e.ms_commit(slots, acc)
        self.e.ms_commit([0], [0])                                 # the pass stayed open through the refusals
        with self.assertRaisesRegex(YasakaniError, "no pass to commit"):
            self.e.ms_commit([], [])
        self.stub.draft_mode(self.e, 1)                            # all right: three drafts confirmed
        r = self.e.ms_pass([0] * 4, [r.argmax[0], 0, 0, 0], [1, 0, 0, 0])
        a, new = ms_accept(r.toks, r.argmax)
        self.assertEqual(a, 3)
        self.e.ms_commit([0], [a])
        self.assertEqual(self.e.ms_slot_pos(0), 3 + 1 + 1 + 3)
        self.assertEqual(self.stub.slot_front(self.e, 0), new[-1])

    def test_a_pass_left_uncommitted_accepts_nothing(self):
        self.e.ms_open(1)
        self.stub.draft_mode(self.e, 1)
        ids = self.read([5, 6, 7], GREEDY, 0)
        r = self.e.ms_pass([0] * 3, [ids[-1], 0, 0], [1, 0, 0])
        self.assertEqual(ms_accept(r.toks, r.argmax)[0], 2)
        self.e.ms_flush()
        with self.assertRaisesRegex(YasakaniError, "no pass to commit"):
            self.e.ms_commit([0], [2])
        self.assertEqual(self.e.ms_slot_pos(0), 4)                 # only the first row went in
        after = r.argmax[2] if r.argmax[2] != r.argmax[0] else r.argmax[0] + 1
        with self.assertRaisesRegex(YasakaniError, "emitted"):     # the drafts were not committed: the frontier is the first answer
            self.e.ms_pass([0], [after])
        self.e.ms_pass([0], [r.argmax[0]])
        self.e.ms_commit([], [])

    def test_engine_fault_in_a_pass(self):
        self.e.ms_open(1)
        ids = self.read([5, 6], GREEDY, 0)
        self.stub.ms_fail_in(self.e, 1)
        with self.assertRaisesRegex(YasakaniError, "injected failure"):
            self.e.ms_pass([0], [ids[-1]])

    def test_tops(self):
        self.e.ms_open(2)
        a = self.read([5, 6], GREEDY, 0)
        b = self.read([9, 8, 7], MIXED[1], 1)
        r = self.e.ms_pass([0, 0, 1], [a[-1], 0, b[-1]], [1, 0, 1], tops=True)
        self.assertEqual(r.top_ids.shape, (3, MS_TOP))
        self.assertEqual(r.top_vals.shape, (3, MS_TOP))
        self.assertEqual(r.top_ids[:, 0].tolist(), r.argmax)
        self.e.ms_commit([], [])
        r = self.e.ms_pass([0, 1], [r.argmax[0], r.argmax[2]])
        self.assertIsNone(r.top_ids)
        self.e.ms_commit([], [])

    # ---- sampling ---------------------------------------------------------------------------------------------------------------
    def finish_alone(self, slot, ids, max_new, stops=EOS_IDS):
        """Passes of one slot (the drafter's drafts) until its stream ends."""
        while len(ids) < max_new and ids[-1] not in stops:
            want = min(7, max_new - len(ids) - 1)
            r = self.e.ms_pass([slot] * (1 + want), [ids[-1]] + [0] * want, [1] + [0] * want)
            a, new = ms_accept(r.toks, r.argmax, stops)
            self.e.ms_commit([slot], [a])
            ids += new
        return ids

    def test_slot_sampling(self):
        self.e.set_sampling(MIXED[1])
        self.e.ms_open(2)
        self.assertEqual(self.e.slot_sampling(1), MIXED[1].normalized())   # a new slot starts with the carrier's sampling
        p, k = [41, 42, 43], 30
        ids = self.read(p, MIXED[2], 0)
        self.assertEqual(self.e.slot_sampling(0), MIXED[2].normalized())   # the load gives the slot the carrier's values
        with self.assertRaisesRegex(YasakaniError, "sampler"):
            self.e.ms_set_sampling(0, Sampling(1.0, 0.9, 20, 0.0, 1, mode=5))
        self.assertEqual(self.e.slot_sampling(0), MIXED[2].normalized())   # refused: nothing changed
        self.assertEqual(self.e.ms_set_sampling(0, MIXED[2]), MIXED[2].normalized())   # the same values: the same ids
        self.assertEqual(self.finish_alone(0, ids, k), self.solo(p, k, MIXED[2]).ids)
        self.assertIsNone(self.e.slot_sampling(7))

    def test_set_sampling_changes_the_stream(self):
        """After the first id, a slot set to other values goes on as the carrier would with those values."""
        self.stub.model(self.e, True, eos_every=0)
        self.e.ms_open(1)
        p = [41, 42, 43]
        first = self.read(p, GREEDY, 0)
        self.e.ms_set_sampling(0, MIXED[3])
        ids = list(first)
        while len(ids) < 20:
            r = self.e.ms_pass([0], [ids[-1]])
            self.e.ms_commit([], [])
            ids.append(r.argmax[0])
        self.e.reset()
        self.e.set_sampling(MIXED[3])
        ref = self.e.generate(p + first, 19)                       # the same stream on the carrier, from the second id on
        self.assertEqual(ids[1:], ref.ids)
        self.assertNotEqual(ids[1:], self.solo(p, 20, GREEDY).ids[1:])

    # ---- back to one request at a time ------------------------------------------------------------------------------------------
    def _run_some(self, slot, prompt, s, n_passes):
        ids = self.read(prompt, s, slot)
        self.stub.draft_mode(self.e, 0)
        for _ in range(n_passes):
            r = self.e.ms_pass([slot] * 4, [ids[-1], 0, 0, 0], [1, 0, 0, 0])
            a, new = ms_accept(r.toks, r.argmax, EOS_IDS)
            self.e.ms_commit([slot], [a])
            ids += new
        return ids

    def test_unload_and_continue(self):
        self.stub.model(self.e, True, eos_every=0)
        p, s, k = [3, 1, 4, 1, 5], MIXED[2], 60
        ref = self.solo(p, k, s).ids
        for frontier in (False, True):
            with self.subTest(frontier=frontier):
                self.e.reset()
                self.e.ms_open(2)
                self.e.set_sampling(GREEDY)
                other = self.read([9, 9, 9], GREEDY, 1)                   # another stream shares the passes
                ids = self._run_some(0, p, s, 5)
                self.e.ms_pass([1], [other[-1]])
                self.e.ms_commit([], [])
                held = p + ids[:-1]
                self.assertEqual(self.e.ms_slot_pos(0), len(held))
                h = self.e.ms_slot_hash(0)
                self.e.set_sampling(GREEDY)                                 # the carrier holds other values now
                self.e.ms_unload(0, held)
                self.assertEqual(self.e.sampling, s.normalized())           # the carrier took the slot's sampling
                self.assertEqual(self.stub.opt(self.e, 13), s.seed)
                self.assertEqual(self.e.state_hash().tolist(), h.tolist())
                live, front, _ = self.stub.live(self.e)
                self.assertEqual((live, front), (held, ids[-1]))
                with self.assertRaisesRegex(YasakaniError, "moved to the carrier"):
                    self.e.ms_pass([0], [ids[-1]])
                with self.assertRaisesRegex(YasakaniError, "already"):
                    self.e.ms_unload(0, held)
                r = self.e.generate(held + [ids[-1]], k - len(ids), frontier=frontier)
                self.assertEqual((r.snap_mode, r.cached_tokens), (1, len(held)))
                self.assertEqual(ids + r.ids, ref)
                self.e.ms_load(0)                                           # and back into the slot
                self.assertEqual(self.e.ms_slot_pos(0), len(p) + len(ref) - 1)
                self.e.ms_pass([0, 1], [ref[-1], self.stub.slot_front(self.e, 1)])
                self.e.ms_commit([], [])

    def test_unload_checks(self):
        self.e.ms_open(1)
        p = [3, 1, 4]
        ids = self._run_some(0, p, GREEDY, 2)
        held = p + ids[:-1]
        for bad, msg in ((held[:-1], "tokens the slot holds"), ([0] + held[1:], "start with"), (held[:-1] + [0], "end with")):
            with self.assertRaisesRegex(YasakaniError, msg):
                self.e.ms_unload(0, bad)
        self.e.ms_pass([0], [ids[-1]])
        with self.assertRaisesRegex(YasakaniError, "not committed"):
            self.e.ms_unload(0, held + [ids[-1]])
        self.e.ms_commit([], [])
        self.e.set_sampling(MIXED[1])
        self.e.ms_unload(0, held + [ids[-1]])
        self.assertEqual(self.e.sampling, GREEDY.normalized())

    # ---- the live sequence and the frontier form ---------------------------------------------------------------------------------
    def test_generate_sends_46_values_unless_frontier(self):
        self.stub.model(self.e, True, eos_every=0)
        r = self.e.generate([5, 6, 7], 4)
        cfg = self.stub.last_cfg()
        self.assertEqual(len(cfg), 46)
        r2 = self.e.generate([5, 6, 7] + r.ids, 4, frontier=True)
        cfg = self.stub.last_cfg()
        self.assertEqual((len(cfg), cfg[CFG_FRONTIER]), (47, 1))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (1, 3 + len(r.ids) - 1))
        self.assertEqual(r.ids + r2.ids, self.solo([5, 6, 7], 8, GREEDY).ids)

    def test_frontier_form_refusals(self):
        p = [5, 6, 7]
        r = self.e.generate(p, 4)
        before = self.stub.live(self.e)
        with self.assertRaisesRegex(YasakaniError, "emitted last"):
            self.e.generate(p + r.ids[:-1] + [r.ids[-1] + 1], 4, frontier=True)
        with self.assertRaisesRegex(YasakaniError, "live sequence"):
            self.e.generate(p + r.ids + [1], 4, frontier=True)
        self.assertEqual(self.stub.live(self.e), before)            # refused: nothing changed
        self.e.live_drop()
        self.assertIsNone(self.stub.live(self.e)[0])
        with self.assertRaisesRegex(YasakaniError, "live sequence"):
            self.e.generate(p + r.ids, 4, frontier=True)
        with self.assertRaises(ValueError):
            self.e.generate(p, 4, spec={"frontier": 1})

    def test_what_ends_the_live_sequence(self):
        p = [5, 6, 7]

        def continued(prepare):
            self.e.reset()
            r = self.e.generate(p, 4)
            prepare()
            return self.e.generate(p + r.ids + [11], 4).snap_mode == 1

        self.assertTrue(continued(lambda: None))
        self.assertTrue(continued(self.e.ms_flush))                 # a flush keeps it
        self.assertFalse(continued(self.e.live_drop))
        self.assertFalse(continued(lambda: self.e.ms_open(2)))      # a new binding drops it
        self.assertFalse(continued(lambda: self.e.ms_load(0)))      # the stream lives in the slot now
        self.assertTrue(continued(lambda: self.e.ms_set_sampling(1, GREEDY)))
        self.assertTrue(continued(lambda: (self.e.ms_slot_pos(0), self.e.ms_slot_hash(0), self.e.ms_info())))


if __name__ == "__main__":
    unittest.main()
