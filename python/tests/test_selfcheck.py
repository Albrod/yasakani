# SPDX-License-Identifier: Apache-2.0
"""0.1.4: the self-check on GPUs other than the RTX 5090 (yasakani/selfcheck.py): teacher forcing, the reference data, the tolerances.

The tests of the tolerances need no engine (numpy only) and run anywhere; the teacher forcing through the engine library runs with the
stub library (POSIX).  The command line with a stand-in nvidia-smi is tested in tests/test_gpu.py.
"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import support
from yasakani import selfcheck as sc
from yasakani._native import YasakaniError
from yasakani.engine import forced_passes

# the cases of the two published reference files (selfcheck.json of the released models; sha256 4d4bb670... and 3e5efb03...)
_P = dict(short=("short-answer", 19, "181481ee96e0a5a5705b34769fd04a590ab38936f8ccfac931ce71d24131acb1", 3,
                 "18de4b801242d19586e669bdce6e897d4de7b7862f280a5ebbc241ce299ec65d"),
          plain=("plain-text", 5, "fa7bdc58b8d16bb0db3dd7cd461fbccc9484ea7115884605ae249040d9f7bf29", 32,
                 "611bde9f7b74f1ed372de23fb42121311a2bad2407fed84db266f657ae73abf6"))
_THINK_PROMPT = "2c2892dba638a3dbd7463c3a31f27f4f50121aee77b61ae391fce43249215c83"
PUBLISHED = {
    "quality": [_P["short"], ("thinking", 69, _THINK_PROMPT, 160, "8d20d06b210ca9d5baecee6e6421d8270602111839885da7d693ff5e3804bd55"), _P["plain"]],
    "speed": [_P["short"], ("thinking", 69, _THINK_PROMPT, 160, "afe2fd9847111d834ef0f725f197949760512330ac6486df5d786c34a4122a81"), _P["plain"]],
}


def published(model):
    return dict(format=1, model_id="qwen3.8-27b", yasakani="0.1.0.dev0",
                cases=[dict(name=n, prompt_tokens=p, prompt_sha256=ps, tokens=t, sha256=s) for n, p, ps, t, s in PUBLISHED[model]])


def forced_of(model):
    """name -> (prompt length, the RTX 5090's predictions) of a published reference."""
    ref = published(model)
    data = sc.full_data(ref)
    return {c["name"]: (c["prompt_tokens"], data[c["name"]]["forced"]) for c in ref["cases"]}


def compare_all(model, change=None):
    """CloseResult of the RTX 5090's own predictions against themselves, after change(name, argmax, nll) altered a copy."""
    res = sc.CloseResult()
    for name, (n_prompt, f) in forced_of(model).items():
        a, n = np.array(f["argmax"], np.int64), np.array(f["nll"], np.float64)
        if change:
            change(name, n_prompt, a, n, np.array(f["margin"]))
        res.cases.append(sc.compare_case(name, n_prompt, f, a, n))
    return res


class PassTests(unittest.TestCase):
    def test_forced_passes(self):
        self.assertEqual(forced_passes(1, 16), [1])
        self.assertEqual(forced_passes(2, 16), [2])
        self.assertEqual(forced_passes(16, 16), [16])
        self.assertEqual(forced_passes(17, 16), [15, 2])                  # never a last pass of one row
        self.assertEqual(forced_passes(32, 16), [16, 16])
        self.assertEqual(forced_passes(33, 16), [16, 15, 2])
        self.assertEqual(forced_passes(227, 16), [16] * 14 + [3])
        for n in range(1, 300):
            s = forced_passes(n, 16)
            self.assertEqual(sum(s), n)
            self.assertTrue(all(1 <= x <= 16 for x in s))
            self.assertTrue(len(s) == 1 or s[-1] >= 2)
        with self.assertRaises(ValueError):
            forced_passes(0, 16)


class ReferenceDataTests(unittest.TestCase):
    def test_known_tables_cover_the_published_files(self):
        for model in PUBLISHED:
            ref = published(model)
            data = sc.full_data(ref)
            self.assertEqual(sorted(data), sorted(c["name"] for c in ref["cases"]), model)
            for c in ref["cases"]:
                d = data[c["name"]]
                self.assertEqual(sc.ids_sha(d["ids"]), c["sha256"])
                f = d["forced"]
                n = c["prompt_tokens"] + c["tokens"] - 1
                self.assertEqual((f["rows"], len(f["argmax"]), len(f["nll"]), len(f["margin"])), (16, n, n, n))
                # the reference output is what the RTX 5090 predicts while it reads it
                self.assertEqual(f["argmax"][c["prompt_tokens"] - 1:], d["ids"], (model, c["name"]))
                self.assertTrue(all(x >= 0 for x in f["nll"]) and all(x >= 0 for x in f["margin"]))

    def test_the_two_models_have_their_own_tables(self):
        q, s = sc.full_data(published("quality")), sc.full_data(published("speed"))
        self.assertEqual(q["plain-text"]["ids"], s["plain-text"]["ids"])          # the same output ...
        self.assertNotEqual(q["plain-text"]["forced"]["nll"], s["plain-text"]["forced"]["nll"])     # ... with other predictions
        self.assertNotEqual(sc.reference_key(published("quality")), sc.reference_key(published("speed")))

    def test_key_ignores_order_and_other_fields(self):
        ref = published("quality")
        other = copy.deepcopy(ref)
        other["cases"].reverse()
        other["yasakani"] = "9.9"
        for c in other["cases"]:
            c["head"] = [1, 2, 3]
        self.assertEqual(sc.reference_key(ref), sc.reference_key(other))
        other["cases"][0]["sha256"] = "0" * 64
        self.assertNotEqual(sc.reference_key(ref), sc.reference_key(other))
        self.assertEqual(sc.full_data(other), {})                                   # a changed file is not a published one

    def test_data_of_the_file_itself(self):
        ids = [5, 6, 7, 8]
        n = 3 + len(ids) - 1
        forced = dict(rows=16, argmax=[1] * n, nll=[0.5] * n, margin=[1.0] * n)
        case = dict(name="x", prompt_tokens=3, prompt_sha256="p", tokens=len(ids), sha256=sc.ids_sha(ids), ids=ids, forced=forced)
        self.assertEqual(sc.full_data(dict(cases=[case])), {"x": dict(ids=ids, forced=forced)})
        for bad in (dict(ids=[5, 6, 7, 9]), dict(tokens=5), dict(forced=dict(forced, rows=8)), dict(forced=dict(forced, nll=[0.5] * (n - 1))),
                    dict(ids="5678"), dict(forced=None)):
            self.assertEqual(sc.full_data(dict(cases=[dict(case, **bad)])), {}, bad)

    def test_dumps(self):
        obj = dict(format=1, cases=[dict(name="a", ids=[1, 2, 3], forced=dict(rows=16, nll=[0.5, 1e-05, 0.0], margin=[]))])
        text = sc.dumps(obj)
        self.assertEqual(json.loads(text), obj)
        self.assertIn('"ids": [1,2,3]', text)
        self.assertIn('"nll": [0.5,1e-05,0.0]', text)
        self.assertIn('"margin": []', text)


class ToleranceTests(unittest.TestCase):
    """The RTX 5090's predictions of the published files, changed on purpose."""

    def test_identical_passes(self):
        for model in PUBLISHED:
            res = compare_all(model)
            self.assertEqual(res.failures(), [])
            self.assertEqual((res.positions, res.output_positions), (285, 195))
            self.assertEqual((res.top1, res.output_mean_nll, res.output_max_nll, res.sure_changed), (1.0, 0.0, 0.0, 0))

    def test_small_differences_pass(self):
        rng = np.random.default_rng(7)

        def noise(name, n_prompt, a, n, m):
            n += rng.normal(0, 0.004, len(n))                                     # every position, output included
            low = np.argsort(m)[:1]                                                # one near tie of this case changes
            a[low] += 1
        res = compare_all("quality", noise)
        self.assertEqual(res.failures(), [], res.failures())
        self.assertLess(res.top1, 1.0)

    def test_mean_nll(self):
        def shift(name, n_prompt, a, n, m):
            n[n_prompt - 1:] += 0.02
        res = compare_all("quality", shift)
        self.assertAlmostEqual(res.output_mean_nll, 0.02, places=6)
        f = res.failures()
        self.assertEqual(len(f), 1)
        self.assertIn("differs by 0.0200 on average", f[0])

    def test_prompt_positions_do_not_count_for_the_nll(self):
        def shift(name, n_prompt, a, n, m):
            n[:n_prompt - 1] += 1.0                                                # the prompt's ids are unlikely ones: noisy
        self.assertEqual(compare_all("quality", shift).failures(), [])

    def test_max_nll(self):
        def spike(name, n_prompt, a, n, m):
            if name == "thinking":
                n[n_prompt + 10] += 0.3
        f = compare_all("quality", spike).failures()
        self.assertEqual(len(f), 1)
        self.assertIn("0.300 at one position", f[0])

    def test_top1(self):
        def flips(k):
            def change(name, n_prompt, a, n, m):
                if name == "thinking":
                    a[np.argsort(m)[:k]] += 1                                      # the k nearest ties (all under 2 nats)
            return change
        self.assertEqual(compare_all("quality", flips(5)).failures(), [])           # 280 of 285 = 98.2 %
        f = compare_all("quality", flips(6)).failures()                             # 279 of 285 = 97.9 %
        self.assertEqual(len(f), 1)
        self.assertIn("97.9 %", f[0])

    def test_sure_position(self):
        def change(name, n_prompt, a, n, m):
            if name == "plain-text":
                i = int(np.argmax(m))
                a[i] += 1
        f = compare_all("quality", change).failures()
        self.assertEqual(len(f), 1)
        self.assertIn("changed at 1 position(s)", f[0])

    def test_nan(self):
        for where in ("prompt", "output"):
            def change(name, n_prompt, a, n, m):
                if name == "thinking":
                    n[3 if where == "prompt" else n_prompt + 3] = math.nan
            f = compare_all("quality", change).failures()
            self.assertTrue(any("NaN" in x for x in f), (where, f))

    def test_problems_and_nothing(self):
        self.assertEqual(sc.CloseResult().failures(), ["nothing was compared"])
        r = compare_all("quality")
        r.problems.append("thinking: the reference file has no full output")
        self.assertEqual(r.failures(), ["thinking: the reference file has no full output"])

    def test_lines(self):
        lines = sc.close_lines(compare_all("speed"))
        self.assertEqual(len(lines), 5)
        self.assertIn("all 285 positions: same most likely id at 100.0 %", lines[3])
        self.assertIn("the 195 positions of the reference output", lines[4])


class ConsistencyTests(unittest.TestCase):
    """The self-consistency check of every GPU (0.1.4): exact comparisons, no tolerance."""

    def test_copy_case(self):
        c = sc.copy_case()
        self.assertEqual(c["template_kwargs"], {"enable_thinking": False})
        text = c["messages"][0]["content"]
        self.assertEqual(text.count('{"id":"R-'), 60)
        self.assertIn('{"id":"R-017","qty":4,"status":"hold"}', text)
        self.assertEqual(sc.copy_case(), c)                                       # fixed, no randomness

    def test_failures(self):
        ok = sc.Consistency(prompt_tokens=1127, tokens=1080, passes=27, wide_passes=15, max_rows=64, compared=1080,
                            wide_checked=True, wide_positions=2206)
        self.assertEqual(ok.failures(), [])
        lines = sc.consistency_lines(ok)
        self.assertIn("15 of them wider than 16 positions, the widest 64", lines[0])
        self.assertIn("every generated id is the most likely id (1080 ids)", lines[1])
        self.assertIn("bit for bit (2206 positions)", lines[2])
        bad = copy.deepcopy(ok)
        bad.mismatches, bad.first_mismatch = 2, 99
        self.assertIn("2 of the 1080 generated ids", bad.failures()[0])
        self.assertIn("(first: id #100)", bad.failures()[0])
        bad = copy.deepcopy(ok)
        bad.wide_nll_diff = 1
        self.assertIn("passes of 64 positions do not give the same predictions", bad.failures()[0])
        bad = copy.deepcopy(ok)
        bad.wide_argmax_diff = 1
        self.assertEqual(len(bad.failures()), 1)
        narrow = sc.Consistency(tokens=10, compared=10, notes=["this GPU does not use passes of more than 16 positions"])
        self.assertEqual(narrow.failures(), [])
        self.assertIn("note: this GPU does not use", sc.consistency_lines(narrow)[-1])
        self.assertEqual(sc.Consistency(problems=["x"]).failures(), ["x"])


class StubForcingTests(unittest.TestCase):
    """Engine.teacher_forced through the stub library (POSIX; skipped where the stub cannot be built)."""

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=512)

    def tearDown(self):
        self.e.close()
        self.tmp.cleanup()

    def test_teacher_forced(self):
        ids = list(range(100, 141))                                                # 40 predictions: passes of 16, 16, 8
        f = self.e.teacher_forced(ids, margins=True)
        self.assertEqual(f.argmax.tolist(), ids[1:])
        want_nll = [0.05 + 0.001 * ((t * 31 + p * 7) % 97) for p, t in enumerate(ids[1:])]
        np.testing.assert_allclose(f.nll, want_nll, rtol=0, atol=1e-6)
        np.testing.assert_allclose(f.margin, [0.5 + ((t + p) % 8) * 0.5 for p, t in enumerate(ids[1:])], rtol=0, atol=1e-6)
        self.assertEqual(self.e.position(), 0)                                     # reset afterwards
        g = self.e.teacher_forced(ids)
        self.assertIsNone(g.margin)
        np.testing.assert_array_equal(g.nll, f.nll)

    def test_record_and_limits(self):
        rec = sc.forced_record(self.e, [1, 2, 3], [4, 5, 6])
        self.assertEqual(rec["rows"], 16)
        self.assertEqual(rec["argmax"], [2, 3, 4, 5, 6])
        self.assertEqual(len(rec["nll"]), 5)
        with self.assertRaises(ValueError):
            self.e.teacher_forced(list(range(600)))                                # longer than the context
        with self.assertRaises(ValueError):
            self.e.teacher_forced([1])

    def test_missing_exports(self):
        saved = dict(self.e._f)
        try:
            del self.e._f["yskn_verify_nll"]
            self.assertFalse(self.e.supports_forced)
            with self.assertRaises(YasakaniError):
                self.e.teacher_forced([1, 2, 3])
            res = sc.close_check(self.e, [], published("quality"))
            self.assertIn("no yskn_verify_nll", res.failures()[0])
        finally:
            self.e._f.clear()
            self.e._f.update(saved)


if __name__ == "__main__":
    unittest.main()
