# SPDX-License-Identifier: Apache-2.0
"""``yasakani bench --edit`` (0.1.6): the edit tasks, the comparison with the expected answer, the header, and runs on the stub."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests import support
from yasakani import editbench
from yasakani._native import YasakaniError
from yasakani.gpu import GpuInfo

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT.parent / "bench"                       # the measurement records (the repository only, not the engine zip)
EOS = 248046
POSIX_TOK = support.tokenizer_dir() is not None and os.name == "posix"

FAKE_SMI = r'''#!{python}
import json, os, sys
gpus = json.loads(os.environ["FAKE_GPUS"])
q = [a for a in sys.argv[1:] if a.startswith("--query-gpu=")]
if not q:
    sys.exit(0)
fields = q[0].split("=", 1)[1].split(",")
units = "nounits" not in " ".join(sys.argv)
for g in gpus:
    print(", ".join(str(g[f]) + (" MiB" if units and f.startswith("memory.") else "") for f in fields))
'''


class TaskDataTests(unittest.TestCase):
    def test_the_three_tasks(self):
        tasks = editbench.load_tasks()
        self.assertEqual([t.name for t in tasks], ["edit_code", "edit_jsonl", "edit_jsonl_long"])
        self.assertEqual([t.title for t in tasks], ["code edit", "JSONL edit", "long JSONL edit"])
        self.assertEqual([t.max_tokens for t in tasks], [8192, 8192, 12000])
        for t in tasks:
            self.assertEqual([m["role"] for m in t.messages], ["user"])
            self.assertTrue(t.expected.strip())

    def test_the_answers_make_the_asked_edit(self):
        code, jsonl, long_ = editbench.load_tasks()
        before = code.messages[0]["content"].split("\n\n", 1)[1]
        self.assertNotIn("common_margin", before)
        self.assertIn("common_margin", code.expected)
        self.assertEqual(code.expected.replace("common_margin", "margin").strip(), before.strip())    # only the rename
        self.assertIn("<email>", code.expected)                       # the address of textwrap.py is replaced, as in bench/
        for t, ids in ((jsonl, ["ORD-0042"]), (long_, ["ORD-0017", "ORD-0088", "ORD-0143"])):
            src = t.messages[0]["content"].split("\n\n", 1)[1].strip().split("\n")
            exp = t.expected.split("\n")
            self.assertEqual(len(src), len(exp))
            changed = [i for i, (a, b) in enumerate(zip(src, exp)) if a != b]
            self.assertEqual([json.loads(exp[i])["id"] for i in changed], ids)
            self.assertTrue(all(json.loads(exp[i])["status"] == "returned" for i in changed))

    def test_checksum_and_format_are_checked(self):
        data = json.loads(editbench.TASKS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(data["format"], editbench.FORMAT)
        for t in data["tasks"]:
            self.assertEqual(hashlib.sha256(t["expected"].encode("utf-8")).hexdigest(), t["expected_sha256"])
        tmp = Path(tempfile.mkdtemp(prefix="yedit_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        bad = json.loads(json.dumps(data))
        bad["tasks"][1]["expected"] += " "
        (tmp / "a.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(YasakaniError):
            editbench.load_tasks(tmp / "a.json")
        bad = dict(data, format="other")
        (tmp / "b.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(YasakaniError):
            editbench.load_tasks(tmp / "b.json")
        with self.assertRaises(YasakaniError):
            editbench.load_tasks(tmp / "missing.json")

    def test_think_settings(self):
        # --think (0.1.6 round e): the server's default output limit for every task and the package's default sampling
        import inspect
        from yasakani import chat, server
        self.assertEqual(editbench.THINK_MAX_TOKENS, 32768)
        self.assertEqual(editbench.THINK_MAX_TOKENS, inspect.signature(server.ServerConfig).parameters["default_max_tokens"].default)
        self.assertEqual(editbench.THINK_MAX_TOKENS, chat.ChatParams(messages=[]).max_tokens)
        self.assertEqual((editbench.THINK_TEMPERATURE, editbench.THINK_TOP_P, editbench.THINK_TOP_K),
                         (chat.DEFAULT_TEMPERATURE, chat.DEFAULT_TOP_P, chat.DEFAULT_TOP_K))
        for t in editbench.load_tasks():
            self.assertEqual(editbench.max_tokens_of(t, False), t.max_tokens)
            self.assertEqual(editbench.max_tokens_of(t, True), 32768)
        self.assertEqual(editbench.sampling_of(False, 3).normalized().temperature, 0.0)            # greedy
        for seed in (0, 1, 7):
            s = editbench.sampling_of(True, seed)
            self.assertEqual((s.temperature, s.top_p, s.top_k, s.min_p, s.seed), (1.0, 0.95, 20, 0.0, seed))
        self.assertEqual(editbench.footer(False), editbench.FOOTER)
        ft = editbench.footer(True)
        for want in ("the thinking included", "temperature 1.0", "32768", "varies from run to run", "not sped up much",
                     "thinking columns of the README tables"):
            self.assertIn(want, ft)
        self.assertNotIn("2 runs per task", ft)

    @unittest.skipUnless((BENCH / "20261103" / "prompts" / "cases.json").is_file(), "the bench/ records are in the repository only")
    def test_same_as_the_measurement_records(self):
        cases = json.loads((BENCH / "20261103" / "prompts" / "cases.json").read_text(encoding="utf-8"))
        cases += json.loads((BENCH / "20261115" / "prompts" / "cases_plus.json").read_text(encoding="utf-8"))
        by = {c["name"]: c for c in cases}
        for t in editbench.load_tasks():
            c = by[t.name]
            self.assertEqual((t.messages, t.expected, t.max_tokens), (c["messages"], c["expected"], int(c["max_new"])))


class CompareTests(unittest.TestCase):
    def test_compare(self):
        exp = "a\nb\nc"
        self.assertEqual(editbench.compare(exp, "a\nb\nc"), (True, 0))
        self.assertEqual(editbench.compare(exp, "\n  a\nb\nc\n\n"), (True, 0))
        self.assertEqual(editbench.compare(exp, "```\na\nb\nc\n```"), (True, 0))
        self.assertEqual(editbench.compare(exp, "```python\na\nb\nc\n```\n"), (True, 0))
        self.assertEqual(editbench.compare(exp, "a\nB\nc"), (False, 2))
        self.assertEqual(editbench.compare(exp, "a\nb"), (False, 1))
        self.assertEqual(editbench.compare(exp, ""), (False, 4))


def fake_engine(rows=64, wide_export=True, path=None):
    fn = {"yskn_prepare_wide": object()} if wide_export else {}
    return SimpleNamespace(spec={"copy_rows": rows}, lib=SimpleNamespace(fn=fn, path=path), max_ctx=262144)


class HeaderTests(unittest.TestCase):
    def gpu(self, cc, name="NVIDIA GeForce RTX 5090"):
        return GpuInfo(index=0, name=name, uuid="GPU-0", compute_cap=cc, total_mib=32607, used_mib=1000, driver="616.92")

    def test_wide_status(self):
        s, line = editbench.wide_status(fake_engine(), self.gpu((12, 0)))
        self.assertEqual(s, "on")
        self.assertIn("compute capability 12.0", line)
        self.assertIn("12.x (RTX 50 series) only", line)
        self.assertIn("measured on the RTX 5090 only", line)
        s, line = editbench.wide_status(fake_engine(), self.gpu((8, 9), "NVIDIA GeForce RTX 4090"))
        self.assertEqual(s, "off")
        self.assertIn("compute capability 8.9", line)
        self.assertIn("up to 16 tokens", line)
        self.assertEqual(editbench.wide_status(fake_engine(), self.gpu(None))[0], "unknown")
        self.assertEqual(editbench.wide_status(fake_engine(), None)[0], "unknown")
        self.assertEqual(editbench.wide_status(fake_engine(rows=16), self.gpu((12, 0)))[0], "off")
        self.assertEqual(editbench.wide_status(fake_engine(wide_export=False), self.gpu((12, 0)))[0], "off")

    def test_header_says_what_ran(self):
        tmp = Path(tempfile.mkdtemp(prefix="yedit_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        dll = tmp / "yskn.dll"
        dll.write_bytes(b"engine")
        pack = SimpleNamespace(model_id="qwen3.8-27b", weights=tmp / "qwen38_27b_speed.yskn")
        h = editbench.header("0.1.6", pack, fake_engine(path=dll), self.gpu((12, 0)), ["x"], False, 1)
        self.assertEqual(h["engine"], dict(file="yskn.dll", sha256=hashlib.sha256(b"engine").hexdigest()))
        self.assertEqual(h["gpu"], dict(index=0, name="NVIDIA GeForce RTX 5090", compute_capability="12.0", driver="616.92"))
        self.assertEqual((h["wide_copy_passes"], h["thinking"], h["greedy"], h["weights"]), ("on", False, True, "qwen38_27b_speed.yskn"))
        self.assertEqual((h["sampling"], h["output_limit"]), (None, None))
        text = "\n".join(editbench.header_lines(h))
        for want in ("greedy, thinking off, 1 run each", "yasakani 0.1.6", "sha256 " + hashlib.sha256(b"engine").hexdigest()[:16],
                     "GPU 0: NVIDIA GeForce RTX 5090, compute capability 12.0, driver 616.92", "qwen38_27b_speed.yskn",
                     "(context 262144 tokens)", "wide copy passes: on", "tokens/pass"):
            self.assertIn(want, text)
        h = editbench.header("0.1.6", pack, fake_engine(path=dll), None, ["nvidia-smi not found"], True, 2, seed0=5)
        text = "\n".join(editbench.header_lines(h))
        self.assertIn("(thinking on: temperature 1.0, top-p 0.95, top-k 20, seeds from 5 (one per run), output limit 32768 tokens, "
                      "2 runs each)", text)
        self.assertNotIn("greedy", text)
        self.assertEqual((h["thinking"], h["greedy"], h["output_limit"]), (True, False, 32768))
        self.assertEqual(h["sampling"], dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, first_seed=5))
        self.assertIn("gpu      nvidia-smi not found", text)
        self.assertIn("wide copy passes: not known", text)

    def test_row_line(self):
        r = dict(task="edit_code", title="code edit", run=1, prompt_tokens=4769, tokens=4708, seconds=2.33, tokens_per_s=2020.1,
                 tokens_per_pass=9.12, wide_passes=15, finish="stop", max_tokens=8192, same=True, diff_lines=0)
        self.assertIn("= expected", editbench.row_line(r))
        self.assertIn("2020.1", editbench.row_line(r))
        r.update(same=False, diff_lines=2, finish="length")
        self.assertIn("differs (2 lines), hit the limit of 8192 tokens", editbench.row_line(r))
        self.assertEqual(editbench.means([dict(task="a", tokens_per_s=1.0), dict(task="a", tokens_per_s=2.0), dict(task="b", tokens_per_s=3.0)]),
                         {"a": 1.5, "b": 3.0})


@unittest.skipUnless(POSIX_TOK, "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class RunOnStubTests(unittest.TestCase):
    """The runs themselves, in this process, with the stub library (scripted outputs) and the real tokenizer."""

    @classmethod
    def setUpClass(cls):
        from yasakani.chat import ChatRunner
        from yasakani.engine import Engine
        from yasakani.runtime import ModelPack
        from yasakani.textio import ChatFormatter
        cls.stub = support.Stub()
        cls.folder = support.make_real_folder("edit")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        cls.fmt = ChatFormatter(cls.folder)
        cls.tasks = editbench.load_tasks()
        cls.engines = []

        def runner(max_ctx=32768):
            eng = Engine(ModelPack.load(cls.folder), max_ctx=max_ctx, lib=cls.stub.lib)
            cls.engines.append(eng)
            return ChatRunner(eng, cls.fmt)
        cls.runner = staticmethod(runner)

    @classmethod
    def tearDownClass(cls):
        for e in cls.engines:
            e.close()

    def tearDown(self):
        os.environ.pop("STUB_GEN_WIDE_PASSES", None)

    def ids(self, text):
        return [int(x) for x in self.fmt.encode(text)] + [EOS]

    def test_expected_answers(self):
        r = self.runner()
        for t in self.tasks:
            self.stub.push(self.ids(t.expected))
        os.environ["STUB_GEN_WIDE_PASSES"] = "7"
        seen = []
        rows, skipped = editbench.run(r, self.tasks, on_row=seen.append)
        self.assertEqual(skipped, [])
        self.assertEqual(seen, rows)
        self.assertEqual([x["task"] for x in rows], ["edit_code", "edit_jsonl", "edit_jsonl_long"])
        for t, x in zip(self.tasks, rows):
            prompt, in_reasoning = self.fmt.prompt(t.messages, enable_thinking=False)
            self.assertFalse(in_reasoning)
            self.assertEqual(x["prompt_tokens"], len(prompt))
            self.assertEqual(x["tokens"], len(self.ids(t.expected)))
            self.assertEqual((x["same"], x["diff_lines"], x["finish"], x["reasoning_tokens"], x["wide_passes"]), (True, 0, "stop", 0, 7))
            self.assertAlmostEqual(x["tokens_per_s"], 10000.0 * (x["tokens"] - 1) / x["tokens"], delta=0.06)   # the stub: 0.1 ms per id
            self.assertAlmostEqual(x["tokens_per_pass"], x["tokens"] / x["passes"], places=2)
        self.assertEqual(self.stub.opt(r.engine, 10), 0)                  # greedy: temperature 0
        self.assertEqual([x["max_tokens"] for x in rows], [8192, 8192, 12000])
        self.assertEqual([x["seed"] for x in rows], [None, None, None])

    def test_a_different_answer(self):
        r = self.runner()
        t = self.tasks[1]
        self.stub.push(self.ids(t.expected.replace('"id":"ORD-0042","sku":"A-1042"', '"id":"ORD-0042","sku":"A-9999"')))
        rows, _ = editbench.run(r, [t])
        self.assertEqual((rows[0]["same"], rows[0]["diff_lines"]), (False, 2))
        self.stub.push(self.ids("```json\n" + t.expected + "\n```"))     # a code fence around it is ignored, as by the bench client
        rows, _ = editbench.run(r, [t])
        self.assertTrue(rows[0]["same"])

    def test_thinking_on(self):
        r = self.runner(max_ctx=65536)                                        # --think: the prompt + 32,768 + 64 must fit
        t = self.tasks[1]
        script = self.ids("Only ORD-0042 changes.")[:-1] + [self.fmt.think_end] + self.ids("\n\n" + t.expected)
        self.stub.push(script)
        self.stub.push(script)
        seeds = []
        rows, skipped = editbench.run(r, [t], think=True, runs=2, on_row=lambda x: seeds.append(self.stub.opt(r.engine, 13)))
        self.assertEqual(skipped, [])
        self.assertEqual(seeds, [1, 2])                         # the runner's seeds (seed0 = 1), as the engine got them for each run
        self.assertEqual([x["seed"] for x in rows], [1, 2])
        self.assertEqual([x["max_tokens"] for x in rows], [32768, 32768])
        self.assertEqual([self.stub.opt(r.engine, k) for k in (10, 11, 12, 29)], [1000, 950, 20, 0])   # temperature 1.0, top-p, top-k, min-p
        prompt, in_reasoning = self.fmt.prompt(t.messages, enable_thinking=True)
        self.assertTrue(in_reasoning)
        self.assertEqual([x["run"] for x in rows], [1, 2])
        for x in rows:
            self.assertEqual(x["prompt_tokens"], len(prompt))
            self.assertTrue(x["same"])
            self.assertEqual(x["reasoning_tokens"], len(self.ids("Only ORD-0042 changes.")) - 1)
            self.assertEqual(x["tokens"], len(script))

    def test_think_needs_room_for_32768_tokens(self):
        r = self.runner(max_ctx=32768)                                        # thinking off, all three fit; with --think none does
        self.assertEqual(editbench.needed_context(r, self.tasks[2], False), 18640)          # the numbers of python/README.md
        self.assertEqual(editbench.needed_context(r, self.tasks[2], True), 39448)
        rows, skipped = editbench.run(r, self.tasks, think=True)
        self.assertEqual(rows, [])
        self.assertEqual([s["task"] for s in skipped], ["edit_code", "edit_jsonl", "edit_jsonl_long"])
        for t, s in zip(self.tasks, skipped):
            prompt, _ = self.fmt.prompt(t.messages, enable_thinking=True)
            self.assertEqual(s["needs_context"], len(prompt) + 32768 + 64)

    def test_a_task_that_does_not_fit_is_skipped(self):
        r = self.runner(max_ctx=12000)
        self.stub.push(self.ids("x"))                                          # the one task that fits
        skipped_seen = []
        rows, skipped = editbench.run(r, self.tasks, on_skip=lambda task, need: skipped_seen.append((task.name, need)))
        self.assertEqual([x["task"] for x in rows], ["edit_jsonl"])
        self.assertEqual([s["task"] for s in skipped], ["edit_code", "edit_jsonl_long"])
        self.assertEqual([s[0] for s in skipped_seen], ["edit_code", "edit_jsonl_long"])
        code_prompt, _ = self.fmt.prompt(self.tasks[0].messages, enable_thinking=False)
        self.assertEqual(skipped[0]["needs_context"], len(code_prompt) + 8192 + 64)


@unittest.skipUnless(POSIX_TOK, "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliEditTests(unittest.TestCase):
    """``yasakani bench --edit`` from the command line, with the stub library and a stand-in nvidia-smi."""

    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder("cli_edit")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        tmp = Path(tempfile.mkdtemp(prefix="fake_smi_"))
        cls.addClassCleanup(shutil.rmtree, tmp, True)
        cls.smi = tmp / "nvidia-smi"
        cls.smi.write_text(FAKE_SMI.replace("{python}", sys.executable), encoding="utf-8")
        cls.smi.chmod(0o755)

    def run_cli(self, *args, name="NVIDIA GeForce RTX 5090", cc="12.0", stub=None):
        gpus = [dict(index=0, uuid="GPU-0", name=name, compute_cap=cc, driver_version="616.92",
                     **{"memory.total": 32607, "memory.used": 1200})]
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_NVIDIA_SMI=str(self.smi), FAKE_GPUS=json.dumps(gpus))
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_"))] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "CUDA_VISIBLE_DEVICES"]:
            env.pop(k, None)
        env.update(stub or {})
        return subprocess.run([sys.executable, "-m", "yasakani", "bench", "--edit", "--model-dir", str(self.folder), "--dll", self.dll,
                               *args], capture_output=True, text=True, timeout=120, cwd=str(ROOT), env=env)

    def test_json(self):
        r = self.run_cli("--max-ctx", "32768", "--json", stub={"STUB_GEN_WIDE_PASSES": "3"})
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(d["test"], "edit")
        self.assertEqual([x["task"] for x in d["runs"]], ["edit_code", "edit_jsonl", "edit_jsonl_long"])
        self.assertTrue(all(x["tokens"] == 41 and not x["same"] and x["wide_passes"] == 3 for x in d["runs"]), d["runs"])   # the stub's default output
        self.assertEqual(d["skipped"], [])
        self.assertEqual(d["gpu"], dict(index=0, name="NVIDIA GeForce RTX 5090", compute_capability="12.0", driver="616.92"))
        self.assertEqual(d["engine"]["sha256"], hashlib.sha256(Path(self.dll).read_bytes()).hexdigest())
        self.assertEqual(d["wide_copy_passes"], "off")              # the stub has no yskn_prepare_wide: an engine library before 0.1.4
        self.assertEqual((d["thinking"], d["greedy"], d["context"]), (False, True, 32768))
        self.assertEqual(set(d["mean_tokens_per_s"]), {"edit_code", "edit_jsonl", "edit_jsonl_long"})
        from yasakani import __version__
        self.assertEqual(d["version"], __version__)

    def test_text(self):
        r = self.run_cli("--max-ctx", "32768", "--runs", "2", name="NVIDIA GeForce RTX 4090", cc="8.9")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        for want in ("yasakani bench --edit: the 3 edit tasks", "GPU 0: NVIDIA GeForce RTX 4090, compute capability 8.9, driver 616.92",
                     "wide copy passes: off", "code edit", "JSONL edit", "long JSONL edit", "differs", "mean tok/s: code edit",
                     "tok/s = (tokens - 1) / seconds"):
            self.assertIn(want, out)
        self.assertEqual(sum(1 for ln in out.splitlines() if ln.startswith(("code edit ", "JSONL edit ", "long JSONL edit "))), 6)

    def test_think(self):
        r = self.run_cli("--max-ctx", "65536", "--think", "--runs", "2")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        for want in ("(thinking on: temperature 1.0, top-p 0.95, top-k 20, seeds from 1 (one per run), output limit 32768 tokens, "
                     "2 runs each)", "mean tok/s: code edit", "With --think the model samples", "thinking columns of the README tables"):
            self.assertIn(want, out)
        self.assertNotIn("2 runs per task", out)
        self.assertEqual(sum(1 for ln in out.splitlines() if ln.startswith(("code edit ", "JSONL edit ", "long JSONL edit "))), 6)
        r = self.run_cli("--max-ctx", "65536", "--think", "--seed0", "10", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual((d["thinking"], d["greedy"], d["output_limit"], d["skipped"]), (True, False, 32768, []))
        self.assertEqual(d["sampling"], dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, first_seed=10))
        self.assertEqual([(x["max_tokens"], x["seed"]) for x in d["runs"]], [(32768, 10), (32768, 11), (32768, 12)])
        r = self.run_cli("--max-ctx", "32768", "--think")                     # enough with thinking off, not with --think
        self.assertEqual(r.returncode, 1, r.stderr)
        for title in ("code edit", "JSONL edit", "long JSONL edit"):
            self.assertIn("%-17s skipped: needs a context of" % title, r.stdout)

    def test_too_small_context_and_wrong_options(self):
        r = self.run_cli("--max-ctx", "12000")
        self.assertEqual(r.returncode, 1, r.stderr)
        for title in ("code edit", "long JSONL edit"):
            self.assertIn("%-17s skipped: needs a context of" % title, r.stdout)
        self.assertEqual(sum(1 for ln in r.stdout.splitlines() if ln.startswith("JSONL edit ")), 1)
        for extra in (["--n", "64"], ["--prompts-file", "x.json"]):
            r = self.run_cli("--max-ctx", "32768", *extra)
            self.assertEqual(r.returncode, 1)
            self.assertIn("do not apply to --edit", r.stderr)
            self.assertNotIn("loading the model", r.stderr)
