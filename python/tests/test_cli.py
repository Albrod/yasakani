# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

from tests import support

ROOT = Path(__file__).resolve().parent.parent
HAVE_TOK = support.tokenizer_dir() is not None
IS_POSIX = os.name == "posix"


def run_cli(*args, input_text=None, timeout=60):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    env.pop("YASAKANI_DLL", None)
    env.pop("YASAKANI_MODEL_DIR", None)
    return subprocess.run([sys.executable, "-m", "yasakani", *args], input=input_text, capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT), env=env)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PrefillOptionTests(unittest.TestCase):
    """--prefill auto | new | old on the commands that open the engine; not given = None (then YASAKANI_PREFILL, else auto)."""

    def test_option(self):
        from yasakani import cli
        p = cli.build_parser()
        for cmd in ("chat", "serve", "bench", "selfcheck"):
            self.assertIsNone(p.parse_args([cmd]).prefill, cmd)
            for mode in ("auto", "new", "old"):
                self.assertEqual(p.parse_args([cmd, "--prefill", mode]).prefill, mode, cmd)
        with open(os.devnull, "w") as null, unittest.mock.patch("sys.stderr", null), self.assertRaises(SystemExit):
            p.parse_args(["chat", "--prefill", "fast"])

    def test_edit_copy_option(self):
        """--edit-copy on | off on the commands that open the engine; not given = None (then YASAKANI_EDIT_COPY, else the default)."""
        from yasakani import cli
        p = cli.build_parser()
        for cmd in ("chat", "serve", "bench", "selfcheck"):
            self.assertIsNone(p.parse_args([cmd]).edit_copy, cmd)
            for mode in ("on", "off"):
                self.assertEqual(p.parse_args([cmd, "--edit-copy", mode]).edit_copy, mode, cmd)
        with open(os.devnull, "w") as null, unittest.mock.patch("sys.stderr", null), self.assertRaises(SystemExit):
            p.parse_args(["chat", "--edit-copy", "auto"])

    def test_wrong_value_fails_first(self):
        """A wrong YASAKANI_PREFILL / YASAKANI_EDIT_COPY / option is reported before anything else can fail (here: no model folder)."""
        from yasakani import cli
        from yasakani._native import YasakaniError
        missing = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no such model folder")
        p = cli.build_parser()
        cases = ((["--prefill", "new"], {"YASAKANI_PREFILL": "", "YASAKANI_EDIT_COPY": "yes"}, "--edit-copy must be one of on, off"),
                 ([], {"YASAKANI_PREFILL": "fast", "YASAKANI_EDIT_COPY": ""}, "--prefill must be one of auto, new, old"),
                 (["--edit-copy", "on"], {"YASAKANI_PREFILL": "0", "YASAKANI_EDIT_COPY": ""}, "--prefill must be one of auto, new, old"),
                 ([], {"YASAKANI_PREFILL": "", "YASAKANI_EDIT_COPY": ""}, "model folder not found"))
        for extra, env, want in cases:
            args = p.parse_args(["chat", "--model-dir", missing, "--dll", "none"] + extra)
            with unittest.mock.patch.dict(os.environ, env), self.assertRaises(YasakaniError) as cm:
                cli._open_stack(args, warm=False)
            self.assertIn(want, str(cm.exception), (extra, env))

    def test_info_lines(self):
        """yasakani info: the setting, and per GPU the line the start would print."""
        from yasakani import cli
        from yasakani import gpu as G
        found = G.parse("0, GPU-0, NVIDIA GeForce RTX 5090, 12.0, 32607, 1000, 616.92\n"
                        "1, GPU-1, NVIDIA GeForce RTX 4090, 8.9, 24564, 1000, 616.92\n")
        with unittest.mock.patch.dict(os.environ, {"YASAKANI_PREFILL": ""}):
            lines = cli._prefill_info_lines(found)
        self.assertEqual(lines[0], "prefill  prompt reading: setting auto (YASAKANI_PREFILL / --prefill)")
        self.assertEqual(lines[1], "         GPU 0: prompt reading: new (setting auto: 32607 MiB card)")
        self.assertTrue(lines[2].startswith("         GPU 1: prompt reading: as in 0.1.7 (setting auto: 24564 MiB card"))
        with unittest.mock.patch.dict(os.environ, {"YASAKANI_PREFILL": "new"}):
            self.assertEqual(cli._prefill_info_lines(found)[2],
                             "         GPU 1: prompt reading: new (setting new; 24564 MiB card, so prompt sizes of 0.1.7)")
        with unittest.mock.patch.dict(os.environ, {"YASAKANI_PREFILL": "fast", "YASAKANI_EDIT_COPY": ""}):
            lines = cli._prefill_info_lines(found)
        self.assertEqual(len(lines), 2)
        self.assertIn("--prefill must be one of auto, new, old", lines[0])
        self.assertEqual(lines[1], "edit     edit copy: setting %s (YASAKANI_EDIT_COPY / --edit-copy)" % G.EDIT_COPY_DEFAULT)
        with unittest.mock.patch.dict(os.environ, {"YASAKANI_PREFILL": "", "YASAKANI_EDIT_COPY": "on"}):
            self.assertEqual(cli._prefill_info_lines(found)[-1], "edit     edit copy: setting on (YASAKANI_EDIT_COPY / --edit-copy)")
        with unittest.mock.patch.dict(os.environ, {"YASAKANI_PREFILL": "", "YASAKANI_EDIT_COPY": "yes"}):
            self.assertIn("--edit-copy must be one of on, off", cli._prefill_info_lines(found)[-1])

    def test_report_row(self):
        """yasakani report: the prompt reading row under the wide passes, when the plan is known."""
        from yasakani import report
        m = dict(package="yasakani x", os="o", python="3",
                 library=dict(file="yskn.dll", sha256="0" * 16, build_info="b"),
                 gpu=dict(name="g", cc="12.0", total_mib=32607, sms=170, code="sm_120", driver="d", driver_model=None, power_limit_w=None,
                          max_sm_mhz=None, max_mem_mhz=None, display_attached=None, display_active=None, pstate=None, temperature_c=None,
                          sm_mhz=None, slowdown_reasons=None, free_mib_now=None),
                 model=dict(id="m", weights="w", context=1),
                 wide=dict(setting="on", status="-", mask=0, widths="-", reason="", cache="-", check_seconds=0.0, free_after_mib=None))
        self.assertNotIn("Prompt reading", report.render(dict(machine=dict(m))))
        self.assertNotIn("Prompt reading", report.render(dict(machine=dict(m, prefill=None))))
        row = "| Prompt reading | prompt reading: new (setting auto: 32607 MiB card) |"
        rows = report.render(dict(machine=dict(m, prefill=dict(reading="new", line="prompt reading: new (setting auto: 32607 MiB card)")))).split("\n")
        self.assertIn(row, rows)
        self.assertTrue(rows[rows.index(row) - 1].startswith("| Wide passes"))
        self.assertNotIn("Edit copy", "\n".join(rows))
        for mode in ("on", "off"):
            self.assertIn("| Edit copy | %s |" % mode, report.render(dict(machine=dict(m, edit_copy=mode))))


@unittest.skipUnless(HAVE_TOK and IS_POSIX, "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder()
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)

    def test_help_and_version(self):
        r = run_cli("--version")
        self.assertEqual(r.returncode, 0)
        self.assertIn("yasakani", r.stdout)
        r = run_cli("serve", "--help")
        self.assertIn("--port", r.stdout)
        r = run_cli()
        self.assertNotEqual(r.returncode, 0)

    def test_info(self):
        r = run_cli("info", "--dll", self.dll, "--model-dir", str(self.folder))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("tokenizer  ok", r.stdout)
        self.assertIn("weights", r.stdout)
        r = run_cli("info", "--dll", "/nonexistent/yskn.dll")
        self.assertEqual(r.returncode, 1)
        self.assertIn("NOT FOUND", r.stdout)
        r = run_cli("info", "--dll", self.dll, "--model-dir", str(self.folder.parent / "missing"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("PROBLEM", r.stdout)

    def test_verify(self):
        d = self.folder.parent / "verify"
        shutil.copytree(self.folder, d, dirs_exist_ok=True)
        sums = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("bdraft2_k8.yskn", "kvq_u5.bin")}
        (d / "yasakani.json").write_text(json.dumps({"sha256": sums}), encoding="utf-8")
        r = run_cli("verify", "--model-dir", str(d))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("all files match", r.stdout)
        (d / "kvq_u5.bin").write_bytes(b"changed")
        r = run_cli("verify", "--model-dir", str(d))
        self.assertEqual(r.returncode, 1)
        self.assertIn("DIFFERENT", r.stdout)
        (d / "yasakani.json").unlink()
        r = run_cli("verify", "--model-dir", str(d))
        self.assertEqual(r.returncode, 2)

    def test_chat(self):
        r = run_cli("chat", "--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096",
                    input_text="/think off\nhello\n/reset\n/bogus\n/exit\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("(thinking off)", r.stdout)
        self.assertIn("tokens/s", r.stdout)
        self.assertIn("complete", r.stdout)
        self.assertIn("(conversation cleared)", r.stdout)
        self.assertIn("commands:", r.stdout)

    def test_chat_ends_on_end_of_input(self):
        r = run_cli("chat", "--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096", "--no-think", input_text="hi\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("complete", r.stdout)

    def test_bench(self):
        pf = self.folder.parent / "prompts.json"
        pf.write_text(json.dumps(["one", {"name": "two", "prompt": "second"}]), encoding="utf-8")
        r = run_cli("bench", "--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096", "--prompts-file", str(pf), "--n", "16",
                    "--runs", "2", "--json")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(len(d["runs"]), 4)
        self.assertTrue(all(x["tokens"] == 16 for x in d["runs"]), d["runs"])
        self.assertEqual({x["test"] for x in d["runs"]}, {"test1", "two"})
        self.assertGreater(d["mean_tokens_per_s"], 0)
        r = run_cli("bench", "--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096", "--n", "8")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("mean over 4 run(s)", r.stdout)

    def test_selfcheck_roundtrip(self):
        ref = self.folder.parent / "ref.json"
        common = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        r = run_cli("selfcheck", *common, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = json.loads(ref.read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in d["cases"]], ["short-answer", "thinking", "plain-text"])
        r = run_cli("selfcheck", *common, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("PASS"), 3)
        d["cases"][1]["sha256"] = "0" * 64                          # a different output
        d["cases"][2]["prompt_sha256"] = "1" * 64                   # a different prompt
        ref.write_text(json.dumps(d), encoding="utf-8")
        r = run_cli("selfcheck", *common, "--reference", str(ref))
        self.assertEqual(r.returncode, 1)
        self.assertIn("output differs", r.stdout)
        self.assertIn("prompt ids differ", r.stdout)
        self.assertIn("FAILED for 2 case(s)", r.stdout)
        r = run_cli("selfcheck", *common)                           # no selfcheck.json in the folder
        self.assertEqual(r.returncode, 2)

    def test_gpu_memory_warning(self):
        from unittest import mock
        from yasakani import cli

        def smi(text):
            return mock.patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=text, stderr=""))

        with mock.patch.object(cli.shutil, "which", return_value="nvidia-smi"):
            with smi("32607, 22003\n"):
                self.assertIsNone(cli._gpu_memory_warning())
            with smi("32607, 31880\n"):
                self.assertIn("almost full (31880 of 32607 MiB", cli._gpu_memory_warning())
            with smi("24564, 24000\n32607, 22003\n"):               # two GPUs: the one the engine was opened on
                self.assertIsNone(cli._gpu_memory_warning(1))
                self.assertIn("almost full", cli._gpu_memory_warning(0))
            with smi("no devices were found\n"):
                self.assertIsNone(cli._gpu_memory_warning())
        with mock.patch.object(cli.shutil, "which", return_value=None):
            self.assertIsNone(cli._gpu_memory_warning())

    def test_bad_inputs(self):
        r = run_cli("chat", "--dll", self.dll)
        self.assertEqual(r.returncode, 1)
        self.assertIn("--model-dir", r.stderr)
        r = run_cli("serve", "--model-dir", str(self.folder), "--dll", self.dll, "--host", "0.0.0.0", "--port", str(free_port()))
        self.assertEqual(r.returncode, 1)
        self.assertIn("without --api-key", r.stderr)
        bad = self.folder.parent / "nofiles"
        bad.mkdir()
        r = run_cli("chat", "--model-dir", str(bad), "--dll", self.dll)
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing", r.stderr)

    def test_serve_port_in_use(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            r = run_cli("serve", "--model-dir", str(self.folder), "--dll", self.dll, "--port", str(port))
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot listen", r.stderr)

    def test_serve_end_to_end(self):
        port = free_port()
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YSKN_STALE_SETTING="1")
        env.pop("YASAKANI_DLL", None)
        proc = subprocess.Popen([sys.executable, "-m", "yasakani", "serve", "--model-dir", str(self.folder), "--dll", self.dll, "--port", str(port),
                                 "--max-ctx", "4096", "--api-key", "k3y"], cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            base = "http://127.0.0.1:%d" % port
            status = None
            t0 = time.time()
            while time.time() - t0 < 40:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                        status = json.loads(resp.read())["status"]
                except Exception:                                   # noqa: BLE001  (not up yet / loading: 503)
                    status = None
                if status == "ok":
                    break
                time.sleep(0.2)
            self.assertEqual(status, "ok", "server did not become ready")
            req = urllib.request.Request(base + "/v1/chat/completions", method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer k3y"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                                                          "chat_template_kwargs": {"enable_thinking": False}}).encode())
            with urllib.request.urlopen(req, timeout=20) as resp:
                d = json.loads(resp.read())
            self.assertEqual(d["choices"][0]["finish_reason"], "length")
            self.assertEqual(d["usage"]["completion_tokens"], 8)
            req = urllib.request.Request(base + "/v1/models")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 401)
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=20)
            self.assertEqual(proc.returncode, 0, err)
            self.assertIn("stopping", err)
            self.assertIn("ignored 1 YSKN_* setting", err)           # the stale setting from the shell was dropped
            self.assertIn("API key required", err)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


if __name__ == "__main__":
    unittest.main()
