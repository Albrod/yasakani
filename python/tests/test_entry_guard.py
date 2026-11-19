# SPDX-License-Identifier: Apache-2.0
"""0.1.6 (issue #20): every exported function of src/c_api.cpp that takes the engine handle and can reach CUDA makes the engine's GPU
current on the calling thread first (``YSKN_TRY``, or an explicit ``on_gpu(e)``).

CUDA's current device belongs to the host thread.  The package generates on a new thread for every request and the server answers on
request threads, and a thread that never chose a device runs on CUDA device 0.  This is the likely cause of "graph instantiate failed"
with two GPUs visible and ``--device 1`` (issue #20), found by reading the code and with a stand-in test on one GPU; it is not yet
confirmed on a PC with two GPUs.  This reads the C++ source only (no GPU, no compiler).  The engine zip has no sources: the tests are
skipped there unless YASAKANI_C_API names src/c_api.cpp.
"""
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_CUDA = {"yskn_position", "yskn_cancel", "yskn_kv_format",
           "yskn_edit_stats"}      # read host fields or an atomic only; cancel is called mid-generation


def c_api_path():
    for p in (os.environ.get("YASAKANI_C_API"), ROOT.parent / "src" / "c_api.cpp"):
        if p and Path(p).is_file():
            return Path(p)
    return None


def functions(text):
    """(name, parameters, body) of every ``extern "C"`` function defined in ``text``."""
    for m in re.finditer(r'extern "C"[^\n{]*?\b(yskn_\w+)\s*\(([^)]*)\)\s*\{', text):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        yield m.group(1), m.group(2), text[m.end():i - 1]


def unguarded(text):
    """(number of entry points checked, the ones that do not make the engine's GPU current)."""
    bad, n = [], 0
    macro = re.search(r"#define YSKN_TRY\(body\)(.*)", text)
    if not macro or "on_gpu(e)" not in macro.group(1):
        bad.append("YSKN_TRY")
    for name, params, body in functions(text):
        if "yskn_engine*" not in params.replace(" *", "*") or name in NO_CUDA:
            continue
        n += 1
        if "YSKN_TRY" not in body and "on_gpu(e)" not in body:
            bad.append(name)
    return n, bad


class EntryGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = c_api_path()
        if p is None:
            raise unittest.SkipTest("src/c_api.cpp not found (set YASAKANI_C_API)")
        cls.text = p.read_text(encoding="utf-8")
        cls.bodies = {name: body for name, _, body in functions(cls.text)}

    def test_every_entry_selects_the_gpu(self):
        n, bad = unguarded(self.text)
        self.assertGreaterEqual(n, 50)
        self.assertEqual(bad, [])

    def test_open_remembers_the_device(self):
        self.assertIn("r->dev = device", self.bodies["yskn_open"])

    def test_cancel_stays_light(self):
        for name in NO_CUDA:
            self.assertNotIn("on_gpu", self.bodies[name], name)
            self.assertNotIn("YSKN_TRY", self.bodies[name], name)

    def test_the_check_finds_a_missing_guard(self):
        n, bad = unguarded(self.text.replace("    on_gpu(e);\n", "", 1))
        self.assertEqual(bad, ["yskn_spec_generate"])
        n, bad = unguarded(self.text.replace("try { on_gpu(e); body;", "try { body;", 1))
        self.assertIn("YSKN_TRY", bad)


if __name__ == "__main__":
    unittest.main()
