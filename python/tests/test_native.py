# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani import _native


class NativeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.default = self.root / "build" / "bin" / ("yskn.dll" if sys.platform == "win32" else "libyskn.so")
        self.default.parent.mkdir(parents=True)
        self.default.touch()
        patcher = mock.patch.object(_native, "__file__", str(self.root / "python" / "yasakani" / "_native.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"YASAKANI_DLL": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_explicit_path_takes_precedence(self):
        explicit = self.root / "explicit"
        explicit.touch()
        with mock.patch.dict(os.environ, {"YASAKANI_DLL": str(self.default)}):
            self.assertEqual(_native.find_library(str(explicit)), explicit.resolve())
            with self.assertRaises(_native.YasakaniError):
                _native.find_library(str(self.root / "missing"))

    def test_environment_path_takes_precedence(self):
        env = self.root / "environment"
        env.touch()
        with mock.patch.dict(os.environ, {"YASAKANI_DLL": str(env)}):
            self.assertEqual(_native.find_library(), env.resolve())
            env.unlink()
            with self.assertRaises(_native.YasakaniError):
                _native.find_library()

    def test_default_discovery(self):
        self.assertEqual(_native.find_library(), self.default.resolve())

    def test_load_library_from_current_directory(self):
        stub = support.stub_path()
        old = Path.cwd()
        try:
            os.chdir(stub.parent)
            for path in (stub.name, "./" + stub.name):
                with self.subTest(path=path):
                    self.assertEqual(_native.NativeLib(path).path, stub.resolve())
        finally:
            os.chdir(old)
