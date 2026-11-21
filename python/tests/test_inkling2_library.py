# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""How the package finds and opens the Inkling-2 engine library (yskn_inkling2.dll): --dll, YASAKANI_INK2_DLL, the neighbour of
yskn.dll, and the errors for a missing, foreign or too old library.  No library is loaded (the binding is replaced by a fake)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2 as ink
from yasakani._native import YasakaniError


class FindLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.engine = self.root / "engine"
        self.engine.mkdir()
        (self.engine / "yskn.dll").write_bytes(b"main")
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("YASAKANI_INK2_DLL", "YASAKANI_DLL"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_neighbour_of_the_main_library(self):
        lib = self.engine / ink.LIBRARY
        lib.write_bytes(b"ink2")
        self.assertEqual(ink.find_dll(str(self.engine / "yskn.dll")), lib)      # what the Windows launcher passes
        self.assertEqual(ink.find_dll(str(self.engine)), lib)                    # the engine folder
        os.environ["YASAKANI_DLL"] = str(self.engine / "yskn.dll")
        self.assertEqual(ink.find_dll(None), lib)                                # the main library of the environment

    def test_explicit_library_and_environment_come_first(self):
        (self.engine / ink.LIBRARY).write_bytes(b"ink2")
        own = self.root / "ink2_host.dll"
        own.write_bytes(b"host")
        self.assertEqual(ink.find_dll(str(own)), own)
        os.environ["YASAKANI_INK2_DLL"] = str(own)
        self.assertEqual(ink.find_dll(str(self.engine / "yskn.dll")), own)
        self.assertEqual(ink.find_dll(None), own)

    def test_missing_library(self):
        with self.assertRaises(YasakaniError) as cm:
            ink.find_dll(str(self.engine / "yskn.dll"))
        self.assertIn(ink.LIBRARY, str(cm.exception))
        self.assertIn("0.3.1", str(cm.exception))
        with self.assertRaises(YasakaniError):
            ink.find_dll(str(self.root / "nothing.dll"))
        os.environ["YASAKANI_INK2_DLL"] = str(self.root / "nothing.dll")
        with self.assertRaises(YasakaniError) as cm:
            ink.find_dll(None)
        self.assertIn("YASAKANI_INK2_DLL", str(cm.exception))

    def test_binding_is_the_package_module(self):
        mod = ink.load_binding(ink.find_binding(self.engine / ink.LIBRARY))
        self.assertEqual(mod.__name__, "yasakani._ink2_native")
        self.assertTrue(hasattr(mod.ink2_ckpt, "Checkpoint"))
        self.assertEqual(ink.find_binding(self.engine / ink.LIBRARY).name, "_ink2_native.py")


class OpenLibraryTests(unittest.TestCase):
    class Lib:
        def __init__(self, abi):
            self.abi = abi

    def fake(self, result=None, error=None):
        class Mod:
            @staticmethod
            def open_dll(path):
                if error is not None:
                    raise error
                return result
        return Mod

    def test_version_check(self):
        lib = self.Lib(ink.REQUIRED_ABI)
        self.assertIs(ink.open_library(self.fake(lib), Path("yskn_inkling2.dll")), lib)
        with self.assertRaises(YasakaniError) as cm:
            ink.open_library(self.fake(self.Lib(1)), Path("yskn_inkling2.dll"))
        self.assertIn("too old", str(cm.exception))

    def test_foreign_or_unloadable_library(self):
        with self.assertRaises(YasakaniError) as cm:
            ink.open_library(self.fake(error=AttributeError("function 'ink2_build_info' not found")), Path("yskn.dll"))
        self.assertIn("not the Inkling-2 engine library", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:
            ink.open_library(self.fake(error=OSError("[WinError 126]")), Path("yskn_inkling2.dll"))
        self.assertIn("Visual C++", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
