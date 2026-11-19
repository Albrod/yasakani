# SPDX-License-Identifier: Apache-2.0
"""Test support: build the CPU-only stub library, make a fake model folder, optional real tokenizer folder."""
import ctypes as C
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from yasakani._native import NativeLib          # noqa: E402
from yasakani.engine import Engine               # noqa: E402
from yasakani.runtime import ModelPack           # noqa: E402

_BUILD = {}


def stub_path(variant: str = "", no_ms: bool = False) -> Path:
    """Compile tests/stub/yskn_stub.c once per run.  ``variant="wide"``: with the 0.1.7 exports of the wide copy passes (-DSTUB_WIDE),
    in a library of its own, so that the tests of older behaviour still see a library without them.  ``no_ms=True``: a build
    without the multi-stream entries, as an older library."""
    key = "path" + variant + ("_no_ms" if no_ms else "")
    if key not in _BUILD:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc or sys.platform == "win32":
            raise unittest.SkipTest("no C compiler for the stub library")
        d = Path(tempfile.mkdtemp(prefix="yskn_stub_"))
        out = d / "libyskn.so"
        flags = (["-DSTUB_WIDE"] if variant == "wide" else []) + (["-DSTUB_NO_MS"] if no_ms else [])
        subprocess.run([cc, "-shared", "-fPIC", "-O1", "-pthread"] + flags + [str(HERE / "stub" / "yskn_stub.c"), "-o", str(out)],
                       check=True)
        _BUILD[key] = out
    return _BUILD[key]


def tokenizer_dir():
    d = os.environ.get("YASAKANI_TEST_MODEL_DIR")
    if d and (Path(d) / "tokenizer.json").is_file():
        return Path(d)
    return None


def make_pack(tmp: Path, name: str = "ok") -> ModelPack:
    root = tmp / name
    root.mkdir(parents=True, exist_ok=True)
    for f in ("qwen38_27b_e3x.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin", "tokenizer.json", "tokenizer_config.json"):
        (root / f).write_bytes(b"x")
    return ModelPack.load(root)


class Stub:
    """The stub library plus its test-only exports."""

    def __init__(self, variant: str = "", no_ms: bool = False):
        self.lib = NativeLib(str(stub_path(variant, no_ms)))
        d = self.lib.dll
        d.stub_push_script.argtypes = [C.c_void_p, C.c_int, C.c_int]
        d.stub_set_delay_us.argtypes = [C.c_int]
        d.stub_get_opt.argtypes = [C.c_void_p, C.c_int]
        d.stub_get_opt.restype = C.c_int
        d.stub_calls.restype = C.c_int
        d.stub_last_cfg.argtypes = [C.c_void_p]
        d.stub_last_cfg.restype = C.c_int
        d.stub_nhist.argtypes = [C.c_void_p]
        d.stub_nhist.restype = C.c_int
        d.stub_last_n_prompt.restype = C.c_int
        d.stub_model.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
        d.stub_ms_draft_mode.argtypes = [C.c_void_p, C.c_int]
        d.stub_ms_fail_in.argtypes = [C.c_void_p, C.c_int]
        for name in ("stub_ms_steps", "stub_ms_drafts", "stub_live_ok", "stub_front"):
            getattr(d, name).argtypes = [C.c_void_p]
            getattr(d, name).restype = C.c_int
        d.stub_ms_front.argtypes = [C.c_void_p, C.c_int]
        d.stub_ms_front.restype = C.c_int
        d.yskn_snap_info.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
        d.yskn_snap_info.restype = C.c_int
        d.stub_clear_scripts.argtypes = []
        d.stub_clear_scripts.restype = None
        for name in ("stub_visual_on", "stub_visual_sets"):
            getattr(d, name).argtypes = [C.c_void_p]
            getattr(d, name).restype = C.c_int
        d.stub_set_visual.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_int]
        d.stub_set_visual.restype = C.c_int
        self.d = d

    def push(self, ids, loop_from=-1):
        a = np.ascontiguousarray(ids, np.int32)
        self.d.stub_push_script(a.ctypes.data, len(a), loop_from)

    def delay_us(self, us):
        self.d.stub_set_delay_us(us)

    def opt(self, engine, key):
        return self.d.stub_get_opt(engine._e, key)

    def last_cfg(self):
        out = np.zeros(64, np.int32)
        n = self.d.stub_last_cfg(out.ctypes.data)
        return out[:n].tolist()

    # ---- the stand-in model and the multi-stream pass (see the notes at the top of stub/yskn_stub.c) ----
    def model(self, engine, on=True, eos_every=0, vocab=64):
        """The stand-in model for engine.generate (the multi-stream passes always use it); about one id in eos_every is a stop id."""
        self.d.stub_model(engine._e, int(on), int(eos_every), int(vocab))

    def draft_mode(self, engine, mode):
        """Drafts of the drafter: 0 = right up to a place that depends on the stream, 1 = all right, 2 = the first one wrong."""
        self.d.stub_ms_draft_mode(engine._e, int(mode))

    def ms_fail_in(self, engine, k):
        """The k-th pass from now fails (an engine fault)."""
        self.d.stub_ms_fail_in(engine._e, int(k))

    def ms_steps(self, engine):
        return self.d.stub_ms_steps(engine._e)

    def ms_drafts(self, engine):
        return self.d.stub_ms_drafts(engine._e)

    def live(self, engine):
        """The carrier's live sequence: (live ids or None, its frontier or -1, length of the saved prompt state)."""
        out = np.zeros(4, np.int32)
        ids = np.zeros(1 << 16, np.int32)
        self.d.yskn_snap_info(engine._e, out.ctypes.data, ids.ctypes.data, len(ids))
        live = ids[:out[3]].tolist() if out[2] else None
        return live, self.d.stub_front(engine._e), int(out[1])

    def slot_front(self, engine, slot):
        return self.d.stub_ms_front(engine._e, int(slot))

    def clear_scripts(self):
        """Drop scripts that no test used (they would come before the stand-in model)."""
        self.d.stub_clear_scripts()

    def give_visual(self, engine):
        """Image input for this engine only: the stub's stand-in for yskn_set_visual (the stub's own table stays without it)."""
        engine._f = dict(engine._f, yskn_set_visual=self.d.stub_set_visual)

    def visual(self, engine):
        """(image input on, calls of set_visual so far)"""
        return self.d.stub_visual_on(engine._e), self.d.stub_visual_sets(engine._e)


def open_engine(tmp: Path, stub: Stub, max_ctx: int = 4096, name: str = "ok") -> Engine:
    return Engine(make_pack(tmp, name), max_ctx=max_ctx, lib=stub.lib)


def make_real_folder(name: str = "model") -> Path:
    """A model folder with the real Qwen3.8 tokenizer files (from YASAKANI_TEST_MODEL_DIR) and placeholder weight files.  The path has
    no letters in its variable part, so it can never contain the words the stub reacts to."""
    src = tokenizer_dir()
    if src is None:
        raise unittest.SkipTest("set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
    root = Path(tempfile.gettempdir()) / ("yct%d_%d" % (os.getpid(), len(list(Path(tempfile.gettempdir()).glob("yct*")))))
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in ("qwen38_27b_e3x.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin"):
        (d / f).write_bytes(b"placeholder-" + f.encode())
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (src / f).is_file():
            shutil.copyfile(src / f, d / f)
    return d
