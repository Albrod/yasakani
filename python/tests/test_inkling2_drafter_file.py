# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2: the converted drafter file (mtp_d4.yskn, docs/FORMAT.md section 10.6).

The converter writes it for --scheme quality (from a random drafter with the layout of the real one, scaled to the tiny model),
the manifest names it, verify checks its SHA-256 and the CRC-32 of its parts, a package of an older kind refuses its format, and
an engine library without the capability generates without drafting (an error with --draft on).  With the host build of the
engine library (YASAKANI_TEST_INK2_DLL) the drafter is loaded packed and greedy output with drafting equals that without.

Needs (else the tests skip): YASAKANI_TEST_INK2_SOURCE (the tiny-inkling2-mini folder); for the engine test also
YASAKANI_TEST_INK2_DLL.
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2 as ink
from yasakani._native import YasakaniError

HERE = Path(__file__).resolve().parent
CONVERT_DIR = HERE.parent.parent / "tools" / "convert"
if not (CONVERT_DIR / "inkling2_convert.py").is_file():
    raise unittest.SkipTest("tools/convert is not beside the package (run these tests from the source tree)")
sys.path.insert(0, str(CONVERT_DIR))
import inkling2_convert as conv  # noqa: E402

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
DLL = os.environ.get("YASAKANI_TEST_INK2_DLL")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "model.safetensors").is_file() and (Path(SOURCE) / "tokenizer.json").is_file()
HAVE_DLL = bool(DLL) and Path(DLL).is_file()
NEED_SOURCE = "set YASAKANI_TEST_INK2_SOURCE to the downloaded tiny-inkling2-mini folder"

TARGET = [5, 14, 22, 30]          # the tiny model: 30 visits, loop passes start at visits 12 and 18
_DIRS = {}


def REV():
    return conv.hf_metadata(Path(SOURCE), "model.safetensors")[0] or "test"


def tiny_mtp_config(text):
    return {"hidden_size": text["hidden_size"], "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 128, "intermediate_size": 512, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "local_layer_ids": [0, 2],
            "sliding_window_size": 32, "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
            "target_layer_ids": TARGET}


def _bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def write_random_drafter(path: Path, mc: dict, w1_rows: int, seed: int = 1234) -> None:
    """A drafter with the tensors of the published mtp.safetensors (BF16), random values, the sizes of ``mc``."""
    rng = np.random.default_rng(seed)
    H, I, r, D = mc["hidden_size"], mc["intermediate_size"], mc["markov_rank"], mc["head_dim"]
    A, C = mc["num_attention_heads"] * D, mc["num_key_value_heads"] * D
    t = {}

    def mat(rows, cols, scale=1.0):
        return (rng.standard_normal((rows, cols), dtype=np.float32) * (scale / np.sqrt(cols))).astype(np.float32)

    def vec(n, base=1.0, s=0.1):
        return (base + s * rng.standard_normal(n, dtype=np.float32)).astype(np.float32)

    t["fuse_weight"] = rng.standard_normal((4, 4), dtype=np.float32)
    for i in range(3):
        p = "layers.%d." % i
        t[p + "dflare_norm.weight"] = vec(H)
        t[p + "input_layernorm.weight"] = vec(H)
        t[p + "post_attention_layernorm.weight"] = vec(H)
        t[p + "self_attn.q_proj.weight"] = mat(A, H, 2.0)
        t[p + "self_attn.k_proj.weight"] = mat(C, H, 2.0)
        t[p + "self_attn.v_proj.weight"] = mat(C, H)
        t[p + "self_attn.o_proj.weight"] = mat(H, A)
        t[p + "self_attn.q_norm.weight"] = vec(D)
        t[p + "self_attn.k_norm.weight"] = vec(D)
        t[p + "self_attn.k_proj_ctx.weight"] = mat(C, H, 2.0)
        t[p + "self_attn.v_proj_ctx.weight"] = mat(C, H)
        t[p + "self_attn.k_norm_ctx.weight"] = vec(D)
        t[p + "mlp.gate_proj.weight"] = mat(I, H)
        t[p + "mlp.up_proj.weight"] = mat(I, H)
        t[p + "mlp.down_proj.weight"] = mat(H, I)
    t["norm.weight"] = vec(H)
    t["mask_embed"] = vec(H, 0.0, 1.0)
    t["markov_head.markov_w1.weight"] = (rng.standard_normal((w1_rows, r), dtype=np.float32) * 0.5).astype(np.float32)
    t["markov_head.rnn_proj.weight"] = mat(3 * r, 2 * r + 2 * H, 2.0)
    t["markov_head.rnn_proj.bias"] = vec(3 * r, 0.0, 0.3)
    t["markov_head.hidden_out.weight"] = mat(H, r, 8.0)
    t["markov_head.hidden_out.bias"] = vec(H, 0.0, 0.5)
    t["markov_head.dflare_norm.weight"] = vec(H)
    header, blobs, off = {}, [], 0
    for name in sorted(t):
        b = _bf16_bits(t[name]).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(t[name].shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def source_with_drafter() -> Path:
    """The tiny checkpoint (hard links) with a config.json that has an mtp_config and a random drafter next to the weights."""
    if "src" not in _DIRS:
        d = Path(tempfile.mkdtemp(prefix="ink2_dsrc_"))

        def link(a, b):
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
        shutil.copytree(SOURCE, d / "src", copy_function=link)
        cfg_p = d / "src" / "config.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        cfg_p.unlink()                                        # a hard link to the download: write a new file, not into it
        mc = tiny_mtp_config(cfg["text_config"])
        cfg["mtp_config"] = mc
        cfg_p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        write_random_drafter(d / "src" / conv.DRAFTER_FILE, mc, int(cfg["text_config"]["vocab_size"]))
        _DIRS["src"] = d
    return _DIRS["src"] / "src"


def quality_folder() -> Path:
    if "q" not in _DIRS:
        d = Path(tempfile.mkdtemp(prefix="ink2_dq_"))
        conv.convert(source_with_drafter(), d / "q", repo="hf-internal-testing/tiny-inkling2-mini", revision=REV(),
                     scheme="quality", threads=4)
        _DIRS["q"] = d
    return _DIRS["q"] / "q"


def tearDownModule():
    for k in list(_DIRS):
        shutil.rmtree(_DIRS.pop(k), ignore_errors=True)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class ConvertedDrafterFolderTests(unittest.TestCase):
    def test_quality_writes_the_converted_drafter(self):
        from yasakani import ink2_yskn as Y
        from yasakani import _ink2_drafter_common as C
        root = quality_folder()
        meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
        d = meta["drafter"]
        self.assertEqual((d["file"], d["format"], d["version"], d["scheme"]), ("mtp_d4.yskn", "yskn", 2, "d4"))
        self.assertFalse((root / conv.DRAFTER_FILE).exists())          # the published file is not placed beside it
        self.assertEqual(d["sha256"], conv.sha256_of(root / "mtp_d4.yskn"))
        self.assertEqual(meta["sha256"]["mtp_d4.yskn"], d["sha256"])
        self.assertEqual(d["source"]["sha256"], conv.sha256_of(source_with_drafter() / conv.DRAFTER_FILE))
        with Y.YsknFile(root / "mtp_d4.yskn") as yf:
            self.assertEqual(yf.model["arch"], C.DRAFTER_ARCH)
            self.assertEqual(yf.model["fold"]["unembed_sha256"], d["folded_with"]["unembed_sha256"])
            encs = {n: yf.entry(n)["encoding"] for n in yf.tensors}
            self.assertEqual(encs["layers.0.mlp.down_proj.weight"], "nvfp4")
            self.assertEqual(encs[C.FOLD_W], "nvfp4")
            self.assertEqual(encs["markov_head.hidden_out.weight"], "bf16")
            self.assertEqual(encs["layers.0.input_layernorm.weight"], "bf16")
            self.assertEqual(yf.check(), [])
            # the fingerprint names the output projection of the folder's own .yskn file
            with Y.YsknFile(root / meta["weights"]["file"]) as mf:
                V = int(json.loads((root / "config.json").read_text(encoding="utf-8"))["text_config"]["unpadded_vocab_size"])
                self.assertEqual(C.unembed_fingerprint(Y.CheckpointView(mf, dense="packed"), V)[1], d["folded_with"]["unembed_sha256"])
        self.assertIn("mtp_d4.yskn) was made from the published mtp.safetensors", (root / "NOTICE").read_text(encoding="utf-8"))
        pack = ink.Ink2Pack.load(root)
        self.assertEqual((pack.drafter, pack.drafter_format, pack.drafter_problem), (root / "mtp_d4.yskn", "yskn", ""))
        self.assertTrue(pack.verify()["mtp_d4.yskn"])
        self.assertEqual(pack.check_drafter(), [])

    def test_conversion_is_reproducible_and_d8_and_published_drafter(self):
        with tempfile.TemporaryDirectory() as t:
            conv.convert(source_with_drafter(), Path(t) / "a", repo="hf-internal-testing/tiny-inkling2-mini", revision=REV(),
                         scheme="quality", threads=2, drafter_scheme="d4")
            self.assertEqual(conv.sha256_of(Path(t) / "a" / "mtp_d4.yskn"), conv.sha256_of(quality_folder() / "mtp_d4.yskn"))
            conv.convert(source_with_drafter(), Path(t) / "b", repo="hf-internal-testing/tiny-inkling2-mini", revision=REV(),
                         scheme="quality", threads=4, drafter_scheme="published")
            meta = json.loads((Path(t) / "b" / "yasakani.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["drafter"]["file"], meta["drafter"]["format"]), ("mtp.safetensors", "hf-safetensors"))
            self.assertNotIn("mtp_d4.yskn", meta["sha256"])
            with self.assertRaises(SystemExit):                  # a converted drafter is for the quality scheme
                conv.convert(source_with_drafter(), Path(t) / "c", repo="hf-internal-testing/tiny-inkling2-mini",
                             revision=REV(), drafter_scheme="d4")

    def test_verify_finds_a_damaged_drafter_part(self):
        from yasakani import ink2_yskn as Y
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(quality_folder(), root)
            p = root / "mtp_d4.yskn"
            with Y.YsknFile(p) as yf:
                off = yf.part_entry("layers.1.self_attn.o_proj.weight", "weight")["offset"] + 11
            with open(p, "r+b") as f:
                f.seek(off)
                b = f.read(1)
                f.seek(off)
                f.write(bytes([b[0] ^ 0x10]))
            pack = ink.Ink2Pack.load(root)
            self.assertFalse(pack.verify()["mtp_d4.yskn"])
            self.assertEqual(pack.check_drafter(), ["layers.1.self_attn.o_proj.weight"])

    def test_formats_and_versions_of_the_entry(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(quality_folder(), root)
            mp = root / "yasakani.json"
            meta = json.loads(mp.read_text(encoding="utf-8"))
            meta["drafter"]["version"] = 3
            mp.write_text(json.dumps(meta), encoding="utf-8")
            pack = ink.Ink2Pack.load(root)
            self.assertIsNone(pack.drafter)
            self.assertIn("version 3", pack.drafter_problem)
            meta["drafter"]["version"], meta["drafter"]["format"] = 2, "other"
            mp.write_text(json.dumps(meta), encoding="utf-8")
            self.assertIn("not supported", ink.Ink2Pack.load(root).drafter_problem)

    def test_library_without_the_capability(self):
        """A library that cannot multiply a converted drafter file: auto and on are told before anything is loaded."""
        pack = ink.Ink2Pack.load(quality_folder())
        calls = []
        native = types.SimpleNamespace(lib=types.SimpleNamespace(caps=3, ink2_draft_abi=lambda: 1),
                                       draft_load=lambda p: calls.append(p), draft_mode=lambda *a: {}, spec_step=None,
                                       spec_stats=lambda *a: {}, reset=lambda: None)
        native.spec_step = lambda *a: None
        eng = ink.Ink2Engine.__new__(ink.Ink2Engine)
        eng.native, eng.fed, eng.last_logits = native, [], None
        note = eng.set_drafter(pack, "auto")
        self.assertIn("cannot multiply", note)
        self.assertIn("generation without drafting", note)
        self.assertFalse(eng.drafting)
        with self.assertRaisesRegex(YasakaniError, "0.3.1 or newer"):
            eng.set_drafter(pack, "on")
        self.assertEqual(calls, [])
        native.lib = types.SimpleNamespace(caps=15, ink2_draft_abi=lambda: 2)
        self.assertTrue(eng.takes_converted_drafter())


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and set YASAKANI_TEST_INK2_DLL to the host build of the library")
class ConvertedDrafterEngineTests(unittest.TestCase):
    def _greedy(self, pack, draft, n=10):
        fmt, eng, runner = ink.open_runner(pack, 160, dll=DLL, warm=False, draft=draft, draft_depth=3 if draft == "on" else None)
        try:
            ids, _ = fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
            logits, _ = eng.read_prompt(ids)
            got = []

            def on_token(tok, got=got):
                got.append(int(tok))
                return ink.END if len(got) >= n else None

            ink.generate(eng, int(np.argmax(logits)), on_token)
            return got, eng.draft_note
        finally:
            eng.close()

    def test_drafting_with_the_converted_file_keeps_the_output(self):
        pack = ink.Ink2Pack.load(quality_folder())
        plain, note_off = self._greedy(pack, "off")
        drafted, note_on = self._greedy(pack, "on")
        self.assertEqual(drafted, plain)
        self.assertIn("mtp_d4.yskn loaded (converted, scheme d4,", note_on)
        self.assertNotIn(" 0 matrices", note_on)
        self.assertIn("--draft off", note_off)


if __name__ == "__main__":
    unittest.main()
