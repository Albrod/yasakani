# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Shared by the drafter converter (tools/convert/inkling2_convert.py) and the engine wrapper (_ink2_drafter_native.py): the schemes,
which tensors are packed, the fingerprint of the output projection U the folded head was made with, and U's rows as the
engine multiplies them.  numpy only.
"""
import hashlib
import json
import os

import numpy as np

DRAFTER_ARCH = "inkling2-drafter"     # model.arch of a converted drafter file (.yskn version 2)
LARGE_BYTES = 65536                   # a 2-D tensor larger than this (as BF16) is a large matrix
KEEP_BF16 = ("markov_head.hidden_out.weight",)  # read only at load (to fold U again when U differs): kept exact
FOLD_W, FOLD_B = "fold.weight", "fold.bias"     # the folded head U @ hidden_out [V, rank] and U @ bias [V]
# scheme -> encoding of the large matrices (and of the folded head)
SCHEMES = {"d8": "mxfp8", "d4": "nvfp4", "d3": "q3"}
FORMAT_CODE = {"nvfp4": 1, "mxfp8": 2, "q3": 3}   # INK2_DFMT_* of the engine
UNEMBED = "model.llm.unembed.weight"

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def _e4m3_table():
    t = np.zeros(256, np.float64)
    for b in range(256):
        e, m = (b >> 3) & 0xF, b & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (float("nan") if (e == 15 and m == 7) else (1.0 + m / 8.0) * 2.0 ** (e - 7))
        t[b] = -v if b & 0x80 else v
    return t


E4M3_F64 = _e4m3_table()


def is_large(name, shape):
    n = 1
    for s in shape:
        n *= int(s)
    return len(shape) == 2 and n * 2 > LARGE_BYTES and name not in KEEP_BF16


def unembed_source(ckpt, V):
    """(encoding, parts) of U as the engine is given it: ("bf16", {"data": [V, H] u16}) or ("nvfp4" / "mxfp8", parts cut to
    V rows; nvfp4 with "factor" = amax / denominator in float32). ckpt: ink2_ckpt.Checkpoint or a CheckpointView."""
    dt = ckpt.info(UNEMBED)[0]
    if dt in ("NVFP4", "MXFP8") and hasattr(ckpt, "parts"):
        pr = ckpt.parts(UNEMBED)
        out = {"weight": np.asarray(pr["weight"])[:V].view(np.uint8), "scale": np.asarray(pr["scale"])[:V].view(np.uint8)}
        if dt == "NVFP4":
            amax = np.asarray(pr["amax"], dtype=np.float32).reshape(-1)
            den = np.asarray(pr["denominator"], dtype=np.float32).reshape(-1)
            out["factor"] = (amax / den).astype(np.float32).reshape(1)
        return dt.lower(), out
    return "bf16", {"data": np.asarray(ckpt.raw(UNEMBED))[:V]}


def unembed_fingerprint(ckpt, V):
    """SHA-256 of the encoding name and of the bytes of U that the engine is given (V rows), in the order data | weight,
    scale, factor. The folded head of a converted drafter file is only valid for this U."""
    enc, parts = unembed_source(ckpt, V)
    h = hashlib.sha256(enc.encode("ascii") + b"\0")
    for k in ("data", "weight", "scale", "factor"):
        if k in parts:
            a = np.ascontiguousarray(parts[k])
            mv = memoryview(a).cast("B")
            for s in range(0, len(mv), 1 << 26):
                h.update(mv[s:s + (1 << 26)])
    return enc, h.hexdigest()


def unembed_rows(ckpt, V):
    """U as the engine multiplies it: function (start, stop) -> float64 rows (the rule of _ink2_drafter_native.unembed_rows)."""
    enc, parts = unembed_source(ckpt, V)
    if enc == "bf16":
        U = parts["data"]
        return lambda a, b: (np.asarray(U[a:b]).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    w, sc = parts["weight"], parts["scale"]
    if enc == "nvfp4":
        fac = float(parts["factor"][0])

        def rows(a, b):
            ww = np.asarray(w[a:b])
            v = np.empty((ww.shape[0], ww.shape[1] * 2), dtype=np.float64)
            v[:, 0::2] = E2M1[ww & 15]
            v[:, 1::2] = E2M1[ww >> 4]
            return v * np.repeat(E4M3_F64[np.asarray(sc[a:b])], 16, axis=1) * fac
        return rows

    def rows(a, b):
        e = np.ldexp(1.0, np.asarray(sc[a:b]).astype(np.int64) - 127)
        return E4M3_F64[np.asarray(w[a:b])] * np.repeat(e, 32, axis=1)
    return rows


def fold_head(rows, V, out_w, out_b, chunk=4096):
    """U @ hidden_out and U @ bias: float64 products in chunks of 4096 rows, stored as float32 (as the engine wrapper does)."""
    proj = out_w.astype(np.float64)
    bias = out_b.astype(np.float64)
    fw = np.empty((V, proj.shape[1]), dtype=np.float32)
    fb = np.empty(V, dtype=np.float32)
    for s in range(0, V, chunk):
        head = rows(s, min(V, s + chunk))
        fw[s:s + chunk] = (head @ proj).astype(np.float32)
        fb[s:s + chunk] = (head @ bias).astype(np.float32)
    return fw, fb


def main_checkpoint(model_dir, reader_module=None):
    """The main weights of a model folder as the engine wrapper reads them: (reader, description). A folder whose
    yasakani.json names a .yskn file is opened with the .yskn reader (dense matrices packed), else the safetensors file."""
    meta_p = os.path.join(model_dir, "yasakani.json")
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            w = (json.load(f).get("weights") or {})
        if w.get("format") == "yskn":
            if reader_module is None:
                from . import ink2_yskn as reader_module
            yf = reader_module.YsknFile(os.path.join(model_dir, w["file"]))
            return reader_module.CheckpointView(yf, dense="packed"), "yskn " + w["file"]
    from . import _ink2_ckpt as ink2_ckpt
    for n in ("model-00001-of-00001.safetensors", "model.safetensors"):
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            return ink2_ckpt.Checkpoint(p), "safetensors " + n
    raise FileNotFoundError("no main weights in %s" % model_dir)
