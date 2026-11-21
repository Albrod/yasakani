# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""inkling2_quant.py - the quantisers of the Inkling-2 conversion (numpy; an optional PyTorch version for a GPU).

Both write the encodings that the experts of the published checkpoint already use (docs/FORMAT.md, section 10):

* NVFP4: value = (e2m1[code] * e4m3[block scale]) * factor, factor = amax / 2688 (float32), one factor per matrix, blocks of 16
  along the columns.  The block scale is chosen among the E4M3 codes base - search .. base + 1, base being the plain rule
  (largest magnitude of the block / 6 / factor, rounded to E4M3), by the smallest squared error of the decoded block; the codes
  are the nearest E2M1 values (ties to the even code), with saturation at 6.
* MXFP8: value = e4m3[byte] * 2 ** (exponent - 127), blocks of 32 along the columns.  The exponent is chosen between
  ceil(log2(largest magnitude / 448)) - 1 and that value, by the smallest squared error.

Every step is the same float32 operation, in the same order, in both versions, so that they give the same bytes.  This includes
the sum of the 16 (32) squared errors of a block, which decides between candidates whose errors are close: it is summed as a
halving tree (x[:8] + x[8:], then x[:4] + x[4:], ...), the order in which PyTorch sums the last axis of such blocks on a CUDA
device, so that the bytes are also those of a plain PyTorch implementation on a GPU.  ``nvfp4_quant`` / ``mxfp8_quant`` take a
float32 matrix [rows, cols] and return the parts as numpy arrays.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np

E4M3_MAX = 448.0
NVFP4_DEN = 2688.0
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), np.float32)


def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float64)
    for b in range(256):
        e, m = (b >> 3) & 0xF, b & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (float("nan") if (e == 15 and m == 7) else (1.0 + m / 8.0) * 2.0 ** (e - 7))
        t[b] = -v if b & 0x80 else v
    return t.astype(np.float32)


E4M3 = _e4m3_table()
THREADS = max(1, min(32, os.cpu_count() or 1))


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """float32 -> E4M3 bytes: saturated to +-448, round to nearest, ties to even; the sign bit is kept (also of -0)."""
    x = np.clip(np.asarray(x, np.float32), np.float32(-E4M3_MAX), np.float32(E4M3_MAX))
    a = np.abs(x)
    e = ((a.view(np.uint32) >> 23).astype(np.int32) - 127)
    np.maximum(e, -6, out=e)
    m = np.rint(a * np.ldexp(np.float32(1.0), 3 - e)).astype(np.int32)        # 0 .. 16 (a power of two: exact)
    normal = m >= 8
    E = np.where(normal, e + 7, 0)
    M = np.where(normal, m - 8, m)
    over = m >= 16
    E = np.where(over, E + 1, E)
    M = np.where(over, 0, M)
    return ((np.signbit(x).astype(np.int32) << 7) | (E << 3) | M).astype(np.uint8)


def e2m1_encode(x: np.ndarray) -> np.ndarray:
    """float32 -> 4-bit E2M1 code: nearest value, ties to the even code, saturation at 6; bit 3 is the sign (x < 0)."""
    mag = np.abs(x)
    idx = np.zeros(mag.shape, np.uint8)
    for b in E2M1_BOUNDS:
        idx += mag > np.float32(b)
    idx += (mag == np.float32(0.75)) | (mag == np.float32(1.75)) | (mag == np.float32(3.5))
    return idx + (x < 0).astype(np.uint8) * np.uint8(8)


def block_sum(sq):
    """Sum over the last axis (16 or 32 values) as a halving tree, in float32 (numpy arrays or PyTorch tensors)."""
    while sq.shape[-1] > 1:
        h = sq.shape[-1] // 2
        sq = sq[..., :h] + sq[..., h:]
    return sq[..., 0]


def _nvfp4_rows(w: np.ndarray, f: np.float32, search: int):
    r, c = w.shape
    blocks = w.reshape(r, c // 16, 16)
    bmax = np.abs(blocks).max(axis=-1)
    per = bmax / (np.float32(6.0) * f)
    per = np.where(np.isfinite(per) & (per > 0), per, np.float32(1.0)).astype(np.float32)
    base = e4m3_encode(np.clip(per, np.float32(2.0 ** -9), np.float32(E4M3_MAX))).astype(np.int32)
    best_err = best_code = best_q = None
    for off in (range(-search, 2) if search > 0 else range(0, 1)):
        code = np.clip(base + off, 1, 126).astype(np.uint8)
        s = E4M3[code]
        inv = np.float32(1.0) / (s * f)
        q = e2m1_encode(blocks * inv[..., None])
        val = (E2M1_VALUES[q] * s[..., None]) * f
        d = val - blocks
        err = block_sum(d * d)
        if best_err is None:
            best_err, best_code, best_q = err, code, q
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_code = np.where(better, code, best_code)
            best_q = np.where(better[..., None], q, best_q)
    q = best_q.reshape(r, c)
    return ((q[:, 1::2] << 4) | q[:, 0::2]).astype(np.uint8), best_code.astype(np.uint8)


def _mxfp8_rows(w: np.ndarray, search: int):
    r, c = w.shape
    blocks = w.reshape(r, c // 32, 32)
    descale = np.abs(blocks).max(axis=-1) / np.float32(E4M3_MAX)
    with np.errstate(divide="ignore"):
        e = np.where(descale > 0, np.ceil(np.log2(descale)), np.float32(-127.0))
    e = np.clip(e, -127, 127).astype(np.int32)
    best_err = best_e = best_b = None
    for off in ((-1, 0) if search > 0 else (0,)):
        ee = np.clip(e + off, -127, 127)
        b = e4m3_encode(blocks * np.ldexp(np.float32(1.0), -ee)[..., None])
        val = E4M3[b] * np.ldexp(np.float32(1.0), ee)[..., None]
        d = val - blocks
        err = block_sum(d * d)
        if best_err is None:
            best_err, best_e, best_b = err, ee, b
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_e = np.where(better, ee, best_e)
            best_b = np.where(better[..., None], b, best_b)
    return best_b.reshape(r, c), (best_e + 127).astype(np.uint8)


def _chunks(rows: int, cols: int, search: int):
    step = max(1, (1 << 18) // (cols * max(1, search + 2) // 8 + 1))
    return [(a, min(rows, a + step)) for a in range(0, rows, step)]


def nvfp4_quant(w: np.ndarray, search: int = 6, threads: int = THREADS) -> Dict[str, np.ndarray]:
    """w float32 [rows, cols] -> {weight [rows, cols/2] uint8, scale [rows, cols/16] E4M3 bytes, amax [1], denominator [1]}."""
    w = np.ascontiguousarray(w, np.float32)
    rows, cols = w.shape
    assert cols % 16 == 0, w.shape
    amax = np.float32(np.abs(w).max()) if w.size else np.float32(0)
    if not amax > 0:
        amax = np.float32(1.0)
    f = np.float32(amax / np.float32(NVFP4_DEN))
    weight = np.empty((rows, cols // 2), np.uint8)
    scale = np.empty((rows, cols // 16), np.uint8)

    def one(span):
        a, b = span
        weight[a:b], scale[a:b] = _nvfp4_rows(w[a:b], f, search)

    _run(one, _chunks(rows, cols, search), threads)
    return {"weight": weight, "scale": scale, "amax": np.array([amax], np.float32),
            "denominator": np.array([NVFP4_DEN], np.float32)}


def mxfp8_quant(w: np.ndarray, search: int = 1, threads: int = THREADS) -> Dict[str, np.ndarray]:
    """w float32 [rows, cols] -> {weight [rows, cols] E4M3 bytes, scale [rows, cols/32] exponent bytes}."""
    w = np.ascontiguousarray(w, np.float32)
    rows, cols = w.shape
    assert cols % 32 == 0, w.shape
    weight = np.empty((rows, cols), np.uint8)
    scale = np.empty((rows, cols // 32), np.uint8)

    def one(span):
        a, b = span
        weight[a:b], scale[a:b] = _mxfp8_rows(w[a:b], search)

    _run(one, _chunks(rows, cols, 2), threads)
    return {"weight": weight, "scale": scale}


def _run(fn, spans, threads):
    if threads <= 1 or len(spans) == 1:
        for s in spans:
            fn(s)
        return
    with ThreadPoolExecutor(threads) as ex:
        for _ in ex.map(fn, spans):
            pass


# ---- the same on a PyTorch device (optional) ------------------------------------------------------------------------------------
def torch_quantisers(device: str):
    """(nvfp4_quant, mxfp8_quant) running on a PyTorch device, with the results moved back to numpy.  Same steps as above."""
    import torch

    F8 = torch.float8_e4m3fn
    dev = torch.device(device)
    bounds = torch.tensor(E2M1_BOUNDS, dtype=torch.float32, device=dev)
    grid = torch.from_numpy(E2M1_VALUES.copy()).to(dev)

    def e4m3_byte(x):
        return x.clamp(-E4M3_MAX, E4M3_MAX).to(F8).view(torch.uint8)

    def e2m1(x):
        mag = x.abs().contiguous()
        idx = torch.searchsorted(bounds, mag, right=False)
        idx = idx + ((mag == 0.75) | (mag == 1.75) | (mag == 3.5)).long()
        return (idx + 8 * (x < 0).long()).to(torch.uint8)

    def nv(w, search=6, threads=None):
        w = torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(dev)
        rows, cols = w.shape
        amax = w.abs().amax().reshape(1)
        amax = torch.where(amax > 0, amax, torch.ones_like(amax))
        f = (amax / torch.tensor(NVFP4_DEN, dtype=torch.float32, device=dev)).float().view(1, 1)
        ws, ss = [], []
        step = max(1, (1 << 22) // (cols * max(1, search + 2)))
        for r0 in range(0, rows, step):
            blocks = w[r0:r0 + step].view(-1, cols // 16, 16)
            per = blocks.abs().amax(-1) / (6.0 * f)
            per = torch.where(torch.isfinite(per) & (per > 0), per, torch.ones_like(per))
            base = e4m3_byte(per.clamp(2.0 ** -9, E4M3_MAX)).long()
            be = bc = bq = None
            for off in (range(-search, 2) if search > 0 else range(0, 1)):
                code = (base + off).clamp(1, 126)
                s = code.to(torch.uint8).view(F8).float()
                inv = 1.0 / (s * f)
                q = e2m1(blocks * inv[..., None])
                val = (grid[q.long()] * s[..., None]) * f[..., None]
                err = block_sum((val - blocks).square())
                if be is None:
                    be, bc, bq = err, code, q
                else:
                    better = err < be
                    be, bc = torch.where(better, err, be), torch.where(better, code, bc)
                    bq = torch.where(better[..., None], q, bq)
            q = bq.reshape(-1, cols)
            ws.append(((q[:, 1::2] << 4) | q[:, 0::2]).to(torch.uint8).cpu())
            ss.append(bc.to(torch.uint8).cpu())
        return {"weight": torch.cat(ws).numpy(), "scale": torch.cat(ss).numpy(), "amax": amax.cpu().numpy().astype(np.float32),
                "denominator": np.array([NVFP4_DEN], np.float32)}

    def mx(w, search=1, threads=None):
        w = torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(dev)
        rows, cols = w.shape
        vs, ss = [], []
        step = max(1, (1 << 23) // cols)
        for r0 in range(0, rows, step):
            blocks = w[r0:r0 + step].reshape(-1, cols // 32, 32)
            descale = blocks.abs().amax(-1) / E4M3_MAX
            e = torch.where(descale > 0, torch.ceil(torch.log2(descale)), torch.full_like(descale, -127.0)).clamp(-127, 127)
            be = bee = bb = None
            for off in ((-1, 0) if search > 0 else (0,)):
                ee = (e + off).clamp(-127, 127)
                b = e4m3_byte(blocks * torch.exp2(-ee)[..., None])
                val = b.view(F8).float() * torch.exp2(ee)[..., None]
                err = block_sum((val - blocks).square())
                if be is None:
                    be, bee, bb = err, ee, b
                else:
                    better = err < be
                    be, bee = torch.where(better, err, be), torch.where(better, ee, bee)
                    bb = torch.where(better[..., None], b, bb)
            vs.append(bb.reshape(-1, cols).cpu())
            ss.append((bee + 127).to(torch.uint8).cpu())
        return {"weight": torch.cat(vs).numpy(), "scale": torch.cat(ss).numpy()}

    return nv, mx
