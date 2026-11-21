# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The model file ``.yskn``, version 2: the weights of an Inkling-2 model in one file, every tensor in its stored encoding
(docs/FORMAT.md, section 10).

* ``YsknFile``: the reader (memory map; every tensor part is a numpy view of the file, nothing is copied).
* ``Writer``: the writer (the converter, tools/convert/inkling2_convert.py).
* ``CheckpointView``: the file seen through the interface of the safetensors reader that the engine binding uses
  (``names``, ``has``, ``info``, ``raw``, ``f32``, ``expert_format``, ``expert_weight``), with the checkpoint's tensor names.
  Experts are handed over packed, as they are stored.  Dense matrices that are stored packed (NVFP4, MXFP8) are handed over
  packed too (``parts()``: the stored bytes) when the engine library says through its capability query that it multiplies them as
  stored (``ink2_capabilities``, bit CAPABILITY_PACKED_DENSE, engine interface version 3); for an older engine they are decoded
  on the CPU to bfloat16.

Number formats (the same as the experts of the published checkpoint):
  NVFP4  weight: two 4-bit E2M1 codes per byte, the even element in the low nibble; scale: one E4M3 byte per 16 consecutive values
         of a row; amax: float32, one per matrix (per expert); denominator: float32 (2688 = 6 * 448)
         value = (e2m1[code] * e4m3[scale]) * (amax / denominator)          (float32, in this order)
  MXFP8  weight: one E4M3 byte per value; scale: one byte per 32 consecutive values of a row
         value = e4m3[weight] * 2 ** (scale - 127)

numpy only.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"YSKN"
VERSION = 2
PREFIX_BYTES = 40            # magic, version, JSON length, data offset, file length (as version 1), JSON CRC-32, reserved
ALIGN = 4096                 # every part starts at a multiple of this (a memory page)
NVFP4_DEN = 2688.0

PART_DTYPES = {"bf16": np.dtype("<u2"), "f32": np.dtype("<f4"), "u8": np.dtype("u1"), "e4m3": np.dtype("u1")}
ENCODINGS = {
    "bf16": (("data", "bf16"),),
    "f32": (("data", "f32"),),
    "nvfp4": (("weight", "u8"), ("scale", "e4m3"), ("amax", "f32"), ("denominator", "f32")),
    "mxfp8": (("weight", "e4m3"), ("scale", "u8")),
}
# the checkpoint's names and dtypes of the parts of an expert tensor ("<prefix>" + suffix)
EXPERT_SUFFIX = {"weight": "_weight", "scale": "_weight_scale", "amax": "_weight_amax", "denominator": "_weight_outer_scale_denominator"}
CKPT_DTYPE = {("nvfp4", "weight"): "U8", ("nvfp4", "scale"): "F8_E4M3", ("nvfp4", "amax"): "F32", ("nvfp4", "denominator"): "F32",
              ("mxfp8", "weight"): "F8_E4M3", ("mxfp8", "scale"): "U8"}
CAPABILITY_PACKED_DENSE = 1   # the bit of the engine's capability query that says "dense matrices may be given packed"
PACKED_DENSE_ABI = 3          # ink2_abi_version() of the first engine with the capability query


class FormatError(ValueError):
    pass


# ---- number formats ------------------------------------------------------------------------------------------------------------------
def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float64)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = float("nan")
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = sign * v
    return t.astype(np.float32)


E4M3 = _e4m3_table()
E2M1 = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), np.float32)
NVFP4_PAIRS = np.stack((E2M1[np.arange(256) & 0xF], E2M1[np.arange(256) >> 4]), axis=-1)     # byte -> (even, odd)
with np.errstate(over="ignore"):
    EXP2 = np.ldexp(np.float32(1.0), np.arange(256, dtype=np.int32) - 127).astype(np.float32)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits (uint16), round to nearest, ties to even (NaN and overflow are not needed for weights)."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((b + np.uint32(0x7FFF) + ((b >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def decode_nvfp4(weight: np.ndarray, scale: np.ndarray, amax: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """weight [..., rows, C/2], scale [..., rows, C/16], amax [n] (one per matrix), denominator [1] -> float32 [..., rows, C]."""
    lead = weight.shape[:-1]
    cols = weight.shape[-1] * 2
    factor = (np.asarray(amax, np.float32) / np.float32(np.asarray(denominator, np.float32).reshape(-1)[0])).astype(np.float32)
    v = NVFP4_PAIRS[np.asarray(weight)].reshape(lead + (cols // 16, 16))
    v = v * E4M3[np.asarray(scale)][..., None]
    if weight.ndim == 3:
        v = v * factor[:, None, None, None]
    else:
        v = v * factor.reshape(-1)[0]
    return v.reshape(lead + (cols,))


def decode_mxfp8(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    lead = weight.shape[:-1]
    cols = weight.shape[-1]
    v = E4M3[np.asarray(weight)].reshape(lead + (cols // 32, 32))
    v = v * EXP2[np.asarray(scale)][..., None]
    return v.reshape(lead + (cols,))


def part_shapes(encoding: str, shape: Sequence[int]) -> Dict[str, Tuple[int, ...]]:
    shape = tuple(int(s) for s in shape)
    if encoding in ("bf16", "f32"):
        return {"data": shape}
    if len(shape) not in (2, 3):
        raise FormatError("a %s tensor has 2 axes (a matrix) or 3 (experts), not %r" % (encoding, shape))
    lead, cols = shape[:-1], shape[-1]
    if encoding == "nvfp4":
        if cols % 16:
            raise FormatError("NVFP4 needs a multiple of 16 columns: %r" % (shape,))
        return {"weight": lead + (cols // 2,), "scale": lead + (cols // 16,), "amax": (shape[0] if len(shape) == 3 else 1,),
                "denominator": (1,)}
    if encoding == "mxfp8":
        if cols % 32:
            raise FormatError("MXFP8 needs a multiple of 32 columns: %r" % (shape,))
        return {"weight": shape, "scale": lead + (cols // 32,)}
    raise FormatError("unknown encoding %r" % encoding)


def _int(value, what: str) -> int:
    """An integer of the header (not a bool, not a float); FormatError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError("%s is not an integer: %r" % (what, value))
    return value


def _sizes(value, what: str) -> Tuple[int, ...]:
    """A shape of the header: a list of integers >= 0; FormatError otherwise."""
    if not isinstance(value, list):
        raise FormatError("%s is not a list of sizes: %r" % (what, value))
    out = tuple(_int(v, what) for v in value)
    if any(v < 0 for v in out):
        raise FormatError("%s has a negative size: %r" % (what, value))
    return out


def _nbytes(dtype: str, shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * PART_DTYPES[dtype].itemsize


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# ---- writer ---------------------------------------------------------------------------------------------------------------------------
class Writer:
    """Plan every tensor first (name, encoding, shape), then write the parts in any order, each in one piece or as a sequence of
    pieces (``stream``).  ``close`` writes the header with the CRC-32 of every part.

        w = Writer(path, model={...}, tensors=[dict(name=..., encoding=..., shape=[...], role=...), ...])
        w.put(name, "data", array)
        with w.stream(name, "weight") as s:  s.write(piece); ...
        w.close()
    """

    def __init__(self, path, model: dict, tensors: List[dict], extra: Optional[dict] = None):
        self.path = str(path)
        entries = []
        for t in tensors:
            enc = t["encoding"]
            shapes = part_shapes(enc, t["shape"])
            parts = [dict(part=p, dtype=dt, shape=list(shapes[p]), offset=0, nbytes=_nbytes(dt, shapes[p]), crc32="00000000")
                     for p, dt in ENCODINGS[enc]]
            e = dict(t)
            e["shape"] = [int(s) for s in t["shape"]]
            e["parts"] = parts
            entries.append(e)
        names = [e["name"] for e in entries]
        if len(set(names)) != len(names):
            raise FormatError("a tensor name is listed twice")
        self.header = {"format": "yskn", "version": VERSION, "alignment": ALIGN, "checksum": "crc32 (zlib) of every part",
                       "model": model, "tensors": entries}
        if extra:
            self.header.update(extra)
        # the header with placeholders as large as any real offset, so that the real header is never longer
        for e in entries:
            for p in e["parts"]:
                p["offset"] = 10 ** 13
        est = len(self._json())
        self.data_offset = _align(PREFIX_BYTES + est)
        off = self.data_offset
        for e in entries:
            for p in e["parts"]:
                p["offset"] = off
                off = _align(off + p["nbytes"])
        last = max((p["offset"] + p["nbytes"] for e in entries for p in e["parts"]), default=self.data_offset)
        self.file_length = last
        self.parts = {(e["name"], p["part"]): p for e in entries for p in e["parts"]}
        self.done = set()
        self.f = open(self.path, "wb")
        self.f.truncate(self.file_length)

    def _json(self) -> bytes:
        return json.dumps(self.header, separators=(",", ":"), ensure_ascii=True).encode("ascii")

    def _part(self, name: str, part: str) -> dict:
        try:
            p = self.parts[(name, part)]
        except KeyError:
            raise FormatError("no part %s of %s in the plan" % (part, name))
        if (name, part) in self.done:
            raise FormatError("part %s of %s written twice" % (part, name))
        return p

    def put(self, name: str, part: str, array) -> None:
        with self.stream(name, part) as s:
            s.write(array)

    def stream(self, name: str, part: str) -> "_PartStream":
        return _PartStream(self, name, part, self._part(name, part))

    def close(self) -> dict:
        missing = [k for k in self.parts if k not in self.done]
        if missing:
            self.f.close()
            raise FormatError("%d parts were not written, first: %s %s" % (len(missing), missing[0][0], missing[0][1]))
        js = self._json()
        assert PREFIX_BYTES + len(js) <= self.data_offset
        prefix = MAGIC + struct.pack("<IQQQII", VERSION, len(js), self.data_offset, self.file_length, zlib.crc32(js), 0)
        assert len(prefix) == PREFIX_BYTES
        self.f.seek(0)
        self.f.write(prefix + js + b"\0" * (self.data_offset - PREFIX_BYTES - len(js)))
        self.f.close()
        return self.header


class _PartStream:
    def __init__(self, w: Writer, name: str, part: str, p: dict):
        self.w, self.key, self.p = w, (name, part), p
        self.dtype = PART_DTYPES[p["dtype"]]
        self.written = 0
        self.crc = 0

    def __enter__(self):
        self.w.f.seek(self.p["offset"])
        return self

    def write(self, array) -> None:
        a = np.ascontiguousarray(array)
        if a.dtype.itemsize != self.dtype.itemsize or a.dtype.kind not in (self.dtype.kind, "u", "i", "f"):
            raise FormatError("%s %s: %s data for a %s part" % (self.key[0], self.key[1], a.dtype, self.p["dtype"]))
        b = memoryview(a).cast("B")
        if self.written + len(b) > self.p["nbytes"]:
            raise FormatError("%s %s: more bytes than the shape %r holds" % (self.key[0], self.key[1], self.p["shape"]))
        self.w.f.seek(self.p["offset"] + self.written)
        self.w.f.write(b)
        self.crc = zlib.crc32(b, self.crc)
        self.written += len(b)

    def finish(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, et, ev, tb):
        if et is None:
            if self.written != self.p["nbytes"]:
                raise FormatError("%s %s: %d of %d bytes written" % (self.key[0], self.key[1], self.written, self.p["nbytes"]))
            self.p["crc32"] = "%08x" % self.crc
            self.w.done.add(self.key)
        return False


# ---- reader ---------------------------------------------------------------------------------------------------------------------------
def read_prefix(path) -> Tuple[int, int]:
    """(version, JSON length) of a file that starts with YSKN; FormatError otherwise."""
    with open(path, "rb") as f:
        head = f.read(PREFIX_BYTES)
    if len(head) < 32 or head[:4] != MAGIC:
        raise FormatError("not a YSKN file (the first four bytes are not 'YSKN')")
    version, json_len = struct.unpack_from("<IQ", head, 4)
    return version, json_len


class YsknFile:
    def __init__(self, path):
        self.path = str(path)
        self.size = os.path.getsize(self.path)
        version, _ = read_prefix(self.path)
        if version != VERSION:
            raise FormatError("YSKN version %d; this reader takes version %d (Inkling-2 files)" % (version, VERSION))
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        if self.size < PREFIX_BYTES:
            raise FormatError("file too short")
        _, json_len, data_off, file_len, json_crc, _r = struct.unpack_from("<IQQQII", self.mm[:PREFIX_BYTES].tobytes(), 4)
        if file_len != self.size:
            raise FormatError("the header says %d bytes, the file has %d (truncated or extended)" % (file_len, self.size))
        if PREFIX_BYTES + json_len > data_off or data_off > self.size:
            raise FormatError("header length out of range")
        js = self.mm[PREFIX_BYTES:PREFIX_BYTES + json_len].tobytes()
        if zlib.crc32(js) != json_crc:
            raise FormatError("the header is damaged (CRC-32 differs)")
        try:
            self.header = json.loads(js.decode("ascii"))
        except ValueError as ex:
            raise FormatError("the header is not JSON: %s" % ex)
        if not isinstance(self.header, dict) or self.header.get("format") != "yskn" or self.header.get("version") != VERSION:
            raise FormatError("the header does not describe a YSKN version 2 file")
        self.data_offset = data_off
        self.model = self.header.get("model") or {}
        if not isinstance(self.model, dict):
            raise FormatError("the header's \"model\" is not an object")
        self.tensors: Dict[str, dict] = {}
        spans = []
        align = _int(self.header.get("alignment") or ALIGN, "the header's \"alignment\"")
        if align <= 0:
            raise FormatError("the header's \"alignment\" is %d" % align)
        tensors = self.header.get("tensors") or []
        if not isinstance(tensors, list):
            raise FormatError("the header's \"tensors\" is not a list")
        for k, e in enumerate(tensors):
            if not isinstance(e, dict) or not isinstance(e.get("name"), str) or not e["name"]:
                raise FormatError("tensor entry %d of the header has no name" % k)
            enc = e.get("encoding")
            if enc not in ENCODINGS:
                raise FormatError("%s: unknown encoding %r" % (e["name"], enc))
            if e.get("role") == "experts" and enc not in ("nvfp4", "mxfp8"):
                raise FormatError("%s: experts stored as %s (expected nvfp4 or mxfp8)" % (e["name"], enc))
            want = part_shapes(enc, _sizes(e.get("shape"), "%s: the shape" % e["name"]))
            parts = e.get("parts")
            if not isinstance(parts, list) or not all(isinstance(p, dict) and isinstance(p.get("part"), str) for p in parts):
                raise FormatError("%s: the parts are not a list of named parts" % e["name"])
            got = {p["part"]: p for p in parts}
            if set(got) != set(want) or len(parts) != len(got):
                raise FormatError("%s: parts %s, expected %s" % (e["name"], sorted(p["part"] for p in parts), sorted(want)))
            for pname, dt in ENCODINGS[enc]:
                p = got[pname]
                what = "%s %s" % (e["name"], pname)
                fits = p.get("dtype") == dt and _sizes(p.get("shape"), what + ": the shape") == want[pname]
                if not fits or _int(p.get("nbytes"), what + ": nbytes") != _nbytes(dt, want[pname]):
                    raise FormatError("%s: dtype, shape or size do not fit the encoding" % what)
                _int(p.get("offset"), what + ": the offset")
                if p["offset"] % align or p["offset"] < data_off or p["offset"] + p["nbytes"] > self.size:
                    raise FormatError("%s %s: offset out of range or not aligned" % (e["name"], pname))
                spans.append((p["offset"], p["offset"] + p["nbytes"], e["name"]))
            if e["name"] in self.tensors:
                raise FormatError("tensor %s listed twice" % e["name"])
            self.tensors[e["name"]] = e
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            if b[0] < a[1]:
                raise FormatError("the data of %s and %s overlap" % (a[2], b[2]))

    def close(self) -> None:
        mm = getattr(self, "mm", None)
        self.mm = None
        if mm is not None and getattr(mm, "_mmap", None) is not None:
            try:
                mm._mmap.close()
            except (BufferError, ValueError):
                pass                                         # views of the file are still alive: it closes with the last one

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- parts -------------------------------------------------------------------------------------------------------------------------
    def entry(self, name: str) -> dict:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError("no tensor %s in %s" % (name, os.path.basename(self.path)))

    def part_entry(self, name: str, part: str) -> dict:
        for p in self.entry(name)["parts"]:
            if p["part"] == part:
                return p
        raise KeyError("tensor %s has no part %s" % (name, part))

    def part(self, name: str, part: str) -> np.ndarray:
        """A view of the stored bytes of one part (no copy)."""
        p = self.part_entry(name, part)
        return self.mm[p["offset"]:p["offset"] + p["nbytes"]].view(PART_DTYPES[p["dtype"]]).reshape(p["shape"])

    def parts(self, name: str) -> Dict[str, np.ndarray]:
        return {p["part"]: self.part(name, p["part"]) for p in self.entry(name)["parts"]}

    def check(self, names: Optional[Iterable[str]] = None, threads: int = 8) -> List[Tuple[str, str]]:
        """CRC-32 of every part of the given tensors (default: all) -> [(tensor, part)] that differ."""
        todo = [(n, p) for n in (names if names is not None else self.tensors) for p in self.entry(n)["parts"]]

        def one(item):
            n, p = item
            view = self.mm[p["offset"]:p["offset"] + p["nbytes"]]
            crc = 0
            for a in range(0, p["nbytes"], 1 << 26):
                crc = zlib.crc32(memoryview(view[a:a + (1 << 26)]), crc)
            return (n, p["part"]) if "%08x" % crc != p["crc32"] else None

        todo.sort(key=lambda it: -it[1]["nbytes"])
        with ThreadPoolExecutor(max(1, threads)) as ex:
            return [r for r in ex.map(one, todo) if r is not None]

    # -- values --------------------------------------------------------------------------------------------------------------------------
    def decode(self, name: str) -> np.ndarray:
        """The float32 values the tensor stands for (exact: the stored numbers, or the decode formula of its encoding)."""
        e = self.entry(name)
        enc = e["encoding"]
        if enc == "bf16":
            return bf16_to_f32(self.part(name, "data"))
        if enc == "f32":
            return np.array(self.part(name, "data"), dtype=np.float32)
        if enc == "nvfp4":
            pr = self.parts(name)
            return decode_nvfp4(pr["weight"], pr["scale"], pr["amax"], pr["denominator"])
        pr = self.parts(name)
        return decode_mxfp8(pr["weight"], pr["scale"])

    def decode_bf16(self, name: str, threads: int = 8, rows: int = 4096) -> np.ndarray:
        """A matrix stored packed, decoded and rounded to bfloat16 (bits, uint16), in pieces of ``rows`` rows on ``threads``
        threads (bounded memory for the large matrices)."""
        e = self.entry(name)
        if e["encoding"] not in ("nvfp4", "mxfp8") or len(e["shape"]) != 2:
            return f32_to_bf16(self.decode(name))
        pr = self.parts(name)
        R = e["shape"][0]
        out = np.empty(tuple(e["shape"]), np.uint16)

        def one(a):
            b = min(R, a + rows)
            if e["encoding"] == "nvfp4":
                v = decode_nvfp4(pr["weight"][a:b], pr["scale"][a:b], pr["amax"], pr["denominator"])
            else:
                v = decode_mxfp8(pr["weight"][a:b], pr["scale"][a:b])
            out[a:b] = f32_to_bf16(v)

        with ThreadPoolExecutor(max(1, threads)) as ex:
            list(ex.map(one, range(0, R, rows)))
        return out


# ---- the interface of the engine binding ------------------------------------------------------------------------------------------
class CheckpointView:
    """A .yskn file with the interface and the tensor names of the safetensors reader of the engine binding.

    dense="bf16": a dense matrix stored packed is decoded and rounded to bfloat16 (``raw`` gives the bits, ``f32`` the rounded
    values): for an engine that multiplies dense matrices from bfloat16 only (interface version 2).  dense="exact": ``f32`` gives the exact decoded float32 values (``raw`` still the
    bfloat16 bits).  dense="packed" (an engine that takes packed matrices): ``info`` gives ("NVFP4" or "MXFP8", logical shape)
    for such a matrix, ``parts`` its stored parts, and ``raw`` refuses.  Experts are always given as stored (``raw`` of ``<prefix>_weight`` etc. is a view of the file).  A decoded
    tensor is made again on every call (nothing is kept)."""

    def __init__(self, yf: YsknFile, dense: str = "bf16"):
        if dense not in ("bf16", "exact", "packed"):
            raise ValueError("dense must be bf16, exact or packed")
        self.yf = yf
        self.path = yf.path
        self.dense = dense
        self.metadata = {"format": "yskn2"}
        self.used = {}
        self._map: Dict[str, Tuple[str, Optional[str]]] = {}    # checkpoint name -> (tensor, part or None = the whole tensor)
        for name, e in yf.tensors.items():
            if e.get("role") == "experts":
                for p in e["parts"]:
                    self._map[name + EXPERT_SUFFIX[p["part"]]] = (name, p["part"])
            else:
                self._map[name] = (name, None)

    # the reader interface ------------------------------------------------------------------------------------------------------
    def names(self) -> List[str]:
        return list(self._map)

    def has(self, name: str) -> bool:
        return name in self._map

    def info(self, name: str):
        t, part = self._map[name]
        e = self.yf.entry(t)
        if part is not None:
            p = self.yf.part_entry(t, part)
            return CKPT_DTYPE[(e["encoding"], part)], tuple(p["shape"])
        enc = e["encoding"]
        if enc in ("nvfp4", "mxfp8") and self.dense == "packed":
            return enc.upper(), tuple(e["shape"])
        return ("F32" if enc == "f32" else "BF16"), tuple(e["shape"])

    def encoding(self, name: str) -> str:
        return self.yf.entry(self._map[name][0])["encoding"]

    def parts(self, name: str) -> Dict[str, np.ndarray]:
        """The stored parts of a dense tensor (views of the file): what an engine that takes packed matrices is given."""
        return self.yf.parts(self._map[name][0])

    def raw(self, name: str, use=None) -> np.ndarray:
        t, part = self._map[name]
        if use is not None:
            self.used[name] = use
        if part is not None:
            return self.yf.part(t, part)
        enc = self.yf.entry(t)["encoding"]
        if enc in ("bf16", "f32"):
            return self.yf.part(t, "data")
        if self.dense == "packed":
            raise ValueError("%s is stored as %s: take its parts (packed mode)" % (name, enc))
        return self.yf.decode_bf16(t)

    def f32(self, name: str, use=None) -> np.ndarray:
        t, part = self._map[name]
        if use is not None:
            self.used[name] = use
        if part is not None:
            dt = self.info(name)[0]
            v = self.yf.part(t, part)
            return np.array(v, np.float32) if dt == "F32" else (E4M3[np.asarray(v)] if dt == "F8_E4M3" else v.astype(np.float32))
        enc = self.yf.entry(t)["encoding"]
        if enc in ("bf16", "f32") or self.dense == "exact":
            return self.yf.decode(t)
        return bf16_to_f32(self.yf.decode_bf16(t))

    def expert_format(self, prefix: str) -> str:
        return self.yf.entry(prefix)["encoding"]

    def expert_weight(self, prefix: str, experts, use=None, official_bf16=False) -> np.ndarray:
        if isinstance(experts, int):
            experts = [experts]
        pr = self.yf.parts(prefix)
        if self.expert_format(prefix) == "nvfp4":
            v = decode_nvfp4(np.asarray(pr["weight"][experts]), np.asarray(pr["scale"][experts]),
                             np.asarray(pr["amax"][experts]), pr["denominator"])
        else:
            v = decode_mxfp8(np.asarray(pr["weight"][experts]), np.asarray(pr["scale"][experts]))
        return bf16_to_f32(f32_to_bf16(v)) if official_bf16 else v

    def unused(self):
        return []


def engine_takes_packed_dense(lib) -> bool:
    """True when the engine library is of interface version 3 or newer (``ink2_abi_version``) and says through
    ``ink2_capabilities`` (bit CAPABILITY_PACKED_DENSE) that its dense matrix products read NVFP4 / MXFP8 tensors as stored."""
    abi = getattr(lib, "abi", None)
    if abi is None:
        try:
            abi = int(lib.ink2_abi_version())
        except (AttributeError, TypeError, ValueError, OSError):
            abi = 1
    if int(abi) < PACKED_DENSE_ABI:
        return False
    fn = getattr(lib, "ink2_capabilities", None)
    if fn is None:
        return False
    try:
        return bool(int(fn()) & CAPABILITY_PACKED_DENSE)
    except (TypeError, ValueError, OSError):
        return False
