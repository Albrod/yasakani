# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2 Mini (``"architecture": "inkling2"`` in ``yasakani.json``): model folder, chat template, output stream, sampling and
the chat runner on the Inkling-2 engine library.

Text only, one request at a time.  The engine library is a separate DLL (``yskn_inkling2.dll``, next to ``yskn.dll`` in the engine
zip) with its own C API; its ctypes binding is the package module ``_ink2_native.py`` (with the safetensors reader ``_ink2_ckpt.py``).
The library is found as ``find_dll`` says.  Model folders are made by ``tools/convert/inkling2_convert.py``.

Two weights formats (the ``weights`` entry of yasakani.json): ``hf-safetensors`` (the published file, read by the binding) and
``yskn`` (a re-quantised ``.yskn`` file of version 2, yasakani/ink2_yskn.py).  For a ``.yskn`` file the CRC-32 of every tensor is
checked first (``YASAKANI_INK2_SKIP_TENSOR_CHECK=1`` skips it), and the binding is given the file through an object with the interface of its safetensors reader: the experts as
stored, the dense matrices that are stored packed decoded to bfloat16 (what the engine multiplies today), unless the engine says
through its capability query that it takes them packed.

How the model writes a reply (``response_template`` of its tokenizer_config.json): blocks ``<|message_model|>[name]<|content_X|>
text<|end_message|>``, where X is ``thinking`` (reasoning) or ``text`` (the answer), and ``<|content_model_end_sampling|>`` (the
stop token) at the end of the turn.  The prompt ends after the messages, without an assistant header: the model writes it.
Thinking controls: ``reasoning_effort`` goes to the template's "Thinking effort level" line (none, minimal, low, medium, high,
xhigh, max or a number from 0 to 0.99; default 0.9).  The model card says that even the lowest effort does not fully turn the
thinking off, so ``enable_thinking: false`` sets the effort to 0 and starts the reply with an answer block
(``<|message_model|><|content_text|>``) so that no thinking is written.

The drafter (the ``drafter`` entry of yasakani.json: the model's published ``mtp.safetensors``) proposes a few tokens that the
model checks in one pass (speculative decoding).  The output does not change: greedy gives the same tokens, sampling the same
distribution (and with a fixed seed the same tokens, see ``Sampler.seek``).  ``--draft auto | on | off`` / ``$YASAKANI_DRAFT``:
auto (default) lets the engine use the drafter only while it makes generation faster, on always drafts, off does not load it.
A folder without the drafter, or an engine library without the drafter calls, generates without drafting.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._native import YasakaniError

ARCHITECTURE = "inkling2"
MANIFEST = "yasakani.json"
WEIGHTS_FORMATS = ("hf-safetensors", "yskn")
CONTEXT_MARGIN = 16                    # positions kept free at the end of the context
DRAFT_MODES = ("auto", "on", "off")
DRAFT_ENV, DRAFT_DEPTH_ENV = "YASAKANI_DRAFT", "YASAKANI_DRAFT_DEPTH"
MAX_DRAFT_DEPTH = 7
DRAFTER_FORMATS = ("hf-safetensors",)
DRAFT_CALLS = ("draft_load", "draft_mode", "spec_step", "spec_stats")   # what the engine wrapper needs for drafting
# reasoning_effort values of the chat template (its effort_map); the server and the chat accept these names
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {e: e for e in EFFORTS}

# special tokens of the reply format (names; the ids are read from tokenizer.json)
T_MODEL, T_THINKING, T_TEXT, T_END, T_STOP = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
                                              "<|content_model_end_sampling|>")
ANSWER_OPENERS = ("<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>", "<|content_invoke_tool_json|>")


def architecture_of(folder) -> Optional[str]:
    """The ``architecture`` of the folder's yasakani.json; None when there is no readable manifest or it names none (Qwen3.8)."""
    try:
        meta = json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return meta.get("architecture") if isinstance(meta, dict) else None


def is_inkling2(folder) -> bool:
    return bool(folder) and architecture_of(folder) == ARCHITECTURE


# ---- model folder -------------------------------------------------------------------------------------------------------------------
@dataclass
class Ink2Pack:
    root: Path
    meta: dict
    weights: Path
    weights_format: str
    model_id: str
    max_ctx: int
    default_ctx: int
    sampling: Dict[str, float]
    stop_ids: List[int]
    sha256: Dict[str, str] = field(default_factory=dict)
    sizes: Dict[str, int] = field(default_factory=dict)
    drafter_entry: dict = field(default_factory=dict)       # the "drafter" entry of yasakani.json ({} when there is none)
    drafter: Optional[Path] = None                          # its file, when it is there and of a known format
    drafter_problem: str = ""                               # why an entry that is listed is not used

    @classmethod
    def load(cls, root) -> "Ink2Pack":
        root = Path(root)
        if not root.is_dir():
            raise YasakaniError("model folder not found: %s" % root)
        try:
            meta = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise YasakaniError("cannot read %s in %s: %s" % (MANIFEST, root, ex))
        if meta.get("architecture") != ARCHITECTURE:
            raise YasakaniError("%s does not describe an Inkling-2 model folder" % MANIFEST)
        w = meta.get("weights") or {}
        if not isinstance(w, dict) or not w.get("file"):
            raise YasakaniError("%s has no weights entry" % MANIFEST)
        fmt = str(w.get("format") or "hf-safetensors")
        if fmt not in WEIGHTS_FORMATS or (fmt == "yskn" and int(w.get("version") or 2) != 2):
            raise YasakaniError("weights format %r is not supported by this version of the package" % fmt)
        need = [w["file"], "config.json", "tokenizer.json", "chat_template.jinja"]
        missing = [n for n in need if not (root / n).is_file()]
        if missing:
            raise YasakaniError("model folder %s is missing: %s" % (root, ", ".join(missing)))
        max_ctx = int(meta.get("max_ctx") or 32768)
        s = meta.get("sampling") or {}
        d = meta.get("drafter")
        d = dict(d) if isinstance(d, dict) and d.get("file") else {}
        drafter, problem = None, ""
        if d:
            if str(d.get("format") or "hf-safetensors") not in DRAFTER_FORMATS:
                problem = "drafter format %r is not supported by this version of the package" % d.get("format")
            elif not (root / d["file"]).is_file():
                problem = "the drafter file %s is missing" % d["file"]
            else:
                drafter = root / d["file"]
        return cls(root=root, drafter_entry=d, drafter=drafter, drafter_problem=problem, meta=meta, weights=root / w["file"], weights_format=fmt, model_id=str(meta.get("model_id") or "inkling-2"),
                   max_ctx=max_ctx, default_ctx=min(int(meta.get("default_ctx") or max_ctx), max_ctx),
                   sampling=dict(temperature=float(s.get("temperature", 1.0)), top_p=float(s.get("top_p", 1.0)),
                                 top_k=int(s.get("top_k", 0) or 0)),
                   stop_ids=[int(i) for i in (meta.get("stop_token_ids") or [])],
                   sha256=dict(meta.get("sha256") or {}), sizes={k: int(v) for k, v in (meta.get("sizes") or {}).items()})

    def verify(self, progress=None) -> Dict[str, bool]:
        """Size, then SHA-256 of every listed file (the weights file is about 20 GB: a minute or so)."""
        res = {}
        sums, sizes = dict(self.sha256), dict(self.sizes)
        d = self.drafter_entry
        if d.get("sha256") and d["file"] not in sums:            # the drafter entry carries its own checksum
            sums[d["file"]] = str(d["sha256"])
            if d.get("size") is not None:
                sizes[d["file"]] = int(d["size"])
        for name, want in sorted(sums.items(), key=lambda kv: (sizes.get(kv[0], 0), kv[0])):
            p = self.root / name
            ok = p.is_file() and (name not in sizes or p.stat().st_size == sizes[name])
            if ok:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(1 << 24), b""):
                        h.update(b)
                ok = h.hexdigest() == want
            res[name] = ok
            if progress:
                progress(name, ok)
        return res

    def check_tensors(self, threads: int = 8) -> List[str]:
        """A .yskn weights file: the CRC-32 of every tensor part -> the names of the tensors that differ (empty: all match)."""
        if self.weights_format != "yskn":
            return []
        from .ink2_yskn import YsknFile
        with YsknFile(self.weights) as yf:
            return sorted({n for n, _ in yf.check(threads=threads)})


# ---- chat template and tokenizer ---------------------------------------------------------------------------------------------------
_GENERATION_TAG = re.compile(r"\{%-?\s*(end)?generation\s*-?%\}")      # a tag of the transformers template environment (no effect)


def _raise_exception(message):
    from jinja2.exceptions import TemplateError
    raise TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Ink2Formatter:
    """chat_template.jinja + tokenizer.json of one Inkling-2 folder."""

    def __init__(self, folder):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        d = Path(folder)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        source = _GENERATION_TAG.sub("", (d / "chat_template.jinja").read_text(encoding="utf-8"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        self.template = env.from_string(source)
        ids = {}
        for name in (T_MODEL, T_THINKING, T_TEXT, T_END, T_STOP) + ANSWER_OPENERS:
            i = self.tok.token_to_id(name)
            if i is None:
                raise YasakaniError("the tokenizer has no token %s (not an Inkling-2 tokenizer?)" % name)
            ids[name] = int(i)
        self.ids = ids
        cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        self.special = frozenset(int(t["id"]) for t in cfg.get("added_tokens", []) if t.get("special"))
        self.answer_prefix = [ids[T_MODEL], ids[T_TEXT]]

    def render(self, messages: Sequence[dict], **template_kwargs) -> str:
        return self.template.render(messages=list(messages), add_generation_prompt=True, tools=None, **template_kwargs)

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def prompt(self, messages: Sequence[dict], template_kwargs: Optional[dict] = None) -> Tuple[List[int], bool]:
        """(prompt ids, the reply starts inside the answer).  ``enable_thinking`` False: effort 0 and an answer block opened in the
        prompt (see the module text).  A wrong effort raises the template's error (jinja2 TemplateError)."""
        kw = dict(template_kwargs or {})
        thinking = kw.pop("enable_thinking", None)
        if thinking is False:
            kw["reasoning_effort"] = "none"
        ids = self.encode(self.render(messages, **kw))
        if thinking is False:
            return ids + self.answer_prefix, True
        return ids, False


class StreamDecoder:
    """Incremental detokenizer (as yasakani.textio.StreamDecoder): ``push`` returns the text that became complete."""

    def __init__(self, tok):
        self.tok = tok
        self.ids: List[int] = []
        self.prefix = 0
        self.read = 0

    def _dec(self, a: int, b: Optional[int] = None) -> str:
        return self.tok.decode(self.ids[a:b], skip_special_tokens=False)

    def push(self, new_ids) -> str:
        self.ids.extend(int(i) for i in new_ids)
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        if len(self.ids) == self.read:
            return ""
        old = self._dec(self.prefix, self.read)
        new = self._dec(self.prefix)
        self.prefix = self.read = len(self.ids)
        return new[len(old):]


class OutputStream:
    """Reply ids -> ``("reasoning", text)`` / ``("content", text)`` pieces, one piece per id that completes some text.

    Text inside ``<|content_thinking|>`` blocks is reasoning; text inside ``<|content_text|>`` (and the other answer blocks) is the
    answer; the block headers (``<|message_model|>`` and the name after it), ``<|end_message|>``, the stop token and every other
    special token are not shown."""

    def __init__(self, fmt: Ink2Formatter, in_answer: bool = False):
        self.fmt = fmt
        i = fmt.ids
        self.opens = {i[T_THINKING]: "reasoning"}
        self.opens.update({i[n]: "content" for n in ANSWER_OPENERS})
        self.t_model, self.t_end = i[T_MODEL], i[T_END]
        self.mode = "content" if in_answer else "head"      # head: between blocks (headers and names are not shown)
        self.dec = StreamDecoder(fmt.tok)
        self.n = 0
        self.reasoning_tokens = 0
        self.answer_at: Optional[int] = 0 if in_answer else None
        self.saw_reasoning = False

    def feed(self, token: int) -> List[Tuple[str, str]]:
        token = int(token)
        out: List[Tuple[str, str]] = []
        k = self.n
        self.n += 1
        if token in self.fmt.special:
            if self.mode in ("reasoning", "content"):
                tail = self.dec.flush()
                if tail:
                    out.append((self.mode, tail))
            if token in self.opens:
                self.mode = self.opens[token]
                if self.mode == "reasoning":
                    self.saw_reasoning = True
                elif self.answer_at is None:
                    self.answer_at = k
            elif token in (self.t_model, self.t_end):
                self.mode = "head"
            self.dec = StreamDecoder(self.fmt.tok)
            if self.answer_at is None:
                self.reasoning_tokens = self.n
            return out
        if self.answer_at is None:
            self.reasoning_tokens = self.n
        if self.mode in ("reasoning", "content"):
            piece = self.dec.push((token,))
            if piece:
                out.append((self.mode, piece))
        return out

    def flush(self) -> List[Tuple[str, str]]:
        if self.mode in ("reasoning", "content"):
            tail = self.dec.flush()
            if tail:
                return [(self.mode, tail)]
        return []

    @property
    def reasoning_count(self) -> int:
        return self.reasoning_tokens if self.saw_reasoning else 0


# ---- sampling ---------------------------------------------------------------------------------------------------------------------
class Sampler:
    """Greedy (temperature 0) or temperature / top-k / top-p / min-p sampling with a seeded generator (numpy).  Exact: the
    vocabulary is cut only where the rule says so; the draw goes through block sums so that a token costs well under a
    millisecond on 200,000 entries."""

    BLOCK = 1024

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0, seed: int = 0):
        self.t, self.top_p, self.top_k, self.min_p = float(temperature), float(top_p), int(top_k or 0), float(min_p or 0.0)
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        self._state0 = self.rng.bit_generator.state
        self._k = 0                                          # draws made since the seed

    def seek(self, k: int) -> None:
        """The next draw is draw number ``k`` of the seed (one draw per token): the token at output index k is drawn with the same
        random number whether earlier draws were made for drafted rows that were not kept, or not."""
        k = int(k)
        if k != self._k:
            bg = self.rng.bit_generator
            bg.state = self._state0
            if k:
                bg.advance(k)                                # PCG64: random() takes one 64-bit output
            self._k = k

    def _draw(self, p: np.ndarray) -> int:
        """Index drawn with probability p[i] / sum(p) (p >= 0, float32)."""
        n, B = p.size, self.BLOCK
        nb = -(-n // B)
        if nb * B != n:
            p = np.concatenate([p, np.zeros(nb * B - n, p.dtype)])
        cb = np.cumsum(p.reshape(nb, B).sum(axis=1, dtype=np.float64))
        r = self.rng.random() * cb[-1]
        self._k += 1
        b = min(int(np.searchsorted(cb, r, side="right")), nb - 1)
        while b > 0 and cb[b] == cb[b - 1]:                 # an empty block (rounding at its edge): the one before
            b -= 1
        r -= cb[b - 1] if b else 0.0
        blk = p[b * B:(b + 1) * B]
        c = np.cumsum(blk, dtype=np.float64)
        j = min(int(np.searchsorted(c, r, side="right")), B - 1)
        while j > 0 and blk[j] == 0.0:
            j -= 1
        return b * B + j

    def __call__(self, logits: np.ndarray) -> int:
        if self.t <= 0.0:
            return int(np.argmax(logits))
        x = np.asarray(logits, np.float32)
        idx = None
        if 0 < self.top_k < x.size:
            idx = np.argpartition(x, -self.top_k)[-self.top_k:]
            x = x[idx]
        p = np.exp((x - x.max()) * np.float32(1.0 / self.t))      # the most likely token has p = 1
        if self.min_p > 0.0:
            p[p < self.min_p] = 0.0
        if self.top_p < 1.0:
            total = float(p.sum(dtype=np.float64))
            want = self.top_p * total
            thr = 1e-4
            while True:                                      # every token above thr: if they hold top_p, the nucleus is among them
                cand = np.flatnonzero(p >= thr) if thr > 0 else np.arange(p.size)
                if thr <= 0 or float(p[cand].sum(dtype=np.float64)) >= want:
                    break
                thr = thr * 1e-4 if thr > 1e-30 else 0.0
            cand = cand[np.argsort(-p[cand], kind="stable")]
            cum = np.cumsum(p[cand], dtype=np.float64)
            keep = max(1, min(int(np.searchsorted(cum - p[cand], want, side="left")), cand.size))
            cand = cand[:keep]
            j = int(cand[self._draw(p[cand])])
        else:
            j = self._draw(p)
        return int(idx[j]) if idx is not None else j


# ---- engine ------------------------------------------------------------------------------------------------------------------------
LIBRARY = "yskn_inkling2.dll"          # the Inkling-2 engine library, next to yskn.dll in the engine zip
MAIN_LIBRARIES = ("yskn.dll", "libyskn.so")
REQUIRED_ABI = 2                       # ink2_abi_version() of the library this binding was written for (or newer)


def find_dll(path: Optional[str] = None) -> Path:
    """The Inkling-2 engine library, in this order: ``path`` (``--dll``) when it names a file other than the main library
    (yskn.dll), or a folder that holds yskn_inkling2.dll; ``$YASAKANI_INK2_DLL``; yskn_inkling2.dll next to the main library
    (``--dll`` naming yskn.dll, as the Windows launcher passes it, else ``$YASAKANI_DLL`` or the bin folder next to the package)."""
    tried = []
    main = None
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / LIBRARY
        elif p.name.lower() in MAIN_LIBRARIES:
            main = p
            p = None
        if p is not None:
            if p.is_file():
                return p
            raise YasakaniError("Inkling-2 engine library not found: %s" % p)
    env = os.environ.get("YASAKANI_INK2_DLL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise YasakaniError("Inkling-2 engine library not found: %s (YASAKANI_INK2_DLL)" % p)
    if main is None:
        from ._native import find_library
        try:
            main = find_library(None)
        except YasakaniError:
            main = None
    if main is not None:
        p = main.parent / LIBRARY
        if p.is_file():
            return p
        tried.append(str(p))
    raise YasakaniError("the Inkling-2 engine library (%s) was not found%s.  It is in the engine zip of YASAKANI 0.3.0 or newer, next "
                        "to yskn.dll: update the engine (START.cmd does this), or give its path with --dll or YASAKANI_INK2_DLL"
                        % (LIBRARY, " (looked for %s)" % "; ".join(tried) if tried else ""))


def find_binding(dll: Path) -> Path:
    """The binding of the engine library: the package module _ink2_native.py (nothing is loaded from outside the package)."""
    return Path(__file__).resolve().parent / "_ink2_native.py"


def load_binding(path: Optional[Path] = None):
    from . import _ink2_native
    return _ink2_native


def open_library(mod, dll: Path):
    """Opens the engine library through the binding and checks that it is the Inkling-2 library of a version this package can use."""
    try:
        lib = mod.open_dll(str(dll))
    except OSError as ex:
        from ._native import _VC_HINT
        raise YasakaniError("%s: %s%s" % (dll.name, _VC_HINT, ex)) from ex
    except AttributeError as ex:
        raise YasakaniError("%s is not the Inkling-2 engine library (%s)" % (dll, ex)) from ex
    abi = int(getattr(lib, "abi", 1))
    if abi < REQUIRED_ABI:
        raise YasakaniError("%s is too old for this package (interface version %d, %d or newer is needed): use the yskn_inkling2.dll "
                            "of the same release as the package" % (dll, abi, REQUIRED_ABI))
    return lib


def draft_mode_of(value: Optional[str] = None, environ=None) -> str:
    """auto | on | off: ``value`` (--draft), else ``$YASAKANI_DRAFT``, else auto."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(DRAFT_ENV, "")
    v = (v or "auto").strip().lower()
    if v not in DRAFT_MODES:
        raise YasakaniError("%s / --draft must be one of %s, not %r" % (DRAFT_ENV, ", ".join(DRAFT_MODES), v))
    return v


def draft_depth_of(value=None, environ=None) -> Optional[int]:
    """1 .. 7 tokens proposed per pass: ``value`` (--draft-depth), else ``$YASAKANI_DRAFT_DEPTH``, else None (the engine's)."""
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else (env.get(DRAFT_DEPTH_ENV) or "").strip()
    if v in (None, ""):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= MAX_DRAFT_DEPTH:
        raise YasakaniError("%s / --draft-depth must be a number from 1 to %d, not %r" % (DRAFT_DEPTH_ENV, MAX_DRAFT_DEPTH, v))
    return n


def backend_of(lib_info: str) -> str:
    b = (os.environ.get("YASAKANI_INK2_BACKEND") or "").strip().lower()
    if b:
        if b not in ("cuda", "host"):
            raise YasakaniError("YASAKANI_INK2_BACKEND must be cuda or host, not %r" % b)
        return b
    return "cuda" if "cuda" in lib_info else "host"


MAX_LAYERS = 64                  # the engine library takes at most this many layers (num_hidden_layers of config.json)


def read_config(root) -> dict:
    """config.json of a model folder, with the values the package and the engine rely on checked -> the parsed file.
    A value that the engine library does not take gives a YasakaniError here, before the library is called."""
    path = Path(root) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise YasakaniError("cannot read config.json in %s: %s" % (Path(root), ex))
    t = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(t, dict):
        raise YasakaniError("config.json in %s has no text_config object" % Path(root))
    n = t.get("num_hidden_layers", 24)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_LAYERS:
        raise YasakaniError("config.json: num_hidden_layers is %r; this version of the engine takes 1 to %d layers" % (n, MAX_LAYERS))
    types = t.get("layer_types")
    if types is not None and (not isinstance(types, list) or len(types) != n or not all(isinstance(x, str) for x in types)):
        raise YasakaniError("config.json: layer_types is not a list of %d names (one per layer)" % n)
    if t.get("loop_config") is not None and not isinstance(t["loop_config"], dict):
        raise YasakaniError("config.json: loop_config is not an object")
    return config


class Ink2Engine:
    """The engine library with the state of one sequence: the ids it has read since the last reset, so that a prompt that only
    extends them (the next turn of a chat) reads only the new ids."""

    def __init__(self, pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, graph: bool = True, log=None):
        read_config(pack.root)                               # values the library does not take stop here, with a message
        self.dll = find_dll(dll)
        self.binding_path = find_binding(self.dll)
        self.mod = load_binding(self.binding_path)
        lib = open_library(self.mod, self.dll)
        self.build_info = lib.ink2_build_info().decode("utf-8", "replace")
        self.backend = backend_of(self.build_info)
        self.max_ctx = int(max_ctx)
        self._cancel = threading.Event()
        t0 = time.perf_counter()
        self.weights_note = pack.weights_format
        self._yskn = None
        try:
            if pack.weights_format == "yskn":
                self.native = self._load_yskn(pack, lib, log)
            else:
                self.native = self.mod.Ink2Native().load(str(pack.root), backend=self.backend, max_positions=self.max_ctx,
                                                         dll=str(self.dll), weights=pack.weights.name, log=log)
        except (RuntimeError, OSError, AssertionError, KeyError, ValueError) as ex:
            if self._yskn is not None:
                self._yskn.close()
                self._yskn = None
            raise YasakaniError("the Inkling-2 engine could not load the model: %s" % ex) from ex
        if graph and self.backend == "cuda":
            try:
                self.native.set_option("graph", 1)
            except RuntimeError:
                pass
        self.load_seconds = time.perf_counter() - t0
        self.vocab = int(self.native.vocab)
        self.fed: List[int] = []
        self.last_logits: Optional[np.ndarray] = None
        self.drafting = False                                # a drafter is loaded (set_drafter)
        self.draft_mode = "off"
        self.draft_depth = 0
        self.draft_note = "drafter: not set up"

    # -- drafter ------------------------------------------------------------------------------------------------------------------
    def set_drafter(self, pack: "Ink2Pack", mode: str = "auto", depth: Optional[int] = None) -> str:
        """Load the model folder's drafter through the engine wrapper (mode auto or on) -> the line for the start-up output.
        Without a drafter in the folder, or with a wrapper that has no drafter calls, generation goes on without drafting; with
        ``mode == "on"`` a drafter that is there but cannot be loaded is an error."""
        self.drafting, self.draft_mode, self.draft_depth = False, "off", 0
        if mode == "off":
            note = "drafter: off (--draft off)"
        elif pack.drafter is None:
            note = "drafter: %s; generation without drafting" % (pack.drafter_problem or "none in this model folder")
        elif not all(callable(getattr(self.native, n, None)) for n in DRAFT_CALLS):
            note = "drafter: drafting is not available with this engine library (no drafter calls); generation without drafting"
        else:
            try:
                self.native.draft_load(str(pack.drafter))
                st = self.native.draft_mode(mode, depth)
            except (RuntimeError, OSError, ValueError) as ex:
                try:
                    if callable(getattr(self.native, "draft_loaded", None)) and self.native.draft_loaded():
                        self.native.draft_mode("off")
                    self.reset()
                except RuntimeError:
                    pass
                if mode == "on":
                    raise YasakaniError("the drafter %s could not be loaded: %s" % (pack.drafter.name, ex)) from ex
                note = "drafter: %s could not be loaded (%s); generation without drafting" % (pack.drafter.name, ex)
            else:
                self.drafting, self.draft_mode, self.draft_depth = True, str(st.get("mode", mode)), int(st.get("depth") or 0)
                self.reset()                                 # the drafter's context starts with the next prompt
                note = "drafter: %s loaded, mode %s, depth %d%s" % (
                    pack.drafter.name, self.draft_mode, self.draft_depth,
                    " (used only while it makes generation faster)" if self.draft_mode == "auto" else "")
        self.draft_note = note
        return note

    def can_draft(self) -> bool:
        """A drafting pass fits: drafter loaded and room for the anchor and every proposed token in the context."""
        return self.drafting and self.fed is not None and len(self.fed) + 1 + self.draft_depth <= self.max_ctx

    def spec_step(self, anchor: int, pick=None) -> Tuple[List[int], int]:
        """Feed ``anchor`` and check the drafter's proposals in one pass -> (accepted tokens, next token; not fed yet)."""
        try:
            accepted, nxt, _ = self.native.spec_step(int(anchor), pick)
        except BaseException as ex:
            self.fed = None                                  # state unknown: the next prompt starts from a reset
            if isinstance(ex, RuntimeError):
                raise YasakaniError("engine step failed: %s" % ex) from ex
            raise
        accepted = [int(a) for a in accepted]
        self.fed.append(int(anchor))
        self.fed.extend(accepted)
        self.last_logits = None
        return accepted, int(nxt)

    def truncate(self, n: int) -> None:
        """Drop the positions from ``n`` on (the rest of a burst that was not emitted)."""
        if self.fed is None or n >= len(self.fed):
            return
        try:
            self.native.truncate(int(n))
        except RuntimeError as ex:
            self.fed = None
            raise YasakaniError("engine truncate failed: %s" % ex) from ex
        del self.fed[n:]
        self.last_logits = None

    def draft_stats(self) -> dict:
        if not self.drafting:
            return {}
        try:
            return dict(self.native.spec_stats())
        except RuntimeError:
            return {}

    def _load_yskn(self, pack: "Ink2Pack", lib, log):
        """Check the tensors of the .yskn file, then let the binding load it through a CheckpointView: the binding opens its weights
        with ``ink2_ckpt.Checkpoint(path)``, which is pointed at the view for this one path while ``load`` runs."""
        from . import ink2_yskn as Y
        t0 = time.perf_counter()
        yf = Y.YsknFile(pack.weights)
        self._yskn = yf
        if os.environ.get("YASAKANI_INK2_SKIP_TENSOR_CHECK") != "1":
            bad = yf.check(threads=8)
            if bad:
                raise ValueError("%s is damaged: the CRC-32 of %d tensor part(s) differs, first %s (%s); convert the folder again"
                                 % (pack.weights.name, len(bad), bad[0][0], bad[0][1]))
        t_check = time.perf_counter() - t0
        # packed dense matrices only when the engine says it takes them AND the binding says it hands them over
        packed = Y.engine_takes_packed_dense(lib) and bool(getattr(self.mod, "TAKES_PACKED_DENSE", False))
        view = Y.CheckpointView(yf, dense="packed" if packed else "bf16")
        kw = dict(backend=self.backend, max_positions=self.max_ctx, dll=str(self.dll), weights=pack.weights.name, log=log)
        try:
            takes_reader = "checkpoint" in inspect.signature(self.mod.Ink2Native.load).parameters
        except (TypeError, ValueError):
            takes_reader = False
        if takes_reader:                                     # a binding that accepts a reader object
            native = self.mod.Ink2Native().load(str(pack.root), checkpoint=view, **kw)
        else:                                                # else: its ink2_ckpt.Checkpoint gives the view for this one path
            ckpt_mod = getattr(self.mod, "ink2_ckpt", None)
            if ckpt_mod is None or not hasattr(ckpt_mod, "Checkpoint"):
                raise RuntimeError("this binding has no ink2_ckpt.Checkpoint to give the .yskn file to")
            original = ckpt_mod.Checkpoint
            target = os.path.normcase(os.path.abspath(str(pack.weights)))

            def checkpoint(path, *a, **k):
                if os.path.normcase(os.path.abspath(str(path))) == target:
                    return view
                return original(path, *a, **k)

            ckpt_mod.Checkpoint = checkpoint
            try:
                native = self.mod.Ink2Native().load(str(pack.root), **kw)
            finally:
                ckpt_mod.Checkpoint = original
        self.weights_note = "yskn %s; tensor check %.1f s; dense matrices %s" % (
            (pack.meta.get("weights") or {}).get("scheme", "?"), t_check, "packed" if packed else "decoded to bfloat16")
        return native

    # -- state --------------------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.native.reset()
        self.fed = []
        self.last_logits = None

    def cancel(self) -> None:
        self._cancel.set()

    def step(self, token: int) -> np.ndarray:
        if len(self.fed) >= self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        try:
            logits = self.native.step(int(token))
        except RuntimeError as ex:
            self.fed = None                                  # state unknown: the next prompt starts from a reset
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed.append(int(token))
        self.last_logits = logits
        return logits

    def read_prompt(self, ids: Sequence[int]) -> Tuple[np.ndarray, int]:
        """Read the prompt; -> (logits after its last id, ids that did not have to be read again).

        The ids already read are kept as far as the new prompt starts with them: when the prompt goes on from all of them (the
        next turn of a chat) only the new ids are read; when it shares only a first part, the positions after that part are dropped
        (``truncate``, up to the engine's rollback) and the rest is read.  All ids but the last are read in passes of several rows
        (``feed_many``), the last one in a step that gives the logits."""
        ids = [int(i) for i in ids]
        if not ids:
            self.reset()
            return None, 0
        if len(ids) > self.max_ctx:
            raise YasakaniError("the context of %d positions is full" % self.max_ctx)
        fed = self.fed
        common = 0
        if fed is not None:
            limit = min(len(fed), len(ids) - 1)              # the last id is always read: its logits are needed
            while common < limit and fed[common] == ids[common]:
                common += 1
        try:
            if fed is None or common == 0:
                self.reset()
                common = 0
            elif common < len(fed):
                if len(fed) - common <= int(getattr(self.native, "rollback", 0)):
                    self.native.truncate(common)
                else:                                        # too far back for the caches: read again from the start
                    self.reset()
                    common = 0
            rest = ids[common:]
            self.fed = None                                  # state unknown until the reading ends
            self.last_logits = None
            if len(rest) > 1:
                self.native.feed_many(rest[:-1])
            logits = self.native.step(rest[-1])
        except (RuntimeError, ValueError) as ex:
            self.fed = None                                  # the next prompt starts from a reset
            raise YasakaniError("engine step failed: %s" % ex) from ex
        self.fed = ids
        self.last_logits = logits
        return logits, common

    # -- information ----------------------------------------------------------------------------------------------------------------
    def memory_used(self) -> int:
        return int(self.native.memory_used())

    def device_memory(self):
        return self.native.device_memory()

    def close(self) -> None:
        n = getattr(self, "native", None)
        if n is not None:
            n.close()
            self.native = None
        y = getattr(self, "_yskn", None)
        if y is not None:
            if n is not None:
                n.ckpt = None
            y.close()
            self._yskn = None


def bytes_per_position(config: dict) -> int:
    """Key/value cache of the full-attention layer visits per position (float32), from config.json."""
    t = config.get("text_config") or config
    n = int(t.get("num_hidden_layers", 24))
    types = t.get("layer_types") or ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]
    loop = t.get("loop_config") or {}
    a, b, passes = loop.get("core_start", 6), loop.get("core_end", 18), loop.get("num_passes", 3)
    visits = sum((passes if a <= i < b else 1) for i, ty in enumerate(types) if ty != "hybrid_sliding")
    return visits * 2 * int(t.get("num_key_value_heads", 8)) * int(t.get("head_dim", 128)) * 4


# ---- generation loop -----------------------------------------------------------------------------------------------------------------
END, DROP = "end", "drop"        # on_token results: the reply ends with this token / ends before it (the token is not emitted)


def sampler_picks(sample: Sampler):
    """k -> a pick function for the rows of a pass whose first row gives output token k; one draw per token (Sampler.seek)."""
    def at(k: int):
        n = [int(k)]

        def pick(logits) -> int:
            sample.seek(n[0])
            n[0] += 1
            return sample(logits)
        return pick
    return at


def generate(eng: Ink2Engine, first: int, on_token, pick_at=None) -> int:
    """Emit tokens from ``first`` (chosen from the logits of the prompt) on, with the drafter when the engine has one.

    ``on_token(tok)`` emits a token and returns None (go on), END (the reply ends with it), DROP (the reply ends before it; it is
    not emitted) or a list of tokens that must come next (the thinking budget).  ``pick_at``: None = greedy, else ``k -> pick``
    (sampler_picks).  The engine state always matches what was emitted: every emitted token is fed except the last one, as in
    plain stepping, and the rest of a burst after the end is truncated away.  Returns the passes of the model (the prompt's last
    pass counts as the one that gave ``first``)."""
    k, tok, forced, passes = 0, int(first), [], 1
    while True:
        r = on_token(tok)
        if r is END or r is DROP:
            return passes
        k += 1
        if r:
            forced = list(r)
        if not forced and eng.can_draft():
            base = len(eng.fed)
            accepted, nxt = eng.spec_step(tok, pick_at(k) if pick_at else None)
            passes += 1
            for i, a in enumerate(accepted):
                r = on_token(a)
                if r is END or r is DROP:                    # the anchor and accepted[:i] stay fed
                    eng.truncate(base + 1 + i)
                    return passes
                k += 1
                if r:                                        # forced tokens after accepted[i]: it stays fed
                    eng.truncate(base + 2 + i)
                    forced = list(r)
                    break
            else:
                tok = nxt
                continue
            tok = forced.pop(0)
            continue
        if len(eng.fed) >= eng.max_ctx:
            return passes
        logits = eng.step(tok)
        passes += 1
        if forced:
            tok = forced.pop(0)
        else:
            tok = pick_at(k)(logits) if pick_at else int(np.argmax(logits))


def _spec_delta(a: dict, b: dict) -> Tuple[int, int]:
    return (int(b.get("proposed", 0)) - int(a.get("proposed", 0)), int(b.get("accepted", 0)) - int(a.get("accepted", 0)))


# ---- chat runner --------------------------------------------------------------------------------------------------------------------
class Ink2Runner:
    """Chat requests on one Inkling-2 engine, one at a time; the same calls as yasakani.chat.ChatRunner (prepare_checked, run,
    close), so the server and the terminal chat use it unchanged."""

    vision = None

    def __init__(self, engine: Ink2Engine, fmt: Ink2Formatter, pack: Ink2Pack, seed0: int = 1):
        self.engine = engine
        self.fmt = fmt
        self.pack = pack
        self._seed = int(seed0)
        self._seed_lock = threading.Lock()
        self.stop_ids = frozenset(pack.stop_ids or [fmt.ids[T_STOP]])

    def next_seed(self) -> int:
        with self._seed_lock:
            s = self._seed
            self._seed += 1
            return s

    def warmup(self, say=None) -> None:
        ids, _ = self.fmt.prompt([{"role": "user", "content": "warm-up"}], {"enable_thinking": False})
        logits, _ = self.engine.read_prompt(ids)
        for _ in range(4):
            logits = self.engine.step(int(np.argmax(logits)))
        tok = int(np.argmax(logits))
        for _ in range(2):                                   # the drafting pass too (a CUDA graph is recorded on first use)
            if not self.engine.can_draft():
                break
            _, tok = self.engine.spec_step(tok)
        self.engine.reset()

    def prepare(self, p):
        from .chat import RequestError
        if p.tools:
            raise RequestError("tools are not supported for this model yet (Inkling-2: text chat only)")
        for i, m in enumerate(p.messages):
            if m.get("role") == "tool" or m.get("tool_calls"):
                raise RequestError("messages[%d]: tool calls and tool results are not supported for this model yet" % i)
            if isinstance(m.get("content"), list):
                if any(isinstance(q, dict) and q.get("type") not in ("text", None) for q in m["content"]):
                    raise RequestError("messages[%d]: only text content is supported for this model" % i)
                m = dict(m, content="".join(str(q.get("text", "")) for q in m["content"] if isinstance(q, dict)))
                p.messages[i] = m
        try:
            from jinja2.exceptions import TemplateError
            return self.fmt.prompt(p.messages, p.template_kwargs)
        except TemplateError as ex:
            raise RequestError("the chat template does not accept these messages: %s" % ex)

    def prepare_checked(self, p):
        from .chat import context_length_error
        prompt, in_answer = self.prepare(p)
        if len(prompt) >= self.engine.max_ctx - CONTEXT_MARGIN:
            raise context_length_error(len(prompt), self.engine.max_ctx)
        return prompt, in_answer

    def close(self) -> None:
        pass

    def run(self, p, emit: Callable[[str, str], None], prepared=None, eng=None):
        """Generate the reply, calling ``emit("reasoning" | "content", text)``; returns yasakani.chat.Usage."""
        from .chat import StopFilter, Usage
        eng = self.engine
        prompt, in_answer = prepared if prepared is not None else self.prepare_checked(p)
        eng._cancel.clear()
        seed = p.seed if p.seed is not None else self.next_seed()
        sample = Sampler(p.temperature, p.top_p, p.top_k, p.min_p, seed)
        stops = frozenset() if p.ignore_eos else self.stop_ids
        max_new = max(1, int(p.max_tokens))
        usage = Usage(prompt_tokens=len(prompt))
        stream = OutputStream(self.fmt, in_answer)
        stopf = StopFilter(p.stop)
        budget = max(0, int(getattr(p, "thinking_budget", 0) or 0))

        def deliver(events):
            for kind, text in events:
                if kind == "content":
                    text = stopf.push(text)
                if text:
                    emit(kind, text)

        t0 = time.perf_counter()
        logits, usage.cached_tokens = eng.read_prompt(prompt)
        t1 = time.perf_counter()
        usage.prefill_ms = usage.prompt_ms = (t1 - t0) * 1000.0
        st0 = eng.draft_stats()
        pick_at = None if sample.t <= 0.0 else sampler_picks(sample)     # greedy: arg-max on the device, no logits copied
        first = int(np.argmax(logits)) if pick_at is None else pick_at(0)(logits)
        state = dict(finish="length", n=0)
        force = [self.fmt.ids[T_END], self.fmt.ids[T_MODEL], self.fmt.ids[T_TEXT]]

        def on_token(tok: int):
            if eng._cancel.is_set():
                return DROP
            state["n"] += 1
            deliver(stream.feed(tok))
            if tok in stops or stopf.hit:
                state["finish"] = "stop"
                return END
            if state["n"] >= max_new:
                return END
            if budget and stream.mode == "reasoning" and stream.answer_at is None and stream.n >= budget:
                return force
            return None

        passes = generate(eng, first, on_token, pick_at)
        n_out, finish = state["n"], state["finish"]
        deliver(stream.flush())
        tail = stopf.flush()
        if tail:
            emit("content", tail)
        usage.gen_ms = (time.perf_counter() - t1) * 1000.0
        usage.completion_tokens = n_out
        usage.passes = n_out
        if eng.drafting:
            usage.passes = passes
            usage.draft_n, usage.draft_n_accepted = _spec_delta(st0, eng.draft_stats())
            usage.spec = dict(draft_mode=eng.draft_mode, draft_passes=passes,
                              tokens_per_pass=round(n_out / passes, 3) if passes else 0.0)
        usage.reasoning_tokens = stream.reasoning_count
        usage.finish = finish
        return usage


def open_runner(pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, seed0: int = 1, say=None, warm: bool = True,
                draft: str = "auto", draft_depth: Optional[int] = None):
    """Model folder -> (formatter, engine, runner).  The tokenizer and template are read first, so a broken folder fails before
    the load.  ``draft``: auto | on | off (see the module text); ``eng.draft_note`` says what became of the drafter."""
    fmt = Ink2Formatter(pack.root)
    eng = Ink2Engine(pack, max_ctx, dll=dll)
    try:
        eng.set_drafter(pack, draft, draft_depth)
        runner = Ink2Runner(eng, fmt, pack, seed0=seed0)
        if warm:
            runner.warmup(say)
    except BaseException:
        eng.close()
        raise
    return fmt, eng, runner


# ---- self-check ----------------------------------------------------------------------------------------------------------------------
def selfcheck_cases(pack: Ink2Pack, reference: Optional[str] = None) -> List[dict]:
    path = Path(reference) if reference else pack.root / "selfcheck.json"
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise YasakaniError("cannot read the reference file %s: %s" % (path, ex))
    if ref.get("format") != 1 or ref.get("architecture") != ARCHITECTURE:
        raise YasakaniError("%s is not an Inkling-2 reference file" % path)
    return list(ref.get("cases") or [])


def run_selfcheck(eng: Ink2Engine, fmt: Ink2Formatter, cases: Sequence[dict], say=print) -> int:
    """Greedy generation of every case against its expected tokens.  0 all pass, 1 a case fails."""
    bad = 0
    for c in cases:
        ids = fmt.encode(fmt.render(c["messages"], **(c.get("template_kwargs") or {})))
        if ids != list(c["prompt_ids"]):
            say("  %-10s FAIL  the prompt ids differ (tokenizer or chat template are not the reference ones)" % c["name"])
            bad += 1
            continue
        want = [int(t) for t in c["tokens"]]
        eng.reset()
        t0 = time.perf_counter()
        logits, _ = eng.read_prompt(ids)
        got: List[int] = []

        def on_token(tok: int, got=got, n=len(want)):
            got.append(int(tok))
            return END if len(got) >= n else None

        passes = generate(eng, int(np.argmax(logits)), on_token)
        dt = time.perf_counter() - t0
        eng.reset()
        per_pass = ", %.2f tokens per pass" % (len(got) / passes) if eng.drafting and passes else ""
        if got == want:
            say("  %-10s PASS  (%d prompt ids, %d tokens equal to the reference, %.2f s%s)" % (c["name"], len(ids), len(want), dt,
                                                                                            per_pass))
        else:
            n = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            say("  %-10s FAIL  the output differs from token #%d on: %r" % (c["name"], n + 1, fmt.decode(got)))
            bad += 1
    return 1 if bad else 0
