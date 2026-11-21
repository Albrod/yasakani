# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Inkling-2 Mini (``"architecture": "inkling2"`` in ``yasakani.json``): model folder, chat template, output stream, sampling and
the chat runner on the Inkling-2 engine library.

Text only, one request at a time.  The engine library is a separate DLL with its own C API (one token per call) and a ctypes
binding module (``ink2_native.py``); both are found through ``--dll`` / ``$YASAKANI_INK2_DLL`` (the binding next to the DLL or one
folder up, or ``$YASAKANI_INK2_BINDING``).  Model folders are made by ``tools/convert/inkling2_convert.py``.

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
        return cls(root=root, meta=meta, weights=root / w["file"], weights_format=fmt, model_id=str(meta.get("model_id") or "inkling-2"),
                   max_ctx=max_ctx, default_ctx=min(int(meta.get("default_ctx") or max_ctx), max_ctx),
                   sampling=dict(temperature=float(s.get("temperature", 1.0)), top_p=float(s.get("top_p", 1.0)),
                                 top_k=int(s.get("top_k", 0) or 0)),
                   stop_ids=[int(i) for i in (meta.get("stop_token_ids") or [])],
                   sha256=dict(meta.get("sha256") or {}), sizes={k: int(v) for k, v in (meta.get("sizes") or {}).items()})

    def verify(self, progress=None) -> Dict[str, bool]:
        """Size, then SHA-256 of every listed file (the weights file is about 20 GB: a minute or so)."""
        res = {}
        for name, want in sorted(self.sha256.items(), key=lambda kv: (self.sizes.get(kv[0], 0), kv[0])):
            p = self.root / name
            ok = p.is_file() and (name not in self.sizes or p.stat().st_size == self.sizes[name])
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

    def _draw(self, p: np.ndarray) -> int:
        """Index drawn with probability p[i] / sum(p) (p >= 0, float32)."""
        n, B = p.size, self.BLOCK
        nb = -(-n // B)
        if nb * B != n:
            p = np.concatenate([p, np.zeros(nb * B - n, p.dtype)])
        cb = np.cumsum(p.reshape(nb, B).sum(axis=1, dtype=np.float64))
        r = self.rng.random() * cb[-1]
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
def find_dll(path: Optional[str] = None) -> Path:
    p = path or os.environ.get("YASAKANI_INK2_DLL")
    if not p:
        raise YasakaniError("the Inkling-2 engine library is not set: give --dll (or set YASAKANI_INK2_DLL) to ink2_cuda.dll")
    p = Path(p)
    if not p.is_file():
        raise YasakaniError("Inkling-2 engine library not found: %s" % p)
    return p


def find_binding(dll: Path) -> Path:
    p = os.environ.get("YASAKANI_INK2_BINDING")
    cands = [Path(p)] if p else [dll.parent / "ink2_native.py", dll.parent.parent / "ink2_native.py"]
    for c in cands:
        if c.is_file():
            return c
    raise YasakaniError("the binding of the Inkling-2 engine (ink2_native.py) is not next to %s; set YASAKANI_INK2_BINDING" % dll.name)


def load_binding(path: Path):
    name = "_yasakani_ink2_binding"
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == str(path):
        return mod
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def backend_of(lib_info: str) -> str:
    b = (os.environ.get("YASAKANI_INK2_BACKEND") or "").strip().lower()
    if b:
        if b not in ("cuda", "host"):
            raise YasakaniError("YASAKANI_INK2_BACKEND must be cuda or host, not %r" % b)
        return b
    return "cuda" if "cuda" in lib_info else "host"


class Ink2Engine:
    """The engine library with the state of one sequence: the ids it has read since the last reset, so that a prompt that only
    extends them (the next turn of a chat) reads only the new ids."""

    def __init__(self, pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, graph: bool = True, log=None):
        self.dll = find_dll(dll)
        self.binding_path = find_binding(self.dll)
        self.mod = load_binding(self.binding_path)
        lib = self.mod.open_dll(str(self.dll))
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
        """Read the prompt; -> (logits after its last id, ids that did not have to be read again)."""
        ids = [int(i) for i in ids]
        fed = self.fed
        if fed is not None and 0 < len(fed) < len(ids) and ids[:len(fed)] == fed:
            cached = len(fed)
        else:
            self.reset()
            cached = 0
        logits = None
        for t in ids[cached:]:
            logits = self.step(t)
        return logits, cached

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
        finish, n_out = "length", 0
        forced: List[int] = []
        while n_out < max_new:
            if eng._cancel.is_set():
                break
            tok = forced.pop(0) if forced else sample(logits)
            n_out += 1
            deliver(stream.feed(tok))
            if tok in stops:
                finish = "stop"
                break
            if stopf.hit:
                finish = "stop"
                break
            if len(eng.fed) >= eng.max_ctx:
                break
            if (budget and not forced and stream.mode == "reasoning" and stream.answer_at is None and stream.n >= budget):
                forced = [self.fmt.ids[T_END], self.fmt.ids[T_MODEL], self.fmt.ids[T_TEXT]]
            if n_out < max_new:
                logits = eng.step(tok)
        deliver(stream.flush())
        tail = stopf.flush()
        if tail:
            emit("content", tail)
        usage.gen_ms = (time.perf_counter() - t1) * 1000.0
        usage.passes = usage.completion_tokens = n_out
        usage.reasoning_tokens = stream.reasoning_count
        usage.finish = finish
        return usage


def open_runner(pack: Ink2Pack, max_ctx: int, dll: Optional[str] = None, seed0: int = 1, say=None, warm: bool = True):
    """Model folder -> (formatter, engine, runner).  The tokenizer and template are read first, so a broken folder fails before
    the load."""
    fmt = Ink2Formatter(pack.root)
    eng = Ink2Engine(pack, max_ctx, dll=dll)
    try:
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
        got = []
        for k in range(len(want)):
            got.append(int(np.argmax(logits)))
            if k + 1 < len(want):
                logits = eng.step(got[-1])
        dt = time.perf_counter() - t0
        eng.reset()
        if got == want:
            say("  %-10s PASS  (%d prompt ids, %d tokens equal to the reference, %.2f s)" % (c["name"], len(ids), len(want), dt))
        else:
            n = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            say("  %-10s FAIL  the output differs from token #%d on: %r" % (c["name"], n + 1, fmt.decode(got)))
            bad += 1
    return 1 if bad else 0
