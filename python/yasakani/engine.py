# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The engine: open the model, set sampling, generate with live token progress, cancel."""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import locale
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ._native import NativeLib, YasakaniError
from .runtime import (DEFAULT_SPEC, EOS_IDS, N_STATS, STAT_INDEX, STATE_FIELDS, VOCAB_SIZE, ModelPack, RuntimeProfile, spec_array)

# engine option keys (yskn_set_opt)
OPT_TEMPERATURE, OPT_TOP_P, OPT_TOP_K, OPT_SEED, OPT_SAMPLER, OPT_MIN_P = 10, 11, 12, 13, 16, 29
_NO_STOP = 0x7FFFFFFF                      # a stop id no token can have (ignore_eos)
CONTEXT_MARGIN = 64                        # ids kept free at the end of the context


@dataclass(frozen=True)
class Sampling:
    """Seeded sampling.  ``temperature <= 0`` is greedy.  top_k is capped at 20 by the engine.  The speculation is built so that the
    result for a given seed is the same as with plain decoding; what was measured is in docs/MEASUREMENTS.md of the repository."""
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    seed: int = 0
    mode: int = 3

    def normalized(self) -> "Sampling":
        if self.temperature <= 0:
            return Sampling(0.0, 0.95, 20, 0.0, 0, self.mode)
        k = int(self.top_k) if self.top_k else 20
        if k <= 0 or k > 20:
            k = 20
        return Sampling(float(self.temperature), float(self.top_p), k, float(self.min_p or 0.0), int(self.seed), self.mode)


@dataclass
class GenResult:
    ids: List[int]                      # generated ids: the first one (predicted by the prompt pass) and all that followed, a stop id last
    finish: str                         # "stop" (a stop id was produced) | "length" | "cancelled"
    prompt_tokens: int
    cached_tokens: int                  # prompt ids the engine did not have to compute again
    snap_mode: int                      # 0 fresh prefill, 1 continued the live state, 2 restored the snapshot (engine's report)
    passes: int
    prefill_ms: float
    gen_ms: float
    stats: Dict[str, float] = field(default_factory=dict)
    draft_n: int = 0                    # tokens the look-ahead proposed (its three sources together)
    draft_n_accepted: int = 0           # of these, the tokens the model's own pass confirmed

    @property
    def completion_tokens(self) -> int:
        return len(self.ids)


@dataclass
class Forced:
    """What :meth:`Engine.teacher_forced` measured.  Index ``i`` is the prediction made after ``ids[:i + 1]``."""
    argmax: np.ndarray                  # int32: the most likely next id
    nll: np.ndarray                     # float32: -log p(ids[i + 1]), natural log
    margin: Optional[np.ndarray] = None # float32: logit of the most likely id minus the logit of the second one (margins=True only)


def forced_passes(n: int, rows: int) -> List[int]:
    """Rows of each pass for ``n`` positions: ``rows`` at a time, and never a last pass of one row (a one-row pass can take a path of
    its own in the engine)."""
    if n < 1 or rows < 2:
        raise ValueError("teacher forcing needs at least one position and passes of two rows or more")
    sizes = [rows] * (n // rows) + ([n % rows] if n % rows else [])
    if len(sizes) > 1 and sizes[-1] == 1:
        sizes[-2] -= 1
        sizes[-1] = 2
    return sizes


def _narrow_paths_as_utf8(paths) -> bool:
    """Windows: the engine library gets file names as UTF-8 bytes, but parts of it read narrow file names in the code page of the C
    runtime's locale (the system code page by default), so a folder name with characters outside ASCII could not be opened.  When a
    path has such characters, the character type of the C locale is switched to UTF-8 for this process, so that every part agrees.
    Paths that are plain ASCII (the configuration all published checks were run with) change nothing.  Returns True when switched."""
    if sys.platform != "win32" or all(str(p).isascii() for p in paths):
        return False
    try:
        locale.setlocale(locale.LC_CTYPE, ".UTF-8")
        return True
    except locale.Error:
        return False


def _put_narrow_env_utf8(env: Dict[str, str]) -> None:
    """Store the path settings in the C runtime's narrow environment as UTF-8 bytes (``os.environ`` stores the narrow copy in the
    system code page; with the UTF-8 locale above the engine library would then read a file name it cannot open)."""
    crt = C.CDLL("ucrtbase")                    # the C runtime that Python and the engine library share
    crt._putenv_s.argtypes = [C.c_char_p, C.c_char_p]
    crt._putenv_s.restype = C.c_int
    for k, v in env.items():
        if crt._putenv_s(k.encode("ascii"), str(v).encode("utf-8")) != 0:
            raise YasakaniError("could not pass %s to the engine library" % k)


class Engine:
    """One model on one GPU.  Calls are serialized; ``cancel()`` may be called from any thread."""

    def __init__(self, pack: ModelPack, *, max_ctx: Optional[int] = None, device: int = 0, dll: Optional[str] = None,
                 profile: Optional[RuntimeProfile] = None, lib: Optional[NativeLib] = None):
        self.pack = pack
        self.max_ctx = int(max_ctx or pack.max_ctx)
        self.profile = profile or RuntimeProfile.default(pack)
        self.utf8_paths = _narrow_paths_as_utf8([pack.weights, pack.draft, pack.draft_head, pack.kv_profile])
        self.removed_env = self.profile.apply()
        if self.utf8_paths:
            _put_narrow_env_utf8({k: v for k, v in self.profile.env.items() if k in ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")})
        self.lib = lib or NativeLib(dll)
        self._f = self.lib.fn
        self._lock = threading.RLock()
        self._cancelled = False
        self._busy = False
        self._e = self._f["yskn_open"](str(pack.weights).encode("utf-8"), self.max_ctx, int(device), 0)
        if not self._e:
            msg = self.lib.last_error()
            hint = ""
            if "memory" in msg.lower():
                hint = "  (not enough free GPU memory: close other programs that use the GPU, or lower --max-ctx)"
            raise YasakaniError("could not open the model: %s%s" % (msg or "unknown error", hint))
        try:
            self._check(self._f["yskn_load_bdraft"](self._e, str(pack.draft).encode("utf-8")))
        except Exception:
            self.close()
            raise
        self.spec: Dict[str, int] = dict(DEFAULT_SPEC)
        self.sampling: Optional[Sampling] = None

    # ---- plumbing --------------------------------------------------------------------------------------------------------------
    def _check(self, rc: int) -> None:
        if rc != 0:
            raise YasakaniError(self.lib.last_error() or "engine call failed (code %d)" % rc)

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_e", None):
                self._f["yskn_close"](self._e)
                self._e = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _need_open(self):
        if not self._e:
            raise YasakaniError("the engine is closed")

    # ---- state -----------------------------------------------------------------------------------------------------------------
    def position(self) -> int:
        self._need_open()
        return int(self._f["yskn_position"](self._e))

    def reset(self) -> None:
        with self._lock:
            self._need_open()
            self._check(self._f["yskn_reset"](self._e))

    def state_hash(self) -> np.ndarray:
        """Hashes of the whole decode state (for checking that two runs are bit-identical)."""
        with self._lock:
            self._need_open()
            out = np.zeros(512, np.uint64)
            n = C.c_int(0)
            self._check(self._f["yskn_state_hash"](self._e, out.ctypes.data, len(out), C.byref(n)))
            return out[:n.value].copy()

    def cancel(self) -> None:
        """Ask a running generation to stop at its next pass boundary (the state stays valid)."""
        if self._e:
            self._cancelled = True
            self._f["yskn_cancel"](self._e)

    # ---- sampling --------------------------------------------------------------------------------------------------------------
    @property
    def supports_visual(self) -> bool:
        return "yskn_set_visual" in self._f

    @contextmanager
    def visual_input(self, features, slots, positions, delta):
        """Keep visual input and generation under the same lock; always clear it afterwards.

        Features are bfloat16 bits, not float16 values. The arrays stay alive until generation finishes.
        Clearing also resets the native prefix cache, so images cannot leak into later text requests.
        """
        if not self.supports_visual:
            raise YasakaniError("the engine library has no yskn_set_visual export")
        features, slots, positions = map(np.asarray, (features, slots, positions))
        if features.dtype != np.uint16 or features.ndim != 2 or features.shape != (slots.size, 5120):
            raise ValueError("visual features must be bfloat16 bits with shape (image tokens, 5120)")
        if slots.dtype != np.int32 or slots.ndim != 1 or not slots.size:
            raise ValueError("visual slots must be a nonempty int32 vector")
        if positions.dtype != np.int32 or positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("visual positions must be int32 with shape (3, prompt tokens)")
        n_prompt = positions.shape[1]
        if n_prompt >= self.max_ctx or np.any(slots < 0) or np.any(slots >= n_prompt) or len(np.unique(slots)) != len(slots):
            raise ValueError("invalid visual slots or prompt length")
        if np.any(positions < 0) or np.any(positions >= self.max_ctx) or int(delta) != int(positions.max()) + 1 - n_prompt:
            raise ValueError("invalid visual positions or delta")
        features, slots, positions = map(np.ascontiguousarray, (features, slots, positions))
        with self._lock:
            self._need_open()
            f = self._f["yskn_set_visual"]
            try:
                self._check(f(self._e, features.ctypes.data, slots.ctypes.data, len(slots), positions.ctypes.data,
                              n_prompt, int(delta)))
                yield
            finally:
                self._check(f(self._e, None, None, 0, None, 0, 0))

    def set_sampling(self, s: Sampling) -> Sampling:
        s = s.normalized()
        with self._lock:
            self._need_open()
            f, e = self._f["yskn_set_opt"], self._e
            self._check(f(e, OPT_TOP_P, int(round(s.top_p * 1000))))
            self._check(f(e, OPT_TOP_K, int(s.top_k)))
            self._check(f(e, OPT_SEED, int(s.seed)))
            self._check(f(e, OPT_TEMPERATURE, int(round(s.temperature * 1000))))
            self._check(f(e, OPT_SAMPLER, int(s.mode)))
            self._check(f(e, OPT_MIN_P, int(round(s.min_p * 1000))))
            self.sampling = s
        return s

    # ---- generation ------------------------------------------------------------------------------------------------------------
    def plan(self, n_prompt: int, max_new: int) -> int:
        """Number of ids to ask for after the first one; raises when the prompt does not fit."""
        if n_prompt < 1:
            raise ValueError("the prompt is empty")
        room = self.max_ctx - n_prompt - CONTEXT_MARGIN
        if room < 1:
            raise ValueError("the prompt (%d tokens) leaves no room in the context of %d tokens" % (n_prompt, self.max_ctx))
        return max(0, min(int(max_new) - 1, room))

    def generate(self, prompt: Sequence[int], max_new: int, *, stop_ids: Optional[Sequence[int]] = EOS_IDS,
                 on_tokens: Optional[Callable[[np.ndarray], None]] = None, spec: Optional[Dict[str, int]] = None,
                 poll_s: float = 0.001) -> GenResult:
        """Generate up to ``max_new`` ids (the first one comes from the prompt pass).  ``on_tokens(ids)`` is called on the calling
        thread with all ids produced so far whenever new ones are committed.  If it raises, the generation is cancelled and the
        exception is re-raised after the engine has stopped.  ``stop_ids=None`` ignores stop ids (always runs to ``max_new``)."""
        prompt_arr = np.ascontiguousarray(np.asarray(prompt, dtype=np.int32))
        n_gen = self.plan(len(prompt_arr), max_new)
        cfg = dict(self.spec)
        for k, v in (spec or {}).items():
            if k in STATE_FIELDS:
                raise ValueError("%s decides how the engine keeps its state and is not a per-request setting" % k)
            if k not in cfg:
                raise ValueError("unknown speculation setting: " + k)
            cfg[k] = int(v)
        if stop_ids is None:
            cfg["eos1"] = cfg["eos2"] = _NO_STOP
            stops = ()
        else:
            stops = tuple(int(x) for x in stop_ids)
            if stops:
                cfg["eos1"] = stops[0]
                cfg["eos2"] = stops[1] if len(stops) > 1 else stops[0]
        cfa = spec_array(cfg)
        out = np.full(n_gen + 1, -1, np.int32)
        st = np.zeros(N_STATS, np.float64)
        tm = np.zeros(2, np.float64)
        box: Dict[str, object] = {}
        with self._lock:
            self._need_open()
            self._cancelled = False
            self._busy = True
            e = self._e
            fn = self._f["yskn_spec_generate"]

            def work():
                try:
                    rc = fn(e, prompt_arr.ctypes.data, len(prompt_arr), n_gen, cfa.ctypes.data, len(cfa), out.ctypes.data,
                            st.ctypes.data, tm.ctypes.data)
                    box["rc"] = rc
                    if rc != 0:
                        box["err"] = self.lib.last_error()          # thread-local in the engine: read it here
                except BaseException as ex:                          # noqa: BLE001
                    box["exc"] = ex

            th = threading.Thread(target=work, name="yasakani-generate", daemon=True)
            th.start()
            seen = 0
            failure: Optional[BaseException] = None
            try:
                while True:
                    alive = th.is_alive()
                    n = seen
                    while n < len(out) and out[n] >= 0:
                        n += 1
                    if n > seen:
                        seen = n
                        if on_tokens is not None and failure is None:
                            try:
                                on_tokens(out[:n])
                            except BaseException as ex:              # noqa: BLE001  (a consumer that fails stops the engine)
                                failure = ex
                                self.cancel()
                    if not alive and seen >= n:
                        break
                    th.join(poll_s) if alive else None
            except BaseException:                                    # e.g. Ctrl+C: stop the engine, wait for it, then re-raise
                self.cancel()
                raise
            finally:
                th.join()
                self._busy = False
            if failure is not None:
                raise failure
        if "exc" in box:
            raise box["exc"]                                       # type: ignore[misc]
        if box.get("rc") != 0:
            raise YasakaniError(str(box.get("err") or "engine call failed"))
        produced = int(st[STAT_INDEX["produced"]])
        gen = min(produced + 1, len(out))
        ids = [int(x) for x in out[:gen]]
        stopped = int(st[STAT_INDEX["eos"]]) == 1
        finish = "stop" if stopped else ("cancelled" if self._cancelled else "length")
        stats = {k: float(st[i]) for k, i in STAT_INDEX.items()}
        return GenResult(ids=ids, finish=finish, prompt_tokens=len(prompt_arr), cached_tokens=int(st[STAT_INDEX["cached_tokens"]]),
                         snap_mode=int(st[STAT_INDEX["snap_mode"]]), passes=int(st[STAT_INDEX["passes"]]),
                         prefill_ms=float(tm[0]), gen_ms=float(tm[1]), stats=stats,
                         draft_n=int(sum(st[STAT_INDEX["drafted_%d" % s]] for s in range(3))),
                         draft_n_accepted=int(sum(st[STAT_INDEX["accepted_%d" % s]] for s in range(3))))

    # ---- teacher forcing -------------------------------------------------------------------------------------------------------
    @property
    def supports_forced(self) -> bool:
        return all(k in self._f for k in ("yskn_verify_nll", "yskn_commit"))

    def teacher_forced(self, ids: Sequence[int], *, rows: int = 16, margins: bool = False) -> Forced:
        """Let the engine read ``ids`` in passes of ``rows`` positions without generating anything (teacher forcing) and return, for
        every position but the last, the most likely next id and the negative log-likelihood of the id that really follows.
        ``margins=True`` also reads the logits of the whole vocabulary to give the margin of the most likely id.  The state is reset
        before and after; the sampling settings are not used."""
        if not self.supports_forced or (margins and not all(k in self._f for k in ("yskn_logits_rows", "yskn_verify"))):
            raise YasakaniError("the engine library has no exports for teacher forcing (yskn_verify_nll, yskn_commit)")
        seq = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        n = len(seq) - 1
        if n + 1 > self.max_ctx - CONTEXT_MARGIN:
            raise ValueError("%d ids do not fit the context of %d tokens" % (len(seq), self.max_ctx))
        sizes = forced_passes(n, rows)
        am = np.empty(n, np.int32)
        nll = np.empty(n, np.float32)
        mg = np.empty(n, np.float32) if margins else None
        f = self._f
        with self._lock:
            self._need_open()
            e = self._e
            self._check(f["yskn_reset"](e))
            done = False
            try:
                pos = 0
                for m in sizes:
                    x = np.ascontiguousarray(seq[pos:pos + m])
                    t = np.ascontiguousarray(seq[pos + 1:pos + m + 1])
                    a = np.empty(m, np.int32)
                    v = np.empty(m, np.float32)
                    self._check(f["yskn_verify_nll"](e, x.ctypes.data, t.ctypes.data, m, 1, a.ctypes.data, v.ctypes.data))
                    if margins:
                        lg = np.empty((m, VOCAB_SIZE), np.float32)
                        self._check(f["yskn_logits_rows"](e, 0, m, lg.ctypes.data))
                        if not np.array_equal(lg.argmax(1), a):        # the logits may not be kept by this pass: make it again
                            b = np.empty(m, np.int32)
                            self._check(f["yskn_verify"](e, x.ctypes.data, m, 1, b.ctypes.data, None, None))
                            self._check(f["yskn_logits_rows"](e, 0, m, lg.ctypes.data))
                            if not (np.array_equal(b, a) and np.array_equal(lg.argmax(1), a)):
                                raise YasakaniError("teacher forcing: the logits of the rows do not belong to the pass")
                        top2 = np.partition(lg, -2, axis=1)[:, -2:]
                        mg[pos:pos + m] = top2.max(1) - top2.min(1)
                    self._check(f["yskn_commit"](e, m - 1))
                    am[pos:pos + m] = a
                    nll[pos:pos + m] = v
                    pos += m
                done = True
            finally:
                rc = f["yskn_reset"](e)
                if done:                                             # an earlier error is the one to report
                    self._check(rc)
        return Forced(argmax=am, nll=nll, margin=mg)

    # ---- warm-up ---------------------------------------------------------------------------------------------------------------
    def warmup(self, prompts: Sequence[Sequence[int]], n_gen: int = 32) -> None:
        """Run short generations so that the CUDA graphs and the prompt-snapshot buffer exist before the first real request.  Give one
        plain prompt and one prompt that contains the chat start marker; the state is reset afterwards."""
        self.set_sampling(Sampling(1.0, 0.95, 20, 0.0, 999, 3))
        for p in prompts:
            self.reset()
            self.generate(p, n_gen + 1)
        self.reset()
        self.prepare_wide()

    def prepare_wide(self) -> None:
        """Build the graphs of the wide copy passes for greedy and for sampled generation (engine libraries of 0.1.4 or newer; nothing
        runs, the state is unchanged).  The sampling settings are restored afterwards."""
        f = self._f.get("yskn_prepare_wide")
        rows = int(self.spec.get("copy_rows", 16))
        if f is None or rows <= 16:
            return
        with self._lock:
            self._need_open()
            keep = self.sampling
            self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 0))
            self._check(f(self._e, rows))
            self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 1000))
            self._check(f(self._e, rows))
            if keep is not None:
                self.set_sampling(keep)
            else:
                self._check(self._f["yskn_set_opt"](self._e, OPT_TEMPERATURE, 0))
