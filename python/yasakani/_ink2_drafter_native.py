# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Drafter methods of Ink2Native (drafter of Inkling-2 Mini, mtp.safetensors).

Installed into _ink2_native.Ink2Native by install() (one line at the end of _ink2_native.py).

    eng.draft_load(path)                       # mode "auto" afterwards
    eng.draft_mode("on", depth=4)              # / "off" / "auto"
    accepted, nxt, logits = eng.spec_step(anchor, pick=None, want_logits=False)
    eng.spec_stats(reset=False)

The drafter context lives in the DLL and is committed by every pass of the engine, so
feed_many / step / reset / truncate keep it in step without anything here.
Needs numpy only.
"""

import ctypes
import json
import os
import time

import numpy as np

from . import _ink2_ckpt as ink2_ckpt  # numpy only

DRAFT_TENSORS = (
    ("dflare", "dflare_norm.weight"), ("in_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"),
    ("wq", "self_attn.q_proj.weight"), ("wk", "self_attn.k_proj.weight"), ("wv", "self_attn.v_proj.weight"),
    ("wo", "self_attn.o_proj.weight"), ("qn", "self_attn.q_norm.weight"), ("kn", "self_attn.k_norm.weight"),
    ("wkc", "self_attn.k_proj_ctx.weight"), ("wvc", "self_attn.v_proj_ctx.weight"), ("knc", "self_attn.k_norm_ctx.weight"),
    ("wg", "mlp.gate_proj.weight"), ("wu", "mlp.up_proj.weight"), ("wd", "mlp.down_proj.weight"),
)
MAX_DEPTH = 7
# Verification pass of R = 2 .. 8 rows / one token, RTX 5090 (measured): a prior
# for depths not measured yet in this run.
PASS_RATIO = [1.0, 8.60 / 5.98, 9.84 / 5.98, 10.84 / 5.98, 11.79 / 5.98, 12.68 / 5.98, 13.59 / 5.98, 14.63 / 5.98]
# Acceptance by depth of the reference (8 prompts pooled): prior.
PRIOR_ACCEPT = [0.684, 0.686, 0.711, 0.730, 0.742, 0.736, 0.733]


class Ink2DraftConfig(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_int) for n in ("hidden", "layers", "heads", "kv_heads", "head_dim", "inter", "window", "block",
                                     "rank", "w1_rows")]
        + [("local", ctypes.c_int * 8), ("feat_visits", ctypes.c_int * 3)]
        + [(n, ctypes.c_float) for n in ("eps", "rope_theta", "softcap")]
    )


def _bind(lib):
    if getattr(lib, "_drafter_bound", False):
        return True
    try:
        lib.ink2_draft_abi.restype = ctypes.c_int
    except AttributeError:
        return False
    vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    lib.ink2_draft_abi.argtypes = []
    lib.ink2_draft_create.argtypes = [vp, ctypes.POINTER(Ink2DraftConfig)]
    lib.ink2_draft_set_tensor.argtypes = [vp, ctypes.c_char_p, vp, cll]
    lib.ink2_draft_finalize.argtypes = [vp]
    lib.ink2_draft_unload.argtypes = [vp]
    lib.ink2_draft_loaded.argtypes = [vp]
    lib.ink2_draft_set_option.argtypes = [vp, ctypes.c_char_p, ci]
    lib.ink2_spec_pass.argtypes = [vp, ci, ci, vp, vp, ci]
    lib.ink2_draft_propose.argtypes = [vp, ci, ci, ci, vp]
    lib.ink2_draft_commit_features.argtypes = [vp, vp, ci, ci]
    lib.ink2_draft_time.argtypes = [vp, ci, ci, ctypes.POINTER(ctypes.c_double)]
    lib.ink2_draft_memory.argtypes = [vp]
    lib.ink2_draft_memory.restype = cll
    lib.ink2_draft_get.argtypes = [vp, ci, ci, vp, cll]
    lib._drafter_bound = True
    return True


def bf16_round(x):
    """float32 -> BF16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def unembed_rows(eng):
    """The output projection U as the engine multiplies it, as a function (start, stop) -> float64 rows:
    BF16 from the checkpoint, or decoded from packed dense parts (an engine with packed dense matrices:
    a dense_dir folder of ink2_pack_dense.py, or a reader whose info() says NVFP4 / MXFP8), so that the
    folded head matches the base logits of the engine."""
    V = eng.vocab
    folder = getattr(eng, "dense_dir", None)
    parts = None
    if folder and getattr(eng, "dense_mode", "packed") == "packed" and os.path.exists(os.path.join(folder, "unembed.w.npy")):
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        info = (man.get("matrices") or man)["unembed"]
        w = np.load(os.path.join(folder, "unembed.w.npy"), mmap_mode="r")
        sc = np.load(os.path.join(folder, "unembed.s.npy"), mmap_mode="r")
        fac = float(np.load(os.path.join(folder, "unembed.f.npy")).reshape(-1)[0]) if info["format"] == "nvfp4" else None
        parts = (info["format"], w, sc, fac)
    else:
        name = "model.llm.unembed.weight"
        dt = eng.ckpt.info(name)[0]
        if dt in ("NVFP4", "MXFP8") and hasattr(eng.ckpt, "parts"):
            pr = eng.ckpt.parts(name)
            fac = None
            if dt == "NVFP4":
                fac = float(np.float32(np.asarray(pr["amax"], dtype=np.float32).reshape(-1)[0] / np.asarray(pr["denominator"], dtype=np.float32).reshape(-1)[0]))
            parts = (dt.lower(), np.asarray(pr["weight"]).view(np.uint8), np.asarray(pr["scale"]).view(np.uint8), fac)
    if parts is None:
        U = eng.ckpt.raw("model.llm.unembed.weight")[:V]
        return lambda a, b: ink2_ckpt.bf16_to_f32(np.asarray(U[a:b])).astype(np.float64)
    fmt, w, sc, fac = parts

    def rows(a, b):
        ww = np.asarray(w[a:b])
        ss = ink2_ckpt.E4M3_TABLE[np.asarray(sc[a:b])].astype(np.float64) if fmt == "nvfp4" else None
        if fmt == "nvfp4":  # value = E2M1(code) * E4M3(block scale) * (amax / denominator); even element in the low nibble
            v = np.empty((ww.shape[0], ww.shape[1] * 2), dtype=np.float64)
            v[:, 0::2] = E2M1[ww & 15]
            v[:, 1::2] = E2M1[ww >> 4]
            return v * np.repeat(ss, 16, axis=1) * fac
        e = np.ldexp(1.0, np.asarray(sc[a:b]).astype(np.int64) - 127)  # mxfp8: E4M3 bytes * 2 ** (byte - 127)
        return ink2_ckpt.E4M3_TABLE[ww].astype(np.float64) * np.repeat(e, 32, axis=1)

    return rows


def fold_head(rows, V, out_w, out_b, chunk=4096):
    """U @ hidden_out folded (S Inkling2MTPHead.refresh_fold, ink2_draft.Ink2Drafter.__init__):
    float64 products in chunks of 4096 rows, stored as float32. rows(start, stop): float64 rows of U."""
    proj = out_w.astype(np.float64)  # [H, rank]
    bias = out_b.astype(np.float64)
    fw = np.empty((V, proj.shape[1]), dtype=np.float32)
    fb = np.empty(V, dtype=np.float32)
    for s in range(0, V, chunk):
        head = rows(s, min(V, s + chunk))
        fw[s:s + chunk] = (head @ proj).astype(np.float32)
        fb[s:s + chunk] = (head @ bias).astype(np.float32)
    return fw, fb


class Governor:
    """Decides per spec_step whether to draft and how deep ("auto").

    Acceptance: EWMA of the conditional acceptance p_j of draft j (given j - 1 accepted), from every
    pass that reached row j. Costs: EWMA of the wall time of a plain step and of a pass with a draft of
    each depth (the first call of each kind records a graph and is not counted); a depth not measured
    yet is estimated from the plain step and PASS_RATIO plus the measured draft call. Expected rate of
    depth d: (1 + sum_j prod_{i<=j} p_i) / time(d), against 1 / time(plain).
    On when the best depth beats plain by `on_margin`, off below `off_margin` (hysteresis). While off,
    one probe pass (depth >= probe_depth) every `probe_every` plain steps keeps the acceptance current
    (the probe still emits >= 1 token, so it costs at most time(d) - time(plain)); a probe moves the
    acceptance by probe_beta; off only after off_patience passes below off_margin in a row. Defaults:
    replay of the measured acceptance of 7 prompts in 3 orders, cold and warm, with today's costs
    (gov_sim.py), then measured on the GPU (out_gpu/check_drafter_log.txt).
    """

    def __init__(self, max_depth=MAX_DEPTH, beta=0.15, on_margin=1.04, off_margin=1.0, probe_every=16, explore_every=12,
                 off_patience=3, probe_depth=2, probe_beta=0.5):
        self.max_depth = max_depth
        self.probe_beta = probe_beta
        self.probe_depth = probe_depth
        self.off_patience, self.below = off_patience, 0
        self.beta, self.on_margin, self.off_margin = beta, on_margin, off_margin
        self.probe_every, self.explore_every = probe_every, explore_every
        self.p = list(PRIOR_ACCEPT[:max_depth])
        self.t_plain = None
        self.t_spec = [None] * (max_depth + 1)
        self.n_spec = [0] * (max_depth + 1)
        self.n_plain = 0
        self.t_draft = [0.0] * (max_depth + 1)  # ms of a draft call, measured at load
        self.active = True
        self.since_probe = 0
        self.since_explore = 0
        self.depth = 3

    def expected_tokens(self, d):
        e, run = 1.0, 1.0
        for j in range(d):
            run *= self.p[j]
            e += run
        return e

    def time_of(self, d):
        if self.t_spec[d] is not None:
            return self.t_spec[d]
        tp = self.t_plain if self.t_plain is not None else 6e-3
        return tp * PASS_RATIO[d] + 1e-3 * self.t_draft[d]

    def best(self):
        tp = self.t_plain if self.t_plain is not None else 6e-3
        gains = [self.expected_tokens(d) / self.time_of(d) * tp for d in range(1, self.max_depth + 1)]
        d = int(np.argmax(gains)) + 1
        return d, gains[d - 1]

    def choose(self):
        """-> depth to draft now (0 = plain step), is_probe."""
        if self.t_plain is None or self.n_plain < 2:
            return 0, False  # measure a plain step first (the first one records its graph)
        d, gain = self.best()
        if self.active:
            self.below = self.below + 1 if gain < self.off_margin else 0
            if self.below >= self.off_patience:
                self.active = False
                self.below = 0
                self.since_probe = 0
            else:
                self.depth = d
                self.since_explore += 1
                if self.since_explore >= self.explore_every:
                    self.since_explore = 0
                    nb = d + 1 if d < self.max_depth else d - 1
                    if self.n_spec[nb] < 3 or nb == d + 1:
                        return nb, False
                return d, False
        else:
            if gain >= self.on_margin:
                self.active = True
                self.depth = d
                return d, False
        self.since_probe += 1
        if self.since_probe >= self.probe_every:
            self.since_probe = 0
            return min(self.max_depth, max(d, self.probe_depth)), True
        return 0, False

    def update_plain(self, seconds):
        self.n_plain += 1
        if self.n_plain == 1:
            return
        b = self.beta
        self.t_plain = seconds if self.t_plain is None else (1 - b) * self.t_plain + b * seconds

    def update_spec(self, d, accepted, seconds, probe=False):
        self.n_spec[d] += 1
        b = self.probe_beta if (probe and self.probe_beta) else self.beta
        for j in range(1, d + 1):
            if accepted >= j - 1:
                self.p[j - 1] = (1 - b) * self.p[j - 1] + b * (1.0 if accepted >= j else 0.0)
        if self.n_spec[d] == 1:
            return
        b = self.beta
        self.t_spec[d] = seconds if self.t_spec[d] is None else (1 - b) * self.t_spec[d] + b * seconds


def _lib_ok(self):
    if self.lib is None or self.h is None:
        raise RuntimeError("load the model first")
    if not _bind(self.lib):
        raise RuntimeError("this engine library has no drafter (ink2_draft_* missing)")


def draft_load(self, path, config=None, fold="bf16", calibrate=True):
    """Loads the drafter weights (mtp.safetensors) to the device. config: the mtp_config dict
    (default: config.json of the model folder). fold: "bf16" (default), "f32" or "both".
    Positions seen before are fed again (the drafter needs their features). Mode "auto" afterwards."""
    _lib_ok(self)
    if not os.path.exists(path):
        raise RuntimeError("drafter file not found: %s" % path)
    if config is None:
        with open(os.path.join(self.model_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f).get("mtp_config")
        if config is None:
            raise RuntimeError("config.json of the model has no mtp_config")
    mc = config
    if mc.get("markov_head_type", "rnn_hidden") != "rnn_hidden" or mc.get("block_size", 8) != 8:
        raise RuntimeError("drafter: unsupported head type or block size")
    tl = list(mc["target_layer_ids"])
    if len(tl) != 4 or tl[-1] != self.num_visits:
        raise RuntimeError("drafter: target_layer_ids %s do not fit %d visits" % (tl, self.num_visits))
    feat_visits = []
    for v in tl[:3]:
        if v < 1 or v >= self.num_visits:
            raise RuntimeError("drafter: target visit %d out of range" % v)
        layer, pas = self.visits[v]
        if pas > 0 and layer == self.cfg["loop_start"]:
            raise RuntimeError("drafter: the input of visit %d is not the output of visit %d" % (v, v - 1))
        feat_visits.append(v - 1)
    ck = ink2_ckpt.Checkpoint(path)
    H, nl = mc["hidden_size"], mc["num_hidden_layers"]
    nh, nkv, D, I, r = mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"], mc["intermediate_size"], mc["markov_rank"]
    w1_rows = ck.info("markov_head.markov_w1.weight")[1][0] if ck.has("markov_head.markov_w1.weight") else -1
    expect = {"fuse_weight": (nl + 1, 4), "norm.weight": (H,), "mask_embed": (H,),
              "markov_head.markov_w1.weight": (w1_rows, r), "markov_head.rnn_proj.weight": (3 * r, 2 * r + 2 * H),
              "markov_head.rnn_proj.bias": (3 * r,), "markov_head.hidden_out.weight": (H, r),
              "markov_head.hidden_out.bias": (H,), "markov_head.dflare_norm.weight": (H,)}
    shapes = {"dflare_norm.weight": (H,), "input_layernorm.weight": (H,), "post_attention_layernorm.weight": (H,),
              "self_attn.q_proj.weight": (nh * D, H), "self_attn.k_proj.weight": (nkv * D, H),
              "self_attn.v_proj.weight": (nkv * D, H), "self_attn.o_proj.weight": (H, nh * D),
              "self_attn.q_norm.weight": (D,), "self_attn.k_norm.weight": (D,), "self_attn.k_proj_ctx.weight": (nkv * D, H),
              "self_attn.v_proj_ctx.weight": (nkv * D, H), "self_attn.k_norm_ctx.weight": (D,),
              "mlp.gate_proj.weight": (I, H), "mlp.up_proj.weight": (I, H), "mlp.down_proj.weight": (H, I)}
    for i in range(nl):
        for k, s in shapes.items():
            expect["layers.%d.%s" % (i, k)] = s
    for name, shape in expect.items():
        if not ck.has(name):
            raise RuntimeError("drafter file: missing tensor %s" % name)
        dt, sh = ck.info(name)
        if dt != "BF16" or tuple(sh) != tuple(shape):
            raise RuntimeError("drafter file: %s is %s %s, expected BF16 %s" % (name, dt, sh, shape))
    if w1_rows < self.vocab or H != self.hidden_size:
        raise RuntimeError("drafter file does not fit the model (hidden %d, markov_w1 rows %d)" % (H, w1_rows))

    c = Ink2DraftConfig()
    c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.inter = H, nl, nh, nkv, D, I
    c.window, c.block, c.rank, c.w1_rows = mc["sliding_window_size"], mc["block_size"], r, w1_rows
    for i in range(8):
        c.local[i] = 1 if i in mc.get("local_layer_ids", []) else 0
    for i in range(3):
        c.feat_visits[i] = feat_visits[i]
    c.eps, c.rope_theta, c.softcap = mc["rms_norm_eps"], mc["rope_theta"], self.cfg["final_logit_softcapping"]
    t0 = time.perf_counter()
    self._check(self.lib.ink2_draft_create(self.h, ctypes.byref(c)), "draft_create")

    def put(name, array):
        array = np.ascontiguousarray(array)
        self._check(self.lib.ink2_draft_set_tensor(self.h, name.encode(), ctypes.c_void_p(array.ctypes.data), array.nbytes), "draft " + name)

    try:
        fw = ink2_ckpt.bf16_to_f32(np.asarray(ck.raw("fuse_weight"))).astype(np.float32)
        ex = np.exp(fw - fw.max(axis=1, keepdims=True))
        put("fuse_alpha", (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32))
        for i in range(nl):
            for short, name in DRAFT_TENSORS:
                put("L%d.%s" % (i, short), ck.raw("layers.%d.%s" % (i, name)))
        put("norm", ck.raw("norm.weight"))
        put("mask", ck.raw("mask_embed"))
        put("w1", ck.raw("markov_head.markov_w1.weight"))
        put("rnn_w", ck.raw("markov_head.rnn_proj.weight"))
        put("rnn_b", ck.raw("markov_head.rnn_proj.bias"))
        put("head_norm", ck.raw("markov_head.dflare_norm.weight"))
        t1 = time.perf_counter()
        fold_w, fold_b = fold_head(unembed_rows(self), self.vocab, ink2_ckpt.bf16_to_f32(np.asarray(ck.raw("markov_head.hidden_out.weight"))),
                                   ink2_ckpt.bf16_to_f32(np.asarray(ck.raw("markov_head.hidden_out.bias"))))
        self.draft_fold_seconds = time.perf_counter() - t1
        if fold in ("bf16", "both"):
            put("fold16", bf16_round(fold_w))
        if fold in ("f32", "both"):
            put("fold32", fold_w)
        put("fold_b", fold_b)
        del fold_w
        self._check(self.lib.ink2_draft_finalize(self.h), "draft_finalize")
    except Exception:
        self.lib.ink2_draft_unload(self.h)
        raise
    self.draft_load_seconds = time.perf_counter() - t0
    self.draft_path = path
    self._gov = Governor()
    self._d_mode, self._d_depth = "auto", 4
    self.spec_stats(reset=True)
    if self.num_seen:  # the drafter needs the features of every position seen
        keep = list(self.seen_ids)
        self.reset()
        self.feed_many(keep)
    if calibrate:
        self.draft_calibrate()
    return self


def draft_calibrate(self, reps=5):
    """Measures a draft call of every depth (records their graphs); used by "auto" and spec_stats."""
    for d in range(1, MAX_DEPTH + 1):
        self._gov.t_draft[d] = self.draft_time(d, reps)["ms"]


def draft_time(self, depth, reps=10):
    """A draft call alone (device synchronised): ms per call as used, and its parts (direct launches)."""
    out = (ctypes.c_double * 5)()
    self._check(self.lib.ink2_draft_time(self.h, int(depth), int(reps), out), "draft_time")
    return {"ms": out[0], "block_ms": out[1], "base_ms": out[2], "head_ms": out[3], "launches": int(out[4])}


def draft_loaded(self):
    return bool(self.lib is not None and self.h and _bind(self.lib) and self.lib.ink2_draft_loaded(self.h))


def draft_unload(self):
    if self.draft_loaded():
        self._check(self.lib.ink2_draft_unload(self.h), "draft_unload")


def draft_mode(self, mode=None, depth=None):
    if not self.draft_loaded():
        if mode in (None, "off"):
            return {"mode": "off", "depth": getattr(self, "_d_depth", 4), "active": False}
        raise RuntimeError("no drafter loaded")
    if mode is not None:
        if mode not in ("off", "on", "auto"):
            raise ValueError("mode must be off, on or auto")
        self._d_mode = mode
        if mode == "auto":
            self._gov.active = True
    if depth is not None:
        depth = int(depth)
        if not 1 <= depth <= MAX_DEPTH:
            raise ValueError("depth must be 1 .. %d" % MAX_DEPTH)
        self._d_depth = depth
        self._gov.max_depth = depth if self._d_mode == "auto" else MAX_DEPTH
    active = self._d_mode == "on" or (self._d_mode == "auto" and self._gov.active)
    return {"mode": self._d_mode, "depth": self._d_depth if self._d_mode != "auto" else self._gov.depth, "active": active}


def draft_set(self, name, value):
    """Drafter options: "fold" 16 / 32 (the folded head in BF16 / float32)."""
    self._check(self.lib.ink2_draft_set_option(self.h, name.encode(), int(value)), "draft_set_option")


def draft_propose(self, anchor, pos, depth=MAX_DEPTH):
    """A draft call alone for `anchor` at position `pos` (state unchanged): the proposed tokens."""
    out = (ctypes.c_int * depth)()
    self._check(self.lib.ink2_draft_propose(self.h, int(anchor), int(pos), int(depth), out), "draft_propose")
    return list(out)


def draft_commit_features(self, feats, p0):
    """Debug: drafter context of positions p0 .. from given features [n, 4, hidden] (engine untouched)."""
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    n = feats.shape[0]
    step = self.max_rows
    for s in range(0, n, step):
        part = np.ascontiguousarray(feats[s:s + step])
        self._check(self.lib.ink2_draft_commit_features(self.h, ctypes.c_void_p(part.ctypes.data), int(p0 + s), int(part.shape[0])),
                    "draft_commit_features")


def draft_memory(self):
    return int(self.lib.ink2_draft_memory(self.h)) if self.draft_loaded() else 0


def _row_logits(self, row):
    out = np.empty(self.vocab, dtype=np.float32)
    self._check(self.lib.ink2_get_row(self.h, 6, 0, int(row), ctypes.c_void_p(out.ctypes.data), out.nbytes), "get_row")
    return out


def spec_step(self, anchor, pick=None, want_logits=False):
    """Feeds `anchor` and, while drafting is active, up to `depth` proposed tokens in one verification pass.
    -> (accepted drafts, next token (not fed yet), its logits if want_logits else None). See DRAFT_CONTRACT.md."""
    anchor = int(anchor)
    depth, probe = 0, False
    if self.draft_loaded():
        mode = self._d_mode
        if mode == "on":
            depth = self._d_depth
        elif mode == "auto":
            depth, probe = self._gov.choose()
        pos = self.position()
        if pos < 1 or pos + depth + 1 > self.lib_max_positions():
            depth = 0
        depth = min(depth, self.max_step_rows - 1)
    if getattr(self, "_sstats", None) is None:
        self.spec_stats(reset=True)
    st = self._sstats
    t0 = time.perf_counter()
    if depth <= 0:
        logits = self.step(anchor)
        nxt = int(np.argmax(logits)) if pick is None else int(pick(logits))
        dt = time.perf_counter() - t0
        if self.draft_loaded():
            self._gov.update_plain(dt)
        st["passes"] += 1
        st["tokens"] += 1
        st["verify_ms"] += 1e3 * dt
        return [], nxt, (logits if want_logits else None)
    pos0 = self.position()
    drafts = (ctypes.c_int * depth)()
    top = (ctypes.c_int * (depth + 1))()
    acc = self.lib.ink2_spec_pass(self.h, anchor, depth, drafts, top, 1 if pick is None else 0)
    if acc < 0:
        self._check(acc, "spec_pass")
    logits = None
    if pick is None:
        accepted = list(drafts[:acc])
        nxt = int(top[acc])
        if want_logits:
            logits = self._row_logits(acc)
    else:
        accepted = []
        for r in range(depth + 1):
            lg = self._row_logits(r)
            y = int(pick(lg))
            if r < depth and y == drafts[r]:
                accepted.append(y)
                continue
            nxt, logits = y, lg
            break
        self._check(self.lib.ink2_truncate(self.h, pos0 + 1 + len(accepted)), "truncate")
        if not want_logits:
            logits = None
    dt = time.perf_counter() - t0
    self.num_seen += 1 + len(accepted)
    self.seen_ids.append(anchor)
    self.seen_ids.extend(accepted)
    self._gov.update_spec(depth, len(accepted), dt, probe)
    dms = self._gov.t_draft[depth]
    st["passes"] += 1
    st["active_passes"] += 1
    st["probe_passes"] += 1 if probe else 0
    st["proposed"] += depth
    st["accepted"] += len(accepted)
    st["tokens"] += 1 + len(accepted)
    st["draft_ms"] += dms
    st["verify_ms"] += 1e3 * dt - dms
    st["depth_hist"][depth] += 1
    return accepted, nxt, logits


def spec_stats(self, reset=False):
    """Counters since the last reset: passes (verification passes, plain steps included), proposed, accepted,
    tokens (fed by spec_step), draft_ms (draft calls, at their measured time), verify_ms (the rest of the
    wall time of spec_step), active_passes (passes with a draft), probe_passes, depth_hist."""
    old = getattr(self, "_sstats", None)
    if reset or old is None:
        self._sstats = {"passes": 0, "proposed": 0, "accepted": 0, "tokens": 0, "draft_ms": 0.0, "verify_ms": 0.0,
                        "active_passes": 0, "probe_passes": 0, "depth_hist": [0] * (MAX_DEPTH + 1)}
    if old is None:
        return dict(self._sstats)
    out = dict(old)
    out["depth_hist"] = list(old["depth_hist"])
    return out


def lib_max_positions(self):
    lim = (ctypes.c_int * 4)()
    self.lib.ink2_limits(self.h, lim)
    return lim[3]


def install(cls):
    for f in (draft_load, draft_calibrate, draft_time, draft_loaded, draft_unload, draft_mode, draft_set, draft_propose,
              draft_commit_features, draft_memory, _row_logits, spec_step, spec_stats, lib_max_positions):
        setattr(cls, f.__name__, f)
