# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""ctypes wrapper of the native Inkling-2 Mini engine (text only).

    eng = Ink2Native().load(model_dir, backend="cuda", max_positions=4096)
    eng.feed(token_id)                    # a prompt token whose prediction is not needed
    logits = eng.step(token_id)           # numpy float32 [unpadded vocabulary]
    tokens = eng.generate(prompt_ids, 200, stop_ids=eng.stop_ids)
  DLL ABI version 2 (eng.abi >= 2):
    eng.feed_many(ids)                    # prompt tokens in passes of several rows
    logits, top, top_logit = eng.step_many(ids)   # several positions in one pass, per-row outputs
    eng.truncate(n)                       # drop the positions from n on (drafted tokens that were rejected)
  DLL ABI version 3 (eng.abi >= 3, eng.caps & CAP_PACKED_DENSE): dense matrices given packed
    eng = Ink2Native().load(model_dir, backend="cuda", checkpoint=reader)
        # reader: info() says "NVFP4" / "MXFP8" for a dense matrix, parts() gives its stored parts (ready bytes)
    eng = Ink2Native().load(model_dir, backend="cuda", dense_dir="packed_q4g4un4")   # folder of ink2_pack_dense.py

Needs numpy only (jinja2 and tokenizers for the chat helpers); no torch.
The checkpoint is read through yasakani/_ink2_ckpt.py (memory map) and handed to
the DLL tensor by tensor; the DLL copies every tensor to its backend.
"""

import ctypes
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import _ink2_ckpt as ink2_ckpt  # numpy only

TAKES_PACKED_DENSE = True  # load() hands packed dense matrices to a DLL that accepts them (ABI 3, CAP_PACKED_DENSE)
CAP_PACKED_DENSE = 1       # bit of ink2_capabilities()
CAP_DRAFTER = 2            # bit of ink2_capabilities(): the drafter calls of ink2_drafter_api.h
CAP_DRAFTER_PACKED = 4     # bit: the drafter takes packed matrices and a packed folded head (ink2_draft_abi() >= 2)
CAP_Q3 = 8                 # bit: the 3-bit format q3 for the drafter matrices
PACKED_ENCODINGS = {"NVFP4": "nvfp4", "MXFP8": "mxfp8"}  # info() dtype of a packed dense matrix -> its encoding

FULL_WEIGHTS = "model-00001-of-00001.safetensors"
TINY_WEIGHTS = "model.safetensors"
PREFIX = "model.llm."

GET_HIDDEN, GET_ROUTER_INDEX, GET_ROUTER_WEIGHT, GET_ROUTER_LOGITS, GET_EMBED, GET_FINAL_HIDDEN, GET_LOGITS = range(7)
PROFILE_SLOTS = (
    "expert products",
    "attention",
    "other matrix products",
    "output projection",
    "mixing and norms",
    "message gates",
    "convolutions and router",
    "copy of the logits",
)


class Ink2Config(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_int) for n in (
            "vocab", "embed_rows", "hidden", "layers", "nh_full", "nkv_full", "hd_full", "nh_sw", "nkv_sw", "hd_sw",
            "window", "d_rel", "rel_extent", "conv_k", "expert_inner", "experts", "top_k", "bar_block", "loop_start",
            "loop_end", "loop_passes", "gate_rank", "n_floor", "max_positions",
        )]
        + [(n, ctypes.c_float) for n in ("eps", "qk_thr", "rel_thr", "alpha", "route_scale", "gate_cap", "up_cap", "logit_cap")]
        + [("sliding", ctypes.c_int * 64)]
    )


def text_config(config):
    """The values of ink2_ref.Config of our CPU reference implementation (same defaults and aliases), as a dict."""
    g = config.get
    out = {
        "vocab_size": g("vocab_size", 201024),
        "unpadded_vocab_size": g("unpadded_vocab_size", 200058),
        "hidden_size": g("hidden_size", 2304),
        "num_hidden_layers": g("num_hidden_layers", 24),
        "num_attention_heads": g("num_attention_heads", 24),
        "num_key_value_heads": g("num_key_value_heads", 8),
        "head_dim": g("head_dim", 128),
        "swa_num_attention_heads": g("swa_num_attention_heads", 24),
        "swa_num_key_value_heads": g("swa_num_key_value_heads", 8),
        "swa_head_dim": g("swa_head_dim", 128),
        "sliding_window_size": g("sliding_window_size", g("sliding_window", 1024)),
        "d_rel": g("d_rel", 16),
        "rel_extent": g("rel_extent", 1024),
        "log_scaling_n_floor": g("log_scaling_n_floor", 8192),
        "log_scaling_alpha": g("log_scaling_alpha", 0.1),
        "rms_norm_eps": g("rms_norm_eps", 1e-6),
        "conv_kernel_size": g("conv_kernel_size", g("sconv_kernel_size", 4)),
        "moe_intermediate_size": g("moe_intermediate_size", g("intermediate_size", 1152)),
        "n_routed_experts": g("n_routed_experts", g("num_local_experts", 128)),
        "num_experts_per_tok": g("num_experts_per_tok", 8),
        "route_scale": g("route_scale", g("routed_scaling_factor", 8.0)),
        "qk_l2_clip_threshold": g("qk_l2_clip_threshold", 22.627416997969522),
        "rel_l2_clip_threshold": g("rel_l2_clip_threshold", 8.0),
        "n_layers_per_attn_res_block": g("n_layers_per_attn_res_block", 6),
        "mlp_gate_softcap": g("mlp_gate_softcap", 4.0),
        "mlp_up_softcap": g("mlp_up_softcap", 25.0),
        "final_logit_softcapping": g("final_logit_softcapping", 30.0),
    }
    if g("scoring_func", "sigmoid") != "sigmoid":
        raise ValueError("config.json: scoring_func %r is not supported (sigmoid only)" % (g("scoring_func"),))
    loop = g("loop_config") or {}
    out["loop_start"] = loop.get("core_start", g("loop_layer_index_start", 6))
    out["loop_end"] = loop.get("core_end", g("loop_layer_index_end", 18))
    out["loop_passes"] = loop.get("num_passes", g("loop_num_passes", 3))
    out["gate_rank"] = loop.get("gate_rank", g("loop_gate_rank", 64))
    n = out["num_hidden_layers"]
    types = g("layer_types") or ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]
    local = g("local_layer_ids")
    if local is not None:
        if [i for i, t in enumerate(types) if t == "hybrid_sliding"] != list(local):
            raise ValueError("config.json: layer_types disagrees with local_layer_ids")
    if len(types) != n:
        raise ValueError("config.json: layer_types has %d entries for %d layers" % (len(types), n))
    out["sliding"] = [t == "hybrid_sliding" for t in types]
    return out


def render_chat(model_dir, messages, **template_kwargs):
    """chat_template.jinja -> text without the transformers package (in jinja2's sandbox, as the package's formatter)."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8") as f:
        source = f.read()
    source = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", source)

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source).render(messages=messages, add_generation_prompt=True, tools=None, **template_kwargs)


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def stop_token_ids(model_dir):
    """eos_token_id of generation_config.json (config.json as fallback)."""
    for name in ("generation_config.json", "config.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id")
            if eos is not None:
                return [int(e) for e in (eos if isinstance(eos, list) else [eos])]
    return []


def make_sampler(temperature=0.0, top_p=1.0, seed=0):
    """Greedy when temperature <= 0, otherwise temperature / top-p sampling (numpy, seeded)."""
    if temperature <= 0:
        return lambda logits: int(np.argmax(logits))
    rng = np.random.default_rng(seed)

    def sample(logits):
        x = logits.astype(np.float64) / temperature
        p = np.exp(x - x.max())
        p /= p.sum()
        if top_p < 1.0:
            order = np.argsort(-p)
            sp = p[order]
            keep = (np.cumsum(sp) - sp) < top_p
            sp = np.where(keep, sp, 0.0)
            return int(order[rng.choice(len(sp), p=sp / sp.sum())])
        return int(rng.choice(len(p), p=p))

    return sample


def default_dll(backend):
    name = "ink2_cuda.dll" if backend == "cuda" else "ink2_host.dll"
    return os.path.join(HERE, "build_cuda" if backend == "cuda" else "build_host", name)


def open_dll(path):
    lib = ctypes.CDLL(path)
    vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    lib.ink2_build_info.restype = ctypes.c_char_p
    lib.ink2_create.restype = vp
    lib.ink2_create.argtypes = [ctypes.POINTER(Ink2Config), ctypes.c_char_p]
    lib.ink2_destroy.argtypes = [vp]
    lib.ink2_destroy.restype = None
    lib.ink2_last_error.restype = ctypes.c_char_p
    lib.ink2_last_error.argtypes = [vp]
    lib.ink2_set_tensor.argtypes = [vp, ctypes.c_char_p, vp, cll]
    lib.ink2_finalize.argtypes = [vp]
    lib.ink2_reset.argtypes = [vp]
    lib.ink2_step.argtypes = [vp, ci, vp]
    lib.ink2_feed.argtypes = [vp, ci]
    lib.ink2_set_option.argtypes = [vp, ctypes.c_char_p, ci]
    lib.ink2_timing.argtypes = [vp, ctypes.POINTER(ctypes.c_double)]
    lib.ink2_num_visits.argtypes = [vp]
    lib.ink2_visit_info.argtypes = [vp, ci, ctypes.POINTER(ci), ctypes.POINTER(ci)]
    lib.ink2_get.argtypes = [vp, ci, ci, vp, cll]
    lib.ink2_memory_used.argtypes = [vp]
    lib.ink2_memory_used.restype = cll
    lib.ink2_device_memory.argtypes = [vp, ctypes.POINTER(cll), ctypes.POINTER(cll)]
    lib.ink2_profile.argtypes = [vp, ci, ctypes.POINTER(ctypes.c_double), ci]
    lib.ink2_kernel_name.restype = ctypes.c_char_p
    lib.ink2_kernel_name.argtypes = [ci]
    lib.ink2_bench.argtypes = [vp, ci, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(cll), ci]
    lib.ink2_capture.argtypes = [vp, ci]
    lib.ink2_capture_count.argtypes = [vp]
    lib.ink2_capture_get.argtypes = [vp, ci, ctypes.c_char_p, ci, vp, cll]
    lib.ink2_selftest.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ci]
    try:
        lib.ink2_abi_version.restype = ci
        lib.ink2_abi_version.argtypes = []
        lib.abi = lib.ink2_abi_version()
    except AttributeError:  # a DLL of version 1
        lib.abi = 1
    if lib.abi >= 2:
        lib.ink2_feed_many.argtypes = [vp, vp, ci]
        lib.ink2_step_many.argtypes = [vp, vp, ci, vp, vp, vp]
        lib.ink2_truncate.argtypes = [vp, ci]
        lib.ink2_position.argtypes = [vp]
        lib.ink2_limits.argtypes = [vp, ctypes.POINTER(ci)]
        lib.ink2_get_row.argtypes = [vp, ci, ci, ci, vp, cll]
    lib.caps = 0
    if lib.abi >= 3:
        lib.ink2_capabilities.restype = ci
        lib.ink2_capabilities.argtypes = []
        lib.ink2_packed_dense.restype = ci
        lib.ink2_packed_dense.argtypes = [vp]
        lib.caps = int(lib.ink2_capabilities())
    return lib


def capabilities(dll):
    """ink2_capabilities() of a DLL (0 for a DLL before version 3)."""
    return open_dll(dll).caps


def selftest(backend="cuda", dll=None):
    """Every kernel on random inputs, CPU emulation against `backend`. -> (failures, report text)"""
    lib = open_dll(dll or default_dll(backend))
    buf = ctypes.create_string_buffer(1 << 16)
    rc = lib.ink2_selftest(backend.encode(), buf, len(buf))
    return rc, buf.value.decode("utf-8", "replace")


def microbench(dll=None):
    """Launch and barrier costs of the device (cuda build). -> (return code, report text)"""
    lib = open_dll(dll or default_dll("cuda"))
    lib.ink2_microbench.argtypes = [ctypes.c_char_p, ctypes.c_int]
    buf = ctypes.create_string_buffer(1 << 16)
    rc = lib.ink2_microbench(buf, len(buf))
    return rc, buf.value.decode("utf-8", "replace")


class Ink2Native:
    def __init__(self):
        self.lib = None
        self.h = None
        self.cfg = None

    # -- loading -----------------------------------------------------------
    def load(self, model_dir, backend="host", max_positions=4096, dll=None, weights=None, log=None, max_rows=None,
             rollback=None, chunk=None, dense_dir=None, dense_mode="packed", dense_only=None, checkpoint=None):
        """max_rows / rollback / chunk (DLL version 2): largest pass, positions truncate() can always drop,
        rows per pass of feed_many(); None = the defaults of the DLL.
        dense_dir: folder of packed dense matrices (ink2_pack_dense.py); every matrix listed in its
        manifest.json replaces the bfloat16 one of the checkpoint. dense_mode "packed" sends the packed bytes
        (multiplied by the packed kernels); "decoded" sends the same values decoded to bfloat16 (exact), the
        nvfp4 matrices without their matrix factor, which is sent as "<name>_factor" (check of the packed path).
        dense_only: optional function name -> bool; only the matrices it accepts are taken from dense_dir.
        checkpoint (DLL version 3): a reader object to take the weights from instead of opening the safetensors
        file, with the interface of ink2_ckpt.Checkpoint (info, raw, f32, expert_format). A dense matrix whose
        info() dtype is "NVFP4" or "MXFP8" is taken from reader.parts(name), ready bytes as stored: nvfp4 weight
        u8 [R, C/2], scale e4m3 [R, C/16], amax f32 [1], denominator f32 [1] (the engine's factor is
        amax / denominator in float32); mxfp8 weight e4m3 [R, C], scale u8 [R, C/32]. dense_mode "packed" sends
        them as stored (needs CAP_PACKED_DENSE); "decoded" sends the same values in bfloat16 (check of the
        packed path)."""
        t0 = time.perf_counter()
        self.model_dir = model_dir
        self.backend = backend
        self.lib = open_dll(dll or default_dll(backend))
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        self.cfg = cfg = text_config(config["text_config"])
        if weights is None:
            weights = FULL_WEIGHTS if os.path.exists(os.path.join(model_dir, FULL_WEIGHTS)) else TINY_WEIGHTS
        self.ckpt = ckpt = checkpoint if checkpoint is not None else ink2_ckpt.Checkpoint(os.path.join(model_dir, weights))

        c = Ink2Config()
        c.vocab = cfg["unpadded_vocab_size"]
        c.embed_rows = ckpt.info(PREFIX + "embed.weight")[1][0]
        c.hidden, c.layers = cfg["hidden_size"], cfg["num_hidden_layers"]
        c.nh_full, c.nkv_full, c.hd_full = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        c.nh_sw, c.nkv_sw, c.hd_sw = cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"]
        c.window, c.d_rel, c.rel_extent = cfg["sliding_window_size"], cfg["d_rel"], cfg["rel_extent"]
        c.conv_k, c.expert_inner = cfg["conv_kernel_size"], cfg["moe_intermediate_size"]
        c.experts, c.top_k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
        c.bar_block = cfg["n_layers_per_attn_res_block"]
        c.loop_start, c.loop_end, c.loop_passes = cfg["loop_start"], cfg["loop_end"], cfg["loop_passes"]
        c.gate_rank, c.n_floor, c.max_positions = cfg["gate_rank"], cfg["log_scaling_n_floor"], max_positions
        c.eps, c.qk_thr, c.rel_thr = cfg["rms_norm_eps"], cfg["qk_l2_clip_threshold"], cfg["rel_l2_clip_threshold"]
        c.alpha, c.route_scale = cfg["log_scaling_alpha"], cfg["route_scale"]
        c.gate_cap, c.up_cap, c.logit_cap = cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"], cfg["final_logit_softcapping"]
        for i, s in enumerate(cfg["sliding"]):
            c.sliding[i] = int(s)
        self.h = self.lib.ink2_create(ctypes.byref(c), backend.encode())
        if not self.h:
            raise RuntimeError("ink2_create: " + self.lib.ink2_last_error(None).decode("utf-8", "replace"))
        self.vocab, self.hidden_size = c.vocab, c.hidden
        self.abi = self.lib.abi
        for name, value in (("max_rows", max_rows), ("rollback", rollback), ("chunk", chunk)):
            if value is not None:
                if self.abi < 2:
                    raise RuntimeError("option %s needs a DLL of version 2" % name)
                self.set_option(name, value)

        sent = [0, 0]
        dense = None
        assert dense_mode in ("packed", "decoded"), dense_mode
        if dense_dir is not None:
            with open(os.path.join(dense_dir, "manifest.json"), encoding="utf-8") as f:
                dense = json.load(f)["matrices"]
        self.dense_dir, self.dense_mode = dense_dir, (dense_mode if dense is not None else "bf16")
        self.dense_inexact = 0
        self.dense_packed_sent = 0  # dense matrices handed over in a packed encoding (as stored, or decoded)
        packed_ok = bool(self.lib.caps & CAP_PACKED_DENSE)

        def put(name, ckpt_name, dtype, rows=None):
            if dense is not None and name in dense and (dense_only is None or dense_only(name)):
                sent[1] += self._set_dense(name, dense[name], dense_dir, dense_mode, ckpt_name, rows)
                sent[0] += 1
                return
            stored, _ = ckpt.info(PREFIX + ckpt_name)
            if stored in PACKED_ENCODINGS and "BF16" in dtype:  # a dense matrix the reader holds packed
                if self.abi < 3 or (dense_mode == "packed" and not packed_ok):
                    raise RuntimeError("%s is stored packed (%s) but this DLL (version %d, capabilities %d) does not take packed"
                                       " dense matrices" % (ckpt_name, stored, self.abi, self.lib.caps))
                if self.dense_mode == "bf16":
                    self.dense_mode = dense_mode
                sent[1] += self._send_packed(name, PACKED_ENCODINGS[stored], ckpt.parts(PREFIX + ckpt_name), dense_mode, rows)
                sent[0] += 1
                return
            assert stored in dtype, "%s: stored as %s, expected %s" % (ckpt_name, stored, dtype)
            view = ckpt.raw(PREFIX + ckpt_name)
            if rows is not None:
                view = view[:rows]
            self._set(name, view)
            sent[0] += 1
            sent[1] += view.nbytes

        put("embed", "embed.weight", ("BF16",))
        put("embed_norm", "embed_norm.weight", ("BF16",))
        put("norm", "norm.weight", ("BF16",))
        put("output_query", "attn_res.output_query", ("F32",))
        put("unembed", "unembed.weight", ("BF16",), rows=c.vocab)  # Inkling2UnpadVocab
        if c.loop_passes > 1:
            merged = ink2_ckpt.merge_message_queries(
                ckpt.f32(PREFIX + "middle_cycle_cross_pass_queries"), ckpt.f32(PREFIX + "middle_cycle_cross_pass_kv_queries")
            )
            self._set("msg_q", np.ascontiguousarray(merged, dtype=np.float32))
        for i in range(c.layers):
            lp, n = "layers.%d." % i, "L%d." % i
            a = lp + "attn."
            if i > 0:
                put(n + "attn_query", "attn_res.attn_queries.%d" % (i - 1), ("F32",))
            put(n + "mlp_query", "attn_res.mlp_queries.%d" % i, ("F32",))
            if cfg["sliding"][i]:
                put(n + "attn_norm", lp + "attn_norm.weight", ("BF16",))
                put(n + "k_conv", a + "k_sconv.weight", ("BF16",))
                put(n + "v_conv", a + "v_sconv.weight", ("BF16",))
            else:
                for s in "qkv":
                    put(n + s + "_norm", lp + s + "_input_norm.weight", ("BF16",))
                for s in "kv":
                    put(n + s + "_static", a + s + "_sconv.static_w", ("BF16",))
                    put(n + s + "_wproj", a + s + "_sconv.weight_proj", ("BF16",))
            put(n + "mlp_norm", lp + "mlp_norm.weight", ("BF16",))
            for short, name in (("wq", "wq_du.weight"), ("wk", "wk_dv.weight"), ("wv", "wv_dv.weight"), ("wr", "wr_du.weight"),
                                ("wo", "wo_ud.weight"), ("w_gate", "gate_attn.weight"), ("w_sink", "attn_sink_proj.weight"),
                                ("q_scale", "q_l2_clip.scale"), ("rel_scale", "rel_l2_clip.scale"), ("rel_proj", "rel_logits_proj.proj")):
                put(n + short, a + name, ("BF16",))
            put(n + "attn_sconv", lp + "attn_sconv.weight", ("BF16",))
            put(n + "mlp_sconv", lp + "mlp_sconv.weight", ("BF16",))
            put(n + "router_w", lp + "mlp.gate.weight", ("BF16",))
            put(n + "router_bias", lp + "mlp.gate.bias", ("F32",))
            for which in ("w13", "w2"):
                ep = lp + "mlp.experts." + which
                put(n + which, ep + "_weight", ("U8", "F8_E4M3"))
                put(n + which + "_scale", ep + "_weight_scale", ("U8", "F8_E4M3"))
                if ckpt.expert_format(PREFIX + ep) == "nvfp4":
                    put(n + which + "_amax", ep + "_weight_amax", ("F32",))
                    put(n + which + "_den", ep + "_weight_outer_scale_denominator", ("F32",))
            if cfg["loop_start"] <= i < cfg["loop_end"] and c.loop_passes > 1:
                for ckpt_gate, short in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                    for part in ("down", "forget", "gain"):
                        put(n + "gate_%s_%s" % (short, part), lp + "controllers.%s.%s_weight" % (ckpt_gate, part), ("BF16",))
            if log is not None:
                log("  layer %2d sent, %.2f GB so far" % (i, sent[1] / 1e9))
        self._check(self.lib.ink2_finalize(self.h), "finalize")
        self.num_visits = self.lib.ink2_num_visits(self.h)
        self.visits = []
        for v in range(self.num_visits):
            layer, pas = ctypes.c_int(), ctypes.c_int()
            self.lib.ink2_visit_info(self.h, v, ctypes.byref(layer), ctypes.byref(pas))
            self.visits.append((layer.value, pas.value))
        self.top_k, self.experts = c.top_k, c.experts
        self.stop_ids = stop_token_ids(model_dir)
        self.tensors_sent, self.bytes_sent = sent
        self.packed_dense = int(self.lib.ink2_packed_dense(self.h)) if self.abi >= 3 else 0  # matrices the engine holds packed
        if backend == "cuda":  # replay the kernels of a token as one recorded graph (about 1.35 times faster)
            self.set_option("graph", 1)
        self.load_seconds = time.perf_counter() - t0
        self.max_rows = self.max_step_rows = 1
        self.rollback = 0
        if self.abi >= 2:
            lim = (ctypes.c_int * 4)()
            self.lib.ink2_limits(self.h, lim)
            self.max_rows, self.max_step_rows, self.rollback = lim[0], lim[1], lim[2]
        self.reset()
        return self

    def _set_dense(self, name, info, folder, mode, ckpt_name, rows):
        """Sends one matrix of the packed folder (see load). Returns the bytes sent."""
        stored, shape = self.ckpt.info(PREFIX + ckpt_name)
        want = [shape[0] if rows is None else rows, shape[1]]
        assert list(info["shape"]) == want, (name, info["shape"], want)
        w = np.load(os.path.join(folder, name + ".w.npy"))
        sc = np.load(os.path.join(folder, name + ".s.npy"))
        nv = info["format"] == "nvfp4"
        fac = np.load(os.path.join(folder, name + ".f.npy")).astype(np.float32).reshape(1) if nv else None
        return self._send_dense(name, w, sc, fac, nv, mode, want)

    def _send_packed(self, name, encoding, parts, mode, rows):
        """Sends one dense matrix given as its stored parts (ready bytes of the caller's reader). rows: the rows
        the engine uses (the unpadded vocabulary of the output projection), None = all. Returns the bytes sent."""
        nv = encoding == "nvfp4"
        w = np.asarray(parts["weight"])
        sc = np.asarray(parts["scale"])
        assert w.dtype.itemsize == 1 and sc.dtype.itemsize == 1 and w.ndim == 2 and sc.ndim == 2, (name, w.dtype, w.shape, sc.shape)
        cols = w.shape[1] * (2 if nv else 1)
        assert sc.shape == (w.shape[0], cols // (16 if nv else 32)), (name, w.shape, sc.shape)
        r = w.shape[0] if rows is None else int(rows)
        assert r <= w.shape[0], (name, r, w.shape)
        w, sc = w[:r].view(np.uint8), sc[:r].view(np.uint8)
        fac = None
        if nv:  # the matrix factor amax / denominator, in float32 as the decode formula
            amax = np.asarray(parts["amax"], dtype=np.float32).reshape(-1)
            den = np.asarray(parts["denominator"], dtype=np.float32).reshape(-1)
            assert amax.size == 1 and den.size == 1, (name, amax.shape, den.shape)
            fac = (amax / den).astype(np.float32).reshape(1)
        return self._send_dense(name, w, sc, fac, nv, mode, [r, cols])

    def _send_dense(self, name, w, sc, fac, nv, mode, want):
        """Packed bytes w / sc (+ the nvfp4 factor) of one matrix [want rows, want cols], as stored or decoded."""
        self.dense_packed_sent += 1
        if mode == "packed":
            self._set(name, w)
            self._set(name + "_scale", sc)
            if nv:
                self._set(name + "_factor", fac)
            return w.nbytes + sc.nbytes + (4 if nv else 0)
        # decoded: exact bfloat16 values (nvfp4 without the matrix factor), in row chunks
        r, cols = want
        assert w.shape[0] == r, (name, w.shape, want)
        out = np.empty((r, cols), dtype=np.uint16)
        step = 8192
        for r0 in range(0, r, step):
            if nv:
                v = ink2_ckpt.decode_nvfp4(w[None, r0:r0 + step], sc[None, r0:r0 + step], np.ones(1, np.float32), np.float32(1.0))[0]
            else:
                v = ink2_ckpt.decode_mxfp8(w[None, r0:r0 + step], sc[None, r0:r0 + step])[0]
            bits = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
            self.dense_inexact += int(np.count_nonzero(bits & 0xFFFF))  # values bfloat16 cannot hold (must stay 0)
            out[r0:r0 + step] = (bits >> 16).astype(np.uint16)
        self._set(name, out)
        if nv:
            self._set(name + "_factor", fac)
        return out.nbytes + (4 if nv else 0)

    def _set(self, name, array):
        array = np.ascontiguousarray(array)
        self._check(self.lib.ink2_set_tensor(self.h, name.encode(), ctypes.c_void_p(array.ctypes.data), array.nbytes), name)

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError("%s: %s" % (what, self.lib.ink2_last_error(self.h).decode("utf-8", "replace")))

    def close(self):
        if self.h:
            self.lib.ink2_destroy(self.h)
            self.h = None

    # -- running -----------------------------------------------------------
    def reset(self):
        self._check(self.lib.ink2_reset(self.h), "reset")
        self.num_seen = 0
        self.seen_ids = []  # the token ids of the positions seen (for truncate beyond the rollback)

    def step(self, token):
        logits = np.empty(self.vocab, dtype=np.float32)
        self._check(self.lib.ink2_step(self.h, int(token), ctypes.c_void_p(logits.ctypes.data)), "step")
        self.num_seen += 1
        self.seen_ids.append(int(token))
        return logits

    def _need(self, version, what):
        if self.abi < version:
            raise RuntimeError("%s needs a DLL of version %d (this one: %d)" % (what, version, self.abi))

    def feed_many(self, ids):
        """Prompt tokens whose predictions are not needed, in passes of several rows (same results as feed())."""
        self._need(2, "feed_many")
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32).reshape(-1))
        if ids.size:
            self._check(self.lib.ink2_feed_many(self.h, ctypes.c_void_p(ids.ctypes.data), int(ids.size)), "feed_many")
        self.num_seen += int(ids.size)
        self.seen_ids.extend(int(t) for t in ids)

    def step_many(self, ids, logits=True):
        """len(ids) <= max_step_rows consecutive tokens in one pass (same results as step() on each).
        Returns (logits [n, vocab] float32 or None, argmax [n] int32, largest logit [n] float32)."""
        self._need(2, "step_many")
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32).reshape(-1))
        n = int(ids.size)
        out = np.empty((n, self.vocab), dtype=np.float32) if logits else None
        top = np.empty(n, dtype=np.int32)
        val = np.empty(n, dtype=np.float32)
        self._check(self.lib.ink2_step_many(self.h, ctypes.c_void_p(ids.ctypes.data), n,
                                            None if out is None else ctypes.c_void_p(out.ctypes.data),
                                            ctypes.c_void_p(top.ctypes.data), ctypes.c_void_p(val.ctypes.data)), "step_many")
        self.num_seen += n
        self.seen_ids.extend(int(t) for t in ids)
        return out, top, val

    def truncate(self, n):
        """Drops the positions from n on. Up to `rollback` positions without recomputing; a deeper rollback
        resets and feeds the kept token ids again."""
        self._need(2, "truncate")
        n = int(n)
        if not 0 <= n <= self.num_seen:
            raise ValueError("truncate to %d of %d positions" % (n, self.num_seen))
        if self.num_seen - n <= self.rollback:
            self._check(self.lib.ink2_truncate(self.h, n), "truncate")
            self.num_seen = n
            del self.seen_ids[n:]
        else:
            keep = self.seen_ids[:n]
            self.reset()
            self.feed_many(keep)

    def position(self):
        return int(self.lib.ink2_position(self.h)) if self.abi >= 2 else self.num_seen

    def set_option(self, name, value):
        """"graph": 1 = record the kernels of a step once and replay them with one call (cuda backend)."""
        self._check(self.lib.ink2_set_option(self.h, name.encode(), int(value)), "set_option")

    def timing(self):
        """Since the last call: seconds queueing kernels, seconds waiting for the logits, steps, launches, graphs recorded."""
        out = (ctypes.c_double * 4)()
        graphs = self.lib.ink2_timing(self.h, out)
        return {"queue_seconds": out[0], "wait_seconds": out[1], "steps": int(out[2]), "launches": int(out[3]), "graphs_recorded": graphs}

    def feed(self, token):
        """A token whose prediction is not needed (prompt tokens before the last): skips the output projection."""
        self._check(self.lib.ink2_feed(self.h, int(token)), "feed")
        self.num_seen += 1
        self.seen_ids.append(int(token))

    def _get(self, what, visit, count, dtype=np.float32, row=0):
        out = np.empty(count, dtype=dtype)
        if row == 0:
            self._check(self.lib.ink2_get(self.h, what, visit, ctypes.c_void_p(out.ctypes.data), out.nbytes), "get")
        else:
            self._need(2, "rows of a pass")
            self._check(self.lib.ink2_get_row(self.h, what, visit, int(row), ctypes.c_void_p(out.ctypes.data), out.nbytes), "get_row")
        return out

    def hidden(self, visit, row=0):
        """Output of a layer visit for the last token (row: of the last pass), float32 [hidden]."""
        return self._get(GET_HIDDEN, visit, self.hidden_size, row=row)

    def router(self, visit, row=0):
        """-> (expert indices int32 [top_k] increasing, weights [top_k], logits [experts]) of the last token (row: of the last pass)."""
        return (
            self._get(GET_ROUTER_INDEX, visit, self.top_k, np.int32, row=row),
            self._get(GET_ROUTER_WEIGHT, visit, self.top_k, row=row),
            self._get(GET_ROUTER_LOGITS, visit, self.experts, row=row),
        )

    def embed(self, row=0):
        return self._get(GET_EMBED, 0, self.hidden_size, row=row)

    def final_hidden(self, row=0):
        return self._get(GET_FINAL_HIDDEN, 0, self.hidden_size, row=row)

    def memory_used(self):
        return int(self.lib.ink2_memory_used(self.h))

    def device_memory(self):
        """-> (free bytes, total bytes) of the device, or None on the host backend."""
        free, total = ctypes.c_longlong(), ctypes.c_longlong()
        if self.lib.ink2_device_memory(self.h, ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return free.value, total.value

    def profile(self, enable):
        """Switches the per-part timers (device synchronised after every kernel) and returns the seconds so far."""
        out = (ctypes.c_double * len(PROFILE_SLOTS))()
        self.lib.ink2_profile(self.h, int(enable), out, len(PROFILE_SLOTS))
        return dict(zip(PROFILE_SLOTS, list(out)))

    def bench(self, reps):
        """Timing of every kernel alone: with reps > 0 each kernel of the following steps runs reps times in a row
        (those steps give no valid result; reset afterwards). Returns {kernel: (seconds of one run summed over the
        launches, launches)} since the last call."""
        secs = (ctypes.c_double * 128)()
        counts = (ctypes.c_longlong * 128)()
        n = min(128, self.lib.ink2_bench(self.h, int(reps), secs, counts, 128))
        return {self.lib.ink2_kernel_name(i).decode(): (secs[i], counts[i]) for i in range(n)}

    def capture(self, visit):
        """Debug: keep the buffers written by every operation of one visit (-1 = off) during the next steps."""
        self.lib.ink2_capture(self.h, int(visit))

    def captured(self):
        """-> list of (operation name, float32 array) of the last step, in execution order."""
        out = []
        name = ctypes.create_string_buffer(64)
        for i in range(self.lib.ink2_capture_count(self.h)):
            n = self.lib.ink2_capture_get(self.h, i, name, len(name), None, 0)
            data = np.empty(n, dtype=np.float32)
            self.lib.ink2_capture_get(self.h, i, name, len(name), ctypes.c_void_p(data.ctypes.data), n)
            out.append((name.value.decode(), data))
        return out

    def generate(self, ids, max_new_tokens, stop_ids=(), sampler=None, on_token=None):
        """Feeds `ids` on top of the current state, then generates token by token.
        Returns (tokens, stopped on a stop token). sampler: logits -> token id (default greedy)."""
        sampler = sampler or make_sampler()
        stop_ids = set(stop_ids)
        logits = None
        ids = list(ids)
        if self.abi >= 2:
            self.feed_many(ids[:-1])
        else:
            for token in ids[:-1]:
                self.feed(token)
        if ids:
            logits = self.step(ids[-1])
        tokens, stopped = [], False
        for i in range(max_new_tokens):
            token = sampler(logits)
            tokens.append(token)
            if on_token is not None:
                on_token(tokens)
            if token in stop_ids:
                stopped = True
                break
            if i + 1 < max_new_tokens:
                logits = self.step(token)
        return tokens, stopped

    def chat_ids(self, user_text, **template_kwargs):
        """One user message through chat_template.jinja -> (text, ids)."""
        text = render_chat(self.model_dir, [{"role": "user", "content": user_text}], **template_kwargs)
        return text, load_tokenizer(self.model_dir).encode(text, add_special_tokens=False).ids

from . import _ink2_drafter_native as ink2_drafter_native  # noqa: E402  (the drafter calls)

ink2_drafter_native.install(Ink2Native)
