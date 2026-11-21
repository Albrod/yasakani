# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Checkpoint reader for Inkling-2 Mini (text decoder).

Reads a safetensors file through a memory map and returns tensors by their
checkpoint name. The quantized expert tensors are decoded to float32 with the
same arithmetic as the official loader:

    core_model_loading.py :: Inkling2LoopDequantizeExperts.convert

Where the official loader rounds to bfloat16 (when the model is loaded with
dtype=bfloat16, as in the official tests), and this reader does not:
  1. core_model_loading.py, the loading loop: every source tensor of a
     converter goes through spawn_materialize(..., _dtype) = tensor.to(dtype)
     BEFORE the dequantizer runs. Codes, block scales and the MXFP8 bytes are
     exact in bfloat16; `amax` (float32) is rounded to bfloat16 there.
  2. Inkling2LoopDequantizeExperts.convert ends with `.to(dtype)`: the float32
     product is rounded to bfloat16.
This reader returns the float32 product of the stored values (equal to the
official loader with dtype=float32). `expert_weight(..., official_bf16=True)`
applies both roundings so that their effect can be measured.

Number formats (cross-checked with nvfp4_tensor.py and mxfp8_tensor.py):
  NVFP4  weight : uint8, two E2M1 codes per byte, EVEN element in the LOW nibble
         scale  : one E4M3 byte for every 16 consecutive values of a row
         amax, outer_scale_denominator : float32, one factor per expert
         value  = (e2m1[code] * e4m3[scale]) * (amax[expert] / denominator)
  MXFP8  weight : E4M3 byte per value
         scale  : uint8 biased exponent for every 32 consecutive values of a row
         value  = e4m3[weight] * 2 ** (scale - 127)

No dependency besides numpy.
"""

import json
import re
import struct

import numpy as np

# Inkling2LoopDequantizeExperts.E2M1_VALUES (index = 4-bit code; bit 3 is the sign)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

NVFP4_BLOCK = 16  # values per E4M3 block scale
MXFP8_BLOCK = 32  # values per power-of-two block scale

_NP_DTYPES = {
    "F32": "<f4",
    "BF16": "<u2",  # raw bits, see bf16_to_f32
    "U8": "u1",
    "F8_E4M3": "u1",  # raw bits, see E4M3_TABLE
    "I64": "<i8",
    "I32": "<i4",
    "F16": "<f2",
    "F64": "<f8",
}


def make_e4m3_table():
    """float8_e4m3fn: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.

    exponent 0 is subnormal (m/8 * 2**-6); there is no infinity; only the
    all-ones pattern (0x7F / 0xFF) is NaN; the largest value is 448.
    """
    table = np.zeros(256, dtype=np.float64)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            value = (mantissa / 8.0) * 2.0**-6
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = sign * value
    return table.astype(np.float32)


E4M3_TABLE = make_e4m3_table()


def make_nvfp4_byte_table():
    """byte -> (even element, odd element), as in the official loader:
    byte_to_values = stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)"""
    e2m1 = np.array(E2M1_VALUES, dtype=np.float32)
    byte = np.arange(256)
    return np.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), axis=-1)


NVFP4_BYTE_TABLE = make_nvfp4_byte_table()


def bf16_to_f32(bits_u16):
    """bfloat16 bits -> float32 (exact: bfloat16 is the upper half of a float32)."""
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)


def round_to_bf16(x_f32):
    """float32 -> nearest bfloat16 (ties to even) -> float32. This is the cast
    the official loader applies to the dequantized experts when the model runs
    in bfloat16. NaN and overflow are not handled (not needed for weights)."""
    bits = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((bits >> 16) << 16).astype(np.uint32).view(np.float32)


def decode_nvfp4(codes_u8, scale_e4m3_u8, amax_f32, denominator_f32):
    """codes [n, rows, cols/2] uint8, scale [n, rows, cols/16] E4M3 bytes,
    amax [n] float32, denominator scalar float32 -> float32 [n, rows, cols].
    Same order of float32 operations as the official loader."""
    n, rows, half = codes_u8.shape
    cols = half * 2
    block_scale = E4M3_TABLE[scale_e4m3_u8]  # scale[experts].float()
    assert block_scale.shape == (n, rows, cols // NVFP4_BLOCK), (block_scale.shape, codes_u8.shape)
    expert_factor = (amax_f32.astype(np.float32) / np.float32(denominator_f32)).astype(np.float32)
    values = NVFP4_BYTE_TABLE[codes_u8].reshape(n, rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
    values = values * block_scale[..., None]
    values = values * expert_factor[:, None, None, None]
    return values.reshape(n, rows, cols)


def decode_mxfp8(weight_e4m3_u8, scale_u8):
    """weight [n, rows, cols] E4M3 bytes, scale [n, rows, cols/32] uint8
    -> float32 [n, rows, cols]. The official loader uses
    torch.exp2(scale.float() - 127); here the exact power of two (ldexp)."""
    n, rows, cols = weight_e4m3_u8.shape
    assert scale_u8.shape == (n, rows, cols // MXFP8_BLOCK), (scale_u8.shape, weight_e4m3_u8.shape)
    block_scale = np.ldexp(np.float32(1.0), scale_u8.astype(np.int32) - 127).astype(np.float32)
    values = E4M3_TABLE[weight_e4m3_u8].reshape(n, rows, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    values = values * block_scale[..., None]
    return values.reshape(n, rows, cols)


def deinterleave_rows(w13):
    """conversion_mapping.py: Interleave(dim=1) applied to gate_up_proj.
    The checkpoint stores w13 with rows interleaved (row 2i = gate i, row
    2i+1 = up i). The official model wants [all gate rows, all up rows]:
    reshape [n, I, 2, H] -> transpose(1, 2) -> reshape [n, 2I, H]."""
    n, rows, cols = w13.shape
    return np.ascontiguousarray(w13.reshape(n, rows // 2, 2, cols).transpose(0, 2, 1, 3)).reshape(n, rows, cols)


def merge_message_queries(queries, kv_queries):
    """core_model_loading.py :: Inkling2LoopMergeMessageQueries.convert.
    queries    [2 * n_loop_layers, H]: (query, mlp) of every loop layer
    kv_queries [2 * n_loop_layers, H]: (key, value) of every loop layer
    result     [n_loop_layers, 4, H]:  (query, key, value, mlp)"""
    q = queries.reshape(-1, 2, queries.shape[-1])
    kv = kv_queries.reshape(-1, 2, kv_queries.shape[-1])
    return np.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)


# ---------------------------------------------------------------------------
# Official key mapping (conversion_mapping.py, "inkling2" and
# "inkling2_loop_text"). Each entry is (regex, replacement); they are applied
# in this order, first match of each only, like WeightRenaming. The result is
# the parameter name inside Inkling2LoopForConditionalGeneration. It is used
# for documentation and for the used / unused table; the reference model reads
# tensors by their checkpoint name.
# ---------------------------------------------------------------------------
_COMPOSITE_RENAMES = [
    (r"model\.llm\.unembed\.", "lm_head."),
    (r"model\.llm\.", "model.language_model."),
    (r"model\.visual\.", "model.vision_tower."),
    (r"vision_tower\.backbone\.embeddings\.", "vision_tower.embeddings."),
    (r"vision_tower\.embeddings\.position_embedding", "vision_tower.embeddings.position_embedding.weight"),
    (r"vision_tower\.backbone\.ln_pre\.", "vision_tower.pre_layernorm."),
    (r"vision_tower\.backbone\.layers\.", "vision_tower.layers."),
    (r"vision_tower\.proj_in\.", "vision_tower.merger.linear_1."),
    (r"vision_tower\.proj_out\.", "vision_tower.merger.linear_2."),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_1\.gamma", r"\1.lambda_1"),
    (r"(vision_tower\.layers\.\d+)\.ls_2\.gamma", r"\1.lambda_2"),
]

_TEXT_RENAMES = [
    # inkling2_loop_text specific
    (r"controllers\.attn\.", "q_message_gate."),
    (r"controllers\.k\.", "k_message_gate."),
    (r"controllers\.v\.", "v_message_gate."),
    (r"controllers\.mlp\.", "mlp_message_gate."),
    (r"_message_gate\.down_weight", "_message_gate.down_proj.weight"),
    (r"_message_gate\.gain_weight", "_message_gate.message_gate_proj.weight"),
    (r"_message_gate\.forget_weight", "_message_gate.forget_gate_proj.weight"),
    # inkling2_text (the expert entries are replaced by the dequantizer, see official_key)
    (r"embed\.weight", "embed_tokens.weight"),
    (r"embed_norm\.", "embed_tokens.embed_norm."),
    (r"mlp.gate.bias", "mlp.gate.e_score_correction_bias"),
    (r"\.attn\.", ".self_attn."),
    (r"self_attn\.wq_du", "self_attn.q_proj.linear"),
    (r"self_attn\.wk_dv", "self_attn.k_proj.linear"),
    (r"self_attn\.wv_dv", "self_attn.v_proj.linear"),
    (r"self_attn\.k_sconv\.", "self_attn.k_proj.sconv."),
    (r"self_attn\.v_sconv\.", "self_attn.v_proj.sconv."),
    (r"self_attn\.rel_logits_proj\.", "self_attn.relative_position_bias."),
    (r"self_attn\.wr_du", "self_attn.relative_position_bias.linear"),
    (r"self_attn\.q_l2_clip\.", "self_attn.q_proj.norm."),
    (r"self_attn\.rel_l2_clip\.", "self_attn.relative_position_bias.norm."),
    (r"self_attn\.wo_ud", "self_attn.o_proj.linear"),
    (r"self_attn\.gate_attn", "self_attn.o_proj.gate_proj"),
    (r"self_attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"attn_sconv\.", "post_attention_convolution."),
    (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"),
    (r"(convolution|sconv)\.static_w", r"\1.static_weight"),
    (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"),
    (r"mlp_norm", "post_attention_layernorm"),
    (r"attn_norm", "input_layernorm"),
    (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
    (r"attn_res\.mlp_queries\.(\d+)", r"layers.\1.mlp_input.query"),
    (r"attn_res\.output_query", "final_input.query"),
]

_EXPERT_RE = re.compile(r"mlp\.experts\.(w13|w2)_weight(_scale|_amax|_outer_scale_denominator)?$")


def official_key(name):
    """Checkpoint tensor name -> (parameter name in the official model, note)."""
    key = name
    for pattern, replacement in _COMPOSITE_RENAMES:
        key = re.sub(pattern, replacement, key, count=1)
    if not key.startswith("model.language_model."):
        if key == "lm_head.weight":
            return key, "Inkling2UnpadVocab: only the first unpadded_vocab_size rows are kept"
        return key, ""
    # sources that are merged by a WeightConverter
    if "middle_cycle_cross_pass" in key:
        note = "Inkling2LoopMergeMessageQueries: (query, key, value, mlp) per loop layer"
        return "model.language_model.message_input.query", note
    match = _EXPERT_RE.search(key)
    if match:
        target = "gate_up_proj" if match.group(1) == "w13" else "down_proj"
        note = "Inkling2LoopDequantizeExperts" + (" + Interleave(dim=1)" if match.group(1) == "w13" else "")
        return key[: match.start()] + "mlp.experts." + target, note
    # Inkling2IndexOffsetWeightRename(offset=1): attn_queries.i belongs to layer i + 1
    match = re.search(r"attn_res\.attn_queries\.(\d+)", key)
    if match:
        layer = int(match.group(1)) + 1
        return key[: match.start()] + "layers.%d.attention_input.query" % layer, "index offset +1"
    for pattern, replacement in _TEXT_RENAMES:
        key = re.sub(pattern, replacement, key, count=1)
    return key, ""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_entry(path, name, entry, room):
    """One tensor entry of the header: dtype (a string), shape (sizes >= 0), data_offsets [begin, end] inside the data
    (``room`` bytes after the header). ValueError with the tensor's name otherwise."""
    if not isinstance(entry, dict):
        raise ValueError("%s: the header entry of %s is not an object" % (path, name))
    if not isinstance(entry.get("dtype"), str):
        raise ValueError("%s: %s has no dtype" % (path, name))
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(_is_int(n) and n >= 0 for n in shape):
        raise ValueError("%s: the shape of %s is not a list of sizes: %r" % (path, name, shape))
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_int(n) for n in offsets):
        raise ValueError("%s: the data_offsets of %s are not two integers: %r" % (path, name, offsets))
    begin, end = offsets
    if begin < 0 or end < begin or end > room:
        raise ValueError("%s: the data of %s (bytes %d to %d) is not inside the file (%d bytes of data): truncated or damaged"
                         % (path, name, begin, end, room))


class Checkpoint:
    """Memory-mapped safetensors file.

    File layout: 8 bytes little-endian header length N, N bytes of JSON
    {name: {dtype, shape, data_offsets: [begin, end]}}, then the data; the
    offsets are relative to the first byte after the header.
    """

    def __init__(self, path):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        size = int(self.data.shape[0])
        if size < 8:
            raise ValueError("%s is not a safetensors file (%d bytes)" % (path, size))
        (header_len,) = struct.unpack("<Q", self.data[:8].tobytes())
        if header_len > size - 8:
            raise ValueError("%s: the header length %d is larger than the file (%d bytes): truncated or not a safetensors file"
                             % (path, header_len, size))
        try:
            header = json.loads(self.data[8 : 8 + header_len].tobytes().decode("utf-8"))
        except ValueError as ex:
            raise ValueError("%s: the header is not JSON (%s)" % (path, ex)) from None
        if not isinstance(header, dict):
            raise ValueError("%s: the header is not a JSON object" % path)
        metadata = header.pop("__metadata__", {})
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.base = 8 + header_len
        room = size - self.base
        for name, entry in header.items():
            _check_entry(path, name, entry, room)
        self.entries = header
        self.used = {}  # checkpoint name -> what the reference model uses it for

    def names(self):
        return list(self.entries.keys())

    def has(self, name):
        return name in self.entries

    def info(self, name):
        entry = self.entries[name]
        return entry["dtype"], tuple(entry["shape"])

    def raw(self, name, use=None):
        """Zero-copy view of the stored bytes in the stored layout (bfloat16 as
        uint16 bits, float8 as uint8 bits)."""
        entry = self.entries[name]
        begin, end = entry["data_offsets"]
        if entry["dtype"] not in _NP_DTYPES:
            raise ValueError("%s: dtype %s is not supported" % (name, entry["dtype"]))
        np_dtype = np.dtype(_NP_DTYPES[entry["dtype"]])
        count = 1
        for n in entry["shape"]:
            count *= n
        if (end - begin) != count * np_dtype.itemsize:
            raise ValueError("%s: %d bytes of data for a %s tensor of shape %s (%d bytes)"
                             % (name, end - begin, entry["dtype"], list(entry["shape"]), count * np_dtype.itemsize))
        view = self.data[self.base + begin : self.base + end].view(np_dtype).reshape(entry["shape"])
        if use is not None:
            self.used[name] = use
        return view

    def f32(self, name, use=None):
        """Tensor as a float32 numpy array (a copy). F32 as stored; BF16 widened
        exactly; F8_E4M3 through the table."""
        dtype = self.entries[name]["dtype"]
        view = self.raw(name, use)
        if dtype == "F32":
            return np.array(view, dtype=np.float32)
        if dtype == "BF16":
            return bf16_to_f32(np.asarray(view))
        if dtype == "F8_E4M3":
            return E4M3_TABLE[np.asarray(view)]
        raise ValueError("no float32 view for dtype %s (%s)" % (dtype, name))

    # -- experts -----------------------------------------------------------
    def expert_format(self, prefix):
        """prefix = '<...>.mlp.experts.w13' or '...w2'. NVFP4 tensors come with
        an amax tensor; MXFP8 tensors do not (same test as the official loader)."""
        return "nvfp4" if self.has(prefix + "_weight_amax") else "mxfp8"

    def expert_weight(self, prefix, experts, use=None, official_bf16=False):
        """Decoded float32 weights [len(experts), rows, cols] of the given
        expert indices (a slice or a list). Rows of w13 are still interleaved
        as in the checkpoint; see deinterleave_rows.

        official_bf16=True reproduces the two bfloat16 roundings of the
        official loader in a bfloat16 model (see the module docstring): amax
        and the denominator are rounded before the division, the result after."""
        if isinstance(experts, int):
            experts = [experts]
        weight = self.raw(prefix + "_weight", use)
        scale = self.raw(prefix + "_weight_scale", use)
        if self.expert_format(prefix) == "nvfp4":
            amax = np.asarray(self.raw(prefix + "_weight_amax", use)[experts], dtype=np.float32)
            denominator = np.asarray(self.raw(prefix + "_weight_outer_scale_denominator", use), dtype=np.float32)
            assert self.info(prefix + "_weight")[0] == "U8" and self.info(prefix + "_weight_scale")[0] == "F8_E4M3"
            if official_bf16:
                amax, denominator = round_to_bf16(amax), round_to_bf16(denominator)
            values = decode_nvfp4(
                np.asarray(weight[experts]), np.asarray(scale[experts]), amax, np.float32(denominator.reshape(-1)[0])
            )
        else:
            assert self.info(prefix + "_weight")[0] == "F8_E4M3" and self.info(prefix + "_weight_scale")[0] == "U8"
            values = decode_mxfp8(np.asarray(weight[experts]), np.asarray(scale[experts]))
        return round_to_bf16(values) if official_bf16 else values

    # -- bookkeeping -------------------------------------------------------
    def key_table(self):
        """One row per checkpoint tensor: (name, dtype, shape, official name,
        note, use). `use` is None for tensors the reference model did not read."""
        rows = []
        for name in self.names():
            dtype, shape = self.info(name)
            official, note = official_key(name)
            rows.append((name, dtype, shape, official, note, self.used.get(name)))
        return rows

    def unused(self):
        """Tensors not read by the reference model, with the reason."""
        out = []
        for name in self.names():
            if name in self.used:
                continue
            if name.startswith("model.visual."):
                reason = "vision tower: out of scope (text only)"
            else:
                reason = "UNEXPECTED: not consumed by the reference model"
            out.append((name, reason))
        return out
