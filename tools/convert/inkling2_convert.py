#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""inkling2_convert.py - make a YASAKANI model folder from a downloaded Inkling-2 Mini folder (Hugging Face layout).

  python inkling2_convert.py --source <downloaded folder> --out <new model folder> [--scheme published | quality]
                             [--repo ID] [--revision SHA] [--copy] [--device cpu | cuda] [--threads N]
                             [--reference-code DIR [--reference-device cpu | cuda]] [--selfcheck-tokens FILE]
  python inkling2_convert.py --selfcheck-only --out <model folder> --reference-code DIR [--reference-device cuda]

Two schemes:

* ``published`` (default): the published weights file, unchanged (a hard link when the source and the new folder are on the same
  drive, else a copy).  The vision tensors stay in the file and are not read.  selfcheck.json holds the prompt of the model's
  official test and the greedy tokens that test expects.
* ``quality``: one ``.yskn`` file (version 2, docs/FORMAT.md section 10) with the text weights re-quantised:
    - NVFP4 (4-bit codes, an E4M3 scale per 16 values, one factor per matrix, scale search): the attention query projection
      (wq_du), the output gate (gate_attn) and the output projection (unembed; its 200,058 used rows),
    - MXFP8 (E4M3 values, an exponent per 32 values): wk_dv, wv_dv, wo_ud, wr_du, attn_sink_proj, the weight_proj of the key and
      value convolutions, every message-gate matrix (controllers: down, forget, gain) and the embedding,
    - the experts: the NVFP4 experts of the published file unchanged; the MXFP8 experts (layers 20 to 23 of Inkling-2 Mini)
      re-packed to NVFP4 with one factor per expert and the scale search,
    - everything else as published (bfloat16 or float32): the router, the norms, the convolutions, the clip scales,
      rel_logits_proj, the attention-residual queries.
  The vision tower and the drafter are not in the file.  Measured on 43 windows (14,024 positions) against the
  published model: KL 0.0110, top-1 96.0 %; the file has 15.05 GB (the published file 19.5 GB, with the vision tower).
  The expected tokens of selfcheck.json are made FROM THE CONVERTED WEIGHTS, by a reference implementation of the model (PyTorch;
  not part of this repository): ``--reference-code`` names the folder of its ``ink2_dev.py``, or ``--selfcheck-tokens`` a JSON
  file {case name: [token ids]} made by such a run.  Without either, selfcheck.json has no cases.

With both schemes the model's drafter (mtp.safetensors, a few layers that propose tokens for speculative decoding) is placed in
the model folder unchanged (a hard link where possible, else a copy) and named by the ``drafter`` entry of the manifest; a source
without it gives a folder that generates without drafting.

The model folder also holds config.json, generation_config.json, the tokenizer files, chat_template.jinja and the model card
(README.md) of the source, unchanged; NOTICE (where the files come from, the licence, the acceptable-use policy, and what this
converter changed); yasakani.json, the manifest (architecture, source, context limit, sampling, stop token, the weights entry, and
the size and SHA-256 of every file, so that ``yasakani verify`` works).

Nothing is downloaded; nothing in the source folder is executed or changed.  numpy for the quality scheme (PyTorch only with
--device cuda or --reference-code); tokenizers and jinja2 for the prompts of the self-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHITECTURE = "inkling2"
MANIFEST = "yasakani.json"
WEIGHT_NAMES = ("model-00001-of-00001.safetensors", "model.safetensors")
TEXT_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja")
OPTIONAL_TEXT_FILES = ("README.md",)
DEFAULT_CTX = 32768                  # context the engine opens by default (the model allows more; the cache grows with it)
DRAFTER_FILE = "mtp.safetensors"     # the model's drafter (multi-token prediction layers), published next to the weights
POLICY = "https://thinkingmachines.ai/model-acceptable-use-policy/"
SCHEMES = ("published", "quality")
NVFP4_SEARCH, MXFP8_SEARCH = 6, 1

# The prompt of the official integration test of the model's code in the transformers library (test_modeling_inkling2_loop.py),
# rendered with chat_template.jinja and tokenized with tokenizer.json: 24 ids.
OFFICIAL_MESSAGES = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
OFFICIAL_TEMPLATE_KWARGS = {"reasoning_effort": "none"}
OFFICIAL_PROMPT_IDS = [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 100887, 149881, 150011,
                       156021, 151661, 149876, 158808, 149884, 152978, 152778, 13, 200010]

# The prompts of the self-check of a re-quantised folder (greedy, SELFCHECK_TOKENS tokens each).
SELFCHECK_CASES = [
    dict(name="official", messages=OFFICIAL_MESSAGES, template_kwargs=OFFICIAL_TEMPLATE_KWARGS),
    dict(name="japanese", messages=[{"role": "user", "content": "日本で一番高い山は何ですか。理由も一文で教えてください。"}],
         template_kwargs={"reasoning_effort": "none"}),
    dict(name="code", messages=[{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
         template_kwargs={"reasoning_effort": "none"}),
]
SELFCHECK_TOKENS = 48

# Greedy tokens the official tests expect after that prompt (their expected text, tokenized with tokenizer.json).
KNOWN = {
    "thinkingmachines/Inkling-2-Mini": dict(
        model_id="inkling-2-mini", hidden_size=2304,
        tokens=[200001, 200008, 371, 150409, 171960, 149887, 150011, 156021, 161043, 151661, 158808, 149884, 152978, 152778, 13,
                170289, 150893, 149911, 4123, 13, 170289, 151791, 153606, 25, 220, 47157, 180553, 25, 177113, 152778],
        expected_text=("<|message_model|><|content_thinking|>The user asks for one sentence explanation why sky is usually blue. "
                       "Need answer concise. Need ensure accuracy: Rayleigh scattering: shorter blue")),
    "hf-internal-testing/tiny-inkling2-mini": dict(
        model_id="tiny-inkling2-mini", hidden_size=256,
        tokens=[34066, 91318, 123193] + [169626] * 17,
        expected_text="floursigne턽" + " перек" * 17),
}


def say(*a) -> None:
    print(*a, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def hf_metadata(source: Path, name: str):
    """(revision, etag) from the download metadata that huggingface_hub writes next to the files, or (None, None)."""
    p = source / ".cache" / "huggingface" / "download" / (name + ".metadata")
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    rev = lines[0].strip() if lines else None
    etag = lines[1].strip() if len(lines) > 1 else None
    return rev or None, etag or None


def guess_repo(config: dict):
    hidden = (config.get("text_config") or {}).get("hidden_size")
    hits = [k for k, v in KNOWN.items() if v["hidden_size"] == hidden]
    return hits[0] if len(hits) == 1 else None


def place(src: Path, dst: Path, copy: bool) -> str:
    """Hard link (same drive) or copy; an existing file that is already the same file is kept."""
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "hard link"
        except OSError:
            pass
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return "hard link"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "copy"


def package_modules():
    """yasakani.ink2_yskn (the container) from the python/ folder of this repository."""
    sys.path.insert(0, str(HERE.parent.parent / "python"))
    from yasakani import ink2_yskn
    return ink2_yskn


# ---- the published file --------------------------------------------------------------------------------------------------------------
class SafeTensors:
    """The tensors of a safetensors file as numpy views of a memory map (8 bytes header length, JSON header, data)."""

    DTYPES = {"F32": "<f4", "BF16": "<u2", "U8": "u1", "F8_E4M3": "u1", "F16": "<f2", "I32": "<i4", "I64": "<i8"}

    def __init__(self, path: Path):
        import numpy as np
        self.np = np
        self.mm = np.memmap(str(path), dtype=np.uint8, mode="r")
        (n,) = struct.unpack("<Q", self.mm[:8].tobytes())
        head = json.loads(self.mm[8:8 + n].tobytes().decode("utf-8"))
        head.pop("__metadata__", None)
        self.entries = head
        self.base = 8 + n

    def names(self):
        return sorted(self.entries, key=lambda k: self.entries[k]["data_offsets"][0])

    def dtype(self, name):
        return self.entries[name]["dtype"]

    def shape(self, name):
        return tuple(self.entries[name]["shape"])

    def view(self, name):
        e = self.entries[name]
        a, b = e["data_offsets"]
        return self.mm[self.base + a:self.base + b].view(self.np.dtype(self.DTYPES[e["dtype"]])).reshape(e["shape"])

    def f32(self, name):
        np = self.np
        v = self.view(name)
        if self.dtype(name) == "BF16":
            return (np.asarray(v).astype(np.uint32) << 16).view(np.float32)
        return np.array(v, np.float32)


# ---- the quality scheme ---------------------------------------------------------------------------------------------------------------
NVFP4_DENSE = (".attn.wq_du.weight", ".attn.gate_attn.weight")
MXFP8_DENSE = (".attn.wk_dv.weight", ".attn.wv_dv.weight", ".attn.wo_ud.weight", ".attn.wr_du.weight", ".attn.attn_sink_proj.weight",
               ".attn.k_sconv.weight_proj", ".attn.v_sconv.weight_proj")
EXPERT_PARTS = {"_weight": "weight", "_weight_scale": "scale", "_weight_amax": "amax", "_weight_outer_scale_denominator": "denominator"}
SCHEME_RULES = [
    "NVFP4 (one factor per matrix, scale search over 8 E4M3 codes): attn.wq_du, attn.gate_attn, unembed (the used rows)",
    "MXFP8 (exponent search over 2 values): attn.wk_dv, attn.wv_dv, attn.wo_ud, attn.wr_du, attn.attn_sink_proj, "
    "attn.k_sconv.weight_proj, attn.v_sconv.weight_proj, controllers.*, embed",
    "experts: NVFP4 tensors of the published file unchanged; MXFP8 tensors re-packed to NVFP4 (one factor per expert, scale search)",
    "as published (bfloat16 / float32): everything else (router, norms, convolutions, clip scales, rel_logits_proj, queries)",
]


def dense_encoding(name: str) -> str:
    if name == "model.llm.unembed.weight" or name.endswith(NVFP4_DENSE):
        return "nvfp4"
    if name == "model.llm.embed.weight" or name.endswith(MXFP8_DENSE) or ".controllers." in name:
        return "mxfp8"
    return ""


def plan_quality(st: SafeTensors, text_config: dict):
    """-> list of tensor descriptions (name, encoding, shape, role, source) in the order of the published file."""
    unpadded = int(text_config.get("unpadded_vocab_size") or text_config.get("vocab_size"))
    plan, seen = [], set()
    for n in st.names():
        if not n.startswith("model.llm."):
            continue                                      # the vision tower (model.visual.*)
        ex = next((s for s in EXPERT_PARTS if n.endswith(".mlp.experts.w13" + s) or n.endswith(".mlp.experts.w2" + s)), None)
        if ex is not None:
            prefix = n[:-len(ex)]
            if prefix in seen:
                continue
            seen.add(prefix)
            fmt = "nvfp4" if (prefix + "_weight_amax") in st.entries else "mxfp8"
            w = st.shape(prefix + "_weight")
            shape = [w[0], w[1], w[2] * (2 if fmt == "nvfp4" else 1)]
            plan.append(dict(name=prefix, encoding="nvfp4", shape=shape, role="experts",
                             source="published NVFP4, unchanged" if fmt == "nvfp4" else "published MXFP8, re-quantised to NVFP4",
                             row_order="as published (w13: gate and up rows interleaved)"))
            continue
        dt, shape = st.dtype(n), list(st.shape(n))
        enc = dense_encoding(n)
        if enc:
            if dt != "BF16" or len(shape) != 2:
                raise SystemExit("%s: expected a bfloat16 matrix, found %s %s" % (n, dt, shape))
            d = dict(name=n, encoding=enc, shape=shape, role="dense", source="published BF16, quantised")
            if n == "model.llm.unembed.weight":
                d["shape"] = [unpadded, shape[1]]
                d["source"] = "published BF16, the first %d rows (the unpadded vocabulary), quantised" % unpadded
            plan.append(d)
        elif dt in ("BF16", "F32"):
            plan.append(dict(name=n, encoding=dt.lower(), shape=shape, role="vector" if len(shape) < 2 else "dense",
                             source="published, unchanged"))
        else:
            raise SystemExit("%s: unexpected dtype %s" % (n, dt))
    return plan


def write_quality(st: SafeTensors, out_file: Path, model: dict, text_config: dict, device: str, threads: int) -> dict:
    import numpy as np
    Y = package_modules()
    sys.path.insert(0, str(HERE))
    import inkling2_quant as NQ
    if device == "cpu":
        nvq = lambda w: NQ.nvfp4_quant(w, NVFP4_SEARCH, threads)               # noqa: E731
        mxq = lambda w: NQ.mxfp8_quant(w, MXFP8_SEARCH, threads)               # noqa: E731
        nvq1 = lambda w: NQ.nvfp4_quant(w, NVFP4_SEARCH, 1)                    # noqa: E731
    else:
        tnv, tmx = NQ.torch_quantisers(device)
        nvq = nvq1 = lambda w: tnv(w, NVFP4_SEARCH)                            # noqa: E731
        mxq = lambda w: tmx(w, MXFP8_SEARCH)                                    # noqa: E731
    plan = plan_quality(st, text_config)
    w = Y.Writer(out_file, model=model, tensors=plan)
    t0 = time.time()
    stats = dict(tensors=len(plan), quantised=0, repacked_experts=0, copied_bytes=0)
    for k, t in enumerate(plan):
        n, enc = t["name"], t["encoding"]
        if t["role"] == "experts":
            if (n + "_weight_amax") in st.entries:          # NVFP4 as published: the bytes unchanged
                for suf, part in EXPERT_PARTS.items():
                    v = st.view(n + suf)
                    with w.stream(n, part) as s:
                        for e0 in range(0, v.shape[0] if v.ndim > 1 else 1, 8):
                            s.write(v[e0:e0 + 8] if v.ndim > 1 else v)
                    stats["copied_bytes"] += v.nbytes
            else:                                            # MXFP8 -> NVFP4, one factor per expert
                vw, vs = st.view(n + "_weight"), st.view(n + "_weight_scale")
                E = vw.shape[0]
                sw, ss = w.stream(n, "weight").__enter__(), w.stream(n, "scale").__enter__()
                amax = np.empty(E, np.float32)

                def one(e):
                    dec = Y.decode_mxfp8(np.asarray(vw[e]), np.asarray(vs[e]))
                    return nvq1(dec)

                pool = ThreadPoolExecutor(threads if device == "cpu" else 1)
                try:
                    for e, q in enumerate(pool.map(one, range(E))):
                        sw.write(q["weight"])
                        ss.write(q["scale"])
                        amax[e] = q["amax"][0]
                finally:
                    pool.shutdown()
                sw.finish()
                ss.finish()
                w.put(n, "amax", amax)
                w.put(n, "denominator", np.array([Y.NVFP4_DEN], np.float32))
                stats["repacked_experts"] += E
        elif enc in ("nvfp4", "mxfp8"):
            x = st.f32(n)
            if x.shape[0] != t["shape"][0]:
                x = np.ascontiguousarray(x[:t["shape"][0]])
            q = nvq(x) if enc == "nvfp4" else mxq(x)
            del x
            for part, arr in q.items():
                w.put(n, part, arr)
            stats["quantised"] += 1
        else:
            w.put(n, "data", st.view(n))
        if k % 50 == 0 or t["role"] == "experts" or enc in ("nvfp4", "mxfp8") and t["shape"][0] > 100000:
            say("  %4d/%d  %-56s %-6s %6.0f s" % (k + 1, len(plan), n[len("model.llm."):], enc, time.time() - t0))
    header = w.close()
    stats["seconds"] = round(time.time() - t0, 1)
    return dict(stats=stats, header=header)


# ---- texts of the folder ---------------------------------------------------------------------------------------------------------
def notice_text(repo: str, revision: str, weights: str, how: str, scheme: str = "published", source_weights: str = "") -> str:
    head = [
        "Inkling-2 Mini for YASAKANI",
        "",
        "This folder holds the model files that the YASAKANI engine loads. The model was not made by the YASAKANI authors.",
        "",
        "* Source: %s on Hugging Face, revision %s." % (repo, revision),
        "  Inkling-2 Mini is a model of Thinking Machines Lab, published under the Apache License 2.0",
        "  (https://www.apache.org/licenses/LICENSE-2.0). Its model card (README.md in this folder, when the source has one)",
        "  also names a Model Acceptable Use Policy (%s)." % POLICY,
        "  Read both before you use the model.",
    ]
    if scheme == "published":
        body = [
            "* The weights (%s), the configuration, the tokenizer and the chat template are the published files," % weights,
            "  unchanged (weights: %s of the published file). The vision tensors in the weights file are not used." % how,
            "  The drafter (%s), when this folder has it, is the published file, unchanged." % DRAFTER_FILE,
            "* selfcheck.json and yasakani.json were written by the YASAKANI converter (tools/convert/inkling2_convert.py).",
            "  The expected tokens in selfcheck.json are those of the official tests of the model's code.",
        ]
    else:
        body = [
            "* MODIFIED WEIGHTS: %s was made from the published %s by the YASAKANI converter" % (weights, source_weights),
            "  (tools/convert/inkling2_convert.py, scheme \"%s\"). The weights were re-quantised by this tool: most dense" % scheme,
            "  matrices are stored in 4-bit (NVFP4) or 8-bit (MXFP8) formats, and the experts that the published file stores in 8 bits",
            "  were re-packed to 4 bits. The other tensors are the published values. The outputs of this model differ slightly",
            "  from those of the published model. The vision tower is not included. The drafter (%s), when this folder" % DRAFTER_FILE,
            "  has it, is the published file, unchanged.",
            "  The format is described in docs/FORMAT.md (section 10) of the YASAKANI repository.",
            "* The configuration, the tokenizer, the chat template and the model card are the published files, unchanged.",
            "* selfcheck.json and yasakani.json were written by the converter. The expected tokens in selfcheck.json were made",
            "  from these converted weights by a reference implementation of the model.",
            "* The acceptable-use policy above applies to the use of these converted weights as well.",
        ]
    return "\n".join(head + body + [""])


def selfcheck_prompts(folder: Path):
    """The cases of SELFCHECK_CASES with their prompt ids (chat template and tokenizer of the folder)."""
    sys.path.insert(0, str(HERE.parent.parent / "python"))
    from yasakani.inkling2 import Ink2Formatter
    fmt = Ink2Formatter(folder)
    out = []
    for c in SELFCHECK_CASES:
        ids = fmt.encode(fmt.render(c["messages"], **c["template_kwargs"]))
        if c["name"] == "official" and ids != OFFICIAL_PROMPT_IDS:
            raise SystemExit("the official prompt does not give the official ids (tokenizer or chat template differ)")
        out.append(dict(c, prompt_ids=ids))
    return out, fmt


def reference_tokens(folder: Path, weights: str, cases, code_dir: str, device: str, n_tokens: int):
    """Greedy tokens of every case with the reference implementation (Ink2DevModel of ``code_dir``/ink2_dev.py) on the converted
    weights, decoded as the engine is given them (dense matrices rounded to bfloat16)."""
    t0 = time.time()
    sys.path.insert(0, str(Path(code_dir).resolve()))
    import torch
    import ink2_dev
    Y = package_modules()
    yf = Y.YsknFile(folder / weights)
    view = Y.CheckpointView(yf, dense="bf16")
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))["text_config"]
    model = ink2_dev.Ink2DevModel(view, cfg, device=device)
    say("reference model on %s ready in %.0f s" % (device, time.time() - t0))
    res = {}
    with torch.no_grad():
        for c in cases:
            t1 = time.time()
            state = model.new_state()
            logits = model.forward(c["prompt_ids"], state)[-1]
            toks = []
            for _ in range(n_tokens):
                toks.append(int(torch.argmax(logits)))
                if len(toks) < n_tokens:
                    logits = model.forward([toks[-1]], state)[-1]
            res[c["name"]] = toks
            say("  reference %-9s %d tokens in %.1f s" % (c["name"], len(toks), time.time() - t1))
    del model
    yf.close()
    impl = "Ink2DevModel (PyTorch %s, float32, %s), dense matrices decoded and rounded to bfloat16, experts decoded from the " \
           "stored codes" % (torch.__version__, device)
    return res, impl


def read_tokens_file(path: str):
    """{case: [ids]} or {"implementation": text, "tokens": {case: [ids]}} -> (tokens, implementation)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d.get("tokens"), dict):
        return d["tokens"], str(d.get("implementation") or "a reference run (%s)" % Path(path).name)
    return d, "a reference run (%s)" % Path(path).name


def write_selfcheck(folder: Path, model_id: str, cases, tokens: dict, impl: str, fmt) -> dict:
    out = []
    for c in cases:
        t = tokens.get(c["name"])
        if not t:
            continue
        out.append(dict(name=c["name"], messages=c["messages"], template_kwargs=c["template_kwargs"], prompt_ids=c["prompt_ids"],
                        tokens=[int(x) for x in t], expected_text=fmt.decode(t),
                        source="greedy output of the converted weights: %s" % impl))
    sc = dict(format=1, architecture=ARCHITECTURE, model_id=model_id, cases=out)
    if not out:
        sc["note"] = "no expected tokens were made (give --reference-code or --selfcheck-tokens to the converter)"
    (folder / "selfcheck.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sc


def write_manifest(out: Path, manifest: dict, listed, known_sums: dict = None) -> dict:
    sizes, sums = {}, {}
    for n in listed:
        sizes[n] = (out / n).stat().st_size
        sums[n] = (known_sums or {}).get(n) or sha256_of(out / n)
    manifest["sizes"], manifest["sha256"] = sizes, sums
    w = manifest.get("weights") or {}
    if w.get("format") == "yskn":
        w["sha256"], w["size"] = sums[w["file"]], sizes[w["file"]]
    d = manifest.get("drafter")
    if d and d.get("file") in sums:
        d["sha256"], d["size"] = sums[d["file"]], sizes[d["file"]]
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


# ---- the conversion ------------------------------------------------------------------------------------------------------------------
def convert(source: Path, out: Path, repo: str = None, revision: str = None, copy: bool = False, ctx: int = DEFAULT_CTX,
            scheme: str = "published", device: str = "cpu", threads: int = 0, reference_code: str = None,
            reference_device: str = "cpu", selfcheck_tokens: str = None) -> dict:
    t0 = time.time()
    if scheme not in SCHEMES:
        raise SystemExit("unknown scheme %r (%s)" % (scheme, ", ".join(SCHEMES)))
    if not source.is_dir():
        raise SystemExit("source folder not found: %s" % source)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "inkling2_loop":
        raise SystemExit("not an Inkling-2 folder: config.json has model_type %r" % config.get("model_type"))
    weights = next((n for n in WEIGHT_NAMES if (source / n).is_file()), None)
    if weights is None:
        raise SystemExit("no weights file (%s) in %s" % (" or ".join(WEIGHT_NAMES), source))
    for n in TEXT_FILES:
        if not (source / n).is_file():
            raise SystemExit("the source folder is missing %s" % n)
    repo = repo or guess_repo(config)
    if repo is None:
        raise SystemExit("give the source repository with --repo")
    rev_meta, etag = hf_metadata(source, weights)
    revision = revision or rev_meta
    if not revision:
        raise SystemExit("give the source revision with --revision (no download metadata found)")
    if rev_meta and rev_meta != revision:
        raise SystemExit("the download metadata names revision %s, not %s" % (rev_meta, revision))
    gen = json.loads((source / "generation_config.json").read_text(encoding="utf-8"))
    eos = gen.get("eos_token_id", config.get("eos_token_id"))
    stop_ids = [int(e) for e in (eos if isinstance(eos, list) else [eos])] if eos is not None else []
    if not stop_ids:
        raise SystemExit("no eos_token_id in generation_config.json or config.json")
    tok_cfg = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    max_ctx = int(tok_cfg.get("model_max_length") or 1048576)
    known = KNOWN.get(repo)
    model_id = (known or {}).get("model_id", repo.split("/")[-1].lower())
    if scheme != "published":
        model_id += "-" + scheme

    out.mkdir(parents=True, exist_ok=True)
    for n in TEXT_FILES + OPTIONAL_TEXT_FILES:
        if (source / n).is_file():
            shutil.copyfile(source / n, out / n)
    sums = {}
    if scheme == "published":
        how = place(source / weights, out / weights, copy)
        say("weights  %-36s %s (%.2f GB)" % (weights, how, (out / weights).stat().st_size / 1e9))
        out_weights = weights
        weights_entry = {"file": weights, "format": "hf-safetensors", "note": "published file, unchanged; vision tensors unused"}
        (out / "NOTICE").write_text(notice_text(repo, revision, weights, how), encoding="utf-8", newline="\n")
        cases = []
        if known is not None:
            cases.append(dict(name="official", messages=OFFICIAL_MESSAGES, template_kwargs=OFFICIAL_TEMPLATE_KWARGS,
                              prompt_ids=OFFICIAL_PROMPT_IDS, tokens=known["tokens"], expected_text=known["expected_text"],
                              source="expected output of the official tests of the model's code (transformers, inkling2_loop)"))
        else:
            say("note: no reference output is known for %s; selfcheck.json has no cases" % repo)
        selfcheck = dict(format=1, architecture=ARCHITECTURE, model_id=model_id, cases=cases)
        (out / "selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        src_sha = None
    else:
        src_sha = etag if etag and len(etag) == 64 else None
        if src_sha is not None:
            got = sha256_of(source / weights)
            if got != src_sha:
                raise SystemExit("%s: SHA-256 %s differs from the published %s (download it again)" % (weights, got, src_sha))
            say("source   %s: SHA-256 equals the published one (%s...)" % (weights, src_sha[:16]))
        out_weights = "%s.yskn" % model_id.replace("-", "_")
        if out_weights in (weights,):
            raise SystemExit("output file name clashes with the source")
        st = SafeTensors(source / weights)
        model = {"arch": ARCHITECTURE, "model_id": model_id,
                 "source": {"repository": repo, "revision": revision, "file": weights, "sha256": src_sha, "license": "Apache-2.0",
                            "acceptable_use_policy": POLICY},
                 "scheme": {"name": scheme, "rules": SCHEME_RULES, "nvfp4_search": NVFP4_SEARCH, "mxfp8_search": MXFP8_SEARCH,
                            "quantiser": "tools/convert/inkling2_quant.py (%s)" % ("numpy" if device == "cpu" else "PyTorch, " + device)},
                 "text_config": config.get("text_config") or {},
                 "converter": "tools/convert/inkling2_convert.py"}
        say("writing  %s (scheme %s, quantiser on %s) ..." % (out_weights, scheme, device))
        r = write_quality(st, out / out_weights, model, config.get("text_config") or {}, device,
                          threads or max(1, min(32, os.cpu_count() or 1)))
        say("wrote    %s: %.2f GB, %d tensors, %d quantised, %d experts re-packed, %.0f s" % (
            out_weights, (out / out_weights).stat().st_size / 1e9, r["stats"]["tensors"], r["stats"]["quantised"],
            r["stats"]["repacked_experts"], r["stats"]["seconds"]))
        weights_entry = {"file": out_weights, "format": "yskn", "version": 2, "scheme": scheme,
                         "note": "re-quantised from the published weights; vision tensors not included"}
        (out / "NOTICE").write_text(notice_text(repo, revision, out_weights, "", scheme, weights), encoding="utf-8", newline="\n")
        cases, fmt = selfcheck_prompts(out)
        tokens, impl = {}, ""
        if selfcheck_tokens:
            tokens, impl = read_tokens_file(selfcheck_tokens)
        elif reference_code:
            tokens, impl = reference_tokens(out, out_weights, cases, reference_code, reference_device, SELFCHECK_TOKENS)
        write_selfcheck(out, model_id, cases, tokens, impl, fmt)

    drafter_entry = None
    if (source / DRAFTER_FILE).is_file():
        how_d = place(source / DRAFTER_FILE, out / DRAFTER_FILE, copy)
        say("drafter  %-36s %s (%.2f GB)" % (DRAFTER_FILE, how_d, (out / DRAFTER_FILE).stat().st_size / 1e9))
        drafter_entry = {"file": DRAFTER_FILE, "format": "hf-safetensors", "note": "published file, unchanged"}
    else:
        say("note: the source has no %s (the drafter); the folder generates without drafting" % DRAFTER_FILE)
    listed = [out_weights] + ([DRAFTER_FILE] if drafter_entry else []) + \
        [n for n in TEXT_FILES + OPTIONAL_TEXT_FILES if (out / n).is_file()] + ["NOTICE", "selfcheck.json"]
    for n in listed:
        _, tag = hf_metadata(source, n)
        published = (scheme == "published" and n == weights) or n == DRAFTER_FILE
        if published and tag and len(tag) == 64:                         # Hugging Face's etag of a large file is its SHA-256
            sums[n] = sha256_of(out / n)
            if tag != sums[n]:
                raise SystemExit("%s: SHA-256 %s differs from the published %s (download it again)" % (n, sums[n], tag))
            say("%-8s SHA-256 equals the published one (%s...)" % ("drafter" if n == DRAFTER_FILE else "weights", tag[:16]))
    manifest = {
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "source": {"repository": repo, "revision": revision, "license": "Apache-2.0", "acceptable_use_policy": POLICY},
        "max_ctx": max_ctx,
        "default_ctx": min(int(ctx), max_ctx),
        "weights": weights_entry,
        **({"drafter": drafter_entry} if drafter_entry else {}),
        "sampling": {"temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 1.0)),
                     "top_k": int(gen.get("top_k", 0) or 0)},
        "stop_token_ids": stop_ids,
        "converter": "tools/convert/inkling2_convert.py",
    }
    write_manifest(out, manifest, listed, sums)
    say("wrote %s: %d files listed, %.0f s" % (out / MANIFEST, len(listed), time.time() - t0))
    return manifest


def refresh_selfcheck(out: Path, reference_code: str = None, reference_device: str = "cpu", selfcheck_tokens: str = None) -> dict:
    """Make selfcheck.json (and NOTICE) of an existing re-quantised folder again (no quantisation) and update the manifest."""
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    w = manifest.get("weights") or {}
    if w.get("format") != "yskn":
        raise SystemExit("--selfcheck-only is for re-quantised folders (weights format yskn)")
    cases, fmt = selfcheck_prompts(out)
    if selfcheck_tokens:
        tokens, impl = read_tokens_file(selfcheck_tokens)
    elif reference_code:
        tokens, impl = reference_tokens(out, w["file"], cases, reference_code, reference_device, SELFCHECK_TOKENS)
    else:
        raise SystemExit("give --reference-code or --selfcheck-tokens")
    write_selfcheck(out, manifest["model_id"], cases, tokens, impl, fmt)
    Y = package_modules()
    with Y.YsknFile(out / w["file"]) as yf:
        src = yf.model.get("source") or {}
    (out / "NOTICE").write_text(notice_text(src.get("repository"), src.get("revision"), w["file"], "", w.get("scheme", "quality"),
                                            src.get("file", "")), encoding="utf-8", newline="\n")
    known = {n: s for n, s in manifest.get("sha256", {}).items() if n not in ("selfcheck.json", "NOTICE")}
    listed = list(manifest.get("sha256", {}))
    write_manifest(out, manifest, listed, known)
    say("selfcheck.json: %d cases; %s updated" % (len(tokens), MANIFEST))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", help="the downloaded folder (config.json, the .safetensors weights, tokenizer files)")
    ap.add_argument("--out", required=True, help="the new model folder")
    ap.add_argument("--scheme", default="published", choices=SCHEMES, help="published (unchanged weights) or quality (a .yskn file)")
    ap.add_argument("--repo", help="source repository on Hugging Face (default: recognised from config.json)")
    ap.add_argument("--revision", help="source revision (default: from the download metadata of huggingface_hub)")
    ap.add_argument("--copy", action="store_true", help="copy the weights even when a hard link is possible (scheme published)")
    ap.add_argument("--default-ctx", type=int, default=DEFAULT_CTX, help="context the engine opens by default (%d)" % DEFAULT_CTX)
    ap.add_argument("--device", default="cpu", help="where the quantiser runs: cpu (numpy, default) or a PyTorch device such as cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads of the quantiser (default: the CPU count, at most 32)")
    ap.add_argument("--reference-code", help="folder of the reference implementation (ink2_dev.py) for the expected tokens")
    ap.add_argument("--reference-device", default="cpu", help="PyTorch device of the reference run (cpu or cuda)")
    ap.add_argument("--selfcheck-tokens", help="JSON file {case: [token ids]} with the expected tokens of a reference run")
    ap.add_argument("--selfcheck-only", action="store_true", help="only make selfcheck.json of an existing re-quantised folder again")
    a = ap.parse_args(argv)
    if a.selfcheck_only:
        refresh_selfcheck(Path(a.out), a.reference_code, a.reference_device, a.selfcheck_tokens)
        return 0
    if not a.source:
        ap.error("--source is required")
    convert(Path(a.source), Path(a.out), a.repo, a.revision, a.copy, a.default_ctx, a.scheme, a.device, a.threads,
            a.reference_code, a.reference_device, a.selfcheck_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
