#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""inkling2_convert.py - make a YASAKANI model folder from a downloaded Inkling-2 Mini folder (Hugging Face layout).

  python inkling2_convert.py --source <downloaded folder> --out <new model folder> [--repo ID] [--revision SHA] [--copy]

What the model folder holds:

* the published weights file, unchanged: a hard link when the source and the new folder are on the same drive (no extra disk
  space), else a copy.  The vision tensors stay in the file and are not read.  The drafter file (mtp.safetensors) is not used yet
  and is left out.
* config.json, generation_config.json, the tokenizer files and chat_template.jinja, unchanged,
* README.md of the source (the model card, with its licence and the acceptable-use policy), unchanged,
* NOTICE: where the files come from and that they are unchanged,
* selfcheck.json: the prompt of the model's official test and the greedy tokens it expects,
* yasakani.json: the manifest (architecture, source repository and revision, context limit, default sampling, stop token, the
  weights entry and the size and SHA-256 of every file, so that ``yasakani verify`` works).

The weights entry of the manifest names a file and its format ("hf-safetensors" here).  A smaller re-quantised file in another
format only changes that entry; the rest of the folder stays as it is.

Standard library only.  Nothing is downloaded; nothing in the source folder is executed or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ARCHITECTURE = "inkling2"
MANIFEST = "yasakani.json"
WEIGHT_NAMES = ("model-00001-of-00001.safetensors", "model.safetensors")
TEXT_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja")
OPTIONAL_TEXT_FILES = ("README.md",)
DEFAULT_CTX = 32768                  # context the engine opens by default (the model allows more; the cache grows with it)

# The prompt of the official integration test of the model's code in the transformers library (test_modeling_inkling2_loop.py),
# rendered with chat_template.jinja and tokenized with tokenizer.json: 24 ids.
OFFICIAL_MESSAGES = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
OFFICIAL_TEMPLATE_KWARGS = {"reasoning_effort": "none"}
OFFICIAL_PROMPT_IDS = [200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 100887, 149881, 150011,
                       156021, 151661, 149876, 158808, 149884, 152978, 152778, 13, 200010]

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


def notice_text(repo: str, revision: str, weights: str, how: str) -> str:
    return "\n".join([
        "Inkling-2 Mini for YASAKANI",
        "",
        "This folder holds the model files that the YASAKANI engine loads. They were not made by the YASAKANI authors.",
        "",
        "* Source: %s on Hugging Face, revision %s." % (repo, revision),
        "  Inkling-2 Mini is a model of Thinking Machines Lab, published under the Apache License 2.0",
        "  (https://www.apache.org/licenses/LICENSE-2.0). Its model card (README.md in this folder, when the source has one)",
        "  also names a Model Acceptable Use Policy (https://thinkingmachines.ai/model-acceptable-use-policy/).",
        "  Read both before you use the model.",
        "* The weights (%s), the configuration, the tokenizer and the chat template are the published files," % weights,
        "  unchanged (weights: %s of the published file). The vision tensors in the weights file are not used." % how,
        "* selfcheck.json and yasakani.json were written by the YASAKANI converter (tools/convert/inkling2_convert.py).",
        "  The expected tokens in selfcheck.json are those of the official tests of the model's code.",
        "",
    ])


def convert(source: Path, out: Path, repo: str = None, revision: str = None, copy: bool = False, ctx: int = DEFAULT_CTX) -> dict:
    t0 = time.time()
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

    out.mkdir(parents=True, exist_ok=True)
    how = place(source / weights, out / weights, copy)
    say("weights  %-36s %s (%.2f GB)" % (weights, how, (out / weights).stat().st_size / 1e9))
    for n in TEXT_FILES + OPTIONAL_TEXT_FILES:
        if (source / n).is_file():
            shutil.copyfile(source / n, out / n)
    (out / "NOTICE").write_text(notice_text(repo, revision, weights, how), encoding="utf-8", newline="\n")
    known = KNOWN.get(repo)
    cases = []
    if known is not None:
        cases.append(dict(name="official", messages=OFFICIAL_MESSAGES, template_kwargs=OFFICIAL_TEMPLATE_KWARGS,
                          prompt_ids=OFFICIAL_PROMPT_IDS, tokens=known["tokens"], expected_text=known["expected_text"],
                          source="expected output of the official tests of the model's code (transformers, inkling2_loop)"))
    else:
        say("note: no reference output is known for %s; selfcheck.json has no cases" % repo)
    selfcheck = dict(format=1, architecture=ARCHITECTURE, model_id=(known or {}).get("model_id", repo.split("/")[-1].lower()),
                     cases=cases)
    (out / "selfcheck.json").write_text(json.dumps(selfcheck, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    listed = [weights] + [n for n in TEXT_FILES + OPTIONAL_TEXT_FILES if (out / n).is_file()] + ["NOTICE", "selfcheck.json"]
    sizes, sums = {}, {}
    for n in listed:
        sizes[n] = (out / n).stat().st_size
        sums[n] = sha256_of(out / n)
        _, tag = hf_metadata(source, n)
        if n == weights and tag and len(tag) == 64:          # Hugging Face's etag of a large file is its SHA-256
            if tag != sums[n]:
                raise SystemExit("%s: SHA-256 %s differs from the published %s (download it again)" % (n, sums[n], tag))
            say("weights  SHA-256 equals the published one (%s...)" % tag[:16])
    manifest = {
        "architecture": ARCHITECTURE,
        "model_id": selfcheck["model_id"],
        "source": {"repository": repo, "revision": revision, "license": "Apache-2.0",
                   "acceptable_use_policy": "https://thinkingmachines.ai/model-acceptable-use-policy/"},
        "max_ctx": max_ctx,
        "default_ctx": min(int(ctx), max_ctx),
        "weights": {"file": weights, "format": "hf-safetensors", "note": "published file, unchanged; vision tensors unused"},
        "sampling": {"temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 1.0)),
                     "top_k": int(gen.get("top_k", 0) or 0)},
        "stop_token_ids": stop_ids,
        "converter": "tools/convert/inkling2_convert.py",
        "sizes": sizes,
        "sha256": sums,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    say("wrote %s: %d files listed, %.0f s" % (out / MANIFEST, len(sums), time.time() - t0))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="the downloaded folder (config.json, the .safetensors weights, tokenizer files)")
    ap.add_argument("--out", required=True, help="the new model folder")
    ap.add_argument("--repo", help="source repository on Hugging Face (default: recognised from config.json)")
    ap.add_argument("--revision", help="source revision (default: from the download metadata of huggingface_hub)")
    ap.add_argument("--copy", action="store_true", help="copy the weights even when a hard link is possible")
    ap.add_argument("--default-ctx", type=int, default=DEFAULT_CTX, help="context the engine opens by default (%d)" % DEFAULT_CTX)
    a = ap.parse_args(argv)
    convert(Path(a.source), Path(a.out), a.repo, a.revision, a.copy, a.default_ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
