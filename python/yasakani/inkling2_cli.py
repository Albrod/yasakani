# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The commands info, verify, selfcheck, chat, serve and bench for an Inkling-2 model folder (yasakani/inkling2.py).  yasakani/cli.py
sends a command here when the folder's yasakani.json says ``"architecture": "inkling2"``; the Qwen3.8 path is not changed."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import List

from . import __version__
from . import inkling2 as ink
from ._native import YasakaniError

UNSUPPORTED = {
    "parallel": "--parallel (several requests at once) is not supported for Inkling-2 models yet",
    "concurrency": "--concurrency is not supported for Inkling-2 models yet",
    "edit": "--edit (the edit tasks of the speed tables) is not supported for Inkling-2 models yet",
    "mmproj": "image input is not supported for Inkling-2 models yet",
    "write": "selfcheck --write is not supported for Inkling-2 models yet",
}


def _say(*a) -> None:
    from .cli import _say as say
    say(*a)


def _gb(n: float) -> str:
    return "%.2f GB" % (n / 1e9)


def refuse_unsupported(args) -> None:
    for key, why in UNSUPPORTED.items():
        v = getattr(args, key, None)
        if (v not in (None, 1)) if key in ("parallel", "concurrency") else bool(v):
            raise YasakaniError(why)


def context_of(args, pack: ink.Ink2Pack) -> int:
    n = args.max_ctx if getattr(args, "max_ctx", None) else pack.default_ctx
    if n < 256 or n > pack.max_ctx:
        raise YasakaniError("--max-ctx must be between 256 and %d (the model's limit)" % pack.max_ctx)
    return int(n)


def _select_gpu(args) -> None:
    """--device N on a PC with several GPUs: the engine sees only that GPU (as for Qwen models)."""
    if getattr(args, "device", 0) in (0, None):
        return
    from . import gpu as gpus
    from .cli import _visible_gpu
    _visible_gpu(gpus.detect(args.device), args.device)


def draft_options(args):
    """(mode, depth) of --draft / --draft-depth, else $YASAKANI_DRAFT / $YASAKANI_DRAFT_DEPTH; a wrong value fails here."""
    return ink.draft_mode_of(getattr(args, "draft", None)), ink.draft_depth_of(getattr(args, "draft_depth", None))


def open_stack(args, warm: bool = True):
    refuse_unsupported(args)
    pack = ink.Ink2Pack.load(args.model_dir_path)
    ctx = context_of(args, pack)
    draft, depth = draft_options(args)
    _select_gpu(args)
    kv = ink.bytes_per_position(ink.read_config(pack.root)) * ctx
    _say("loading %s (%s of weights, context %d tokens, key/value cache %s) ..." % (pack.model_id, _gb(pack.weights.stat().st_size), ctx, _gb(kv)))
    t0 = time.time()
    try:
        fmt, eng, runner = ink.open_runner(pack, ctx, dll=getattr(args, "dll", None), seed0=getattr(args, "seed0", 1), warm=warm,
                                           draft=draft, draft_depth=depth)
    except YasakaniError as ex:
        raise YasakaniError("%s  (a smaller --max-ctx needs less GPU memory)" % ex) if "memory" in str(ex).lower() else ex
    dm = eng.device_memory()
    mem = "engine allocations %s" % _gb(eng.memory_used())
    if dm:
        mem += "; device %s used of %s" % (_gb(dm[1] - dm[0]), _gb(dm[1]))
    _say("model ready in %.1f s (engine load %.1f s, %s backend; %s)" % (time.time() - t0, eng.load_seconds, eng.backend, mem))
    if pack.weights_format == "yskn":
        _say("weights: %s" % eng.weights_note)
    _say(eng.draft_note)
    return pack, eng, fmt, runner


# ---- info / verify ---------------------------------------------------------------------------------------------------------------
def cmd_info(args) -> int:
    from .cli import _gpu_lines, _human
    ok = True
    print("yasakani", __version__)
    print("python  ", platform.python_version(), platform.system(), platform.machine())
    for mod in ("numpy", "tokenizers", "jinja2"):
        try:
            m = importlib.import_module(mod)
            print("%-8s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-8s MISSING  (pip install %s)" % (mod, mod))
            ok = False
    try:
        dll = ink.find_dll(getattr(args, "dll", None))
        binding = ink.find_binding(dll)
        mod = ink.load_binding(binding)
        lib = ink.open_library(mod, dll)
        lib_info = "%s, interface version %d" % (lib.ink2_build_info().decode("utf-8", "replace"), lib.abi)
        print("engine   %s  %s  sha256 %s  (%s; uses the %s backend)" % (dll.name, _human(dll.stat().st_size),
              hashlib.sha256(dll.read_bytes()).hexdigest()[:16], lib_info, ink.backend_of(lib_info)))
        print("binding  %s  (drafter calls: %s)" % (binding.name, "yes" if all(callable(getattr(mod.Ink2Native, n, None))
                                                                             for n in ink.DRAFT_CALLS) else "no, drafting not available"))
    except (YasakaniError, OSError, AttributeError) as ex:
        print("engine   NOT FOUND:", ex)
        ok = False
    for ln in _gpu_lines():
        print("gpu      " + ln)
    try:
        pack = ink.Ink2Pack.load(args.model_dir_path)
        src = pack.meta.get("source") or {}
        print("model    %s  (id %s, architecture inkling2, context up to %d tokens, %d by default)"
              % (pack.root.name, pack.model_id, pack.max_ctx, pack.default_ctx))
        print("  source     %s @ %s, licence %s" % (src.get("repository"), str(src.get("revision"))[:12], src.get("license")))
        wmeta = pack.meta.get("weights") or {}
        print("  weights    %-36s %s  (%s%s)" % (pack.weights.name, _human(pack.weights.stat().st_size), pack.weights_format,
                                                ", version %s, scheme %s" % (wmeta.get("version", 2), wmeta.get("scheme"))
                                                if pack.weights_format == "yskn" else ""))
        d = pack.drafter_entry
        if not d:
            print("  drafter    none (generation without drafting)")
        else:
            print("  drafter    %-36s %s  (%s)%s" % (d["file"], _human((pack.root / d["file"]).stat().st_size) if pack.drafter else "-",
                                                 d.get("format", "hf-safetensors"), "  PROBLEM: " + pack.drafter_problem
                                                 if pack.drafter_problem else ""))
        print("  sampling   temperature %g, top_p %g, top_k %d (generation_config.json); stop ids %s"
              % (pack.sampling["temperature"], pack.sampling["top_p"], pack.sampling["top_k"], pack.stop_ids))
        print("  checksums  %d files listed in yasakani.json" % len(pack.sha256))
        fmt = ink.Ink2Formatter(pack.root)
        ids, _ = fmt.prompt([{"role": "user", "content": "hello"}])
        print("  tokenizer  ok (chat template renders a %d-id prompt)" % len(ids))
        print("  selfcheck  %s" % ("%d case(s)" % len(ink.selfcheck_cases(pack)) if (pack.root / "selfcheck.json").is_file() else "no selfcheck.json"))
    except (YasakaniError, ImportError, OSError, ValueError) as ex:
        print("model    PROBLEM:", ex)
        ok = False
    return 0 if ok else 1


def cmd_verify(args) -> int:
    from .cli import _human
    pack = ink.Ink2Pack.load(args.model_dir_path)
    if not pack.sha256:
        print("no checksums are listed for this folder (yasakani.json without \"sha256\").")
        return 2
    print("checking %d files (reads about %s) ..." % (len(pack.sha256), _human(sum(pack.sizes.values()))))
    res = pack.verify(progress=lambda n, ok: print("  %-36s %s" % (n, "OK" if ok else "DIFFERENT"), flush=True))
    bad = [n for n, ok in res.items() if not ok]
    print("all files match." if not bad else "%d file(s) differ: %s  (convert the folder again)" % (len(bad), ", ".join(bad)))
    if pack.weights_format == "yskn" and pack.weights.is_file():
        try:
            tbad = pack.check_tensors()
        except ValueError as ex:                          # ink2_yskn.FormatError: the header itself is damaged
            print("tensors: %s cannot be read: %s" % (pack.weights.name, ex))
            return 1
        if tbad:
            print("tensors: the CRC-32 of %d tensor(s) of %s differs: %s" % (len(tbad), pack.weights.name, ", ".join(tbad[:8])
                                                                            + (" ..." if len(tbad) > 8 else "")))
            return 1
        print("tensors: the CRC-32 of every tensor of %s matches." % pack.weights.name)
    return 0 if not bad else 1


# ---- selfcheck --------------------------------------------------------------------------------------------------------------------
def cmd_selfcheck(args) -> int:
    refuse_unsupported(args)
    pack = ink.Ink2Pack.load(args.model_dir_path)
    cases = ink.selfcheck_cases(pack, getattr(args, "reference", None))
    if not cases:
        print("the reference file has no cases")
        return 2
    pack, eng, fmt, runner = open_stack(args, warm=False)
    try:
        rc = ink.run_selfcheck(eng, fmt, cases)
    finally:
        eng.close()
    print("self-check passed: the output equals the reference." if rc == 0 else "self-check FAILED.")
    return rc


# ---- chat -------------------------------------------------------------------------------------------------------------------------
CHAT_HELP = """commands:  /exit  quit      /reset  forget the conversation      /think on|off  thinking on or off (default on)
           /effort none|minimal|low|medium|high|xhigh|max  how long the model thinks (default: the template's 0.9)
           /help  this text      Ctrl+C while it writes: stop the answer"""


def _chat_params(args, pack, messages, kw):
    from .chat import ChatParams
    s = pack.sampling
    return ChatParams(messages=[dict(m) for m in messages], max_tokens=args.max_tokens,
                      temperature=s["temperature"] if args.temperature is None else args.temperature,
                      top_p=s["top_p"] if args.top_p is None else args.top_p,
                      top_k=s["top_k"] if args.top_k is None else args.top_k, seed=args.seed, template_kwargs=dict(kw))


def cmd_chat(args) -> int:
    from .chat import RequestError
    from .cli import _use_color
    pack, eng, fmt, runner = open_stack(args, warm=True)
    color = _use_color()
    dim = (lambda s: "\x1b[2m%s\x1b[0m" % s) if color else (lambda s: s)
    kw = {}
    if args.no_think:
        kw["enable_thinking"] = False
    if args.effort:
        kw["reasoning_effort"] = args.effort
    base: List[dict] = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    one_shot = getattr(args, "ask", None)
    if not one_shot:
        print("YASAKANI chat (%s) - type /help for commands, /exit to quit." % pack.model_id)
    try:
        while True:
            if one_shot:
                line = one_shot
            else:
                try:
                    line = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not line:
                    continue
                if line.startswith("/"):
                    cmd, _, rest = line[1:].partition(" ")
                    rest = rest.strip().lower()
                    if cmd in ("exit", "quit", "q"):
                        return 0
                    if cmd == "reset":
                        messages = list(base)
                        print("(conversation cleared)")
                    elif cmd == "think" and rest in ("on", "off"):
                        kw["enable_thinking"] = rest == "on"
                        print("(thinking %s)" % rest)
                    elif cmd == "effort" and rest in ink.EFFORTS:
                        kw["reasoning_effort"] = rest
                        print("(reasoning effort %s)" % rest)
                    else:
                        print(CHAT_HELP)
                    continue
            messages.append({"role": "user", "content": line})
            params = _chat_params(args, pack, messages, kw)
            parts = {"reasoning": [], "content": []}
            last = [None]

            def emit(kind: str, text: str) -> None:
                parts[kind].append(text)
                if kind == "reasoning" and args.hide_thinking:
                    return
                if last[0] != kind:
                    if kind == "reasoning":
                        print(dim("[thinking]"), flush=True)
                    elif last[0] == "reasoning":
                        print(dim("\n[/thinking]\n"), flush=True)
                    last[0] = kind
                print(dim(text) if kind == "reasoning" else text, end="", flush=True)

            try:
                u = runner.run(params, emit)
            except KeyboardInterrupt:
                print("\n[stopped]")
                messages.pop()
                eng.reset()
                if one_shot:
                    return 130
                continue
            except RequestError as ex:
                print("cannot do that: %s" % ex)
                messages.pop()
                if one_shot:
                    return 1
                continue
            if parts["reasoning"] and not parts["content"] and u.finish != "stop":
                print(dim("\n(the limit of %d tokens was reached while the model was still thinking, so no answer was written.  "
                          "/effort low shortens the thinking, /think off skips it, --max-tokens raises the limit)" % args.max_tokens))
            tps = (u.completion_tokens / (u.gen_ms / 1000.0)) if u.gen_ms > 0 else 0.0
            print(dim("\n\n[%d tokens, %.1f tokens/s%s, prompt %d tokens in %.2f s%s, %s]" % (
                u.completion_tokens, tps, ", %.2f tokens per pass" % u.spec["tokens_per_pass"] if u.spec else "",
                u.prompt_tokens, u.prompt_ms / 1000.0,
                " (%d kept from the last turn)" % u.cached_tokens if u.cached_tokens else "",
                "complete" if u.finish == "stop" else "hit the length limit")))
            if one_shot:
                return 0
            msg = {"role": "assistant", "content": "".join(parts["content"])}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"])
            messages.append(msg)
    finally:
        eng.close()


# ---- serve ------------------------------------------------------------------------------------------------------------------------
def cmd_serve(args) -> int:
    from .cli import _stop_on_signals
    from .server import ApiState, ServerConfig, bind
    refuse_unsupported(args)
    local = args.host in ("127.0.0.1", "localhost", "::1")
    if not local and not args.api_key:
        raise YasakaniError("refusing to listen on %s without --api-key: anyone who can reach this computer could use the GPU" % args.host)
    pack = ink.Ink2Pack.load(args.model_dir_path)
    context_of(args, pack)
    cfg = ServerConfig(host=args.host, port=args.port, api_key=args.api_key, model_id=pack.model_id,
                       default_max_tokens=args.default_max_tokens, max_tokens_cap=args.max_tokens_cap)
    cfg.sampling_defaults = dict(pack.sampling)
    cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
    state = ApiState(cfg)
    state.version = __version__
    try:
        httpd = bind(state)
    except OSError as ex:
        raise YasakaniError("cannot listen on %s:%d (%s).  Is another server already using this port?  Use --port to pick another."
                            % (args.host, args.port, ex.strerror or type(ex).__name__))
    cfg.port = httpd.server_address[1]
    _stop_on_signals()
    threading.Thread(target=httpd.serve_forever, kwargs=dict(poll_interval=0.2), name="http", daemon=True).start()
    base = "http://%s:%d" % ("[%s]" % args.host if ":" in args.host else args.host, cfg.port)
    _say("listening on %s  (answers /health while the model loads)" % base)
    eng = None
    try:
        pack, eng, fmt, runner = open_stack(args, warm=not args.no_warmup)
        state.runner = runner
        _say("ready: OpenAI-compatible API at %s/v1   (model id %r%s; one request at a time, text only)"
             % (base, cfg.model_id, ", API key required" if cfg.api_key else ""))
        _say("stop with Ctrl+C")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _say("stopping ...")
        return 0
    finally:
        state.runner = None
        httpd.shutdown()
        if eng is not None:
            eng.cancel()
            with state.gen_lock:
                eng.close()
        httpd.server_close()


# ---- bench ------------------------------------------------------------------------------------------------------------------------
def cmd_bench(args) -> int:
    from .chat import ChatParams
    from .cli import _gpu_lines, _load_prompts
    refuse_unsupported(args)
    n_tokens = 256 if args.n is None else args.n
    prompts = _load_prompts(args.prompts_file)
    pack, eng, fmt, runner = open_stack(args, warm=True)
    rows = []
    try:
        _say("GPU: " + "; ".join(_gpu_lines()))
        kw = {} if args.think else {"enable_thinking": False}
        for name, text in prompts:
            for run in range(args.runs):
                eng.reset()
                p = ChatParams(messages=[{"role": "user", "content": text}], max_tokens=n_tokens, temperature=0.0, ignore_eos=True,
                               loop_guard=False, template_kwargs=dict(kw))
                u = runner.run(p, lambda kind, t: None)
                tps = u.completion_tokens / (u.gen_ms / 1000.0) if u.gen_ms > 0 else 0.0
                rows.append(dict(test=name, run=run + 1, prompt_tokens=u.prompt_tokens, tokens=u.completion_tokens,
                                 seconds=round(u.gen_ms / 1000.0, 3), prompt_seconds=round(u.prompt_ms / 1000.0, 3), tokens_per_s=round(tps, 1)))
                if u.spec:
                    rows[-1].update(tokens_per_pass=round(u.spec["tokens_per_pass"], 2), draft_n=u.draft_n, draft_n_accepted=u.draft_n_accepted)
                if not args.json:
                    r = rows[-1]
                    print("%-12s run %d  prompt %5d (%.2f s)  generated %5d  %7.1f tokens/s%s" %
                          (name, r["run"], r["prompt_tokens"], r["prompt_seconds"], r["tokens"], r["tokens_per_s"],
                           "  %.2f tokens per pass" % r["tokens_per_pass"] if "tokens_per_pass" in r else ""), flush=True)
    finally:
        eng.close()
    mean = sum(r["tokens_per_s"] for r in rows) / len(rows)
    if args.json:
        print(json.dumps(dict(version=__version__, model=pack.model_id, gpu=_gpu_lines(), tokens_per_test=n_tokens, runs=rows,
                              mean_tokens_per_s=round(mean, 1), drafter=eng.draft_note), indent=1))
    else:
        print("mean over %d run(s): %.1f tokens/s   (greedy, thinking %s, %d tokens each; %s)" % (
            len(rows), mean, "on" if args.think else "off", n_tokens, eng.draft_note))
    return 0


COMMANDS = dict(info=cmd_info, verify=cmd_verify, selfcheck=cmd_selfcheck, chat=cmd_chat, serve=cmd_serve, bench=cmd_bench)


def dispatch(name: str, args):
    """The Inkling-2 command for ``name`` when the model folder of ``args`` is an Inkling-2 folder, else None."""
    d = getattr(args, "model_dir", None) or os.environ.get("YASAKANI_MODEL_DIR")
    if name not in COMMANDS or not d or not ink.is_inkling2(d):
        return None
    args.model_dir_path = Path(d)
    return COMMANDS[name]
