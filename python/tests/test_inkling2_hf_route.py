# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The two Inkling-2 rules stay apart: a folder with yasakani.json ("architecture": "inkling2") is the 0.3.0 path; a folder of the
published safetensors without yasakani.json runs on the engine of #28 only with an engine library built with YSKN_WITH_INK2."""
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests import support                                   # noqa: F401
from yasakani import cli, engine, inkling2, runtime, server, textio
from yasakani._native import YasakaniError
from yasakani.engine import Sampling
from yasakani.runtime import ModelPack

QWEN_MISSING = "is missing: qwen38_27b_e3x.yskn, bdraft2_k8.yskn, bdraft2_k8.shead, kvq_u5.bin"


def _hf_folder(root: Path, manifest: bool = False) -> Path:
    """A folder shaped like the published Inkling-2 Mini download (files are placeholders)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "inkling2_loop", "eos_token_id": 200006,
                                                  "text_config": {"model_type": "inkling2_loop_text", "vocab_size": 201024,
                                                                  "unpadded_vocab_size": 200058}}), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"do_sample": True, "temperature": 1.0, "eos_token_id": 200006,
                                                             "pad_token_id": 199999}), encoding="utf-8")
    for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / n).write_text("{}", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 16)
    (root / "mtp.safetensors").write_bytes(b"\0" * 16)
    if manifest:
        (root / "yasakani.json").write_text(json.dumps({"architecture": "inkling2", "format_version": 1,
                                                        "weights": {"file": "model-00001-of-00001.safetensors",
                                                                    "format": "hf-safetensors"}}), encoding="utf-8")
    return root


class _Lib:
    def __init__(self, info):
        self.info, self.fn = info, {}

    def build_info(self):
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


class HfInkling2RouteTests(unittest.TestCase):
    def test_default_load_gives_the_error_of_030_with_a_hint(self):
        with tempfile.TemporaryDirectory() as t:
            d = _hf_folder(Path(t) / "Inkling-2-Mini")
            with self.assertRaises(YasakaniError) as cm:
                ModelPack.load(d)
            msg = str(cm.exception)
            self.assertIn(QWEN_MISSING, msg)
            self.assertIn("inkling2_convert.py", msg)
            self.assertIn("-DYSKN_WITH_INK2=ON", msg)
            self.assertEqual(str(runtime.hf_inkling2_error(d)), msg)

    def test_hf_load_uses_the_model_sampling(self):
        with tempfile.TemporaryDirectory() as t:
            d = _hf_folder(Path(t) / "Inkling-2-Mini")
            pack = ModelPack.load(d, hf_inkling2=True)
            self.assertTrue(pack.is_hf_inkling2)
            self.assertEqual(pack.weights.name, "model-00001-of-00001.safetensors")
            self.assertEqual(pack.eos_ids, (200006, 199999))
            self.assertEqual(pack.vocab_size, 200058)
            self.assertEqual(pack.sampling_defaults, dict(temperature=1.0, top_p=1.0, top_k=0))

    def test_a_manifest_keeps_the_030_rule(self):
        with tempfile.TemporaryDirectory() as t:
            d = _hf_folder(Path(t) / "ink", manifest=True)
            self.assertTrue(inkling2.is_inkling2(d))
            self.assertFalse(runtime.is_hf_inkling2_folder(d))
            with self.assertRaises(YasakaniError) as cm:                # never the path of #28, even when the caller allows it
                ModelPack.load(d, hf_inkling2=True)
            self.assertIn(QWEN_MISSING, str(cm.exception))
            self.assertNotIn("YSKN_WITH_INK2", str(cm.exception))
            args = cli.build_parser().parse_args(["chat", "--model-dir", str(d)])
            self.assertIsNotNone(cli._inkling2_command(args))          # main() sends it to the 0.3.0 engine first

    def test_qwen_folder_is_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            pack = support.make_pack(Path(t))
            (pack.root / "config.json").write_text("not json", encoding="utf-8")    # 0.3.0 did not read it either
            p2 = ModelPack.load(pack.root, hf_inkling2=True)
            self.assertFalse(p2.is_hf_inkling2)
            self.assertEqual(p2.weights, pack.weights)
            self.assertEqual(p2.sampling_defaults, {})

    def test_library_flag(self):
        self.assertFalse(runtime.lib_has_hf_inkling2(_Lib("cuda=13020;wide=1")))
        self.assertTrue(runtime.lib_has_hf_inkling2(_Lib("cuda=13010;wide=1;ink2=1")))
        self.assertFalse(runtime.lib_has_hf_inkling2(_Lib("cuda=13010;wide=1;ink2=10")))
        self.assertFalse(runtime.lib_has_hf_inkling2(_Lib("")))
        self.assertFalse(runtime.lib_has_hf_inkling2(_Lib(RuntimeError("old library"))))

    def test_engine_refuses_a_library_without_the_engine(self):
        with tempfile.TemporaryDirectory() as t:
            pack = ModelPack.load(_hf_folder(Path(t) / "m"), hf_inkling2=True)
            lib = _Lib("cuda=13020;wide=1")
            lib.fn["yskn_open"] = mock.Mock(side_effect=AssertionError("yskn_open must not be called"))
            with mock.patch.dict(os.environ, {}, clear=False):
                with self.assertRaises(YasakaniError) as cm:
                    engine.Engine(pack, max_ctx=4096, lib=lib)
            self.assertIn(QWEN_MISSING, str(cm.exception))

    def test_open_stack_checks_the_library_and_refuses_qwen_options(self):
        with tempfile.TemporaryDirectory() as t:
            d = _hf_folder(Path(t) / "m")
            dll = Path(t) / "fake.so"
            dll.write_bytes(b"x")
            args = cli.build_parser().parse_args(["serve", "--model-dir", str(d), "--dll", str(dll)])
            with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), \
                 mock.patch.object(textio, "ChatFormatter", return_value=object()), \
                 mock.patch.object(cli, "NativeLib", return_value=_Lib("cuda=13020;wide=1")), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(YasakaniError) as cm:
                    cli._open_stack(args, warm=False)
            self.assertIn(QWEN_MISSING, str(cm.exception))
            for extra, what in ((["--yarn2"], "--yarn2"),):
                a = cli.build_parser().parse_args(["serve", "--model-dir", str(d), "--dll", str(dll)] + extra)
                with mock.patch.dict(os.environ, {"YASAKANI_NVIDIA_SMI": "none"}), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(YasakaniError, what + " is for Qwen3.8 model folders only"):
                        cli._open_stack(a, warm=False)

    def test_supports_forced_is_false(self):
        with tempfile.TemporaryDirectory() as t:
            pack = ModelPack.load(_hf_folder(Path(t) / "m"), hf_inkling2=True)
            e = object.__new__(engine.Engine)
            e.pack, e._f = pack, {"yskn_verify_nll": 1, "yskn_commit": 1}
            self.assertFalse(e.supports_forced)
            e.pack = support.make_pack(Path(t) / "q")
            self.assertTrue(e.supports_forced)

    def test_full_vocabulary_sampling(self):
        self.assertEqual(Sampling(1.0, 1.0, 0, 0.0, 7).normalized_full(), Sampling(1.0, 1.0, 0, 0.0, 7))
        self.assertEqual(Sampling(0.7, 0.9, 50, 0.05, 3).normalized_full().top_k, 50)
        self.assertEqual(Sampling(0.0, 0.5, 9, 0.0, 3).normalized_full(), Sampling(0.0, 1.0, 1, 0.0, 0))
        self.assertEqual(Sampling(1.0, 1.0, 0, 0.0, 7).normalized().top_k, 20)          # the Qwen3.8 engine: unchanged

    def test_server_reasoning_effort_on_both_paths(self):
        body = lambda eff: {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": eff}
        cfg = server.ServerConfig()
        self.assertEqual(server.parse_chat_request(body("xhigh"), cfg).template_kwargs["reasoning_effort"], "xhigh")
        with self.assertRaises(server.RequestError):
            server.parse_chat_request(body("max"), cfg)
        hf = server.ServerConfig(hf_inkling2=True)
        self.assertEqual(server.parse_chat_request(body("max"), hf).template_kwargs["reasoning_effort"], "max")
        self.assertEqual(server.parse_chat_request(body(0.5), hf).template_kwargs["reasoning_effort"], 0.5)
        for bad in ("turbo", 1.5, True):
            with self.assertRaises(server.RequestError):
                server.parse_chat_request(body(bad), hf)
        ours = server.ServerConfig()                                       # the 0.3.0 Inkling-2 path (inkling2_cli)
        ours.effort_aliases = dict(inkling2.EFFORT_ALIASES)
        self.assertEqual(server.parse_chat_request(body("none"), ours).template_kwargs["reasoning_effort"], "none")
        tool_body = {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "f", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "1"}]}
        self.assertNotIn("tool_call_id", server.parse_chat_request(tool_body, cfg).messages[1])
        ours.keep_tool_ids = True
        self.assertEqual(server.parse_chat_request(tool_body, ours).messages[1]["tool_call_id"], "c1")


if __name__ == "__main__":
    unittest.main()
