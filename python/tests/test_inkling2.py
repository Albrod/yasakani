# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini path: converter (published and quality schemes), the .yskn file of version 2, manifest and verify, chat
template, output stream, sampling, the engine on its host backend, the server, and the dispatch of the commands.

Needs (else the tests skip):
  YASAKANI_TEST_INK2_SOURCE  the downloaded hf-internal-testing/tiny-inkling2-mini folder (config.json, model.safetensors, tokenizer)
  YASAKANI_TEST_INK2_DLL     the host build of the Inkling-2 engine library (ink2_host.dll), with its binding ink2_native.py next to
                             it or one folder up (only for the engine and server tests)
The engine tests run on the CPU (about 20 s each class).
"""
import argparse
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2 as ink

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools" / "convert"))
import inkling2_convert as conv  # noqa: E402

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
DLL = os.environ.get("YASAKANI_TEST_INK2_DLL")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "model.safetensors").is_file() and (Path(SOURCE) / "tokenizer.json").is_file()
HAVE_DLL = bool(DLL) and Path(DLL).is_file()
NEED_SOURCE = "set YASAKANI_TEST_INK2_SOURCE to the downloaded tiny-inkling2-mini folder"
NEED_DLL = "set YASAKANI_TEST_INK2_DLL to the host build of the Inkling-2 engine library"

_FOLDER = {}


def tiny_folder() -> Path:
    """The tiny checkpoint converted once per test run (hard link of the weights where possible)."""
    if "dir" not in _FOLDER:
        d = Path(tempfile.mkdtemp(prefix="ink2_tiny_"))
        conv.convert(Path(SOURCE), d / "tiny", repo="hf-internal-testing/tiny-inkling2-mini",
                     revision=conv.hf_metadata(Path(SOURCE), "model.safetensors")[0] or "test")
        _FOLDER["dir"] = d
    return _FOLDER["dir"] / "tiny"


def tiny_quality_folder() -> Path:
    """The tiny checkpoint converted once per test run with --scheme quality (no reference run: selfcheck.json has no cases)."""
    if "qdir" not in _FOLDER:
        d = Path(tempfile.mkdtemp(prefix="ink2_tinyq_"))
        conv.convert(Path(SOURCE), d / "tinyq", repo="hf-internal-testing/tiny-inkling2-mini",
                     revision=conv.hf_metadata(Path(SOURCE), "model.safetensors")[0] or "test", scheme="quality", threads=4)
        _FOLDER["qdir"] = d
    return _FOLDER["qdir"] / "tinyq"


def tearDownModule():
    for k in ("dir", "qdir"):
        d = _FOLDER.pop(k, None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class ConverterTests(unittest.TestCase):
    def test_manifest_and_verify(self):
        root = tiny_folder()
        meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["architecture"], "inkling2")
        self.assertEqual(meta["source"]["repository"], "hf-internal-testing/tiny-inkling2-mini")
        self.assertEqual(meta["weights"], dict(file="model.safetensors", format="hf-safetensors",
                                               note="published file, unchanged; vision tensors unused"))
        self.assertEqual(meta["stop_token_ids"], [200006])
        self.assertEqual(meta["sampling"], dict(temperature=1.0, top_p=1.0, top_k=0))
        for name in ("model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja", "NOTICE", "selfcheck.json"):
            self.assertIn(name, meta["sha256"])
            self.assertEqual(meta["sizes"][name], (root / name).stat().st_size)
        pack = ink.Ink2Pack.load(root)
        self.assertTrue(ink.is_inkling2(root))
        self.assertEqual(pack.weights.name, "model.safetensors")
        self.assertTrue(all(pack.verify().values()))

    def test_verify_finds_a_changed_file(self):
        src = tiny_folder()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "copy"
            root.mkdir()
            for n in json.loads((src / "yasakani.json").read_text(encoding="utf-8"))["sha256"]:
                if n != "model.safetensors":
                    shutil.copyfile(src / n, root / n)
            shutil.copyfile(src / "yasakani.json", root / "yasakani.json")
            try:
                os.link(src / "model.safetensors", root / "model.safetensors")
            except OSError:
                shutil.copyfile(src / "model.safetensors", root / "model.safetensors")
            p = root / "chat_template.jinja"
            p.write_bytes(p.read_bytes().replace(b"Thinking effort level", b"Thinking effort Level"))   # same size, other bytes
            res = ink.Ink2Pack.load(root).verify()
            self.assertEqual([n for n, ok in res.items() if not ok], ["chat_template.jinja"])

    def test_weights_entry_names_the_file(self):
        """A later weights format changes only the weights entry: an unknown format is refused with a clear error."""
        src = tiny_folder()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "f"
            shutil.copytree(src, root, ignore=shutil.ignore_patterns("model.safetensors"))
            (root / "small.bin").write_bytes(b"x")
            meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
            meta["weights"] = {"file": "small.bin", "format": "yskn-ink2-v1"}
            (root / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "weights format 'yskn-ink2-v1' is not supported"):
                ink.Ink2Pack.load(root)

    def test_not_an_inkling2_folder(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(ink.is_inkling2(t))
            (Path(t) / "yasakani.json").write_text(json.dumps({"model_id": "qwen3.8-27b"}), encoding="utf-8")
            self.assertFalse(ink.is_inkling2(t))


# greedy tokens of the PyTorch reference implementation of the model on the decoded weights of the tiny checkpoint converted with
# --scheme quality (official prompt; dense matrices rounded to bfloat16 as the engine is given them)
TINY_QUALITY_TOKENS = [165105, 147669, 95722, 153201, 34066, 31311, 193569, 41817, 169626, 169626, 169626, 169626]


def flip_byte(path: Path, offset: int) -> None:
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0x10]))


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class QualitySchemeTests(unittest.TestCase):
    """--scheme quality on the tiny checkpoint: the .yskn file read back bit for bit, the manifest, the refusals, damage."""

    @classmethod
    def setUpClass(cls):
        from yasakani import ink2_yskn
        cls.Y = ink2_yskn
        cls.root = tiny_quality_folder()
        cls.meta = json.loads((cls.root / "yasakani.json").read_text(encoding="utf-8"))
        cls.path = cls.root / cls.meta["weights"]["file"]

    def test_read_back_bit_exact(self):
        """Every tensor: copied ones equal to the published bytes, quantised ones equal to the quantiser on the published values,
        the MXFP8 experts re-packed expert by expert; the decode gives the values of the formula."""
        import inkling2_quant as NQ
        st = conv.SafeTensors(Path(SOURCE) / "model.safetensors")
        counts = dict(copied=0, quantised=0, experts_copied=0, experts_repacked=0)
        with self.Y.YsknFile(self.path) as yf:
            self.assertEqual(yf.check(), [])
            for name, e in yf.tensors.items():
                enc = e["encoding"]
                if e["role"] == "experts":
                    if name + "_weight_amax" in st.entries:
                        for suf, part in conv.EXPERT_PARTS.items():
                            self.assertTrue(np.array_equal(np.asarray(yf.part(name, part)).ravel(),
                                                           np.asarray(st.view(name + suf)).ravel()), name)
                        counts["experts_copied"] += 1
                    else:
                        w, sc = st.view(name + "_weight"), st.view(name + "_weight_scale")
                        for j in range(e["shape"][0]):
                            q = NQ.nvfp4_quant(self.Y.decode_mxfp8(np.asarray(w[j]), np.asarray(sc[j])), conv.NVFP4_SEARCH, 1)
                            self.assertTrue(np.array_equal(q["weight"], yf.part(name, "weight")[j]), (name, j))
                            self.assertTrue(np.array_equal(q["scale"], yf.part(name, "scale")[j]), (name, j))
                            self.assertEqual(q["amax"][0], yf.part(name, "amax")[j])
                        counts["experts_repacked"] += 1
                elif enc in ("nvfp4", "mxfp8"):
                    x = st.f32(name)[:e["shape"][0]]
                    q = NQ.nvfp4_quant(x, conv.NVFP4_SEARCH) if enc == "nvfp4" else NQ.mxfp8_quant(x, conv.MXFP8_SEARCH)
                    for k, v in q.items():
                        self.assertTrue(np.array_equal(v.ravel(), np.asarray(yf.part(name, k)).ravel()), (name, k))
                    counts["quantised"] += 1
                else:
                    self.assertTrue(np.array_equal(np.asarray(yf.part(name, "data")).ravel(), np.asarray(st.view(name)).ravel()), name)
                    counts["copied"] += 1
            self.assertEqual(counts["experts_copied"] + counts["experts_repacked"], 2 * 18)
            self.assertEqual(counts["experts_repacked"], 8)                     # layers 14 to 17 of the tiny model are MXFP8
            self.assertGreater(counts["quantised"], 100)
            self.assertFalse(any(n.startswith("model.visual.") for n in yf.tensors))
            self.assertEqual(yf.entry("model.llm.unembed.weight")["shape"], [200058, 256])
            # the decode follows the formula of docs/FORMAT.md, section 10.3: element (3, 10) is in the low nibble of byte (3, 5)
            pr = yf.parts("model.llm.layers.0.attn.wq_du.weight")
            code = int(pr["weight"][3, 5]) & 0xF
            factor = np.float32(pr["amax"][0]) / np.float32(pr["denominator"][0])
            want = np.float32(self.Y.E2M1[code] * self.Y.E4M3[pr["scale"][3, 0]]) * factor
            self.assertEqual(yf.decode("model.llm.layers.0.attn.wq_du.weight")[3, 10], want)
            for p in (p for e in yf.tensors.values() for p in e["parts"]):
                self.assertEqual(p["offset"] % 4096, 0)

    def test_manifest_notice_and_verify(self):
        w = self.meta["weights"]
        self.assertEqual((w["format"], w["version"], w["scheme"], w["file"]), ("yskn", 2, "quality", self.path.name))
        self.assertEqual(w["size"], self.path.stat().st_size)
        self.assertEqual(w["sha256"], self.meta["sha256"][self.path.name])
        self.assertEqual(self.meta["model_id"], "tiny-inkling2-mini-quality")
        self.assertNotIn("model.safetensors", self.meta["sha256"])
        notice = (self.root / "NOTICE").read_text(encoding="utf-8")
        for text in ("re-quantised by this tool", "https://thinkingmachines.ai/model-acceptable-use-policy/", "Apache License 2.0",
                     "hf-internal-testing/tiny-inkling2-mini", "revision"):
            self.assertIn(text, notice)
        pack = ink.Ink2Pack.load(self.root)
        self.assertEqual(pack.weights_format, "yskn")
        self.assertTrue(all(pack.verify().values()))
        self.assertEqual(pack.check_tensors(), [])
        sc = json.loads((self.root / "selfcheck.json").read_text(encoding="utf-8"))
        self.assertEqual((sc["architecture"], sc["cases"]), ("inkling2", []))      # no reference run in this test

    def test_old_readers_refuse(self):
        """The reader of version 1 stops at the version field; the version 2 reader refuses a version 1 file; the package refuses
        a weights entry of another version."""
        sys.path.insert(0, str(HERE.parent.parent / "tools"))
        import yskn_decode
        with self.assertRaisesRegex(yskn_decode.FormatError, "unsupported version 2"):
            yskn_decode.Container(str(self.path))
        v1 = HERE.parent.parent / "tools" / "testdata" / "tiny.yskn"
        if v1.is_file():
            with self.assertRaisesRegex(self.Y.FormatError, "version 1"):
                self.Y.YsknFile(v1)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "f"
            shutil.copytree(self.root, root, ignore=shutil.ignore_patterns("*.yskn"))
            (root / self.path.name).write_bytes(b"YSKN")
            meta = dict(self.meta)
            meta["weights"] = dict(self.meta["weights"], version=3)
            (root / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "weights format 'yskn' is not supported"):
                ink.Ink2Pack.load(root)

    def test_damage_is_found(self):
        """A flipped byte in a tensor: verify (file hash and tensor CRC) names it; a damaged header is refused."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(self.root, root)
            path = root / self.path.name
            with self.Y.YsknFile(path) as yf:
                p = yf.part_entry("model.llm.unembed.weight", "weight")
                off = p["offset"] + p["nbytes"] // 2
            flip_byte(path, off)
            pack = ink.Ink2Pack.load(root)
            res = pack.verify()
            self.assertEqual([n for n, ok in res.items() if not ok], [path.name])
            self.assertEqual(pack.check_tensors(), ["model.llm.unembed.weight"])
            flip_byte(path, off)                                                 # back
            self.assertEqual(pack.check_tensors(), [])
            flip_byte(path, 200)                                                 # in the JSON header
            with self.assertRaisesRegex(self.Y.FormatError, "header is damaged"):
                self.Y.YsknFile(path)


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostEngineQualityTests(unittest.TestCase):
    """The engine on its host backend loads the .yskn folder (packed experts, decoded dense matrices): greedy tokens equal to those
    of the reference implementation on the same decoded weights; a damaged tensor is refused by the loader."""

    def test_greedy_equals_the_reference(self):
        pack = ink.Ink2Pack.load(tiny_quality_folder())
        fmt, eng, runner = ink.open_runner(pack, 128, dll=DLL, warm=False)
        try:
            ids, _ = fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
            logits, _ = eng.read_prompt(ids)
            got = []
            for k in range(len(TINY_QUALITY_TOKENS)):
                got.append(int(np.argmax(logits)))
                if k + 1 < len(TINY_QUALITY_TOKENS):
                    logits = eng.step(got[-1])
            self.assertEqual(got, TINY_QUALITY_TOKENS)
            self.assertIn("decoded to bfloat16", eng.weights_note)
        finally:
            eng.close()

    def test_loader_refuses_a_damaged_tensor(self):
        from yasakani import ink2_yskn as Y
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(tiny_quality_folder(), root)
            pack = ink.Ink2Pack.load(root)
            with Y.YsknFile(pack.weights) as yf:
                p = yf.part_entry("model.llm.layers.3.attn.wo_ud.weight", "scale")
                off = p["offset"] + 7
            flip_byte(pack.weights, off)
            with self.assertRaisesRegex(Exception, r"model\.llm\.layers\.3\.attn\.wo_ud\.weight"):
                ink.open_runner(pack, 64, dll=DLL, warm=False)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class TemplateAndStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmt = ink.Ink2Formatter(tiny_folder())

    def test_official_prompt_ids(self):
        ids, in_answer = self.fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
        self.assertEqual(ids, conv.OFFICIAL_PROMPT_IDS)
        self.assertFalse(in_answer)
        text = self.fmt.render(conv.OFFICIAL_MESSAGES, **conv.OFFICIAL_TEMPLATE_KWARGS)
        self.assertEqual(text, "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                               "<|message_user|><|content_text|>Explain in one sentence why the sky is usually blue.<|end_message|>")

    def test_effort_and_thinking_off(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        default = self.fmt.render(msgs)
        self.assertTrue(default.startswith("<|message_system|><|content_text|>Be brief.<|end_message|>"
                                           "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"))
        self.assertIn("Thinking effort level: 0.2<", self.fmt.render(msgs, reasoning_effort="low"))
        ids, in_answer = self.fmt.prompt(msgs, {"enable_thinking": False})
        self.assertTrue(in_answer)
        self.assertEqual(ids[-2:], [self.fmt.ids["<|message_model|>"], self.fmt.ids["<|content_text|>"]])
        self.assertIn("Thinking effort level: 0<", self.fmt.decode(ids))
        from jinja2.exceptions import TemplateError
        with self.assertRaises(TemplateError):
            self.fmt.prompt(msgs, {"reasoning_effort": "huge"})

    def test_earlier_turn_with_reasoning(self):
        msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "R"},
                {"role": "user", "content": "Q2"}]
        self.assertIn("<|message_model|><|content_thinking|>R<|end_message|><|message_model|><|content_text|>A<|end_message|>"
                      "<|content_model_end_sampling|><|message_user|>", self.fmt.render(msgs))

    def _split(self, ids, in_answer=False):
        s = ink.OutputStream(self.fmt, in_answer)
        parts = {"reasoning": [], "content": []}
        pieces = 0
        for t in ids:
            for kind, text in s.feed(t):
                parts[kind].append(text)
                pieces += 1
        for kind, text in s.flush():
            parts[kind].append(text)
        return "".join(parts["reasoning"]), "".join(parts["content"]), s, pieces

    def test_thinking_answer_split(self):
        think = "The user asks 2+2. Need: 4 — 日本語も 😀."
        answer = "The answer is 4. 答えは４です 👍🏽"
        ids = self.fmt.encode("<|message_model|><|content_thinking|>" + think + "<|end_message|>"
                              "<|message_model|><|content_text|>" + answer + "<|end_message|><|content_model_end_sampling|>")
        r, c, s, pieces = self._split(ids)
        self.assertEqual((r, c), (think, answer))
        n_think = len(self.fmt.encode("<|message_model|><|content_thinking|>" + think + "<|end_message|><|message_model|>"))
        self.assertEqual(s.reasoning_count, n_think)
        self.assertNotIn("�", r + c)
        self.assertLessEqual(pieces, len(ids))

    def test_answer_only_and_name_header(self):
        ids = self.fmt.encode("Sure, here it is.<|end_message|><|content_model_end_sampling|>")
        r, c, s, _ = self._split(ids, in_answer=True)
        self.assertEqual((r, c, s.reasoning_count), ("", "Sure, here it is.", 0))
        ids = self.fmt.encode("<|message_model|>helper<|content_text|>x<|end_message|><|message_model|><|content_text|>y<|end_message|>")
        r, c, s, _ = self._split(ids)
        self.assertEqual((r, c), ("", "xy"))           # the name after <|message_model|> is not shown; blocks are joined

    def test_split_characters_are_held_back(self):
        text = "𠮷野家 👨‍👩‍👧‍👦 🇯🇵"
        ids = self.fmt.encode("<|message_model|><|content_text|>" + text)
        s = ink.OutputStream(self.fmt)
        out = []
        for t in ids:
            for kind, piece in s.feed(t):
                self.assertNotIn("�", piece)
                out.append(piece)
        out += [p for _, p in s.flush()]
        self.assertEqual("".join(out), text)


class SamplerTests(unittest.TestCase):
    def test_greedy_and_seeded(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=5000).astype(np.float32)
        self.assertEqual(ink.Sampler(0.0)(logits), int(np.argmax(logits)))
        self.assertEqual(ink.Sampler(1.0, top_k=1, seed=4)(logits), int(np.argmax(logits)))
        self.assertEqual(ink.Sampler(1.0, top_p=1e-6, seed=4)(logits), int(np.argmax(logits)))
        a = [ink.Sampler(1.0, 0.9, 0, 0.0, seed=7)(logits) for _ in range(3)]
        s = ink.Sampler(1.0, 0.9, 0, 0.0, seed=7)
        b = [s(logits), s(logits), s(logits)]
        self.assertEqual(a[0], b[0])
        self.assertEqual(len(set(ink.Sampler(1.0, seed=i)(logits) for i in range(20))) > 1, True)

    def test_top_p_keeps_the_nucleus(self):
        logits = np.full(10000, -50.0, np.float32)
        logits[[3, 9, 77]] = [5.0, 4.9, 4.8]
        picks = {ink.Sampler(1.0, top_p=0.9, seed=i)(logits) for i in range(200)}
        self.assertTrue(picks <= {3, 9, 77} and len(picks) >= 2)
        picks = {ink.Sampler(1.0, top_k=2, seed=i)(logits) for i in range(200)}
        self.assertEqual(picks, {3, 9})


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = ink.Ink2Pack.load(tiny_folder())
        cls.fmt, cls.eng, cls.runner = ink.open_runner(cls.pack, 256, dll=DLL, warm=False)

    @classmethod
    def tearDownClass(cls):
        cls.eng.close()

    def test_selfcheck_reference_tokens(self):
        """Greedy on the official prompt: the 20 tokens the official test expects."""
        cases = ink.selfcheck_cases(self.pack)
        self.assertEqual(cases[0]["tokens"], [34066, 91318, 123193] + [169626] * 17)
        out = []
        self.assertEqual(ink.run_selfcheck(self.eng, self.fmt, cases, say=out.append), 0, out)

    def test_runner_usage_and_prefix_reuse(self):
        from yasakani.chat import ChatParams
        p = ChatParams(messages=[{"role": "user", "content": "hi"}], max_tokens=3, temperature=0.0, ignore_eos=True,
                       template_kwargs={"reasoning_effort": "none"})
        u = self.runner.run(p, lambda kind, text: None)
        self.assertEqual((u.completion_tokens, u.finish, u.cached_tokens), (3, "length", 0))
        first = list(self.eng.fed)
        self.assertEqual(len(first), u.prompt_tokens + 2)          # the last sampled token is not read
        # the same conversation one turn on: only the new ids are read
        ids, _ = self.fmt.prompt(p.messages, p.template_kwargs)
        longer = first + self.fmt.encode("<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>more<|end_message|>")
        _, cached = self.eng.read_prompt(longer)
        self.assertEqual(cached, len(first))
        _, cached = self.eng.read_prompt(ids)                      # shorter: read again from the start
        self.assertEqual(cached, 0)

    def test_requests_that_are_refused(self):
        from yasakani.chat import ChatParams, ContextLengthError, RequestError
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with self.assertRaisesRegex(RequestError, "tools are not supported"):
            self.runner.prepare_checked(ChatParams(messages=[{"role": "user", "content": "x"}], tools=tools))
        with self.assertRaises(ContextLengthError):
            self.runner.prepare_checked(ChatParams(messages=[{"role": "user", "content": "word " * 400}]))


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostServerTests(unittest.TestCase):
    """The OpenAI-compatible server on the tiny model (host backend): models, health, a streamed request with usage, a refused one."""

    @classmethod
    def setUpClass(cls):
        from yasakani.server import ApiState, ServerConfig, bind
        cls.pack = ink.Ink2Pack.load(tiny_folder())
        cls.fmt, cls.eng, runner = ink.open_runner(cls.pack, 256, dll=DLL, warm=False)
        cfg = ServerConfig(port=0, model_id=cls.pack.model_id)
        cfg.sampling_defaults = dict(cls.pack.sampling)
        cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
        cls.state = ApiState(cfg)
        cls.httpd = bind(cls.state)
        cfg.port = cls.httpd.server_address[1]
        cls.state.runner = runner
        threading.Thread(target=cls.httpd.serve_forever, kwargs=dict(poll_interval=0.1), daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.eng.close()

    def _req(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.state.cfg.port, timeout=120)
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json", "Host": "127.0.0.1:%d" % self.state.cfg.port})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_models_health_stream(self):
        st, data = self._req("GET", "/v1/models")
        self.assertEqual((st, json.loads(data)["data"][0]["id"]), (200, "tiny-inkling2-mini"))
        st, data = self._req("GET", "/health")
        self.assertEqual((st, json.loads(data)["max_context_tokens"]), (200, 256))
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                                                              "stream": True, "stream_options": {"include_usage": True},
                                                              "reasoning_effort": "none", "temperature": 0})
        self.assertEqual(st, 200)
        events = [json.loads(ln[6:]) for ln in data.decode().splitlines() if ln.startswith("data: {")]
        self.assertTrue(data.decode().rstrip().endswith("data: [DONE]"))
        usage = [e["usage"] for e in events if e.get("usage")][0]
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["prompt_tokens"], len(self.fmt.prompt([{"role": "user", "content": "hi"}], {"reasoning_effort": "none"})[0]))
        self.assertIn("length", [c.get("finish_reason") for e in events for c in e.get("choices", [])])

    def test_refused_requests(self):
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                                              "tools": [{"type": "function", "function": {"name": "f"}}]})
        self.assertEqual(st, 400)
        self.assertIn("tools are not supported", json.loads(data)["error"]["message"])
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"})
        self.assertEqual(st, 400)
        self.assertIn("none, minimal, low, medium, high, xhigh, max", json.loads(data)["error"]["message"])
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]})
        self.assertEqual(st, 400)


class ServerDefaultsTests(unittest.TestCase):
    def test_model_defaults_and_efforts(self):
        from yasakani.server import ServerConfig, parse_chat_request
        q = ServerConfig()
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, q)
        self.assertEqual((p.temperature, p.top_p, p.top_k, p.template_kwargs["reasoning_effort"]), (1.0, 0.95, 20, "xhigh"))
        c = ServerConfig(model_id="inkling-2-mini")
        c.sampling_defaults = dict(temperature=1.0, top_p=1.0, top_k=0)
        c.effort_aliases = dict(ink.EFFORT_ALIASES)
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, c)
        self.assertEqual((p.temperature, p.top_p, p.top_k, p.template_kwargs["reasoning_effort"]), (1.0, 1.0, 0, "high"))
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "top_p": 0.5, "temperature": 0.3, "seed": 9}, c)
        self.assertEqual((p.temperature, p.top_p, p.seed), (0.3, 0.5, 9))


class DispatchTests(unittest.TestCase):
    def test_commands_go_by_architecture(self):
        from yasakani import cli, inkling2_cli
        with tempfile.TemporaryDirectory() as t:
            q = Path(t) / "qwen"
            q.mkdir()
            (q / "yasakani.json").write_text(json.dumps({"model_id": "qwen3.8-27b"}), encoding="utf-8")
            k = Path(t) / "ink"
            k.mkdir()
            (k / "yasakani.json").write_text(json.dumps({"architecture": "inkling2"}), encoding="utf-8")
            for cmd in ("info", "verify", "selfcheck", "chat", "serve", "bench"):
                args = cli.build_parser().parse_args([cmd, "--model-dir", str(q)])
                self.assertIsNone(cli._inkling2_command(args), cmd)
                args = cli.build_parser().parse_args([cmd, "--model-dir", str(k)])
                self.assertIs(cli._inkling2_command(args), inkling2_cli.COMMANDS[cmd], cmd)
            args = cli.build_parser().parse_args(["update", "--check"])
            self.assertIsNone(cli._inkling2_command(args))

    def test_unsupported_options(self):
        from yasakani import inkling2_cli
        from yasakani._native import YasakaniError
        with self.assertRaisesRegex(YasakaniError, "--parallel"):
            inkling2_cli.refuse_unsupported(argparse.Namespace(parallel=2))
        inkling2_cli.refuse_unsupported(argparse.Namespace(parallel=None, edit=False, write=None, mmproj=None))

    def test_chat_defaults_left_to_the_model(self):
        from yasakani import cli
        args = cli.build_parser().parse_args(["chat"])
        self.assertEqual((args.temperature, args.top_p, args.top_k, args.prompt), (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
