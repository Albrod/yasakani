# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini path: tool calls (the parsing draws on PR #28 by @audreyt).  The chat template with tools, the reply stream that
turns call blocks into tool calls, tool_choice (the forced call header and the names the call header may take), the thinking budget with
tools, and the OpenAI-compatible server on a stand-in engine (no engine library is needed).

Needs (else the tests skip):
  YASAKANI_TEST_INK2_SOURCE  the downloaded hf-internal-testing/tiny-inkling2-mini folder (its tokenizer.json and chat_template.jinja
                             are those of Inkling-2 Mini)
"""
import http.client
import json
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2 as ink
from yasakani.chat import ChatParams, RequestError, clean_messages, clean_tools

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "tokenizer.json").is_file() and (Path(SOURCE) / "chat_template.jinja").is_file()
NEED_SOURCE = "set YASAKANI_TEST_INK2_SOURCE to the downloaded tiny-inkling2-mini folder"

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}
SHELL = {"type": "function", "function": {"name": "run_command", "parameters": {"type": "object",
                                                                              "properties": {"command": {"type": "string"}}}}}
WRITE = {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object",
                                                                             "properties": {"path": {"type": "string"}}}}}


def fn(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}

_FMT = {}


def fmt() -> "ink.Ink2Formatter":
    if "f" not in _FMT:
        _FMT["f"] = ink.Ink2Formatter(SOURCE)
    return _FMT["f"]


class FakeEngine:
    """Stands in for Ink2Engine in Ink2Runner.run: the "model" writes ``reply(prompt)`` token by token (its logits put that token
    first), and at the reply positions listed in ``decoys`` it would rather write the decoy (a higher logit).  A decoy is not
    written only where the runner forces its own tokens or masks the decoy."""

    def __init__(self, f, reply, decoys=None, max_ctx=4096):
        self.f, self.reply, self.decoys = f, reply, dict(decoys or {})
        self.vocab = f.tok.get_vocab_size()
        self.max_ctx = max_ctx
        self.fed = []
        self.prompts = []
        self.n_prompt = 0
        self.want = []
        self._cancel = threading.Event()
        self.drafting = False
        self.draft_mode = "off"

    def _logits(self):
        g = len(self.fed) - self.n_prompt
        x = np.zeros(self.vocab, np.float32)
        x[self.want[g] if g < len(self.want) else self.f.ids[ink.T_STOP]] = 10.0
        if g in self.decoys:
            x[self.decoys[g]] = 20.0
        return x

    def read_prompt(self, ids):
        self.prompts.append(list(ids))
        self.fed, self.n_prompt = list(ids), len(ids)
        self.want = list(self.reply(list(ids)))
        return self._logits(), 0

    def step(self, tok):
        self.fed.append(int(tok))
        return self._logits()

    def can_draft(self):
        return False

    def draft_stats(self):
        return {}

    def truncate(self, n):
        del self.fed[n:]

    def reset(self):
        self.fed = []


class DraftingFakeEngine(FakeEngine):
    """FakeEngine with drafting passes: a pass feeds the anchor and ``depth`` ids, each the best of its logits (all accepted), and
    gives the best id after them (as the engine's spec_step: the anchor and the accepted ids are fed, the next id is not)."""

    def __init__(self, *a, depth=4, **k):
        super().__init__(*a, **k)
        self.depth, self.drafting, self.draft_mode, self.passes = depth, True, "on", 0

    def can_draft(self):
        return True

    def spec_step(self, anchor, pick=None):
        self.passes += 1
        self.fed.append(int(anchor))
        out = []
        for _ in range(self.depth):
            out.append(int(np.argmax(self._logits())))
            self.fed.append(out[-1])
        return out, int(np.argmax(self._logits()))


def runner_for(eng, f):
    return ink.Ink2Runner(eng, f, SimpleNamespace(stop_ids=[f.ids[ink.T_STOP]]))


def run(runner, p):
    got = []
    u = runner.run(p, lambda kind, item: got.append((kind, item)))
    return u, got


def joined(got, kind):
    return "".join(x for k, x in got if k == kind)


def calls(got):
    return [x for k, x in got if k == "tool_call"]


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class TemplateTests(unittest.TestCase):
    def test_declaration_block(self):
        f = fmt()
        msgs = [{"role": "user", "content": "Kyoto?"}]
        text = f.render(msgs, tools=clean_tools([WEATHER]))
        self.assertTrue(text.startswith(
            '<|message_system|>tool_declare<|content_xml|>[{"description":"Weather of a city","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]'
            '<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>'))
        self.assertEqual(f.render(msgs), f.render(msgs, tools=None))          # no tools: the prompt of before
        self.assertNotIn("tool_declare", f.render(msgs))
        no_desc = f.render(msgs, tools=clean_tools([SHELL]))                   # a missing description is "" in the spec
        self.assertIn('[{"description":"","name":"run_command"', no_desc)

    def test_history_with_a_call_and_its_result(self):
        f = fmt()
        raw = [{"role": "user", "content": "Kyoto?"},
               {"role": "assistant", "content": None, "reasoning_content": "Use the tool.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"京都\"}"}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
               {"role": "tool", "name": "run_command", "content": "ok"}]
        msgs = clean_messages(raw, tool_ids=True)
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msgs[2]["tool_call_id"], "call_1")
        self.assertNotIn("tool_call_id", clean_messages(raw)[2])                # the Qwen3.8 path: as before
        self.assertNotIn("id", clean_messages(raw)[1]["tool_calls"][0])
        text = f.render(msgs, tools=clean_tools([WEATHER, SHELL]))
        self.assertIn('<|message_user|><|content_text|>Kyoto?<|end_message|>'
                      '<|message_model|><|content_thinking|>Use the tool.<|end_message|>'
                      '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"京都"}}<|end_message|>'
                      '<|content_model_end_sampling|>'
                      '<|message_tool|>get_weather<|content_text|>sunny<|end_message|>'
                      '<|message_tool|>run_command<|content_text|>ok<|end_message|>', text)

    def test_runner_reads_string_arguments(self):
        """The runner gives the template the arguments as an object even when the client sends JSON text (the template refuses it)."""
        f = fmt()
        r = runner_for(FakeEngine(f, lambda p: []), f)
        p = ChatParams(messages=[{"role": "user", "content": "x"},
                                 {"role": "assistant", "content": "", "tool_calls": [
                                     {"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Nara\"}"}}]},
                                 {"role": "tool", "name": "get_weather", "content": "rain"}],
                       tools=clean_tools([WEATHER]))
        ids, start, plan = r.prepare_checked(p)
        self.assertIn('{"name":"get_weather","args":{"city":"Nara"}}', f.decode(ids))
        self.assertEqual((start, plan["call"], plan["ban_first"]), ("free", None, []))
        bad = ChatParams(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": "{oops"}}]}], tools=clean_tools([WEATHER]))
        with self.assertRaisesRegex(RequestError, "not valid JSON"):
            r.prepare_checked(bad)
        ids2, in_answer = r.prepare_checked(ChatParams(messages=[{"role": "user", "content": "x"}]))     # no tools: as before
        self.assertEqual((ids2, in_answer), tuple(f.prompt([{"role": "user", "content": "x"}])))

    def test_prompt_endings(self):
        f = fmt()
        i = f.ids
        msgs = [{"role": "user", "content": "Kyoto?"}]
        tools = clean_tools([WEATHER, SHELL])
        base = f.encode(f.render(msgs, tools=tools))
        base0 = f.encode(f.render(msgs, tools=tools, reasoning_effort="none"))
        off = {"enable_thinking": False}
        self.assertEqual(f.tool_prompt(msgs, tools, {}, None), (base, "free", []))
        self.assertEqual(f.tool_prompt(msgs, tools, off, None), (base0, "free", []))      # effort 0; the model writes its header
        hdr = [i[ink.T_MODEL]] + f.encode("get_weather") + [i[ink.T_CALL]]
        self.assertEqual(f.call_header("get_weather"), hdr)
        self.assertEqual(f.tool_prompt(msgs, tools, off, "get_weather"), (base0 + hdr, "call", []))
        self.assertEqual(f.tool_prompt(msgs, tools, {}, "get_weather"), (base + [i[ink.T_MODEL], i[ink.T_THINKING]], "reasoning", []))
        self.assertEqual(f.tool_prompt(msgs, tools, {}, ""), (base + [i[ink.T_MODEL], i[ink.T_THINKING]], "reasoning", []))
        ids, start, allow = f.tool_prompt(msgs, tools, off, "")
        self.assertEqual((ids, start), (base0 + [i[ink.T_MODEL]], "head"))
        self.assertIsInstance(allow, ink.Allow)                              # only the first id of one of the names
        self.assertEqual(set(allow), {f.encode("get_weather")[0], f.encode("run_command")[0]})
        self.assertIn("Thinking effort level: 0<", f.decode(base0))


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class StreamTests(unittest.TestCase):
    def split(self, text, **kw):
        f = fmt()
        s = ink.OutputStream(f, **kw)
        out = []
        for t in f.encode(text):
            out += s.feed(t)
        out += s.flush()
        return out, s

    def test_thinking_and_two_calls(self):
        out, s = self.split('<|message_model|><|content_thinking|>Two cities.<|end_message|>'
                            '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"大阪"}}<|end_message|>'
                            '<|message_model|>get_weather<|content_invoke_tool_json|>{"name": "get_weather", "args": {"city": "東京", "days": 1}}'
                            '<|end_message|><|content_model_end_sampling|>', tools=True)
        self.assertEqual(joined(out, "reasoning"), "Two cities.")
        self.assertEqual(joined(out, "content"), "")
        self.assertEqual(calls(out), [{"name": "get_weather", "arguments": {"city": "大阪"}},
                                      {"name": "get_weather", "arguments": {"city": "東京", "days": 1}}])
        self.assertEqual(s.calls, 2)

    def test_text_then_call_and_header_name(self):
        out, _ = self.split('<|message_model|><|content_text|>Let me check.<|end_message|>'
                            '<|message_model|>run_command<|content_invoke_tool_json|>{"args":{"command":"dir"}}<|end_message|>', tools=True)
        self.assertEqual(joined(out, "content"), "Let me check.")
        self.assertEqual(calls(out), [{"name": "run_command", "arguments": {"command": "dir"}}])
        out, _ = self.split('<|message_model|>run_command<|content_invoke_tool_json|>{"command":"ls"}<|end_message|>', tools=True)
        self.assertEqual(calls(out), [{"name": "run_command", "arguments": {"command": "ls"}}])   # only the arguments
        out, _ = self.split('<|message_model|><|content_invoke_tool_json|>{"name":"f","arguments":"{\\"a\\": 1}"}<|end_message|>', tools=True)
        self.assertEqual(calls(out), [{"name": "f", "arguments": {"a": 1}}])                 # arguments as JSON text

    def test_unreadable_blocks_come_back_as_text(self):
        for body in ('{"name":"get_weather","args":', 'not json', '["a"]', '{"name":"f","args":[1]}', '{"args":{}}'):
            out, s = self.split('<|message_model|><|content_invoke_tool_json|>' + body + '<|end_message|>', tools=True)
            self.assertEqual((calls(out), joined(out, "content"), s.calls), ([], body, 0), body)
        out, _ = self.split('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Kobe"}}',
                            tools=True)                                    # the reply ended (length) right after a complete object
        self.assertEqual(calls(out), [{"name": "get_weather", "arguments": {"city": "Kobe"}}])
        out, _ = self.split('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"ci', tools=True)
        self.assertEqual((calls(out), joined(out, "content")), ([], '{"name":"get_weather","args":{"ci'))

    def test_without_tools_a_call_block_is_text(self):
        body = '{"name":"get_weather","args":{"city":"Kobe"}}'
        out, _ = self.split('<|message_model|>get_weather<|content_invoke_tool_json|>' + body + '<|end_message|>')
        self.assertEqual((calls(out), joined(out, "content")), ([], body))

    def test_starts(self):
        out, s = self.split('{"name":"get_weather","args":{"city":"Nara"}}<|end_message|>', tools=True, start="call",
                            call_name="get_weather")
        self.assertEqual(calls(out), [{"name": "get_weather", "arguments": {"city": "Nara"}}])
        out, s = self.split('plan<|end_message|><|message_model|><|content_text|>A<|end_message|>', tools=True, start="reasoning")
        self.assertEqual((joined(out, "reasoning"), joined(out, "content"), s.saw_reasoning), ("plan", "A", True))
        with self.assertRaises(ValueError):
            ink.OutputStream(fmt(), start="nowhere")


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class RunnerTests(unittest.TestCase):
    """Ink2Runner.run on the stand-in engine: what is emitted, the finish reason, forcing and masking."""

    def ids(self, text):
        return fmt().encode(text)

    def test_auto_thinking_on(self):
        f = fmt()
        reply = self.ids('<|message_model|><|content_thinking|>Need weather.<|end_message|>'
                         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"京都"}}'
                         '<|end_message|><|content_model_end_sampling|>')
        eng = FakeEngine(f, lambda p: reply)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Kyoto?"}], tools=clean_tools([WEATHER]),
                                                    temperature=0.0))
        self.assertEqual(calls(got), [{"name": "get_weather", "arguments": {"city": "京都"}}])
        self.assertEqual((u.finish, u.tool_calls, joined(got, "reasoning")), ("tool_calls", 1, "Need weather."))
        self.assertEqual(eng.fed[eng.n_prompt:], reply[:-1])                # every emitted token but the last one is fed

    def test_plain_answer_with_tools_offered(self):
        f = fmt()
        reply = self.ids('<|message_model|><|content_thinking|>Easy.<|end_message|><|message_model|><|content_text|>15<|end_message|>'
                         '<|content_model_end_sampling|>')
        eng = FakeEngine(f, lambda p: reply)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "1+...+5?"}], tools=clean_tools([WEATHER]),
                                                    temperature=0.0, template_kwargs={"enable_thinking": False}))
        self.assertEqual((joined(got, "content"), joined(got, "reasoning"), calls(got), u.finish), ("15", "Easy.", [], "stop"))
        self.assertIn("Thinking effort level: 0<", f.decode(eng.prompts[0]))
        self.assertEqual(eng.prompts[0][-1], f.ids[ink.T_END])              # nothing opened for the model

    def test_named_function_thinking_off(self):
        f = fmt()
        reply = self.ids('{"name":"get_weather","args":{"city":"Nara"}}<|end_message|><|content_model_end_sampling|>')
        eng = FakeEngine(f, lambda p: reply)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Nara"}], tools=clean_tools([WEATHER]),
                                                    tool_choice="required", temperature=0.0, template_kwargs={"enable_thinking": False}))
        self.assertEqual(eng.prompts[0][-len(f.call_header("get_weather")):], f.call_header("get_weather"))
        self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {"city": "Nara"}}], "tool_calls"))

    def test_named_function_after_thinking(self):
        """The model ends its thinking and would write a text block; the call header is written for it."""
        f = fmt()
        head = self.ids('Think.<|end_message|>')
        hdr = f.call_header("get_weather")
        reply = head + hdr + self.ids('{"name":"get_weather","args":{"city":"Kobe"}}<|end_message|><|content_model_end_sampling|>')
        decoys = {len(head) + k: f.ids[ink.T_TEXT] for k in range(len(hdr))}
        eng = FakeEngine(f, lambda p: reply, decoys=decoys)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Kobe"}], tools=clean_tools([WEATHER]),
                                                    tool_choice="required", temperature=0.0))
        self.assertEqual(eng.prompts[0][-2:], [f.ids[ink.T_MODEL], f.ids[ink.T_THINKING]])
        self.assertEqual((joined(got, "reasoning"), calls(got), u.finish),
                         ("Think.", [{"name": "get_weather", "arguments": {"city": "Kobe"}}], "tool_calls"))

    def test_required_among_several(self):
        """tool_choice required with two functions: after the thinking a header, then only ids that spell one of the names."""
        f = fmt()
        i = f.ids
        head = self.ids('Think.<|end_message|><|message_model|>')
        rest = self.ids('run_command<|content_invoke_tool_json|>{"name":"run_command","args":{"command":"dir"}}<|end_message|>'
                        '<|content_model_end_sampling|>')
        reply = head + rest
        eng = FakeEngine(f, lambda p: reply, decoys={len(head) - 1: i[ink.T_TEXT], len(head): i[ink.T_TEXT]})
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "list"}],
                                                    tools=clean_tools([WEATHER, SHELL]), tool_choice="required", temperature=0.0))
        self.assertEqual((calls(got), u.finish), ([{"name": "run_command", "arguments": {"command": "dir"}}], "tool_calls"))
        self.assertEqual((joined(got, "reasoning"), joined(got, "content")), ("Think.", ""))
        self.assertEqual(eng.fed[eng.n_prompt:], reply[:-1])
        # thinking off: the first pick is masked
        eng = FakeEngine(f, lambda p: rest, decoys={0: i[ink.T_TEXT]})
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "list"}], tools=clean_tools([WEATHER, SHELL]),
                                                    tool_choice="required", temperature=0.0, template_kwargs={"enable_thinking": False}))
        self.assertEqual((calls(got), u.finish), ([{"name": "run_command", "arguments": {"command": "dir"}}], "tool_calls"))
        self.assertEqual((joined(got, "reasoning"), joined(got, "content")), ("", ""))

    def junk_header_run(self, tools, want, thinking=True, lead="Think.<|end_message|>"):
        """tool_choice required, and a "model" that after <|message_model|> would rather write a junk id, then close the block and
        stop (what the real model did: its answer as the name of a block, so no call) and only second best spells ``want``."""
        f = fmt()
        i = f.ids
        junk = f.encode("\u4e00")[0]
        head = self.ids(lead + "<|message_model|>") if thinking else []
        body = f.encode(want) + [i[ink.T_CALL]] + self.ids('{"name":"%s","args":{}}<|end_message|><|content_model_end_sampling|>' % want)
        g = len(head)
        n = len(f.encode(want)) + 1                                      # the name and the call opener
        decoys = {g + k: (junk, i[ink.T_END], i[ink.T_STOP], i[ink.T_TEXT])[min(k, 3)] for k in range(n)}
        if thinking and lead.endswith("<|end_message|>"):
            decoys[g - 1] = i[ink.T_TEXT]                                # it would open a text block instead of the header
        eng = FakeEngine(f, lambda p: head + body, decoys=decoys)
        kw = {} if thinking else {"template_kwargs": {"enable_thinking": False}}
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "What is 17 times 3?"}],
                                                    tools=clean_tools(tools), tool_choice="required", temperature=0.0, **kw))
        return u, got, eng, head + body

    def test_required_among_several_junk_header(self):
        """The real model's failure (thinking on and off): a junk id after the header and the block closed.  The name is written."""
        for thinking in (True, False):
            u, got, eng, reply = self.junk_header_run([WEATHER, WRITE, SHELL], "run_command", thinking)
            self.assertEqual((calls(got), u.finish), ([{"name": "run_command", "arguments": {}}], "tool_calls"), thinking)
            self.assertEqual(joined(got, "content"), "", thinking)
            self.assertEqual(eng.fed[eng.n_prompt:], reply[:-1], thinking)
            self.assertEqual(joined(got, "reasoning"), "Think." if thinking else "")

    def test_required_names_with_a_shared_beginning(self):
        """get, get_weather and get_weather_week: a complete name that begins another may end or go on; each is reachable."""
        f = fmt()
        i = f.ids
        tools = [fn("get"), fn("get_weather"), fn("get_weather_week")]
        trie = f.name_trie(clean_tools(tools))
        g, us, w, us2, wk = f.encode("get_weather_week")
        self.assertEqual(us2, us)
        self.assertEqual((f.encode("get"), f.encode("get_weather")), ([g], [g, us, w]))
        self.assertEqual(set(trie.root), {g})
        self.assertEqual(set(trie.root[g]), {us, i[ink.T_CALL]})               # "get" is complete and goes on
        self.assertEqual(set(trie.root[g][us][w]), {us, i[ink.T_CALL]})        # "get_weather" too
        self.assertEqual(set(trie.root[g][us][w][us][wk]), {i[ink.T_CALL]})
        for want in ("get", "get_weather", "get_weather_week"):
            for thinking in (True, False):
                u, got, _, _ = self.junk_header_run(tools, want, thinking)
                self.assertEqual((calls(got), u.finish), ([{"name": want, "arguments": {}}], "tool_calls"), (want, thinking))

    def test_required_with_one_tool_junk_header(self):
        """One function: the whole header is written for the model (thinking on) or is in the prompt (thinking off)."""
        for thinking in (True, False):
            f = fmt()
            i = f.ids
            junk = f.encode("\u4e00")[0]
            head = self.ids("Think.<|end_message|>") if thinking else []
            body = self.ids('{"name":"get_weather","args":{"city":"Kobe"}}<|end_message|><|content_model_end_sampling|>')
            hdr = f.call_header("get_weather") if thinking else []
            reply = head + hdr + body
            decoys = {len(head) + k: (junk, i[ink.T_END], i[ink.T_STOP])[min(k, 2)] for k in range(len(hdr))}
            eng = FakeEngine(f, lambda p: reply, decoys=decoys)
            kw = {} if thinking else {"template_kwargs": {"enable_thinking": False}}
            u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Kobe"}], tools=clean_tools([WEATHER]),
                                                        tool_choice="required", temperature=0.0, **kw))
            self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {"city": "Kobe"}}], "tool_calls"), thinking)
            if not thinking:
                self.assertEqual(eng.prompts[0][-len(f.call_header("get_weather")):], f.call_header("get_weather"))

    def test_required_header_without_end_of_thinking(self):
        """The thinking ends with <|message_model|> instead of <|end_message|>: the name is still held to the offered ones."""
        u, got, _, _ = self.junk_header_run([WEATHER, SHELL], "get_weather", True, lead="Think.")
        self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {}}], "tool_calls"))
        self.assertEqual(joined(got, "reasoning"), "Think.")

    def test_required_header_inside_a_drafting_pass(self):
        """The thinking ends with <|message_model|> inside a drafting pass whose next ids are junk: the rest of the pass is dropped
        and the name is picked under the mask (the engine state still matches what was emitted)."""
        f = fmt()
        i = f.ids
        junk = f.encode("\u4e00")[0]
        head = self.ids("Think.<|message_model|>")
        body = f.encode("get_weather") + [i[ink.T_CALL]] + self.ids('{"name":"get_weather","args":{}}<|end_message|>'
                                                                    '<|content_model_end_sampling|>')
        g = len(head)
        decoys = {g: junk, g + 1: i[ink.T_END], g + 2: i[ink.T_STOP], g + 3: i[ink.T_STOP]}
        eng = DraftingFakeEngine(f, lambda p: head + body, decoys=decoys, depth=6)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools([WEATHER, SHELL]),
                                                    tool_choice="required", temperature=0.0))
        self.assertGreater(eng.passes, 0)
        self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {}}], "tool_calls"))
        self.assertEqual(eng.fed[eng.n_prompt:], (head + body)[:-1])

    def test_thinking_budget_with_required_among_several(self):
        """The budget closes the thinking; the header is written and the name held to the offered ones."""
        f = fmt()
        i = f.ids
        junk = f.encode("\u4e00")[0]
        think = self.ids("a b c d e f g h")
        n = 4
        body = f.encode("run_command") + [i[ink.T_CALL]] + self.ids('{"name":"run_command","args":{}}<|end_message|>'
                                                                    '<|content_model_end_sampling|>')
        reply = think[:n] + [i[ink.T_END], i[ink.T_MODEL]] + body
        decoys = {n: think[n], n + 1: think[n + 1], n + 2: junk, n + 3: i[ink.T_END], n + 4: i[ink.T_STOP]}
        eng = FakeEngine(f, lambda p: reply, decoys=decoys)
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools([WEATHER, SHELL]),
                                                    tool_choice="required", temperature=0.0, thinking_budget=n))
        self.assertEqual((calls(got), u.finish), ([{"name": "run_command", "arguments": {}}], "tool_calls"))
        self.assertEqual(eng.fed[eng.n_prompt:], reply[:-1])

    def test_a_forced_call_ends_the_reply(self):
        """required (several functions or one) and a named function: the reply ends after the first call that can be read, whatever
        the model would write next (another call, text, thinking).  Thinking on and off."""
        f = fmt()
        one = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>'
        cases = [("Think.<|end_message|>" + one, {}, [WEATHER, SHELL]),
                 (one[len("<|message_model|>"):], {"enable_thinking": False}, [WEATHER, SHELL]),
                 ('{"name":"get_weather","args":{}}<|end_message|>', {"enable_thinking": False}, [WEATHER]),
                 ("Think.<|end_message|>" + one, {}, [WEATHER])]
        for lead, kw, tools in cases:
            for tail in ('<|message_model|>run_command<|content_invoke_tool_json|>{"name":"run_command","args":{}}<|end_message|>'
                         '<|content_model_end_sampling|>',
                         '<|message_model|><|content_text|>51<|end_message|><|content_model_end_sampling|>',
                         '<|message_model|><|content_thinking|>Now answer.<|end_message|><|content_model_end_sampling|>',
                         '<|content_model_end_sampling|>'):
                reply = self.ids(lead + tail)
                eng = FakeEngine(f, lambda p: reply)
                u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools(tools),
                                                            tool_choice="required", temperature=0.0, template_kwargs=kw))
                self.assertEqual((calls(got), joined(got, "content")), ([{"name": "get_weather", "arguments": {}}], ""), (lead, tail))
                self.assertEqual((u.finish, u.tool_calls, u.completion_tokens), ("tool_calls", 1, len(self.ids(lead)) + 1), (lead, tail))
                self.assertEqual(eng.fed[eng.n_prompt:], self.ids(lead))             # the stop token is the last one, not fed

    def test_an_unreadable_call_gets_another_header(self):
        """A call block that is not JSON: another call header of an offered function may follow (here it does), or the end."""
        f = fmt()
        bad = 'run_command<|content_invoke_tool_json|>{"name":"run_command","args":<|end_message|>'
        good = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>'
        for tail, want in ((good + '<|content_model_end_sampling|>', [{"name": "get_weather", "arguments": {}}]),
                           ('<|message_model|><|content_text|>51<|end_message|><|content_model_end_sampling|>', [])):
            reply = self.ids(bad + tail)
            eng = FakeEngine(f, lambda p: reply)
            u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools([WEATHER, SHELL]),
                                                        tool_choice="required", temperature=0.0,
                                                        template_kwargs={"enable_thinking": False}))
            self.assertEqual(calls(got), want, tail)
            self.assertEqual(joined(got, "content"), '{"name":"run_command","args":', tail)   # the unreadable block, as text
            self.assertEqual(u.finish, "tool_calls" if want else "stop", tail)

    def test_forced_header_names_the_call(self):
        """The arguments name another function: the call is that of the header the runner held (named and required)."""
        f = fmt()
        for tools, kw, reply_text in (([WEATHER], {"enable_thinking": False}, '{"name":"run_command","args":{"city":"Nara"}}'),
                                      ([WEATHER, SHELL], {"enable_thinking": False},
                                       'get_weather<|content_invoke_tool_json|>{"name":"other","args":{"city":"Nara"}}')):
            reply = self.ids(reply_text + '<|end_message|><|content_model_end_sampling|>')
            eng = FakeEngine(f, lambda p: reply)
            u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools(tools),
                                                        tool_choice="required", temperature=0.0, template_kwargs=kw))
            self.assertEqual(calls(got), [{"name": "get_weather", "arguments": {"city": "Nara"}}])
        out = []                                                         # tool_choice auto: the JSON name, as before
        s = ink.OutputStream(f, tools=True)
        for t in self.ids('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"other","args":{}}<|end_message|>'):
            out += s.feed(t)
        self.assertEqual(calls(out), [{"name": "other", "arguments": {}}])

    def test_after_a_call_junk_is_held_to_calls_or_the_end(self):
        """After a call the model would write a junk id: only <|message_model|> or the stop token may come (here the stop)."""
        f = fmt()
        junk = f.encode("\u4e00")[0]
        reply = self.ids('Think.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}'
                         '<|end_message|><|content_model_end_sampling|>')
        eng = FakeEngine(f, lambda p: reply, decoys={len(reply) - 1: junk})
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], tools=clean_tools([WEATHER, SHELL]),
                                                    tool_choice="required", temperature=0.0))
        self.assertEqual((calls(got), joined(got, "content"), u.finish), ([{"name": "get_weather", "arguments": {}}], "", "tool_calls"))

    def test_masked_allow(self):
        x = np.arange(6, dtype=np.float32)
        self.assertEqual(list(ink.masked(x, ink.Allow({1, 3}))), [-np.inf, 1.0, -np.inf, 3.0, -np.inf, -np.inf])
        self.assertEqual(list(ink.masked(x, [0, 5])), [-np.inf, 1.0, 2.0, 3.0, 4.0, -np.inf])
        self.assertEqual(list(x), [0, 1, 2, 3, 4, 5])                      # the engine's array is not changed

    def test_thinking_budget_with_tools_leaves_room_for_a_call(self):
        f = fmt()
        i = f.ids
        think = self.ids('<|message_model|><|content_thinking|>a b c d e f g h')
        rest = self.ids('get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Kobe"}}<|end_message|>'
                        '<|content_model_end_sampling|>')
        n = 5                                                            # the budget: the thinking is closed after 5 reply tokens
        reply = think[:n] + [i[ink.T_END], i[ink.T_MODEL]] + rest
        # it would go on thinking at the two positions the budget forces, and would think again after the header (masked)
        eng = FakeEngine(f, lambda p: reply, decoys={n: think[n], n + 1: think[n + 1], n + 2: i[ink.T_THINKING]})
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Kobe"}], tools=clean_tools([WEATHER]),
                                                    temperature=0.0, thinking_budget=n))
        self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {"city": "Kobe"}}], "tool_calls"))
        self.assertEqual((joined(got, "reasoning"), joined(got, "content")), (f.decode(think[2:n]), ""))
        self.assertEqual(eng.fed[eng.n_prompt:], reply[:-1])
        self.assertEqual(eng.fed[eng.n_prompt:eng.n_prompt + n + 2], think[:n] + [i[ink.T_END], i[ink.T_MODEL]])

    def test_sampled_request_respects_the_mask(self):
        f = fmt()
        reply = self.ids('get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Kobe"}}<|end_message|>'
                         '<|content_model_end_sampling|>')
        junk = f.encode("\u4e00")[0]
        eng = FakeEngine(f, lambda p: reply, decoys={0: f.ids[ink.T_TEXT], 1: junk, 2: f.ids[ink.T_END], 3: f.ids[ink.T_STOP]})
        eng_logit = eng._logits

        def sharp():                                                     # sampling at temperature 1 picks the best token here
            x = eng_logit()
            return x * 10.0
        eng._logits = sharp
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "Kobe"}], tools=clean_tools([WEATHER, SHELL]),
                                                    tool_choice="required", temperature=1.0, top_p=1.0, top_k=0, seed=7,
                                                    template_kwargs={"enable_thinking": False}))
        self.assertEqual((calls(got), u.finish), ([{"name": "get_weather", "arguments": {"city": "Kobe"}}], "tool_calls"))
        self.assertEqual((joined(got, "reasoning"), joined(got, "content")), ("", ""))

    def test_without_tools_nothing_changes(self):
        f = fmt()
        call = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>'
        eng = FakeEngine(f, lambda p: self.ids(call + '<|content_model_end_sampling|>'))
        u, got = run(runner_for(eng, f), ChatParams(messages=[{"role": "user", "content": "x"}], temperature=0.0))
        self.assertEqual((calls(got), u.finish, u.tool_calls), ([], "stop", 0))
        self.assertEqual(joined(got, "content"), '{"name":"get_weather","args":{}}')


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class ServerTests(unittest.TestCase):
    """yasakani serve's HTTP layer for an Inkling-2 folder (as inkling2_cli.cmd_serve configures it) on the stand-in engine."""

    @classmethod
    def setUpClass(cls):
        from yasakani.server import ApiState, ServerConfig, bind
        f = fmt()
        call = f.encode('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"京都"}}'
                        '<|end_message|><|content_model_end_sampling|>')
        answer = f.encode('<|message_model|><|content_text|>晴れです。<|end_message|><|content_model_end_sampling|>')
        tool_turn = f.encode("<|message_tool|>")

        def reply(prompt):
            if prompt[-len(f.call_header("get_weather")):] == f.call_header("get_weather"):
                return f.encode('{"name":"get_weather","args":{"city":"Nara"}}<|end_message|><|content_model_end_sampling|>')
            return answer if any(prompt[k:k + len(tool_turn)] == tool_turn for k in range(len(prompt))) else call
        cls.eng = FakeEngine(f, reply)
        cfg = ServerConfig(port=0, model_id="inkling-2-mini-quality")
        cfg.sampling_defaults = dict(temperature=1.0, top_p=1.0, top_k=0)
        cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
        cfg.keep_tool_ids = True
        cls.state = ApiState(cfg)
        cls.httpd = bind(cls.state)
        cfg.port = cls.httpd.server_address[1]
        cls.state.runner = runner_for(cls.eng, f)
        threading.Thread(target=cls.httpd.serve_forever, kwargs=dict(poll_interval=0.1), daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.state.cfg.port, timeout=60)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, data

    def test_call_result_answer(self):
        msgs = [{"role": "user", "content": "京都の天気は？"}]
        st, data = self.post({"messages": msgs, "tools": [WEATHER], "temperature": 0})
        self.assertEqual(st, 200, data)
        d = json.loads(data)
        ch = d["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertIsNone(ch["message"]["content"])
        tc = ch["message"]["tool_calls"]
        self.assertEqual((len(tc), tc[0]["type"], tc[0]["function"]["name"]), (1, "function", "get_weather"))
        self.assertEqual(json.loads(tc[0]["function"]["arguments"]), {"city": "京都"})
        self.assertTrue(tc[0]["id"].startswith("call_"))
        msgs2 = msgs + [{"role": "assistant", "content": None, "tool_calls": tc},
                        {"role": "tool", "tool_call_id": tc[0]["id"], "content": "{\"weather\": \"晴れ\"}"}]
        st, data = self.post({"messages": msgs2, "tools": [WEATHER], "temperature": 0})
        self.assertEqual(st, 200, data)
        ch = json.loads(data)["choices"][0]
        self.assertEqual((ch["finish_reason"], ch["message"]["content"]), ("stop", "晴れです。"))
        text = fmt().decode(self.eng.prompts[-1])
        self.assertIn('<|message_tool|>get_weather<|content_text|>{"weather": "晴れ"}<|end_message|>', text)   # name via tool_call_id
        self.assertIn('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"京都"}}', text)

    def test_stream(self):
        st, data = self.post({"messages": [{"role": "user", "content": "京都？"}], "tools": [WEATHER], "stream": True,
                              "stream_options": {"include_usage": True}, "temperature": 0})
        self.assertEqual(st, 200)
        events = [json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: {")]
        deltas = [c["delta"] for e in events for c in e.get("choices", [])]
        tcs = [t for d in deltas for t in d.get("tool_calls", [])]
        self.assertEqual(len(tcs), 1)
        self.assertEqual((tcs[0]["index"], tcs[0]["function"]["name"], json.loads(tcs[0]["function"]["arguments"])),
                         (0, "get_weather", {"city": "京都"}))
        self.assertIn("tool_calls", [c.get("finish_reason") for e in events for c in e.get("choices", [])])
        self.assertTrue(data.rstrip().endswith("data: [DONE]"))

    def test_tool_choice(self):
        st, data = self.post({"messages": [{"role": "user", "content": "Nara"}], "tools": [WEATHER, SHELL], "temperature": 0,
                              "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "enable_thinking": False})
        self.assertEqual(st, 200, data)
        ch = json.loads(data)["choices"][0]
        self.assertEqual((ch["finish_reason"], ch["message"]["tool_calls"][0]["function"]["name"]), ("tool_calls", "get_weather"))
        self.assertNotIn("run_command", fmt().decode(self.eng.prompts[-1]))     # only the named function is offered
        st, data = self.post({"messages": [{"role": "user", "content": "x"}], "tools": [WEATHER], "tool_choice": "none",
                              "temperature": 0})
        self.assertEqual(st, 200, data)
        self.assertNotIn("tool_declare", fmt().decode(self.eng.prompts[-1]))
        ch = json.loads(data)["choices"][0]
        self.assertEqual((ch["finish_reason"], "tool_calls" in ch["message"]), ("stop", False))


class RollbackTests(unittest.TestCase):
    """The engine's rollback option (no engine library: a stand-in for the binding)."""

    class Native:
        def __init__(self, rollback):
            self.rollback, self.calls = rollback, []

        def load(self, root, backend="host", max_positions=4096, dll=None, weights=None, log=None, rollback=None):
            pass

        def reset(self):
            self.calls.append("reset")

        def truncate(self, n):
            self.calls.append(("truncate", n))

        def feed_many(self, ids):
            self.calls.append(("feed", len(ids)))

        def step(self, tok):
            return np.zeros(4, np.float32)

    def engine(self, rollback, max_ctx=32768):
        e = ink.Ink2Engine.__new__(ink.Ink2Engine)
        e.max_ctx, e.mod = max_ctx, SimpleNamespace(Ink2Native=self.Native)
        e.native, e.fed, e.last_logits = self.Native(rollback), [], None
        return e

    def test_option(self):
        e = self.engine(64)
        old = os.environ.pop(ink.ROLLBACK_ENV, None)
        try:
            self.assertEqual(e._rollback_kw(), {"rollback": ink.DEFAULT_ROLLBACK})
            self.assertEqual(self.engine(64, max_ctx=256)._rollback_kw(), {"rollback": 256})
            os.environ[ink.ROLLBACK_ENV] = "0"
            self.assertEqual(e._rollback_kw(), {})                             # the engine's own default
            os.environ[ink.ROLLBACK_ENV] = "4096"
            self.assertEqual(e._rollback_kw(), {"rollback": 4096})
            for bad in ("x", "-1", "40000"):
                os.environ[ink.ROLLBACK_ENV] = bad
                with self.assertRaises(Exception):
                    e._rollback_kw()
        finally:
            os.environ.pop(ink.ROLLBACK_ENV, None)
            if old is not None:
                os.environ[ink.ROLLBACK_ENV] = old

    def test_a_dropped_reply_keeps_the_prompt(self):
        """The next request leaves out the last reply's thinking: the reply is dropped, the conversation before it is not read again."""
        prompt, reply = list(range(1000)), list(range(5000, 5300))
        for rollback, want in ((1024, [("truncate", 1001), ("feed", 49)]), (64, ["reset", ("feed", 1050)])):
            e = self.engine(rollback)
            e.fed = prompt + reply
            _, cached = e.read_prompt(prompt + [5000] + list(range(9000, 9050)))
            self.assertEqual((e.native.calls, cached), (want, 1001 if rollback == 1024 else 0))


class ParseRequestTests(unittest.TestCase):
    """parse_chat_request: tool_choice and the kept ids (no tokenizer needed)."""

    def test_fields(self):
        from yasakani.server import ServerConfig, parse_chat_request
        cfg = ServerConfig()
        body = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "r"}], "tools": [WEATHER, SHELL]}
        p = parse_chat_request(dict(body), cfg)
        self.assertIsNone(p.tool_choice)
        self.assertNotIn("id", p.messages[1]["tool_calls"][0])                # Qwen3.8 (the default): as before
        self.assertEqual(set(p.messages[2]), {"role", "content"})
        self.assertEqual(parse_chat_request(dict(body, tool_choice="required"), cfg).tool_choice, "required")
        p = parse_chat_request(dict(body, tool_choice={"type": "function", "function": {"name": "run_command"}}), cfg)
        self.assertEqual((p.tool_choice, [t["function"]["name"] for t in p.tools]), ("required", ["run_command"]))
        self.assertIsNone(parse_chat_request(dict(body, tool_choice="none"), cfg).tools)
        cfg.keep_tool_ids = True
        p = parse_chat_request(dict(body), cfg)
        self.assertEqual(p.messages[1]["tool_calls"][0]["id"], "c1")
        self.assertEqual((p.messages[2]["tool_call_id"], p.messages[2]["name"]), ("c1", "get_weather"))


if __name__ == "__main__":
    unittest.main()
