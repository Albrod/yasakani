# SPDX-License-Identifier: Apache-2.0
"""The server with --parallel N (the scheduler, yasakani/sched.py) against the stub library: requests sent at once get what the same
requests get one at a time from a server without --parallel; a full queue is answered 503 with Retry-After; a client that goes away
(waiting, running without a stream, or streaming) cancels its request; /health and the timings carry the scheduler's fields; an engine
failure ends the requests with 500 and /health says so; without --parallel the server serves as before.  Also: the context of N
requests at once (yasakani/gpu.py), the start-up check (sched.consistency_check) and the command line (serve, selfcheck, bench)."""
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from tests import support
from yasakani import gpu as G
from yasakani._native import YasakaniError
from yasakani.engine import GenResult, MsInfo
from yasakani.runtime import EOS_IDS

ROOT = Path(__file__).resolve().parent.parent
HAVE_TOK = support.tokenizer_dir() is not None
IS_POSIX = os.name == "posix"
FASTEST = (10311344128, 1164969472, 1411010584)       # the published files: weights, draft model, its output head
QUALITY = (12365242368, 1164969472, 1411010584)
MIB = 1 << 20
ENGINE_SLOT_BYTES = 195887104                         # what yskn_ms_info says of one slot of the published models (besides its KV)
NO_WAIT_BOUND = 600.0                                 # max_wait in the tests of the keep itself: a waiting request never takes it


def wait_for(cond, timeout=15.0, what="a condition"):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for " + what)


def sse(raw: bytes):
    """A stream's events -> (role chunk seen, reasoning, content, finish, usage, timings, errors)."""
    events = [e[len("data: "):] for e in raw.decode("utf-8").split("\n\n") if e.startswith("data: ")]
    objs = [json.loads(e) for e in events if e != "[DONE]"]
    out = dict(role=any(o.get("choices") and o["choices"][0]["delta"].get("role") == "assistant" for o in objs),
               reasoning="".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o.get("choices")),
               content="".join(o["choices"][0]["delta"].get("content", "") for o in objs if o.get("choices")),
               finish=next((o["choices"][0]["finish_reason"] for o in objs if o.get("choices") and o["choices"][0]["finish_reason"]), None),
               usage=next((o["usage"] for o in objs if "usage" in o), None),
               timings=next((o["timings"] for o in objs if "timings" in o), None),
               errors=[o["error"] for o in objs if "error" in o], done=bool(events) and events[-1] == "[DONE]")
    return out


class Server:
    """A server in this process on one stub engine: with a scheduler (``parallel`` >= 2) or one request at a time."""

    def __init__(self, engine, fmt, parallel=1, max_queue=64, seed0=1, **cfg_kw):
        from yasakani.chat import ChatRunner
        from yasakani.sched import Scheduler
        from yasakani.server import ApiState, ServerConfig, bind
        self.engine = engine
        self.state = ApiState(ServerConfig(port=0, **cfg_kw))
        self.state.runner = ChatRunner(engine, fmt, seed0=seed0)
        self.sched = None
        if parallel >= 2:
            self.sched = Scheduler(engine, parallel, max_queue=max_queue)
            self.state.sched = self.sched
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs=dict(poll_interval=0.05), daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.sched is not None:
            self.sched.close()

    def request(self, method, path, body=None, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            h = {"Host": "127.0.0.1:%d" % self.port}
            data = None
            if body is not None:
                h["Content-Type"] = "application/json"
                data = json.dumps(body)
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def chat(self, body, timeout=30):
        status, headers, data = self.request("POST", "/v1/chat/completions", body, timeout)
        if body.get("stream") and status == 200:
            return status, headers, sse(data)
        return status, headers, json.loads(data)

    def health(self):
        status, _, data = self.request("GET", "/health")
        return status, json.loads(data)

    def raw(self, body):
        """Send a request on a socket of its own and return the socket (the test reads or closes it)."""
        b = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % (self.port, len(b)) + b)
        return s


def answer(r):
    """What a request's answer says, stream or not: (reasoning, content, finish, usage)."""
    if "choices" in r and r.get("object") == "chat.completion":
        m = r["choices"][0]["message"]
        return m.get("reasoning_content") or "", m.get("content") or "", r["choices"][0]["finish_reason"], r["usage"]
    return r["reasoning"], r["content"], r["finish"], r["usage"]


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ParallelServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from yasakani.textio import ChatFormatter
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.stub.clear_scripts()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        self.ref = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000, name="ref")
        self.model(eos_every=40)
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()
        self.stub.delay_us(0)
        self.e.close()
        self.ref.close()
        self.tmp.cleanup()

    def model(self, eos_every=40):
        for e in (self.e, self.ref):
            self.stub.model(e, True, eos_every=eos_every, vocab=64)

    def server(self, engine, parallel=1, **kw):
        s = Server(engine, self.fmt, parallel, **kw)
        self.servers.append(s)
        return s

    REQUESTS = [
        dict(messages=[{"role": "user", "content": "one"}], max_tokens=60, temperature=0, chat_template_kwargs={"enable_thinking": False}),
        dict(messages=[{"role": "user", "content": "two"}], max_tokens=80, seed=3, stream=True, stream_options={"include_usage": True},
             chat_template_kwargs={"enable_thinking": False}),
        dict(messages=[{"role": "user", "content": "three"}], max_tokens=70, seed=4, thinking_budget=12),
        dict(messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "four"}], max_tokens=1, seed=5,
             chat_template_kwargs={"enable_thinking": False}),
        dict(messages=[{"role": "user", "content": "five"}], max_tokens=50, seed=6, ignore_eos=True, stream=True,
             stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False}),
        dict(messages=[{"role": "user", "content": "six"}], max_tokens=90, seed=7, temperature=0.7, top_k=5, min_p=0.05,
             chat_template_kwargs={"enable_thinking": False}),
    ]

    # ---- the same answers ---------------------------------------------------------------------------------------------------------
    def test_at_once_equals_one_at_a_time(self):
        """Requests sent at once to --parallel 3 (more requests than slots: some wait) get the text, finish and token counts the same
        requests get one at a time from a server without --parallel."""
        self.stub.delay_us(300)
        par = self.server(self.e, parallel=3)
        one = self.server(self.ref)
        got = [None] * len(self.REQUESTS)

        def send(i):
            got[i] = par.chat(dict(self.REQUESTS[i]))
        ths = [threading.Thread(target=send, args=(i,), daemon=True) for i in range(len(self.REQUESTS))]
        for t in ths:
            t.start()
        for t in ths:
            t.join(60)
        want = [one.chat(dict(b)) for b in self.REQUESTS]
        for i, (g, w) in enumerate(zip(got, want)):
            with self.subTest(request=i):
                self.assertEqual((g[0], w[0]), (200, 200))
                ga, wa = answer(g[2]), answer(w[2])
                self.assertEqual(ga[:3], wa[:3])
                for f in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.assertEqual(ga[3][f], wa[3][f], f)
                self.assertEqual(ga[3]["completion_tokens_details"], wa[3]["completion_tokens_details"])
                self.assertEqual(ga[3].get("loop_guard"), wa[3].get("loop_guard"))
        self.assertGreater(par.sched.perf["passes"], 0)                       # the slots were used

    def test_stop_string_up_to_the_stop(self):
        par = self.server(self.e, parallel=2)
        one = self.server(self.ref)
        base = dict(messages=[{"role": "user", "content": "say something"}], max_tokens=60, temperature=0,
                    chat_template_kwargs={"enable_thinking": False})
        text = answer(one.chat(dict(base))[2])[1]
        k = next(k for k in range(len(text) // 2, len(text) - 3) if text[k:k + 3].strip() == text[k:k + 3])
        body = dict(base, stop=[text[k:k + 3]])
        self.stub.delay_us(300)
        other = threading.Thread(target=par.chat, args=(dict(self.REQUESTS[5], max_tokens=200),), daemon=True)
        other.start()
        g = answer(par.chat(dict(body))[2])
        other.join(30)
        w = answer(one.chat(dict(body))[2])
        self.assertEqual((g[1], g[2]), (w[1], w[2]))
        self.assertEqual((g[1], g[2]), (text[:text.find(text[k:k + 3])], "stop"))

    def test_tool_calls_keep_the_stream_for_the_next_turn(self):
        """An answer with tool calls (stream or not) closes its request with next_turn: the client sends the results at once, and that
        next turn continues the stream, kept as long as the scheduler keeps a conversation of that length (Scheduler.turn_keep_s).
        Other answers close without a keep."""
        from yasakani import sched as S
        par = self.server(self.e, parallel=2)
        seen = []
        orig = S.ScheduledEngine.close

        def rec(se, keep_s=0.0, next_turn=False):
            orig(se, keep_s, next_turn=next_turn)
            last = se.results[-1]
            seen.append((next_turn, se._keep_s, par.sched.turn_keep_s(last.prompt_tokens + len(last.ids))))
        tools = [dict(type="function", function=dict(name="get_weather", description="The weather of a city.",
                                                     parameters=dict(type="object", properties=dict(city=dict(type="string")),
                                                                     required=["city"])))]
        call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n</tool_call>"
        base = dict(messages=[{"role": "user", "content": "weather in Osaka?"}], max_tokens=200, temperature=0,
                    chat_template_kwargs={"enable_thinking": False})
        with mock.patch.object(S.ScheduledEngine, "close", rec):
            for stream in (False, True):
                self.stub.push(self.fmt.encode(call).tolist() + [EOS_IDS[0]])
                st, _, r = par.chat(dict(base, tools=tools, stream=stream))
                self.assertEqual((st, answer(r)[2]), (200, "tool_calls"))
            st, _, r = par.chat(dict(base, max_tokens=20))
            self.assertEqual(st, 200)
            self.assertNotEqual(answer(r)[2], "tool_calls")
        self.assertEqual([x[0] for x in seen], [True, True, False])
        self.assertEqual([x[1] for x in seen], [seen[0][2], seen[1][2], 0.0])
        self.assertEqual(seen[0][2], S.KEEP_TURN_MIN_S)                        # a short conversation: the shortest keep

    def answer_at_once(self, srv, body):
        """Send a request on a socket of its own and return its answer as soon as it is complete (a stream: at "data: [DONE]"; else:
        its Content-Length), without waiting for the server to close the connection: a client that sends its next turn at once."""
        s = srv.raw(body)
        got = b""
        try:
            while True:
                b = s.recv(65536)
                if not b:
                    break
                got += b
                head, sep, rest = got.partition(b"\r\n\r\n")
                if not sep:
                    continue
                if body.get("stream"):
                    if b"data: [DONE]\n\n" in rest:
                        return sse(rest)
                else:
                    m = re.search(rb"(?i)content-length:\s*(\d+)", head)
                    if m and len(rest) >= int(m.group(1)):
                        return json.loads(rest[:int(m.group(1))])
        finally:
            s.close()
        raise AssertionError("the connection closed before the answer was complete: %r" % got[:300])

    def test_the_next_turn_right_after_tool_calls_continues_the_stream(self):
        """The server closes a request's view of the scheduler (with keep_s after tool calls) before the end of the answer goes out: a
        client that sends its next turn the moment it has the answer finds the stream kept for it, and the turn continues it (the
        snap_mode and the answer of the single path), also when the request's thread is slow to go on after the answer (a delay before
        the close: with the answer written first, the next turn came while the request was still open and was read from the start)."""
        from yasakani import sched as S
        one = self.server(self.ref)
        orig = S.ScheduledEngine.close

        def late_close(se, keep_s=0.0, next_turn=False):
            time.sleep(0.6)                                       # the request's thread goes on late
            return orig(se, keep_s, next_turn=next_turn)
        tools = [dict(type="function", function=dict(name="get_weather", description="The weather of a city.",
                                                     parameters=dict(type="object", properties=dict(city=dict(type="string")),
                                                                     required=["city"])))]
        call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n</tool_call>"
        base = dict(tools=tools, max_tokens=60, temperature=0, chat_template_kwargs={"enable_thinking": False})
        for stream in (False, True):
            with self.subTest(stream=stream):
                e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000, name="par%d" % stream)
                self.addCleanup(e.close)
                self.stub.model(e, True, eos_every=40, vocab=64)
                par = self.server(e, parallel=2)                 # a scheduler of its own: nothing kept from another subtest
                msgs = [{"role": "user", "content": "weather in Osaka? (%s)" % ("stream" if stream else "plain")}]
                got = []
                for srv in (par, one):
                    self.stub.push(self.fmt.encode(call).tolist() + [EOS_IDS[0]])
                    with mock.patch.object(S.ScheduledEngine, "close", late_close):
                        r1 = self.answer_at_once(srv, dict(base, messages=msgs, stream=stream))
                        if stream:
                            calls = r1.get("tool_calls") or []
                            finish = r1["finish"]
                        else:
                            calls = r1["choices"][0]["message"].get("tool_calls") or []
                            finish = r1["choices"][0]["finish_reason"]
                        self.assertEqual(finish, "tool_calls")
                        nxt = msgs + [{"role": "assistant", "content": None, "tool_calls": [
                                           {"id": "call_0", "type": "function", "function": {"name": "get_weather",
                                                                                            "arguments": json.dumps({"city": "Osaka"})}}]},
                                       {"role": "tool", "tool_call_id": "call_0", "content": "sunny, 21 C"}]
                        st, _, r2 = srv.chat(dict(base, messages=nxt))
                    self.assertEqual(st, 200)
                    got.append(r2)
                g, w = got
                self.assertEqual(g["timings"]["snap_mode"], 1)    # the next turn continued the stream kept for it
                self.assertEqual(answer(g)[:3], answer(w)[:3])
                self.assertEqual(g["usage"]["completion_tokens"], w["usage"]["completion_tokens"])

    # ---- the queue ---------------------------------------------------------------------------------------------------------------
    def test_queue_full_is_503_and_a_waiting_client_that_leaves_is_cancelled(self):
        self.model(eos_every=0)
        self.stub.delay_us(2000)
        par = self.server(self.e, parallel=2, max_queue=1)
        long = dict(messages=[{"role": "user", "content": "go on"}], max_tokens=100000, stream=True, loop_guard=False,
                    chat_template_kwargs={"enable_thinking": False})
        running = [par.raw(dict(long, seed=i)) for i in range(2)]
        wait_for(lambda: par.sched.status()["slots_busy"] == 2, what="two running requests")
        waiting = par.raw(dict(long, stream=False, seed=9))
        wait_for(lambda: par.sched.status()["queued"] == 1, what="a waiting request")
        status, headers, d = par.chat(dict(long, stream=False, max_tokens=5))
        self.assertEqual(status, 503)
        self.assertEqual(headers.get("Retry-After"), "2")
        self.assertEqual(d["error"]["type"], "unavailable")
        self.assertIn("waiting", d["error"]["message"])
        status, headers, raw = par.request("POST", "/v1/chat/completions", dict(long, max_tokens=5))     # a stream: no header sent yet
        self.assertEqual((status, headers.get("Retry-After")), (503, "2"))
        self.assertEqual(json.loads(raw)["error"]["type"], "unavailable")
        waiting.close()                                                       # the waiting client goes away: it leaves the queue
        wait_for(lambda: par.sched.status()["queued"] == 0, timeout=5, what="the waiting request to be cancelled")
        self.assertEqual(par.sched.status()["slots_busy"], 2)
        for s in running:                                                     # streams that have started find out by writing
            s.close()
        wait_for(lambda: par.sched.status()["slots_busy"] == 0, timeout=10, what="the streams to be cancelled")
        status, _, d = par.chat(dict(long, stream=False, max_tokens=5))
        self.assertEqual((status, d["usage"]["completion_tokens"]), (200, 5))

    def test_a_client_that_leaves_a_running_request_without_a_stream_cancels_it(self):
        self.model(eos_every=0)
        self.stub.delay_us(2000)
        par = self.server(self.e, parallel=2)
        s = par.raw(dict(messages=[{"role": "user", "content": "go on"}], max_tokens=100000, loop_guard=False,
                         chat_template_kwargs={"enable_thinking": False}))
        wait_for(lambda: par.sched.status()["slots_busy"] == 1, what="the request to run")
        time.sleep(0.2)
        self.assertEqual(par.sched.status()["slots_busy"], 1)
        s.close()
        wait_for(lambda: par.sched.status()["slots_busy"] == 0, timeout=5, what="the request to be cancelled")

    def test_a_stream_whose_client_leaves_is_cancelled(self):
        self.model(eos_every=0)
        self.stub.delay_us(1000)
        par = self.server(self.e, parallel=2)
        long = dict(messages=[{"role": "user", "content": "go on"}], max_tokens=100000, stream=True, loop_guard=False,
                    chat_template_kwargs={"enable_thinking": False})
        a, b = par.raw(dict(long, seed=1)), par.raw(dict(long, seed=2))
        got = b""
        while got.count(b"data: ") < 4:
            got += a.recv(4096)
        self.assertIn(b"200 OK", got)
        wait_for(lambda: par.sched.status()["slots_busy"] == 2, what="two streams")
        a.close()
        wait_for(lambda: par.sched.status()["slots_busy"] == 1, timeout=5, what="the stream to be cancelled")
        b.close()
        wait_for(lambda: par.sched.status()["slots_busy"] == 0, timeout=5, what="the other stream to be cancelled")

    # ---- what the server reports ---------------------------------------------------------------------------------------------------
    def test_health_and_timings(self):
        par = self.server(self.e, parallel=3)
        status, h = par.health()
        self.assertEqual(status, 200)
        self.assertEqual({k: h[k] for k in ("status", "parallel", "slots_busy", "slots_idle", "queued", "max_context_tokens")},
                         dict(status="ok", parallel=3, slots_busy=0, slots_idle=0, queued=0, max_context_tokens=20000))
        status, _, d = par.chat(dict(self.REQUESTS[0]))
        t = d["timings"]
        self.assertEqual(t["parallel_avg"], 1.0)                                # alone: on the carrier, no shared pass
        self.assertGreaterEqual(t["queue_ms"], 0.0)
        self.assertEqual(t["snap_mode"], 0)
        self.assertEqual(t["predicted_n"], d["usage"]["completion_tokens"])
        status, h = par.health()
        self.assertEqual((h["slots_busy"], h["queued"]), (0, 0))
        one = self.server(self.ref)                                            # without --parallel: as before
        status, h = one.health()
        self.assertEqual(set(h), {"status", "version", "model", "max_context_tokens", "thinking_budget"})
        status, _, d = one.chat(dict(self.REQUESTS[0]))
        self.assertFalse({"queue_ms", "parallel_avg", "snap_mode"} & set(d["timings"]))

    def test_streams_that_share_passes_say_so(self):
        self.stub.delay_us(500)
        par = self.server(self.e, parallel=2)
        body = dict(messages=[{"role": "user", "content": "x"}], max_tokens=120, ignore_eos=True, seed=1, stream=True,
                    stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
        out = [None, None]

        def send(i):
            out[i] = par.chat(dict(body, seed=i + 1, messages=[{"role": "user", "content": "x%d" % i}]))
        ths = [threading.Thread(target=send, args=(i,)) for i in range(2)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(30)
        avgs = [o[2]["timings"]["parallel_avg"] for o in out]
        self.assertTrue(any(a > 1.0 for a in avgs), avgs)
        self.assertTrue(all(o[2]["done"] and o[2]["role"] for o in out))

    def test_engine_failure(self):
        self.model(eos_every=0)
        self.stub.delay_us(300)
        par = self.server(self.e, parallel=2)
        self.stub.ms_fail_in(self.e, 5)
        body = dict(messages=[{"role": "user", "content": "go"}], max_tokens=4000, chat_template_kwargs={"enable_thinking": False})
        out = [None, None]

        def send(i):
            out[i] = par.chat(dict(body, seed=i))
        ths = [threading.Thread(target=send, args=(i,)) for i in range(2)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(30)
        for status, _, d in out:
            self.assertEqual(status, 500)
            self.assertIn("engine failed", d["error"]["message"])
        status, h = par.health()
        self.assertEqual((status, h["status"]), (503, "error"))
        self.assertIn("injected failure", h["error"])
        status, _, d = par.chat(dict(body, max_tokens=4))
        self.assertEqual(status, 500)

    # ---- one request at a time stays as it was -------------------------------------------------------------------------------------
    def test_without_parallel_the_generation_lock_serves(self):
        """Without a scheduler a request waits for the generation lock (one at a time); with one, it does not use the lock."""
        one = self.server(self.ref)
        par = self.server(self.e, parallel=2)
        body = dict(self.REQUESTS[0])
        for srv, blocked in ((one, True), (par, False)):
            res = {}
            with srv.state.gen_lock:
                th = threading.Thread(target=lambda: res.setdefault("r", srv.chat(dict(body))), daemon=True)
                th.start()
                th.join(1.0)
                self.assertEqual(th.is_alive(), blocked)
            th.join(20)
            self.assertEqual(res["r"][0], 200)

    def test_a_request_that_does_not_fit_is_a_400_before_it_waits(self):
        par = self.server(self.e, parallel=2)
        status, _, d = par.chat(dict(messages=[{"role": "user", "content": "word " * 30000}], max_tokens=10))
        self.assertEqual(status, 400)
        self.assertEqual(d["error"]["code"], "context_length_exceeded")
        self.assertEqual(par.sched.perf["reads"], 0)

    # ---- image input goes through the request's engine ----------------------------------------------------------------------------
    def test_image_request_runs_through_the_scheduler(self):
        """ChatRunner.run(eng=...) sets the image input through that engine (the scheduler then gives the request the carrier to
        itself) and never through the runner's own engine; the text is the one the engine gives alone."""
        from yasakani.chat import ChatParams, ChatRunner
        from yasakani.sched import ScheduledEngine, Scheduler
        from yasakani.vision import PreparedVisual
        for e in (self.e, self.ref):
            self.stub.give_visual(e)
        trace = []
        sched = Scheduler(self.e, 2, trace=trace)
        self.addCleanup(sched.close)
        n_prompt, n_img = 12, 4
        prompt = np.arange(100, 100 + n_prompt, dtype=np.int32)
        vis = dict(features=np.ones((n_img, 5120), np.uint16), slots=np.arange(2, 2 + n_img, dtype=np.int32),
                   positions=np.tile(np.arange(n_prompt, dtype=np.int32), (3, 1)), delta=0)
        p = ChatParams(messages=[{"role": "user", "content": "look"}], max_tokens=30, seed=4)
        runner = ChatRunner(self.e, self.fmt)
        calls = []
        orig = self.e.visual_input

        def visual_input(**kw):                                # from another thread it would hold the engine while the scheduler
            calls.append(threading.current_thread().name)      # waits for it: fail at once instead
            if threading.current_thread().name != "yasakani-scheduler":
                raise AssertionError("the image input was set from the request's thread")
            return orig(**kw)
        self.e.visual_input = visual_input
        try:
            ev = []
            with ScheduledEngine(sched) as se:
                u = runner.run(p, lambda k, t: ev.append((k, t)), PreparedVisual(prompt, False, vis), eng=se)
        finally:
            del self.e.visual_input
        self.assertEqual(calls, ["yasakani-scheduler"])                        # set on the scheduler's thread, not the request's
        self.assertEqual(self.stub.visual(self.e), (0, 2))
        self.assertIn("visual", [t[0] for t in trace])
        ev2 = []
        u2 = ChatRunner(self.ref, self.fmt).run(p, lambda k, t: ev2.append((k, t)), PreparedVisual(prompt, False, vis))
        self.assertEqual(ev, ev2)
        self.assertEqual((u.completion_tokens, u.finish), (u2.completion_tokens, u2.finish))


class QueueLimitTests(unittest.TestCase):
    """max_queue counts the requests that wait beyond the slots a request can take: requests that come at once all find a place while
    there are free slots, and one more than the free slots and the queue together is refused."""

    def test_requests_that_come_at_once_take_the_free_slots(self):
        from yasakani.engine import Sampling
        from yasakani.sched import QueueFull, Scheduler, Stage
        stub = support.Stub()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        stub.clear_scripts()
        e = support.open_engine(Path(tmp.name), stub, max_ctx=8192)
        self.addCleanup(e.close)
        stub.model(e, True, eos_every=0, vocab=64)
        stub.delay_us(1000)
        self.addCleanup(stub.delay_us, 0)
        greedy = Sampling(temperature=0.0)
        for trial in range(5):
            s = Scheduler(e, 3, max_queue=1)
            try:
                ts = [s.submit(Stage([10 + i, 2, 3 + trial], 100000, EOS_IDS, greedy)) for i in range(4)]    # 3 slots + 1 waiting
                with self.assertRaises(QueueFull):
                    s.submit(Stage([20, 2], 10, EOS_IDS, greedy))
                wait_for(lambda: s.status()["slots_busy"] == 3 and s.status()["queued"] == 1, what="three running, one waiting")
                with self.assertRaises(QueueFull):
                    s.submit(Stage([21, 2], 10, EOS_IDS, greedy))
                for t in ts:
                    t.cancel()
                for t in ts:
                    kind, r = t.q.get(timeout=30)
                    while kind == "ids":
                        kind, r = t.q.get(timeout=30)
                    self.assertEqual((kind, r.finish), ("done", "cancelled"))
            finally:
                s.close()


class SlotReserveTests(unittest.TestCase):
    """The slot of a finished stream stays with its request while the request is open (its next stage continues the stream there, read
    as alone), for at most sched.RESERVE_S; a new request waits for another slot meanwhile."""

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    def setUp(self):
        from yasakani.engine import Sampling
        self.greedy = Sampling(temperature=0.0)
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(1000)
        self.stub.clear_scripts()
        self.e = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=8192)
        self.ref = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=8192, name="ref")
        for e in (self.e, self.ref):
            self.stub.model(e, True, eos_every=0, vocab=64)

    def tearDown(self):
        self.stub.delay_us(0)
        self.e.close()
        self.ref.close()
        self.tmp.cleanup()

    def background(self, sched, prompt):
        from yasakani.sched import ScheduledEngine
        se = ScheduledEngine(sched)
        se.set_sampling(self.greedy)
        th = threading.Thread(target=lambda: se.generate(prompt, 100000), daemon=True)
        th.start()
        return se, th

    def test_next_stage_keeps_its_slot(self):
        from yasakani.sched import ScheduledEngine, Scheduler
        s = Scheduler(self.e, 2)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 5)                                  # in a slot next to the first request; then its slot is kept
        b, tb = self.background(s, [50, 51])
        time.sleep(0.5)
        self.assertEqual((s.status()["queued"], s.status()["slots_busy"]), (1, 1))     # the new request waits
        r2 = a.generate(p + r1.ids, 5)                         # the next stage continues the stream in its slot
        a.close()
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids, 5)
        self.assertEqual((r1.ids, r2.ids), (w1.ids, w2.ids))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(r2.snap_mode, 1)
        wait_for(lambda: s.status()["slots_busy"] == 2 and s.status()["queued"] == 0, what="the new request to run")
        x.cancel()
        b.cancel()
        tx.join(10)
        tb.join(10)

    def test_the_slot_is_given_up_after_a_while(self):
        from yasakani import sched as S
        s = S.Scheduler(self.e, 2)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)                               # open, and it sends nothing more
        a.set_sampling(self.greedy)
        a.generate([40, 41, 42], 5)
        t0 = time.monotonic()
        b, tb = self.background(s, [50, 51])
        time.sleep(0.5)
        self.assertEqual(s.status()["queued"], 1)
        wait_for(lambda: s.status()["slots_busy"] == 2, timeout=S.RESERVE_S + 3, what="the new request to take the slot")
        self.assertGreater(time.monotonic() - t0, S.RESERVE_S - 0.6)
        a.close()
        x.cancel()
        b.cancel()
        tx.join(10)
        tb.join(10)

    def test_a_slow_client_keeps_its_slot(self):
        """RESERVE_S counts from the moment the request's thread has taken the end of its stage, not from the end of the stream: a
        client that reads the last ids of a stage late still continues its stream in its slot (read as alone); a new request waits."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        s = S.Scheduler(self.e, 2)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        late = []

        def slow(ids):                                         # the client takes its time while the stage ends
            if not late:
                late.append(self.background(s, [50, 51]))     # a new request comes meanwhile
                time.sleep(S.RESERVE_S + 0.8)
        r1 = a.generate(p, 5, on_tokens=slow)
        self.assertEqual((s.status()["queued"], s.status()["slots_busy"]), (1, 1))     # the new request still waits
        r2 = a.generate(p + r1.ids, 5)                         # the next stage continues the stream in its slot
        a.close()
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids, 5)
        self.assertEqual((r1.ids, r2.ids), (w1.ids, w2.ids))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(r2.snap_mode, 1)
        b, tb = late[0]
        wait_for(lambda: s.status()["slots_busy"] == 2 and s.status()["queued"] == 0, what="the new request to run")
        x.cancel()
        b.cancel()
        tx.join(10)
        tb.join(10)

    def test_next_turn_after_tool_calls_keeps_its_slot(self):
        """A request closed with keep_s (an answer with tool calls) keeps the stream of its last stage for its next turn: a new request
        that waits does not take the slot meanwhile, and the next turn (a new request whose prompt continues the stream) is read before
        it and continues the stream in its slot, read as alone (the ids and prompt-reuse mode of the single path)."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        trace = []
        s = S.Scheduler(self.e, 2, trace=trace, max_wait=NO_WAIT_BOUND)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 5)                                  # in a slot next to the first request
        b, tb = self.background(s, [50, 51])                   # a new request: it waits for a's slot
        wait_for(lambda: s.status()["queued"] == 1 and b._ticket is not None, what="the new request to wait")
        b_id = b._ticket.id
        a.close(keep_s=2.0)                                    # the answer asked for its next turn (tool calls)
        time.sleep(0.6)
        self.assertEqual((s.status()["queued"], s.status()["slots_busy"]), (1, 1))     # the new request still waits
        c = S.ScheduledEngine(s)                               # the next turn: a new request on the same conversation
        c.set_sampling(self.greedy)
        p2 = p + r1.ids + [60, 61]
        r2 = c.generate(p2, 5)
        c.close()
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids + [60, 61], 5)
        self.assertEqual((r1.ids, r2.ids), (w1.ids, w2.ids))
        self.assertEqual((r2.snap_mode, r2.cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(r2.snap_mode, 1)
        wait_for(lambda: s.status()["slots_busy"] == 2 and s.status()["queued"] == 0, what="the new request to run")
        reads = [t[1] for t in trace if t[0] == "read"]
        self.assertLess(reads.index(max(reads)), reads.index(b_id))   # the next turn (submitted last) was read before the new request
        x.cancel()
        b.cancel()
        tx.join(10)
        tb.join(10)

    def test_next_turn_keep_runs_out(self):
        """The stream kept for a next turn is given up keep_s after the close: a new request that waits then takes the slot; a request
        closed without keep_s gives its slot up at once."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        s = S.Scheduler(self.e, 2, max_wait=NO_WAIT_BOUND)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        for keep in (0.8, 0.0):
            with self.subTest(keep=keep):
                a = S.ScheduledEngine(s)
                a.set_sampling(self.greedy)
                a.generate([40, 41, 42 + int(keep * 10)], 5)
                b, tb = self.background(s, [50, 51 + int(keep * 10)])
                wait_for(lambda: s.status()["queued"] == 1, what="the new request to wait")
                t0 = time.monotonic()
                a.close(keep_s=keep)
                wait_for(lambda: s.status()["slots_busy"] == 2, timeout=5, what="the new request to take the slot")
                waited = time.monotonic() - t0
                if keep:
                    self.assertGreater(waited, keep - 0.2)
                else:
                    self.assertLess(waited, 0.5)
                b.cancel()
                tb.join(10)
                wait_for(lambda: s.status()["slots_busy"] == 1, what="the slot to come free")
        x.cancel()
        tx.join(10)

    def test_a_next_turn_that_came_in_time_keeps_the_stream_while_it_waits(self):
        """A next turn that comes within keep_s claims the stream kept for it: the stream stays kept while the turn waits for the
        scheduler (here it reads another request's long prompt meanwhile, past keep_s), a new request that waited before does not take
        it, and the turn continues the stream (the ids and prompt-reuse mode of the single path), read before that request."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        trace = []
        s = S.Scheduler(self.e, 3, trace=trace, max_wait=NO_WAIT_BOUND)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 5)                                  # in a slot next to the first request
        orig = self.e.generate

        def long_read(prompt, *args, **kw):                    # the read of a long prompt: the scheduler does nothing else meanwhile
            if len(prompt) and int(prompt[0]) == 77:
                time.sleep(1.2)
            return orig(prompt, *args, **kw)
        keep = 0.4
        out = {}

        def run(key, se, prompt, n):
            try:
                out[key] = se.generate(prompt, n)
            except Exception as ex:                            # noqa: BLE001
                out[key] = ex
            finally:
                se.close()
        with mock.patch.object(self.e, "generate", long_read):
            a.close(keep_s=keep)                               # the answer asked for its next turn (tool calls)
            b, tb = self.background(s, [77, 78])               # a new request with a long read: it takes the third slot
            wait_for(lambda: b._ticket is not None and any(t[0] == "read" and t[1] == b._ticket.id for t in trace),
                     what="the long read to start")
            d = S.ScheduledEngine(s)                           # a new request: it waits (no slot is free)
            d.set_sampling(self.greedy)
            td = threading.Thread(target=run, args=("d", d, [90, 91], 5), daemon=True)
            td.start()
            wait_for(lambda: d._ticket is not None, what="the new request to wait")
            c = S.ScheduledEngine(s)                           # the next turn, within keep_s of the close
            c.set_sampling(self.greedy)
            p2 = p + r1.ids + [60, 61]
            tc = threading.Thread(target=run, args=("c", c, p2, 5), daemon=True)
            tc.start()
            wait_for(lambda: c._ticket is not None, what="the next turn to wait")
            c_id, d_id = c._ticket.id, d._ticket.id
            tc.join(15)
            td.join(15)
        self.assertNotIsInstance(out.get("c"), Exception)
        self.assertNotIsInstance(out.get("d"), Exception)
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids + [60, 61], 5)
        self.assertEqual((r1.ids, out["c"].ids), (w1.ids, w2.ids))
        self.assertEqual((out["c"].snap_mode, out["c"].cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(out["c"].snap_mode, 1)
        reads = [t[1] for t in trace if t[0] == "read"]
        self.assertLess(reads.index(c_id), reads.index(d_id))  # the next turn first, though it came after the new request
        x.cancel()
        b.cancel()
        tx.join(10)
        tb.join(10)

    def test_a_request_that_came_before_the_answer_does_not_take_its_stream(self):
        """Two conversations with the same tokens: a request whose prompt continues the stream kept for another request's next turn,
        but that came before that answer ended (it waited for a slot), is not that next turn and does not take the stream; the next turn
        that comes afterwards continues it (the ids and prompt-reuse mode of the single path), and the other request is served after it
        with the same ids, read from the start."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        trace = []
        s = S.Scheduler(self.e, 2, back_to_one=False, trace=trace, max_wait=NO_WAIT_BOUND)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 5)                                  # in the second slot; the slot is kept while a is open
        p2 = p + r1.ids + [60, 61]
        out = {}

        def run(key, se, prompt, n):
            try:
                out[key] = se.generate(prompt, n)
            except Exception as ex:                            # noqa: BLE001
                out[key] = ex
            finally:
                se.close()
        y = S.ScheduledEngine(s)                               # the same tokens as a's next turn, sent before a's answer ended
        y.set_sampling(self.greedy)
        ty = threading.Thread(target=run, args=("y", y, p2, 5), daemon=True)
        ty.start()
        wait_for(lambda: s.status()["queued"] == 1 and y._ticket is not None, what="the other request to wait")
        a.close(keep_s=2.0)                                    # a's answer ends with tool calls
        time.sleep(0.3)
        self.assertNotIn("y", out)                             # it still waits: the kept stream is not for it
        c = S.ScheduledEngine(s)                               # a's next turn
        c.set_sampling(self.greedy)
        tc = threading.Thread(target=run, args=("c", c, p2, 5), daemon=True)
        tc.start()
        y_id = y._ticket.id
        tc.join(15)
        ty.join(15)
        self.assertNotIsInstance(out.get("c"), Exception)
        self.assertNotIsInstance(out.get("y"), Exception)
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids + [60, 61], 5)
        self.assertEqual((r1.ids, out["c"].ids, out["y"].ids), (w1.ids, w2.ids, w2.ids))
        self.assertEqual((out["c"].snap_mode, out["c"].cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(out["c"].snap_mode, 1)
        self.assertNotEqual(out["y"].snap_mode, 1)             # read from the start: the stream was the next turn's
        reads = [t[1] for t in trace if t[0] == "read"]
        self.assertLess(reads.index(max(reads)), reads.index(y_id))   # the next turn (submitted last) was read before the other one
        x.cancel()
        tx.join(10)

    def test_a_next_turn_that_comes_while_its_stream_moves_into_a_slot_claims_it(self):
        """The stream of an answer with tool calls ended on the carrier (the request ran alone) and is kept for the next turn.  Another
        request comes, and before its read the kept stream moves into a slot (ms_load).  The next turn comes exactly then (here
        ms_load waits for it): it still finds the stream, claims it and continues it (the ids and prompt-reuse mode of the single
        path), read right after the move.  Before this change the stream was on neither the carrier nor the slot during ms_load: the
        turn claimed nothing and waited until the keep ran out (or was read from the start)."""
        from yasakani import sched as S
        s = S.Scheduler(self.e, 2)
        self.addCleanup(s.close)
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 20)                                 # alone on the carrier: the stream ends there
        self.assertIsNotNone(s._carrier)
        keep = 3.0
        orig = self.e.ms_load
        moving, release = threading.Event(), threading.Event()

        def slow_load(slot, *args, **kw):                      # the first ms_load: the kept stream moves into a slot
            if not moving.is_set():
                moving.set()
                release.wait(10)
            return orig(slot, *args, **kw)
        out = {}

        def run(key, se, prompt, n):
            try:
                out[key] = se.generate(prompt, n)
            except Exception as ex:                            # noqa: BLE001
                out[key] = ex
            finally:
                se.close()
        with mock.patch.object(self.e, "ms_load", slow_load):
            a.close(keep_s=keep)                               # a's answer ends with tool calls
            b = S.ScheduledEngine(s)                           # another request: its admission moves a's stream into a slot
            b.set_sampling(self.greedy)
            tb = threading.Thread(target=run, args=("b", b, [90, 91], 100000), daemon=True)
            tb.start()
            self.assertTrue(moving.wait(10), "the kept stream did not move into a slot")
            c = S.ScheduledEngine(s)                           # a's next turn, while the stream moves
            c.set_sampling(self.greedy)
            p2 = p + r1.ids + [60, 61]
            tc = threading.Thread(target=run, args=("c", c, p2, 5), daemon=True)
            tc.start()
            wait_for(lambda: c._ticket is not None or "c" in out, what="the next turn to be submitted")
            release.set()
            tc.join(15)
        b.cancel()
        tb.join(10)
        self.assertNotIsInstance(out.get("c"), Exception)
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 20)
        w2 = self.ref.generate(p2, 5)
        self.assertEqual((r1.ids, out["c"].ids), (w1.ids, w2.ids))
        self.assertEqual((out["c"].snap_mode, out["c"].cached_tokens), (w2.snap_mode, w2.cached_tokens))
        self.assertEqual(out["c"].snap_mode, 1)
        self.assertLess(out["c"].stats["queue_ms"], 1000.0 * keep / 2)   # read after the move, not when the keep ran out

    def test_a_claimed_stream_is_kept_until_the_turn_is_served(self):
        """sched._kept for a closed request (keep_s): kept for keep_s after the close; once its next turn has claimed it, until that turn
        is served or cancelled (RESERVE_MAX_S after the close at most); a stream given up is not kept again."""
        from yasakani import sched as S

        class Owner:
            _open = False
        o = Owner()
        h = S._Held([1, 2, 3], 4, o, tid=7)
        o._taken_tid, o._t_taken, o._t_closed, o._keep_s = 7, h.t_end + 0.1, h.t_end + 0.2, 0.5
        t0 = o._t_closed
        self.assertTrue(S._kept(h, t0 + 0.4))
        self.assertFalse(S._kept(h, t0 + 0.6))
        claim = S.Ticket(S.Stage([1, 2, 3, 4, 5], 5), None)
        h.claim = claim
        self.assertTrue(S._kept(h, t0 + 0.6))                  # the next turn came in time and waits
        self.assertTrue(S._kept(h, t0 + S.RESERVE_MAX_S - 0.1))
        self.assertFalse(S._kept(h, t0 + S.RESERVE_MAX_S + 0.1))
        claim.closed = True                                    # served (or cancelled while it waited)
        self.assertFalse(S._kept(h, t0 + 0.6))
        self.assertTrue(S._kept(h, t0 + 0.4))
        o._keep_s = 0.0                                        # closed without keep_s: never kept
        self.assertFalse(S._kept(h, t0 + 0.1))

    def test_a_later_stage_is_not_a_new_request(self):
        """The next stage of a request whose stream was given away (its client sent the stage later than RESERVE_S) is not a new
        request: it does not count against max_queue (not refused while the queue is full, where a new request is) and it runs before
        the requests that wait.  Its ids are those of the single path."""
        from yasakani import sched as S
        self.stub.draft_mode(self.e, 2)                        # one id a pass: the first request runs through the test
        trace = []
        s = S.Scheduler(self.e, 2, max_queue=1, trace=trace)
        self.addCleanup(s.close)
        x, tx = self.background(s, [1, 2, 3])
        wait_for(lambda: s.status()["slots_busy"] == 1, what="the first request")
        a = S.ScheduledEngine(s)
        a.set_sampling(self.greedy)
        p = [40, 41, 42]
        r1 = a.generate(p, 5)                                  # in a slot next to the first request
        b, tb = self.background(s, [50, 51])                   # takes a's slot once a is late
        wait_for(lambda: s.status()["slots_busy"] == 2, timeout=S.RESERVE_S + 3, what="the new request to take a's slot")
        d = S.ScheduledEngine(s)
        d.set_sampling(self.greedy)
        out = {}

        def run(key, se, prompt, n):
            try:
                out[key] = se.generate(prompt, n)
            except Exception as ex:                            # noqa: BLE001
                out[key] = ex
        td = threading.Thread(target=run, args=("d", d, [60, 61], 8), daemon=True)
        td.start()
        wait_for(lambda: s.status()["queued"] == 1 and d._ticket is not None, what="a waiting request")
        with self.assertRaises(S.QueueFull):                   # the queue is full: a new request is refused
            s.submit(S.Stage([70, 71], 5, EOS_IDS, self.greedy))
        ta = threading.Thread(target=run, args=("a", a, p + r1.ids, 6), daemon=True)
        ta.start()
        wait_for(lambda: (s.status()["queued"] == 2 and a._ticket is not None) or "a" in out, what="a's next stage to wait")
        self.assertNotIn("a", out)                             # not refused: it waits
        a_id, d_id = a._ticket.id, d._ticket.id
        b.cancel()                                             # a slot comes free: a's next stage takes it before d
        ta.join(15)
        a.close()
        td.join(15)
        self.assertNotIsInstance(out.get("a"), Exception)
        self.assertNotIsInstance(out.get("d"), Exception)
        self.ref.set_sampling(self.greedy)
        w1 = self.ref.generate(p, 5)
        w2 = self.ref.generate(p + w1.ids, 6)
        self.assertEqual((r1.ids, out["a"].ids, out["a"].finish), (w1.ids, w2.ids, w2.finish))
        reads = [t[1] for t in trace if t[0] == "read"]
        self.assertLess(reads.index(a_id), reads.index(d_id))
        x.cancel()
        tx.join(10)
        tb.join(10)


class ClientGoneTests(unittest.TestCase):
    """server.client_gone: an open connection is not gone (also with data waiting on it), a closed one is; on POSIX also for a socket
    numbered 1024 or more, which select() cannot take (it raised ValueError, which was read as a closed connection)."""

    def check(self, sock, peer):
        from yasakani.server import client_gone
        self.assertFalse(client_gone(sock))
        peer.sendall(b"x")
        time.sleep(0.05)
        self.assertFalse(client_gone(sock))                    # data waiting (a request sent ahead): not gone
        self.assertEqual(sock.recv(1), b"x")
        peer.close()
        wait_for(lambda: client_gone(sock), timeout=5, what="the closed connection to be seen")

    def test_open_and_closed(self):
        a, b = socket.socketpair()
        try:
            self.check(a, b)
        finally:
            a.close()
            b.close()

    @unittest.skipUnless(IS_POSIX, "socket numbers above 1024: POSIX")
    def test_a_socket_numbered_above_1024(self):
        import resource
        import select
        fd = 1101
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and soft <= fd + 1:
            if hard != resource.RLIM_INFINITY and hard <= fd + 1:
                self.skipTest("the limit of open files is %d" % hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (fd + 100 if hard == resource.RLIM_INFINITY else min(hard, fd + 100), hard))
            self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        a, b = socket.socketpair()
        try:
            os.dup2(a.fileno(), fd)
        finally:
            a.close()
        hi = socket.socket(fileno=fd)
        try:
            with self.assertRaises(ValueError):
                select.select([hi], [], [], 0)
            self.check(hi, b)
        finally:
            hi.close()
            b.close()


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class StartupCheckTests(unittest.TestCase):
    """sched.consistency_check: N requests at once against one at a time."""

    @classmethod
    def setUpClass(cls):
        from yasakani.textio import ChatFormatter
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def test_passes_on_the_stub(self):
        from yasakani.sched import Scheduler, check_prompts, consistency_check
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        e = support.open_engine(Path(tmp.name), self.stub, max_ctx=4096)
        self.addCleanup(e.close)
        self.stub.clear_scripts()
        self.stub.model(e, True, eos_every=0, vocab=64)
        self.stub.delay_us(100)
        self.addCleanup(self.stub.delay_us, 0)
        for n in (2, 5):
            trace = []
            s = Scheduler(e, n, trace=trace)
            try:
                prompts = check_prompts(self.fmt, n)
                self.assertEqual(len({tuple(p[:1]) for p in prompts}), n)          # each starts with other ids
                res = consistency_check(s, prompts, max_new=40)
            finally:
                s.close()
            self.assertTrue(res.ok, res.line())
            self.assertEqual((res.n, res.compared), (n, n))
            self.assertTrue(any(t[0] == "rows" for t in trace))                 # the second round ran in the slots
        self.assertEqual(len(check_prompts(self.fmt, 40)), 40)


class CompareRunsTests(unittest.TestCase):
    def r(self, ids, finish="length", mode=0, cached=0):
        return GenResult(ids=list(ids), finish=finish, prompt_tokens=3, cached_tokens=cached, snap_mode=mode, passes=1, prefill_ms=0.0,
                         gen_ms=0.0)

    def test_compare(self):
        from yasakani.sched import compare_runs
        a = [self.r([1, 2, 3]), self.r([4, 5]), self.r([6, 7], mode=2, cached=8), self.r([9])]
        b = [self.r([1, 2, 3]), self.r([4, 6]), self.r([6, 8]), self.r([9], finish="stop")]
        res = compare_runs(a, b)
        self.assertFalse(res.ok)
        self.assertEqual((res.n, res.compared, res.differ, res.first_diff), (4, 3, [1, 3], {1: 1, 3: 1}))
        self.assertIn("#2 from id 2", res.line())
        res = compare_runs(a[:1], b[:1])
        self.assertTrue(res.ok)
        self.assertIn("the same ids for 1 of 1", res.line())


class ParallelPlanTests(unittest.TestCase):
    """The context of N requests at once: (N + 1) contexts and N slots beside the model, on every card."""

    def pack(self, sizes, max_ctx=262144):
        from tests.test_gpu import FakePack
        return FakePack(sizes, max_ctx)

    def det(self, total, used, name="NVIDIA GeForce RTX 5090", cc="12.0"):
        from tests.test_gpu import det_of, smi_line
        return det_of(smi_line(0, name, cc, total, used))

    def test_rtx_5090(self):
        for sizes, n, ctx, head_off in ((FASTEST, 2, 245760, False), (FASTEST, 4, 131072, False), (FASTEST, 8, 65536, False),
                                        (QUALITY, 2, 212992, False), (QUALITY, 4, 114688, False), (QUALITY, 8, 65536, True)):
            with self.subTest(sizes=sizes[0], n=n):
                plan = G.plan_parallel(self.pack(sizes), self.det(32607, 1107), n)          # 31,500 MiB free
                self.assertEqual((plan.max_ctx, plan.parallel, plan.env.get("YSKN_BDR_SHEAD") == "0"), (ctx, n, head_off))
                self.assertEqual(plan.free_mib, 31500)
                self.assertIn("context: %d tokens for each of %d requests at once" % (ctx, n), plan.notes[-1])
                files = sum(sizes) - (sizes[2] if head_off else 0)
                fit = G.parallel_fit(31500, files, G.overhead_mib(sizes[0]), n)
                self.assertTrue(ctx <= fit < ctx + G.CONTEXT_STEP, fit)                 # the longest multiple of 16,384 that fits
                need = G.parallel_need_mib(files, sizes[0], ctx, n, G.KV_BYTES_PER_TOKEN - G.POSITION_TABLE_BYTES, ENGINE_SLOT_BYTES)
                self.assertLessEqual(need, 31500)                                      # also by the engine's own way of counting
                self.assertLessEqual(ENGINE_SLOT_BYTES / MIB + G.SLOT_ALLOC_MIB, G.SLOT_MIB)  # the plan counts what the engine adds

    def test_slot_memory_as_measured(self):
        """The engine's count with SLOT_ALLOC_MIB covers what the server took on the RTX 5090 (MiB above the empty card, at the peak
        under load; 16 requests at once missed the estimate by 376 MiB without it)."""
        measured = ((FASTEST, 2, 245760, 30615), (FASTEST, 4, 131072, 29350), (FASTEST, 8, 65536, 28853), (FASTEST, 16, 32768, 30039),
                    (QUALITY, 2, 212992, 30758), (QUALITY, 8, 65536, 30994))
        for sizes, n, ctx, used in measured:
            with self.subTest(weights=sizes[0], n=n):
                need = G.parallel_need_mib(sum(sizes), sizes[0], ctx, n, 22144, ENGINE_SLOT_BYTES)
                self.assertGreaterEqual(need, used)

    def test_model_max_and_small_n(self):
        plan = G.plan_parallel(self.pack(FASTEST, max_ctx=32768), self.det(32607, 1107), 2)
        self.assertEqual(plan.max_ctx, 32768)

    def test_too_many(self):
        with self.assertRaises(YasakaniError) as cm:
            G.plan_parallel(self.pack(QUALITY), self.det(32607, 1107), 40)
        self.assertIn("--parallel 40 does not fit", str(cm.exception))
        self.assertRegex(str(cm.exception), r"--parallel \d+ fits")
        with self.assertRaises(YasakaniError) as cm:
            G.plan_parallel(self.pack(QUALITY), self.det(24564, 8000, "NVIDIA GeForce RTX 4090", "8.9"), 2)
        self.assertIn("Even --parallel 2 does not fit", str(cm.exception))

    def test_unknown_memory(self):
        from yasakani.gpu import Detected
        plan = G.plan_parallel(self.pack(FASTEST), Detected(), 4)
        self.assertEqual((plan.parallel, plan.max_ctx), (1, None))
        self.assertIn("nvidia-smi could not tell", plan.notes[0])
        plan = G.plan_given_parallel(self.pack(FASTEST), Detected(), 8192, 4)
        self.assertEqual((plan.parallel, plan.max_ctx, plan.env, plan.notes), (4, 8192, {}, []))

    def test_given_context(self):
        det = self.det(32607, 1107)
        plan = G.plan_given_parallel(self.pack(FASTEST), det, 65536, 4)
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (65536, {}, []))
        plan = G.plan_given_parallel(self.pack(FASTEST), det, 155648, 4)               # fits only without the head
        self.assertEqual(plan.env, {"YSKN_BDR_SHEAD": "0"})
        self.assertIn("is not loaded", plan.notes[0])
        plan = G.plan_given_parallel(self.pack(FASTEST), det, 262144, 4)
        self.assertEqual(plan.env, {})
        self.assertIn("trying anyway", plan.notes[0])

    def info(self, **kw):
        v = dict(unsupported=0, kv_format=5, kv5_bits=5, kv_bytes_per_position=22144, slot_bytes=195887104, max_ctx=131072, max_rows=16,
                 max_slots=64, drafter=True, draft_head=True, slots_made=0, slots_bound=0, snapshot_bytes=0, snapshot_slots=8)
        v.update(kw)
        return MsInfo(**v)

    def test_check_with_the_engine_numbers(self):
        pack = self.pack(FASTEST)
        plan = G.plan_parallel(pack, self.det(32607, 1107), 4)
        self.assertEqual(G.check_parallel(pack, plan, self.info(max_ctx=plan.max_ctx), 4), (4, None))
        n, note = G.check_parallel(pack, plan, self.info(max_ctx=plan.max_ctx, unsupported=2, kv_format=8, kv5_bits=0), 4)
        self.assertEqual(n, 1)
        self.assertIn("8 bits", note)
        n, note = G.check_parallel(pack, plan, self.info(max_ctx=plan.max_ctx, kv_bytes_per_position=33280), 4)  # a bigger KV
        self.assertEqual(n, 2)
        self.assertIn("serving 2 at a time", note)
        n, note = G.check_parallel(pack, plan, self.info(max_ctx=plan.max_ctx, kv_bytes_per_position=80000), 4)
        self.assertEqual(n, 1)
        self.assertIn("not even 2 fit", note)
        self.assertEqual(G.check_parallel(pack, G.ContextPlan(max_ctx=8192, parallel=4), self.info(), 4), (4, None))   # unknown memory
        no_head = self.info(max_ctx=plan.max_ctx, draft_head=False, kv_bytes_per_position=25500)
        self.assertEqual(G.check_parallel(pack, plan, no_head, 4), (4, None))          # the head left out leaves room
        self.assertEqual(G.check_parallel(pack, plan, self.info(max_ctx=plan.max_ctx, kv_bytes_per_position=25500), 4)[0], 3)
        # --prefill new on a card under 30,000 MiB: the reserve of the plan is not given to the slots
        det = self.det(24564, 500)
        plain, kept = G.plan_parallel(pack, det, 2), G.plan_parallel(pack, det, 2, G.NEW_PREFILL_RESERVE_MIB)
        self.assertEqual((plain.reserve_mib, kept.reserve_mib), (0, G.NEW_PREFILL_RESERVE_MIB))
        big = self.info(max_ctx=kept.max_ctx, kv_bytes_per_position=26000)       # the engine's KV a little larger than the planner's
        self.assertEqual(G.check_parallel(pack, plain, self.info(max_ctx=kept.max_ctx, kv_bytes_per_position=26000), 2), (2, None))
        n, note = G.check_parallel(pack, kept, big, 2)
        self.assertEqual(n, 1)
        self.assertIn("(besides 1800 MiB kept for the new prompt reading)", note)
        self.assertEqual(G.check_parallel(pack, kept, self.info(max_ctx=kept.max_ctx), 2), (2, None))


class CliOptionTests(unittest.TestCase):
    def test_options(self):
        from yasakani import cli
        p = cli.build_parser()
        a = p.parse_args(["serve", "--parallel", "4", "--max-queue", "8"])
        self.assertEqual((a.parallel, a.max_queue, a.parallel_rows, a.no_slot_reuse, a.parallel_check), (4, 8, 16, False, "auto"))
        a = p.parse_args(["serve"])
        self.assertEqual((a.parallel, a.max_queue), (None, 64))
        a = p.parse_args(["serve", "--parallel-rows", "8", "--no-slot-reuse", "--parallel-check", "always"])
        self.assertEqual((a.parallel_rows, a.no_slot_reuse, a.parallel_check), (8, True, "always"))
        self.assertEqual(p.parse_args(["selfcheck", "--parallel", "3"]).parallel, 3)
        self.assertEqual(p.parse_args(["bench", "--concurrency", "5"]).concurrency, 5)
        subs = p._subparsers._group_actions[0].choices
        helps = {cmd: subs[cmd].format_help() for cmd in ("serve", "selfcheck", "bench")}
        self.assertIn("--parallel N", helps["serve"])
        self.assertIn("--max-queue Q", helps["serve"])
        self.assertIn("--concurrency N", helps["bench"])
        for h in helps.values():                                                   # the options for tests stay out of the help
            for hidden in ("--parallel-rows", "--no-slot-reuse", "--parallel-check"):
                self.assertNotIn(hidden, h)

    def test_count(self):
        from yasakani import cli
        with mock.patch.dict(os.environ, {"YASAKANI_PARALLEL": ""}):
            self.assertEqual(cli._parallel_count(None, env=True), 1)
        with mock.patch.dict(os.environ, {"YASAKANI_PARALLEL": "4"}):
            self.assertEqual(cli._parallel_count(None, env=True), 4)
            self.assertEqual(cli._parallel_count(None), 1)                         # selfcheck and bench: only their own option
            self.assertEqual(cli._parallel_count(2, env=True), 2)                  # the option wins
        with mock.patch.dict(os.environ, {"YASAKANI_PARALLEL": "many"}):
            with self.assertRaisesRegex(YasakaniError, "whole number"):
                cli._parallel_count(None, env=True)
        for bad in (0, 65, -1):
            with self.assertRaisesRegex(YasakaniError, "between 1 and 64"):
                cli._parallel_count(bad)

    def test_keep_turn(self):
        """--keep-turn auto|off|SECONDS, else $YASAKANI_KEEP_TURN, else auto; the scheduler gets None (auto), 0 or the seconds."""
        from yasakani import cli
        p = cli.build_parser()
        self.assertIn("--keep-turn auto|off|SECONDS", p._subparsers._group_actions[0].choices["serve"].format_help())
        with mock.patch.dict(os.environ, {"YASAKANI_KEEP_TURN": ""}):
            for argv, want in ((["serve"], None), (["serve", "--keep-turn", "auto"], None), (["serve", "--keep-turn", "off"], 0.0),
                               (["serve", "--keep-turn", "0"], 0.0), (["serve", "--keep-turn", "4.5"], 4.5),
                               (["serve", "--keep-turn", "AUTO"], None)):
                self.assertEqual(cli._turn_settings(p.parse_args(argv))["keep_turn"], want, argv)
            for bad in ("-1", "601", "soon", "nan"):
                with self.assertRaisesRegex(YasakaniError, "--keep-turn"):
                    cli._turn_settings(p.parse_args(["serve", "--keep-turn", bad]))
        with mock.patch.dict(os.environ, {"YASAKANI_KEEP_TURN": "off"}):
            self.assertEqual(cli._turn_settings(p.parse_args(["serve"]))["keep_turn"], 0.0)
            self.assertEqual(cli._turn_settings(p.parse_args(["serve", "--keep-turn", "3"]))["keep_turn"], 3.0)   # the option wins
            self.assertEqual(cli._turn_settings(p.parse_args(["bench"]))["keep_turn"], 0.0)                        # others: the variable
        with mock.patch.dict(os.environ, {"YASAKANI_KEEP_TURN": "later"}):
            with self.assertRaisesRegex(YasakaniError, "YASAKANI_KEEP_TURN"):
                cli._turn_settings(p.parse_args(["serve"]))

    def test_max_wait(self):
        """--max-wait auto|SECONDS, else $YASAKANI_MAX_WAIT, else auto (no "off": a new request always gets a bound); the start-up
        line names both settings."""
        from yasakani import cli
        from yasakani import sched as S
        p = cli.build_parser()
        self.assertIn("--max-wait auto|SECONDS", p._subparsers._group_actions[0].choices["serve"].format_help())
        with mock.patch.dict(os.environ, {"YASAKANI_MAX_WAIT": "", "YASAKANI_KEEP_TURN": ""}):
            for argv, want in ((["serve"], None), (["serve", "--max-wait", "auto"], None), (["serve", "--max-wait", "0"], 0.0),
                               (["serve", "--max-wait", "2.5"], 2.5)):
                self.assertEqual(cli._turn_settings(p.parse_args(argv))["max_wait"], want, argv)
            for bad in ("off", "-0.5", "700", "x"):
                with self.assertRaisesRegex(YasakaniError, "--max-wait"):
                    cli._turn_settings(p.parse_args(["serve", "--max-wait", bad]))
        with mock.patch.dict(os.environ, {"YASAKANI_MAX_WAIT": "1.25", "YASAKANI_KEEP_TURN": ""}):
            self.assertEqual(cli._turn_settings(p.parse_args(["serve"])), dict(keep_turn=None, max_wait=1.25))

        class Sch:
            keep_turn, max_wait = None, None
        line = cli._turn_line(Sch)
        self.assertIn("%g to %g s" % (S.KEEP_TURN_MIN_S, S.KEEP_TURN_MAX_S), line)
        self.assertIn("and %g s" % S.WAIT_FLOOR_S, line)
        Sch.keep_turn, Sch.max_wait = 3.0, 0.5
        self.assertIn("next turn 3 s; a new request waits for next turns at most 0.5 s", cli._turn_line(Sch))
        Sch.keep_turn = 0.0
        self.assertEqual(cli._turn_line(Sch), "after tool calls nothing is kept for the next turn")

    def test_check_cache(self):
        from yasakani import cli
        d = Path(tempfile.mkdtemp(prefix="yct_cache_"))
        self.addCleanup(shutil.rmtree, d, True)
        with mock.patch.dict(os.environ, {"YASAKANI_CACHE_DIR": str(d / "c")}):
            path = cli._cache_dir() / cli.CHECK_FILE
            self.assertEqual(path, d / "c" / "parallel_check.json")
            self.assertEqual(cli._check_cache(path), {})
            cli._check_remember(path, "k1", dict(ok=True, detail="x"))
            cli._check_remember(path, "k2", dict(ok=False, detail="y"))
            self.assertEqual(cli._check_cache(path), {"k1": dict(ok=True, detail="x"), "k2": dict(ok=False, detail="y")})
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(cli._check_cache(path), {})


FAKE_SMI_GPU = dict(index=0, uuid="GPU-00000000-0000-0000-0000-000000000000", name="NVIDIA GeForce RTX 5090", compute_cap="12.0",
                    driver_version="616.92", **{"memory.total": 32607, "memory.used": 1107})


@unittest.skipUnless(HAVE_TOK and IS_POSIX, "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliParallelTests(unittest.TestCase):
    """The commands with the stub library: serve --parallel (the check once, then from the cache; /health), selfcheck --parallel,
    bench --concurrency, and an engine library without the entries (one request at a time)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_gpu import FAKE_SMI
        cls.dll = str(support.stub_path())
        cls.old_dll = str(support.stub_path(no_ms=True))
        cls.folder = support.make_real_folder("par")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="yct_par_"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.smi = cls.tmp / "nvidia-smi"
        cls.smi.write_text(FAKE_SMI.replace("{python}", sys.executable), encoding="utf-8")
        cls.smi.chmod(0o755)

    def env(self, cache, smi=True):
        """The stand-in model of the stub for every engine (no stop ids), the fake nvidia-smi (an RTX 5090 with 31,500 MiB free) or
        none, and a cache folder of the test's own."""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_CACHE_DIR=str(cache),
                   YASAKANI_NVIDIA_SMI=str(self.smi) if smi else "none", FAKE_GPUS=json.dumps([FAKE_SMI_GPU]))
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_"))] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "YASAKANI_PARALLEL",
                                                                          "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        env.update(STUB_MODEL="0", STUB_DELAY_US="200")                       # passes that take time: the requests of a check overlap
        return env

    def run_cli(self, env, *args, timeout=120):
        return subprocess.run([sys.executable, "-m", "yasakani", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
                              env=env)

    def serve(self, env, *extra):
        """The server on a port the system gives it (--port 0; the port is read from its "listening on" line, so that tests run at
        once never pick the same one)."""
        proc = subprocess.Popen([sys.executable, "-m", "yasakani", "serve", "--model-dir", str(self.folder), "--port", "0",
                                 "--max-ctx", "8192", *extra], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True)
        lines, port = [], []

        def read():
            for line in proc.stderr:
                lines.append(line)
                m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", line)
                if m and not port:
                    port.append(int(m.group(1)))
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            health = None
            t0 = time.time()
            while not port and time.time() - t0 < 60 and proc.poll() is None:
                time.sleep(0.05)
            base = "http://127.0.0.1:%d" % (port[0] if port else 0)
            while port and time.time() - t0 < 60 and proc.poll() is None:
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                        health = json.loads(resp.read())
                except Exception:                                   # noqa: BLE001  (not up yet / loading: 503)
                    health = None
                if health and health["status"] == "ok":
                    break
                time.sleep(0.2)
            reply = None
            if health and health["status"] == "ok":
                req = urllib.request.Request(base + "/v1/chat/completions", method="POST", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                                                              "chat_template_kwargs": {"enable_thinking": False}}).encode())
                with urllib.request.urlopen(req, timeout=20) as resp:
                    reply = json.loads(resp.read())
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=30)
            reader.join(10)
            return proc.returncode, health, reply, "".join(lines)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            reader.join(10)

    def test_serve_check_once_then_cached(self):
        cache = self.tmp / "cache1"
        env = self.env(cache)
        rc, health, reply, err = self.serve(env, "--dll", self.dll, "--parallel", "3")
        self.assertEqual(rc, 0, err)
        self.assertEqual((health["parallel"], health["max_context_tokens"], health["slots_busy"], health["queued"]), (3, 8192, 0, 0))
        self.assertEqual(reply["usage"]["completion_tokens"], 8)
        self.assertIn("queue_ms", reply["timings"])
        self.assertIn("parallel check passed: the same ids for 3 of 3 requests", err)
        self.assertIn("serving up to 3 requests at once", err)
        rec = json.loads((cache / "parallel_check.json").read_text(encoding="utf-8"))
        self.assertEqual([v["ok"] for v in rec.values()], [True])
        rc, health, reply, err = self.serve(env, "--dll", self.dll, "--parallel", "3")
        self.assertEqual(rc, 0, err)
        self.assertIn("parallel check: passed before", err)
        self.assertNotIn("checking that", err)
        self.assertEqual(health["parallel"], 3)
        rc, health, reply, err = self.serve(dict(env, YASAKANI_PARALLEL="2"), "--dll", self.dll)      # the variable: another key
        self.assertEqual(rc, 0, err)
        self.assertEqual(health["parallel"], 2)
        self.assertIn("parallel check passed: the same ids for 2 of 2 requests", err)
        self.assertEqual(len(json.loads((cache / "parallel_check.json").read_text(encoding="utf-8"))), 2)

    def test_serve_check_fails(self):
        """Passes that give other ids than one request alone: the check fails, the server serves one request at a time, and the
        next start does not check again."""
        cache = self.tmp / "cache6"
        env = dict(self.env(cache), STUB_MS_SALT="12345")
        rc, health, reply, err = self.serve(env, "--dll", self.dll, "--parallel", "2")
        self.assertEqual(rc, 0, err)
        self.assertIn("parallel check FAILED: different ids for 2 of 2 requests", err)
        self.assertIn("--parallel 2 is off: 2 requests at once gave other output", err)
        self.assertNotIn("parallel", health)
        self.assertEqual(reply["usage"]["completion_tokens"], 8)
        rec = json.loads((cache / "parallel_check.json").read_text(encoding="utf-8"))
        self.assertEqual([v["ok"] for v in rec.values()], [False])
        rc, health, reply, err = self.serve(env, "--dll", self.dll, "--parallel", "2")
        self.assertIn("parallel check: failed before", err)
        self.assertNotIn("parallel", health)
        rc, health, reply, err = self.serve(env, "--dll", self.dll, "--parallel", "2", "--parallel-check", "always")
        self.assertIn("parallel check FAILED", err)

    def test_serve_falls_back_to_one_at_a_time(self):
        cache = self.tmp / "cache2"
        rc, health, reply, err = self.serve(self.env(cache), "--dll", self.old_dll, "--parallel", "3")
        self.assertEqual(rc, 0, err)
        self.assertIn("no multi-stream entries", err)
        self.assertNotIn("parallel", health)                                           # served as before
        self.assertEqual(reply["usage"]["completion_tokens"], 8)
        rc, health, reply, err = self.serve(dict(self.env(cache), STUB_MS_UNSUPPORTED="2"), "--dll", self.dll, "--parallel", "3")
        self.assertEqual(rc, 0, err)
        self.assertIn("8 bits", err)
        self.assertNotIn("parallel", health)
        rc, health, reply, err = self.serve(dict(self.env(cache), STUB_MS_BIND_FAIL="1"), "--dll", self.dll, "--parallel", "3")
        self.assertEqual(rc, 0, err)
        self.assertIn("the stream slots could not be made", err)
        self.assertNotIn("parallel", health)
        self.assertFalse((cache / "parallel_check.json").exists())

    def test_serve_too_many_for_the_memory(self):
        env = self.env(self.tmp / "cache3")
        env["FAKE_GPUS"] = json.dumps([dict(FAKE_SMI_GPU, **{"memory.used": 30000})])
        r = self.run_cli(env, "serve", "--model-dir", str(self.folder), "--dll", self.dll, "--parallel", "8", "--port", "0")
        self.assertEqual(r.returncode, 1)
        self.assertIn("--parallel 8 does not fit", r.stderr)
        self.assertNotIn("loading the model", r.stderr)

    def test_selfcheck_parallel(self):
        cache = self.tmp / "cache4"
        env = self.env(cache, smi=False)
        common = ["--model-dir", str(self.folder), "--max-ctx", "4096"]
        ref = self.tmp / "ref.json"
        r = self.run_cli(env, "selfcheck", *common, "--dll", self.dll, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.run_cli(env, "selfcheck", *common, "--dll", self.dll, "--reference", str(ref), "--parallel", "3")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("PASS"), 4)
        self.assertIn("PASS  (3 requests at once)", r.stdout)
        r = self.run_cli(env, "selfcheck", *common, "--dll", self.old_dll, "--reference", str(ref), "--parallel", "3")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("NOT RUN", r.stdout)
        r = self.run_cli(dict(env, STUB_MS_SALT="7"), "selfcheck", *common, "--dll", self.dll, "--reference", str(ref), "--parallel", "2")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("self-check FAILED: 2 requests at once gave other output", r.stdout)

    def test_bench_concurrency(self):
        env = self.env(self.tmp / "cache5", smi=False)
        r = self.run_cli(env, "bench", "--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096", "--n", "16", "--runs", "2",
                         "--concurrency", "3", "--json")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual((d["concurrency"], len(d["runs"])), (3, 8))
        self.assertTrue(all(x["tokens"] == 16 for x in d["runs"]), d["runs"])
        self.assertEqual(d["total_tokens"], 128)
        self.assertGreater(d["total_tokens_per_s"], 0)
        r = self.run_cli(env, "bench", "--model-dir", str(self.folder), "--dll", self.old_dll, "--max-ctx", "4096", "--n", "8",
                         "--concurrency", "2")
        self.assertEqual(r.returncode, 1)
        self.assertIn("--concurrency 2 cannot run here", r.stderr)


if __name__ == "__main__":
    unittest.main()
