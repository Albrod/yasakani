# SPDX-License-Identifier: Apache-2.0
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests import support
from yasakani.chat import ChatRunner, Usage
from yasakani.runtime import EOS_IDS
from yasakani.server import ApiState, ServerConfig, bind, timings_json
from yasakani.textio import ChatFormatter

EOS = EOS_IDS[0]
HAVE_TOK = support.tokenizer_dir() is not None


class TimingsTests(unittest.TestCase):
    LAYOUT = ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
              "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"]   # the llama.cpp server's fields

    def test_llama_cpp_layout(self):
        t = timings_json(Usage(prompt_tokens=120, cached_tokens=20, completion_tokens=50, passes=10, prefill_ms=200.0, prompt_ms=200.0,
                               gen_ms=250.0, draft_n=60, draft_n_accepted=39))
        self.assertEqual(list(t), self.LAYOUT + ["draft_n", "draft_n_accepted"])
        self.assertEqual((t["cache_n"], t["prompt_n"], t["prompt_ms"], t["prompt_per_token_ms"], t["prompt_per_second"]),
                         (20, 100, 200.0, 2.0, 500.0))
        self.assertEqual((t["predicted_n"], t["predicted_ms"], t["predicted_per_token_ms"], t["predicted_per_second"]), (50, 250.0, 5.0, 200.0))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (60, 39))

    def test_later_readings_count_as_generation(self):
        # a second stage (thinking budget, loop guard) reads its prompt again: that time comes after the first token
        t = timings_json(Usage(prompt_tokens=100, completion_tokens=40, prefill_ms=130.0, prompt_ms=100.0, gen_ms=170.0))
        self.assertEqual((t["prompt_ms"], t["predicted_ms"], t["predicted_per_token_ms"]), (100.0, 200.0, 5.0))
        self.assertEqual(list(t), self.LAYOUT)                                  # no look-ahead counts when nothing was proposed

    def test_no_time_measured(self):
        t = timings_json(Usage(prompt_tokens=5, cached_tokens=5))            # everything from the cache, nothing generated
        self.assertEqual((t["prompt_n"], t["prompt_per_token_ms"], t["prompt_per_second"]), (0, 0.0, 0.0))
        self.assertEqual((t["predicted_n"], t["predicted_per_token_ms"], t["predicted_per_second"]), (0, 0.0, 0.0))


@unittest.skipUnless(HAVE_TOK, "set YASAKANI_TEST_MODEL_DIR to a folder with the Qwen3.8 tokenizer files")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()
        cls.fmt = ChatFormatter(support.tokenizer_dir())

    def start(self, **cfg_kw):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.delay_us(0)
        self.engine = support.open_engine(Path(self.tmp.name), self.stub, max_ctx=20000)
        cfg = ServerConfig(port=0, **cfg_kw)
        self.state = ApiState(cfg)
        self.state.runner = ChatRunner(self.engine, self.fmt, seed0=1)
        self.httpd = bind(self.state)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
            self.engine.close()
            self.tmp.cleanup()
            self.httpd = None

    def ids(self, text):
        return self.fmt.encode(text).tolist()

    def request(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        self.addCleanup(c.close)
        h = {"Host": host or "127.0.0.1:%d" % self.port}
        if body is not None:
            h["Content-Type"] = "application/json"
            body = json.dumps(body)
        h.update(headers or {})
        c.request(method, path, body=body, headers=h)
        return c, c.getresponse()

    def chat(self, **body):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        c, r = self.request("POST", "/v1/chat/completions", body)
        data = r.read()
        c.close()
        return r.status, json.loads(data)

    def test_health_and_models(self):
        self.start()
        c, r = self.request("GET", "/health")
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["status"], "ok")
        c, r = self.request("GET", "/v1/models")
        self.assertEqual(json.loads(r.read())["data"][0]["id"], "qwen3.8-27b")
        self.assertEqual(r.getheader("X-Frame-Options"), "DENY")
        c, r = self.request("GET", "/nope")
        self.assertEqual(r.status, 404)

    def test_loading_state(self):
        self.start()
        self.state.runner = None
        c, r = self.request("GET", "/health")
        self.assertEqual((r.status, json.loads(r.read())["status"]), (503, "loading"))
        status, body = self.chat()
        self.assertEqual(status, 503)

    def test_completion(self):
        self.start()
        self.stub.push(self.ids("Thinking it over.") + [self.fmt.think_end] + self.ids("\n\nThe answer is 4.") + [EOS])
        status, d = self.chat(max_tokens=200, seed=5)
        self.assertEqual(status, 200)
        m = d["choices"][0]["message"]
        self.assertEqual((m["content"], m["reasoning_content"]), ("The answer is 4.", "Thinking it over."))
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        u = d["usage"]
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], 0)
        t = d["timings"]
        self.assertEqual((t["prompt_n"], t["predicted_n"]), (u["prompt_tokens"], u["completion_tokens"]))
        # the stub proposes 4 tokens per pass after the prompt pass and confirms all but the last token it writes in a pass
        passes = t["draft_n"] // 4
        self.assertEqual((t["draft_n"] % 4, t["draft_n_accepted"]), (0, u["completion_tokens"] - 1 - passes))
        self.assertEqual(self.stub.opt(self.engine, 13), 5)

    def test_streaming(self):
        self.start()
        self.stub.push(self.ids("Let me see.") + [self.fmt.think_end] + self.ids("\n\nHello, world.") + [EOS])
        self.stub.delay_us(2000)
        c, r = self.request("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(r.status, 200)
        self.assertIn("text/event-stream", r.getheader("Content-Type"))
        raw = r.read().decode("utf-8")
        events = [e[len("data: "):] for e in raw.split("\n\n") if e.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        objs = [json.loads(e) for e in events[:-1]]
        self.assertEqual(objs[0]["choices"][0]["delta"], {"role": "assistant"})
        reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o["choices"])
        content = "".join(o["choices"][0]["delta"].get("content", "") for o in objs if o["choices"])
        self.assertEqual((reasoning, content), ("Let me see.", "Hello, world."))
        finals = [o for o in objs if o["choices"] and o["choices"][0]["finish_reason"]]
        self.assertEqual(finals[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(objs[-1]["choices"], [])
        self.assertEqual(objs[-1]["usage"]["completion_tokens"] > 0, True)
        self.assertEqual(objs[-1]["timings"]["predicted_n"], objs[-1]["usage"]["completion_tokens"])
        self.assertEqual([o for o in objs[:-1] if "timings" in o], [])          # only the last chunk carries them

    def test_streaming_without_usage_has_timings_in_the_last_chunk(self):
        self.start()
        self.stub.push(self.ids("Let me see.") + [self.fmt.think_end] + self.ids("\n\nHello, world.") + [EOS])
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        raw = r.read().decode("utf-8")
        objs = [json.loads(e[len("data: "):]) for e in raw.split("\n\n") if e.startswith("data: ") and not e.endswith("[DONE]")]
        self.assertEqual(objs[-1]["choices"][0]["finish_reason"], "stop")    # as the llama.cpp server: the finish chunk, no usage chunk
        self.assertNotIn("usage", objs[-1])
        self.assertGreater(objs[-1]["timings"]["predicted_n"], 0)
        self.assertEqual([o for o in objs[:-1] if "timings" in o], [])

    def test_stream_has_one_chunk_per_token(self):
        self.start()
        text = "The quick brown fox jumps over the lazy dog, again and again."
        answer = self.ids(text)
        self.stub.push([self.fmt.think_end] + answer + [EOS])        # the stub writes one to four ids per pass
        self.stub.delay_us(3000)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        raw = r.read().decode("utf-8")
        objs = [json.loads(e[len("data: "):]) for e in raw.split("\n\n") if e.startswith("data: ") and not e.endswith("[DONE]")]
        pieces = [o["choices"][0]["delta"]["content"] for o in objs if o["choices"] and o["choices"][0]["delta"].get("content")]
        self.assertEqual("".join(pieces), text)
        self.assertEqual(len(pieces), len(answer))                   # a client that counts chunks counts tokens

    def test_rejections(self):
        self.start()
        for body, text in (({"messages": []}, "messages"), ({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}, "tools"),
                           ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n must"),
                           ({"messages": [{"role": "user", "content": "x"}], "temperature": 5}, "temperature"),
                           ({"messages": [{"role": "user", "content": "x"}], "stop": ["a"] * 5}, "stop"),
                           ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, "text content")):
            status, d = self.chat(**body)
            self.assertEqual(status, 400, body)
            self.assertIn(text, d["error"]["message"])
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        c.request("POST", "/v1/chat/completions", body=b"", headers={"Content-Length": str(64 * 1024 * 1024), "Host": "127.0.0.1:%d" % self.port})
        self.assertEqual(c.getresponse().status, 413)                # refused from the header alone (the body is never read)
        c.close()

    def test_bad_json(self):
        self.start()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        c.request("POST", "/v1/chat/completions", body=b"{not json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        c.close()

    def test_host_and_origin_checks(self):
        self.start()
        c, r = self.request("GET", "/health", host="evil.example:%d" % self.port)
        self.assertEqual(r.status, 403)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Origin": "http://evil.example"})
        self.assertEqual(r.status, 403)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status, 403)

    def test_api_key(self):
        self.start(api_key="s3cret")
        self.stub.push([1000, EOS])
        status, d = self.chat()
        self.assertEqual(status, 401)
        c, r = self.request("GET", "/v1/models")
        self.assertEqual(r.status, 401)
        c, r = self.request("GET", "/health")                    # health needs no key
        self.assertEqual(r.status, 200)
        c, r = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]},
                            {"Authorization": "Bearer s3cret"})
        self.assertEqual(r.status, 200)
        r.read()

    def test_engine_failure_is_a_500(self):
        self.start()
        # a prompt whose first id is 7 makes the stub fail; render by raw messages is not possible, so fail via the runner
        orig = self.state.runner.engine.generate

        def failing(*a, **k):
            from yasakani._native import YasakaniError
            raise YasakaniError("CUDA error: something")
        self.state.runner.engine.generate = failing
        status, d = self.chat()
        self.assertEqual(status, 500)
        self.assertIn("CUDA error", d["error"]["message"])
        self.state.runner.engine.generate = orig

    def test_client_disconnect_stops_the_engine(self):
        self.start()
        self.stub.push(self.ids("endless text goes on ") * 3, loop_from=0)
        self.stub.delay_us(1000)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 1_000_000,
                           "chat_template_kwargs": {"enable_thinking": False}, "loop_guard": False}).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % (self.port, len(body)) + body)
        got = b""
        while got.count(b"data: ") < 3:                         # headers, the role chunk and some text
            got += s.recv(4096)
        s.close()                                              # the client goes away mid-stream
        t0 = time.time()
        self.stub.push([1000, EOS])
        status = None
        while time.time() - t0 < 10:
            c2, r2 = self.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "again"}], "max_tokens": 20})
            status = r2.status
            r2.read()
            c2.close()
            if status == 200:
                break
        self.assertEqual(status, 200)                           # the engine was cancelled and the lock released
        self.assertLess(time.time() - t0, 8)


if __name__ == "__main__":
    unittest.main()
