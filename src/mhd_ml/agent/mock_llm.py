"""A local mock of an OpenAI-compatible /chat/completions server (tests and offline dry runs).

    $MHD_ML_PY -m mhd_ml.agent.mock_llm --port 8480 --delay 0.4     # answers like a perfect model

By default each reply is the keyword grammar's Intent for the user message, as JSON, so a dry run of
the whole agent behaves sensibly without a network. Per-model behaviour can be scripted for tests:

    MockLLM(port=8480, behaviours={"glimmer": {"delay_s": 3.0}, "muse-spark-1.3": {"mode": "invalid"}})

Modes: "valid" (default), "invalid" (prose, no JSON), "partial" (truncated JSON), "error" (HTTP 500),
"fixed" (return behaviour["content"] verbatim). Streaming (SSE) and non-streaming are both supported.
Ports: the agent workstream owns 8480-8489 for mock LLM servers.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class MockLLM:
    def __init__(self, port: int = 8480, host: str = "127.0.0.1", behaviours: Optional[dict] = None,
                 default: Optional[dict] = None):
        self.host, self.port = host, port
        self.behaviours = behaviours or {}
        self.default = default or {"delay_s": 0.0, "mode": "valid"}
        self.requests: list[dict] = []
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def behaviour_for(self, model: str) -> dict:
        b = dict(self.default)
        for key, val in self.behaviours.items():
            if key == model or key in model:
                b.update(val)
        return b

    def reply_content(self, body: dict, b: dict) -> str:
        from .intent import keyword_intent

        user = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        mode = b.get("mode", "valid")
        if mode == "fixed":
            return b.get("content", "")
        if mode == "invalid":
            return "Sure! I would be happy to help you follow someone."
        js = keyword_intent(user).model_dump_json()
        if mode == "partial":
            return js[: len(js) // 2]
        return b.get("prefix", "") + js

    def start(self) -> "MockLLM":
        from . import intent  # noqa: F401  (import before serving: a cold import on the first request took >2 s)

        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    data = json.dumps({"data": [{"id": "mock"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self.send_error(404)
                    return
                rec = {"path": self.path, "body": body, "auth": bool(self.headers.get("Authorization")),
                       "t": time.monotonic()}
                mock.requests.append(rec)
                b = mock.behaviour_for(body.get("model", ""))
                time.sleep(float(b.get("delay_s", 0.0)))
                if b.get("mode") == "error":
                    data = b'{"error": {"message": "mock failure"}}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                content = mock.reply_content(body, b)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    step = max(1, int(b.get("chunk_chars", 12)))
                    try:
                        for i in range(0, len(content), step):
                            chunk = {"id": "mock", "object": "chat.completion.chunk", "model": body.get("model"),
                                     "choices": [{"index": 0, "delta": {"content": content[i:i + step]},
                                                  "finish_reason": None}]}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(float(b.get("chunk_delay_s", 0.0)))
                        end = {"id": "mock", "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 40}}
                        self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass   # the client stops reading once it has a valid JSON object
                    self.close_connection = True
                    return
                data = json.dumps({"id": "mock", "object": "chat.completion", "model": body.get("model"),
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 40}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="mock-llm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock OpenAI-compatible intent server")
    ap.add_argument("--port", type=int, default=8480)
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    m = MockLLM(port=args.port, default={"delay_s": args.delay, "mode": "valid"}).start()
    print(f"mock LLM on {m.base_url} (Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        m.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
