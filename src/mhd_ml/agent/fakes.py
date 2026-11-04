"""Local stand-ins for tests and dry runs (loopback only).

- FakeRealtimeASR: speaks the Muse Voice Transcribe realtime protocol (handshake, ack, binary PCM,
  partials, endStream -> final, close 1000). Ports 8485-8489.
- FakeBackend: the agent-facing part of the drone backend: WS /ws/agent?role=agent|ui (relays
  AgentEvents between roles), POST /command, POST /mission, GET /state. Ports 8400-8409.

    $MHD_ML_PY -m mhd_ml.agent.fakes backend --port 8400     # dry-run backend for the service

Both run their own event loop in a background thread, so synchronous tests can use them directly.
"""
# (no "from __future__ import annotations": FastAPI must resolve the locally imported WebSocket type)

import argparse
import asyncio
import json
import threading
import time
from typing import Optional


class _LoopThread:
    """An asyncio loop on a daemon thread."""

    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self.loop.run_forever, name=name, daemon=True)
        self._t.start()

    def run(self, coro, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._t.join(timeout=3)


class FakeRealtimeASR:
    """Fake wss://api.meta.ai/v1/asr/realtime on ws://127.0.0.1:<port>."""

    def __init__(self, port: int = 8485, transcript: str = "follow the person in the blue shirt",
                 ack_delay_s: float = 0.0, final_delay_s: float = 0.05, partial_every: int = 5,
                 expected_key: Optional[str] = None, early_final: Optional[str] = None):
        self.port = port
        self.transcript = transcript
        self.ack_delay_s = ack_delay_s
        self.final_delay_s = final_delay_s
        self.partial_every = partial_every
        self.expected_key = expected_key
        self.early_final = early_final          # if set: a final:true with this text after 3 chunks
        self.sessions: list[dict] = []
        self._lt: Optional[_LoopThread] = None
        self._server = None

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handler(self, ws) -> None:
        sess = {"t_open": time.monotonic(), "bytes": 0, "chunks": 0, "chunk_sizes": set(),
                "first_audio": None, "last_audio": None, "ended": False, "handshake": None}
        self.sessions.append(sess)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
        except asyncio.TimeoutError:
            await ws.close(1008, "handshake timeout")
            return
        hs = json.loads(raw)
        token = ((hs.get("authorization") or {}).get("accessToken") or "")
        # Record the handshake without the credential.
        sess["handshake"] = {k: v for k, v in hs.items() if k != "authorization"}
        sess["has_bearer"] = token.startswith("Bearer ") and len(token) > 7
        problems = []
        if not sess["has_bearer"] or (self.expected_key and token != f"Bearer {self.expected_key}"):
            problems.append("bad credential")
        if hs.get("audioEncoding") not in ("PCM_24KHZ", "PCM_16KHZ"):
            problems.append("audioEncoding")
        if hs.get("model") != "muse-voice-transcribe-1.0":
            problems.append("model")
        if problems:
            await ws.send(json.dumps({"type": "error", "message": "; ".join(problems), "sessionId": "x"}))
            await ws.close(1008, "invalid request")
            return
        await asyncio.sleep(self.ack_delay_s)
        sid = f"fake-{len(self.sessions)}"
        await ws.send(json.dumps({"sessionId": sid}))
        words = self.transcript.split()
        async for msg in ws:
            if isinstance(msg, bytes):
                now = time.monotonic()
                sess["bytes"] += len(msg)
                sess["chunks"] += 1
                sess["chunk_sizes"].add(len(msg))
                sess["first_audio"] = sess["first_audio"] or now
                sess["last_audio"] = now
                if self.early_final and sess["chunks"] == 3:
                    await ws.send(json.dumps({"type": "transcript", "transcript": self.early_final,
                                              "final": True, "audioProcessedMs": 240}))
                elif self.partial_every and sess["chunks"] % self.partial_every == 0:
                    k = min(len(words), sess["chunks"] // self.partial_every)
                    await ws.send(json.dumps({"type": "transcript", "transcript": " ".join(words[:k]),
                                              "final": False, "audioProcessedMs": sess["chunks"] * 80}))
                continue
            m = json.loads(msg)
            if m.get("type") == "endStream":
                sess["ended"] = True
                await asyncio.sleep(self.final_delay_s)
                await ws.send(json.dumps({"type": "transcript", "transcript": self.transcript, "final": True,
                                          "audioProcessedMs": sess["chunks"] * 80}))
                await ws.close(1000)
                return

    def start(self) -> "FakeRealtimeASR":
        from websockets.asyncio.server import serve

        self._lt = _LoopThread("fake-asr")

        async def _start():
            self._server = await serve(self._handler, "127.0.0.1", self.port)

        self._lt.run(_start())
        return self

    def stop(self) -> None:
        if self._lt:
            async def _stop():
                self._server.close()
                await self._server.wait_closed()
            try:
                self._lt.run(_stop(), timeout=5)
            finally:
                self._lt.stop()
                self._lt = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class FakeBackend:
    """Agent-facing subset of the drone backend, recording everything it receives."""

    def __init__(self, port: int = 8400, command_delay_s: float = 0.0, fail_http: bool = False,
                 refuse_commands: Optional[str] = None):
        self.port = port
        self.command_delay_s = command_delay_s
        self.fail_http = fail_http
        self.refuse_commands = refuse_commands   # e.g. "estopped": reply 200 {"ok": false, "reason": ...}
        self.commands: list[dict] = []        # POST /command and WS cmd messages, with "via"
        self.missions: list[dict] = []        # POST /mission and WS mission messages, with "via"
        self.agent_msgs: list[dict] = []      # everything the agent sent over the WS
        self.flight_state = "AIRBORNE"
        self._agents: set = set()
        self._uis: set = set()
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def ws_base(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def http_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _apply(self, cmd: dict) -> None:
        c = cmd.get("command")
        if c == "land":
            self.flight_state = "LANDING"
        elif c == "estop":
            self.flight_state = "ESTOPPED"

    def build_app(self):
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse

        fb = self

        @asynccontextmanager
        async def lifespan(app):
            fb.loop = asyncio.get_running_loop()
            yield

        app = FastAPI(lifespan=lifespan)

        @app.post("/command")
        async def command(req: Request):
            if fb.fail_http:
                return JSONResponse({"ok": False}, status_code=503)
            body = await req.json()
            await asyncio.sleep(fb.command_delay_s)
            fb.commands.append(dict(body, via="http", t=time.monotonic()))
            if fb.refuse_commands:
                return {"ok": False, "flight_state": fb.flight_state, "reason": fb.refuse_commands}
            fb._apply(body)
            return {"ok": True, "flight_state": fb.flight_state}

        @app.post("/mission")
        async def mission(req: Request):
            if fb.fail_http:
                return JSONResponse({"ok": False}, status_code=503)
            body = await req.json()
            fb.missions.append(dict(body, via="http", t=time.monotonic()))
            await fb._broadcast(fb._uis, {"type": "mission_ack", "mission_id": body.get("mission_id")})
            return {"ok": True, "mission_id": body.get("mission_id")}

        @app.get("/state")
        async def state():
            return {"type": "telemetry", "flight_state": fb.flight_state}

        @app.websocket("/ws/agent")
        async def ws_agent(ws: WebSocket):
            role = ws.query_params.get("role", "ui")
            await ws.accept()
            peers = fb._agents if role == "agent" else fb._uis
            peers.add(ws)
            try:
                while True:
                    msg = await ws.receive_json()
                    if role == "agent":
                        fb.agent_msgs.append(dict(msg, t=time.monotonic()))
                        if msg.get("type") == "cmd":
                            fb.commands.append(dict(msg, via="ws", t=time.monotonic()))
                            fb._apply(msg)
                        elif msg.get("type") == "mission":
                            fb.missions.append(dict(msg, via="ws", t=time.monotonic()))
                        await fb._broadcast(fb._uis, msg)
                    else:
                        await fb._broadcast(fb._agents, msg)
            except WebSocketDisconnect:
                pass
            finally:
                peers.discard(ws)

        return app

    async def _broadcast(self, peers: set, msg: dict) -> None:
        for p in list(peers):
            try:
                await p.send_json(msg)
            except Exception:
                peers.discard(p)

    def send_to_agents(self, msg: dict, timeout: float = 2.0) -> None:
        """Push a message to every connected agent (as if a UI had sent it)."""
        fut = asyncio.run_coroutine_threadsafe(self._broadcast(self._agents, msg), self.loop)
        fut.result(timeout)

    def agent_count(self) -> int:
        return len(self._agents)

    def start(self) -> "FakeBackend":
        import uvicorn

        cfg = uvicorn.Config(self.build_app(), host="127.0.0.1", port=self.port, log_level="warning",
                             lifespan="on", ws="websockets-sansio" if _has_sansio() else "auto")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, name="fake-backend", daemon=True)
        self._thread.start()
        t0 = time.monotonic()
        while not self._server.started:
            if time.monotonic() - t0 > 10:
                raise RuntimeError("fake backend did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
            self._thread.join(timeout=5)
            self._server = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _has_sansio() -> bool:
    try:
        import uvicorn.protocols.websockets.websockets_sansio_impl  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Fake backend / fake ASR for agent dry runs")
    ap.add_argument("what", choices=["backend", "asr"])
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    if args.what == "backend":
        fb = FakeBackend(port=args.port or 8400).start()
        print(f"fake backend on {fb.http_base} (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(1)
                if fb.commands or fb.missions:
                    pass
        except KeyboardInterrupt:
            fb.stop()
    else:
        fa = FakeRealtimeASR(port=args.port or 8485).start()
        print(f"fake realtime ASR on {fa.url} (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            fa.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
