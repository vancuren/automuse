"""The agent's connection to the drone backend.

- WebSocket ws://<backend>/ws/agent?role=agent, reconnecting with backoff (0.5 s doubling to 5 s).
  Incoming JSON (AgentEvent ptt_down/ptt_up relayed from the UI, Select, ...) goes to `on_message`.
  Outgoing: AgentEvent, Mission, OperatorCommand.
- HTTP fallbacks on the same host: POST /command (OperatorCommand) and POST /mission (Mission). Both
  acknowledge, so they are tried first for anything that changes flight; the WebSocket is the fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

from pydantic import BaseModel

log = logging.getLogger("mhd.agent.link")


def http_base_from_ws(ws_base: str) -> str:
    u = urlparse(ws_base)
    scheme = "https" if u.scheme == "wss" else "http"
    return f"{scheme}://{u.netloc}"


class BackendLink:
    def __init__(self, backend: str = "ws://127.0.0.1:8000",
                 on_message: Optional[Callable[[dict], Awaitable[None] | None]] = None,
                 http_timeout_s: float = 1.5):
        b = backend.rstrip("/")
        if b.startswith("http"):
            b = "ws" + b[4:]
        self.ws_base = b
        self.ws_url = f"{b}/ws/agent?role=agent"
        self.http_base = http_base_from_ws(b)
        self.on_message = on_message
        self.http_timeout_s = http_timeout_s
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._connected = asyncio.Event()
        self._stop = False
        self._task: Optional[asyncio.Task] = None
        self._http = None
        self.reconnects = 0

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._connected.is_set()

    async def start(self) -> None:
        import httpx

        self.loop = asyncio.get_running_loop()
        self._http = httpx.AsyncClient(timeout=self.http_timeout_s)
        self._task = asyncio.create_task(self._run(), name="backend-link")

    async def wait_connected(self, timeout: float = 5.0) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _run(self) -> None:
        from websockets.asyncio.client import connect

        delay = 0.5
        while not self._stop:
            try:
                async with connect(self.ws_url, open_timeout=3, ping_interval=10, ping_timeout=10) as ws:
                    self._ws = ws
                    self._connected.set()
                    delay = 0.5
                    log.info("connected to %s", self.ws_url)
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if self.on_message:
                            try:
                                r = self.on_message(msg)
                                if asyncio.iscoroutine(r):
                                    await r
                            except Exception:
                                log.exception("on_message failed for %s", str(msg)[:200])
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stop:
                    log.warning("backend link down (%s); retrying in %.1f s", e.__class__.__name__, delay)
            finally:
                self._ws = None
                self._connected.clear()
            if self._stop:
                break
            self.reconnects += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def send(self, msg: BaseModel | dict) -> bool:
        """Send over the WebSocket. False if not connected or the send failed."""
        ws = self._ws
        if ws is None:
            return False
        data = msg.model_dump_json() if isinstance(msg, BaseModel) else json.dumps(msg)
        try:
            await ws.send(data)
            return True
        except Exception:
            return False

    def send_threadsafe(self, msg: BaseModel | dict, timeout: float = 0.5) -> bool:
        """For other threads (the safety sender): run send() on the link's loop and wait briefly."""
        if self.loop is None or self.loop.is_closed():
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(self.send(msg), self.loop)
            return bool(fut.result(timeout))
        except Exception:
            return False

    async def post(self, path: str, msg: BaseModel) -> Optional[dict]:
        """POST JSON; returns the reply body on HTTP 200, else None."""
        if self._http is None:
            return None
        try:
            r = await self._http.post(self.http_base + path, json=json.loads(msg.model_dump_json()))
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"ok": True}
            log.warning("POST %s -> HTTP %s", path, r.status_code)
        except Exception as e:
            log.warning("POST %s failed: %s", path, e.__class__.__name__)
        return None

    async def deliver(self, path: str, msg: BaseModel) -> dict:
        """Flight-changing messages: HTTP first (acknowledged), then the WebSocket."""
        t0 = time.monotonic()
        body = await self.post(path, msg)
        if body is not None:
            return {"ok": True, "via": "http", "reply": body, "ms": (time.monotonic() - t0) * 1000}
        ok = await self.send(msg)
        return {"ok": ok, "via": "ws" if ok else None, "ms": (time.monotonic() - t0) * 1000}

    async def close(self) -> None:
        self._stop = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._http is not None:
            await self._http.aclose()
