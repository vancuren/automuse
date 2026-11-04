"""Newest-frame store for WS /ws/frames (pull protocol, docs/interfaces.md).

The frame pump thread calls put(VideoFrame); perception clients await wait_newer(after_id).
Frame ids are assigned here and only ever increase (the upstream adapter restarts its own ids at
1 after every reconnect, which would stall a client asking for "after 5000").
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable, Optional

from mhd.contracts import FrameMeta, now_ms


class FrameStore:
    def __init__(self, flight_mode: str = "twin", on_frame: Optional[Callable[[float, float], None]] = None):
        self.flight_mode = flight_mode
        self.on_frame = on_frame            # e.g. governor.note_frame(capture_ms, receive_ms)
        self._lock = threading.Lock()
        self._latest: Optional[tuple[FrameMeta, bytes]] = None
        self._next_id = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: Optional[asyncio.Event] = None
        self.count = 0
        self.capture_rise_ms: Optional[float] = None   # real-mode pts baseline rise (telemetry)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()

    def put(self, frame) -> FrameMeta:
        """Thread-safe. `frame` is a turbodrone VideoFrame (JPEG bytes in .data)."""
        receive = getattr(frame, "receive_ms", None) or now_ms()
        capture = getattr(frame, "capture_ms", None)
        if capture is None:
            capture = receive
        rise = getattr(frame, "capture_rise_ms", None)
        if isinstance(rise, (int, float)):
            self.capture_rise_ms = float(rise)
        with self._lock:

            self._next_id += 1
            meta = FrameMeta(frame_id=self._next_id, capture_ms=float(capture), receive_ms=float(receive),
                             width=int(getattr(frame, "width", None) or 1280),
                             height=int(getattr(frame, "height", None) or 720),
                             render_id=getattr(frame, "render_id", None), flight_mode=self.flight_mode)
            self._latest = (meta, bytes(frame.data))
            self.count += 1
        if self.on_frame is not None:
            self.on_frame(meta.capture_ms, meta.receive_ms)
        loop, ev = self._loop, self._event
        if loop is not None and ev is not None:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:        # loop closed during shutdown
                pass
        return meta

    def latest(self) -> Optional[tuple[FrameMeta, bytes]]:
        with self._lock:
            return self._latest

    async def wait_newer(self, after_id: int, timeout: float = 1.0) -> Optional[tuple[FrameMeta, bytes]]:
        """Newest frame with id > after_id, waiting up to `timeout` seconds. None on timeout."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            cur = self.latest()
            if cur is not None and cur[0].frame_id > after_id:
                return cur
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            if self._event is None:
                await asyncio.sleep(min(0.01, remaining))
                continue
            self._event.clear()
            cur = self.latest()        # re-check after clear to avoid missing a set
            if cur is not None and cur[0].frame_id > after_id:
                return cur
            try:
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
