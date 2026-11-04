"""Capture-time estimate for X69 RTSP frames (VideoFrame.timestamp upstream is decode time).

Order of preference, per frame:
  1. twin mode: the twin's counter block, if mhd.twin.counter is importable and decodes:
     capture_ms = render_ms (same monotonic clock as the backend).
  2. RTSP presentation time: capture_ms = pts + min over 10 s of (receive - pts) - offset, where
     offset = MHD_CAPTURE_OFFSET_MS (the camera-to-backend minimum latency measured in Session A;
     0 until measured). A burst of late frames after a stall keeps its old pts, so it shows up
     as old, not fresh.
  3. a frame without a usable pts in a stream that had them keeps the last good estimate
     ("pts_hold"), so it can never make old video look fresh.
  4. receive time, only when the stream has never carried a usable pts.

TimedX69RtspAdapter is the upstream X69 RTSP adapter plus: OpenCV open/read timeouts of 2 s, a
thin wrapper around cv2.VideoCapture.read() that records receive time, pts and the counter, and
handle_payload() attaching capture_ms / receive_ms / render_id to each VideoFrame.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Callable, Optional

import cv2

from mhd.contracts import now_ms

logger = logging.getLogger(__name__)

try:  # built by the twin workstream: decode(img_bgr) -> Optional[(render_id, render_ms)]
    from mhd.twin.counter import decode as _counter_decode  # type: ignore
except Exception:  # noqa: BLE001 - absent until the twin lands; any import error means "no counter"
    _counter_decode = None


class CaptureClock:
    """capture = pts + min over a sliding window of (receive - pts), minus a fixed offset."""

    def __init__(self, window_ms: float = 10_000.0, offset_ms: float = 0.0):
        self.window_ms = window_ms
        self.offset_ms = offset_ms
        self._dq: deque[tuple[float, float]] = deque()    # (receive_ms, receive - pts), increasing
        self._last_pts: Optional[float] = None
        self.last_capture: Optional[float] = None

    def reset(self) -> None:
        self._dq.clear()
        self._last_pts = None
        self.last_capture: Optional[float] = None

    @property
    def has_history(self) -> bool:
        return bool(self._dq)

    def update(self, pts_ms: float, receive_ms: float) -> float:
        if self._last_pts is not None and pts_ms < self._last_pts - 100.0:
            self.reset()                                   # new RTSP session: pts restarted
        self._last_pts = pts_ms
        off = receive_ms - pts_ms
        while self._dq and self._dq[-1][1] >= off:
            self._dq.pop()
        self._dq.append((receive_ms, off))
        while len(self._dq) > 1 and self._dq[0][0] < receive_ms - self.window_ms:
            self._dq.popleft()
        self.last_capture = pts_ms + self._dq[0][1] - self.offset_ms
        return self.last_capture


def estimate(img, pts_ms: float, receive_ms: float, clock: CaptureClock, flight_mode: str,
             counter: Optional[Callable] = None) -> tuple[float, Optional[int], str]:
    """Returns (capture_ms, render_id, source)."""
    counter = _counter_decode if counter is None else counter
    if flight_mode == "twin" and counter is not None and img is not None:
        try:
            res = counter(img)
        except Exception:  # noqa: BLE001
            res = None
        if res:
            render_id, render_ms = res
            return float(render_ms), int(render_id), "counter"
    if pts_ms is not None and pts_ms == pts_ms and pts_ms > 0:
        return clock.update(float(pts_ms), receive_ms), None, "pts"
    if clock.last_capture is not None:
        # A frame without a usable pts in a stream that had them (seen right after a read timeout):
        # never let it look fresher than the last good estimate.
        return clock.last_capture, None, "pts_hold"
    return receive_ms, None, "receive"      # stream has never carried pts


def _base_adapter():
    from protocols.x69_lg_rtsp_video_protocol import X69LgRtspVideoProtocolAdapter, _rtsp_capture_options
    return X69LgRtspVideoProtocolAdapter, _rtsp_capture_options


_Base, _rtsp_capture_options = _base_adapter()


class _TimedCapture:
    """Wraps cv2.VideoCapture: after every successful read, records timing for that frame."""

    def __init__(self, cap, on_frame: Callable):
        self._cap = cap
        self._on_frame = on_frame

    def read(self):
        ok, img = self._cap.read()
        if ok and img is not None:
            self._on_frame(img, self._cap)
        return ok, img

    def __getattr__(self, name):
        return getattr(self._cap, name)


class TimedX69RtspAdapter(_Base):
    """Upstream X69 RTSP adapter with capture-time estimates and 2 s OpenCV timeouts."""

    OPEN_TIMEOUT_MS = 2000
    READ_TIMEOUT_MS = 2000

    def __init__(self, *args, flight_mode: str = "twin", **kwargs):
        super().__init__(*args, **kwargs)
        self.flight_mode = flight_mode
        self.capture_clock = CaptureClock(
            window_ms=1000.0 * float(os.getenv("MHD_CAPTURE_WINDOW_S", "10") or 10),
            offset_ms=float(os.getenv("MHD_CAPTURE_OFFSET_MS", "0") or 0))
        self._pending: Optional[tuple] = None
        self.capture_source_counts: dict[str, int] = {}
        self.decoder_threads: Optional[float] = None

    def _open_stream(self) -> bool:
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _rtsp_capture_options()
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.OPEN_TIMEOUT_MS,
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.READ_TIMEOUT_MS]
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG, params)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self._cap = _TimedCapture(cap, self._on_frame)
                self.capture_clock.reset()
                self._last_frame_time = time.time()
                self.decoder_threads = cap.get(cv2.CAP_PROP_N_THREADS)
                logger.info("[x69-lg-rtsp] Stream opened (timed, %d decoder threads): %s",
                            int(self.decoder_threads), self.rtsp_url)
                return True
            cap.release()
            self._cap = None
            logger.warning("[x69-lg-rtsp] Failed to open stream: %s", self.rtsp_url)
            return False

    def _on_frame(self, img, cap) -> None:
        receive = now_ms()
        try:
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:  # noqa: BLE001
            pts = -1.0
        capture, render_id, source = estimate(img, pts, receive, self.capture_clock, self.flight_mode)
        self.capture_source_counts[source] = self.capture_source_counts.get(source, 0) + 1
        h, w = img.shape[:2]
        self._pending = (capture, receive, render_id, w, h, source)

    def handle_payload(self, payload: bytes):
        frame = super().handle_payload(payload)
        pending, self._pending = self._pending, None
        if frame is not None and pending is not None:
            capture, receive, render_id, w, h, source = pending
            frame.capture_ms = capture
            frame.receive_ms = receive
            frame.render_id = render_id
            frame.width, frame.height = w, h
            frame.capture_source = source
        return frame
