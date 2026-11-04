"""Twin video path: renderer JPEG -> decode -> blur/noise -> counter stamp -> latency buffer
-> ffmpeg (hevc_videotoolbox) -> RTSP publish to mediamtx, plus an RTSP probe that measures
the real end-to-end latency with the same OpenCV reader settings the backend uses.

Latency control. Every frame carries its own timestamps:
    render_ms   state snapshot sent to the renderer (the "capture" instant)
    ready_ms    decoded, noised, and stamped in this process
    release_ms  written to the encoder
    decode_ms   decoded by the probe's RTSP reader (OpenCV, FFmpeg, TCP, one decoder thread)
total = decode_ms - render_ms. The part after release (encode + RTSP + decode) is measured by
the probe; its running median is the "downstream" estimate. A frame is held until
    render_ms + target_total - downstream_estimate
so the injected delay is the target total minus the measured baseline, frame by frame.

Stall-then-burst: during a stall nothing is released; when it ends every held frame goes out
back to back, as a stalled Wi-Fi link would deliver them. mode "drop" discards them instead.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable, Optional

import cv2
import numpy as np

from mhd.contracts import FRAME_H, FRAME_W, now_ms

from . import counter, media

log = logging.getLogger("mhd.twin.video")

# Same options as turbodrone's X69 RTSP adapter, plus one decoder thread (see docs/ws/twin.md:
# FFmpeg frame threading otherwise adds about 0.7 s of HEVC decode delay).
PROBE_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


def open_rtsp_capture(url: str, timeout_ms: int = 4000) -> cv2.VideoCapture:
    """cv2.VideoCapture for the twin stream: CAP_FFMPEG, TCP transport, one decoder thread."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = PROBE_CAPTURE_OPTIONS
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")    # silence "Could not find ref" while joining mid-GOP
    params = [cv2.CAP_PROP_N_THREADS, 1, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


@dataclass
class FrameMetaT:
    render_id: int
    render_ms: float
    arrival_ms: float
    ready_ms: float = 0.0
    due_ms: float = 0.0
    release_ms: float = 0.0
    held_ms: float = 0.0
    stalled: bool = False


@dataclass
class StallPlan:
    every_s: float = 0.0          # 0 = no periodic stalls
    duration_ms: float = 0.0
    mode: str = "burst"           # burst | drop
    next_ms: float = 0.0
    until_ms: float = 0.0
    count: int = 0


class NoiseBank:
    """Precomputed sensor noise (and its negative) so per-frame noise costs two saturating adds."""

    def __init__(self, sigma: float, n: int = 8, seed: int = 7) -> None:
        self.sigma = sigma
        rng = np.random.default_rng(seed)
        self.pos, self.neg = [], []
        if sigma > 0:
            for _ in range(n):
                z = rng.normal(0.0, sigma, (FRAME_H, FRAME_W, 1)).astype(np.float32)
                z = np.repeat(z, 3, axis=2) * np.array([1.0, 0.9, 1.1], np.float32)  # mostly luma noise
                self.pos.append(np.clip(z, 0, 255).astype(np.uint8))
                self.neg.append(np.clip(-z, 0, 255).astype(np.uint8))
        self.i = 0

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.pos or img.shape[:2] != (FRAME_H, FRAME_W):
            return img
        k = self.i % len(self.pos)
        self.i += 1
        cv2.add(img, self.pos[k], dst=img)
        cv2.subtract(img, self.neg[k], dst=img)
        return img


class VideoPipeline:
    """Threads: process (decode/noise/stamp), writer (latency buffer -> ffmpeg), probe (RTSP read)."""

    def __init__(self, rtsp_url: str, fps: int = 25, bitrate: str = "3M", gop: int = 25,
                 noise_sigma: float = 3.0, blur_sigma: float = 0.7, probe: bool = True,
                 on_probe_frame: Optional[Callable[[int, float, np.ndarray], None]] = None) -> None:
        self.url = rtsp_url
        self.fps = fps
        self.bitrate = bitrate
        self.gop = gop
        self.blur_sigma = blur_sigma
        self.noise = NoiseBank(noise_sigma)
        self.probe_enabled = probe
        self.on_probe_frame = on_probe_frame
        self.target_total_ms: Optional[float] = None
        self.stall = StallPlan()
        self._in_q: queue.Queue = queue.Queue(maxsize=3)
        self._buf: deque = deque()
        self._cond = threading.Condition()
        self._ff: Optional[subprocess.Popen] = None
        self._ff_lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._meta_by_id: dict[int, FrameMetaT] = {}
        self._meta_order: deque = deque(maxlen=2000)
        # statistics
        self.downstream_ms: deque = deque(maxlen=150)   # decode - release
        self.total_ms: deque = deque(maxlen=300)        # decode - render (non-stalled frames)
        self.baseline_ms: deque = deque(maxlen=300)     # total - held
        self.upstream_ms: deque = deque(maxlen=300)     # ready - render
        self.processed = 0
        self.dropped_in = 0
        self.released = 0
        self.stall_dropped = 0
        self.encoder_restarts = 0
        self.probe_frames = 0
        self.probe_decoded = 0
        self.probe_reopens = 0
        self.decode_ok_recent: deque = deque(maxlen=500)  # steady-state decode success (after first good frame)
        self.release_times: deque = deque(maxlen=200)
        self.probe_times: deque = deque(maxlen=200)
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_probe_bgr: Optional[np.ndarray] = None
        self.frame_log: deque = deque(maxlen=5000)       # per-frame timing records (for tests/logs)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._running = True
        for target, name in ((self._process_loop, "twin-process"), (self._writer_loop, "twin-writer")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)
        if self.probe_enabled:
            th = threading.Thread(target=self._probe_loop, name="twin-probe", daemon=True)
            th.start()
            self._threads.append(th)

    def begin_stop(self) -> None:
        """Stop accepting work and end the encoder, so the RTSP reader unblocks quickly."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        with self._ff_lock:
            media.stop_process(self._ff)
            self._ff = None

    def join(self) -> None:
        for th in self._threads:
            th.join(timeout=5.0)

    def stop(self) -> None:
        self.begin_stop()
        self.join()

    # ------------------------------------------------------------------ input
    def submit(self, render_id: int, render_ms: float, jpeg: Optional[bytes] = None,
               bgr: Optional[np.ndarray] = None) -> None:
        """Called from the asyncio thread; never blocks. Drops the oldest input if processing lags."""
        meta = FrameMetaT(render_id=render_id, render_ms=render_ms, arrival_ms=now_ms())
        item = (meta, jpeg, bgr)
        try:
            self._in_q.put_nowait(item)
        except queue.Full:
            try:
                self._in_q.get_nowait()
                self.dropped_in += 1
            except queue.Empty:
                pass
            try:
                self._in_q.put_nowait(item)
            except queue.Full:
                self.dropped_in += 1

    def set_target(self, total_ms: Optional[float]) -> None:
        self.target_total_ms = None if total_ms is None or total_ms <= 0 else float(total_ms)

    def set_stall(self, spec: Optional[dict]) -> None:
        """{"every_s": 20, "duration_ms": 1500, "mode": "burst"} periodic; {"duration_ms": 1500} once now;
        {} or {"every_s": 0} with no duration cancels."""
        spec = spec or {}
        now = now_ms()
        dur = float(spec.get("duration_ms", 0) or 0)
        every = float(spec.get("every_s", 0) or 0)
        mode = spec.get("mode", "burst")
        self.stall.mode = mode if mode in ("burst", "drop") else "burst"
        self.stall.duration_ms = dur
        self.stall.every_s = every
        if dur > 0 and every <= 0:
            self.stall.until_ms = now + dur          # one-shot, starting now
            self.stall.count += 1
            self.stall.next_ms = 0.0
        elif dur > 0:
            self.stall.next_ms = now + every * 1000.0
        else:
            self.stall.until_ms = 0.0
            self.stall.next_ms = 0.0

    def downstream_estimate(self) -> float:
        if len(self.downstream_ms) >= 5:
            return median(self.downstream_ms)
        return 60.0

    # ------------------------------------------------------------------ threads
    def _process_loop(self) -> None:
        while self._running:
            try:
                meta, jpeg, bgr = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                img = bgr if bgr is not None else cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[1] != FRAME_W or img.shape[0] != FRAME_H:
                    img = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                if not img.flags.writeable or not img.flags.c_contiguous:
                    img = np.ascontiguousarray(img).copy()
                if self.blur_sigma > 0:
                    img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)
                self.noise.apply(img)
                counter.stamp(img, meta.render_id, meta.render_ms)
                meta.ready_ms = now_ms()
                tgt = self.target_total_ms
                meta.due_ms = meta.render_ms + tgt - self.downstream_estimate() if tgt else 0.0
                self.processed += 1
                self.latest_bgr = img
                with self._cond:
                    self._buf.append((meta, img))
                    self._cond.notify_all()
            except Exception:
                log.exception("frame processing failed")

    def _update_stall(self, now: float) -> None:
        st = self.stall
        if st.every_s > 0 and st.duration_ms > 0 and st.next_ms and now >= st.next_ms:
            st.until_ms = now + st.duration_ms
            st.next_ms = now + st.every_s * 1000.0
            st.count += 1

    def stalled(self, now: Optional[float] = None) -> bool:
        return (now or now_ms()) < self.stall.until_ms

    def _ensure_ffmpeg(self) -> Optional[subprocess.Popen]:
        with self._ff_lock:
            if self._ff is not None and self._ff.poll() is None:
                return self._ff
            if self._ff is not None:
                self.encoder_restarts += 1
                log.warning("ffmpeg exited with %s; restarting", self._ff.returncode)
            cmd = media.ffmpeg_publish_cmd(self.url, FRAME_W, FRAME_H, self.fps, self.bitrate, self.gop)
            self._ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            return self._ff

    def _writer_loop(self) -> None:
        while self._running:
            with self._cond:
                if not self._buf:
                    self._cond.wait(timeout=0.05)
                    continue
                meta, img = self._buf[0]
            now = now_ms()
            self._update_stall(now)
            in_stall = now < self.stall.until_ms
            if in_stall and self.stall.mode == "drop":
                with self._cond:
                    self._buf.popleft()
                self.stall_dropped += 1
                continue
            gate = max(meta.due_ms, self.stall.until_ms if in_stall else 0.0)
            if now < gate:
                time.sleep(min(gate - now, 5.0) / 1000.0)
                continue
            with self._cond:
                self._buf.popleft()
            ff = self._ensure_ffmpeg()
            try:
                ff.stdin.write(img.data)
                ff.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                time.sleep(0.3)
                continue
            meta.release_ms = now_ms()
            meta.held_ms = max(0.0, meta.release_ms - meta.ready_ms)
            # Late = released well after it was due (held by a stall, or the writer fell behind).
            # Late frames are excluded from the latency statistics that drive the injection.
            meta.stalled = meta.release_ms > max(meta.due_ms, meta.ready_ms) + 30.0
            self.released += 1
            self.release_times.append(meta.release_ms)
            self._meta_by_id[meta.render_id] = meta
            self._meta_order.append(meta.render_id)
            while len(self._meta_by_id) > 1500:
                self._meta_by_id.pop(self._meta_order.popleft(), None)

    def _probe_loop(self) -> None:
        time.sleep(1.0)
        while self._running:
            cap = open_rtsp_capture(self.url)
            if not cap.isOpened():
                cap.release()
                self.probe_reopens += 1
                time.sleep(1.0)
                continue
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = now_ms()
                self.probe_frames += 1
                self.probe_times.append(t)
                d = counter.decode(frame, ref_ms=t)
                self.latest_probe_bgr = frame
                if d is None:
                    if self.probe_decoded:            # frames before the first keyframe do not count
                        self.decode_ok_recent.append(False)
                    continue
                self.probe_decoded += 1
                self.decode_ok_recent.append(True)
                rid, rms = d
                meta = self._meta_by_id.get(rid)
                total = t - rms
                rec = {"render_id": rid, "render_ms": rms, "decode_ms": t, "total_ms": total}
                if meta is not None:
                    down = t - meta.release_ms
                    rec.update(ready_ms=meta.ready_ms, release_ms=meta.release_ms, held_ms=meta.held_ms,
                               stalled=meta.stalled, downstream_ms=down)
                    if not meta.stalled:
                        self.downstream_ms.append(down)
                        self.total_ms.append(total)
                        self.baseline_ms.append(total - meta.held_ms)
                        self.upstream_ms.append(meta.ready_ms - meta.render_ms)
                self.frame_log.append(rec)
                if self.on_probe_frame is not None:
                    try:
                        self.on_probe_frame(rid, rms, frame)
                    except Exception:
                        log.exception("probe callback failed")
            cap.release()
            self.probe_reopens += 1
            time.sleep(0.5)

    # ------------------------------------------------------------------ status
    @staticmethod
    def _rate(times: deque, window_ms: float = 2000.0) -> float:
        if len(times) < 2:
            return 0.0
        now = now_ms()
        recent = [t for t in times if now - t <= window_ms]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(1e-3, (recent[-1] - recent[0]) / 1000.0)

    @staticmethod
    def _p(vals, q) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)

    def status(self) -> dict:
        with self._ff_lock:
            ff_alive = self._ff is not None and self._ff.poll() is None
        return {
            "encoder_alive": ff_alive, "encoder_restarts": self.encoder_restarts,
            "processed": self.processed, "released": self.released, "input_dropped": self.dropped_in,
            "buffered": len(self._buf), "stall_dropped": self.stall_dropped,
            "encode_fps": round(self._rate(self.release_times), 2),
            "probe_fps": round(self._rate(self.probe_times), 2),
            "probe_frames": self.probe_frames, "probe_decoded": self.probe_decoded,
            "counter_decode_rate": round(self.probe_decoded / self.probe_frames, 4) if self.probe_frames else None,
            "counter_decode_rate_recent": (round(sum(self.decode_ok_recent) / len(self.decode_ok_recent), 4)
                                           if self.decode_ok_recent else None),
            "target_total_ms": self.target_total_ms,
            "baseline_ms_p50": self._p(self.baseline_ms, 0.5),
            "upstream_ms_p50": self._p(self.upstream_ms, 0.5),
            "downstream_ms_p50": self._p(self.downstream_ms, 0.5),
            "total_ms_p50": self._p(self.total_ms, 0.5), "total_ms_p95": self._p(self.total_ms, 0.95),
            "stall": {"every_s": self.stall.every_s, "duration_ms": self.stall.duration_ms,
                      "mode": self.stall.mode, "count": self.stall.count, "active": self.stalled()},
            "noise_sigma": self.noise.sigma, "blur_sigma": self.blur_sigma,
        }

    def set_noise(self, sigma: Optional[float] = None, blur: Optional[float] = None) -> None:
        if sigma is not None and float(sigma) != self.noise.sigma:
            self.noise = NoiseBank(float(sigma))
        if blur is not None:
            self.blur_sigma = max(0.0, float(blur))


class FileRecorder:
    """Writes BGR frames to an MP4 through ffmpeg (chase-view recording for the demo)."""

    def __init__(self, path: Path, width: int, height: int, fps: int = 25) -> None:
        self.path = path
        self.size = (width, height)
        self.fps = fps
        self.frames = 0
        self._proc = subprocess.Popen(media.ffmpeg_file_cmd(path, width, height, fps), stdin=subprocess.PIPE)
        self._q: queue.Queue = queue.Queue(maxsize=10)
        self._th = threading.Thread(target=self._loop, name="chase-recorder", daemon=True)
        self._th.start()

    def submit_jpeg(self, jpeg: bytes) -> None:
        try:
            self._q.put_nowait(jpeg)
        except queue.Full:
            pass

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            img = cv2.imdecode(np.frombuffer(item, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != self.size:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            try:
                self._proc.stdin.write(img.tobytes())
                self.frames += 1
            except (BrokenPipeError, OSError):
                break

    def close(self) -> None:
        self._q.put(None)
        self._th.join(timeout=5.0)
        media.stop_process(self._proc, timeout=10.0)
