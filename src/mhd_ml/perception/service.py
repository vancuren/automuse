"""Perception service: pulls frames from the backend, finds and tracks the subject, reports evidence.

  $MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 \
      --detector sam_mlx|sam_api --tracker auto|sam_mlx|vit|kcf|csrt

Connections (docs/interfaces.md):
- WS /ws/frames: we send NextFrame {"type":"next","after_frame_id":N} and receive one binary frame
  (mhd.contracts.framing). One request in flight, newest only.
- WS /ws/subject: we receive FindRequest, Release, Select; we send SubjectObs (one per processed
  frame, i.e. camera rate) and Candidates (when look-alikes tie).

Threads: the asyncio loop only does I/O. One "frames" thread decodes JPEGs, runs the tracker and
the fusion (PerceptionCore); one "detector" thread runs SAM (all MLX calls stay on it). The
detector runs at 2 Hz while finding, lost, or in doubt, and 1 Hz while following.
Perception reports evidence only; the autopilot in the backend decides what to do with it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from mhd.contracts import FindRequest, NextFrame, Release, Select, decode_frame, now_ms

from .pipeline import DetJob, PerceptionConfig, PerceptionCore
from .trackers import make_tracker, resolve_tracker_name

log = logging.getLogger("percep.service")


def build_detector(name: str, args):
    if name == "sam_mlx":
        from .detectors.sam_mlx import SamMlxDetector
        return SamMlxDetector(resolution=args.sam_res, threshold=min(0.25, args.min_score))
    if name == "sam_api":
        from .detectors.sam_api import SamApiDetector
        return SamApiDetector(cap_usd=args.spend_cap)
    if name == "fake":
        from .detectors.fake import ColorBlobDetector
        return ColorBlobDetector()
    raise SystemExit(f"unknown detector {name!r}")


def build_tracker_factory(name: str, detector):
    name = resolve_tracker_name(name)
    if name == "sam_mlx":
        engine = getattr(detector, "engine", None)
        if engine is None:
            raise SystemExit("--tracker sam_mlx needs --detector sam_mlx (they share the model)")
        log.warning("SAM 3.1 MLX tracker: about 1 s per step on this Mac (G1); offline use only")
    return name, (lambda: make_tracker(name, engine=getattr(detector, "engine", None)))


class PerceptionService:
    def __init__(self, args, detector=None):
        self.args = args
        self.backend = args.backend.rstrip("/")
        # The detector is built in run(), on the detector thread: MLX streams belong to the thread
        # that created them, so SAM must be loaded and called on one thread (MLX 0.32 raises
        # "There is no Stream(gpu, 1) in current thread" otherwise).
        self.detector = detector
        self.core: Optional[PerceptionCore] = None
        self.frame_pool = ThreadPoolExecutor(1, thread_name_prefix="frames")
        self.det_pool = ThreadPoolExecutor(1, thread_name_prefix="detector")
        self.out_q: asyncio.Queue = asyncio.Queue(maxsize=300)
        self.stop = asyncio.Event()
        self.obs_log = open(args.obs_log, "a") if args.obs_log else None
        self.det_latencies: list[float] = []
        self.sent = 0
        self.frames = 0

    def _setup(self) -> None:
        """Runs on the detector thread: build the detector, the tracker factory, and the core."""
        args = self.args
        if self.detector is None:
            self.detector = build_detector(args.detector, args)
        if args.detector == "sam_api" and not os.environ.get("MODEL_API_KEY"):
            log.error("--detector sam_api but MODEL_API_KEY is not set: nothing will be detected "
                      "(source env.sh, which loads $MHD_HOME/.env)")
        self.tracker_name, factory = build_tracker_factory(args.tracker, self.detector)
        if self.tracker_name == "sam_mlx":
            # The SAM tracker uses the same MLX engine, so frames must run on the detector thread.
            self.frame_pool = self.det_pool
        cfg = PerceptionConfig(min_score=args.min_score, hfov_deg=args.hfov_deg, mask_counter=args.mask_counter,
                               find_period_ms=1000.0 / args.find_hz, follow_period_ms=1000.0 / args.follow_hz)
        self.core = PerceptionCore(cfg, getattr(self.detector, "name", args.detector), factory)
        self.warm_up()

    # ----- helpers --------------------------------------------------------------------------
    def _enqueue(self, msgs) -> None:
        for m in msgs:
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()          # drop the oldest: stale evidence is useless
                except asyncio.QueueEmpty:
                    pass
            self.out_q.put_nowait(m)

    def _process_frame(self, meta, jpeg: bytes):
        import cv2

        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("frame %d: JPEG decode failed", meta.frame_id)
            return []
        if img.shape[1] != 1280 or img.shape[0] != 720:
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        try:
            return self.core.on_frame(meta, img)
        except Exception:                 # a pipeline bug must not drop the connection
            log.exception("frame %d: pipeline error", meta.frame_id)
            return []

    def _fuse(self, job, dets, latency):
        try:
            return self.core.on_detection(job, dets, latency, now_ms())
        except Exception:
            log.exception("fusing a detection failed")
            self.core.detection_failed(job)
            return []

    def _run_detector(self, job: DetJob):
        t0 = time.perf_counter()
        try:
            dets = self.detector.detect(job.image, job.phrase)
            err = getattr(self.detector, "last_error", None)
        except Exception as e:                       # a detector crash must not kill the service
            log.exception("detector failed")
            dets, err = [], repr(e)
        return job, dets, (time.perf_counter() - t0) * 1e3, err

    async def _detect(self, job: DetJob) -> None:
        loop = asyncio.get_running_loop()
        job, dets, latency, err = await loop.run_in_executor(self.det_pool, self._run_detector, job)
        if err:
            log.debug("detector: %s", err)
        self.det_latencies = (self.det_latencies + [latency])[-50:]
        msgs = await loop.run_in_executor(self.frame_pool, self._fuse, job, dets, latency)
        self._enqueue(msgs)

    def _maybe_detect(self) -> None:
        job = self.core.detection_due(now_ms())
        if job is not None:
            asyncio.get_running_loop().create_task(self._detect(job))

    # ----- /ws/frames -----------------------------------------------------------------------
    async def frames_loop(self) -> None:
        import websockets

        loop = asyncio.get_running_loop()
        url = f"{self.backend}/ws/frames"
        while not self.stop.is_set():
            try:
                async with websockets.connect(url, max_size=16 * 1024 * 1024, open_timeout=3,
                                              ping_interval=5, ping_timeout=5) as ws:
                    log.info("connected %s", url)
                    last_id, waits = -1, 0
                    await ws.send(NextFrame(after_frame_id=last_id).model_dump_json())
                    while not self.stop.is_set():
                        try:
                            blob = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        except asyncio.TimeoutError:
                            waits += 1
                            log.warning("no frame for %.0f s", 2.0 * waits)
                            if waits >= 3:
                                break                # reconnect
                            continue
                        waits = 0
                        if isinstance(blob, str):
                            log.warning("frames: unexpected text message %s", blob[:200])
                            await ws.send(NextFrame(after_frame_id=last_id).model_dump_json())
                            continue
                        meta, jpeg = decode_frame(blob)
                        last_id = max(last_id, meta.frame_id)
                        # Ask for the next frame first so the backend can prepare it meanwhile.
                        await ws.send(NextFrame(after_frame_id=last_id).model_dump_json())
                        msgs = await loop.run_in_executor(self.frame_pool, self._process_frame, meta, jpeg)
                        self.frames += 1
                        self._enqueue(msgs)
                        self._maybe_detect()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("frames connection: %s", e)
            if not self.stop.is_set():
                await asyncio.sleep(0.5)

    # ----- /ws/subject ----------------------------------------------------------------------
    def _handle_command(self, raw: str) -> list:
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "find":
                self.core.on_find(FindRequest.model_validate(msg))
            elif kind == "release":
                self.core.on_release(Release.model_validate(msg))
            elif kind == "select":
                return self.core.on_select(Select.model_validate(msg))
            else:
                log.debug("subject: ignoring message type %r", kind)
        except Exception as e:
            log.warning("subject: bad message (%s): %s", e, raw[:200])
        return []

    async def subject_loop(self) -> None:
        import websockets

        url = f"{self.backend}/ws/subject"
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            try:
                async with websockets.connect(url, open_timeout=3, ping_interval=5, ping_timeout=5) as ws:
                    log.info("connected %s", url)

                    async def reader():
                        async for raw in ws:
                            if isinstance(raw, bytes):
                                raw = raw.decode("utf-8", "replace")
                            msgs = await loop.run_in_executor(self.frame_pool, self._handle_command, raw)
                            self._enqueue(msgs)
                            self._maybe_detect()

                    async def writer():
                        while True:
                            m = await self.out_q.get()
                            # Evidence older than 1 s (e.g. queued while disconnected) is not sent.
                            if getattr(m, "capture_ms", None) is not None and now_ms() - m.capture_ms > 1000:
                                continue
                            data = m.model_dump_json(exclude_none=False)
                            await ws.send(data)
                            self.sent += 1
                            if self.obs_log:
                                self.obs_log.write(data + "\n")

                    tasks = [asyncio.create_task(reader()), asyncio.create_task(writer())]
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            if t.exception():
                                raise t.exception()
                    finally:
                        for t in tasks:
                            t.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("subject connection: %s", e)
            if not self.stop.is_set():
                await asyncio.sleep(0.5)

    async def status_loop(self) -> None:
        last_frames, last_t = 0, time.monotonic()
        while not self.stop.is_set():
            await asyncio.sleep(5.0)
            t = time.monotonic()
            fps = (self.frames - last_frames) / (t - last_t)
            last_frames, last_t = self.frames, t
            lat = sorted(self.det_latencies)
            p50 = lat[len(lat) // 2] if lat else float("nan")
            if self.core is None:
                continue
            m = self.core.mission
            tr = self.core.track
            mem = ""
            if getattr(self.detector, "engine", None) is not None:
                try:
                    import mlx.core as mx
                    mem = f", MLX {mx.get_active_memory() / 1e9:.1f} GB"
                except Exception:
                    pass
            log.info("status: %.1f fps, sent %d, detector p50 %.0f ms%s, mission %s, track %s, stats %s",
                     fps, self.sent, p50, mem, m.mission_id if m else None,
                     (f"{tr.track_id} confirms={tr.n_confirms} score={tr.last_score:.2f} lost={tr.lost}"
                      if tr else None), self.core.stats)
            if self.obs_log:
                self.obs_log.flush()

    def warm_up(self) -> None:
        """One detection on a blank frame so the first real one is not slowed by kernel compiles."""
        if self.args.detector != "sam_mlx":
            return
        t0 = time.perf_counter()
        self.detector.detect(np.full((720, 1280, 3), 127, np.uint8), "person")
        log.info("detector warm-up %.1f s", time.perf_counter() - t0)

    async def run(self, install_signals: bool = True) -> None:
        loop = asyncio.get_running_loop()
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self.stop.set)
                except (NotImplementedError, RuntimeError):
                    pass
        await loop.run_in_executor(self.det_pool, self._setup)
        log.info("perception ready: detector %s, tracker %s, backend %s", self.args.detector,
                 self.tracker_name, self.backend)
        tasks = [asyncio.create_task(self.frames_loop()), asyncio.create_task(self.subject_loop()),
                 asyncio.create_task(self.status_loop())]
        await self.stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.close()

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if close:
            close()
        self.frame_pool.shutdown(wait=False, cancel_futures=True)
        self.det_pool.shutdown(wait=False, cancel_futures=True)
        if self.obs_log:
            self.obs_log.close()
            self.obs_log = None
        log.info("perception stopped")


def parse_args(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description="SAM 3.1 perception service (evidence only).")
    ap.add_argument("--backend", default="ws://127.0.0.1:8000")
    ap.add_argument("--detector", default="sam_mlx", choices=["sam_mlx", "sam_api", "fake"])
    ap.add_argument("--tracker", default="auto", choices=["auto", "sam_mlx", "vit", "kcf", "csrt"])
    ap.add_argument("--sam-res", type=int, default=672, help="SAM MLX square input (G1: 672 = 234 ms)")
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--find-hz", type=float, default=2.0, help="detector rate while finding/lost/in doubt")
    ap.add_argument("--follow-hz", type=float, default=1.0, help="detector rate while following")
    ap.add_argument("--hfov-deg", type=float, default=float(os.environ.get("MHD_HFOV_DEG", "80")))
    ap.add_argument("--mask-counter", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--spend-cap", type=float, default=8.0, help="USD cap for sam_api (whole workstream)")
    ap.add_argument("--obs-log", default=None, help="append every message sent to this JSONL file")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    svc = PerceptionService(args)
    asyncio.run(svc.run())


if __name__ == "__main__":
    main()
