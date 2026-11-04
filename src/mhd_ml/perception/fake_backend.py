"""A fake drone-backend for perception: serves frames on /ws/frames and records /ws/subject.

For tests and for running perception without the real backend (ports 8300-8309 are perception's):

  $MHD_ML_PY -m mhd_ml.perception.fake_backend --port 8300 --source media/shared/twin/frames \
      --fps 15 --find "person in blue shirt" --color blue --record logs/fake_backend_obs.jsonl

Sources: a folder of images (sorted, looped), a video file (looped), or "synthetic" (a blue-shirt
person walking across a textured background, for tests). Frames are JPEG q85 at 1280x720, framed
with mhd.contracts.framing. /ws/frames honours NextFrame(after_frame_id): it answers with the
newest frame newer than that id, waiting for one if needed (one in flight, newest only).
On each /ws/subject connection it sends the FindRequest (if configured) and records everything
the client sends.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mhd.contracts import FindRequest, FrameMeta, NextFrame, encode_frame, now_ms

log = logging.getLogger("percep.fake_backend")


def synthetic_source(n: int = 600, fps: float = 15.0):
    """Frames of a blue-shirt person walking right then left on a textured background."""
    import cv2

    rng = np.random.default_rng(0)
    gray = rng.integers(70, 160, size=(45, 80)).astype(np.uint8)       # low-saturation texture
    small = np.dstack([gray * 0.9, gray, gray * 0.95]).astype(np.uint8)
    bg = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_CUBIC)

    def frame(i: int):
        t = i / fps
        x = 200 + 300 * (1 + np.sin(t * 0.6))        # 200..800 px, smooth back and forth
        img = bg.copy()
        w, h = 120, 330
        y = 200
        cv2.circle(img, (int(x + w / 2), int(y + 26)), 26, (120, 160, 210), -1)
        cv2.rectangle(img, (int(x), int(y + 0.15 * h)), (int(x + w), int(y + 0.55 * h)), (200, 60, 20), -1)
        cv2.rectangle(img, (int(x + 12), int(y + 0.55 * h)), (int(x + w - 12), int(y + h)), (40, 40, 40), -1)
        return img, (x, y, x + w, y + h)

    return frame


def folder_source(path: str):
    import cv2

    files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(str(Path(path) / f"*.{ext}")))
    if not files:
        raise SystemExit(f"no images in {path}")

    def frame(i: int):
        img = cv2.imread(files[i % len(files)])
        if img.shape[1] != 1280 or img.shape[0] != 720:
            img = cv2.resize(img, (1280, 720))
        return img, None

    return frame


def video_source(path: str):
    import cv2

    cap = cv2.VideoCapture(path)

    def frame(i: int):
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
        if img.shape[1] != 1280 or img.shape[0] != 720:
            img = cv2.resize(img, (1280, 720))
        return img, None

    return frame


class FakeBackend:
    def __init__(self, frame_fn: Callable[[int], tuple], fps: float = 15.0,
                 find: Optional[FindRequest] = None, latency_ms: float = 80.0,
                 record: Optional[str] = None, flight_mode: str = "twin"):
        self.frame_fn = frame_fn
        self.fps = fps
        self.find = find
        self.latency_ms = latency_ms
        self.flight_mode = flight_mode
        self.received: list[dict] = []
        self.truth: dict[int, tuple] = {}
        self._newest: Optional[tuple[FrameMeta, bytes]] = None
        self._cond: Optional[asyncio.Condition] = None
        self._record = open(record, "a") if record else None
        self._subject_clients: set = set()
        self.frames_served = 0

    async def produce(self) -> None:
        import cv2

        self._cond = self._cond or asyncio.Condition()
        i = 0
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while True:
            img, truth = self.frame_fn(i)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            t = now_ms()
            meta = FrameMeta(frame_id=i, capture_ms=t - self.latency_ms, receive_ms=t, width=1280,
                             height=720, render_id=100000 + i, flight_mode=self.flight_mode)
            if truth is not None:
                self.truth[i] = truth
            async with self._cond:
                self._newest = (meta, buf.tobytes())
                self._cond.notify_all()
            i += 1
            next_t += period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    async def _frames(self, ws) -> None:
        self._cond = self._cond or asyncio.Condition()
        async for raw in ws:
            req = NextFrame.model_validate_json(raw)
            async with self._cond:
                await self._cond.wait_for(lambda: self._newest is not None
                                          and self._newest[0].frame_id > req.after_frame_id)
                meta, jpeg = self._newest
            await ws.send(encode_frame(meta, jpeg))
            self.frames_served += 1

    async def _subject(self, ws) -> None:
        self._subject_clients.add(ws)
        try:
            if self.find is not None:
                await ws.send(self.find.model_dump_json())
            async for raw in ws:
                msg = json.loads(raw)
                msg["_rx_ms"] = now_ms()
                self.received.append(msg)
                if self._record:
                    self._record.write(json.dumps(msg) + "\n")
        finally:
            self._subject_clients.discard(ws)

    async def send_subject(self, msg) -> None:
        """Send a FindRequest / Release / Select to every connected perception client."""
        for ws in list(self._subject_clients):
            await ws.send(msg.model_dump_json())

    async def handler(self, ws) -> None:
        path = ws.request.path
        try:
            if path.startswith("/ws/frames"):
                await self._frames(ws)
            elif path.startswith("/ws/subject"):
                await self._subject(ws)
            else:
                await ws.close(code=4404, reason="unknown path")
        except Exception as e:  # client went away
            log.debug("connection %s closed: %s", path, e)

    async def serve(self, host: str = "127.0.0.1", port: int = 8300):
        from websockets.asyncio.server import serve

        self._cond = asyncio.Condition()
        return await serve(self.handler, host, port, max_size=16 * 1024 * 1024)

    def close(self) -> None:
        if self._record:
            self._record.close()
            self._record = None


async def _main(args) -> None:
    if args.source == "synthetic":
        fn = synthetic_source(fps=args.fps)
    elif Path(args.source).is_dir():
        fn = folder_source(args.source)
    else:
        fn = video_source(args.source)
    find = (FindRequest(mission_id=args.mission, subject_phrase=args.find, category=args.category,
                        color=args.color) if args.find else None)
    fb = FakeBackend(fn, fps=args.fps, find=find, record=args.record)
    server = await fb.serve(port=args.port)
    log.info("fake backend on ws://127.0.0.1:%d (frames %s at %.0f fps)", args.port, args.source, args.fps)
    prod = asyncio.create_task(fb.produce())
    try:
        if args.seconds:
            await asyncio.sleep(args.seconds)
        else:
            await asyncio.Future()
    finally:
        prod.cancel()
        server.close()
        fb.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fake backend for perception (ports 8300-8309).")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--source", default="synthetic", help="synthetic, a folder of images, or a video")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--find", default="person in blue shirt")
    ap.add_argument("--category", default="person")
    ap.add_argument("--color", default="blue")
    ap.add_argument("--mission", default="fake-1")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = forever)")
    args = ap.parse_args()
    if not 8300 <= args.port <= 8309:
        raise SystemExit("perception's ports are 8300-8309")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
