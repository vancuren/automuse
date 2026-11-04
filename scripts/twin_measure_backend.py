"""Measure camera-to-backend latency at the backend's /mjpeg output using the twin counter block.

    $MHD_PY scripts/twin_measure_backend.py --backend http://127.0.0.1:8200 --seconds 15

Reads the multipart MJPEG stream, decodes each JPEG, reads the counter (render time on the shared
monotonic clock), and prints p05/p50/p95 of now - render_ms. Works for any backend that serves the
twin's frames (upstream turbodrone or the patched one).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mhd.contracts import now_ms  # noqa: E402
from mhd.twin import counter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8200")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--skip-s", type=float, default=3.0, help="ignore the first seconds (stream warm-up)")
    a = ap.parse_args()
    lat, frames, decoded = [], 0, 0
    buf = b""
    t0 = time.monotonic()
    with httpx.stream("GET", a.backend.rstrip("/") + "/mjpeg", timeout=10) as r:
        for chunk in r.iter_bytes():
            buf += chunk
            while True:
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2)
                if s < 0 or e < 0:
                    break
                jpg, buf = buf[s:e + 2], buf[e + 2:]
                t = now_ms()
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                frames += 1
                d = counter.decode(img, ref_ms=t) if img is not None else None
                if d:
                    decoded += 1
                    if time.monotonic() - t0 > a.skip_s:
                        lat.append(t - d[1])
            if time.monotonic() - t0 > a.seconds:
                break
    q = np.percentile(lat, [5, 50, 95]) if lat else [None] * 3
    print(json.dumps({"frames": frames, "decoded": decoded, "fps": round(frames / a.seconds, 2),
                      "latency_ms": {"p05": round(q[0], 1), "p50": round(q[1], 1), "p95": round(q[2], 1),
                                     "n": len(lat)} if lat else None}))


if __name__ == "__main__":
    main()
