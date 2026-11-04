"""Screen counter: measure the REAL drone's camera-to-backend latency (PLAN 4.2, Session A step 3).

1. On the Mac, serve the counter page (loopback only):
       $MHD_PY -m mhd.twin.screen_counter serve --port 8766
   and open http://127.0.0.1:8766/counter full screen. The page shows the twin's 64-bit counter
   code (render id + render time, CRC-16) as an 8x8 grid framed by four ArUco markers, redrawn
   every display frame. Its clock is synced to this process's mhd.contracts.now_ms() (the shared
   monotonic clock) over loopback, so render_ms is on the same clock as the decoder.
2. Point the drone camera at the screen (30-60 cm, the whole block in view) and run
       $MHD_PY -m mhd.twin.screen_counter measure --rtsp <the drone's RTSP URL> --seconds 60 --csv lat.csv
   which decodes every frame (ArUco -> homography -> cells -> CRC) and prints the latency histogram.
   It uses the same OpenCV reader settings as the twin probe (TCP, one decoder thread).

The measured latency includes the display's own latency (compositor + panel, typically 1-3
frames at 60 Hz); --display-ms subtracts a fixed estimate if you have one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mhd.contracts import now_ms

from . import counter

# Page layout, in page units (the page scales it to the screen). The grid is 8x8 cells of CELL;
# marker k is MARKER units square, centered at MARKER_CENTERS[k] (TL, TR, BR, BL).
CELL = 48
GRID = 8 * CELL
MARKER = 96
GAP = 24
_off = GAP + MARKER / 2
MARKER_CENTERS = {0: (-_off, -_off), 1: (GRID + _off, -_off), 2: (GRID + _off, GRID + _off), 3: (-_off, GRID + _off)}
ARUCO_DICT = cv2.aruco.DICT_4X4_50
WEB_DIR = Path(__file__).resolve().parents[3] / "twin" / "web"


def marker_png(marker_id: int, px: int = 240) -> bytes:
    """ArUco marker image (black border included) for the page."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(d, marker_id, px)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    return _detector


def decode_screen(img_bgr: np.ndarray, ref_ms: Optional[float] = None) -> Optional[tuple[int, float]]:
    """Find the four markers, rectify the grid, read the 64 cells, check the CRC."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = _get_detector().detectMarkers(gray)
    if ids is None:
        return None
    centers = {}
    for c, i in zip(corners, ids.flatten()):
        if int(i) in MARKER_CENTERS:
            centers[int(i)] = c.reshape(4, 2).mean(axis=0)
    if len(centers) < 4:
        return None
    src = np.float32([MARKER_CENTERS[k] for k in range(4)])
    dst = np.float32([centers[k] for k in range(4)])
    H = cv2.getPerspectiveTransform(src, dst)
    # sample a 3x3 patch around each cell center (in page units), mapped into the image
    offs = [(-CELL * 0.2, -CELL * 0.2), (0, -CELL * 0.2), (CELL * 0.2, -CELL * 0.2), (-CELL * 0.2, 0), (0, 0),
            (CELL * 0.2, 0), (-CELL * 0.2, CELL * 0.2), (0, CELL * 0.2), (CELL * 0.2, CELL * 0.2)]
    pts = []
    for i in range(64):
        r, c = divmod(i, 8)
        cx, cy = c * CELL + CELL / 2, r * CELL + CELL / 2
        pts += [(cx + dx, cy + dy) for dx, dy in offs]
    mapped = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = gray.shape[:2]
    xs = np.clip(np.round(mapped[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(int), 0, h - 1)
    vals = gray[ys, xs].astype(np.float32).reshape(64, len(offs)).mean(axis=1)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 40:
        return None
    thr = (lo + hi) / 2
    # on screen, a 1 bit is drawn dark (black ink on white): see counter_page.js
    bits = [1 if v < thr else 0 for v in vals]
    return counter.decode_bits(bits, ref_ms)


# ---------------------------------------------------------------- serving the page
def mount(app) -> None:
    """Add /counter, /clock and the marker images to a FastAPI app (the twin server uses this)."""
    from fastapi.responses import FileResponse, JSONResponse, Response

    @app.get("/counter")
    async def counter_page():
        return FileResponse(WEB_DIR / "counter.html", headers={"Cache-Control": "no-store"})

    @app.get("/clock")
    async def clock():
        return JSONResponse({"now_ms": now_ms()}, headers={"Cache-Control": "no-store"})

    @app.get("/counter/marker/{marker_id}.png")
    async def marker(marker_id: int):
        if marker_id not in MARKER_CENTERS:
            return Response(status_code=404)
        return Response(content=marker_png(marker_id), media_type="image/png")

    @app.get("/counter/layout")
    async def layout():
        return {"cell": CELL, "grid": GRID, "marker": MARKER, "gap": GAP,
                "marker_centers": {str(k): v for k, v in MARKER_CENTERS.items()}}


def serve(port: int) -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="screen counter")
    mount(app)
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
    print(f"open http://127.0.0.1:{port}/counter full screen", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------- measuring
def measure(source: str, seconds: float, csv_path: Optional[str], display_ms: float) -> dict:
    from .video import open_rtsp_capture
    if source.startswith("rtsp://"):
        cap = open_rtsp_capture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    rows, frames = [], 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        ok, f = cap.read()
        if not ok:
            if not source.startswith("rtsp://"):
                break
            continue
        t = now_ms()
        frames += 1
        d = decode_screen(f, ref_ms=t)
        if d:
            rows.append({"decode_ms": t, "render_id": d[0], "render_ms": d[1],
                         "latency_ms": t - d[1] - display_ms})
    cap.release()
    lat = np.array([r["latency_ms"] for r in rows])
    out = {"frames": frames, "decoded": len(rows), "display_ms_subtracted": display_ms}
    if lat.size:
        out["latency_ms"] = {f"p{q:02d}": round(float(np.percentile(lat, q)), 1) for q in (5, 25, 50, 75, 95)}
        out["latency_ms"]["min"] = round(float(lat.min()), 1)
        out["latency_ms"]["max"] = round(float(lat.max()), 1)
        hist, edges = np.histogram(lat, bins=np.arange(0, max(100, lat.max() + 50), 50))
        out["histogram_50ms"] = {f"{int(edges[i])}-{int(edges[i + 1])}": int(hist[i]) for i in range(len(hist)) if hist[i]}
    if csv_path:
        with open(csv_path, "w") as fh:
            fh.write("decode_ms,render_id,render_ms,latency_ms\n")
            for r in rows:
                fh.write(f'{r["decode_ms"]:.1f},{r["render_id"]},{r["render_ms"]:.1f},{r["latency_ms"]:.1f}\n')
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Screen counter for camera latency")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the counter page on loopback")
    s.add_argument("--port", type=int, default=8766)
    m = sub.add_parser("measure", help="decode the counter from a camera stream or video file")
    m.add_argument("--rtsp", required=True, help="RTSP URL or a recorded video file")
    m.add_argument("--seconds", type=float, default=60.0)
    m.add_argument("--csv", default=None)
    m.add_argument("--display-ms", type=float, default=0.0, help="subtract this display latency estimate")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.port)
    else:
        print(json.dumps(measure(a.rtsp, a.seconds, a.csv, a.display_ms), indent=2))


if __name__ == "__main__":
    main()
