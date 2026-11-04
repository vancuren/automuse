"""Capture frames that went through the full twin video path (render -> HEVC -> RTSP -> decode),
each joined to its ground truth by the counter block's render id.

    $MHD_PY -m mhd.twin.capture --twin http://127.0.0.1:8766 --out DIR --count 60 --every 5 [--overlay]

Writes DIR/frame_<render_id>.jpg (quality 95, the decoded RTSP frame, counter included),
DIR/truth.jsonl (one Truth per saved frame), and with --overlay DIR/overlay/<...>.jpg with the
truth boxes drawn, to check that the Python projection matches the rendered pixels.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import httpx
from websockets.sync.client import connect

from mhd.contracts import Truth, now_ms

from . import counter
from .video import open_rtsp_capture


class TruthBuffer:
    """Keeps the last few hundred Truth messages from /ws/truth, by render id."""

    def __init__(self, ws_url: str) -> None:
        self.by_id: dict[int, dict] = {}
        self.order: list[int] = []
        self.lock = threading.Lock()
        self.stop = False
        self.th = threading.Thread(target=self._run, args=(ws_url,), daemon=True)
        self.th.start()

    def _run(self, url: str) -> None:
        while not self.stop:
            try:
                with connect(url, max_size=None, open_timeout=5) as ws:
                    while not self.stop:
                        line = ws.recv(timeout=2.0)
                        d = json.loads(line)
                        with self.lock:
                            self.by_id[d["render_id"]] = d
                            self.order.append(d["render_id"])
                            if len(self.order) > 1000:
                                self.by_id.pop(self.order.pop(0), None)
            except Exception:
                time.sleep(0.5)

    def get(self, rid: int, wait_s: float = 0.5):
        t_end = time.monotonic() + wait_s
        while time.monotonic() < t_end:
            with self.lock:
                if rid in self.by_id:
                    return self.by_id[rid]
            time.sleep(0.01)
        return None


def draw_overlay(img, truth: dict):
    out = img.copy()
    for a in truth["actors"]:
        b = a.get("box")
        if not b:
            continue
        color = (0, 255, 255) if a["visible_frac"] > 0.5 else (0, 128, 255)
        cv2.rectangle(out, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), color, 2)
        cv2.putText(out, f'{a["label"]} vis={a["visible_frac"]:.2f} d={a["distance_m"]:.1f}m',
                    (int(b["x1"]), max(14, int(b["y1"]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    x1, y1, x2, y2 = counter.COUNTER_ROI
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 255), 1)
    return out


def capture(twin: str, out: Path, count: int = 60, every: int = 5, overlay: bool = False,
            timeout_s: float = 120.0) -> dict:
    st = httpx.get(f"{twin}/status", timeout=5).json()
    rtsp = st["rtsp_url"]
    ws_url = twin.replace("http://", "ws://") + "/ws/truth"
    out.mkdir(parents=True, exist_ok=True)
    if overlay:
        (out / "overlay").mkdir(exist_ok=True)
    truths = TruthBuffer(ws_url)
    time.sleep(0.5)
    cap = open_rtsp_capture(rtsp)
    saved, seen, decoded, missing_truth = 0, 0, 0, 0
    t_end = time.monotonic() + timeout_s
    with open(out / "truth.jsonl", "w", encoding="utf-8") as tf:
        while saved < count and time.monotonic() < t_end:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            seen += 1
            d = counter.decode(frame, ref_ms=now_ms())
            if d is None:
                continue
            decoded += 1
            if decoded % every:
                continue
            rid = d[0]
            tr = truths.get(rid)
            if tr is None:
                missing_truth += 1
                continue
            Truth.model_validate(tr)
            cv2.imwrite(str(out / f"frame_{rid:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            tf.write(json.dumps(tr) + "\n")
            if overlay:
                cv2.imwrite(str(out / "overlay" / f"overlay_{rid:06d}.jpg"), draw_overlay(frame, tr),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            saved += 1
    cap.release()
    truths.stop = True
    summary = {"saved": saved, "frames_read": seen, "counter_decoded": decoded, "missing_truth": missing_truth,
               "scenario": st.get("scenario"), "out": str(out)}
    (out / "capture.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Save HEVC-path twin frames with ground truth")
    ap.add_argument("--twin", default="http://127.0.0.1:8766")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--every", type=int, default=5, help="keep one decoded frame in N")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    a = ap.parse_args(argv)
    print(json.dumps(capture(a.twin, Path(a.out), a.count, a.every, a.overlay, a.timeout_s)))


if __name__ == "__main__":
    main()
