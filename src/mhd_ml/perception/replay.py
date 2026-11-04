"""Replay perception over a video file or a folder of frames: SubjectObs JSONL + annotated MP4.

  $MHD_ML_PY -m mhd_ml.perception.replay --input clip.mp4 --phrase "person in blue shirt" \
      --color blue --category person --detector sam_mlx --out logs/replay/clip

Writes <out>/obs.jsonl (one SubjectObs per frame, plus Candidates lines), <out>/annotated.mp4
(H.264 via ffmpeg) and <out>/summary.json. The clock is synthetic (frame i is captured at
i * 1000 / fps ms) and the detector's result reaches the pipeline after its measured latency in
that clock, exactly as it would live, so re-seeding and confirmation ages are realistic.
Optional --truth JSONL ({"frame": i, "box": [x1, y1, x2, y2]} per line, 1280x720 px) adds lock
rate and IoU to the summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from mhd.contracts import Candidates, FindRequest, FrameMeta, SubjectObs

from .benchutil import to_1280x720
from .detectors import build_detector, build_tracker_factory
from .pipeline import PerceptionConfig, PerceptionCore
from .trackers import REPLAY_MAX_STEPS
from .types import iou

log = logging.getLogger("percep.replay")

STATE_COLORS = {"searching": (160, 160, 160), "candidate": (0, 200, 255), "confirmed": (0, 220, 0),
                "lost": (0, 0, 255)}


def iter_frames(src: str, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
    import cv2

    n = 0
    p = Path(src)
    if p.is_dir():
        files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(str(p / f"*.{ext}")))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            yield img if img.shape[:2] == (720, 1280) else to_1280x720(img)
            n += 1
            if max_frames and n >= max_frames:
                return
    else:
        cap = cv2.VideoCapture(src)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img if img.shape[:2] == (720, 1280) else to_1280x720(img)
            n += 1
            if max_frames and n >= max_frames:
                break
        cap.release()


class Mp4Writer:
    """Pipes BGR frames to ffmpeg (libx264, yuv420p) so the MP4 plays everywhere."""

    def __init__(self, path: str, fps: float, w: int = 1280, h: int = 720):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait(timeout=60)


def annotate(img: np.ndarray, obs: Optional[SubjectObs], cands: Optional[Candidates],
             truth=None, phrase: str = "") -> np.ndarray:
    import cv2

    out = img.copy()
    if truth is not None:
        x1, y1, x2, y2 = (int(v) for v in truth)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    if cands is not None:
        for c in cands.items:
            b = c.box
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 255), 2)
            cv2.putText(out, str(c.index), (int(b.x1) + 4, int(b.y1) + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 200, 255), 2)
    if obs is not None:
        color = STATE_COLORS.get(obs.state.value, (255, 255, 255))
        if obs.box is not None:
            b = obs.box
            if obs.mask_poly:
                pts = np.array(obs.mask_poly, np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, color, 1)
            cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 3)
        label = (f"{obs.state.value}  track {obs.track_id}  trk {obs.tracker_score:.2f}  "
                 f"conf {obs.confidence:.2f}  age {min(obs.confirm_age_ms, 99999):.0f} ms  {obs.source.value}")
        cv2.rectangle(out, (0, 0), (1280, 34), (0, 0, 0), -1)
        cv2.putText(out, f"{phrase!r}  {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def load_truth(path: Optional[str], label: Optional[str] = None) -> dict[int, tuple]:
    """Truth boxes by frame index. Two formats:
    - {"frame": i, "box": [x1, y1, x2, y2]} per line;
    - the twin's mhd.contracts.Truth per line (a frame folder's truth.jsonl, in frame order): the
      box of the actor whose label equals `label` (frames where it is off screen are skipped)."""
    if not path:
        return {}
    out = {}
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row.get("type") == "truth":
                for a in row.get("actors", []):
                    if a.get("label") == label and a.get("box"):
                        b = a["box"]
                        out[i] = (b["x1"], b["y1"], b["x2"], b["y2"])
            elif row.get("box"):
                out[int(row["frame"])] = tuple(row["box"])
    return out


def replay(frames, phrase: str, category: str, color: str, detector, tracker: str, fps: float,
           out_dir: str, truth: Optional[dict] = None, det_latency_ms: Optional[float] = None,
           cfg: Optional[PerceptionConfig] = None, write_video: bool = True, flight_mode: str = "real") -> dict:
    """Run the pipeline over frames. det_latency_ms None = use each call's measured latency."""
    os.makedirs(out_dir, exist_ok=True)
    truth = truth or {}
    tname, factory = build_tracker_factory(tracker, detector)
    if cfg is None:
        cfg = PerceptionConfig(replay_max_steps=REPLAY_MAX_STEPS.get(tname, 30))
    core = PerceptionCore(cfg, getattr(detector, "name", "sam_mlx"), factory)
    core.on_find(FindRequest(mission_id="replay", subject_phrase=phrase, category=category, color=color))
    obs_path = Path(out_dir) / "obs.jsonl"
    writer = Mp4Writer(str(Path(out_dir) / "annotated.mp4"), fps) if write_video else None
    pending = []
    n = 0
    det_ms, frame_ms = [], []
    ious, locked, with_truth, first_confirm, switches, last_tid = [], 0, 0, None, 0, None
    locked_after, truth_after = 0, 0          # counted from the first confirmed frame on
    last_cands = None
    t_wall0 = time.perf_counter()
    with open(obs_path, "w") as fo:
        for i, img in enumerate(frames):
            t = i * 1000.0 / fps
            meta = FrameMeta(frame_id=i, capture_ms=t, receive_ms=t, width=1280, height=720,
                             flight_mode=flight_mode)
            t0 = time.perf_counter()
            msgs = core.on_frame(meta, img)
            frame_ms.append((time.perf_counter() - t0) * 1e3)
            for item in [p for p in pending if p[0] <= t]:
                pending.remove(item)
                msgs += core.on_detection(item[1], item[2], item[3], now_ms=t)
            job = core.detection_due(t)
            if job is not None:
                t0 = time.perf_counter()
                dets = detector.detect(job.image, job.phrase)
                lat = (time.perf_counter() - t0) * 1e3
                det_ms.append(lat)
                pending.append((t + (lat if det_latency_ms is None else det_latency_ms), job, dets, lat))
            obs = None
            for m in msgs:
                fo.write(m.model_dump_json() + "\n")
                if isinstance(m, SubjectObs):
                    obs = m
                elif isinstance(m, Candidates):
                    last_cands = m
            if obs is not None:
                if obs.state.value == "confirmed" and first_confirm is None:
                    first_confirm = t
                if obs.track_id is not None and last_tid is not None and obs.track_id != last_tid:
                    switches += 1
                if obs.track_id is not None:
                    last_tid = obs.track_id
                if i in truth:
                    with_truth += 1
                    after = first_confirm is not None
                    truth_after += int(after)
                    if obs.box is not None:
                        v = iou((obs.box.x1, obs.box.y1, obs.box.x2, obs.box.y2), truth[i])
                        ious.append(v)
                        locked += int(v >= 0.5)
                        locked_after += int(after and v >= 0.5)
            if writer:
                writer.write(annotate(img, obs, last_cands if (obs and obs.box is None) else None,
                                      truth.get(i), phrase))
            n += 1
    if writer:
        writer.close()
    summary = {
        "frames": n, "fps": fps, "phrase": phrase, "detector": getattr(detector, "name", "?"),
        "tracker": tname, "detector_calls": len(det_ms),
        "detector_ms_p50": round(float(np.percentile(det_ms, 50)), 1) if det_ms else None,
        "frame_ms_p50": round(float(np.percentile(frame_ms, 50)), 2) if frame_ms else None,
        "time_to_confirm_s": round(first_confirm / 1000.0, 2) if first_confirm is not None else None,
        "track_switches": switches, "stats": core.stats,
        "wall_s": round(time.perf_counter() - t_wall0, 1),
    }
    if with_truth:
        summary.update({"truth_frames": with_truth, "lock_rate": round(locked / with_truth, 3),
                        "lock_rate_after_confirm": (round(locked_after / truth_after, 3)
                                                    if truth_after else None),
                        "mean_iou_when_boxed": round(float(np.mean(ious)), 3) if ious else 0.0})
    with open(Path(out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="video file or folder of frames")
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--category", default="person")
    ap.add_argument("--color", default="none")
    ap.add_argument("--detector", default="sam_mlx", choices=["sam_mlx", "sam_api", "fake"])
    ap.add_argument("--tracker", default="auto", choices=["auto", "sam_mlx", "vit", "kcf", "csrt"])
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default=None, help="truth JSONL (frame/box lines, or a twin truth.jsonl)")
    ap.add_argument("--truth-label", default=None, help="actor label to score in a twin truth.jsonl (default: --phrase)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--det-latency-ms", type=float, default=None,
                    help="fixed detector latency in the synthetic clock (default: measured)")
    ap.add_argument("--sam-res", type=int, default=672)
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--mask-counter", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--flight-mode", default="real", choices=["twin", "real"],
                    help="twin frames get the counter block masked (with --mask-counter auto)")
    ap.add_argument("--spend-cap", type=float, default=8.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    det = build_detector(args.detector, args.sam_res, args.min_score, args.spend_cap)
    cfg = PerceptionConfig(min_score=args.min_score, mask_counter=args.mask_counter)
    try:
        summary = replay(iter_frames(args.input, args.max_frames), args.phrase, args.category, args.color,
                         det, args.tracker, args.fps, args.out,
                         load_truth(args.truth, args.truth_label or args.phrase), args.det_latency_ms,
                         cfg, not args.no_video, args.flight_mode)
    finally:
        close = getattr(det, "close", None)
        if close:
            close()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
