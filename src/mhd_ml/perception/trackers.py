"""One tracker interface over VitTrack, KCF, CSRT, and the SAM 3.1 MLX tracker.

    tr = make_tracker("vit")
    tr.init(img, box)            # img: BGR at any size (640x360 in the service); box: 1280x720 px
    box, score = tr.update(img)  # box in 1280x720 px or None when lost; score 0..1

Scores are meant to be honest evidence for the autopilot:
- VitTrack: its own tracking score (getTrackingScore), 0 when update() reports failure.
- KCF and CSRT: no native score, so score = (update ok) x (color-histogram similarity between the
  tracked patch now and at init). This catches CSRT's habit of "succeeding" on the background.
- SAM 3.1 MLX: the object presence probability from the multiplex tracker.

G1 (docs/benchmarks-perception.md): VitTrack 2.6 ms, CSRT 14 ms, KCF 26 ms per update at
640x360, all well under 0.20 s; the SAM 3.1 MLX tracker takes about 1 s per step, so "auto"
never picks it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from mhd.contracts import FRAME_W

Box4 = tuple[float, float, float, float]


def vittrack_model_path() -> Path:
    return Path(os.environ.get("MHD_MODELS", "")) / "opencv" / "object_tracking_vittrack_2023sep.onnx"


def _to_xywh(box: Box4, scale: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """1280x720 (x1, y1, x2, y2) -> integer (x, y, w, h) in image pixels, kept inside the image."""
    x1, y1, x2, y2 = (v * scale for v in box)
    x1 = int(np.clip(round(x1), 0, img_w - 2))
    y1 = int(np.clip(round(y1), 0, img_h - 2))
    x2 = int(np.clip(round(x2), x1 + 2, img_w))
    y2 = int(np.clip(round(y2), y1 + 2, img_h))
    return x1, y1, x2 - x1, y2 - y1


def patch_hist(img: np.ndarray, xywh) -> Optional[np.ndarray]:
    """Normalized hue-saturation histogram of the central part of a patch (for appearance checks)."""
    import cv2

    x, y, w, h = (int(v) for v in xywh)
    x0, y0 = max(0, x + w // 6), max(0, y + h // 6)
    x1, y1 = min(img.shape[1], x + w - w // 6), min(img.shape[0], y + h - h // 6)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    s = hist.sum()
    return hist / s if s > 0 else None


def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """1 - Bhattacharyya distance, 0..1 (1 = identical)."""
    import cv2

    if a is None or b is None:
        return 0.0
    return float(np.clip(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA), 0.0, 1.0))


class CvTracker:
    """Wraps an OpenCV tracker; boxes in and out are 1280x720 pixels."""

    def __init__(self, kind: str):
        self.name = kind
        self.kind = kind
        self._t = None
        self._scale = 1.0
        self._ref_hist = None

    def _create(self):
        import cv2

        if self.kind == "vit":
            p = cv2.TrackerVit_Params()
            p.net = str(vittrack_model_path())
            return cv2.TrackerVit_create(p)
        if self.kind == "kcf":
            return cv2.TrackerKCF_create()
        if self.kind == "csrt":
            return cv2.TrackerCSRT_create()
        raise ValueError(f"unknown OpenCV tracker {self.kind!r}")

    def init(self, img: np.ndarray, box: Box4) -> None:
        self._scale = img.shape[1] / FRAME_W
        xywh = _to_xywh(box, self._scale, img.shape[1], img.shape[0])
        self._t = self._create()
        self._t.init(img, xywh)
        self._ref_hist = patch_hist(img, xywh)

    def update(self, img: np.ndarray) -> tuple[Optional[Box4], float]:
        if self._t is None:
            return None, 0.0
        try:
            ok, bb = self._t.update(img)
        except Exception:
            return None, 0.0
        if not ok or bb[2] <= 1 or bb[3] <= 1:
            return None, 0.0
        s = 1.0 / self._scale
        box = (bb[0] * s, bb[1] * s, (bb[0] + bb[2]) * s, (bb[1] + bb[3]) * s)
        if self.kind == "vit":
            score = float(self._t.getTrackingScore())
        else:
            score = hist_similarity(self._ref_hist, patch_hist(img, bb))
        return box, float(np.clip(score, 0.0, 1.0))


class SamMlxTracker:
    """The SAM 3.1 multiplex tracker (about 1 s per step on this Mac: offline use only).

    Needs the full-resolution frame and a SamMlxEngine at 1008. init() runs a detection-free start
    from a box: it asks SAM for the mask inside the box with the box prompt."""

    name = "sam_mlx"

    def __init__(self, engine, phrase: str = "object"):
        self.engine = engine
        self.phrase = phrase
        self._state = None
        self._i = 0

    def init(self, img: np.ndarray, box: Box4) -> None:
        import cv2

        from .types import WORK_H, WORK_W

        full = img if img.shape[1] == FRAME_W else cv2.resize(img, (FRAME_W, 720))
        from .types import iou

        with self.engine.lock:
            prev = self.engine.resolution
            try:
                self.engine.set_resolution(1008)
                feats = self.engine.backbone(self.engine.preprocess(full))
                mask = np.zeros((WORK_H, WORK_W), np.uint8)
                x1, y1, x2, y2 = (int(v / 2) for v in box)
                dets = self.engine.to_detections(self.engine.detect_features(feats, self.phrase), self.phrase)
                best = max(dets, key=lambda d: iou(d.box, box), default=None)
                if best is not None and best.mask is not None and iou(best.box, box) > 0.3:
                    mask = best.mask
                else:
                    mask[y1:y2, x1:x2] = 1
                self._state = self.engine.track_init(feats, [mask])
            finally:
                self.engine.set_resolution(prev)      # the detector must stay at its own size
        self._i = 0

    def update(self, img: np.ndarray) -> tuple[Optional[Box4], float]:
        import cv2

        if self._state is None:
            return None, 0.0
        full = img if img.shape[1] == FRAME_W else cv2.resize(img, (FRAME_W, 720))
        self._i += 1
        with self.engine.lock:
            prev = self.engine.resolution
            try:
                self.engine.set_resolution(1008)
                feats = self.engine.backbone(self.engine.preprocess(full))
                r = self.engine.track_step(self._state, feats, self._i)
            finally:
                self.engine.set_resolution(prev)
        if len(r.scores) == 0 or not r.masks[0].any():
            return None, 0.0
        b = r.boxes[0] * 2.0
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(r.scores[0])


# Replay cost per step (G1): VitTrack 3 ms, CSRT 14-17 ms, KCF 26-29 ms. A re-seed replays up to
# this many steps on the frame thread, so slow trackers get fewer (about 0.2 s worst case).
REPLAY_MAX_STEPS = {"vit": 30, "kcf": 8, "csrt": 12, "sam_mlx": 3}


def resolve_tracker_name(name: str) -> str:
    """auto -> vit when the VitTrack ONNX is present, else kcf."""
    if name == "auto":
        return "vit" if vittrack_model_path().exists() else "kcf"
    return name


def make_tracker(name: str, engine=None, phrase: str = "object"):
    name = resolve_tracker_name(name)
    if name in ("vit", "kcf", "csrt"):
        return CvTracker(name)
    if name == "sam_mlx":
        if engine is None:
            raise ValueError("the sam_mlx tracker needs a SamMlxEngine")
        return SamMlxTracker(engine, phrase)
    raise ValueError(f"unknown tracker {name!r} (auto, vit, kcf, csrt, sam_mlx)")
