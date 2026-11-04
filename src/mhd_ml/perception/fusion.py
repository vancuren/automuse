"""Fusion: Kalman filter on bearing and size, association, re-seeding, color score, counter mask.

Everything here is plain numpy/OpenCV and is unit-tested without a GPU (tests/percep).
Boxes are (x1, y1, x2, y2) in 1280x720 pixels; times are capture times in mono ms.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mhd.contracts import FRAME_H, FRAME_W

from .types import WORK_H, WORK_W, iou

Box4 = tuple[float, float, float, float]

# =============================================================================================
# Camera geometry (docs/conventions.md): beta = atan((cx - 640) / f), f = 640 / tan(HFOV / 2).
# =============================================================================================
DEFAULT_HFOV_DEG = float(os.environ.get("MHD_HFOV_DEG", "80"))   # replaced by the Session A value


class Camera:
    def __init__(self, hfov_deg: float = DEFAULT_HFOV_DEG):
        self.hfov_deg = hfov_deg
        self.f = (FRAME_W / 2) / math.tan(math.radians(hfov_deg) / 2)

    def bearing(self, cx: float) -> float:
        return math.atan((cx - FRAME_W / 2) / self.f)

    def cx(self, beta: float) -> float:
        return FRAME_W / 2 + self.f * math.tan(beta)

    def elevation(self, cy: float) -> float:
        """Vertical angle of the box center, positive below the image center."""
        return math.atan((cy - FRAME_H / 2) / self.f)

    def cy(self, el: float) -> float:
        return FRAME_H / 2 + self.f * math.tan(el)


# =============================================================================================
# Constant-velocity Kalman filter on bearing and size, keyed on capture time.
# =============================================================================================
class BoxKalman:
    """State x = [beta, beta_rate, s, s_rate, el, el_rate] with s = ln(box width in px).

    - predict/update are keyed on the frame's capture_ms, never on arrival time, so a burst of
      late frames does not fake a fast subject, and a stall widens the uncertainty correctly.
    - Measurements older than the filter time are refused (returns None): late detector results
      enter through re-seeding instead, which brings them to the present first.
    - The aspect ratio ln(h / w) is smoothed separately (it has no useful dynamics).
    """

    # process noise spectral densities (units^2 / s^3) for (beta, s, el)
    Q = (4.0, 1.0, 2.0)
    # measurement standard deviations (rad, ln px, rad) for tracker and detector measurements
    R_TRACKER = (0.010, 0.06, 0.010)
    R_DETECTOR = (0.006, 0.04, 0.006)
    MAX_DT_S = 2.0

    def __init__(self, cam: Camera, box: Box4, t_ms: float):
        self.cam = cam
        beta, s, el, a = self._measure(box)
        self.x = np.array([beta, 0.0, s, 0.0, el, 0.0], dtype=float)
        self.P = np.diag([0.02 ** 2, 0.5 ** 2, 0.1 ** 2, 0.5 ** 2, 0.02 ** 2, 0.5 ** 2])
        self.aspect = a
        self.t_ms = float(t_ms)
        self.n_updates = 0

    # ----- conversions ---------------------------------------------------------------------
    def _measure(self, box: Box4) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        return (self.cam.bearing((x1 + x2) / 2), math.log(w), self.cam.elevation((y1 + y2) / 2),
                math.log(h / w))

    def _box_from(self, x: np.ndarray) -> Box4:
        cx, cy = self.cam.cx(float(x[0])), self.cam.cy(float(x[4]))
        w = math.exp(float(x[2]))
        h = w * math.exp(self.aspect)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    # ----- filter --------------------------------------------------------------------------
    def _FQ(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        Q = np.zeros((6, 6))
        for i, q in enumerate(self.Q):
            j = 2 * i
            F[j, j + 1] = dt
            Q[j:j + 2, j:j + 2] = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return F, Q

    def predict_state(self, t_ms: float) -> tuple[np.ndarray, np.ndarray]:
        """State and covariance predicted to t_ms (>= filter time), without changing the filter."""
        dt = min(max((t_ms - self.t_ms) / 1000.0, 0.0), self.MAX_DT_S)
        F, Q = self._FQ(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    def predict(self, t_ms: float) -> None:
        if t_ms <= self.t_ms:
            return
        self.x, self.P = self.predict_state(t_ms)
        self.t_ms = float(t_ms)

    def predicted_box(self, t_ms: float) -> Box4:
        x, _ = self.predict_state(max(t_ms, self.t_ms))
        return self._box_from(x)

    def update(self, t_ms: float, box: Box4, kind: str = "tracker", score: float = 1.0) -> Optional[float]:
        """Fuse a measurement taken at capture time t_ms. Returns the innovation's Mahalanobis
        distance, or None if the measurement is older than the filter (refused)."""
        if t_ms < self.t_ms - 1e-6:
            return None
        self.predict(t_ms)
        beta, s, el, a = self._measure(box)
        z = np.array([beta, s, el])
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 2] = H[2, 4] = 1.0
        sd = np.array(self.R_DETECTOR if kind == "detector" else self.R_TRACKER, dtype=float)
        if kind != "detector":
            sd = sd / max(score, 0.25)          # weak tracker scores count less
        R = np.diag(sd ** 2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.aspect = 0.7 * self.aspect + 0.3 * a
        self.n_updates += 1
        return float(math.sqrt(max(y @ np.linalg.solve(S, y), 0.0)))

    def box(self) -> Box4:
        return self._box_from(self.x)

    @property
    def bearing(self) -> float:
        return float(self.x[0])

    @property
    def bearing_rate(self) -> float:
        return float(self.x[1])


# =============================================================================================
# Color score: hue histogram of the upper-body mask region.
# =============================================================================================
COLOR_ALIASES = {
    "grey": "gray", "navy": "blue", "violet": "purple", "magenta": "pink", "crimson": "red",
    "maroon": "red", "scarlet": "red", "teal": "cyan", "turquoise": "cyan", "aqua": "cyan",
    "tan": "brown", "beige": "brown", "gold": "yellow", "silver": "gray", "lime": "green",
}
KNOWN_COLORS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown",
                "white", "gray", "black"}


def normalize_color(color: Optional[str]) -> Optional[str]:
    """Map a color word to one of KNOWN_COLORS, or None when there is no usable color."""
    if not color:
        return None
    c = color.strip().lower()
    c = COLOR_ALIASES.get(c, c)
    return c if c in KNOWN_COLORS else None


def color_match(hsv: np.ndarray, color: str) -> np.ndarray:
    """Boolean per pixel: does it look like `color`? hsv is OpenCV HSV (H 0-179, S, V 0-255)."""
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    if color == "red":
        return ((h <= 8) | (h >= 165)) & (s >= 90) & (v >= 50)
    if color == "orange":
        return (h >= 9) & (h <= 20) & (s >= 100) & (v >= 80)
    if color == "yellow":
        return (h >= 21) & (h <= 34) & (s >= 80) & (v >= 80)
    if color == "green":
        return (h >= 35) & (h <= 85) & (s >= 60) & (v >= 40)
    if color == "cyan":
        return (h >= 86) & (h <= 99) & (s >= 60) & (v >= 50)
    if color == "blue":
        return (h >= 100) & (h <= 130) & (s >= 70) & (v >= 35)
    if color == "purple":
        return (h >= 131) & (h <= 150) & (s >= 50) & (v >= 35)
    if color == "pink":
        return (((h >= 151) & (h <= 172) & (s >= 40) & (v >= 80))
                | (((h <= 10) | (h >= 165)) & (s >= 30) & (s < 90) & (v >= 150)))
    if color == "brown":
        return (h >= 5) & (h <= 22) & (s >= 60) & (v >= 30) & (v <= 150)
    if color == "white":
        return (s <= 40) & (v >= 180)
    if color == "gray":
        return (s <= 40) & (v >= 70) & (v < 180)
    if color == "black":
        return v < 60
    return np.zeros(h.shape, bool)


def region_pixels(img_work: np.ndarray, box: Box4, mask_work: Optional[np.ndarray],
                  category: str = "person") -> np.ndarray:
    """HSV pixels (N, 3) of the color-bearing region of a subject, at work size (640x360).

    People: the upper body (15-55% of the box height, central 70% of its width), inside the mask
    when there is one: the shirt, not the face, legs, or background. Cars and other things: the
    middle band (25-85% of the height). Pixels in the twin counter block are excluded."""
    import cv2

    sx, sy = img_work.shape[1] / FRAME_W, img_work.shape[0] / FRAME_H
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    w, h = x2 - x1, y2 - y1
    if category == "person":
        rx1, rx2, ry1, ry2 = x1 + 0.15 * w, x2 - 0.15 * w, y1 + 0.15 * h, y1 + 0.55 * h
    else:
        rx1, rx2, ry1, ry2 = x1, x2, y1 + 0.25 * h, y1 + 0.85 * h
    H, W = img_work.shape[:2]
    rx1, rx2 = int(np.clip(rx1, 0, W)), int(np.clip(math.ceil(rx2), 0, W))
    ry1, ry2 = int(np.clip(ry1, 0, H)), int(np.clip(math.ceil(ry2), 0, H))
    if rx2 - rx1 < 1 or ry2 - ry1 < 1:
        return np.zeros((0, 3), np.uint8)
    hsv = cv2.cvtColor(img_work[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    keep = np.ones(hsv.shape[:2], bool)
    if mask_work is not None and mask_work.shape[:2] == (H, W):
        m = mask_work[ry1:ry2, rx1:rx2] > 0
        if m.sum() >= 20:                  # a mask that misses the region is worse than none
            keep &= m
    cx1, cy1, cx2, cy2 = (int(v * s) for v, s in zip(counter_roi(), (sx, sy, sx, sy)))
    ys, xs = np.mgrid[ry1:ry2, rx1:rx2]
    keep &= ~((xs >= cx1) & (xs < cx2) & (ys >= cy1) & (ys < cy2))
    return hsv[keep]


def color_fraction(img_work, box, mask_work, color: Optional[str], category: str = "person") -> Optional[float]:
    """Fraction (0..1) of region pixels that match the color word; None if no usable color."""
    c = normalize_color(color)
    if c is None:
        return None
    px = region_pixels(img_work, box, mask_work, category)
    if len(px) < 10:
        return 0.0
    return float(color_match(px[None, :, :], c).mean())


def color_score(frac: Optional[float], full_at: float = 0.35) -> float:
    """Map a color fraction to a ranking factor: 1.0 at >= full_at of the region, down to 0.1.
    No color word -> 1.0 (neutral)."""
    if frac is None:
        return 1.0
    return 0.1 + 0.9 * min(1.0, frac / full_at)


HIST_BINS = 18 + 3     # 18 hue bins for colorful pixels, then white, gray, black


def hue_hist(img_work, box, mask_work, category: str = "person") -> Optional[np.ndarray]:
    """Normalized 21-bin histogram of the region: 18 hue bins (S >= 60, V >= 50) + white/gray/black."""
    px = region_pixels(img_work, box, mask_work, category)
    if len(px) < 10:
        return None
    h, s, v = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    colorful = (s >= 60) & (v >= 50)
    hist = np.zeros(HIST_BINS)
    np.add.at(hist, np.minimum(h[colorful] // 10, 17), 1.0)
    rest = ~colorful
    hist[18] = np.sum(rest & (v >= 180))
    hist[19] = np.sum(rest & (v >= 60) & (v < 180))
    hist[20] = np.sum(rest & (v < 60))
    return hist / hist.sum()


def hist_intersection(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.minimum(a, b).sum())


def size_prior(box: Box4) -> float:
    """Prefer subjects of a followable size: tiny boxes (< 0.5% of the frame) and boxes that fill
    the frame (often a mis-detection of the whole scene) rank lower."""
    a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0) / (FRAME_W * FRAME_H)
    p = float(np.clip(math.sqrt(a / 0.005), 0.3, 1.0))
    return p * (0.7 if a > 0.8 else 1.0)


# =============================================================================================
# Twin counter block masking.
# =============================================================================================
def counter_roi() -> Box4:
    """(x1, y1, x2, y2) in 1280x720 of the twin's counter block, which perception must ignore."""
    try:
        from mhd.twin.counter import COUNTER_ROI  # provided by the twin workstream
        x1, y1, x2, y2 = (float(v) for v in COUNTER_ROI)
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    except Exception:
        pass
    return (FRAME_W - 160.0, FRAME_H - 160.0, float(FRAME_W), float(FRAME_H))


def mask_counter(img: np.ndarray, value: int = 128) -> np.ndarray:
    """Paint the counter block flat gray (in place) on an image of any size; returns the image."""
    sx, sy = img.shape[1] / FRAME_W, img.shape[0] / FRAME_H
    x1, y1, x2, y2 = counter_roi()
    img[int(y1 * sy):int(math.ceil(y2 * sy)), int(x1 * sx):int(math.ceil(x2 * sx))] = value
    return img


def overlaps_counter(box: Box4, frac: float = 0.5) -> bool:
    """True if more than `frac` of the box lies inside the counter block."""
    x1, y1, x2, y2 = counter_roi()
    ix = max(0.0, min(box[2], x2) - max(box[0], x1))
    iy = max(0.0, min(box[3], y2) - max(box[1], y1))
    area = max(box[2] - box[0], 1e-6) * max(box[3] - box[1], 1e-6)
    return ix * iy / area > frac


# =============================================================================================
# Ring buffer of work-size frames, and re-seeding by replay.
# =============================================================================================
@dataclass
class BufferedFrame:
    frame_id: int
    capture_ms: float
    img: np.ndarray          # WORK_W x WORK_H BGR, counter block already masked


class FrameRing:
    """The last `span_ms` of processed frames at 640x360 (about 0.7 MB each)."""

    def __init__(self, span_ms: float = 2000.0, max_frames: int = 90):
        self.span_ms = span_ms
        self._d: deque[BufferedFrame] = deque(maxlen=max_frames)

    def add(self, frame_id: int, capture_ms: float, img: np.ndarray) -> None:
        self._d.append(BufferedFrame(frame_id, capture_ms, img))
        while self._d and capture_ms - self._d[0].capture_ms > self.span_ms:
            self._d.popleft()

    def get(self, frame_id: int) -> Optional[BufferedFrame]:
        for f in reversed(self._d):
            if f.frame_id == frame_id:
                return f
        return None

    def after(self, frame_id: int) -> list[BufferedFrame]:
        return [f for f in self._d if f.frame_id > frame_id]

    def latest(self) -> Optional[BufferedFrame]:
        return self._d[-1] if self._d else None

    def __len__(self) -> int:
        return len(self._d)


def box_offset(a: Box4, b: Box4) -> float:
    """Center distance between two boxes, as a fraction of b's size (sqrt of its area)."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = math.sqrt(max(b[2] - b[0], 1.0) * max(b[3] - b[1], 1.0))
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / size


@dataclass
class ReseedResult:
    tracker: object
    box: Box4                # replayed box on the newest buffered frame
    score: float
    frame_id: int            # the frame `box` belongs to
    capture_ms: float
    steps: int


def reseed(make_tracker: Callable[[], object], ring: FrameRing, det_frame_id: int, det_box: Box4,
           max_steps: int = 30) -> Optional[ReseedResult]:
    """Start a fresh tracker on the detection's own frame, then replay it through the buffered
    frames up to the newest one. Returns None if the frame left the buffer or the replay lost it.

    max_steps caps the replay cost: with more frames than that, every k-th frame is used (always
    ending on the newest)."""
    start = ring.get(det_frame_id)
    if start is None:
        return None
    tr = make_tracker()
    tr.init(start.img, det_box)
    frames = ring.after(det_frame_id)
    if len(frames) > max_steps:
        k = math.ceil(len(frames) / max_steps)
        frames = frames[::-1][::k][::-1]          # every k-th, counted back from the newest
    box, score, fid, t = det_box, 1.0, start.frame_id, start.capture_ms
    for f in frames:
        b, sc = tr.update(f.img)
        if b is None:
            return None
        box, score, fid, t = b, sc, f.frame_id, f.capture_ms
    return ReseedResult(tr, box, score, fid, t, len(frames))


def associate(track_box: Box4, track_hist: Optional[np.ndarray], dets: list, det_hists: list,
              min_iou: float = 0.2, max_offset: float = 0.5) -> Optional[int]:
    """Index of the detection that continues the track, by IoU plus color, or None.

    track_box is the track's box on the detection's own frame (from history), so detector latency
    does not count as motion. A detection passes the gate with IoU >= min_iou or a center offset
    <= max_offset of the box size; among those, the score is IoU + 0.5 x color similarity."""
    best, best_score = None, -1.0
    for i, d in enumerate(dets):
        ov = iou(d.box, track_box)
        off = box_offset(d.box, track_box)
        if ov < min_iou and off > max_offset:
            continue
        sim = hist_intersection(track_hist, det_hists[i] if i < len(det_hists) else None)
        score = ov + 0.5 * (sim if sim is not None else 0.5)
        if score > best_score:
            best, best_score = i, score
    return best


def work_image(img_bgr: np.ndarray, mask_counter_block: bool) -> np.ndarray:
    """Resize any frame to 640x360 (INTER_AREA) and optionally mask the counter block."""
    import cv2

    if img_bgr.shape[1] != WORK_W or img_bgr.shape[0] != WORK_H:
        img = cv2.resize(img_bgr, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr.copy()
    if mask_counter_block:
        mask_counter(img)
    return img
