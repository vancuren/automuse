"""Small shared types for perception. Boxes are always in 1280x720 pixels (FRAME_W x FRAME_H)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mhd.contracts import FRAME_H, FRAME_W, Box

# Internal working size for masks, color scoring and the re-seed ring buffer (half of 1280x720).
WORK_W, WORK_H = 640, 360


@dataclass
class Detection:
    """One detector candidate.

    box:       (x1, y1, x2, y2) in 1280x720 pixels.
    score:     detector confidence 0..1.
    mask_poly: simplified outline polygon in 1280x720 pixels (may be empty).
    mask:      optional binary mask (uint8 0/1) at WORK_W x WORK_H, used for the color score.
    label:     the noun phrase that produced it.
    """
    box: tuple[float, float, float, float]
    score: float
    mask_poly: list[tuple[float, float]] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    label: str = ""

    def to_box(self) -> Box:
        x1, y1, x2, y2 = self.box
        return Box(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def w(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def h(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


def clip_box(box, w: float = FRAME_W, h: float = FRAME_H) -> tuple[float, float, float, float]:
    """Clip an (x1, y1, x2, y2) box to the frame and make sure x2 >= x1 and y2 >= y1."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0.0), w), min(max(x2, 0.0), w)))
    y1, y2 = sorted((min(max(y1, 0.0), h), min(max(y2, 0.0), h)))
    return (x1, y1, x2, y2)


def iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def scale_box(box, sx: float, sy: float) -> tuple[float, float, float, float]:
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def mask_to_polygon(mask: np.ndarray, sx: float = 1.0, sy: float = 1.0,
                    max_points: int = 40) -> list[tuple[float, float]]:
    """Outline of the largest blob in a binary mask, simplified, scaled by (sx, sy)."""
    import cv2

    if mask is None or mask.size == 0 or not mask.any():
        return []
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    while len(approx) > max_points:
        eps *= 1.5
        approx = cv2.approxPolyDP(c, eps, True)
    return [(round(float(p[0][0]) * sx, 1), round(float(p[0][1]) * sy, 1)) for p in approx]
