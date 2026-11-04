"""A test double for the detector: finds blobs of the requested color by HSV thresholding.

Used by the tests and the fake-backend integration test so they run without SAM, a GPU, or the
network. Its observations carry source "oracle" (the contract has no "fake" source)."""
from __future__ import annotations

import time

import numpy as np

from mhd.contracts import FRAME_H, FRAME_W

from ..fusion import color_match, normalize_color
from ..types import Detection, clip_box, mask_to_polygon


class ColorBlobDetector:
    name = "fake"

    def __init__(self, default_color: str = "blue", min_area_px: int = 1600, delay_s: float = 0.0):
        self.default_color = default_color
        self.min_area_px = min_area_px
        self.delay_s = delay_s
        self.calls = 0

    def detect(self, image_bgr: np.ndarray, phrase: str) -> list[Detection]:
        import cv2

        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        color = next((w for w in phrase.lower().split() if normalize_color(w)), self.default_color)
        small = cv2.resize(image_bgr, (640, 360), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = color_match(hsv, normalize_color(color)).astype(np.uint8)
        n, labels, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        sx, sy = FRAME_W / 640, FRAME_H / 360
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < self.min_area_px / 4:      # min_area_px is in 1280x720 pixels
                continue
            blob = (labels == i).astype(np.uint8)
            # Pretend the blob is a shirt: the "person" box extends above and below it.
            box = clip_box(((x - 0.1 * w) * sx, (y - 0.5 * h) * sy, (x + 1.1 * w) * sx, (y + 2.2 * h) * sy))
            out.append(Detection(box=box, score=0.9, mask_poly=mask_to_polygon(blob, sx, sy),
                                 mask=None, label=phrase))
        out.sort(key=lambda d: -(d.w * d.h))
        return out
