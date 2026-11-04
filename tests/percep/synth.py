"""Synthetic camera frames for perception tests: textured background plus simple "people"
(head, colored shirt, dark trousers) that move. Ground-truth boxes are in 1280x720 pixels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLUE = (200, 60, 20)       # BGR
RED = (30, 30, 210)
GREEN = (40, 170, 40)
SKIN = (120, 160, 210)
TROUSERS = (40, 40, 40)


@dataclass
class Person:
    x: float                 # left edge at t = 0, px
    y: float                 # top edge, px
    vx: float = 0.0          # px per second
    vy: float = 0.0
    w: float = 120.0
    h: float = 330.0
    shirt: tuple = BLUE
    trousers: tuple = TROUSERS

    def box(self, t: float):
        x, y = self.x + self.vx * t, self.y + self.vy * t
        return (x, y, x + self.w, y + self.h)


def background(seed: int = 0) -> np.ndarray:
    """A fixed, textured, mostly gray-green background (trackers need texture)."""
    import cv2

    rng = np.random.default_rng(seed)
    small = rng.integers(70, 150, size=(45, 80, 3)).astype(np.uint8)
    small[..., 0] = (small[..., 0] * 0.8).astype(np.uint8)
    bg = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(-12, 12, size=bg.shape)
    return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)


def render(people, t: float, bg: np.ndarray) -> tuple[np.ndarray, list]:
    import cv2

    img = bg.copy()
    boxes = []
    for p in people:
        x1, y1, x2, y2 = p.box(t)
        boxes.append((x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        head_r = int(w * 0.22)
        cv2.circle(img, (int(cx), int(y1 + head_r)), head_r, SKIN, -1)
        cv2.rectangle(img, (int(x1), int(y1 + 0.15 * h)), (int(x2), int(y1 + 0.55 * h)), p.shirt, -1)
        cv2.rectangle(img, (int(x1 + 0.1 * w), int(y1 + 0.55 * h)), (int(x2 - 0.1 * w), int(y2)), p.trousers, -1)
    return img, boxes


def mask_for(box, shape=(360, 640)) -> np.ndarray:
    """A work-size (640x360) mask covering a 1280x720 box."""
    m = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = (int(round(v / 2)) for v in box)
    m[max(0, y1):y2, max(0, x1):x2] = 1
    return m
