"""Corner counter block: render id and render time, stamped into every twin frame.

The block survives HEVC at about 3 Mb/s because it uses only two luma levels in large cells:
- 64 cells of 16x16 px (in the 1280x720 frame), 16 columns by 4 rows, flush with the
  bottom-right corner and aligned to the 16-px grid the encoder uses.
- Cells are gray (B=G=R), so all information is in luma; chroma subsampling cannot hurt it.
- A 16-px black quiet zone on the left and top isolates the cells from scene content.
- Decoding samples only the central 8x8 px of each cell and thresholds adaptively.

Bit layout (MSB first, 64 bits, row-major from the top-left cell):
    4 bits   magic 0b1011 (format version 1)
    20 bits  render_id mod 2**20       (11.6 hours at 25 fps)
    24 bits  render_ms mod 2**24       (twin clock mhd.contracts.now_ms, 4.66 hours of range)
    16 bits  CRC-16/CCITT-FALSE over the 48 bits above (6 bytes)

The full render_ms is recovered from the modulo field as the value closest to a reference
time (by default now_ms() at decode), which is unambiguous for any latency under 2.3 hours.

Perception should ignore pixels inside COUNTER_ROI (x1, y1, x2, y2 in 1280x720).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from mhd.contracts import FRAME_H, FRAME_W, now_ms

CELL = 16
COLS = 16
ROWS = 4
QUIET = 16
GRID_X0 = FRAME_W - COLS * CELL      # 1024
GRID_Y0 = FRAME_H - ROWS * CELL      # 656
COUNTER_ROI = (GRID_X0 - QUIET, GRID_Y0 - QUIET, FRAME_W, FRAME_H)   # (1008, 640, 1280, 720)

MAGIC = 0b1011
ID_BITS = 20
MS_BITS = 24
ID_MOD = 1 << ID_BITS
MS_MOD = 1 << MS_BITS
LEVEL_0 = 16      # luma for a 0 bit (video black)
LEVEL_1 = 235     # luma for a 1 bit (video white)
MIN_CONTRAST = 60  # below this spread between darkest and brightest cell, there is no block


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final xor)."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _message(render_id: int, render_ms: float) -> int:
    rid = int(render_id) % ID_MOD
    ms = int(round(render_ms)) % MS_MOD
    return (MAGIC << 44) | (rid << 24) | ms


def encode_bits(render_id: int, render_ms: float) -> list[int]:
    """64 bits, MSB first."""
    msg = _message(render_id, render_ms)
    crc = crc16_ccitt(msg.to_bytes(6, "big"))
    word = (msg << 16) | crc
    return [(word >> (63 - i)) & 1 for i in range(64)]


def decode_bits(bits: list[int], ref_ms: Optional[float] = None) -> Optional[tuple[int, float]]:
    if len(bits) != 64:
        return None
    word = 0
    for b in bits:
        word = (word << 1) | (1 if b else 0)
    msg, crc = word >> 16, word & 0xFFFF
    if crc16_ccitt(msg.to_bytes(6, "big")) != crc:
        return None
    if (msg >> 44) != MAGIC:
        return None
    rid = (msg >> 24) & (ID_MOD - 1)
    ms_mod = msg & (MS_MOD - 1)
    ref = now_ms() if ref_ms is None else float(ref_ms)
    # value congruent to ms_mod (mod 2**24) that is closest to the reference time
    k = round((ref - ms_mod) / MS_MOD)
    return rid, float(ms_mod + k * MS_MOD)


def stamp(img_bgr: np.ndarray, render_id: int, render_ms: float) -> np.ndarray:
    """Draw the counter block into img_bgr (in place) and return it.

    Designed for 1280x720; other sizes are stamped at the proportional position.
    """
    h, w = img_bgr.shape[:2]
    sx, sy = w / FRAME_W, h / FRAME_H
    x1, y1, x2, y2 = COUNTER_ROI
    img_bgr[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)] = LEVEL_0
    bits = encode_bits(render_id, render_ms)
    for i, bit in enumerate(bits):
        if not bit:
            continue
        r, c = divmod(i, COLS)
        cx0 = GRID_X0 + c * CELL
        cy0 = GRID_Y0 + r * CELL
        img_bgr[int(cy0 * sy):int((cy0 + CELL) * sy), int(cx0 * sx):int((cx0 + CELL) * sx)] = LEVEL_1
    return img_bgr


def read_cells(img_bgr: np.ndarray) -> np.ndarray:
    """Mean luma of the central half of each cell, shape (64,)."""
    h, w = img_bgr.shape[:2]
    sx, sy = w / FRAME_W, h / FRAME_H
    x0, y0 = int(GRID_X0 * sx), int(GRID_Y0 * sy)
    roi = img_bgr[y0:h, x0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    means = np.empty(64, dtype=np.float32)
    q = CELL / 4.0
    for i in range(64):
        r, c = divmod(i, COLS)
        ya = int((r * CELL + q) * sy)
        yb = max(ya + 1, int((r * CELL + CELL - q) * sy))
        xa = int((c * CELL + q) * sx)
        xb = max(xa + 1, int((c * CELL + CELL - q) * sx))
        means[i] = float(gray[ya:yb, xa:xb].mean())
    return means


def decode(img_bgr: np.ndarray, ref_ms: Optional[float] = None) -> Optional[tuple[int, float]]:
    """Return (render_id, render_ms) or None if no valid block is present.

    ref_ms: reference time used to unwrap render_ms (default: now_ms() at the call).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    means = read_cells(img_bgr)
    lo, hi = float(means.min()), float(means.max())
    if hi - lo < MIN_CONTRAST:
        return None
    threshold = (lo + hi) / 2.0
    bits = [1 if m > threshold else 0 for m in means]
    return decode_bits(bits, ref_ms)


def mask_roi(img_bgr: np.ndarray, value: int = 0) -> np.ndarray:
    """Blank the counter block (for perception or display). Works at any frame size."""
    h, w = img_bgr.shape[:2]
    sx, sy = w / FRAME_W, h / FRAME_H
    x1, y1, x2, y2 = COUNTER_ROI
    img_bgr[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)] = value
    return img_bgr
