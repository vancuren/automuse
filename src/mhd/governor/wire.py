"""Normalized sticks (-1..1) to X69 wire counts (0..255, center 128).

The governor is the only place this conversion happens (docs/conventions.md). Order:
  1. per-axis sign
  2. firmware deadband compensation (autopilot only): sign(u) * (d + |u| * 127) when |u| > threshold
  3. cap, in counts
  4. symmetric rounding (half away from zero), so +u and -u give mirror-image counts

Upstream turbodrone maps +1 -> 255 and -1 -> 0 (half-ranges of 127 and 128) and truncates with int(),
so +0.3 became +38 and -0.3 became -39. Here both sides use 127 counts and the rounding is symmetric.
"""
from __future__ import annotations

import math

from .config import AXES, AxisInts, WireCfg

CENTER = 128


def round_half_away(x: float) -> int:
    """Symmetric rounding: 0.5 -> 1, -0.5 -> -1, 1.49 -> 1."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def axis_offset(u: float, axis: str, wire: WireCfg, cap: int, compensate: bool) -> int:
    """Signed offset from center, in counts, for one axis. NaN or inf give 0."""
    if u != u or u in (math.inf, -math.inf):  # NaN / inf guard: never send garbage
        return 0
    u = max(-1.0, min(1.0, float(u))) * wire.sign.get(axis)
    counts = u * wire.full_scale_counts
    if compensate:
        d = wire.deadband_counts.get(axis)
        if abs(u) <= wire.deadband_threshold:
            counts = 0.0
        elif d:
            counts = math.copysign(d + abs(counts), u)
    counts = max(-cap, min(cap, counts))
    return round_half_away(counts)


def to_wire(sticks: dict[str, float], wire: WireCfg, caps: AxisInts, compensate: bool) -> dict[str, int]:
    """Normalized stick dict -> wire bytes (0..255) for throttle, yaw, pitch, roll."""
    out = {}
    for axis in AXES:
        off = axis_offset(sticks.get(axis, 0.0), axis, wire, caps.get(axis), compensate)
        out[axis] = max(0, min(255, CENTER + off))
    return out


def neutral_wire() -> dict[str, int]:
    return {axis: CENTER for axis in AXES}


def wire_to_normalized(counts: dict[str, int], wire: WireCfg) -> dict[str, float]:
    """Inverse of to_wire without deadband compensation (for telemetry and the wrong-way detector).

    Undoes the per-axis sign, so the result is in the conventions' frame (+yaw = clockwise).
    """
    fs = float(wire.full_scale_counts)
    return {axis: (counts[axis] - CENTER) / fs * wire.sign.get(axis) for axis in AXES}
