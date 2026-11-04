"""Deterministic follow autopilot: a pure function the governor calls every RC tick (PLAN.md 4.4)."""
from .core import (  # noqa: F401
    AutopilotRun, autopilot_start, autopilot_step, autopilot_stop, bearing_rad, focal_px, yaw_kp,
    yaw_rate_per_unit,
)
