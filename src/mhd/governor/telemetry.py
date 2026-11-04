"""Build the contract Telemetry message (backend -> UI at 10 Hz) from the governor's last tick.

Runs on the telemetry thread / event loop, never on the RC thread: the RC tick only swaps in a
small snapshot object, and pydantic validation happens here.
"""
from __future__ import annotations

from typing import Optional

from mhd.contracts import Telemetry, now_ms

from .core import Governor


def build_telemetry(gov: Governor, *, alarm: bool = False, extra_flags: Optional[list] = None,
                    extra_latency: Optional[dict] = None) -> Telemetry:
    snap = gov.snapshot()
    flags = list(snap.safety_flags)
    if gov.flight_mode == "twin" and gov.cfg.twin_headless_bypass:
        flags.append("headless_bypass")
    if alarm:
        flags.append("governor_fault")
    last = gov.last_tick_ts
    now = now_ms()
    if last is None or now - last > 500.0:
        flags.append("rc_tick_stale")
    if extra_flags:
        flags.extend(extra_flags)
    latency = dict(snap.latency_ms)
    if last is not None:
        latency["rc_tick_age"] = round(now - last, 1)
    if extra_latency:
        latency.update({k: float(v) for k, v in extra_latency.items() if v is not None})
    return Telemetry(
        ts_ms=now,
        flight_mode=gov.flight_mode,
        flight_state=gov.state,            # live: reflects a command issued since the last tick
        autopilot_state=snap.autopilot_state,
        armed=gov.armed,
        sticks_wire=dict(snap.wire),
        sticks_by_source={k: {a: float(v) for a, v in d.items()} for k, d in snap.by_source.items()},
        active_source=snap.source,
        frame_age_ms=snap.frame_age_ms,
        obs_age_ms=snap.obs_age_ms,
        latency_ms=latency,
        flight_time_s=snap.flight_time_s,
        pack_id=snap.pack_id,
        safety_flags=flags,
        mission=gov.mission,
        subject=gov.latest_obs,
    )
