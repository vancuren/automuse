"""Strict loaders for config/governor.yaml and config/autopilot.yaml.

Each YAML file maps onto a tree of frozen dataclasses. Unknown keys and wrong types raise
ConfigError, so a typo in a safety parameter can never silently fall back to a default.
Defaults below mirror the YAML files; the YAML files are the documented source of truth.
"""
from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

AXES = ("throttle", "yaw", "pitch", "roll")


class ConfigError(ValueError):
    pass


def repo_root() -> Path:
    """The checkout this package lives in (MHD_TREE when env.sh was sourced)."""
    env = os.environ.get("MHD_TREE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def logs_dir() -> Path:
    """Where control logs and the flight timer live ($MHD_LOGS, else <repo>/logs)."""
    env = os.environ.get("MHD_LOGS")
    return Path(env) if env else repo_root() / "logs"


# ----------------------------------------------------------------------------------------------
# Small helpers


@dataclass(frozen=True)
class AxisInts:
    throttle: int = 0
    yaw: int = 0
    pitch: int = 0
    roll: int = 0

    def get(self, axis: str) -> int:
        return getattr(self, axis)


@dataclass(frozen=True)
class AxisSigns(AxisInts):
    """Per-axis sign; an axis left out of the YAML keeps +1 (never 0)."""
    throttle: int = 1
    yaw: int = 1
    pitch: int = 1
    roll: int = 1


# ----------------------------------------------------------------------------------------------
# Governor


@dataclass(frozen=True)
class WireCfg:
    full_scale_counts: int = 127
    sign: AxisSigns = AxisSigns()
    deadband_counts: AxisInts = AxisInts(0, 0, 0, 0)
    deadband_threshold: float = 0.01


@dataclass(frozen=True)
class CapsCfg:
    autopilot: AxisInts = AxisInts(throttle=0, yaw=38, pitch=32, roll=0)
    human: AxisInts = AxisInts(throttle=64, yaw=64, pitch=64, roll=64)


@dataclass(frozen=True)
class SlewCfg:
    autopilot_per_s: float = 1.0


@dataclass(frozen=True)
class HumanCfg:
    stale_s: float = 0.25
    deadzone: float = 0.05
    inc_ramp_per_s: float = 1.0


@dataclass(frozen=True)
class HeartbeatCfg:
    neutral_after_s: float = 1.5
    land_after_s: float = 5.0


@dataclass(frozen=True)
class VideoCfg:
    neutral_lag_s: float = 0.7
    land_lag_s: float = 2.0
    floor_window_s: float = 10.0
    floor_cap_s: float = 1.5


@dataclass(frozen=True)
class ObservationCfg:
    stale_min_s: float = 0.4
    stale_periods: float = 2.5
    floor_window_s: float = 10.0
    floor_cap_s: float = 2.0


@dataclass(frozen=True)
class AutopilotGuardCfg:
    max_step_ms: float = 5.0
    fault_neutral_s: float = 1.0


@dataclass(frozen=True)
class TakeoffCfg:
    hold_s: float = 4.0
    packets: int = 1
    require_heartbeat: bool = True


@dataclass(frozen=True)
class LandingCfg:
    reassert_per_s: float = 5.0


@dataclass(frozen=True)
class EstopCfg:
    rearm_after_reset: bool = True


@dataclass(frozen=True)
class ArmingCfg:
    phrase: str = "ARM"


@dataclass(frozen=True)
class FlightTimerCfg:
    warn_s: float = 300.0
    land_s: float = 390.0


@dataclass(frozen=True)
class WrongWayCfg:
    enabled: bool = True
    window_thetas: float = 1.5
    min_change_rad: float = 0.15


@dataclass(frozen=True)
class TiltCfg:
    max_hold_s: float = 3.0


@dataclass(frozen=True)
class FaultsCfg:
    land_after_s: float = 2.0


@dataclass(frozen=True)
class GovernorConfig:
    rc_rate_hz: float = 25.0
    wire: WireCfg = WireCfg()
    caps: CapsCfg = CapsCfg()
    slew: SlewCfg = SlewCfg()
    human: HumanCfg = HumanCfg()
    heartbeat: HeartbeatCfg = HeartbeatCfg()
    video: VideoCfg = VideoCfg()
    observation: ObservationCfg = ObservationCfg()
    autopilot: AutopilotGuardCfg = AutopilotGuardCfg()
    takeoff: TakeoffCfg = TakeoffCfg()
    landing: LandingCfg = LandingCfg()
    estop: EstopCfg = EstopCfg()
    arming: ArmingCfg = ArmingCfg()
    flight_timer: FlightTimerCfg = FlightTimerCfg()
    wrong_way: WrongWayCfg = WrongWayCfg()
    tilt: TiltCfg = TiltCfg()
    faults: FaultsCfg = FaultsCfg()
    twin_headless_bypass: bool = False
    tick_log: bool = True

    def validate(self) -> "GovernorConfig":
        fs = self.wire.full_scale_counts
        if not 1 <= fs <= 127:
            raise ConfigError("wire.full_scale_counts must be 1..127")
        for axis in AXES:
            if self.wire.sign.get(axis) not in (-1, 1):
                raise ConfigError(f"wire.sign.{axis} must be 1 or -1")
            if not 0 <= self.wire.deadband_counts.get(axis) <= 40:
                raise ConfigError(f"wire.deadband_counts.{axis} must be 0..40")
            for src in ("autopilot", "human"):
                cap = getattr(self.caps, src).get(axis)
                if not 0 <= cap <= 127:
                    raise ConfigError(f"caps.{src}.{axis} must be 0..127")
        if self.heartbeat.land_after_s <= self.heartbeat.neutral_after_s:
            raise ConfigError("heartbeat.land_after_s must exceed neutral_after_s")
        if self.video.land_lag_s <= self.video.neutral_lag_s:
            raise ConfigError("video.land_lag_s must exceed neutral_lag_s")
        if self.flight_timer.land_s <= self.flight_timer.warn_s:
            raise ConfigError("flight_timer.land_s must exceed warn_s")
        if self.takeoff.hold_s < 0 or self.takeoff.packets < 1:
            raise ConfigError("takeoff.hold_s >= 0 and takeoff.packets >= 1")
        if self.landing.reassert_per_s <= 0:
            raise ConfigError("landing.reassert_per_s must be > 0")
        if self.human.stale_s <= 0 or self.observation.stale_min_s <= 0:
            raise ConfigError("staleness thresholds must be > 0")
        return self


# ----------------------------------------------------------------------------------------------
# Autopilot


@dataclass(frozen=True)
class CameraCfg:
    hfov_deg: float = 75.0
    frame_w: int = 1280
    frame_h: int = 720


@dataclass(frozen=True)
class PlantCfg:
    yaw_rate_per_count: float = 0.012
    forward_speed_per_count: float = 0.012
    yaw_lag_s: float = 0.2


@dataclass(frozen=True)
class DelayCfg:
    theta_s: float = 1.0


@dataclass(frozen=True)
class FeedForwardCfg:
    enabled: bool = False
    gain: float = 0.6
    lowpass_s: float = 0.6


@dataclass(frozen=True)
class YawCfg:
    loop_gain: float = 0.4
    bearing_deadband_rad: float = 0.03
    feedforward: FeedForwardCfg = FeedForwardCfg()


@dataclass(frozen=True)
class GateCfg:
    edge_margin_frac: float = 0.02
    max_box_h_frac: float = 0.8
    max_abs_bearing_rad: float = 0.4
    max_confirm_age_s: float = 1.5


@dataclass(frozen=True)
class StandoffCfg:
    max_box_h_frac: float = 0.85
    max_box_w_frac: float = 0.6
    hover_s: float = 0.5
    back_off_pitch: float = 0.08


@dataclass(frozen=True)
class RefWidthCfg:
    close: float = 0.22
    medium: float = 0.14
    far: float = 0.09


@dataclass(frozen=True)
class CategoryScaleCfg:
    person: float = 1.0
    car: float = 0.8
    animal: float = 0.7
    object: float = 0.6
    none: float = 1.0


@dataclass(frozen=True)
class ForwardCfg:
    size_deadband: float = 0.08
    kp: float = 0.6
    cap: float = 0.25
    back_cap_ratio: float = 0.5
    backward_budget_m: float = 1.0
    gate: GateCfg = GateCfg()
    standoff: StandoffCfg = StandoffCfg()
    ref_width_frac: RefWidthCfg = RefWidthCfg()
    category_scale: CategoryScaleCfg = CategoryScaleCfg()


@dataclass(frozen=True)
class SearchCfg:
    step_deg: float = 30.0
    max_deg: float = 90.0
    yaw: float = 0.2
    stare_extra_s: float = 0.5


@dataclass(frozen=True)
class LifecycleCfg:
    confirm_s: float = 1.0
    confirm_min_score: float = 0.5
    confirm_max_jump_frac: float = 0.15
    follow_min_score: float = 0.4
    jump_frac: float = 0.25
    jump_size_ratio: float = 1.6
    detector_p95_default_s: float = 1.5
    detector_period_default_s: float = 1.0
    doubt_extra_s: float = 0.5
    lost_confirm_age_min_s: float = 2.5
    find_timeout_s: float = 25.0
    give_up_s: float = 25.0
    search: SearchCfg = SearchCfg()


@dataclass(frozen=True)
class AutopilotConfig:
    camera: CameraCfg = CameraCfg()
    plant: PlantCfg = PlantCfg()
    delay: DelayCfg = DelayCfg()
    yaw: YawCfg = YawCfg()
    forward: ForwardCfg = ForwardCfg()
    lifecycle: LifecycleCfg = LifecycleCfg()

    def validate(self) -> "AutopilotConfig":
        if not 20.0 <= self.camera.hfov_deg <= 170.0:
            raise ConfigError("camera.hfov_deg must be 20..170")
        if self.plant.yaw_rate_per_count <= 0:
            raise ConfigError("plant.yaw_rate_per_count must be > 0")
        if self.delay.theta_s <= 0:
            raise ConfigError("delay.theta_s must be > 0")
        if self.yaw.loop_gain <= 0:
            raise ConfigError("yaw.loop_gain must be > 0")
        if not self.yaw.feedforward.enabled and self.yaw.loop_gain > 0.7:
            raise ConfigError("yaw.loop_gain above 0.7 needs feed-forward (PLAN 4.4)")
        if self.yaw.feedforward.gain > 0.6:
            raise ConfigError("yaw.feedforward.gain must be <= 0.6 (PLAN 4.4)")
        if not 0 <= self.forward.cap <= 1 or not 0 <= self.forward.back_cap_ratio <= 1:
            raise ConfigError("forward.cap and back_cap_ratio must be 0..1")
        if self.lifecycle.search.max_deg > 180:
            raise ConfigError("lifecycle.search.max_deg must be <= 180")
        return self


# ----------------------------------------------------------------------------------------------
# Generic strict dict -> dataclass conversion


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path or '<root>'}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    origin = typing.get_origin(tp)
    if origin is typing.Union:  # Optional[...]
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if value is None else _coerce(args[0], value, path)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported type {tp!r}")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def governor_config_from_dict(data: Optional[dict]) -> GovernorConfig:
    return _build(GovernorConfig, data or {}, "").validate()


def autopilot_config_from_dict(data: Optional[dict]) -> AutopilotConfig:
    return _build(AutopilotConfig, data or {}, "").validate()


def load_governor_config(path: Optional[str | Path] = None) -> GovernorConfig:
    """Load config/governor.yaml (or GOVERNOR_CONFIG, or an explicit path)."""
    p = Path(path or os.environ.get("GOVERNOR_CONFIG") or repo_root() / "config" / "governor.yaml")
    if not p.is_absolute():
        p = repo_root() / p
    return governor_config_from_dict(_load_yaml(p))


def load_autopilot_config(path: Optional[str | Path] = None) -> AutopilotConfig:
    """Load config/autopilot.yaml (or AUTOPILOT_CONFIG, or an explicit path)."""
    p = Path(path or os.environ.get("AUTOPILOT_CONFIG") or repo_root() / "config" / "autopilot.yaml")
    if not p.is_absolute():
        p = repo_root() / p
    return autopilot_config_from_dict(_load_yaml(p))


def replace(cfg: Any, **changes: Any) -> Any:
    """dataclasses.replace with dotted keys, for tests: replace(cfg, **{"human.stale_s": 0.5})."""
    top: dict[str, dict] = {}
    direct: dict[str, Any] = {}
    for key, value in changes.items():
        if "." in key:
            head, rest = key.split(".", 1)
            top.setdefault(head, {})[rest] = value
        else:
            direct[key] = value
    for head, sub in top.items():
        direct[head] = replace(getattr(cfg, head), **sub)
    return dataclasses.replace(cfg, **direct)
