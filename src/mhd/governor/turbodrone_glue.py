"""Install the governor into turbodrone's X69 model: the single per-packet choke point.

FlightController calls model.update(dt, axes) once per RC tick, in the RC thread, right before
protocol.build_control_packet(model). GovernorStrategy.update is that call. It ignores `axes`
(nothing may write sticks around the governor) and writes the final raw values into the model:
roll/pitch/throttle/yaw as exact wire counts, speed_index pinned to 2 (scale 1.0), and the
one-shot flags. Any exception falls back to neutral sticks (keeping a latched E-stop) and raises
an alarm, so the RC thread can never die in model.update.

Importing this module needs turbodrone/backend on sys.path (web_server.py runs from there;
tests call ensure_turbodrone_on_path()).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import repo_root

logger = logging.getLogger(__name__)


def ensure_turbodrone_on_path() -> Path:
    backend = repo_root() / "turbodrone" / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    return backend


ensure_turbodrone_on_path()

from control.strategies import ControlStrategy  # noqa: E402  (turbodrone)
from models.x69_lg_rc import X69LgRcModel  # noqa: E402  (turbodrone)

from .core import Governor  # noqa: E402

PINNED_SPEED_INDEX = 2       # scale 1.0 on roll and pitch, so governor caps are exact on the wire


class GovernorStrategy(ControlStrategy):
    """turbodrone ControlStrategy whose update() is one governor tick."""

    def __init__(self, governor: Governor):
        self.governor = governor
        self.alarm = False           # set after any tick exception; telemetry shows it
        self.last_error = None
        self._land_toggle = 0

    def update(self, model, dt, axes) -> None:  # noqa: ARG002 - axes are deliberately ignored
        try:
            out = self.governor.tick()
            model.speed_index = PINNED_SPEED_INDEX
            model.throttle = float(out.wire["throttle"])
            model.yaw = float(out.wire["yaw"])
            model.pitch = float(out.wire["pitch"])
            model.roll = float(out.wire["roll"])
            model.takeoff_flag = bool(out.takeoff)
            model.land_flag = bool(out.land)
            model.stop_flag = bool(out.stop)
            model.flip_flag = False          # masked: nothing in this system may flip
            model.calibration_flag = False   # masked: never calibrate from the governor
            model.camera_tilt_state = int(out.tilt_state)
        except Exception as exc:  # noqa: BLE001 - the RC thread must survive anything
            self._fallback(model, exc)

    def _fallback(self, model, exc: BaseException) -> None:
        """Neutral sticks; keep a latched E-stop; after faults.land_after_s also re-assert land."""
        try:
            self.alarm = True
            self.last_error = repr(exc)
            center = 128.0
            model.speed_index = PINNED_SPEED_INDEX
            model.throttle = model.yaw = model.pitch = model.roll = center
            model.takeoff_flag = False
            model.flip_flag = False
            model.calibration_flag = False
            model.camera_tilt_state = 0
            estop = bool(getattr(self.governor, "_estop_latched", False))
            model.stop_flag = estop
            land = False
            try:
                land = self.governor.note_fault()
            except Exception:  # noqa: BLE001
                land = True
            self._land_toggle = (self._land_toggle + 1) % 5   # every 5th packet: 5 Hz at 25 Hz
            model.land_flag = bool(land and not estop and self._land_toggle == 0)
            logger.error("[governor] tick failed, neutral fallback: %r", exc)
        except Exception:  # noqa: BLE001
            pass


class GovernedX69LgRcModel(X69LgRcModel):
    """X69 model that refuses to let anything replace the governor once installed.

    Also routes the legacy one-shot methods (takeoff, land, emergency_stop) to the governor so
    any upstream code path that calls them still goes through the state machine, and locks the
    speed tier and profile (expo) because the governor writes raw wire counts directly.
    """

    def __init__(self, governor: Governor):
        super().__init__(profile="normal")
        self.governor = governor
        self.speed_index = PINNED_SPEED_INDEX
        self.strategy = GovernorStrategy(governor)

    def set_strategy(self, strategy) -> None:
        if isinstance(getattr(self, "strategy", None), GovernorStrategy) and strategy is not self.strategy:
            logger.warning("[governor] refused set_strategy(%s): the governor owns the sticks",
                           type(strategy).__name__)
            return
        self.strategy = strategy

    def set_speed_index(self, speed_index: int) -> None:
        self.speed_index = PINNED_SPEED_INDEX   # pinned; ignore UI speed tiers

    def set_profile(self, name: str) -> None:
        return None                             # expo and ramps are not used under the governor

    def set_camera_tilt_state(self, state: int) -> None:
        d = {2: 1, 1: -1}.get(int(state), 0)
        self.governor.set_tilt(d)

    def takeoff(self):
        return self.governor.command("takeoff", source="model")

    def land(self):
        return self.governor.command("land", source="model")

    def emergency_stop(self):
        return self.governor.command("estop", source="model")

    def calibrate_gyro(self):
        logger.warning("[governor] calibrate refused (masked)")

    def flip(self):
        logger.warning("[governor] flip refused (masked)")


def install_governor(governor: Governor) -> GovernedX69LgRcModel:
    """Build the X69 model with the governor installed as its (irreplaceable) strategy."""
    model = GovernedX69LgRcModel(governor)
    governor.start_background()
    return model
