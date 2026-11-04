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
import math
import select
import socket
import sys

import threading
import time
from pathlib import Path
from typing import Callable, Optional

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
from protocols.x69_lg_rc_protocol_adapter import X69LgRcProtocolAdapter  # noqa: E402  (turbodrone)
from services.flight_controller import FlightController  # noqa: E402  (turbodrone)

from mhd.contracts import FlightState  # noqa: E402

from .core import Governor  # noqa: E402

PINNED_SPEED_INDEX = 2       # scale 1.0 on roll and pitch, so governor caps are exact on the wire


class GovernorStrategy(ControlStrategy):
    """turbodrone ControlStrategy whose update() is one governor tick."""

    def __init__(self, governor: Governor):
        self.governor = governor
        self.alarm = False           # set after any tick exception; telemetry shows it
        self.last_error = None
        self._land_toggle = 0
        self._last_log = -1e9

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
            # Already LANDING (or shutting down): keep landing now, not after faults.land_after_s.
            # Read without the lock, like _estop_latched: the tick that holds it may be the fault.
            if getattr(self.governor, "_state", None) == FlightState.landing or getattr(self.governor, "_shutdown", False):
                land = True

            self._land_toggle = (self._land_toggle + 1) % 5   # every 5th packet: 5 Hz at 25 Hz
            model.land_flag = bool(land and not estop and self._land_toggle == 0)
            now = time.monotonic()
            if now - self._last_log >= 5.0:           # stderr I/O on the RC thread: rate-limited
                self._last_log = now
                logger.error("[governor] tick failed, neutral fallback (estop=%s, land=%s): %r",
                             estop, land, exc)
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


class PacedFlightController(FlightController):
    """Upstream FlightController with deadline pacing instead of a fixed sleep after each tick.

    Upstream sleeps update_interval after the tick's work; on macOS time.sleep(0.040) really takes
    about 46 ms, so the X69 loop ran at ~20.5 Hz with an RC inter-arrival p99 near 60 ms. Here each
    tick sleeps until an absolute 40 ms deadline, so the average rate is the configured 25 Hz.
    After a stall it resynchronizes instead of bursting. Packet building is unchanged upstream code.
    """

    def __init__(self, drone_model, protocol_adapter, update_rate=25.0):
        super().__init__(drone_model, protocol_adapter, validate_control_rate(update_rate))

    def _control_loop(self):
        period = self.update_interval

        prev = time.monotonic()
        deadline = prev
        while self.running:
            now = time.monotonic()
            dt = now - prev
            prev = now
            try:
                self.model.update(dt, {
                    "throttle": self.throttle_dir,
                    "yaw": self.yaw_dir,
                    "pitch": self.pitch_dir,
                    "roll": self.roll_dir,
                })
            except Exception:  # noqa: BLE001 - GovernorStrategy already catches; belt and braces
                logger.exception("[rc] model.update raised")
            try:
                packet = self.protocol.build_control_packet(self.model)
                self.protocol.send_control_packet(packet)
            except Exception:  # noqa: BLE001 - same as upstream: socket may be recreated
                pass
            deadline += period
            delay = deadline - time.monotonic()
            if delay < -period:          # fell behind by more than a tick: resync, no burst
                deadline = time.monotonic()
                delay = 0.0
            if delay > 0:
                time.sleep(delay)


def validate_control_rate(rate) -> float:
    """The X69 RC loop runs at 25 Hz. NaN, negative or absurd rates used to make it spin flat out."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        raise ValueError(f"CONTROL_RATE={rate!r} is not a number") from None
    if not (math.isfinite(r) and 10.0 <= r <= 50.0):
        raise ValueError(f"CONTROL_RATE={rate!r} must be within 10..50 Hz (X69: 25)")
    return r


def _age_ms(since: Optional[float], now: float) -> Optional[float]:
    """Milliseconds from `since` to `now` (same clock), or None when `since` never happened."""
    return None if since is None else round(max(0.0, now - since) * 1000.0, 1)


class GovernedX69LgRcAdapter(X69LgRcProtocolAdapter):
    """Upstream X69 RC adapter plus visibility into the UDP link (review finding).

    Upstream's 1 Hz D1 keepalive thread returned on the first OSError (ENOBUFS / EHOSTUNREACH in a
    Wi-Fi blip), ending keepalives for the whole session, and RC sends swallowed OSError silently.
    Here the keepalive logs and keeps going, and both paths count errors and successes for
    telemetry (link_stats()). Packet bytes are unchanged upstream code. link_stats() also gives the
    age of the last failed RC send, so the hub's rc_send_errors flag shows recent failures only.
    """

    board = "x69_lg"                             # the hub reads this (link flags, RC pause wording)

    def __init__(self, *args, **kwargs):
        self._init_link_state()
        super().__init__(*args, **kwargs)        # starts the keepalive thread (our _keepalive_loop)

    def _init_link_state(self) -> None:
        self._stats_lock = threading.Lock()
        self.rc_sent = 0
        self.rc_errors = 0
        self.rc_last_error: str = ""
        self.rc_last_error_t: Optional[float] = None   # monotonic s of the last failed RC send
        self.keepalive_sent = 0
        self.keepalive_errors = 0
        self.keepalive_last_ok: float = time.monotonic()
        self._ka_last_log = -1e9
        self._rc_last_log = -1e9
        # Debug RC pause (Session A link-loss test 3: control packets stop, the keepalive goes on).
        # Set by the hub's POST /debug/rc_pause, which owns the policy (real mode: ON_GROUND, <= 15 s).
        self._pause_until = 0.0                  # monotonic s; 0 = not paused
        self._pause_keep: Optional[Callable[[], bool]] = None
        self._pause_info: dict = {}
        self.rc_pause_dropped_total = 0

    # ---------------------------------------------------------------------------------- RC pause
    def pause_rc(self, seconds: float, keep_paused: Optional[Callable[[], bool]] = None,
                 note: str = "") -> dict:
        """Stop sending RC control packets for `seconds`; the D1 keepalive thread is untouched.

        DEBUG ONLY (Session A link-loss test 3). `keep_paused` is called before every dropped packet
        (RC thread, 25 Hz): when it returns False or raises, the pause ends at once and that very
        packet goes out (the hub passes "flight state unchanged and not shutting down", so LAND,
        E-STOP, TAKEOFF or a safety landing always reach the drone). The policy lives in the hub.
        """
        s = float(seconds)
        if not (math.isfinite(s) and s > 0):
            raise ValueError(f"pause seconds must be a finite number > 0, got {seconds!r}")
        now = time.monotonic()
        with self._stats_lock:
            self._pause_keep = keep_paused
            self._pause_info = {"seconds": s, "started": now, "dropped": 0, "ended": None,
                                "end_reason": None, "note": note,
                                "rc_sent_at_start": self.rc_sent, "keepalive_sent_at_start": self.keepalive_sent}
            self._pause_until = now + s
        return self.rc_pause_status()

    def resume_rc(self, reason: str = "resumed") -> dict:
        """End a pause now (idempotent)."""
        with self._stats_lock:
            self._end_pause_locked(reason, time.monotonic())
        return self.rc_pause_status()

    def _end_pause_locked(self, reason: str, now: float) -> bool:
        if not self._pause_until:
            return False
        self._pause_until = 0.0
        self._pause_keep = None
        info = self._pause_info
        info["ended"] = now
        info["end_reason"] = reason
        info["keepalive_sent_at_end"] = self.keepalive_sent
        info["rc_sent_at_end"] = self.rc_sent
        logger.warning("[rc] RC PAUSE ENDED (%s): %d control packets dropped in %.2f s; "
                       "control packets flow again", reason, info.get("dropped", 0),
                       now - info.get("started", now))
        return True

    def rc_pause_status(self) -> dict:
        with self._stats_lock:
            info = dict(self._pause_info)
            active = bool(self._pause_until)
            now = time.monotonic()
            left = max(0.0, self._pause_until - now) if active else 0.0
            out = {"active": active, "left_s": round(left, 3), "seconds": info.get("seconds"),
                   "dropped": info.get("dropped", 0), "dropped_total": self.rc_pause_dropped_total,
                   "end_reason": info.get("end_reason"), "note": info.get("note", ""),
                   "rc_sent": self.rc_sent, "keepalive_sent": self.keepalive_sent}
            if info.get("started") is not None:
                end = info.get("ended") or now
                out["elapsed_s"] = round(end - info["started"], 3)
                ka_end = info.get("keepalive_sent_at_end", self.keepalive_sent)
                rc_end = info.get("rc_sent_at_end", self.rc_sent)
                out["keepalives_during"] = ka_end - info.get("keepalive_sent_at_start", 0)
                out["rc_sent_during"] = rc_end - info.get("rc_sent_at_start", 0)
            return out

    def _drop_for_pause(self) -> bool:
        """RC thread: True if this control packet must be dropped (pause active and still wanted)."""
        until = self._pause_until
        if not until:
            return False
        now = time.monotonic()
        keep = self._pause_keep
        reason = None
        if now >= until:
            reason = "timeout"
        else:
            try:
                if keep is not None and not keep():
                    reason = "state_changed"
            except Exception:  # noqa: BLE001 - fail safe: a broken predicate ends the pause
                reason = "predicate_error"
        with self._stats_lock:
            if reason is not None:
                self._end_pause_locked(reason, now)
                return False
            if not self._pause_until:           # resumed meanwhile
                return False
            self._pause_info["dropped"] = self._pause_info.get("dropped", 0) + 1
            self.rc_pause_dropped_total += 1
        return True

    def send_control_packet(self, packet: bytes) -> None:
        if self._pause_until and self._drop_for_pause():
            return                                   # debug RC pause: drop it, keepalive continues
        try:
            self.sock.sendto(packet, (self.drone_ip, self.control_port))
        except OSError as exc:
            now = time.monotonic()
            with self._stats_lock:
                self.rc_errors += 1
                self.rc_last_error = f"{type(exc).__name__}({exc.errno})"
                self.rc_last_error_t = now
            if now - self._rc_last_log >= 5.0:      # RC thread: rate-limited, through the log queue
                self._rc_last_log = now
                logger.error("[rc] RC packet send failed (%d so far): %r", self.rc_errors, exc)
            return
        with self._stats_lock:
            self.rc_sent += 1
        if self.debug_packets:
            self.packet_counter += 1
            logger.debug("[x69-lg-rc] #%05d %s", self.packet_counter, packet.hex(" "))

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(1.0):
            try:
                self.sock.sendto(self.D1_KEEPALIVE, (self.drone_ip, self.control_port))
            except OSError as exc:
                if self._keepalive_stop.is_set():
                    return                            # socket closed by stop()
                with self._stats_lock:
                    self.keepalive_errors += 1
                now = time.monotonic()
                if now - self._ka_last_log >= 5.0:
                    self._ka_last_log = now
                    logger.error("[rc] D1 keepalive send failed (%d so far), retrying: %r",
                                 self.keepalive_errors, exc)
                continue
            with self._stats_lock:
                self.keepalive_sent += 1
                self.keepalive_last_ok = time.monotonic()

    def link_stats(self) -> dict:
        with self._stats_lock:
            now = time.monotonic()
            return {"rc_sent": self.rc_sent, "rc_errors": self.rc_errors,
                    "rc_error_age_ms": _age_ms(self.rc_last_error_t, now),
                    "keepalive_sent": self.keepalive_sent, "keepalive_errors": self.keepalive_errors,
                    "keepalive_age_ms": round((now - self.keepalive_last_ok) * 1000.0, 1),
                    "keepalive_alive": self._keepalive_thread.is_alive()}


class CameraOnlyViolation(RuntimeError):
    """Something tried to use the RC socket in camera-only mode (there is none). A bug, never a link error."""


class _NoRcSocket:
    """Stands in for the RC adapter's UDP socket in camera-only mode: refuses every send."""

    def _refuse(self, *_args, **_kwargs):
        raise CameraOnlyViolation("camera-only mode: the RC adapter has no socket; nothing may be sent")

    sendto = send = sendmsg = sendall = connect = bind = _refuse

    def close(self) -> None:
        return None


class CameraOnlyRcAdapter(GovernedX69LgRcAdapter):
    """MHD_CAMERA_ONLY=1 (twin mode with a real camera, no aircraft): the RC adapter that never sends.

    The governor's RC packets are built exactly as in the twin (same model, same packet builder, same
    one-shot flag handling) and then dropped and counted. No socket is ever created, so nothing can
    be sent to any address, not even loopback. The keepalive thread only refreshes keepalive_last_ok,
    so telemetry shows a healthy link (no keepalive_stale, no rc_send_errors) and the governor,
    autopilot and telemetry run exactly as in twin mode.
    """

    camera_only = True

    def __init__(self, drone_ip: str = "127.0.0.1", control_port: int = 0, *, local_port: int = 0) -> None:
        # Deliberately NOT X69LgRcProtocolAdapter.__init__: it opens a UDP socket, binds local_port,
        # and starts a keepalive thread that sends D1 datagrams.
        self._init_link_state()
        self.drone_ip = drone_ip                 # informational only: never used as a destination
        self.control_port = int(control_port)
        self.local_port = 0                      # nothing is bound
        self.debug_packets = False
        self.packet_counter = 0
        self.speed_scale_by_index = {0: 0.55, 1: 0.75, 2: 1.0}   # upstream's table; the governor pins 2
        self.sock = _NoRcSocket()
        self.rc_suppressed = 0                   # RC packets built and dropped (would have been sent)
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True,
                                                  name="CameraOnlyKeepalive")
        self._keepalive_thread.start()

    def send_control_packet(self, packet: bytes) -> None:
        if self._pause_until and self._drop_for_pause():
            return                               # debug RC pause: same bookkeeping as the twin
        with self._stats_lock:
            self.rc_suppressed += 1
        if self.debug_packets:
            self.packet_counter += 1
            logger.debug("[camera-only-rc] #%05d not sent: %s", self.packet_counter, packet.hex(" "))

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(1.0):
            with self._stats_lock:
                self.keepalive_last_ok = time.monotonic()   # nothing is sent: the link is "fine"

    def stop(self) -> None:
        self._keepalive_stop.set()
        if self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=1.0)

    def link_stats(self) -> dict:
        stats = super().link_stats()
        with self._stats_lock:
            stats["rc_suppressed"] = self.rc_suppressed
        stats["camera_only"] = True
        return stats


# ------------------------------------------------------------------------------------ Lewei board

FC_TELEMETRY_LEN = 10
FC_TELEMETRY_START = 0x66


def _udp_socket() -> socket.socket:
    """The Lewei adapter's UDP socket. A function so tests can hand in a fake: no test may reach a drone."""
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def parse_fc_telemetry(data: bytes) -> Optional[tuple[float, int, bytes]]:
    """One Lewei flight-controller downlink packet, 10 bytes: 66 VV S2 S3 S4 S5 S6 S7 S8 CK, where
    CK = XOR of bytes 1..8, VV = battery voltage in 0.1 V and S2 = status flags (0x00 idle, 0x08 seen
    while the motors ran after the takeoff flag). Returns (volts, status, bytes 3..8), or None."""
    if len(data) != FC_TELEMETRY_LEN or data[0] != FC_TELEMETRY_START:
        return None
    ck = 0
    for b in data[1:9]:
        ck ^= b
    if ck != data[9]:
        return None
    return data[1] / 10.0, data[2], bytes(data[3:9])


class LeweiRcAdapter(GovernedX69LgRcAdapter):
    """MHD_BOARD=lewei: our X69 unit's Lewei Wi-Fi board (FLIGHT_MODE=real only, see mhd.backend.modes).

    Verified on the drone (props off, 2026-11-04): flight control is UDP to 192.168.0.1:50000 with
    exactly the 20-byte X69 flight-controller payload that build_control_payload() makes
    (66 14 RR PP TT YY F1 F2 00x10 XOR 99), sent raw: no `ca 47 d5 00` + length wrapper and no D1
    keepalive. Same governor model, payload builder and one-shot flag handling as the X69 adapter.

    The socket is non-blocking, so a send can never stall the 25 Hz RC tick. A daemon thread reads the
    drone's downlink on the same socket (it replies from :50000 to our source port: ~20/s, ~5/s while
    the motors run, nothing for ~5 s during LAND): battery volts, status byte, age, good/bad counts.
    Nothing here feeds a governor decision.

    Link health is that downlink, never a successful send: with the iPhone USB tether holding the
    default route, datagrams to 192.168.0.1 leave through the tether once the drone's Wi-Fi is gone and
    still "succeed". There is no keepalive on this board, so link_stats() puts the downlink in the
    keepalive fields: keepalive_age_ms = time since the last VALID downlink packet (since the adapter
    started, before the first one), keepalive_alive = the reader thread is alive. The hub turns them
    into drone_link_stale (never keepalive_stale). `clock` is the monotonic clock (tests pass a fake).
    """

    board = "lewei"                              # the hub: drone_link_stale, and "all packets stop" on RC pause

    def __init__(self, drone_ip: str = "192.168.0.1", control_port: int = 50000, *, local_port: int = 0,
                 sock=None, clock: Callable[[], float] = time.monotonic) -> None:
        # Not X69LgRcProtocolAdapter.__init__: it would start the D1 keepalive sender.
        self._init_link_state()
        self._clock = clock
        self._started_at = clock()               # the downlink age counts from here until the first packet
        self.drone_ip = drone_ip
        self.control_port = int(control_port)
        self.local_port = local_port
        self.debug_packets = False
        self.packet_counter = 0
        self.speed_scale_by_index = {0: 0.55, 1: 0.75, 2: 1.0}   # upstream's table; the governor pins 2
        self.sock = sock if sock is not None else _udp_socket()
        self._bind_socket(local_port)            # upstream helper: binds only when local_port > 0
        self.sock.setblocking(False)
        self._fc_lock = threading.Lock()
        self.fc_ok = 0
        self.fc_bad = 0
        self.fc_battery_v: Optional[float] = None
        self.fc_status: Optional[int] = None
        self.fc_rest: Optional[bytes] = None
        self.fc_last_rx: Optional[float] = None
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True, name="LeweiFcTelemetry")
        self._reader_thread.start()

    # ----------------------------------------------------------------------------------- uplink
    def build_control_packet(self, drone_model) -> bytes:
        return self.build_control_payload(drone_model)   # the bare 20 bytes: no wrapper on this board

    def send_control_packet(self, packet: bytes) -> None:
        if self._pause_until and self._drop_for_pause():
            return          # debug RC pause (real mode: ON_GROUND, <= 15 s): no keepalive, so ALL packets stop
        try:
            self.sock.sendto(packet, (self.drone_ip, self.control_port))
        except OSError as exc:                       # no route at all (EHOSTUNREACH...) or a full buffer
            now = self._clock()
            with self._stats_lock:
                self.rc_errors += 1
                self.rc_last_error = f"{type(exc).__name__}({exc.errno})"
                self.rc_last_error_t = now
            if now - self._rc_last_log >= 5.0:
                self._rc_last_log = now
                logger.error("[rc] Lewei RC packet send failed (%d so far): %r", self.rc_errors, exc)
            return
        with self._stats_lock:
            self.rc_sent += 1           # proves nothing about the drone: the tether route also "succeeds"
        if self.debug_packets:
            self.packet_counter += 1
            logger.debug("[lewei-rc] #%05d %s", self.packet_counter, packet.hex(" "))

    def _keepalive_loop(self) -> None:               # never started; there is no keepalive on this board
        return None

    # --------------------------------------------------------------------------------- downlink
    def on_fc_packet(self, data: bytes, addr, now: Optional[float] = None) -> bool:
        """One datagram from the socket: counted good only if it is from the drone and well formed."""
        from_drone = tuple(addr[:2]) == (self.drone_ip, self.control_port)
        parsed = parse_fc_telemetry(data) if from_drone else None
        with self._fc_lock:
            if parsed is None:
                self.fc_bad += 1
                return False
            self.fc_ok += 1
            self.fc_battery_v, self.fc_status, self.fc_rest = parsed
            self.fc_last_rx = self._clock() if now is None else now
        return True

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            try:
                ready, _, _ = select.select([self.sock], [], [], 0.2)
            except (OSError, ValueError, TypeError):  # socket closed by stop()
                if self._reader_stop.wait(0.2):
                    return
                continue
            if not ready:
                continue
            try:
                data, addr = self.sock.recvfrom(256)
            except BlockingIOError:
                continue
            except OSError:
                if self._reader_stop.wait(0.05):
                    return
                continue
            self.on_fc_packet(data, addr)

    def fc_telemetry(self) -> dict:
        """Informational telemetry fields (hub telemetry latency_ms; never a governor input)."""
        with self._fc_lock:
            return {"fc_battery_v": self.fc_battery_v, "fc_status": self.fc_status,
                    "fc_telemetry_age": _age_ms(self.fc_last_rx, self._clock()),
                    "fc_packets_ok": self.fc_ok, "fc_packets_bad": self.fc_bad}

    def link_stats(self) -> dict:
        """The keepalive fields are the drone's downlink here (class docstring): keepalive_age_ms since
        the last valid downlink packet (or since the adapter started), keepalive_alive = reader thread.
        One lock at a time, never nested (the reader holds only _fc_lock, the RC thread _stats_lock)."""
        with self._fc_lock:
            last_rx = self.fc_last_rx
        with self._stats_lock:
            stats = {"rc_sent": self.rc_sent, "rc_errors": self.rc_errors, "keepalive_sent": 0, "keepalive_errors": 0}
            err_t = self.rc_last_error_t
        now = self._clock()
        stats["rc_error_age_ms"] = _age_ms(err_t, now)
        stats["keepalive_age_ms"] = _age_ms(self._started_at if last_rx is None else last_rx, now)
        stats["keepalive_alive"] = self._reader_thread.is_alive()
        return stats

    def stop(self) -> None:
        self._reader_stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        try:
            self.sock.close()
        except OSError:
            pass

