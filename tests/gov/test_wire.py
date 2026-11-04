"""Wire test: the real turbodrone FlightController + X69LgRcProtocolAdapter, with the governor
installed, sending 28-byte datagrams to a local UDP socket (127.0.0.1:23510, this workstream's range).

Proves on the wire: caps, latched E-stop (even with dropped packets), land re-assert, neutral within
0.5 s of perception dying, plus RC inter-arrival and per-tick governor time.
"""
from __future__ import annotations

import random
import socket
import threading
import time

import pytest

from conftest import make_mission, make_obs
from mhd.contracts import now_ms
from mhd.governor import Governor
from mhd.governor.config import load_autopilot_config, load_governor_config, replace
from mhd.governor.turbodrone_glue import PacedFlightController, install_governor

from protocols.x69_lg_rc_protocol_adapter import X69LgRcProtocolAdapter
from services.flight_controller import FlightController

RX_PORT = 23510
TX_PORT = 23511
MAGIC = b"\xca\x47\xd5\x00"


def decode(pkt: bytes) -> dict:
    """Decode and validate one 28-byte X69 control datagram."""
    assert len(pkt) == 28 and pkt[:4] == MAGIC
    assert int.from_bytes(pkt[4:8], "little") == 20
    p = pkt[8:]
    assert p[0] == 0x66 and p[1] == 0x14 and p[19] == 0x99
    x = 0
    for b in p[2:18]:
        x ^= b
    assert x == p[18], "bad XOR checksum"
    assert p[8:18] == bytes(10)
    return {"roll": p[2], "pitch": p[3], "throttle": p[4], "yaw": p[5], "f1": p[6], "f2": p[7]}


class Receiver(threading.Thread):
    def __init__(self, port: int, drop: float = 0.0):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.1)
        self.drop = drop
        self.rng = random.Random(7)
        self.packets: list[tuple[float, dict]] = []
        self.keepalives = 0
        self._stop = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            t = now_ms()
            if len(data) == 8:
                self.keepalives += 1
                continue
            if self.drop and self.rng.random() < self.drop:
                continue                       # simulated radio loss
            with self.lock:
                self.packets.append((t, decode(data)))

    def since(self, t0: float) -> list[tuple[float, dict]]:
        with self.lock:
            return [p for p in self.packets if p[0] >= t0]

    def close(self):
        self._stop.set()
        self.join(timeout=1.0)
        self.sock.close()


class Feeder(threading.Thread):
    """Stands in for the UI (heartbeat), the video receiver (frames) and perception (obs)."""

    def __init__(self, gov: Governor):
        super().__init__(daemon=True)
        self.g = gov
        self.obs_cx = None          # None: no observations
        self.last_obs_ms = None
        self._stop = threading.Event()
        self._fid = 0

    def run(self):
        while not self._stop.wait(0.04):
            t = now_ms()
            self.g.heartbeat(t)
            self.g.note_frame(t - 120.0, t)
            if self.obs_cx is not None:
                self._fid += 1
                cx = self.obs_cx + (1.0 if self._fid % 2 else -1.0)     # a real tracker's box jitters
                self.g.submit_obs(make_obs(self._fid, t - 200.0, cx=cx, w=60, h=200), receive_ms=t)

                self.last_obs_ms = t

    def close(self):
        self._stop.set()
        self.join(timeout=1.0)


@pytest.fixture(params=["paced", "upstream"])
def stack(tmp_path, request):
    cfg = replace(load_governor_config(), **{"takeoff.hold_s": 0.5})
    ap = replace(load_autopilot_config(), **{"delay.theta_s": 0.3})      # high gain so caps bind
    gov = Governor(cfg, ap, "twin", log_dir=tmp_path)
    rx = Receiver(RX_PORT)
    rx.start()
    model = install_governor(gov)
    adapter = X69LgRcProtocolAdapter("127.0.0.1", RX_PORT, local_port=TX_PORT)
    fc_cls = PacedFlightController if request.param == "paced" else FlightController
    fc = fc_cls(model, adapter, 25.0)
    feeder = Feeder(gov)
    feeder.start()
    fc.start()
    time.sleep(0.3)
    yield gov, rx, feeder, fc
    fc.stop()
    feeder.close()
    rx.close()
    gov.close()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_wire_caps_estop_land_and_stale_obs(stack, record_property):
    gov, rx, feeder, fc = stack

    # Idle on the ground: neutral, no flags, 25 Hz.
    t0 = now_ms()
    time.sleep(1.0)
    idle = rx.since(t0)
    # Upstream FlightController sleeps a fixed 40 ms after each tick's work, so macOS delivers
    # about 21-22 Hz, not 25 Hz (sleep overshoot). Unchanged upstream; measured and reported.
    record_property("idle_rc_packets_per_s", len(idle))
    assert len(idle) >= 18
    assert all(p["yaw"] == 128 and p["pitch"] == 128 and p["f1"] == 0 for _, p in idle)

    # Takeoff flag goes out once.
    t0 = now_ms()
    assert gov.command("takeoff").ok
    assert _wait(lambda: gov.state.value == "AIRBORNE")
    takeoff = [p for _, p in rx.since(t0) if p["f1"] & 0x01]
    assert len(takeoff) == 1

    # Human caps on the wire.
    t0 = now_ms()
    end = time.monotonic() + 0.6
    while time.monotonic() < end:
        gov.submit_human({"yaw": 1.0, "pitch": -1.0, "roll": 1.0, "throttle": 1.0}, "abs")
        time.sleep(0.03)
    human = [p for _, p in rx.since(t0 + 100)]
    assert human and all(p["yaw"] <= 128 + 64 and p["pitch"] >= 128 - 64 and p["throttle"] <= 128 + 64 for p in human)
    assert max(p["yaw"] for p in human) == 128 + 64
    time.sleep(0.4)                                    # human goes stale

    # Autopilot caps on the wire: subject far right, yaw pinned at +38, roll and throttle 0.
    gov.set_mission(make_mission())
    feeder.obs_cx = 1250.0
    assert _wait(lambda: gov.autopilot.mode.value == "FOLLOW", 3.0)
    time.sleep(1.0)
    t0 = now_ms()
    time.sleep(0.8)
    ap = [p for _, p in rx.since(t0)]
    assert ap and all(p["yaw"] <= 128 + 38 and 128 - 32 <= p["pitch"] <= 128 + 32 for p in ap)
    assert all(p["roll"] == 128 and p["throttle"] == 128 for p in ap)
    assert max(p["yaw"] for p in ap) == 128 + 38

    # Perception dies: neutral on the wire within 0.5 s of the last observation.
    feeder.obs_cx = None
    time.sleep(0.05)
    last_obs = feeder.last_obs_ms
    assert _wait(lambda: any(p["yaw"] == 128 for _, p in rx.since(last_obs)), 1.0)
    first_neutral = min(t for t, p in rx.since(last_obs) if p["yaw"] == 128)
    stale_s = (first_neutral - last_obs) / 1000.0
    record_property("stale_obs_to_neutral_s", round(stale_s, 3))
    assert stale_s <= 0.5, stale_s

    # Land: re-asserted 5x per second, sticks neutral.
    t0 = now_ms()
    assert gov.command("land").ok
    time.sleep(2.0)
    landing = rx.since(t0 + 50)
    n_land = sum(1 for _, p in landing if p["f1"] & 0x80)
    assert 9 <= n_land <= 11, n_land
    assert all(p["yaw"] == 128 and p["pitch"] == 128 for _, p in landing)

    # E-stop latched: every packet carries 0x02 with neutral sticks, even with 50% packet loss,
    # and a takeoff command cannot clear it.
    with rx.lock:
        clean_ts = [t for t, _ in rx.packets]          # before simulated loss starts
    rx.drop = 0.5
    assert gov.command("estop").ok
    time.sleep(0.1)
    t0 = now_ms()
    gov.command("takeoff")
    gov.command("land")
    time.sleep(1.5)
    est = rx.since(t0)
    assert len(est) >= 10
    assert all(p["f1"] == 0x02 and p["yaw"] == 128 and p["pitch"] == 128 for _, p in est)

    # Report: RC inter-arrival and per-tick governor time inside the real RC thread.
    gaps = sorted(b - a for a, b in zip(clean_ts, clean_ts[1:]))
    p50, p99 = gaps[len(gaps) // 2], gaps[int(0.99 * len(gaps))]
    stats = gov.tick_stats()
    record_property("rc_interarrival_p50_ms", round(p50, 1))
    record_property("rc_interarrival_p99_ms", round(p99, 1))
    record_property("tick_us", stats)
    print(f"\nwire[{type(fc).__name__}]: stale->neutral {stale_s:.3f} s; RC gap p50 {p50:.1f} ms p99 {p99:.1f} ms "
          f"(no loss); governor tick (RC thread, with feeder and receiver threads running) {stats}; keepalives {rx.keepalives}")
    assert stats["p99_us"] < 1000.0
    assert not fc.model.strategy.alarm
