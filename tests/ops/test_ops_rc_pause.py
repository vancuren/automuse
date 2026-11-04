"""POST /debug/rc_pause: RC control packets stop for N s while the 1 Hz D1 keepalive continues.

Two layers:
- the real patched backend (TestClient, real lifespan, PacedFlightController) in TWIN mode, sending
  real X69 datagrams to a UDP listener on 127.0.0.1:23551 (ops port range), with a fake video adapter;
- the hub's real-mode policy with a stub governor and a governed RC adapter aimed at loopback
  (nothing is ever sent to the real drone).
"""
from __future__ import annotations

import importlib
import json
import queue
import socket
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mhd.contracts import FlightState, now_ms
from mhd.governor.turbodrone_glue import ensure_turbodrone_on_path

ensure_turbodrone_on_path()

RC_PORT, LOCAL_PORT, STUB_RC_PORT = 23551, 23552, 23553


class FakeVideoAdapter:
    """30 fps synthetic JPEG frames captured 80 ms before receipt (same as tests/gov)."""

    def __init__(self, *a, **kw):
        self._running = False
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = None
        ok, buf = cv2.imencode(".jpg", np.zeros((36, 64, 3), np.uint8))
        self._jpeg = buf.tobytes()

    def start(self):
        from models.video_frame import VideoFrame
        self._running = True

        def loop():
            i = 0
            while self._running:
                i += 1
                f = VideoFrame(i, self._jpeg)
                t = now_ms()
                f.capture_ms, f.receive_ms, f.width, f.height, f.render_id = t - 80.0, t, 1280, 720, None
                try:
                    self._q.put_nowait(f)
                except queue.Full:
                    pass
                time.sleep(1 / 30)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_running(self):
        return self._running

    def get_frame(self, timeout=1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_packets(self):
        return []


class UdpCounter:
    """Counts X69 control datagrams (28 bytes, ca 47 d5) and D1 keepalives (8 bytes, ca 47 d1)."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.events: list[tuple[float, str, bytes]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            kind = "rc" if data[:3] == b"\xca\x47\xd5" else ("ka" if data[:3] == b"\xca\x47\xd1" else "other")
            self.events.append((time.monotonic(), kind, data))

    def count(self, kind: str, t0: float, t1: float) -> int:
        return sum(1 for t, k, _ in list(self.events) if k == kind and t0 <= t < t1)

    def first_rc_after(self, t0: float):
        for t, k, d in list(self.events):
            if k == "rc" and t >= t0:
                return t, d
        return None

    def close(self):
        self._stop.set()
        self.sock.close()
        self._t.join(timeout=1.0)


BASE_ENV = {
    "FLIGHT_MODE": "twin", "DRONE_TYPE": "x69_lg", "DRONE_IP": "127.0.0.1", "CONTROL_PORT": str(RC_PORT),
    "X69_LG_LOCAL_CONTROL_PORT": str(LOCAL_PORT), "X69_LG_VIDEO_MODE": "rtsp", "PLUGINS_ENABLED": "false",
    "X69_LG_RTSP_URL": "rtsp://127.0.0.1:8650/live/ch00_1", "LOG_LEVEL": "WARNING",
}


@pytest.fixture
def backend(monkeypatch, tmp_path):
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    udp = UdpCounter(RC_PORT)
    ws_mod = importlib.import_module("web_server")
    monkeypatch.setattr(ws_mod, "TimedX69RtspAdapter", FakeVideoAdapter)
    from fastapi.testclient import TestClient
    try:
        with TestClient(ws_mod.app) as client:
            yield client, ws_mod.HUB, udp
    finally:
        udp.close()


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_twin_pause_stops_control_packets_but_not_keepalives(backend):
    client, hub, udp = backend
    assert _wait(lambda: udp.count("rc", 0, 1e18) > 10), "backend is not sending RC packets"
    assert _wait(lambda: udp.count("ka", 0, 1e18) >= 1, 2.5), "no keepalive seen"
    r = client.post("/debug/rc_pause", json={"seconds": 2.5}).json()
    assert r["ok"] is True and r["flight_mode"] == "twin" and r["seconds"] == 2.5
    t0 = time.monotonic()
    tele = client.get("/state").json()
    assert "rc_paused" in tele["safety_flags"]
    assert client.get("/debug/rc_pause").json()["active"] is True
    time.sleep(2.3)
    t1 = time.monotonic()
    # 50 ms grace at the start for a packet already in flight
    assert udp.count("rc", t0 + 0.05, t1) == 0
    assert udp.count("ka", t0, t1) >= 2          # 1 Hz keepalive kept going
    assert _wait(lambda: udp.count("rc", t1, 1e18) > 5, 2.0), "control packets did not come back"
    st = client.get("/debug/rc_pause").json()
    assert st["active"] is False and st["end_reason"] == "timeout"
    assert 55 <= st["dropped"] <= 70             # ~25 Hz x 2.5 s
    assert st["keepalives_during"] >= 2
    assert "rc_paused" not in client.get("/state").json()["safety_flags"]


def test_twin_estop_ends_the_pause_at_once(backend):
    client, hub, udp = backend
    assert _wait(lambda: udp.count("rc", 0, 1e18) > 5)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hb", "client": "ui"}))
        assert client.post("/debug/rc_pause", json={"seconds": 10}).json()["ok"]
        time.sleep(0.5)
        t_cmd = time.monotonic()
        ws.send_text(json.dumps({"type": "cmd", "command": "estop"}))
        hit = None
        end = time.monotonic() + 1.0
        while hit is None and time.monotonic() < end:
            hit = udp.first_rc_after(t_cmd)
            time.sleep(0.01)
        assert hit is not None, "E-stop never reached the wire during the pause"
        t_hit, data = hit
        assert t_hit - t_cmd < 0.25
        assert data[8 + 6] & 0x02, "first packet after the pause must carry E-stop (0x02)"
    st = client.get("/debug/rc_pause").json()
    assert st["active"] is False and st["end_reason"] == "state_changed"


def test_bad_requests_are_refused(backend):
    client, hub, udp = backend
    for body, reason in (({"seconds": 0}, "seconds_out_of_range"), ({"seconds": -1}, "seconds_out_of_range"),
                         ({"seconds": 61}, "seconds_out_of_range"), ({"seconds": "ten"}, "seconds_not_a_number")):
        r = client.post("/debug/rc_pause", json=body).json()
        assert r["ok"] is False and r["reason"] == reason, (body, r)
    assert client.post("/debug/rc_pause", json={}).status_code == 422
    assert client.post("/debug/rc_pause", content=b'{"seconds": NaN}',
                       headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/debug/rc_pause", json={"seconds": 1},
                       headers={"origin": "http://evil.example"}).status_code == 403
    assert client.post("/debug/rc_pause", json={"seconds": 1}).json()["ok"] is True
    r = client.post("/debug/rc_pause", json={"seconds": 1}).json()
    assert r["ok"] is False and r["reason"] == "already_paused"
    assert client.post("/debug/rc_resume").json()["was_active"] is True
    assert client.get("/debug/rc_pause").json()["active"] is False


# ---------------------------------------------------------------- real-mode policy (stub governor)

@pytest.fixture
def stub_hub():
    from mhd.backend.hub import Hub
    from mhd.governor.turbodrone_glue import GovernedX69LgRcAdapter
    udp = UdpCounter(STUB_RC_PORT)
    hub = Hub()
    hub.gov = SimpleNamespace(flight_mode="real", state=FlightState.on_ground, estop_latched=False)
    ad = GovernedX69LgRcAdapter("127.0.0.1", STUB_RC_PORT)     # loopback only, never the drone
    hub.rc_adapter = ad
    try:
        yield hub, ad, udp
    finally:
        ad.stop()
        udp.close()


@pytest.mark.parametrize("state", [FlightState.taking_off, FlightState.airborne, FlightState.landing,
                                   FlightState.estopped])
def test_real_mode_refuses_unless_on_ground(stub_hub, state):
    hub, ad, _ = stub_hub
    hub.gov.state = state
    r = hub.rc_pause(10)
    assert r["ok"] is False and r["reason"] == "real_mode_needs_on_ground"
    assert ad.rc_pause_status()["active"] is False


def test_real_mode_caps_at_15_s_and_ends_on_state_change(stub_hub):
    hub, ad, udp = stub_hub
    r = hub.rc_pause(15.5)
    assert r["ok"] is False and r["reason"] == "seconds_out_of_range" and r["max_s"] == 15.0
    r = hub.rc_pause(15)
    assert r["ok"] is True and r["flight_mode"] == "real"
    pkt = b"\xca\x47\xd5\x00" + (20).to_bytes(4, "little") + bytes(20)
    t0 = time.monotonic()
    for _ in range(5):
        ad.send_control_packet(pkt)                  # dropped: still ON_GROUND
    hub.gov.state = FlightState.taking_off           # operator pressed TAKEOFF (or LAND, E-STOP)
    ad.send_control_packet(pkt)                      # goes out, and ends the pause
    assert _wait(lambda: udp.count("rc", t0, 1e18) == 1, 1.0)
    st = ad.rc_pause_status()
    assert st["active"] is False and st["end_reason"] == "state_changed" and st["dropped"] == 5


def test_broken_predicate_fails_safe(stub_hub):
    hub, ad, udp = stub_hub
    hub.gov.flight_mode = "twin"
    assert hub.rc_pause(5)["ok"]
    ad._pause_keep = lambda: 1 / 0                   # a bug in the predicate must never block packets
    t0 = time.monotonic()
    ad.send_control_packet(b"\xca\x47\xd5\x00" + (20).to_bytes(4, "little") + bytes(20))
    assert _wait(lambda: udp.count("rc", t0, 1e18) == 1, 1.0)
    assert ad.rc_pause_status()["end_reason"] == "predicate_error"
