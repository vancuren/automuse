"""MHD_BOARD=lewei: our X69 unit's Lewei Wi-Fi board (real mode only).

Flight control is the bare 20-byte X69 payload over UDP to 192.168.0.1:50000 (no `ca 47 d5 00` wrapper,
no D1 keepalive); the drone's 10-byte downlink is read for telemetry only. NOTHING in this file may
reach a drone: every adapter aimed at 192.168.0.1 gets a fake socket, under a guard that fails the
test on any real UDP socket; the reader thread is exercised with both ends on 127.0.0.1.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import time

import pytest

from conftest import ROOT
from test_camera_only import no_udp  # noqa: F401  (fixture: fails on any real UDP socket / datagram)

from mhd.backend.modes import BackendEnv, ModeError, validate_env

LEWEI_ENV = {"FLIGHT_MODE": "real", "DRONE_TYPE": "x69_lg", "MHD_BOARD": "lewei", "DRONE_IP": "192.168.0.1",
             "CONTROL_PORT": "50000", "X69_LG_VIDEO_MODE": "rtsp", "PLUGINS_ENABLED": "false",
             "X69_LG_RTSP_URL": "rtsp://127.0.0.1:8554/live/ch00_1"}
DRONE = ("192.168.0.1", 50000)


def fc_packet(volts_tenths: int, status: int, rest: bytes = b"\x00\x80\x00\x80\x00\x80") -> bytes:
    body = bytes([volts_tenths, status]) + rest
    ck = 0
    for b in body:
        ck ^= b
    return bytes([0x66]) + body + bytes([ck])


class FakeUdp:
    """Stands in for the adapter's UDP socket: records datagrams, never sends anything anywhere. Its
    fileno is an idle pipe, so the adapter's reader select()s on it and simply never wakes up."""

    def __init__(self, fail_with: int | None = None):
        self.sent: list[tuple[bytes, tuple]] = []
        self.fail_with = fail_with
        self._r, self._w = os.pipe()
        self.closed = False

    def sendto(self, data, addr):
        if self.fail_with is not None:
            raise OSError(self.fail_with, os.strerror(self.fail_with))
        self.sent.append((bytes(data), tuple(addr)))
        return len(data)

    def recvfrom(self, n):
        raise BlockingIOError(errno.EAGAIN, "no data")

    def fileno(self):
        return -1 if self.closed else self._r

    def setblocking(self, flag):
        self.blocking = flag

    def bind(self, addr):
        self.bound = addr

    def close(self):
        if not self.closed:
            self.closed = True
            os.close(self._r)
            os.close(self._w)


@pytest.fixture
def lewei(no_udp):  # noqa: F811
    from mhd.governor.turbodrone_glue import LeweiRcAdapter
    sock = FakeUdp()
    ad = LeweiRcAdapter(*DRONE, sock=sock)
    yield ad, sock
    ad.stop()
    assert no_udp == []


def _model(**flags):
    from models.x69_lg_rc import X69LgRcModel
    m = X69LgRcModel(profile="normal")
    m.speed_index = 2
    for k, v in flags.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------------------- mode gate

def test_lewei_is_accepted_only_for_the_real_drone_exactly():
    e = validate_env(dict(LEWEI_ENV))
    assert (e.flight_mode, e.board, e.drone_ip, e.camera_only) == ("real", "lewei", "192.168.0.1", False)
    x69 = validate_env({"FLIGHT_MODE": "real", "DRONE_IP": "172.16.11.1"})
    assert x69.board == "x69_lg"
    assert validate_env(dict(LEWEI_ENV, MHD_BOARD="X69_LG", DRONE_IP="172.16.11.1", X69_LG_RTSP_URL="")).board == "x69_lg"


@pytest.mark.parametrize("env,msg", [
    ({"FLIGHT_MODE": "twin", "DRONE_IP": "127.0.0.1"}, "only with FLIGHT_MODE=real"),
    ({"FLIGHT_MODE": "twin", "DRONE_IP": "127.0.0.1", "MHD_CAMERA_ONLY": "1"}, "only with FLIGHT_MODE=real"),
    ({"MHD_CAMERA_ONLY": "1"}, "MHD_CAMERA_ONLY=1"),
    ({"MHD_BOARD": "lw"}, "must be one of"),
    ({"DRONE_IP": "172.16.11.1"}, "DRONE_IP=192.168.0.1"),
    ({"DRONE_IP": "127.0.0.1"}, "DRONE_IP=192.168.0.1"),
    ({"DRONE_IP": "192.168.0.10"}, "DRONE_IP=192.168.0.1"),
    ({"CONTROL_PORT": "23458"}, "CONTROL_PORT=50000"),
    ({"CONTROL_PORT": ""}, "CONTROL_PORT=50000"),
    ({"CONTROL_PORT": "5e4"}, "CONTROL_PORT=50000"),
    ({"X69_LG_RTSP_URL": ""}, "loopback X69_LG_RTSP_URL"),
    ({"X69_LG_RTSP_URL": "rtsp://192.168.0.1:7060/live"}, "loopback X69_LG_RTSP_URL"),
    ({"X69_LG_RTSP_URL": "rtsp://172.16.11.1/live/ch00_1"}, "loopback X69_LG_RTSP_URL"),
    ({"X69_LG_VIDEO_MODE": "h265"}, "X69_LG_VIDEO_MODE=rtsp"),
    ({"X69_LG_RTSP_FFMPEG_OPTIONS": "rtsp_transport;udp"}, "RTSP over TCP"),
    ({"CONTROL_RATE": "30"}, "25 Hz"),
    ({"PLUGINS_ENABLED": "true"}, "PLUGINS_ENABLED"),
])
def test_lewei_gate_refuses_everything_else(env, msg):
    with pytest.raises(ModeError, match=msg):
        validate_env(dict(LEWEI_ENV, **env))


def test_x69_real_mode_checks_are_unchanged():
    with pytest.raises(ModeError, match="172.16.11.1"):
        validate_env({"FLIGHT_MODE": "real", "DRONE_IP": "192.168.0.1"})          # no MHD_BOARD: x69_lg
    with pytest.raises(ModeError, match="its host must be"):
        validate_env({"FLIGHT_MODE": "real", "DRONE_IP": "172.16.11.1",
                      "X69_LG_RTSP_URL": "rtsp://127.0.0.1:8554/live/ch00_1"})


# ------------------------------------------------------------------------------------ uplink

def test_sends_exactly_the_bare_20_byte_payload_to_the_drone(lewei):
    ad, sock = lewei
    m = _model(takeoff_flag=True, throttle=128.0, yaw=128.0, pitch=150.0, roll=128.0)
    pkt = ad.build_control_packet(m)
    ad.send_control_packet(pkt)
    assert len(pkt) == 20 and pkt[:2] == b"\x66\x14" and pkt[19] == 0x99
    assert pkt[2:6] == bytes([128, 150, 128, 128]) and pkt[6] == 0x01 and pkt[7] == 0x00
    assert pkt[8:18] == bytes(10)
    xor = 0
    for b in pkt[2:18]:
        xor ^= b
    assert pkt[18] == xor
    assert not pkt.startswith(b"\xca\x47")                      # no `ca 47 d5 00` + length wrapper
    assert sock.sent == [(pkt, DRONE)]
    assert m.takeoff_flag is False                              # one-shot flags clear after one packet
    pkt2 = ad.build_control_packet(_model(land_flag=True))
    assert pkt2[6] == 0x80 and ad.build_control_packet(_model())[6] == 0x00
    assert sock.blocking is False                               # a send can never stall the RC tick


def test_no_keepalive_only_rc_packets_go_out(lewei):
    ad, sock = lewei
    time.sleep(1.3)                                              # the X69 adapter would have sent D1 by now
    assert sock.sent == []
    for _ in range(5):
        ad.send_control_packet(ad.build_control_packet(_model()))
    assert len(sock.sent) == 5 and all(len(p) == 20 and a == DRONE for p, a in sock.sent)
    s = ad.link_stats()
    assert (s["rc_sent"], s["rc_errors"], s["keepalive_sent"]) == (5, 0, 0) and s["keepalive_age_ms"] < 100


def test_failed_sends_raise_the_existing_link_flags(no_udp):  # noqa: F811
    """No keepalive on this board: keepalive_last_ok means "an RC send succeeded", so a dead Wi-Fi shows
    as rc_send_errors at once and keepalive_stale after 3 s, exactly as on the X69."""
    from mhd.backend.hub import Hub
    from mhd.governor.turbodrone_glue import LeweiRcAdapter
    ad = LeweiRcAdapter(*DRONE, sock=FakeUdp(fail_with=errno.EHOSTUNREACH))
    try:
        hub = Hub()
        hub.rc_adapter = ad
        ad.send_control_packet(ad.build_control_packet(_model()))
        flags, extra = hub.link_flags_and_stats()
        assert flags == ["rc_send_errors"] and extra["rc_send_errors"] == 1
        ad.keepalive_last_ok = time.monotonic() - 4.0          # 4 s without one good send
        ad.send_control_packet(ad.build_control_packet(_model()))
        flags, _ = hub.link_flags_and_stats()
        assert flags == ["rc_send_errors", "keepalive_stale"]
        assert ad.link_stats()["rc_sent"] == 0
    finally:
        ad.stop()
    assert no_udp == []


# ---------------------------------------------------------------------------------- downlink

def test_parse_fc_telemetry():
    from mhd.governor.turbodrone_glue import parse_fc_telemetry
    assert parse_fc_telemetry(fc_packet(0x27, 0x00)) == (3.9, 0x00, b"\x00\x80\x00\x80\x00\x80")
    assert parse_fc_telemetry(fc_packet(0x28, 0x08))[:2] == (4.0, 0x08)
    good = fc_packet(0x27, 0x08)
    assert parse_fc_telemetry(good[:9] + bytes([good[9] ^ 1])) is None        # bad checksum
    assert parse_fc_telemetry(good[:9]) is None and parse_fc_telemetry(good + b"\x00") is None   # length
    assert parse_fc_telemetry(b"\x67" + good[1:]) is None                       # start byte
    assert parse_fc_telemetry(b"") is None


def test_downlink_counts_good_bad_and_foreign_packets_and_ages(lewei):
    ad, _ = lewei
    assert ad.fc_telemetry() == {"fc_battery_v": None, "fc_status": None, "fc_telemetry_age": None,
                                 "fc_packets_ok": 0, "fc_packets_bad": 0}
    assert ad.on_fc_packet(fc_packet(0x27, 0x08), DRONE)
    assert not ad.on_fc_packet(fc_packet(0x27, 0x08)[:-1] + b"\x00", DRONE)       # bad checksum
    assert not ad.on_fc_packet(fc_packet(0x27, 0x08), ("192.168.0.1", 50001))     # not the drone's port
    assert not ad.on_fc_packet(fc_packet(0x27, 0x08), ("192.168.0.7", 50000))     # not the drone
    t = ad.fc_telemetry()
    assert (t["fc_battery_v"], t["fc_status"], t["fc_packets_ok"], t["fc_packets_bad"]) == (3.9, 8, 1, 3)
    assert t["fc_telemetry_age"] < 200
    ad.on_fc_packet(fc_packet(0x26, 0x00), DRONE, now=time.monotonic() - 2.5)     # last good one 2.5 s ago
    t = ad.fc_telemetry()
    assert t["fc_battery_v"] == 3.8 and t["fc_telemetry_age"] >= 2500.0


def test_reader_thread_on_loopback_never_slows_the_rc_tick():
    """Both ends on 127.0.0.1: a stand-in drone floods the adapter's socket with downlink packets while
    the RC path sends; every send stays fast, and the reader counts what it got."""
    from mhd.governor.turbodrone_glue import LeweiRcAdapter
    drone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    drone.bind(("127.0.0.1", 0))
    drone.settimeout(2.0)
    mine = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mine.bind(("127.0.0.1", 0))
    ad = LeweiRcAdapter("127.0.0.1", drone.getsockname()[1], sock=mine)
    try:
        my_addr = mine.getsockname()
        worst = 0.0
        for i in range(200):
            for _ in range(10):                                    # 10 downlink packets per RC tick
                drone.sendto(fc_packet(0x27 + (i % 2), 0x08), my_addr)
            if i % 50 == 0:
                drone.sendto(b"\x66\x27", my_addr)                 # malformed
            t0 = time.perf_counter()
            ad.send_control_packet(ad.build_control_packet(_model()))
            worst = max(worst, time.perf_counter() - t0)
        assert worst < 0.02, f"an RC send took {worst * 1000:.1f} ms"
        got = [drone.recvfrom(64) for _ in range(200)]
        assert all(len(p) == 20 and p[:2] == b"\x66\x14" and a == my_addr for p, a in got)
        end = time.monotonic() + 3
        while ad.fc_telemetry()["fc_packets_ok"] < 2000 and time.monotonic() < end:
            time.sleep(0.02)
        t = ad.fc_telemetry()
        assert t["fc_packets_ok"] == 2000 and t["fc_packets_bad"] == 4, t
        assert t["fc_battery_v"] in (3.9, 4.0) and t["fc_status"] == 8
        assert ad.link_stats()["rc_sent"] == 200
    finally:
        ad.stop()
        drone.close()
    assert not ad._reader_thread.is_alive()


# ---------------------------------------------------------------------------------------- hub

def test_hub_installs_the_lewei_adapter_and_shows_its_telemetry(monkeypatch, no_udp, tmp_path):  # noqa: F811
    from mhd.backend.hub import Hub
    from mhd.governor import Governor
    from mhd.governor import turbodrone_glue as glue
    from mhd.governor.config import load_autopilot_config, load_governor_config
    fake = FakeUdp()
    monkeypatch.setattr(glue, "_udp_socket", lambda: fake)
    monkeypatch.delenv("MHD_CAMERA_ONLY", raising=False)
    hub = Hub()
    hub.env = BackendEnv("real", "x69_lg", "192.168.0.1", board="lewei")
    ad = hub.make_rc_adapter("192.168.0.1", 50000)
    try:
        assert isinstance(ad, glue.LeweiRcAdapter) and ad.sock is fake and fake.sent == []
        hub.gov = Governor(load_governor_config(ROOT / "config" / "governor.yaml"),
                           load_autopilot_config(ROOT / "config" / "autopilot.yaml"), "real",
                           log_dir=tmp_path, persist_timer=False, tick_log=False)
        ad.on_fc_packet(fc_packet(0x27, 0x08), DRONE)
        tel = json.loads(hub.telemetry_json())
        assert tel["flight_mode"] == "real" and tel["armed"] is False      # nothing else changes
        lat = tel["latency_ms"]
        assert lat["fc_battery_v"] == 3.9 and lat["fc_status"] == 8.0 and lat["fc_telemetry_age"] < 500
        assert lat["fc_packets_ok"] == 1.0 and "camera_only" not in tel["safety_flags"]
    finally:
        ad.stop()
    assert fake.sent == [] and no_udp == []


# ------------------------------------------------- the whole real-mode backend, Lewei board (fake socket)

@pytest.fixture
def lewei_backend(monkeypatch, tmp_path, no_udp):  # noqa: F811
    """The real lifespan (web_server + governor in REAL mode + PacedFlightController at 25 Hz) with the
    Lewei env. The adapter's socket is a FakeUdp, so every datagram is recorded and none is sent; the
    no-UDP guard would fail the test if anything created a real UDP socket."""
    import importlib

    from test_backend_api import FakeVideoAdapter
    from mhd.governor import turbodrone_glue as glue
    env = dict(LEWEI_ENV, LOG_LEVEL="WARNING", MHD_LOGS=str(tmp_path),
               GOVERNOR_CONFIG=str(ROOT / "config" / "governor.yaml"),
               AUTOPILOT_CONFIG=str(ROOT / "config" / "autopilot.yaml"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("MHD_CAMERA_ONLY", "GOVERNOR_TWIN_HEADLESS", "X69_LG_LOCAL_CONTROL_PORT", "CONTROL_RATE"):
        monkeypatch.delenv(k, raising=False)
    fake = FakeUdp()
    monkeypatch.setattr(glue, "_udp_socket", lambda: fake)
    ws_mod = importlib.import_module("web_server")
    monkeypatch.setattr(ws_mod, "TimedX69RtspAdapter", FakeVideoAdapter)
    from fastapi.testclient import TestClient
    with TestClient(ws_mod.app) as client:
        yield client, ws_mod.HUB, fake


def _until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_real_lewei_backend_end_to_end_bytes(lewei_backend, no_udp):  # noqa: F811
    from mhd.governor.turbodrone_glue import LeweiRcAdapter
    client, hub, fake = lewei_backend
    assert hub.env.board == "lewei" and hub.gov.flight_mode == "real" and isinstance(hub.rc_adapter, LeweiRcAdapter)
    assert _until(lambda: len(fake.sent) >= 25)                         # the 25 Hz RC loop is running
    assert {a for _, a in fake.sent} == {DRONE}
    assert all(len(p) == 20 and p[:2] == b"\x66\x14" and p[19] == 0x99 for p, _ in fake.sent)
    assert all(p[2:6] == bytes([128] * 4) and p[6:8] == b"\x00\x00" for p, _ in fake.sent)   # disarmed: neutral
    hub.rc_adapter.on_fc_packet(fc_packet(0x27, 0x00), DRONE)
    st = client.get("/state").json()
    assert st["flight_mode"] == "real" and st["armed"] is False and st["latency_ms"]["fc_battery_v"] == 3.9
    assert not {"rc_send_errors", "keepalive_stale", "camera_only"} & set(st["safety_flags"])
    hb = json.dumps({"type": "hb", "client": "ui"})
    with client.websocket_connect("/ws") as ws:
        ws.send_text(hb)
        ws.send_text(json.dumps({"type": "cmd", "command": "takeoff"}))         # not armed: refused
        time.sleep(0.2)
        assert hub.gov.state.value == "ON_GROUND"
        ws.send_text(json.dumps({"type": "cmd", "command": "arm", "arm_phrase": "ARM"}))
        assert _until(lambda: hub.gov.armed)
        n0 = len(fake.sent)
        ws.send_text(hb)
        ws.send_text(json.dumps({"type": "cmd", "command": "takeoff"}))
        assert _until(lambda: any(p[6] & 0x01 for p, _ in fake.sent[n0:]))       # 0x01 reached the wire
        assert hub.gov.state.value == "TAKING_OFF"
        n1 = len(fake.sent)
        ws.send_text(hb)
        ws.send_text(json.dumps({"type": "cmd", "command": "land"}))
        assert _until(lambda: any(p[6] & 0x80 for p, _ in fake.sent[n1:]))       # LAND 0x80 too
    takeoffs = [p for p, _ in fake.sent if p[6] & 0x01]
    assert len(takeoffs) == 1, "takeoff.packets is 1: exactly one packet carries 0x01"
    assert not any(p[6] & 0x0E for p, _ in fake.sent), "no E-STOP, calibrate or flip bits were sent"
    client.__exit__(None, None, None)                                    # shutdown: neutral + land, then stop
    assert fake.closed and no_udp == []
