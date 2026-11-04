"""Camera-only mode (MHD_CAMERA_ONLY=1, scripts/run_stack.sh --mode camera): the real camera feeds the
pipeline, the governor and autopilot run as in twin mode, and NOTHING is sent over UDP, to any address.

Covers the mode gate, the RC adapter without a socket (every UDP socket creation and datagram send in
this process fails the test), the telemetry flag, the config/camera files and the RTSP readiness check
that run_stack.sh uses to wait for the Lewei bridge.
"""
from __future__ import annotations

import importlib
import json
import socket
import threading
import time

import pytest

from conftest import ROOT
from test_backend_api import BASE_ENV, FakeVideoAdapter, _airborne

from mhd.backend.modes import CAMERA_ONLY_FLAG, ModeError, validate_env

CAMERA_ENV = dict(BASE_ENV, MHD_CAMERA_ONLY="1", CONTROL_PORT="23999",
                  GOVERNOR_CONFIG=str(ROOT / "config" / "camera" / "governor.yaml"),
                  AUTOPILOT_CONFIG=str(ROOT / "config" / "camera" / "autopilot.yaml"))


@pytest.fixture
def no_udp(monkeypatch):
    """Records (and refuses) every UDP socket and every datagram send made in this process. Threads
    swallow the AssertionError (the RC loop survives anything), so tests assert the list is empty."""
    events: list = []
    orig_init = socket.socket.__init__

    def guarded_init(self, family=-1, type=-1, proto=-1, fileno=None):
        if fileno is None and type == socket.SOCK_DGRAM:
            events.append(("udp_socket", family))
            raise AssertionError("camera-only mode created a UDP socket")
        orig_init(self, family, type, proto, fileno)

    def refuse(name):
        def send(self, *args, **kwargs):
            events.append((name, args[1:] if len(args) > 1 else args))
            raise AssertionError(f"camera-only mode called socket.{name}")
        return send

    monkeypatch.setattr(socket.socket, "__init__", guarded_init)
    for name in ("sendto", "sendmsg"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    return events


# ---------------------------------------------------------------------------------------- mode gate

@pytest.mark.parametrize("env,msg", [
    ({"FLIGHT_MODE": "real", "DRONE_IP": "172.16.11.1", "X69_LG_RTSP_URL": ""}, "only with FLIGHT_MODE=twin"),
    ({"FLIGHT_MODE": "real", "DRONE_IP": "172.16.11.1",
      "X69_LG_RTSP_URL": "rtsp://172.16.11.1/live/ch00_1"}, "only with FLIGHT_MODE=twin"),
    ({"FLIGHT_MODE": "real"}, "only with FLIGHT_MODE=twin"),
    ({"DRONE_IP": "192.168.0.1"}, "non-loopback DRONE_IP"),
    ({"DRONE_IP": "172.16.11.1"}, "non-loopback DRONE_IP"),
    ({"X69_LG_RTSP_URL": "rtsp://192.168.0.1:8554/live/ch00_1"}, "non-loopback X69_LG_RTSP_URL"),
    ({"X69_LG_RTSP_URL": ""}, "needs a loopback X69_LG_RTSP_URL"),
    ({"X69_LG_VIDEO_MODE": "h265"}, "X69_LG_VIDEO_MODE=rtsp"),
    ({"X69_LG_VIDEO_MODE": "jpeg"}, "X69_LG_VIDEO_MODE=rtsp"),
    ({"X69_LG_RTSP_FFMPEG_OPTIONS": "rtsp_transport;udp"}, "RTSP over TCP"),
    ({"MHD_CAMERA_ONLY": "maybe"}, "must be 1 or 0"),
])
def test_camera_only_is_refused_outside_twin_on_loopback(env, msg):
    with pytest.raises(ModeError, match=msg):
        validate_env(dict(CAMERA_ENV, **env))


def test_camera_only_is_accepted_in_twin_on_loopback():
    e = validate_env(dict(CAMERA_ENV))
    assert (e.flight_mode, e.drone_ip, e.camera_only) == ("twin", "127.0.0.1", True)
    assert validate_env(dict(CAMERA_ENV, X69_LG_RTSP_URL="rtsp://localhost:8554/live/ch00_1")).camera_only
    for off in ("0", "", "false", "off"):
        assert validate_env(dict(CAMERA_ENV, MHD_CAMERA_ONLY=off)).camera_only is False
    plain = dict(BASE_ENV)
    assert validate_env(plain).camera_only is False


# ------------------------------------------------------------------------------- RC adapter

def test_the_udp_guard_is_not_vacuous(no_udp):
    """The twin's adapter opens a UDP socket at once: the guard must catch exactly that."""
    from mhd.governor.turbodrone_glue import GovernedX69LgRcAdapter
    with pytest.raises(AssertionError, match="UDP socket"):
        GovernedX69LgRcAdapter("127.0.0.1", 23514)
    assert no_udp and no_udp[0][0] == "udp_socket"
    no_udp.clear()


def test_camera_only_adapter_never_sends_and_keeps_the_keepalive_fresh(no_udp):
    from models.x69_lg_rc import X69LgRcModel
    from mhd.backend.hub import Hub
    from mhd.governor.turbodrone_glue import CameraOnlyRcAdapter, CameraOnlyViolation

    ad = CameraOnlyRcAdapter("127.0.0.1", 23999, local_port=23515)
    try:
        model = X69LgRcModel(profile="normal")
        model.takeoff_flag = True
        for _ in range(50):
            ad.send_control_packet(ad.build_control_packet(model))
        assert model.takeoff_flag is False          # one-shot flags handled exactly as in the twin
        first_ok = ad.keepalive_last_ok
        time.sleep(2.3)
        s = ad.link_stats()
        assert ad.keepalive_last_ok >= first_ok + 1.5, "the keepalive loop must refresh keepalive_last_ok"
        assert (s["rc_sent"], s["rc_errors"], s["keepalive_sent"], s["keepalive_errors"]) == (0, 0, 0, 0)
        assert s["rc_suppressed"] == 50 and s["camera_only"] is True
        assert s["keepalive_alive"] and s["keepalive_age_ms"] < 1500.0
        hub = Hub()
        hub.rc_adapter = ad
        flags, extra = hub.link_flags_and_stats()
        assert flags == [] and extra["rc_send_errors"] == 0
        with pytest.raises(CameraOnlyViolation):    # there is no socket behind the adapter at all
            ad.sock.sendto(b"\xca\x47\xd1\x00", ("127.0.0.1", 23999))
    finally:
        ad.stop()
    assert not ad._keepalive_thread.is_alive()
    assert no_udp == []


@pytest.mark.parametrize("value", ["1", "true", "maybe"])
def test_hub_picks_the_socketless_adapter_whenever_the_flag_is_set(monkeypatch, no_udp, value):
    """Fail safe: even before setup() (no validated env), any MHD_CAMERA_ONLY that is not clearly off
    gets the adapter without a socket."""
    from mhd.backend.hub import Hub
    from mhd.governor.turbodrone_glue import CameraOnlyRcAdapter
    monkeypatch.setenv("MHD_CAMERA_ONLY", value)
    hub = Hub()
    ad = hub.make_rc_adapter("127.0.0.1", 23999, local_port=23515)
    try:
        assert isinstance(ad, CameraOnlyRcAdapter) and hub.camera_only
    finally:
        ad.stop()
    assert no_udp == []


def test_hub_keeps_the_twin_adapter_without_the_flag(monkeypatch):
    from mhd.backend.hub import Hub
    from mhd.governor.turbodrone_glue import CameraOnlyRcAdapter, GovernedX69LgRcAdapter
    monkeypatch.delenv("MHD_CAMERA_ONLY", raising=False)
    hub = Hub()
    ad = hub.make_rc_adapter("127.0.0.1", 23514)
    try:
        assert isinstance(ad, GovernedX69LgRcAdapter) and not isinstance(ad, CameraOnlyRcAdapter)
        assert not hub.camera_only
    finally:
        ad.stop()


# --------------------------------------------------------------------- the whole backend, camera-only

@pytest.fixture
def camera_backend(monkeypatch, tmp_path, no_udp):
    for k, v in CAMERA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    ws_mod = importlib.import_module("web_server")
    monkeypatch.setattr(ws_mod, "TimedX69RtspAdapter", FakeVideoAdapter)
    from fastapi.testclient import TestClient
    with TestClient(ws_mod.app) as client:
        yield client, ws_mod.HUB, no_udp


def test_camera_only_backend_runs_the_governor_and_sends_nothing(camera_backend):
    """The real lifespan (governor, PacedFlightController at 25 Hz, frame pump) with MHD_CAMERA_ONLY=1:
    takeoff -> AIRBORNE as in the twin, telemetry says camera_only with a healthy link, and not one
    UDP socket or datagram, through the landing at shutdown."""
    from mhd.governor.turbodrone_glue import CameraOnlyRcAdapter
    client, hub, events = camera_backend
    assert hub.env.camera_only and isinstance(hub.rc_adapter, CameraOnlyRcAdapter)
    assert hub.gov.flight_mode == "twin" and hub.gov.cfg.wrong_way.enabled is False
    with client.websocket_connect("/ws") as ws:
        _airborne(client, hub, ws)
        ws.send_text(json.dumps({"type": "hb"}))
        st = client.get("/state").json()
    flags = st["safety_flags"]
    assert st["flight_mode"] == "twin" and st["flight_state"] == "AIRBORNE"
    assert CAMERA_ONLY_FLAG in flags
    assert not {"keepalive_stale", "rc_send_errors", "rc_tick_stale", "governor_fault"} & set(flags), flags
    stats = hub.rc_adapter.link_stats()
    assert stats["rc_sent"] == 0 and stats["keepalive_sent"] == 0 and stats["rc_suppressed"] > 75
    assert stats["keepalive_age_ms"] < 1500.0
    gov = hub.gov
    client.__exit__(None, None, None)         # lifespan shutdown: neutral + land for 0.5 s, RC loop stops
    assert gov._shutdown is True
    assert events == [], events


def test_twin_backend_telemetry_has_no_camera_only_flag(monkeypatch, tmp_path):
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MHD_CAMERA_ONLY", raising=False)
    monkeypatch.setenv("MHD_LOGS", str(tmp_path))
    ws_mod = importlib.import_module("web_server")
    monkeypatch.setattr(ws_mod, "TimedX69RtspAdapter", FakeVideoAdapter)
    from fastapi.testclient import TestClient
    with TestClient(ws_mod.app) as client:
        assert not ws_mod.HUB.env.camera_only
        assert CAMERA_ONLY_FLAG not in client.get("/state").json()["safety_flags"]


# ------------------------------------------------------------------------------------ configs

def test_camera_configs_are_twin_only_with_the_wrong_way_detectors_off(tmp_path):
    from mhd.governor import Governor
    from mhd.governor.config import WrongWayCfg, load_autopilot_config, load_governor_config
    from mhd.governor.core import GovernorError
    gpath, apath = ROOT / "config" / "camera" / "governor.yaml", ROOT / "config" / "camera" / "autopilot.yaml"
    gov, ap = load_governor_config(gpath), load_autopilot_config(apath)
    assert gov.twin_only and ap.twin_only
    assert gov.wrong_way.enabled is False
    assert ap.camera.hfov_deg == 75.0 and "Lewei camera, not measured yet" in apath.read_text()
    for p in (gpath, apath):
        assert "CAMERA ONLY, NO FLIGHT" in p.read_text()
    with pytest.raises(GovernorError, match="twin-only"):
        Governor(gov, ap, "real", log_dir=tmp_path, tick_log=False)
    # Everything else keeps the detectors on (the default is unchanged).
    assert WrongWayCfg().enabled is True
    for name in ("governor.yaml", "twin/governor.yaml"):
        assert load_governor_config(ROOT / "config" / name).wrong_way.enabled is True


# ------------------------------------------------------------------- RTSP readiness (run_stack.sh)

def _fake_rtsp_server(reply: bytes):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got: list[bytes] = []

    def run():
        try:
            c, _ = srv.accept()
            with c:
                got.append(c.recv(4096))
                c.sendall(reply)
        finally:
            srv.close()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return srv.getsockname()[1], got, th


@pytest.mark.parametrize("reply,ok", [
    (b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Type: application/sdp\r\n\r\nv=0\r\n", True),
    (b"RTSP/1.0 404 Not Found\r\nCSeq: 1\r\n\r\n", False),
    (b"garbage\r\n", False),
])
def test_rtsp_has_publisher_reads_the_describe_status(reply, ok):
    from mhd.twin.media import rtsp_has_publisher
    port, got, th = _fake_rtsp_server(reply)
    url = f"rtsp://127.0.0.1:{port}/live/ch00_1"
    assert rtsp_has_publisher(url, timeout=2.0) is ok
    th.join(2)
    assert got and got[0].startswith(f"DESCRIBE {url} RTSP/1.0\r\n".encode())


def test_rtsp_has_publisher_is_loopback_only_and_false_when_nothing_listens():
    from mhd.twin.media import rtsp_has_publisher
    with socket.socket() as s:                 # a port that was free a moment ago
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert rtsp_has_publisher(f"rtsp://127.0.0.1:{port}/live/ch00_1", timeout=0.5) is False
    for url in ("rtsp://192.168.0.1:8554/live/ch00_1", "rtsp://172.16.11.1/live/ch00_1", "http://127.0.0.1:8554/x"):
        with pytest.raises(ValueError):
            rtsp_has_publisher(url)
