"""Twin server over real UDP, driven by the real turbodrone adapter (no video, no backend).

Covers: packet decode and stats on the wire, takeoff/land/E-stop through the server,
link loss while the adapter's keepalive thread keeps running, RC loss injection, control API.
"""
from __future__ import annotations

import threading
import time

import pytest

from procs import start_twin, twin_control, twin_status, wait_until

RC_PORT = 23521

pytestmark = pytest.mark.slow


class AdapterLoop:
    """Runs the real upstream model + adapter at 25 Hz, like turbodrone's FlightController."""

    def __init__(self, upstream, port: int):
        Model, Adapter = upstream
        self.model = Model()
        self.model.speed_index = 2
        self.adapter = Adapter("127.0.0.1", port)
        self.running = True
        self.paused = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while self.running:
            if not self.paused:
                self.adapter.send_control_packet(self.adapter.build_control_packet(self.model))
            time.sleep(0.04)

    def stop(self):
        self.running = False
        self.th.join(timeout=1)
        self.adapter.stop()


@pytest.fixture(scope="module")
def twin():
    p = start_twin("s1", rc_port=RC_PORT, extra=["--no-video"])
    yield p
    p.stop()


@pytest.fixture
def rc(upstream, twin):
    twin_control(reset=True, rc_loss=0.0, rc_burst_rate=0.0, rc_jitter_ms=0.0)
    loop = AdapterLoop(upstream, RC_PORT)
    yield loop
    loop.stop()


def phase():
    return twin_status()["drone"]["phase"]


def test_packets_and_keepalives_are_counted(rc):
    time.sleep(2.5)
    st = twin_status()["rc"]
    assert st["packets"] >= 40
    assert st["keepalives"] >= 1           # the adapter's own d1 keepalive thread
    assert not st["errors"]
    assert 30 <= st["interarrival_p50_ms"] <= 60


def test_takeoff_yaw_land_through_the_wire(rc):
    rc.model.takeoff()
    wait_until(lambda: phase() == "flying", 6, what="flying")
    z = twin_status()["drone"]["z"]
    assert 0.9 <= z <= 1.1
    rc.model.yaw = 230.0                      # DirectStrategy-style raw value
    time.sleep(1.5)
    d = twin_status()["drone"]
    assert d["yaw_rate_dps"] > 60
    rc.model.yaw = 128.0
    rc.model.land()
    wait_until(lambda: phase() == "landed", 6, what="landed")
    assert twin_status()["drone"]["events"]["land"] == 1


def test_estop_drops_the_drone(rc):
    rc.model.takeoff()
    wait_until(lambda: phase() == "flying", 6, what="flying")
    rc.model.emergency_stop()
    wait_until(lambda: not twin_status()["drone"]["motors_on"], 1.0, what="motors off")
    wait_until(lambda: phase() == "landed", 2, what="on the ground")


def test_link_loss_ignores_keepalive(rc):
    """Stop RC packets while the adapter's keepalive thread keeps sending: the twin must land."""
    rc.model.takeoff()
    wait_until(lambda: phase() == "flying", 6, what="flying")
    rc.paused = True
    ka0 = twin_status()["rc"]["keepalives"]
    st = wait_until(lambda: (lambda s: s if s["link"]["lost"] else None)(twin_status()), 3.0, what="link lost")
    assert st["drone"]["phase"] in ("landing", "landed")
    time.sleep(1.2)
    assert twin_status()["rc"]["keepalives"] > ka0        # keepalives kept coming the whole time
    wait_until(lambda: phase() == "landed", 6, what="landed after link loss")
    rc.paused = False
    wait_until(lambda: not twin_status()["link"]["lost"], 2, what="link restored")


def test_injected_total_rc_loss_triggers_link_loss(rc):
    rc.model.takeoff()
    wait_until(lambda: phase() == "flying", 6, what="flying")
    twin_control(rc_loss=1.0)
    try:
        wait_until(lambda: twin_status()["link"]["lost"], 3.0, what="link lost under 100% loss")
        assert twin_status()["rc"]["faults"]["dropped"] > 10
    finally:
        twin_control(rc_loss=0.0)


def test_control_api_rejects_unknown_fields_and_scenarios(twin):
    import httpx
    r = httpx.post("http://127.0.0.1:8266/control", json={"bogus": 1}, timeout=3)
    assert r.status_code == 422
    r = httpx.post("http://127.0.0.1:8266/control", json={"scenario": "nope"}, timeout=3)
    assert r.status_code == 404
    out = twin_control(scenario="s2")
    assert out["status"]["scenario"] == "s2"
    twin_control(scenario="s1")
