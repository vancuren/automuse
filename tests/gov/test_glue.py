"""GovernorStrategy and the guarded X69 model: the turbodrone choke point."""
from __future__ import annotations

from mhd.governor.turbodrone_glue import GovernedX69LgRcModel, GovernorStrategy, install_governor

from control.strategies import DirectStrategy, IncrementalStrategy
from protocols.x69_lg_rc_protocol_adapter import X69LgRcProtocolAdapter


def _adapter():
    a = X69LgRcProtocolAdapter.__new__(X69LgRcProtocolAdapter)    # no socket, no keepalive thread
    a.speed_scale_by_index = {0: 0.55, 1: 0.75, 2: 1.0}
    return a


def test_set_strategy_cannot_replace_the_governor(rig):
    r = rig()
    model = GovernedX69LgRcModel(r.g)
    gs = model.strategy
    assert isinstance(gs, GovernorStrategy)
    model.set_strategy(DirectStrategy())
    model.set_strategy(IncrementalStrategy())
    assert model.strategy is gs
    model.set_speed_index(0)
    model.set_profile("aggressive")
    assert model.speed_index == 2


def test_update_writes_exact_wire_counts_and_ignores_axes(rig):
    r = rig()
    r.airborne()
    model = GovernedX69LgRcModel(r.g)
    r.g.submit_human({"yaw": 0.3, "pitch": -0.3}, "abs", now=r.t)
    model.update(0.04, {"throttle": 1.0, "yaw": -1.0, "pitch": 1.0, "roll": 1.0})   # ignored
    pkt = _adapter().build_control_payload(model)
    assert pkt[5] == 128 + 38 and pkt[3] == 128 - 38 and pkt[2] == 128 and pkt[4] == 128


def test_legacy_methods_route_through_the_governor(rig):
    r = rig()
    model = GovernedX69LgRcModel(r.g)
    r.tick(2)
    model.takeoff()
    assert r.g.state.value == "TAKING_OFF"
    model.emergency_stop()
    assert r.g.state.value == "ESTOPPED"
    model.flip()
    model.calibrate_gyro()
    model.update(0.04, {})
    assert not model.flip_flag and not model.calibration_flag and model.stop_flag


def test_flip_and_calibrate_masked_even_if_set_directly(rig):
    r = rig()
    r.airborne()
    model = GovernedX69LgRcModel(r.g)
    model.flip_flag = True
    model.calibration_flag = True
    model.update(0.04, {})
    pkt = _adapter().build_control_payload(model)
    assert pkt[6] & 0x08 == 0 and pkt[6] & 0x04 == 0


def test_tick_exception_falls_back_to_neutral_keeping_estop(rig, monkeypatch):
    r = rig()
    r.airborne()
    model = GovernedX69LgRcModel(r.g)
    r.g.command("estop")

    def boom(*a, **k):
        raise RuntimeError("synthetic tick failure")
    monkeypatch.setattr(r.g, "tick", boom)
    model.update(0.04, {})
    pkt = _adapter().build_control_payload(model)
    assert pkt[2:6] == bytes([128, 128, 128, 128]) and pkt[6] == 0x02
    assert model.strategy.alarm


def test_persistent_faults_also_reassert_land(rig, monkeypatch):
    r = rig()
    r.airborne()
    model = GovernedX69LgRcModel(r.g)

    def boom(*a, **k):
        raise RuntimeError("synthetic tick failure")
    monkeypatch.setattr(r.g, "tick", boom)
    lands = []
    for _ in range(100):                      # 4 s of failures at 25 Hz
        model.update(0.04, {})
        lands.append(_adapter().build_control_payload(model)[6] & 0x80 != 0)
        r.t += 40.0
    assert not any(lands[:45])                # first 2 s: neutral only
    assert sum(lands[55:]) >= 8               # then land at 5 Hz


def test_install_governor_starts_housekeeping(rig):
    r = rig()
    model = install_governor(r.g)
    assert isinstance(model.strategy, GovernorStrategy)
    assert r.g._hk_thread is not None and r.g._hk_thread.is_alive()
