"""RC packets built by the real turbodrone X69 adapter decode correctly in the twin."""
from __future__ import annotations

import random
import socket
import time

import pytest

from mhd.twin import protocol as P


def _model(upstream, speed_index=2):
    Model, _ = upstream
    m = Model()
    m.speed_index = speed_index
    return m


def test_idle_packet_decodes_to_center(upstream, offline_adapter):
    pkt = offline_adapter.build_control_packet(_model(upstream))
    assert len(pkt) == P.DATAGRAM_LEN == 28
    d = P.decode_datagram(pkt)
    assert isinstance(d, P.RcPacket)
    assert d.sticks() == {"roll": 128, "pitch": 128, "throttle": 128, "yaw": 128}
    assert d.flags1 == 0 and d.flags2 == 0 and not d.reserved_nonzero


def test_random_sticks_roundtrip_all_speed_tiers(upstream, offline_adapter):
    rng = random.Random(7)
    for _ in range(500):
        speed = rng.choice([0, 1, 2])
        m = _model(upstream, speed)
        m.roll, m.pitch, m.throttle, m.yaw = (rng.uniform(0, 255) for _ in range(4))
        expected_payload = offline_adapter.build_control_payload(m)
        # build_control_payload clears nothing stick-related, so build the full packet the same way
        pkt = offline_adapter._wrap_control_payload(expected_payload)
        d = P.decode_datagram(pkt)
        assert isinstance(d, P.RcPacket)
        assert (d.roll, d.pitch, d.throttle, d.yaw) == tuple(expected_payload[2:6])
        # our own encoder produces the identical bytes
        assert P.encode_control(d.roll, d.pitch, d.throttle, d.yaw, d.flags1, d.flags2) == pkt


def test_one_shot_flags(upstream, offline_adapter):
    m = _model(upstream)
    for setter, attr, value in ((m.takeoff, "takeoff", P.FLAG_TAKEOFF), (m.land, "land", P.FLAG_LAND),
                                (m.emergency_stop, "estop", P.FLAG_ESTOP),
                                (m.calibrate_gyro, "calibrate", P.FLAG_CALIBRATE), (m.flip, "flip", P.FLAG_FLIP)):
        setter()
        d = P.decode_datagram(offline_adapter.build_control_packet(m))
        assert isinstance(d, P.RcPacket) and d.flags1 == value and getattr(d, attr)
        # one-shot: the next packet has the flag cleared
        d2 = P.decode_datagram(offline_adapter.build_control_packet(m))
        assert d2.flags1 == 0
    # all at once (upstream test expects 0x8F)
    m.takeoff(); m.land(); m.emergency_stop(); m.calibrate_gyro(); m.flip()
    d = P.decode_datagram(offline_adapter.build_control_packet(m))
    assert d.flags1 == 0x8F and d.unknown_flags1 == 0
    assert set(d.flag_names()) == {"takeoff", "land", "estop", "calibrate", "flip"}


def test_camera_tilt_bits(upstream, offline_adapter):
    m = _model(upstream)
    m.set_camera_tilt_state(2)
    d = P.decode_datagram(offline_adapter.build_control_packet(m))
    assert d.tilt_up and not d.tilt_down and d.flags2 == P.FLAG2_TILT_UP
    m.set_camera_tilt_state(1)
    d = P.decode_datagram(offline_adapter.build_control_packet(m))
    assert d.tilt_down and not d.tilt_up and d.flags2 == P.FLAG2_TILT_DOWN
    # tilt is held, not one-shot
    d = P.decode_datagram(offline_adapter.build_control_packet(m))
    assert d.tilt_down
    m.set_camera_tilt_state(0)
    assert P.decode_datagram(offline_adapter.build_control_packet(m)).flags2 == 0


def test_keepalive_is_not_a_control_packet(upstream):
    _, Adapter = upstream
    assert isinstance(P.decode_datagram(Adapter.D1_KEEPALIVE), P.Keepalive)
    assert Adapter.D1_KEEPALIVE == P.KEEPALIVE


def test_corrupt_packets_are_rejected(upstream, offline_adapter):
    good = bytearray(offline_adapter.build_control_packet(_model(upstream)))
    bad = bytearray(good); bad[8 + 5] ^= 0x01          # yaw byte flipped, checksum now wrong
    assert P.decode_datagram(bytes(bad)) == P.DecodeError("bad_checksum", bytes(bad[8:]))
    bad = bytearray(good); bad[8] = 0x65               # start marker
    assert P.decode_datagram(bytes(bad)).reason == "bad_marker"
    bad = bytearray(good); bad[8 + 19] = 0x98          # end marker
    assert P.decode_datagram(bytes(bad)).reason == "bad_marker"
    assert P.decode_datagram(bytes(good[:-1])).reason == "bad_length"
    assert P.decode_datagram(b"\x00" * 28).reason == "bad_magic"
    assert P.decode_datagram(b"").reason == "empty"


def test_count_to_unit():
    assert P.count_to_unit(128) == 0.0
    assert P.count_to_unit(255) == 1.0
    assert P.count_to_unit(0) == -1.0
    assert abs(P.count_to_unit(160) - 32 / 127) < 1e-12


def test_real_adapter_over_udp_including_keepalive(upstream):
    """The real adapter object (socket + keepalive thread) talks to a loopback UDP socket."""
    Model, Adapter = upstream
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 23529))
    rx.settimeout(2.5)
    adapter = Adapter("127.0.0.1", 23529)
    try:
        m = Model()
        m.speed_index = 2
        m.yaw = 200.0
        m.takeoff()
        adapter.send_control_packet(adapter.build_control_packet(m))
        got_control = got_keepalive = False
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and not (got_control and got_keepalive):
            data, _ = rx.recvfrom(64)
            d = P.decode_datagram(data)
            if isinstance(d, P.RcPacket):
                assert d.yaw == 200 and d.takeoff
                got_control = True
            elif isinstance(d, P.Keepalive):
                got_keepalive = True
        assert got_control and got_keepalive
    finally:
        adapter.stop()
        rx.close()
