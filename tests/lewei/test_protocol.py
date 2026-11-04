"""Lewei camera client: only the three video commands can go out, and frames are parsed and unscrambled exactly."""
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mhd.lewei import protocol as p
from mhd.lewei.tables import TABLE1, TABLE2, TABLE3

TREE = Path(__file__).resolve().parents[2]


def _hdr(cmd, body_size=0, stream_type=0, dec1=0, dec2=0):
    return b"lewei_cmd\x00" + struct.pack("<9I", cmd, 0, 0, body_size, stream_type, dec1, dec2, 0, 0)


def _munge_new(frame: bytes):
    """Scramble like the camera (type 129): 3 bytes at size//2 become junk; the header carries them."""
    idx = len(frame) >> 1
    o0, o1, o2 = frame[idx], frame[idx + 1], frame[idx + 2]
    dec1 = TABLE1[o0] | (TABLE2[o1 ^ o0] << 16)
    dec2 = TABLE3[o2 ^ o0 ^ o1]
    junk = bytearray(frame)
    junk[idx:idx + 3] = bytes([(o0 + 77) & 255, (o1 + 13) & 255, (o2 + 5) & 255])
    return bytes(junk), dec1, dec2


def _frame_packet(data: bytes, count: int):
    munged, dec1, dec2 = _munge_new(data)
    body = struct.pack("<LLQL", 1, len(munged), count, 0) + b"\0" * 12 + munged
    return _hdr(p.RET_STREAM, len(body), p.STREAM_NEW, dec1, dec2) + body


def test_only_the_three_video_commands_can_be_encoded():
    # Wi-Fi settings, SD card format, flight-controller baud rate, config writes...: all refused.
    for cmd in (0, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 32, 34, 35, 36, 37, 38, 41, 49, 257):
        with pytest.raises(ValueError):
            p.encode_command(cmd)
    assert p.encode_command(p.CMD_START_STREAM, 1) == b"lewei_cmd\x00" + struct.pack("<9I", 2, 1, 0, 0, 0, 0, 0, 0, 0)
    assert len(p.encode_command(p.CMD_HEARTBEAT)) == p.HDR_LEN == 46


def test_parse_header_rejects_garbage():
    with pytest.raises(ValueError):
        p.parse_header(b"x" * 46)
    with pytest.raises(ValueError):
        p.parse_header(b"lewei_cmd\x00")
    h = p.parse_header(_hdr(257, 1234, 129, 7, 9))
    assert (h.cmd, h.body_size, h.stream_type, h.dec1, h.dec2) == (257, 1234, 129, 7, 9)


def test_type_129_unscramble_restores_every_byte_value():
    for v in range(256):
        data = bytes((i * 31 + v) & 255 for i in range(1001))
        idx = len(data) >> 1
        data = data[:idx] + bytes([v, (v * 7) & 255, (v * 13 + 1) & 255]) + data[idx + 3:]
        munged, dec1, dec2 = _munge_new(data)
        frame = bytearray(munged)
        p.unmunge(frame, p.STREAM_NEW, dec1, dec2, count=0)
        assert bytes(frame) == data, v


def test_type_1_inverts_one_byte_and_type_0_is_untouched():
    frame = bytearray(100)
    p.unmunge(frame, p.STREAM_ONE, 0, 0, count=7)
    assert frame.count(0xFF) == 1 and frame[p._fix_byte(7, 100)] == 0xFF
    frame = bytearray(b"abc" * 10)
    p.unmunge(frame, p.STREAM_PLAIN, 1, 2, count=3)
    assert bytes(frame) == b"abc" * 10


def test_incomplete_frame_is_an_error():
    body = struct.pack("<LLQL", 1, 500, 0, 0) + b"\0" * 12 + b"\1" * 10
    with pytest.raises(ValueError):
        p.parse_video(p.parse_header(_hdr(257, len(body), 0)), body, 0.0)


class _FakeCamera:
    """A loopback stand-in for the camera's video port: records every byte the client sends."""

    def __init__(self, frames, keep_streaming=False):
        self.frames, self.keep_streaming = frames, keep_streaming
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.received = bytearray()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        c, _ = self.srv.accept()
        c.settimeout(0.05)
        try:
            c.sendall(_hdr(p.CMD_HEARTBEAT, 64) + b"\0" * 64)      # a heartbeat reply comes first
            n = 0
            while True:
                if n < len(self.frames) or self.keep_streaming:
                    c.sendall(_frame_packet(self.frames[n % len(self.frames)], n))
                    n += 1
                try:
                    d = c.recv(4096)
                    if not d:
                        break
                    self.received += d
                except socket.timeout:
                    pass
        except OSError:
            pass
        finally:
            c.close()
            self.srv.close()

    def commands(self):
        sent = bytes(self.received)
        assert len(sent) % p.HDR_LEN == 0
        for i in range(0, len(sent), p.HDR_LEN):
            assert sent[i:i + 10] == b"lewei_cmd\x00"
        return [struct.unpack_from("<I", sent, i + 10)[0] for i in range(0, len(sent), p.HDR_LEN)]


def test_client_reads_unscrambled_frames_and_sends_only_video_commands():
    originals = [bytes((i * k) & 255 for i in range(2048)) for k in (3, 5, 7)]
    cam = _FakeCamera(originals)
    with p.LeweiStream("127.0.0.1", cam.port, heartbeat_s=0.0, read_timeout=2.0) as s:
        it = s.frames()
        got = [next(it) for _ in originals]
    cam.th.join(5)
    assert [f.data for f in got] == originals
    assert all(f.stream_type == p.STREAM_NEW for f in got)
    cmds = cam.commands()
    assert cmds[0] == p.CMD_START_STREAM and cmds[-1] == p.CMD_STOP_STREAM
    assert set(cmds) <= {p.CMD_HEARTBEAT, p.CMD_START_STREAM, p.CMD_STOP_STREAM}
    assert p.CMD_HEARTBEAT in cmds


def test_bridge_script_captures_and_stops_cleanly(tmp_path):
    originals = [bytes((i * k + 1) & 255 for i in range(4096)) for k in (2, 9)]
    cam = _FakeCamera(originals, keep_streaming=True)
    out = tmp_path / "clip.h264"
    r = subprocess.run([sys.executable, str(TREE / "scripts" / "lewei_bridge.py"), "--cam-ip", "127.0.0.1",
                        "--port", str(cam.port), "--no-rtsp", "--out-h264", str(out), "--seconds", "1",
                        "--log-dir", str(tmp_path)], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": str(TREE / "src")})
    cam.th.join(5)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "receiving video (no RTSP)" in r.stdout
    data = out.read_bytes()
    assert len(data) >= 2 * 4096 and len(data) % 4096 == 0
    assert data[:4096] == originals[0] and data[4096:8192] == originals[1]
    cmds = cam.commands()
    assert cmds[0] == p.CMD_START_STREAM and cmds[-1] == p.CMD_STOP_STREAM
    assert set(cmds) <= {1, 2, 3}
    assert (tmp_path / "lewei_bridge.log").exists()
