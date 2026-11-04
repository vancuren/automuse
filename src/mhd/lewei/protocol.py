"""Minimal client for the Lewei Wi-Fi camera board ("lewei_cam") on our X69 unit.

It can send exactly three commands, all on the video port: heartbeat (1), start-stream (2) and
stop-stream (3). Everything else the board understands (Wi-Fi settings, SD card, the flight
controller's baud rate) is refused here on purpose: this module reads video and never changes
the drone. It never talks to the flight controller.

Protocol facts (the 46-byte "lewei_cmd" header, the frame layout, and how a few bytes of each
frame are scrambled) come from pylwdrone 0.5.0, Copyright 2020 meekworth, Apache-2.0
(https://github.com/meekworth/pylwdrone, licenses/pylwdrone-LICENSE). The code here is our own.
"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Iterator

from mhd.lewei.tables import TABLE1, TABLE2, TABLE3

CAM_IP = "192.168.0.1"
STREAM_PORT = 7060
MAGIC = b"lewei_cmd\x00"
HDR_LEN = len(MAGIC) + 9 * 4            # 46 bytes: magic + command + 8 argument ints
CMD_HEARTBEAT = 1
CMD_START_STREAM = 2
CMD_STOP_STREAM = 3
RET_STREAM = 257                         # a video frame from the camera
ALLOWED_COMMANDS = frozenset({CMD_HEARTBEAT, CMD_START_STREAM, CMD_STOP_STREAM})
FRAME_OFF = 32                           # frame bytes start 32 bytes into a video packet body
MAX_BODY = 8 * 1024 * 1024               # sanity bound on one packet
STREAM_PLAIN, STREAM_ONE, STREAM_NEW = 0, 1, 129   # scrambling types


def _first_index(table: tuple[int, ...]) -> dict[int, int]:
    """code -> the first byte value that maps to it (same rule as a linear search)."""
    inv: dict[int, int] = {}
    for i, code in enumerate(table):
        inv.setdefault(code, i)
    return inv


_INV1, _INV2, _INV3 = _first_index(TABLE1), _first_index(TABLE2), _first_index(TABLE3)


def encode_command(cmd: int, arg1: int = 0) -> bytes:
    """Bytes for one command. Only heartbeat, start-stream and stop-stream are allowed."""
    if cmd not in ALLOWED_COMMANDS:
        raise ValueError(f"refusing Lewei command {cmd}: only heartbeat, start-stream and stop-stream are allowed")
    return MAGIC + struct.pack("<9I", cmd, arg1 & 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class Header:
    cmd: int
    args: tuple[int, ...]                # the 8 ints after the command

    @property
    def body_size(self) -> int:
        return self.args[2]

    @property
    def stream_type(self) -> int:
        return self.args[3]

    @property
    def dec1(self) -> int:
        return self.args[4]

    @property
    def dec2(self) -> int:
        return self.args[5]


def parse_header(buf: bytes) -> Header:
    if len(buf) != HDR_LEN:
        raise ValueError(f"header is {len(buf)} bytes, expected {HDR_LEN}")
    if buf[:len(MAGIC)] != MAGIC:
        raise ValueError("bad magic: not a lewei_cmd header")
    ints = struct.unpack_from("<9I", buf, len(MAGIC))
    return Header(ints[0], tuple(ints[1:]))


def _fix_byte(count: int, size: int) -> int:
    """Index of the inverted byte for scrambling type 1 (same arithmetic as pylwdrone)."""
    size &= 0xFFFFFFFF
    if size & 1 == 0:
        v2 = (size + 1 + (size ^ count)) ^ size
    else:
        v2 = ((size ^ count) + size) ^ size
    v1 = int(v2 / size) if size else 0
    return (v2 - v1 * size) & 0xFFFFFFFF


def unmunge(frame: bytearray, stream_type: int, dec1: int, dec2: int, count: int) -> None:
    """Undo the camera's per-frame scrambling in place."""
    size = len(frame)
    if stream_type == STREAM_NEW:
        idx = size >> 1
        if idx + 2 >= size:
            return
        i = _INV1.get(dec1 & 0xFFFF)
        if i is not None:
            frame[idx] = i
        j = _INV2.get((dec1 >> 16) & 0xFFFF)
        if j is not None:
            frame[idx + 1] = frame[idx] ^ j
        k = _INV3.get(dec2 & 0xFFFF)
        if k is not None:
            frame[idx + 2] = frame[idx] ^ frame[idx + 1] ^ k
    elif stream_type == STREAM_ONE:
        idx = _fix_byte(count, size)
        if 0 <= idx < size:
            frame[idx] = ~frame[idx] & 0xFF


@dataclass(frozen=True)
class VideoFrame:
    data: bytes                          # H.264 Annex B access unit, unscrambled
    count: int
    flag: int
    stream_type: int
    t_recv: float                        # time.time() when the frame arrived


def parse_video(hdr: Header, body: bytes, t_recv: float) -> VideoFrame:
    if len(body) < FRAME_OFF:
        raise ValueError("video packet shorter than its frame header")
    flag, size, count, _gphoto = struct.unpack_from("<LLQL", body)
    frame = bytearray(body[FRAME_OFF:FRAME_OFF + size])
    if len(frame) != size:
        raise ValueError(f"incomplete video frame: {len(frame)} of {size} bytes")
    unmunge(frame, hdr.stream_type, hdr.dec1, hdr.dec2, count)
    return VideoFrame(bytes(frame), count, flag, hdr.stream_type, t_recv)


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("camera closed the video connection")
        buf += chunk
    return bytes(buf)


class LeweiStream:
    """One live-video session: connect, start, read frames (heartbeat every second), stop."""

    def __init__(self, ip: str = CAM_IP, port: int = STREAM_PORT, hd: bool = True,
                 connect_timeout: float = 5.0, read_timeout: float = 5.0, heartbeat_s: float = 1.0):
        self.addr = (ip, port)
        self.hd = hd
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.heartbeat_s = heartbeat_s
        self._sock: socket.socket | None = None

    def __enter__(self) -> "LeweiStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _send(self, cmd: int, arg1: int = 0) -> None:
        assert self._sock is not None
        self._sock.sendall(encode_command(cmd, arg1))

    def frames(self) -> Iterator[VideoFrame]:
        self._sock = socket.create_connection(self.addr, timeout=self.connect_timeout)
        self._sock.settimeout(self.read_timeout)
        self._send(CMD_START_STREAM, int(self.hd))
        last_hb = time.monotonic()
        while True:
            hdr = parse_header(_recvall(self._sock, HDR_LEN))
            if hdr.body_size > MAX_BODY:
                raise ValueError(f"packet body of {hdr.body_size} bytes is not plausible")
            body = _recvall(self._sock, hdr.body_size)
            now = time.monotonic()
            if now - last_hb >= self.heartbeat_s:
                self._send(CMD_HEARTBEAT)
                last_hb = now
            if hdr.cmd == RET_STREAM:
                yield parse_video(hdr, body, time.time())
            # heartbeat replies and anything else are ignored

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is None:
            return
        try:
            sock.sendall(encode_command(CMD_STOP_STREAM))
        except OSError:
            pass
        sock.close()
