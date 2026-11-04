"""Decode X69 RC datagrams, byte for byte as turbodrone builds them.

Source of truth: turbodrone/backend/protocols/x69_lg_rc_protocol_adapter.py (upstream 2c09cf8).

Control datagram, 28 bytes, sent at 25 Hz to UDP 23458:
    ca 47 d5 00            wrapper magic
    14 00 00 00            payload length, uint32 little-endian (= 20)
    payload (20 bytes):
      [0]  0x66            start marker
      [1]  0x14            length marker
      [2]  roll            0..255, center 128 (already scaled by the speed tier on the sender)
      [3]  pitch           0..255
      [4]  throttle        0..255
      [5]  yaw             0..255
      [6]  flags1          0x01 takeoff, 0x02 E-stop, 0x04 gyro calibrate, 0x08 flip, 0x80 land
      [7]  flags2          0x08 camera tilt up, 0x10 camera tilt down (held while pressed)
      [8..17]              zeros
      [18] checksum        XOR of bytes [2..17]
      [19] 0x99            end marker

Keepalive datagram, 8 bytes, sent once per second on its own thread:
    ca 47 d1 00 00 00 00 00

The keepalive carries no stick data. The twin ignores it for link-loss purposes, because the
upstream keepalive thread keeps running even if the RC thread dies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

CONTROL_MAGIC = b"\xca\x47\xd5\x00"
KEEPALIVE = b"\xca\x47\xd1\x00\x00\x00\x00\x00"
KEEPALIVE_MAGIC = KEEPALIVE[:4]
PAYLOAD_LEN = 20
DATAGRAM_LEN = 8 + PAYLOAD_LEN
START_MARKER = 0x66
LENGTH_MARKER = 0x14
END_MARKER = 0x99
CENTER = 128

# flags1 (payload byte 6)
FLAG_TAKEOFF = 0x01
FLAG_ESTOP = 0x02
FLAG_CALIBRATE = 0x04
FLAG_FLIP = 0x08
FLAG_LAND = 0x80
KNOWN_FLAGS1 = FLAG_TAKEOFF | FLAG_ESTOP | FLAG_CALIBRATE | FLAG_FLIP | FLAG_LAND
# flags2 (payload byte 7)
FLAG2_TILT_UP = 0x08
FLAG2_TILT_DOWN = 0x10
KNOWN_FLAGS2 = FLAG2_TILT_UP | FLAG2_TILT_DOWN


@dataclass(frozen=True)
class RcPacket:
    """One decoded 28-byte control datagram. Stick values are raw wire counts (0..255)."""
    roll: int
    pitch: int
    throttle: int
    yaw: int
    flags1: int
    flags2: int
    reserved_nonzero: bool = False     # bytes 8..17 were not all zero (upstream always sends zeros)

    @property
    def takeoff(self) -> bool:
        return bool(self.flags1 & FLAG_TAKEOFF)

    @property
    def estop(self) -> bool:
        return bool(self.flags1 & FLAG_ESTOP)

    @property
    def calibrate(self) -> bool:
        return bool(self.flags1 & FLAG_CALIBRATE)

    @property
    def flip(self) -> bool:
        return bool(self.flags1 & FLAG_FLIP)

    @property
    def land(self) -> bool:
        return bool(self.flags1 & FLAG_LAND)

    @property
    def tilt_up(self) -> bool:
        return bool(self.flags2 & FLAG2_TILT_UP)

    @property
    def tilt_down(self) -> bool:
        return bool(self.flags2 & FLAG2_TILT_DOWN)

    @property
    def unknown_flags1(self) -> int:
        return self.flags1 & ~KNOWN_FLAGS1 & 0xFF

    @property
    def unknown_flags2(self) -> int:
        return self.flags2 & ~KNOWN_FLAGS2 & 0xFF

    def sticks(self) -> dict[str, int]:
        return {"roll": self.roll, "pitch": self.pitch, "throttle": self.throttle, "yaw": self.yaw}

    def flag_names(self) -> list[str]:
        names = []
        for name, on in (("takeoff", self.takeoff), ("estop", self.estop), ("calibrate", self.calibrate),
                         ("flip", self.flip), ("land", self.land), ("tilt_up", self.tilt_up),
                         ("tilt_down", self.tilt_down)):
            if on:
                names.append(name)
        return names


@dataclass(frozen=True)
class Keepalive:
    """The 8-byte d1 keepalive. Carries nothing; never counts as an RC packet."""


@dataclass(frozen=True)
class DecodeError:
    reason: str            # bad_magic | bad_length | bad_marker | bad_checksum | empty
    raw: bytes = field(default=b"", repr=False)


Decoded = Union[RcPacket, Keepalive, DecodeError]


def xor_checksum(data: bytes | bytearray) -> int:
    value = 0
    for b in data:
        value ^= b
    return value & 0xFF


def decode_payload(payload: bytes) -> Decoded:
    """Decode the 20-byte inner payload (without the 8-byte wrapper)."""
    if len(payload) != PAYLOAD_LEN:
        return DecodeError("bad_length", bytes(payload))
    if payload[0] != START_MARKER or payload[1] != LENGTH_MARKER or payload[19] != END_MARKER:
        return DecodeError("bad_marker", bytes(payload))
    if xor_checksum(payload[2:18]) != payload[18]:
        return DecodeError("bad_checksum", bytes(payload))
    return RcPacket(
        roll=payload[2], pitch=payload[3], throttle=payload[4], yaw=payload[5],
        flags1=payload[6], flags2=payload[7],
        reserved_nonzero=any(payload[8:18]),
    )


def decode_datagram(data: bytes) -> Decoded:
    """Decode one UDP datagram as received on the X69's RC port."""
    if not data:
        return DecodeError("empty", b"")
    if data == KEEPALIVE:
        return Keepalive()
    if data[:4] == KEEPALIVE_MAGIC:
        # Same magic as the keepalive but different body: treat as a keepalive variant.
        return Keepalive()
    if data[:4] != CONTROL_MAGIC:
        return DecodeError("bad_magic", bytes(data))
    if len(data) < 8:
        return DecodeError("bad_length", bytes(data))
    declared = int.from_bytes(data[4:8], "little")
    if declared != PAYLOAD_LEN or len(data) != 8 + declared:
        return DecodeError("bad_length", bytes(data))
    return decode_payload(data[8:])


def encode_control(roll: int = CENTER, pitch: int = CENTER, throttle: int = CENTER, yaw: int = CENTER,
                   flags1: int = 0, flags2: int = 0) -> bytes:
    """Build a 28-byte control datagram the same way the upstream adapter does (for tests and tools)."""
    payload = bytearray(PAYLOAD_LEN)
    payload[0] = START_MARKER
    payload[1] = LENGTH_MARKER
    payload[2] = max(0, min(255, int(roll)))
    payload[3] = max(0, min(255, int(pitch)))
    payload[4] = max(0, min(255, int(throttle)))
    payload[5] = max(0, min(255, int(yaw)))
    payload[6] = flags1 & 0xFF
    payload[7] = flags2 & 0xFF
    payload[18] = xor_checksum(payload[2:18])
    payload[19] = END_MARKER
    return CONTROL_MAGIC + PAYLOAD_LEN.to_bytes(4, "little") + bytes(payload)


def count_to_unit(count: int) -> float:
    """Wire count (0..255, center 128) to a normalized stick value in [-1, 1]."""
    offset = int(count) - CENTER
    return offset / 127.0 if offset >= 0 else offset / 128.0
