"""Binary framing for JPEG frames over WebSocket: 4-byte big-endian header length + JSON header + JPEG."""
from __future__ import annotations

import json
import struct
import time

from .messages import FrameMeta


def now_ms() -> float:
    """The one shared clock (time.monotonic, milliseconds)."""
    return time.monotonic() * 1000.0


def encode_frame(meta: FrameMeta, jpeg: bytes) -> bytes:
    header = meta.model_dump_json().encode("utf-8")
    return struct.pack(">I", len(header)) + header + jpeg


def decode_frame(blob: bytes) -> tuple[FrameMeta, bytes]:
    (n,) = struct.unpack(">I", blob[:4])
    meta = FrameMeta.model_validate(json.loads(blob[4:4 + n].decode("utf-8")))
    return meta, blob[4 + n:]
