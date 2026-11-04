"""Frozen message contracts shared by every process. See docs/conventions.md.

Change only through the integrator. Workstreams that need something extra subclass locally.
"""
from .messages import *  # noqa: F401,F403
from .framing import encode_frame, decode_frame, now_ms  # noqa: F401
