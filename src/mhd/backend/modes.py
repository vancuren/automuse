"""FLIGHT_MODE gate for the backend (PLAN.md 2 and 6). Refuses to start unless the mode and the
drone address agree: twin talks only to loopback, real only to the X69 at 172.16.11.1.
"""
from __future__ import annotations

import ipaddress
import math
import os

from dataclasses import dataclass

REAL_DRONE_IP = "172.16.11.1"
# RTSP through OpenCV/FFmpeg 7: TCP, low delay, and a 2 s socket I/O timeout (microseconds) so a
# frozen stream fails fast instead of blocking for tens of seconds.
DEFAULT_RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|timeout;2000000"


class ModeError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendEnv:
    flight_mode: str          # "twin" | "real"
    drone_type: str           # always "x69_lg"
    drone_ip: str


def apply_env_defaults() -> None:
    """Call before turbodrone opens any RTSP stream."""
    os.environ.setdefault("X69_LG_RTSP_FFMPEG_OPTIONS", DEFAULT_RTSP_FFMPEG_OPTIONS)
    # One HEVC decoder thread: OpenCV's FFmpeg backend otherwise frame-threads the decoder
    # (16 threads here), which the twin workstream measured as ~0.7 s of added latency. A pts-based
    # capture estimate cannot see that constant delay, so it would silently add to theta.
    os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
    os.environ.setdefault("PLUGINS_ENABLED", "false")


def validate_env(environ=None) -> BackendEnv:
    env = os.environ if environ is None else environ
    mode = (env.get("FLIGHT_MODE") or "").strip().lower()
    if mode not in ("twin", "real"):
        raise ModeError("FLIGHT_MODE must be set to 'twin' or 'real' (no default). Refusing to start.")
    drone_type = (env.get("DRONE_TYPE") or "x69_lg").strip().lower()
    if drone_type != "x69_lg":
        raise ModeError(f"DRONE_TYPE={drone_type!r}: the safety governor supports only x69_lg. Refusing to start.")
    ip = (env.get("DRONE_IP") or "").strip()
    if mode == "twin":
        if not ip:
            raise ModeError("FLIGHT_MODE=twin needs DRONE_IP=127.0.0.1 (loopback only).")
        try:
            loop = ipaddress.ip_address(ip).is_loopback
        except ValueError:
            loop = False
        if not loop:
            raise ModeError(f"FLIGHT_MODE=twin refuses non-loopback DRONE_IP={ip!r}.")
        url = (env.get("X69_LG_RTSP_URL") or "").strip()
        if url and not _url_is_loopback(url):
            raise ModeError(f"FLIGHT_MODE=twin refuses a non-loopback X69_LG_RTSP_URL={url!r}.")
    else:
        if ip != REAL_DRONE_IP:
            raise ModeError(f"FLIGHT_MODE=real requires DRONE_IP={REAL_DRONE_IP}, got {ip!r}.")
        # The video must come from the drone too: flying the X69 while the video-age checks and the
        # autopilot watch the twin's stream would be blind flight (review finding).
        url = (env.get("X69_LG_RTSP_URL") or "").strip()
        if url and _url_host(url) != REAL_DRONE_IP:
            raise ModeError(f"FLIGHT_MODE=real refuses X69_LG_RTSP_URL={url!r}: its host must be {REAL_DRONE_IP}.")
    rate = (env.get("CONTROL_RATE") or "").strip()
    if rate:
        try:
            r = float(rate)
        except ValueError:
            r = math.nan
        if not (math.isfinite(r) and 10.0 <= r <= 50.0):
            raise ModeError(f"CONTROL_RATE={rate!r} must be a number within 10..50 Hz.")
        if mode == "real" and r != 25.0:
            raise ModeError(f"FLIGHT_MODE=real requires the X69's 25 Hz RC rate, got CONTROL_RATE={rate!r}.")
    if (env.get("PLUGINS_ENABLED") or "false").strip().lower() in ("1", "true", "yes", "on"):
        raise ModeError("PLUGINS_ENABLED must stay false: plugins would bypass the governor.")
    return BackendEnv(flight_mode=mode, drone_type=drone_type, drone_ip=ip)


def _url_host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _url_is_loopback(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
