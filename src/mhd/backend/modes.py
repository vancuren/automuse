"""FLIGHT_MODE gate for the backend (PLAN.md 2 and 6). Refuses to start unless the mode and the
drone address agree: twin talks only to loopback, real only to the X69 at 172.16.11.1.

MHD_CAMERA_ONLY=1 (real camera, NO flight): a twin-mode run whose video is a real camera republished
on loopback RTSP (scripts/lewei_bridge.py). It is accepted only with FLIGHT_MODE=twin, a loopback
DRONE_IP, an explicit loopback X69_LG_RTSP_URL and RTSP video (the other X69 video modes send UDP).
The backend then installs an RC adapter that has no socket at all (turbodrone_glue.CameraOnlyRcAdapter)
and telemetry carries the CAMERA_ONLY_FLAG safety flag.
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


# Telemetry safety flag in camera-only mode. The UI shows the REAL CAMERA - NO FLIGHT banner on it
# (turbodrone/frontend/src/lib/gates.ts CAMERA_ONLY_FLAG); unknown flags never block takeoff there.
CAMERA_ONLY_FLAG = "camera_only"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


class ModeError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendEnv:
    flight_mode: str          # "twin" | "real"
    drone_type: str           # always "x69_lg"
    drone_ip: str
    camera_only: bool = False  # MHD_CAMERA_ONLY=1: real camera over loopback RTSP, no RC packets at all


def camera_only_requested(environ=None) -> bool:
    """Fail-safe reading of MHD_CAMERA_ONLY for choosing the RC adapter: anything that is not
    clearly "off" counts as camera-only (validate_env refuses values it does not understand)."""
    env = os.environ if environ is None else environ
    return (env.get("MHD_CAMERA_ONLY") or "").strip().lower() not in _FALSE


def _camera_only(env) -> bool:
    raw = (env.get("MHD_CAMERA_ONLY") or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ModeError(f"MHD_CAMERA_ONLY={env.get('MHD_CAMERA_ONLY')!r} must be 1 or 0. Refusing to start.")


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
    camera_only = _camera_only(env)
    if camera_only and mode != "twin":
        raise ModeError("MHD_CAMERA_ONLY=1 (real camera, no flight) runs only with FLIGHT_MODE=twin; "
                        f"refusing FLIGHT_MODE={mode}.")
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
    if camera_only:
        _check_camera_only(env)
    return BackendEnv(flight_mode=mode, drone_type=drone_type, drone_ip=ip, camera_only=camera_only)


def _check_camera_only(env) -> None:
    """MHD_CAMERA_ONLY=1, after the twin checks (FLIGHT_MODE=twin, loopback DRONE_IP, no plugins)."""
    # The camera must come in on loopback too, and named explicitly (an unset URL would mean the
    # adapter's own default).
    url = (env.get("X69_LG_RTSP_URL") or "").strip()
    if not url or not _url_is_loopback(url):
        raise ModeError("MHD_CAMERA_ONLY=1 needs a loopback X69_LG_RTSP_URL "
                        f"(e.g. rtsp://127.0.0.1:8554/live/ch00_1), got {url!r}.")
    # The X69's other video modes (h265, jpeg) send UDP start commands to DRONE_IP: RTSP (TCP) only.
    video = (env.get("X69_LG_VIDEO_MODE") or "rtsp").strip().lower()
    if video != "rtsp":
        raise ModeError(f"MHD_CAMERA_ONLY=1 needs X69_LG_VIDEO_MODE=rtsp, got {video!r}: "
                        "the other video modes send UDP to the drone address.")
    # ... and RTSP interleaved over TCP (RTP over UDP would send RTCP datagrams).
    opts = (env.get("X69_LG_RTSP_FFMPEG_OPTIONS") or DEFAULT_RTSP_FFMPEG_OPTIONS).replace(" ", "").lower()
    if "rtsp_transport;tcp" not in opts:
        raise ModeError("MHD_CAMERA_ONLY=1 needs RTSP over TCP: X69_LG_RTSP_FFMPEG_OPTIONS must contain "
                        "rtsp_transport;tcp.")


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
