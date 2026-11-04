#!/usr/bin/env python3
"""Camera-only mode: pull the drone's live video from its Lewei camera board and republish it as RTSP
on loopback, so the backend, perception and UI see the real camera. Nothing is ever sent to the flight
controller: the only bytes sent to the drone are start-video, heartbeat and stop-video.

    $MHD_PY scripts/lewei_bridge.py --rtsp-out rtsp://127.0.0.1:8554/live/ch00_1 --log-dir logs/x
    $MHD_PY scripts/lewei_bridge.py --no-rtsp --out-h264 clip.h264 --seconds 10      # capture test

Runs until Ctrl-C or SIGTERM, and reconnects when the camera drops. It prints
"[lewei_bridge] publishing <url>" once frames reach ffmpeg.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE / "src"))
from mhd.lewei.protocol import CAM_IP, STREAM_PORT, LeweiStream  # noqa: E402

DEFAULT_RTSP = "rtsp://127.0.0.1:8554/live/ch00_1"


class _Stop(Exception):
    pass


def ffmpeg_cmd(url: str) -> list[str]:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("[lewei_bridge] ffmpeg not found on PATH")
    # Copy the camera's H.264 as is; stamp each frame with the wall clock on arrival, so the backend's
    # capture-time estimate and staleness checks see real time.
    return [ff, "-hide_banner", "-loglevel", "warning", "-nostats", "-nostdin",
            "-fflags", "nobuffer", "-probesize", "500000", "-analyzeduration", "500000",
            "-use_wallclock_as_timestamps", "1", "-f", "h264", "-i", "pipe:0",
            "-an", "-c:v", "copy",
            "-flush_packets", "1", "-muxdelay", "0", "-max_delay", "0",
            "-f", "rtsp", "-rtsp_transport", "tcp", url]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-ip", default=CAM_IP)
    ap.add_argument("--port", type=int, default=STREAM_PORT)
    ap.add_argument("--rtsp-out", default=DEFAULT_RTSP, help="RTSP URL to publish to (mediamtx on loopback)")
    ap.add_argument("--no-rtsp", action="store_true", help="do not start ffmpeg (capture or test only)")
    ap.add_argument("--low-def", action="store_true", help="ask the camera for its low-definition stream")
    ap.add_argument("--out-h264", default=None, help="also write the raw H.264 to this file")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = run until stopped)")
    ap.add_argument("--log-dir", default=None, help="also append the log to <dir>/lewei_bridge.log")
    a = ap.parse_args()

    logf = None
    if a.log_dir:
        Path(a.log_dir).mkdir(parents=True, exist_ok=True)
        logf = open(Path(a.log_dir) / "lewei_bridge.log", "a", buffering=1)

    def log(msg: str) -> None:
        line = f"[lewei_bridge] {time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")

    def on_signal(signum, _frame):
        raise _Stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    t_start = time.monotonic()
    ff: subprocess.Popen | None = None
    raw = open(a.out_h264, "wb") if a.out_h264 else None
    announced = False
    backoff = 1.0
    frames = nbytes = 0
    t_stats = time.monotonic()
    log(f"camera {a.cam_ip}:{a.port} ({'low' if a.low_def else 'high'} definition) -> "
        f"{'no RTSP' if a.no_rtsp else a.rtsp_out}")
    try:
        while True:
            if a.seconds and time.monotonic() - t_start >= a.seconds:
                break
            try:
                with LeweiStream(a.cam_ip, a.port, hd=not a.low_def) as cam:
                    for f in cam.frames():
                        if not a.no_rtsp and (ff is None or ff.poll() is not None):
                            if ff is not None:
                                log(f"ffmpeg exited with {ff.returncode}; restarting it")
                            ff = subprocess.Popen(ffmpeg_cmd(a.rtsp_out), stdin=subprocess.PIPE)
                        if ff is not None:
                            try:
                                ff.stdin.write(f.data)
                                ff.stdin.flush()
                            except BrokenPipeError:
                                log("ffmpeg's input closed; it will be restarted on the next frame")
                                ff.wait(timeout=5)
                                continue
                        if raw:
                            raw.write(f.data)
                        frames += 1
                        nbytes += len(f.data)
                        backoff = 1.0
                        if not announced:
                            log(f"publishing {a.rtsp_out}" if not a.no_rtsp else "receiving video (no RTSP)")
                            announced = True
                        now = time.monotonic()
                        if now - t_stats >= 10.0:
                            dt = now - t_stats
                            log(f"{frames / dt:.1f} fps, {nbytes * 8 / dt / 1e6:.2f} Mbit/s, scrambling type {f.stream_type}")
                            frames = nbytes = 0
                            t_stats = now
                        if a.seconds and now - t_start >= a.seconds:
                            break
            except (OSError, ValueError) as e:
                log(f"camera link: {type(e).__name__}: {e}; reconnecting in {backoff:.0f} s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
    except _Stop:
        log("stopping")
    finally:
        if ff is not None and ff.poll() is None:
            try:
                ff.stdin.close()
            except OSError:
                pass
            try:
                ff.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ff.terminate()
        if raw:
            raw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
