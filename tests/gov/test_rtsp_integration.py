"""RTSP integration: mediamtx (127.0.0.1:8105, TCP) + ffmpeg HEVC test stream -> TimedX69RtspAdapter.

Checks the capture-time estimate (pts + 10 s min of receive - pts) on a real FFmpeg/OpenCV RTSP
path, that the 2 s timeouts keep a frozen stream from blocking, and that a stall followed by a
catch-up burst (publisher paused with SIGSTOP, ffmpeg -re then bursts) is seen as OLD video by the
governor (video lag above 2 s means land), not as fresh frames.

Known limit (by design, per the 10 s window): if a stream's timeline shifts permanently, the
estimate re-adapts after 10 s and the extra latency is no longer visible. See docs/ws/gov.md.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from mhd.contracts import now_ms

ROOT = Path(__file__).resolve().parents[2]
URL = "rtsp://127.0.0.1:8105/live/ch00_1"


def _mediamtx() -> Path:
    home = Path(os.environ.get("MHD_TOOLS") or ROOT / "tools")
    return home / "mediamtx" / "mediamtx"


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def rtsp_stream(tmp_path):
    mtx = _mediamtx()
    ff = shutil.which("ffmpeg")
    if not mtx.exists() or not ff:
        pytest.skip("mediamtx or ffmpeg not available")
    if _port_open(8105):
        pytest.skip("port 8105 busy")
    procs = []
    log = open(tmp_path / "rtsp.log", "w")
    try:
        procs.append(subprocess.Popen([str(mtx), str(Path(__file__).with_name("mediamtx_gov.yml"))],
                                      stdout=log, stderr=log, cwd=tmp_path))   # any generated file stays in tmp
        for _ in range(50):
            if _port_open(8105):
                break
            time.sleep(0.1)
        pub = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                                "testsrc2=size=1280x720:rate=15", "-c:v", "hevc_videotoolbox", "-b:v", "3M",
                                "-g", "15", "-bf", "0", "-realtime", "1", "-f", "rtsp", "-rtsp_transport", "tcp",
                                URL], stdout=log, stderr=log)
        procs.append(pub)
        time.sleep(2.0)
        if pub.poll() is not None:
            pytest.skip("ffmpeg publisher failed (see rtsp.log)")
        yield pub
    finally:
        for p in reversed(procs):
            try:
                p.send_signal(signal.SIGCONT)
                p.terminate()
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                p.kill()
        log.close()


def test_capture_estimate_timeouts_and_stall_burst(rtsp_stream, tmp_path, record_property):
    from mhd.backend.capture import TimedX69RtspAdapter
    from mhd.governor import Governor
    from mhd.governor.config import load_autopilot_config, load_governor_config

    gov = Governor(load_governor_config(), load_autopilot_config(), "twin", log_dir=tmp_path, tick_log=False)
    ad = TimedX69RtspAdapter(drone_ip="127.0.0.1", video_port=8105, rtsp_url=URL, flight_mode="real")
    ad.start()
    try:
        frames = []
        end = time.monotonic() + 4.0
        while time.monotonic() < end:
            f = ad.get_frame(timeout=0.5)
            if f is not None:
                frames.append(f)
                gov.note_frame(f.capture_ms, f.receive_ms)
        assert len(frames) >= 30, len(frames)
        assert all(getattr(f, "capture_source", None) == "pts" for f in frames[5:])
        assert (frames[-1].width, frames[-1].height) == (1280, 720)
        excess = [f.receive_ms - f.capture_ms for f in frames[5:]]
        assert min(excess) >= -1.0 and max(excess) < 300.0
        record_property("steady_receive_minus_capture_ms_max", round(max(excess), 1))
        assert gov._frame_lag(now_ms()) < 300.0

        # Freeze the publisher for 3 s, then let it burst to catch up.
        rtsp_stream.send_signal(signal.SIGSTOP)
        t_stop = time.monotonic()
        gap_frames = 0
        lag_peak = 0.0
        while time.monotonic() - t_stop < 3.0:
            f = ad.get_frame(timeout=0.2)
            if f is not None:
                gap_frames += 1
                gov.note_frame(f.capture_ms, f.receive_ms)
            lag_peak = max(lag_peak, gov._frame_lag(now_ms()))
        assert lag_peak > 2000.0                     # governor would land (video lag > 2 s)
        rtsp_stream.send_signal(signal.SIGCONT)
        burst = []
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            f = ad.get_frame(timeout=0.5)
            if f is not None:
                burst.append((f.receive_ms - f.capture_ms, f))
                gov.note_frame(f.capture_ms, f.receive_ms)
        assert burst, "stream did not resume within 6 s"
        first_age = burst[0][0]
        last_age = burst[-1][0]
        record_property("stall_frames_during_freeze", gap_frames)
        record_property("first_frame_after_stall_age_ms", round(first_age, 1))
        record_property("age_5s_after_resume_ms", round(last_age, 1))
        print(f"\nrtsp: steady (receive-capture) max {max(excess):.1f} ms; frames during 3 s freeze "
              f"{gap_frames}; first frame after resume looks {first_age:.0f} ms old; 5 s later "
              f"{last_age:.0f} ms; reconnects {ad.reconnect_count}; sources {ad.capture_source_counts}")
        assert first_age > 1500.0                    # the late frames are recognized as old video
        # This ffmpeg -re source never catches up after SIGCONT (its timeline stays ~3 s behind), so
        # every later frame really is ~3 s old: the estimate must not forget that inside its 10 s
        # window. (A real camera's clock never pauses; its TCP backlog drains in a burst instead.)
        assert all(age > 1500.0 for age, _ in burst)
    finally:
        ad.stop()
        gov.close()
