"""mediamtx is RTSP only, TCP only, loopback only: check the sockets it actually opens."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from mhd.twin import media

pytestmark = pytest.mark.slow


def test_generated_config_disables_everything_else():
    cfg = media.mediamtx_config(8254)
    for key in ("api", "metrics", "pprof", "playback", "rtmp", "hls", "webrtc", "srt", "moq"):
        assert f"\n{key}: false" in cfg, key
    assert "rtspAddress: 127.0.0.1:8254" in cfg and "rtspTransports: [tcp]" in cfg


def test_mediamtx_listens_only_on_loopback_rtsp(env_paths, tmp_path):
    from procs import wait_ports_free
    wait_ports_free([8254])
    p = media.start_mediamtx(8254, tmp_path)
    try:
        time.sleep(1.0)
        out = subprocess.run(["lsof", "-nP", "-a", "-p", str(p.pid), "-i"], capture_output=True, text=True).stdout
        lines = [l for l in out.splitlines()[1:] if l.strip()]
        assert lines, out
        assert all("TCP 127.0.0.1:8254 (LISTEN)" in l for l in lines), out
    finally:
        media.stop_process(p)
