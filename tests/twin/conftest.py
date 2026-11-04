"""Shared fixtures for the twin tests.

Ports (twin workstream only): twin HTTP 8266, RTSP 8254, UDP RC 23520-23529, test backend 8200-8209.
Slow tests (real processes, Chrome, HEVC) are marked `slow`; run them with `-m slow` or plain pytest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
UPSTREAM_BACKEND = TREE / "turbodrone" / "backend"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: starts real processes (mediamtx, ffmpeg, Chrome, backend)")


def upstream_importable() -> bool:
    return (UPSTREAM_BACKEND / "models" / "x69_lg_rc.py").exists()


@pytest.fixture(scope="session")
def upstream():
    """Make turbodrone/backend importable (appended, so its generic module names never shadow ours)."""
    if not upstream_importable():
        pytest.skip("turbodrone/backend/models is missing in this checkout (git-ignored by 'models/'); "
                    "copy it from the main checkout, see docs/ws/twin.md")
    path = str(UPSTREAM_BACKEND)
    if path not in sys.path:
        sys.path.append(path)
    from models.x69_lg_rc import X69LgRcModel  # noqa: E402
    from protocols.x69_lg_rc_protocol_adapter import X69LgRcProtocolAdapter  # noqa: E402
    return X69LgRcModel, X69LgRcProtocolAdapter


@pytest.fixture
def offline_adapter(upstream):
    """The real adapter class without its socket or keepalive thread (same trick as upstream tests)."""
    _, Adapter = upstream
    adapter = Adapter.__new__(Adapter)
    adapter.speed_scale_by_index = {0: 0.55, 1: 0.75, 2: 1.0}
    return adapter


@pytest.fixture(scope="session")
def env_paths():
    home = os.environ.get("MHD_HOME")
    if not home:
        pytest.skip("source env.sh first (MHD_HOME not set)")
    return {
        "home": Path(home),
        "tree": TREE,
        "tools": Path(os.environ.get("MHD_TOOLS", Path(home) / "tools")),
        "logs": Path(os.environ.get("MHD_LOGS", TREE / "logs")),
    }
