"""Shared helpers for the UI tests (ports, paths, process control, the mock client).

A uniquely named module, so tests never import `conftest` by name: every workstream has a
tests/<ws>/conftest.py, and a plain `from conftest import` could pick up the wrong one.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

TREE = Path(os.environ.get("MHD_TREE") or Path(__file__).resolve().parents[2])
HOME = Path(os.environ.get("MHD_HOME") or TREE)
FRONTEND = TREE / "turbodrone" / "frontend"
MOCK_SCRIPT = FRONTEND / "mock" / "mock_backend.py"
VITE_BIN = FRONTEND / "node_modules" / ".bin" / "vite"

VITE_PORT = 5273
MOCK_PORT = 8601
UI_URL = f"http://127.0.0.1:{VITE_PORT}/"
MOCK_URL = f"http://127.0.0.1:{MOCK_PORT}"

LOG_DIR = TREE / "logs" / "ui_tests"
MOCK_LOG = LOG_DIR / "mock_messages.jsonl"
SCREENS = TREE / "media" / "ui_screens"
SHARED = HOME / "media" / "shared" / "ui"
CHROME_PROFILE = HOME / ".cache" / "chrome-ui"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_http(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code < 500:
                return
        except Exception as e:  # not up yet
            last = e
        time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up in {timeout_s}s: {last}")


def stop(proc: subprocess.Popen) -> None:
    """Terminate a process group started with start_new_session=True."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


class Mock:
    """Client for the mock backend's test controls and its JSONL message log."""

    def __init__(self, url: str, log_path: Path):
        self.url = url
        self.log_path = log_path

    def control(self, **kw: Any) -> dict:
        return httpx.post(self.url + "/mock/control", json=kw, timeout=5).json()

    def reset(self, **kw: Any) -> None:
        httpx.post(self.url + "/mock/reset", json=kw, timeout=5).raise_for_status()

    def candidates(self) -> dict:
        return httpx.post(self.url + "/mock/candidates", timeout=5).json()

    def now_ms(self) -> float:
        return httpx.get(self.url + "/mock/log", params={"since_ms": 1e18}, timeout=5).json()["now_ms"]

    def entries(self, channel: Optional[str] = None, since_ms: float = 0.0) -> list[dict]:
        """Read the JSONL log file (what the tests are specified to check)."""
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if (channel is None or e["channel"] == channel) and e["t_ms"] >= since_ms:
                out.append(e)
        return out

    def of_type(self, kind: str, channel: Optional[str] = None, since_ms: float = 0.0) -> list[dict]:
        return [e for e in self.entries(channel, since_ms) if isinstance(e["msg"], dict) and e["msg"].get("type") == kind]

    def commands(self, command: str, channel: Optional[str] = None, since_ms: float = 0.0) -> list[dict]:
        return [e for e in self.of_type("cmd", channel, since_ms) if e["msg"].get("command") == command]

    def wait_for(self, find: Callable[[], list], timeout_s: float = 3.0, min_count: int = 1) -> list:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            got = find()
            if len(got) >= min_count:
                return got
            time.sleep(0.05)
        return find()
