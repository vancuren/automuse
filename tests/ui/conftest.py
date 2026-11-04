"""Fixtures for the operator UI tests.

Browser tests start, once per session:
  - the mock backend (turbodrone/frontend/mock/mock_backend.py) on 127.0.0.1:8601, scenario "manual";
  - vite on 127.0.0.1:5273 with VITE_BACKEND_URL pointing at the mock;
  - Google Chrome through Playwright (channel="chrome", profile $MHD_HOME/.cache/chrome-ui).
Everything is stopped at the end of the session. Run: $MHD_PY -m pytest tests/ui -v
Set UI_HEADED=1 to watch the browser.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest

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


@pytest.fixture(scope="session")
def mock_backend():
    if not port_free(MOCK_PORT):
        pytest.fail(f"port {MOCK_PORT} is busy: stop whatever runs there")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / "mock_backend.out", "w")
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SCRIPT), "--port", str(MOCK_PORT), "--scenario", "manual", "--log", str(MOCK_LOG)],
        stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        wait_http(MOCK_URL + "/state", 20)
        yield Mock(MOCK_URL, MOCK_LOG)
    finally:
        stop(proc)
        out.close()


@pytest.fixture(scope="session")
def vite_server(mock_backend):
    if not VITE_BIN.exists():
        pytest.fail("node_modules missing: run `npm ci` in turbodrone/frontend")
    if not port_free(VITE_PORT):
        pytest.fail(f"port {VITE_PORT} is busy: stop the other vite first")
    env = dict(os.environ, VITE_BACKEND_URL=MOCK_URL, VITE_TWIN_CHASE_URL=MOCK_URL + "/chase")
    out = open(LOG_DIR / "vite.out", "w")
    proc = subprocess.Popen(
        [str(VITE_BIN), "--port", str(VITE_PORT), "--strictPort"],
        cwd=FRONTEND, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        wait_http(UI_URL, 30)
        yield UI_URL
    finally:
        stop(proc)
        out.close()


@pytest.fixture(scope="session")
def browser_ctx(vite_server):
    from playwright.sync_api import sync_playwright

    CHROME_PROFILE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(CHROME_PROFILE),
            channel="chrome",  # the installed Google Chrome; never a downloaded browser
            headless=os.environ.get("UI_HEADED") != "1",
            viewport={"width": 1920, "height": 1080},
            args=[f"--disk-cache-dir={CHROME_PROFILE / 'disk-cache'}"],
            # Keep Chrome's normal background throttling: Playwright disables it by default.
            ignore_default_args=["--disable-background-timer-throttling",
                                 "--disable-renderer-backgrounding",
                                 "--disable-backgrounding-occluded-windows"],
        )
        try:
            yield ctx
        finally:
            ctx.close()


@pytest.fixture
def ui(browser_ctx, mock_backend):
    """A fresh UI page against a reset mock (twin mode, on the ground, manual scenario).

    After the test, every message the UI sent must validate against the frozen contracts.
    """
    mock_backend.reset(flight_mode="twin", scenario="manual")
    page = browser_ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(UI_URL)
    page.wait_for_selector('[data-testid="link-control"][data-status="open"]', timeout=15000)
    page.wait_for_selector('[data-testid="flight-state"][data-state="ON_GROUND"]', timeout=15000)
    page.wait_for_selector('[data-testid="link-agent"][data-status="open"]', timeout=15000)
    yield page
    page.close()
    invalid = [e for e in mock_backend.entries() if not e["valid"]]
    assert not invalid, f"UI sent messages that break the contracts: {invalid[:5]}"
    assert not errors, f"page errors: {errors}"


@pytest.fixture(scope="session")
def screens_dir() -> Path:
    SCREENS.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    return SCREENS
