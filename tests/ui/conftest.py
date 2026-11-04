"""Fixtures for the operator UI tests.

Browser tests start, once per session:
  - the mock backend (turbodrone/frontend/mock/mock_backend.py) on 127.0.0.1:8601, scenario "manual";
  - vite on 127.0.0.1:5273 with VITE_BACKEND_URL pointing at the mock;
  - Google Chrome through Playwright (channel="chrome", profile $MHD_HOME/.cache/chrome-ui).
Everything is stopped at the end of the session. Run: $MHD_PY -m pytest tests/ui -v
Set UI_HEADED=1 to watch the browser.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ui_harness import (CHROME_PROFILE, FRONTEND, LOG_DIR, MOCK_LOG, MOCK_PORT, MOCK_SCRIPT, MOCK_URL, SCREENS,
                        SHARED, UI_URL, VITE_BIN, VITE_PORT, Mock, port_free, stop, wait_http)


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
    # The mock, like the governor, flags heartbeat_missing until the worker's first heartbeat.
    page.wait_for_selector('[data-testid="takeoff-blocker"] >> text=ready', timeout=15000)
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


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs npm (build or lint)")
    config.addinivalue_line("markers", "browser: drives Google Chrome against vite and the mock (deselect: -m 'not browser')")
