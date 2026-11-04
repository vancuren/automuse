"""The production bundle (what flies) works: built with `npm run build`, served same-origin by
the mock on :8602 at /ui/, driven in Chrome. Checks the worker chunk, heartbeat, hotkeys and
the HTTP fallback in the minified build."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from conftest import FRONTEND, LOG_DIR, MOCK_SCRIPT, Mock, port_free, stop, wait_http

PROD_PORT = 8602
PROD_URL = f"http://127.0.0.1:{PROD_PORT}"
OUT_DIR = LOG_DIR / "dist-prod"
PROD_LOG = LOG_DIR / "mock_prod_messages.jsonl"


@pytest.fixture(scope="module")
def prod_mock():
    env = dict(os.environ, VITE_BACKEND_URL=PROD_URL, VITE_TWIN_CHASE_URL=PROD_URL + "/chase")
    build = subprocess.run(
        ["npm", "run", "build", "--", "--base", "/ui/", "--outDir", str(OUT_DIR), "--emptyOutDir"],
        cwd=FRONTEND, env=env, capture_output=True, text=True, timeout=180,
    )
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    if not port_free(PROD_PORT):
        pytest.fail(f"port {PROD_PORT} is busy")
    out = open(LOG_DIR / "mock_prod.out", "w")
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SCRIPT), "--port", str(PROD_PORT), "--scenario", "manual",
         "--log", str(PROD_LOG), "--static-ui", str(OUT_DIR)],
        stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        wait_http(PROD_URL + "/ui/", 20)
        yield Mock(PROD_URL, PROD_LOG)
    finally:
        stop(proc)
        out.close()


def test_build_points_at_the_configured_backend_only(prod_mock):
    bundles = list((OUT_DIR / "assets").glob("index-*.js"))
    assert bundles, "no bundle (the prod_mock fixture builds it)"
    js = bundles[0].read_text()
    assert PROD_URL in js and "localhost:8000" not in js
    assert list((OUT_DIR / "assets").glob("heartbeat.worker-*.js")), "worker chunk missing"


def test_production_bundle_heartbeat_and_safety_keys(prod_mock, browser_ctx):
    prod_mock.reset(flight_mode="twin", scenario="manual")
    page = browser_ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(PROD_URL + "/ui/")
        page.wait_for_selector('[data-testid="link-control"][data-status="open"]', timeout=15000)
        page.wait_for_selector('[data-testid="flight-state"][data-state="ON_GROUND"]', timeout=15000)

        t0 = prod_mock.now_ms()
        time.sleep(3.0)
        hbs = prod_mock.of_type("hb", "ws", since_ms=t0)
        assert 4.5 <= len(hbs) / 3.0 <= 5.5, f"{len(hbs)} heartbeats in 3 s"
        assert len({e["conn"] for e in hbs}) == 1  # production has no StrictMode double mount
        assert prod_mock.of_type("axes") == []

        page.keyboard.press("Escape")
        assert prod_mock.wait_for(lambda: prod_mock.commands("estop", "ws"))
        page.keyboard.down("Space")
        time.sleep(0.4)
        page.keyboard.up("Space")
        kinds = [e["msg"]["kind"] for e in prod_mock.wait_for(lambda: prod_mock.of_type("agent_event"), min_count=2)]
        assert kinds[:2] == ["ptt_down", "ptt_up"]

        prod_mock.control(ws_mode="down")
        page.wait_for_selector('[data-testid="disconnected-banner"]', timeout=3000)
        page.keyboard.press("KeyL")
        assert prod_mock.wait_for(lambda: prod_mock.commands("land", "http"))
        prod_mock.control(ws_mode="normal")
        page.wait_for_selector('[data-testid="disconnected-banner"]', state="detached", timeout=5000)
    finally:
        page.close()
    invalid = [e for e in prod_mock.entries() if not e["valid"]]
    assert not invalid, invalid[:5]
    assert not errors, errors


def test_a_page_reload_keeps_the_heartbeat_gap_under_the_governor_threshold(prod_mock, browser_ctx):
    """Reloading the production page mid-flight: the heartbeat gap must stay well under the
    governor's 1.5 s neutral threshold."""
    page = browser_ctx.new_page()
    try:
        page.goto(PROD_URL + "/ui/")
        page.wait_for_selector('[data-testid="link-control"][data-status="open"]', timeout=15000)
        time.sleep(1.0)
        t0 = prod_mock.now_ms()
        page.reload()
        page.wait_for_selector('[data-testid="link-control"][data-status="open"]', timeout=15000)
        time.sleep(1.5)
        hbs = prod_mock.of_type("hb", "ws", since_ms=t0 - 500)
        gap = max(b["t_ms"] - a["t_ms"] for a, b in zip(hbs, hbs[1:]))
        (LOG_DIR / "reload_gap_ms.txt").write_text(f"{gap:.1f}\n")
        assert gap < 1000, f"heartbeat gap across a reload: {gap:.0f} ms"
    finally:
        page.close()
