"""Opt-in integration smoke test: the operator UI against the REAL patched drone-backend
(governor installed), in twin mode, with no twin running.

    UI_GOV_TREE=/path/to/tree/with/patched/backend $MHD_PY -m pytest tests/ui/test_ui_gov_backend.py -v

After integration the backend lives in this tree, so UI_GOV_TREE=$MHD_TREE works. Skipped when
UI_GOV_TREE is unset. Nothing is written to UI_GOV_TREE: configs are read from it, logs go to
$MHD_TREE/logs/ui_tests/gov_backend, bytecode writing is off. RC packets go to UDP 127.0.0.1:8604
and RTSP to 127.0.0.1:8606 where nothing listens, so no twin or drone is ever commanded.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from conftest import LOG_DIR, UI_URL, port_free, stop, wait_http

GOV_TREE = os.environ.get("UI_GOV_TREE")
pytestmark = pytest.mark.skipif(not GOV_TREE, reason="set UI_GOV_TREE to a tree with the patched backend")

BACKEND_PORT = 8603
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
RC_UDP_PORT = 8604     # nothing listens: RC datagrams go nowhere
RTSP_PORT = 8606       # nothing listens: no video


def sel(testid: str) -> str:
    return f'[data-testid="{testid}"]'


@pytest.fixture(scope="module")
def gov_backend():
    tree = Path(GOV_TREE).resolve()
    backend_dir = tree / "turbodrone" / "backend"
    assert (tree / "src" / "mhd" / "governor").is_dir(), f"no governor in {tree}"
    if not port_free(BACKEND_PORT):
        pytest.fail(f"port {BACKEND_PORT} is busy")
    logs = LOG_DIR / "gov_backend"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "FLIGHT_MODE": "twin", "DRONE_IP": "127.0.0.1", "DRONE_TYPE": "x69_lg", "CONTROL_PORT": str(RC_UDP_PORT),
        "X69_LG_VIDEO_MODE": "rtsp", "X69_LG_RTSP_URL": f"rtsp://127.0.0.1:{RTSP_PORT}/live/ch00_1",
        "PLUGINS_ENABLED": "false", "MHD_LOGS": str(logs), "PYTHONDONTWRITEBYTECODE": "1",
        "GOVERNOR_CONFIG": str(tree / "config" / "governor.yaml"),
        "AUTOPILOT_CONFIG": str(tree / "config" / "autopilot.yaml"),
        "PYTHONPATH": f"{tree / 'src'}:{backend_dir}",
    })
    env.pop("GOVERNOR_TWIN_HEADLESS", None)
    out = open(logs / "backend.out", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web_server:app", "--app-dir", str(backend_dir),
         "--host", "127.0.0.1", "--port", str(BACKEND_PORT), "--log-level", "warning"],
        cwd=logs, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        wait_http(BACKEND_URL + "/state", 30)
        yield BACKEND_URL
    finally:
        stop(proc)
        out.close()


@pytest.fixture(scope="module")
def gov_vite(gov_backend):
    """vite on 5273 pointed at the real backend (the session dev server is not used here)."""
    frontend = Path(__file__).resolve().parents[2] / "turbodrone" / "frontend"
    if not port_free(5273):
        pytest.skip("vite port 5273 busy (run this module on its own)")
    env = dict(os.environ, VITE_BACKEND_URL=gov_backend)
    out = open(LOG_DIR / "gov_vite.out", "w")
    proc = subprocess.Popen([str(frontend / "node_modules/.bin/vite"), "--port", "5273", "--strictPort"],
                            cwd=frontend, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_http(UI_URL, 30)
        yield UI_URL
    finally:
        stop(proc)
        out.close()


@pytest.fixture(scope="module")
def gov_page(gov_vite):
    from playwright.sync_api import sync_playwright

    from conftest import CHROME_PROFILE
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(CHROME_PROFILE), channel="chrome", headless=True,
                                                   viewport={"width": 1920, "height": 1080})
        page = ctx.new_page()
        page.goto(gov_vite)
        page.wait_for_selector(f'{sel("link-control")}[data-status="open"]', timeout=15000)
        page.wait_for_selector(f'{sel("mode-banner")}[data-mode="twin"]', timeout=15000)
        try:
            yield page
        finally:
            ctx.close()


def state(url: str) -> dict:
    return httpx.get(url + "/state", timeout=3).json()


def test_telemetry_renders_and_twin_needs_no_arm(gov_backend, gov_page):
    s = state(gov_backend)
    assert s["flight_mode"] == "twin"
    gov_page.wait_for_selector(f'{sel("flight-state")}[data-state="{s["flight_state"]}"]', timeout=3000)
    assert gov_page.locator(sel("arm-gate")).count() == 0
    # No twin, so no video: the governor would refuse takeoff, and the button says why.
    if "video_stale" in s["safety_flags"]:
        gov_page.wait_for_selector(f'{sel("takeoff-blocker")} >> text=no video', timeout=3000)
        assert gov_page.is_disabled(sel("takeoff-btn"))


def test_heartbeat_reaches_the_governor(gov_backend, gov_page):
    time.sleep(2.0)
    s = state(gov_backend)
    hb_age = s["latency_ms"].get("hb_age")
    flags = " ".join(s["safety_flags"])
    assert hb_age is not None and hb_age < 600, f"governor sees no UI heartbeat: hb_age={hb_age} flags={flags}"


def test_escape_estops_the_real_governor_and_reset_works(gov_backend, gov_page):
    gov_page.keyboard.press("Escape")
    gov_page.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=3000)
    assert state(gov_backend)["flight_state"] == "ESTOPPED"
    gov_page.click(sel("estop-reset-btn"))
    gov_page.click(sel("estop-reset-btn"))
    gov_page.wait_for_selector(f'{sel("flight-state")}[data-state="ON_GROUND"]', timeout=3000)


def test_http_fallback_reaches_the_real_governor(gov_backend):
    r = httpx.post(gov_backend + "/command", json={"type": "cmd", "command": "estop"}, timeout=3,
                   headers={"Origin": "http://127.0.0.1:5273"})
    assert r.status_code == 200 and r.json().get("ok") is True, r.text
    assert r.headers.get("access-control-allow-origin") in ("*", "http://127.0.0.1:5273")
    httpx.post(gov_backend + "/command", json={"type": "cmd", "command": "estop_reset"}, timeout=3)


def test_no_contract_errors_in_backend_log(gov_backend, gov_page):
    log = (LOG_DIR / "gov_backend" / "backend.out").read_text()
    for bad in ("ValidationError", "Traceback", "extra_forbidden"):
        assert bad not in log, log[-3000:]


def test_screenshot_on_the_real_backend(gov_backend, gov_page):
    import shutil

    from conftest import SCREENS, SHARED
    SCREENS.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    time.sleep(1.5)
    path = SCREENS / "ui_gov_backend_twin_no_video.png"
    gov_page.screenshot(path=str(path))
    shutil.copy2(path, SHARED / path.name)
