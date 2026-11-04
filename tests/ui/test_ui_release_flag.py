"""Opt-in build flag VITE_AXES_RELEASE_MESSAGE=1: exactly one all-zero axes message when stick
input stops (key up, blur), so the governor releases at once instead of after 0.25 s.
Built and served like test_ui_production (mock on :8604, same origin, /ui/)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from ui_harness import FRONTEND, LOG_DIR, MOCK_SCRIPT, Mock, port_free, stop, wait_http

pytestmark = pytest.mark.browser

PORT = 8604
URL = f"http://127.0.0.1:{PORT}"
OUT_DIR = LOG_DIR / "dist-release"
LOG = LOG_DIR / "mock_release_messages.jsonl"


@pytest.fixture(scope="module")
def release_mock():
    env = dict(os.environ, VITE_BACKEND_URL=URL, VITE_AXES_RELEASE_MESSAGE="1")
    build = subprocess.run(["npm", "run", "build", "--", "--base", "/ui/", "--outDir", str(OUT_DIR), "--emptyOutDir"],
                           cwd=FRONTEND, env=env, capture_output=True, text=True, timeout=180)
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    if not port_free(PORT):
        pytest.fail(f"port {PORT} is busy")
    out = open(LOG_DIR / "mock_release.out", "w")
    proc = subprocess.Popen([sys.executable, str(MOCK_SCRIPT), "--port", str(PORT), "--scenario", "manual",
                             "--log", str(LOG), "--static-ui", str(OUT_DIR)],
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_http(URL + "/ui/", 20)
        yield Mock(URL, LOG)
    finally:
        stop(proc)
        out.close()


def test_one_release_message_on_key_up_and_on_blur(release_mock, browser_ctx):
    release_mock.reset(flight_mode="twin", scenario="manual")
    page = browser_ctx.new_page()
    try:
        page.goto(URL + "/ui/")
        page.wait_for_selector('[data-testid="link-control"][data-status="open"]', timeout=15000)
        time.sleep(1.0)
        assert release_mock.of_type("axes") == []  # still nothing while idle

        page.keyboard.down("KeyW")
        time.sleep(0.4)
        page.keyboard.up("KeyW")
        time.sleep(0.5)
        axes = release_mock.of_type("axes", "ws")
        zeros = [e for e in axes if not any(e["msg"][a] for a in ("throttle", "yaw", "pitch", "roll"))]
        assert len(zeros) == 1 and zeros[0] is axes[-1] and zeros[0]["note"] == "release", axes[-3:]

        page.keyboard.down("KeyD")
        time.sleep(0.4)
        page.evaluate("() => window.dispatchEvent(new Event('blur'))")
        time.sleep(0.5)
        page.keyboard.up("KeyD")
        axes = release_mock.of_type("axes", "ws")
        zeros = [e for e in axes if not any(e["msg"][a] for a in ("throttle", "yaw", "pitch", "roll"))]
        assert len(zeros) == 2 and zeros[1] is axes[-1], "no release message on blur"
    finally:
        page.close()
    assert all(e["valid"] for e in release_mock.entries()), [e for e in release_mock.entries() if not e["valid"]][:3]
