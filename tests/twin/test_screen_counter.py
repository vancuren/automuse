"""Screen counter page (for the real drone's latency measurement): render in Chrome, then decode
screenshots after a camera-like perspective warp, blur, noise, and JPEG. Also checks the live
page's clock sync: a fresh screenshot decodes to a render time only milliseconds old."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mhd.contracts import now_ms
from mhd.twin.screen_counter import decode_screen

from procs import HTTP_PORT, PY, Proc, wait_http, wait_ports_free

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright
    wait_ports_free([HTTP_PORT])
    srv = Proc("screen-counter", [PY, "-m", "mhd.twin.screen_counter", "serve", "--port", str(HTTP_PORT)])
    wait_http(f"http://127.0.0.1:{HTTP_PORT}/counter/layout", proc=srv)
    home = Path(os.environ.get("MHD_HOME", "."))
    profile = home / ".cache" / "chrome-twin" / "test-profile"
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=False, viewport={"width": 1280, "height": 720},
            args=["--use-mock-keychain", "--no-first-run", "--disable-background-timer-throttling",
                  f"--disk-cache-dir={profile / 'cache'}"])
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        yield pg
        ctx.close()
    srv.stop()


def shot(pg) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(pg.screenshot(), np.uint8), cv2.IMREAD_COLOR)


def camera_like(img: np.ndarray, seed: int) -> np.ndarray:
    """Perspective view of the screen, resampled to 1280x720, blurred, noisy, JPEG q70."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = rng.uniform(0.03, 0.12, (4, 2)) * [w, h]
    dst = np.float32([[j[0, 0], j[0, 1]], [w - j[1, 0], j[1, 1]], [w - j[2, 0], h - j[2, 1]], [j[3, 0], h - j[3, 1]]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 90, 90))
    warped = cv2.resize(warped, (1280, 720), interpolation=cv2.INTER_AREA)
    warped = cv2.GaussianBlur(warped, (0, 0), 1.2)
    warped = np.clip(warped.astype(np.int16) + rng.normal(0, 4, warped.shape).astype(np.int16), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_static_code_decodes_through_a_camera_like_view(page):
    rid, ms = 54321, 987_654_321
    page.goto(f"http://127.0.0.1:{HTTP_PORT}/counter?static={rid},{ms}")
    page.wait_for_function("window.__ready === true", timeout=10000)
    img = shot(page)
    assert decode_screen(img, ref_ms=ms + 50) == (rid, float(ms))
    ok = sum(decode_screen(camera_like(img, s), ref_ms=ms + 50) == (rid, float(ms)) for s in range(20))
    assert ok >= 19, f"{ok}/20 camera-like views decoded"


def test_live_page_clock_is_synced(page):
    page.goto(f"http://127.0.0.1:{HTTP_PORT}/counter")
    page.wait_for_function("window.__ready === true && window.__counter !== undefined", timeout=10000)
    time.sleep(0.5)
    lat = []
    for _ in range(5):
        img = shot(page)
        t = now_ms()
        d = decode_screen(img, ref_ms=t)
        assert d is not None
        lat.append(t - d[1])
    rtt = page.evaluate("window.__counter.bestRtt")
    assert rtt < 20
    # a screenshot is a few frames old at most; the clocks must agree to well under 0.2 s
    assert all(-20 < x < 200 for x in lat), lat
