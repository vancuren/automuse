"""Camera-only runs (REAL CAMERA, NO flight): the backend raises the camera_only safety flag in twin
mode, and the UI shows its own amber banner for it. Source-level checks that need no browser; the
browser check is test_ui_browser.py::test_camera_only_banner_keeps_takeoff_and_drops_the_chase_view."""
from __future__ import annotations

import re
from pathlib import Path

from mhd.backend.modes import CAMERA_ONLY_FLAG

SRC = Path(__file__).resolve().parents[2] / "turbodrone" / "frontend" / "src"


def test_the_ui_reads_the_same_flag_the_backend_raises():
    gates = (SRC / "lib" / "gates.ts").read_text()
    assert f'export const CAMERA_ONLY_FLAG = "{CAMERA_ONLY_FLAG}";' in gates
    # Twin mode only: the same flag on a REAL telemetry stream must never relabel the real banner.
    assert 'telemetry?.flight_mode === "twin" && (telemetry.safety_flags ?? []).includes(CAMERA_ONLY_FLAG)' in gates
    blocking = re.search(r"TAKEOFF_BLOCKING_FLAGS[^=]*= \{(.*?)\};", gates, re.S).group(1)
    assert CAMERA_ONLY_FLAG not in blocking, "camera_only must not block the (simulated) takeoff"


def test_mode_banner_has_a_distinct_camera_banner_checked_before_twin():
    banner = (SRC / "components" / "ModeBanner.tsx").read_text()
    i_real = banner.index('data-mode="real"')
    i_cam = banner.index('data-mode="camera"')
    i_twin = banner.index('data-mode="twin"')
    assert i_real < i_cam < i_twin
    cam = banner[i_cam:i_twin]
    assert "REAL CAMERA – NO FLIGHT" in cam
    assert "amber" in cam and "banner-real" not in cam and "sky-" not in cam   # neither red stripes nor twin blue
    assert 'if (mode === "twin" && isCameraOnly(telemetry))' in banner


def test_no_twin_chase_view_in_camera_mode():
    app = (SRC / "App.tsx").read_text()
    assert 'const twin = tlm.telemetry?.flight_mode === "twin" && !isCameraOnly(tlm.telemetry);' in app
    assert "{twin && <ChaseView" in app
