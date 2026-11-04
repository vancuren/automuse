"""Static checks on the operator UI source, plus the production build and lint."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "turbodrone" / "frontend"
SRC = FRONTEND / "src"
SOURCES = sorted(p for p in SRC.rglob("*") if p.suffix in {".ts", ".tsx", ".css"})


def text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_backend_url_lives_only_in_config():
    offenders = [p.name for p in SOURCES if "localhost" in text(p) and p.name != "config.ts"]
    assert offenders == [], f"hard-coded backend URL in {offenders}"
    assert 'VITE_BACKEND_URL || "http://localhost:8000"' in text(SRC / "config.ts")


def test_speed_tier_profile_and_plugin_controls_are_gone():
    for word in ("set_speed_index", "speed_index", "set_profile", "SpeedControl", "/plugins/", "requestPointerLock"):
        offenders = [p.name for p in SOURCES if word in text(p)]
        assert offenders == [], f"{word!r} still used in {offenders}"


def test_axes_never_carry_extra_fields():
    # OperatorInput is extra="forbid": upstream's camera_tilt_direction inside axes is invalid.
    assert not [p.name for p in SOURCES if "camera_tilt_direction" in text(p)]


def test_only_the_worker_sends_heartbeats():
    # an object literal {type: "hb", ...}; the interface declaration `type: "hb";` does not count
    senders = [p.relative_to(SRC).as_posix() for p in SOURCES if re.search(r'\{[ \t]*type:[ \t]*"hb"', text(p))]
    assert senders == ["workers/heartbeat.worker.ts"], senders


def test_every_button_goes_through_UiButton():
    raw = [p.name for p in SOURCES if p.suffix == ".tsx" and "<button" in text(p) and p.name != "UiButton.tsx"]
    assert raw == [], f"raw <button> (can take focus) in {raw}"
    ui_button = text(SRC / "components" / "UiButton.tsx")
    assert "tabIndex={-1}" in ui_button and "e.preventDefault()" in ui_button


def test_subject_never_called_target():
    files = SOURCES + [FRONTEND / "index.html", FRONTEND / "mock" / "mock_backend.py"]
    offenders = [p.name for p in files if re.search(r"target", text(p), re.I)]
    assert offenders == [], f"'target' used in {offenders} (the followed thing is the subject)"


def test_no_web_fonts_or_cdn():
    for p in SOURCES + [FRONTEND / "index.html"]:
        assert "fonts.googleapis" not in text(p) and not re.search(r"https://\w", text(p)), p.name


def test_dev_server_binds_loopback_only():
    assert "host: '127.0.0.1'" in text(FRONTEND / "vite.config.ts")


@pytest.mark.slow
def test_production_build_passes():
    r = subprocess.run(["npm", "run", "build"], cwd=FRONTEND, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert any((FRONTEND / "dist" / "assets").glob("heartbeat.worker-*.js")), "worker chunk missing"


@pytest.mark.slow
def test_lint_is_clean():
    r = subprocess.run(["npx", "eslint", "src", "--max-warnings", "0"], cwd=FRONTEND,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
