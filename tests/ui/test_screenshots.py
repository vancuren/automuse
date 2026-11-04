"""Screenshots of the operator UI in twin and real modes, for the demo video and the README.

Saved to $MHD_TREE/media/ui_screens/ and copied to $MHD_HOME/media/shared/ui/.
"""
from __future__ import annotations

import shutil
import time

from conftest import SHARED

SHOTS: list[str] = []


def sel(testid: str) -> str:
    return f'[data-testid="{testid}"]'


def shoot(page, screens_dir, name: str) -> None:
    path = screens_dir / f"{name}.png"
    page.screenshot(path=str(path))
    assert path.stat().st_size > 50_000, f"{name}.png looks empty"
    shutil.copy2(path, SHARED / path.name)
    SHOTS.append(path.name)


def speak(page, mock_backend, hold_s: float = 1.6) -> None:
    """A full push-to-talk turn through the mock agent: transcript, mission, reply."""
    page.keyboard.down("Space")
    time.sleep(hold_s)
    page.keyboard.up("Space")
    page.wait_for_selector(sel("model-badge"), timeout=5000)
    page.wait_for_selector(f'{sel("transcript")} >> text=Following the person', timeout=5000)


def test_screens_twin(ui, mock_backend, screens_dir):
    mock_backend.control(time_offset_s=4.0)
    speak(ui, mock_backend)
    mock_backend.control(flight_state="AIRBORNE", autopilot_state="FOLLOW", freeze_autopilot=True, pack_time_s=252)
    ui.wait_for_selector(f'{sel("subject-box")}[data-state="FOLLOW"]', timeout=3000)
    ui.wait_for_selector(f'{sel("chase-view")} iframe', timeout=6000)
    time.sleep(1.0)
    shoot(ui, screens_dir, "ui_twin_follow")

    ui.set_viewport_size({"width": 1512, "height": 982})  # MacBook Pro 14" default scaling
    time.sleep(0.8)
    shoot(ui, screens_dir, "ui_twin_follow_1512x982")
    ui.set_viewport_size({"width": 1920, "height": 1080})

    mock_backend.control(autopilot_state="CONFIRM")
    mock_backend.candidates()
    ui.wait_for_selector(sel("candidate-2"), timeout=3000)
    time.sleep(0.5)
    shoot(ui, screens_dir, "ui_twin_candidates")
    ui.click(sel("candidate-1"))  # the operator picks one; the picker clears
    ui.wait_for_selector(sel("candidate-picker"), state="detached", timeout=4000)

    mock_backend.control(autopilot_state="DOUBT", extra_flags=[], frame_age_ms=820)
    ui.keyboard.down("Space")
    ui.wait_for_selector(sel("partial-transcript"), timeout=3000)
    time.sleep(0.9)
    shoot(ui, screens_dir, "ui_twin_listening_doubt")
    ui.keyboard.up("Space")
    mock_backend.control(frame_age_ms=None)


def test_screens_real(ui, mock_backend, screens_dir):
    mock_backend.control(flight_mode="real", pack_time_s=0)
    ui.wait_for_selector(f'{sel("mode-banner")}[data-mode="real"]', timeout=3000)
    ui.click(sel("arm-input"))
    ui.keyboard.type("AR")
    time.sleep(0.4)
    shoot(ui, screens_dir, "ui_real_locked")

    ui.keyboard.type("M")
    ui.keyboard.press("Enter")
    ui.wait_for_selector(sel("armed-ok"), timeout=3000)
    mock_backend.control(time_offset_s=9.0)
    speak(ui, mock_backend)
    mock_backend.control(flight_state="AIRBORNE", autopilot_state="FOLLOW", freeze_autopilot=True, pack_time_s=318)
    ui.wait_for_selector(f'{sel("pack-timer")}[data-level="amber"]', timeout=3000)
    time.sleep(1.0)
    shoot(ui, screens_dir, "ui_real_armed_follow")

    ui.keyboard.press("Escape")
    ui.wait_for_selector(f'{sel("flight-state")}[data-state="ESTOPPED"]', timeout=3000)
    time.sleep(0.6)
    shoot(ui, screens_dir, "ui_real_estopped")

    mock_backend.control(ws_mode="down")
    ui.wait_for_selector(sel("disconnected-banner"), timeout=3000)
    time.sleep(0.6)
    shoot(ui, screens_dir, "ui_real_disconnected")
    mock_backend.control(ws_mode="normal")
    ui.wait_for_selector(sel("disconnected-banner"), state="detached", timeout=5000)
