"""Flight replay in the UI: a saved flight plays back with its commands, and it never gets in the way of flying."""
from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.browser


def sel(testid: str) -> str:
    return f'[data-testid="{testid}"]'


def open_replay(ui):
    ui.click(sel("replay-open"))
    ui.wait_for_selector(sel("replay-frame"), timeout=5000)


def test_replay_lists_the_saved_flight_and_its_commands(ui):
    open_replay(ui)
    assert ui.locator(sel("replay-flight")).count() == 1
    assert "follow the blue shirt" in ui.inner_text(sel("replay-flights"))
    marks = ui.inner_text(sel("replay-marks"))
    for line in ("Take off", "Airborne", "“follow the blue shirt”", "Mission: follow · person in blue shirt", "Land"):
        assert line in marks, (line, marks)
    assert "tilt" not in marks.lower(), "camera tilt is not part of the flight's story"
    assert "0:03" in ui.inner_text(sel("replay-time"))


def test_replay_plays_seeks_and_draws_the_subject(ui):
    open_replay(ui)
    ui.wait_for_selector(f'{sel("replay-play")}[data-playing="0"]', timeout=6000)      # 3 s flight: plays to the end
    assert ui.get_attribute(sel("replay-state"), "data-state") == "ON_GROUND"
    ui.locator(sel("replay-mark"), has_text="Mission: follow").click()                 # jump to 0:01.2
    ui.wait_for_selector(f'{sel("replay-mark")}[data-kind="mission"][data-current="1"]', timeout=2000)
    assert ui.get_attribute(sel("replay-state"), "data-state") == "AIRBORNE"
    box = ui.locator(sel("replay-track")).bounding_box()
    ui.mouse.click(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)        # 1.8 s: following
    ui.wait_for_selector(f'{sel("replay-view")} {sel("subject-box")}[data-state="FOLLOW"]', timeout=2000)
    assert "140" in ui.inner_text(sel("replay-sticks")), "the sticks sent to the drone at the playhead"
    ui.click(sel("replay-close"))
    ui.wait_for_selector(sel("replay-view"), state="detached", timeout=2000)


def test_safety_keys_stay_live_while_the_replay_is_open(ui, mock_backend):
    open_replay(ui)
    ui.wait_for_selector(f'{sel("replay-play")}[data-playing="0"]', timeout=6000)
    since = mock_backend.now_ms()
    ui.keyboard.down("Space")                                                          # push-to-talk, not play/pause
    time.sleep(0.3)
    ui.keyboard.up("Space")
    assert mock_backend.wait_for(lambda: [e for e in mock_backend.of_type("agent_event", "agent", since)
                                          if e["msg"].get("kind") == "ptt_down"])
    assert ui.get_attribute(sel("replay-play"), "data-playing") == "0"
    assert ui.locator(sel("land-btn")).is_visible() and ui.locator(sel("estop-btn")).is_visible()
    ui.keyboard.press("Escape")
    assert mock_backend.wait_for(lambda: mock_backend.commands("estop", since_ms=since))


def test_replay_closes_and_locks_when_the_drone_takes_off(ui, mock_backend):
    open_replay(ui)
    mock_backend.control(flight_state="AIRBORNE")
    ui.wait_for_selector(sel("replay-view"), state="detached", timeout=3000)
    assert ui.is_disabled(sel("replay-open")), "the live camera is never hidden in the air"
