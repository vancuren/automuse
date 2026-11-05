"""Shared helpers for the UI tests (ports, paths, process control, the mock client).

A uniquely named module, so tests never import `conftest` by name: every workstream has a
tests/<ws>/conftest.py, and a plain `from conftest import` could pick up the wrong one.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

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
FLIGHTS_DIR = LOG_DIR / "flights"          # the mock serves these to the replay (never the real logs/flights)
SAMPLE_FLIGHT = "20261104-120000-twin"
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


def write_sample_flight(root: Path) -> Path:
    """A 3 s saved flight in the recorder's format (src/mhd/backend/recorder.py) for the replay tests."""
    import shutil

    import cv2
    import numpy as np

    d = root / SAMPLE_FLIGHT
    shutil.rmtree(root, ignore_errors=True)
    d.mkdir(parents=True)
    lines: list[tuple[float, dict]] = []
    for i in range(31):
        t = i * 100.0
        state = "TAKING_OFF" if t < 500 else "AIRBORNE" if t < 2600 else "ON_GROUND"
        following = 1300 <= t < 2600
        x = 400 + i * 10
        lines.append((t, {"k": "tlm", "d": {
            "type": "telemetry", "ts_ms": 1000.0 + t, "flight_mode": "twin", "flight_state": state,
            "autopilot_state": "FOLLOW" if following else "IDLE", "armed": False,
            "sticks_wire": {"throttle": 128, "yaw": 128 + i, "pitch": 140, "roll": 128},
            "sticks_by_source": {}, "active_source": "autopilot" if following else "none",
            "frame_age_ms": 90.0, "obs_age_ms": 120.0, "latency_ms": {}, "flight_time_s": t / 1000.0, "pack_id": "1",
            "safety_flags": [],
            "mission": {"type": "mission", "mission_id": "m-1", "transcript": "follow the blue shirt",
                        "intent": {"action": "follow", "subject_phrase": "person in blue shirt"},
                        "model": "glimmer-30b+lora", "latency_ms": 1200.0, "reply": "Following."} if following else None,
            "subject": {"type": "subject_obs", "mission_id": "m-1", "frame_id": i, "capture_ms": 1000.0 + t,
                        "state": "confirmed", "track_id": 1, "box": {"x1": x, "y1": 200, "x2": x + 160, "y2": 620},
                        "confidence": 0.9, "tracker_score": 0.9, "confirm_age_ms": 100.0, "mask_poly": None,
                        "source": "sam_mlx", "detector_latency_ms": None, "render_id": None} if following else None}}))
        lines.append((t + 1, {"k": "frame", "i": i}))
    lines += [
        (200.0, {"k": "cmd", "d": {"command": "takeoff", "source": "ui", "ok": True, "reason": None}}),
        (500.0, {"k": "agent", "d": {"type": "agent_event", "kind": "status", "text": "flight TAKING_OFF -> AIRBORNE",
                                     "data": {"kind": "flight_state", "from": "TAKING_OFF", "to": "AIRBORNE"}}}),
        (1000.0, {"k": "agent", "d": {"type": "agent_event", "kind": "final", "text": "follow the blue shirt",
                                      "data": {"model": "muse-voice-transcribe-1.0"}}}),
        (1200.0, {"k": "mission", "d": {"mission_id": "m-1", "transcript": "follow the blue shirt", "ok": True,
                                        "intent": {"action": "follow", "subject_phrase": "person in blue shirt"},
                                        "model": "glimmer-30b+lora", "latency_ms": 1200.0}}),
        (1250.0, {"k": "cmd", "d": {"command": "tilt_up", "source": "ui", "ok": True, "reason": None}}),
        (2500.0, {"k": "cmd", "d": {"command": "land", "source": "ui", "ok": True, "reason": None}}),
    ]
    lines.sort(key=lambda e: e[0])
    (d / "events.jsonl").write_text("".join(json.dumps({"t": t, **e}) + "\n" for t, e in lines), encoding="utf-8")
    offset, idx = 0, []
    with open(d / "frames.bin", "wb") as fb:
        for i in range(31):
            img = np.full((180, 320, 3), (60 + i * 4, 90, 40), np.uint8)
            jpeg = cv2.imencode(".jpg", img)[1].tobytes()
            fb.write(jpeg)
            idx.append(f"{offset} {len(jpeg)}")
            offset += len(jpeg)
    (d / "frames.idx").write_text("\n".join(idx) + "\n", encoding="ascii")
    (d / "meta.json").write_text(json.dumps({
        "id": SAMPLE_FLIGHT, "started_unix": 1793811600.0, "flight_mode": "twin", "duration_ms": 3000, "frames": 31,
        "complete": True, "commands": [{"t": 200, "kind": "cmd", "command": "takeoff"},
                                       {"t": 1200, "kind": "mission", "transcript": "follow the blue shirt"},
                                       {"t": 2500, "kind": "cmd", "command": "land"}]}), encoding="utf-8")
    return d
