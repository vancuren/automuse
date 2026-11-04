"""scripts/session_a.py in its INTERACTIVE (real-drone) code paths, against the twin (opt-in: OPS_STACK=1).

The real flow cannot be run tonight (nothing may reach the drone), so this drives every prompt with
scripted answers against a twin and a twin-mode backend (headless, since no UI runs here), on the
ops ports and loopback only. It checks that no step crashes and that every step is recorded.
link-loss test 1 (kill -9 the backend) is skipped: it needs the operator to restart the backend.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest

TREE = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("OPS_STACK") != "1", reason="opt-in: OPS_STACK=1")

ANSWERS = [  # (prompt fragment, answer); first match wins; anything else gets "?"
    ("press Enter", ""), ("type ARM", "ARM"), ("props OFF", "y"), ("props marked", "y"),
    ("re-run preflight", "n"), ("G0 over the tether", "?"), ("raw RTSP", "y"), ("subjects clip", "n"),
    ("mic check", "?"), ("webcam", "n"), ("UI shows live", "y"), ("open the page in Chrome", "n"),
    ("open the snapshot", "n"), ("stopwatch", "450"), ("distance camera to wall", "1.0"),
    ("visible tape width", "1.6"), ("did the motors SPIN", "y"), ("roughly how long", "1"),
    ("did the motors STOP", "y"), ("take to stop", "2"), ("CUT at once", "y"), ("how fast", "0"),
    ("stayed OFF", "y"), ("after the E-STOP reset", "y"), ("no-op", "y"), ("anything else", ""),
    ("how many degrees", "5"), ("total travel", "60"), ("leave the camera", "slightly down"),
    ("what did the motors do", "r"), ("after how many seconds", "2"), ("rejoin", "y"),
    ("keep streaming", "y"), ("LED change", ""), ("did the motors spin?", "y"), ("remote ON", "n"),
    ("remote's stop", "y"), ("which remote control", "power button"), ("WITH battery", "260"),
    ("low-battery LED", "blinks twice"), ("motors still spinning", "n"),
]


def answer(prompt: str) -> str:
    for frag, ans in ANSWERS:
        if frag.lower() in prompt.lower():
            return ans
    return "?"


def test_session_a_interactive_paths_against_the_twin(tmp_path, monkeypatch):
    run = TREE / "logs" / f"ops-test/sa-int-{time.strftime('%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    twin = subprocess.Popen(["bash", "scripts/run_twin.sh", "--ports", "8666,23550,8650", "--latency-ms", "500",
                             "--no-renderer", "--", "--log-dir", str(run / "twin")], cwd=TREE,
                            stdout=open(run / "twin.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    backend = None
    try:
        for _ in range(60):
            try:
                if httpx.get("http://127.0.0.1:8666/status", timeout=1).json().get("fps", 0) > 5:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        backend = subprocess.Popen(["bash", "scripts/run_backend.sh", "--mode", "twin", "--port", "8150",
                                    "--control-port", "23550", "--rtsp-url", "rtsp://127.0.0.1:8650/live/ch00_1",
                                    "--headless", "--logs", str(run / "backend")], cwd=TREE,
                                   stdout=open(run / "backend.log", "w"), stderr=subprocess.STDOUT,
                                   start_new_session=True)
        for _ in range(80):
            try:
                if httpx.get("http://127.0.0.1:8150/state", timeout=1).json()["latency_ms"].get("frames_received", 0) > 5:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        spec = importlib.util.spec_from_file_location("session_a", TREE / "scripts" / "session_a.py")
        sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa)
        asked: list[tuple[str, str]] = []

        def fake_input(self, prompt):
            a = answer(prompt)
            asked.append((prompt, a))
            print(f"  ?  {prompt} -> {a!r}")
            if a.lower() == "s":
                raise sa.SkipStep()
            return a
        monkeypatch.setattr(sa.IO, "_input", fake_input)
        monkeypatch.setenv("MHD_OPS_LOOPBACK_ONLY", "1")
        code = sa.main(["--drone-ip", "127.0.0.1", "--backend-port", "8150", "--rtsp-url",
                        "rtsp://127.0.0.1:8650/live/ch00_1", "--counter-port", "8157", "--skip-ml",
                        "--skip", "linkloss_kill", "--latency-s", "5", "--link-s", "4", "--motor-s", "3",
                        "--media-dir", str(run / "media"), "--no-color"])
        rec = sorted(TREE.glob("logs/session_a-*.json"), key=os.path.getmtime)[-1]
        data = json.loads(rec.read_text())
        statuses = {k: v["status"] for k, v in data["steps"].items()}
        (run / "asked.json").write_text(json.dumps(asked, indent=1))
        print(json.dumps(statuses, indent=1), data["summary"]["verdict"])
        assert code == 0
        assert all(v == "done" for v in statuses.values()), statuses
        assert len(statuses) == 16
        assert data["steps"]["hfov"]["measured"]["hfov_deg"] == pytest.approx(77.3, abs=0.1)
        assert data["steps"]["linkloss_rcpause"]["measured"]["rc_pause_response"]["ok"] is True
        assert data["steps"]["flag_estop"]["measured"]["takeoff_refused_while_estopped"] is True
    finally:
        for p in (backend, twin):
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
