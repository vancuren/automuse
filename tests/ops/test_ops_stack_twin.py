"""scripts/run_stack.sh in TWIN mode, end to end (opt-in: OPS_STACK=1; needs Chrome and about 2 min).

Starts the whole stack on the ops ports (twin 8666/23550/8650 with its Chrome renderer, backend 8150,
oracle perception, an offline agent, UI preview 8158), opens the UI in Playwright Chrome (profile
under $MHD_HOME/.cache/chrome-ops), flies takeoff -> follow -> FOLLOW through the governor, then
sends SIGINT to the launcher exactly as a terminal Ctrl-C would and checks that the backend landed
the twin and that every process and port is gone.

    OPS_STACK=1 $MHD_PY -m pytest tests/ops/test_ops_stack_twin.py -q -s
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

TREE = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("MHD_HOME", TREE))
BACKEND, UI, T_HTTP, T_RC, T_RTSP = 8150, 8158, 8666, 23550, 8650
pytestmark = pytest.mark.skipif(os.environ.get("OPS_STACK") != "1", reason="opt-in: OPS_STACK=1")


def port_busy(p: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", p)) == 0


def wait(pred, timeout, what):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        try:
            v = pred()
            if v:
                return v
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting for {what} ({last!r})")


def state():
    return httpx.get(f"http://127.0.0.1:{BACKEND}/state", timeout=3).json()


def twin():
    return httpx.get(f"http://127.0.0.1:{T_HTTP}/status", timeout=3).json()


def test_stack_twin_end_to_end(tmp_path):
    for p in (BACKEND, UI, T_HTTP, T_RTSP):
        assert not port_busy(p), f"port {p} busy"
    run = TREE / "logs" / f"ops-test/stack-{time.strftime('%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", "scripts/run_stack.sh", "--mode", "twin", "--twin-ports", f"{T_HTTP},{T_RC},{T_RTSP}",
           "--port", str(BACKEND), "--ui-port", str(UI), "--perception", "oracle", "--agent", "local",
           "--agent-extra", "--no-mic --mute --tts none --no-warm", "--scenario", "s1", "--latency-ms", "500",
           "--chrome-profile", str(HOME / ".cache" / "chrome-ops" / "renderer"), "--logs", str(run)]
    out = open(run / "launcher.out", "w")
    proc = subprocess.Popen(cmd, cwd=TREE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    report = {"run": str(run)}
    try:
        wait(lambda: "UP:" in (run / "stack.log").read_text(), 240, "the stack to come up")
        wait(lambda: twin()["fps"] >= 15, 60, "the renderer at 15 fps")
        report["twin_fps"] = twin()["fps"]
        assert httpx.get(f"http://127.0.0.1:{UI}/", timeout=3).status_code == 200
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(str(HOME / ".cache" / "chrome-ops" / "ui"), channel="chrome",
                                                        headless=True, viewport={"width": 1600, "height": 1000})
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{UI}/")
            page.wait_for_selector("text=TWIN", timeout=20000)
            wait(lambda: (state()["latency_ms"].get("hb_age") or 1e9) < 600, 20, "the UI worker heartbeat")
            report["hb_age_ms"] = state()["latency_ms"]["hb_age"]
            import websockets.sync.client as wsc
            with wsc.connect(f"ws://127.0.0.1:{BACKEND}/ws") as ws:
                ws.send(json.dumps({"type": "cmd", "command": "takeoff"}))
            wait(lambda: state()["flight_state"] == "AIRBORNE", 20, "AIRBORNE")
            m = {"type": "mission", "mission_id": f"ops-{uuid.uuid4().hex[:6]}",
                 "intent": {"action": "follow", "subject_phrase": "person in blue shirt", "category": "person",
                            "color": "blue", "distance": "medium", "reason_code": "none"},
                 "transcript": "follow the person in the blue shirt", "model": "ops-test", "latency_ms": 0.0}
            r = httpx.post(f"http://127.0.0.1:{BACKEND}/mission", json=m, timeout=5).json()
            report["mission_ack"] = r
            if r.get("paused"):
                with wsc.connect(f"ws://127.0.0.1:{BACKEND}/ws") as ws:
                    ws.send(json.dumps({"type": "cmd", "command": "resume"}))
            wait(lambda: state()["autopilot_state"] == "FOLLOW", 30, "FOLLOW")
            time.sleep(4)
            page.screenshot(path=str(run / "ui_follow.png"))
            report["autopilot"] = state()["autopilot_state"]
            report["agent_log_tail"] = (run / "agent.log").read_text()[-300:]
            ctx.close()
        land0 = twin()["drone"]["events"]["land"]
        t0 = time.monotonic()
        os.kill(proc.pid, signal.SIGINT)                    # a terminal Ctrl-C reaches only the launcher
        wait(lambda: twin()["drone"]["events"]["land"] > land0 or twin()["drone"]["phase"] in ("landing", "landed"),
             10, "the twin to receive LAND after Ctrl-C")
        report["land_after_ctrl_c_s"] = round(time.monotonic() - t0, 2)
        proc.wait(60)
        report["launcher_exit_s"] = round(time.monotonic() - t0, 2)
        report["launcher_code"] = proc.returncode
        time.sleep(1)
        busy = [p for p in (BACKEND, UI, T_HTTP, T_RTSP) if port_busy(p)]
        report["ports_still_busy"] = busy
        left = subprocess.run(["pgrep", "-f", f"{run}|--port {BACKEND}|http-port {T_HTTP}|--port {UI}"],
                              capture_output=True, text=True).stdout.split()
        report["processes_left"] = left
        (run / "check.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        assert proc.returncode == 0 and not busy and not left
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        out.close()


def test_stack_in_a_terminal_restarts_a_killed_backend_and_stops_on_ctrl_c():
    """Under a real pseudo-terminal (as the owner runs it): kill -9 the backend (Session A link-loss
    test 1), press Enter in the launcher window to restart it, then Ctrl-C (byte 0x03 -> SIGINT from
    the tty driver to the foreground group)."""
    import pty
    import threading
    for p in (BACKEND, T_HTTP, T_RTSP):
        assert not port_busy(p), f"port {p} busy"
    run = TREE / "logs" / f"ops-test/stack-pty-{time.strftime('%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    argv = ["bash", "scripts/run_stack.sh", "--mode", "twin", "--twin-ports", f"{T_HTTP},{T_RC},{T_RTSP}",
            "--port", str(BACKEND), "--no-ui", "--perception", "none", "--agent", "none", "--no-renderer",
            "--logs", str(run)]
    pid, fd = pty.fork()
    if pid == 0:                                   # child: the pty is its controlling terminal
        os.chdir(TREE)
        os.execvp("bash", argv)
    buf = bytearray()
    stop = threading.Event()

    def drain():
        while not stop.is_set():
            try:
                data = os.read(fd, 4096)
            except OSError:
                return
            if not data:
                return
            buf.extend(data)
    th = threading.Thread(target=drain, daemon=True)
    th.start()
    report = {}
    try:
        wait(lambda: b"UP:" in buf, 120, "the stack to come up")
        b1 = int(subprocess.run(["lsof", "-nP", f"-iTCP:{BACKEND}", "-sTCP:LISTEN", "-t"], capture_output=True,
                                text=True).stdout.split()[0])
        os.kill(b1, signal.SIGKILL)
        wait(lambda: b"press Enter in THIS window" in buf, 10, "the dead-backend banner")
        os.write(fd, b"\n")
        wait(lambda: state()["flight_state"] == "ON_GROUND", 40, "the restarted backend")
        b2 = int(subprocess.run(["lsof", "-nP", f"-iTCP:{BACKEND}", "-sTCP:LISTEN", "-t"], capture_output=True,
                                text=True).stdout.split()[0])
        report["backend_pids"] = [b1, b2]
        assert b2 != b1
        t0 = time.monotonic()
        os.write(fd, b"\x03")                      # Ctrl-C at the terminal
        status = None
        end = time.monotonic() + 45
        while status is None and time.monotonic() < end:
            done, st = os.waitpid(pid, os.WNOHANG)
            if done:
                status = st
            else:
                time.sleep(0.1)
        assert status is not None, "the launcher did not exit within 45 s of Ctrl-C"
        report["exit_s"] = round(time.monotonic() - t0, 2)
        report["exit_status"] = status
        time.sleep(1)
        report["ports_still_busy"] = [p for p in (BACKEND, T_HTTP, T_RTSP) if port_busy(p)]
        (run / "check.json").write_text(json.dumps(report, indent=1))
        (run / "terminal.txt").write_bytes(bytes(buf))
        print(json.dumps(report, indent=1))
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert not report["ports_still_busy"]
        assert b"restarted backend" in buf and b"stopped. logs:" in buf
    finally:
        stop.set()
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pat in (f"http-port {T_HTTP}", f"--port {BACKEND}"):  # never leave a twin or backend behind
            subprocess.run(["pkill", "-f", pat])
        try:
            os.close(fd)
        except OSError:
            pass
