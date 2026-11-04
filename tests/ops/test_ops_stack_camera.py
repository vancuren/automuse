"""scripts/run_stack.sh --mode camera (REAL CAMERA, NO flight), checked with --dry-run: it would start
mediamtx and the Lewei bridge instead of the twin, the backend with --camera, perception without the
twin's counter mask and with the camera HFOV, and it refuses the oracle and the twin options.
--dry-run starts nothing (and writes only its stack.log and the backend's empty log dir). Ports are unusual ones so a
stack running on the defaults does not matter."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts" / "run_stack.sh"
PORTS = ["--port", "8171", "--ui-port", "8179", "--rtsp-port", "8673"]
URL = "rtsp://127.0.0.1:8673/live/ch00_1"


def run(*args) -> tuple[int, str]:
    p = subprocess.run(["bash", str(SCRIPT), *args], cwd=TREE, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def started(out: str) -> dict[str, str]:
    """name -> command, from the "DRY RUN: would start NAME: CMD" lines, in order."""
    return dict(re.findall(r"DRY RUN: would start (\w+): (.*)", out))


def test_camera_dry_run_starts_mediamtx_and_the_bridge_and_never_the_twin(tmp_path):
    run_dir = tmp_path / "run"
    rc, out = run("--mode", "camera", "--dry-run", *PORTS, "--logs", str(run_dir))
    assert rc == 0, out
    cmds = started(out)
    assert list(cmds) == ["mediamtx", "bridge", "backend", "perception", "agent"], out
    assert "twin" not in cmds and "renderer" not in cmds
    assert "run_twin.sh" not in out and "launch_renderer" not in out and "mhd.twin.oracle" not in out
    assert cmds["mediamtx"].endswith(f"tools/mediamtx/mediamtx {run_dir}/mediamtx_8673.yml")
    assert cmds["bridge"].endswith(f"scripts/lewei_bridge.py --rtsp-out {URL} --log-dir {run_dir}")
    b = cmds["backend"]
    assert b.startswith("bash scripts/run_backend.sh --camera --port 8171 ") and f"--rtsp-url {URL}" in b
    assert "--mode" not in b and "--control-port" not in b
    assert "config/camera/governor.yaml" in b and "config/camera/autopilot.yaml" in b
    # run_backend.sh's own dry run, printed by run_stack: camera-only twin, loopback, the camera configs
    assert "mode=twin drone=127.0.0.1:23999" in out and f"rtsp={URL}" in out and "camera_only=1" in out
    assert "governor config=" in out and "config/camera/governor.yaml" in out
    p = cmds["perception"]
    assert "--mask-counter off" in p and "--hfov-deg 75.0" in p and "--backend ws://127.0.0.1:8171" in p
    assert "--backend ws://127.0.0.1:8171" in cmds["agent"]
    assert "would build the UI" in out and "VITE_TWIN_CHASE_URL" not in out
    assert "REAL CAMERA, NO FLIGHT" in out and "DRY RUN done: nothing was started" in out
    # Nothing ran: no mediamtx config, no bridge log (run_backend.sh --dry-run makes its empty log dir).
    assert sorted(x.name for x in run_dir.iterdir()) == ["backend", "stack.log"]
    assert not any((run_dir / "backend").iterdir())


def test_camera_logs_go_to_a_camera_run_dir_and_extra_bridge_flags_pass_through():
    rc, out = run("--mode", "camera", "--dry-run", *PORTS, "--no-agent", "--no-ui", "--no-perception",
                  "--bridge-extra", "--low-def --cam-ip 192.168.0.1")
    m = re.search(r"logs (\S+/logs/stack-\d{8}-\d{6}-camera)\b", out)
    try:
        assert rc == 0, out
        assert m, out
        assert list(started(out)) == ["mediamtx", "bridge", "backend"]
        assert started(out)["bridge"].endswith(f"--rtsp-out {URL} --log-dir {m.group(1)} --low-def --cam-ip 192.168.0.1")
    finally:
        if m and Path(m.group(1)).is_dir() and Path(m.group(1)).parent == TREE / "logs":
            shutil.rmtree(m.group(1))


def test_camera_refuses_the_oracle_twin_options_and_a_second_rtsp_url():
    for extra, msg in ((["--perception", "oracle"], "refuses --perception oracle"),
                       (["--scenario", "s2"], "refuses twin options"),
                       (["--twin-ports", "8766,23458,8554"], "refuses twin options"),
                       (["--no-renderer"], "refuses twin options"),
                       (["--bridge-extra", "--rtsp-out rtsp://127.0.0.1:9/x"], "may not change --rtsp-out"),
                       (["--bridge-extra", "--no-rtsp"], "may not change --rtsp-out"),
                       (["--rtsp-port", "80"], "--rtsp-port must be"),
                       (["--camera-wait", "0"], "--camera-wait must be")):
        rc, out = run("--mode", "camera", "--dry-run", *PORTS[:4], *extra)
        assert rc == 2 and msg in out, (extra, out)
        assert "would start" not in out


def test_camera_options_belong_to_camera_mode_and_camera_configs_never_fly():
    rc, out = run("--mode", "twin", "--dry-run", "--rtsp-port", "8673")
    assert rc == 2 and "camera-mode options" in out
    # The real stack refuses config/camera (twin_only, "TWIN calibration") before it starts anything.
    rc, out = run("--mode", "real", "--dry-run", "--config-dir", "config/camera")
    assert rc == 2 and "REAL mode refuses" in out and "would start" not in out


def test_twin_dry_run_installs_nothing(tmp_path):
    """--dry-run used to run npm ci in twin/web when three.js was missing (found while adding camera mode)."""
    rc, out = run("--mode", "twin", "--dry-run", "--port", "8171", "--twin-ports", "8766,23458,8673",
                  "--no-ui", "--no-agent", "--no-perception", "--logs", str(tmp_path / "run"))
    assert rc == 0, out
    assert "npm install in twin/web" not in out
    assert list(started(out))[:1] == ["twin"]


# ------------------------------------------------------------------ end to end (opt-in: OPS_CAMERA=1)
# The real launcher with a loopback stand-in for the Lewei camera board: run_stack.sh --mode camera
# starts mediamtx, the real scripts/lewei_bridge.py (--cam-ip 127.0.0.1) and the backend; the test
# takes off through the governor, checks that no process of the stack has a UDP socket, then Ctrl-Cs
# the launcher and checks that everything stopped.
#     OPS_CAMERA=1 $MHD_PY -m pytest tests/ops/test_ops_stack_camera.py -q -s -k end_to_end

E2E_BACKEND, E2E_RTSP = 8172, 8674


def _h264_access_units(tmp: Path) -> list[bytes]:
    out = tmp / "testsrc.h264"
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=1280x720:rate=12", "-t", "4", "-c:v", "libx264", "-profile:v", "baseline",
                    "-g", "12", "-x264-params", "repeat-headers=1", "-pix_fmt", "yuv420p", "-f", "h264", str(out)],
                   check=True, timeout=60)
    data = out.read_bytes()
    starts = [m.start() for m in re.finditer(b"\x00\x00\x01", data)]
    aus, cur = [], b""
    for i, s in enumerate(starts):
        b = s - 1 if s > 0 and data[s - 1] == 0 else s          # include a 4-byte start code's leading 0
        e = starts[i + 1] if i + 1 < len(starts) else len(data)
        e = e - 1 if i + 1 < len(starts) and data[e - 1] == 0 else e
        nal = data[b:e]
        cur += nal
        if data[s + 3] & 0x1F in (1, 5):                        # a slice ends the access unit
            aus.append(cur)
            cur = b""
    assert len(aus) >= 40, len(aus)
    return aus


class _LoopbackLeweiCamera:
    """Serves H.264 access units the way the Lewei board does (plain stream type), 12 fps, forever."""

    def __init__(self, aus: list[bytes]):
        self.aus, self.commands, self.stop = aus, [], threading.Event()
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(2)
        self.srv.settimeout(0.2)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        from mhd.lewei import protocol as p

        def hdr(cmd, size):
            return p.MAGIC + struct.pack("<9I", cmd, 0, 0, size, p.STREAM_PLAIN, 0, 0, 0, 0)
        c.settimeout(0.01)
        buf = bytearray()

        def take(data: bytes) -> None:          # the client's 46-byte commands, in order
            buf.extend(data)
            while len(buf) >= p.HDR_LEN:
                self.commands.append(p.parse_header(bytes(buf[:p.HDR_LEN])).cmd)
                del buf[:p.HDR_LEN]
        try:
            c.sendall(hdr(p.CMD_HEARTBEAT, 0))
            n, nxt = 0, time.monotonic()
            while not self.stop.is_set():
                if time.monotonic() >= nxt:
                    au = self.aus[n % len(self.aus)]
                    body = struct.pack("<LLQL", 1, len(au), n, 0) + b"\0" * (p.FRAME_OFF - 20) + au
                    c.sendall(hdr(p.RET_STREAM, len(body)) + body)
                    n, nxt = n + 1, nxt + 1 / 12
                try:
                    d = c.recv(4096)
                    if not d:
                        break
                    take(d)
                except socket.timeout:
                    pass
        except OSError:
            pass
        finally:
            try:                                 # read what the client sent last (stop-stream)
                c.settimeout(0.5)
                while d := c.recv(4096):
                    take(d)
            except OSError:
                pass
            c.close()

    def close(self):
        self.stop.set()
        self.srv.close()


def _udp_sockets(pgid: int) -> str:
    r = subprocess.run(["lsof", "-nP", "-a", "-g", str(pgid), "-i", "UDP"], capture_output=True, text=True)
    return r.stdout.strip()


@pytest.mark.skipif(os.environ.get("OPS_CAMERA") != "1", reason="opt-in: OPS_CAMERA=1 (about 15 s)")
def test_camera_stack_end_to_end_with_a_loopback_camera(tmp_path):
    import httpx
    from websockets.sync.client import connect

    def busy(port):
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) == 0
    for port in (E2E_BACKEND, E2E_RTSP):
        assert not busy(port), f"port {port} busy"
    cam = _LoopbackLeweiCamera(_h264_access_units(tmp_path))
    run_dir = tmp_path / "run"
    cmd = ["bash", str(SCRIPT), "--mode", "camera", "--port", str(E2E_BACKEND), "--rtsp-port", str(E2E_RTSP),
           "--no-ui", "--no-agent", "--perception", "none", "--camera-wait", "30",
           "--bridge-extra", f"--cam-ip 127.0.0.1 --port {cam.port}", "--logs", str(run_dir)]
    out = open(tmp_path / "launcher.out", "w")
    proc = subprocess.Popen(cmd, cwd=TREE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    stack_log = run_dir / "stack.log"

    def wait(pred, timeout, what):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                v = pred()
                if v:
                    return v
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(0.3)
        raise AssertionError(f"timed out waiting for {what}; stack.log:\n" + stack_log.read_text())

    def state():
        return httpx.get(f"http://127.0.0.1:{E2E_BACKEND}/state", timeout=3).json()
    pgids = {}
    try:
        wait(lambda: "UP:" in stack_log.read_text(), 90, "the camera stack to come up")
        pgids = {m.group(1): int(m.group(2)) for m in
                 re.finditer(r"started (\w+) \(pid (\d+)\)", stack_log.read_text())}
        assert set(pgids) == {"mediamtx", "bridge", "backend"}, pgids
        st = wait(lambda: (s := state())["latency_ms"].get("frames_received", 0) > 24 and s, 30, "camera frames")
        flags = set(st["safety_flags"])
        assert st["flight_mode"] == "twin" and "camera_only" in flags, flags
        assert not flags & {"keepalive_stale", "rc_send_errors", "video_stale", "rc_tick_stale"}, flags
        with connect(f"ws://127.0.0.1:{E2E_BACKEND}/ws") as ws:
            ws.send(json.dumps({"type": "hb", "client": "ui"}))
            ws.send(json.dumps({"type": "cmd", "command": "takeoff"}))
            end = time.monotonic() + 10
            while state()["flight_state"] != "AIRBORNE" and time.monotonic() < end:
                ws.send(json.dumps({"type": "hb", "client": "ui"}))
                time.sleep(0.2)
            st = state()
        assert st["flight_state"] == "AIRBORNE", st["flight_state"]
        flags = set(st["safety_flags"])
        assert "camera_only" in flags and not flags & {"keepalive_stale", "rc_send_errors"}, flags
        for name, pgid in pgids.items():
            assert _udp_sockets(pgid) == "", f"{name} has UDP sockets:\n{_udp_sockets(pgid)}"
    finally:
        proc.send_signal(signal.SIGINT)                     # a terminal Ctrl-C
        try:
            proc.wait(timeout=60)
        finally:
            out.close()
            cam.close()
    log = stack_log.read_text()
    assert "stopped. logs:" in log
    for name, pgid in pgids.items():
        r = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
        assert r.stdout.strip() == "", f"{name} still running: {r.stdout}"
    assert not busy(E2E_BACKEND) and not busy(E2E_RTSP)
    assert cam.commands and set(cam.commands) <= {1, 2, 3}, cam.commands     # video commands only
    assert cam.commands[0] == 2 and 3 in cam.commands                         # start ... stop
    assert "publishing rtsp://127.0.0.1:8674/live/ch00_1" in (run_dir / "lewei_bridge.log").read_text()
    assert os.path.exists(run_dir / f"mediamtx_{E2E_RTSP}.yml")
