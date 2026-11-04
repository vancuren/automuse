"""Process helpers for the twin's process-level tests (twin server, renderer, upstream backend).

Every process is started in its own session so teardown can kill its whole group
(mediamtx and ffmpeg are children of the twin server). Ports are the twin workstream's:
HTTP 8266, RTSP 8254, UDP RC 23520-23529, backend 8200-8209.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

TREE = Path(__file__).resolve().parents[2]
PY = os.environ.get("MHD_PY", sys.executable)
LOGS = Path(os.environ.get("MHD_LOGS", TREE / "logs")) / "tests"
HTTP_PORT = 8266
RTSP_PORT = 8254
BACKEND_PORT = 8200


def port_busy(port: int, udp: bool = False) -> bool:
    """True if something is bound there. SO_REUSEADDR like uvicorn, so TIME_WAIT leftovers don't count."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM if udp else socket.SOCK_STREAM)
    if not udp:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def wait_ports_free(ports, timeout: float = 15.0) -> None:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if not any(port_busy(p) for p in ports):
            return
        time.sleep(0.2)
    raise RuntimeError(f"ports still busy: {[p for p in ports if port_busy(p)]}")


class Proc:
    def __init__(self, name: str, cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log_path = LOGS / f"{name}-{time.strftime('%H%M%S')}.log"
        self.log = open(self.log_path, "wb")
        full_env = dict(os.environ)
        full_env.update(env or {})
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=full_env,
                                  cwd=str(cwd or TREE), start_new_session=True)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.p.pid, signal.SIGKILL)       # anything left in the group (ffmpeg, mediamtx, Chrome)
        except (ProcessLookupError, PermissionError):
            pass
        self.log.close()

    def tail(self, n: int = 30) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""


def wait_http(url: str, timeout: float = 20.0, proc: Proc | None = None) -> dict:
    t_end = time.monotonic() + timeout
    last = None
    while time.monotonic() < t_end:
        if proc is not None and not proc.alive():
            raise RuntimeError(f"{proc.name} exited:\n{proc.tail()}")
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(0.2)
    raise TimeoutError(f"{url} not ready: {last}")


def start_twin(scenario: str = "s1", rc_port: int = 23520, extra: list[str] | None = None) -> Proc:
    wait_ports_free([HTTP_PORT, RTSP_PORT])
    cmd = [PY, "-m", "mhd.twin.server", "--http-port", str(HTTP_PORT), "--rc-port", str(rc_port),
           "--rtsp-port", str(RTSP_PORT), "--scenario", scenario, "--log-dir", str(LOGS)] + (extra or [])
    p = Proc("twin-server", cmd, env={"FLIGHT_MODE": "twin", "DRONE_IP": "127.0.0.1"})
    wait_http(f"http://127.0.0.1:{HTTP_PORT}/status", proc=p)
    return p


def start_renderer(chase_rec: bool = False) -> Proc:
    cmd = [PY, "-m", "mhd.twin.launch_renderer", "--twin", f"http://127.0.0.1:{HTTP_PORT}"]
    if chase_rec:
        cmd.append("--chase-rec")
    return Proc("twin-renderer", cmd)


def start_backend(rc_port: int = 23520, port: int = BACKEND_PORT) -> Proc:
    wait_ports_free([port])
    cmd = ["bash", str(TREE / "scripts" / "run_upstream_backend_twin.sh"), "--port", str(port),
           "--rc-port", str(rc_port), "--rtsp-port", str(RTSP_PORT)]
    # In the integrated tree this is the governed fork of the backend. Twin-only headless mode lets
    # these wire tests run without a UI heartbeat; the governor refuses the flag in real mode.
    p = Proc("backend", cmd, env={"GOVERNOR_TWIN_HEADLESS": "1"})
    wait_http(f"http://127.0.0.1:{port}/capabilities", timeout=30, proc=p)
    return p


def twin_status() -> dict:
    return httpx.get(f"http://127.0.0.1:{HTTP_PORT}/status", timeout=3).json()


def twin_control(**kw) -> dict:
    r = httpx.post(f"http://127.0.0.1:{HTTP_PORT}/control", json=kw, timeout=5)
    r.raise_for_status()
    return r.json()


def wait_until(pred, timeout: float, interval: float = 0.1, what: str = "condition"):
    t_end = time.monotonic() + timeout
    last = None
    while time.monotonic() < t_end:
        last = pred()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")
