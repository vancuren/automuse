"""scripts/run_backend.sh: real mode takes configs only from config/ or config/real/, never a twin_only
file, and ignores GOVERNOR_CONFIG / AUTOPILOT_CONFIG exported in the shell (review, 2026-11-04).
All runs use --dry-run, which checks the arguments and exits before anything starts, except the busy-port
test, which runs a copy of the script whose MHD_PY is a stub that only records its arguments."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from contextlib import contextmanager

from conftest import ROOT

SCRIPT = ROOT / "scripts" / "run_backend.sh"


def dry(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("GOVERNOR_CONFIG", "AUTOPILOT_CONFIG")}
    env.update(env_extra or {})
    p = subprocess.run(["bash", str(SCRIPT), *args, "--dry-run"], capture_output=True, text=True,
                       env=env, timeout=30)
    return p.returncode, p.stderr


def test_real_mode_refuses_twin_config_files():
    for flag, name in (("--governor-config", "governor"), ("--autopilot-config", "autopilot")):
        rc, err = dry("--mode", "real", flag, f"config/twin/{name}.yaml")
        assert rc == 2 and "only from config/ or config/real/" in err


def test_real_mode_ignores_config_env_vars_from_the_shell():
    rc, err = dry("--mode", "real", env_extra={"GOVERNOR_CONFIG": str(ROOT / "config/twin/governor.yaml"),
                                               "AUTOPILOT_CONFIG": str(ROOT / "config/twin/autopilot.yaml")})
    assert rc == 0, err
    assert "ignores GOVERNOR_CONFIG/AUTOPILOT_CONFIG" in err
    assert "governor config=config/governor.yaml autopilot config=config/autopilot.yaml" in err


def test_real_mode_accepts_the_base_configs():
    rc, err = dry("--mode", "real", "--governor-config", "config/governor.yaml",
                  "--autopilot-config", str(ROOT / "config" / "autopilot.yaml"))
    assert rc == 0, err
    assert "dry run: not starting" in err


def test_real_mode_refuses_a_missing_file():
    rc, err = dry("--mode", "real", "--governor-config", "config/real/nope.yaml")
    assert rc == 2 and "no such file" in err


def test_twin_mode_takes_the_twin_configs():
    rc, err = dry("--mode", "twin", "--governor-config", "config/twin/governor.yaml",
                  "--autopilot-config", "config/twin/autopilot.yaml")
    assert rc == 0, err
    assert "governor config=config/twin/governor.yaml" in err


# ---------------------------------------------------------------- --camera (real camera, NO flight)

def test_camera_runs_twin_mode_with_camera_only_and_the_camera_configs():
    rc, err = dry("--camera")
    assert rc == 0, err
    assert "mode=twin board=x69_lg drone=127.0.0.1:23999 rtsp=rtsp://127.0.0.1:8554/live/ch00_1" in err
    assert "camera_only=1" in err and "CAMERA ONLY (real camera, NO FLIGHT)" in err
    assert "governor config=config/camera/governor.yaml autopilot config=config/camera/autopilot.yaml" in err
    assert "dry run: not starting" in err


def test_camera_takes_an_rtsp_url_and_explicit_configs_but_not_the_shell_ones():
    env = {"GOVERNOR_CONFIG": str(ROOT / "config/governor.yaml"), "AUTOPILOT_CONFIG": str(ROOT / "config/autopilot.yaml"),
           "CONTROL_PORT": "23458"}
    rc, err = dry("--camera", "--rtsp-url", "rtsp://127.0.0.1:8650/live/ch00_1", env_extra=env)
    assert rc == 0, err
    assert "rtsp=rtsp://127.0.0.1:8650/live/ch00_1" in err and "drone=127.0.0.1:23999" in err
    assert "governor config=config/camera/governor.yaml autopilot config=config/camera/autopilot.yaml" in err
    rc, err = dry("--mode", "twin", "--camera", "--governor-config", "config/twin/governor.yaml")
    assert rc == 0, err
    assert "governor config=config/twin/governor.yaml autopilot config=config/camera/autopilot.yaml" in err


def test_camera_refuses_real_mode_and_rc_ports():
    for args in (("--camera", "--mode", "real"), ("--mode", "real", "--camera")):
        rc, err = dry(*args)
        assert rc == 2 and "refusing --mode real" in err, err
    for flag in ("--control-port", "--local-control-port"):
        rc, err = dry("--camera", flag, "23458")
        assert rc == 2 and "sends no RC packets" in err, err


def test_camera_only_from_the_shell_never_reaches_real_or_plain_twin_mode():
    rc, err = dry("--mode", "real", env_extra={"MHD_CAMERA_ONLY": "1"})
    assert rc == 2 and "MHD_CAMERA_ONLY=1 is set in the shell: refusing --mode real" in err
    rc, err = dry("--mode", "twin", env_extra={"MHD_CAMERA_ONLY": "1"})
    assert rc == 0 and "camera_only=0" in err          # only --camera turns it on


# ---------------------------------------------------------------- --board lewei (our X69 unit)

def test_real_lewei_targets_the_lewei_board_and_the_loopback_bridge():
    rc, err = dry("--mode", "real", "--board", "lewei")
    assert rc == 0, err
    assert ("mode=real board=lewei drone=192.168.0.1:50000 rtsp=rtsp://127.0.0.1:8554/live/ch00_1" in err
            and "camera_only=0" in err)
    assert "governor config=config/governor.yaml autopilot config=config/autopilot.yaml" in err
    rc, err = dry("--mode", "real", "--board", "lewei", "--rtsp-url", "rtsp://127.0.0.1:8650/live/ch00_1",
                  "--governor-config", "config/governor.yaml")
    assert rc == 0 and "rtsp=rtsp://127.0.0.1:8650/live/ch00_1" in err and "config/governor.yaml" in err
    rc, err = dry("--mode", "real")                                     # the default board is unchanged
    assert rc == 0 and "board=x69_lg drone=172.16.11.1:23458" in err


def test_lewei_refusals():
    cases = [(("--mode", "twin", "--board", "lewei"), "only with --mode real"),
             (("--camera", "--board", "lewei"), "only with --mode real"),
             (("--board", "lewei"), "only with --mode real"),
             (("--mode", "real", "--board", "bogus"), "--board x69_lg|lewei"),
             (("--mode", "real", "--board", "lewei", "--control-port", "23458"), "CONTROL_PORT 50000"),
             (("--mode", "real", "--board", "lewei", "--rtsp-url", "rtsp://192.168.0.1/live/ch00_1"), "loopback bridge"),
             (("--mode", "real", "--board", "lewei", "--governor-config", "config/twin/governor.yaml"),
              "only from config/ or config/real/"),
             (("--mode", "real", "--board", "lewei", "--headless"), "twin-only")]
    for args, msg in cases:
        rc, err = dry(*args)
        assert rc == 2 and msg in err, (args, err)
    rc, err = dry("--mode", "real", env_extra={"MHD_BOARD": "lewei"})     # a stray shell value never wins
    assert rc == 2 and "pass --board lewei" in err
    rc, err = dry("--mode", "real", "--board", "lewei", env_extra={"MHD_CAMERA_ONLY": "1"})
    assert rc == 2 and "MHD_CAMERA_ONLY" in err


# ---------------------------------------------------------------- a busy --port (a second backend)
# uvicorn runs the app's startup (the RC loop starts sending) before it binds the port, so a second backend
# on a busy port would send RC packets until its bind failed: run_backend.sh refuses first (lsof).

LIVE_STACK_PORTS = (8000, 5173, 8554, 8766)


@contextmanager
def listening_port():
    """A TCP listener on a loopback port the kernel picks (ephemeral: never one of the live stack's ports)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert port not in LIVE_STACK_PORTS
        yield port
    finally:
        s.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port not in LIVE_STACK_PORTS
    return port


def test_dry_run_only_notes_a_busy_port():
    with listening_port() as port:
        for args in (("--mode", "twin"), ("--mode", "real", "--board", "lewei"), ("--camera",)):
            rc, err = dry(*args, "--port", str(port))
            assert rc == 0, (args, err)
            assert f"note: port {port} is already in use, so a real start would refuse" in err, err
            assert f"{os.getpid()}" in err and f"127.0.0.1:{port} (LISTEN)" in err      # lsof names the listener
            assert "dry run: not starting" in err
    rc, err = dry("--mode", "twin", "--port", str(free_port()))
    assert rc == 0 and "already in use" not in err and "dry run: not starting" in err


def hermetic_copy(tmp_path):
    """run_backend.sh in a scratch tree whose env.sh points MHD_PY at a stub that records its arguments, so
    a real (not dry-run) start can be checked without starting anything."""
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "turbodrone" / "backend").mkdir(parents=True)
    shutil.copy2(SCRIPT, tree / "scripts" / "run_backend.sh")
    marker = tmp_path / "started.txt"
    stub = tmp_path / "python-stub"
    stub.write_text(f'#!/bin/bash\necho "$@" > "{marker}"\n')
    stub.chmod(0o755)
    (tree / "env.sh").write_text(f'export MHD_PY="{stub}" MHD_LOGS="{tmp_path / "logs"}"\n')
    env = {k: v for k, v in os.environ.items()
           if k not in ("GOVERNOR_CONFIG", "AUTOPILOT_CONFIG", "MHD_BOARD", "MHD_CAMERA_ONLY", "LOG_LEVEL")}
    env["MHD_PY"] = str(stub)
    return tree / "scripts" / "run_backend.sh", marker, env


def test_a_real_start_refuses_a_busy_port_before_anything_runs(tmp_path):
    script, marker, env = hermetic_copy(tmp_path)
    args = ["--mode", "twin", "--control-port", "23999", "--rtsp-url", "rtsp://127.0.0.1:9/live/ch00_1"]
    with listening_port() as port:
        p = subprocess.run(["bash", str(script), *args, "--port", str(port)], capture_output=True, text=True,
                           env=env, timeout=30)
    assert p.returncode == 2, p.stderr
    assert f"port {port} is already in use" in p.stderr and f"127.0.0.1:{port} (LISTEN)" in p.stderr
    assert "refusing to start a second backend" in p.stderr
    assert not marker.exists()                         # uvicorn (here the stub) never ran
    port = free_port()                                 # the same start on a free port gets to uvicorn
    p = subprocess.run(["bash", str(script), *args, "--port", str(port)], capture_output=True, text=True,
                       env=env, timeout=30)
    assert p.returncode == 0, p.stderr
    assert marker.read_text().startswith(f"-m uvicorn web_server:app --host 127.0.0.1 --port {port} ")
    assert "already in use" not in p.stderr
