"""scripts/run_stack.sh --mode camera (REAL CAMERA, NO flight), checked with --dry-run: it would start
mediamtx and the Lewei bridge instead of the twin, the backend with --camera, perception without the
twin's counter mask and with the camera HFOV, and it refuses the oracle and the twin options.
--dry-run starts nothing (and writes only its stack.log and the backend's empty log dir). Ports are unusual ones so a
stack running on the defaults does not matter."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

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


def test_twin_dry_run_installs_nothing():
    """--dry-run used to run npm ci in twin/web when three.js was missing (found while adding camera mode)."""
    rc, out = run("--mode", "twin", "--dry-run", "--port", "8171", "--twin-ports", "8766,23458,8673",
                  "--no-ui", "--no-agent", "--no-perception", "--logs", str(TREE / "logs" / "ops-test" / "dry-twin"))
    try:
        assert rc == 0, out
        assert "npm install in twin/web" not in out
        assert list(started(out))[:1] == ["twin"]
    finally:
        shutil.rmtree(TREE / "logs" / "ops-test" / "dry-twin", ignore_errors=True)
