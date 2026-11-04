"""scripts/run_stack.sh --mode real --board lewei, checked with --dry-run only (nothing starts, nothing
is contacted): mediamtx and the Lewei bridge come first, then the real-mode backend aimed at the Lewei
board, perception with real-mode settings, the agent and the UI; every real-mode refusal still holds."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts" / "run_stack.sh"
PORTS = ["--port", "8174", "--ui-port", "8182", "--rtsp-port", "8676"]
URL = "rtsp://127.0.0.1:8676/live/ch00_1"


def run(*args) -> tuple[int, str]:
    p = subprocess.run(["bash", str(SCRIPT), *args], cwd=TREE, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def started(out: str) -> dict[str, str]:
    return dict(re.findall(r"DRY RUN: would start (\w+): (.*)", out))


def test_real_lewei_dry_run_starts_the_bridge_first_then_the_real_stack(tmp_path):
    run_dir = tmp_path / "run"
    rc, out = run("--mode", "real", "--board", "lewei", "--dry-run", *PORTS, "--logs", str(run_dir))
    assert rc == 0, out
    cmds = started(out)
    assert list(cmds) == ["mediamtx", "bridge", "backend", "perception", "agent"], out
    assert cmds["mediamtx"].endswith(f"tools/mediamtx/mediamtx {run_dir}/mediamtx_8676.yml")
    assert cmds["bridge"].endswith(f"scripts/lewei_bridge.py --rtsp-out {URL} --log-dir {run_dir}")
    b = cmds["backend"]
    assert b.startswith("bash scripts/run_backend.sh --mode real --port 8174 ")
    assert b.endswith(f"--board lewei --rtsp-url {URL}") and "--logs" not in b
    assert f"{TREE}/config/governor.yaml" in b and f"{TREE}/config/autopilot.yaml" in b
    assert "mode=real board=lewei drone=192.168.0.1:50000" in out and f"rtsp={URL}" in out
    p = cmds["perception"]
    assert "--hfov-deg 75.0" in p and "--mask-counter" not in p        # real-mode perception settings
    assert "would build the UI" in out and "DRY RUN done: nothing was started" in out
    assert "run_twin.sh" not in out and "REAL DRONE, Lewei board" in out
    assert sorted(x.name for x in run_dir.iterdir()) == ["stack.log"]


def test_real_lewei_keeps_every_real_mode_refusal():
    for extra, msg in ((["--headless"], "refuses --headless"),
                       (["--perception", "oracle"], "refuses --perception oracle"),
                       (["--scenario", "s1"], "refuses twin options"),
                       (["--config-dir", "config/twin"], "REAL mode refuses"),
                       (["--config-dir", "config/camera"], "REAL mode refuses"),
                       (["--bridge-extra", "--rtsp-out rtsp://127.0.0.1:9/x"], "may not change --rtsp-out")):
        rc, out = run("--mode", "real", "--board", "lewei", "--dry-run", *PORTS[:4], *extra)
        assert rc == 2 and msg in out, (extra, out)
        assert "would start" not in out


def test_board_lewei_only_in_real_mode_and_bridge_options_only_with_a_bridge():
    for mode in ("twin", "camera"):
        rc, out = run("--mode", mode, "--board", "lewei", "--dry-run", *PORTS[:4])
        assert rc == 2 and "only with --mode real" in out
    rc, out = run("--mode", "real", "--board", "nope", "--dry-run")
    assert rc == 2 and "--board x69_lg|lewei" in out
    rc, out = run("--mode", "real", "--rtsp-port", "8676", "--dry-run", *PORTS[:4])     # x69_lg: no bridge
    assert rc == 2 and "options of --mode camera and --mode real --board lewei" in out
