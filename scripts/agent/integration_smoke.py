"""Agent <-> real backend smoke test (twin mode, loopback only, no microphone, no speaker).

Starts the patched turbodrone backend (governor installed) from a checkout that has it, then replays
two `say` recordings through the agent with real Muse Voice Transcribe and Spark:
  1. "follow the person in the blue shirt"  -> POST /mission -> autopilot FIND, subject set
  2. "land now"                             -> safety listener -> POST /command land -> LANDING
and checks /state after each. RC datagrams go to an unused loopback UDP port; video is absent.

    cd $MHD_TREE && source env.sh
    $MHD_ML_PY scripts/agent/integration_smoke.py                        # backend from $MHD_TREE
    $MHD_ML_PY scripts/agent/integration_smoke.py --backend-tree ../gov  # before the merge

Costs about $0.003 (two short transcriptions and one Spark call). Exit code 0 = pass.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

TREE = Path(os.environ.get("MHD_TREE", Path(__file__).resolve().parents[2]))
HTTP_PORT, UDP_PORT, UDP_LOCAL = 8407, 8408, 8409


def start_backend(backend_tree: Path, logs: Path) -> subprocess.Popen:
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{backend_tree / 'src'}:{env.get('PYTHONPATH', '')}",
        "PYTHONPYCACHEPREFIX": str(TREE / "logs" / "pycache"),   # never write into another checkout
        "PYTHONDONTWRITEBYTECODE": "1",
        "MHD_LOGS": str(logs),
        "GOVERNOR_CONFIG": str(backend_tree / "config" / "governor.yaml"),
        "AUTOPILOT_CONFIG": str(backend_tree / "config" / "autopilot.yaml"),
        "FLIGHT_MODE": "twin", "DRONE_IP": "127.0.0.1", "DRONE_TYPE": "x69_lg",
        "X69_LG_VIDEO_MODE": "rtsp", "PLUGINS_ENABLED": "false",
        "CONTROL_PORT": str(UDP_PORT), "X69_LG_LOCAL_CONTROL_PORT": str(UDP_LOCAL),
        "X69_LG_RTSP_URL": "rtsp://127.0.0.1:8403/live/ch00_1",     # nothing there: no video
        "GOVERNOR_TWIN_HEADLESS": "1", "LOG_LEVEL": "WARNING",
    })
    out = open(logs / "backend.log", "w")
    p = subprocess.Popen([os.environ["MHD_PY"], "-m", "uvicorn", "web_server:app", "--host", "127.0.0.1",
                          "--port", str(HTTP_PORT), "--log-level", "warning"],
                         cwd=backend_tree / "turbodrone" / "backend", env=env, stdout=out, stderr=out)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20:
        try:
            if httpx.get(f"http://127.0.0.1:{HTTP_PORT}/state", timeout=1).status_code == 200:
                return p
        except Exception:
            pass
        time.sleep(0.3)
    p.terminate()
    raise RuntimeError(f"backend did not start; see {logs / 'backend.log'}")


def state() -> dict:
    return httpx.get(f"http://127.0.0.1:{HTTP_PORT}/state", timeout=2).json()


async def run_agent() -> list[str]:
    sys.path.insert(0, str(TREE / "src"))
    from mhd_ml.agent.service import AgentService, load_config

    cfg = load_config()
    cfg.update({"backend": f"ws://127.0.0.1:{HTTP_PORT}"})
    cfg.setdefault("tts", {})["mute"] = True
    svc = AgentService(cfg, start_mic=False)
    failures = []
    await svc.start()
    try:
        svc.speaker.wait_ready(30)
        audio = TREE / "media" / "test_audio" / "commands"
        await svc.simulate_file(audio / "c01_24k.wav")
        await svc.wait_idle()
        await asyncio.sleep(0.5)
        s = state()
        subject = ((s.get("mission") or {}).get("intent") or {}).get("subject_phrase")
        print("after follow:", s["flight_state"], s["autopilot_state"], subject)
        if s["autopilot_state"] not in ("FIND", "CONFIRM", "FOLLOW") or subject != "person in blue shirt":
            failures.append("follow mission did not start the autopilot")
        await svc.simulate_file(audio / "c07_24k.wav")
        await svc.wait_idle()
        await asyncio.sleep(0.5)
        s = state()
        print("after land:", s["flight_state"], s["autopilot_state"])
        if s["flight_state"] != "LANDING":
            failures.append("spoken land did not start a landing")
        if not any(h["result"].get("ok") and h["hit"]["command"] == "land" for h in svc.safety_hits):
            failures.append("safety listener did not deliver land")
    finally:
        await svc.stop()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend-tree", default=str(TREE), help="checkout with src/mhd/backend and the patched turbodrone")
    args = ap.parse_args()
    bt = Path(args.backend_tree).resolve()
    if not (bt / "src" / "mhd" / "backend").exists():
        print(f"no backend in {bt} (src/mhd/backend missing)")
        return 2
    from mhd_ml.agent.make_test_audio import generate

    generate()
    p = start_backend(bt, TREE / "logs" / "agent_integration")
    try:
        failures = asyncio.run(run_agent())
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    print("PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
