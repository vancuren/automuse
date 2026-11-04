"""Per-tick cost of the governor while the autopilot follows a subject.

    $MHD_PY -m mhd.governor.bench [--ticks 5000] [--threads 0]

Drives Governor.tick on a fake clock with a fresh SubjectObs every tick (worst case: the
wrong-way check and the autopilot run every tick). Reports wall time (what the RC loop waits)
and per-thread CPU time (the governor's own compute). --threads N adds N busy Python threads
to show GIL contention, which is what the RC thread sees inside the backend.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
from pathlib import Path

from mhd.contracts import Box, Intent, Mission, SubjectObs

from .config import load_autopilot_config, load_governor_config
from .core import Governor


def _pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def run(ticks: int = 5000, threads: int = 0) -> dict:
    clock = {"t": 10_000.0}
    tmp = Path(tempfile.mkdtemp(prefix="mhd-bench-"))       # tick log on (realistic cost), removed after
    gov = Governor(load_governor_config(), load_autopilot_config(), "twin", clock=lambda: clock["t"],
                   log_dir=tmp, persist_timer=False, tick_log=True)
    stop = threading.Event()

    def busy():
        x = 0
        while not stop.is_set():
            x = (x + 1) % 1000

    workers = [threading.Thread(target=busy, daemon=True) for _ in range(threads)]
    for w in workers:
        w.start()

    def step(fid=None, cx=900.0):
        t = clock["t"]
        gov.heartbeat(t)
        gov.note_frame(t - 100.0, t)
        if fid is not None:
            gov.submit_obs(SubjectObs(mission_id="b", frame_id=fid, capture_ms=t - 150.0, state="confirmed",
                                      track_id=1, box=Box(x1=cx - 70, y1=200, x2=cx + 70, y2=560),
                                      confidence=0.9, tracker_score=0.9, confirm_age_ms=300.0,
                                      source="oracle"), receive_ms=t)
        clock["t"] += 40.0

    for _ in range(3):
        step()
        gov.tick(clock["t"])
    gov.command("takeoff")
    for _ in range(110):
        step()
        gov.tick(clock["t"])
    gov.set_mission(Mission(mission_id="b", intent=Intent(action="follow", subject_phrase="person",
                                                          category="person", color="none", distance="medium",
                                                          reason_code="none"),
                            transcript="b", model="bench", latency_ms=0.0))
    wall, cpu = [], []
    for i in range(ticks):
        step(fid=i + 1, cx=900.0 + 50.0 * ((i // 50) % 2))
        w0, c0 = time.perf_counter(), time.thread_time()
        gov.tick(clock["t"])
        cpu.append((time.thread_time() - c0) * 1e6)
        wall.append((time.perf_counter() - w0) * 1e6)
    stop.set()
    mode = gov.autopilot.mode.value
    gov.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"ticks": ticks, "threads": threads, "autopilot": mode,
            "wall_us": {"p50": round(_pct(wall, 0.5), 1), "p99": round(_pct(wall, 0.99), 1), "max": round(max(wall), 1)},
            "cpu_us": {"p50": round(_pct(cpu, 0.5), 1), "p99": round(_pct(cpu, 0.99), 1), "max": round(max(cpu), 1)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    print(run(a.ticks, a.threads))


if __name__ == "__main__":
    main()
