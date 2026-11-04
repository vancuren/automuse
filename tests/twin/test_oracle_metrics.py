"""Oracle perception + metrics on a live twin (no Chrome: the placeholder path still publishes Truth).

The truth-driven demo pilot flies S1 and follows the walker while the oracle logs SubjectObs;
metrics.py then joins its log with the twin's truth log by render id.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from mhd.twin import metrics
from mhd.twin.demo_pilot import DemoPilot

from procs import HTTP_PORT, PY, Proc, start_twin, twin_status, wait_until

RC_PORT = 23524
pytestmark = pytest.mark.slow


def test_oracle_follow_metrics(tmp_path):
    twin = start_twin("s1", rc_port=RC_PORT, extra=["--no-probe"])
    obs_log = tmp_path / "obs.jsonl"
    oracle = None
    pilot = None
    try:
        pilot = DemoPilot(f"http://127.0.0.1:{HTTP_PORT}", RC_PORT, "person in blue shirt")
        pilot.start(takeoff=True)
        wait_until(lambda: twin_status()["drone"]["phase"] == "flying", 10, what="flying")
        oracle = Proc("oracle", [PY, "-m", "mhd.twin.oracle", "--twin", f"http://127.0.0.1:{HTTP_PORT}",
                                 "--subject", "person in blue shirt", "--latency-ms", "250", "--dropout", "0.02",
                                 "--log", str(obs_log), "--duration-s", "14"])
        oracle.p.wait(timeout=30)
        log_dir = Path(twin_status()["log_dir"])
    finally:
        if pilot:
            pilot.stop()
        if oracle:
            oracle.stop()
        twin.stop()
    truth = metrics.load_jsonl(log_dir / "truth.jsonl")
    obs = metrics.load_jsonl(obs_log)
    assert len(obs) > 50
    rep = metrics.compute(truth, obs, subject="person in blue shirt")["report"]
    print(rep)
    assert rep["time_to_find_s"] == pytest.approx(2.0, abs=0.6)      # find delay 1.2 s + confirm 0.8 s
    assert rep["lock_rate"] > 0.85
    assert rep["id_switches"] == 0
    assert rep["perception_bearing_err_rms_rad"] < 0.02
    assert rep["bearing_rms_rad"] < 0.35                               # the demo pilot keeps it near center
