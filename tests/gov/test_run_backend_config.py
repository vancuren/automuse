"""scripts/run_backend.sh: real mode takes configs only from config/ or config/real/, never a twin_only
file, and ignores GOVERNOR_CONFIG / AUTOPILOT_CONFIG exported in the shell (review, 2026-11-04).
All runs use --dry-run, which checks the arguments and exits before anything starts."""
from __future__ import annotations

import os
import subprocess

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
