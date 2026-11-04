"""Promote an intermediate mlx_lm checkpoint to the adapter the server loads.

mlx_lm writes NNNNNNN_adapters.safetensors every --save-every iterations and adapters.safetensors only at
the end. If a run is stopped early (SIGINT), this copies the chosen checkpoint to adapters.safetensors
and records the choice in train_summary.json.

  $MHD_ML_PY training/finalize_checkpoint.py --adapter-dir $MHD_MODELS/adapters/glimmer-intent-v1 --iteration 2000
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--reason", default="stopped early: validation loss had plateaued and the run was behind schedule")
    a = ap.parse_args()
    ckpt = a.adapter_dir / f"{a.iteration:07d}_adapters.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint {ckpt}")
    shutil.copyfile(ckpt, a.adapter_dir / "adapters.safetensors")
    summary_path = a.adapter_dir / "train_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"ok": True, "final_iteration": a.iteration, "finalized_from": ckpt.name,
                    "finalize_reason": a.reason, "finalized": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
