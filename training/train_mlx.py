"""Run mlx_lm's LoRA trainer in-process so we can record wall time and MLX peak memory.

Usage: $MHD_ML_PY training/train_mlx.py <mlx_lm lora arguments...>   (see training/train_mlx.sh)
Writes <adapter-path>/train_summary.json with the command line, wall time, and peak memory.
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import lora


def main() -> None:
    args = sys.argv[1:]
    t0 = time.time()
    sys.argv = ["mlx_lm.lora"] + args
    ok = False
    try:
        lora.main()
        ok = True
    finally:
        summary = {"ok": ok, "wall_s": round(time.time() - t0, 1), "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
                   "args": args, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        out = Path(args[args.index("--adapter-path") + 1]) if "--adapter-path" in args else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("TRAIN_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
