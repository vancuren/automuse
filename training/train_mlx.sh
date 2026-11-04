#!/usr/bin/env bash
# Local QLoRA fine-tune of Muse Glimmer 30B (MLX 4-bit) for intent parsing.
# Usage: training/train_mlx.sh [smoke|full]     (run from the repo root after `source env.sh`)
# Only one Glimmer process may run at a time (about 16-25 GB of unified memory).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MHD_ML_PY:?source env.sh first}"
MODE="${1:-full}"
MODEL="$MHD_MODELS/Muse-Glimmer-30B-4bit"
COMMON=(--model "$MODEL" --train --data training/data/mlx --mask-prompt --num-layers 16 --batch-size 1
        --grad-accumulation-steps 4 --max-seq-length 512 --grad-checkpoint --learning-rate 1e-4 --seed 0)
if [ "$MODE" = smoke ]; then
  exec "$MHD_ML_PY" training/train_mlx.py "${COMMON[@]}" --iters 20 --steps-per-report 5 --steps-per-eval 20 \
    --val-batches 8 --save-every 1000 --adapter-path logs/mlx/adapters-smoke -c training/mlx_lora.yaml
fi
OUT="${ADAPTER_OUT:-$MHD_MODELS/adapters/glimmer-intent-v1}"
mkdir -p "$OUT"
exec "$MHD_ML_PY" training/train_mlx.py "${COMMON[@]}" --iters 2600 --steps-per-report 50 --steps-per-eval 250 \
  --val-batches 25 --save-every 500 --adapter-path "$OUT" -c training/mlx_lora_full.yaml
