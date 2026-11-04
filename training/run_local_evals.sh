#!/usr/bin/env bash
# After training: evaluate Glimmer + LoRA and base Glimmer locally (one mlx_lm.server at a time on :8580),
# then rebuild docs/eval. Waits for training/train_mlx.py to exit first. Log: logs/mlx/evals.log
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./env.sh
while pgrep -f "train_mlx.py" >/dev/null 2>&1; do sleep 20; done
ADAPTER_DIR="${ADAPTER:-$MHD_MODELS/adapters/glimmer-intent-v1}"
if ! grep -q '"ok": true' "$ADAPTER_DIR/train_summary.json" 2>/dev/null; then
  echo "training did not finish cleanly; evaluating the newest checkpoint is up to a human" >&2; exit 1
fi
NOTE_BASE="MLX 4-bit on M5 Pro 48 GB, mlx_lm.server :8580, other workstreams may have shared the GPU (twin Chrome/WebGL renderer seen 00:25)"
echo "[$(date +%H:%M:%S)] lora quality run"
training/eval_mlx.sh lora --sets test,gold,gold_asr --concurrency 4 --max-tokens 200 --fresh \
  --note "$NOTE_BASE; client concurrency 4 (batched decode)"
echo "[$(date +%H:%M:%S)] lora latency run"
LABEL=glimmer-lora-mlx-seq training/eval_mlx.sh lora --sets gold --limit 40 --concurrency 1 --max-tokens 200 --fresh \
  --note "$NOTE_BASE; sequential latency run"
echo "[$(date +%H:%M:%S)] base quality run"
training/eval_mlx.sh base --sets test,gold,gold_asr --concurrency 8 --max-tokens 1500 --fresh \
  --note "$NOTE_BASE; client concurrency 8 (batched decode); template defaults, no schema"
echo "[$(date +%H:%M:%S)] base latency run"
LABEL=glimmer-base-mlx-seq training/eval_mlx.sh base --sets gold --limit 20 --concurrency 1 --max-tokens 1500 --fresh \
  --note "$NOTE_BASE; sequential latency run"
"$MHD_PY" training/eval.py report > /dev/null
echo "[$(date +%H:%M:%S)] all local evals done"
