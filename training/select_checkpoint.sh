#!/usr/bin/env bash
# Model selection on the VALID set only: score each saved checkpoint with the real metrics, print a table.
#   training/select_checkpoint.sh 1000 1500 [2000 ...]
# Each checkpoint gets a temporary adapter dir under logs/mlx/ckpt-N (config + that checkpoint's weights).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./env.sh
SRC="$MHD_MODELS/adapters/glimmer-intent-v1"
for N in "$@"; do
  D="logs/mlx/ckpt-$N"
  mkdir -p "$D"
  cp "$SRC/adapter_config.json" "$D/"
  cp "$SRC/$(printf '%07d' "$N")_adapters.safetensors" "$D/adapters.safetensors"
  ADAPTER="$D" LABEL="select-ckpt-$N" training/eval_mlx.sh lora --sets valid --concurrency 4 --max-tokens 200 --fresh \
    --note "checkpoint selection on valid; MLX 4-bit, :8580"
done
"$MHD_PY" - "$@" <<'PY'
import json, sys
sys.path.insert(0, ".")
from training.eval import metrics
from training.common import read_jsonl
for n in sys.argv[1:]:
    m = metrics(read_jsonl(f"logs/eval/select-ckpt-{n}.jsonl"))
    print(f"ckpt {n}: valid JSON {m['valid_json_rate']:.3f}  action {m['action_accuracy']:.3f}  "
          f"subject {m['subject_normalized']:.3f}  clarify F1 {m['clarify']['f1']:.2f}  refuse F1 {m['refuse']['f1']:.2f}")
PY
