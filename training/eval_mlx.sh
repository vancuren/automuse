#!/usr/bin/env bash
# Evaluate Glimmer locally: start mlx_lm.server on 127.0.0.1:8580, run training/eval.py, always stop the server.
#   training/eval_mlx.sh lora [eval.py run args...]   # adapter $ADAPTER (default glimmer-intent-v1)
#   training/eval_mlx.sh base [eval.py run args...]
# Labels: glimmer-lora-mlx / glimmer-base-mlx (override with LABEL=...). Only one Glimmer process at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./env.sh
MODE="${1:?lora or base}"; shift
PORT="${PORT:-8580}"
MODEL="$MHD_MODELS/Muse-Glimmer-30B-4bit"
if pgrep -f "mlx_lm.*Muse-Glimmer|train_mlx.py" >/dev/null 2>&1; then
  echo "another Glimmer process is running; refusing to start a second one" >&2; exit 1
fi
ARGS=(--model "$MODEL" --host 127.0.0.1 --port "$PORT" --temp 0 --max-tokens 400 --log-level WARNING)
if [ "$MODE" = lora ]; then
  ARGS+=(--adapter-path "${ADAPTER:-$MHD_MODELS/adapters/glimmer-intent-v1}")
  LABEL="${LABEL:-glimmer-lora-mlx}"
else
  LABEL="${LABEL:-glimmer-base-mlx}"
fi
mkdir -p logs/mlx
"$MHD_ML_PY" -m mlx_lm server "${ARGS[@]}" > "logs/mlx/server-$LABEL.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died; see logs/mlx/server-$LABEL.log" >&2; exit 1; }
  sleep 2
done
"$MHD_PY" training/eval.py run --backend mlx --url "http://127.0.0.1:$PORT/v1" --label "$LABEL" "$@"
