#!/usr/bin/env bash
# Edge mode: serve Muse Glimmer 30B (MLX 4-bit) plus the intent LoRA as an OpenAI-compatible API.
#
#   scripts/serve_glimmer.sh                 # http://127.0.0.1:8080/v1, adapter glimmer-intent-v1
#   scripts/serve_glimmer.sh --port 8580     # another port (the train workstream evaluates on 8580)
#   scripts/serve_glimmer.sh --base          # base Glimmer, no adapter
#   scripts/serve_glimmer.sh --adapter DIR   # a different adapter directory
#
# The agent calls POST /v1/chat/completions with the system prompt mhd.intent_spec.GLIMMER_SYSTEM_PROMPT
# (exactly as in training) and the transcript as the user message; mlx_lm splits Glimmer's reasoning
# channel into "reasoning" and puts the intent JSON in "content". Parse it with mhd.intent_spec.parse_intent.
# Do not pass chat_template_kwargs: the LoRA was trained with the template defaults.
#
# Memory: about 16-17 GB resident. Only one Glimmer process may run at a time on this Mac, so this script
# refuses to start if another mlx_lm Glimmer server or trainer is running. Stop it with Ctrl-C (or kill the
# PID it prints). Loopback only.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./env.sh

PORT=8080
ADAPTER="$MHD_MODELS/adapters/glimmer-intent-v1"
USE_ADAPTER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --base) USE_ADAPTER=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

MODEL="$MHD_MODELS/Muse-Glimmer-30B-4bit"
if [ ! -f "$MODEL/model-00004-of-00004.safetensors" ]; then
  echo "Glimmer weights missing at $MODEL (run scripts/fetch_models.sh)" >&2; exit 1
fi
if pgrep -f "mlx_lm.*Muse-Glimmer|train_mlx.py" >/dev/null 2>&1; then
  echo "another Glimmer process is already running (only one fits in memory):" >&2
  pgrep -fl "mlx_lm.*Muse-Glimmer|train_mlx.py" >&2 || true
  exit 1
fi
ARGS=(--model "$MODEL" --host 127.0.0.1 --port "$PORT" --temp 0 --max-tokens 200 --prompt-cache-size 4
      --log-level WARNING)
if [ "$USE_ADAPTER" = 1 ]; then
  if [ ! -f "$ADAPTER/adapters.safetensors" ]; then
    echo "adapter not found: $ADAPTER/adapters.safetensors (train with training/train_mlx.sh, or use --base)" >&2
    exit 1
  fi
  ARGS+=(--adapter-path "$ADAPTER")
fi
echo "Glimmer on http://127.0.0.1:$PORT/v1  adapter: $([ "$USE_ADAPTER" = 1 ] && echo "$ADAPTER" || echo none)  pid $$"
exec "$MHD_ML_PY" -m mlx_lm server "${ARGS[@]}"
