#!/usr/bin/env bash
# Start the patched turbodrone backend (safety governor installed) from this checkout or worktree.
#
#   scripts/run_backend.sh --mode twin [--port 8000] [--control-port 23458] [--local-control-port N]
#                          [--rtsp-url rtsp://127.0.0.1:8554/live/ch00_1] [--headless] [--host 127.0.0.1]
#                          [--logs DIR]   (default $MHD_LOGS: control-*.jsonl and flight_timer.json)
#   scripts/run_backend.sh --mode real [--port 8000]          # DRONE_IP=172.16.11.1, RTSP from the drone
#
# --mode is required (FLIGHT_MODE has no default). Twin mode talks only to loopback. --headless sets
# GOVERNOR_TWIN_HEADLESS=1 (skip UI-heartbeat checks); the governor refuses it in real mode.
# Binds 127.0.0.1 by default so nothing else on the network can command the drone. No --reload, ever.
# Control logs: $MHD_LOGS/control-*.jsonl; flight timer: $MHD_LOGS/flight_timer.json.
set -euo pipefail
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$TREE/env.sh"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

MODE="" PORT="" CONTROL_PORT="" LOCAL_CONTROL_PORT="" RTSP_URL="" HEADLESS=0 HOST="127.0.0.1" LOGS=""
LOG_LEVEL="${LOG_LEVEL:-INFO}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --control-port) CONTROL_PORT="$2"; shift 2 ;;
    --local-control-port) LOCAL_CONTROL_PORT="$2"; shift 2 ;;
    --rtsp-url) RTSP_URL="$2"; shift 2 ;;
    --governor-config) export GOVERNOR_CONFIG="$2"; shift 2 ;;
    --autopilot-config) export AUTOPILOT_CONFIG="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --logs) LOGS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_backend.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  twin)
    export FLIGHT_MODE=twin DRONE_IP=127.0.0.1
    export X69_LG_RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/live/ch00_1}"
    ;;
  real)
    if [[ "$HEADLESS" == 1 ]]; then echo "run_backend.sh: --headless is twin-only" >&2; exit 2; fi
    export FLIGHT_MODE=real DRONE_IP=172.16.11.1
    if [[ -n "$RTSP_URL" ]]; then export X69_LG_RTSP_URL="$RTSP_URL"; else unset X69_LG_RTSP_URL; fi
    ;;
  *) echo "run_backend.sh: --mode twin|real is required" >&2; usage >&2; exit 2 ;;
esac

export DRONE_TYPE=x69_lg X69_LG_VIDEO_MODE=rtsp PLUGINS_ENABLED=false LOG_LEVEL
# One HEVC decoder thread: frame threading adds ~0.7 s of latency (measured by the twin workstream).
export OPENCV_FFMPEG_THREADS="${OPENCV_FFMPEG_THREADS:-1}"
export CONTROL_PORT="${CONTROL_PORT:-23458}"
[[ -n "$LOCAL_CONTROL_PORT" ]] && export X69_LG_LOCAL_CONTROL_PORT="$LOCAL_CONTROL_PORT"
if [[ "$HEADLESS" == 1 ]]; then export GOVERNOR_TWIN_HEADLESS=1; else unset GOVERNOR_TWIN_HEADLESS; fi
PORT="${PORT:-8000}"
if [[ -n "$LOGS" ]]; then mkdir -p "$LOGS"; export MHD_LOGS="$(cd "$LOGS" && pwd)"; fi

echo "[run_backend] tree=$TREE logs=$MHD_LOGS mode=$FLIGHT_MODE drone=$DRONE_IP:$CONTROL_PORT rtsp=${X69_LG_RTSP_URL:-<drone default>} http=$HOST:$PORT headless=$HEADLESS" >&2
cd "$TREE/turbodrone/backend"
exec "$MHD_PY" -m uvicorn web_server:app --host "$HOST" --port "$PORT" --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"
