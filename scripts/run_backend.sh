#!/usr/bin/env bash
# Start the patched turbodrone backend (safety governor installed) from this checkout or worktree.
#
#   scripts/run_backend.sh --mode twin [--port 8000] [--control-port 23458] [--local-control-port N]
#                          [--rtsp-url rtsp://127.0.0.1:8554/live/ch00_1] [--headless] [--host 127.0.0.1]
#                          [--logs DIR]   (default $MHD_LOGS: control-*.jsonl and flight_timer.json)
#   scripts/run_backend.sh --mode real [--port 8000]          # DRONE_IP=172.16.11.1, RTSP from the drone
#   [--governor-config FILE] [--autopilot-config FILE]   twin: any file (default config/*.yaml);
#       real: only config/{governor,autopilot}.yaml or config/real/ (Session B values), never a
#       twin_only file, and GOVERNOR_CONFIG / AUTOPILOT_CONFIG from the shell are ignored.
#   --dry-run: check the arguments, print what would run, and exit (contacts nothing).
#
# --mode is required (FLIGHT_MODE has no default). Twin mode talks only to loopback. --headless sets
# GOVERNOR_TWIN_HEADLESS=1 (skip UI-heartbeat checks); the governor refuses it in real mode.
# Binds 127.0.0.1 by default so nothing else on the network can command the drone. No --reload, ever.
# Control logs: $MHD_LOGS/control-*.jsonl; flight timer: $MHD_LOGS/flight_timer.json.
set -euo pipefail
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$TREE/env.sh"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

MODE="" PORT="" CONTROL_PORT="" LOCAL_CONTROL_PORT="" RTSP_URL="" HEADLESS=0 HOST="127.0.0.1" LOGS=""
GOV_CFG_ARG="" AP_CFG_ARG="" DRY_RUN=0
LOG_LEVEL="${LOG_LEVEL:-INFO}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --control-port) CONTROL_PORT="$2"; shift 2 ;;
    --local-control-port) LOCAL_CONTROL_PORT="$2"; shift 2 ;;
    --rtsp-url) RTSP_URL="$2"; shift 2 ;;
    --governor-config) GOV_CFG_ARG="$2"; shift 2 ;;
    --autopilot-config) AP_CFG_ARG="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --logs) LOGS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_backend.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Real mode: the governor/autopilot configs may only come from config/ or config/real/ (the values
# Session A/B measured), and never from a file marked twin_only (config/twin/). The Governor refuses
# twin_only files in real mode as well; this is the earlier, clearer error.
real_config() {   # real_config governor|autopilot FILE -> prints the absolute path or exits
  local kind="$1" f="$2" dir tree_p
  [[ "$f" = /* ]] || f="$TREE/$f"           # relative paths are relative to the tree, as in the loader
  [[ -f "$f" ]] || { echo "run_backend.sh: --$kind-config $f: no such file" >&2; exit 2; }
  dir="$(cd "$(dirname "$f")" && pwd -P)"; f="$dir/$(basename "$f")"
  tree_p="$(cd "$TREE" && pwd -P)"
  case "$f" in
    "$tree_p/config/$kind.yaml"|"$tree_p/config/real/$kind.yaml") ;;
    *) echo "run_backend.sh: real mode takes --$kind-config only from config/ or config/real/, got $f" >&2; exit 2 ;;
  esac
  if grep -qE '^twin_only:[[:space:]]*true' "$f"; then
    echo "run_backend.sh: $f is a twin-only config (twin_only: true): refusing real mode" >&2; exit 2
  fi
  echo "$f"
}

case "$MODE" in
  twin)
    export FLIGHT_MODE=twin DRONE_IP=127.0.0.1
    export X69_LG_RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/live/ch00_1}"
    [[ -n "$GOV_CFG_ARG" ]] && export GOVERNOR_CONFIG="$GOV_CFG_ARG"
    [[ -n "$AP_CFG_ARG" ]] && export AUTOPILOT_CONFIG="$AP_CFG_ARG"
    ;;
  real)
    if [[ "$HEADLESS" == 1 ]]; then echo "run_backend.sh: --headless is twin-only" >&2; exit 2; fi
    export FLIGHT_MODE=real DRONE_IP=172.16.11.1
    if [[ -n "$RTSP_URL" ]]; then export X69_LG_RTSP_URL="$RTSP_URL"; else unset X69_LG_RTSP_URL; fi
    if [[ -n "${GOVERNOR_CONFIG:-}${AUTOPILOT_CONFIG:-}" ]]; then
      echo "run_backend.sh: real mode ignores GOVERNOR_CONFIG/AUTOPILOT_CONFIG from the shell" >&2
    fi
    unset GOVERNOR_CONFIG AUTOPILOT_CONFIG
    if [[ -n "$GOV_CFG_ARG" ]]; then GOVERNOR_CONFIG="$(real_config governor "$GOV_CFG_ARG")"; export GOVERNOR_CONFIG; fi
    if [[ -n "$AP_CFG_ARG" ]]; then AUTOPILOT_CONFIG="$(real_config autopilot "$AP_CFG_ARG")"; export AUTOPILOT_CONFIG; fi
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
echo "[run_backend] governor config=${GOVERNOR_CONFIG:-config/governor.yaml} autopilot config=${AUTOPILOT_CONFIG:-config/autopilot.yaml}" >&2
if [[ "$DRY_RUN" == 1 ]]; then echo "[run_backend] dry run: not starting" >&2; exit 0; fi
cd "$TREE/turbodrone/backend"
# --timeout-graceful-shutdown 1: on Ctrl-C the governor starts landing at once (signal hook in
# hub.py); uvicorn then waits at most 1 s for open responses (the UI's /mjpeg) before the lifespan
# shutdown, which keeps sending land until 0.5 s have passed and then stops the RC loop.
exec "$MHD_PY" -m uvicorn web_server:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 1 \
  --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

