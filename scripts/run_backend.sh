#!/usr/bin/env bash
# Start the patched turbodrone backend (safety governor installed) from this checkout or worktree.
#
#   scripts/run_backend.sh --mode twin [--port 8000] [--control-port 23458] [--local-control-port N]
#                          [--rtsp-url rtsp://127.0.0.1:8554/live/ch00_1] [--headless] [--host 127.0.0.1]
#                          [--logs DIR]   (default $MHD_LOGS: control-*.jsonl and flight_timer.json)
#   scripts/run_backend.sh --mode real [--port 8000]          # DRONE_IP=172.16.11.1, RTSP from the drone
#   scripts/run_backend.sh --mode real --board lewei [--rtsp-url rtsp://127.0.0.1:8554/live/ch00_1]
#       our X69 unit's Lewei board: MHD_BOARD=lewei, DRONE_IP=192.168.0.1, CONTROL_PORT=50000 (raw 20-byte
#       payload), video from scripts/lewei_bridge.py on loopback. Refused with --mode twin / --camera.
#   scripts/run_backend.sh --camera [--rtsp-url rtsp://127.0.0.1:8554/live/ch00_1] [--port 8000]
#       REAL CAMERA, NO FLIGHT: twin mode (DRONE_IP=127.0.0.1) with MHD_CAMERA_ONLY=1, video from the
#       Lewei bridge on loopback, configs from config/camera/; the RC adapter has no socket and sends
#       nothing. Refuses --mode real, --control-port and --local-control-port.
#   [--governor-config FILE] [--autopilot-config FILE]   twin: any file (default config/*.yaml);
#       real: only config/{governor,autopilot}.yaml or config/real/ (Session B values), never a
#       twin_only file, and GOVERNOR_CONFIG / AUTOPILOT_CONFIG from the shell are ignored.
#   --dry-run: check the arguments, print what would run, and exit (contacts nothing; a busy --port is only noted).
#
# --mode (or --camera) is required (FLIGHT_MODE has no default). Twin mode talks only to loopback. --headless sets
# GOVERNOR_TWIN_HEADLESS=1 (skip UI-heartbeat checks); the governor refuses it in real mode.
# Binds 127.0.0.1 by default so nothing else on the network can command the drone. No --reload, ever.
# Refuses to start while anything listens on --port: uvicorn runs the app's startup (the RC loop starts
# sending) before it binds the port, so a second backend would send RC packets before it failed.
# Control logs: $MHD_LOGS/control-*.jsonl; flight timer: $MHD_LOGS/flight_timer.json.
set -euo pipefail
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$TREE/env.sh"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

# port_listeners PORT: one "COMMAND PID USER ... NAME" line per TCP listener on PORT (any address), or
# status 3 without lsof. lsof only reads the kernel's socket table: it connects to nothing.
port_listeners() {
  local lsof_bin
  lsof_bin="$(command -v lsof 2>/dev/null || true)"
  if [[ -z "$lsof_bin" && -x /usr/sbin/lsof ]]; then lsof_bin=/usr/sbin/lsof; fi
  [[ -n "$lsof_bin" ]] || return 3
  { "$lsof_bin" -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null || true; } | sed 1d
}

MODE="" PORT="" CONTROL_PORT="" LOCAL_CONTROL_PORT="" RTSP_URL="" HEADLESS=0 HOST="127.0.0.1" LOGS=""
GOV_CFG_ARG="" AP_CFG_ARG="" DRY_RUN=0 CAMERA=0 BOARD=x69_lg
# Camera-only mode: no RC socket exists, so this port is never used. It is still not the twin's RC port
# (23458), so even a bug could not steer a twin that happens to run on this Mac.
CAMERA_CONTROL_PORT=23999
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
    --camera) CAMERA=1; shift ;;
    --board) BOARD="$2"; shift 2 ;;
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

case "$BOARD" in x69_lg|lewei) ;; *) echo "run_backend.sh: --board x69_lg|lewei, got '$BOARD'" >&2; exit 2 ;; esac
if [[ -n "${MHD_BOARD:-}" && "${MHD_BOARD}" != "$BOARD" ]]; then
  echo "run_backend.sh: MHD_BOARD=${MHD_BOARD} is set in the shell but the board is $BOARD: pass --board ${MHD_BOARD}" \
       "(or unset MHD_BOARD)" >&2; exit 2
fi
if [[ "$BOARD" == lewei && ( "$MODE" != real || "$CAMERA" == 1 ) ]]; then
  echo "run_backend.sh: --board lewei is the real drone's flight board: only with --mode real (not twin, not --camera)" >&2; exit 2
fi
if [[ "$CAMERA" == 1 ]]; then
  # Real camera, no flight: always twin mode. Refuse anything that hints at the real drone or an RC link.
  if [[ -n "$MODE" && "$MODE" != twin ]]; then
    echo "run_backend.sh: --camera (real camera, NO flight) runs in twin mode only; refusing --mode $MODE" >&2; exit 2
  fi
  if [[ -n "$CONTROL_PORT$LOCAL_CONTROL_PORT" ]]; then
    echo "run_backend.sh: --camera sends no RC packets; --control-port/--local-control-port do not apply" >&2; exit 2
  fi
  MODE=twin
fi

case "$MODE" in
  twin)
    export FLIGHT_MODE=twin DRONE_IP=127.0.0.1
    export X69_LG_RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/live/ch00_1}"
    if [[ "$CAMERA" == 1 ]]; then
      export MHD_CAMERA_ONLY=1 CONTROL_PORT="$CAMERA_CONTROL_PORT"
      unset X69_LG_LOCAL_CONTROL_PORT
      # config/camera/ unless given here (GOVERNOR_CONFIG / AUTOPILOT_CONFIG from the shell are ignored)
      export GOVERNOR_CONFIG="${GOV_CFG_ARG:-config/camera/governor.yaml}"
      export AUTOPILOT_CONFIG="${AP_CFG_ARG:-config/camera/autopilot.yaml}"
    else
      unset MHD_CAMERA_ONLY                  # only --camera turns it on
      [[ -n "$GOV_CFG_ARG" ]] && export GOVERNOR_CONFIG="$GOV_CFG_ARG"
      [[ -n "$AP_CFG_ARG" ]] && export AUTOPILOT_CONFIG="$AP_CFG_ARG"
    fi
    ;;
  real)
    if [[ "$HEADLESS" == 1 ]]; then echo "run_backend.sh: --headless is twin-only" >&2; exit 2; fi
    case "$(echo "${MHD_CAMERA_ONLY:-}" | tr '[:upper:]' '[:lower:]')" in
      ""|0|false|no|off) ;;
      *) echo "run_backend.sh: MHD_CAMERA_ONLY=${MHD_CAMERA_ONLY} is set in the shell: refusing --mode real" \
              "(camera-only runs use --camera)" >&2; exit 2 ;;
    esac
    if [[ "$BOARD" == lewei ]]; then
      # Our X69 unit's Lewei board: the raw payload to 192.168.0.1:50000, video from the loopback bridge.
      if [[ -n "$CONTROL_PORT" && "$CONTROL_PORT" != 50000 ]]; then
        echo "run_backend.sh: --board lewei uses CONTROL_PORT 50000, not $CONTROL_PORT" >&2; exit 2
      fi
      export FLIGHT_MODE=real MHD_BOARD=lewei DRONE_IP=192.168.0.1 CONTROL_PORT=50000
      export X69_LG_RTSP_URL="${RTSP_URL:-rtsp://127.0.0.1:8554/live/ch00_1}"
      case "$X69_LG_RTSP_URL" in
        rtsp://127.0.0.1:*|rtsp://127.0.0.1/*|rtsp://localhost:*|rtsp://localhost/*) ;;
        *) echo "run_backend.sh: --board lewei takes the video from the loopback bridge, got --rtsp-url $X69_LG_RTSP_URL" >&2; exit 2 ;;
      esac
    else
      export FLIGHT_MODE=real DRONE_IP=172.16.11.1
      unset MHD_BOARD
      if [[ -n "$RTSP_URL" ]]; then export X69_LG_RTSP_URL="$RTSP_URL"; else unset X69_LG_RTSP_URL; fi
    fi
    if [[ -n "${GOVERNOR_CONFIG:-}${AUTOPILOT_CONFIG:-}" ]]; then
      echo "run_backend.sh: real mode ignores GOVERNOR_CONFIG/AUTOPILOT_CONFIG from the shell" >&2
    fi
    unset GOVERNOR_CONFIG AUTOPILOT_CONFIG
    if [[ -n "$GOV_CFG_ARG" ]]; then GOVERNOR_CONFIG="$(real_config governor "$GOV_CFG_ARG")"; export GOVERNOR_CONFIG; fi
    if [[ -n "$AP_CFG_ARG" ]]; then AUTOPILOT_CONFIG="$(real_config autopilot "$AP_CFG_ARG")"; export AUTOPILOT_CONFIG; fi
    ;;
  *) echo "run_backend.sh: --mode twin|real (or --camera) is required" >&2; usage >&2; exit 2 ;;
esac

export DRONE_TYPE=x69_lg X69_LG_VIDEO_MODE=rtsp PLUGINS_ENABLED=false LOG_LEVEL
# One HEVC decoder thread: frame threading adds ~0.7 s of latency (measured by the twin workstream).
export OPENCV_FFMPEG_THREADS="${OPENCV_FFMPEG_THREADS:-1}"
export CONTROL_PORT="${CONTROL_PORT:-23458}"
[[ -n "$LOCAL_CONTROL_PORT" ]] && export X69_LG_LOCAL_CONTROL_PORT="$LOCAL_CONTROL_PORT"
if [[ "$HEADLESS" == 1 ]]; then export GOVERNOR_TWIN_HEADLESS=1; else unset GOVERNOR_TWIN_HEADLESS; fi
PORT="${PORT:-8000}"
if [[ -n "$LOGS" ]]; then mkdir -p "$LOGS"; export MHD_LOGS="$(cd "$LOGS" && pwd)"; fi

echo "[run_backend] tree=$TREE logs=$MHD_LOGS mode=$FLIGHT_MODE board=${MHD_BOARD:-x69_lg} drone=$DRONE_IP:$CONTROL_PORT rtsp=${X69_LG_RTSP_URL:-<drone default>} http=$HOST:$PORT headless=$HEADLESS camera_only=${MHD_CAMERA_ONLY:-0}" >&2
[[ "$CAMERA" == 1 ]] && echo "[run_backend] CAMERA ONLY (real camera, NO FLIGHT): MHD_CAMERA_ONLY=1, the RC adapter has no socket and sends nothing" >&2
echo "[run_backend] governor config=${GOVERNOR_CONFIG:-config/governor.yaml} autopilot config=${AUTOPILOT_CONFIG:-config/autopilot.yaml}" >&2
# A second backend must never send a single RC packet. uvicorn runs the app's startup (the RC loop starts
# sending to the drone) BEFORE it binds the HTTP port, so on a busy port a second backend would send packets
# alongside the first one until its bind failed. So: refuse here, before anything starts.
BUSY="" PORT_CHECKED=1
BUSY="$(port_listeners "$PORT")" || PORT_CHECKED=0
if [[ "$PORT_CHECKED" == 0 ]]; then
  echo "[run_backend] WARNING: lsof not found: cannot check that port $PORT is free" >&2
elif [[ -n "$BUSY" && "$DRY_RUN" == 1 ]]; then
  echo "[run_backend] note: port $PORT is already in use, so a real start would refuse:" >&2
  printf '%s\n' "$BUSY" | sed 's/^/[run_backend]     /' >&2
elif [[ -n "$BUSY" ]]; then
  echo "run_backend.sh: port $PORT is already in use:" >&2
  printf '%s\n' "$BUSY" | sed 's/^/    /' >&2
  echo "run_backend.sh: refusing to start a second backend (it would send RC packets to the drone before it found" \
       "the port taken). Stop the other one first (Ctrl-C in its window), or use another --port." >&2
  exit 2
fi
if [[ "$DRY_RUN" == 1 ]]; then echo "[run_backend] dry run: not starting" >&2; exit 0; fi
cd "$TREE/turbodrone/backend"
# --timeout-graceful-shutdown 1: on Ctrl-C the governor starts landing at once (signal hook in
# hub.py); uvicorn then waits at most 1 s for open responses (the UI's /mjpeg) before the lifespan
# shutdown, which keeps sending land until 0.5 s have passed and then stops the RC loop.
exec "$MHD_PY" -m uvicorn web_server:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 1 \
  --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

