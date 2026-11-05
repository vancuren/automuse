#!/usr/bin/env bash
# One launcher for the whole stack, real drone, twin or real camera: backend (governor), perception, voice
# agent, the operator UI (production build + vite preview), optional local Glimmer, in twin mode the
# twin, and in camera mode mediamtx plus the Lewei camera bridge.
# Logs go to logs/stack-<time>-<mode>/. Ctrl-C stops everything: the backend first (it lands the drone
# at once on SIGINT), then the clients, the UI, and the twin (or the bridge and mediamtx) last (so the
# land packets reach it). More Ctrl-Cs are ignored while it stops; anything still running after its
# timeout gets TERM, then KILL. If the backend dies, press Enter in this window to restart it.
#
#   scripts/run_stack.sh --mode real [--config-dir config/real] [--glimmer local]
#   scripts/run_stack.sh --mode real --no-perception --no-agent          # Session A / B: backend + UI
#   scripts/run_stack.sh --mode real --board lewei [--config-dir config]  # our X69 unit (Lewei board)
#   scripts/run_stack.sh --mode twin [--scenario s1] [--latency-ms 500] [--perception oracle]
#   scripts/run_stack.sh --mode camera [--bridge-extra "--low-def"]       # REAL CAMERA, NO FLIGHT
#
# Options (defaults in brackets):
#   --mode real|twin|camera    required
#   --board x69_lg|lewei       real mode: the drone's board [x69_lg]. lewei: flight control is the raw X69
#                              payload to 192.168.0.1:50000 and the video comes from the Lewei bridge, so
#                              mediamtx and scripts/lewei_bridge.py start FIRST and stop LAST (the drone
#                              powers itself off a few minutes after its video stops)
#   --config-dir DIR           governor.yaml + autopilot.yaml [real: config, twin: config/twin,
#                              camera: config/camera]
#   --port N                   backend HTTP port [8000]
#   --ui-port N                UI preview port [5173]          --no-ui     no UI
#   --perception sam_mlx|sam_api|oracle|none   [sam_mlx]       --no-perception
#   --agent auto|glimmer|keyword|local|none    [auto]          --no-agent
#       auto: Voice Transcribe, then Glimmer and Spark raced (Glimmer wins only if valid within 2.5 s,
#             so with LOCAL Glimmer (~3.6 s) Spark usually answers); glimmer: Voice Transcribe, then the
#             tuned Glimmer only (needs --glimmer local; keyword fallback at 9 s): use it for the clip
#             where Glimmer must answer; keyword: no LLM; local: on-device ASR (edge mode)
#   --agent-extra "ARGS"       extra agent flags, e.g. "--no-mic --mute --tts none"
#   --glimmer local|none       local: start scripts/serve_glimmer.sh (Glimmer + LoRA on :8080, ~17 GB) [none]
#   --no-build                 serve the existing UI build (dist/) instead of rebuilding
#   --open                     open the UI in its own Chrome window once everything is up (./fly does)
#   --no-record                do not save flights for the UI's replay (default: every flight is saved
#                              to logs/flights/, about 40 MB per minute in the air)
#   --dry-run                  validate, print every command it would run, start nothing
#   --logs DIR                 [logs/stack-<time>]
#   twin only:
#   --twin-ports HTTP,RC,RTSP  [8766,23458,8554]    --scenario s1..s6 [s1]    --latency-ms N [500]
#   --headless                 no UI heartbeat needed (twin only)    --no-renderer   placeholder video
#   --chrome-profile DIR       renderer Chrome profile [$MHD_HOME/.cache/chrome-twin]
#   camera mode and real --board lewei (the drone's Lewei camera board feeds the pipeline):
#   --rtsp-port N              loopback mediamtx RTSP port [8554]
#   --bridge-extra "ARGS"      extra scripts/lewei_bridge.py flags, e.g. "--low-def"
#   --camera-wait S            seconds to wait for the bridge to publish [60]
#
# Real mode refuses --headless, twin configs, the oracle, and --twin-* options. It never starts the
# twin. Run scripts/preflight.py first (the X69 may allow only one RTSP client: the backend).
# Camera mode is twin mode with MHD_CAMERA_ONLY=1 (run_backend.sh --camera): it starts mediamtx on
# 127.0.0.1 and scripts/lewei_bridge.py (camera -> RTSP), never the twin; the backend's RC adapter has
# no socket, so no RC packet is sent anywhere. It refuses the oracle and the twin options.
set -o pipefail             # no -u: bash 3.2 (macOS) treats empty arrays as unbound
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$TREE/env.sh"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "run_stack: $*" >&2; exit 2; }

MODE="" CONFIG_DIR="" PORT=8000 UI_PORT=5173 UI=1 PERCEPTION=sam_mlx AGENT=auto AGENT_EXTRA="" GLIMMER=none
BUILD=1 LOGS="" TWIN_PORTS="" SCENARIO=s1 LATENCY=500 HEADLESS=0 RENDERER=1 CHROME_PROFILE="" TWIN_OPT=0 DRY=0
RTSP_PORT=8554 BRIDGE_EXTRA="" CAMERA_WAIT=60 CAMERA_OPT=0 BOARD=x69_lg OPEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --board) BOARD="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ui-port) UI_PORT="$2"; shift 2 ;;
    --no-ui) UI=0; shift ;;
    --open) OPEN=1; shift ;;
    --no-record) export MHD_RECORD=0; shift ;;
    --perception) PERCEPTION="$2"; shift 2 ;;
    --no-perception) PERCEPTION=none; shift ;;
    --agent) AGENT="$2"; shift 2 ;;
    --no-agent) AGENT=none; shift ;;
    --agent-extra) AGENT_EXTRA="$2"; shift 2 ;;
    --glimmer) GLIMMER="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --dry-run) DRY=1; shift ;;
    --logs) LOGS="$2"; shift 2 ;;
    --twin-ports) TWIN_PORTS="$2"; TWIN_OPT=1; shift 2 ;;
    --scenario) SCENARIO="$2"; TWIN_OPT=1; shift 2 ;;
    --latency-ms) LATENCY="$2"; TWIN_OPT=1; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --no-renderer) RENDERER=0; TWIN_OPT=1; shift ;;
    --chrome-profile) CHROME_PROFILE="$2"; TWIN_OPT=1; shift 2 ;;
    --rtsp-port) RTSP_PORT="$2"; CAMERA_OPT=1; shift 2 ;;
    --bridge-extra) BRIDGE_EXTRA="$2"; CAMERA_OPT=1; shift 2 ;;
    --camera-wait) CAMERA_WAIT="$2"; CAMERA_OPT=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

# ------------------------------------------------------------------------------- validation
case "$MODE" in real|twin|camera) ;; *) usage >&2; die "--mode real|twin|camera is required" ;; esac
case "$PERCEPTION" in sam_mlx|sam_api|oracle|none) ;; *) die "--perception sam_mlx|sam_api|oracle|none" ;; esac
case "$AGENT" in auto|glimmer|keyword|local|none) ;; *) die "--agent auto|glimmer|keyword|local|none" ;; esac
[[ "$AGENT" == glimmer && "$GLIMMER" != local ]] && die "--agent glimmer needs --glimmer local"
case "$GLIMMER" in local|none) ;; *) die "--glimmer local|none" ;; esac
case "$BOARD" in x69_lg|lewei) ;; *) die "--board x69_lg|lewei" ;; esac
[[ "$BOARD" == lewei && "$MODE" != real ]] && die "--board lewei is the real drone's flight board: only with --mode real"
BRIDGE_RUN=0                # mediamtx + the Lewei bridge: camera mode, and the real drone with the Lewei board
[[ "$MODE" == camera || "$BOARD" == lewei ]] && BRIDGE_RUN=1
if [[ "$BRIDGE_RUN" == 1 ]]; then
  if [[ "$MODE" == camera ]]; then
    [[ "$PERCEPTION" == oracle ]] && die "CAMERA mode refuses --perception oracle (twin ground truth; there is no twin)"
    [[ "$TWIN_OPT" == 1 ]] && die "CAMERA mode refuses twin options (--twin-ports, --scenario, --latency-ms, --no-renderer, --chrome-profile)"
  fi
  [[ "$RTSP_PORT" =~ ^[0-9]+$ ]] && (( RTSP_PORT >= 1024 && RTSP_PORT <= 65535 )) || die "--rtsp-port must be 1024..65535"
  [[ "$CAMERA_WAIT" =~ ^[0-9]+$ ]] && (( CAMERA_WAIT >= 1 )) || die "--camera-wait must be a whole number of seconds >= 1"
  case " $BRIDGE_EXTRA " in
    *" --rtsp-out"*|*" --no-rtsp"*) die "--bridge-extra may not change --rtsp-out or --no-rtsp (the backend reads that URL)" ;;
  esac
  MTX_BIN="$("$MHD_PY" -c "from mhd.twin.media import mediamtx_binary as b; print(b())" 2>/dev/null)" \
    || die "mediamtx not found under \$MHD_TOOLS/mediamtx (the twin's loopback RTSP server)"
  BRIDGE="$TREE/scripts/lewei_bridge.py"
  if [[ ! -f "$BRIDGE" ]]; then
    if [[ "$DRY" == 1 ]]; then echo "run_stack: WARNING: $BRIDGE is missing (needed for a real run)" >&2
    else die "$BRIDGE is missing: merge the Lewei camera bridge first"; fi
  fi
elif [[ "$CAMERA_OPT" == 1 ]]; then
  die "--rtsp-port, --bridge-extra and --camera-wait are options of --mode camera and --mode real --board lewei"
fi
if [[ -z "$CONFIG_DIR" ]]; then
  case "$MODE" in real) CONFIG_DIR="config" ;; twin) CONFIG_DIR="config/twin" ;; camera) CONFIG_DIR="config/camera" ;; esac
fi
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$TREE/$CONFIG_DIR"
GOV_CFG="$CONFIG_DIR/governor.yaml"; AP_CFG="$CONFIG_DIR/autopilot.yaml"
[[ -f "$GOV_CFG" && -f "$AP_CFG" ]] || die "missing $GOV_CFG or $AP_CFG"
if [[ "$MODE" == real ]]; then
  [[ "$HEADLESS" == 1 ]] && die "REAL mode refuses --headless (the UI heartbeat is a safety check)"
  [[ "$TWIN_OPT" == 1 ]] && die "REAL mode refuses twin options (--twin-ports, --scenario, --latency-ms, --no-renderer, --chrome-profile)"
  [[ "$PERCEPTION" == oracle ]] && die "REAL mode refuses --perception oracle (twin ground truth)"
  case "$(cd "$CONFIG_DIR" && pwd)/" in */twin/*) die "REAL mode refuses twin configs ($CONFIG_DIR)" ;; esac
  if grep -qE '^twin_only:[[:space:]]*true' "$GOV_CFG" "$AP_CFG" || \
     grep -qiE "TWIN calibration|Never use this file for the real drone|Never for the real drone" "$GOV_CFG" "$AP_CFG"; then
    die "REAL mode refuses $CONFIG_DIR: its YAML says it is a twin calibration"
  fi
  if pgrep -f "mhd\.twin\.server" >/dev/null 2>&1; then
    echo "run_stack: WARNING: a twin server is running; stop it for real flights (GPU/CPU load)" >&2
  fi
fi
for p in "$PORT" $([[ "$UI" == 1 ]] && echo "$UI_PORT") $([[ "$BRIDGE_RUN" == 1 ]] && echo "$RTSP_PORT"); do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN -t >/dev/null 2>&1; then die "port $p is already in use (another stack?)"; fi
done
HFOV="$("$MHD_PY" -c "from mhd.governor.config import load_autopilot_config as l; print(l('$AP_CFG').camera.hfov_deg)" 2>/dev/null)" \
  || die "cannot load $AP_CFG"
"$MHD_PY" -c "from mhd.governor.config import load_governor_config as l; l('$GOV_CFG').validate()" 2>/dev/null \
  || die "cannot load $GOV_CFG"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN="${LOGS:-$MHD_LOGS/stack-$STAMP-$MODE}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
STACK_LOG="$RUN/stack.log"
log() { echo "[stack $(date +%H:%M:%S)] $*" | tee -a "$STACK_LOG" >&2; }
# Flight replay: the backend saves every flight here (src/mhd/backend/recorder.py); one place for all modes.
export MHD_RECORD="${MHD_RECORD:-1}" MHD_FLIGHTS_DIR="${MHD_FLIGHTS_DIR:-$TREE/logs/flights}"

# ------------------------------------------------------------------------------- process table
# Each component runs in its OWN process group (perl setpgrp, SIGINT/SIGQUIT back to default), so a
# terminal Ctrl-C reaches only this script, which then stops them in a safe order. No bash job control
# (set -m): with a real terminal, bash 3.2 spun at 100% CPU in the trap handler (seen in testing).
NAMES=() PIDS=() CMDS=() DEAD=()
LAST_PID=""
spawn_pg() {                # spawn_pg LOGFILE CMD...  -> LAST_PID (= its process group id)
  local logf="$1"; shift
  (cd "$TREE" && exec perl -e '$SIG{INT} = "DEFAULT"; $SIG{QUIT} = "DEFAULT"; setpgrp(0, 0) or die "setpgrp: $!";
                               exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!"' -- "$@") </dev/null >>"$logf" 2>&1 &
  LAST_PID=$!
}
start() {                   # start NAME LOGFILE CMD...  (cwd $TREE)
  local name="$1" logf="$2"; shift 2
  if [[ "$DRY" == 1 ]]; then log "DRY RUN: would start $name: $*"; return 0; fi
  : >"$logf"
  spawn_pg "$logf" "$@"
  local pid=$LAST_PID
  NAMES+=("$name"); PIDS+=("$pid"); CMDS+=("$*"); DEAD+=(0)
  log "started $name (pid $pid): $*  -> $(basename "$logf")"
}
pid_of() { local i; for i in "${!NAMES[@]}"; do [[ "${NAMES[$i]}" == "$1" ]] && { echo "${PIDS[$i]}"; return; }; done; }
alive() {                   # running and not a zombie (kill -0 succeeds on an unreaped zombie)
  local st; st="$(ps -o stat= -p "$1" 2>/dev/null)"
  [[ -n "$st" && "$st" != Z* ]]
}
stop_group() {              # stop_group NAME SIGNAL WAIT_S
  local pid; pid="$(pid_of "$1")"; [[ -z "$pid" ]] && return 0
  alive "$pid" || return 0
  kill "-$2" -- "-$pid" 2>/dev/null || kill "-$2" "$pid" 2>/dev/null
  local n=0 max=$(( $3 * 10 ))
  while alive "$pid" && (( n < max )); do sleep 0.1; n=$((n + 1)); done
  if alive "$pid"; then
    kill -TERM -- "-$pid" 2>/dev/null; sleep 1
    alive "$pid" && kill -KILL -- "-$pid" 2>/dev/null
  fi
  return 0
}

STOPPING=0
cleanup() {
  [[ "$STOPPING" == 1 ]] && return
  STOPPING=1
  # From here on nothing may cut the ordered stop short: bash 3.2 abandons a SIGINT trap handler when
  # a foreground child (sleep) is interrupted, which once left the twin running. The backend is
  # already landing; stop_group escalates to TERM and KILL on its own timeouts.
  trap '' INT TERM
  log "stopping: backend first (it sends LAND on SIGINT), then the clients, the UI, the twin (camera: bridge, mediamtx) last"
  local b; b="$(pid_of backend)"
  [[ -n "$b" && "$BACKEND_SIGNALED" != 1 ]] && alive "$b" && kill -INT -- "-$b" 2>/dev/null
  local w=()
  stop_group agent INT 5 & w+=($!)
  stop_group perception INT 5 & w+=($!)
  stop_group glimmer INT 5 & w+=($!)
  stop_group backend INT 8
  local j; for j in "${w[@]}"; do wait "$j" 2>/dev/null; done
  stop_group ui INT 3
  stop_group renderer INT 6
  stop_group twin INT 10
  stop_group bridge INT 5                  # camera mode: sends stop-video to the camera, then exits
  stop_group mediamtx INT 3
  [[ -n "${CAFF:-}" ]] && kill "$CAFF" 2>/dev/null
  log "stopped. logs: $RUN"
  exit 0
}
# The signal handler only records the request and sends SIGINT to the backend's group (it lands at
# once). The ordered stop runs from the main loop: in bash 3.2, `read -t` arms SIGALRM and longjmps
# back into read when it fires, which abandoned a cleanup running inside the trap (seen in testing).
STOP_REQ=0 BACKEND_PGID="" BACKEND_SIGNALED=0
on_signal() {
  STOP_REQ=1
  if [[ -n "$BACKEND_PGID" && "$BACKEND_SIGNALED" != 1 ]]; then
    BACKEND_SIGNALED=1
    kill -INT -- "-$BACKEND_PGID" 2>/dev/null
  fi
}
check_stop() { [[ "$STOP_REQ" == 1 ]] && cleanup; return 0; }
trap on_signal INT TERM
trap '[[ "$STOPPING" == 1 ]] || cleanup' EXIT

CAFF=""
if [[ "$DRY" != 1 ]]; then caffeinate -dimsu -w $$ >/dev/null 2>&1 & CAFF=$!; fi

{
  echo "mode=$MODE board=$BOARD config=$CONFIG_DIR hfov=$HFOV port=$PORT ui=$UI:$UI_PORT perception=$PERCEPTION agent=$AGENT glimmer=$GLIMMER"
  echo "MHD_CAPTURE_OFFSET_MS=${MHD_CAPTURE_OFFSET_MS:-<unset>} tree=$TREE"
  git -C "$TREE" log -1 --format='commit %h %s' 2>/dev/null
} >> "$STACK_LOG"
log "mode $MODE, configs $CONFIG_DIR (HFOV $HFOV), logs $RUN"
[[ "$MODE" == real ]] && log "REAL DRONE: run scripts/preflight.py first; MHD_CAPTURE_OFFSET_MS=${MHD_CAPTURE_OFFSET_MS:-unset}"
[[ "$BOARD" == lewei ]] && log "REAL DRONE, Lewei board: flight control to 192.168.0.1:50000; video via scripts/lewei_bridge.py -> mediamtx on 127.0.0.1:$RTSP_PORT (started first, stopped last)"
[[ "$MODE" == camera ]] && log "REAL CAMERA, NO FLIGHT: Lewei camera -> scripts/lewei_bridge.py -> mediamtx on 127.0.0.1:$RTSP_PORT; the backend sends no RC packets (MHD_CAMERA_ONLY=1)"

wait_http() {               # wait_http URL SECONDS NAME
  local i
  [[ "$DRY" == 1 ]] && return 0
  for ((i = 0; i < $2 * 4; i++)); do
    check_stop
    curl -sf -o /dev/null "$1" && return 0
    local p; p="$(pid_of "$3")"
    [[ -n "$p" ]] && ! alive "$p" && { log "$3 exited while starting; see its log"; return 1; }
    sleep 0.25
  done
  log "timed out waiting for $1"; return 1
}
wait_listen() {             # wait_listen PORT SECONDS NAME: until something listens on 127.0.0.1:PORT
  [[ "$DRY" == 1 ]] && return 0
  local end=$((SECONDS + $2)) p
  while (( SECONDS < end )); do
    check_stop
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1 && return 0
    p="$(pid_of "$3")"
    [[ -n "$p" ]] && ! alive "$p" && { log "$3 exited while starting; see $RUN/$3.log"; return 1; }
    sleep 0.2
  done
  log "timed out waiting for $3 to listen on 127.0.0.1:$1"; return 1
}
wait_publisher() {          # wait_publisher URL SECONDS: until mediamtx has a publisher (or the bridge says so)
  [[ "$DRY" == 1 ]] && return 0
  local end=$((SECONDS + $2)) n p
  log "waiting up to $2 s for the Lewei bridge to publish $1"
  while (( SECONDS < end )); do
    check_stop
    if grep -qE '^\[lewei_bridge\] .*publishing rtsp://' "$RUN/bridge.log" 2>/dev/null || \
       "$MHD_PY" -c 'import sys; from mhd.twin.media import rtsp_has_publisher as f; sys.exit(0 if f(sys.argv[1]) else 1)' \
         "$1" >/dev/null 2>&1; then
      log "camera video is up: $1"
      return 0
    fi
    for n in mediamtx bridge; do
      p="$(pid_of "$n")"
      [[ -n "$p" ]] && ! alive "$p" && { log "$n exited while waiting for the camera; see $RUN/$n.log"; return 1; }
    done
    sleep 0.5
  done
  log "TIMED OUT after $2 s: the Lewei bridge did not publish $1. Is this Mac on the drone's Wi-Fi" \
      "(camera board at 192.168.0.1)? See $RUN/bridge.log (raise the wait with --camera-wait S)"
  return 1
}

# ------------------------------------------------------------------------------- twin
if [[ "$MODE" == twin ]]; then
  IFS=, read -r T_HTTP T_RC T_RTSP <<< "${TWIN_PORTS:-8766,23458,8554}"
  if [[ "$RENDERER" == 1 && ! -d "$TREE/twin/web/node_modules/three" && "$DRY" == 1 ]]; then
    log "DRY RUN: would npm ci in twin/web (first run: three.js for the renderer)"
  elif [[ "$RENDERER" == 1 && ! -d "$TREE/twin/web/node_modules/three" ]]; then
    log "npm install in twin/web (first run: three.js for the renderer)"
    (cd "$TREE/twin/web" && { [[ -f package-lock.json ]] && npm ci --no-audit --no-fund || npm install --no-audit --no-fund; }) \
      >"$RUN/twin-npm.log" 2>&1 || { log "twin/web npm failed: $RUN/twin-npm.log"; cleanup; }
  fi
  start twin "$RUN/twin.log" bash scripts/run_twin.sh --ports "$T_HTTP,$T_RC,$T_RTSP" --scenario "$SCENARIO" \
    --latency-ms "$LATENCY" --no-renderer -- --log-dir "$RUN/twin"
  wait_http "http://127.0.0.1:$T_HTTP/status" 30 twin || cleanup
  if [[ "$RENDERER" == 1 ]]; then
    start renderer "$RUN/renderer.log" "$MHD_PY" -m mhd.twin.launch_renderer --twin "http://127.0.0.1:$T_HTTP" \
      --user-data-dir "${CHROME_PROFILE:-$MHD_HOME/.cache/chrome-twin}"
  fi
  RTSP_URL="rtsp://127.0.0.1:$T_RTSP/live/ch00_1"
fi

# ------------------------------------------------------------------------------- Lewei camera bridge
# Camera mode (real camera, NO flight) and the real drone with the Lewei board. No twin and no renderer:
# mediamtx with the twin's loopback, TCP-only, RTSP-only config, then the Lewei bridge (camera board ->
# H.264 with wall-clock pts -> RTSP on loopback). The backend comes after it. The bridge runs for the
# whole session: the drone powers itself off a few minutes after its video stream stops.
if [[ "$BRIDGE_RUN" == 1 ]]; then
  RTSP_URL="rtsp://127.0.0.1:$RTSP_PORT/live/ch00_1"
  MTX_CFG="$RUN/mediamtx_$RTSP_PORT.yml"
  if [[ "$DRY" != 1 ]]; then
    "$MHD_PY" -c 'import sys; from pathlib import Path; from mhd.twin.media import write_mediamtx_config as w; w(Path(sys.argv[1]), int(sys.argv[2]))' \
      "$RUN" "$RTSP_PORT" || { log "cannot write $MTX_CFG"; cleanup; }
  fi
  start mediamtx "$RUN/mediamtx.log" "$MTX_BIN" "$MTX_CFG"
  wait_listen "$RTSP_PORT" 10 mediamtx || cleanup
  BR_ARGS=(--rtsp-out "$RTSP_URL" --log-dir "$RUN")
  # shellcheck disable=SC2206
  [[ -n "$BRIDGE_EXTRA" ]] && BR_ARGS+=($BRIDGE_EXTRA)
  start bridge "$RUN/bridge.log" "$MHD_PY" scripts/lewei_bridge.py "${BR_ARGS[@]}"
  wait_publisher "$RTSP_URL" "$CAMERA_WAIT" || cleanup
fi

check_stop
# ------------------------------------------------------------------------------- glimmer (optional)
if [[ "$GLIMMER" == local ]]; then
  start glimmer "$RUN/glimmer.log" bash scripts/serve_glimmer.sh --port 8080
fi

check_stop
# ------------------------------------------------------------------------------- backend
# Real mode: control logs and the per-pack flight timer stay in $MHD_LOGS (logs/), so a restarted
# backend keeps counting the same pack. Twin mode: the run dir, so twin flights never add pack time.
B_ARGS=(--mode "$MODE" --port "$PORT" --governor-config "$GOV_CFG" --autopilot-config "$AP_CFG")
if [[ "$MODE" == twin ]]; then
  B_ARGS+=(--logs "$RUN/backend" --control-port "$T_RC" --rtsp-url "$RTSP_URL")
  [[ "$HEADLESS" == 1 ]] && B_ARGS+=(--headless)
elif [[ "$MODE" == camera ]]; then      # twin mode + MHD_CAMERA_ONLY=1: no RC socket, nothing is sent
  B_ARGS=(--camera --port "$PORT" --governor-config "$GOV_CFG" --autopilot-config "$AP_CFG"
          --logs "$RUN/backend" --rtsp-url "$RTSP_URL")
  [[ "$HEADLESS" == 1 ]] && B_ARGS+=(--headless)
elif [[ "$BOARD" == lewei ]]; then      # real mode, Lewei board: 192.168.0.1:50000, video from the bridge
  B_ARGS+=(--board lewei --rtsp-url "$RTSP_URL")
fi
if [[ "$DRY" == 1 ]]; then              # the backend launcher's own checks (real: config/ or config/real/ only)
  (cd "$TREE" && bash scripts/run_backend.sh "${B_ARGS[@]}" --dry-run) 2>&1 | sed 's/^/    /' | tee -a "$STACK_LOG" >&2
  [[ "${PIPESTATUS[0]}" == 0 ]] || die "run_backend.sh --dry-run refused these arguments"
fi
start backend "$RUN/backend.log" bash scripts/run_backend.sh "${B_ARGS[@]}"
BACKEND_PGID="$(pid_of backend)"
wait_http "http://127.0.0.1:$PORT/state" 40 backend || cleanup
BACKEND_RESTART="scripts/run_backend.sh ${B_ARGS[*]}"

check_stop
# ------------------------------------------------------------------------------- perception
MASK=()                                  # camera: no twin counter block to mask in the real camera's image
[[ "$MODE" == camera ]] && MASK=(--mask-counter off)
case "$PERCEPTION" in
  sam_mlx|sam_api)
    start perception "$RUN/perception.log" "$MHD_ML_PY" -m mhd_ml.perception.service --backend "ws://127.0.0.1:$PORT" \
      --detector "$PERCEPTION" --tracker auto --hfov-deg "$HFOV" "${MASK[@]}" --obs-log "$RUN/obs.jsonl" --log-level INFO ;;
  oracle)
    start perception "$RUN/perception.log" "$MHD_PY" -m mhd.twin.oracle --twin "http://127.0.0.1:$T_HTTP" \
      --backend "ws://127.0.0.1:$PORT" --latency-ms 300 --log "$RUN/obs.jsonl" ;;
esac

check_stop
# ------------------------------------------------------------------------------- agent
if [[ "$AGENT" != none ]]; then
  A_ARGS=(--backend "ws://127.0.0.1:$PORT")
  case "$AGENT" in
    auto) A_ARGS+=(--asr meta --intent auto) ;;
    glimmer) A_ARGS+=(--asr meta --intent glimmer) ;;
    keyword) A_ARGS+=(--asr meta --intent keyword) ;;
    local) if [[ "$GLIMMER" == local ]]; then A_ARGS+=(--asr local --intent glimmer); else A_ARGS+=(--asr local --intent keyword); fi ;;
  esac
  [[ "$GLIMMER" == local ]] && A_ARGS+=(--glimmer-url http://127.0.0.1:8080/v1 --glimmer-model default_model --glimmer-label glimmer-30b+lora)
  # shellcheck disable=SC2206
  [[ -n "$AGENT_EXTRA" ]] && A_ARGS+=($AGENT_EXTRA)
  start agent "$RUN/agent.log" "$MHD_ML_PY" -m mhd_ml.agent.service "${A_ARGS[@]}"
fi

check_stop
# ------------------------------------------------------------------------------- UI (production build)
if [[ "$UI" == 1 && "$DRY" == 1 ]]; then
  log "DRY RUN: would build the UI (VITE_BACKEND_URL=http://localhost:$PORT VITE_AXES_RELEASE_MESSAGE=1) and serve it on 127.0.0.1:$UI_PORT"
  UI=0
fi
if [[ "$UI" == 1 ]]; then
  FE="$TREE/turbodrone/frontend"
  if [[ ! -d "$FE/node_modules" ]]; then
    log "npm ci in turbodrone/frontend (first run; cache in \$MHD_HOME/.cache/npm)"
    (cd "$FE" && npm ci --no-audit --no-fund) >"$RUN/ui-npm.log" 2>&1 || { log "npm ci failed: $RUN/ui-npm.log"; cleanup; }
  fi
  if [[ "$BUILD" == 1 || ! -d "$FE/dist" ]]; then
    log "building the UI for http://localhost:$PORT"
    CHASE=""; [[ "$MODE" == twin ]] && CHASE="http://localhost:$T_HTTP/chase"
    (cd "$FE" && env VITE_BACKEND_URL="http://localhost:$PORT" VITE_AXES_RELEASE_MESSAGE=1 \
      ${CHASE:+VITE_TWIN_CHASE_URL=$CHASE} npm run build) >"$RUN/ui-build.log" 2>&1 \
      || { log "UI build failed: $RUN/ui-build.log"; cleanup; }
  fi
  start ui "$RUN/ui.log" bash -c "cd '$FE' && exec npx vite preview --host 127.0.0.1 --port '$UI_PORT' --strictPort"
  wait_http "http://127.0.0.1:$UI_PORT/" 30 ui || log "UI preview did not answer; see $RUN/ui.log"
fi

check_stop
# ------------------------------------------------------------------------------- status and watch
if [[ "$DRY" == 1 ]]; then
  log "DRY RUN done: nothing was started"
  STOPPING=1; exit 0
fi
echo
log "UP: backend http://127.0.0.1:$PORT  $([[ "$UI" == 1 ]] && echo "UI http://127.0.0.1:$UI_PORT (open it in its own Chrome window)")"
[[ "$MODE" == twin ]] && log "twin http://127.0.0.1:$T_HTTP (chase: /chase)  RTSP $RTSP_URL"
[[ "$MODE" == camera ]] && log "REAL CAMERA, NO FLIGHT: RTSP $RTSP_URL from scripts/lewei_bridge.py; no RC packets are sent"
[[ "$BOARD" == lewei ]] && log "REAL DRONE, Lewei board: RC to 192.168.0.1:50000; video $RTSP_URL from scripts/lewei_bridge.py (keep it running)"
log "Ctrl-C stops everything (the backend lands first). Logs: $RUN"
if [[ "$OPEN" == 1 && "$UI" == 1 ]]; then
  # Its own window, so no other tab can take the keyboard from Esc / L / Space.
  open -na "Google Chrome" --args --new-window "http://127.0.0.1:$UI_PORT/" 2>/dev/null \
    || open "http://127.0.0.1:$UI_PORT/" 2>/dev/null || log "could not open a browser: open http://127.0.0.1:$UI_PORT/"
fi
while true; do
  check_stop
  for i in "${!NAMES[@]}"; do
    if [[ "${DEAD[$i]}" == 0 ]] && ! alive "${PIDS[$i]}"; then
      DEAD[$i]=1
      wait "${PIDS[$i]}" 2>/dev/null; code=$?
      name="${NAMES[$i]}"
      log "!!!!! $name EXITED (code $code). See $RUN/$name.log"
      if [[ "$name" == backend ]]; then
        log "!!!!! NO BACKEND: the drone gets no RC packets (Session A link-loss behavior applies)."
        if [[ -t 0 ]]; then
          log "!!!!! press Enter in THIS window to restart it (same settings; it comes back ON_GROUND, disarmed)"
        else
          log "!!!!! restart it in another window of this terminal app:  $BACKEND_RESTART"
        fi
      fi
    fi
  done
  if [[ -t 0 ]]; then
    if read -t 1 -r _line && [[ "$STOP_REQ" != 1 ]]; then
      bi=""; for i in "${!NAMES[@]}"; do [[ "${NAMES[$i]}" == backend ]] && bi=$i; done
      if [[ -n "$bi" && "${DEAD[$bi]}" == 1 ]]; then
        spawn_pg "$RUN/backend.log" bash scripts/run_backend.sh "${B_ARGS[@]}"
        PIDS[$bi]=$LAST_PID; DEAD[$bi]=0; BACKEND_PGID=$LAST_PID; BACKEND_SIGNALED=0
        log "restarted backend (pid ${PIDS[$bi]}); type ARM again in the UI"
      fi
    fi
  else
    sleep 1 &
    wait $! 2>/dev/null
  fi
done
