#!/usr/bin/env bash
# One launcher for the whole stack, real drone or twin: backend (governor), perception, voice agent,
# the operator UI (production build + vite preview), optional local Glimmer, and in twin mode the twin.
# Logs go to logs/stack-<time>/. Ctrl-C stops everything: the backend first (it lands the drone at
# once on SIGINT), then the clients, the UI, and the twin last (so the land packets reach it).
#
#   scripts/run_stack.sh --mode real [--config-dir config/real] [--glimmer local]
#   scripts/run_stack.sh --mode real --no-perception --no-agent          # Session A / B: backend + UI
#   scripts/run_stack.sh --mode twin [--scenario s1] [--latency-ms 500] [--perception oracle]
#
# Options (defaults in brackets):
#   --mode real|twin           required
#   --config-dir DIR           governor.yaml + autopilot.yaml [real: config, twin: config/twin]
#   --port N                   backend HTTP port [8000]
#   --ui-port N                UI preview port [5173]          --no-ui     no UI
#   --perception sam_mlx|sam_api|oracle|none   [sam_mlx]       --no-perception
#   --agent auto|keyword|local|none            [auto]          --no-agent
#       auto: Voice Transcribe + Glimmer/Spark race; keyword: no LLM; local: on-device ASR (edge mode)
#   --agent-extra "ARGS"       extra agent flags, e.g. "--no-mic --mute --tts none"
#   --glimmer local|none       local: start scripts/serve_glimmer.sh (Glimmer + LoRA on :8080, ~17 GB) [none]
#   --no-build                 serve the existing UI build (dist/) instead of rebuilding
#   --logs DIR                 [logs/stack-<time>]
#   twin only:
#   --twin-ports HTTP,RC,RTSP  [8766,23458,8554]    --scenario s1..s6 [s1]    --latency-ms N [500]
#   --headless                 no UI heartbeat needed (twin only)    --no-renderer   placeholder video
#   --chrome-profile DIR       renderer Chrome profile [$MHD_HOME/.cache/chrome-twin]
#
# Real mode refuses --headless, twin configs, the oracle, and --twin-* options. It never starts the
# twin. Run scripts/preflight.py first (the X69 may allow only one RTSP client: the backend).
set -o pipefail             # no -u: bash 3.2 (macOS) treats empty arrays as unbound
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$TREE/env.sh"

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "run_stack: $*" >&2; exit 2; }

MODE="" CONFIG_DIR="" PORT=8000 UI_PORT=5173 UI=1 PERCEPTION=sam_mlx AGENT=auto AGENT_EXTRA="" GLIMMER=none
BUILD=1 LOGS="" TWIN_PORTS="" SCENARIO=s1 LATENCY=500 HEADLESS=0 RENDERER=1 CHROME_PROFILE="" TWIN_OPT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ui-port) UI_PORT="$2"; shift 2 ;;
    --no-ui) UI=0; shift ;;
    --perception) PERCEPTION="$2"; shift 2 ;;
    --no-perception) PERCEPTION=none; shift ;;
    --agent) AGENT="$2"; shift 2 ;;
    --no-agent) AGENT=none; shift ;;
    --agent-extra) AGENT_EXTRA="$2"; shift 2 ;;
    --glimmer) GLIMMER="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --logs) LOGS="$2"; shift 2 ;;
    --twin-ports) TWIN_PORTS="$2"; TWIN_OPT=1; shift 2 ;;
    --scenario) SCENARIO="$2"; TWIN_OPT=1; shift 2 ;;
    --latency-ms) LATENCY="$2"; TWIN_OPT=1; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --no-renderer) RENDERER=0; TWIN_OPT=1; shift ;;
    --chrome-profile) CHROME_PROFILE="$2"; TWIN_OPT=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

# ------------------------------------------------------------------------------- validation
case "$MODE" in real|twin) ;; *) usage >&2; die "--mode real|twin is required" ;; esac
case "$PERCEPTION" in sam_mlx|sam_api|oracle|none) ;; *) die "--perception sam_mlx|sam_api|oracle|none" ;; esac
case "$AGENT" in auto|keyword|local|none) ;; *) die "--agent auto|keyword|local|none" ;; esac
case "$GLIMMER" in local|none) ;; *) die "--glimmer local|none" ;; esac
[[ -z "$CONFIG_DIR" ]] && { [[ "$MODE" == real ]] && CONFIG_DIR="config" || CONFIG_DIR="config/twin"; }
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$TREE/$CONFIG_DIR"
GOV_CFG="$CONFIG_DIR/governor.yaml"; AP_CFG="$CONFIG_DIR/autopilot.yaml"
[[ -f "$GOV_CFG" && -f "$AP_CFG" ]] || die "missing $GOV_CFG or $AP_CFG"
if [[ "$MODE" == real ]]; then
  [[ "$HEADLESS" == 1 ]] && die "REAL mode refuses --headless (the UI heartbeat is a safety check)"
  [[ "$TWIN_OPT" == 1 ]] && die "REAL mode refuses twin options (--twin-ports, --scenario, --latency-ms, --no-renderer, --chrome-profile)"
  [[ "$PERCEPTION" == oracle ]] && die "REAL mode refuses --perception oracle (twin ground truth)"
  case "$(cd "$CONFIG_DIR" && pwd)/" in */twin/*) die "REAL mode refuses twin configs ($CONFIG_DIR)" ;; esac
  if grep -qiE "TWIN calibration|Never use this file for the real drone|Never for the real drone" "$GOV_CFG" "$AP_CFG"; then
    die "REAL mode refuses $CONFIG_DIR: its YAML says it is a twin calibration"
  fi
  if pgrep -f "mhd\.twin\.server" >/dev/null 2>&1; then
    echo "run_stack: WARNING: a twin server is running; stop it for real flights (GPU/CPU load)" >&2
  fi
fi
for p in "$PORT" $([[ "$UI" == 1 ]] && echo "$UI_PORT"); do
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

# ------------------------------------------------------------------------------- process table
set -m                      # each component in its own process group: Ctrl-C reaches only this script
NAMES=() PIDS=() CMDS=() DEAD=()
start() {                   # start NAME LOGFILE CMD...  (cwd $TREE)
  local name="$1" logf="$2"; shift 2
  (cd "$TREE" && exec "$@") >"$logf" 2>&1 &
  local pid=$!
  NAMES+=("$name"); PIDS+=("$pid"); CMDS+=("$*"); DEAD+=(0)
  log "started $name (pid $pid): $*  -> $(basename "$logf")"
}
pid_of() { local i; for i in "${!NAMES[@]}"; do [[ "${NAMES[$i]}" == "$1" ]] && { echo "${PIDS[$i]}"; return; }; done; }
alive() { kill -0 "$1" 2>/dev/null; }
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
  if [[ "$STOPPING" == 1 ]]; then        # second Ctrl-C: the backend's own force-quit (lands 0.5 s first)
    local b; b="$(pid_of backend)"; [[ -n "$b" ]] && kill -INT -- "-$b" 2>/dev/null
    return
  fi
  STOPPING=1
  trap 'cleanup' INT TERM
  log "stopping: backend first (it sends LAND on SIGINT), then the clients, the UI, the twin last"
  local b; b="$(pid_of backend)"
  [[ -n "$b" ]] && alive "$b" && kill -INT -- "-$b" 2>/dev/null
  local w=()
  stop_group agent INT 5 & w+=($!)
  stop_group perception INT 5 & w+=($!)
  stop_group glimmer INT 5 & w+=($!)
  stop_group backend INT 8
  local j; for j in "${w[@]}"; do wait "$j" 2>/dev/null; done
  stop_group ui INT 3
  stop_group renderer INT 6
  stop_group twin INT 10
  [[ -n "${CAFF:-}" ]] && kill "$CAFF" 2>/dev/null
  log "stopped. logs: $RUN"
  exit 0
}
trap 'cleanup' INT TERM
trap '[[ "$STOPPING" == 1 ]] || cleanup' EXIT

caffeinate -dimsu -w $$ >/dev/null 2>&1 &
CAFF=$!

{
  echo "mode=$MODE config=$CONFIG_DIR hfov=$HFOV port=$PORT ui=$UI:$UI_PORT perception=$PERCEPTION agent=$AGENT glimmer=$GLIMMER"
  echo "MHD_CAPTURE_OFFSET_MS=${MHD_CAPTURE_OFFSET_MS:-<unset>} tree=$TREE"
  git -C "$TREE" log -1 --format='commit %h %s' 2>/dev/null
} >> "$STACK_LOG"
log "mode $MODE, configs $CONFIG_DIR (HFOV $HFOV), logs $RUN"
[[ "$MODE" == real ]] && log "REAL DRONE: run scripts/preflight.py first; MHD_CAPTURE_OFFSET_MS=${MHD_CAPTURE_OFFSET_MS:-unset}"

wait_http() {               # wait_http URL SECONDS NAME
  local i
  for ((i = 0; i < $2 * 4; i++)); do
    curl -sf -o /dev/null "$1" && return 0
    local p; p="$(pid_of "$3")"
    [[ -n "$p" ]] && ! alive "$p" && { log "$3 exited while starting; see its log"; return 1; }
    sleep 0.25
  done
  log "timed out waiting for $1"; return 1
}

# ------------------------------------------------------------------------------- twin
if [[ "$MODE" == twin ]]; then
  IFS=, read -r T_HTTP T_RC T_RTSP <<< "${TWIN_PORTS:-8766,23458,8554}"
  if [[ "$RENDERER" == 1 && ! -d "$TREE/twin/web/node_modules/three" ]]; then
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

# ------------------------------------------------------------------------------- glimmer (optional)
if [[ "$GLIMMER" == local ]]; then
  start glimmer "$RUN/glimmer.log" bash scripts/serve_glimmer.sh --port 8080
fi

# ------------------------------------------------------------------------------- backend
B_ARGS=(--mode "$MODE" --port "$PORT" --governor-config "$GOV_CFG" --autopilot-config "$AP_CFG" --logs "$RUN/backend")
if [[ "$MODE" == twin ]]; then
  B_ARGS+=(--control-port "$T_RC" --rtsp-url "$RTSP_URL")
  [[ "$HEADLESS" == 1 ]] && B_ARGS+=(--headless)
fi
start backend "$RUN/backend.log" bash scripts/run_backend.sh "${B_ARGS[@]}"
wait_http "http://127.0.0.1:$PORT/state" 40 backend || cleanup
BACKEND_RESTART="scripts/run_backend.sh ${B_ARGS[*]}"

# ------------------------------------------------------------------------------- perception
case "$PERCEPTION" in
  sam_mlx|sam_api)
    start perception "$RUN/perception.log" "$MHD_ML_PY" -m mhd_ml.perception.service --backend "ws://127.0.0.1:$PORT" \
      --detector "$PERCEPTION" --tracker auto --hfov-deg "$HFOV" --obs-log "$RUN/obs.jsonl" --log-level INFO ;;
  oracle)
    start perception "$RUN/perception.log" "$MHD_PY" -m mhd.twin.oracle --twin "http://127.0.0.1:$T_HTTP" \
      --backend "ws://127.0.0.1:$PORT" --latency-ms 300 --log "$RUN/obs.jsonl" ;;
esac

# ------------------------------------------------------------------------------- agent
if [[ "$AGENT" != none ]]; then
  A_ARGS=(--backend "ws://127.0.0.1:$PORT")
  case "$AGENT" in
    auto) A_ARGS+=(--asr meta --intent auto) ;;
    keyword) A_ARGS+=(--asr meta --intent keyword) ;;
    local) if [[ "$GLIMMER" == local ]]; then A_ARGS+=(--asr local --intent glimmer); else A_ARGS+=(--asr local --intent keyword); fi ;;
  esac
  [[ "$GLIMMER" == local ]] && A_ARGS+=(--glimmer-url http://127.0.0.1:8080/v1 --glimmer-model default_model --glimmer-label glimmer-30b+lora)
  # shellcheck disable=SC2206
  [[ -n "$AGENT_EXTRA" ]] && A_ARGS+=($AGENT_EXTRA)
  start agent "$RUN/agent.log" "$MHD_ML_PY" -m mhd_ml.agent.service "${A_ARGS[@]}"
fi

# ------------------------------------------------------------------------------- UI (production build)
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

# ------------------------------------------------------------------------------- status and watch
echo
log "UP: backend http://127.0.0.1:$PORT  $([[ "$UI" == 1 ]] && echo "UI http://127.0.0.1:$UI_PORT (open it in its own Chrome window)")"
[[ "$MODE" == twin ]] && log "twin http://127.0.0.1:$T_HTTP (chase: /chase)  RTSP $RTSP_URL"
log "Ctrl-C stops everything (the backend lands first). Logs: $RUN"
while true; do
  for i in "${!NAMES[@]}"; do
    if [[ "${DEAD[$i]}" == 0 ]] && ! alive "${PIDS[$i]}"; then
      DEAD[$i]=1
      wait "${PIDS[$i]}" 2>/dev/null; code=$?
      name="${NAMES[$i]}"
      log "!!!!! $name EXITED (code $code). See $RUN/$name.log"
      if [[ "$name" == backend ]]; then
        log "!!!!! NO BACKEND: the drone gets no RC packets (Session A link-loss behavior applies)."
        log "!!!!! restart it in another window of this terminal app:  $BACKEND_RESTART"
      fi
    fi
  done
  sleep 1 &
  wait $! 2>/dev/null
done
