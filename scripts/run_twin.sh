#!/usr/bin/env bash
# Run the X69 digital twin: twin server (which starts its own RTSP-only mediamtx and the ffmpeg
# HEVC publisher) plus the Three.js renderer in Chrome, all under caffeinate -i.
#
#   scripts/run_twin.sh [--ports HTTP,RC,RTSP | --ports default | --ports ws-twin] [--scenario s1]
#                       [--latency-ms 500] [--chase] [--chase-rec] [--no-renderer] [-- extra server args]
#
#   --ports default   8766,23458,8554 (docs/interfaces.md)
#   --ports ws-twin   8266,23520,8254 (the overnight twin workstream's ports)
#   --latency-ms      TOTAL camera-to-decode latency set point. Default 500 (the X69's reported
#                     lag, until Session A measures it); 0 = no injection (about 100 ms baseline).
# Ctrl-C (or SIGTERM) stops everything it started.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
PORTS=default; SCENARIO=s1; LATENCY=500; RENDER=1; CHASE=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --ports) PORTS="$2"; shift 2;;
    --scenario) SCENARIO="$2"; shift 2;;
    --latency-ms) LATENCY="$2"; shift 2;;
    --chase) CHASE="$CHASE --chase"; shift;;
    --chase-rec) CHASE="$CHASE --chase-rec"; shift;;
    --no-renderer) RENDER=0; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
case "$PORTS" in
  default) PORTS=8766,23458,8554;;
  ws-twin) PORTS=8266,23520,8254;;
esac
IFS=, read -r HTTP RC RTSP <<< "$PORTS"
export FLIGHT_MODE=twin DRONE_IP=127.0.0.1
mkdir -p "$MHD_LOGS"
STAMP="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="$MHD_LOGS/twin-server-$STAMP.log"; RENDER_LOG="$MHD_LOGS/twin-renderer-$STAMP.log"
ARGS=(--http-port "$HTTP" --rc-port "$RC" --rtsp-port "$RTSP" --scenario "$SCENARIO")
[ -n "$LATENCY" ] && [ "$LATENCY" != 0 ] && ARGS+=(--latency-ms "$LATENCY")
caffeinate -i -w $$ &            # keep the Mac awake while this script runs
"$MHD_PY" -m mhd.twin.server "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "$SERVER_LOG" 2>&1 &
SERVER=$!
RENDERER=""
cleanup() {
  trap - EXIT INT TERM
  [ -n "$RENDERER" ] && kill "$RENDERER" 2>/dev/null || true
  kill "$SERVER" 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "$SERVER" 2>/dev/null || { [ -n "$RENDERER" ] && kill -0 "$RENDERER" 2>/dev/null; } || break
    sleep 0.25
  done
  for p in $SERVER $RENDERER; do pkill -9 -P "$p" 2>/dev/null || true; kill -9 "$p" 2>/dev/null || true; done
  echo "twin stopped"
}
trap cleanup EXIT INT TERM
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$HTTP/status" > /dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "twin server failed; see $SERVER_LOG" >&2; tail -20 "$SERVER_LOG" >&2; exit 1; }
  sleep 0.25
done
echo "twin: http://127.0.0.1:$HTTP  RC udp 127.0.0.1:$RC  RTSP rtsp://127.0.0.1:$RTSP/live/ch00_1  scenario $SCENARIO"
echo "logs: $SERVER_LOG"
if [ "$RENDER" = 1 ]; then
  "$MHD_PY" -m mhd.twin.launch_renderer --twin "http://127.0.0.1:$HTTP" $CHASE > "$RENDER_LOG" 2>&1 &
  RENDERER=$!
  echo "renderer log: $RENDER_LOG"
fi
wait "$SERVER"
