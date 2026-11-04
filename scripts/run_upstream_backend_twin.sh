#!/usr/bin/env bash
# Upstream turbodrone backend (unchanged code) pointed at the twin, exactly as for the real drone
# except DRONE_IP=127.0.0.1, CONTROL_PORT and X69_LG_RTSP_URL.
#
#   scripts/run_upstream_backend_twin.sh [--port 8200] [--rc-port 23520] [--rtsp-port 8254] [--default-decoder-threads]
#
# OPENCV_FFMPEG_THREADS=1: OpenCV's FFmpeg backend otherwise frame-threads the HEVC decoder,
# which adds about 0.7 s of latency (measured; see docs/ws/twin.md). This is an environment
# setting, not a code change, and the real-drone backend needs it too.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"
PORT=8200; RC=23520; RTSP=8254; THREADS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --rc-port) RC="$2"; shift 2;;
    --rtsp-port) RTSP="$2"; shift 2;;
    --default-decoder-threads) THREADS=""; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [ ! -f "$MHD_TREE/turbodrone/backend/models/x69_lg_rc.py" ]; then
  echo "turbodrone/backend/models is missing (git-ignored by 'models/'); copy it from the main checkout:" >&2
  echo "  cp -R \"\$MHD_HOME/turbodrone/backend/models\" \"$MHD_TREE/turbodrone/backend/\"" >&2
  exit 1
fi
export FLIGHT_MODE=twin DRONE_TYPE=x69_lg DRONE_IP=127.0.0.1 CONTROL_PORT="$RC" PLUGINS_ENABLED=false
export X69_LG_VIDEO_MODE=rtsp X69_LG_RTSP_URL="rtsp://127.0.0.1:${RTSP}/live/ch00_1"
if [ -n "$THREADS" ]; then export OPENCV_FFMPEG_THREADS="$THREADS"; fi
cd "$MHD_TREE/turbodrone/backend"
# upstream modules (models, protocols, ...) are imported relative to turbodrone/backend
export PYTHONPATH="$MHD_TREE/turbodrone/backend${PYTHONPATH:+:$PYTHONPATH}"
exec "$MHD_PY" -m uvicorn web_server:app --host 127.0.0.1 --port "$PORT" --log-level warning
