#!/usr/bin/env bash
# Downloads every model and tool into the project (never outside it). Resumable; safe to re-run.
set -u
source "$(dirname "$0")/../env.sh"
M="$MHD_MODELS"; T="$MHD_TOOLS"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
fetch(){ # url dest
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ] && [ ! -f "$2.part" ]; then log "have $(basename "$2")"; return 0; fi
  log "get  $1"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2" && log "done $(basename "$2") $(du -h "$2" | cut -f1)" || log "FAIL $1"
}
hf_repo(){ # repo local_dir
  local repo="$1" dir="$2"
  mkdir -p "$dir"
  /usr/bin/python3 - "$repo" <<'PY' > "$dir/.files"
import json, sys, urllib.request
repo = sys.argv[1]
with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}") as r:
    d = json.load(r)
for s in d.get("siblings", []):
    print(s["rfilename"])
PY
  while read -r f; do
    [ -z "$f" ] && continue
    fetch "https://huggingface.co/$repo/resolve/main/$f" "$dir/$f"
  done < "$dir/.files"
}
log "=== mediamtx"
if [ ! -x "$T/mediamtx/mediamtx" ]; then
  fetch "https://github.com/bluenviron/mediamtx/releases/download/v1.21.3/mediamtx_v1.21.3_darwin_arm64.tar.gz" "$T/mediamtx/mediamtx.tar.gz"
  (cd "$T/mediamtx" && tar xzf mediamtx.tar.gz && log "mediamtx unpacked: $(ls)")
fi
log "=== sherpa-onnx Omnilingual ASR 300M int8 + silero VAD"
SH="$M/sherpa"
if [ ! -d "$SH/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05" ]; then
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05.tar.bz2" "$SH/omni300m.tar.bz2"
  (cd "$SH" && tar xjf omni300m.tar.bz2 && rm -f omni300m.tar.bz2 && log "omnilingual unpacked")
fi
fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" "$SH/silero_vad.onnx"
log "=== OpenCV VitTrack ONNX"
fetch "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx" "$M/opencv/object_tracking_vittrack_2023sep.onnx"
log "=== MMS-TTS (facebook/mms-tts-eng, CC-BY-NC)"
hf_repo facebook/mms-tts-eng "$M/mms-tts-eng"
log "=== SAM 3.1 MLX (mlx-community/sam3.1-bf16)"
hf_repo mlx-community/sam3.1-bf16 "$M/sam3.1-bf16"
log "=== Muse Glimmer 30B MLX 4-bit (mlx-community/Muse-Glimmer-30B-4bit)"
hf_repo mlx-community/Muse-Glimmer-30B-4bit "$M/Muse-Glimmer-30B-4bit"
log "=== ALL DONE"
du -sh "$M"/* "$T"/* 2>/dev/null
