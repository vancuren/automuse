"""Gold set through a real speech path: macOS `say` speaks each gold command, Omnilingual ASR transcribes it.

This gives a third evaluation set with genuine recognizer errors (the local safety listener's ASR,
Omnilingual CTC 300M v2 int8 via sherpa-onnx), instead of only synthetic noise. Labels are the gold labels.

  $MHD_ML_PY training/gold_asr.py        # writes training/data/gold_asr.jsonl (audio stays in logs/gold_asr/)

CPU only (sherpa-onnx, 2 threads); it does not touch the GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"
AUDIO = ROOT / "logs" / "gold_asr"
MODEL_DIR = Path(os.environ["MHD_MODELS"]) / "sherpa" / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"
# Natural macOS voices with six English accents (US, UK, Australia, Ireland, India, South Africa), one per
# command in turn. The robotic "Eloquence" voices (Eddy, Flo, Reed, ...) were dropped: Omnilingual, which has
# no language hint here, often transcribes them in Thai or Chinese script, which is not what a person causes.
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa"]


def synth(text: str, voice: str, out_wav: Path) -> None:
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(out_wav)], check=True)
    aiff.unlink(missing_ok=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


def main() -> None:
    import sherpa_onnx

    AUDIO.mkdir(parents=True, exist_ok=True)
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(MODEL_DIR / "model.int8.onnx"), tokens=str(MODEL_DIR / "tokens.txt"), num_threads=2)
    gold = [json.loads(l) for l in (DATA / "gold.jsonl").read_text().splitlines() if l.strip()]
    out, times = [], []
    for k, g in enumerate(gold):
        voice = VOICES[k % len(VOICES)]
        wav = AUDIO / f"{g['id']}.wav"
        synth(g["transcript"], voice, wav)
        samples, sr = read_wav(wav)
        t0 = time.perf_counter()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        rec.decode_stream(stream)
        text = stream.result.text.strip()
        times.append((time.perf_counter() - t0) * 1000)
        out.append({"id": g["id"].replace("gold-", "goldasr-"), "transcript": text, "intent": g["intent"],
                    "group": g.get("group", ""), "note": g.get("note", ""), "tts_text": g["transcript"],
                    "voice": voice, "audio_s": round(len(samples) / sr, 2)})
        print(f"{g['id']} [{voice}] {g['transcript']!r} -> {text!r}", flush=True)
    (DATA / "gold_asr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    times.sort()
    print(f"ASR decode ms: p50 {times[len(times) // 2]:.0f}, max {times[-1]:.0f} (CPU, 2 threads, "
          f"MLX training running on the GPU)", file=sys.stderr)


if __name__ == "__main__":
    main()
