"""Omnilingual ASR CTC-300M (int8) through sherpa-onnx: offline transcription on this Mac.

Two users share one recognizer instance (about 370 MB on disk, loaded once):
- the always-on safety listener (short VAD segments, every few hundred ms while someone talks);
- push-to-talk in edge mode (`--asr local`), via LocalASR.

The model takes 16 kHz mono float audio. Output is lowercase and unpunctuated, and may be misspelled;
everything downstream (safety matching, keyword grammar, intent prompts) is written for that.
Measured on this Mac (M5 Pro, nothing else heavy running, Glimmer download in progress): load 0.39 s,
decode 0.15 s for a 2 s clip and 0.29 s for a 3.8 s clip with 2 threads.

Carrier phrase. The CTC model has no language conditioning, and on very short clips it often guesses
the wrong language and script: a lone "land" came back as Arabic script, "time to land" as Thai,
"stop, stop" as Japanese kana. Prepending a fixed English carrier ("the command is", spoken by `say`)
gives it English context; tokens whose timestamps fall inside the carrier are then dropped. On the test
set this turned "stob" into "stop", "preese" into "freeze", and fixed every non-Latin output.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import resample

log = logging.getLogger("mhd.agent.asr_local")

MODEL_SUBDIR = "sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"
ASR_SAMPLE_RATE = 16000


def default_model_dir() -> Path:
    models = os.environ.get("MHD_MODELS")
    if not models:
        raise RuntimeError("MHD_MODELS is not set; run `source env.sh` first")
    return Path(models) / MODEL_SUBDIR


@dataclass
class Transcript:
    text: str
    latency_ms: float                 # time spent producing the final text after audio ended
    model: str
    final: bool = True
    details: dict = field(default_factory=dict)


class OmnilingualRecognizer:
    """Thread-safe wrapper around sherpa_onnx.OfflineRecognizer for the Omnilingual CTC model."""

    def __init__(self, model_dir: Optional[str | Path] = None, num_threads: int = 2,
                 provider: str = "cpu", use_carrier: bool = True):
        import sherpa_onnx  # lazy: heavy native import

        d = Path(model_dir) if model_dir else default_model_dir()
        model, tokens = d / "model.int8.onnx", d / "tokens.txt"
        if not model.exists() or not tokens.exists():
            raise FileNotFoundError(f"Omnilingual model files missing in {d}")
        t0 = time.monotonic()
        self._rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=str(model), tokens=str(tokens), num_threads=num_threads, provider=provider)
        self.load_s = time.monotonic() - t0
        self._lock = threading.Lock()
        log.info("omnilingual CTC loaded in %.2f s (%d threads)", self.load_s, num_threads)

        self.carrier: Optional[np.ndarray] = None
        if use_carrier:
            try:
                self.carrier = load_carrier()
            except Exception as e:  # no `say`? Work without it.
                log.warning("carrier phrase unavailable (%s); short clips may decode in the wrong script", e)

    def transcribe(self, samples: np.ndarray, sample_rate: int = ASR_SAMPLE_RATE,
                   carrier: bool = True) -> str:
        """Decode a whole clip. Audio at other rates is resampled to 16 kHz first."""
        x = np.asarray(samples, dtype=np.float32)
        if sample_rate != ASR_SAMPLE_RATE:
            x = resample(x, sample_rate, ASR_SAMPLE_RATE)
        if len(x) < ASR_SAMPLE_RATE // 10:      # under 100 ms: nothing to decode
            return ""
        prefix = self.carrier if (carrier and self.carrier is not None) else None
        if prefix is not None:
            gap = np.zeros(int(CARRIER_GAP_S * ASR_SAMPLE_RATE), dtype=np.float32)
            x = np.concatenate([prefix, gap, x])
        with self._lock:
            s = self._rec.create_stream()
            s.accept_waveform(ASR_SAMPLE_RATE, x)
            self._rec.decode_stream(s)
            res = s.result
        if prefix is None:
            return res.text.strip()
        # Drop the carrier's tokens by time: keep tokens that start after the middle of the gap.
        cut = len(prefix) / ASR_SAMPLE_RATE + CARRIER_GAP_S / 2
        kept = [tok for tok, t in zip(res.tokens, res.timestamps) if t >= cut]
        return " ".join("".join(kept).split())


CARRIER_TEXT = "the command is"
CARRIER_GAP_S = 0.15


def carrier_path() -> Path:
    tree = os.environ.get("MHD_TREE") or str(Path(__file__).resolve().parents[3])
    return Path(tree) / "media" / "agent" / "carrier_16k.wav"


def load_carrier() -> np.ndarray:
    """The English carrier clip (16 kHz float), made once with `say` and cached under media/agent/."""
    from .audio import load_wav, speech_bounds, write_wav

    p = carrier_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        aiff = p.with_suffix(".aiff")
        raw = p.with_name("carrier_raw.wav")
        subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), CARRIER_TEXT], check=True, timeout=20)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{ASR_SAMPLE_RATE}", "-c", "1",
                        str(aiff), str(raw)], check=True, timeout=20)
        x, sr = load_wav(raw, target_sr=ASR_SAMPLE_RATE)
        start, end = speech_bounds(x, sr)
        x = x[max(0, int((start - 0.05) * sr)):int((end + 0.05) * sr)]
        write_wav(p, x, sr)
        aiff.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
    x, _ = load_wav(p, target_sr=ASR_SAMPLE_RATE)
    return x


_shared: Optional[OmnilingualRecognizer] = None
_shared_lock = threading.Lock()


def get_shared_recognizer(model_dir: Optional[str | Path] = None, num_threads: int = 2) -> OmnilingualRecognizer:
    """The process-wide recognizer, loaded on first use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = OmnilingualRecognizer(model_dir=model_dir, num_threads=num_threads)
        return _shared


class LocalASR:
    """Push-to-talk transcription in edge mode (`--asr local`): whole utterance in, final text out."""

    name = "omnilingual-ctc-300m"

    def __init__(self, recognizer: Optional[OmnilingualRecognizer] = None):
        self._rec = recognizer

    @property
    def recognizer(self) -> OmnilingualRecognizer:
        if self._rec is None:
            self._rec = get_shared_recognizer()
        return self._rec

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        t0 = time.monotonic()
        text = self.recognizer.transcribe(samples, sample_rate)
        return Transcript(text=text, latency_ms=(time.monotonic() - t0) * 1000.0, model=self.name)

    def transcribe_file(self, path: str | Path) -> Transcript:
        from .audio import load_wav

        x, sr = load_wav(path, target_sr=ASR_SAMPLE_RATE)
        return self.transcribe(x, sr)
