"""Spoken replies: MMS-TTS (facebook/mms-tts-eng, a Meta model, CC-BY-NC) with macOS `say` as fallback.

The reply text always comes from mhd.intent_spec.build_reply(intent), built locally from the mission
JSON, so models emit fewer tokens.

- torch and transformers are imported lazily, in the worker thread, the first time MMS is needed.
- start() loads the model and synthesizes one warm-up sentence (without playing it), so the first real
  reply is not slow. Measured: load 0.4 s (after a 2.3 s torch import), synthesis 0.12 s for
  "Landing now." and 0.39 s for "Following the person in blue shirt." (CPU, 2 threads).
- Playback uses sounddevice (imported lazily). A newer reply interrupts the one playing.
- If MMS fails to load or synthesize, the same text goes to `say` instead.

    $MHD_ML_PY -m mhd_ml.agent.tts "Following the person in blue shirt."          # speak it
    $MHD_ML_PY -m mhd_ml.agent.tts --out media/agent/reply.wav "Landing now."      # write a file
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from mhd.contracts import now_ms

log = logging.getLogger("mhd.agent.tts")


def default_model_dir() -> Path:
    return Path(os.environ.get("MHD_MODELS", "")) / "mms-tts-eng"


class MmsTts:
    """facebook/mms-tts-eng through transformers.VitsModel (16 kHz output)."""

    def __init__(self, model_dir: Optional[str | Path] = None, num_threads: int = 2, seed: int = 0):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.num_threads = num_threads
        self.seed = seed
        self._model = None
        self._tok = None
        self.sample_rate = 16000
        self.load_s = 0.0

    def load(self) -> None:
        if self._model is not None:
            return
        t0 = time.monotonic()
        import torch  # lazy: heavy
        from transformers import AutoTokenizer, VitsModel

        torch.set_num_threads(self.num_threads)
        self._torch = torch
        self._model = VitsModel.from_pretrained(str(self.model_dir))
        self._model.eval()
        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.sample_rate = int(self._model.config.sampling_rate)
        self.load_s = time.monotonic() - t0

    def synthesize(self, text: str) -> np.ndarray:
        self.load()
        torch = self._torch
        torch.manual_seed(self.seed)          # VITS samples noise; fix it so replies sound the same
        inputs = self._tok(text, return_tensors="pt")
        with torch.inference_mode():
            wav = self._model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if peak > 0:
            wav = wav * (0.9 / peak)
        return wav


class Speaker:
    """Speaks replies on a worker thread. engine: "mms" | "say" | "none"."""

    def __init__(self, engine: str = "mms", voice: str = "Samantha", model_dir: Optional[str] = None,
                 device=None, play: bool = True):
        self.engine = engine
        self.voice = voice
        self.device = device
        self.play = play
        self.mms = MmsTts(model_dir) if engine == "mms" else None
        self._q: "queue.Queue[tuple[str, float]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._say_proc: Optional[subprocess.Popen] = None
        self.mms_failed = False
        self._busy = False
        self._gen = 0                 # bumped by every interrupting speak()
        self._audible: list[tuple[float, float, str]] = []   # (start, end, text) of lines actually played
        self.history: list[dict] = []          # {text, engine, queued_ms, started_ms, synth_ms}
        self.warmup_ms: Optional[float] = None

    def start(self, warm_up: bool = True) -> "Speaker":
        self._thread = threading.Thread(target=self._run, args=(warm_up,), name="tts", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def speak(self, text: str, interrupt: bool = True) -> None:
        """Queue a line. By default a newer line interrupts the one being played (safety replies
        must not wait); interrupt=False plays it after the current line instead."""
        if self.engine == "none" or not text:
            return
        if interrupt:
            self._gen += 1            # lines queued before this one are now stale
            self._interrupt()
        self._q.put((text, now_ms(), self._gen))

    def audible_words(self, now: Optional[float] = None, tail_s: float = 1.5) -> set[str]:
        """Words of every line that was playing at some point in the last `tail_s` seconds (monotonic
        clock). The safety listener uses this to ignore its own voice coming back through the mic."""
        from .words import words

        now = time.monotonic() if now is None else now
        out: set[str] = set()
        for start, end, text in list(self._audible):
            if start <= now and (end == float("inf") or now <= end + tail_s):
                out.update(words(text))
        return out

    def _mark_audible(self, text: str, start: float, end: float) -> None:
        self._audible = [a for a in self._audible[-5:] if a[1] != float("inf")] + [(start, end, text)]

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until every queued line has finished playing."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._q.empty() and not self._busy:
                return True
            time.sleep(0.05)
        return False

    def close(self, interrupt: bool = True) -> None:
        self._stop.set()
        if interrupt:
            self._interrupt()
        self._q.put(("", 0.0, self._gen))
        if self._thread:
            self._thread.join(timeout=3)

    # ---- worker ------------------------------------------------------------------------------
    def _run(self, warm_up: bool) -> None:
        if self.mms is not None:
            try:
                t0 = time.monotonic()
                self.mms.load()
                if warm_up:
                    self.mms.synthesize("Ready.")
                self.warmup_ms = (time.monotonic() - t0) * 1000.0
                log.info("MMS-TTS ready in %.0f ms", self.warmup_ms)
            except Exception as e:
                self.mms_failed = True
                log.warning("MMS-TTS unavailable (%s); replies will use `say`", e)
        self._ready.set()
        while not self._stop.is_set():
            text, queued, gen = self._q.get()
            if self._stop.is_set() or not text:
                continue
            # A line is stale if an interrupting line was queued after it (its generation is older).
            # Non-interrupting lines keep their generation, so they play in order after the current one.
            if gen != self._gen:
                continue
            self._busy = True
            try:
                self._say_one(text, queued)
            finally:
                self._busy = False

    def _say_one(self, text: str, queued_ms: float) -> None:
        rec = {"text": text, "queued_ms": queued_ms, "engine": None, "started_ms": None, "synth_ms": None}
        self.history.append(rec)
        if self.mms is not None and not self.mms_failed:
            try:
                t0 = time.monotonic()
                wav = self.mms.synthesize(text)
                rec["synth_ms"] = (time.monotonic() - t0) * 1000.0
                rec["engine"] = "mms"
                rec["started_ms"] = now_ms()
                if self.play:
                    t0 = time.monotonic()
                    self._mark_audible(text, t0, float("inf"))
                    try:
                        self._play(wav, self.mms.sample_rate)
                    finally:
                        self._mark_audible(text, t0, time.monotonic())
                return
            except Exception as e:
                log.warning("MMS-TTS failed (%s); falling back to `say`", e)
        rec["engine"] = "say"
        rec["started_ms"] = now_ms()
        if self.play:
            t0 = time.monotonic()
            self._mark_audible(text, t0, float("inf"))
            try:
                self._say(text)
            finally:
                self._mark_audible(text, t0, time.monotonic())

    def _play(self, wav: np.ndarray, sr: int) -> None:
        import sounddevice as sd  # lazy

        sd.play(wav, sr, device=self.device)
        sd.wait()

    def _say(self, text: str) -> None:
        try:
            self._say_proc = subprocess.Popen(["say", "-v", self.voice, text])
            self._say_proc.wait(timeout=20)
        except Exception as e:
            log.error("`say` failed: %s", e)
        finally:
            self._say_proc = None

    def _interrupt(self) -> None:
        try:
            if self.play and self.mms is not None and "sounddevice" in __import__("sys").modules:
                import sounddevice as sd

                sd.stop()
        except Exception:
            pass
        p = self._say_proc
        if p is not None and p.poll() is None:
            p.terminate()


def say_to_file(text: str, path: str | Path, voice: str = "Samantha") -> Path:
    """`say` fallback rendered to an AIFF file (used by the smoke test; plays nothing)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["say", "-v", voice, "-o", str(p), text], check=True, timeout=30)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Speak a reply with MMS-TTS (fallback: say)")
    ap.add_argument("text")
    ap.add_argument("--engine", choices=["mms", "say"], default="mms")
    ap.add_argument("--out", help="write a WAV instead of playing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.out:
        from .audio import write_wav

        m = MmsTts()
        wav = m.synthesize(args.text)
        write_wav(args.out, wav, m.sample_rate)
        print(f"wrote {args.out} ({len(wav) / m.sample_rate:.2f} s)")
        return 0
    s = Speaker(engine=args.engine).start(warm_up=False)
    s.wait_ready()
    s.speak(args.text)
    time.sleep(0.1)
    s.wait_idle()
    s.close(interrupt=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
