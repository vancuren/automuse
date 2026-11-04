"""Always-on local safety listener: hears stop, hover, land, and abort and tells the backend at once.

Pipeline (all on this Mac, no network, every mode):
    16 kHz mono audio -> silero VAD (sherpa-onnx) -> Omnilingual ASR CTC-300M -> safety-word match
    -> OperatorCommand: "land" for land/abort/emergency, "pause" (hover) for stop/hover/hold/freeze.

Two detection paths, so a safety word is caught quickly even inside a long sentence:
- final: when the VAD closes a speech segment (min_silence_s of quiet), the whole segment is decoded;
- partial: while speech continues, the last few seconds are decoded every partial_interval_s. A word
  at the very end of a partial transcript is ignored (it may be the start of "landscape"), and so is
  the first word of a window that began mid-utterance (it may be the end of "island").
Each action fires at most once per VAD segment, plus a debounce, so one "land" sends one command.
Land beats hover. Voice never sends an E-stop and never takes off.

Run:
    # live microphone (needs the macOS Microphone permission for the terminal app)
    $MHD_ML_PY -m mhd_ml.agent.safety_listener --mic --backend http://127.0.0.1:8000
    # file input (tests, benchmarks): prints hits and latency after speech end
    $MHD_ML_PY -m mhd_ml.agent.safety_listener --file media/test_audio/safety/land_mid.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from mhd.contracts import now_ms

from .asr_local import ASR_SAMPLE_RATE, OmnilingualRecognizer, get_shared_recognizer
from .audio import load_wav, resample, speech_bounds
from .words import SafetyHit, match_safety

log = logging.getLogger("mhd.agent.safety")

VAD_WINDOW = 512          # silero window at 16 kHz (32 ms)
SEG_PRE_S = 0.30          # the VAD trims word onsets ("freeze" -> "reeze"); decode with this pre-roll
SEG_POST_S = 0.15


class SampleRing:
    """Fixed-capacity ring of the most recent samples, addressed by absolute sample index."""

    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.cap = capacity
        self.end = 0                      # absolute index one past the newest sample

    def append(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= self.cap:
            self.buf[:] = x[-self.cap:]
            self.end += n
            # keep the invariant: buf[(i) % cap] holds absolute sample i
            self.buf = np.roll(self.buf, self.end % self.cap)
            return
        i = self.end % self.cap
        first = min(n, self.cap - i)
        self.buf[i:i + first] = x[:first]
        if first < n:
            self.buf[:n - first] = x[first:]
        self.end += n

    @property
    def start(self) -> int:
        return max(0, self.end - self.cap)

    def get(self, a: int, b: int) -> np.ndarray:
        """Samples [a, b) clipped to what is still held."""
        a, b = max(a, self.start), min(b, self.end)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(a, b) % self.cap
        return self.buf[idx]


def default_vad_model() -> str:
    return str(Path(os.environ.get("MHD_MODELS", "")) / "sherpa" / "silero_vad.onnx")


@dataclass
class SafetyConfig:
    vad_model: str = ""
    vad_threshold: float = 0.5
    min_silence_s: float = 0.25      # quiet needed to close a segment (sets the final-path latency)
    min_speech_s: float = 0.10       # a clipped "stop" is short
    max_speech_s: float = 8.0        # force-close very long segments
    partial_interval_s: float = 0.4  # decode cadence while speech continues
    partial_window_s: float = 3.0    # how much recent audio each partial decode sees
    debounce_s: float = 1.0          # same action again within this time is ignored
    fuzzy: bool = True
    num_threads: int = 2

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SafetyConfig":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


class SafetyListener:
    """Feed it 16 kHz float audio in any chunk size; it returns (and reports) SafetyHits.

    Not thread-safe: one thread feeds it. `on_hit` is called synchronously from feed(), so keep it
    fast (the service hands hits to a sender thread).
    """

    def __init__(self, cfg: Optional[SafetyConfig] = None,
                 recognizer: Optional[OmnilingualRecognizer] = None,
                 on_hit: Optional[Callable[[SafetyHit], None]] = None,
                 on_transcript: Optional[Callable[[str, str], None]] = None):
        import sherpa_onnx  # lazy

        self.cfg = cfg or SafetyConfig()
        vad_path = self.cfg.vad_model or default_vad_model()
        if not Path(vad_path).exists():
            raise FileNotFoundError(f"silero VAD model not found: {vad_path}")
        silero = sherpa_onnx.SileroVadModelConfig(
            model=vad_path, threshold=self.cfg.vad_threshold,
            min_silence_duration=self.cfg.min_silence_s, min_speech_duration=self.cfg.min_speech_s,
            window_size=VAD_WINDOW, max_speech_duration=self.cfg.max_speech_s)
        vcfg = sherpa_onnx.VadModelConfig(silero_vad=silero, sample_rate=ASR_SAMPLE_RATE, num_threads=1)
        self.vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=30)
        self.rec = recognizer or get_shared_recognizer(num_threads=self.cfg.num_threads)
        self.on_hit = on_hit
        self.on_transcript = on_transcript
        # Recent audio by absolute sample index (VAD index == ring index), for pre-roll and partials.
        self._ring = SampleRing(int((max(self.cfg.max_speech_s, self.cfg.partial_window_s) + 3.0)
                                    * ASR_SAMPLE_RATE))
        self._pending = np.zeros(0, dtype=np.float32)   # samples not yet a whole VAD window
        self._fed = 0                                    # samples handed to the VAD so far
        self._speech_start: Optional[int] = None
        self._last_partial_at = 0
        self._fired_in_segment: set[str] = set()
        self._last_fire: dict[str, float] = {}           # action -> stream time (s) of last fire
        self.hits: list[SafetyHit] = []
        self.decodes = 0
        self.decode_ms_total = 0.0

    # ---- feeding -------------------------------------------------------------------------------
    def feed(self, samples: np.ndarray, sample_rate: int = ASR_SAMPLE_RATE) -> list[SafetyHit]:
        x = np.asarray(samples, dtype=np.float32)
        if sample_rate != ASR_SAMPLE_RATE:
            x = resample(x, sample_rate, ASR_SAMPLE_RATE)
        buf = np.concatenate([self._pending, x]) if len(self._pending) else x
        n_whole = (len(buf) // VAD_WINDOW) * VAD_WINDOW
        self._pending = buf[n_whole:].copy()
        out: list[SafetyHit] = []
        for i in range(0, n_whole, VAD_WINDOW):
            out.extend(self._step(buf[i:i + VAD_WINDOW]))
        return out

    def flush(self) -> list[SafetyHit]:
        """End of stream: close any open segment and decode it."""
        out: list[SafetyHit] = []
        if len(self._pending):
            pad = np.zeros(VAD_WINDOW - len(self._pending), dtype=np.float32)
            out.extend(self._step(np.concatenate([self._pending, pad])))
            self._pending = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        out.extend(self._drain_segments())
        return out

    def _step(self, win: np.ndarray) -> list[SafetyHit]:
        self.vad.accept_waveform(win)
        self._ring.append(win)
        self._fed += len(win)
        out = self._drain_segments()
        if self.vad.is_speech_detected():
            if self._speech_start is None:
                # The VAD reports speech min_speech_s after it starts; look back a little further.
                back = int((self.cfg.min_speech_s + 0.3) * ASR_SAMPLE_RATE)
                self._speech_start = max(self._ring.start, self._fed - back)
                self._last_partial_at = self._fed
            if self._fed - self._last_partial_at >= int(self.cfg.partial_interval_s * ASR_SAMPLE_RATE):
                self._last_partial_at = self._fed
                out.extend(self._partial_decode())
        else:
            self._speech_start = None
        return out

    # ---- decoding ------------------------------------------------------------------------------
    def _decode(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.monotonic()
        text = self.rec.transcribe(audio, ASR_SAMPLE_RATE)
        ms = (time.monotonic() - t0) * 1000.0
        self.decodes += 1
        self.decode_ms_total += ms
        return text, ms

    def _drain_segments(self) -> list[SafetyHit]:
        out: list[SafetyHit] = []
        while not self.vad.empty():
            seg = self.vad.front
            seg_start, seg_len = int(seg.start), len(seg.samples)
            self.vad.pop()
            # Decode the segment with a little pre- and post-roll from the ring (the VAD clips onsets).
            a = seg_start - int(SEG_PRE_S * ASR_SAMPLE_RATE)
            b = seg_start + seg_len + int(SEG_POST_S * ASR_SAMPLE_RATE)
            samples = self._ring.get(a, b)
            text, ms = self._decode(samples)
            if self.on_transcript and text:
                self.on_transcript("final", text)
            hit = match_safety(text, partial=False, fuzzy=self.cfg.fuzzy)
            if hit:
                h = self._fire(hit, ms)
                if h:
                    out.append(h)
            self._fired_in_segment.clear()     # the next segment may fire the same action again
            self._speech_start = None
        return out

    def _partial_decode(self) -> list[SafetyHit]:
        if self._speech_start is None:
            return []
        win_start = max(self._speech_start, self._fed - int(self.cfg.partial_window_s * ASR_SAMPLE_RATE))
        truncated = win_start > self._speech_start
        audio = self._ring.get(win_start, self._fed)
        text, ms = self._decode(audio)
        if self.on_transcript and text:
            self.on_transcript("partial", text)
        hit = match_safety(text, partial=True, truncated_start=truncated, fuzzy=self.cfg.fuzzy)
        if hit:
            h = self._fire(hit, ms)
            return [h] if h else []
        return []

    def _fire(self, hit: SafetyHit, compute_ms: float) -> Optional[SafetyHit]:
        t = self._fed / ASR_SAMPLE_RATE
        if hit.action in self._fired_in_segment:
            return None
        last = self._last_fire.get(hit.action)
        if last is not None and t - last < self.cfg.debounce_s:
            return None
        self._fired_in_segment.add(hit.action)
        self._last_fire[hit.action] = t
        hit.audio_t_s = t
        hit.compute_ms = compute_ms
        hit.wall_ms = now_ms()
        self.hits.append(hit)
        log.warning("SAFETY %s (%s) heard %r in %r [%s, decode %.0f ms]", hit.command, hit.action,
                    hit.word, hit.transcript, hit.kind, compute_ms)
        if self.on_hit:
            try:
                self.on_hit(hit)
            except Exception:  # a broken callback must never stop the listener
                log.exception("safety on_hit callback failed")
        return hit

    # ---- drivers -------------------------------------------------------------------------------
    def process_file(self, path: str | Path, chunk_ms: int = 80) -> list[SafetyHit]:
        """Feed a recording in 80 ms chunks as fast as possible. Hit audio_t_s values are exact
        stream positions, so latency after speech end = audio_t_s - speech_end + compute_ms."""
        x, _ = load_wav(path, target_sr=ASR_SAMPLE_RATE)
        return self.process_array(x, chunk_ms)

    def process_array(self, x: np.ndarray, chunk_ms: int = 80) -> list[SafetyHit]:
        n = ASR_SAMPLE_RATE * chunk_ms // 1000
        out: list[SafetyHit] = []
        for i in range(0, len(x), n):
            out.extend(self.feed(x[i:i + n]))
        out.extend(self.flush())
        return out

    def run_queue(self, q: "queue.Queue[np.ndarray]", sample_rate: int, stop: threading.Event) -> None:
        """Live mode: consume float chunks from a MicHub subscription until `stop` is set."""
        while not stop.is_set():
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.feed(x, sample_rate)
            except Exception:
                log.exception("safety listener feed failed; continuing")


def latency_after_speech_end(hit: SafetyHit, speech_end_s: float) -> float:
    """Seconds from the end of speech to the moment the hit would reach the sender (file mode)."""
    return hit.audio_t_s - speech_end_s + hit.compute_ms / 1000.0


class CommandSender:
    """Delivers safety commands to the backend: POST /command first (it acknowledges with the flight
    state), then the agent WebSocket as a fallback. Runs on its own thread so a slow network call never
    blocks the listener. `ws_send` is an optional callable(OperatorCommand) -> bool from BackendLink.
    """

    def __init__(self, http_base: Optional[str], ws_send: Optional[Callable] = None,
                 on_result: Optional[Callable[[SafetyHit, dict], None]] = None, timeout_s: float = 1.0):
        self.http_base = http_base.rstrip("/") if http_base else None
        self.ws_send = ws_send
        self.on_result = on_result
        self.timeout_s = timeout_s
        self._q: "queue.Queue[SafetyHit]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="safety-sender", daemon=True)
        self._thread.start()

    def submit(self, hit: SafetyHit) -> None:
        self._q.put(hit)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        import httpx

        from mhd.contracts import OperatorCommand

        client = httpx.Client(timeout=self.timeout_s)
        try:
            while not self._stop.is_set():
                try:
                    hit = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                cmd = OperatorCommand(command=hit.command)
                result = {"command": hit.command, "via": None, "ok": False, "ms": 0.0}
                t0 = time.monotonic()
                for attempt in range(2):
                    if self.http_base:
                        try:
                            r = client.post(f"{self.http_base}/command", json=cmd.model_dump())
                            result.update(via="http", ok=r.status_code == 200, status=r.status_code)
                            try:
                                result["reply"] = r.json()
                            except Exception:
                                pass
                        except Exception as e:
                            result.update(error=f"http: {e.__class__.__name__}")
                    if not result["ok"] and self.ws_send is not None:
                        try:
                            result.update(via="ws", ok=bool(self.ws_send(cmd)))
                        except Exception as e:
                            result.update(error=f"ws: {e.__class__.__name__}")
                    if result["ok"]:
                        break
                result["ms"] = (time.monotonic() - t0) * 1000.0
                if not result["ok"]:
                    log.error("SAFETY command %s NOT delivered: %s", hit.command, result)
                if self.on_result:
                    try:
                        self.on_result(hit, result)
                    except Exception:
                        log.exception("on_result failed")
        finally:
            client.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Local safety-word listener (stop, hover, land, abort)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", nargs="+", help="audio file(s) to scan instead of the microphone")
    src.add_argument("--mic", action="store_true", help="listen on the default microphone")
    ap.add_argument("--backend", default=None, help="backend HTTP base, e.g. http://127.0.0.1:8000 "
                    "(omit to only print hits)")
    ap.add_argument("--sample-rate", type=int, default=16000, help="mic sample rate")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sender = CommandSender(args.backend) if args.backend else None
    on_hit = sender.submit if sender else None
    if args.file:
        for f in args.file:
            lst = SafetyListener(on_hit=on_hit)
            hits = lst.process_file(f)
            x, sr = load_wav(f, target_sr=ASR_SAMPLE_RATE)
            _, end = speech_bounds(x, sr)
            print(json.dumps({"file": f, "speech_end_s": round(end, 3), "hits": [
                {"command": h.command, "word": h.word, "kind": h.kind, "transcript": h.transcript,
                 "latency_s": round(latency_after_speech_end(h, end), 3)} for h in hits]}))
        if sender:
            time.sleep(0.5)
            sender.close()
        return 0
    from .audio import MicHub

    hub = MicHub(sample_rate=args.sample_rate)
    lst = SafetyListener(on_hit=on_hit)
    stop = threading.Event()
    q = hub.subscribe()
    hub.start()
    print("listening for stop / hover / land / abort; Ctrl-C to quit", flush=True)
    try:
        lst.run_queue(q, args.sample_rate, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        hub.stop()
        if sender:
            sender.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
