"""Audio helpers shared by the safety listener, the ASR clients, and the service.

Conventions:
- Float audio is mono float32 in [-1, 1]. PCM is signed 16-bit little-endian mono bytes.
- Chunks are 80 ms by default (the Muse Voice Transcribe cookbook size: 3,840 bytes at 24 kHz,
  2,560 bytes at 16 kHz).
- The microphone is opened lazily: `sounddevice` is imported only inside MicHub.start(), so tests and
  file-input modes never touch the audio hardware (or its macOS permission prompt).
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger("mhd.agent.audio")

CHUNK_MS = 80


def chunk_samples(sample_rate: int, chunk_ms: int = CHUNK_MS) -> int:
    """Samples per chunk (1,920 at 24 kHz, 1,280 at 16 kHz for 80 ms)."""
    return sample_rate * chunk_ms // 1000


def float_to_pcm16(x: np.ndarray) -> bytes:
    """Float32 [-1, 1] -> int16 little-endian bytes (clipped)."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(b: bytes) -> np.ndarray:
    """int16 little-endian bytes -> float32 [-1, 1]."""
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float audio. Linear interpolation, with a short moving-average low-pass when
    downsampling. Good enough for speech recognition, not for music."""
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or len(x) == 0:
        return x
    if sr_out < sr_in:
        k = int(round(sr_in / sr_out))
        if k >= 2:
            kernel = np.ones(k, dtype=np.float32) / k
            x = np.convolve(x, kernel, mode="same").astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x), dtype=np.float64) / sr_in
    t_out = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def load_wav(path: str | Path, target_sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Read an audio file (WAV/AIFF/FLAC via soundfile) as mono float32, optionally resampled."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1).astype(np.float32)
    if target_sr and target_sr != sr:
        x, sr = resample(x, sr, target_sr), target_sr
    return x, int(sr)


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sample_rate, subtype="PCM_16")


def wav_bytes(x: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float audio as an in-memory 16-bit PCM WAV file."""
    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, dtype=np.float32), sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def speech_bounds(x: np.ndarray, sample_rate: int, threshold_db: float = -35.0,
                  frame_ms: int = 20) -> tuple[float, float]:
    """(start_s, end_s) of the region whose 20 ms RMS is within `threshold_db` of the loudest frame.
    Used by tests and benchmarks to find where speech ends in a `say` recording."""
    n = max(1, sample_rate * frame_ms // 1000)
    frames = len(x) // n
    if frames == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    idx = np.nonzero(db > threshold_db)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    return idx[0] * n / sample_rate, (idx[-1] + 1) * n / sample_rate


def iter_chunks(x: np.ndarray, sample_rate: int, chunk_ms: int = CHUNK_MS) -> Iterator[np.ndarray]:
    """Split float audio into fixed-size chunks (the last one may be shorter)."""
    n = chunk_samples(sample_rate, chunk_ms)
    for i in range(0, len(x), n):
        yield x[i:i + n]


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class MicHub:
    """One microphone stream fanned out to several consumers (safety listener, push-to-talk).

    Every 80 ms the PortAudio callback delivers an int16 chunk; the hub converts it to float32 and puts
    it on each subscriber queue (dropping the oldest chunk for a subscriber that falls behind). A short
    ring buffer keeps the last second so push-to-talk can include a little pre-roll.

    `sounddevice` is imported in start(), never at module import. On macOS a process without Microphone
    permission receives all-zero audio instead of an error; `suspect_no_permission` turns True after
    2 s of exact digital silence so the service can tell the operator.
    """

    def __init__(self, sample_rate: int = 24000, chunk_ms: int = CHUNK_MS, device=None,
                 preroll_s: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._ring: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(preroll_s * 1000 / chunk_ms)))
        self._stream = None
        self._zero_run_s = 0.0
        self.suspect_no_permission = False
        self.overflows = 0

    def start(self) -> None:
        import sounddevice as sd  # lazy: only the live mic path touches audio hardware

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            blocksize=chunk_samples(self.sample_rate, self.chunk_ms),
            device=self.device, callback=self._callback)
        self._stream.start()
        log.info("mic open: %d Hz, %d ms chunks, device=%s", self.sample_rate, self.chunk_ms, self.device)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status:
            self.overflows += 1
        x = pcm16_to_float(bytes(indata))
        # Exact zeros for 2 s almost always means macOS denied the Microphone permission.
        if not np.any(x):
            self._zero_run_s += len(x) / self.sample_rate
            if self._zero_run_s >= 2.0:
                self.suspect_no_permission = True
        else:
            self._zero_run_s = 0.0
            self.suspect_no_permission = False
        self.publish(x)

    def publish(self, x: np.ndarray) -> None:
        """Deliver one float chunk to every subscriber (also used to inject audio in tests)."""
        self._ring.append(x)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(x)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(x)

    def subscribe(self, maxsize: int = 250, preroll_s: float = 0.0) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        if preroll_s > 0:
            n = int(preroll_s * 1000 / self.chunk_ms)
            for x in list(self._ring)[-n:]:
                q.put_nowait(x)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


def play_file_into(hub: MicHub, x: np.ndarray, sample_rate: int, realtime: bool = True) -> None:
    """Feed a recording through a MicHub as if it came from the microphone (dry runs and tests)."""
    if sample_rate != hub.sample_rate:
        x = resample(x, sample_rate, hub.sample_rate)
    t0 = time.monotonic()
    step = hub.chunk_ms / 1000.0
    for i, chunk in enumerate(iter_chunks(x, hub.sample_rate, hub.chunk_ms)):
        if realtime:
            delay = t0 + i * step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        hub.publish(chunk)
