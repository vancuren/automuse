"""Muse Voice Transcribe (muse-voice-transcribe-1.0) clients: realtime WebSocket and file upload.

Realtime protocol (dev.meta.ai/docs/api-reference/voice/realtime and .../voice/schemas):
- connect wss://api.meta.ai/v1/asr/realtime; the HTTP Authorization header is ignored.
- first text frame, within 10 s: {"authorization": {"accessToken": "Bearer <key>"},
  "audioEncoding": "PCM_24KHZ" | "PCM_16KHZ", "model": "muse-voice-transcribe-1.0",
  "mode": "PUSH_TO_TALK", "partialMode": "CUMULATIVE", "emitAudioProgress": false,
  "keywords": [...], "languageBias": ["English"]}
- ack: {"sessionId": "..."} (the only server frame without "type").
- binary frames: s16le mono PCM, 80 ms each (3,840 B at 24 kHz, 2,560 B at 16 kHz), at real-time pace.
  Below real time for 10 s, or more than 5 s of backlog, closes the socket with 1008.
- {"type": "endStream"} ends input; read until the server closes with 1000.
- server events: transcript {transcript, final, audioProcessedMs}, speechStart/speechEnd/speechComplete,
  audioProgress, error {message, errorType?, errorCode?}.
File endpoint: POST https://api.meta.ai/v1/asr/transcribe, Bearer header, multipart with a JSON part
"request" ({"model", "audioEncoding": "WAV", "keywords", "languageBias"}) and a WAV part "audio".
Response: {"sessionId", "transcript", "audioDurationMs", "turns": [...]}.

Latency: a session per utterance opens the socket when push-to-talk goes down, buffering mic audio
until the ack arrives, so the connect overlaps with speaking. WarmRealtimeASR goes further and keeps
one session open ahead of time, streaming 80 ms of silence until the next utterance (cost: the
$0.18/hour audio rate for as long as the agent runs).

The API key is read from MODEL_API_KEY and never logged or printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Optional

import numpy as np

from .asr_local import Transcript
from .audio import chunk_samples, float_to_pcm16, iter_chunks, load_wav, wav_bytes

log = logging.getLogger("mhd.agent.asr_meta")

REALTIME_URL = "wss://api.meta.ai/v1/asr/realtime"
FILE_URL = "https://api.meta.ai/v1/asr/transcribe"
MODEL = "muse-voice-transcribe-1.0"

# Vocabulary bias for drone missions: verbs, colors, subjects, clothing, distances.
DEFAULT_KEYWORDS = [
    "follow", "track", "hover", "land", "stop", "abort", "drone",
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray",
    "person", "man", "woman", "kid", "child", "dog", "cat", "car", "truck", "bike", "ball",
    "shirt", "jacket", "hoodie", "hat", "dress", "backpack", "umbrella", "balloon",
    "close", "closer", "far", "farther",
]


class AsrError(RuntimeError):
    pass


@dataclass
class RealtimeConfig:
    url: str = REALTIME_URL
    model: str = MODEL
    sample_rate: int = 24000                 # 24000 -> PCM_24KHZ, 16000 -> PCM_16KHZ
    mode: str = "PUSH_TO_TALK"
    keywords: list[str] = field(default_factory=lambda: list(DEFAULT_KEYWORDS))
    language_bias: list[str] = field(default_factory=lambda: ["English"])
    chunk_ms: int = 80
    open_timeout_s: float = 10.0
    final_timeout_s: float = 6.0             # endStream -> final transcript
    api_key_env: str = "MODEL_API_KEY"
    max_session_s: float = 50 * 60           # the server limit is 60 min; recycle warm sessions earlier

    @property
    def encoding(self) -> str:
        if self.sample_rate == 24000:
            return "PCM_24KHZ"
        if self.sample_rate == 16000:
            return "PCM_16KHZ"
        raise ValueError("Muse Voice Transcribe takes 16 kHz or 24 kHz PCM")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RealtimeConfig":
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "")
    if not key:
        raise AsrError(f"{env_name} is not set (source env.sh; the key lives in .env)")
    return key


class RealtimeSession:
    """One realtime session = one push-to-talk turn.

    Usage: await open(); await send_pcm(...) repeatedly; t = await finish(). Partial transcripts go to
    `on_partial(text)`. Timing fields (monotonic seconds) are kept for the latency gates.
    """

    def __init__(self, cfg: RealtimeConfig, on_partial: Optional[Callable[[str], None]] = None,
                 api_key: Optional[str] = None):
        self.cfg = cfg
        self.on_partial = on_partial
        self._key = api_key
        self.ws = None
        self.session_id: Optional[str] = None
        self.t_connect_start = 0.0
        self.t_open = 0.0              # ack received
        self.t_last_audio = 0.0
        self.t_end_sent = 0.0
        self.t_final = 0.0
        self.partials: list[str] = []
        self.finals: list[str] = []
        self.events: list[dict] = []   # every non-audio server message, for debugging
        self.error: Optional[str] = None
        self.close_code: Optional[int] = None
        self.bytes_sent = 0
        self._recv_task: Optional[asyncio.Task] = None
        self._final_evt = asyncio.Event()
        self._closed_evt = asyncio.Event()

    async def open(self) -> None:
        from websockets.asyncio.client import connect

        self.t_connect_start = time.monotonic()
        key = self._key or _api_key(self.cfg.api_key_env)
        self.ws = await asyncio.wait_for(
            connect(self.cfg.url, open_timeout=self.cfg.open_timeout_s, max_size=2 ** 22,
                    ping_interval=20, ping_timeout=20),
            timeout=self.cfg.open_timeout_s + 1)
        handshake = {
            "authorization": {"accessToken": f"Bearer {key}"},
            "audioEncoding": self.cfg.encoding,
            "model": self.cfg.model,
            "mode": self.cfg.mode,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        if self.cfg.keywords:
            handshake["keywords"] = list(self.cfg.keywords)
        if self.cfg.language_bias:
            handshake["languageBias"] = list(self.cfg.language_bias)
        await self.ws.send(json.dumps(handshake))
        del handshake, key  # do not keep the credential around
        # The ack is the first server frame; anything else (an error) means the handshake failed.
        raw = await asyncio.wait_for(self.ws.recv(), timeout=self.cfg.open_timeout_s)
        msg = json.loads(raw)
        if "type" not in msg and "sessionId" in msg:
            self.session_id = msg["sessionId"]
        else:
            await self._close_quietly()
            raise AsrError(f"handshake rejected: {_safe(msg)}")
        self.t_open = time.monotonic()
        self._recv_task = asyncio.create_task(self._receiver())

    async def _receiver(self) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ != "audioProgress":
                    self.events.append(msg)
                if typ == "transcript":
                    text = (msg.get("transcript") or "").strip()
                    if msg.get("final"):
                        self.finals.append(text)
                        self.t_final = time.monotonic()
                        self._final_evt.set()
                    else:
                        self.partials.append(text)
                        if self.on_partial and text:
                            try:
                                self.on_partial(text)
                            except Exception:
                                log.exception("on_partial failed")
                elif typ == "speechComplete":
                    # ENDPOINTING mode delivers the turn text here.
                    text = (msg.get("transcript") or "").strip()
                    if self.cfg.mode == "ENDPOINTING":
                        self.finals.append(text)
                        self.t_final = time.monotonic()
                        self._final_evt.set()
                elif typ == "error":
                    self.error = _safe(msg)
                    log.error("voice transcribe error: %s", self.error)
                    self._final_evt.set()
        except ConnectionClosed as e:
            self.close_code = e.rcvd.code if e.rcvd else None
        except Exception as e:  # malformed frame etc.
            self.error = f"receiver: {e.__class__.__name__}: {e}"
        finally:
            if self.close_code is None and self.ws is not None:
                self.close_code = getattr(self.ws, "close_code", None)
            self._closed_evt.set()
            self._final_evt.set()

    async def send_pcm(self, pcm: bytes) -> None:
        if self.ws is None:
            raise AsrError("session not open")
        await self.ws.send(pcm)
        self.bytes_sent += len(pcm)
        self.t_last_audio = time.monotonic()

    async def finish(self) -> Transcript:
        """Send endStream, wait for the final transcript (or close), return it."""
        if self.ws is None:
            raise AsrError("session not open")
        self.t_end_sent = time.monotonic()
        # Wait for a final that arrives AFTER endStream (PUSH_TO_TALK sends one per turn; if the server
        # ever sent one mid-stream, it must not end the wait early).
        n_before = len(self.finals)
        self._final_evt.clear()
        if self._closed_evt.is_set():
            self._final_evt.set()
        await self.ws.send(json.dumps({"type": "endStream"}))
        try:
            while len(self.finals) == n_before and not self._closed_evt.is_set() and not self.error:
                self._final_evt.clear()
                await asyncio.wait_for(self._final_evt.wait(),
                                       timeout=max(0.05, self.cfg.final_timeout_s - (time.monotonic() - self.t_end_sent)))
            # More finals may follow; the server closes with 1000 once it has flushed everything.
            if not self._closed_evt.is_set():
                try:
                    await asyncio.wait_for(self._closed_evt.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        except asyncio.TimeoutError:
            self.error = self.error or "timeout waiting for final transcript"
        await self._close_quietly()
        if self.error and not self.finals:
            raise AsrError(self.error)
        finals = [t for t in self.finals if t]
        if finals and all(finals[-1].startswith(f) for f in finals[:-1]):
            text = finals[-1]                 # cumulative finals: the last one holds everything
        else:
            text = " ".join(finals)
        text = text.strip()
        t_ref = self.t_end_sent
        return Transcript(
            text=text, latency_ms=((self.t_final or time.monotonic()) - t_ref) * 1000.0, model=self.cfg.model,
            details={"session_id": self.session_id, "connect_ms": (self.t_open - self.t_connect_start) * 1000.0,
                     "audio_s": self.bytes_sent / 2 / self.cfg.sample_rate, "finals": list(self.finals),
                     "partials": len(self.partials), "close_code": self.close_code})

    async def _close_quietly(self) -> None:
        if self.ws is not None:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=1.0)
            except Exception:
                pass
        if self._recv_task is not None and not self._recv_task.done():
            try:
                await asyncio.wait_for(self._recv_task, timeout=1.0)
            except Exception:
                self._recv_task.cancel()

    async def abort(self) -> None:
        await self._close_quietly()


def _safe(msg: dict) -> str:
    """Server messages never contain our key, but strip anything credential-shaped anyway."""
    m = {k: v for k, v in msg.items() if "auth" not in k.lower() and "token" not in k.lower()}
    return json.dumps(m)[:500]


def pcm_chunks_from_array(x: np.ndarray, sample_rate: int, chunk_ms: int = 80) -> list[bytes]:
    return [float_to_pcm16(c) for c in iter_chunks(x, sample_rate, chunk_ms)]


async def paced(chunks: Iterable[bytes], chunk_ms: int = 80) -> AsyncIterator[bytes]:
    """Yield chunks on an absolute real-time schedule (sleep only when ahead), like a microphone."""
    t0 = time.monotonic()
    for i, c in enumerate(chunks):
        delay = t0 + i * chunk_ms / 1000.0 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        yield c


class MetaRealtimeASR:
    """Push-to-talk over the realtime WebSocket, one session per utterance."""

    name = MODEL

    def __init__(self, cfg: Optional[RealtimeConfig] = None):
        self.cfg = cfg or RealtimeConfig()

    async def start_session(self, on_partial=None) -> RealtimeSession:
        s = RealtimeSession(self.cfg, on_partial=on_partial)
        await s.open()
        return s

    async def transcribe_stream(self, chunks: AsyncIterator[bytes], on_partial=None,
                                session: Optional[RealtimeSession] = None) -> Transcript:
        """Stream PCM chunks (already real-time paced, e.g. from the mic) and return the final text.
        The socket opens concurrently with the first chunks; audio is buffered until the ack."""
        opener = None
        if session is None:
            session = RealtimeSession(self.cfg, on_partial=on_partial)
            opener = asyncio.create_task(session.open())
        else:
            session.on_partial = on_partial
        backlog: list[bytes] = []
        try:
            async for c in chunks:
                if opener is not None and not opener.done():
                    backlog.append(c)
                    continue
                if opener is not None:
                    opener.result()          # raises if the handshake failed
                    opener = None
                while backlog:               # catch up: a burst well under the 5 s backlog limit
                    await session.send_pcm(backlog.pop(0))
                await session.send_pcm(c)
            if opener is not None:
                await opener
            for c in backlog:
                await session.send_pcm(c)
            t_speech_end = time.monotonic()
            t = await session.finish()
            t.details["end_of_speech_to_final_ms"] = ((session.t_final or time.monotonic()) - t_speech_end) * 1000.0
            return t
        except BaseException:
            if opener is not None and not opener.done():
                opener.cancel()
            await session.abort()
            raise

    async def transcribe_array(self, x: np.ndarray, sample_rate: int, realtime: bool = True,
                               on_partial=None, session: Optional[RealtimeSession] = None) -> Transcript:
        from .audio import resample

        if sample_rate != self.cfg.sample_rate:
            x = resample(x, sample_rate, self.cfg.sample_rate)
        chunks = pcm_chunks_from_array(x, self.cfg.sample_rate, self.cfg.chunk_ms)

        async def fast():
            for c in chunks:
                yield c

        src = paced(chunks, self.cfg.chunk_ms) if realtime else fast()
        return await self.transcribe_stream(src, on_partial=on_partial, session=session)

    async def transcribe_file(self, path: str | Path, realtime: bool = True, on_partial=None) -> Transcript:
        x, sr = load_wav(path)
        return await self.transcribe_array(x, sr, realtime=realtime, on_partial=on_partial)


class WarmRealtimeASR(MetaRealtimeASR):
    """Keeps one session open ahead of the next utterance, streaming 80 ms of silence at real-time
    pace so the server does not time it out. take() hands that session over (connect and handshake
    already done) and immediately starts warming the next one. Falls back to a fresh session if the
    warm one is missing or broken (for example after a network drop)."""

    def __init__(self, cfg: Optional[RealtimeConfig] = None):
        super().__init__(cfg)
        self._warm: Optional[RealtimeSession] = None
        self._keepalive: Optional[asyncio.Task] = None
        self._warming: Optional[asyncio.Task] = None
        self._closed = False
        self.warm_hits = 0
        self.warm_misses = 0

    async def start(self) -> None:
        self._closed = False
        self._schedule_warm()

    def _schedule_warm(self) -> None:
        if self._closed or (self._warming and not self._warming.done()):
            return
        self._warming = asyncio.create_task(self._warm_up())

    async def _warm_up(self) -> None:
        delay = 0.5
        while not self._closed:
            s = RealtimeSession(self.cfg)
            try:
                await s.open()
            except Exception as e:
                log.warning("warm voice session failed to open (%s); retrying in %.1f s", e.__class__.__name__, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            self._warm = s
            self._keepalive = asyncio.create_task(self._feed_silence(s))
            return

    async def _feed_silence(self, s: RealtimeSession) -> None:
        silence = b"\x00\x00" * chunk_samples(self.cfg.sample_rate, self.cfg.chunk_ms)
        t0 = time.monotonic()
        i = 0
        try:
            while self._warm is s:
                if s._closed_evt.is_set() or time.monotonic() - s.t_open > self.cfg.max_session_s:
                    break
                await s.send_pcm(silence)
                i += 1
                delay = t0 + i * self.cfg.chunk_ms / 1000.0 - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("warm voice session dropped (%s)", e.__class__.__name__)
        if self._warm is s:      # died or aged out while idle: replace it
            self._warm = None
            await s.abort()
            self._schedule_warm()

    async def take(self) -> Optional[RealtimeSession]:
        """The warm session, or None if there is none ready (the caller then opens a fresh one)."""
        s = self._warm
        if s is None or s._closed_evt.is_set():
            self.warm_misses += 1
            self._schedule_warm()
            return None
        self._warm = None
        if self._keepalive:
            self._keepalive.cancel()
            try:
                await self._keepalive
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive = None
        self.warm_hits += 1
        self._schedule_warm()
        return s

    async def transcribe_stream(self, chunks, on_partial=None, session=None) -> Transcript:
        if session is None:
            session = await self.take()
        return await super().transcribe_stream(chunks, on_partial=on_partial, session=session)

    async def close(self) -> None:
        self._closed = True
        for t in (self._warming, self._keepalive):
            if t and not t.done():
                t.cancel()
        if self._warm is not None:
            await self._warm.abort()
            self._warm = None


class MetaFileASR:
    """POST /v1/asr/transcribe with the whole utterance as WAV (simplest path, no pacing)."""

    name = MODEL + " (file)"

    def __init__(self, url: str = FILE_URL, model: str = MODEL, keywords: Optional[list[str]] = None,
                 language_bias: Optional[list[str]] = None, api_key_env: str = "MODEL_API_KEY",
                 timeout_s: float = 15.0):
        self.url = url
        self.model = model
        self.keywords = DEFAULT_KEYWORDS if keywords is None else keywords
        self.language_bias = ["English"] if language_bias is None else language_bias
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    async def transcribe_wav_bytes(self, wav: bytes) -> Transcript:
        import httpx

        req = {"model": self.model, "audioEncoding": "WAV"}
        if self.keywords:
            req["keywords"] = list(self.keywords)
        if self.language_bias:
            req["languageBias"] = list(self.language_bias)
        files = {"request": (None, json.dumps(req), "application/json"),
                 "audio": ("audio.wav", wav, "audio/wav")}
        headers = {"Authorization": f"Bearer {_api_key(self.api_key_env)}", "Accept": "application/json"}
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(self.url, files=files, headers=headers)
        ms = (time.monotonic() - t0) * 1000.0
        if r.status_code != 200:
            raise AsrError(f"transcribe HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        return Transcript(text=(body.get("transcript") or "").strip(), latency_ms=ms, model=self.model,
                          details={"session_id": body.get("sessionId"),
                                   "audio_ms": body.get("audioDurationMs"), "turns": body.get("turns")})

    async def transcribe_array(self, x: np.ndarray, sample_rate: int) -> Transcript:
        from .audio import resample

        if sample_rate not in (16000, 24000):
            x, sample_rate = resample(x, sample_rate, 24000), 24000
        return await self.transcribe_wav_bytes(wav_bytes(x, sample_rate))

    async def transcribe_file(self, path: str | Path) -> Transcript:
        x, sr = load_wav(path)
        return await self.transcribe_array(x, sr)


async def _cli(args) -> int:
    cfg = RealtimeConfig(sample_rate=args.rate)
    if args.endpoint == "file":
        t = await MetaFileASR().transcribe_file(args.file)
    else:
        asr = MetaRealtimeASR(cfg)
        t = await asr.transcribe_file(args.file, realtime=True, on_partial=lambda p: print("partial:", p))
    print(json.dumps({"text": t.text, "latency_ms": round(t.latency_ms, 1), "details": t.details}, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe a WAV with Muse Voice Transcribe")
    ap.add_argument("file")
    ap.add_argument("--endpoint", choices=["realtime", "file"], default="realtime")
    ap.add_argument("--rate", type=int, default=24000, choices=[16000, 24000])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
