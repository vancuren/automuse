"""Demo-only runner for the voice agent (ml venv): the real AgentService with a simulated microphone.

    source env.sh
    $MHD_ML_PY scripts/demo_agent.py --config config/demo/agent.yaml --backend ws://127.0.0.1:8000 \
        --out logs/demo-<run>/agent < commands

It runs `mhd_ml.agent.service.AgentService` unchanged (Voice Transcribe, the hedged router with the
fine-tuned Glimmer, MMS-TTS, the always-on local stop/land listener), with three demo-only differences:

- No microphone is opened (no Microphone permission tonight, and nobody to speak). A feeder thread
  publishes a quiet room tone into the agent's MicHub in real time, 80 ms chunks, the way the microphone
  would, and mixes recorded utterances (WAV from `say` + `afconvert`) into it on request.
- Push-to-talk is driven by the UI: holding Space in the dashboard sends ptt_down / ptt_up, which the
  backend relays to the agent, exactly as in a flight. (`ptt <wav>` below lets the agent press it itself.)
- Replies are synthesized with MMS-TTS as usual, but written to <out>/tts_NN.wav instead of the
  speakers (it is the middle of the night); each line's start time is printed, so the demo can lay the
  agent's voice into the video's sound track at the moment it would have been heard.

Commands on stdin, one per line:
    play <wav>    mix the WAV into the microphone now. Open mic: only the safety listener acts on it,
                  unless push-to-talk is held (then it is also the push-to-talk utterance).
    ptt <wav>     the agent presses push-to-talk itself: ptt_down, play, ptt_up (no UI needed).
    quit          stop the agent and exit (EOF does the same).
stdout: one JSON line per event: agent_event (every AgentEvent the agent emits), agent_rx (messages
relayed to it: ptt_down, ptt_up, announcements), tts (a spoken line), mic (utterance start and end).
Times are `t_ms` on the shared monotonic clock (mhd.contracts.now_ms).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from mhd.contracts import now_ms
from mhd_ml.agent.audio import MicHub, chunk_samples, load_wav, resample, write_wav
from mhd_ml.agent.service import AgentService, apply_cli, build_arg_parser, load_config
from mhd_ml.agent.tts import Speaker

log = logging.getLogger("mhd.demo.agent")
_out_lock = threading.Lock()


def emit_line(rec: dict) -> None:
    rec.setdefault("t_ms", round(now_ms(), 1))
    with _out_lock:
        sys.stdout.write(json.dumps(rec, default=str) + "\n")
        sys.stdout.flush()


class FileSpeaker(Speaker):
    """The agent's Speaker, but each synthesized line goes to a WAV file (and takes as long as playing
    it would) instead of the loudspeaker."""

    def __init__(self, *a, out_dir: Path, **kw):
        super().__init__(*a, **kw)
        self.out_dir = out_dir
        self.n = 0

    def _line_text(self) -> str:
        return self.history[-1]["text"] if self.history else ""

    def _play(self, wav: np.ndarray, sr: int) -> None:
        self.n += 1
        p = self.out_dir / f"tts_{self.n:02d}.wav"
        write_wav(p, wav, sr)
        dur = len(wav) / float(sr)
        emit_line({"type": "tts", "engine": "mms", "text": self._line_text(), "file": str(p),
                   "duration_s": round(dur, 3)})
        time.sleep(dur)

    def _say(self, text: str) -> None:
        self.n += 1
        p = self.out_dir / f"tts_{self.n:02d}.aiff"
        try:
            subprocess.run(["say", "-v", self.voice, "-o", str(p), text], check=True, timeout=20)
            x, sr = load_wav(p)
            dur = len(x) / float(sr)
        except Exception as e:
            log.error("`say` to file failed: %s", e)
            return
        emit_line({"type": "tts", "engine": "say", "text": text, "file": str(p), "duration_s": round(dur, 3)})
        time.sleep(dur)


class MicFeeder(threading.Thread):
    """Publishes 80 ms chunks of quiet room tone into a MicHub in real time, with WAVs mixed in."""

    def __init__(self, hub: MicHub, level_dbfs: float = -60.0, seed: int = 7):
        super().__init__(name="demo-mic", daemon=True)
        self.hub = hub
        self.n = chunk_samples(hub.sample_rate, hub.chunk_ms)
        self.step = hub.chunk_ms / 1000.0
        self.amp = 10 ** (level_dbfs / 20.0)
        self.rng = np.random.default_rng(seed)
        self.q: "queue.Queue[tuple[np.ndarray, threading.Event, str]]" = queue.Queue()
        self.stop_ev = threading.Event()
        self.cur = None

    def play(self, x: np.ndarray, sr: int, name: str) -> threading.Event:
        if sr != self.hub.sample_rate:
            x = resample(x, sr, self.hub.sample_rate)
        ev = threading.Event()
        self.q.put((x.astype(np.float32), ev, name))
        return ev

    def run(self) -> None:
        t0, i = time.monotonic(), 0
        pos = 0
        while not self.stop_ev.is_set():
            i += 1
            delay = t0 + i * self.step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:                       # fell far behind (machine busy): resync, don't burst
                t0, i = time.monotonic(), 0
            chunk = self.rng.normal(0.0, self.amp, self.n).astype(np.float32)
            if self.cur is None:
                try:
                    self.cur = self.q.get_nowait()
                    pos = 0
                    emit_line({"type": "mic", "event": "start", "file": self.cur[2],
                               "duration_s": round(len(self.cur[0]) / self.hub.sample_rate, 3)})
                except queue.Empty:
                    pass
            if self.cur is not None:
                x, ev, name = self.cur
                seg = x[pos:pos + self.n]
                chunk[:len(seg)] += seg
                pos += self.n
                if pos >= len(x):
                    self.cur = None
                    emit_line({"type": "mic", "event": "end", "file": name})
                    ev.set()
            self.hub.publish(chunk)


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = apply_cli(load_config(args.config), args)
    t = cfg.get("tts", {})
    speaker = FileSpeaker(engine=t.get("engine", "mms"), voice=t.get("voice", "Samantha"), play=True,
                          out_dir=out_dir).start()
    hub = MicHub(sample_rate=int(cfg.get("mic", {}).get("sample_rate", 24000)))
    svc = AgentService(cfg, hub=hub, start_mic=False, speaker=speaker,
                       metrics_path=out_dir / "ptt_metrics.jsonl")

    orig_emit, orig_rx = svc.emit, svc._on_message

    async def emit(kind, text=None, data=None):
        emit_line({"type": "agent_event", "kind": kind, "text": text, "data": data})
        await orig_emit(kind, text, data)

    async def on_message(msg: dict):
        if msg.get("type") == "agent_event":
            emit_line({"type": "agent_rx", "kind": msg.get("kind"), "text": msg.get("text"),
                       "data": msg.get("data")})
        await orig_rx(msg)

    svc.emit, svc._on_message = emit, on_message      # instance attributes: start() wires these
    feeder = MicFeeder(hub, level_dbfs=args.room_dbfs)
    feeder.start()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await svc.start()
    ready = await loop.run_in_executor(None, speaker.wait_ready, 60)
    emit_line({"type": "ready", "tts_ready": ready, "tts_warmup_ms": speaker.warmup_ms, "status": svc.status()})

    async def play(path: str) -> None:
        x, sr = load_wav(path)
        ev = feeder.play(x, sr, path)
        await loop.run_in_executor(None, ev.wait, 60)

    async def commands() -> None:
        while not stop.is_set():
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            cmd, _, arg = line.strip().partition(" ")
            try:
                if cmd == "play":
                    await play(arg)
                elif cmd == "ptt":
                    await svc.ptt_down()
                    await asyncio.sleep(0.3)
                    await play(arg)
                    await asyncio.sleep(0.4)
                    await svc.ptt_up()
                elif cmd == "quit":
                    break
                elif cmd:
                    emit_line({"type": "error", "text": f"unknown command {cmd!r}"})
            except Exception as e:                    # keep the agent alive whatever the command did
                emit_line({"type": "error", "text": f"{cmd} {arg}: {e.__class__.__name__}: {e}"})
        stop.set()

    task = asyncio.create_task(commands())
    await stop.wait()
    task.cancel()
    await svc.wait_idle(timeout=15)
    await loop.run_in_executor(None, speaker.wait_idle, 15)
    feeder.stop_ev.set()
    await svc.stop()
    emit_line({"type": "stopped", "missions": [m.model_dump() for m in svc.missions],
               "safety_hits": svc.safety_hits, "tts": speaker.history})
    return 0


def main(argv=None) -> int:
    ap = build_arg_parser()
    ap.description = __doc__
    ap.formatter_class = argparse.RawDescriptionHelpFormatter
    ap.add_argument("--out", required=True, help="directory for TTS WAVs and the push-to-talk metrics")
    ap.add_argument("--room-dbfs", type=float, default=-60.0, help="level of the simulated room tone")
    args = ap.parse_args(argv)
    args.no_mic = True                                # never open the real microphone here
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
