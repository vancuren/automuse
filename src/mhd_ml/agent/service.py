"""The voice and intent agent process (PLAN.md 4.6).

    $MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000 --asr meta --intent auto \
        [--glimmer-url https://api.fireworks.ai/inference/v1 --glimmer-model accounts/<a>/deployedModels/<d>]

What it does:
- Connects to the backend at /ws/agent?role=agent (reconnecting) and listens for AgentEvents relayed
  from the UI: ptt_down starts recording the microphone, ptt_up ends it.
- Push-to-talk: audio -> ASR (Muse Voice Transcribe realtime, streamed while the button is held; or
  Omnilingual locally with --asr local) -> hedged intent router (Glimmer + Spark, keyword fallback)
  -> POST /mission (WebSocket fallback) with the model that answered and its latency -> spoken reply.
  If Voice Transcribe fails (network down), the same audio is transcribed locally instead.
- Emits AgentEvents for the UI: partial, final, intent, reply, error, safety_word, status.
- Runs the local safety listener on the same microphone the whole time (every mode): stop / hover /
  land / abort go straight to POST /command, without any language model.
- Voice never takes off and never sends an E-stop. Only follow / hover / land intents become
  Missions; clarify and refuse are spoken and shown, and change nothing in flight.

Settings live in config/agent.yaml; flags override them. --simulate FILE replays a WAV as one
push-to-talk utterance (dry runs without a microphone or UI).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mhd.contracts import AgentEvent, Intent, Mission, OperatorCommand, now_ms
from mhd.intent_spec import build_reply

from .asr_local import LocalASR, Transcript
from .asr_meta import MetaFileASR, MetaRealtimeASR, RealtimeConfig, WarmRealtimeASR
from .audio import MicHub, float_to_pcm16, load_wav, play_file_into, resample
from .backend_link import BackendLink
from .intent import HedgedRouter, glimmer_client, spark_client
from .safety_listener import CommandSender, SafetyConfig, SafetyListener
from .tts import Speaker
from .words import SafetyHit, safety_action_for_word

log = logging.getLogger("mhd.agent.service")

SORRY = "Sorry, I didn't catch that."


def tree_root() -> Path:
    return Path(os.environ.get("MHD_TREE") or Path(__file__).resolve().parents[3])


def load_config(path: Optional[str | Path] = None) -> dict:
    import yaml

    p = Path(path) if path else tree_root() / "config" / "agent.yaml"
    if not p.exists():
        log.warning("config %s not found; using built-in defaults", p)
        return {}
    return yaml.safe_load(p.read_text()) or {}


def apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(cfg))   # deep copy
    for k in ("backend", "asr", "intent"):
        v = getattr(args, k, None)
        if v:
            cfg[k] = v
    g = cfg.setdefault("glimmer", {})
    if args.glimmer_url:
        g["base_url"] = args.glimmer_url
    if args.glimmer_model:
        g["model"] = args.glimmer_model
    if args.glimmer_label:
        g["label"] = args.glimmer_label
    if args.glimmer_url and args.glimmer_url.startswith("http://127.0.0.1"):
        g.setdefault("api_key_env", None)
        if g.get("api_key_env") == "FIREWORKS_API_KEY":
            g["api_key_env"] = None          # local servers take no key
    if args.tts:
        cfg.setdefault("tts", {})["engine"] = args.tts
    if args.mute:
        cfg.setdefault("tts", {})["mute"] = True
    if args.no_safety:
        cfg.setdefault("safety", {})["enabled"] = False
    if args.mic_rate:
        cfg.setdefault("mic", {})["sample_rate"] = args.mic_rate
    if args.asr_endpoint:
        cfg.setdefault("voice_transcribe", {})["endpoint"] = args.asr_endpoint
    if args.no_warm:
        cfg.setdefault("voice_transcribe", {})["warm_session"] = False
    return cfg


@dataclass
class PttState:
    t_down_ms: float
    q: "queue.Queue[np.ndarray]"
    done: asyncio.Event
    chunks: list = field(default_factory=list)      # float audio at the hub rate, for fallbacks
    task: Optional[asyncio.Task] = None
    t_up_ms: Optional[float] = None


class AgentService:
    def __init__(self, cfg: dict, hub: Optional[MicHub] = None, start_mic: bool = True,
                 speaker: Optional[Speaker] = None, metrics_path: Optional[Path] = None):
        self.cfg = cfg
        mic = cfg.get("mic", {})
        self.hub = hub or MicHub(sample_rate=int(mic.get("sample_rate", 24000)), device=mic.get("device"))
        self.start_mic = start_mic
        self.speaker = speaker
        self.metrics_path = metrics_path or (tree_root() / "logs" / "agent" / "ptt_metrics.jsonl")
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.link: Optional[BackendLink] = None
        self.asr = None
        self.local_asr: Optional[LocalASR] = None
        self.router: Optional[HedgedRouter] = None
        self.listener: Optional[SafetyListener] = None
        self.sender: Optional[CommandSender] = None
        self._safety_stop = threading.Event()
        self._safety_thread: Optional[threading.Thread] = None
        self._ptt: Optional[PttState] = None
        self._pipeline_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.events: list[dict] = []          # every AgentEvent emitted (tests, debugging)
        self.missions: list[Mission] = []
        self.safety_hits: list[dict] = []
        self.stopped = asyncio.Event()

    # ---- setup --------------------------------------------------------------------------------
    def _build_asr(self):
        vt = self.cfg.get("voice_transcribe", {})
        if self.cfg.get("asr", "meta") == "local":
            return self._local()
        if vt.get("endpoint", "realtime") == "file":
            return MetaFileASR(url=vt.get("file_url", "https://api.meta.ai/v1/asr/transcribe"),
                               model=vt.get("model", "muse-voice-transcribe-1.0"),
                               keywords=vt.get("keywords"), language_bias=vt.get("language_bias"))
        rc = RealtimeConfig(url=vt.get("url", "wss://api.meta.ai/v1/asr/realtime"),
                            model=vt.get("model", "muse-voice-transcribe-1.0"),
                            sample_rate=self.hub.sample_rate if self.hub.sample_rate in (16000, 24000) else 24000,
                            mode=vt.get("mode", "PUSH_TO_TALK"),
                            final_timeout_s=float(vt.get("final_timeout_s", 6.0)))
        if vt.get("keywords") is not None:
            rc.keywords = list(vt["keywords"])
        if vt.get("language_bias") is not None:
            rc.language_bias = list(vt["language_bias"])
        return WarmRealtimeASR(rc) if vt.get("warm_session", True) else MetaRealtimeASR(rc)

    def _local(self) -> LocalASR:
        if self.local_asr is None:
            self.local_asr = LocalASR()
        return self.local_asr

    def _build_router(self) -> HedgedRouter:
        mode = self.cfg.get("intent", "auto")
        g, s, r = self.cfg.get("glimmer", {}), self.cfg.get("spark", {}), self.cfg.get("router", {})
        glimmer = None
        if g.get("model") and mode in ("auto", "glimmer"):
            glimmer = glimmer_client(g.get("base_url", "https://api.fireworks.ai/inference/v1"), g["model"],
                                     label=g.get("label", "glimmer-30b+lora"), api_key_env=g.get("api_key_env"),
                                     response_format=g.get("response_format", "json_schema"),
                                     timeout_s=float(g.get("timeout_s", 8.0)), max_tokens=int(g.get("max_tokens", 256)))
        elif mode == "glimmer":
            log.warning("--intent glimmer but no Glimmer model configured; the keyword grammar will answer")
        spark = None
        if s.get("enabled", True) and mode in ("auto", "spark"):
            spark = spark_client(reasoning_effort=s.get("reasoning_effort", "minimal"),
                                 base_url=s.get("base_url", "https://api.meta.ai/v1"),
                                 model=s.get("model", "muse-spark-1.3"), timeout_s=float(s.get("timeout_s", 8.0)))
        return HedgedRouter(glimmer=glimmer, spark=spark, mode=mode,
                            prefer_window_s=float(r.get("glimmer_window_s", 2.5)),
                            total_timeout_s=float(r.get("total_timeout_s", 8.0)),
                            safety_fast_path=bool(r.get("safety_fast_path", True)))

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.link = BackendLink(self.cfg.get("backend", "ws://127.0.0.1:8000"), on_message=self._on_message)
        await self.link.start()
        self.router = self._build_router()
        self.asr = self._build_asr()
        if isinstance(self.asr, WarmRealtimeASR):
            await self.asr.start()
        if self.speaker is None:
            t = self.cfg.get("tts", {})
            self.speaker = Speaker(engine=t.get("engine", "mms"), voice=t.get("voice", "Samantha"),
                                   play=not t.get("mute", False)).start()
        if self.start_mic:
            try:
                self.hub.start()
            except Exception as e:
                log.error("microphone unavailable: %s", e)
                self._spawn(self.emit("error", f"Microphone unavailable ({e.__class__.__name__}); "
                                               "voice commands and the safety listener are off."))
        if self.cfg.get("safety", {}).get("enabled", True):
            self._start_safety()
        await self.link.wait_connected(timeout=3.0)
        await self.emit("status", "agent ready", data=self.status())
        self._spawn(self._status_heartbeat(), background=True)
        if self.start_mic:
            self._spawn(self._mic_watchdog(), background=True)

    def status(self) -> dict:
        return {"asr": self.cfg.get("asr", "meta"), "intent": self.cfg.get("intent", "auto"),
                "glimmer": self.router.glimmer.name if self.router and self.router.glimmer else None,
                "spark": self.router.spark.name if self.router and self.router.spark else None,
                "safety_listener": bool(self._safety_thread and self._safety_thread.is_alive()),
                "backend_connected": bool(self.link and self.link.connected),
                "mic_rate": self.hub.sample_rate, "mic_silent": self.hub.suspect_no_permission}

    async def _status_heartbeat(self, period_s: float = 5.0) -> None:
        """Re-send status every few seconds so a UI that connects later (or reconnects) sees it, and
        can tell when the agent or its safety listener has died."""
        while not self.stopped.is_set():
            await asyncio.sleep(period_s)
            await self.emit("status", "agent alive", data=self.status())

    def _start_safety(self) -> None:
        try:
            self.listener = SafetyListener(SafetyConfig.from_dict(self.cfg.get("safety")),
                                           on_hit=self._on_safety_hit)
        except Exception as e:
            log.error("SAFETY LISTENER FAILED TO START: %s", e)
            self._spawn(self.emit("error", f"Safety listener failed to start: {e}"))
            return
        self.sender = CommandSender(self.link.http_base, ws_send=self.link.send_threadsafe,
                                    on_result=self._on_safety_result)
        q = self.hub.subscribe(maxsize=500)
        self._safety_thread = threading.Thread(
            target=self.listener.run_queue, args=(q, self.hub.sample_rate, self._safety_stop),
            name="safety-listener", daemon=True)
        self._safety_thread.start()
        log.info("safety listener running")

    async def _mic_watchdog(self) -> None:
        warned = False
        while not self.stopped.is_set():
            await asyncio.sleep(1.0)
            if self.hub.suspect_no_permission and not warned:
                warned = True
                await self.emit("error", "The microphone is delivering pure silence: grant Microphone "
                                         "permission to this terminal app (System Settings > Privacy).")

    def _spawn(self, coro, background: bool = False) -> asyncio.Task:
        t = asyncio.create_task(coro, name=("bg-" if background else "") + getattr(coro, "__name__", "task"))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    # ---- events -------------------------------------------------------------------------------
    async def emit(self, kind: str, text: Optional[str] = None, data: Optional[dict] = None) -> None:
        ev = AgentEvent(kind=kind, text=text, data=data, ts_ms=now_ms())
        self.events.append(ev.model_dump())
        if self.link is not None:
            await self.link.send(ev)

    async def _on_message(self, msg: dict) -> None:
        if msg.get("type") != "agent_event":
            return                                   # mission_ack, cmd_ack, candidates: nothing to do
        kind = msg.get("kind")
        if kind == "ptt_down":
            await self.ptt_down()
        elif kind == "ptt_up":
            await self.ptt_up()
        elif kind == "status":
            # The backend relays autopilot announcements ("Got them. Following.", "I lost the subject.
            # Looking around.") as status events; the agent owns the voice, so it says them.
            data = msg.get("data") or {}
            if data.get("kind") == "announce" and msg.get("text") and \
                    self.cfg.get("tts", {}).get("speak_announcements", True):
                if _has_hover_word(msg["text"]):
                    log.info("not speaking announcement %r: the listener would hear a hover word", msg["text"])
                else:
                    self.speaker.speak(msg["text"], interrupt=False)

    # ---- safety -------------------------------------------------------------------------------
    def _on_safety_hit(self, hit: SafetyHit) -> None:          # listener thread
        # Our own reply coming back through the microphone ("Landing now." contains "landing"): ignore a
        # safety word that is part of a line the speaker played in the last 1.5 s. Anything else the
        # operator says during a reply still goes through.
        audible = getattr(self.speaker, "audible_words", None)
        if audible is not None and self.cfg.get("safety", {}).get("ignore_own_voice", True):
            try:
                aw = audible()
                # Only land is ever skipped, and only while a line with a land word is audible
                # ("Landing now.", possibly heard as "land in now"): we are already landing. A hover
                # word is never skipped, so an operator's "stop" always goes through; instead the
                # agent never speaks a hover word (see _speakable).
                land_echo = hit.action == "land" and any(
                    (safety_action_for_word(w, fuzzy=False) or ("",))[0] == "land" for w in aw)
                if land_echo:
                    log.info("safety word %r ignored: it is the agent's own voice", hit.word)
                    self.safety_hits.append({"hit": hit.as_dict(), "result": {"ok": False, "via": "echo"}})
                    return
            except Exception:
                log.exception("echo check failed; sending the command anyway")
        self.sender.submit(hit)                                # first: get the command out
        if self.loop is not None:
            self.loop.call_soon_threadsafe(lambda: self._spawn(self._safety_feedback(hit)))

    async def _safety_feedback(self, hit: SafetyHit) -> None:
        await self.emit("safety_word", hit.word, data={"command": hit.command, "action": hit.action,
                                                       "transcript": hit.transcript, "kind": hit.kind,
                                                       "decode_ms": round(hit.compute_ms, 1)})
        intent = Intent(action=hit.action, subject_phrase="", category="none", color="none",
                        distance="medium", reason_code="none")
        self.speaker.speak(build_reply(intent))

    def _on_safety_result(self, hit: SafetyHit, result: dict) -> None:   # sender thread
        self.safety_hits.append({"hit": hit.as_dict(), "result": result})
        if self.loop is None:
            return
        if not result.get("ok"):
            msg = f"Safety command {hit.command} could not reach the backend!"
        elif result.get("accepted") is False:
            msg = f"The governor refused {hit.command}: {result.get('reason') or 'no reason given'}"
        else:
            return
        self.loop.call_soon_threadsafe(lambda: self._spawn(self.emit("error", msg)))

    # ---- push-to-talk -------------------------------------------------------------------------
    async def ptt_down(self) -> None:
        if self._ptt is not None:
            return                                   # already recording; ignore repeats
        preroll = float(self.cfg.get("mic", {}).get("preroll_s", 0.3))
        st = PttState(t_down_ms=now_ms(), q=self.hub.subscribe(maxsize=1500, preroll_s=preroll),
                      done=asyncio.Event())
        self._ptt = st
        if isinstance(self.asr, MetaRealtimeASR):
            st.task = asyncio.create_task(self._stream_to_meta(st))
        self._spawn(self._ptt_watchdog(st), background=True)
        log.info("push-to-talk down")

    async def _ptt_watchdog(self, st: PttState) -> None:
        """A lost ptt_up (UI crash, dropped socket) must not record forever."""
        max_s = float(self.cfg.get("mic", {}).get("max_ptt_s", 15.0))
        await asyncio.sleep(max_s)
        if self._ptt is st:
            log.warning("push-to-talk held for %.0f s without release; ending the utterance", max_s)
            await self.emit("status", f"Push-to-talk released automatically after {max_s:.0f} s")
            await self.ptt_up()

    async def _chunks(self, st: PttState):
        """Mic chunks for this utterance as PCM at the ASR rate, until ptt_up and the queue is empty."""
        asr_rate = self.asr.cfg.sample_rate if isinstance(self.asr, MetaRealtimeASR) else self.hub.sample_rate
        while True:
            try:
                x = st.q.get_nowait()
            except queue.Empty:
                if st.done.is_set():
                    return
                await asyncio.sleep(0.01)
                continue
            st.chunks.append(x)
            if asr_rate != self.hub.sample_rate:
                x = resample(x, self.hub.sample_rate, asr_rate)
            yield float_to_pcm16(x)

    async def _stream_to_meta(self, st: PttState) -> Transcript:
        def partial(text: str) -> None:
            self._spawn(self.emit("partial", text))
        return await self.asr.transcribe_stream(self._chunks(st), on_partial=partial)

    async def _drain(self, st: PttState) -> None:
        async for _ in self._chunks(st):
            pass

    async def ptt_up(self) -> None:
        st = self._ptt
        if st is None:
            return
        self._ptt = None
        st.t_up_ms = now_ms()
        await asyncio.sleep(0.09)          # the mic delivers 80 ms chunks; take the last one
        st.done.set()
        self.hub.unsubscribe(st.q)
        log.info("push-to-talk up after %.1f s", (st.t_up_ms - st.t_down_ms) / 1000)
        self._spawn(self._finish_utterance(st))

    async def _finish_utterance(self, st: PttState) -> None:
        t, err = None, None
        try:
            if st.task is not None:
                t = await st.task
            else:
                await self._drain(st)
                audio = np.concatenate(st.chunks) if st.chunks else np.zeros(0, np.float32)
                if isinstance(self.asr, MetaFileASR):
                    t = await self.asr.transcribe_array(audio, self.hub.sample_rate)
                else:
                    t = await asyncio.get_running_loop().run_in_executor(
                        None, self._local().transcribe, audio, self.hub.sample_rate)
        except Exception as e:
            err = f"{e.__class__.__name__}: {e}"[:200]
            log.warning("ASR failed: %s", err)
        if (t is None or not t.text) and not isinstance(self.asr, LocalASR):
            # Cloud ASR failed or heard nothing: try the same audio locally (works offline). If the
            # stream failed early, the rest of the utterance is still queued: collect it first.
            await self._drain(st)
        if (t is None or not t.text) and not isinstance(self.asr, LocalASR) and st.chunks:
            try:
                audio = np.concatenate(st.chunks)
                t2 = await asyncio.get_running_loop().run_in_executor(
                    None, self._local().transcribe, audio, self.hub.sample_rate)
                if t2.text:
                    await self.emit("status", "Voice Transcribe unavailable; used local Omnilingual ASR",
                                    data={"error": err})
                    t = t2
            except Exception as e:
                log.warning("local ASR fallback failed: %s", e)
        t_final_ms = now_ms()
        if t is None or not t.text:
            await self.emit("error", f"No transcript ({err or 'empty'})")
            self.speaker.speak(SORRY)
            return
        await self.emit("final", t.text, data={"model": t.model, "asr_ms": round(t_final_ms - st.t_up_ms, 1),
                                               "details": _jsonable(t.details)})
        async with self._pipeline_lock:
            await self._handle_transcript(t.text, st, t_final_ms, t.model)

    async def _handle_transcript(self, text: str, st: PttState, t_final_ms: float, asr_model: str) -> None:
        tcfg = self.cfg.get("tts", {})
        ack_after = tcfg.get("ack_after_s", 1.2)
        acked = False

        async def ack():
            nonlocal acked
            await asyncio.sleep(float(ack_after))
            acked = True
            self.speaker.speak(tcfg.get("ack_text", "Okay."))

        ack_task = asyncio.create_task(ack()) if ack_after else None
        try:
            res = await self.router.route(text)
        finally:
            if ack_task:
                ack_task.cancel()
        intent = res.intent
        reply = build_reply(intent)
        t_intent_ms = now_ms()
        flies = intent.action in ("follow", "hover", "land")   # clarify / refuse change nothing in flight
        mission_id = f"m-{uuid.uuid4().hex[:8]}" if flies else None
        await self.emit("intent", intent.action, data={
            "intent": intent.model_dump(), "model": res.model, "latency_ms": round(res.latency_ms, 1),
            "reason": res.reason, "attempts": res.attempts, "transcript": text, "reply": reply,
            "mission_id": mission_id})
        delivery = None
        if flies:
            if intent.action in ("land", "hover"):
                # Same path as the safety listener: the governor acts on the command immediately.
                cmd = OperatorCommand(command="land" if intent.action == "land" else "pause")
                r = await self.link.deliver("/command", cmd)
                if not r.get("ok"):
                    await self.emit("error", f"Command {cmd.command} could not reach the backend!")
                elif r.get("accepted") is False:
                    await self.emit("error", f"The governor refused {cmd.command}: {r.get('reason')}")
            mission = Mission(mission_id=mission_id, intent=intent, transcript=text,
                              model=res.model, latency_ms=round(res.latency_ms, 1), reply=reply)
            self.missions.append(mission)
            delivery = await self.link.deliver("/mission", mission)
            if not delivery.get("ok"):
                await self.emit("error", "Mission could not reach the backend")
            elif delivery.get("accepted") is False:
                await self.emit("error", f"The backend refused the mission: {delivery.get('reason')}")
        await self.emit("reply", reply)
        self.speaker.speak(_speakable(reply, intent.action), interrupt=not acked)   # "Okay." finishes first
        rec = {"t_up_to_final_ms": round(t_final_ms - st.t_up_ms, 1), "acked": acked,
               "final_to_intent_ms": round(t_intent_ms - t_final_ms, 1),
               "t_up_to_reply_queued_ms": round(now_ms() - st.t_up_ms, 1),
               "asr": asr_model, "intent_model": res.model, "reason": res.reason, "action": intent.action,
               "transcript": text, "delivered": delivery}
        log.info("utterance: %s", rec)
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a") as f:
                f.write(json.dumps(_jsonable(rec)) + "\n")
        except Exception:
            pass

    # ---- dry runs -----------------------------------------------------------------------------
    async def simulate_file(self, path: str | Path, realtime: bool = True) -> None:
        """Replay a WAV through the mic hub as one push-to-talk utterance."""
        x, sr = load_wav(path)
        await self.ptt_down()
        await asyncio.get_running_loop().run_in_executor(None, play_file_into, self.hub, x, sr, realtime)
        await self.ptt_up()

    async def wait_idle(self, timeout: float = 20.0) -> None:
        """Wait until no utterance is being recorded or processed."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            busy = [t for t in self._tasks if not t.get_name().startswith("bg-")]
            if not busy and self._ptt is None:
                return
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self.stopped.set()
        self._safety_stop.set()
        if self._safety_thread:
            self._safety_thread.join(timeout=2)
        if self.sender:
            self.sender.close()
        try:
            self.hub.stop()
        except Exception:
            pass
        if isinstance(self.asr, WarmRealtimeASR):
            await self.asr.close()
        if self.router:
            await self.router.aclose()
        for t in list(self._tasks):
            t.cancel()
        if self.link:
            await self.link.close()
        if self.speaker:
            self.speaker.close()


def _has_hover_word(text: str) -> bool:
    """Would the safety listener hear a hover word ("stop", "hold", ...) in this line?"""
    from .words import words

    return any((safety_action_for_word(w, fuzzy=True) or ("",))[0] == "hover" for w in words(text))


def _speakable(reply: str, action: str) -> str:
    """The line to say out loud. A reply like "Following the person by the stop sign." would come back
    through the microphone as "stop" and pause the follow it just started, so say it without the
    subject instead. (Mission.reply keeps the full text for the UI.)"""
    if action == "follow" and _has_hover_word(reply):
        return "Following now."
    return reply


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Voice and intent agent")
    ap.add_argument("--config", default=None, help="YAML settings (default config/agent.yaml)")
    ap.add_argument("--backend", help="backend WebSocket base, e.g. ws://127.0.0.1:8000")
    ap.add_argument("--asr", choices=["meta", "local"])
    ap.add_argument("--intent", choices=["auto", "glimmer", "spark", "keyword"])
    ap.add_argument("--glimmer-url", help="OpenAI-compatible base URL for Glimmer")
    ap.add_argument("--glimmer-model", help="Glimmer model or deployment name")
    ap.add_argument("--glimmer-label", help="name recorded in Mission.model (glimmer-30b+lora, glimmer-30b)")
    ap.add_argument("--asr-endpoint", choices=["realtime", "file"])
    ap.add_argument("--no-warm", action="store_true", help="open a Voice Transcribe session per utterance")
    ap.add_argument("--tts", choices=["mms", "say", "none"])
    ap.add_argument("--mute", action="store_true", help="synthesize replies but do not play them")
    ap.add_argument("--no-safety", action="store_true", help="disable the local safety listener")
    ap.add_argument("--no-mic", action="store_true", help="do not open the microphone (dry runs)")
    ap.add_argument("--mic-rate", type=int, choices=[16000, 24000])
    ap.add_argument("--simulate", nargs="*", help="replay WAV file(s) as push-to-talk utterances, then exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


async def amain(args: argparse.Namespace) -> int:
    cfg = apply_cli(load_config(args.config), args)
    svc = AgentService(cfg, start_mic=not args.no_mic)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await svc.start()
    try:
        if args.simulate:
            svc.speaker.wait_ready(30)
            for f in args.simulate:
                await svc.simulate_file(f)
                await svc.wait_idle()
                svc.speaker.wait_idle(15)
            for e in svc.events:
                print(json.dumps(e, default=str))
        else:
            print(f"agent running (backend {cfg.get('backend')}); Ctrl-C to stop", flush=True)
            await stop.wait()
    finally:
        await svc.stop()
    return 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
