# Workstream: voice and intent agent (PLAN.md 4.6, gates G0 voice and Spark)

Branch `ws/agent`, code in `src/mhd_ml/agent/`, tests in `tests/agent/`, settings in `config/agent.yaml`.
Built overnight 2026-11-03/04. Every number below says what else was running.

## What it does

```
 microphone (24 kHz mono, 80 ms chunks; sounddevice opened lazily)
   |-- safety listener (always on, local, every mode)
   |     silero VAD -> Omnilingual CTC-300M (+ English carrier phrase) -> safety-word match
   |     -> POST /command {"type":"cmd","command":"land"|"pause"}   (WS /ws/agent fallback)
   |     -> AgentEvent safety_word, spoken "Landing now." / "Holding position."
   |
   |-- push-to-talk (UI sends AgentEvent ptt_down / ptt_up on /ws/agent?role=ui; the backend relays)
         audio -> Muse Voice Transcribe realtime (streamed while the button is held)
                  (fallbacks: same audio through local Omnilingual; --asr local for edge mode)
         -> hedged router: Glimmer (Fireworks or local) and Spark 1.3 at once,
            Glimmer wins if valid within 2.5 s, else first valid, else keyword grammar;
            "stop"/"land" in the command skip the models entirely (safety fast path)
         -> follow / hover / land: POST /mission (WS fallback) with model + latency;
            hover / land also POST /command pause / land
            clarify / refuse: spoken and shown only, nothing changes in flight
         -> AgentEvents partial, final, intent, reply (+ error, status every 5 s)
         -> reply from mhd.intent_spec.build_reply, voiced by MMS-TTS (fallback `say`);
            "Okay." first if the intent takes over 1.2 s
   autopilot announcements relayed by the backend ("Got them. Following.") are spoken too.
```

Voice can never take off or E-stop: the agent only ever sends `pause` and `land` commands, and the
Intent schema has no takeoff or E-stop action. The gov backend also refuses anything else from
`source=agent`.

| Module | Role |
|---|---|
| `safety_listener.py` | VAD-gated listener; final decode per VAD segment plus partial decodes every 0.4 s while speech continues; `CommandSender` (HTTP first, WS fallback); CLI with `--file` and `--mic` |
| `words.py` | transcript normalization and safety-word matching (SAFETY_WORDS + inflections + observed CTC spellings + glued words + edit distance 1, with a near-miss list) |
| `asr_meta.py` | Muse Voice Transcribe: `RealtimeSession`, `MetaRealtimeASR` (session per utterance, audio buffered until the ack), `WarmRealtimeASR` (pre-opened session streaming 80 ms silence), `MetaFileASR` (POST /v1/asr/transcribe, multipart `request` JSON + `audio` WAV) |
| `asr_local.py` | Omnilingual CTC-300M int8 via sherpa-onnx, shared by the listener and `--asr local`; English carrier phrase |
| `intent.py` | keyword grammar, `ChatIntentClient` (any OpenAI-compatible server, streamed, stops at the first complete valid JSON), `glimmer_client`, `spark_client`, `HedgedRouter` |
| `tts.py` | MMS-TTS (`transformers.VitsModel`, lazy import, warm-up) played with sounddevice; `say` fallback |
| `backend_link.py` | `/ws/agent?role=agent` with reconnect; `deliver()` = POST then WS |
| `service.py` | the agent process |
| `mic_check.py` | morning microphone check (permission, levels, live listener) |
| `bench.py` | gate measurements |
| `fakes.py`, `mock_llm.py` | fake backend (8400-8409), fake Voice Transcribe (8485-8489), mock OpenAI server (8480-8484) |
| `make_test_audio.py` | `say` + `afconvert` test audio, synthetic propeller noise |

## Run it

Always `cd $MHD_TREE && source env.sh` first.

```bash
# The agent (backend must be up; it reconnects if not)
$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000 --asr meta --intent auto
#   with the tuned Glimmer on Fireworks:
$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000 --intent auto \
    --glimmer-url https://api.fireworks.ai/inference/v1 --glimmer-model accounts/<acct>/deployedModels/<id>
#   edge mode (no internet): local ASR, local Glimmer server, or keyword only
$MHD_ML_PY -m mhd_ml.agent.service --asr local --intent glimmer \
    --glimmer-url http://127.0.0.1:8080/v1 --glimmer-model <name> --glimmer-label glimmer-30b
$MHD_ML_PY -m mhd_ml.agent.service --asr local --intent keyword
#   other flags: --tts mms|say|none, --mute, --no-safety, --no-mic, --mic-rate 16000, --no-warm,
#   --asr-endpoint file, --config FILE, -v

# Dry run without UI or microphone: fake backend + replayed recordings (TTS muted)
$MHD_ML_PY -m mhd_ml.agent.fakes backend --port 8400 &
$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8400 --no-mic --mute \
    --simulate media/test_audio/commands/c01_24k.wav media/test_audio/commands/c07_24k.wav -v
kill %1

# Safety listener alone
$MHD_ML_PY -m mhd_ml.agent.safety_listener --file media/test_audio/safety/land_mid_16k.wav
$MHD_ML_PY -m mhd_ml.agent.safety_listener --mic --backend http://127.0.0.1:8000

# Morning microphone check (grant Microphone permission to the terminal app first)
$MHD_ML_PY -m mhd_ml.agent.mic_check --list
$MHD_ML_PY -m mhd_ml.agent.mic_check --seconds 20           # say a sentence, then "stop", then "land"
$MHD_ML_PY -m mhd_ml.agent.mic_check --ptt 4                # local + Meta transcription of 4 s

# Test audio (also generated on demand by the tests), tests, gates
$MHD_ML_PY -m mhd_ml.agent.make_test_audio
$MHD_ML_PY -m pytest tests/agent -q                          # offline; ~3 min
MHD_REAL_API=1 $MHD_ML_PY -m pytest tests/agent -q -k real   # + 2 real Voice Transcribe calls
$MHD_ML_PY -m mhd_ml.agent.bench spark --n 20                # ~$0.07
$MHD_ML_PY -m mhd_ml.agent.bench voice --n 10 [--rate 16000]
$MHD_ML_PY -m mhd_ml.agent.bench safety
$MHD_ML_PY -m mhd_ml.agent.bench e2e --intent auto           # 20 scripted commands, ~$0.03
$MHD_ML_PY scripts/agent/integration_smoke.py [--backend-tree ../gov]   # vs the real backend
```

## Test results

`$MHD_ML_PY -m pytest tests/agent -q`: **all pass** (see the final count in the report; 2 real-API tests
are opt-in and also pass with `MHD_REAL_API=1`).

- `test_keyword_grammar.py`: 62 grammar cases (follow with color, person + garment/object, distance,
  safety words, refusals, clarify, plurals) plus invariants.
- `test_intent_clients.py`: lenient parsing (reasoning text, `<think>`, code fences, case repair,
  rejects); clients against the mock server on 8480 (stream and non-stream, invalid, truncated,
  HTTP 500, timeout, Glimmer prompt, Spark request body: strict schema, `reasoning_effort`,
  `max_completion_tokens`, no temperature, no tools).
- `test_router.py`: both fast -> Glimmer; slow Glimmer -> Spark at the 2.5 s window; invalid Glimmer
  -> Spark without waiting; Glimmer at 2.8 s beats a slower Spark; both invalid and total timeout ->
  keyword; safety fast path calls no model; late answers logged.
- `test_safety_listener.py`: 18 safety clips detected within 1.5 s of speech end; 0 false triggers on
  20 neutral sentences (near misses: island, landscape, hand, gold, shop, top, over, cover, lens);
  silence, noise, 24 kHz input; one command per utterance; stop then land escalates; command
  delivery over HTTP and the WS fallback; synthetic propeller noise.
- `test_asr_meta.py`: fake realtime server on 8485-8488: handshake fields, PCM_24KHZ/PCM_16KHZ,
  80 ms chunks, every byte sent, real-time pacing, buffering until a slow ack, rejected handshake,
  missing key, warm session hand-over and replacement, fresh-session fallback, a final sent before
  endStream; file endpoint multipart against a local FastAPI fake; 2 opt-in real calls.
- `test_tts_and_local_asr.py`: MMS-TTS synthesis checked by transcribing it back with Omnilingual;
  `say` fallback to file; Speaker queue (no playback at night unless `MHD_TTS_PLAY=1`); MMS missing
  -> `say`; Omnilingual push-to-talk transcripts.
- `test_service.py`: the whole process with fake backend, mock Glimmer and fake Voice Transcribe:
  push-to-talk -> Mission over HTTP with model and latency and the events the UI reads; clarify sends
  no Mission; the listener alone lands the drone in real time within 1.5 s of speech end; Voice
  Transcribe down -> local ASR fallback; keyword mode; spoken acknowledgment; lost ptt_up watchdog;
  `--asr local`; relayed autopilot announcements are spoken.
- `scripts/agent/integration_smoke.py` against the gov workstream's real patched backend (its
  worktree, run read-only on my ports 8407-8409): **PASS**. Follow -> autopilot FIND with subject
  "person in blue shirt"; spoken "land" -> governor LANDING (`{"ok": true, "flight_state": "LANDING"}`).

## Measurements

Conditions for all cloud numbers: home Wi-Fi, model downloads finished (fetch.log ALL DONE 23:40:51),
measured 00:20-01:05 ET while the train workstream ran MLX LoRA training on the GPU, the perception
service ran, and the UI workstream drove Chrome through Playwright. Not yet repeated over the tether.

**G0 Spark (muse-spark-1.3, Chat Completions, strict json_schema, streamed, 20 varied commands per
effort, interleaved):** time to the complete, valid JSON

| effort | p50 | p95 | valid | right action | completion tokens (reasoning) | cost / 20 |
|---|---|---|---|---|---|---|
| minimal | 1.64 s | 2.47 s | 20/20 | 20/20 | 178 (139) | $0.026 |
| low | 3.58 s | 4.50 s | 20/20 | 20/20 | 421 (382) | $0.047 |

Gate "Spark p95 <= 4 s": **GO at minimal** (low would fail). Minimal is the default.

**G0 Voice Transcribe (`say` Samantha, button released 150 ms after speech ends):** end of speech to
final transcript

| path | 24 kHz p50 / p95 (n=10) | 16 kHz p50 / p95 (n=5) | exact transcripts |
|---|---|---|---|
| realtime, session per utterance | 226 / 267 ms | 245 / 253 ms | 10/10 |
| realtime, warm persistent session | 239 / 279 ms | 220 / 227 ms | 10/10 |
| file endpoint (POST /v1/asr/transcribe) | 913 / 1126 ms | 895 / 1246 ms | 8/10 (doubled words) |

endStream to final: p50 108 ms. Socket connect + handshake: p50 375 ms, hidden because the socket
opens at ptt_down while you speak. Gate "voice <= 1 s": **GO** (realtime). The warm session gave no
gain on good Wi-Fi; it stays on as insurance for a slow tethered link (`--no-warm` turns it off; it
costs the $0.18/hour audio rate while the agent runs).

**Push-to-talk acceptance (PLAN 4.6), 20 scripted commands** through the real service (warm Voice
Transcribe, Spark minimal since no Glimmer deployment existed yet, keyword fast path, MMS-TTS
synthesized but muted, fake backend):

- button release to start of the spoken reply: **p50 2.35 s, p95 3.35 s** (target 4 s / 6 s): pass.
- release to final transcript p50 210 ms; release to intent p50 2.11 s, p95 2.92 s.
- right mission: **19/20** (target 18/20). The miss: "follow the guy with the green backpack" ->
  "person with green backpack" (my checker wanted "man"; fine for SAM).

**Safety listener (local CPU, 2 threads, same concurrent load):** 18/18 safety clips detected, 0/20
false triggers; latency after speech end p50 ~0 s (partial decodes catch the word before the VAD
closes the segment), worst 0.64 s; one decode (with carrier) 0.22 s mean; listener CPU about a third
of one core while someone talks, near zero in silence. Through the whole service in real time, the
land command reached the backend within 1.5 s of speech end (`test_spoken_land_during_flight...`).

Synthetic propeller noise (four motor tones + harmonics + broadband, same seeds per setting):

| SNR | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| safety words detected (of 17) | 16 | 13 | 6 | 0 |

No false triggers at any SNR, nor on 30-60 s of noise alone. VAD thresholds 0.3-0.4 detected no
more and raised the worst latency to 1.35 s, so the default stays 0.5. Real prop noise must be
checked in Session B with `mic_check`.

Other: Omnilingual load 0.32-0.39 s, decode 0.15 s for a 2 s clip. MMS-TTS: torch import 2.3 s +
load 0.4 s at warm-up, then 0.12 s ("Landing now.") to 0.47 s (refusal line) per reply.

Spend for this workstream: about **$0.12** of the $3 cap (logged in `$MHD_HOME/logs/spend.jsonl`).

## Decisions made overnight

- **English carrier phrase for Omnilingual.** The CTC model has no language conditioning; short clips
  came back in the wrong script ("land" as Arabic, "time to land" as Thai, "stop, stop" as kana).
  Prepending a fixed `say` clip of "the command is" and dropping its tokens by timestamp fixed every
  case. Cached at `media/agent/carrier_16k.wav`, regenerated with `say` if missing.
- **VAD segments decoded with 0.3 s pre-roll** (the VAD clipped onsets: "freeze" became "preeze").
- **Partial decodes ignore the last word** (could be the start of "landscape") and the first word of
  a truncated window (could be the end of "island"). Final decodes use every word.
- **Land beats hover**; one command per action per VAD segment plus a 1 s debounce.
- **Fuzzy matching stays narrow:** edit distance 1, same first letter, 4+ letters, not a near-miss
  word. Distance 2 restricted to non-dictionary words was tested and rejected: it turned "shopp"
  (coffee shop) into stop. Observed CTC spellings are listed explicitly (`stolk`, `stog`, `preeze`,
  `aboard`, ...) and apply only in the listener, not the grammar ("stalk him" is a refusal).
- **Commands go HTTP first** (`POST /command` acknowledges with the flight state), WS fallback.
  Missions too. Voice hover = `pause` (the governor holds position), voice land/abort = `land`.
- **Only follow/hover/land become Missions**; clarify and refuse are spoken only.
- **Safety fast path in push-to-talk:** a transcript with a safety word answers instantly from the
  keyword grammar (consequence: "follow the person by the stop sign" hovers).
- **Spark:** Chat Completions, `response_format` json_schema strict, `reasoning_effort: minimal`,
  `max_completion_tokens: 1024` (reasoning counts against it), no temperature, no tools
  (`tool_choice` only supports auto, so strict schema is the reliable route). Streamed; the client
  stops reading at the first complete valid JSON object.
- **Router losers keep running in the background** (bounded by their timeouts) so their latency and
  validity are logged for the evaluation; they never change the decision.
- **Spoken "Okay."** if the intent takes over 1.2 s (the plan's early acknowledgment, done with a
  timer rather than the streamed subject phrase); the real reply waits for it instead of cutting it off.
- **Agent speaks autopilot announcements** relayed by the backend (`status` with `data.kind == "announce"`).
- **24 kHz capture by default** (PLAN), 16 kHz for headsets; if a device refuses the rate, the hub
  captures at the device rate and resamples.
- **Push-to-talk auto-release after 15 s** (lost `ptt_up`).

## Known gaps and risks

1. **Live microphone never exercised** (no permission overnight). Everything was tested by injecting
   audio into the same `MicHub` the mic feeds. Run `mic_check` in the morning; it prints ALL ZEROS if
   the terminal app lacks the Microphone permission (the service also emits an error event).
2. **Glimmer not exercised for real.** No Fireworks deployment existed at 01:00 and Fireworks has no
   serverless Glimmer (only `muse-spark-open`, which the plan rules out). The Glimmer client and the
   hedged timing are tested against the mock. `response_format: json_schema` is sent; if the
   Fireworks deployment or `mlx_lm.server` rejects it, set `glimmer.response_format` to `json_object`
   or `none` in `config/agent.yaml` (parsing tolerates prose around the JSON).
3. **Propeller noise:** detection falls off below about 10 dB SNR (table above). Use a headset or a
   mic close to the mouth for flights; the UI LAND button and Esc E-STOP remain the primary controls.
4. **Echo:** with laptop speakers, the listener hears the agent's own voice. "Landing now." contains
   "landing" and re-sends land, which is harmless (already landing). A subject phrase containing a
   safety word would be heard too. A headset avoids both.
5. **Known false lands (safe direction):** `SAFETY_WORDS` (frozen) maps "lend" and "lan" to land, so
   "lend me a hand" lands; in heavy noise "camera lens" was once heard as "camer land".
6. The Voice Transcribe file endpoint occasionally doubles a word ("woman woman"); realtime is the default.
7. MMS-TTS is CC-BY-NC (already noted in the plan). `--tts say` avoids it.
8. Not done: multi-step missions via Spark tool calls (Tier 3); hands-free ENDPOINTING mode.

## What the integrator must wire up

- **Backend:** nothing new. The gov hub already implements `/ws/agent?role=agent|ui` relaying,
  `POST /command`, `POST /mission`, `mission_ack`/`cmd_ack`, and relays governor events as `status`
  AgentEvents. Verified end to end by `scripts/agent/integration_smoke.py`.
- **UI:** already sends `{"type":"agent_event","kind":"ptt_down"|"ptt_up"}` on `/ws/agent?role=ui` and
  reads the agent's events. Event payloads the agent emits:
  - `status` every 5 s: `data = {asr, intent, glimmer, spark, safety_listener, backend_connected, mic_rate, mic_silent}`
  - `partial` (text), `final` (text, `data.model`, `data.asr_ms`, `data.details`)
  - `intent` (text = action, `data = {intent, model, latency_ms, reason, attempts, transcript, reply, mission_id}`)
  - `reply` (text), `error` (text), `safety_word` (text = word, `data = {command, action, transcript, kind, decode_ms}`)
- **Start order:** backend, then `$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000`.
  The agent needs `MODEL_API_KEY` (and `FIREWORKS_API_KEY` for Glimmer on Fireworks) from `.env`.
- **Glimmer:** put the deployment's model id in `config/agent.yaml` (`glimmer.model`) or pass
  `--glimmer-model`. Mission.model then reads `glimmer-30b+lora` when Glimmer answers.
- **Memory/GPU:** the agent uses CPU only (Omnilingual 2 threads, MMS-TTS 2 threads) and loads no
  MLX model. Peak RSS measured 1.9 GB (1.2 GB after Omnilingual + VAD, 1.8 GB after MMS-TTS + torch).
- **Contracts:** no changes needed. No frozen file was edited.
