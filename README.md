# Voice Follow Drone

**A $34 toy drone that follows a person or a car you describe by voice, built on Meta models.**

![The operator UI flying the digital twin: the spoken command, the mission from our fine-tuned Muse Glimmer, and the SAM 3.1 mask on the subject in FOLLOW](docs/img/demo_s1_2_voice_mission_follow.png)

*The operator UI flying the digital twin. The command (voiced by macOS `say` into the agent's
microphone input) went through Muse Voice Transcribe and our fine-tuned Muse Glimmer
(`glimmer-30b+lora` under the mission); SAM 3.1 outlines the subject, and the autopilot is following.*

Hold Space and say "follow the person in the blue shirt". Muse Voice Transcribe turns your words
into text, our fine-tuned Muse Glimmer turns the text into a mission, SAM 3.1 finds that person in
the drone's video, and a classical controller inside a safety governor turns a Loiley X69 to keep
them in frame. Say "land" at any time: a Meta speech model running on the laptop hears it, even
with no internet.

It is a consumer follow-me camera: a hands-free, voice-directed camera operator for creators,
coaches, and people with limited mobility. It picks the subject by appearance only ("the person in
the blue shirt", "the green car"), never by identity. Ask it to "follow John" and it answers "I
can't follow someone by identity. Tell me what they're wearing instead." It films only people who
agreed to be filmed, and it calls the person or thing it follows the *subject*.

> **Status, 2026-11-04 evening.** The real drone flew the whole chain. In the 16:45 ET flight it
> took off from the UI and heard "follow the person with the blue shirt" twice. Each time our
> fine-tuned Glimmer on the laptop turned the words into a mission, SAM 3.1 found the subject in
> the drone's video, and the autopilot turned the drone to keep them in frame. It was airborne for
> 102 s and landed from the Land button. It followed for 17 s of that, by turning only (it never
> flew forward), and lost the subject for 44 s. A third flight at 17:41 followed for 26 s of its
> 81 s in the air. The same chain runs end to end in a wire-level digital twin on one Mac
> (pictured above), which is where the measurements below come from.

**Demo video (3 min):** linked on the hackathon submission.

| The twin's chase view: the drone (center) turns to keep the walker in blue in frame | S2: a walker in red crosses between the drone and the subject in blue |
|---|---|
| ![Chase view of the twin](docs/img/demo_s1_chase_follow.png) | ![The S2 crossing in the operator UI](docs/img/demo_s2_crossing_follow.png) |
| **S3: "Drone, follow the green car."** The LoRA's mission is `green car`, and SAM 3.1 boxes the RC car, not the parked red one | **"Okay drone, please land now."** The local listener (Omnilingual ASR on the laptop, no language model) hears "land", and the governor lands |
| ![The S3 green car mission in the operator UI](docs/img/demo_s3_2_voice_mission_follow.png) | ![A spoken land command in the operator UI](docs/img/demo_s1_5_landing_voice.png) |

## How it works

1. **You speak.** Push-to-talk audio streams to Muse Voice Transcribe. The final transcript arrives
   about 0.23 s after you stop talking.
2. **A model picks the mission; it never flies.** Our LoRA-tuned Muse Glimmer 30B turns the
   transcript into a mission that must validate against a fixed JSON schema, with Muse Spark 1.3
   racing it as a fallback:
   `{"action":"follow","subject_phrase":"person in blue shirt","category":"person","color":"blue",...}`.
   Clarify and refuse answers are spoken and change nothing in flight.
3. **SAM 3.1 finds the subject** in the newest frame from the words alone, 1-2 times a second.
   OpenCV's VitTrack follows the box at camera rate, and a Kalman filter keyed on capture time
   smooths it.
4. **A deterministic autopilot steers.** Every 40 ms RC tick, a pure function turns the box into yaw
   and forward sticks. It runs inside the safety governor, which owns every packet: latched E-stop,
   re-asserted land, watchdogs, caps, and human override.
5. **The same packets fly the drone or the twin.** turbodrone's X69 driver (its files unchanged; our
   subclasses add capture timing and keepalive retries) sends real 28-byte UDP packets at 25 Hz and
   reads HEVC video over RTSP, from the X69 or from the twin. Our own X69 turned out to be a
   different hardware revision with a Lewei camera board: it takes the same 20-byte stick payload
   raw on UDP 50000, and `scripts/lewei_bridge.py` relays its H.264 video to RTSP on the Mac.
6. **"Stop" and "land" never wait for the cloud.** An always-on listener (Omnilingual ASR on the
   laptop) sends them straight to the governor.

```
 you: hold Space, "follow the person in the blue shirt"       you: "land" (works offline)
   |                                                            |
 agent (CPU)                                                  safety listener: silero VAD +
   Muse Voice Transcribe (Meta API) -> transcript             Omnilingual ASR CTC-300M -> land / pause
   Glimmer 30B + LoRA (Mac) | Muse Spark 1.3 (Meta API)         |
     race -> JSON mission (schema-checked); spoken replies      |
   | POST /mission                                              | POST /command
   v                                                            v
 drone backend: patched turbodrone FastAPI, :8000  <----  operator UI: patched turbodrone React, :5173
   hub --FindRequest-->  perception: newest frame -> SAM 3.1 (MLX on the Mac, or Meta API)
   <------SubjectObs---    -> VitTrack at camera rate -> Kalman filter on capture time
   safety governor, every 40 ms RC tick (the only path to the sticks):
     E-stop latch > land > human sticks > autopilot; staleness watchdogs; caps; wrong-way checks
     autopilot(SubjectObs) -> yaw, pitch    FIND -> CONFIRM -> FOLLOW -> DOUBT -> LOST -> SEARCH
   X69 RC adapter (turbodrone, unchanged) -> 28-byte UDP packets, 25 Hz
   |                                                 ^ HEVC 1280x720 over RTSP
   v                                                 |
 real: Loiley X69 at 172.16.11.1        twin: 127.0.0.1, same packets and video. X69 dynamics, Three.js
                                        world in Chrome, ffmpeg hevc_videotoolbox -> mediamtx, ground truth
```

| Process | What it does | Ports | Start |
|---|---|---|---|
| drone backend | patched turbodrone FastAPI: RC link, video in, safety governor with the autopilot, mission API, hub, flight recorder | 8000 | `scripts/run_backend.sh --mode twin` or `--mode real` |
| perception | newest frame to SAM 3.1, VitTrack and a Kalman filter; one `SubjectObs` per frame | client of 8000 | `$MHD_ML_PY -m mhd_ml.perception.service` |
| agent | safety listener, push-to-talk, Voice Transcribe, Glimmer/Spark router, spoken replies (ElevenLabs, MMS-TTS fallback) | client of 8000 | `$MHD_ML_PY -m mhd_ml.agent.service` |
| twin | X69 emulator: RC decoder, 100 Hz dynamics, Three.js renderer, HEVC/RTSP through mediamtx, ground truth | UDP 23458, HTTP/WS 8766, RTSP 8554 | `scripts/run_twin.sh` |
| operator UI | turbodrone's React UI shell with safety patches, redesigned: the camera with the subject outline, voice, mission, safety, the chase view, and flight replay | 5173 | `npm run dev` in `turbodrone/frontend` |
| Glimmer server (edge mode) | `mlx_lm.server`: Muse Glimmer 30B 4-bit plus our LoRA, OpenAI-compatible | 8080 | `scripts/serve_glimmer.sh` |

### Meta models

Six Meta models: three through the Meta Model API and four as open weights on the laptop (SAM 3.1
runs both ways), one of them fine-tuned by us.

| Model | ID | Runs on | Role |
|---|---|---|---|
| Muse Voice Transcribe | `muse-voice-transcribe-1.0` | Meta Model API, realtime WebSocket | push-to-talk speech to text |
| Muse Glimmer 30B + our LoRA | `mlx-community/Muse-Glimmer-30B-4bit` + adapter `glimmer-intent-v1` | this Mac (MLX, `mlx_lm.server`) | transcript to mission JSON. We trained the LoRA locally with MLX after Fireworks blocked our dataset and model uploads. Base Glimmer (`accounts/fireworks/models/muse-glimmer-30b`) also ran on a Fireworks H200 deployment for the evaluation. |
| Muse Spark 1.3 | `muse-spark-1.3` | Meta Model API (strict JSON schema, reasoning effort minimal) | races Glimmer for the mission; reworded training commands (labels are ours); evaluation baseline |
| SAM 3.1 | `sam-3.1` (Meta API); `mlx-community/sam3.1-bf16` (MLX) | this Mac by default, or the Meta Model API | finds the subject in the video from the words; the mask shown in the UI |
| Omnilingual ASR CTC-300M | `sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05` | this Mac (sherpa-onnx, CPU) | always-on "stop" / "land" listener; offline transcription fallback; the spoken test set |
| MMS-TTS | `facebook/mms-tts-eng` | this Mac (transformers VITS, CPU) | spoken replies when ElevenLabs is off or unreachable (CC-BY-NC 4.0; macOS `say` is the last fallback) |

Two small non-Meta models help: the silero VAD finds speech for the safety listener, and OpenCV's
VitTrack follows SAM's box between detections. Spoken replies are voiced by ElevenLabs
(`eleven_flash_v2_5`, cloud, needs `ELEVENLABS_API_KEY`) and fall back to MMS-TTS on the Mac;
speech to text stays on Muse Voice Transcribe. Everything else in the loop is ordinary code: a
Kalman filter, a proportional controller, and the governor's rules.

## Results

All numbers come from runs on one MacBook Pro (M5 Pro, 48 GB). Conditions, source files, and the
full tables are in [docs/results.md](docs/results.md). The twin numbers were measured the night
before the flights; the real-drone numbers come from the flight logs.

**The real drone** (2026-11-04, our X69 unit, `config/real`, the local Glimmer LoRA answering).
All three flights can be played back in the UI (see [Run it](#run-it)). The first two were rebuilt
from their logs and have no camera picture; the flight recorder saved the third with its video.
In every flight the forward stick stayed at neutral: the autopilot turned the drone with yaw only.

| Flight | Airborne | What happened |
|---|---|---|
| 15:53 ET | 10 s | Take off and Land from the UI, then Emergency stop 7 s into the landing. No mission. |
| 16:45 ET | 102 s | Take off from the UI. "Follow the person with the blue shirt", spoken twice (at 0:48 and 1:31); the Glimmer LoRA answered in 6.0 s and 5.3 s. The autopilot reached FOLLOW after each one, 17 s in total over five spells. The subject was lost for 44 s in total. Land from the UI. |
| 17:41 ET | 81 s | Take off from the UI. "Can you follow the person with the cellphone in their hands" (heard as "selphone") became the mission `person with phone` in 6.0 s, and the autopilot was in FOLLOW 1.5 s later, for 16 s. A second, badly transcribed command became the mission `person` and was acknowledged with `wrong_way_hold`. FOLLOW for 26 s in total over four spells; Pause pressed three times; Land from the UI. |

**End to end in the wire-level twin** (`scripts/e2e_twin.py`): a follow mission posted to the
backend, SAM 3.1 on the Mac, VitTrack and the Kalman filter, the autopilot with feed-forward, the
governor, real X69 packets, 500 ms total video latency, 60 s of following.

| Scenario | Subject phrase | Time in FOLLOW | Bearing error RMS | Lock rate | Frames on the wrong subject | Mission to FOLLOW |
|---|---|---|---|---|---|---|
| S1: one walker | "person in blue shirt" | 100% | 0.092 rad | 95.5% | 0 | 1.57 s |
| S2: blue and red shirts cross | "person in blue shirt" | 99% | 0.103 rad | 94.2% | 49 of 1,706 (2.9%), at the crossings | 2.51 s |
| S3: RC car on a track, a walker, a parked red car | "green car" | 100% | 0.089 rad | 96.0% | 0 | 1.55 s |

**Perception.** SAM 3.1 on the Mac (MLX, 672 input) detects in 234 ms (p50, idle GPU); VitTrack
tracks in 2.6 ms. On twin frames after the full HEVC path, SAM finds the S1 subject in 60 of 60
frames (gate: 90%). The hosted `sam-3.1` answers in 697 ms p50 / 1,277 ms p95 at 640x360. The SAM
3.1 MLX tracker takes about 1 s per step, so it stays out of the control loop.

![SAM 3.1 detections on twin frames: white boxes are ground truth, green boxes are SAM's](docs/img/percep/g2_examples.jpg)

**Intent model.** On 300 held-out commands from unseen phrasings, our LoRA picks the right action
98.0% of the time (base Glimmer 93.7%, Muse Spark 1.3 95.3%), and 100% on 80 hand-written commands
(93.8% and 95.0%), with 100% valid JSON. It writes the subject phrase exactly in our label convention
("person in blue shirt") 92.9% of the time (base 41.4%, Spark 52.7%), and answers in 3.6 s p50 on
the Mac. Its weak spot is noisy speech: 88.8% on the hand-written set spoken through `say` and
Omnilingual ASR, vs. Spark's 90.0%.

![Intent model evaluation: our Glimmer LoRA vs. base Glimmer and Muse Spark 1.3](docs/eval/intent_eval.png)

**Voice.** Muse Voice Transcribe: final transcript 226 ms p50 / 267 ms p95 after you stop talking.
Muse Spark 1.3 at minimal effort: 1.64 s p50 / 2.47 s p95 to a valid mission. 20 of 20 scripted
push-to-talk commands, with Spark answering, gave the right mission; button release to the start of
the spoken reply was 2.40 s p50 / 3.97 s p95. With the local Glimmer LoRA answering instead, release
to mission took 4.3 s in the demo takes. The local listener caught 18 of 18 spoken safety words with 0 false
triggers on 20 ordinary sentences.

**Safety.** 185 governor and autopilot tests. An adversarial review found 4 major and 10 minor
problems; all 14 are fixed, each with a regression test. Killing perception mid-follow gives neutral
sticks on the wire in 0.42-0.44 s.

## Safety design

- **The language models never touch the sticks.** They return a mission (follow, hover, land, clarify,
  refuse). The agent may send only `pause` and `land`; the governor and the backend refuse anything
  else from it. Voice cannot take off (a button does) and cannot E-stop.
- **One choke point.** The governor is installed as the X69 driver's control strategy, so every RC
  packet passes through it, once per 40 ms tick. Priority: E-stop, then land, then human sticks,
  then the autopilot.
- **Latched E-stop.** Once pressed (Esc in the UI), every packet carries the E-stop flag with neutral
  sticks until an explicit reset, which goes to LANDING. With 50% simulated packet loss, every
  received packet was still E-stop. Land is re-asserted 5 times a second until you confirm touchdown.
- **Watchdogs.** Stale observations (perception lag beyond the best-case pipeline): autopilot neutral.
  Video lag over 0.7 s: autopilot neutral; over 2 s: land. UI heartbeat (a Web Worker, 5 Hz) missing
  for 1.5 s: all sticks neutral; for 5 s: land. Human input counts as zero 0.25 s after the last key.
  A flight timer warns at 5:00 and lands at 6:30. Ctrl-C lands first (first land packet 3 ms after
  the signal).
- **Wrong-way checks.** These latch a hover until you press Resume:
  - the subject's bearing moves against the commanded yaw;
  - the box keeps shrinking while the drone flies forward;
  - the box freezes while the drone turns.

  What the reviews added:
  - Two more adversarial reviews tested a reversed pitch sign in closed loop: 360 combinations of speed (0.5-1.5x), deadband and latency.
  - These limits came out of them:
    - the autopilot never backs up, because that is its only path toward a person under a reversed sign;
    - it flies forward only toward a subject within about 6 m, because a farther runaway is too slow to see.
  - Result: a reversed drone never moved toward the person.
  - It drifted at most about 2.4 m away at the modeled speed. It drifted farther if the drone is much faster than modeled, or if a person walks toward it.
  - So the morning runbook requires a hand sign check before any forward follow.
- **Caps.** First-flight autopilot caps: yaw 38 and pitch 32 wire counts out of 127, roll and throttle
  0. Forward motion only when the box is clear of the top and bottom edges, near the center,
  within about 6 m (at least 64 px wide), and confirmed by SAM within the last 1.5 s; otherwise
  the autopilot only turns. It never backs up.
- **Real mode is gated.** `FLIGHT_MODE=twin|real` has no default; twin mode refuses any non-loopback
  address. Real mode accepts only the drone's address (172.16.11.1, or 192.168.0.1 with `--board lewei`), refuses the twin's configs
  (`twin_only: true`), and the operator types ARM once per session. No props-on flight happens until Session A (props off) has checked the flags, link
  loss, latency and field of view. The real-drone configs (`config/*.yaml`) stay conservative, with
  feed-forward off, until Session B measures the drone; the twin uses its own calibration
  (`config/twin/`).

Details: [docs/ws/gov.md](docs/ws/gov.md).

## Run it

You need an Apple Silicon Mac (tested on an M5 Pro, 48 GB, macOS 26), Python 3.13 at
`/opt/homebrew/bin/python3.13`, [uv](https://docs.astral.sh/uv/), ffmpeg with `hevc_videotoolbox`,
Google Chrome, and Node.js with npm. Everything else installs inside the repository folder.
`env.sh` points every cache, model, and virtualenv into the project.

```bash
source env.sh                     # caches, venvs, models, and tools inside the project
scripts/sync_envs.sh              # two Python 3.13 venvs: core (.venv) and ML (ml/.venv)
scripts/fetch_models.sh           # about 22 GB into .cache/models (18 GB is Muse Glimmer); mediamtx into tools/
(cd twin/web && npm ci)           # three.js for the twin renderer
(cd turbodrone/frontend && npm ci)  # the operator UI
$MHD_PY scripts/doctor.py         # environment check
```

Cloud models need a Meta Model API key: copy `.env.example` to `.env` (gitignored) and fill in
`MODEL_API_KEY`. The LoRA adapter is not in git. Train it with
`$MHD_PY training/generate.py --rebuild && training/train_mlx.sh full` (the first command rebuilds the
chat files from the committed data; training took 77 minutes and 17.7 GB peak on the M5 Pro), or
download it from the release page if one is published.

**One-command twin run.** This starts the twin (Chrome renderer, HEVC/RTSP), the governed backend
and perception, takes off, sends a follow mission, follows for 60 s, lands, and scores the run
against the twin's ground truth. Twin only: loopback addresses and `FLIGHT_MODE=twin`.

```bash
$MHD_PY scripts/e2e_twin.py --scenario s1 --subject "person in blue shirt"
$MHD_PY scripts/e2e_twin.py --scenario s3 --subject "green car" --category car --color green
$MHD_PY scripts/e2e_twin.py --perception oracle     # ground-truth perception, no GPU
$MHD_PY scripts/e2e_twin.py --perception sam_api    # hosted SAM 3.1 (paid, $0.0025 per image)
```

Results land in `logs/e2e-<time>-<scenario>-<perception>/e2e_result.json`, with `report.json` and a
per-frame `rows.csv`.

**The recorded demo.** `scripts/demo_twin.py` (from the demo workstream) starts everything, opens
the UI, speaks "Drone, follow the person in the blue shirt." into the agent's microphone input, lets
the fine-tuned Glimmer answer, follows, says "land", and records the UI and the chase view to
`media/demo/`:

```bash
$MHD_PY scripts/demo_twin.py                        # S1, "person in blue shirt"
$MHD_PY scripts/demo_twin.py --demo-scenario s2     # S2: the red-shirt walker crosses
$MHD_PY scripts/demo_twin.py --stills               # also write the README stills to docs/img/
MHD_DEMO_PORTS=8010,8767,23459,5183 $MHD_PY scripts/demo_twin.py   # beside another running stack
```

**The whole stack with one command.** `scripts/run_stack.sh` (from the ops workstream) starts the
backend, perception, the agent, the production UI, optionally the local Glimmer, and in twin mode
the twin; Ctrl-C stops it all, the backend first so the drone lands:

```bash
./fly                 # the real drone: Lewei board, config/real, tuned Glimmer; opens the UI in its own window
./fly sim             # the digital twin, no aircraft
./fly camera          # the drone's real camera, no flight
```

In the UI: type ARM (real drone only), press **Take off**, then hold Space and say what to follow.
**Land** (L) and **Emergency stop** (Esc) are always on screen. **Details** in the header shows the
sticks and the latency of each stage.

Every flight is saved to `logs/flights/` when it lands (about 40 MB per minute in the air;
`--no-record` turns it off). **Flights** in the header plays one back: the camera, the subject
outline, the sticks sent to the drone, and every command on a timeline you can scrub or click.
Replay only shows a saved flight; it never sends its commands to the drone again. It is available
on the ground, and taking off closes it. `scripts/import_flights.py` rebuilds flights flown before
the recorder existed from their logs (no camera picture for those).

![Flights: a saved twin flight played back, with its commands on the right](docs/img/ui_replay.png)

*Flights, playing back a twin flight of the same mission as the top of this page. The list on the
left also holds today's three real flights.*

`./fly` is a short wrapper around `scripts/run_stack.sh`, which takes every option:

```bash
scripts/run_stack.sh --mode twin --scenario s1 --glimmer local --agent glimmer
scripts/run_stack.sh --mode real --no-perception --no-agent     # Session A and B: backend and UI only
scripts/run_stack.sh --mode real --config-dir config/real       # flights, with the Session B values
```

`--agent glimmer` makes the local LoRA answer every command; the default `--agent auto` races it
against Muse Spark 1.3 (see below). `scripts/apply_sysid.py` writes `config/real/` after Session B.
docs/RUNBOOK.md says when to run the local Glimmer during flights.

**The full stack by hand** (one terminal each):

```bash
scripts/run_twin.sh --scenario s1
scripts/run_backend.sh --mode twin --governor-config config/twin/governor.yaml \
    --autopilot-config config/twin/autopilot.yaml
$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 --detector sam_mlx --tracker auto
scripts/serve_glimmer.sh          # Glimmer + LoRA on 127.0.0.1:8080, about 17 GB
$MHD_ML_PY -m mhd_ml.agent.service --backend ws://127.0.0.1:8000 --intent glimmer \
    --glimmer-url http://127.0.0.1:8080/v1 --glimmer-model default_model
(cd turbodrone/frontend && npm run dev)   # open http://127.0.0.1:5173, hold Space to talk
```

`--intent glimmer` uses the fine-tuned Glimmer alone (the keyword grammar answers if it fails).
`--intent auto` also races Muse Spark 1.3 and keeps Glimmer's answer only if it is valid within
2.5 s, which suits a cloud-hosted Glimmer; the local LoRA takes about 4 s, so Spark usually answers
first.

**The real drone.** Follow [docs/RUNBOOK.md](docs/RUNBOOK.md), the morning guide from the ops
workstream. In short:

1. From the terminal app that will launch the backend (macOS grants Local Network permission per
   app): `$MHD_PY scripts/preflight.py` checks the drone Wi-Fi, routes, internet and video.
2. Props off: `$MHD_PY scripts/session_a.py` walks through Session A (flags, latency, field of view,
   link loss) and ends with GO or NO-GO. No props-on flight without GO.
3. `scripts/apply_sysid.py ... --write` writes `config/real/` from the session's measurements (the
   runbook has the exact command).
4. `./fly` starts the backend, perception, the agent, the local Glimmer and the UI. Type ARM in
   the UI, then press Take off. `./fly` does not run the preflight for you.
5. A second person keeps a hand on Land (L) and Emergency stop (Esc). Never fly near people who
   have not agreed to it.

`./fly` is set up for our unit's Lewei board. For it, add `--board lewei` to the preflight and
Session A commands. An X69 with the stock board uses `scripts/run_stack.sh --mode real` instead.

**Tests:** `$MHD_PY -m pytest tests/gov tests/twin tests/train tests/ui tests/docs` and
`$MHD_ML_PY -m pytest tests/percep tests/agent`. Before publishing: `$MHD_PY scripts/secret_scan.py`.

## Built during the event vs. reused

- **Reused:** [turbodrone](https://github.com/marshallrichards/turbodrone) by Marshall Richards
  (Apache-2.0): the X69 Wi-Fi protocol driver, the FastAPI backend, and the React UI shell. It was
  imported unchanged from upstream commit 2c09cf8 in commit `1aff2b5`. Commit `09d005e` (and part of
  `15a5f80`) later added 16 upstream files that ignore rules had hidden from the import,
  byte-identical to 2c09cf8.
- **New, written November 3-4, 2026:** every other commit. That is the safety governor and
  autopilot, the backend patches, the digital twin, the perception service, the voice agent, the
  intent dataset and the Glimmer LoRA, the operator UI's safety patches and new panels, the tests,
  and these documents. Changed turbodrone files are listed in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **The diff:** [`1aff2b5...main`](https://github.com/vancuren/automuse/compare/1aff2b5...main), or
  `git diff 1aff2b5 HEAD` locally.

The video captions say the same: "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else
built Nov 3-4."

## Where things are

| Path | What |
|---|---|
| `src/mhd/governor/`, `src/mhd/autopilot/` | the safety governor (the only path to the sticks) and the pure-function autopilot |
| `src/mhd/backend/`, `turbodrone/backend/web_server.py` | the backend hub, the flight recorder and the patched turbodrone server |
| `src/mhd/contracts/`, `docs/interfaces.md`, `docs/conventions.md` | the messages between processes, units and sign conventions |
| `src/mhd/twin/`, `twin/web/`, `config/twin/` | the wire-level digital twin: RC decoder, dynamics, Three.js renderer, video path, metrics, scenarios |
| `src/mhd_ml/perception/` | SAM 3.1 (MLX and API), trackers, the Kalman filter, the perception service |
| `src/mhd_ml/agent/` | safety listener, Voice Transcribe clients, intent router, MMS-TTS, the agent service |
| `training/` | intent data generator, gold sets, Glimmer LoRA training, evaluation |
| `turbodrone/frontend/` | the operator UI and its mock backend |
| `config/` | real-drone settings for the governor, autopilot and agent (first-flight values) |
| `fly`, `scripts/` | the one-command launcher, run scripts, the end-to-end twin test, the environment doctor, the secret scan |
| `tests/` | one suite per component |

## More documents

| Document | What is in it |
|---|---|
| [docs/results.md](docs/results.md) | every measurement, with conditions and source files |
| [docs/ws/](docs/ws/) | one report per workstream: twin, percep, gov, agent, train, ui |
| [docs/eval/summary.md](docs/eval/summary.md) | intent-model evaluation tables |
| [docs/benchmarks-perception.md](docs/benchmarks-perception.md) | perception gates G0, G1, G2 |
| [PLAN.md](PLAN.md) | the plan, written at the start of the event |
| [docs/video_script.md](docs/video_script.md) | the 3-minute video, two storyboards |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | submission checklist |

## Licenses

Our code is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); turbodrone keeps its own Apache-2.0
license in `turbodrone/`. The models and libraries have their own licenses, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Three matter most:

- SAM 3.1 weights and `meta-sam-parser`: the SAM License ([licenses/SAM_LICENSE.txt](licenses/SAM_LICENSE.txt)),
  which forbids military, warfare and ITAR uses, among others.
- Muse Glimmer: Apache-2.0, plus Meta's usage policy linked from its model card.
- MMS-TTS: CC-BY-NC 4.0, non-commercial only. Use `--tts say` for anything commercial.
- ElevenLabs voices the replies by default (`tts.engine: elevenlabs` in `config/agent.yaml`); it is a
  paid cloud service under its own terms. `--tts mms` keeps every reply on the Mac.

No model weights, raw footage of people, or API keys are stored in this repository.

## Acknowledgments

Marshall Richards for turbodrone, which already spoke the X69's protocol. Meta for the Muse, SAM,
Omnilingual and MMS models and the Model API. mlx-community for the MLX ports of SAM 3.1 and Muse
Glimmer, k2-fsa for sherpa-onnx and its Omnilingual and silero VAD exports, OpenCV for VitTrack,
the mediamtx and three.js projects.

## Contributors

- **Russell Van Curen** (owner).
- **AI assistance, disclosed:** the code, tests and documents were written with Claude Code (Claude
  Opus 5.5, and Claude Fable 5.1 for the final pass), working as parallel agents from the owner's plan. Every commit carries a
  `Co-Authored-By: Claude` trailer.
