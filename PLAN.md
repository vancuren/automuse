# Plan: voice-commanded autonomous follow drone (Loiley X69), Meta Global AI Developer Hackathon

Draft 3, written 2026-11-03 at about 23:15 ET. Submission deadline: 2026-11-04, 21:00 ET (about 21.5 hours away). Research: `docs/research/`. Draft 3 folds in an independent review by a robotics/safety engineer, a delivery manager, and a hackathon judge with a fact-checker.

## 0. The short version

- Fork turbodrone (Apache-2.0), which already speaks the X69's Wi-Fi protocol. Add four things:
  - a digital twin that speaks the same wire protocol;
  - a perception service built on SAM 3.1;
  - a deterministic autopilot that runs inside a safety governor;
  - a voice agent driven by Meta language models.
- Every machine-learning job is done by a Meta model:
  - Muse Voice Transcribe hears you.
  - A LoRA-fine-tuned Muse Glimmer turns your words into a mission, with Muse Spark 1.3 as teacher and as a parallel fallback.
  - SAM 3.1 finds and tracks "the person in the blue shirt".
  - Omnilingual ASR listens locally for "stop" and "land" even when the internet drops.
- The trained Glimmer runs live, hosted on Fireworks, in the twin demo and in at least one recorded real flight. That meets the brief "train an open Meta model and have it run on Fireworks or Baseten".
- Language models choose missions; they never move the sticks. A classical controller does, and it runs inside the governor, which owns every RC packet. The governor adds a latched E-stop, a re-asserted land, watchdogs on stale data, stick caps, and human override.
- The twin is wire-compatible. The unmodified X69 driver sends real 28-byte RC packets to it and receives HEVC over RTSP from it. That proves the same build talks to the drone. Radio behavior, permissions, and real dynamics are proven only in three hardware sessions, and the first, with props off, is a hard gate.
- With about 21.5 hours left:
  - the first hour does accounts and a props-off hardware check;
  - a thin end-to-end slice runs on fake perception by about 03:00;
  - real flights happen in the morning;
  - features freeze at 15:00, and the entry is submitted by 19:00 ET.

## 1. Facts that shape the plan

### Hackathon
Sources: the official event page and Official Rules, plus participant copies of the login-walled guide.

- Submissions close Nov 4 at 9:00 PM ET, and late entries are not considered. Applications closed Oct 25, so this assumes you are an accepted participant. The $150 Meta Model API credit reportedly must be claimed by the same deadline and needs a card pre-authorization.
- Judging, from the Official Rules:
  - Impact, 30%;
  - Use of Meta AI tooling, 25% ("beyond a basic integration");
  - Technical depth, 20%;
  - Demo, 25% ("holds up live" plus video quality).
- Stage-1 gate: the project fits a theme and genuinely uses the Meta Model API or open-weight models. "Push the Frontier" names robotics and controlling a physical system.
- Participant copies of the guide list:
  - a public repo;
  - a 3-minute demo video;
  - all contributors listed;
  - "new work only": the demo must make clear what was built during the event;
  - a deployed or live product that judges can try.
  Confirm all of this in the portal tonight. Discord office hours run Wednesday 10:00 to 11:00 ET.
- Reusing open source is allowed if licenses are followed and the submission builds on it. One participant copy claims only startups may build on an existing codebase, so ask at office hours.
- Reported bonus prizes:
  - Best Build on Open Weight Models, $10K: Glimmer or Spark Open on your own hardware or partner inference. SAM through the API does not count.
  - Best Agentic Use-Case, $5K.
  - Most Viral, $20K: needs an X post linked on the form.
  - Community Vote, $10K.
  - Best Creative, $5K.
- Framing is a real risk:
  - A participant copy of the restrictions lists public-space tracking, police use, identity recognition, "machines judging people", and anything involving war as disqualifying.
  - It also lists "drones, target recognition, ISR, anything defense" as banned in spirit. That copy is unofficial, but it is how a reviewer may read this project.
  - The SAM License bans military use.
  - So the project is framed as a consumer follow-me camera. It selects people by appearance only, never by identity, and only consenting people are filmed.
  - Code and copy avoid the word "target": the person or object followed is the "subject".

### Drone
Sources: turbodrone code and its notes on the decompiled app. Most hardware anecdotes online are unverified.

- The X69 is a 2.4 GHz Wi-Fi access point at 172.16.11.1.
- Control: a 28-byte UDP datagram to port 23458 at 25 Hz, plus a 1 Hz keepalive on a separate thread.
- Video: HEVC 1280x720 over RTSP at /live/ch00_1, per the decompiled app.
- Flag bits:
  - E-stop (0x02) and the camera-tilt bits are readable in the decompiled code.
  - Takeoff (0x01), land (0x80) and flip (0x08) came from obfuscated code.
  - None has been verified on hardware, so all are tested with props off in Session A.
- "Speed tier" is client-side scaling of roll and pitch only. No speed flag goes to the drone.
- No telemetry comes back: no battery, altitude, or link status.
- Flight time is reportedly 8 to 10 minutes per pack, and there are two packs.
- Hover uses optical flow, so it needs a textured floor and bright light.
- Reported video lag is about half a second.
- Listed weights disagree around 250 g. Flying indoors sidesteps the FAA questions.

### turbodrone behaviors that would defeat a naive autopilot (verified in code)
- The X69 model starts in IncrementalStrategy. There, any positive axis value ramps the stick toward full deflection, so a 0.05 command becomes a full-rate ramp.
- The stock UI sends `axes` at 30 Hz even when idle. Each message re-selects the stick strategy, writes zero sticks, and resets camera tilt, unless a turbodrone plugin is running. An autopilot outside the plugin system is overwritten 30 times a second.
- DirectStrategy applies the profile's expo (0.45 for the X69 "normal" profile), so a 0.25 command becomes 0.134 and small commands vanish.
- The flight loop re-sends the last sticks forever, with no staleness check. `model.update` runs outside its try block, so an exception there kills the RC thread while the keepalive thread keeps the link looking alive.
- One-shot flags go out in exactly one packet: takeoff, land, and E-stop. A single dropped datagram loses an E-stop.
- `VideoFrame.timestamp` is the decode time, not the capture time. After a network stall, OpenCV delivers a burst of old frames stamped as fresh.
- Plugins read a 100-frame drop-oldest queue, so a slow reader sees frames that are seconds old.
- The included follow plugin imports ultralytics, which is AGPL. The plugin manager imports every module at startup, so removing ultralytics without deleting `plugins/follow/` crashes the backend.

### Meta models
Checked Nov 3 against dev.meta.ai docs, Hugging Face cards, and provider docs.

- The Meta Model API at https://api.meta.ai/v1 is OpenAI-SDK compatible. It serves muse-spark-1.3, muse-wonder-1.0, muse-image-1.0, muse-voice-transcribe-1.0, and sam-3.1. It serves no Glimmer and no Llama.
- SAM 3.1 on the API:
  - Request and response only: one short noun phrase plus one image, or a whole video clip. No live-stream mode.
  - No published latency.
  - Images cost $2.50 per 1,000, which is $9 to $18 an hour at 1 to 2 calls per second.
  - Object IDs are stable only within one request.
- SAM 3.1 running locally:
  - The official PyTorch code needs CUDA, and the facebook/sam3.1 weights need manual approval.
  - The MLX port in mlx-vlm 0.7.11 runs on Apple Silicon with ungated weights: mlx-community/sam3.1-bf16, 3.5 GB, stored as F32.
  - Its author measured about 0.2 s per tracked frame on an M3 Max. This M5 Pro has a 20-core GPU and has not been measured.
- Muse Spark 1.3:
  - Supports strict JSON-schema output and tool calling.
  - Reasoning cannot be turned off; effort "minimal" is the fastest setting.
  - Live time to first token has been reported around 4.5 s at p95.
- Muse Glimmer 30B:
  - Apache-2.0 open weights; takes text and images.
  - Fireworks fine-tunes it with LoRA for free until Nov 30. Trained LoRAs need an on-demand deployment: an H100 at $8 an hour, with a card on file.
  - Baseten runs it on a dedicated H100 at $6.50 an hour. Baseten Training checkpoints are downloadable.
  - Locally: the official GGUF (17 GB, plus a 1.4 GB vision projector) runs on the already-installed llama-server.
  - Merging a LoRA into the BF16 base needs about 60 GB of memory, so that cannot happen on this Mac.
- Speech:
  - Muse Voice Transcribe (realtime WebSocket, $0.18 an hour) is Meta's only hosted speech model. The API has no text-to-speech.
  - Omnilingual ASR CTC-300M transcribed a 5-second command in 0.13 to 0.19 s on this Mac, through sherpa-onnx on Python 3.13.
  - MMS-TTS is the only practical Meta text-to-speech. It is licensed CC-BY-NC.

### This Mac
- M5 Pro with 48 GB and no CUDA. Python 3.13 is at /opt/homebrew/bin; bare `python3` is Apple's 3.9.6.
- ffmpeg 9 has the hevc_videotoolbox encoder. llama-server (build 11960) and Ollama 0.35.1 are installed. 265 GB is free.
- The Mac has one Wi-Fi radio, and today its only network services are Wi-Fi and Thunderbolt Bridge. Joining the drone's network drops internet unless iPhone USB tethering or USB-C Ethernet is added. The USB Wi-Fi dongle turbodrone recommends likely has no macOS 26 driver.
- macOS gates network and media access with privacy permissions:
  - Local Network permission is attached to the app that launches Python. Without it, traffic to 172.16.11.1 can fail. Loopback is exempt, so the twin works either way and only the real drone breaks.
  - Microphone, Camera, and Screen Recording prompts must be granted while you are awake.
- Dependency conflicts:
  - opencv-python 5.0 lacks the CSRT and KCF trackers; opencv-contrib-python has them.
  - mlx-vlm and ultralytics both pull in opencv-python, which overwrites contrib's `cv2`.
  - So the project overrides opencv-python and pins opencv-contrib-python.
- Binding RTSP's port 554 on 127.0.0.1 needs root (tested). The twin uses port 8554, and the backend points at it with `X69_LG_RTSP_URL`.

## 2. Architecture

```
 Operator UI (forked turbodrone React app, :5173)
   REAL/TWIN banner | drone camera + SAM mask | push-to-talk | transcript + mission JSON + model badge
   latency bars | flight state + timer | LAND (L) and E-STOP (Esc), far apart | heartbeat from a Web Worker
        |  WebSocket (auto-reconnect) + HTTP fallback for LAND and E-STOP
        v
 drone-backend (forked turbodrone FastAPI, :8000, Python 3.13; no MLX or torch imports)
   VideoReceiver (x69_lg RTSP, capture-time estimate) -> /mjpeg (UI) and /ws/frames (pull, newest only) -> perception
   /ws (patched): UI axes, tilt and commands -> governor.submit(...)      never set_strategy
   /mission, /state, /ws/telemetry, /ws/agent <---------------------------------------------- agent
   /ws/subject <----------------------------------------------------------------------------- perception
   FlightController (unchanged, 25 Hz) -> model.update() -> GovernorStrategy.update()   [the only choke point]
        flight state machine, latched E-stop, re-asserted land, watchdogs, caps, slew, wire-count rounding
        autopilot(subject_obs, t) -> sticks          (pure function, called every RC tick)
     -> X69 RC adapter (unchanged) -> UDP :23458
        |                                         ^ HEVC over RTSP
        v                                         |
 real: Loiley X69 at 172.16.11.1        twin: 127.0.0.1, UDP :23458 + RTSP :8554 (mediamtx, loopback only)

 perception (own process):  newest frame -> SAM 3.1 detector (Meta API or MLX) at 1-2 Hz
                            -> tracker at camera rate -> Kalman filter keyed on capture time -> SubjectObs
 agent (own process):       local safety listener (Omnilingual + VAD; stop/hover/land always on)
                            push-to-talk -> Muse Voice Transcribe -> Glimmer-LoRA and Spark in parallel -> /mission
 twin (own process):        RC decoder -> X69 dynamics (lag, drift, deadband, loss) -> Three.js renderer
                            (Chrome via Playwright) -> counter block -> latency and stall injection
                            -> ffmpeg hevc_videotoolbox -> mediamtx; ground truth
```

| Process | Role | Ports |
|---|---|---|
| drone-backend | RC link, video in, governor with autopilot, mission API, hub for the UI | 8000 |
| perception | newest frame to SAM 3.1 detector, tracker, and Kalman filter, then subject observations | client of 8000 |
| agent | local safety listener, push-to-talk, ASR, hedged intent routing, spoken replies | client of 8000 |
| twin | X69 emulator: RC decoder, dynamics, renderer bridge, RTSP publisher, ground truth | UDP 23458, HTTP/WS 8766, RTSP 8554 |
| frontend | operator UI | 5173 |
| glimmer-local (edge mode only) | `llama-server -m ./.cache/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj ... -c 8192` | 8080 |

The governor is installed as the X69 model's control strategy. FlightController calls it once per RC tick, in the RC thread, right before the packet is built, so every packet passes through it.
- `set_strategy` refuses to replace it.
- The patched `/ws` routes UI input to the governor instead of writing sticks.
- It ignores `set_profile` and locks `set_speed_index`.
- The speed tier is pinned at full scale, so the governor's caps are exact wire counts.
- Expo is 0, and rounding is symmetric.
- Per-tick work stays under 1 ms, and logging goes through a queue, never file I/O on the RC thread.

The autopilot is a pure function the governor calls every tick, not a turbodrone plugin. `plugins/follow/` is deleted. Perception and the agent are separate processes, so GPU work, network calls, and crashes never stall the RC thread. The governor watches observation age, so a perception crash produces neutral sticks.

`FLIGHT_MODE=twin|real` is required, with no default:
- Twin mode refuses any `DRONE_IP` that is not loopback.
- Real mode requires `DRONE_IP=172.16.11.1`, plus typing ARM in the UI once per session before takeoff is accepted.
- Real flights run from a tagged git worktree, with a production frontend build and uvicorn without reload. No agent edits or restarts it during a session.
- The twin, the renderer, and the local LLM are stopped during real flights.
- On shutdown, the backend sends neutral sticks plus land for 0.5 s.

Contracts live in `contracts/` as pydantic models, each with a fake producer. Conventions live in `docs/conventions.md`:
- Boxes are in 1280x720 pixels.
- Yaw is positive clockwise, seen from above.
- Positive pitch is forward.
- All timestamps share one clock: time.monotonic, in ms.

| Contract | Direction | Contents |
|---|---|---|
| `Frame` | backend to perception, pull | frame_id, capture_ts estimate, receive_ts, width, height, JPEG |
| `FindRequest` / `Release` | backend to perception | mission_id, subject_phrase, category, color |
| `SubjectObs` | perception to backend | frame_id, capture_ts, state (searching, candidate, confirmed, lost), track_id, box, confidence, tracker score, age of last detector confirmation, mask polygon, source (sam_api, sam_mlx, tracker, oracle) |
| `Candidates` / `Select` | perception to agent to UI, and back | numbered boxes when look-alikes tie, and the user's choice |
| `OperatorInput` | UI to backend | source, axes, nonzero flag, ts; all-zero messages are not input |
| `Heartbeat` | UI to backend | 5 Hz, from a Web Worker |
| `OperatorCommand` | UI to backend | takeoff, land, estop, estop_reset, pause, resume, confirm_touchdown |
| `Mission` | agent to backend | intent JSON (section 4.6), id, model that answered, latency |
| `AgentEvent` / `PttEvent` | agent to and from UI | push-to-talk down and up, partial and final transcript, spoken reply |
| `Telemetry` | backend to UI, 10 Hz | flight state, mode, sticks by source, frame age, latency budget, flight timer per pack, safety flags, mission |
| `FrameStamp` / `Truth` / `TwinControl` | twin | render id and time in the counter block; per-frame actor boxes; scenario, latency, and drift controls |

## 3. Meta model map

| Job | Cloud mode (Meta Model API or partner) | Edge mode (open weights on this Mac) | Non-Meta fallback |
|---|---|---|---|
| Mission speech to text | muse-voice-transcribe-1.0, realtime, push-to-talk, keyword bias | Omnilingual ASR CTC-300M v2 (sherpa-onnx) | none |
| Always-on stop, hover, land | Omnilingual ASR CTC-300M v2 + VAD, local, in every mode | same | keyboard |
| Command to mission | Muse Glimmer 30B + our LoRA on Fireworks, raced against muse-spark-1.3 (effort minimal, strict schema) | base Glimmer GGUF through llama-server with a JSON schema, 50-token output | keyword grammar |
| Find the subject from words | sam-3.1 API on single frames at 1-2 Hz | SAM 3.1 MLX | none |
| Track between detections | SAM 3.1 MLX tracker if it holds 5 Hz or more under load | same | OpenCV VitTrack (ONNX from opencv_zoo) or KCF |
| Pick among look-alikes | color score first, then muse-spark-1.3 vision on numbered boxes (the SAM 3 Agent pattern) | Glimmer vision through llama-server | ask the user |
| Spoken replies | built locally from the mission JSON; voiced by MMS-TTS | same | macOS `say` |
| Training data | muse-spark-1.3 generates and labels; muse-wonder-1.0 optionally adjudicates the gold labels | none | hand-written gold set |
| Twin actors (Tier 3) | muse-image-1.0 images, cut out with SAM 3.1 | none | CC0 3D models |

Deliberately not used:
- Muse Spark Open: its license bars operating transportation technology.
- Llama: not on the API, and no weights newer than Llama 4.
- V-JEPA 2: no role in a drone loop.
- CoTracker3 and Seamless: non-commercial licenses.
- The official CUDA SAM 3.1: the gated weights and a rented GPU will not clear approval and deployment in time.

## 4. Components

### 4.1 Repository, environment, and footprint
- The first commit is `.gitignore`: `.cache/`, `models/`, `tools/*`, `logs/`, `media/`, `captures/`, `.env`, `node_modules/`, `.worktrees/`.
- Then run `git init`. The next commit imports turbodrone at upstream commit 2c09cf8, unchanged, with LICENSE and NOTICE. Every later commit is new work, which the README links.
- One `env.sh`, sourced by every script, keeps everything inside this folder:
  - `UV_PYTHON=/opt/homebrew/bin/python3.13` and `UV_PYTHON_DOWNLOADS=never`;
  - `UV_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `LLAMA_CACHE`, `PIP_CACHE_DIR`, `npm_config_cache`, and `PLAYWRIGHT_BROWSERS_PATH`, all under `./.cache`;
  - Chrome launched with `--user-data-dir=./.cache/chrome`;
  - recordings saved to `./media`.
  Fireworks is used through its REST API with `FIREWORKS_API_KEY`, never `firectl signin`, which writes to the home folder.
- `pyproject.toml` overrides opencv-python out of the dependency set and pins `opencv-contrib-python==5.0.0.93`. A test asserts that `cv2.TrackerCSRT_create` exists.
- Perception gets its own uv project, so mlx-vlm's transformers stack never enters the backend. MMS-TTS is imported lazily inside the agent only.
- ultralytics and `plugins/follow/` are deleted in the same commit.
- `scripts/doctor.py` runs before any work and at the start of every agent session. It checks:
  - the CSRT tracker exists, and hevc_videotoolbox works;
  - a 5-second test stream goes out through mediamtx and comes back in through turbodrone's RTSP client;
  - MLX sees Metal, and the SAM 3.1 MLX predictor loads;
  - sherpa-onnx imports, and MMS-TTS speaks one sentence;
  - every cache path resolves inside the project.
- mediamtx is configured to serve RTSP only, on 127.0.0.1.
- Overnight agents work in git worktrees under `./.worktrees`:
  - each has its own port range;
  - one agent owns `pyproject.toml` and `uv.lock`;
  - a lock file stops GPU benchmarks from overlapping renders or llama-server;
  - one launcher writes all logs to `./logs`.
- Secrets live in `.env`. Before the repo goes public, a project-local regex scan runs over the files and `git log -p`.
- Acceptance: the doctor passes, the upstream x69_lg unit tests pass on Python 3.13, and the contract tests pass against their fakes.

### 4.2 Digital twin
What it proves: the same build, packets, and video path drive the drone. What it cannot prove is left to Sessions A to C: the radio, macOS permissions, real flag behavior, and real dynamics.

Two layers:
- An in-process `x69_sim` adapter pair. This is the fast path for the thin slice, unit tests, and gain tuning.
- A wire-level emulator. It is the acceptance test. The backend runs exactly as for the real drone, with `DRONE_TYPE=x69_lg`, `FLIGHT_MODE=twin`, `DRONE_IP=127.0.0.1`, and `X69_LG_RTSP_URL=rtsp://127.0.0.1:8554/live/ch00_1`.

RC side:
- A UDP server on 127.0.0.1:23458 parses the 8-byte wrapper and the 20-byte payload, checks the markers and the XOR checksum, and decodes sticks and flags.
- Link loss means no RC packet for T seconds. T is measured in Session A; the keepalive is ignored, because it runs on its own thread.
- Packet inter-arrival time is logged, and an alarm fires if p99 exceeds 60 ms. That is the evidence the governor tick shares the backend process safely.
- Injectable faults: RC packet loss, burst loss, and jitter.

Dynamics, at 100 Hz, all parameters in `twin/params/x69.yaml`, refit from Session B:
- Sticks pass through a per-axis firmware deadband, then become velocity and yaw rate through first-order lags with drag. Throttle is climb rate over an altitude hold.
- Camera pitch tilts by about atan(forward acceleration / g).
- Headless mode can be toggled.
- Flags: takeoff climbs to the hover height measured in Session A, land descends and disarms, and E-stop cuts the motors.
- Optical-flow drift is a random walk, stronger over low texture.
- Tilt servo travel per press is configurable, as are link-loss behavior and a battery timer.

Rendering, in Three.js:
- It runs in its own Chrome instance through Playwright's chrome channel. Nothing is downloaded, and the flags `--use-angle=metal --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows` keep it rendering. Everything runs under `caffeinate`.
- At startup it checks that WebGL is not running on the SwiftShader software renderer.
- It renders a drone camera at 1280x720 using the HFOV measured in Session A, plus a chase camera.
- Frames are pulled newest-only, one in flight.

Counter block:
- Luma-only 16x16-pixel cells carry 24 data bits plus an 8-bit CRC. They encode the twin's render time and render ID.
- The block is stamped after noise is added, and decoded by the backend right after video decode.
- It gives exact camera-to-backend latency in the twin. Pointing the real X69 at the counter page on the Mac's screen gives the same measurement for the real drone.
- Injected latency is the target total minus the measured baseline. Acceptance tests use total latency.

Video encode, settings matched to the drone's stream as measured by ffprobe in Session A:
- `-c:v hevc_videotoolbox`;
- `-b:v` at the drone's bitrate, or 3 Mb/s if unknown;
- `-bf 0`;
- `-g` equal to the drone's keyframe interval;
- `-realtime 1`.
Stall-then-burst events are injected, not only drops. If Session A finds no RTSP on this unit, the twin adds the H.265-over-UDP mode.

Actors:
- CC0 people with recolorable shirts and CC0 cars with recolorable paint, from the Quaternius and Kenney packs.
- Confirm by 01:30 that they download with curl. If not, you fetch them before sleeping.
- The SAM detectability check runs on frames that went through the full HEVC path.

Scenarios:
- S1: one walker in a blue shirt.
- S2: blue and red shirts crossing.
- S3: a green RC car on a private track, plus a walker.
- Tier 3: S4 occlusion, S5 the subject leaves the frame, S6 heavy drift.

Ground truth: actor boxes projected into the drone camera every frame, joined to perception output by render ID. `twin/metrics.py` reports time-to-find, lock rate, bearing-error RMS, identity switches, and settling time.

Acceptance:
1. The upstream UI flies the twin by keyboard, and takeoff, land, latched E-stop, and tilt behave.
2. The counter block decodes, and total latency matches the set point within ±50 ms.
3. SAM finds the S1 subject in at least 90% of frames where it covers at least 5% of the image.
4. RC inter-arrival p99 stays at or below 60 ms while the autopilot runs.

### 4.3 Perception service
Input:
- The newest frame, pulled from `/ws/frames`.
- On the real drone, capture time is estimated from the RTSP presentation timestamp, offset by the minimum latency measured in Session A. Decode time is not used.
- The adapter gets a 2-second socket timeout, so a frozen stream fails fast instead of blocking for tens of seconds.

Detector, `detect(image, phrase) -> candidates`:
- `sam_api` calls the Meta API with model sam-3.1 and one noun phrase. Frames are sent as 640x360 JPEG data URLs, with confidence scores and one-bit masks, decoded by meta-sam-parser, with a 2.5-second timeout. Responses are cached by frame hash, and a hard overnight spend cap of $15 applies.
- `sam_mlx` runs Sam3Predictor from mlx-vlm on this Mac.

Tracker, `init(image, box)` then `update(image) -> box, score`:
- SAM 3.1 MLX tracking, if it holds 5 Hz or more (0.20 s per step or less) while the twin renders and llama-server generates.
- Otherwise OpenCV VitTrack, with its ONNX file stored under `./models`, or KCF. Both report an honest loss score. CSRT is the last resort, because it often reports success while drifting onto the background.

Fusion:
- A constant-velocity Kalman filter runs in bearing and size, keyed on capture time.
- Detector results are matched to the state predicted for their own frame's capture time.
- To re-seed, the tracker is replayed from the detection frame through a 2-second ring buffer of 640x360 frames. Offsets larger than 0.3 of the box size are rejected.

Finding the subject:
- Candidates above a confidence threshold (starting at 0.4) are ranked by confidence × color score × size prior. The color score is the hue histogram of the upper-body mask region.
- Ties go to Spark vision with numbered boxes, then to a spoken clarification ("I see two people in blue. Left or right?").
- Perception reports evidence: tracker score and confirmation age. The autopilot alone decides state changes.

Cadence: the detector runs at 2 Hz while finding or lost, and 1 Hz while following. The tracker runs at camera rate.

Acceptance, separate from the twin: on replayed X69 and webcam clips from Session A, the right subject is boxed in at least 80% of frames. Then on twin S1 and S2: found within 3 s, locked at least 90% of the visible time, and at most 1 identity switch per minute.

### 4.4 Autopilot
A pure function, called every 40 ms RC tick with the newest observation, returning sticks to the governor.

Geometry:
- Everything is in radians. Bearing β = atan((cx − W/2)/f), with focal length f from the measured HFOV.
- Size error e_s = ln(w_ref / w) uses box width for every class. Pitching shifts the image vertically by about 145 px per 10 degrees, and close standoffs crop box height.

Delay: θ = measured camera-to-backend latency + perception p50 + half the observation period + half the RC period + yaw lag from system identification. Expect 0.85 to 1.2 s, not the 0.6 to 0.8 s assumed before.

Yaw:
- Starts P-only, at loop gain Kp·K = 0.4/θ, with no D term. K is yaw rate per wire count, from Session B.
- Firmware deadband compensation: u = sign(u) × (d_fw + |u|) above a small threshold.
- Feed-forward is off until a twin run with ±30% error in K shows no ringing. When on, it uses the subject's own rate, ω = β̇ + K × u(t − θ), low-passed, at gain 0.6 or less.
- Never exceed 0.7/θ without it.

Forward and back:
- Proportional on deadbanded e_s (±0.08).
- Forward motion is allowed only when all of these hold: the box is at least 2% from the top and bottom edges, box height is under 0.8 H, |β| is under 0.4 rad, and the last detector confirmation is under 1.5 s old. Otherwise the autopilot is yaw-only.
- Minimum standoff: if box height exceeds 0.85 H or box width exceeds 0.6 W, hover, then back off slowly.
- Backward speed is half the forward cap, with a net-backward travel budget of 1 m.
- People stand beside the flight line, never behind the drone.

Vertical: tilt is set once before takeoff. Throttle and tilt are out of the loop in version 1, and altitude is held by the drone.

Caps for first flights, in wire counts from the 128 center:
- yaw ±38 (0.30);
- pitch ±32 (0.25);
- roll 0;
- throttle 0.
Session B checks that the forward cap gives at least 1.5 times the subject's walking speed. The subject is asked to walk at 0.3 m/s or slower.

Lifecycle, owned by the autopilot:
- IDLE, then FIND (hover, detector at 2 Hz), then CONFIRM (hover, about 1 s of consistent lock), then FOLLOW.
- FOLLOW to DOUBT when the tracker score drops or the box jumps: sticks go neutral and the detector re-runs at once.
- DOUBT to LOST no sooner than detector p95 + 0.5 s.
- LOST also triggers when confirmation age exceeds max(2.5 s, 2 × detector period + detector p95).
- SEARCH is stop-and-stare: a 30-degree yaw step, a hover for detector p95 + 0.5 s, then detection, within ±90 degrees of the last-seen bearing.
- GIVE_UP after 25 s: hover and announce it. Re-finding always passes through CONFIRM.

Acceptance in the twin, at the Session A total latency and at 500 ms above it:
- yaw step overshoot of 10% or less, settling within 7θ;
- bearing error RMS of 0.15 rad or less in S1 at an assumed 0.06 rad/s subject bearing rate;
- the subject stays in frame up to 0.3 rad/s.

### 4.5 Safety governor
The governor is the X69 model's control strategy. Each RC tick it does the following, in order.

1. Flight state:
   - ON_GROUND: neutral sticks only.
   - TAKING_OFF: neutral for at least 4 s.
   - AIRBORNE: the autopilot is allowed.
   - LANDING: land re-asserted for 5 packets every second, sticks neutral, until you confirm touchdown.
   - ESTOPPED: 0x02 in every packet, sticks neutral, until an explicit reset.
   - Flip and calibrate are masked while airborne. If Session A shows 0x80 does not land the drone, LANDING becomes a throttle-down ramp, tuned in the twin and checked at low height in Session B.
2. Priority: E-stop, then land (human or safety), then human sticks, then the autopilot. A human cancels a safety landing only with an explicit button.
3. Staleness checks:
   - Observation age is now minus the capture time of the frame behind the newest observation. Above max(0.4 s, 2.5 × the measured observation period), the autopilot is forced neutral.
   - The autopilot function failing to return is a separate liveness check.
   - Human input counts as zero after 0.25 s without a message. The UI zeros its axes on window blur.
   - The UI heartbeat comes from a Web Worker at 5 Hz. Missing for 1.5 s means neutral; missing for 5 s means land.
   - Video frame age (capture estimate) above 0.7 s means neutral; above 2 s means land.
   - Safety neutralization bypasses the slew limiter.
4. Flight timer: counts from takeoff, per battery pack (chosen at each swap), and saved to a file. It warns at 5:00 and lands at 6:30 until Session A and B measure the real endurance; then it lands 60 s before the low-battery warning.
5. Wrong-way detector: if the observed bearing change over 1.5θ has the opposite sign to the change predicted from commanded yaw, and both exceed 0.15 rad, hover and alert.
6. Output: caps, a slew limit of 1 unit per second for the autopilot, symmetric rounding to wire counts, and the camera tilt state.
7. Errors: any exception in the tick means neutral sticks, an alarm, and a health flag. `last_tick_ts` is published.

No language-model output can produce an E-stop. Voice "stop" and "hover" mean hover; voice "land" means land. Each rule gets a unit test.

Acceptance in the twin:
- killing perception mid-follow gives neutral sticks within 0.5 s;
- freezing video gives a landing within 2.5 s;
- spamming E-stop from a dropped-packet twin still latches;
- a stuck key after focus loss goes stale in 0.25 s.

### 4.6 Voice and intent agent
- The local safety listener runs in every mode: Omnilingual ASR CTC-300M int8 through sherpa-onnx, plus silero VAD. It listens continuously for stop, hover, land, and abort, and calls the governor directly. False alarms only ever hover or land.
  - Twin acceptance: with the network cut, saying "land" starts a landing within 1.5 s.
- Push-to-talk is for new missions only:
  - It is the on-screen button, or Space through a capture-phase key listener. Buttons never take keyboard focus, so Space cannot re-fire them.
  - Audio is 24 kHz mono 16-bit in 80 ms chunks, or 16 kHz for a Bluetooth headset.
  - Muse Voice Transcribe runs in PUSH_TO_TALK mode, with a keyword bias list.
  - If per-utterance sessions are slow to open, one persistent session streams 80 ms of silence between utterances.
- Takeoff is a button press only. Voice cannot take off.
- Intent schema, strict, about 50 output tokens:

```json
{
  "action": "follow | hover | land | clarify | refuse",
  "subject_phrase": "person in blue shirt",
  "category": "person | car | animal | object | none",
  "color": "blue | green | red | ... | none",
  "distance": "close | medium | far",
  "reason_code": "none | ambiguous | identity_request | unsafe | unsupported"
}
```

- Hedged routing: every command goes to the tuned Glimmer on Fireworks and to muse-spark-1.3 (effort minimal, strict schema, streaming) at the same time.
  - If Glimmer's answer is valid within 2.5 s, it is used. Otherwise the first valid answer wins.
  - The UI badge shows which model answered and how long it took.
- The spoken reply is built locally from the JSON and voiced by MMS-TTS, falling back to `say`. A short acknowledgment plays as soon as the subject phrase arrives.
- Stretch (Tier 3): multi-step missions through Spark tool calls, run by a small mission runner. The model never commands the sticks.
- Acceptance:
  - from button release to the start of the spoken reply: p50 at most 4 s and p95 at most 6 s;
  - 18 of 20 scripted commands give the right mission.

### 4.7 Training: fine-tune Muse Glimmer, then run it live
What it learns:
- A LoRA on Muse Glimmer 30B maps noisy transcripts to the schema above.
- It also learns policy: clarify ambiguous requests, and refuse identity requests ("follow John") and surveillance requests while offering to follow by appearance instead.

Data, built by `training/generate.py`:
- Templates crossed with slots give 2,500 training and 300 held-out examples. The held-out split is by template family and paraphrase seed, so no near-duplicates leak.
- Spark generates and labels them in batches of 10 per strict-schema call, at effort minimal, under a $10 cap. The actual cost is reported.
- ASR-style noise matches what Omnilingual and Voice Transcribe emit.
- About 15% clarify cases and 5% refuse cases.
- Optionally, muse-wonder-1.0 adjudicates the held-out labels, for about $2.
- A gold set not produced by Spark: 80 hand-written commands, plus Voice Transcribe transcripts of `say` audio, plus about 40 commands you speak in 10 minutes in the morning.
- Check the dev.meta.ai terms on training with model outputs tonight.

Training:
- Fireworks LoRA SFT on muse-glimmer-30b, free through Nov 30.
- Free-path defaults are rank 8, 1 epoch, and a 2,048-token limit. Rank 16 and 2 epochs are used only if the API accepts them.
- A 50-row smoke job validates the chat format, including a short reasoning field, before the full run.

Time box:
- data by 02:30;
- smoke job by 02:45;
- full job by 03:30;
- evaluation by 05:00.
If Fireworks quota or format fails by 03:30, switch to Baseten Training (Axolotl QLoRA on one H100, downloadable checkpoints).

Evaluation:
- One on-demand deployment with LoRA add-ons scores base Glimmer and the LoRA in the same window, for about $8 to $12. Spark is scored too. Then the deployment scales to zero.
- Metrics, on the held-out set and the gold set: valid-JSON rate, action accuracy, subject-phrase match, clarify and refuse F1, and latency.

Live:
- The deployment is warm from about 10:15 to 12:45 for flights and recording, about $20, then scaled to zero by hand. The default idle wait before scaling down is an hour.

Acceptance: the LoRA beats base Glimmer on action accuracy on both sets, with at least 95% valid JSON. It answers live in at least one recorded flight and in the twin demo.

Tier 2: download the Baseten checkpoint, merge and convert it to a Q4 GGUF in the cloud, and serve it on llama-server. That gives offline flights, and it counts as "own hardware" for the open-weights bonus.

### 4.8 Operator UI
The turbodrone frontend, patched:
- `axes` is sent only when nonzero.
- Axes zero on blur and on visibility change.
- Auto-reconnect, with a full-width DISCONNECTED banner while the socket is down.
- LAND and E-STOP fall back to HTTP POST if the WebSocket is down.
- The heartbeat comes from a Web Worker, so tab throttling doesn't stop it.
- E-STOP sits on its own hotkey (Esc), far from LAND (L).
- A REAL/TWIN banner shows the flight mode.

Added panels:
- push-to-talk with live transcript;
- mission card, with the model badge and latency;
- SAM mask overlay colored by state;
- flight-state badge and timer per pack;
- latency bars: capture to backend, detector, intent;
- stick bars;
- candidate picker;
- the twin chase view, in twin mode.

Captions mark what is turbodrone and what is new. Acceptance: one screen recording tells the whole story.

### 4.9 Go/no-go gates, with clock times

| Gate | When | Measure | Go if | Otherwise |
|---|---|---|---|---|
| G0 cloud latency | 00:00, home Wi-Fi, downloads paused; repeated over the tether in Session A | sam-3.1 image round trip at 640x360, p50 and p95; Spark at minimal; Voice Transcribe end of speech to final | SAM p95 1.5 s or less; Spark p95 4 s or less; voice 1 s or less | SAM at 0.5-1 Hz with a stronger tracker; Glimmer-first routing; persistent voice session |
| G1 perception | 01:30, on Session A clips | SAM accuracy on "person in blue shirt" with two people and "green car"; MLX detect and track speed under load | right subject in 8 of 10 frames; track step 0.20 s or less | color score plus clarification; VitTrack or KCF in the loop |
| G2 twin actors | first wire-level render, about 03:30 | SAM on HEVC-path twin frames | 90% recall in S1 | textured CC0 glTF variants, then billboards |
| G3 local Glimmer | only if edge mode is triggered | llama-server time for a 50-token intent | 6 s or less | keyword grammar in flight |
| Both SAM paths fail | any time | none | none | SAM API at 0.5 Hz or less, KCF, Kalman, gains at 0.35/θ, hover on DOUBT |
| Hardware | 13:00 | one clean voice, follow, land run | recorded | twin-led storyboard plus a manual or yaw-only clip; stop flying |

## 5. Networking for real flights
- Use iPhone USB tethering for internet, with the Mac's Wi-Fi on the X69. In System Settings, under Network, put the iPhone service above Wi-Fi.
- Check whether the X69's DHCP lease installs a default route.
- Grant Local Network permission to the terminal app that launches the backend.
- `scripts/preflight.py` runs from that same app. It does a raw send to 172.16.11.1:23458 and a TCP connect to port 554, and prints the errno: 65 means the permission is missing. It also checks the routes:

```
route -n get 172.16.11.1     # interface should be en0, the drone Wi-Fi
route -n get api.meta.ai     # interface should be the iPhone one
```

- For sessions, turn off AirDrop and Handoff. They hop the Wi-Fi radio off-channel and cause 100 ms spikes. With your OK, `sudo ifconfig awdl0 down` turns this off until reboot.
- Turn off auto-join for other networks.
- Film with a second phone, not the tether iPhone.
- If the tether fails in Session A, minimal edge mode moves to Tier 1 for flights: SAM MLX, a CPU tracker, the local safety listener, and Spark only when online.

## 6. Hardware sessions
Session A is a hard gate. No props-on flight happens until it passes.

### Session A: tonight, about 00:15 to 01:00, props off
Before starting:
- Mark each prop and motor A or B, and photograph them.
- Use the prop tool.
- Keep motor runs to 10 to 15 s.
- Work on a non-flammable surface, with a hand ready to pull the battery.

1. Tether, routes, Local Network grant, and preflight script. Repeat G0 over the tether.
2. Run ffprobe on the RTSP stream for frame rate, keyframe interval, bitrate, and whether it allows only one client. Then run the backend with `DRONE_TYPE=x69_lg FLIGHT_MODE=real` and confirm video.
3. Latency: point the X69 camera at the counter page on the Mac's screen for 60 s, to build a camera-to-backend histogram. If the page isn't ready, film a phone stopwatch instead.
4. HFOV: put the drone 1.00 m from a wall with a tape measure across the view. HFOV = 2·atan(w/2d).
5. Flags:
   - takeoff makes the motors spin;
   - land stops them;
   - E-stop works, both once and repeated;
   - land while on the ground is a no-op;
   - tilt up and down travel per press;
   - whether takeoff needs an arm step.
6. Link loss, three ways, 10 s each, filmed:
   - kill the backend;
   - turn off the Mac's Wi-Fi;
   - stop RC packets while the keepalive continues.
   Note what the motors do and how fast.
7. Physical remote: with turbodrone streaming and the motors running, does the remote's stop cut them? Does switching the remote on change anything?
8. Recordings:
   - 60 s of raw RTSP (`ffmpeg -c copy`);
   - 2 minutes of hand-carried video of the subjects (a person in a blue shirt, a person in another color, and a green toy car), for perception replay;
   - a 5 s mic test and a webcam clip, which grant the macOS permissions.
9. Weigh the drone with its battery. Note the low-battery LED behavior.

Exit: the flag table, latency, link-loss behavior, remote behavior, HFOV, and stream format are known. Props-on is go or no-go.

### Session B: about 07:30 to 08:15, props on, indoors, pack 1
Space and safety:
- At least 5 by 5 m, with a ceiling of 2.5 m or more, a patterned mat, and bright light.
- Spectators stand beside the flight line, never behind the drone.
- E-stop is armed for the whole session.

Steps:
- Take off by button, hover for 30 s, note the drift, and land with 0x80.
- Sign checks: positive yaw moves the scene the expected way. For headless, yaw 90 degrees, then pitch forward for 1 s; the motion must follow the camera axis.
- System identification uses an ArUco marker on the wall or a tablet, with opencv-contrib measuring yaw angle and range per frame. Step each axis at three amplitudes per sign, with expo 0 and raw counts logged: yaw at 0.10, 0.20, and 0.35; pitch at 0.10, 0.20, and 0.25. This gives firmware deadband, yaw rate and speed per count, and lag.

Exit: θ, K, deadband, and the caps are verified, and the twin is refit within 20% on rise time.

### Session C: about 09:30 to 11:30, packs 2 then 1, with a second person
1. Yaw-only follow at 60% of the twin-optimal gain: a subject standing, then walking slowly sideways at 0.3 m/s or less.
2. Add forward and back at a 2 to 4 m standoff.
3. The full loop:
   - take off by button;
   - "follow the person in the blue shirt" through push-to-talk;
   - the tuned Glimmer answers live;
   - "land" through the local safety listener.
4. Record a screen capture into `./media` plus a second camera on the room.
5. Re-flights and fixes run from 11:30 to 13:00. The hardware go/no-go is at 13:00.

Rules:
- One person stays on LAND and E-STOP.
- Battery out when not flying, and charging is attended.
- Stop at the first unexpected behavior.
- Filmed people consent on camera.
- If working solo:
  - give missions at the laptop before stepping into frame;
  - rely on the always-on local "land";
  - keep the physical remote in hand if Session A showed it can stop the motors;
  - cut the caps by a third;
  - aim for one clean clip.

## 7. Schedule
Times are ET and assume you say go by 23:45. Each hour later compresses the morning.

| When | Who | Work | Exit check |
|---|---|---|---|
| now to 23:45 | You | answer the questions; claim credits; create the Meta API key with a spend cap; create a Fireworks account, add a card, prepay about $20; read the portal (form fields, live URL required?, editable after submit?, credit coverage and expiry); line up a second person for 09:30 to 12:00; find a blue shirt and a green toy car; start battery charging (attended); test iPhone USB hotspot | keys in `./.env` |
| 23:45 to 00:15 | Claude | `.gitignore`, git init, turbodrone import, `env.sh`, uv projects with the opencv override, doctor, preflight, counter page, upstream backend and UI runnable, contracts v0 with fakes | doctor passes |
| 00:00 to 00:15 | Claude | G0 on home Wi-Fi; start batched Spark data generation; prove the Fireworks H100 quota (create a base Glimmer deployment, one completion, delete it); then start the 3.5 GB SAM MLX download | G0 table in `docs/benchmarks.md` |
| 00:15 to 01:00 | You + Claude | Session A | flag table and latency known; props-on go/no-go |
| 01:00 to 01:10 | You | hand-off: rejoin home Wi-Fi with auto-join on; Mac on AC, lid open, auto-lock off, Amphetamine on; then sleep until 07:15 | permissions granted |
| 01:10 to 03:00 | Claude | G1 on Session A clips; pick providers; freeze contracts v1; smoke fine-tune; thin slice: typed text to Spark to Mission to oracle perception (from twin ground truth, with latency, noise and dropouts) to autopilot to governor to FlightController to in-process twin | S1 closes the loop |
| 03:00 to 05:00 | Claude, parallel worktrees | real perception on replayed clips; governor rules and unit tests; voice from WAV files plus the safety listener; full fine-tune, evaluation, scale to zero; twin S2 and S3, wire-level RTSP, counter block; UI panels | real SAM on the in-process twin; eval chart |
| 05:00 to 07:15 | Claude | wire-level integration; S1 to S3 metrics; gain tuning at the measured total latency and 500 ms above it; twin footage; README draft | twin acceptance passes |
| 07:30 to 08:15 | You | Session B (pack 1) | θ, K, deadband measured |
| 08:15 to 09:15 | Claude / You | Claude refits the twin and retunes; you do a voice and UI dry run with the real mic and record the 40 gold commands | twin within 20%; dry run clean |
| 09:30 to 11:30 | You + second person | Session C (packs 2 then 1); LoRA warm from 10:15 | one clean run |
| 11:30 to 13:00 | You + Claude | re-flights, fixes, the live-LoRA recording; LoRA scaled to zero by 12:45 | hardware go/no-go at 13:00 |
| 13:00 to 15:00 | Claude | Tier 2 only if Tier 1 is green; the static replay page if a live URL is required | feature freeze at 15:00 |
| 15:00 to 18:00 | You + Claude | video (rough cut by 16:30), README, licenses, secret scan, repo public by 17:30 | assets final |
| 18:00 to 19:00 | You | X post first if going for Most Viral, video uploaded unlisted, submit | submitted by 19:00 |
| 19:00 to 21:00 | | buffer | |

Priority tiers. Protect Tier 1 in this order:
1. The wire-level twin with S1 and S2 and ground-truth metrics.
2. Cloud perception, autopilot, and governor.
3. Voice with Spark intent and the local stop/land listener.
4. The Glimmer LoRA, trained, evaluated, and answering live in at least one recorded window.
5. One real follow clip, or the 13:00 fallback.

Tier 2:
- Minimal edge mode. This moves to Tier 1 if the tether fails.
- Spark disambiguation.
- The offline LoRA GGUF.
- The static replay page. This moves to Tier 1 if the portal requires a live URL.

Tier 3: Muse Image actors, S4 to S6, the chase camera, the twin's battery and link-loss modes, agentic missions, and the EdgeTAM and EfficientTAM benchmarks.

When time runs short, cut in this order:
1. edge mode;
2. Tier 3;
3. Spark disambiguation, keeping the color score and the spoken clarification;
4. MMS-TTS, falling back to `say`.

## 8. Submission package
- Two storyboards; the 13:00 gate picks one.
  - Real-flight-led:
    - 0:00 a voice command, and the drone turns to follow;
    - 0:15 why it matters: a $34 toy becomes a hands-free, voice-directed camera operator for creators, coaches, and people with limited mobility;
    - 0:50 the architecture, with each Meta model named and its role;
    - 1:30 the twin: same packets, metrics;
    - 2:00 the training chart, including identity-request refusals;
    - 2:30 the safety layer;
    - 2:50 close.
  - Twin-led: the same structure, opening on the twin, with a manual or yaw-only real clip.
- The video captions what was reused: "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else built Nov 3-4."
- README:
  - new work vs. reused, with commit links;
  - setup, and a one-command twin demo;
  - flying the real drone;
  - the models actually used, each with the commit that uses it;
  - evaluation results, safety, licenses;
  - the framing: the model picks an intent, and a deterministic controller flies.
- Licenses:
  - keep Apache-2.0 and the turbodrone NOTICE;
  - include the SAM License text, which covers the weights and meta-sam-parser;
  - Glimmer is Apache-2.0, plus its usage policy (read it tonight);
  - MMS-TTS is CC-BY-NC, stated plainly.
- Privacy: raw footage of people never enters the repo. Only metrics, charts, and the video link do.
- If the portal requires a live URL: a static GitHub Pages "twin replay", with Three.js playing back recorded poses, masks, transcripts, and missions. No API keys, about one hour of work.

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| About 21.5 hours to deadline | high | high | thin slice by 03:00; staged gates; tiers; freeze at 15:00; submit by 19:00 |
| Upstream stick handling defeats the autopilot | was certain | high | governor as the model's strategy; patched `/ws`; expo 0; tier pinned |
| Flag bits wrong, or the drone keeps flying on link loss | medium | high | Session A hard gate; latched E-stop; re-asserted land; throttle-ramp landing fallback |
| No internet on the drone Wi-Fi, or Local Network permission missing | medium | high | tether plus preflight in Session A; minimal edge mode |
| SAM slow, or picks the wrong person | medium | medium | two-rate design; capture-time Kalman; color score; clarification |
| Total delay about 1 s makes the loop oscillate | high | medium | measured θ; gain 0.4/θ; feed-forward only after a robustness test; yaw-only gating |
| Indoor collision while backing up or near a wall | medium | high | standoff limits; 1 m backward budget; people beside the flight line; E-stop armed |
| Fireworks quota, card, or cold start | medium | medium | quota proven at midnight; Spark races Glimmer; Baseten fallback by 03:30 |
| Twin actors not detected by SAM | medium | medium | G2 on HEVC-path frames; textured variants; billboards |
| Eligibility framing | medium | high | follow-me camera, appearance only, "subject", consent; ask at office hours |
| Overnight work silently stalls (permissions, throttled tabs, sleep) | medium | medium | hand-off checklist; Playwright flags; caffeinate; doctor |
| Only 16-20 minutes of airtime | high | medium | sessions sized to packs; the twin for iteration |
| API keys leak through the public repo | low | high | gitignored `.env`; secret scan; spend caps |

## 10. Decision log

| Decision | Choice | Alternatives | Why |
|---|---|---|---|
| Base | fork turbodrone | from scratch | X69 protocol already reverse-engineered; Apache-2.0 |
| Choke point | governor as the model's control strategy | plugin; wrapper around FlightController | the only per-packet hook in the RC thread; upstream code stays unchanged |
| Autopilot placement | pure function inside the governor tick | turbodrone plugin; separate process | no 20/25 Hz beat; no UI overwrite; deterministic |
| Twin boundary | wire level as the acceptance test; in-process for speed | one or the other | exercises the exact driver while keeping iteration fast |
| Renderer | Three.js in Playwright-driven Chrome; dynamics in Python | Webots, MuJoCo, OpenCV 2.5D, Godot | visuals and CC0 animated people; nothing installed; Webots needs a system install and its humans are licensed for Webots only; MuJoCo has no people |
| Perception | SAM 3.1 detector at 1-2 Hz, tracker, capture-time Kalman | SAM on every frame | the API has no stream mode |
| SAM host | Meta API, plus MLX on the Mac | CUDA SAM on Baseten or Modal | gated weights and deploy time vs. the deadline |
| Control | P with measured-delay gains, gated forward motion, governor | LLM control; learned policy | safe, tunable, explainable |
| Trained model | Glimmer LoRA for intent, live on Fireworks | SAM fine-tune; behavior cloning | fits the time box, measurable, visible; SAM training needs CUDA |
| Intent routing | race Glimmer and Spark | Glimmer, then Spark sequentially | keeps latency inside the budget during cold starts |
| Safety words | always-on local listener | push-to-talk only | works with no internet and with no hands |
| Speech | Muse Voice Transcribe plus Omnilingual | Whisper | Meta-only and offline-capable |
| Python | 3.13 | 3.15 | wheel availability |

## 11. Open questions and the defaults I will use
Questions for you:
1. Are you submitting by Nov 4, 9 PM ET as an accepted participant? Should I start the answer-independent first hour now? Default: yes and yes.
2. Second internet path while flying? Default: iPhone USB tethering.
3. Where should the fine-tuned Glimmer be trained and served? Default: Fireworks, with Baseten as the fallback.
4. What is in place: a Meta API key with credits; a Fireworks account with a card; a second person for 09:30 to 12:00; the X69 already flown with turbodrone?

Assumptions unless you say otherwise:
- The demo is indoors, following a consenting person. "Green car" is a toy car indoors, plus a private-track scenario in the twin.
- Solo builder, Indie track, Push the Frontier theme.
- Non-AI glue is fine: OpenCV tracker, Kalman filter, Three.js, mediamtx, `say` as a fallback. Every ML role uses a Meta model.
- Everything stays inside this folder: about 4 GB of models for Tier 1, about 22 GB with edge mode. Already-installed llama-server, ffmpeg, and Chrome may be used; nothing new is installed system-wide.
- Approved exceptions: macOS permission grants, the network service order, and optionally `sudo tcpdump` (writing to `./captures`) and `sudo ifconfig awdl0 down`.
- Spend beyond credits stays at $50 or less.
- The Mac may stay awake and unlocked overnight on AC power.

## 12. Before you sleep (hand-off checklist)
1. Session A done, or rescheduled to 07:00.
2. Microphone, Camera, Screen Recording, and Local Network permissions granted to the terminal app.
3. Home Wi-Fi rejoined, with auto-join on.
4. Mac on AC power, lid open, auto-lock off, Amphetamine on.
5. Batteries charged and removed from the drone.
6. Second person confirmed for 09:30.

## 13. Footprint so far
Research subagents wrote outside this folder before the rule was passed to them. Nothing else outside it was changed. Cleanup, if you want it:

```
brew uninstall libsndfile && brew uninstall libvorbis flac && brew uninstall libogg
rm -rf ~/.cache/fairseq2
rm -rf ~/.cache/huggingface/hub/models--facebook--wav2vec2-large-960h-lv60-self ~/.cache/huggingface/hub/models--facebook--mms-tts-eng
rm -f ~/.cache/huggingface/hub/blobs/b3/b37dcb74594da028322793f7bd638569e82b73831b26efe8a41cef4597660d69 \
      ~/.cache/huggingface/hub/blobs/e6/e6725a55c0fb20b65d5d5575905cdc7c5811060f9290a7773b576c26354699a4 \
      ~/.cache/huggingface/hub/blobs/ff/ffc775fd4a0db0116986f200d15f41bae284412678f8a891db1e13bedc052629
```

The shared uv package cache (`~/.cache/uv`) also grew during benchmarks. Its new entries can't be separated from older ones, so it is left alone.
