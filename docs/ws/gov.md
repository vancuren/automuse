# Workstream gov: backend patches, safety governor, autopilot

Branch `ws/gov`. Built overnight 2026-11-03/04. PLAN.md sections 2, 4.1, 4.4, 4.5 and 6.

## What exists

| Path | What it is |
|---|---|
| `config/governor.yaml`, `config/autopilot.yaml` | Every safety and control parameter, with commented first-flight defaults. Values marked [Session A] or [Session B] are guesses until measured. Unknown keys are errors. |
| `src/mhd/governor/config.py` | Strict YAML loaders (frozen dataclasses). `GOVERNOR_CONFIG` and `AUTOPILOT_CONFIG` override the paths. |
| `src/mhd/governor/core.py` | `Governor`. One `tick()` per RC packet: flight state machine, priorities, staleness checks, flight timer, wrong-way detector, slew limit, wire counts, flags. Inputs arrive from other threads through `heartbeat`, `submit_human`, `set_tilt`, `note_frame`, `submit_obs`, `set_mission` and `command`. |
| `src/mhd/governor/wire.py` | Normalized to wire counts: per-axis sign, deadband compensation (autopilot only), caps in counts, symmetric rounding (127 counts each side). |
| `src/mhd/governor/turbodrone_glue.py` | `GovernorStrategy` (turbodrone ControlStrategy: the choke point). `GovernedX69LgRcModel` (`set_strategy` refuses to replace the governor; speed tier pinned at 2; legacy `takeoff`/`land`/`emergency_stop` go through the governor; `flip`/`calibrate` refused). `PacedFlightController` (deadline-paced 25 Hz). |
| `src/mhd/governor/sysid.py` | Session B step identification. Flies ± steps at chosen amplitudes through the governor (human priority, so LAND/E-STOP still win), measures an ArUco marker on `/ws/frames`, and fits K per count, firmware deadband and lag. |
| `src/mhd/governor/flight_timer.py`, `tick_log.py`, `logq.py`, `stats.py`, `telemetry.py`, `bench.py` | Per-pack timer persisted off the RC thread; per-tick JSONL through a QueueHandler; non-blocking governor logging; sliding minimum; contract `Telemetry` builder; tick benchmark. |
| `src/mhd/autopilot/core.py` | `autopilot_step(obs, state, params, now_ms, *, can_move, applied_yaw, applied_pitch) -> (sticks, new_state, events)`: a pure function with the IDLE, FIND, CONFIRM, FOLLOW, DOUBT, LOST, SEARCH, GIVE_UP lifecycle. |
| `src/mhd/autopilot/sim.py` | In-process closed-loop sim: the real `Governor` flying a kinematic X69 after a walking subject, with delayed oracle observations. |
| `src/mhd/backend/` | `modes.py` (FLIGHT_MODE gate), `capture.py` (capture-time estimate plus `TimedX69RtspAdapter`), `frames.py` (newest-frame store for `/ws/frames`), `hub.py` (all new endpoints; routes every UI, perception and agent message to the governor). |
| `turbodrone/backend/web_server.py` | Patched, +30/-78 lines. Every change is marked `MHD patch`. |
| `turbodrone/backend/plugins/follow/` | Deleted, and `ultralytics` removed from `requirements.txt`. `PLUGINS_ENABLED=true` is refused. |
| `turbodrone/backend/models/` | Restored. The root `.gitignore` rule `models/` matched it at any depth, so the import commit left the package out and the backend could not start from a clean checkout. The rule is now `/models/`, and the 15 upstream files are added byte-identical. |
| `scripts/run_backend.sh` | Starts the patched backend from the checkout it lives in. |
| `tests/gov/` | 106 pytest tests (see below). |

## Run it

```bash
cd <checkout> && source env.sh
# twin, with a UI sending heartbeats
scripts/run_backend.sh --mode twin --port 8000 --rtsp-url rtsp://127.0.0.1:8554/live/ch00_1
# twin, headless (no UI heartbeat needed; refused in real mode)
scripts/run_backend.sh --mode twin --port 8100 --control-port 23512 --rtsp-url rtsp://127.0.0.1:8105/live/ch00_1 --headless
# real drone (DRONE_IP=172.16.11.1, RTSP from the drone); type ARM in the UI before takeoff
scripts/run_backend.sh --mode real --port 8000
```

Without `--mode` it refuses to start. The server binds 127.0.0.1 and never uses `--reload`. Control logs go to `$MHD_LOGS/control-*.jsonl`, the flight timer to `$MHD_LOGS/flight_timer.json`.

Tests and tools:

```bash
$MHD_PY -m pytest tests/gov -q -p no:logging            # 106 tests, about 105 s (RTSP and live tests ~50 s)
(cd turbodrone/backend && $MHD_PY -m unittest tests.test_x69_lg_protocol tests.test_x69_lg_rtsp_video_protocol tests.test_x69_lg_video_mode)
$MHD_PY -m mhd.autopilot.sim                             # step + S1 at total delays 0.6 / 1.0 / 1.5 s
$MHD_PY -m mhd.autopilot.sim --ff --k-scale 0.7 1.0 1.3  # feed-forward robustness to K error
$MHD_PY -m mhd.governor.bench                            # per-tick wall and CPU time
```

Session B system identification (drone AIRBORNE, UI open, ArUco DICT_4X4_50 marker in view; results in `$MHD_LOGS/sysid-*.json`):

```bash
$MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis yaw --amps 0.10 0.20 0.35 --marker-size 0.15
$MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis pitch --amps 0.10 0.20 0.25 --hold 1.5
```

It refuses unless AIRBORNE, aborts the moment the drone isn't, and flies each amplitude + then −. Put the fitted K, deadband and lag into `config/autopilot.yaml` (`plant.*`) and `config/governor.yaml` (`wire.deadband_counts`).


`-p no:logging` only hides pytest's captured-log dump on failures.

## Test results (final run, 00:43 ET)

- `tests/gov`: **106 passed, 0 failed.**
  - 42 governor rule tests (`test_governor_rules.py`), including the throttle-ramp landing fallback.
  - 14 autopilot lifecycle and geometry tests.
  - 7 turbodrone glue tests.
  - 2 UDP wire tests (paced and upstream FlightController).
  - 15 closed-loop sim tests.
  - 16 FastAPI TestClient tests.
  - 4 capture-clock tests.
  - 1 RTSP integration test (mediamtx with an ffmpeg HEVC stream).
  - 1 live end-to-end test (`run_backend.sh` with uvicorn, RTSP and a UDP drone stand-in).
  - 4 sysid tests: schedule limits, ArUco bearing and range on a synthetic frame, fit recovering K / deadband / lag, and a live run through the real backend with an ArUco HEVC stream.
- Upstream x69_lg unit tests: **15 passed.**
- Repo-level `tests/test_contracts.py` and `tests/test_intent_spec.py`: **6 passed.**

## Measurements

All measured 23:58–00:43 ET on the M5 Pro after the model downloads finished (23:40:51, so the network was idle). Five other workstreams were running alongside: Chrome/Playwright and node (twin/ui), Python processes, and the percep workstream's ffmpeg. Load average was 3.5–4.0. No GPU model was loaded by gov.

| What | Result |
|---|---|
| Governor tick CPU time, following with a fresh observation every tick, tick log on (`mhd.governor.bench`) | p50 35 µs, p99 57 µs, max 160 µs |
| Same, with 2 busy Python threads | CPU p99 81 µs; wall max 20–34 ms. This is waiting for the GIL (5 ms switch interval), not governor work. A 1 ms switch interval made p99 worse (3.1 ms), so the default is kept. |
| Tick wall time inside the real RC thread (wire test, with feeder and receiver threads) | p50 0.10–0.15 ms, p99 0.21–0.40 ms, max 1.2–5.5 ms |
| RC inter-arrival, upstream FlightController | p50 47.7 ms, p99 63.0 ms (about 20.7 Hz). On this Mac `time.sleep(0.040)` takes 45.9 ms. |
| RC inter-arrival, `PacedFlightController` (used by the backend) | p50 40.4 ms, p99 48.3 ms (25 Hz) |
| Perception killed mid-follow, time to neutral sticks on the wire | 0.42–0.44 s (PLAN: within 0.5 s) |
| Land re-assert on the wire | 10 land packets in 2 s, evenly spaced |
| Latched E-stop with 50% simulated packet loss | every received packet is 0x02 with neutral sticks |
| Graceful shutdown (SIGINT, live backend) | 12 land packets with neutral sticks |
| RTSP capture estimate, steady (mediamtx on loopback, HEVC 1280x720 at 15 fps) | receive minus capture at most 11 ms |
| RTSP freeze of 3 s | no reconnect (2 s read timeout); frames after the freeze read 2.9 s old, so the governor would land |

Closed loop (`mhd.autopilot.sim`): kinematic plant with yaw lag 0.15 s and forward lag 0.4 s, 15 fps camera, HFOV 75°, subject at 3.2 m. Total delay θ includes the observation delay, half the camera and RC periods, and the yaw lag.

| θ | Yaw step 20°: overshoot / settle (7θ) | S1 bearing RMS, P-only (default) | S1 RMS, feed-forward on | FF on, K ±30%: worst step overshoot / S1 RMS |
|---|---|---|---|---|
| 0.6 s | 0% / 2.0 s (4.2) | 0.083 rad | 0.040 | 2.5% / 0.055 |
| 1.0 s | 0% / 3.1 s (7.0) | 0.135 rad | 0.066 | 0% / 0.093 |
| 1.5 s | 0% / 4.6 s (10.5) | **0.196 rad** (PLAN wants ≤ 0.15) | 0.104 | 0% / 0.139 |

- S1 is one walker at about 0.06 rad/s bearing rate, walking right, pausing and walking back. The drone never came closer than 2.76 m, and the subject stayed in frame 100% of the time.
- With a reversed yaw sign, the wrong-way detector stops the runaway: maximum bearing error 0.68 rad, then hover. With the detector off, the error reaches 2.77 rad.
- A 0.3 rad/s walker (0.96 m/s lateral at 3.2 m) is **not** held in frame (40–80% in frame). The loop is delay-limited, and SEARCH reacquires the subject.

## Behavior and decisions (written down because nobody could be asked)

1. **Staleness is "lag beyond the best-case pipeline".** Literally, `now - capture_ms > 0.4 s` would block the autopilot forever, because normal latency is 0.5–1.2 s and the twin injects up to 1.5 s.
   - Video lag = (now − capture_ms) − min over 10 s of (receive − capture). The floor is capped at 1.5 s for video and 2.0 s for observations, so a very slow pipeline can't excuse itself.
   - In real mode with the pts estimate and offset 0, the floor is about 0, so this is the literal age.
   - Thresholds:
     - observation lag above max(0.4 s, 2.5 × measured observation period): autopilot neutral;
     - video lag above 0.7 s: autopilot neutral; above 2.0 s: land.
   - Killing perception gives neutral in 0.42 s on the wire.
2. **The heartbeat is any `/ws` message.** That includes the Web Worker's `{"type":"hb"}` and the legacy UI's 30 Hz `axes`, so the stock UI works too.
   - `POST /command` does not refresh it, on purpose.
   - Missing 1.5 s: all sticks neutral. Missing 5 s: land.
   - Takeoff needs a fresh heartbeat (unless twin headless) and fresh video.
3. **Video stale neutralizes only the autopilot.** The human keeps the sticks for line-of-sight flying. A heartbeat loss neutralizes everything.
4. **Human input.**
   - Stale 0.25 s after the last nonzero message.
   - The deadzone is 0.05, so gamepad drift is not input.
   - An all-zero message releases the sticks immediately but never counts as input.
   - `inc` mode (keyboard) ramps at 1.0/s toward the human cap. A reversal goes through zero, and release is immediate.
   - Human caps are 64 counts (0.5) on every axis. Autopilot caps follow PLAN: yaw 38, pitch 32, roll 0, throttle 0.
5. **Priority:** E-stop > land > human > autopilot. Human sticks during LANDING are ignored.
6. **LAND works from every state, including ON_GROUND.**
   - It goes to LANDING and re-asserts 0x80 on a fixed 200 ms grid.
   - `confirm_touchdown` returns to ON_GROUND. The drone sends no telemetry, so the operator must confirm.
7. **`resume` during LANDING cancels the landing**, whatever caused it.
   - The autopilot stays paused; a second `resume` restarts following.
   - A cause that still holds (heartbeat missing, video frozen) lands again on the next tick.
   - The flight-timer landing fires once per flight, so one explicit resume overrides it.
   - Every landing pauses the autopilot, so a re-takeoff never starts following by itself.
8. **E-stop.**
   - `estop_reset` returns to ON_GROUND.
   - In real mode ARM must be typed again (`estop.rearm_after_reset: true`).
   - A mission is stopped by E-stop, and perception gets `Release`.
9. **Agent (voice/LLM) commands are limited to `pause` and `land`**, enforced in the governor and the hub.
   - No E-stop and no takeoff from a model.
   - A Mission `hover` pauses; `land` lands; `follow` starts FIND and unpauses.
10. **Slew.** Increases in magnitude are limited to 1.0 per second. Moving toward zero is immediate, and every safety neutral bypasses the slew.
11. **Deadband compensation** applies to autopilot output only (default 0 counts). Human sticks are mapped plainly.
12. **The autopilot runs whenever a mission is active**, including on the ground, so it can already be in FOLLOW. It moves only when AIRBORNE and allowed.
    - SEARCH waits in LOST while it may not move.
    - FIND gives up after 25 s, and LOST/SEARCH after 25 s.
    - GIVE_UP holds until `resume` (which restarts FIND) or a new mission.
13. **Wrong-way detector.**
    - Compares the observed bearing change over 1.5θ with the change predicted from the yaw actually sent (accounting for yaw lag).
    - It latches a hold until `resume` or a new follow mission.
    - It can false-alarm when the subject outruns the drone. The result is a hover, which is safe.
14. **Reference widths were re-derived.**
    - With HFOV 75° and a hover height of about 1.2 m, a standing person only fits the frame (box height under 0.8 H, feet inside the 2% margin) from about 2.9 m.
    - New values: close 0.13 W (about 2.25 m), medium 0.10 (about 2.9 m), far 0.075 (about 3.9 m).
    - The old 0.14 W "medium" sat inside the standoff zone.
15. **Capture time.**
    - Twin: `mhd.twin.counter.decode` when importable and decodable.
    - Otherwise: pts + min over 10 s of (receive − pts) − `MHD_CAPTURE_OFFSET_MS` (window: `MHD_CAPTURE_WINDOW_S`).
    - A frame without pts in a stream that had them keeps the last estimate (`pts_hold`), so it can't look fresh.
    - Receive time is used only if the stream never carried pts.
16. **PacedFlightController** replaces upstream's fixed `sleep(0.04)` after work, which gave 20.7 Hz with p99 63 ms. The upstream file is unchanged, and both loops are tested.
17. **One HEVC decoder thread.** `OPENCV_FFMPEG_THREADS=1` is set by default in `apply_env_defaults()` and `run_backend.sh`. OpenCV otherwise frame-threads the decoder (16 threads here), which the twin workstream measured at about 0.7 s of extra latency. A pts-based capture estimate cannot see that constant delay, so it would silently add to θ.
    - RTSP read timeout is 2 s; open timeout is 10 s, because opening needs a keyframe and the X69's keyframe interval is unknown. Override with `MHD_RTSP_READ_TIMEOUT_MS` / `MHD_RTSP_OPEN_TIMEOUT_MS`.
18. **Logging.** Ticks go to JSONL through a QueueHandler. Governor warnings also go through a queue, because a stalled stderr pipe must never block the RC thread.

## Telemetry vocabulary (for the UI)

- `active_source` values:
  - `none`, `human`, `autopilot`, `takeoff`, `landing`, `estop`, `shutdown`.
- `safety_flags` values:
  - `heartbeat_missing`, `video_stale`, `obs_stale`, `flight_timer_warn`, `wrong_way`, `autopilot_paused`, `autopilot_fault`, `autopilot_slow`;
  - `landing:<cause>`, where the cause is `ui_land`, `http_land`, `agent_land`, `model_land`, `heartbeat_lost`, `video_frozen` or `flight_timer`;
  - `headless_bypass`, `governor_fault`, `rc_tick_stale`.
- `latency_ms` keys:
  - `frame_lag`, `obs_lag`, `frame_floor`, `obs_floor`, `obs_period`, `hb_age`, `autopilot_step`, `rc_tick_age`, `frames_received`, `obs_accepted`.
- `sticks_by_source` keys:
  - `human`, `autopilot`, `output` (effective normalized output after caps).

## What the integrator must wire up

- **UI (ws/ui):**
  - Send anything on `/ws` at 5 Hz or more (`{"type":"hb"}` from the worker).
  - Commands: `{"type":"cmd","command":...}` on `/ws`, with `POST /command` as the fallback.
  - Real mode: `{"type":"cmd","command":"arm","arm_phrase":"ARM"}` (exact, after strip).
  - Needs a **Landed** button (`confirm_touchdown`) and a **Resume** button (cancels a landing, or unpauses).
  - Tilt: `tilt_up`/`tilt_down`/`tilt_stop`. Tilt stops by itself after 3 s; the legacy `camera_tilt_direction` also works.
  - `pack_swap` with `pack_id`, on the ground only.
  - Read `/ws/telemetry`.
  - **`turbodrone/frontend/src/lib/ws.ts` is missing from git**: turbodrone's own `.gitignore` rule `lib/` hid it. It exists in the main checkout's working tree. ws/ui must `git add -f` it or fix the rule. I did not touch the frontend.
- **Agent (ws/agent):**
  - Connect to `/ws/agent?role=agent`.
  - Send `Mission` there or to `POST /mission`; acks come back as `mission_ack` / `cmd_ack`.
  - Safety words: `{"type":"cmd","command":"pause"}` for stop/hover, `{"type":"cmd","command":"land"}` for land. Anything else is rejected with `not_allowed_from_agent`.
  - Autopilot announcements ("Got them. Following.", "I lost the subject. Looking around.", "I can't find the subject. Holding position.") and safety events arrive as `AgentEvent(kind="status")`, along with `Candidates`.
- **Perception (ws/percep):**
  - Connect to `/ws/subject`. You get the current `FindRequest` on connect and on each new mission, plus `Release` and `Select`.
  - Pull frames from `/ws/frames`.
  - Send `SubjectObs` **at camera rate, even when nothing is found** (`state:"searching"`, no box). Otherwise the observations go stale and SEARCH cannot move.
  - `capture_ms` must be the `FrameMeta.capture_ms` of the source frame. Also set `mission_id`, give `confirm_age_ms` as of `capture_ms`, and send `detector_latency_ms` (it drives the DOUBT and LOST timers).
  - On a tracker-score drop, re-run the detector on your own: there is no "redetect" message in the contracts. The autopilot's DOUBT follows from your low score.
- **Twin (ws/twin):**
  - Provide `mhd.twin.counter.decode(img_bgr) -> Optional[(render_id, render_ms)]` on the shared monotonic clock. The backend picks it up automatically in twin mode.
  - Use `scripts/run_backend.sh --mode twin --rtsp-url ... --control-port ...`.
- **After Session A/B, update:**
  - `autopilot.yaml`: `camera.hfov_deg`, `plant.yaw_rate_per_count`, `plant.forward_speed_per_count`, `plant.yaw_lag_s`, `delay.theta_s`.
  - `governor.yaml`: `wire.sign`, `wire.deadband_counts`, `flight_timer.land_s` (60 s before low battery).
  - Environment: `MHD_CAPTURE_OFFSET_MS` (minimum camera-to-backend latency).
- **Merge notes:**
  - `.gitignore` changed (`models/` became `/models/`), and `turbodrone/backend/models/*.py` were added unchanged. Identical additions from other branches merge cleanly.
  - Contracts are unchanged; no adapter was needed.

## Known gaps and risks

- **Not tested against the twin's full wire-level emulator.** Its renderer needs Chrome, which gov may not run. I did run the twin's own RC decoder (`ws/twin` `mhd/twin/protocol.py`, loaded read-only) on my governor's packets: takeoff, sticks at ±38/±64 counts, tilt and land all decode identically.
  - Otherwise tested against a local UDP decoder, an in-process kinematic sim, and real mediamtx/ffmpeg RTSP.
  - The twin's `count_to_unit` divides negative offsets by 128; the governor sends symmetric counts (±127 full scale). Only counts reach the drone, so this matters only for the twin's dynamics (0.8% on negative sticks).
- **Feed-forward is off by default.** The in-process sim shows it is robust to ±30% K error and brings S1 at θ = 1.5 s to 0.104 rad, but PLAN requires a wire-level twin run first. Enable it with `yaw.feedforward.enabled: true`.
- **P-only S1 RMS at θ = 1.5 s is 0.196 rad**, against a 0.15 target. The 0.3 rad/s "stays in frame" target fails with or without feed-forward.
- **No RC-thread watchdog that sends packets itself.** If the RC thread stopped, telemetry shows `rc_tick_stale` and the log says so, but nothing else sends neutral or land. Session A step 6 ("stop RC packets while the keepalive continues") tells us what the drone does.
- **Throttle-ramp landing is built but off** (`landing.throttle_ramp: false`). If Session A shows 0x80 does not land the drone, set it to true. While LANDING, throttle then ramps to −0.35 (44 counts) over 1 s, and the land flag is still re-asserted.
  - It is tested only in unit tests: check it at low height in Session B before relying on it.
- **The capture estimate re-adapts after 10 s** if the stream's timeline shifts permanently (by spec). Stall-then-burst, the realistic case, is detected.
- **GIL contention** in the backend process can delay an RC tick by tens of ms (bench max 34 ms with 2 busy threads). The paced loop keeps the 25 Hz average, and the wire-test p99 was 48 ms.
- **First-flight defaults are guesses** until Sessions A and B: HFOV 75°, K 0.012 rad/s per count, θ 1.0 s, reference widths, deadband 0. The forward gate (box clear of the top and bottom edges) means forward motion only happens beyond about 2.9 m at a 1.2 m hover; closer in, the autopilot is yaw-only.
- No paid API calls were made, so there are no `spend.jsonl` entries.
