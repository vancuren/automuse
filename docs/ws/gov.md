# Workstream gov: backend patches, safety governor, autopilot

Branch `ws/gov`. Built overnight 2026-11-03/04. PLAN.md sections 2, 4.1, 4.4, 4.5 and 6.
An adversarial review (4 major, 10 minor findings) was fixed at 01:30–02:40 ET: see **Review fixes** below.
Several behaviors changed for other workstreams; they are listed there under "What changed for other workstreams".

## What exists

| Path | What it is |
|---|---|
| `config/governor.yaml`, `config/autopilot.yaml` | Every safety and control parameter, with commented first-flight defaults. Values marked [Session A] or [Session B] are guesses until measured. Unknown keys are errors. |
| `src/mhd/governor/config.py` | Strict YAML loaders (frozen dataclasses). `GOVERNOR_CONFIG` and `AUTOPILOT_CONFIG` override the paths. |
| `src/mhd/governor/core.py` | `Governor`. One `tick()` per RC packet: flight state machine, priorities, staleness checks, flight timer, wrong-way detectors (yaw, pitch, frozen box), slew limit, wire counts, flags. Inputs arrive from other threads through `heartbeat`, `submit_human`, `set_tilt`, `note_frame`, `submit_obs`, `set_mission` and `command`; frames and observations with non-finite numbers or boxes outside the frame are refused. |
| `src/mhd/governor/wire.py` | Normalized to wire counts: per-axis sign, deadband compensation (autopilot only), caps in counts, symmetric rounding (127 counts each side). |
| `src/mhd/governor/turbodrone_glue.py` | `GovernorStrategy` (turbodrone ControlStrategy: the choke point). `GovernedX69LgRcModel` (`set_strategy` refuses to replace the governor; speed tier pinned at 2; legacy `takeoff`/`land`/`emergency_stop` go through the governor; `flip`/`calibrate` refused). `PacedFlightController` (deadline-paced 25 Hz; refuses rates outside 10–50 Hz). `GovernedX69LgRcAdapter` (upstream RC adapter whose D1 keepalive survives OSError, with send-error counters for telemetry). |
| `src/mhd/governor/sysid.py` | Session B step identification. Flies ± steps at chosen amplitudes through the governor (human priority, so LAND/E-STOP still win), measures an ArUco marker on `/ws/frames`, and fits K per count, firmware deadband and lag. |
| `src/mhd/governor/flight_timer.py`, `tick_log.py`, `logq.py`, `stats.py`, `telemetry.py`, `bench.py` | Per-pack timer persisted off the RC thread; per-tick JSONL through a QueueHandler; non-blocking governor logging; session minimum (latency floors) and sliding minimum; contract `Telemetry` builder; tick benchmark. |
| `src/mhd/autopilot/core.py` | `autopilot_step(obs, state, params, now_ms, *, can_move, applied_yaw, applied_pitch) -> (sticks, new_state, events)`: a pure function with the IDLE, FIND, CONFIRM, FOLLOW, DOUBT, LOST, SEARCH, GIVE_UP lifecycle. |
| `src/mhd/autopilot/sim.py` | In-process closed-loop sim: the real `Governor` flying a kinematic X69 after a walking subject, with delayed oracle observations. |
| `src/mhd/backend/` | `modes.py` (FLIGHT_MODE gate), `capture.py` (capture-time estimate plus `TimedX69RtspAdapter`), `frames.py` (newest-frame store for `/ws/frames`), `hub.py` (all new endpoints; routes every UI, perception and agent message to the governor). |
| `turbodrone/backend/web_server.py` | Patched, about +45/-80 lines. Every change is marked `MHD patch`: FLIGHT_MODE gate, governed model and RC adapter, paced loop, `/ws` through the hub, signal hooks that land on the first Ctrl-C, MJPEG streams that end on shutdown, loopback-only CORS. |
| `turbodrone/backend/plugins/follow/` | Deleted, and `ultralytics` removed from `requirements.txt`. `PLUGINS_ENABLED=true` is refused. |
| `turbodrone/backend/models/` | Restored. The root `.gitignore` rule `models/` matched it at any depth, so the import commit left the package out and the backend could not start from a clean checkout. The rule is now `/models/`, and the 15 upstream files are added byte-identical. |
| `scripts/run_backend.sh` | Starts the patched backend from the checkout it lives in. |
| `tests/gov/` | 176 pytest tests (see below); `test_review_fixes.py` holds the regression tests for the review. |

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

Without `--mode` it refuses to start. In real mode it also refuses an RTSP URL that is not the drone's (172.16.11.1) and any `CONTROL_RATE` other than 25. Ctrl-C (or SIGTERM) starts the landing at once: every following RC packet is neutral sticks plus land; uvicorn then waits at most 1 s for open responses (`--timeout-graceful-shutdown 1`), and the RC loop stops after at least 0.5 s of land. A second Ctrl-C still sends land for 0.5 s before exiting. The server binds 127.0.0.1 and never uses `--reload`. Control logs go to `$MHD_LOGS/control-*.jsonl`, the flight timer to `$MHD_LOGS/flight_timer.json`; `--logs DIR` puts both elsewhere (the tests use it so they never add flight time to a real pack). Send `pack_swap` with a `pack_id` at the start of each session.

Tests and tools:

```bash
$MHD_PY -m pytest tests/gov -q -p no:logging            # 176 tests, about 135 s (RTSP and live tests ~60 s)
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

It refuses unless AIRBORNE and flies each amplitude + then −. It aborts the moment the drone is not AIRBORNE, the governor is not flying its sticks (0.3 s into a step), or (real mode) the UI heartbeat is older than 1 s. Its own axes never count as the UI heartbeat in real mode, so a dead UI still lands the drone.
 Put the fitted K, deadband and lag into `config/autopilot.yaml` (`plant.*`) and `config/governor.yaml` (`wire.deadband_counts`).


`-p no:logging` only hides pytest's captured-log dump on failures.

## Test results (final run after the review fixes, 02:10 ET)

- `tests/gov`: **176 passed, 0 failed** (about 135 s).
  - 67 review regression tests (`test_review_fixes.py`), one section per finding.
  - 42 governor rule tests (`test_governor_rules.py`), including the throttle-ramp landing fallback.
  - 14 autopilot lifecycle and geometry tests.
  - 7 turbodrone glue tests.
  - 2 UDP wire tests (paced and upstream FlightController).
  - 15 closed-loop sim tests.
  - 17 FastAPI TestClient tests (now including CORS and Origin on the real app).
  - 4 capture-clock tests.
  - 1 RTSP integration test (mediamtx with an ffmpeg HEVC stream).
  - 3 live tests (`run_backend.sh` with uvicorn, RTSP and a UDP drone stand-in): end to end, and Ctrl-C with /mjpeg held open, with and without a force quit.
  - 4 sysid tests: schedule limits, ArUco bearing and range on a synthetic frame, fit recovering K / deadband / lag, and a live run through the real backend with an ArUco HEVC stream.
- Upstream x69_lg unit tests: **15 passed.**
- Repo-level `tests/test_contracts.py` and `tests/test_intent_spec.py`: **6 passed.** All tests together: **197 passed.**
  - The final full run showed one pytest thread warning: sysid's frame-reader thread raised when the backend closed its socket at test teardown. Fixed afterwards (the thread now exits quietly); the sysid tests were re-run and pass.
- Tests changed by the review fixes, on purpose: takeoff and Landed go over `/ws` in the API, live and sysid tests; the E-stop test expects LANDING after reset; the timer test expects the landing to fire again; the wire test presses Resume after a mission sent 2 s after takeoff; the test rig's steady observations jitter ±1 px.


## Measurements

Rows marked "after the review fixes" were measured 01:58–02:10 ET with the other workstreams idle (load average 0.9–2.5, no Chrome, no GPU model; mediamtx and an ffmpeg hevc_videotoolbox stream ran for the live tests). The rest were measured 23:58–00:43 ET on the M5 Pro after the model downloads finished (23:40:51, so the network was idle).
 Five other workstreams were running alongside: Chrome/Playwright and node (twin/ui), Python processes, and the percep workstream's ffmpeg. Load average was 3.5–4.0. No GPU model was loaded by gov.

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
| Graceful shutdown (SIGINT, live backend, no video client) | 12 land packets with neutral sticks (13 after the review fixes) |
| **After the review fixes:** Ctrl-C while a client holds /mjpeg open, drone AIRBORNE (live backend, real HEVC RTSP) | first land packet 3 ms after SIGINT; 29 land packets over 1.1 s; process gone after 2.3 s. Before the fix: 0 land packets, the last decision kept flying. |
| Same, with a second Ctrl-C 50 ms later (force quit) | first land 41 ms; 15 land packets over 0.6 s; gone after 0.7 s |
| **After the review fixes:** governor tick, same benchmark | p50 41 µs, p99 70 µs, max 192 µs (was 35 / 57 / 160; the pitch and frozen-box checks walk two more short histories) |
| **After the review fixes:** real-mode capture clock + governor, true latency creeping 0.5 → 3.5 s over 60 s (reviewer's p07 with `make_capture_clock("real")`) | autopilot neutral (`obs_stale`) at 10.7 s, `video_stale` at 14.4 s, safety landing at 43.0 s. Lasting +1.5 s step at 5 s: neutral at 6.0 s and for the rest of the 40 s run (frame lag 1.47 s at the end). |

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

## Review fixes (01:30–02:40 ET)

An adversarial review found 4 major and 10 minor problems. All 14 are fixed. Each has a regression test (mostly in `tests/gov/test_review_fixes.py`), and the reviewer's own probes in `logs/review/probes` (gitignored) were re-run against the fixed code.

1. **(major) A latency creep or a lasting rise was invisible in real mode.** Both the capture estimate (pts + 10 s minimum of receive − pts) and the governor's floors (10 s minimum of receive − capture) re-baselined.
   - Governor floors are now session minimums (`stats.SessionMin`): they never rise.
   - The real-mode capture clock uses the session minimum of (receive − pts), allowed to rise only by camera clock drift: at most 2 ms/s and 300 ms in total (`MHD_CAPTURE_DRIFT_MS_PER_S`, `MHD_CAPTURE_MAX_RISE_MS`). The rise is in telemetry (`capture_rise`). Twin mode keeps the 10 s window, because the twin's pts may be frame-count based and its counter block is the primary source.
   - I kept "lag beyond the best case" rather than a fixed literal age budget: the literal reading (age > 0.4 s) blocks the autopilot forever at real latencies. With session floors, every rise above the best case counts.
   - Result (reviewer's p07 with the real-mode clock): creep 0.5 → 3.5 s over 60 s gives autopilot neutral at 10.7 s (`obs_stale`), `video_stale` at 14.4 s and a safety landing at 43.0 s (it used to keep yawing and flying forward). A lasting +1.5 s step gives neutral 1.0 s after the step, held for the whole run (it used to resume after 10 s on 2 s old video). 100 ppm of camera clock drift over 7 minutes adds no lag.

   - Tests: `test_latency_creep_is_seen_and_lands`, `test_lasting_latency_step_keeps_the_autopilot_neutral`, `test_slow_camera_clock_drift_is_absorbed`, `test_observation_latency_creep_goes_stale`, `test_governor_floors_never_rise`, `test_capture_env_values_are_validated`.
2. **(major) Ctrl-C with the UI's /mjpeg open never landed.** uvicorn ran the lifespan shutdown only after every open response finished, and skipped it after a second Ctrl-C.
   - `Hub.install_signal_hooks()` wraps uvicorn's SIGINT/SIGTERM handlers: the first signal calls `begin_shutdown()` at once (every following packet is neutral sticks plus land; a latched E-stop stays E-stop), and ends every MJPEG stream. A second Ctrl-C (force quit) first lets the RC thread send land for at least 0.5 s.
   - `run_backend.sh` adds `--timeout-graceful-shutdown 1`.
   - Result, live backend with real HEVC RTSP and a client holding /mjpeg: first land packet 3 ms after SIGINT, 29 land packets over 1.1 s, exit after 2.3 s. With a force quit 50 ms later: first land 41 ms, 15 land packets over 0.6 s, exit after 0.7 s. It used to keep flying the last decision and send 0 land packets.
   - Unchanged by design: the listening socket closes on the first Ctrl-C, so E-STOP over the network is unreachable during that last ≤ 1.5 s. The drone is already landing then.
   - Tests: `test_ctrl_c_lands_even_with_mjpeg_open[False/True]` (live), `test_begin_shutdown_lands_now_and_runs_callbacks`, `test_signal_hook_begins_shutdown_then_chains`, `test_signal_hooks_need_the_main_thread`.
3. **(major) A late follow Mission overrode a voice "stop" and re-armed the autopilot after an E-stop.**
   - `set_mission(follow)` never unpauses on its own authority and never clears a wrong-way hold. It releases a pause only if it was spoken after the last pause / land / E-stop / takeoff: spoken_at = arrival − `latency_ms` − 3 s (`mission.utterance_slack_s`, covering ASR time that `latency_ms` leaves out; a non-finite or out-of-range `latency_ms` counts as 30 s). Otherwise it is accepted but stays paused until Resume, the ack says `"paused": true`, and a status event explains it.
   - The contract has no push-to-talk timestamp (contracts are frozen), so `latency_ms` plus the slack is the bound. A "stop" up to about 2.5 s before the end of the next "follow" utterance also keeps that follow paused: the safe side.
   - E-stop drops the mission; a follow Mission is refused while E-stopped. Takeoff pauses the autopilot, so a mission set before a takeoff waits for Resume.
   - Result (reviewer's p04): the late mission is acked `paused_after_utterance` and the drone stays neutral; after E-stop, reset, ARM and takeoff nothing follows by itself.
   - Tests: `test_late_mission_does_not_override_a_later_pause`, `test_mission_spoken_after_the_pause_releases_it`, `test_estop_drops_the_mission_and_refuses_late_missions`, `test_mission_set_on_the_ground_waits_for_resume_after_takeoff`, `test_new_mission_never_clears_a_wrong_way_hold`, `test_mission_with_unusable_latency_counts_as_slow`.
4. **(major) Config could switch the safety checks off in real mode** (huge thresholds, YAML `.nan`, `require_heartbeat: false`, negative slew, any `CONTROL_RATE`).
   - Every float must be finite. Safety values have upper bounds: heartbeat neutral ≤ 3 s and land ≤ 10 s, video lag ≤ 1 s and ≤ 3 s, observation stale_min ≤ 1 s, flight timer ≤ 600 s, slew in (0, 10], human stale ≤ 1 s, and more (`GovernorConfig.validate`). The Governor validates configs built in code too.
   - Real mode refuses `takeoff.require_heartbeat: false` (like `twin_headless_bypass`). `CONTROL_RATE` must be 10–50 Hz, and exactly 25 in real mode; `PacedFlightController` also refuses bad rates.
   - Result (reviewer's p02): all four configs are refused at load.
   - Tests: `test_config_refuses_unsafe_or_non_finite_values` (10 cases), `test_real_mode_refuses_takeoff_without_heartbeat_requirement`, `test_governor_validates_configs_built_in_code`, `test_control_rate_is_validated`, `test_paced_controller_refuses_bad_rates`.
5. **(minor) A NaN `capture_ms` defeated the staleness checks.** `submit_obs` refuses non-finite `capture_ms`, `confirm_age_ms`, `tracker_score` (and scores outside 0..1), `confidence`, `detector_latency_ms` and box coordinates, and boxes outside 1280x720 by more than 5 %. `note_frame` refuses non-finite and future capture times. The hub parses every socket with `loads_strict`, which refuses NaN and Infinity. Result (p01): the NaN observation is refused, and neutral comes 0.4 s after perception dies. Tests: `test_invalid_observations_are_refused` (10 cases), `test_nan_capture_obs_cannot_keep_the_autopilot_alive`, `test_frames_with_bad_capture_times_are_refused`, `test_hub_json_refuses_nan_and_infinity`.
6. **(minor) No wrong-way check for pitch.** New pitch detector (decision 13). I did not add a separate cap on continuous forward time: the pitch detector catches the runaway. Result (reviewer's p10, pitch reversed in the sim): the detector fires and the drone stops 0.4 m further away; it used to fly from 4.0 m to 12.7 m. With the correct sign nothing fires. Tests: `test_pitch_wrong_way_hovers`, `test_pitch_check_quiet_when_approach_works`, plus the 15 closed-loop sim tests.
7. **(minor) Commands out of context.** Landed only in LANDING; resume returns to the state the landing started from; the timer landing fires again 20 s after a resume; `estop_reset` goes to LANDING (decisions 6–8). No separate "abort takeoff": the `OperatorCommand` contract is frozen, and LAND then Landed clears a lost takeoff. Result (p03): all four cases behave. Tests: `test_confirm_touchdown_only_while_landing`, `test_takeoff_cannot_be_confirmed_away`, `test_resume_after_land_on_ground_stays_on_ground`, `test_resume_during_takeoff_landing_restarts_the_hold`, `test_resume_postpones_timer_landing_it_does_not_cancel_it`, `test_estop_is_latched_until_reset`.
8. **(minor) HTTP had full rights; any web page could command the drone.** `POST /command` allows only `land`, `estop`, `estop_reset` and `pause` (the UI's fallback; ws/ui's tests use `estop_reset` over HTTP), and `?role=agent` applies the agent whitelist. `/ws`, `/ws/subject`, `/ws/agent`, `POST /command` and `POST /mission` refuse non-loopback `Origin`s (`MHD_EXTRA_ORIGIN_HOSTS` adds hosts), and CORS is loopback-only instead of `*`. Tests: `test_post_command_whitelist_and_agent_role`, `test_websockets_refuse_foreign_origins`, `test_cors_and_ws_origin_on_the_real_app`, `test_estop_clears_the_find_request`.
9. **(minor) Legacy UI stuck key; any zero released the override.** Real mode accepts only `OperatorInput` on `/ws` (the legacy stream is refused and flagged); zeros release human input only from the socket that owns it, and a disconnect releases only its own input. Twin mode keeps the legacy UI working on purpose. Tests: `test_real_mode_refuses_legacy_axes_and_counts_only_ui_heartbeats`, `test_zeros_from_another_socket_do_not_release_the_override`.
10. **(minor) Real mode accepted a loopback RTSP URL.** It now refuses any RTSP host other than 172.16.11.1. Test: `test_real_mode_rtsp_url_must_be_the_drone` (4 cases).
11. **(minor) The keepalive died on the first OSError; send errors were silent.** `GovernedX69LgRcAdapter` (subclass; the upstream file is unchanged) logs and retries the keepalive, counts RC and keepalive errors, and telemetry shows `rc_send_errors` / `keepalive_stale`. Test: `test_keepalive_survives_oserror_and_errors_are_counted`.
12. **(minor) A failing tick delayed land by 1.7 s while LANDING.** The fallback now asserts land on the 5 Hz pattern at once when the governor is LANDING or shutting down (read without the lock, like the E-stop latch). Test: `test_fallback_lands_immediately_when_already_landing`.
13. **(minor) Boxes never checked; a stuck tracker spun the drone.** Out-of-frame boxes are refused (fix 5). Instead of the suggested general "no response" check (bearing unchanged while yawing), I added the frozen-box check: a general check false-alarms on good tracking of a moving subject, whose bearing stays constant while the drone turns with it. A truly stuck tracker or repeating frame gives the very same box, which real tracking never does. Tests: `test_frozen_box_while_turning_hovers`, `test_frozen_box_quiet_when_not_turning`. (The test rig's steady observations now jitter ±1 px, as a real tracker's do.)
14. **(minor) sysid's axes counted as the heartbeat; wrong docstring and abort test.** Real mode counts only `{"type":"hb","client":"ui"}`; sysid aborts when the drone is not AIRBORNE, when the governor is not flying its sticks (0.3 s into a step), or when the UI heartbeat is older than 1 s; the docstring says so. Test: `test_sysid_abort_reasons`, plus the live sysid run.

**What changed for other workstreams** (also in "What the integrator must wire up"):
- ws/ui: E-stop reset ends in LANDING (press Landed); Landed only in LANDING; takeoff, arm, resume and Landed only over `/ws`; pages must be on localhost (or `MHD_EXTRA_ORIGIN_HOSTS`); real mode counts only the worker's `hb` and only `OperatorInput` axes.
- ws/agent: use `POST /command?role=agent`; a follow Mission after a stop may be acked `"paused": true`.
- ws/percep: finite numbers only, boxes inside the frame (already true with `clip_box`).
- Operators: a mission set before takeoff, or a follow said during the takeoff hold, needs Resume.

## Behavior and decisions (written down because nobody could be asked)

1. **Staleness is "lag beyond the best-case pipeline".** Literally, `now - capture_ms > 0.4 s` would block the autopilot forever, because normal latency is 0.5–1.2 s and the twin injects up to 1.5 s.
   - Video lag = (now − capture_ms) − the **session minimum** of (receive − capture). The floor only ever goes down, so a latency that creeps up or rises and stays high shows up as lag (review fix 1; it was a 10 s sliding minimum, which re-baselined). The floor is capped at 1.5 s for video and 2.0 s for observations, so a very slow pipeline can't excuse itself.
   - In real mode the capture estimate is pts + a best-case baseline that may rise only by camera clock drift (2 ms/s, 300 ms at most), so the floor is about the Session A offset and the lag is the literal age beyond it.
   - Thresholds:
     - observation lag above max(0.4 s, 2.5 × measured observation period): autopilot neutral;
     - video lag above 0.7 s: autopilot neutral; above 2.0 s: land.
   - Killing perception gives neutral in 0.42 s on the wire.
2. **The heartbeat.** Real mode: only `{"type":"hb","client":"ui"}` (the UI's Web Worker) counts, so a tool streaming axes (sysid) cannot mask a dead UI (review fix 14). Twin mode: any `/ws` message counts, so the stock upstream UI works against the twin.
   - `POST /command` does not refresh it, on purpose.
   - Missing 1.5 s: all sticks neutral. Missing 5 s: land.
   - Takeoff needs a fresh heartbeat (unless twin headless) and fresh video.
3. **Video stale neutralizes only the autopilot.** The human keeps the sticks for line-of-sight flying. A heartbeat loss neutralizes everything.
4. **Human input.**
   - Stale 0.25 s after the last nonzero message.
   - The deadzone is 0.05, so gamepad drift is not input.
   - An all-zero message releases the sticks immediately but never counts as input, and only from the socket that sent the input: an idle second tab streaming zeros cannot break the operator's override (review fix 9).
   - Real mode accepts only the contract's `OperatorInput` on `/ws`. The legacy upstream UI's axes (30 Hz, `camera_tilt_direction`, no blur handler, so a key stuck after focus loss never went stale) are refused and flagged `legacy_axes_refused`.
   - `inc` mode (keyboard) ramps at 1.0/s toward the human cap. A reversal goes through zero, and release is immediate.
   - Human caps are 64 counts (0.5) on every axis. Autopilot caps follow PLAN: yaw 38, pitch 32, roll 0, throttle 0.
5. **Priority:** E-stop > land > human > autopilot. Human sticks during LANDING are ignored.
6. **LAND works from every state, including ON_GROUND.**
   - It goes to LANDING and re-asserts 0x80 on a fixed 200 ms grid.
   - `confirm_touchdown` returns to ON_GROUND, and is accepted **only in LANDING** (review fix 7; "Landed" while AIRBORNE used to silence every safety landing). The drone sends no telemetry, so the operator must confirm. A lost takeoff (0x01 never arrived) is cleared with LAND, then Landed.
7. **`resume` during LANDING cancels the landing**, whatever caused it, and returns to the state the landing started from: AIRBORNE, TAKING_OFF (a fresh neutral hold, no second 0x01) or ON_GROUND (LAND pressed on the ground). So resume can never bypass ARM or "ON_GROUND: neutral only" (review fix 7).
   - The autopilot stays paused; a second `resume` restarts following.
   - A cause that still holds (heartbeat missing, video frozen) lands again on the next tick.
   - A cancelled flight-timer landing fires again after `flight_timer.refire_after_s` (20 s) of flight.
   - Every landing and every takeoff pauses the autopilot, so a (re-)takeoff never starts following by itself.
8. **E-stop.**
   - `estop_reset` goes to **LANDING** (review fix 7): if 0x02 did not cut the motors, the drone still lands. The operator presses Landed as after any landing; a resume from it returns to ON_GROUND.
   - In real mode ARM must be typed again (`estop.rearm_after_reset: true`).
   - E-stop ends the mission (the governor drops it, perception gets `Release`, the hub forgets the FindRequest), and a follow Mission is refused while E-stopped.
9. **Agent (voice/LLM) commands are limited to `pause` and `land`**, enforced in the governor and the hub (`/ws/agent`, and `POST /command?role=agent`).
   - No E-stop and no takeoff from a model. `POST /command` without a role accepts only `land`, `estop`, `estop_reset` and `pause` (the UI's HTTP fallback); takeoff, arm, resume, Landed and the rest need `/ws`.
   - A Mission `hover` pauses; `land` lands; `follow` starts FIND. It releases a pause only if it was spoken after the last pause / land / E-stop / takeoff: spoken_at = arrival − `latency_ms` − 3 s (`mission.utterance_slack_s`). Otherwise it is accepted (FIND runs, perception can lock on) but stays paused until Resume, and the ack says `"paused": true`. A Mission never clears a wrong-way hold (review fix 3).
10. **Slew.** Increases in magnitude are limited to 1.0 per second. Moving toward zero is immediate, and every safety neutral bypasses the slew.
11. **Deadband compensation** applies to autopilot output only (default 0 counts). Human sticks are mapped plainly.
12. **The autopilot runs whenever a mission is active**, including on the ground, so it can already be in FOLLOW. It moves only when AIRBORNE and allowed.
    - SEARCH waits in LOST while it may not move.
    - FIND gives up after 25 s, and LOST/SEARCH after 25 s.
    - GIVE_UP holds until `resume` (which restarts FIND) or a new mission.
13. **Wrong-way detectors.** All latch a hover until the operator's `resume` (a new mission no longer clears it). Flags: `wrong_way` plus `wrong_way:yaw|pitch|frozen_box`.
    - yaw: the observed bearing change over 1.5θ against the change predicted from the yaw actually sent (accounting for yaw lag); both above 0.15 rad with opposite signs.
    - pitch (review fix 6): mean applied pitch over the window at least 0.08 (10 counts) forward while the box width shrank by 10 % or more, or backward while it grew.
    - frozen box (review fix 13): the very same box (within 0.5 px) for 2 s while the yaw sent integrates to 0.15 rad or more: a stuck tracker or a repeating camera frame.
    - They can false-alarm when the subject outruns the drone. The result is a hover, which is safe.
14. **Reference widths were re-derived.**
    - With HFOV 75° and a hover height of about 1.2 m, a standing person only fits the frame (box height under 0.8 H, feet inside the 2% margin) from about 2.9 m.
    - New values: close 0.13 W (about 2.25 m), medium 0.10 (about 2.9 m), far 0.075 (about 3.9 m).
    - The old 0.14 W "medium" sat inside the standoff zone.
15. **Capture time.**
    - Twin: `mhd.twin.counter.decode` when importable and decodable.
    - Real mode: pts + baseline − `MHD_CAPTURE_OFFSET_MS`, where the baseline is the session minimum of (receive − pts), allowed to rise at most `MHD_CAPTURE_DRIFT_MS_PER_S` (2 ms/s, camera clock drift) and never more than `MHD_CAPTURE_MAX_RISE_MS` (300 ms) above the minimum. The rise is in telemetry as `latency_ms.capture_rise`; hitting the cap logs a warning. An RTSP reconnect restarts the baseline (the stall that caused it has already landed the drone).
    - Twin without the counter: pts + min over 10 s of (receive − pts) − offset (window: `MHD_CAPTURE_WINDOW_S`).
    - Every `MHD_CAPTURE_*` value is range-checked; a bad one stops the backend from starting.

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
  - `heartbeat_missing`, `video_stale`, `obs_stale`, `flight_timer_warn`, `wrong_way` (with `wrong_way:yaw`, `wrong_way:pitch` or `wrong_way:frozen_box`), `autopilot_paused`, `autopilot_fault`, `autopilot_slow`;
  - `landing:<cause>`, where the cause is `ui_land`, `http_land`, `agent_land`, `model_land`, `heartbeat_lost`, `video_frozen`, `flight_timer` or `estop_reset`;
  - `headless_bypass`, `governor_fault`, `rc_tick_stale`;
  - link and input health: `rc_send_errors` (an RC datagram failed to send), `keepalive_stale` (no D1 keepalive sent for 3 s), `legacy_axes_refused` (real mode refused the legacy UI's axes).
- `latency_ms` keys:
  - `frame_lag`, `obs_lag`, `frame_floor`, `obs_floor`, `obs_period`, `hb_age`, `autopilot_step`, `rc_tick_age`, `frames_received`, `obs_accepted`;
  - `capture_rise` (real mode: pts baseline above its session minimum, ms), `rc_send_errors`, `keepalive_age`, `keepalive_errors`, `frames_rejected`, `obs_invalid` (counts are counts, not ms).
- `sticks_by_source` keys:
  - `human`, `autopilot`, `output` (effective normalized output after caps).

## What the integrator must wire up

- **UI (ws/ui):**
  - Heartbeat: `{"type":"hb","client":"ui"}` at 5 Hz from the worker (real mode counts nothing else).
  - Commands: `{"type":"cmd","command":...}` on `/ws`. `POST /command` is the fallback for `land`, `estop`, `estop_reset`, `pause` only.
  - The page must be served from localhost / 127.0.0.1 (or a host in `MHD_EXTRA_ORIGIN_HOSTS`): `/ws`, `/ws/subject`, `/ws/agent`, `POST /command` and `POST /mission` refuse other `Origin`s, and CORS allows only those pages.
  - After **E-STOP reset** the state is LANDING, not ON_GROUND: the operator presses **Landed**, then (real mode) ARM. ws/ui's opt-in `test_escape_estops_the_real_governor_and_reset_works` waits for ON_GROUND after reset; it must wait for LANDING and click Landed.
  - **Landed** is refused outside LANDING (`not_landing`).
  - Real mode: `{"type":"cmd","command":"arm","arm_phrase":"ARM"}` (exact, after strip).
  - Needs a **Landed** button (`confirm_touchdown`) and a **Resume** button (cancels a landing, or unpauses).
  - Tilt: `tilt_up`/`tilt_down`/`tilt_stop`. Tilt stops by itself after 3 s; the legacy `camera_tilt_direction` also works.
  - `pack_swap` with `pack_id`, on the ground only.
  - Read `/ws/telemetry`.
  - **`turbodrone/frontend/src/lib/ws.ts` is missing from git**: turbodrone's own `.gitignore` rule `lib/` hid it. It exists in the main checkout's working tree. ws/ui must `git add -f` it or fix the rule. I did not touch the frontend.
- **Agent (ws/agent):**
  - Connect to `/ws/agent?role=agent`.
  - Send `Mission` there or to `POST /mission`; acks come back as `mission_ack` / `cmd_ack`.
  - Safety words: `{"type":"cmd","command":"pause"}` for stop/hover, `{"type":"cmd","command":"land"}` for land. Anything else is rejected with `not_allowed_from_agent`. Over HTTP, post to **`/command?role=agent`** so the same whitelist applies (without the role, HTTP allows land, estop, estop_reset, pause).
  - Set `Mission.latency_ms` to the time since the utterance ended (or the router's latency, as now; the governor adds 3 s for ASR). A follow Mission that may predate the last stop / land / E-stop / takeoff is acked with `"paused": true` and does not move the drone until the operator's Resume; a status event says so. Say the command again after a "stop" to follow without Resume.
  - Autopilot announcements ("Got them. Following.", "I lost the subject. Looking around.", "I can't find the subject. Holding position.") and safety events arrive as `AgentEvent(kind="status")`, along with `Candidates`.
- **Perception (ws/percep):**
  - Connect to `/ws/subject`. You get the current `FindRequest` on connect and on each new mission, plus `Release` and `Select`.
  - Pull frames from `/ws/frames`.
  - Send `SubjectObs` **at camera rate, even when nothing is found** (`state:"searching"`, no box). Otherwise the observations go stale and SEARCH cannot move.
  - Every number must be finite (no NaN / Infinity: the hub refuses such JSON) and boxes must lie in 1280x720 (5 % margin); refused observations count in `latency_ms.obs_invalid`. ws/percep's `clip_box` already does this.
  - `capture_ms` must be the `FrameMeta.capture_ms` of the source frame. Also set `mission_id`, give `confirm_age_ms` as of `capture_ms`, and send `detector_latency_ms` (it drives the DOUBT and LOST timers).
  - On a tracker-score drop, re-run the detector on your own: there is no "redetect" message in the contracts. The autopilot's DOUBT follows from your low score.
- **Twin (ws/twin):**
  - Provide `mhd.twin.counter.decode(img_bgr) -> Optional[(render_id, render_ms)]` on the shared monotonic clock. The backend picks it up automatically in twin mode.
  - Use `scripts/run_backend.sh --mode twin --rtsp-url ... --control-port ...`.
- **After Session A/B, update:**
  - `autopilot.yaml`: `camera.hfov_deg`, `plant.yaw_rate_per_count`, `plant.forward_speed_per_count`, `plant.yaw_lag_s`, `delay.theta_s`.
  - `governor.yaml`: `wire.sign`, `wire.deadband_counts`, `flight_timer.land_s` (60 s before low battery).
  - Environment: `MHD_CAPTURE_OFFSET_MS` (minimum camera-to-backend latency). In Session A, watch `latency_ms.capture_rise` for a minute with a steady stream: if it climbs toward 300 the camera's pts clock drifts more than 2 ms/s (e.g. frame-count timestamps); raise `MHD_CAPTURE_DRIFT_MS_PER_S` to just above the measured drift.
- **Merge notes:**
  - `.gitignore` changed (`models/` became `/models/`), and `turbodrone/backend/models/*.py` were added unchanged. Identical additions from other branches merge cleanly.
  - Contracts are unchanged; no adapter was needed.

## Known gaps and risks

- **Not tested against the twin's full wire-level emulator.** Its renderer needs Chrome, which gov may not run. I did run the twin's own RC decoder (`ws/twin` `mhd/twin/protocol.py`, loaded read-only) on my governor's packets: takeoff, sticks at ±38/±64 counts, tilt and land all decode identically.
  - Otherwise tested against a local UDP decoder, an in-process kinematic sim, and real mediamtx/ffmpeg RTSP.
  - The twin's `count_to_unit` divides negative offsets by 128; the governor sends symmetric counts (±127 full scale). Only counts reach the drone, so this matters only for the twin's dynamics (0.8% on negative sticks).
- **Feed-forward is off by default.** The in-process sim shows it is robust to ±30% K error and brings S1 at θ = 1.5 s to 0.104 rad, but PLAN requires a wire-level twin run first. Enable it with `yaw.feedforward.enabled: true`.
- **P-only S1 RMS at θ = 1.5 s is 0.196 rad**, against a 0.15 target. The 0.3 rad/s "stays in frame" target fails with or without feed-forward.
- **No RC-thread watchdog that sends packets itself.** If the RC thread stopped, telemetry shows `rc_tick_stale` and the log says so, but nothing else sends neutral or land. Session A step 6 ("stop RC packets while the keepalive continues") tells us what the drone does. The opposite case (RC continues, keepalive fails) is now visible (`keepalive_stale`, `rc_send_errors`) and the keepalive retries, but what the X69 does without keepalives is still untested.
- **The legacy upstream UI is still accepted in twin mode** (any message is a heartbeat; its stuck-key behavior is unchanged there). Real mode refuses it.
- **CPU-bound Python threads in the backend process** can still stretch RC gaps (review: p99 112 ms with 2 busy threads, 19.7 Hz with 4); `rc_tick_age` shows it, nothing corrects it.
- **The pitch and frozen-box detectors are new tonight** and tested only in unit tests, the closed-loop sim and the reviewer's probes, not in the wire-level twin.

- **Throttle-ramp landing is built but off** (`landing.throttle_ramp: false`). If Session A shows 0x80 does not land the drone, set it to true. While LANDING, throttle then ramps to −0.35 (44 counts) over 1 s, and the land flag is still re-asserted.
  - It is tested only in unit tests: check it at low height in Session B before relying on it.
- **Real-mode capture estimate:** a latency creep or a lasting rise now shows up (review fix 1), but a creep slower than 2 ms/s is absorbed, up to 300 ms in total. If the X69 stamps pts by frame count and drops frames, its pts clock may drift faster than that and the video would slowly look older than it is (false `video_stale`, then a landing). Session A must check `capture_rise` (see above). The twin without its counter block keeps the 10 s sliding window.

- **GIL contention** in the backend process can delay an RC tick by tens of ms (bench max 34 ms with 2 busy threads). The paced loop keeps the 25 Hz average, and the wire-test p99 was 48 ms.
- **First-flight defaults are guesses** until Sessions A and B: HFOV 75°, K 0.012 rad/s per count, θ 1.0 s, reference widths, deadband 0. The forward gate (box clear of the top and bottom edges) means forward motion only happens beyond about 2.9 m at a 1.2 m hover; closer in, the autopilot is yaw-only.
- No paid API calls were made, so there are no `spend.jsonl` entries.
