# Workstream ops: tools and a runbook for the morning hardware sessions

Branch `ws2/ops`, built overnight 2026-11-04 between 03:05 and 04:30 ET. It covers PLAN.md sections 5 and 6:
the owner's Sessions A, B and C on the real Loiley X69, from about 07:15. The owner's guide is
**`docs/RUNBOOK.md`**. This file covers what exists, how it was tested, and what is still open.

**Rule kept:** nothing was sent to the real drone (172.16.11.1) or to any private address. Every test
ran against the wire-level twin or loopback, on the ops ports:
- backend 8150 (the UI preview on 8158, the counter page on 8157);
- twin HTTP 8666;
- UDP RC 23550-23553;
- RTSP 8650.

`preflight.py` and `session_a.py` also honor `MHD_OPS_LOOPBACK_ONLY=1` (set for every test run tonight),
which refuses any non-loopback drone address.

## What exists

| Path | What it does |
|---|---|
| `scripts/preflight.py` | PASS/FAIL table with a fix for every FAIL; JSON in `logs/preflight-<time>.json`; exit code 1 on any FAIL. Every drone-facing check takes `--drone-ip` (default 172.16.11.1). Groups (`--only` / `--skip`) are listed below the table. |
| `src/mhd/backend/hub.py` | `POST /debug/rc_pause {"seconds": N}`, `POST /debug/rc_resume`, `GET /debug/rc_pause`, policy below the table. |
| `src/mhd/governor/turbodrone_glue.py` | the hook: `GovernedX69LgRcAdapter.pause_rc / resume_rc / rc_pause_status`; `send_control_packet` drops control packets while a pause is active. The D1 keepalive thread is untouched. Packet bytes are unchanged. |
| `scripts/session_a.py` | the guided props-off checklist. Results go to `logs/session_a-<time>.json` (saved after every answer; `--resume`, `--only`, `--skip`, `--list`), with GO / NO-GO gates and the values to carry forward (`MHD_CAPTURE_OFFSET_MS`, HFOV, latency, land fallback, link-loss behavior). Steps listed below the table. |
| `scripts/make_aruco.py`, `docs/img/aruco_sysid.png` | the printable marker that `mhd.governor.sysid` expects: DICT_4X4_50, id 7, black square 149.9 mm, US Letter at 300 DPI, with a 100 mm scale bar and a caption giving `--marker-id 7 --marker-size 0.150`. Id 7 avoids ids 0-3, which the screen-counter page uses. Checked with sysid's own detector, also in a skewed, blurred 1280x720 view. |
| `scripts/apply_sysid.py` | details below the table. |
| `scripts/run_stack.sh` | one launcher, real or twin. Details below the table. |
| `docs/RUNBOOK.md` | the owner's morning guide. Contents below the table. |
| `tests/ops/` | 9 rc_pause tests and 5 apply_sysid tests (fast, always run), plus 3 opt-in end-to-end tests (`OPS_STACK=1`): the full twin stack, the stack under a real pseudo-terminal, and session_a's interactive prompts. |

**preflight.py groups:**
- `env`: `scripts/doctor.py` in both venvs, and the terminal app's name.
- `wifi`: the address on the Wi-Fi interface must be in 172.16.11.0/24. macOS hides the SSID without Location Services, so the address is what counts.
- `routes`: `route -n get 172.16.11.1` must go via the Wi-Fi interface (lo0 for loopback); `api.meta.ai` must not, and is reported as iPhone USB when it is; then the service order.
- `internet`: `GET https://api.meta.ai/v1/models` must return 401 without a key, or 200 with one.
- `localnet`: an EMPTY UDP datagram to `drone-ip:9` and a TCP connect to 554 (and to the RTSP URL's port). errno 65 means the Local Network permission is missing, and the fix names the terminal app.
- `video`: ffprobe codec, size and profile, then a packet window that gives fps, the keyframe interval in frames and seconds, and the bitrate. Then a second-client test: two OpenCV readers with the backend's settings; it reports whether both read, only one, or the new one kicks the first.
- `system`: AWDL (AirDrop) status, and stray twin or Glimmer processes.

**rc_pause policy:**
- Real mode: only while the governor is ON_GROUND, at most 15 s. Twin mode: any state, at most 60 s.
- The pause ends at once if the flight state changes (LAND, E-STOP, TAKEOFF, a safety landing) or the backend shuts down, and that packet goes out.
- Logging: a banner in the log, a `rc_paused` telemetry flag, and `latency_ms.rc_pause_left` and `rc_pause_dropped`.
- UI status events at the start and at the end, with packets dropped and keepalives sent.
- Refusals give HTTP 200 with `ok: false` and a reason. A malformed body gives 422. A foreign Origin gives 403.

**session_a.py steps:**
1. safety;
2. props;
3. preflight (network groups), plus an optional G0 over the tether;
4. ffprobe;
5. recordings (raw RTSP, the 2-minute subjects clip, mic and webcam);
6. backend and UI;
7. latency: serves the screen-counter page, decodes it on the backend's own `/ws/frames` (backend receive time minus render time, on one clock), with the RTSP measure and the stopwatch as fallbacks;
8. HFOV from the wall, with a snapshot;
9. flags, sent over `/ws`: takeoff and land, arm step; E-stop once, latched, and reset to LANDING; land on the ground; tilt per press;
10. link loss 1, `kill -9` the backend;
11. link loss 2, Wi-Fi off;
12. link loss 3, `/debug/rc_pause`;
13. the physical remote;
14. weight and the LED.

Motor runs are bounded: the script sends LAND after `--motor-s`, 6 s by default. `--auto-twin` answers every question from the twin's `/status`, runs and kills its own twin-mode backend, and stands in for the UI heartbeat (twin only).

**apply_sysid.py:**
- It writes `config/real/{autopilot,governor}.yaml`: copies of `config/*.yaml`, comments kept, where only these change, each tagged `[measured <time>: source]`:
  - HFOV;
  - K (`plant.yaw_rate_per_count`);
  - `forward_speed_per_count`;
  - `yaw_lag_s`;
  - `theta_s`;
  - `wire.sign` for yaw and pitch;
  - `wire.deadband_counts` for yaw and pitch.
- It re-fits the sysid samples with the measured HFOV and marker size, using sysid's own `fit_steps`.
- It takes each axis sign from the recorded wire counts.
- It subtracts the latency floor that `MHD_CAPTURE_OFFSET_MS` did not cover.
- It refuses values outside sanity ranges, and validates with the governor's strict loaders.
- It prints a unified diff, and writes only with `--write`.
- Explicit stage switches for Sessions B and C: `--yaw-only`, `--yaw-gain-scale`, `--feedforward`, `--throttle-ramp`.
- It never writes `config/` or `config/twin/`.

**run_stack.sh:**
- It starts:
  - the backend;
  - perception (`sam_mlx`, `sam_api`, `oracle` in the twin only, or none), which gets the autopilot config's HFOV;
  - the agent (`auto`, `glimmer`, `keyword`, `local`, or none);
  - the UI: production build with `VITE_AXES_RELEASE_MESSAGE=1`, then `vite preview`;
  - optionally `serve_glimmer.sh`;
  - in twin mode, the twin and its renderer.
- Logs go to `logs/stack-<time>-<mode>/`.
- Ctrl-C: the backend gets SIGINT at once (it lands), then the clients, the UI, and the twin last.
- A dead component is announced. Enter restarts a dead backend.
- Real mode refuses `--headless`, twin configs (by path and by YAML header), the oracle, and twin options.

**RUNBOOK.md contents:**
- setup: tether and service order, Local Network and Microphone, caffeinate, AWDL, stopping overnight processes;
- Session A, with pass criteria per step and the 0x80 throttle-ramp fallback;
- Session B: hover, sign and headless checks, the marker, sysid per axis and amplitude, apply_sysid, feed-forward;
- Session C: yaw-only at 60 % gain, then forward and back, then the full voice loop with `--glimmer local --agent glimmer`;
- UI banners and every safety flag;
- troubleshooting, shutdown, batteries.

## How to run

```bash
cd <checkout> && source env.sh
$MHD_PY scripts/preflight.py                                  # real drone (default 172.16.11.1)
$MHD_PY scripts/session_a.py                                  # Session A, interactive
scripts/run_stack.sh --mode real --no-perception --no-agent   # Sessions A/B: backend + UI
scripts/run_stack.sh --mode real --config-dir config/real [--glimmer local --agent glimmer]   # Session C
$MHD_PY scripts/apply_sysid.py --yaw logs/sysid-yaw-*.json --pitch logs/sysid-pitch-*.json \
    --hfov-deg <A> --latency-ms <A p50> [--write]
$MHD_PY scripts/make_aruco.py                                 # regenerates docs/img/aruco_sysid.png

# twin-only checks (what was run tonight)
MHD_OPS_LOOPBACK_ONLY=1 $MHD_PY scripts/preflight.py --drone-ip 127.0.0.1 --rtsp-url rtsp://127.0.0.1:8650/live/ch00_1
MHD_OPS_LOOPBACK_ONLY=1 $MHD_PY scripts/session_a.py --auto-twin --backend-port 8150 --twin http://127.0.0.1:8666 \
    --rc-port 23550 --rtsp-url rtsp://127.0.0.1:8650/live/ch00_1 --counter-port 8157   # needs a twin on 8666/23550/8650
scripts/run_stack.sh --mode twin --twin-ports 8666,23550,8650 --port 8150 --ui-port 8158 --perception oracle \
    --agent local --agent-extra "--no-mic --mute --tts none --no-warm" --chrome-profile $MHD_HOME/.cache/chrome-ops/renderer
$MHD_PY -m pytest tests/ops -q -p no:logging                  # 14 fast tests (about 15 s)
OPS_STACK=1 $MHD_PY -m pytest tests/ops -q -s -p no:logging   # + 3 end-to-end twin tests (about 3.5 min, Chrome)
```

## Results (this Mac, 03:20-04:10 ET; the demo workstream's twin, backend, perception and Glimmer were running alongside)

| Test | Result |
|---|---|
| `tests/ops` | 14 fast tests and 3 opt-in end-to-end tests: **17 passed** |
| `tests/gov`, all 185, after the hub and glue changes | **185 passed** (137 s). Run from a temporary copy with only the port numbers remapped into the ops range (8100-8109 to 8150-8159, 23510-23518 to 23550-23558); the copy was deleted afterwards. |
| rc_pause on the wire, real backend (TestClient, real PacedFlightController, UDP listener on 23551) | 2.5 s pause: 0 control packets after a 50 ms grace, at least 2 keepalives, about 62 packets dropped (25 Hz), packets back afterwards, `rc_paused` in telemetry. E-stop during a 10 s pause: the first packet after it carries 0x02, within 0.25 s, end reason `state_changed`. Bad input refused: 0 s, −1 s, 61 s, "ten", NaN (422), a foreign Origin (403), already paused. |
| rc_pause, real-mode policy (stub governor, adapter aimed at loopback) | refused in TAKING_OFF, AIRBORNE, LANDING and ESTOPPED; refused at 15.5 s, accepted at 15 s; TAKEOFF during the pause ends it and that packet goes out; a broken predicate fails safe (`predicate_error`, packet sent) |
| **rc_pause against the wire-level twin** (`session_a --auto-twin` link-loss step 3, AIRBORNE, 6 s) | The twin's control-packet count froze at 1341 for the whole pause while keepalives went from 55 to 61. The twin's link-loss timer (1 s, keepalive ignored) fired at **1.04 s**; it landed and its motors stopped by about 4.1 s. Packets resumed (1341 to 1386 within 1 s). Backend: 150 packets dropped, 6 keepalives sent during the pause, end reason `timeout`. |
| `preflight.py --drone-ip 127.0.0.1` against the twin | doctor PASS (20 checks; ml venv PASS when not skipped); route via lo0; HTTPS api.meta.ai 200 in 0.5-0.7 s; UDP :9 ok; TCP 554 refused (expected on loopback) and TCP 8650 ok; RTSP HEVC 1280x720 Main over TCP. Placeholder video (no renderer): 9.5-9.9 fps, keyframe every 25 frames, 3.0-3.1 Mb/s. Second client: both read (24 of client A's frames arrived while B ran). AWDL is active on this Mac (WARN). About 17 s for localnet and video. |
| `session_a.py --auto-twin` (17 steps, 94 s) | **all 17 done, verdict GO.** Latency on the backend's `/ws/frames`: 97 of 97 frames decoded, p50 497.9 ms, p05 493.1 ms, p95 503.9 ms against the twin's 500 ms set point. HFOV round trip 75.0 degrees. Takeoff spins and land stops. E-stop: cut, latched; takeoff refused while E-stopped; reset goes to LANDING. Land on the ground: a no-op. Kill -9: the twin's link lost at 1.01 s, motors off at 3.4 s. Wi-Fi off (twin `rc_loss=1.0`): 3.49 s. A 5 s raw RTSP recording saved. |
| `session_a.py` interactive prompts (scripted answerer, twin and headless twin-mode backend; kill -9 skipped) | 16 of 16 steps done, 59 prompts, no exception; HFOV 77.3 degrees from 1.6 m at 1.0 m; manual-latency fallback carried forward; verdict INCOMPLETE (correct: link-loss 1 skipped) |
| `run_stack.sh --mode twin`, full stack on the ops ports | renderer 25.0 fps (Chrome, profile `chrome-ops/renderer`); UI preview up after `npm ci` (3 s) and a build (3 s); the UI's worker heartbeat 80-113 ms; takeoff, follow, FOLLOW (oracle); screenshot in `$MHD_HOME/media/shared/ops/run_stack_twin_ui_follow.png`. **Ctrl-C: the twin received LAND 0.31 s later; the launcher exited in 1.7-1.8 s with code 0; no ports or processes left.** |
| `run_stack.sh` under a real pseudo-terminal | `kill -9` of the backend gives the EXITED banner; Enter restarts it with the same settings (a new pid, ON_GROUND); Ctrl-C (byte 0x03) exits with code 0 in 3.2 s, ports free |
| `run_stack.sh --mode real --dry-run` | prints the real-mode command lines and starts nothing: the backend with `--mode real`, the chosen configs and no `--logs`; perception `sam_mlx` with `--hfov-deg 75.0` from the config; the agent `--asr meta --intent auto` (or `--intent glimmer` plus the local Glimmer URL); the UI build for localhost:8000 |
| `run_stack.sh` real-mode refusals | `--headless`, `--config-dir config/twin`, a copied twin config (by its YAML header), `--perception oracle`, `--scenario`, `--twin-ports`, a missing config, and `--agent glimmer` without `--glimmer local`: all refused before anything starts |
| `apply_sysid.py` on synthetic sysid results (K 0.010 rad/s per count, deadband 5, lag 0.25 s; pitch 0.012 m/s per count, deadband 6, sign reversed) | K 0.0098-0.0103 and deadband 4.7-4.8; **the pitch sign flip is detected and flagged**; HFOV re-fit 75 to 78.4 degrees scales K by about 1.064 as expected; theta = latency + perception + half frame + half RC + lag; stage switches are incremental (yaw-only 0 counts, gain 0.24, back to 32 and 0.4); out-of-range K and HFOV refused with nothing written; `config/` and `config/twin` refused as targets or bases; an aborted run needs `--allow-aborted` |

Shared artifacts are in `$MHD_HOME/media/shared/ops/`:
- `run_stack_twin_ui_follow.png`;
- `run_stack_twin_check.json`;
- `session_a_auto_twin.json` and `.log`.

## Decisions (nobody could be asked)

1. **rc_pause in real mode is ON_GROUND only, as specified, so Session A's third link-loss test runs with the motors idle.**
   - The governor leaves ON_GROUND at TAKEOFF. With the props off and the motors spinning, it is in TAKING_OFF or AIRBORNE, so the specified rule cannot observe what the motors do.
   - I kept the safer rule. The constant `REAL_MODE_RC_PAUSE_STATES` in hub.py is the one-line change if the integrator wants it widened (see the notes below).
   - The runbook and session_a explain what the idle test still shows: does the video keep streaming, does an LED change, does the drone still take off after 10 s of keepalive-only traffic?
2. **Every pause ends on any flight-state change**, so LAND, E-STOP, TAKEOFF or a safety landing always reach the drone, even in the twin. A failing predicate ends the pause (fail safe).
3. **The ArUco id is 7**, not 0. Ids 0-3 are on the screen-counter page. sysid takes any id by default, so `--marker-id 7` is optional but recommended.
4. **`apply_sysid` also writes `plant.yaw_lag_s`.** It was not in the task's list, but `docs/ws/gov.md` names it among the values to update after Session B, and the wrong-way detector uses it.
5. **`apply_sysid` is incremental.** Its default base is `config/real` when present, so Session A's HFOV and a throttle-ramp switch survive the Session B apply. `--base config` starts over.
6. **The axis sign comes from the recorded wire counts:** sign(rate) × sign(count), majority over the steps above noise. It does not depend on the config used during the run.
7. **run_stack's real mode keeps the backend's control logs and `flight_timer.json` in `logs/`.** Without this, a backend restart would start a half-used pack at 0:00. Twin runs use the run directory, so twin flights never add pack time.
8. **`--agent glimmer`.**
   - With `--intent auto`, the router's 2.5 s Glimmer window loses to Spark (about 1.6 s) against local Glimmer (about 3.6 s). The "tuned Glimmer answers live" clip would then show the Spark badge.
   - `--agent glimmer` passes `--intent glimmer`: Glimmer only, with the keyword grammar answering at 9 s.
9. **No bash job control in run_stack.** Driving it under a real pseudo-terminal, as the owner will, showed two bash 3.2 problems:
   - `set -m` with a controlling terminal spun at 100 % CPU inside the trap;
   - `read -t` arms SIGALRM and longjmps back into `read`, which abandoned a cleanup running inside the SIGINT trap and left the twin running.

   Fixes:
   - components get their own process groups through `perl setpgrp`, with SIGINT reset to default;
   - the handler only records the request and sends SIGINT to the backend group;
   - the ordered stop runs from the main loop, ignoring further signals;
   - liveness checks are zombie-aware.

   The non-tty test had passed with the old code, which is why the pty test now exists.
10. **The UI is built with `VITE_AXES_RELEASE_MESSAGE=1`**, as `docs/ws/ui.md` recommends for flights: sticks release at once on key up or blur.
11. **The preflight video checks open RTSP clients**, so the runbook and session_a run them before the backend starts. session_a also records the raw and subjects clips before the backend starts.
12. **No paid API calls.** The only API traffic is preflight's `GET /v1/models` (not billed), about 6 calls. Spend: $0, so `spend.jsonl` has no lines.

## Known gaps

- **Never run against the real X69, by rule.** These are exercised only by code review and the loopback runs:
  - the real-mode branches: the 172.16.11.x address check, routes via en0 and the iPhone, errno 65, TCP 554, RTSP from the drone;
  - the interactive Session A flow on hardware.
- **The screen-counter decode (`decode_screen`) was not run on a camera view of the page tonight.**
  - In the twin, session_a measured with the twin's own corner counter, on the same `/ws/frames` path.
  - The page decoder was tested by the twin workstream on warped screenshots. The stopwatch fallback is in the flow.
- **Link-loss test 3 in real mode observes idle motors only** (decision 1).
- **`run_stack.sh --mode real` has never started**, because it would reach the drone.
  - Its real-mode differences from the tested twin path: no twin, the backend's `--mode real` without `--logs`, perception and the agent on the same backend.
  - Only its refusals and its `--dry-run` command lines were tested.
- **`--glimmer local` and `--agent glimmer`**: only the flag plumbing was checked. Only the demo workstream may load Glimmer tonight.
- **`apply_sysid` was tested on synthetic sysid results only.** It reads the format that `mhd.governor.sysid.run` writes, and reuses sysid's own fit.
- **The macOS Wi-Fi name is hidden without Location Services.** preflight reports it as hidden and relies on the address.
- **Session A's kill -9 test in the interactive flow needs the operator to restart the backend.** The auto-twin mode tests the kill and restart itself.

## For the integrator

1. **Merge**: `ws2/ops` adds new files and edits only `src/mhd/backend/hub.py` and `src/mhd/governor/turbodrone_glue.py`. No frozen file was touched (contracts, intent_spec, conventions, interfaces, env.sh, doctor, governor core or config, autopilot core, `config/*.yaml`, `config/twin/*.yaml`). All 185 gov tests pass with the change.
2. **Optional: widen the real-mode rc_pause states** for the props-off Session A, so test 3 can observe spinning motors:
   - in hub.py, `REAL_MODE_RC_PAUSE_STATES = frozenset({FlightState.on_ground, FlightState.taking_off, FlightState.airborne})`;
   - then update the session_a step `linkloss_rcpause` (its real branch calls `ensure_ground_and_armed`; it would take off first, like the auto branch) and RUNBOOK 2.9.
   - The risk: someone uses it in Sessions B or C, props on. The pause still ends on LAND or E-STOP at once, and is capped at 15 s.
3. **The perception service's `--hfov-deg` defaults to 80** (`MHD_HFOV_DEG`), while `config/autopilot.yaml` says 75. run_stack passes the autopilot config's value; `e2e_twin.py` passes 75. Consider aligning the default.
4. **`tests/gov` hardcodes ports 8100-8109 and UDP 23510-23518.** Parallel runs must not overlap them.
5. **`scripts/run_twin.sh` ignores SIGINT when a non-interactive shell starts it in the background** (bash rule for `&` without job control). Stop it with SIGTERM, which is what run_stack and the tests do.
6. **`config/real/` does not exist yet.** The owner creates it with `apply_sysid.py` in the morning, and should commit it after Session B: it is part of the story.
7. **Point docs/STATUS.md's "When you wake up" to `docs/RUNBOOK.md`.**
