# Workstream ui: the operator UI

Branch `ws/ui`, written overnight 2026-11-03/04. This is the forked turbodrone React frontend (`turbodrone/frontend`, React 19, Vite 6, Tailwind 4), patched for safety, plus a dashboard that tells the whole story on one screen. It comes with a mock backend and 77 tests (plus 6 opt-in tests that run against the real patched backend).

Screenshots are in `media/ui_screens/` (copied to `$MHD_HOME/media/shared/ui/`):
- `ui_twin_follow.png`
- `ui_twin_candidates.png`
- `ui_twin_listening_doubt.png`
- `ui_twin_follow_1512x982.png`
- `ui_real_locked.png`
- `ui_real_armed_follow.png`
- `ui_real_estopped.png`
- `ui_real_disconnected.png`
- `ui_gov_backend_twin_no_video.png`: the UI on the real governor backend.
- `ui_demo_mock.mp4`: a 36 s, 1080p, 25 fps recording of the dashboard on the mock, showing the whole story:
  1. takeoff;
  2. a held-Space voice mission, its transcript, and the mission card with model badge and latency;
  3. FIND, then CONFIRM, then a look-alike pick, then FOLLOW, with the mask and sticks steering;
  4. DOUBT;
  5. Esc E-STOP;
  6. a link drop with the DISCONNECTED banner and LAND over HTTP;
  7. auto-reconnect.

  Regenerate it with `$MHD_PY turbodrone/frontend/mock/record_demo.py`. It uses Chrome's CDP screencast and the system ffmpeg, because Playwright's own video needs an ffmpeg download I did not make.

## Run it

All commands assume `cd $MHD_TREE && source env.sh`. Dependencies install into the worktree, with the npm cache inside the project:

```bash
cd turbodrone/frontend && npm ci
```

**Real flights** run a production build. `VITE_BACKEND_URL` is baked in at build time, and the footer shows it.

```bash
cd turbodrone/frontend
VITE_BACKEND_URL=http://localhost:8000 npm run build
npx vite preview            # http://127.0.0.1:5173 (loopback only, strict port)
```

Open the page in its own Chrome window and keep that window visible (see "Known gaps").

**Twin, development:** run `npm run dev`. It serves http://127.0.0.1:5173 against `http://localhost:8000`. The twin chase view comes from `VITE_TWIN_CHASE_URL`, default `http://localhost:8766/chase`.

**Demo or UI work without a backend:** run these in two terminals:

```bash
cd turbodrone/frontend && npm run mock       # mock backend on :8600, scenario "cycle" (flies by itself)
cd turbodrone/frontend && npm run dev:mock   # vite on :5273 pointed at the mock
# open http://127.0.0.1:5273 ; hold Space to "speak" ; Esc, L, W/A/S/D all work
```

The mock has these options:
- `--scenario manual` changes state only on commands.
- `--mode real` shows the REAL banner and the ARM gate.
- `--pack-start-s 280` starts near the pack warning.

Its test controls (`POST /mock/control`) can do the following:
- switch REAL/TWIN;
- drop the control socket (`ws_mode: "down"`) or keep it open but ignore it (`"deaf"`);
- pin flight and autopilot states;
- inject agent events;
- push candidates (`POST /mock/candidates`).

To run the tests (about 90 s), use the commands below. They start the mock on 8601 and 8602 and vite on 5273, run Chrome through Playwright (`channel="chrome"`, profile in `$MHD_HOME/.cache/chrome-ui`), and stop everything when they finish.

```bash
$MHD_PY -m pytest tests/ui -v
UI_GOV_TREE=$MHD_TREE $MHD_PY -m pytest tests/ui/test_ui_gov_backend.py -v   # once the backend is merged here
```

`UI_HEADED=1` shows the browser while the tests run.

## Operator cheat sheet

| Key or button | Does | Notes |
|---|---|---|
| **Esc** / red **E-STOP** (far right) | `estop` | Works always, even while typing in the ARM box. Acts on key down or pointer down. Falls back to HTTP. |
| **L** / orange **LAND** (far left) | `land` | Falls back to HTTP. Ignored while typing in the ARM box. |
| **Space** (hold) / **HOLD TO TALK** | `ptt_down` / `ptt_up` | Missions only, never takeoff. Released automatically if the window loses focus. |
| **TAKEOFF** | `takeoff` | Button only, never voice or a hotkey. Disabled with a reason shown under it (see below). |
| type **ARM** + Enter | `arm` with `arm_phrase:"ARM"` | REAL mode only, once per session. Unlocks TAKEOFF. |
| **Reset E-STOP…** | `estop_reset` | Appears only when E-STOPPED. Needs two clicks within 3 s. |
| **Confirm touchdown** | `confirm_touchdown` | Appears while LANDING. |
| W/A/S/D, arrows | axes: pitch/roll, throttle/yaw | Human override. Sent only while a key is held and the window is focused. |
| PageUp/PageDown, camera ▲▼ | `tilt_up`/`tilt_down`, then `tilt_stop` on release | Hold to tilt. |
| pack 1 / pack 2 | `pack_swap` | On the ground only. |

TAKEOFF is disabled when any of these is true:
- the control link is down;
- telemetry is stale;
- the drone is not on the ground;
- REAL mode and not armed;
- the governor raises `video_stale`, `heartbeat_missing`, `governor_fault` or `rc_tick_stale`.

If a takeoff still leaves the drone on the ground, the command line says "NOT ACCEPTED".

Banners:
- **TWIN** (blue) or **REAL DRONE** (red stripes) comes from `telemetry.flight_mode`. It reads MODE UNKNOWN until telemetry arrives.
- **DISCONNECTED** (red, full width) means the control socket is down. It reconnects by itself, and LAND and E-STOP use HTTP meanwhile.
- **NO TELEMETRY** (amber) means the flight state is unknown and takeoff is locked. LAND and E-STOP still work.

## What I built

### 1. Safety patches to the turbodrone frontend

| Requirement | Where | How |
|---|---|---|
| Backend URL configurable (all 7 hard-coded places) | `src/config.ts` | `VITE_BACKEND_URL` (default `http://localhost:8000`), plus `VITE_TWIN_CHASE_URL`, `VITE_PACK_WARN_S`, `VITE_PACK_LAND_S`, `VITE_PACKS`. A test fails if `localhost` appears anywhere else. |
| axes only while nonzero; zeroed on blur and visibilitychange | `hooks/useControls.ts` | 30 Hz TX loop sends `{"type":"axes",...}` only if a rounded axis is nonzero and the page is visible and focused. Sticks are zeroed on `blur`, on `visibilitychange`, and on any Meta (Cmd) chord, because macOS drops keyups while Cmd is held. Keys use `e.code`: upstream used `e.key`, so Shift or Caps Lock could strand a key at full deflection. |
| Auto-reconnect plus DISCONNECTED banner | `lib/reconnectingSocket.ts`, `components/ModeBanner.tsx` | 250 ms backoff doubling to a 2 s cap, a 3 s connect timeout, and an optional idle timeout (telemetry: 3 s). Used by every socket, the worker's too. The MJPEG stream reopens when the link comes back. |
| LAND/E-STOP HTTP fallback | `lib/commands.ts`, `useControls.sendCommand` | Socket down: `POST /command`, retried 8 times about 250 ms apart. Socket open but telemetry does not show the command took effect within 600 ms (a wedged handler): also sent over HTTP. A command confirmed at any moment after sending is never re-sent, so a quick E-STOP reset is not undone. The result line under the buttons says which channel was used. |
| 5 Hz heartbeat from a dedicated Web Worker | `workers/heartbeat.worker.ts`, `hooks/useHeartbeat.ts` | The worker owns its own WebSocket to `/ws` and sends exactly `{"type":"hb","client":"ui"}` every 200 ms. A test blocks the page's main thread for 2.5 s and heartbeats keep arriving. StrictMode's double mount terminates the first worker, so there is one heartbeat socket. |
| Buttons never take focus | `components/UiButton.tsx`, `useNoButtonFocus` | Every button is a `UiButton` (`tabIndex=-1`, `preventDefault` on mousedown; a test forbids raw `<button>`). A capture-phase `focusin` guard blurs any button that still gets focus. |
| Hotkeys | `hooks/useHotkeys.ts` | Capture-phase listeners on `window`. Esc is E-STOP and always works. L is LAND, but not while typing in a text field and not with Cmd, Ctrl or Alt held. Space is push-to-talk with `preventDefault` on keydown and keyup, so it never activates a focused button or scrolls the page. |
| Speed tier, profile, plugins removed | deleted upstream components | `SpeedControl`, `ControlSchemeToggle` (and TrackPoint pointer-lock mode, which swallows Esc), `PluginControls`/`usePlugins`, `ControlsOverlay`, `CommandButtons`, `CameraTiltControl`, `AxisIndicator`, `DrawingOverlay`/`useVideoSizing`. They stay in git history at 1aff2b5. |
| REAL/TWIN banner and ARM gate | `ModeBanner.tsx`, `CommandBar.tsx` (`ArmGate`), `lib/gates.ts` | ARM box shown in REAL mode only. It uppercases input, enables the ARM button only for exactly `ARM`, sends `{"type":"cmd","command":"arm","arm_phrase":"ARM"}`, then blurs so the hotkeys come back. TAKEOFF follows `telemetry.armed`. |
| `{"type":"cmd"}` for every command | `lib/commands.ts` | Includes `tilt_up`/`tilt_down`/`tilt_stop` (upstream put `camera_tilt_direction` inside axes, which `OperatorInput` forbids). |

### 2. New panels (one screen, 1920x1080; also fits 1512x982)

- **Header**: flight-state badge (E-STOPPED pulses); autopilot badge colored by state; ARMED/DISARMED in REAL mode; per-pack timer, amber from 5:00 and red from 6:00 (30 s before the 6:30 auto-land), with a progress bar and pack-swap buttons; link chips for CTRL, HB rate, TLM rate and VOICE.
- **Camera**: the upstream MJPEG feed with an SVG overlay in 1280x720 space, so contract boxes map 1:1 at any size:
  - SAM mask polygon (fill plus outline);
  - box, dashed until confirmed;
  - label (subject phrase, autopilot state, confidence);
  - image-center crosshair, with a dashed line to the subject (the bearing the autopilot corrects);
  - colors by autopilot state: FIND sky, CONFIRM yellow, FOLLOW green, DOUBT orange, LOST red, SEARCH violet;
  - numbered candidate boxes, which can be clicked.
- **Voice**:
  - push-to-talk state;
  - live partial transcript, then final lines (with ASR model and time), replies, errors and safety words ("heard “land” → land");
  - caption from the agent's 5 s health report: agent alive, local stop/land listener on or off, mic silent, agent silent.
- **Mission card**:
  - transcript;
  - intent JSON;
  - model badge (`glimmer-30b+lora` blue, `muse-spark-1.3` purple, `keyword` gray);
  - latency.

  The agent's `intent` event (no `mission_id`) shows at once, then the backend's mission from telemetry replaces it.
- **Which one?**: candidate picker, which sends `Select`. Candidates expire after 10 s and clear when the autopilot reaches FOLLOW or the mission changes.
- **Sticks**: final wire counts per axis around 128, the active source, the sources, and "you: idle (no axes sent)" or the axes being sent.
- **Latency**: four bars with threshold ticks:
  - frame age: neutral at 0.7 s, full scale 2 s;
  - observation age: neutral at 0.4 s;
  - detector: 1.5 s gate;
  - intent: 2.5 s Glimmer deadline.

  Other `latency_ms` keys appear in one small line.
- **Safety**: governor flags (red for anything that neutralizes, lands or faults; otherwise amber), plus the last 4 governor/autopilot transitions relayed by the hub.
- **Twin chase view**: an iframe of the twin's `/chase`, twin mode only. It is mounted only once reachable. It has `pointer-events:none` and `tabIndex=-1`, so a click can never move keyboard focus (and Esc) into it.
- **Footer**: "UI shell from turbodrone (Apache-2.0) · new: voice, mission, SAM overlay, safety, latency and twin panels", plus the backend URL. The camera HUD also marks "MJPEG (turbodrone) · SAM 3.1 mask overlay (new)".

### 3. Mock backend: `turbodrone/frontend/mock/mock_backend.py`

The mock is a core-venv FastAPI app, default port 8600. Every message it sends is built from the frozen pydantic contracts. Every message it receives is validated against them, and its JSONL log marks all-zero axes and extra fields as invalid. It serves:
- `/mjpeg`: a 1280x720 test pattern with a moving subject in a blue shirt and a look-alike in green;
- `/ws/telemetry`: 10 Hz, cycling through flight and autopilot states, with the box and mask matching the picture;
- `/ws/agent?role=ui`: a fake agent that streams partials while Space is held, then the final transcript, intent, and reply, in the real agent's shapes;
- governor-style transition events and agent health heartbeats;
- `/ws` and `POST /command`, which act like a minimal governor (E-STOP latch, ARM phrase, takeoff rules);
- `/state` and `/capabilities`;
- `/chase`: a 2D canvas stand-in for the twin's chase view;
- with `--static-ui DIR`, a production build at `/ui/`.

## Test results

77 passed and 6 skipped (the opt-in gov tests) in about 90 s at 00:27 ET. Repeated full runs were stable (3 consecutive browser-suite runs, 0 flakes). The 6 opt-in tests passed against `ws/gov` (f4af4b3 plus its working-tree edits at 00:24).

| File | Count | Checks |
|---|---|---|
| `test_ui_browser.py` | 26 | See the list below. |
| `test_ui_production.py` | 2 | The minified build (`--base /ui/`), served same-origin by the mock: worker chunk present, heartbeat 5 Hz from one socket, Esc, Space and L work, HTTP fallback works, only the configured backend URL is in the bundle. |
| `test_ui_latency.py` | 1 | Measurements below, with loose regression bounds. |
| `test_screenshots.py` | 2 | Twin and real screenshots. |
| `test_mock_backend.py` | 11 | The mock validates and judges messages correctly, latches E-STOP, needs the exact ARM phrase, and its masks stay inside their boxes. |
| `test_contract_mirror.py` | 25 | `src/lib/contracts.ts` matches the pydantic models: field names, command names, event kinds, enums, literals. |
| `test_frontend_static.py` | 10 | `npm run build`, `eslint --max-warnings 0`, one backend URL, no speed/profile/plugin code, only the worker sends `hb`, every button is a `UiButton`, the followed thing is always called "subject", no web fonts, loopback dev server. |
| `test_ui_gov_backend.py` | 6 (opt-in) | Real governor backend in twin mode with no twin: telemetry renders, no ARM needed in twin, TAKEOFF shows "no video", the governor sees the worker heartbeat (`hb_age` under 600 ms), Esc E-STOPs the governor and reset works, `POST /command` passes CORS, no contract errors in the backend log. |

The 26 browser tests check that:
- the heartbeat runs at 5 Hz from a single socket, and survives a 2.5 s main-thread block;
- no axes are sent while idle;
- axes go out at about 30 Hz while a key is held and stop within 100 ms of release;
- axes stop on blur and on a hidden tab, even with the key still down;
- Space sends `ptt_down` and `ptt_up`, its default action is prevented, it never re-fires a clicked TAKEOFF, and a scripted focus is undone;
- the PTT button works by mouse;
- PTT is released on blur;
- no button is in the Tab order;
- Esc sends E-STOP over the socket, and there is no HTTP re-send once telemetry confirms it;
- Esc goes over HTTP when the socket is down, the banner appears, and auto-reconnect clears it;
- E-STOP is re-sent over HTTP when the socket is "deaf";
- a quick reset is not re-asserted (this test fails without the fix);
- L lands over WS and over HTTP;
- LAND and E-STOP are more than 800 px apart;
- the twin banner shows and takeoff needs no ARM;
- the REAL banner and ARM gate work: "AR" keeps the button disabled, "arm" plus Enter sends exactly `arm_phrase:"ARM"`, and TAKEOFF then unlocks;
- stick keys and L are ignored while typing, and Esc still works there;
- the candidate picker sends `Select`;
- mask colors follow the autopilot state;
- the pack timer goes amber and red, and pack swap works;
- the footer credit is present;
- telemetry loss gives UNKNOWN with takeoff locked;
- agent and governor events are routed correctly;
- the governor's refusal reasons show on TAKEOFF;
- an ignored takeoff is reported as NOT ACCEPTED.

After every browser test, every message the UI sent must validate against the frozen contracts (`extra="forbid"`), and the page must have no JS errors.

## Measurements

All measurements are loopback (127.0.0.1) against the mock backend, using the vite dev server and headless Google Chrome 157 on this Mac. They ran at 00:28 ET with load average 3.4 to 4.0, while the other overnight workstreams ran (twin Chrome and ffmpeg, percep and train Python with about 14 GB resident). Model downloads had finished ("ALL DONE" at 23:40). The raw data is in `logs/ui_tests/measurements.json`.

| What | p50 | p95 | Notes |
|---|---|---|---|
| Esc key to E-STOP at the backend, over WS | 1.0 ms | 4.1 ms | n=10. One earlier run under load average 5.3 had a 72 ms outlier. |
| Esc key to E-STOP at the backend, over HTTP (socket down) | 2.4 ms | 3.7 ms | n=10, includes the CORS preflight path. |
| Backend state change to DOM | 87 ms | 110 ms | n=20, random phase. About 50 ms of this is waiting for the 10 Hz telemetry tick. |
| Heartbeat inter-arrival at the backend | 199.8 ms | 206.4 ms | Max 209 ms, 5.0 Hz. Also 5 Hz during a 2.5 s main-thread block. |
| Control socket reconnect after the backend comes back | 190 to 1320 ms | | Depends on where the 2 s-capped backoff is. |
| axes rate while a key is held | 30.5 Hz | | |
| Production bundle | 271 kB JS (87 kB gzip), 26 kB CSS, 3 kB worker | | |

## Wire behavior (what the backend will see)

- `/ws`, main socket:
  - `{"type":"axes","mode":"inc"|"abs","throttle":..,"yaw":..,"pitch":..,"roll":..}` at 30 Hz, only while at least one axis is nonzero. There is no trailing all-zero message: the governor's 0.25 s human-input staleness ends the input. Values are rounded to 3 decimals. Keyboard sends ±1; the gamepad sends deadzoned 0.15..1.
  - `{"type":"cmd","command":...}` for everything, plus `arm_phrase` or `pack_id` where needed.
- `/ws`, worker socket: `{"type":"hb","client":"ui"}` every 200 ms.
- `POST /command` with `Content-Type: application/json`: LAND and E-STOP only.
- `/ws/agent?role=ui`: `{"type":"agent_event","kind":"ptt_down"}`, `{"type":"agent_event","kind":"ptt_up"}`, `{"type":"select","mission_id":..,"index":..}`.
- No `ts_ms` anywhere: browser clocks are not the shared `time.monotonic`, so the backend stamps arrival times itself.

## Decisions I made (nobody was awake to ask)

1. No trailing all-zero axes message on release. This follows "sent only while nonzero" literally. Release is bounded by the governor's 0.25 s staleness rule.
2. TrackPoint (pointer-lock) mode removed: pointer lock consumes Esc, which is E-STOP. Keyboard and gamepad remain. The gamepad is untested (no gamepad here).
3. LAND and E-STOP act on pointer down, not click, so a finger that slides off still stops the drone. TAKEOFF stays a click.
4. Safety commands get the HTTP re-send when telemetry does not confirm them in 600 ms. Both paths are idempotent in the governor (latched E-STOP, re-asserted land).
5. TAKEOFF also respects four governor flag names. This couples the UI to non-contract strings, but it can only make the button stricter, and the governor still decides.
6. The upstream overlay channel (`/ws/overlays`) is no longer drawn on the dashboard. It is still used by `/test`.
7. No web fonts (Google Fonts removed): during real flights the internet comes over a phone tether.
8. Vite binds 127.0.0.1 only. The production port stays 5173; my tests use 5273.
9. `turbodrone/.gitignore` ignores every `lib/` folder (a Python template rule), which silently hid `src/lib/`. That is probably why the brief's `lib/ws.ts` did not exist. I re-included it in `turbodrone/frontend/.gitignore` with a comment.
10. `BrowserRouter` now takes `basename` from Vite's base: the production test found that the app rendered nothing under a sub-path.

## Known gaps and risks

- **A frozen or discarded tab stops the heartbeat.** The worker survives timer throttling and a busy main thread. But if Chrome freezes the page (Memory Saver, a long time in the background), its workers freeze too. The governor then goes neutral at 1.5 s and lands at 5 s, which is fail-safe but unwanted. During flights:
  - keep the UI in its own visible Chrome window;
  - exclude it from Memory Saver and Energy Saver;
  - do not minimize it.

  Headless Chrome never hides a page, so this is not covered by tests.
- **Commands over the socket get no reply.** The UI infers acceptance from telemetry: E-STOP and LAND are re-sent over HTTP if unconfirmed, and TAKEOFF is reported as "NOT ACCEPTED". A refusal reason (for example `not_armed`) is only visible through the flags. An ack message would be better; see the suggestions below.
- **One UI at a time.** Two open dashboards mean two heartbeats and two sources of human input.
- **The chase iframe is a full Three.js renderer** (the twin's `/chase`), so the UI page costs GPU in twin mode. It is absent in real mode.
- **Not tested:** the real twin video path into `/mjpeg`, a gamepad, and the real agent and perception sending live events. Their message shapes are mirrored from their branches, and the mock emits the same shapes.
- **Only Chrome was tested.**
- **The vite dev server runs React StrictMode.** It is harmless, but use the production build for flights. The PLAN asks for a production build anyway.

## What the integrator must wire up

1. Merge `turbodrone/frontend` (and `tests/ui`). Run `npm ci` in `turbodrone/frontend`, since node_modules is not in git.
2. The backend endpoints the UI uses are:
   - `/ws` (axes, hb and cmd: `ws/gov` has this);
   - `POST /command` with CORS (`ws/gov` keeps upstream's `allow_origins=["*"]`);
   - `/ws/telemetry`;
   - `/ws/agent?role=ui` (relays the agent's AgentEvents, Candidates and the hub's status events, and forwards `ptt_down`/`ptt_up` and `Select`);
   - `/mjpeg`;
   - `/capabilities` (upstream).

   `ws/gov` at f4af4b3 already serves all of them, and `test_ui_gov_backend.py` passed against it.
3. Telemetry keys the UI reads beyond the contract:
   - `sticks_wire` keys `throttle`/`yaw`/`pitch`/`roll`;
   - `latency_ms.detector` and `latency_ms.intent` are optional (the UI falls back to `subject.detector_latency_ms` and `mission.latency_ms`); other keys are shown as a small line;
   - flag names `video_stale`, `heartbeat_missing`, `governor_fault` and `rc_tick_stale` lock TAKEOFF.
4. Agent event shapes the UI reads beyond the contract (as emitted by `ws/agent`):
   - `final.data.model` and `final.data.asr_ms`;
   - `intent.data.{intent, model, latency_ms, transcript}`;
   - `safety_word.data.{command, decode_ms}`;
   - health: `status` with `data.safety_listener` (also `asr`, `intent`, `mic_silent`);
   - hub transitions: `status` with `data.kind` in `state`, `flight_state`, `announce`, `safety`.
5. Twin: the chase URL defaults to `http://localhost:8766/chase`. Override it with `VITE_TWIN_CHASE_URL`.
6. After the merge, run `UI_GOV_TREE=$MHD_TREE $MHD_PY -m pytest tests/ui/test_ui_gov_backend.py -v` once.
7. Contract changes: none needed. Suggestion for v0.2: have the backend answer each `/ws` `cmd` with its `CommandResult` (`ok`, `reason`), for example as an `AgentEvent` `status` on `/ws/agent` or a field in Telemetry. The UI could then say "refused: not_armed" instead of inferring it.

## Spend

$0. The UI workstream made no paid API calls.
