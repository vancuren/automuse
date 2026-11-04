# Workstream twin: wire-level digital twin of the X69

Built overnight 2026-11-03/04 on branch `ws/twin` (PLAN.md section 4.2). The unmodified turbodrone
`x69_lg` backend flies the twin over real X69 UDP packets and receives HEVC over RTSP from it,
exactly as it would from the drone, except that `DRONE_IP=127.0.0.1`.

## What is here

| Path | What it does |
|---|---|
| `src/mhd/twin/protocol.py` | Decodes the 28-byte X69 control datagram (`ca 47 d5 00` + LE32 length + 20-byte payload, markers 0x66/0x14/0x99, XOR of bytes 2..17) and the 8-byte `ca 47 d1` keepalive; flags; an encoder for tools. Tested against packets from the real upstream adapter class. |
| `src/mhd/twin/dynamics.py`, `config/twin/x69.yaml` | 100 Hz kinematic X69: per-axis deadband, first-order lags, yaw rate per count, quadratic drag, altitude hold, camera pitch from atan(a_thrust/g), headless, takeoff/land/E-stop (free fall), flip, optical-flow drift (Ornstein-Uhlenbeck, worse on low texture), tilt servo per press or while held, link loss after T s without control packets (keepalive ignored), battery timer. Also `RcFaultInjector` (loss, Gilbert-Elliott burst loss, order-preserving jitter) and `RcLinkStats` (inter-arrival p50/p99, alarm when p99 > 60 ms). |
| `src/mhd/twin/scenarios.py`, `config/twin/scenarios/s1-s6.yaml` | Worlds and actors that walk or drive along paths (walk speed capped at 0.3 m/s unless `allow_fast`). Ground truth: every actor projected into the drone camera with the renderer's exact pinhole model; people use the renderer's own skeleton and gait, so boxes follow the stride; visibility counts in-frame points not hidden by walls, trees, or closer actors. |
| `src/mhd/twin/counter.py` | Corner counter block: 64 luma cells of 16x16 px (render_id mod 2^20, render_ms mod 2^24, CRC-16). `stamp(img, render_id, render_ms)`, `decode(img, ref_ms=None) -> (render_id, render_ms) or None`, `COUNTER_ROI = (1008, 640, 1280, 720)`, `mask_roi(img)`. Works at any frame size (e.g. 640x360). |
| `src/mhd/twin/server.py` | The twin process (FastAPI/uvicorn + asyncio): UDP RC, dynamics loop, `/ws/render`, `/ws/truth`, `POST /control`, `GET /status`, pages. Starts its own mediamtx. |
| `src/mhd/twin/video.py` | Frame path: JPEG decode, blur (sigma 0.7) and sensor noise (sigma 3), counter stamp, latency buffer (target total minus measured baseline, per frame), stall-then-burst, ffmpeg `hevc_videotoolbox -b:v 3M -bf 0 -g 25 -realtime 1` publishing RTSP over TCP, and an RTSP probe that measures the real total latency. |
| `src/mhd/twin/media.py` | mediamtx config (RTSP only, TCP only, 127.0.0.1, path `live/ch00_1`; API, metrics, pprof, playback, RTMP, HLS, WebRTC, SRT and MoQ off) and the ffmpeg commands. |
| `twin/web/` | Three.js 0.180 renderer (`npm install` inside `twin/web`): textured lawn, house and fences, trees, props, procedural articulated people (recolorable shirt, pants, skin, hair, shoes; walk cycle), procedural cars (recolorable paint; toy RC buggy and full-size), the drone camera at 1280x720 with HFOV from the params (default 75 deg), `/chase` view with a picture-in-picture of the drone camera. |
| `src/mhd/twin/launch_renderer.py` | Playwright, `channel="chrome"`, profile `$MHD_HOME/.cache/chrome-twin`, flags `--use-angle=metal --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows`; refuses to run on SwiftShader or any software WebGL. |
| `src/mhd/twin/metrics.py` | Joins Truth with SubjectObs and Telemetry logs by render id: time-to-find, lock rate, bearing-error RMS (control and perception), identity switches, track-id switches, settling time. CLI writes a JSON report and a per-frame CSV. |
| `src/mhd/twin/oracle.py` | Oracle perception for the wire twin: SubjectObs from ground truth on the backend's `/ws/subject` (FindRequest/Release honored), with latency, box noise, dropouts, find delay, 2 Hz detector age. For closing the loop before real perception, and for metrics tests. |
| `src/mhd/twin/capture.py` | Saves HEVC-path frames joined to their Truth by the decoded render id (`--overlay` draws the truth boxes). |
| `src/mhd/twin/demo_pilot.py` | Truth-driven demo pilot (twin only, not the autopilot): flies the twin over UDP, follows a subject, records chase video and sample frames. |
| `src/mhd/twin/screen_counter.py`, `twin/web/counter.html` | **Real-drone latency (Session A step 3).** A page that shows the same 64-bit counter code as an 8x8 grid framed by four ArUco markers, redrawn every display frame on the shared monotonic clock (synced over loopback, RTT about 0.3 ms); `measure` decodes it from any RTSP stream or recording via a homography and prints the latency histogram. |
| `scripts/run_twin.sh` | mediamtx + twin server + Chrome renderer under `caffeinate -i`; `--ports`, `--scenario`, `--latency-ms`, `--chase`, `--chase-rec`. |
| `scripts/run_upstream_backend_twin.sh` | The upstream turbodrone backend pointed at the twin (port 8200 by default). |
| `scripts/twin_rc.py`, `scripts/twin_sam_check.py`, `scripts/twin_measure_backend.py` | Hand RC sender; SAM 3.1 API spot-check of actor detectability (paid, logged); latency at a backend's `/mjpeg`. |

## Run it

```bash
cd <checkout> && source env.sh
(cd twin/web && npm install)                      # once: three.js into twin/web/node_modules

# Twin on the interfaces.md defaults (HTTP 8766, UDP RC 23458, RTSP 8554), scenario S1,
# total latency 500 ms (the default until Session A measures the real value; --latency-ms 0 = none):
scripts/run_twin.sh --scenario s1
# The overnight twin workstream's ports:
scripts/run_twin.sh --ports ws-twin --scenario s2 --latency-ms 500 --chase
# Explicit ports: --ports HTTP,RC,RTSP

# Upstream (unchanged) turbodrone backend flying the twin:
scripts/run_upstream_backend_twin.sh --port 8200 --rc-port 23458 --rtsp-port 8554
#   = DRONE_TYPE=x69_lg DRONE_IP=127.0.0.1 CONTROL_PORT=<rc> X69_LG_VIDEO_MODE=rtsp
#     X69_LG_RTSP_URL=rtsp://127.0.0.1:<rtsp>/live/ch00_1 OPENCV_FFMPEG_THREADS=1 PLUGINS_ENABLED=false

# Control and status:
curl -s localhost:8766/status | python3 -m json.tool
curl -s -X POST localhost:8766/control -H 'content-type: application/json' \
     -d '{"scenario":"s3"}'                                   # switch scenario (drone resets, landed)
curl -s -X POST localhost:8766/control -d '{"latency_ms":600}' -H 'content-type: application/json'
curl -s -X POST localhost:8766/control -d '{"stall":{"every_s":20,"duration_ms":1500,"mode":"burst"}}' -H 'content-type: application/json'
curl -s -X POST localhost:8766/control -d '{"rc_loss":0.1,"rc_burst_rate":0.01,"rc_burst_len":8,"rc_jitter_ms":30}' -H 'content-type: application/json'

# Metrics from logs (the twin writes logs/twin-<time>-<port>/truth.jsonl and events.jsonl):
$MHD_PY -m mhd.twin.metrics --truth logs/twin-.../truth.jsonl --obs obs.jsonl --subject "person in blue shirt" --out report.json --csv rows.csv

# Oracle perception into a backend (twin only):
$MHD_PY -m mhd.twin.oracle --twin http://127.0.0.1:8766 --backend ws://127.0.0.1:8000 --latency-ms 300 --log obs.jsonl

# Session A, real-drone latency: serve the page, open it full screen, point the drone at it
$MHD_PY -m mhd.twin.screen_counter serve --port 8766        # then open http://127.0.0.1:8766/counter
$MHD_PY -m mhd.twin.screen_counter measure --rtsp rtsp://172.16.11.1/live/ch00_1 --seconds 60 --csv logs/latency_real.csv
#   (or --rtsp a 60 s recording made with ffmpeg -c copy; the twin server also serves /counter)

# Tests (about 3 minutes; Chrome opens a window for the video-path and counter-page tests):
$MHD_PY -m pytest tests/twin -q
```

## Interfaces

- UDP RC `127.0.0.1:<rc-port>`: X69 control datagrams and keepalives. Only control packets count for the link.
- RTSP `rtsp://127.0.0.1:<rtsp-port>/live/ch00_1`: HEVC 1280x720, about 25 fps, 3 Mb/s, no B-frames, 1 s GOP, counter block bottom-right.
- `GET /` drone-camera page, `GET /chase` passive chase view (embed it in the UI), `GET /chase?rec=1` chase frames for recording.
- `WS /ws/render?role=camera|chase|chase_rec`: server sends `{"type":"scene",...}` then `{"type":"pose","render_id","render_ms","t","camera","drone","phase","actors":[{id,x,y,heading,speed,dist}]}`; camera and chase_rec answer with binary `[uint32 LE render_id][JPEG]`. Newest only, one frame in flight, server-paced (25 fps).
- `WS /ws/truth`: one `mhd.contracts.Truth` JSON per rendered frame. `Truth.drone` carries x, y, z, yaw, pitch, roll, vx, vy, vz plus yaw_rate, cam_tilt, motors_on. Truth is also logged to `logs/twin-*/truth.jsonl`.
- `POST /control`: `TwinControl` fields (scenario, reset, latency_ms = TOTAL target, drift multiplier, rc_loss, stall) plus a local subclass `TwinControlExt` in server.py: rc_burst_rate, rc_burst_len, rc_jitter_ms, headless, link_loss_s, link_loss_action (land, hover, drop), battery_s, fps, noise, blur, hfov_deg, cam_tilt_deg, pause_actors, record_chase, drone_pose. Unknown fields are rejected (422).
- `GET /status`: fps, renderer (WebGL string), RC (packets, keepalives, errors, inter-arrival p50/p99/max, alarm, senders, applied after faults), link, drone (phase, pose, sticks, events), video (encode/probe fps, counter decode rate, baseline/upstream/downstream/total latency, stall), latency summary, log dir.
- Extra: `GET /truth/latest`, `GET /scene`, `GET /scenarios`, `GET /frame.jpg` (latest stamped frame; `?source=rtsp` for the decoded one), `GET /frames/log` (per-frame timing records).

Total latency is defined at an OpenCV RTSP reader (CAP_FFMPEG, TCP, nobuffer/low_delay, one decoder thread): decode time minus render time. The twin measures it with its own probe and holds each frame until `render_ms + target - downstream_estimate`.

## Test results (all on this Mac, 2026-11-04 00:15-00:45 ET)

`$MHD_PY -m pytest tests/twin -q`: 65 tests, all pass in about 3 minutes (final run at 01:00 ET, see the end of this file). The repo's own `tests/test_contracts.py` and `tests/test_intent_spec.py` and upstream's x69_lg unit tests also pass.

| Acceptance item | Result |
|---|---|
| RC packets from the real adapter decode correctly | 8 tests: idle, 500 random stick sets at all speed tiers, every flag, tilt bits, keepalive, corruption, and the real adapter object over UDP including its keepalive thread |
| Upstream backend over its /ws moves the twin; takeoff, land, E-stop behave | pass: incremental (keyboard) yaw turns it and decays on release; pitch moves it forward; land lands; E-stop cuts motors in the air (exactly one packet carries it); UI tilt message moves the servo |
| OpenCV (CAP_FFMPEG, TCP) reads the RTSP stream | pass, with the upstream adapter's exact options |
| Counter decodes on at least 99% of frames | 491/491 and 492/492 through RTSP in two runs (100%); 250/250 through a 3 Mb/s hevc_videotoolbox file in the unit test |
| Total latency within +/-50 ms at 350 and 600 ms | 350: p05 348.3, p50 351.9, p95 358.3 ms. 600: p05 599.3, p50 602.0, p95 609.1 ms (225 frames each) |
| S1 to S3 render at 15 fps or more | S1 24.98, S2 24.92, S3 24.92 fps (minimum of 4 samples; target 25) |
| Screen counter (bonus) | static code decodes in at least 19 of 20 camera-like views (perspective, blur, noise, JPEG 70); live page clock within milliseconds of now_ms |
| Stall-then-burst | 1.5 s stall: 1510 ms gap, 43 frames within 500 ms after it, first frames 1819 ms old, recovers to 378 ms at a 350 ms set point |
| Link loss ignores keepalive | pass: control packets stop, keepalives continue, the twin lands after 1 s |

## Measurements (with every other workstream also running: percep SAM MLX, gov tests, train, agent, ui)

- Twin pipeline baseline (render to decoded at an RTSP reader with one decoder thread): about 97-100 ms at 25 fps. Upstream part (render, JPEG, decode, noise, stamp) about 13 ms; encode + RTSP + decode about 85-125 ms (VideoToolbox holds about 3 frames).
- **OpenCV's default FFmpeg decoder threading adds about 0.6 s.** At the upstream backend's own `/mjpeg` output: p50 102 ms with `OPENCV_FFMPEG_THREADS=1`, p50 697 ms without it (twin with no injected delay). With the twin set to 500 ms total, the default-threads backend sees 1100 ms. The same applies to the real drone's stream.
- Upstream turbodrone RC loop (unchanged code, twin + Chrome + backend running, 60 s): inter-arrival p50 47.4 ms, p99 51.1 ms, max 52.9 ms, so about 21 Hz rather than 25 Hz (it sleeps 40 ms after doing its work). No alarm, but p99 is only 9 ms under the 60 ms threshold before the governor runs in that thread.
- Renderer: Chrome 157, `ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Pro)`, 25 fps camera page plus 25 fps chase recording at the same time.
- SAM 3.1 through the Meta API on HEVC-path twin frames (640x360 JPEG, 10 frames per scenario, gate G2): S1 "person in blue shirt" 10/10 (IoU with truth 0.88-0.99, conf 0.89-0.95); S2 "person in blue shirt" 9/9 visible frames, zero hits on the red-shirt walker; S3 "green car" 10/10 (IoU 0.84-0.97), zero hits on the parked red car. Median call 0.7-1.1 s. 43 images, $0.11 (logged to spend.jsonl as ws twin).
- Soak, 3 minutes at 00:56 ET (`run_twin.sh` S2 at the 500 ms default, Chrome, upstream backend reading video, demo pilot flying): 25.0 fps throughout, total p50 501.5 ms / p95 507 ms at the twin probe and p50 503.3 ms at the backend's `/mjpeg`, 100% counter decode, no encoder restarts or render timeouts, twin server memory flat at 239-241 MB, zero errors in the server, renderer, backend, and pilot logs. Footprint: twin server about 21% of one core, ffmpeg 3%, Chrome about 25% across processes, about 1 GB of memory in total.
- Oracle-driven follow on S1 (demo pilot flying, oracle at 250 ms latency, 2% dropouts): time-to-find 2.1 s, lock rate 0.93, control bearing RMS 0.042 rad, perception bearing error 0.004 rad, 0 identity switches.

## Shared artifacts

`$MHD_HOME/media/shared/twin/`: `s1/`, `s2/`, `s3/` with 60 HEVC-path frames each (`frame_<render_id>.jpg`), `truth.jsonl`, `overlay/` (truth boxes drawn), `capture.json`; `chase-s1|s2|s3-*.mp4` (30 s chase views with drone-camera PiP, HEVC 1280x720 25 fps); `sam_s*.json` (the SAM spot-check rows); `README.md`.

## Decisions (made without the owner)

- Procedural people and cars, not CC0 packs: kenney.nl answers curl with 403 Forbidden, and SAM 3.1 detects the procedural actors (above).
- The truth projection runs in Python from the same skeleton constants as the renderer (`PERSON_SKELETON` in scenarios.py = `SKEL` in world.js; a test checks they match). Change both together.
- Twin children (mediamtx, ffmpeg) stay in the server's process group; scripts and tests kill the group.
- The latency set point is the TOTAL at an OpenCV reader with one decoder thread, per frame (render to decode).
- Placeholder video (flat boxes, labeled "NO RENDERER") streams when no WebGL page is connected, so a backend always has a stream; `--no-placeholder` turns it off.
- The twin's drone does not latch E-stop (the governor does); after an E-stop it falls, and a new takeoff works.
- Unknown flags1 bits are counted (`unknown_flag_packets`); a headless bit can be mapped in x69.yaml once Session A finds one.

## Known gaps

- Dynamics numbers are placeholders until Sessions A and B: `hover_height_m`, `hfov_deg`, `link.loss_timeout_s` and `loss_action`, deadbands, yaw rate and speed per count, lags, tilt step. Refit `config/twin/x69.yaml`.
- No H.265-over-UDP video mode (only RTSP). Add it only if Session A finds no RTSP on the unit.
- Stall bursts are paced by the encoder (VideoToolbox with `-realtime 1`), so a 1.5 s burst drains in about 0.5 s, not instantly.
- Actors are procedural; no textures on clothing, no faces beyond eyes and nose.
- Tier-3 scenarios exist (S4 occlusion behind a shed and a tree, S5 subject leaves the frame, S6 heavy drift over smooth concrete) but have no metrics thresholds or sample frames yet.
- The screen counter was tested on warped screenshots only, never on the real drone (the twin workstream must not talk to it). Its numbers include the display's own latency (1-3 frames at 60 Hz); use `--display-ms` if you have an estimate.
- The upstream React UI was not driven in a browser; the acceptance test drives the backend's /ws with the UI's keyboard messages.
- The chase HUD text is HTML, not in the recorded MP4 (caption in the editor).
- The in-process `x69_sim` adapter pair (PLAN 4.2 first layer) is not part of this workstream.

## For the integrator

1. **Missing upstream files in git.** The root `.gitignore` rule `models/` hides `turbodrone/backend/models/` (15 files), and upstream's own `turbodrone/.gitignore` rule `lib/` hides `turbodrone/frontend/src/lib/ws.ts`. Neither is in the import commit, so a fresh checkout or worktree cannot run the backend. I copied them into this worktree locally (still ignored, not committed). Fix: anchor the rule as `/models/` and `git add -f` those files (unchanged from upstream 2c09cf8).
2. **Set `OPENCV_FFMPEG_THREADS=1` for every backend that reads RTSP, twin and real** (or pass `[cv2.CAP_PROP_N_THREADS, 1]` to `cv2.VideoCapture`). It removes about 0.6 s of video latency, which changes the autopilot's theta and gains. `scripts/run_upstream_backend_twin.sh` already sets it.
3. **gov's mediamtx config leaves MoQ on.** mediamtx 1.21 enables a MoQ server by default (`:8892` TCP+UDP, `:8893` UDP). `tests/gov/mediamtx_gov.yml` was running orphaned (PPID 1) at 00:18 with `*:8892` bound. Add `moq: false` (see `src/mhd/twin/media.py`).
4. `mhd.twin.counter` is already used by gov (`src/mhd/backend/capture.py`, `decode(img)`) and percep (`fusion.py`, `COUNTER_ROI`); the API matches.
5. `TwinControlExt` (server.py) extends the frozen `TwinControl`; promote its fields into the contract if other processes need them.
6. To run the full loop in the twin: `scripts/run_twin.sh` (defaults), the gov backend with `FLIGHT_MODE=twin DRONE_IP=127.0.0.1 CONTROL_PORT=23458 X69_LG_RTSP_URL=rtsp://127.0.0.1:8554/live/ch00_1 OPENCV_FFMPEG_THREADS=1`, then perception (or `mhd.twin.oracle`), and `mhd.twin.metrics` on the twin's truth log plus the perception or backend logs.
7. The UI can embed the chase view: `<iframe src="http://127.0.0.1:8766/chase">` (a passive viewer; it renders in the UI's browser).

## Final test run

```
$ $MHD_PY -m pytest tests/twin -q          # 2026-11-04 00:52-00:55 ET, other workstreams running
65 passed in 171.33s (0:02:51)
```

Spend: Meta Model API sam-3.1, 43 images, about $0.11 (5 lines with ws "twin" in `$MHD_HOME/logs/spend.jsonl`). No Fireworks use.
