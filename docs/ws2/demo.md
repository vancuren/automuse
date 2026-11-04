# Workstream demo: the end-to-end demo in the twin, with the fine-tuned Glimmer answering live

Branch `ws2/demo` (from `integration` at 6bb13fb), 2026-11-04 03:05-04:30 ET. Everything ran on this
Mac in the wire-level twin, loopback only (`FLIGHT_MODE=twin`). Nothing was sent to the drone.

## TL;DR

- **One command records the whole story:** `$MHD_PY scripts/demo_twin.py` starts the twin, the governed
  backend (not headless: the UI's heartbeat keeps it flying), SAM 3.1 perception on MLX, the fine-tuned
  Glimmer (MLX 4-bit + LoRA on 8080), the voice agent and the production UI, then flies one scripted
  mission and records it. About 2 minutes 10 seconds with Glimmer already loaded, plus about 1 minute 45
  seconds for a cold Glimmer load.
- **The story on screen:** the dashboard in TWIN mode, TAKEOFF clicked, Space held while "Drone, follow
  the person in the blue shirt" is heard, the live transcript, the mission card with the
  **`glimmer-30b+lora`** badge (3.8-4.0 s), FIND, CONFIRM and FOLLOW with the SAM mask for about 40 s
  while the subject walks, "Okay drone, please land now" heard by the local stop/land listener (open mic,
  no button, no model), LANDING, then Confirm touchdown.
- **Final takes** (UI 1920x1080 and chase 1280x720, H.264 25 fps with an AAC sound track: the spoken
  commands plus the agent's MMS-TTS replies, each placed when it was spoken). All are in
  `$MHD_HOME/media/demo/` (copied from `media/demo/` of this worktree):

  | Take | Run | UI video | Chase video | Follow window: bearing RMS / in frame / on another actor |
  |---|---|---|---|---|
  | **S1, final** | `logs/demo-20261104-033642-s1` | `demo-s1-20261104-033642-ui.mp4`, 71.4 s | `demo-s1-20261104-033642-chase.mp4`, 71.3 s | 0.082 rad / 100% / 0 frames |
  | **S2, recommended** | `logs/demo-20261104-032802-s2` | `demo-s2-20261104-032802-ui.mp4`, 69.6 s | `demo-s2-20261104-032802-chase.mp4`, 69.4 s | 0.092 rad / 94.0% / 18 of 910 (2.0%) |
  | S2, second take | `logs/demo-20261104-033923-s2` | `demo-s2-20261104-033923-ui.mp4`, 69.5 s | `demo-s2-20261104-033923-chase.mp4`, 68.9 s | 0.094 rad / 94.1% / 54 of 916 (5.9%) |
  | S3, bonus ("green car") | `logs/demo-20261104-034231-s3` | `demo-s3-20261104-034231-ui.mp4`, 68.9 s | `demo-s3-20261104-034231-chase.mp4`, 68.8 s | 0.070 rad / 100% / 0 frames |
  | S1, first take | `logs/demo-20261104-032416-s1` | `demo-s1-20261104-032416-ui.mp4`, 71.3 s | `demo-s1-20261104-032416-chase.mp4`, 71.2 s | 0.084 rad / 100% / 0 frames |

  Run directories are under `.worktrees/demo/logs/` (gitignored). The S2 03:28 take and the first S1 take
  were recorded before the Detector-bar fix, so their latency panel shows "-" for the detector.
- **Six README stills** (1280 px wide, 218-358 KB): `docs/img/demo_s1_1_dashboard_twin.png`,
  `demo_s1_2_voice_mission_follow.png`, `demo_s1_5_landing_voice.png`, `demo_s1_chase_follow.png`,
  `demo_s2_crossing_follow.png`, `demo_s3_2_voice_mission_follow.png`.
- **Fixed on the way:** (1) the twin's mediamtx watchdog restarted mediamtx during shutdown, and that copy
  outlived the twin holding the RTSP port (it is why 8554 is still taken: see Findings 1); (2) the UI's
  Detector latency bar was empty about 98% of the time.
- **Found, not fixed (frozen files):** after a deliberate landing the autopilot keeps its lifecycle
  running, says "I lost the subject. Looking around." on the ground, and in both full S1 takes flapped
  FIND/CONFIRM at about 8 Hz (Findings 2 and 3).
- Spend: about $0.001 (Voice Transcribe, one session per push-to-talk utterance; logged).

## What is here

| Path | What it does |
|---|---|
| `scripts/demo_twin.py` | The demo recorder (core venv). Starts everything on the default ports, records the UI with Chrome's CDP screencast (Playwright `channel="chrome"`, headless with Metal WebGL, viewport 1920x1080, profile `$MHD_HOME/.cache/chrome-demo`), scripts the operator, then encodes the videos, builds the sound track, scores the run with `mhd.twin.metrics`, saves stills, logs spend, and stops every process it started (process-group sweep included). |
| `scripts/demo_agent.py` | Demo-only runner for the voice agent (ml venv): `mhd_ml.agent.service.AgentService` unchanged (Voice Transcribe realtime, hedged router, MMS-TTS, the always-on local stop/land listener), with a simulated microphone: a quiet room tone (-60 dBFS) published into the agent's `MicHub` in real time in 80 ms chunks, with `say` + `afconvert` WAVs mixed in on command. MMS-TTS lines go to WAV files instead of the speakers. Commands on stdin (`play <wav>`, `ptt <wav>`, `quit`); every AgentEvent, relayed message, spoken line and mic event goes to stdout as JSON with `t_ms` on the shared clock. |
| `config/demo/agent.yaml` | Copy of `config/agent.yaml` for the demo only: `intent: glimmer`, Glimmer at `http://127.0.0.1:8080/v1` (`default_model`, no key, label `glimmer-30b+lora`), Glimmer window 9 s and keyword deadline 10 s (the local LoRA takes 3.6-4.0 s; the cloud window is 2.5 s), Spark off, a Voice Transcribe session per utterance (`warm_session: false`), preroll 0.5 s. |
| `docs/img/demo_*.png` | The six README stills. |
| `src/mhd/twin/server.py` | Fix: the mediamtx watchdog no longer restarts a mediamtx that exited cleanly (exit 0 = asked to stop = shutdown). |
| `turbodrone/frontend/src/components/LatencyBars.tsx` | Fix: the Detector bar holds the last detection's latency for 3 s. |

## Run it

```bash
cd .worktrees/demo && source env.sh          # or the main checkout after merging
(cd twin/web && npm ci) && (cd turbodrone/frontend && npm ci)   # once per checkout (node_modules is gitignored)

$MHD_PY scripts/demo_twin.py                              # S1: "person in blue shirt", 40 s of FOLLOW
$MHD_PY scripts/demo_twin.py --demo-scenario s2           # S2: the red-shirt walker crosses (stills at the crossings)
$MHD_PY scripts/demo_twin.py --demo-scenario s3           # S3: "green car" on the private track
$MHD_PY scripts/demo_twin.py --stills                     # also refresh the README stills in docs/img/
$MHD_PY scripts/demo_twin.py --keep-glimmer               # leave Glimmer loaded for the next take
```

Other flags: `--follow-s 40`, `--command "..."`, `--land-text "..."`, `--voice Samantha`, `--latency-ms 500`,
`--scenario NAME|YAML`, `--subject`, `--take NAME`, `--rtsp-port`, `--no-audio`, `--headed`, `--keep-frames`,
`--still-scn-s 32,53`, `--agent-config`, `--governor-config`, `--autopilot-config` (twin-calibrated by default).

Preconditions: ports 8000, 8766, UDP 23458 and 5173 free (it refuses to start otherwise); 8080 either free
or already serving Glimmer (then it is reused and left running); RTSP 8554 free, else it uses 8555 and says
so. Only one Glimmer process fits in memory: the script reuses a running one, `scripts/serve_glimmer.sh`
refuses a second.

What it starts, in order (logs in the run directory):

| Process | Command |
|---|---|
| Glimmer | `scripts/serve_glimmer.sh --port 8080` (mlx_lm.server + `$MHD_MODELS/adapters/glimmer-intent-v1`); ready when `/v1/models` answers, then one warm-up completion of the demo command |
| twin | `scripts/run_twin.sh --ports 8766,23458,<rtsp> --scenario s1 --latency-ms 500 --chase-rec -- --media-dir <run>/twin_media` |
| backend | `scripts/run_backend.sh --mode twin --port 8000 --control-port 23458 --rtsp-url rtsp://127.0.0.1:<rtsp>/live/ch00_1 --logs <run>/backend --governor-config config/twin/governor.yaml --autopilot-config config/twin/autopilot.yaml` (no `--headless`) |
| perception | `$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 --detector sam_mlx --tracker auto --hfov-deg 75 --obs-log <run>/obs.jsonl` (20 s warm-up) |
| agent | `$MHD_ML_PY scripts/demo_agent.py --config config/demo/agent.yaml --backend ws://127.0.0.1:8000 --out <run>/agent` |
| UI | `VITE_BACKEND_URL=http://localhost:8000 npm run build`, then `vite preview` on 127.0.0.1:5173 |
| Chrome | Playwright, headless, `--use-angle=metal` (WebGL on the GPU: `ANGLE Metal Renderer: Apple M5 Pro`), 1920x1080 |

Then: reset the twin scenario (drone on the ground at the start pose, subject at the start of its path),
start the chase recorder and the screencast, and fly the story below. Afterwards: `ui.mp4` (libx264 CRF 20,
1080p25, the repaint-driven screencast resampled to a constant 25 fps), `chase.mp4` (the twin's HEVC chase
recording transcoded to H.264), `soundtrack_*.wav`, `report.json` + `rows.csv` (`mhd.twin.metrics`, as in
`e2e_twin.py`), `follow_window` in `demo_result.json`, stills, and a spend line.

## The story, second by second (S1 final take, video time)

| t (s) | What the viewer sees | Evidence |
|---|---|---|
| 0-4.2 | Dashboard in TWIN mode, ON GROUND, AUTOPILOT IDLE, all four link chips green, "agent alive · local stop/land listener on · ASR meta · intent glimmer" | still 1 |
| 4.2 | TAKEOFF clicked (button only) | governor ON_GROUND -> TAKING_OFF at 4.22 |
| 8.4 | AIRBORNE | TAKING_OFF -> AIRBORNE at 8.27 |
| 10.9-14.2 | Space held: LISTENING; the WAV plays into the agent's mic from 11.4 s; partial transcripts stream in | 8 partials from Muse Voice Transcribe |
| 14.3 | Final transcript "drone follow the person in the blue shirt" | 184 ms after release |
| 15.6 | (sound) "Okay." | the agent's 1.2 s acknowledgment |
| 18.5 | Mission card: `glimmer-30b+lora`, 4.04 s, intent JSON `follow / person in blue shirt / person / blue / medium` | router reason `glimmer_in_window` |
| 18.9-19.9 | FIND -> CONFIRM -> FOLLOW, SAM mask on the subject; (sound) "Following the person in blue shirt." then "Got them. Following." | 1.5 s from the intent to FOLLOW |
| 20-61.7 | FOLLOW while the subject walks its loop; the drone yaws (and backs off once: "backward travel budget used up" at 54.8 s) | follow window below |
| 61.7 | (sound) "Okay drone, please land now." on the open mic, no button | WAV fed at 61.7 s |
| 63.9 | Voice panel: SAFETY heard "land" -> land · local listener · 261 ms; LANDING (`landing:http_land`); (sound) "Landing now." | the listener caught it on a partial decode; LANDING 0.32 s after the WAV ended |
| 66.5 | The twin's drone touches down | phase `landed` |
| 67.7 | (sound) "I lost the subject. Looking around." while still LANDING (Findings 2) | FOLLOW -> LOST (`confirmation_too_old`) |
| 68.0 | Confirm touchdown clicked -> ON GROUND | |
| 68-71.3 | Outro; the autopilot badge flickers CONFIRM/FIND (Findings 3) | the recording ends after the last spoken line |

## Results

### Follow quality (twin ground truth, `mhd.twin.metrics`)

`follow_window` is the FOLLOW segment only, from 5 s after FOLLOW began to the spoken land, computed from
`rows.csv` by `demo_twin.py`. `report.json` is `mhd.twin.metrics` on the whole run from the mission on,
exactly as `e2e_twin.py` scores it; it includes the acquisition turn (the subject was 0.45 rad off-axis when
FIND started) and the landing (the subject keeps walking while the drone no longer turns), so its bearing RMS
is higher than the e2e runs' 60 s follow-only numbers.

| Take | Window | Bearing RMS / max | Subject in frame | Locked | Frames on another actor | Autopilot states in the window |
|---|---|---|---|---|---|---|
| S1 final (03:36) | 36.6 s, 916 frames | 0.082 / 0.118 rad | 100% | 98.9% | 0 | FOLLOW 916 |
| S1 first (03:24) | 36.6 s, 915 frames | 0.084 / 0.132 rad | 100% | 100% | 0 | FOLLOW 915 |
| S2 (03:28) | 36.4 s, 910 frames | 0.092 / 0.178 rad | 94.0% (occlusions) | 100% | 18 (2.0%) | FOLLOW 831, DOUBT 53, CONFIRM 26 |
| S2 (03:39) | 36.6 s, 916 frames | 0.094 / 0.288 rad | 94.1% (occlusions) | 96.2% | 54 (5.9%) | FOLLOW 893, DOUBT 23 |
| S3 (03:42) | 36.6 s, 916 frames | 0.070 / 0.125 rad | 100% | 100% | 0 | FOLLOW 916 |

| Take | `report.json`: duration | time to find | lock rate | bearing RMS | perception bearing error | identity switches |
|---|---|---|---|---|---|---|
| S1 final | 56.9 s | 0.80 s | 0.899 | 0.124 rad | 0.014 rad | 0 (1 track-id change) |
| S1 first | 56.4 s | 0.80 s | 0.953 | 0.125 rad | 0.016 rad | 0 |
| S2 (03:28) | 54.9 s | 0.80 s | 1.000 | 0.200 rad | 0.015 rad | 4 |
| S2 (03:39) | 54.8 s | 0.80 s | 0.970 | 0.199 rad | 0.017 rad | 8 |
| S3 | 54.8 s | 0.80 s | 0.977 | 0.155 rad | 0.004 rad | 0 |

S2 details: in the 03:28 take the first crossing gave a 0.8 s DOUBT (box size jump, 30.7 s); at 42.8 s a
DOUBT (box size jump) timed out into LOST and FOLLOW resumed 1.0 s after the LOST (spoken: "I lost the
subject. Looking around." then "Got them. Following."); the second crossing gave a 0.5 s DOUBT (low tracker
score, 51.9 s). In the 03:39 take the first crossing gave a 0.9 s DOUBT (low tracker score) and the tracker
sat on the red walker twice for 0.3 s; at the second crossing it sat on the red walker for 1.6 s
(51.1-52.7 s) without a DOUBT, then returned to the blue shirt; the drone yawed toward the red walker
briefly (bearing max 0.29 rad). This matches the e2e S2 result (2.9% of frames briefly on the
red walker) and is a perception limit with a closer occluder of the same size, not a demo artifact. For the
video, the 03:28 take is the cleaner one.

Video and link, every take: frame age p50 508-536 ms while airborne (500 ms total latency set in the twin);
observation age p50 518-547 ms; RC gap max 50 ms; Chrome WebGL on the GPU (Metal) for the UI and the twin.
One brief `video_stale` (0.6 s, autopilot neutral) at 24.2 s in the S2 03:39 take; none in the other four.

### Voice and intent (all five takes)

| | S1 final | S1 first | S2 03:28 | S2 03:39 | S3 |
|---|---|---|---|---|---|
| final transcript | drone follow the person in the blue shirt | drone, follow the person in the blue shirt. | drone follow the person in the blue shirt. | drone follow the person in the blue shirt | drone, follow the green car. |
| release -> final (Voice Transcribe) | 184 ms | 178 ms | 220 ms | 160 ms | 197 ms |
| Glimmer + LoRA intent | 4.04 s | 4.03 s | 4.03 s | 3.95 s | 3.79 s |
| intent | follow, person in blue shirt, person, blue | same | same | same | follow, green car, car, green |
| intent -> FOLLOW | 1.53 s | 1.49 s | 1.54 s | 1.50 s | 1.63 s |
| spoken land: listener decode / LANDING after the WAV ended | 261 ms / 0.32 s | 270 ms / 0.33 s | 286 ms / 0.32 s | 260 ms / 0.34 s | 248 ms / 0.26 s |

The land WAV ("Okay drone, please land now.") is 1.97 s long; LANDING came 2.2 s after it started, 0.26-0.34 s
after it ended, every time through the open-mic listener (transcript `oky drone please land now`), never the
L-key fallback. Router reason `glimmer_in_window` every time; no keyword fallback, no errors, no page errors.
Glimmer warm-up (the demo command) 4.6 s; cold load to `/v1/models` about 1 minute 45 seconds (03:12:56 to
03:14:38, 13.9 GB resident at that point).

## Decisions (nobody to ask)

1. **The agent is the real `AgentService`, with a simulated microphone, not `--simulate`.** `--simulate`
   presses push-to-talk itself and exits after the file, so the UI would never show LISTENING and the agent
   would vanish between commands. `scripts/demo_agent.py` keeps the agent alive for the whole flight and lets
   the UI drive push-to-talk exactly as in a flight (Space -> ptt_down relayed by the backend). Its `ptt
   <wav>` command is the same as `simulate_file` for runs without a UI.
2. **"Land" goes through the open microphone, not push-to-talk.** That shows the always-on local listener
   (Omnilingual CTC on the CPU, no model, no button), which is the safety story; push-to-talk "land" would
   also work (the router's safety fast path).
3. **A room tone instead of silence** in the simulated mic, so the VAD sees a real noise floor and closes
   segments, as with a microphone. Level -60 dBFS (`--room-dbfs`).
4. **MMS-TTS is synthesized but written to files** (03:00 at night, speakers off); the files go into the
   sound track at the moment each line started. The `audible_words` echo logic still runs as in a flight.
5. **`intent: glimmer`** in a demo-only config, so the badge always shows the fine-tuned model (the task's
   goal), with the keyword grammar at 10 s as the fallback; Spark is not raced (no spend, no chance of a
   `muse-spark-1.3` badge).
6. **Headless Chrome with `--use-angle=metal`** for the UI: WebGL is on the GPU (checked in every take), the
   page is never hidden (so the heartbeat worker is never frozen), and nothing pops up on the screen.
7. **The scenario is reset just before recording**, so every take starts the same way (subject at the start
   of its path). Takeoff -> mission takes about 14 s, so the governor's "spoken before the takeoff" rule
   never holds the mission (the script would press Resume, as an operator would, if it did; it never had to).
8. **RTSP on 8555 when 8554 is taken** by another process (see Findings 1), with a warning in the log and
   `fallbacks` in `demo_result.json`. RTSP is internal between the twin and the backend.
9. **Metrics:** `report.json` exactly as `e2e_twin.py`, plus a follow-only window (see Results).
10. **README stills:** 1280 px PNG, quantized to 256 colors when needed to stay under 400 KB.
11. **The S2 03:28 take is the recommended S2 video** (lock stayed on the blue shirt; the brief loss is
    recovered on camera, with voice). The 03:39 take shows a 1.6 s flip onto the red walker.

## Findings and gaps (for the integrator)

1. **Orphaned mediamtx on 8554 (PID 45865).** Left by the integrator's 03:02 e2e run (S3,
   `logs/twin-20261104-030221-8766/mediamtx_8554.yml`, parent PID 1). The next `e2e_twin.py` or
   `run_twin.sh` on the default ports fails until it is stopped (`kill 45865`). I was not allowed to stop a
   process I had not started, so my takes used RTSP 8555. **Cause found and fixed in this branch:** on Ctrl-C
   the signal reaches the whole process group, mediamtx exits 0 at once, and the twin's watchdog restarted it
   during uvicorn's graceful shutdown; when the twin server then died, that new mediamtx was orphaned. Both
   runs that left an orphan (the integrator's 03:02 and my 03:28) logged `mediamtx exited with 0;
   restarting` during shutdown; it is a race (the watchdog polls once a second), so most runs got away with
   it. Fix in `src/mhd/twin/server.py`: no restart after a clean exit; no run after the fix logged it. Verified 5 of 5 stops without leftovers (3 without and 2 with the Chrome renderer),
   and `demo_twin.py` additionally sweeps each process group after its leader exits. `e2e_twin.py` has the
   same `Proc.stop` without the sweep.
2. **After a deliberate landing the autopilot keeps running and speaks.** The mission stays active on the
   ground (by design: "the autopilot runs whenever a mission is active, including on the ground"). After
   touchdown the camera sits on the grass, the subject's box is clipped at the top edge, SAM stops
   re-confirming, and about 4 s after the land command the autopilot goes FOLLOW -> LOST
   (`confirmation_too_old`) and announces "I lost the subject. Looking around." (both full S1 takes; not in
   the S1 smoke run, S2 or S3). In a flight the operator hears this right after saying "land". Suggestion (frozen files):
   no announcements unless AIRBORNE, or release the mission on a voice or UI land.
3. **FIND <-> CONFIRM flapping at about 8 Hz** after that, on the ground (S1 final take: 27 transitions in
   3.2 s; first S1 take: 21 in 3.5 s). Cause: the tracker score hovers around `lifecycle.confirm_min_score` 0.5
   (0.47-0.63), and `_is_candidate` (enter CONFIRM) and `_consistent` (stay in CONFIRM) use the same
   threshold with no hysteresis or debounce, so alternate frames enter and leave CONFIRM
   (`src/mhd/autopilot/core.py` lines 199-216 and 302-323, frozen). Perception meanwhile reports `confirmed`
   with `confirm_age_ms` growing past 5 s. On the ground it moves nothing, but the badge flickers and the
   hub sends 10 events/s. The same could happen in the air after a LOST. Suggestion: a debounce like FOLLOW's
   `doubt_debounce_obs`, or enter at >= 0.55 and break below 0.45; and do not count an observation whose
   `confirm_age_ms` exceeds the lost age as a candidate.
4. **S2 occlusion:** the tracker can sit on the crossing red walker for up to 1.6 s without a DOUBT (above).
   Perception's re-seed fixes it; a color check on tracker frames (not only detector frames) would catch it
   sooner.
5. **`detector_latency_ms` only on detection frames** (51 of 1,344 observations in S1). Fixed in the UI
   (bar holds the last value 3 s). Backend alternative: `latency_ms.detector` = the last detection's latency.
6. **"backward travel budget used up"** (governor safety event) in 2 of 3 S1 takes and in S3 while
   following: the subject walked toward the drone and the autopilot used its budget for backing away blind.
   Expected behavior, mentioned because it is visible in the Safety panel.
7. **The mission card shows nothing while Glimmer thinks** (about 4 s between the final transcript and the
   card). The UI could show "intent: thinking (glimmer-30b+lora)" from the agent's final event.
8. The local LoRA (3.8-4.0 s here, 3.6 s p50 in train) does not fit the cloud 2.5 s Glimmer window; the demo
   config gives it 9 s. On Fireworks (if uploads open) it would fit.
9. Not exercised: a real microphone (permission), real speakers (TTS went to files), Spark racing Glimmer,
   the gamepad. The screencast captures the page only, so the sound track is reconstructed from the WAVs and
   TTS files with their timestamps, not recorded from the speakers.
10. The chase HUD text (HTML) is not in the chase MP4 (known twin gap); the UI video shows the chase view
    with its HUD inside the dashboard.
11. Captions for the editor: "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else built
    Nov 3-4." The UI footer already says it.

## Tests

- The demo takes themselves (5 full runs plus one 15 s smoke run): 0 fallbacks other than the RTSP port,
  0 agent errors, 0 page errors, every process stopped at the end.
- `tests/ui` after the LatencyBars change: **89 passed, 6 skipped** (the opt-in gov tests), 130 s.
- `tests/twin` after the watchdog change: **65 passed** in 179 s.
- The twin shutdown check (5 stops of `run_twin.sh` by process-group SIGINT, 2 of them with the Chrome
  renderer): no process left in the group, RTSP port free afterwards.

## Shared artifacts

- `$MHD_HOME/media/demo/`: the ten MP4s in the table above.
- `$MHD_HOME/media/shared/demo/`: `README.md` (index), each take's `demo_result.json` and `report.json`, and
  the README stills.

## Spend

Voice Transcribe realtime, one session per push-to-talk utterance: about 3.5 audio seconds per take, about
$0.0002 each, $0.001 in total (lines with `"ws": "demo"` in `$MHD_HOME/logs/spend.jsonl`). Glimmer, SAM 3.1,
Omnilingual and MMS-TTS ran on this Mac. No Fireworks or Spark calls.
