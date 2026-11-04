# Morning runbook: the real Loiley X69 (Sessions A, B, C)

Written overnight on 2026-11-04 for the owner, who starts at about 07:15 ET. PLAN.md sections 5 and 6
say why each step exists. This file says what to type and what counts as a pass.

Every tool here was tested against the wire-level twin or loopback only. Nothing has been sent to the
real drone yet, so Session A is the first time this code talks to it.

**Rules for the whole morning**
- **Session A is a hard gate.** No props-on flight until its summary says GO.
- **One person stays on LAND (L) and E-STOP (Esc)** whenever the motors can spin. Esc cuts the motors in
  the air, so the drone falls. Use it only for a real emergency. LAND is the normal stop.
- **Stop at the first unexpected behavior.** Land, take the battery out, and write down what happened.
- **People stand beside the flight line, never behind the drone.** Everyone filmed has agreed on camera.
- **Battery out whenever you are not flying.** Charge only while someone is watching.

## 0. At a glance

| When (ET) | What | Done when |
|---|---|---|
| 07:15-07:35 | Setup (section 1) | the preflight prints PREFLIGHT PASS |
| 07:35-08:05 | Session A, props off (section 2) | `session_a.py` prints GO |
| 08:05-08:45 | Session B, props on, pack 1 (section 3) | sysid results applied to `config/real/` |
| 08:45-09:30 | Claude refits the twin; you do a voice dry run | twin within 20 % |
| 09:30-11:30 | Session C, packs 2 then 1, with a second person (section 4) | one clean voice, follow, land clip |
| 13:00 | Hardware go/no-go | real clip, or the twin-led storyboard |

**Bring:**
- both packs, charged;
- the iPhone and its USB cable (for the tether);
- a second phone to film with;
- the prop tool, a paint pen or tape, and a tape measure;
- the printed ArUco marker (`docs/img/aruco_sysid.png`, section 3.5);
- a blue shirt, a shirt in another color, and a green toy car;
- a non-flammable surface for the bench tests.

All commands assume this, in the terminal app you will fly from:

```bash
cd /Users/russellvancuren/Projects/MetaHackathonDrone      # the checkout you fly from
source env.sh
```

## 1. Setup (07:15)

**1.1 Charge the packs.** Charge both, attended (section 8). Pass: both packs show full on the charger.

**1.2 Fly from one fixed commit.**
```bash
git status --short | head        # expect nothing: no edits during a session
git log -1 --format='%h %s'      # write the commit down
git tag -f flight-$(date +%m%d) && git log -1 --format=%h flight-$(date +%m%d)
```
- No agent edits or restarts anything while you fly.
- Pass: the tree is clean, and you have written down the commit.

**1.3 Use one terminal app for everything.**
- macOS grants Local Network and Microphone access to each app separately.
- Pick one app, Terminal or iTerm2, and run the backend, the agent and every script below from it.
- An editor's built-in terminal also works, but then the permission belongs to the editor.
- Pass: every window you use is in that one app.

**1.4 Set up the iPhone USB tether, then the service order.**
1. On the iPhone, open Settings > Personal Hotspot and turn on Allow Others to Join.
2. Plug the iPhone into the Mac by USB. Tap **Trust** on the iPhone if it asks.
3. On the Mac, open System Settings > Network. "iPhone USB" appears in the service list.
4. Click the **(...)** button under the list, then **Set Service Order...**. Drag **iPhone USB above Wi-Fi**, then click OK.
5. Wi-Fi: turn off auto-join for every network except the drone's. That way the Mac rejoins the drone after a Wi-Fi blip.

Pass: the preflight's `routes/route to api.meta.ai` line passes "via enX (iPhone USB)" (step 1.7).

**1.5 Grant the permissions.**
- **Local Network:** System Settings > Privacy & Security > Local Network. Turn on your terminal app.
  - If the app is not listed, run the preflight once from it (step 1.7) and click **Allow** on the prompt.
  - Then quit the app with Cmd-Q and reopen it.
- **Microphone:** System Settings > Privacy & Security > Microphone. Turn on the same app, then restart it.
  Check it worked:
  ```bash
  $MHD_ML_PY -m mhd_ml.agent.mic_check --seconds 20     # speak, then say "stop" and "land"
  ```
  - Pass: the level bars move, and "stop" and "land" each print a SAFETY line.
  - If it prints ALL ZEROS, the permission did not take.

**1.6 Keep the Mac awake, and keep the radio quiet.**
- Plug the Mac into AC power, keep the lid open, and turn auto-lock off.
- `scripts/run_stack.sh` runs `caffeinate -dimsu` for as long as it runs. Also turn on Amphetamine if you use it.
- Turn AirDrop off in Control Center. Turn Handoff off in System Settings > General > AirDrop & Handoff.
- Optional, until the next reboot: `sudo ifconfig awdl0 down`. AWDL hops the Wi-Fi radio off channel, which causes about 100 ms of lag spikes.
- Pass: the preflight's `system/AirDrop / AWDL` line says PASS.

**1.7 Stop the overnight processes, then run the preflight with the drone on (props off).**
```bash
pgrep -fl "run_twin|mhd.twin|mlx_lm|uvicorn web_server|perception.service|agent.service|run_stack"   # leftovers
# stop each with Ctrl-C in its own window; if the window is gone (SIGTERM, then check again):
pkill -f run_twin.sh; pkill -f mhd.twin.server; pkill -f mhd.twin.launch_renderer; pkill -f "mlx_lm.*Muse-Glimmer"
pgrep -fl "mhd.twin|mediamtx|mlx_lm" || echo "all clear"
```
The twin must be off for real flights, because of its GPU and CPU load. The overnight Glimmer must be off too: only one Glimmer fits in memory, and `--glimmer local` starts its own.
Power on the drone with its props off, and join its Wi-Fi on the Mac. Then:
```bash
$MHD_PY scripts/preflight.py
```
- Every check prints PASS or FAIL. Each FAIL prints the fix under "HOW TO FIX".
- The checks cover:
  - the doctor, for both venvs;
  - the Mac's address on the drone's Wi-Fi (172.16.11.x);
  - `route -n get 172.16.11.1` going via en0, and `api.meta.ai` going via the iPhone;
  - HTTPS to api.meta.ai;
  - Local Network access (errno 65 means the permission is missing);
  - ffprobe of the stream;
  - a second-client test;
  - AWDL.
- Pass: it ends with `PREFLIGHT PASS`. Results are saved to `logs/preflight-<time>.json`.
- macOS hides the Wi-Fi name unless the app has Location Services. The 172.16.11.x address check is what counts.
- Run the preflight **before** the backend starts. The video checks open extra RTSP clients, and the X69 may allow only one.

## 2. Session A: props OFF (07:35)

The motors run for at most 6 s per test, and the script sends LAND itself. Film every motor test and
every link-loss test with the second phone.

**2.0 Run the guided checklist.** It does sections 2.1 to 2.11 in order. For each step it prints exactly
what to do, runs what it can, asks for your observation, and saves after every answer.
```bash
$MHD_PY scripts/session_a.py                 # answers: y / n / ? (unknown) / s (skip step) / q (save and quit)
$MHD_PY scripts/session_a.py --list          # step names
$MHD_PY scripts/session_a.py --resume logs/session_a-<time>.json            # continue after a quit
$MHD_PY scripts/session_a.py --only linkloss --resume logs/session_a-<time>.json   # redo some steps
```
- Results go to `logs/session_a-<time>.json` (the per-step data) and `media/session_a-<time>/` (the recordings).
- Pass: the summary prints `GO for Session B (props on)`.

The rest of this section lists what each step does and its pass criterion, so you can also do it by hand.

**2.1 Props.** Mark each prop and its motor arm A or B, photograph every corner, and remove the props with the prop tool.
- Pass: all four props are off and photographed.
- Props are handed. A swapped prop flips that motor's thrust in Session B.

**2.2 Tether, routes, permissions.** The checklist runs `preflight.py` (network groups).
- Pass: no FAIL.
- Optional: repeat gate G0 over the tether (about $0.05):
  `$MHD_ML_PY -m mhd_ml.agent.bench spark --n 10` and `... bench voice --n 5`.
  Pass: Spark p95 is at most 4 s, and voice is at most 1 s.

**2.3 Stream format.** With the backend stopped, the checklist runs `preflight.py --only video`.
- It records the frame rate, the keyframe interval, the bitrate, and the second-client verdict.
- Pass: an HEVC 1280x720 stream at 10 fps or more.
- If it reports "only one client" or "kicks the first one", **never** open a second player while flying. That would freeze the backend's video and land the drone.

**2.4 Recordings, with the backend still stopped.**
- 60 s of raw RTSP (`ffmpeg -c copy`).
- 2 minutes of you hand-carrying the drone, props off, past the subjects: a person in a blue shirt, a person in another color, and a green toy car.
- The microphone test.
- Pass: the files are in `media/session_a-<time>/`, with the size and duration printed. Raw footage of people never goes into git.

**2.5 Backend and UI.** In a second window of the same app:
```bash
scripts/run_stack.sh --mode real --no-perception --no-agent      # backend + UI, logs in logs/stack-<time>-real/
```
Open http://127.0.0.1:5173 in its own Chrome window. Keep it maximized, never minimized, and keep it out of Memory Saver.
- Pass:
  - the banner reads **REAL DRONE** (red stripes);
  - the chips read CTRL open, HB 5.0 Hz and TLM 10 Hz;
  - the video is live.
- Type **ARM** and press Enter. The badge must read ARMED.
- Click pack 1 or pack 2.

**2.6 Latency.** The checklist serves the counter page at http://127.0.0.1:8766/counter.
1. Open the page full screen (Cmd-Ctrl-F), turn screen brightness up, and turn Night Shift off.
2. Hold the drone 30-60 cm from the screen, with the whole code and all four markers in view.
3. It measures for 60 s on the backend's own frames. Latency is the backend's receive time minus the time the code was drawn, on the same clock.

- Pass: 100 or more decoded frames, with p50 written down (expect 300-900 ms).
- If nothing decodes: film a phone stopwatch next to the UI video and type in the difference.
- Then, before Session B:
  ```bash
  export MHD_CAPTURE_OFFSET_MS=<p05 from the summary>      # in the window that starts the backend
  ```
- The checklist also logs `capture_rise`. If it climbs toward 300 within a minute, the camera's clock drifts. Tell Claude: `MHD_CAPTURE_DRIFT_MS_PER_S` needs raising.

**2.7 HFOV.**
1. Put the drone exactly 1.00 m from a flat wall, camera level.
2. Stretch a tape measure across the whole view.
3. Read the tape at the left and right frame edges. A snapshot is saved to read from.

- HFOV = 2·atan(w/2d).
- Pass: between 40 and 140 degrees, and written down (the default guess is 75).

**2.8 Flags, props off.** Each is sent over the backend's `/ws`, exactly as the UI sends it.

| Test | What you watch | Pass |
|---|---|---|
| Takeoff (0x01), then LAND after 6 s | motors spin within about 2 s | they spin with no extra arm step |
| LAND (0x80) | the motors stop | they stop within a few seconds (if not, see 2.8a) |
| E-STOP once while spinning | the motors cut | they cut at once |
| TAKEOFF while E-stopped, then 5 more E-STOPs | nothing | the motors stay off (the E-stop is latched) |
| E-STOP reset | governor goes to LANDING, then Landed, then ARM again | the motors stay off |
| LAND while on the ground | nothing | a no-op: no twitch, no beep |
| Tilt: one 0.3 s press down, then up, then a 3 s hold | the camera moves | degrees per press and full travel written down |

**2.8a If LAND (0x80) does not stop the motors:** E-STOP stops them for the test. Then turn on the
throttle-ramp landing for every later session. While LANDING it ramps the throttle stick down to −0.35
(44 counts) over 1 s, and keeps re-asserting 0x80 (`config/governor.yaml`, `landing:` section).
```bash
$MHD_PY scripts/apply_sysid.py --throttle-ramp on            # shows the diff
$MHD_PY scripts/apply_sysid.py --throttle-ramp on --write    # writes config/real/governor.yaml
```
- Fly with `--config-dir config/real` from then on.
- The first props-on LAND in Session B is from **low height (about 0.5 m)**, to check the ramp really descends.
- Pass: the drone descends and settles. Then press **Landed**. If the motors still idle on the ground, E-STOP once.

**2.9 Link loss, three ways, 10 s each, motors spinning and filmed.**

| Test | How | Write down |
|---|---|---|
| 1. The backend dies | the checklist sends `kill -9` to the backend. Not Ctrl-C: Ctrl-C lands first. | what the motors do, and after how many seconds |
| 2. The Mac's Wi-Fi goes off | you turn Wi-Fi off for 10 s, then on again | the motors; whether the Mac rejoins the drone by itself |
| 3. RC packets stop, keepalive continues | `POST /debug/rc_pause {"seconds": 10}` | see the note below |

- After test 1, press **Enter in the `run_stack.sh` window**. It restarts the backend with the same settings, and the UI reconnects by itself. The backend comes back disarmed, so type ARM again.
- **Note on test 3.** In real mode the backend allows the pause only while the governor is **ON_GROUND**, for at most 15 s.
  - So test 3 runs with the motors **idle**. It shows whether the drone tolerates 10 s of keepalive-only traffic: does the video keep streaming, does an LED change, does it still take off right after?
  - LAND, E-STOP and TAKEOFF end the pause at once.
  - It is also available by hand:
    `curl -s -X POST localhost:8000/debug/rc_pause -H 'content-type: application/json' -d '{"seconds": 10}'`
  - In the twin, the same test, run while airborne, passed: control packets stopped, keepalives continued, and the twin's 1 s link-loss timer fired.
- Pass: in tests 1 and 2 the motors stop or ramp down within about 5 s.
  - If they keep spinning, link loss does not stop this drone. That is **NO-GO**, unless the physical remote can stop the motors (2.10). Then fly with the remote in hand.

**2.10 Physical remote.**
1. Start with the remote off. Send TAKEOFF from the backend.
2. While the motors spin, switch the remote on: does anything change?
3. Press the remote's stop: do the motors stop?
- Pass: written down. If the remote can stop the motors, keep it in hand during every flight.

**2.11 Weight and low-battery LED.** Weigh the drone with its battery. Note how the LED behaves when a pack runs low.
- Until the real endurance is known, the flight timer warns at 5:00 and lands at 6:30.

**2.12 Go or no-go, and carry the values forward.** The summary prints each gate:
- preflight;
- video;
- takeoff;
- land;
- E-stop latched;
- link loss;
- the remote;
- latency;
- HFOV.

It ends with a verdict and the commands to carry the values forward:
```bash
export MHD_CAPTURE_OFFSET_MS=<p05>                                  # then (re)start the backend from this window
$MHD_PY scripts/apply_sysid.py --hfov-deg <HFOV> --latency-ms <p50>     # check the diff
$MHD_PY scripts/apply_sysid.py --hfov-deg <HFOV> --latency-ms <p50> --write
```
- Pass: the verdict starts with `GO`.
- With `NO-GO`, do not put the props on. Use the twin-led storyboard (PLAN 8).

## 3. Session B: props ON, indoors, pack 1 (08:05)

**3.0 Space and safety.**
- At least 5 by 5 m, a ceiling of 2.5 m or more, a patterned mat (the drone holds position by optical flow), and bright light.
- E-STOP is armed the whole session. Nobody stands behind the drone.

**Before any start:** `scripts/run_stack.sh <same options> --dry-run` validates the options and prints every command it would run, without starting anything.

**3.1 Props on, then start the stack.**
1. Fit each prop on its marked arm (A on A, B on B, as in the photos).
2. In the backend window:
   ```bash
   echo $MHD_CAPTURE_OFFSET_MS                                    # the Session A p05; export it if empty
   scripts/run_stack.sh --mode real --config-dir config/real --no-perception --no-agent
   ```
3. In the UI: REAL DRONE, all chips green, ARM, then pick the pack.

- Pass: TAKEOFF is enabled (no reason is shown under it).
- If you have not run `apply_sysid` yet, use `--config-dir config` (the conservative defaults).

**3.2 First hover.**
1. Click TAKEOFF. The governor holds neutral for 4 s.
2. Hover for 30 s and watch the drift.
3. Press LAND (L). When the motors stop, press **Landed**.

- Pass: it holds within about 0.5 m, and LAND lands it. If you set the throttle ramp in 2.8a, land from about 0.5 m first.

**3.3 Sign check, by hand.** Hover, then:
- Tap the **Right arrow** (+yaw). The nose must turn right (clockwise from above), and the scene moves left in the video.
- Tap **W** (+pitch). It must move forward, the way the camera faces.

- Pass: both go the expected way.
- If one is reversed: land, and write it down. The sysid in 3.5 measures the sign, and `apply_sysid` flips `wire.sign`.
- Never run the autopilot on a reversed axis. The wrong-way detector would catch it, but do not rely on that.

**3.4 Headless check.**
1. Hold the Right arrow until the nose has turned about 90 degrees.
2. Then hold W for 1 s.

- Pass: it moves where the camera now points.
- If it moves along the original heading, headless mode is on. That is **NO-GO for following** until it is off (the remote may toggle it).

**3.5 Print and place the marker.**
1. Print `docs/img/aruco_sysid.png` at **100 % / Actual size**, never "Fit to page". It is ArUco DICT_4X4_50 id 7, black square 150 mm.
2. Check the 100 mm scale bar with a ruler, and measure the black square.
3. Tape it flat on the wall at camera height. Hover 2.5-3 m in front of it, centered, in bright light with no glare.

- Pass: the marker is fully in view and sharp in the UI.

**3.6 System identification.** The drone must be AIRBORNE and hovering, facing the marker, with the UI open (real mode needs its heartbeat). Run each in a third window:
```bash
# yaw: 3 amplitudes, + then - each; a short hold keeps the marker in view
$MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis yaw --amps 0.10 0.20 0.35 \
    --hold 1.0 --rest 3 --marker-id 7 --marker-size 0.150 --hfov <Session A HFOV>
# pitch: moves toward the marker and back; start 2.5-3 m from the wall
$MHD_PY -m mhd.governor.sysid --backend http://127.0.0.1:8000 --axis pitch --amps 0.10 0.20 0.25 \
    --hold 1.5 --rest 3 --marker-id 7 --marker-size 0.150 --hfov <Session A HFOV>
```
- Steps are ordinary human sticks, capped at 64 counts, so **LAND and E-STOP always win**.
- It refuses unless the drone is AIRBORNE. It aborts if the flight state changes, if the governor stops flying its sticks, or if the UI heartbeat is older than 1 s. Ctrl-C releases the sticks.
- Results go to `logs/sysid-yaw-<time>.json` and `logs/sysid-pitch-<time>.json`.
- Pass: each prints a fit with K, deadband and lag (not `None`), and `[sysid] <axis> direction OK`. The drone ends near where it started.
- `direction REVERSED` means that axis moved against the stick. `apply_sysid` flips `wire.sign` for it (3.7), and it must match what you saw in 3.3.
- If the marker leaves the view on the largest yaw step, rerun with `--amps 0.10 0.20 0.30`.

**3.7 Apply the results, then turn feed-forward on.**
```bash
$MHD_PY scripts/apply_sysid.py --yaw logs/sysid-yaw-<time>.json --pitch logs/sysid-pitch-<time>.json \
    --hfov-deg <HFOV> --latency-ms <Session A p50> --camera-fps <Session A fps>
```
Read the output:
- the per-step table, and the fit R² (want 0.8 or more);
- the theta arithmetic;
- any **SIGN FLIPPED** warning: it must match what you saw in 3.3.

Then write it:
```bash
$MHD_PY scripts/apply_sysid.py --yaw ... --pitch ... --hfov-deg ... --latency-ms ... --write
$MHD_PY scripts/apply_sysid.py --feedforward on --write        # feed-forward only once K is measured (PLAN 4.4)
```
- It changes only:
  - HFOV;
  - K (`plant.yaw_rate_per_count`);
  - forward speed per count;
  - the yaw lag;
  - theta;
  - `wire.sign` for yaw and pitch;
  - `wire.deadband_counts` for yaw and pitch.
- Every changed line is tagged `[measured <time>: source]`.
- It refuses values outside sanity ranges, unless you pass `--force`.
- Feed-forward passed the ±30 % K robustness test in the twin tonight: bearing RMS 0.062-0.089 rad, against 0.130 for P-only.
- Pass: it prints `wrote config/real/...`, and the next backend start loads it without error.
- Then land, take the battery out, and send `logs/sysid-*.json` to Claude to refit the twin (`config/twin/x69.yaml`).

## 4. Session C: following, with a second person (09:30)

Rules:
- One person stays on LAND and E-STOP.
- The subject walks at 0.3 m/s or less, beside the flight line.
- Stop at the first surprise.

If you are working solo:
- give the mission at the laptop, then step into frame;
- rely on the local "land" listener;
- keep the remote in hand if Session A showed it can stop the motors;
- cut the caps by a third.

**4.1 Yaw-only, at 60 % gain, on pack 2.**
1. Set the configs:
   ```bash
   $MHD_PY scripts/apply_sysid.py --yaw-only on --yaw-gain-scale 0.6 --write   # autopilot pitch cap 0, loop gain 0.24
   ```
2. Start the stack with perception and the voice agent:
   ```bash
   scripts/run_stack.sh --mode real --config-dir config/real
   ```
3. Take off by button.
4. Hold Space and say "follow the person in the blue shirt". Or type the mission:
   ```bash
   curl -s localhost:8000/mission -H 'content-type: application/json' -d '{"type":"mission","mission_id":"typed-1",
     "intent":{"action":"follow","subject_phrase":"person in blue shirt","category":"person","color":"blue",
     "distance":"medium","reason_code":"none"},"transcript":"typed","model":"operator-typed","latency_ms":0}'
   ```
   If the ack says `"paused": true`, press **Resume**.
5. The subject stands still 3 m away, then walks slowly sideways.

- Pass:
  - FIND, then CONFIRM, then FOLLOW;
  - the nose keeps the subject near the center without swinging back and forth;
  - no wrong-way flag.

**4.2 Add forward and back, at a 2-4 m standoff.**
```bash
$MHD_PY scripts/apply_sysid.py --yaw-only off --write                     # keep 0.6 gain for the first run
$MHD_PY scripts/apply_sysid.py --yaw-gain-scale 1.0 --write               # only after a steady run
```
Restart the stack between changes: press Ctrl-C with the drone landed, then run the same command again.
- Pass: it closes in smoothly, and backs up slowly when the subject walks toward it.
  - It moves forward only when the whole subject is in frame.
  - Backing up is limited to **0.5 m per flight**. A new mission or Resume does not refill it.
  - Closer than about 2 m (box over 85 % of the frame height or 60 % of its width), it only **hovers**. It never backs away by itself (integration's 04:09 safety review). The subject steps back to make it follow again.

**4.3 The full loop**, to record:
```bash
scripts/run_stack.sh --mode real --config-dir config/real --glimmer local --agent glimmer   # tuned Glimmer + LoRA on :8080 (~17 GB)
```
`--agent glimmer` makes the tuned Glimmer answer every command. With `--agent auto`, Spark (about 1.6 s) races local Glimmer (about 3.6 s) and usually wins, so the badge would read muse-spark-1.3.
1. Start screen recording (Cmd-Shift-5, saved to `media/`), and film the room with the second phone.
2. Take off by button.
3. Hold Space: "follow the person in the blue shirt". The mission card must show the **glimmer-30b+lora** badge.
4. Say "land". The local safety listener lands the drone.

- Pass: one clean clip, from voice, to follow, to voice land.
- Local Glimmer answers in about 3.6 s, and the keyword grammar answers if it has not replied by 9 s.
- Start the stack about a minute before you need Glimmer, so it can load. SAM and Glimmer share the GPU, so the detector slows to about 0.7 s.
- Without `--glimmer local`, the agent races Spark against the keyword grammar.

**4.4 Re-flights and fixes, 11:30-13:00.** The hardware go/no-go is at 13:00.

## 5. What the UI tells you

Banners:

| Banner | Means | Do |
|---|---|---|
| **REAL DRONE** (red stripes) or **TWIN** (blue) | the backend's `FLIGHT_MODE` | real flights need REAL DRONE |
| **DISCONNECTED** (red, full width) | the control socket is down; LAND and E-STOP fall back to HTTP | wait for the reconnect; if the backend died, see section 6 |
| **NO HEARTBEAT** (red, full width) | the UI's heartbeat worker or its socket stopped; the governor holds after 1.5 s and lands after 5 s | reload the page (a reload costs about 0.3 s of heartbeat) |
| **NO TELEMETRY** (amber) | flight state unknown; takeoff locked | LAND and E-STOP still work; check the backend |
| Flight badge: ON_GROUND, TAKING_OFF, AIRBORNE, LANDING, ESTOPPED | the governor's state machine | after a landing, press **Landed** |
| Autopilot badge: IDLE, FIND, CONFIRM, FOLLOW, DOUBT, LOST, SEARCH, GIVE_UP | the autopilot's state | GIVE_UP holds until Resume or a new mission |
| Pack timer, amber at 5:00, red at 6:00 | flight time on this pack; auto-land at 6:30 | land and swap packs |

Safety flags (the Safety panel):

| Flag | Means | Effect |
|---|---|---|
| `heartbeat_missing` | no UI heartbeat for 1.5 s | all sticks neutral; land after 5 s |
| `video_stale` | video lag beyond the best case is over 0.7 s | autopilot neutral; land after 2 s |
| `obs_stale` | perception is late or dead | autopilot neutral |
| `wrong_way`, `wrong_way:yaw`, `wrong_way:pitch`, `wrong_way:frozen_box` | the drone moved against its command, or the box froze while turning. For pitch, a short probe (pitch held at 0) runs first, so a subject who simply walked away faster does not latch it. | hover, latched until **Resume**; land and check the signs (3.3) |
| `autopilot_paused` | voice "stop" or hover, Pause, or a mission spoken before the last stop | press Resume, or say a new follow |
| `landing:<cause>` | why it is landing: `ui_land`, `http_land`, `agent_land`, `model_land`, `heartbeat_lost`, `video_frozen`, `flight_timer`, `estop_reset` | press **Landed** after touchdown; Resume cancels |
| `flight_timer_warn` | 5:00 on this pack | land soon |
| `rc_send_errors`, `keepalive_stale` | UDP sends to the drone are failing | the Wi-Fi link is bad: land |
| `rc_slow` | an RC packet gap over 100 ms | CPU starved: stop other heavy work |
| `rc_paused` | `/debug/rc_pause` is active (Session A test 3 only) | it ends by itself; LAND or E-STOP end it at once |
| `governor_fault`, `rc_tick_stale`, `autopilot_fault` | an internal error | land, and save the logs |
| `legacy_axes_refused` | an old UI tab is sending axes | close the other tab |
| `headless_bypass` | twin-only test mode | never in real mode (refused) |

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| preflight: errno 65 at Local Network | the terminal app lacks the Local Network permission | section 1.5; quit the app with Cmd-Q and reopen it |
| preflight: `api.meta.ai` goes via en0 | the drone's Wi-Fi is first in the service order | section 1.4: iPhone USB above Wi-Fi |
| preflight: no 172.16.11.x address | not on the drone's Wi-Fi | join it; turn off auto-join for other networks |
| No video in the UI; `video_stale` | another RTSP client is connected, or the drone camera is stuck | close every player; power-cycle the drone; run `preflight.py --only video` with the backend stopped |
| TAKEOFF disabled, "not armed" | real mode needs ARM after every backend start and every E-stop reset | type ARM and press Enter |
| TAKEOFF disabled, "no video" or "no heartbeat" | see those flags | fix the video; bring the UI window to the front |
| TAKEOFF shows NOT ACCEPTED | the governor refused (see its flags) | read the Safety panel |
| Mission acked `"paused": true` | it was spoken before the last stop, land or takeoff | press Resume, or say it again |
| FIND, then GIVE_UP after 25 s | the subject was not seen | better light; say the shirt color; face the subject |
| `wrong_way:yaw` right after FOLLOW | yaw sign reversed, or K far off | land; redo 3.3 and 3.6; check `wire.sign.yaw` |
| Voice not heard | Microphone permission, or prop noise | `mic_check`; use a headset mic (`--agent-extra "--mic-rate 16000"` for Bluetooth) |
| `run_stack: port 8000 is already in use` | an old backend is still running | `lsof -nP -iTCP:8000 -sTCP:LISTEN`; Ctrl-C it in its window |
| A component EXITED banner in run_stack | that process died; see `logs/stack-<time>-real/<name>.log` | backend: restart it with the printed command (the drone gets no RC until then) |
| UI says DISCONNECTED and the backend is dead in the air | backend crashed | E-STOP is unreachable. Use the physical remote if Session A showed it works, otherwise the drone's link-loss behavior applies. Restart the backend at once. |

## 7. Shutdown

1. LAND, then press **Landed** once the motors stop. Then take the battery out.
2. Press Ctrl-C in the `run_stack.sh` window. It sends SIGINT to the backend first, which lands at once (that does not matter on the ground). Then it stops the clients, the UI, and the twin if there is one.
   - More Ctrl-Cs are ignored while it stops. Anything still running after its timeout (the backend's is 8 s) gets TERM, then KILL.
3. Check nothing is left: `pgrep -fl "uvicorn web_server|perception.service|agent.service|mlx_lm"` should print nothing.
4. Rejoin your home Wi-Fi if you need it. Send Claude:
   - `logs/session_a-*.json`;
   - `logs/sysid-*.json`;
   - `logs/stack-*`.

## 8. Battery and charging safety

- Charge only while someone is in the room, on a non-flammable surface, and away from paper and fabric. Use a LiPo bag if you have one.
- Never charge a pack that is hot, puffy, dented, or was in a crash. Let packs cool for 15 minutes after a flight before charging.
- Take the battery out of the drone whenever you are not flying. Never leave a pack plugged into the drone.
- Fly a pack for about 6 minutes, then swap. The timer warns at 5:00 and lands at 6:30. Click the pack button (pack 1 or pack 2) on the ground before each flight, so the timer counts the right pack.
- Stop on the low-battery LED. Never fly a pack down to cutoff indoors.
- If a pack smokes, hisses or swells: move away and keep people back. Do not use water on a burning pack. Let it burn out on the non-flammable surface, and air the room.

## 9. Where things are

| What | Where |
|---|---|
| Preflight results | `logs/preflight-<time>.json` |
| Session A answers and measurements | `logs/session_a-<time>.json`, plus `logs/session_a-<time>/` (latency CSV, HFOV snapshot) |
| Recordings | `media/session_a-<time>/` (gitignored) |
| Stack logs | `logs/stack-<time>-<mode>/` (twin runs also keep the backend's control logs in `backend/`) |
| System identification | `logs/sysid-<axis>-<time>.json` |
| Real-drone configs | `config/real/{autopilot,governor}.yaml`, written by `scripts/apply_sysid.py` |
| Flight timer | `logs/flight_timer.json`: per pack, kept across backend restarts in real mode; twin runs use their own run dir and never add pack time |
| Real-mode control logs | `logs/control-<time>.jsonl` (every RC tick) |
| Tools | `scripts/preflight.py`, `scripts/session_a.py`, `scripts/run_stack.sh`, `scripts/apply_sysid.py`, `scripts/make_aruco.py` (docs/ws2/ops.md) |
