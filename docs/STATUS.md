# Status for the morning

Last updated: 2026-11-04 05:30 ET. Deadline: 2026-11-04 21:00 ET (submit by 19:00).
Everything is merged on `integration`, and `main` matches it.
- Your step-by-step morning guide: **docs/RUNBOOK.md**.
- The submission checklist: **docs/SUBMISSION.md**.

## Read first: the follow safety rules changed overnight
Three adversarial reviews tested the follow code against a reversed pitch sign in closed loop (logs/review2/, logs/review3/). That is the failure where "forward" on the stick flies the drone backward. After the fixes, the autopilot:
- **never backs up**, so it can never fly toward a person; when someone walks up to it, it holds;
- **flies forward only toward a subject within about 6 m** (box at least 64 px wide); farther away, it only turns;
- **hovers until you press Resume** (strict pitch wrong-way check) if the box shrinks while it flies forward; a walker faster than about 0.2 m/s also trips this;
- **refuses the twin's configs in real mode**, even through a flag or an exported variable.

The final reviewer's verdict is "safe with conditions" for forward-only follow:
- **Toward the person:** a reversed drone never moved toward them.
- **Away from the person:** it drifted at most about 2.4 m away at the modeled speed.
- **The one residual:** a person walking toward a reversed drone can push it 5-6 m back before anything latches.

**Conditions before any forward follow** (Session C step 4.2, details in RUNBOOK section 4):
1. Session B: the pitch and yaw sign checks by hand, including the headless check (RUNBOOK 3.3-3.4), and sysid prints `direction OK` for both axes.
2. If sysid measures more than 0.015 m/s per count of pitch, lower `caps.autopilot.pitch` in `config/real/governor.yaml` (RUNBOOK 3.7).
3. Keep at least 6 m clear behind the drone. The operator and the safety person stand to the side, with LAND / E-STOP ready.
4. The subject walks away only, at 0.2 m/s or slower, or in walk, stop, wait steps. They never walk back toward the drone, and stop at least 2 m from it.
5. Launch exactly as the runbook says: `scripts/run_stack.sh --mode real --config-dir config/real`.

## What works now (wire-level twin on this Mac; the real drone is untested)
| Piece | Result |
|---|---|
| Follow "person in blue shirt" (S1), final code | following 100% of the time; lock 95.3%; bearing RMS 0.103 rad; 0 identity switches; mission to FOLLOW 1.6 s |
| Blue and red shirts crossing (S2) | following 90% (brief DOUBT/CONFIRM at a track change); lock 96.8%; 2.5% of frames on the red walker |
| Follow "green car" (S3) | following 100%; lock 96.0%; bearing RMS 0.127 rad; 0 switches |
| Safety in the twin | S1-S3 ran with the real thresholds: 0 wrong-way hovers. Two demo takes (05:01, 05:06) false-tripped the strict pitch check: the 0.3 m/s walker pulled away while the drone accelerated. A third (05:10) tripped because the walker turned and its box narrowed while the drone closed in. The twin config now uses the earlier twin tuning for this check: 12 % and a near-cap requirement (real: 6 % and none, marked TWIN). |
| Recorded demo takes | voice, then Glimmer, then SAM, then follow, then voice land. **New S1 take on the final code: `media/demo/demo-s1-final4-20261104-051519-ui.mp4` (and `-chase.mp4`)**: following 100% of the follow window, locked 99.6%, bearing RMS 0.095 rad, no wrong-way hover, silent landing, mission ended at touchdown. The earlier takes (S2 `demo-s2-20261104-032802`, S3 `demo-s3-20261104-034231`) predate the safety changes; cut those at touchdown (docs/video_script.md). |
| Fine-tuned Muse Glimmer (LoRA) | 98.0% action accuracy on 300 held-out commands, 100% on 80 hand-written ones; base Glimmer 93.7%; Spark 1.3 95.3%; 3.6 s on this Mac |
| Voice | Voice Transcribe about 0.23 s after you stop talking; local stop/land listener 18/18, 0 false triggers |
| Morning tools | `scripts/preflight.py`, `scripts/session_a.py`, `scripts/apply_sysid.py`, `scripts/make_aruco.py`, `scripts/run_stack.sh` |
| Docs | README, docs/results.md, docs/video_script.md, licenses and notices, secret scan |

Tests on `integration`, all passing (733 in total; governor, twin, UI, ops and docs re-run at 05:27 ET on the final code, the rest at 04:56):

| Suite | Passed |
|---|---|
| governor | 224 |
| twin | 65 |
| UI | 90, plus 6 against the real backend |
| perception | 60 |
| voice agent | 200 |
| training | 38 |
| contracts | 6 |
| upstream X69 | 20 |
| morning tools | 14, plus 3 full-stack runs against the twin |
| docs | 5 |

## Your morning, in order (details in docs/RUNBOOK.md)
1. Plug in the iPhone and put "iPhone USB" above Wi-Fi. Grant Local Network and Microphone permission to your terminal app (RUNBOOK 1).
2. Run `scripts/preflight.py`, then **Session A**, props off, with `scripts/session_a.py` (RUNBOOK 2). This is the hard gate.
3. **Session B**, props on, one pack (RUNBOOK 3):
   - the first hover;
   - the sign and headless checks by hand;
   - sysid on the ArUco marker (print `docs/img/aruco_sysid.png` at 100%);
   - `scripts/apply_sysid.py ... --write`, which creates `config/real/`. Commit it after Session B.
4. **Session C**, with a second person (RUNBOOK 4):
   - yaw-only first;
   - then forward (the drone never backs up);
   - then the full voice loop with `--glimmer local --agent glimmer`.
5. 13:00: the hardware go/no-go picks the video storyboard (real-flight-led or twin-led).
6. Portal: check whether a live URL is required and whether the submission can be edited. docs/SUBMISSION.md lists the placeholders, VIDEO_LINK and OWNER/REPO.

## Blocked or changed
- **Fireworks blocks uploads on your account.** Every dataset and model upload fails with "temporarily disabled... contact Fireworks AI". The LoRA was trained locally with MLX and runs on this Mac. To host it on Fireworks too, ask in the hackathon Discord (office hours 10-11 ET), then run `training/fireworks.py train` and `morning`.
- **The Baseten key is empty in `.env`.** It is only needed for Baseten as a second host.
- **Session A did not happen last night.** It moves to 07:15.
- **Minor, known:**
  - perception's own default field of view is 80 degrees and the autopilot config says 75; `run_stack.sh` passes the config value, and Session A measures the real one;
  - local Glimmer (about 3.6 s) loses the 2.5 s race to Spark with `--agent auto`, so use `--agent glimmer` for the Glimmer clip.

## Spend so far: about $2.52
| Item | Cost |
|---|---|
| Fireworks H200, base-Glimmer evaluation (deleted) | $1.10 |
| Muse Spark 1.3 | $0.80 |
| SAM 3.1 API | $0.56 |
| Voice Transcribe | $0.05 |
| Overnight wave 2 (demo Glimmer runs, docs, ops) | under $0.01 |

## Files written outside this folder
- **Before the rule reached the research agents:** `brew` libsndfile, libvorbis, flac and libogg; `~/.cache/fairseq2`; two Hugging Face model folders. The cleanup commands are in PLAN.md section 13.
- **At 04:27 ET:** five of my own scratch files went to the macOS temp folder. They were moved back into `.cache/tmp/integrator/` within minutes.
- **Checking:** a watchdog checked every 4 minutes all night and saw nothing else.
