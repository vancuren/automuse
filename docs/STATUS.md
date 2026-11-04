# Status for the morning

Last updated: 2026-11-04 03:10 ET. Deadline: 2026-11-04 21:00 ET (submit by 19:00).
Branch with everything merged: `integration`. `main` gets it at the end.

## What works now
Everything below was measured in the wire-level digital twin on this Mac. The real drone is untested until Session A.

| Piece | Result |
|---|---|
| Full chain | Voice/mission, then SAM 3.1 on this Mac, then the autopilot inside the safety governor, then real X69 packets to the twin, which streams HEVC/RTSP back |
| Follow "person in blue shirt" (S1) | following 100% of 60 s; subject in frame 100%; bearing error RMS 0.092 rad; lock 95.5%; 0 identity switches; mission to FOLLOW 1.6 s |
| Blue vs red shirts crossing (S2) | bearing RMS 0.103 rad; lock 94%; about 3% of frames briefly on the red walker during the two crossings |
| Follow "green car" (S3) | following 100%; bearing RMS 0.089 rad; lock 96%; 0 switches |
| Fine-tuned Muse Glimmer (LoRA) | 98.0% action accuracy on 300 held-out commands, 100% on 80 hand-written ones; base Glimmer 93.7%; Muse Spark 1.3 95.3%; answers in 3.6 s on this Mac |
| Voice | Muse Voice Transcribe about 0.23 s after you stop talking; Spark intent p50 1.6 s; local stop/land listener 18/18 detected, 0 false triggers |
| Safety governor | 185 tests; an adversarial review found 14 issues, and all are fixed with regression tests |

Tests on `integration`, all passing:
- governor: 185;
- voice agent: 200;
- twin: 65;
- UI: 89;
- perception: 60;
- training: 38;
- upstream X69: 15.

## Still in progress (overnight wave 2, about 03:05-05:30)
- A recorded demo in the twin: UI screen plus chase camera, with the fine-tuned Glimmer answering live.
- Morning tools:
  - `scripts/preflight.py` (network and permission check);
  - `scripts/session_a.py` (guided props-off check);
  - the link-loss test endpoint;
  - `scripts/apply_sysid.py`;
  - `docs/RUNBOOK.md`.
- README, results, video script, licenses, and the submission checklist.

## Blocked or changed
- **Fireworks blocks uploads on your account.** Every dataset and model upload fails with "temporarily disabled... contact Fireworks AI to get access". Base Glimmer deployments do work. The LoRA was trained locally with MLX instead and runs on this Mac. To also host it on Fireworks, ask in the hackathon Discord (office hours 10-11 ET) for upload access, then run `training/fireworks.py train` and `morning`.
- **Baseten key is empty in `.env`.** It is only needed if you want Baseten as a second host.
- **Session A (props off) did not happen last night.** It moves to 07:15.

## Your morning, in order (details in docs/RUNBOOK.md once it lands)
1. Plug in the iPhone, turn on Personal Hotspot over USB, and put "iPhone USB" above Wi-Fi in System Settings, under Network.
2. Grant Local Network and Microphone permission to the terminal app you will use.
3. **Session A**, props off, 45 minutes. This is the hard gate. Use `scripts/session_a.py`.
4. **Session B**, props on, one pack: measure yaw rate, deadband, and signs; apply them with `scripts/apply_sysid.py`.
5. **Session C**, with a second person on LAND/E-STOP: yaw-only first, then forward/back, then the voice loop.
6. At 13:00, the hardware go/no-go picks the video storyboard (real-flight-led or twin-led).
7. Portal: check whether a live URL is required and whether the submission can be edited.

## Spend so far: about $2.52
| Item | Cost |
|---|---|
| Fireworks H200, base-Glimmer evaluation (deleted) | $1.10 |
| Muse Spark 1.3 | $0.80 |
| SAM 3.1 API | $0.56 |
| Voice Transcribe | $0.05 |

## Files written outside this folder, from before the rule reached the research agents
`brew` libsndfile, libvorbis, flac, and libogg; `~/.cache/fairseq2`; and two Hugging Face model folders. The cleanup commands are in PLAN.md section 13. Nothing else was written outside the folder; a watchdog checked every 4 minutes all night.
