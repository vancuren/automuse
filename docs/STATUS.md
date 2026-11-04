# Overnight status

Last updated: 2026-11-03 23:40 ET. Deadline: 2026-11-04 21:00 ET.

## Running now
Six builders, each in its own git worktree under `.worktrees/`, targeting about 03:00 ET:

| Workstream | What it builds |
|---|---|
| gov | backend patches, safety governor, autopilot; then an adversarial safety review and a fix pass |
| twin | wire-level digital twin: X69 packets in, HEVC RTSP out, Three.js world, ground truth |
| percep | SAM 3.1 perception (on-device MLX and Meta API), tracker, Kalman filter |
| agent | voice: always-on local stop/land listener, Muse Voice Transcribe, Glimmer and Spark intent, MMS-TTS |
| train | Muse Glimmer LoRA: data, Fireworks fine-tune and eval, plus a local MLX fine-tune as a hedge |
| ui | operator UI safety patches and the new panels |

Then, from about 03:00: integration in the twin, end-to-end tests, and tuning.

## Your keys
The Meta Model API and Fireworks keys authenticate. The Baseten key is empty, which is fine: it is only the fallback.

Overnight spend caps:
- Meta Model API: about $21 in total (SAM $8, Spark data $10, other $3).
- Fireworks: about $12 of on-demand GPU, deleted after evaluation.

## Changed from the plan
- Session A (props-off hardware check) did not happen tonight. It moves to 07:00-07:30, and Session B to 07:45-08:30.
- The on-device path (local SAM 3.1, Glimmer, Omnilingual ASR) gets built alongside the cloud path tonight.

## When you wake up (07:00)
1. Read this file and the integration notes that replace this section by morning.
2. Plug in the iPhone, turn on Personal Hotspot over USB, and put "iPhone USB" above Wi-Fi in System Settings, under Network.
3. Charge both batteries, attended, if they aren't already.
4. Session A, props off (PLAN.md section 6). Claude will walk you through it.
5. Check the hackathon portal: is a live URL required, and can you edit after submitting?
6. Confirm the second person for 09:30.
