# Demo video script (3:00)

Two storyboards. The hardware go/no-go at **13:00 ET** picks one (PLAN.md sections 4.9 and 8):

- **A, real-flight-led:** use it if Session C recorded one clean run: voice command, follow, land.
- **B, twin-led:** use it otherwise. It opens on the twin and shows a manual or yaw-only real clip.

Both run the same seven beats: the voice command, why it matters, how it works, the twin, training,
safety, close. The judges score impact (30%), use of Meta AI tooling (25%), technical depth (20%) and
the demo (25%), so every Meta model is named on screen, and new work is captioned.

**Rules for the edit**
- Only people who agreed to be filmed appear. Blur anyone else. No names on screen, and no "target":
  the person or thing followed is the "subject".
- Every number on screen comes from [docs/results.md](results.md). Do not round 2.9% to "never".
- Only say "the fine-tuned Glimmer answered live" over a clip whose mission card shows the
  `glimmer-30b+lora` badge. Otherwise say "the mission model".
- Show this caption at 0:50 and again at the close, for at least 4 s each:
  **"UI shell and X69 driver: turbodrone (Apache-2.0). Everything else built Nov 3-4."**
- Keep the narration under about 400 words (about 140 words a minute).

## Recordings to use

| ID | What | File | Status at 03:30 ET |
|---|---|---|---|
| T1 | Twin S1: dashboard, takeoff, held Space, "Drone, follow the person in the blue shirt.", transcript, Glimmer mission card, FIND, CONFIRM, FOLLOW with the SAM mask, spoken "land", landing; audio track of the commands and the MMS-TTS replies | `media/demo/demo-s1-<stamp>-ui.mp4` (1920x1080) | being recorded tonight by the demo workstream (`scripts/demo_twin.py`) |
| T2 | Twin S1 chase view of the same take | `media/demo/demo-s1-<stamp>-chase.mp4` (1280x720) | same |
| T3 | Twin S2: the red-shirt walker crosses in front | `media/demo/demo-s2-<stamp>-ui.mp4`, `-chase.mp4` | same (`--demo-scenario s2`) |
| T4 | Twin S3: "Drone, follow the green car." | `media/demo/demo-s3-<stamp>-ui.mp4`, `-chase.mp4` | same (`--demo-scenario s3`) |
| S1 | Stills for titles and the end card | `docs/img/demo_s1_*.png` (dashboard, voice mission, following, chase) | same (`--stills`) |
| C1 | Intent evaluation chart | [docs/eval/intent_eval.png](eval/intent_eval.png) | ready |
| C2 | SAM 3.1 on twin frames (white = truth, green = SAM) | [docs/img/percep/g2_examples.jpg](img/percep/g2_examples.jpg) | ready |
| C3 | Architecture graphic | redraw the README diagram in the editor | to make |
| R1 | Real flight, room camera (second phone): Space held, "follow the person in the blue shirt", the X69 turns and follows, "land", it lands | TBD (Session C, 09:30-11:30) | TBD |
| R2 | Real flight, screen capture of the operator UI for the same run | TBD, `media/` (Session C) | TBD |
| R3 | Real X69, manual or yaw-only clip through the governor (storyboard B) | TBD (Session B or C) | TBD |
| R4 | Props-off Session A: motors start on takeoff, E-stop cuts them | TBD (Session A) | TBD |

Fallback for the UI if no twin take exists: `media/ui_screens/ui_demo_mock.mp4` (36 s of the
dashboard on the mock backend). Caption it "UI on a mock backend".

## Storyboard A: real-flight-led

| # | Time | On screen | Narration | Captions | Source |
|---|---|---|---|---|---|
| A1 | 0:00-0:15 | Room camera. The operator holds Space and says "Follow the person in the blue shirt." The X69 turns to the subject and follows as they walk. Laptop speaker: "Following the person in blue shirt." | none: live sound only | "Real flight. Loiley X69 toy drone, $34." | R1 |
| A2 | 0:15-0:30 | The UI screen capture of the same moment: transcript, mission card and model badge, the SAM mask turning green in FOLLOW. | "That's a thirty-four dollar toy drone. I told it who to follow, and now it's flying itself. It's a follow-me camera you direct with your voice: for creators, coaches, and people with limited mobility." | "Voice → mission → SAM 3.1 → autopilot" | R2 |
| A3 | 0:30-0:50 | Twin UI: the operator says "follow John"; the reply card reads "I can't follow someone by identity. Tell me what they're wearing instead." | "It picks subjects by what they look like, never by who they are. Ask for someone by name, and it asks what they're wearing instead. And we only film people who agreed to it." | "Appearance only. Never identity. Consenting people only." | a twin take with that command (see the notes below), or T1 |
| A4 | 0:50-1:30 | Architecture graphic, each block lighting up in turn: Muse Voice Transcribe, Muse Glimmer 30B + LoRA, Muse Spark 1.3, SAM 3.1, the governor, Omnilingual ASR, MMS-TTS. Cut to the UI's voice panel and mission card while each is named. | "Meta models do the hearing, the understanding and the finding. Muse Voice Transcribe turns my words into text a quarter second after I stop. Our fine-tuned Muse Glimmer, racing Muse Spark 1.3, turns the text into a mission. SAM 3.1, on this Mac with MLX, finds the subject in the video from those words alone, and a small tracker follows it between detections. Then a plain controller steers, inside a safety governor that owns every radio packet. Omnilingual ASR listens on the laptop for 'stop' and 'land', even offline, and MMS-TTS talks back." | Each model's ID as it is named: `muse-voice-transcribe-1.0`, `Muse Glimmer 30B + LoRA (MLX)`, `muse-spark-1.3`, `sam-3.1 (MLX on device)`, `Omnilingual ASR CTC-300M`, `MMS-TTS`. At 0:50: "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else built Nov 3-4." | C3, T1 |
| A5 | 1:30-2:00 | Twin chase view beside the drone camera (T2), the subject in blue walking; then the S2 crossing (T3); then a two-second cut of the green car (T4). | "To test without crashing, we built a digital twin that speaks the drone's exact protocol: same packets in, same HEVC video out, half a second of video delay. Following the person in blue for a minute, it kept them in frame the whole time. When a walker in red crossed in front, it slipped onto them in under three percent of frames, then came back. It follows a green RC car, too." | "Wire-level twin: real X69 packets, HEVC/RTSP, 500 ms video latency" · "S1: in frame 100%, bearing error RMS 0.092 rad (5.3°)" · "S2: 49 of 1,706 frames on the red walker (2.9%)" · "S3: 'green car', following 100%" | T2, T3, T4 |
| A6 | 2:00-2:30 | The intent chart (C1), the action-accuracy bars highlighted; then the mission card showing `glimmer-30b+lora`. | "Fireworks blocked our training uploads overnight, so we fine-tuned Muse Glimmer 30B right here on the laptop with MLX, in seventy-seven minutes. On 300 commands it had never seen, our LoRA picks the right action 98 percent of the time, against 94 for base Glimmer and 95 for Muse Spark. On 80 hand-written commands: 100 percent. And it learned the policy: follow by clothing, refuse identity." | "Held-out test (300): LoRA 98.0%, base Glimmer 93.7%, Muse Spark 1.3 95.3%" · "Hand-written (80): 100%" · "Trained locally with MLX: 2,600 iterations, 77 min" | C1, T1 |
| A7 | 2:30-2:50 | The UI: Esc pressed, E-STOPPED pulses; the safety panel flags; then the room camera as "land" is spoken and the drone lands. | "Safety isn't a prompt. The models only choose a mission; they can't touch the sticks, take off, or clear an E-stop. The governor latches E-stop, keeps re-sending land, and goes neutral when video or perception goes stale. A review found fourteen problems; all are fixed and tested." | "Latched E-stop · re-asserted land · watchdogs · caps · human override" · "185 governor tests · 14 review findings fixed" | T1 (end), R1 (landing) |
| A8 | 2:50-3:00 | End card: still of the drone following (R1 frame or `docs/img/demo_s1_*`), the repo URL, the team. | "A thirty-four dollar toy, a laptop, and Meta's models: a camera operator you can talk to. The code is open." | Repo URL · "Built on Meta models: Muse Voice Transcribe, Muse Glimmer, Muse Spark, SAM 3.1, Omnilingual ASR, MMS-TTS" · "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else built Nov 3-4." | S1 or R1 |

## Storyboard B: twin-led

Same beats and narration from 0:50 on (B4-B8 = A4-A8), with T-clips in place of the R-clips.

| # | Time | On screen | Narration | Captions | Source |
|---|---|---|---|---|---|
| B1 | 0:00-0:15 | Twin S1 UI take: Space held, "Drone, follow the person in the blue shirt.", the transcript and the mission card fill in, the mask goes green; inset: the chase view as the drone turns to follow. | none: the take's own sound track (the spoken command and the MMS-TTS reply) | "Digital twin: the real X69 software, real packets, real models" | T1, T2 |
| B2 | 0:15-0:30 | The real X69: the manual or yaw-only clip (R3). If there is none, the props-off clip (R4). | "This is the same software flying a thirty-four dollar toy drone. You tell it who to follow, and it flies itself: a follow-me camera you direct with your voice, for creators, coaches, and people with limited mobility." | R3: "Real Loiley X69, yaw-only follow through the governor" or "manual flight through the governor". R4: "Real X69, props off: takeoff and E-stop flags" | R3 or R4 |
| B3 | 0:30-0:50 | As A3. | As A3. | As A3. | as A3 |
| B4-B8 | 0:50-3:00 | As A4-A8. In B7, use the twin for the landing ("land" heard by the local listener in T1), not R1. | As A4-A8. In B5, the twin footage is now the main evidence, so hold the S1 and S2 numbers on screen longer. | As A4-A8. | as A4-A8 |

In B, never call a twin shot a flight. If there is no real clip at all, say in B2: "The real drone
flies this build after its hardware checks; here it is in the twin." and caption the hardware status
honestly.

## Notes for the edit

- **The identity refusal shot (A3).** `scripts/demo_twin.py` speaks one preset command per take
  (`--command` overrides it). Record a short extra take with `--command "Drone, follow John."` and
  `--follow-s 5`, or use the mission card from the morning dry run. The reply text comes from
  `mhd.intent_spec.build_reply`, so it is exactly "I can't follow someone by identity. Tell me what
  they're wearing instead."
- **Model badge.** The mission card shows which model answered (`glimmer-30b+lora`,
  `muse-spark-1.3` or `keyword`) and its latency. Freeze-frame it for A6.
- **Latency on screen.** The UI's latency panel shows frame age, observation age, detector time and
  intent time; one second of it in A5 makes "half a second of video delay" concrete.
- **Sound.** Keep the take's own audio under the narration at about -18 dB in A1/B1, then duck it.
- **Length check.** The narration above is about 370 words. Read it aloud with a timer before
  recording the voice-over; cut A5's last sentence first if it runs long. Budget per beat at 140 words a minute: A2 15 s, A3 15 s, A4 41 s (read it a little faster), A5 31 s, A6 28 s, A7 20 s, A8 9 s; with the 15 s cold open that is about 2:54.
