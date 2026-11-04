# Workstream percep: the perception service

Branch `ws/percep`. Everything lives in `src/mhd_ml/perception/` (ml venv, `$MHD_ML_PY`) and
`tests/percep/`. Measurements: `docs/benchmarks-perception.md`.

## What it does

The service turns words into evidence. A FindRequest ("person in blue shirt") comes in. SAM 3.1
finds the subject, VitTrack follows it at camera rate, and a Kalman filter keyed on capture time
smooths it. One `SubjectObs` per frame goes back to the backend. Perception never decides what
the drone does; the autopilot reads the evidence (`state`, `box`, `tracker_score`,
`confirm_age_ms`) and owns every state change.

| Module | Role |
|---|---|
| `detectors/sam_mlx.py` | SAM 3.1 on this Mac (mlx-vlm 0.7.11). `SamMlxEngine` (detect, ViT backbone, multiplex tracker); `SamMlxDetector.detect(image_bgr, phrase)`. Default input 672 (234 ms idle). |
| `detectors/sam_api.py` | hosted `sam-3.1` on `POST https://api.meta.ai/v1/responses`. 640x360 JPEG data URL, `include_confidence`, `one_bit` masks. A hard 2.5 s total deadline, no retries, a phrase that got HTTP 400 is never re-sent, 401/403 disables it, 429 backs off 5 s, cache by frame hash, $8 workstream cap (`spend.py`, ledger `$MHD_HOME/logs/spend.jsonl`). A key-based record reader replaces meta_sam_parser's slow full validation in the hot path; both give identical records on all 40 recorded responses. Only the 3 best masks are decoded. |
| `detectors/fake.py` | color-blob test double (no GPU or network); its observations say `source: "oracle"`. |
| `trackers.py` | `make_tracker(name)` -> `init(img, box)` / `update(img) -> (box, score)` over VitTrack, KCF, CSRT, and the SAM 3.1 MLX tracker. Boxes are in 1280x720 px; images are any size (640x360 in the service). KCF/CSRT score = ok x color-histogram similarity to the init patch. |
| `fusion.py` | `BoxKalman` (constant velocity on bearing, ln width, elevation; keyed on `capture_ms`; refuses older measurements), color score (hue of the upper-body mask region, 12 color words plus aliases), `hue_hist` + `associate` (IoU plus color), `FrameRing` (2 s of 640x360 frames), `reseed` (replay a fresh tracker from the detection frame to now), counter-block masking (`COUNTER_ROI` from `mhd.twin.counter` when present, else bottom-right 160x160). |
| `pipeline.py` | `PerceptionCore`: mission state, detector cadence, ranking (confidence x color x size prior, min score 0.4), look-alike ties -> `Candidates`, `Select`, association, re-seed with the 0.3-box-size offset check, loss, SubjectObs. Synchronous and lock-protected; shared by service and replay. |
| `service.py` | the process: `/ws/frames` pull loop (newest only, one in flight) and `/ws/subject` (FindRequest / Release / Select in; SubjectObs / Candidates out). Asyncio for I/O, one thread for frames + tracker, one thread for SAM. |
| `replay.py` | the same pipeline over a video or a frame folder: `obs.jsonl`, `annotated.mp4` (H.264), `summary.json` (time to confirm, lock rate, switches). Detector latency is replayed in a synthetic clock. |
| `fake_backend.py` | a stand-in backend on ports 8300-8309 (synthetic scene, image folder or video), for tests and for running perception alone. |
| `bench.py`, `bench_api.py`, `bench_accuracy.py`, `benchutil.py` | G1 speed, G0 for SAM, the G1 accuracy proxy. |

## Run it

Always `cd $MHD_TREE && source env.sh` first.

```bash
# The service (SAM 3.1 MLX detector, VitTrack tracker): the default for tomorrow.
$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 --detector sam_mlx --tracker auto

# The same with the hosted SAM 3.1 (costs $0.0025 per detection, about $3.60 per hour at 1 Hz).
$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8000 --detector sam_api --tracker auto

# Useful flags: --sam-res 672 (504 is faster, 1008 slower), --find-hz 2 --follow-hz 1,
# --min-score 0.4, --hfov-deg 80 (set the Session A value), --mask-counter auto|on|off,
# --obs-log logs/percep_obs.jsonl (everything sent), --log-level INFO.

# Perception alone, against the fake backend (port 8300) serving a frame folder or video:
$MHD_ML_PY -m mhd_ml.perception.fake_backend --port 8300 --source <folder|video|synthetic> \
    --fps 15 --find "person in blue shirt" --color blue --record logs/fake_rx.jsonl &
$MHD_ML_PY -m mhd_ml.perception.service --backend ws://127.0.0.1:8300 --detector sam_mlx

# Replay a clip (Session A footage, twin frames): JSONL + annotated MP4 + summary.
$MHD_ML_PY -m mhd_ml.perception.replay --input clip.mp4 --phrase "person in blue shirt" \
    --color blue --category person --detector sam_mlx --out logs/replay/clip [--truth truth.jsonl]

# Benchmarks and tests.
$MHD_ML_PY -m mhd_ml.perception.bench --images media/bench/coco      # G1 speed (GPU)
$MHD_ML_PY -m mhd_ml.perception.bench_api --n640 30 --n1280 10        # G0 for SAM ($0.10)
$MHD_ML_PY -m mhd_ml.perception.bench_accuracy                        # look-alike accuracy (GPU)
$MHD_ML_PY -m pytest -q tests/percep                                  # offline, no GPU, about 10 s
```

The service needs nothing beyond the ml venv as installed. The SAM weights are in
`$MHD_MODELS/sam3.1-bf16`, and the CLIP tokenizer is in the project HF cache, so the hub is
switched to offline mode automatically. The VitTrack ONNX is in `$MHD_MODELS/opencv`.

## Results (details and conditions in docs/benchmarks-perception.md)

- **G1 speed.** SAM 3.1 MLX detection p50 is 581 ms at 1008, 234 ms at 672, 172 ms at 504, idle.
  With Glimmer training and the twin renderer on the GPU: 698 ms at 672 and 1406 ms at 1008.
  The SAM 3.1 MLX tracker takes about 1 s per step (backbone 431 + propagate 544 ms, idle), so it
  is **no-go in the loop**. VitTrack takes 2.6 ms (2.8 ms under load), KCF 26 ms, CSRT 14 ms. All
  three hold 5 Hz. `--tracker auto` = VitTrack.
- **G1 accuracy proxy.** Two look-alike people that differ only in jacket color: right subject
  24/24 at 672 and 1008. "green car": 4/5. Session A clips are still needed for the real gate.
- **G0 for SAM.** API round trip p50 697 ms, p95 1277 ms at 640x360 (0 errors in 30 calls).
  1280x720 takes the same round trip but 3x the parse time, so 640x360 is the size to send.
  meta_sam_parser parse p50 481 ms (pure-Python mask decoding); the service path takes 74 ms.
  **Go** at 1 Hz.
- **End to end (fake backend, SAM MLX, under GPU load).** SubjectObs at 15.0 Hz (camera rate),
  about 10 ms per frame inside perception, confirmed 1.4 s after FindRequest, 99.9% of frames
  on the right subject, 0 identity switches in 47 s, detector p50 / p95 625 / 695 ms.
  Same with `sam_api`: 15 Hz, 97.5% locked, confirmed after 2.1 s, API p50 645 ms.
- **G2 (twin frames, full HEVC path).** S1 "person in blue shirt" recall 100% (60/60, gate 90%).
  S2: blue 98%, red 83% (100% when at least half visible). S3: "green car" 100%. Every actor that
  is at least half visible is found in every frame. The hosted sam-3.1 matches on a third of the
  frames. Pipeline replays of the twin frames (only 3.1 fps): 0 identity switches, including the
  S2 crossing (92.6% locked); S3 green car 100% locked.
- **The color gate.** With the red walker off screen, "person in red shirt" boxes the blue
  walker (score 0.47-0.72). Candidates whose shirt region matches the requested color in less
  than 8% of its pixels are now dropped: 34 false alarms -> 0, no true detection lost.
- **Tests.** `tests/percep`: 61 passed in about 11 s (offline: no GPU, network, or spend). Fusion: Kalman timing, color score, association,
  re-seed, counter block. Pipeline: states, ties and Select, the color tie-break, loss, cadence,
  re-seed rejection, and the safety fixes. SAM API client: parsing, no retry on 400, cache,
  spend cap, deadline. WebSocket integration with the fake backend. Replay.
- **Spend:** $0.46 of the $8 cap (185 sam-3.1 images).

## Decisions made overnight (nobody to ask)

1. VitTrack is the in-loop tracker. The SAM 3.1 MLX tracker is wired up (`--tracker sam_mlx`,
   correct boxes, IoU 0.98) but is about 1 s per step, so it is for offline replays only.
2. The SAM MLX detector runs at a 672 input by default, not 1008: 2.5x faster, same look-alike
   accuracy, and 2 Hz fits.
3. The weights run as stored (F32). bf16 was no faster, and fp16 found nothing.
4. The API detector sends 640x360. For the hot path I wrote a small record reader that follows
   meta_sam_parser's grammar, parses by key, and uses the same package's mask decoder. Tests
   compare it against meta_sam_parser on recorded responses.
5. Color gate: with a color word in the FindRequest, a candidate whose upper-body mask region
   matches that color in less than 8% of its pixels never becomes a track (`min_color_frac`).
   Found on twin frames (G2), where it removed 34 of 34 wrong-person detections.
6. `SubjectObs.state` semantics: see the table below. A track needs 2 consistent detector
   confirmations to be "confirmed". A Select counts as confirmation.
7. Look-alikes: if the second-best rank is >= 0.85 x the best, and the two boxes are distinct,
   perception sends `Candidates`. They are numbered 1..N **left to right**, and the numbering
   stays fixed until the set is re-sent (every 2 s at most). Perception then reports "candidate"
   with no box until a `Select`. It never picks for the user.
8. Detector cadence: 2 Hz while searching, candidate, lost, or in doubt (tracker score < 0.5, a
   rejected re-seed, or a detection that missed the subject). While following: at most 1 Hz, and
   faster when the detector is slow, so that `confirm_age_ms` stays under about 1.3 s
   (period = 1300 ms - detector p95, never faster than 2 Hz). Under GPU load this keeps every
   frame under the autopilot's 1.5 s forward-motion gate (measured: max 1467 ms).
9. The counter block is masked only on frames whose `flight_mode` is "twin" (`--mask-counter auto`).
10. `confirm_age_ms` is measured from the capture time of the frame the detector saw, not from
   when its result arrived. Detector latency therefore counts as age, as it should.
11. Fixed after review, because a drone is involved:
    - a late Select re-identifies the subject in a fresh detection instead of trusting a stale box;
    - a repeated identical FindRequest does not reset the track;
    - a backend restart (frame ids going backwards) resets the buffer and marks the track lost;
    - a detection older than 300 ms with no frame left to replay from never starts a track;
    - a subject leaving the frame edge becomes "lost" rather than a zero-width box;
    - errors in the pipeline are logged and never drop the connection.
12. Fixed after an independent code review (00:45 ET; regression tests in `test_review_fixes.py`):
    - an API request refused by the in-flight limit is no longer billed;
    - the spend cap re-reads the ledger on every check (concurrent processes share one budget)
      and flushes at exit;
    - a result computed for an older phrase under the same mission_id is discarded (mission
      generation counter);
    - a Select whose track cannot start falls back to re-identification instead of being lost;
    - a repeated frame id (for example the newest frame re-sent after a reconnect) is skipped, not
      treated as a backend restart;
    - the SAM tracker restores the detector's input size even when MLX raises;
    - the follow cadence uses the measured submit-to-fused cycle, not detector time alone;
    - re-seed replays are capped per tracker (VitTrack 30 steps, CSRT 12, KCF 8), so a replay
      holds the frame thread for about 0.2 s at most;
    - `counter_roi()` is resolved once;
    - one detector and tracker builder is shared by service and replay; trackers that prompt
      (SAM) get the mission phrase.

## What the integrator must wire up

1. **Backend `/ws/frames`** as in docs/interfaces.md. Perception sends `NextFrame` and expects
   exactly one binary frame back, the newest with `frame_id > after_frame_id`, waiting if there
   is none yet. `capture_ms` must be the capture estimate (from the RTSP PTS on the real drone),
   and `flight_mode` must be set. Frames that are not 1280x720 are resized.
2. **Backend `/ws/subject`.** Send `FindRequest` when a follow mission starts (subject_phrase,
   category, and color from the Intent). Re-sending the same request is harmless. Send `Release`
   on hover, land, or a new mission, and relay `Select` from the UI or agent. Relay `Candidates`
   to the UI and the agent; it carries numbered boxes for "I see two people in blue. Left or right?".
   Perception sends nothing while there is no mission.
3. **Autopilot reading SubjectObs:**

   | state | box | meaning |
   |---|---|---|
   | searching | None | mission active, no subject found yet |
   | candidate | set | one detector confirmation so far (wait for "confirmed") |
   | candidate | None | look-alikes tie; Candidates sent; waiting for Select |
   | confirmed | set | 2 or more consistent detector confirmations |
   | lost | None | the tracker failed 3 frames in a row or the subject left the frame |

   - `tracker_score` comes from VitTrack (0..1). It is 0 on a frame where the tracker missed and
     the box is the Kalman prediction. Below 0.5 is "doubt".
   - `track_id` changes on every new track. A re-find always gets a new id, so it goes through CONFIRM.
   - `confirm_age_ms` stays under about 1.3-1.5 s while following: measured max 1467 ms with the
     GPU shared with Glimmer training. The cadence adapts to detector latency, so the 1.5 s
     forward-motion gate holds.
   - Measured against the gov autopilot's thresholds (config/autopilot.yaml in ws/gov):
     - VitTrack `tracker_score` p5 / p50 is 0.54 / 0.71 on the live composite run and 0.52 / 0.63
       on twin S1. It is under `follow_min_score` 0.4 in 0-4% of frames (14% for the close, fast
       red walker in S2 at 3.1 fps).
     - Frame-to-frame width ratio is at most 1.26 (limit 1.6); center jumps are at most 32 px
       (limit 320).
   - Detector p95 for the autopilot's timing formulas: about 250 ms (MLX 672, idle), 700 ms (MLX
     672 under load), 1.3 s (API p95 round trip; 2.5 s deadline).
4. **HFOV:** pass `--hfov-deg` (or set `MHD_HFOV_DEG`) to the Session A value. Perception only
   uses it inside its Kalman filter, so a wrong value costs a little smoothing, not correctness.
5. **COUNTER_ROI:** picked up from `mhd.twin.counter` automatically once the twin branch is merged
   (it is `(1008, 640, 1280, 720)` there). Until then the fallback is the bottom-right 160x160,
   which does not cover the twin's whole block; set `MHD_COUNTER_ROI=1008,640,1280,720` for
   unmerged runs.
6. **GPU etiquette:** only perception loads SAM: 3.5-5.6 GB of MLX memory at 672 (the status line
   logs it), or 11.7 GB with the SAM tracker at 1008.
   At 2 Hz it keeps about half of the idle GPU busy, and it slows down 2.5-3x next to Glimmer
   training.
7. **Contracts:** no changes needed. Suggestion: an optional `bearing_rate` (rad/s, from the
   Kalman filter) on `SubjectObs` would give the autopilot its feed-forward term without
   differentiating boxes. Perception already estimates it (`BoxKalman.bearing_rate`).

## Known gaps

- **API rate limits:** the team key returned HTTP 429 at only 1.7 calls/s while other
  workstreams were using it. A 429 makes the API detector back off for 5 s, and the autopilot
  then hovers. Keep batch jobs off the key during API-detector flights, or fly `sam_mlx`.
- No real X69 or webcam footage yet. The G1 accuracy gate on Session A clips is still open
  (`replay.py` is ready for it).
- The SAM accuracy proxy uses pasted look-alikes on COCO backgrounds, not drone-view footage.
  Expect lower scores from far away, from above, and in motion blur.
- Spark-vision tie-breaking (numbered boxes to muse-spark-1.3) belongs to the agent: perception
  only emits `Candidates`.
- `--tracker sam_mlx` in the live service runs frames on the detector thread at about 1 frame
  per second. Do not use it for flying.
- KCF and CSRT have no native confidence. Their score is a color-histogram similarity, which is
  weaker evidence than VitTrack's.
