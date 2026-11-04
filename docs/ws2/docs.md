# Workstream docs: the submission documents

Branch `ws2/docs`, from `integration` at `6bb13fb`. Written 2026-11-04, 03:05-05:30 ET.

## What it delivers

| File | What it is |
|---|---|
| `README.md` | For judges and anyone cloning: the one-line pitch, the follow-me-camera framing, the demo stills, how it works (steps, text diagram, process table), the Meta models with IDs and roles, the results, the safety design, how to run the twin and the real drone, new work vs. reused, licenses, acknowledgments, contributors with the AI assistance disclosed. |
| `docs/results.md` | Every overnight number in tables, with conditions and source paths: twin end to end (oracle tuning progression, feed-forward robustness, SAM S1-S3), perception gates G0/G1/G2, intent training and evaluation, voice G0 and push-to-talk acceptance, the latency budget, governor tests and the 14 review findings, test suites, spend. |
| `docs/video_script.md` | The 3-minute video: storyboard A (real-flight-led) and B (twin-led), shot by shot with timings, on-screen content, narration (368 words, timed per beat), captions, and the recordings to use. |
| `LICENSE` | `turbodrone/LICENSE` byte for byte (CRLF kept), with the appendix line set to "Copyright 2026 Russell Van Curen". |
| `NOTICE` | turbodrone's NOTICE block verbatim (byte-identical prefix), our attribution lines below it. |
| `THIRD_PARTY_NOTICES.md` | turbodrone and the upstream files we modified or deleted (Apache-2.0 4(b)); the Meta models' licenses; silero VAD, VitTrack and the CLIP tokenizer files SAM 3.1 MLX loads; mediamtx, FFmpeg, Chrome; the JS packages; all 105 installed Python packages with licenses read from their metadata. |
| `licenses/SAM_LICENSE.txt` | The SAM License, copied unchanged from the installed meta-sam-parser 0.1.0 wheel. |
| `docs/SUBMISSION.md` | The submission checklist: portal checks, drafts of every form field, the new-work statement, the secret-scan command, making the repo public, the video, submitting. |
| `scripts/secret_scan.py` | The pre-publication scan (stdlib only). |
| `tests/docs/test_secret_scan.py` | 4 tests for it on throwaway git repos. |

## How to run

```bash
cd <checkout> && source env.sh
$MHD_PY scripts/secret_scan.py                 # working tree + git log -p --all; RESULT: clean or FAIL
$MHD_PY scripts/secret_scan.py --no-history    # working tree only
$MHD_PY -m pytest -q tests/docs                # 4 tests, under a second
```

The scan reads the values in `$MHD_HOME/.env` (as the shell reads them: an inline `# comment` is not
a value) and looks for each one, in memory, in every tracked and untracked-but-not-ignored file and
in `git log -p --all --text` (every branch, diff and commit message). It never prints a value: a
hit reports only the variable name and the file or commit. It also flags credential patterns
(masked), `.env` files tracked now or in history, video or media/logs files, and files over 50 MB
(warning) or 100 MB (failure, GitHub's limit), including history blobs.

## Results

- `tests/docs`: **4 passed** (0.8 s): a planted key that was committed and then removed is found in
  history and never printed; a private value warns; a tracked `.env` fails; a token pattern is masked.
- `scripts/secret_scan.py` on this repository at 03:30 ET: **RESULT: clean, 0 failures**, 372 tracked
  files and 135 commits on all branches. 3 warnings, all one private (non-secret) `.env` value that
  also occurs inside local file paths quoted in `docs/research/` (and the planning commit). The
  first run reported a false failure: the Baseten line in `.env` is empty with an inline comment,
  and the scanner took the comment for the value; fixed and covered by a test.
- Every relative link in the new documents resolves, except `docs/RUNBOOK.md` (the ops workstream's,
  not merged yet) and the demo stills (see Gaps).
- Every number in README.md and docs/results.md was taken from, or recomputed from, the sources the
  brief allows: docs/ws/*.md, docs/eval/*, docs/benchmarks-perception.md, PLAN.md, docs/STATUS.md,
  `logs/e2e-*/` (e2e_result.json, report.json, rows.csv, telemetry.jsonl, obs.jsonl), `logs/ffcfg/`,
  `logs/spend.jsonl`, and git; plus, for results.md 1d and the README stills, the demo workstream's
  `logs/demo-<stamp>-<take>/demo_result.json` (its worktree) and committed `docs/img/demo_*.png`.
  Numbers recomputed from logs, and how:
  - S2's 49 red-walker frames come in three episodes about 21 s apart (11 frames at 21.7 s, five
    flickers of 1-4 frames at 41-43 s, 30 frames at 63-64 s): the `identity` column of
    `logs/e2e-20261104-030039-s2-sam_mlx/rows.csv`. 21 s is the scenario's crossing period (6 m at
    0.3 m/s plus a 1 s pause, `config/twin/scenarios/s2.yaml`).
  - Telemetry while following (S1): frame age p50 519 ms, observation age p50 535 ms, observation
    period 40 ms; SAM detection 241 / 247 ms p50 / p95 over 64 detections; confirmation age
    760 / 1,238 / 1,282 ms; VitTrack score p5 0.59 (from `telemetry.jsonl` and `obs.jsonl`).
  - Pitch wrong-way flags in the runs before the fixes: 412 of 547 (02:30), 322 of 546 (02:40) and
    463 of 684 (02:53) telemetry samples.
  - Spend: $2.52 from the 115 ledger entries (matches docs/STATUS.md).
- Test counts collected tonight (`pytest --collect-only`): gov 185, twin 65, train 38, ui 95 (89 plus
  6 opt-in), percep 61, agent 202 (200 plus 2 opt-in real-API tests). The documents use the passing
  counts in docs/STATUS.md (03:10): perception 60, one fewer than collected.

## Decisions (made without the owner)

1. **Name: "Voice Follow Drone".** The UI already uses it (`turbodrone/frontend/index.html`: "Voice
   Follow Drone · Operator"). Renaming touches README.md, NOTICE, docs/SUBMISSION.md and the UI title.
2. **The README says exactly what each run covers.** `scripts/e2e_twin.py` posts the follow Mission
   over HTTP; it does not speak, so the Results table says "a follow mission posted to the backend".
   The claim that the whole chain runs with voice rests on the demo workstream's S1 takes at 03:24
   and 03:36 (`logs/demo-<stamp>-s1/demo_result.json` in its worktree: Voice Transcribe, the Glimmer
   LoRA in about 4.0 s, FOLLOW, a spoken "land"), shown as the README's hero still (03:36 take) and
   in results.md 1d. The caption says the command was voiced by `say`.
3. **Newest numbers win.** Test counts come from docs/STATUS.md at 03:10 (governor 185, agent 200,
   perception 60), not the workstream docs (179, 199, 61).
4. **Results shown in the README:** the last three SAM runs (02:58-03:02), not the earlier S1 run
   (02:53) whose hover latch led to `6bb13fb`. results.md shows that run and why it was superseded.
5. **"Frames on the wrong subject"** is the README column for S2, stated as 49 of 1,706 (2.9%), "at
   the crossings", so nobody reads it as "never".
6. **Muse Wonder is not listed**: nothing in the training docs or code uses it. Nor Muse Image.
7. **Non-Meta models named plainly**: silero VAD and OpenCV VitTrack are small non-Meta models; the
   README does not claim "every model is Meta's". The video narration says "Meta models do the
   hearing, the understanding and the finding".
8. **LICENSE and NOTICE keep turbodrone's CRLF line endings**, so the copies are byte-exact except for
   the one copyright line.
9. **Apache-2.0 4(b) (notices in modified files)** is met with a central list in
   THIRD_PARTY_NOTICES.md and a statement in NOTICE, not per-file headers: the modified files belong
   to other workstreams and some are JSON, which cannot carry comments.
10. **Licenses I could not read locally are marked.** silero VAD (MIT) and VitTrack (Apache-2.0) come
    without license files; both rows say "not checked locally".
11. **The twin workstream's chase videos in `media/shared/twin/` are not in the video script**: a
    ground-truth demo pilot flew them, not the autopilot. The script uses the demo workstream's
    `media/demo/demo-<take>-<stamp>-*.mp4` takes instead.
12. **The identity-refusal shot** is a screen recording in the morning dry run, because
    `scripts/demo_twin.py` waits for FOLLOW.
13. **A secret-scan script instead of a one-line grep**, because the real risk is the exact key
    values in history, which a pattern grep cannot know, and a script can check them without ever
    printing them.
14. **`OWNER/REPO` and `VIDEO_LINK` are the only placeholders**, chosen so one grep finds them
    (docs/SUBMISSION.md step 3).

## Gaps

- **No real-drone numbers.** Sessions A-C happen this morning. README's status line, the results
  page and the submission description each say so and need one edit afterwards (SUBMISSION.md
  step 3 lists them).
- **The demo stills and video are the demo workstream's.** README.md shows
  `docs/img/demo_s1_2_voice_mission_follow.png`, `demo_s1_chase_follow.png` and
  `demo_s2_crossing_follow.png`, committed on ws2/demo (`3ee6018`, the final S1 take at 03:36). They
  resolve once ws2/demo is merged; if that workstream renames them, fix the three image links. The
  video itself (VIDEO_LINK) is cut by the owner from `media/demo/` and the morning's flights.
- **`docs/RUNBOOK.md`** (linked from README "Run it") is the ops workstream's and is not merged yet.
- **`scripts/demo_twin.py` and `scripts/run_stack.sh`** are documented in the README from their
  committed docstrings (ws2/demo `328f7d3`, ws2/ops `2616343`); I did not run them (they start
  servers, Chrome and the models, which this workstream may not).
- The Glimmer usage policy and the Meta API terms on training with model outputs were not read
  (no network use for docs); SUBMISSION.md step 3 asks the owner to.
- One upstream file, `turbodrone/frontend/src/lib/ws.ts`, does not show in
  `git diff 1aff2b5 HEAD` because it was restored and then deleted the same night; the notices list
  it by hand.

## For the integrator

- Merge `ws2/docs`. It adds files only and edits nothing another workstream owns: README.md,
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, licenses/SAM_LICENSE.txt, docs/results.md,
  docs/video_script.md, docs/SUBMISSION.md, docs/ws2/docs.md, scripts/secret_scan.py,
  tests/docs/test_secret_scan.py. No frozen file was touched.
- After merging `ws2/demo`: check `ls docs/img/demo_*` against the three README image links.
- After merging `ws2/ops`: the README's `docs/RUNBOOK.md` link resolves (it says "being written
  tonight" until then; drop those three words once it exists).
- `scripts/run_stack.sh --agent auto --glimmer local` passes `--intent auto`, so the router keeps
  Glimmer's answer only inside its 2.5 s window and Spark usually answers (the local LoRA takes about
  4 s). The README's twin example adds `--agent-extra "--intent glimmer"`; the demo workstream runs
  `intent: glimmer` with a 9 s window (`config/demo/agent.yaml`). Worth a default in run_stack.
- If anything merged tonight changes upstream turbodrone files, refresh the table in
  THIRD_PARTY_NOTICES.md: `git diff --name-status 1aff2b5 HEAD -- turbodrone/ | grep -v '^A'`.
- Optional: one line in docs/STATUS.md pointing the owner to docs/SUBMISSION.md.

Spend: $0. No paid API calls, no model loaded, no Chrome, no servers.
