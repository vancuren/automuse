# Submission checklist

**Deadline: Wednesday 2026-11-04, 21:00 ET. Late entries are not considered. Aim to be submitted by
19:00 ET** (PLAN.md section 7). Work through this top to bottom; every box is something to do or
check.

## 0. In the morning: confirm in the portal

- [ ] The exact form fields and limits (characters, file types). The drafts below assume the fields
      in the participant guide: name, tagline, description, models used, repo URL, video link,
      optional X post. Fix the drafts if the portal differs.
- [ ] Is a deployed or live product required? If yes, PLAN.md section 8 describes a static
      "twin replay" page (about an hour of work).
- [ ] Can the submission be edited after it is sent? If yes, submit early and update.
- [ ] Theme and track: the plan assumes **Push the Frontier** (robotics, controlling a physical
      system), Indie track, solo builder.
- [ ] Discord office hours, 10:00-11:00 ET: ask (1) whether building on an open-source code base
      (turbodrone) is fine for non-startups, and (2) for Fireworks dataset and model upload access.

## 1. Form fields (drafts)

**Name:** Voice Follow Drone

**Tagline:** A $34 toy drone that follows a person or a car you describe by voice, on Meta models.

**Description (about 170 words):**

> Voice Follow Drone turns a $34 Loiley X69 toy drone into a hands-free, voice-directed follow-me
> camera for creators, coaches, and people with limited mobility. Hold Space and say "follow the
> person in the blue shirt": Muse Voice Transcribe transcribes it, our Muse Glimmer 30B LoRA (trained
> on a MacBook with MLX) and Muse Spark 1.3 race to turn it into a mission, SAM 3.1 running on the
> Mac finds the subject in the drone's video, and a deterministic autopilot inside a safety governor
> steers. Omnilingual ASR listens on the laptop for "stop" and "land", even offline, and MMS-TTS
> talks back. The language models choose missions; they never touch the sticks. The governor
> latches E-stop, re-asserts land, and goes neutral when video or perception goes stale. A
> wire-level digital twin speaks the drone's exact protocol (UDP control packets, HEVC over RTSP),
> so the same build was tested end to end before it flew: it followed a walker for 60 s with the
> subject in frame the whole time. It follows by appearance only, never identity, and films only
> people who agreed.

Before pasting: if Session C produced a real follow, add one sentence about it; if not, keep "before
it flew" only if the drone has flown at all, otherwise write "before its first flight".

**Models used:**

| Model | How we use it |
|---|---|
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`, Meta Model API) | push-to-talk speech to text, realtime |
| Muse Glimmer 30B (open weights) + our LoRA | spoken command to mission JSON; fine-tuned and served on the Mac with MLX; base Glimmer also evaluated on a Fireworks H200 deployment |
| Muse Spark 1.3 (`muse-spark-1.3`, Meta Model API) | races Glimmer for the mission; reworded training data; evaluation baseline |
| SAM 3.1 (`sam-3.1` on the Meta Model API, and the open weights on the Mac with MLX) | finds the described subject in the drone's video |
| Omnilingual ASR CTC-300M (open weights) | always-on local "stop" / "land" listener, offline transcription |
| MMS-TTS (open weights) | spoken replies |

Bonus prize to mention: **Best Build on Open Weight Models**: the fine-tuned Muse Glimmer runs on
our own hardware (MLX on the MacBook), as do SAM 3.1, Omnilingual ASR and MMS-TTS.

**Repo URL:** `https://github.com/OWNER/REPO` (fill in after step 4).

**Video link:** the unlisted YouTube (or Vimeo) link from step 5.

**Contributors:** Russell Van Curen. AI assistance: the code, tests and documents were written with
Claude Code (Claude Opus 5.5); every commit carries a `Co-Authored-By: Claude` trailer.

**X post (optional; the Most Viral prize needs it linked on the form).** Post it before submitting.
Use the first version only if a real flight was recorded:

> I told a $34 toy drone "follow the person in the blue shirt" and it did. Muse Voice Transcribe →
> a Muse Glimmer we fine-tuned on a laptop → SAM 3.1 → an autopilot inside a safety governor.
> Built Nov 3-4 for the Meta Global AI Developer Hackathon. [video] [repo]

> A $34 toy drone you direct by voice: "follow the person in the blue shirt". Muse Voice Transcribe,
> a fine-tuned Muse Glimmer, SAM 3.1 and a safety governor, tested end to end in a wire-level
> digital twin. Built Nov 3-4 for the Meta Global AI Developer Hackathon. [video] [repo]

## 2. The "new work" statement (paste into the form)

> Built during the event (November 3-4, 2026): everything except turbodrone. We imported
> turbodrone (Apache-2.0, by Marshall Richards: the X69 Wi-Fi protocol driver, its FastAPI backend
> and React UI shell) unchanged from upstream commit 2c09cf8, in our commit 1aff2b5. Commit 09d005e
> (and part of 15a5f80) added 16 upstream files the import had missed, byte-identical. Every other
> commit is new work: the safety governor and autopilot, the backend patches, the wire-level
> digital twin, the SAM 3.1 perception service, the voice agent, the intent dataset and the Muse
> Glimmer LoRA, the operator UI's safety patches and new panels, and the tests. Diff:
> https://github.com/OWNER/REPO/compare/1aff2b5...main

The video carries the same caption: "UI shell and X69 driver: turbodrone (Apache-2.0). Everything
else built Nov 3-4."

## 3. Before the repo goes public

- [ ] Everything merged to `main` (the integrator's job), tests green.
- [ ] README: replace the status line with the real-flight result, add the video link
      (`VIDEO_LINK`), replace `OWNER/REPO`, and link the demo stills in `docs/img/`. Check with
      `grep -rn "OWNER/REPO\|VIDEO_LINK" README.md docs/SUBMISSION.md`.
- [ ] docs/results.md: add a "Real drone" section with the Session A-C numbers, or say plainly that
      it did not fly.
- [ ] If more turbodrone files changed tonight, refresh the list in THIRD_PARTY_NOTICES.md:
      `git diff --name-status 1aff2b5 HEAD -- turbodrone/ | grep -v '^A'`.
- [ ] **Secret scan** (must end with `RESULT: clean`):

  ```bash
  source env.sh && $MHD_PY scripts/secret_scan.py
  ```

  It looks for the exact `.env` values in every tracked and untracked file and in
  `git log -p --all` (every branch and commit message), plus common key patterns, tracked `.env`
  files, video files, and files over 100 MB. It never prints a secret. `WARN` lines are private
  values or large files to look at; `FAIL` lines must be fixed. If a secret is ever found in
  history, rotate the key at its provider first, then rewrite the history; a scrubbed commit is not
  enough once anything was pushed.
- [ ] No raw footage of people: `git ls-files | grep -iE '\.(mp4|mov|jpe?g|png|wav)$'` and open every
      image. Twin frames are fine (procedural people); real-flight frames only with consent.
- [ ] Read the Muse Glimmer usage policy (linked from its Hugging Face card) and the dev.meta.ai
      terms on training with model outputs (docs/ws/train.md, "Terms check").
- [ ] Check that `licenses/SAM_LICENSE.txt` matches the LICENSE on huggingface.co/facebook/sam3.1.
- [ ] Optional: publish the LoRA adapter so judges can run edge mode. `adapters.safetensors` is
      129 MB, over GitHub's 100 MB file limit: attach it to a GitHub release or upload it to
      Hugging Face, then link it in the README "Run it" section. It is a Muse Glimmer derivative:
      Apache-2.0 plus the usage policy.

## 4. Make the repo public

On the owner's Mac, from the main checkout, with the GitHub CLI logged in:

```bash
source env.sh
git switch main && git log --oneline -3          # the merged, final main
$MHD_PY scripts/secret_scan.py                   # RESULT: clean
gh repo create OWNER/REPO --private --source . --remote origin --description \
  "A \$34 toy drone that follows a person or a car you describe by voice, on Meta models"
git push -u origin main                           # main only; the ws/* branches stay local
```

Then, on github.com, still private:
- [ ] The README renders, both images load, and every link works (results, notices, docs).
- [ ] GitHub detects the license as Apache-2.0.
- [ ] Replace `OWNER/REPO` in README.md and here with the real path, commit, push.

Then flip it and check it logged out:

```bash
gh repo edit OWNER/REPO --visibility public --accept-visibility-change-consequences
```

- [ ] Open the repo in a private browser window: README, images, the compare link.

## 5. Video

- [ ] Cut it from [docs/video_script.md](video_script.md): storyboard A if the 13:00 hardware gate
      passed, otherwise B. 3:00 or less.
- [ ] Captions on, including "UI shell and X69 driver: turbodrone (Apache-2.0). Everything else
      built Nov 3-4."
- [ ] Upload as unlisted (or public). Description: the tagline, the repo URL, the models list.
- [ ] Play it in a logged-out window, start to end.

## 6. Submit

- [ ] X post first, if going for Most Viral.
- [ ] Every form field filled; the repo is public; the video plays logged out; contributors and the
      AI-assistance disclosure are listed; the new-work statement is in.
- [ ] Submitted by 19:00 ET. Save a screenshot of the confirmation in `media/` (not in git).
