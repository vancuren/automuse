# Workstream: train (Muse Glimmer intent LoRA)

Branch `ws/train`. Overnight, 2026-11-03 23:35 to 2026-11-04 (times ET). PLAN.md section 4.7.

## TL;DR

- **Data**: `training/generate.py` builds 2,200 train / 150 valid / 300 test examples from template families
  with labels taken from the slots, never from a model. ASR-style noise and 458 Muse Spark 1.3 paraphrases
  (wording only) are added. `training/data/gold.jsonl` holds 80 hand-written, hand-labeled commands.
- **Fireworks is blocked for custom weights on this account tonight.** Every upload path returns code 9:
  `dataset uploads are temporarily disabled, please contact Fireworks AI to get access` (datasets) and
  `model creation, uploads and preparation are temporarily disabled, please contact Fireworks AI to get
  access` (models). So no SFT job could start. Base-Glimmer deployments do work: I created one H200
  deployment, evaluated base Glimmer on it, and deleted it after 8.3 GPU-minutes (about $1.10).
- **The LoRA was trained locally with MLX** (QLoRA on `Muse-Glimmer-30B-4bit`, last 16 layers, rank 16),
  saved to `$MHD_MODELS/adapters/glimmer-intent-v1/`, and evaluated locally. Results: see
  [Results](#results) and `docs/eval/summary.md`.
- **Edge mode is ready**: `scripts/serve_glimmer.sh` serves Glimmer + LoRA on `http://127.0.0.1:8080/v1`.
- **Cloud mode**: once Fireworks enables uploads, run `training/fireworks.py train` and then
  `training/fireworks.py morning` (about 10 minutes plus training time). Until then cloud mode can only
  serve base Glimmer on Fireworks (`training/fireworks.py deploy --plain`).

## What I built

| Path | What |
|---|---|
| `training/generate.py` | Template and slot generator, ASR noise, family-level splits, Spark paraphrases (`--paraphrase`), chat files for mlx_lm and Fireworks (`--rebuild` re-derives them from the committed records, with no network) |
| `training/data/{train,valid,test}.jsonl` | Records with metadata (family, source, clean text, transcript, intent). Committed. |
| `training/data/gold.jsonl` | 80 hand-written commands with hand labels. Never trained on. |
| `training/data/mlx/`, `training/data/fireworks/` | Derived chat JSONL (gitignored; `--rebuild` recreates them byte-identically) |
| `training/fireworks.py` | Fireworks REST client and CLI: upload, sft, job, deploy, load-lora, warmup, morning, down, delete, list, train |
| `training/train_mlx.sh`, `training/train_mlx.py`, `training/mlx_lora*.yaml` | Local QLoRA training (smoke and full) with recorded wall time and peak memory |
| `training/eval.py`, `training/eval_mlx.sh` | Evaluation (Spark, Fireworks, mlx_lm backends) and report: `docs/eval/results.json`, `summary.md`, `intent_eval.png` |
| `scripts/serve_glimmer.sh` | Edge-mode server: mlx_lm.server on port 8080 with the adapter |
| `src/mhd_ml/glimmer_client.py` | Small client the agent can call for edge (`GlimmerClient.edge()`) or cloud (`GlimmerClient.cloud()`) |
| `tests/train/` | 39 pytest tests (generator, eval metrics, Fireworks client with mocked HTTP, Glimmer client) |

## Data

### Label conventions (the policy the LoRA learns)
- People are always `person` in `subject_phrase`, described by clothing only: `person in blue shirt`,
  `person with red backpack`, `person in blue` (color only). No gender or age words: the drone never
  judges people, only clothes. Vehicles: `<color> <type>` (`green toy car`, `white pickup truck`). Animals and
  objects: `<color> <noun>`. No color gives just the noun and `color: none`.
- When several garments are mentioned, only the first goes in the label (at most two attributes, at most
  four words).
- `distance` is medium unless asked. Meters map to close (2 or less), medium (3 to 5), far (6 or more).
- **clarify / ambiguous**: pronouns or pointing with no description ("follow them", "go after that"), a
  missing subject ("follow uh"), "someone", **"follow me"** (the drone asks what you are wearing), a color with
  no noun ("the red one"), or two subjects at once.
- **refuse / identity_request**: a person by name, relationship ("my neighbor"), role, or fame, or any
  request to identify or recognize someone ("who is the guy in the red hat").
  **A name plus a clothing description is followed by the description only**: "follow Dave, he's in the green
  hoodie" gives follow `person in green hoodie`. Possessive animals are fine ("follow my dog" gives `dog`).
- **refuse / unsafe**: surveillance (everyone, a house, a window, following someone home or covertly) and
  harm (fly into, hit, scare). `unsafe` wins over `identity_request`.
- **refuse / unsupported**: take off (button only, as in PLAN), flips, flying somewhere, altitude changes,
  photos, and non-drone requests.

### Construction
- 77 template families in 11 groups. Whole families are held out: 55 train, 11 valid, 11 test
  (`training/data/stats.json`). Test therefore measures unseen phrasings, not memorized templates.
- Slots: 11 colors (plus silver for vehicles, and shades like "light blue" that the label drops), 17
  garments and accessories, 16 people nouns, 10 vehicle types, 12 animals, 10 objects, 36 names, 20
  relationships.
- ASR noise, applied per example: phonetic misspellings (`metrs`, `dron`, `blew`, `lend`, `grean`), dropped
  function words, duplicated words, fillers (`um`, `uh`, `like`), and wake words (`hey drone`, `hey dron`).
  55% Omnilingual style (lowercase, no punctuation) and 45% Voice Transcribe style (capitalized, punctuated).
- Paraphrases: Muse Spark 1.3 (Meta API, `reasoning_effort: minimal`, strict JSON schema) rewrote clean seed
  sentences 10 per call. Labels are kept from the seed. A filter drops any paraphrase that loses or adds a
  color, garment, vehicle or animal word, a name, or a distance request. Kept: 379/450 train, 26/30 valid,
  53/60 test. Each paraphrase batch draws from one split's families only. Cost about $0.12.
- Mix (train): follow 1,250, clarify 308, refuse 237 (identity 113, unsafe 88, unsupported 36), land 204,
  hover 201. Test: follow 169, clarify 40, refuse 40, hover 27, land 24. Gold: follow 44, refuse 11,
  clarify 9, hover 8, land 8.
- Exact-transcript dedup across splits and against gold (tested).

### Chat format and the reasoning field
Rows are `{"messages":[system GLIMMER_SYSTEM_PROMPT, user transcript, assistant]}`, and the assistant message
carries `content` (compact Intent JSON in schema order) plus a one-line `reasoning_content`
("Follow by appearance: person in blue shirt.", "Identity-based request; I follow by appearance only, so
refuse.").

Why a reasoning field: Glimmer's chat template renders `reasoning_content` as an
`<|start|>assistant to=self<|message|>...<|eom|>` message before the `to=user` answer, and Meta warns that
answer-only SFT can suppress the reasoning channel. One short line keeps the channel format intact while
adding only about 10 tokens. A completion is 39 to 55 tokens, and a row is 320 to 355 tokens. Fireworks SFT
accepts `reasoning_content` (also `thinking` / `reasoning`) on assistant messages, so both formats use the
same rows.

Note: the template always appends `Reasoning strength: high.` to the system message unless
`reasoning_strength` is passed. `GLIMMER_SYSTEM_PROMPT` starts with `Reasoning strength: low`. mlx_lm's
trainer passes no template kwargs, so training and serving both use the template defaults. **Do not pass
`chat_template_kwargs` to the LoRA server**, or the prompt will differ from training.

Terms check: I searched for the Meta Model API terms on training with outputs. No public page I could find
forbids it, and a third-party tracker of the Llama API terms says the October 2026 terms do not use hosted
content for Meta training. Spark only rewrote wording here, the labels are ours, and the student is a Meta
model. The owner should still read the live dev.meta.ai terms before the repo goes public.

## Fireworks (primary path in the plan, blocked tonight)

What I verified with the REST API (key from env, never printed):
- Account READY. Quotas: H100 16, H200 16, B200 16, GB300 0. `muse-glimmer-30b` is `supervisedLoraTunable`,
  Training V2, BF16.
- `POST datasets/{id}:upload` gives 403 and `:getUploadEndpoint` gives 400, both with
  `{"code": 9, "message": "dataset uploads are temporarily disabled, please contact Fireworks AI to get access"}`.
- `POST models` (HF_PEFT_ADDON) gives 400 `model creation, uploads and preparation are temporarily disabled,
  please contact Fireworks AI to get access`. So uploading our MLX adapter is blocked too.
- The Serverless Training API needs the `fireworks` Python SDK (`fireworks.training.sdk`), which is not
  installed. Installing is outside the rules, so I did not try it.
- Multi-LoRA shapes for Glimmer exist only on GB300 (quota 0) and a B200 FP4 "Decision Engine" shape. The
  plan therefore uses the **H200 BF16 shape `accounts/fireworks/deploymentShapes/dkuwotf7` with
  `enableHotReloadLatestAddon`**: base Glimmer first, then one LoRA hot-loaded and merged
  (`POST deployedModels?replaceMergedAddon=true`). Fireworks requires min = max replicas for hot reload.
  This config passes `validateOnly`.
- I ran one plain H200 deployment `glimmer-intent-eval` (min 0, max 1, 10-minute scale-to-zero as a safety
  net). Create to READY with 1 replica took about 4 minutes (00:04:37 to 00:08:43). I evaluated base Glimmer,
  with and without a strict JSON schema, then **deleted it** at 00:12:53: 8.3 GPU-minutes, about $1.10,
  logged in `$MHD_HOME/logs/spend.jsonl`. No deployment is left on the account (`fireworks.py list`).

### Morning runbook (demo window about 10:15 to 12:45)
Ask Fireworks (Discord or support) to enable dataset and model uploads for the account first. Then:
```bash
source env.sh
$MHD_PY training/generate.py --rebuild            # recreate training/data/fireworks/*.jsonl
$MHD_PY training/fireworks.py train               # upload train/valid, SFT rank 16 x 2 epochs (falls back to rank 8 x 1), waits
$MHD_PY training/fireworks.py morning --lora glimmer-intent-v1   # H200 hot-reload deployment + LoRA + warm-up; prints the model string
# ... demo ...
$MHD_PY training/fireworks.py delete --deployment-id glimmer-intent   # by 12:45; stops billing ($8/hour)
```
If uploads stay blocked, serve base Glimmer in the cloud and the LoRA on the Mac:
```bash
$MHD_PY training/fireworks.py deploy --deployment-id glimmer-intent --plain   # base, scales to zero after 10 idle minutes
scripts/serve_glimmer.sh                                                      # LoRA locally on :8080
$MHD_PY training/fireworks.py delete --deployment-id glimmer-intent
```
Allow about 5 minutes from create to READY (measured tonight), so start by 10:05.

## Local MLX path (where the LoRA was actually trained)

- Model: `$MHD_MODELS/Muse-Glimmer-30B-4bit` (mlx-community, text tower; vision weights are dropped by
  mlx_lm). Download complete at 23:40:51. GPU lock `g1-bench.done` was present at 00:01.
- Smoke test (00:02, nothing else heavy running, 87% memory free): 20 iterations, batch 1, grad accumulation
  4, rank 16, scale 2.0, dropout 0.05, lr 1e-4, last 16 layers, `--mask-prompt`, `--grad-checkpoint`,
  max-seq 512. Val loss went 1.428 to 0.434. About 1.45 s per iteration, about 31 trained (completion)
  tokens/s, about 230 processed tokens/s. 32.2M trainable parameters (0.116%). Inference with the smoke
  adapter: 15.7 GB active, 16.6 GB peak, about 12 to 14 tok/s decode.
- Full run: `training/train_mlx.sh full`. 2,600 iterations (about 1.2 epochs), 650 optimizer updates,
  25-step warmup, then cosine decay from 1e-4 to 1e-5. Validation every 250 iterations (25 batches),
  checkpoints every 500. RESULTS_TRAINING

## Results

RESULTS_TABLE

## Exact commands

```bash
source env.sh
# data
$MHD_PY training/generate.py --paraphrase          # templates + Spark paraphrases (about $0.12); writes training/data/
$MHD_PY training/generate.py --rebuild             # or: re-derive the chat files from the committed records (no network)
$MHD_ML_PY training/gold_asr.py                    # gold spoken by `say` + Omnilingual ASR -> training/data/gold_asr.jsonl
$MHD_PY -m pytest -q tests/train                   # tests
# local training (one Glimmer process at a time)
training/train_mlx.sh smoke                        # 20 iterations, about 1 minute
training/train_mlx.sh full                         # 2,600 iterations, about 80 minutes on the M5 Pro -> $MHD_MODELS/adapters/glimmer-intent-v1
# evaluation
training/run_local_evals.sh                        # waits for training, then LoRA and base on :8580 (quality + sequential latency)
$MHD_PY training/eval.py run --backend spark --label spark-1.3 --sets test,gold,gold_asr --concurrency 6
$MHD_PY training/eval.py run --backend fireworks --label glimmer-base-fireworks \
    --model 'accounts/fireworks/models/muse-glimmer-30b#accounts/$FIREWORKS_ACCOUNT_ID/deployments/glimmer-intent-eval' \
    --sets test,gold --concurrency 12 --max-tokens 1500   # needs a live deployment (fireworks.py deploy --plain)
$MHD_PY training/eval.py report                    # docs/eval/{results.json,summary.md,intent_eval.png}
# serving
scripts/serve_glimmer.sh                           # edge mode on :8080 (use --port 8580 for tests)
```

## What the integrator must wire up

- **Agent, edge mode**: start `scripts/serve_glimmer.sh` (port 8080; refuses to start if another Glimmer
  process is running). Call `POST http://127.0.0.1:8080/v1/chat/completions` with
  `[{"role":"system","content":GLIMMER_SYSTEM_PROMPT},{"role":"user","content":transcript}]`,
  `"model": "default_model"` (any other name makes mlx_lm.server try to load that path), `temperature 0`,
  `max_tokens` about 200, and no `chat_template_kwargs`. **mlx_lm 0.32.2 does not split Glimmer's
  channels**: its Muse response format is keyed to the Muse Spark model type, so `content` arrives raw as
  `to=self<|message|>Follow by appearance: ...<|eom|><|start|>assistant to=user<|message|>{json}`. Take
  the text after the last `<|message|>` and parse it with `mhd.intent_spec.parse_intent`.
  `mhd_ml.glimmer_client.split_channels` does this, and `GlimmerClient.edge().intent(text)` wraps it all.
  `parse_intent` on the whole string happens to work for the LoRA (its one-line reasoning has no braces),
  but not reliably for base Glimmer.
  `Mission.model = "glimmer-30b+lora"`.
- **Agent, cloud mode**: `POST https://api.fireworks.ai/inference/v1/chat/completions`, Bearer
  `FIREWORKS_API_KEY`, model
  `accounts/$FIREWORKS_ACCOUNT_ID/models/glimmer-intent-v1#accounts/$FIREWORKS_ACCOUNT_ID/deployments/glimmer-intent`
  once the LoRA exists on Fireworks. Until then use base:
  `accounts/fireworks/models/muse-glimmer-30b#accounts/$FIREWORKS_ACCOUNT_ID/deployments/glimmer-intent`
  with `max_tokens` about 1200 (base reasons for 80 to 170 tokens). `GlimmerClient.cloud(lora=None)` does this.
  A scaled-to-zero deployment answers `503 DEPLOYMENT_SCALING_UP` immediately, so keep it warm in the demo
  window and race Spark as planned.
- **Memory**: only one Glimmer process at a time (about 16 to 17 GB). Stop the edge server before
  running `training/eval_mlx.sh` or training, and stop it during real flights if the plan requires it.
- **Contracts**: no contract changes were needed. `Intent`, `GLIMMER_SYSTEM_PROMPT`, `parse_intent` and
  `build_reply` are used unchanged.

## Known gaps and risks

1. **No LoRA on Fireworks yet.** Dataset and model uploads are disabled for this account (exact errors
   above), so the "trained Glimmer live on Fireworks" goal needs the owner to get uploads enabled, then
   about 10 minutes of commands plus training time (runbook above). The hot-reload LoRA load
   (`deployedModels?replaceMergedAddon=true`) and the live-merge fallback (`morning --merged`) are
   implemented and unit-tested against a mock, but have not run against the real API.
2. The local LoRA is a QLoRA trained on the 4-bit MLX base. If uploads open up, retrain on Fireworks from the
   same data (`fireworks.py train`, BF16) rather than converting this adapter. No MLX-to-PEFT converter
   was written. For the record, the adapter covers layers 36 to 51, 8 linear modules each, with lora_a
   `[in, 16]` and lora_b `[16, out]`. PEFT wants both transposed, with `lora_alpha = 32` (scale 2 x
   rank 16). The HF module prefix of Glimmer's text tower was not verified.
3. The template appends `Reasoning strength: high.` after `GLIMMER_SYSTEM_PROMPT`'s own
   `Reasoning strength: low`. Training and serving are consistent, so the LoRA is unaffected. For a v2, the
   integrator could drop the line from the frozen prompt, or pass `reasoning_strength` in both training
   and serving.
4. Base Glimmer on Fireworks was scored on test and gold only (the spoken set was built after the
   deployment was deleted). Local base Glimmer covers all three sets.
5. Small evaluation sets: 80 gold commands give about plus or minus 5 points of noise on accuracy. Test
   comes from held-out template families and paraphrases, so it is closer to the training distribution
   than gold is.
6. The spoken gold set uses macOS TTS voices, not people. Omnilingual (no language hint) turned 3 of 80
   very short clips into non-Latin script; they are kept because that is what the recognizer did.
7. The edge-mode LoRA takes several seconds per command (see latency). That is above the 2.5 s
   hedged-routing budget written for the cloud path. Use it as the offline path, or give it its own budget.
8. The dev.meta.ai terms on training with model outputs were not conclusively verified (see Terms check).
9. Baseten was not tried: there is no `BASETEN_API_KEY` in `.env`. The Fireworks Serverless Training API
   was not tried: it needs the `fireworks` SDK, which is not installed, and it would still hit the
   model-creation block when promoting the result.

## Spend (this workstream)

SPEND
