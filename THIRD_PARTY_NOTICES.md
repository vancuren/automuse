# Third-party notices

This repository's own code is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). It includes one
third-party code base (turbodrone), three lookup tables from pylwdrone, and uses models, libraries and tools that keep their own
licenses. **No model weights, no node_modules, no virtualenvs and no downloaded tools are stored in
git**: `scripts/sync_envs.sh`, `scripts/fetch_models.sh` and `npm` fetch them into the project
folder, and each comes with its own license.

Each license below was read from the installed package metadata, the model card, or the license
file that ships with the download, on 2026-11-04 (except where marked "not checked locally").

## 1. Included code: turbodrone (Apache-2.0)

[turbodrone](https://github.com/marshallrichards/turbodrone), Copyright 2025 Marshall Richards,
Apache License 2.0. It lives in `turbodrone/` with its own `LICENSE`, `NOTICE` and `README.md`. It
was imported unchanged from upstream commit 2c09cf8 (this repository's commit `1aff2b5`); commits
`09d005e` and `15a5f80` added 16 upstream files that the import had missed, byte-identical to
2c09cf8. Its NOTICE block is kept verbatim at the top of the root [NOTICE](NOTICE).

**Files we changed** (Apache-2.0 section 4(b)). Russell Van Curen modified or deleted these
upstream files in November 2026. List as of commit `6bb13fb`; regenerate it with
`git diff --name-status 1aff2b5 HEAD -- turbodrone/ | grep -v '^A'`.

| Change | File | What and why |
|---|---|---|
| modified | `turbodrone/backend/web_server.py` | governor installed as the control strategy, FLIGHT_MODE gate, paced RC loop, `/ws` routed through our hub, landing on Ctrl-C, loopback-only CORS; each change marked `MHD patch` |
| modified | `turbodrone/backend/requirements.txt` | removed ultralytics (AGPL) with the follow plugin |
| deleted | `turbodrone/backend/plugins/follow/` (4 files) | the follow plugin imported ultralytics (AGPL) |
| modified | `turbodrone/frontend/.gitignore` | re-includes `src/lib/`, which turbodrone's own `lib/` rule hid |
| modified | `turbodrone/frontend/index.html` | page title |
| modified | `turbodrone/frontend/package.json` | scripts for the mock backend (`mock`, `dev:mock`) |
| modified | `turbodrone/frontend/vite.config.ts` | dev and preview servers bound to 127.0.0.1 on a strict port; worker format |
| modified | `turbodrone/frontend/src/App.tsx`, `index.css` | the new operator dashboard and its styles |
| modified | `turbodrone/frontend/src/main.tsx`, `vite-env.d.ts` | router base path; types for the `VITE_*` settings |
| modified | `turbodrone/frontend/src/hooks/useControls.ts` | axes sent only while nonzero and zeroed on blur, keys by `e.code`, commands as `{"type":"cmd"}`, configurable backend URL |
| modified | `turbodrone/frontend/src/hooks/useOverlays.ts`, `pages/TestPage.tsx` | configurable backend URL instead of a hard-coded `localhost`; reconnect timer cleared on unmount |
| deleted | `turbodrone/frontend/src/App.css`; components `AxisIndicator`, `CameraTiltControl`, `CommandButtons`, `ControlSchemeToggle`, `ControlsOverlay`, `DrawingOverlay`, `PluginControls`, `SpeedControl`; hooks `usePlugins`, `useVideoSizing`; `lib/ws.ts` | replaced by the new panels, or removed for safety (speed tier, profiles, plugins, pointer lock that swallowed Esc) |

Every other file under `turbodrone/` is either upstream's, unchanged, or new (added by us).
Details: [docs/ws/gov.md](docs/ws/gov.md) and [docs/ws/ui.md](docs/ws/ui.md).

## 1b. Included data: pylwdrone lookup tables (Apache-2.0)

Our unit's camera turned out to be a Lewei Wi-Fi camera board (192.168.0.1, video on TCP 7060), not the
board turbodrone's `x69_lg` driver expects. `src/mhd/lewei/tables.py` holds three 256-entry lookup tables
copied as data from pylwdrone 0.5.0 (`pylwdrone/responses.py`), Copyright 2020 meekworth, Apache License 2.0
(https://github.com/meekworth/pylwdrone; full text in [licenses/pylwdrone-LICENSE](licenses/pylwdrone-LICENSE)).
The camera scrambles three bytes in the middle of each video frame, and these tables undo it. The client
code in `src/mhd/lewei/protocol.py` and `scripts/lewei_bridge.py` is our own, written from pylwdrone's
description of the protocol; pylwdrone itself is not installed or run.

## 2. Meta models

| Model | Where we get it | License | Notes |
|---|---|---|---|
| SAM 3.1 weights | [`mlx-community/sam3.1-bf16`](https://huggingface.co/mlx-community/sam3.1-bf16), an MLX conversion of [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) | **SAM License** | The model card says the original weights are under the SAM License. Full text: [licenses/SAM_LICENSE.txt](licenses/SAM_LICENSE.txt). It forbids uses subject to ITAR and end uses prohibited by trade controls, including military or warfare, nuclear, espionage, and guns or illegal weapons; anyone who distributes SAM Materials must include the license. |
| SAM 3.1, hosted (`sam-3.1`) | Meta Model API | Meta Model API terms | no weights involved |
| Muse Glimmer 30B | [`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit), an MLX conversion of [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) | **Apache-2.0**, plus Meta's usage policy | `license: apache-2.0` in the model card; the usage policy is linked from the original model card. Base Glimmer was also served on a Fireworks deployment (`accounts/fireworks/models/muse-glimmer-30b`) under Fireworks' terms. |
| Our LoRA adapter `glimmer-intent-v1` | trained by us (`training/train_mlx.sh full`) | derivative of Muse Glimmer: Apache-2.0 plus the usage policy | not in git (129 MB) |
| Omnilingual ASR CTC-300M v2, int8 | [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) `sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05` | **Apache-2.0**, Copyright 2025 Meta Platforms, Inc. and affiliates | `LICENSE` and `README.md` in the download |
| MMS-TTS English | [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) | **CC-BY-NC 4.0** | **Non-commercial use only.** Used for the agent's spoken replies; `--tts say` uses macOS speech instead. |
| Muse Spark 1.3 (`muse-spark-1.3`) | Meta Model API | Meta Model API terms | Its outputs reworded 458 training sentences (labels are ours). The terms on training with model outputs were not conclusively checked; see docs/ws/train.md. |
| Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Meta Model API | Meta Model API terms | |

Not used: Muse Spark Open (its license bars operating transportation technology), Muse Wonder,
Muse Image, Llama.

## 3. Other models

| Model | Where we get it | License | Notes |
|---|---|---|---|
| silero VAD (`silero_vad.onnx`) | sherpa-onnx release assets | MIT (snakers4/silero-vad) | not checked locally: the download has no license file |
| VitTrack (`object_tracking_vittrack_2023sep.onnx`) | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) | Apache-2.0 (opencv_zoo) | not checked locally: the download has no license file |
| CLIP tokenizer files (`openai/clip-vit-base-patch32`) | Hugging Face, loaded by mlx-vlm for SAM 3.1's text encoder | see the model card | not checked locally: the cached files (vocab, merges, tokenizer config) carry no license file |

## 4. Tools run as separate programs (not distributed)

| Tool | Version here | License | Notes |
|---|---|---|---|
| mediamtx | v1.21.3 (`tools/mediamtx/`, from `scripts/fetch_models.sh`) | MIT, Copyright (c) 2019 aler9 | `tools/mediamtx/LICENSE` |
| FFmpeg | 9.0.2 (Homebrew, `/opt/homebrew/bin/ffmpeg`) | this build: GPL-3.0-or-later (`ffmpeg -L`) | invoked as a command for HEVC encoding and recording; not linked, not distributed |
| Google Chrome | system install | Google Chrome terms | the twin's renderer, driven through Playwright (`channel="chrome"`) |
| uv, Node.js, npm | system installs | their own licenses | build tools only |

## 5. JavaScript packages

Fetched by `npm ci` (UI) and `npm install` (twin renderer); not in git. Licenses from each package's
`package.json`.

| Package | Version | License | Used in |
|---|---|---|---|
| three | 0.180.0 | MIT | twin renderer (`twin/web`) |
| react, react-dom | 19.1.0 | MIT | operator UI bundle |
| scheduler | 0.26.0 | MIT | operator UI bundle (react-dom) |
| react-router, react-router-dom | 7.6.2 | MIT | operator UI bundle |
| cookie, set-cookie-parser | 1.0.2, 2.7.1 | MIT | operator UI bundle (react-router) |
| vite | 6.3.5 | MIT | UI build tool |
| tailwindcss | 4.1.7 | MIT | UI build tool |
| typescript | 5.8.3 | Apache-2.0 | UI build tool |
| eslint | 9.27.0 | MIT | UI lint |
| @vitejs/plugin-react, autoprefixer, postcss | 4.4.1, 10.4.21, 8.5.3 | MIT | UI build tools |

## 6. Python packages

Installed by `scripts/sync_envs.sh` from `uv.lock` (core venv) and `ml/uv.lock` (ML venv); not in
git. Licenses from each installed package's metadata (`License-Expression`, else `License`, else
the license classifiers). Packages with the most weight in the project:

| Package | License | Role |
|---|---|---|
| mlx, mlx-metal, mlx-lm, mlx-vlm | MIT | SAM 3.1 and Muse Glimmer on Apple Silicon; the LoRA training |
| meta-sam-parser | SAM License | parsing the hosted SAM 3.1 responses (also covered by [licenses/SAM_LICENSE.txt](licenses/SAM_LICENSE.txt)) |
| sherpa-onnx | Apache-2.0 | Omnilingual ASR and the silero VAD |
| transformers, torch | Apache-2.0; torch: `Apache-2.0 AND ... AND BSD-3-Clause AND ...` (see below) | MMS-TTS |
| opencv-contrib-python | Apache-2.0 | RTSP decode, VitTrack, KCF, CSRT, ArUco |
| fastapi, uvicorn, starlette, pydantic, websockets, httpx | MIT / BSD-3-Clause | backend, twin, and every client: httpx for the Meta Model API, Fireworks and `mlx_lm.server`; websockets for the Voice Transcribe realtime stream and the backend sockets |
| playwright | Apache-2.0 | drives Chrome for the twin renderer and the UI tests |
| numpy, matplotlib, pyyaml, pillow, sounddevice, soundfile | BSD-3-Clause and others / matplotlib license / MIT / MIT-CMU / MIT / BSD-3-Clause | numerics, charts, config, images, audio |

<details>
<summary>Every installed Python package (105), with version and license</summary>

| Package | Version | License (metadata) | Venv |
|---|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | ML |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT | ML |
| aiosignal | 1.4.0 | Apache-2.0 | ML |
| annotated-doc | 0.0.5 | MIT | core, ML |
| annotated-types | 0.8.0 | MIT | core, ML |
| anyio | 4.16.0 | MIT | core, ML |
| attrs | 26.1.0 | MIT | ML |
| certifi | 2026.10.23 | MPL-2.0 | core, ML |
| cffi | 2.1.1 | MIT-0 | ML |
| charset-normalizer | 3.5.2 | MIT | ML |
| click | 8.5.0 | BSD-3-Clause | core, ML |
| contourpy | 1.4.0 | BSD-3-Clause | core, ML |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | ML |
| cycler | 0.12.1 | BSD | core, ML |
| datasets | 5.1.0 | Apache-2.0 | ML |
| dill | 0.4.1 | BSD-3-Clause | ML |
| fastapi | 0.162.0 | MIT | core, ML |
| filelock | 4.2.7 | MIT | ML |
| fonttools | 4.68.0 | MIT | core, ML |
| frozenlist | 1.8.0 | Apache-2.0 | ML |
| fsspec | 2026.7.0 | BSD-3-Clause | ML |
| greenlet | 3.5.6 | MIT AND PSF-2.0 | core |
| h11 | 0.16.0 | MIT | core, ML |
| hf-xet | 1.8.0 | Apache-2.0 | ML |
| httpcore | 1.0.9 | BSD-3-Clause | core, ML |
| httpcore2 | 2.16.0 | BSD-3-Clause | core, ML |
| httptools | 0.9.0 | MIT | core |
| httpx | 0.28.1 | BSD-3-Clause | core, ML |
| httpx2 | 2.16.0 | BSD-3-Clause | core, ML |
| huggingface-hub | 2.4.0 | Apache-2.0 | ML |
| idna | 3.20 | BSD-3-Clause | core, ML |
| iniconfig | 2.3.1 | MIT | core, ML |
| jinja2 | 3.1.6 | BSD | ML |
| jiter | 0.18.0 | MIT | core, ML |
| kiwisolver | 1.5.1 | BSD | core, ML |
| llguidance | 1.9.1 | MIT | ML |
| markdown-it-py | 4.2.0 | MIT | ML |
| markupsafe | 3.0.4 | BSD-3-Clause | ML |
| matplotlib | 3.11.2 | matplotlib license (PSF-based) | core, ML |
| mdurl | 0.1.2 | MIT | ML |
| meta-sam-parser | 0.1.0 | SAM License | ML |
| miniaudio | 1.71 | MIT | ML |
| mlx | 0.32.4 | MIT | ML |
| mlx-audio | 0.6.4 | MIT | ML |
| mlx-lm | 0.32.2 | MIT | ML |
| mlx-metal | 0.32.4 | MIT | ML |
| mlx-vlm | 0.7.11 | MIT | ML |
| mpmath | 1.3.0 | BSD | ML |
| multidict | 7.2.1 | Apache-2.0 | ML |
| multiprocess | 0.70.19 | BSD-3-Clause | ML |
| networkx | 3.7 | BSD-3-Clause | ML |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | core, ML |
| openai | 3.45.0 | Apache-2.0 | core, ML |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 | core, ML |
| opentelemetry-api | 1.45.2 | Apache-2.0 | core, ML |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | core, ML |
| pandas | 3.1.0 | BSD-3-Clause | ML |
| pillow | 13.0.0 | MIT-CMU | core, ML |
| playwright | 1.64.1 | Apache-2.0 | core |
| pluggy | 1.7.0 | MIT | core, ML |
| propcache | 0.5.4 | Apache-2.0 | ML |
| protobuf | 7.36.2 | BSD-3-Clause | ML |
| pyarrow | 26.0.1 | Apache-2.0 | ML |
| pycparser | 3.12 | BSD-3-Clause | ML |
| pydantic | 2.14.2 | MIT | core, ML |
| pydantic-core | 2.50.2 | MIT | core, ML |
| pyee | 14.0.0 | MIT | core |
| pygments | 2.21.0 | BSD-2-Clause | core, ML |
| pyparsing | 3.3.3 | MIT | core, ML |
| pytest | 9.2.0 | MIT | core, ML |
| python-dateutil | 4.0.0 | BSD-3-Clause | core, ML |
| python-dotenv | 1.2.4 | BSD-3-Clause | core |
| python-multipart | 0.0.33 | Apache-2.0 | core, ML |
| pyyaml | 6.0.3 | MIT | core, ML |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | ML |
| requests | 2.34.2 | Apache-2.0 | ML |
| rich | 15.0.0 | MIT | ML |
| safetensors | 0.8.0 | Apache-2.0 | ML |
| scipy | 1.18.1 | BSD | ML |
| sentencepiece | 0.2.2 | Apache-2.0 | ML |
| setuptools | 84.2.1 | MIT | ML |
| shellingham | 1.5.4 | ISC | ML |
| sherpa-onnx | 2.0.1 | Apache-2.0 | ML |
| sherpa-onnx-core | 2.0.1 | Apache-2.0 | ML |
| sniffio | 1.3.1 | MIT OR Apache-2.0 | core, ML |
| sounddevice | 0.5.6 | MIT | ML |
| soundfile | 0.14.0 | BSD-3-Clause | ML |
| starlette | 1.10.0 | BSD-3-Clause | core, ML |
| sympy | 1.14.0 | BSD | ML |
| tokenizers | 0.23.3 | Apache-2.0 | ML |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | ML |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | ML |
| transformers | 5.22.1 | Apache-2.0 | ML |
| truststore | 0.10.5 | MIT | core, ML |
| typer | 0.27.3 | MIT | ML |
| typing-extensions | 4.16.0 | PSF-2.0 | core, ML |
| typing-inspection | 0.4.4 | MIT | core, ML |
| tzdata | 2026.6 | Apache-2.0 | core, ML |
| urllib3 | 2.8.0 | MIT | ML |
| uvicorn | 0.54.0 | BSD-3-Clause | core, ML |
| uvloop | 0.23.0 | MIT OR Apache-2.0 | core |
| watchfiles | 1.3.0 | MIT | core |
| websockets | 17.4 | BSD-3-Clause | core, ML |
| xxhash | 4.0.2 | BSD-2-Clause | ML |
| yarl | 1.25.1 | Apache-2.0 | ML |

</details>

## 7. Data and media

- **COCO val2017 images** (CC BY 4.0) were used for the perception benchmarks. They stay in
  `media/bench/coco`, which is not in git.
- **Twin actors and scenes** are procedural (built in code in `twin/web/`); no third-party 3D assets.
- **Training data** (`training/data/`): template-generated commands with our labels, ASR-style
  noise, and 458 sentences reworded by Muse Spark 1.3. The 80 gold commands were written by hand.
- **Test audio** was generated with macOS `say` and stays in `media/`, which is not in git.
- **No raw footage of people** is stored in the repository; real-flight video appears only in the
  demo video, with the consent of everyone filmed.
