# Perception benchmarks (G1 speed; G0 for SAM; G2)

Machine: MacBook Pro M5 Pro, 48 GB, 20-core GPU, macOS 26. mlx 0.32.4, mlx-vlm 0.7.11,
SAM 3.1 MLX weights `mlx-community/sam3.1-bf16` (stored as F32, 873M parameters), OpenCV 5.0.0.

Raw results: `logs/bench_g1.json` (not in git). Reproduce:
`$MHD_ML_PY -m mhd_ml.perception.bench --images media/bench/coco --runs 30 --out logs/bench_g1.json`

## G1: SAM 3.1 MLX speed on this Mac (measured 2026-11-03 about 23:43 ET)

**What else was running:** nothing else on the GPU (ioreg GPU utilization 0% before the run). No
twin Chrome, no Glimmer/mlx_lm, no llama-server. The model downloads had finished ("ALL DONE" in
`logs/fetch.log`). Background: Cursor, a Virtualization.framework VM (about 8 GB resident),
BambuStudio idle. So these are **idle-machine numbers**; see "under load" below.

Inputs: 12 COCO val2017 images (9 with people, 3 with cars; CC BY 4.0, in `media/bench/coco`,
not in git), center-cropped to 16:9 and resized to 1280x720. Prompt "person" or "car".
3 warm-up calls, then 30 timed calls cycling through the images. Time is end to end: preprocess
(resize to the square model input + normalize), ViT backbone, heads (FPN neck, DETR encoder and
decoder, mask decoder, NMS), and conversion to boxes and polygons in 1280x720.

### Text-prompted detection

| Model input | p50 ms | p95 ms | backbone p50 | heads p50 | mean candidates | mean top score |
|---|---|---|---|---|---|---|
| 1008 (native) | **581** | **597** | 406 | 161 | 12.1 | 0.936 |
| 672 | **234** | **247** | 168 | 57 | 12.2 | 0.922 |
| 504 | **172** | **190** | 129 | 36 | 14.5 | 0.901 |
| 336 | **74** | **102** | 44 | 26 | 21.6 | 0.867 |

- Candidates counted at score > 0.25 after NMS. Lower inputs give more low-quality candidates.
- Casting the weights to bf16 changed nothing (588 ms vs 581 ms at 1008); fp16 returned no
  detections at all. So the service runs the weights as stored (F32).
- MLX memory: about 5.6 GB peak for detection only; 11.7 GB peak with the tracker session below.
- Load time: 0.2 to 1.0 s with the weights in the page cache (cold load not measured).

### SAM 3.1 multiplex tracker (track_init / track_step)

Native 1008 input only (mlx-vlm's tracker needs the 1008 feature maps). One subject, a 34-frame
synthetic camera pan (20 px per step) over the skier image, a fresh ViT backbone every step
(a moving camera needs one; mlx-vlm's "realtime" demo reuses a stale backbone, which only works
for a static camera), and the memory encoder run every step.

| Part | p50 ms | p95 ms |
|---|---|---|
| Full track step (backbone + propagate) | **980** | **1533** |
| ViT backbone | 431 | 441 |
| Propagate (memory attention + mask decoder) | 544 | 1099 |
| Session init from a detection mask | 716 (once) | |

It tracks well (mean IoU 0.96 against the known pan offset, presence score 0.996), but at
about 1 s per step it misses the 0.20 s G1 target by 5x. **Verdict: no-go for the SAM 3.1 MLX
tracker in the control loop.**

### OpenCV trackers (CPU), 640x360, same pan sequence

| Tracker | idle p50 / p95 ms | with SAM 1008 detection running in parallel, p50 / p95 ms | mean IoU vs truth | reported success |
|---|---|---|---|---|
| VitTrack (opencv_zoo 2023sep ONNX) | **2.6 / 2.7** | **2.8 / 2.9** | 0.85 | 100% |
| KCF | 26.4 / 27.1 | 28.8 / 30.9 | 0.92 | 100% |
| CSRT | 13.8 / 14.7 | 17.2 / 20.5 | 0.90 | 100% |

All three hold 5 Hz (0.20 s per step) with a wide margin, also while SAM detection keeps the
GPU busy. "Under load" here means SAM detection only: the twin's Chrome renderer and
Glimmer were not running at measurement time (only the twin and ui workstreams run Chrome, and
only train runs Glimmer). Repeat `--skip-sam` while they run to refresh the CPU tracker numbers.

### G1 decision

- **In-loop tracker: VitTrack** (`--tracker auto` picks it), KCF as the fallback, CSRT last.
  VitTrack also reports an honest score (`getTrackingScore`), which feeds `tracker_score`.
- **SAM 3.1 MLX tracker: not used in the loop** (about 1 s per step). It stays available as
  `--tracker sam_mlx` for offline replays.
- **SAM 3.1 MLX detector:** 672 input is the default (234 ms p50): 2 Hz while finding uses about
  half the GPU; 1 Hz while following about a quarter. 1008 (581 ms) cannot sustain 2 Hz.
- GPU sharing: SAM detection at 2 Hz keeps the GPU about 50% busy, which will slow Glimmer
  (train) and the twin renderer, and they will slow SAM. Measure again once they run together.
