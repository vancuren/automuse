# Intent model evaluation

Generated 2026-11-04 02:14 EST by `training/eval.py report`. Test = 300 held-out examples from unseen template families (some Spark-paraphrased, ASR-noised); gold = 80 hand-written commands. Invalid replies count as wrong. Subject/category/color/distance are scored on follow examples only. Latency is client-side wall time per request.

## test set

| model | n | valid JSON | action acc | subject exact | subject norm | category | color | distance | clarify F1 | refuse F1 | refuse reason | land/hover | p50 ms | p95 ms | out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glimmer base (MLX 4-bit, local) | 100 | 100.0% | 95.0% | 35.3% | 90.2% | 98.0% | 92.2% | 94.1% | 0.85 | 0.97 | 100.0% | 90.9% | 52932 | 105356 | 340 |
| Glimmer base (Fireworks H200) | 300 | 100.0% | 93.7% | 41.4% | 88.2% | 97.6% | 92.9% | 88.8% | 0.84 | 0.85 | 77.5% | 94.1% | 1488 | 2689 | 137 |
| Glimmer base + JSON schema (Fireworks) | 300 | 100.0% | 94.0% | 40.8% | 88.2% | 97.6% | 92.9% | 89.3% | 0.85 | 0.86 | 80.0% | 94.1% | 1426 | 2563 | 138 |
| Glimmer + LoRA (MLX 4-bit, local) | 300 | 100.0% | 98.0% | 92.9% | 92.9% | 98.8% | 93.5% | 95.3% | 0.95 | 0.99 | 97.5% | 98.0% | 5429 | 6060 | 47 |
| Muse Spark 1.3 (Meta API, minimal effort) | 300 | 100.0% | 95.3% | 52.7% | 91.7% | 97.0% | 92.9% | 98.2% | 0.86 | 0.89 | 85.0% | 96.1% | 1886 | 3146 | 192 |

## gold set

| model | n | valid JSON | action acc | subject exact | subject norm | category | color | distance | clarify F1 | refuse F1 | refuse reason | land/hover | p50 ms | p95 ms | out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glimmer base (MLX 4-bit, local) | 80 | 100.0% | 91.2% | 50.0% | 86.4% | 95.5% | 97.7% | 93.2% | 0.73 | 0.76 | 72.7% | 93.8% | 58697 | 128144 | 328 |
| Glimmer base (Fireworks H200) | 80 | 100.0% | 93.8% | 47.7% | 86.4% | 93.2% | 97.7% | 88.6% | 0.82 | 0.80 | 72.7% | 100.0% | 1529 | 2524 | 140 |
| Glimmer base + JSON schema (Fireworks) | 80 | 100.0% | 95.0% | 45.5% | 84.1% | 95.5% | 100.0% | 88.6% | 0.82 | 0.84 | 72.7% | 100.0% | 1503 | 2398 | 139 |
| Glimmer + LoRA (MLX 4-bit, local) | 80 | 100.0% | 100.0% | 93.2% | 95.5% | 100.0% | 95.5% | 100.0% | 1.00 | 1.00 | 100.0% | 100.0% | 5151 | 5884 | 47 |
| Muse Spark 1.3 (Meta API, minimal effort) | 80 | 100.0% | 95.0% | 52.3% | 86.4% | 95.5% | 97.7% | 100.0% | 0.86 | 0.86 | 81.8% | 100.0% | 1970 | 3606 | 200 |

## gold_asr set

| model | n | valid JSON | action acc | subject exact | subject norm | category | color | distance | clarify F1 | refuse F1 | refuse reason | land/hover | p50 ms | p95 ms | out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glimmer base (MLX 4-bit, local) | 80 | 100.0% | 87.5% | 38.6% | 70.5% | 84.1% | 97.7% | 95.5% | 0.61 | 0.91 | 90.9% | 75.0% | 69154 | 128641 | 398 |
| Glimmer + LoRA (MLX 4-bit, local) | 80 | 100.0% | 88.8% | 77.3% | 79.5% | 84.1% | 88.6% | 100.0% | 0.62 | 0.77 | 90.9% | 93.8% | 5050 | 5728 | 47 |
| Muse Spark 1.3 (Meta API, minimal effort) | 80 | 100.0% | 90.0% | 38.6% | 72.7% | 81.8% | 95.5% | 100.0% | 0.72 | 0.90 | 81.8% | 87.5% | 1997 | 3097 | 207 |

## Latency per command

Client-side wall time for one intent request. 'sequential' rows come from a separate run with one request at a time (the agent's real pattern); others were measured under concurrent load.

| model | p50 ms | p95 ms | mean ms | measured |
|---|---|---|---|---|
| Glimmer base (MLX 4-bit, local) | 17580 | 25525 | 17467 | sequential, n=20 |
| Glimmer base (Fireworks H200) | 828 | 1250 | 913 | sequential, n=40 |
| Glimmer base + JSON schema (Fireworks) | 1426 | 2563 | 1571 | concurrency 12 |
| Glimmer + LoRA (MLX 4-bit, local) | 3606 | 4017 | 3615 | sequential, n=40 |
| Muse Spark 1.3 (Meta API, minimal effort) | 1871 | 3403 | 2475 | sequential, n=40 |

## Notes per model

- **Glimmer base (Fireworks H200)**: Fireworks H200 BF16 on-demand, 1 replica, client concurrency 12, home Wi-Fi; prompt GLIMMER_SYSTEM_PROMPT, no schema
- **Glimmer base + JSON schema (Fireworks)**: Fireworks H200 BF16 on-demand, 1 replica, client concurrency 12, home Wi-Fi; strict JSON schema response_format
- **Glimmer base (Fireworks H200), latency run**: latency run: sequential (concurrency 1), Fireworks H200, home Wi-Fi, local MLX training running on the Mac
- **Glimmer base (MLX 4-bit, local)**: MLX 4-bit on M5 Pro 48 GB, mlx_lm.server :8580, other workstreams may have shared the GPU (twin Chrome/WebGL renderer seen 00:25); client concurrency 8 (batched decode); template defaults, no schema; MLX 4-bit on M5 Pro, mlx_lm.server :8580, client concurrency 8-12 (batched decode); template defaults, no schema; test = first 100 examples only (time box)
- **Glimmer base (MLX), latency run**: MLX 4-bit on M5 Pro 48 GB, mlx_lm.server :8580, other workstreams may have shared the GPU (twin Chrome/WebGL renderer seen 00:25); sequential latency run
- **Glimmer + LoRA (MLX 4-bit, local)**: MLX 4-bit on M5 Pro 48 GB, mlx_lm.server :8580, other workstreams may have shared the GPU (twin Chrome/WebGL renderer seen 00:25); client concurrency 4 (batched decode)
- **Glimmer + LoRA (MLX), latency run**: MLX 4-bit on M5 Pro 48 GB, mlx_lm.server :8580, other workstreams may have shared the GPU (twin Chrome/WebGL renderer seen 00:25); sequential latency run
- **Muse Spark 1.3 (Meta API, minimal effort)**: Meta API from home Wi-Fi; other workstreams active on this Mac; concurrency 6; gold_asr run at 00:22, Meta API from home Wi-Fi, local MLX training running
- **Muse Spark 1.3, latency run**: latency run: sequential, Meta API from home Wi-Fi, local MLX training running on the Mac
