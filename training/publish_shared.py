"""Copy what other workstreams need into $MHD_HOME/media/shared/train/ (no secrets).

  $MHD_PY training/publish_shared.py

Writes glimmer_intent.json (adapter path, endpoints, how to call, headline metrics) and copies the eval
chart and summary. The account id is never written; the agent reads FIREWORKS_ACCOUNT_ID from env.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("MHD_HOME", ROOT))
OUT = HOME / "media" / "shared" / "train"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT / "docs" / "eval" / "results.json").read_text())["models"]
    headline = {}
    for m in ("glimmer-lora-mlx", "glimmer-base-mlx", "glimmer-base-fireworks", "spark-1.3"):
        if m in results:
            headline[m] = {s: {k: r[k] for k in ("valid_json_rate", "action_accuracy", "subject_normalized")}
                           for s, r in results[m]["sets"].items()}
    models = os.environ.get("MHD_MODELS", str(HOME / ".cache" / "models"))
    info = {
        "what": "Muse Glimmer 30B intent LoRA (train workstream). Transcript in, mhd.contracts.Intent JSON out.",
        "prompt": "mhd.intent_spec.GLIMMER_SYSTEM_PROMPT as the system message, transcript as the user message",
        "parse": "mhd.intent_spec.parse_intent(choices[0].message.content)",
        "client": "mhd_ml.glimmer_client.GlimmerClient.edge() / .cloud()",
        "edge": {"serve": "scripts/serve_glimmer.sh  (mlx_lm.server, 127.0.0.1:8080)",
                 "url": "http://127.0.0.1:8080/v1/chat/completions",
                 "request": {"model": "default_model", "temperature": 0, "max_tokens": 200},
                 "base_model_dir": f"{models}/Muse-Glimmer-30B-4bit",
                 "adapter_dir": f"{models}/adapters/glimmer-intent-v1",
                 "memory_gb": "about 16-17; only one Glimmer process at a time"},
        "cloud": {"url": "https://api.fireworks.ai/inference/v1/chat/completions",
                  "auth": "Bearer $FIREWORKS_API_KEY",
                  "lora_model": "accounts/$FIREWORKS_ACCOUNT_ID/models/glimmer-intent-v1#accounts/$FIREWORKS_ACCOUNT_ID/deployments/glimmer-intent",
                  "base_model": "accounts/fireworks/models/muse-glimmer-30b#accounts/$FIREWORKS_ACCOUNT_ID/deployments/glimmer-intent",
                  "status": "LoRA not on Fireworks: uploads disabled for the account on 2026-11-04 (see docs/ws/train.md). "
                            "Base deployments work: training/fireworks.py deploy --plain / morning / delete."},
        "mission_model_label": {"lora": "glimmer-30b+lora", "base": "glimmer-30b"},
        "headline_metrics": headline,
        "docs": "docs/ws/train.md, docs/eval/summary.md",
    }
    (OUT / "glimmer_intent.json").write_text(json.dumps(info, indent=2) + "\n")
    for f in ("intent_eval.png", "summary.md", "results.json"):
        shutil.copyfile(ROOT / "docs" / "eval" / f, OUT / f)
    print("published to", OUT)


if __name__ == "__main__":
    main()
