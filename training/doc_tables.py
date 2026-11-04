"""Print compact markdown tables from docs/eval/results.json (pasted into docs/ws/train.md) and spend totals."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["glimmer-base-mlx", "glimmer-lora-mlx", "glimmer-base-fireworks", "glimmer-base-fireworks-schema",
         "spark-1.3"]


def results_table() -> str:
    res = json.loads((ROOT / "docs" / "eval" / "results.json").read_text())["models"]
    out = []
    for s, title in (("test", "Held-out test (300)"), ("gold", "Gold, hand-written (80)"),
                     ("gold_asr", "Gold spoken by `say` + Omnilingual ASR (80)")):
        out += [f"**{title}**", "",
                "| model | valid JSON | action | subject (norm.) | clarify F1 | refuse F1 | land/hover |",
                "|---|---|---|---|---|---|---|"]
        for m in ORDER:
            r = res.get(m, {}).get("sets", {}).get(s)
            if not r:
                continue
            lh = "n/a" if r["land_hover_accuracy"] is None else f"{r['land_hover_accuracy']:.1%}"
            out.append(f"| {res[m]['name']} | {r['valid_json_rate']:.1%} | {r['action_accuracy']:.1%} | "
                       f"{r['subject_normalized']:.1%} | {r['clarify']['f1']:.2f} | {r['refuse']['f1']:.2f} | {lh} |")
        out.append("")
    out += ["**Latency per command, one request at a time** (client wall time)", "",
            "| model | p50 | p95 | n | conditions |", "|---|---|---|---|---|"]
    for m in ORDER:
        seq = res.get(f"{m}-seq")
        if not seq:
            continue
        set_name, r = next(iter(seq["sets"].items()))
        out.append(f"| {res[m]['name'] if m in res else m} | {r['latency_ms']['p50']:.0f} ms | "
                   f"{r['latency_ms']['p95']:.0f} ms | {r['n']} ({set_name}) | {'; '.join(seq['notes'])} |")
    return "\n".join(out) + "\n"


def spend() -> str:
    path = Path(os.environ.get("MHD_HOME", ROOT)) / "logs" / "spend.jsonl"
    tot: dict[str, float] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("ws") == "train":
            tot[r["service"]] = tot.get(r["service"], 0.0) + float(r["est_usd"])
    lines = [f"- `{k}`: ${v:.2f}" for k, v in sorted(tot.items())]
    lines.append(f"- total: ${sum(tot.values()):.2f} (caps: Meta API $10, Fireworks GPU 1.5 hours / about $12)")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(results_table())
    print(spend())
