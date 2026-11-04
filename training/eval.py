"""Evaluate intent models on the held-out test set and the hand-written gold set.

Two steps, so runs can be resumed and merged:
  run     one model on one or more sets -> logs/eval/<label>.jsonl (one prediction per line)
  report  every logs/eval/*.jsonl -> docs/eval/results.json, docs/eval/summary.md, docs/eval/intent_eval.png

Backends (all OpenAI-style chat completions):
  spark      muse-spark-1.3 on the Meta Model API: SYSTEM_PROMPT, strict JSON schema, reasoning_effort minimal
  fireworks  Glimmer on our Fireworks on-demand deployment (--model is '<model>#<deployment>')
  mlx        mlx_lm.server (default http://127.0.0.1:8580/v1), base Glimmer or Glimmer + LoRA

Glimmer backends send GLIMMER_SYSTEM_PROMPT exactly as in training. The answer is parsed with
mhd.intent_spec.parse_intent, the same function the agent uses, so "valid JSON" means "the agent
would have accepted it".

Examples:
  $MHD_PY training/eval.py run --backend spark --label spark-1.3 --sets test,gold
  $MHD_PY training/eval.py run --backend mlx --label glimmer-lora-mlx --sets test,gold --concurrency 4
  $MHD_PY training/eval.py report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mhd.contracts import Intent  # noqa: E402
from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT, SYSTEM_PROMPT, intent_json_schema, parse_intent  # noqa: E402

from training.common import DATA_DIR, LOG_DIR, log_spend, read_jsonl, spark_cost  # noqa: E402

EVAL_LOG = LOG_DIR / "eval"
DOCS = ROOT / "docs" / "eval"
SETS = {"test": DATA_DIR / "test.jsonl", "gold": DATA_DIR / "gold.jsonl", "valid": DATA_DIR / "valid.jsonl"}

# ----------------------------------------------------------------------------------------------
# Loading examples
# ----------------------------------------------------------------------------------------------


def load_set(name: str) -> list[dict]:
    rows = read_jsonl(SETS[name])
    return [{"id": r["id"], "set": name, "transcript": r["transcript"], "gold": r["intent"],
             "group": r.get("group", "")} for r in rows]


# ----------------------------------------------------------------------------------------------
# Parsing model output
# ----------------------------------------------------------------------------------------------

_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def final_answer(text: str) -> str:
    """The user-facing part of a raw Glimmer reply (after the last '<|message|>' if channels leaked)."""
    text = text or ""
    if "<|message|>" in text:
        text = text.rsplit("<|message|>", 1)[1]
    return _SPECIAL.sub("", text).strip()


def parse_reply(content: Optional[str]) -> Optional[Intent]:
    """parse_intent on the content; if channel markers leaked into the content, try the final channel."""
    if not content:
        return None
    got = parse_intent(content)
    if got is None and "<|" in content:
        got = parse_intent(final_answer(content))
    return got


# ----------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------


class Backend:
    label = "model"

    def __call__(self, transcript: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class SparkBackend(Backend):
    """muse-spark-1.3 on the Meta Model API with the strict intent schema (the agent's fallback)."""

    def __init__(self, effort: str = "minimal"):
        from training.common import META_BASE
        self.url = f"{META_BASE}/chat/completions"
        self.key = os.environ["MODEL_API_KEY"]
        self.effort = effort
        self.client = httpx.Client(timeout=90)
        self.lock = threading.Lock()
        self.pending_usd = 0.0
        self.pending_units = 0

    def __call__(self, transcript: str) -> dict:
        body = {"model": "muse-spark-1.3",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": transcript}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "intent", "strict": True, "schema": intent_json_schema()}},
                "reasoning_effort": self.effort, "max_completion_tokens": 1500}
        t0 = time.perf_counter()
        r = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"})
        ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        j = r.json()
        usage = j.get("usage", {})
        with self.lock:
            self.pending_usd += spark_cost(usage)
            self.pending_units += int(usage.get("total_tokens", 0))
        msg = j["choices"][0]["message"]
        return {"content": msg.get("content"), "reasoning": msg.get("reasoning_content"), "latency_ms": ms,
                "usage": usage, "finish": j["choices"][0].get("finish_reason")}

    def flush_spend(self, note: str) -> None:
        with self.lock:
            if self.pending_units:
                log_spend("meta:muse-spark-1.3", self.pending_units, self.pending_usd, note=note)
            self.pending_usd, self.pending_units = 0.0, 0


class OpenAICompatBackend(Backend):
    """Glimmer through an OpenAI-compatible server (Fireworks deployment or mlx_lm.server)."""

    def __init__(self, url: str, model: str, headers: Optional[dict] = None, max_tokens: int = 400,
                 extra: Optional[dict] = None, system: str = GLIMMER_SYSTEM_PROMPT, timeout: float = 180):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = headers or {}
        self.max_tokens = max_tokens
        self.extra = extra or {}
        self.system = system
        self.client = httpx.Client(timeout=timeout)

    def __call__(self, transcript: str) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": transcript}],
                **self.extra}
        t0 = time.perf_counter()
        r = self.client.post(self.url, json=body, headers=self.headers)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        msg = j["choices"][0]["message"]
        return {"content": msg.get("content"), "latency_ms": ms, "usage": j.get("usage", {}),
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                "finish": j["choices"][0].get("finish_reason")}


def make_backend(a: argparse.Namespace) -> Backend:
    extra = json.loads(a.extra) if a.extra else {}
    if a.template_kwargs:
        extra["chat_template_kwargs"] = json.loads(a.template_kwargs)
    if a.backend == "spark":
        return SparkBackend(a.effort)
    if a.backend == "fireworks":
        if not a.model:
            sys.exit("--model '<model>#<deployment>' is required for fireworks")
        model = a.model.replace("$FIREWORKS_ACCOUNT_ID", os.environ.get("FIREWORKS_ACCOUNT_ID", ""))
        return OpenAICompatBackend("https://api.fireworks.ai/inference/v1", model,
                                   {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                                   a.max_tokens, extra)
    if a.backend == "mlx":
        return OpenAICompatBackend(a.url, a.model or "default_model", None, a.max_tokens, extra)
    sys.exit(f"unknown backend {a.backend}")


# ----------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------


def run(a: argparse.Namespace) -> Path:
    backend = make_backend(a)
    out = EVAL_LOG / f"{a.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.fresh:
        done = {(r["set"], r["id"]) for r in read_jsonl(out) if r.get("error") is None}
    examples = []
    for name in a.sets.split(","):
        rows = load_set(name)
        if a.limit:
            rows = rows[:a.limit]
        examples += [e for e in rows if (e["set"], e["id"]) not in done]
    print(f"[{a.label}] {len(examples)} examples to run ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    mode = "w" if a.fresh else "a"
    n_ok = 0
    t_start = time.time()
    with open(out, mode) as f, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = {pool.submit(_one, backend, e): e for e in examples}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rec.update({"model": a.label, "backend": a.backend, "concurrency": a.concurrency,
                        "note": a.note or ""})
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            n_ok += rec.get("pred") is not None
            if k % 25 == 0 or k == len(examples):
                el = time.time() - t_start
                print(f"[{a.label}] {k}/{len(examples)} valid={n_ok} {el:.0f}s", flush=True)
                if isinstance(backend, SparkBackend):
                    backend.flush_spend(f"eval {a.label}")
    if isinstance(backend, SparkBackend):
        backend.flush_spend(f"eval {a.label}")
    return out


def _one(backend: Backend, e: dict, retries: int = 2) -> dict:
    rec: dict[str, Any] = {"id": e["id"], "set": e["set"], "transcript": e["transcript"], "gold": e["gold"],
                           "group": e["group"]}
    for attempt in range(retries + 1):
        try:
            res = backend(e["transcript"])
            pred = parse_reply(res.get("content"))
            rec.update({"raw": res.get("content"), "reasoning": res.get("reasoning"),
                        "latency_ms": round(res["latency_ms"], 1), "usage": res.get("usage"),
                        "finish": res.get("finish"), "pred": pred.model_dump() if pred else None, "error": None})
            return rec
        except Exception as ex:  # network or server error: retry, then record it
            rec.update({"error": f"{type(ex).__name__}: {str(ex)[:300]}", "pred": None})
            time.sleep(2 + 3 * attempt)
    return rec


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

_DROP = {"the", "a", "an", "in", "with", "wearing", "on", "of", "carrying", "holding", "that", "this", "one"}
_SYNONYMS = {
    # people: the label always says person
    "man": "person", "woman": "person", "guy": "person", "lady": "person", "kid": "person", "child": "person",
    "boy": "person", "girl": "person", "dude": "person", "someone": "person", "runner": "person",
    "jogger": "person", "cyclist": "person", "skater": "person", "skateboarder": "person", "people": "person",
    "grey": "gray", "t-shirt": "shirt", "tshirt": "shirt", "tee": "shirt", "top": "shirt", "hoody": "hoodie",
    "sweatshirt": "sweater", "jumper": "sweater", "raincoat": "coat", "overcoat": "coat", "trousers": "pants",
    "rucksack": "backpack", "puppy": "dog", "pup": "dog", "doggy": "dog", "doggo": "dog", "lab": "dog",
    "kitty": "cat", "kitten": "cat", "pony": "horse", "bunny": "rabbit", "sedan": "car", "automobile": "car",
    "minivan": "van", "cab": "taxi", "bicycle": "bike", "pram": "stroller", "hen": "chicken", "jeans": "pants",
}
_PHRASES = {"pickup truck": "pickup", "rc car": "toy car", "remote control car": "toy car", "baseball cap": "cap",
            "ball cap": "cap", "school bus": "bus", "t shirt": "shirt", "tee shirt": "shirt"}


def norm_subject(s: str) -> tuple[str, ...]:
    """Lenient subject comparison: lowercase, synonyms, no articles or prepositions, order-free."""
    s = " " + re.sub(r"[^a-z0-9\- ]", " ", (s or "").lower()) + " "
    for k, v in _PHRASES.items():
        s = s.replace(f" {k} ", f" {v} ")
    toks = [_SYNONYMS.get(t, t) for t in s.split()]
    return tuple(sorted(t for t in toks if t not in _DROP))


def norm_color(c: str) -> str:
    c = (c or "none").lower().strip()
    return {"grey": "gray"}.get(c, c)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def metrics(recs: list[dict]) -> dict:
    """All metrics for one model on one set. Invalid or failed predictions count as wrong."""
    n = len(recs)
    valid = [r for r in recs if r.get("pred")]
    acts = ["follow", "hover", "land", "clarify", "refuse"]
    conf = {g: {p: 0 for p in acts + ["invalid"]} for g in acts}
    for r in recs:
        conf[r["gold"]["action"]][r["pred"]["action"] if r.get("pred") else "invalid"] += 1
    action_ok = sum(1 for r in valid if r["pred"]["action"] == r["gold"]["action"])
    fol = [r for r in recs if r["gold"]["action"] == "follow"]

    def on_follow(fn) -> float:
        ok = sum(1 for r in fol if r.get("pred") and fn(r["pred"], r["gold"]))
        return round(ok / len(fol), 4) if fol else 0.0

    out = {
        "n": n,
        "valid_json_rate": round(len(valid) / n, 4) if n else 0.0,
        "action_accuracy": round(action_ok / n, 4) if n else 0.0,
        "subject_exact": on_follow(lambda p, g: p["action"] == "follow" and
                                   p["subject_phrase"].strip().lower() == g["subject_phrase"]),
        "subject_normalized": on_follow(lambda p, g: p["action"] == "follow" and
                                        norm_subject(p["subject_phrase"]) == norm_subject(g["subject_phrase"])),
        "category_accuracy": on_follow(lambda p, g: p["category"] == g["category"]),
        "color_accuracy": on_follow(lambda p, g: norm_color(p["color"]) == norm_color(g["color"])),
        "distance_accuracy": on_follow(lambda p, g: p["distance"] == g["distance"]),
        "n_follow": len(fol),
        "confusion": conf,
    }
    for cls in ("clarify", "refuse"):
        tp = sum(1 for r in valid if r["pred"]["action"] == cls and r["gold"]["action"] == cls)
        fp = sum(1 for r in valid if r["pred"]["action"] == cls and r["gold"]["action"] != cls)
        fn = sum(1 for r in recs if r["gold"]["action"] == cls and not (r.get("pred") and r["pred"]["action"] == cls))
        out[cls] = _prf(tp, fp, fn)
    ref = [r for r in recs if r["gold"]["action"] == "refuse"]
    out["refuse_reason_accuracy"] = round(sum(1 for r in ref if r.get("pred") and r["pred"]["action"] == "refuse"
                                              and r["pred"]["reason_code"] == r["gold"]["reason_code"]) / len(ref), 4) \
        if ref else None
    # Safety words: a spoken land or stop must never turn into a follow.
    sw = [r for r in recs if r["gold"]["action"] in ("land", "hover")]
    out["land_hover_accuracy"] = round(sum(1 for r in sw if r.get("pred") and r["pred"]["action"] ==
                                           r["gold"]["action"]) / len(sw), 4) if sw else None
    lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None and r.get("error") is None]
    out["latency_ms"] = {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "mean": round(sum(lat) / len(lat), 1) if lat else None}
    toks = [int((r.get("usage") or {}).get("completion_tokens", 0)) for r in recs if r.get("usage")]
    out["completion_tokens_mean"] = round(sum(toks) / len(toks), 1) if toks else None
    out["errors"] = sum(1 for r in recs if r.get("error"))
    out["concurrency"] = sorted({r.get("concurrency") for r in recs if r.get("concurrency") is not None})
    return out


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

ORDER = ["glimmer-base-mlx", "glimmer-base-fireworks", "glimmer-base-fireworks-schema", "glimmer-lora-mlx",
         "glimmer-lora-fireworks", "spark-1.3"]
PRETTY = {"glimmer-base-mlx": "Glimmer base (MLX 4-bit, local)",
          "glimmer-base-fireworks": "Glimmer base (Fireworks H200)",
          "glimmer-base-fireworks-schema": "Glimmer base + JSON schema (Fireworks)",
          "glimmer-lora-mlx": "Glimmer + LoRA (MLX 4-bit, local)",
          "glimmer-lora-fireworks": "Glimmer + LoRA (Fireworks)",
          "spark-1.3": "Muse Spark 1.3 (Meta API, minimal effort)"}


def report(a: argparse.Namespace) -> dict:
    files = sorted(EVAL_LOG.glob("*.jsonl"))
    results: dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "models": {}}
    for fp in files:
        label = fp.stem
        if a.only and label not in a.only.split(","):
            continue
        recs = read_jsonl(fp)
        # keep the latest record per (set, id)
        latest: dict[tuple, dict] = {}
        for r in recs:
            latest[(r["set"], r["id"])] = r
        by_set: dict[str, list] = {}
        for r in latest.values():
            by_set.setdefault(r["set"], []).append(r)
        results["models"][label] = {
            "name": PRETTY.get(label, label),
            "notes": sorted({r.get("note", "") for r in latest.values() if r.get("note")}),
            "sets": {s: metrics(v) for s, v in sorted(by_set.items())}}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (DOCS / "summary.md").write_text(summary_md(results))
    chart(results, DOCS / "intent_eval.png")
    print((DOCS / "summary.md").read_text())
    return results


def _ordered(models: dict) -> list[str]:
    return sorted(models, key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))


def summary_md(results: dict) -> str:
    lines = ["# Intent model evaluation", "",
             f"Generated {results['generated']} by `training/eval.py report`. Test = 300 held-out examples from "
             "unseen template families (some Spark-paraphrased, ASR-noised); gold = 80 hand-written commands. "
             "Invalid replies count as wrong. Subject/category/color/distance are scored on follow examples only. "
             "Latency is client-side wall time per request.", ""]
    for s in ("test", "gold"):
        lines += [f"## {s} set", "",
                  "| model | n | valid JSON | action acc | subject exact | subject norm | category | color | distance |"
                  " clarify F1 | refuse F1 | refuse reason | land/hover | p50 ms | p95 ms | out tok |",
                  "|---|" + "---|" * 15]
        for m in _ordered(results["models"]):
            r = results["models"][m]["sets"].get(s)
            if not r:
                continue
            lines.append(
                f"| {results['models'][m]['name']} | {r['n']} | {r['valid_json_rate']:.1%} | {r['action_accuracy']:.1%} "
                f"| {r['subject_exact']:.1%} | {r['subject_normalized']:.1%} | {r['category_accuracy']:.1%} "
                f"| {r['color_accuracy']:.1%} | {r['distance_accuracy']:.1%} | {r['clarify']['f1']:.2f} "
                f"| {r['refuse']['f1']:.2f} | {_fmt(r['refuse_reason_accuracy'])} | {_fmt(r['land_hover_accuracy'])} "
                f"| {_n(r['latency_ms']['p50'])} | {_n(r['latency_ms']['p95'])} | {_n(r['completion_tokens_mean'])} |")
        lines.append("")
    lines += ["## Notes per model", ""]
    for m in _ordered(results["models"]):
        notes = results["models"][m]["notes"]
        if notes:
            lines.append(f"- **{results['models'][m]['name']}**: " + "; ".join(notes))
    lines.append("")
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _n(x) -> str:
    return "n/a" if x is None else f"{x:.0f}"


def chart(results: dict, path: Path) -> None:
    """Grouped bars per set (quality metrics) plus a latency panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = _ordered(results["models"])
    if not models:
        return
    metrics_shown = [("valid_json_rate", "valid JSON"), ("action_accuracy", "action"),
                     ("subject_normalized", "subject (norm.)"), ("clarify", "clarify F1"), ("refuse", "refuse F1")]
    # Categorical palette, colorblind-safe (Okabe-Ito), one hue per model.
    palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [3, 3, 2]})
    width = 0.8 / len(models)
    for ax, s in zip(axes[:2], ("test", "gold")):
        for k, m in enumerate(models):
            r = results["models"][m]["sets"].get(s)
            if not r:
                continue
            vals = [(r[key]["f1"] if isinstance(r[key], dict) else r[key]) for key, _ in metrics_shown]
            xs = [i + (k - (len(models) - 1) / 2) * width for i in range(len(metrics_shown))]
            ax.bar(xs, vals, width * 0.92, color=palette[k % len(palette)], label=results["models"][m]["name"])
        ax.set_xticks(range(len(metrics_shown)), [lbl for _, lbl in metrics_shown])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{s} set" + (" (300 held-out)" if s == "test" else " (80 hand-written)"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    for k, m in enumerate(models):
        r = results["models"][m]["sets"].get("test") or results["models"][m]["sets"].get("gold")
        if not r or r["latency_ms"]["p50"] is None:
            continue
        ax.bar(k - 0.2, r["latency_ms"]["p50"] / 1000, 0.38, color=palette[k % len(palette)])
        ax.bar(k + 0.2, r["latency_ms"]["p95"] / 1000, 0.38, color=palette[k % len(palette)], alpha=0.5)
    ax.set_xticks(range(len(models)), [str(i + 1) for i in range(len(models))])
    ax.set_xlabel("model (legend order); solid p50, light p95")
    ax.set_ylabel("seconds per command")
    ax.set_title("latency")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{i + 1}. {lbl}" for i, lbl in enumerate(labels)], loc="lower center",
               ncol=min(3, len(labels)), frameon=False)
    fig.suptitle("Voice command to mission intent: Muse Glimmer LoRA vs base vs Muse Spark", fontsize=13)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["spark", "fireworks", "mlx"], required=True)
    r.add_argument("--label", required=True, help="output name, e.g. glimmer-lora-mlx")
    r.add_argument("--sets", default="test,gold")
    r.add_argument("--model", help="fireworks: '<model>#<deployment>'; mlx: optional model name")
    r.add_argument("--url", default="http://127.0.0.1:8580/v1", help="mlx_lm.server base URL")
    r.add_argument("--max-tokens", type=int, default=400)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--limit", type=int, default=0, help="first N examples of each set (0 = all)")
    r.add_argument("--effort", default="minimal", help="spark reasoning_effort")
    r.add_argument("--extra", help="JSON merged into the request body")
    r.add_argument("--template-kwargs", help="JSON chat_template_kwargs (mlx_lm.server)")
    r.add_argument("--note", help="free text stored with every record (what else was running, etc.)")
    r.add_argument("--fresh", action="store_true", help="overwrite instead of resuming")
    rp = sub.add_parser("report")
    rp.add_argument("--only", help="comma-separated labels to include")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print("wrote", run(a))
    else:
        report(a)


if __name__ == "__main__":
    main()
