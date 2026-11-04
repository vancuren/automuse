"""Fireworks REST client for the Glimmer intent LoRA: datasets, SFT jobs, one deployment, warm-up, scale-down.

Uses only the REST API with FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID from the environment (env.sh loads
$MHD_HOME/.env). Never `firectl signin` (it writes to the home folder). Nothing here prints a key.

Deployment design (validated 2026-11-03 with validateOnly):
- Multi-LoRA shapes for muse-glimmer-30b exist only on GB300 (our quota is 0) and a B200 FP4 "decision
  engine" shape. So we use the H200 BF16 shape with `enableHotReloadLatestAddon`: the deployment starts as
  base Glimmer, then one LoRA is hot-loaded and merged (POST deployedModels?replaceMergedAddon=true).
  Hot-reload deployments need minReplicaCount == maxReplicaCount, so "scale down" means max=min=0 or delete.
- Base Glimmer is evaluated first; after the LoRA is merged the same deployment serves the LoRA.

Cost control: H200 on-demand is $8.00/hour. Every deployment this script creates is recorded in
logs/fireworks_state.json, and `down`/`delete` append the GPU time to $MHD_HOME/logs/spend.jsonl.

Commands (run with $MHD_PY training/fireworks.py <cmd> ...):
  upload    --file F --dataset-id ID           create a dataset and upload a JSONL file
  sft       --dataset ID --output-model M ...  start a LoRA SFT job (free for Glimmer until Nov 30)
  job       --job J [--wait]                   show (or wait for) an SFT job
  deploy    --deployment-id D [--lora M]       create the H200 deployment (base Glimmer), optionally load a LoRA
  load-lora --deployment-id D --lora M         hot-load (merge) a LoRA into the deployment
  warmup    --deployment-id D [--lora M]       send test requests until one answers, print latency
  morning   --lora M [--deployment-id D]       re-create + load LoRA + warm up, then print the agent's model string
  down      --deployment-id D                  scale to zero replicas (stops GPU billing)
  delete    --deployment-id D                  delete the deployment
  list                                         list deployments, deployed models, SFT jobs, datasets
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training.common import LOG_DIR, log_spend  # noqa: E402

API = "https://api.fireworks.ai/v1"
INFER = "https://api.fireworks.ai/inference/v1"
BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
H200_SHAPE = "accounts/fireworks/deploymentShapes/dkuwotf7"      # "Muse Glimmer 30B BF16 1x H200"
GPU_USD_PER_HOUR = 8.00                                          # H100/H200 on-demand (fireworks.ai/pricing)
STATE_FILE = LOG_DIR / "fireworks_state.json"
DEFAULT_DEPLOYMENT = "glimmer-intent"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Fireworks:
    """Thin wrapper over the Fireworks gateway REST API."""

    def __init__(self, timeout: float = 60.0):
        key = os.environ.get("FIREWORKS_API_KEY")
        self.account = os.environ.get("FIREWORKS_ACCOUNT_ID")
        if not key or not self.account:
            raise RuntimeError("FIREWORKS_API_KEY / FIREWORKS_ACCOUNT_ID not set (source env.sh)")
        self.c = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    # ---------------- helpers ----------------
    def _url(self, path: str) -> str:
        return f"{API}/accounts/{self.account}/{path}"

    def _req(self, method: str, path: str, **kw) -> dict:
        r = self.c.request(method, self._url(path), **kw)
        if r.status_code >= 400:
            # The error body never contains our key; trim it so logs stay readable.
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:600]}")
        return r.json() if r.content else {}

    def model_name(self, model_id: str) -> str:
        return model_id if model_id.startswith("accounts/") else f"accounts/{self.account}/models/{model_id}"

    def deployment_name(self, deployment_id: str) -> str:
        return f"accounts/{self.account}/deployments/{deployment_id}"

    # ---------------- datasets ----------------
    def upload_dataset(self, dataset_id: str, path: Path) -> dict:
        n = sum(1 for line in Path(path).read_text().splitlines() if line.strip())
        try:
            self._req("POST", "datasets", json={"datasetId": dataset_id, "dataset": {
                "displayName": dataset_id, "format": "CHAT", "exampleCount": str(n), "userUploaded": {}}})
        except RuntimeError as e:
            if "409" not in str(e) and "already exists" not in str(e).lower():
                raise
        with open(path, "rb") as f:
            r = self.c.post(self._url(f"datasets/{dataset_id}:upload"),
                            files={"file": (Path(path).name, f, "application/jsonl")}, timeout=300)
        if r.status_code >= 400:
            raise RuntimeError(f"upload {dataset_id} -> {r.status_code}: {r.text[:600]}")
        for _ in range(60):
            d = self._req("GET", f"datasets/{dataset_id}")
            if d.get("state") == "READY":
                return d
            time.sleep(5)
        return d

    # ---------------- supervised fine-tuning ----------------
    def create_sft(self, job_id: str, dataset_id: str, output_model: str, *, epochs: int = 1,
                   lora_rank: int = 8, learning_rate: Optional[float] = None, max_context: int = 2048,
                   eval_dataset: Optional[str] = None, batch_size_samples: Optional[int] = None) -> dict:
        body: dict[str, Any] = {
            "dataset": f"accounts/{self.account}/datasets/{dataset_id}",
            "baseModel": BASE_MODEL,
            "outputModel": self.model_name(output_model),
            "displayName": job_id,
            "epochs": epochs,
            "loraRank": lora_rank,
            "maxContextLength": max_context,
        }
        if learning_rate:
            body["learningRate"] = learning_rate
        if eval_dataset:
            body["evaluationDataset"] = f"accounts/{self.account}/datasets/{eval_dataset}"
        if batch_size_samples:
            body["batchSizeSamples"] = batch_size_samples
        return self._req("POST", "supervisedFineTuningJobs", params={"supervisedFineTuningJobId": job_id}, json=body)

    def get_sft(self, job_id: str) -> dict:
        return self._req("GET", f"supervisedFineTuningJobs/{job_id}")

    def wait_sft(self, job_id: str, poll_s: int = 60, timeout_s: int = 4 * 3600) -> dict:
        t0 = time.time()
        while True:
            j = self.get_sft(job_id)
            state = j.get("state", "")
            prog = j.get("jobProgress") or {}
            print(f"[{time.strftime('%H:%M:%S')}] {job_id}: {state} {json.dumps(prog)[:200]}", flush=True)
            if state in ("JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
                         "JOB_STATE_EARLY_STOPPED", "JOB_STATE_DELETED", "JOB_STATE_ARCHIVED", "JOB_STATE_PAUSED"):
                return j
            if time.time() - t0 > timeout_s:
                return j
            time.sleep(poll_s)

    # ---------------- deployments ----------------
    def create_deployment(self, deployment_id: str, *, replicas: int = 1, hot_reload: bool = True,
                          base_model: str = BASE_MODEL, shape: Optional[str] = H200_SHAPE) -> dict:
        """hot_reload=True: min=max=replicas (required by Fireworks), a LoRA can be merged in later.
        hot_reload=False: a plain base deployment with min 0 / max 1 that scales to zero after 10 idle
        minutes, so a crashed client can never leave a GPU billing for long."""
        body: dict[str, Any] = {
            "baseModel": base_model,
            "displayName": deployment_id,
            "description": "Muse Glimmer intent LoRA (MetaHackathonDrone, train workstream)",
            "minReplicaCount": replicas if hot_reload else 0,
            "maxReplicaCount": replicas if hot_reload else 1,
            "autoscalingPolicy": {"scaleToZeroWindow": "900s" if hot_reload else "600s"},
        }
        if shape:
            body["deploymentShape"] = shape
        if hot_reload:
            body["enableHotReloadLatestAddon"] = True
        d = self._req("POST", "deployments", params={"deploymentId": deployment_id}, json=body)
        _record(deployment_id, created=_now_iso(), base_model=base_model)
        return d

    def get_deployment(self, deployment_id: str) -> dict:
        return self._req("GET", f"deployments/{deployment_id}")

    def wait_deployment(self, deployment_id: str, timeout_s: int = 40 * 60) -> dict:
        t0 = time.time()
        while True:
            d = self.get_deployment(deployment_id)
            st = d.get("state")
            print(f"[{time.strftime('%H:%M:%S')}] deployment {deployment_id}: {st} replicas={d.get('replicaCount')}"
                  f" status={(d.get('status') or {}).get('code')}", flush=True)
            if st == "READY" and int(d.get("replicaCount") or 0) >= 1:
                return d
            if st in ("FAILED", "DELETED", "DELETING") or time.time() - t0 > timeout_s:
                return d
            time.sleep(20)

    def scale(self, deployment_id: str, replicas: int) -> dict:
        d = self.get_deployment(deployment_id)
        body = {"baseModel": d["baseModel"], "minReplicaCount": replicas, "maxReplicaCount": replicas}
        out = self._req("PATCH", f"deployments/{deployment_id}", json=body)
        if replicas == 0:
            _log_gpu_time(deployment_id, "scaled to zero")
        else:
            _record(deployment_id, created=_now_iso())
        return out

    def delete_deployment(self, deployment_id: str) -> dict:
        out = self._req("DELETE", f"deployments/{deployment_id}", params={"ignoreChecks": "true"})
        _log_gpu_time(deployment_id, "deleted")
        _record(deployment_id, deleted=_now_iso())
        return out

    def load_lora(self, deployment_id: str, lora_model: str, replace_merged: bool = True) -> dict:
        body = {"model": self.model_name(lora_model), "deployment": self.deployment_name(deployment_id)}
        params = {"replaceMergedAddon": "true"} if replace_merged else {}
        return self._req("POST", "deployedModels", params=params, json=body)

    def deployed_models(self) -> list[dict]:
        return self._req("GET", "deployedModels").get("deployedModels", [])

    def wait_lora(self, deployment_id: str, lora_model: str, timeout_s: int = 20 * 60) -> Optional[dict]:
        want, dep = self.model_name(lora_model), self.deployment_name(deployment_id)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            for m in self.deployed_models():
                if m.get("model") == want and m.get("deployment") == dep:
                    print(f"[{time.strftime('%H:%M:%S')}] lora {lora_model}: {m.get('state')}", flush=True)
                    if m.get("state") == "DEPLOYED":
                        return m
            time.sleep(15)
        return None

    # ---------------- inference ----------------
    def route(self, deployment_id: str, model: Optional[str] = None) -> str:
        """Model string for chat completions: '<model>#<deployment>' routes to our on-demand deployment."""
        return f"{self.model_name(model) if model else BASE_MODEL}#{self.deployment_name(deployment_id)}"

    def chat(self, model: str, messages: list[dict], max_tokens: int = 600, temperature: float = 0.0,
             timeout: float = 120.0, **extra) -> dict:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
        r = self.c.post(f"{INFER}/chat/completions", json=body, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"chat -> {r.status_code}: {r.text[:400]}")
        return r.json()

    def list_all(self) -> dict:
        return {"deployments": self._req("GET", "deployments").get("deployments", []),
                "deployedModels": self.deployed_models(),
                "jobs": self._req("GET", "supervisedFineTuningJobs").get("supervisedFineTuningJobs", [])}


# ---------------- local state and spend ----------------

def _load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"deployments": {}}


def _record(deployment_id: str, **fields) -> None:
    st = _load_state()
    rec = st["deployments"].setdefault(deployment_id, {"events": []})
    rec.update({k: v for k, v in fields.items() if k != "created"})
    if "created" in fields:
        rec["up_since"] = fields["created"]
    rec["events"].append({"t": _now_iso(), **fields})
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=2) + "\n")


def _log_gpu_time(deployment_id: str, why: str) -> None:
    """Log GPU time from the last (re)creation to now as spend (conservative: counts provisioning too)."""
    st = _load_state()
    rec = st["deployments"].get(deployment_id, {})
    since = rec.get("up_since")
    if not since:
        return
    hours = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(since)).total_seconds() / 3600
    log_spend("fireworks:h200-on-demand", int(hours * 3600), hours * GPU_USD_PER_HOUR,
              note=f"deployment {deployment_id} {why}; {hours * 60:.1f} GPU-min")
    rec.pop("up_since", None)
    rec.setdefault("gpu_minutes", 0.0)
    rec["gpu_minutes"] = round(rec["gpu_minutes"] + hours * 60, 2)
    STATE_FILE.write_text(json.dumps(st, indent=2) + "\n")


def gpu_minutes_used() -> float:
    st = _load_state()
    total = 0.0
    for rec in st["deployments"].values():
        total += rec.get("gpu_minutes", 0.0)
        if rec.get("up_since"):
            total += (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(rec["up_since"])).total_seconds() / 60
    return total


# ---------------- warm-up ----------------

def warmup(fw: Fireworks, deployment_id: str, lora: Optional[str], tries: int = 40) -> float:
    """Send one real intent request until it succeeds; returns its latency in ms."""
    from mhd.intent_spec import GLIMMER_SYSTEM_PROMPT
    model = fw.route(deployment_id, lora)
    msgs = [{"role": "system", "content": GLIMMER_SYSTEM_PROMPT},
            {"role": "user", "content": "follow the person in the blue shirt"}]
    for k in range(tries):
        t0 = time.perf_counter()
        try:
            out = fw.chat(model, msgs, max_tokens=300)
            ms = (time.perf_counter() - t0) * 1000
            msg = out["choices"][0]["message"]
            print(f"warm: {ms:.0f} ms -> {str(msg.get('content'))[:160]!r}", flush=True)
            return ms
        except Exception as e:
            print(f"warmup try {k}: {str(e)[:200]}", flush=True)
            time.sleep(15)
    raise RuntimeError("deployment did not answer")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("upload"); s.add_argument("--file", type=Path, required=True); s.add_argument("--dataset-id", required=True)
    s = sub.add_parser("sft")
    s.add_argument("--dataset", required=True); s.add_argument("--output-model", required=True)
    s.add_argument("--job-id"); s.add_argument("--epochs", type=int, default=1); s.add_argument("--lora-rank", type=int, default=8)
    s.add_argument("--lr", type=float); s.add_argument("--max-context", type=int, default=2048)
    s.add_argument("--eval-dataset"); s.add_argument("--batch-size-samples", type=int)
    s = sub.add_parser("job"); s.add_argument("--job", required=True); s.add_argument("--wait", action="store_true")
    for name in ("deploy", "load-lora", "warmup", "morning", "down", "delete"):
        s = sub.add_parser(name)
        s.add_argument("--deployment-id", default=DEFAULT_DEPLOYMENT)
        s.add_argument("--lora", help="LoRA model id (accounts/.../models/<id> or just <id>)")
        s.add_argument("--plain", action="store_true",
                       help="deploy: base model only, min 0 / max 1, scale to zero after 10 idle minutes")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    fw = Fireworks()

    if a.cmd == "upload":
        print(json.dumps(fw.upload_dataset(a.dataset_id, a.file), indent=1)[:1500])
    elif a.cmd == "sft":
        job_id = a.job_id or f"{a.output_model}-job"
        j = fw.create_sft(job_id, a.dataset, a.output_model, epochs=a.epochs, lora_rank=a.lora_rank,
                          learning_rate=a.lr, max_context=a.max_context, eval_dataset=a.eval_dataset,
                          batch_size_samples=a.batch_size_samples)
        print(json.dumps({k: j.get(k) for k in ("name", "state", "epochs", "loraRank", "learningRate",
                                                "maxContextLength", "outputModel", "estimatedCost")}, indent=1))
    elif a.cmd == "job":
        j = fw.wait_sft(a.job) if a.wait else fw.get_sft(a.job)
        print(json.dumps({k: j.get(k) for k in ("name", "state", "status", "jobProgress", "epochs", "loraRank",
                                                "learningRate", "outputModel", "createTime", "completedTime",
                                                "estimatedCost")}, indent=1))
    elif a.cmd == "deploy":
        fw.create_deployment(a.deployment_id, hot_reload=not a.plain)
        if a.plain:
            # min 0: the first request wakes it up; warmup() retries through the 503 DEPLOYMENT_SCALING_UP
            time.sleep(10)
            warmup(fw, a.deployment_id, None, tries=120)
        fw.wait_deployment(a.deployment_id)
        if a.lora:
            fw.load_lora(a.deployment_id, a.lora)
            fw.wait_lora(a.deployment_id, a.lora)
        print("model string:", fw.route(a.deployment_id, a.lora).replace(fw.account, "$FIREWORKS_ACCOUNT_ID"))
    elif a.cmd == "load-lora":
        fw.load_lora(a.deployment_id, a.lora)
        fw.wait_lora(a.deployment_id, a.lora)
    elif a.cmd == "warmup":
        warmup(fw, a.deployment_id, a.lora)
    elif a.cmd == "morning":
        if not a.lora:
            sys.exit("--lora is required")
        try:
            d = fw.get_deployment(a.deployment_id)
            if d.get("state") in ("DELETED", "DELETING"):
                raise RuntimeError("deleted")
            if int(d.get("maxReplicaCount") or 0) == 0:
                fw.scale(a.deployment_id, 1)
        except RuntimeError:
            fw.create_deployment(a.deployment_id)
        fw.wait_deployment(a.deployment_id)
        fw.load_lora(a.deployment_id, a.lora)
        fw.wait_lora(a.deployment_id, a.lora)
        warmup(fw, a.deployment_id, a.lora)
        print("agent model string:", fw.route(a.deployment_id, a.lora).replace(fw.account, "$FIREWORKS_ACCOUNT_ID"))
    elif a.cmd == "down":
        fw.scale(a.deployment_id, 0)
        print("scaled to zero:", a.deployment_id)
    elif a.cmd == "delete":
        fw.delete_deployment(a.deployment_id)
        print("deleted:", a.deployment_id)
    elif a.cmd == "list":
        allv = fw.list_all()
        for d in allv["deployments"]:
            print("deployment", d.get("name", "").split("/")[-1], d.get("state"), "replicas", d.get("replicaCount"),
                  "min/max", d.get("minReplicaCount"), d.get("maxReplicaCount"), d.get("baseModel", "").split("/")[-1])
        for m in allv["deployedModels"]:
            print("deployedModel", m.get("model", "").split("/")[-1], "on", m.get("deployment", "").split("/")[-1], m.get("state"))
        for j in allv["jobs"]:
            print("sft job", j.get("name", "").split("/")[-1], j.get("state"), j.get("outputModel", "").split("/")[-1])
        print(f"GPU minutes recorded by this workstream: {gpu_minutes_used():.1f}")


if __name__ == "__main__":
    main()
