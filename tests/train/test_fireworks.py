"""Tests for training/fireworks.py against a mocked Fireworks API (no network, no spend)."""
import json

import httpx
import pytest

from training import common
from training import fireworks as fwm


@pytest.fixture
def fw(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setattr(fwm, "STATE_FILE", tmp_path / "fw_state.json")
    monkeypatch.setattr(common, "SPEND_LOG", tmp_path / "spend.jsonl")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, dict(request.url.params), body,
                      request.headers.get("authorization")))
        if request.method == "GET" and request.url.path.endswith("/deployments/d1"):
            return httpx.Response(200, json={"name": "accounts/acct/deployments/d1", "state": "READY",
                                             "replicaCount": 1, "baseModel": fwm.BASE_MODEL,
                                             "maxReplicaCount": 1})
        return httpx.Response(200, json={"ok": True})

    client = fwm.Fireworks()
    client.c = httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer test-key"})
    client.calls = calls
    return client


def test_create_hot_reload_deployment(fw):
    fw.create_deployment("d1")
    method, path, params, body, auth = fw.calls[-1]
    assert (method, path) == ("POST", "/v1/accounts/acct/deployments")
    assert params == {"deploymentId": "d1"}
    assert body["minReplicaCount"] == body["maxReplicaCount"] == 1
    assert body["enableHotReloadLatestAddon"] is True
    assert body["deploymentShape"] == fwm.H200_SHAPE
    assert auth == "Bearer test-key"
    state = json.loads(fwm.STATE_FILE.read_text())
    assert "up_since" in state["deployments"]["d1"]


def test_create_plain_deployment_scales_to_zero(fw):
    fw.create_deployment("d1", hot_reload=False)
    body = fw.calls[-1][3]
    assert body["minReplicaCount"] == 0 and body["maxReplicaCount"] == 1
    assert body["autoscalingPolicy"]["scaleToZeroWindow"] == "600s"
    assert "enableHotReloadLatestAddon" not in body


def test_load_lora_and_route(fw):
    fw.load_lora("d1", "glimmer-intent-v1")
    method, path, params, body, _ = fw.calls[-1]
    assert path == "/v1/accounts/acct/deployedModels" and params == {"replaceMergedAddon": "true"}
    assert body == {"model": "accounts/acct/models/glimmer-intent-v1", "deployment": "accounts/acct/deployments/d1"}
    assert fw.route("d1") == f"{fwm.BASE_MODEL}#accounts/acct/deployments/d1"
    assert fw.route("d1", "glimmer-intent-v1") == "accounts/acct/models/glimmer-intent-v1#accounts/acct/deployments/d1"


def test_scale_down_and_delete_log_gpu_time(fw):
    fw.create_deployment("d1")
    fw.scale("d1", 0)
    patch = [c for c in fw.calls if c[0] == "PATCH"][-1]
    assert patch[3]["minReplicaCount"] == 0 and patch[3]["maxReplicaCount"] == 0
    spend = [json.loads(l) for l in common.SPEND_LOG.read_text().splitlines()]
    assert spend[-1]["service"] == "fireworks:h200-on-demand" and spend[-1]["ws"] == "train"
    fw.scale("d1", 1)
    fw.delete_deployment("d1")
    dele = [c for c in fw.calls if c[0] == "DELETE"][-1]
    assert dele[1] == "/v1/accounts/acct/deployments/d1" and dele[2] == {"ignoreChecks": "true"}
    st = json.loads(fwm.STATE_FILE.read_text())["deployments"]["d1"]
    assert "deleted" in st and "up_since" not in st


def test_sft_body(fw):
    fw.create_sft("job1", "ds1", "glimmer-intent-v1", epochs=2, lora_rank=16, max_context=2048)
    method, path, params, body, _ = fw.calls[-1]
    assert path == "/v1/accounts/acct/supervisedFineTuningJobs" and params == {"supervisedFineTuningJobId": "job1"}
    assert body["baseModel"] == "accounts/fireworks/models/muse-glimmer-30b"
    assert body["dataset"] == "accounts/acct/datasets/ds1"
    assert body["outputModel"] == "accounts/acct/models/glimmer-intent-v1"
    assert body["loraRank"] == 16 and body["epochs"] == 2


def test_no_key_in_errors(fw):
    def handler(request):
        return httpx.Response(403, json={"code": 9, "message": "dataset uploads are temporarily disabled"})
    fw.c = httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer test-key"})
    with pytest.raises(RuntimeError) as e:
        fw.get_deployment("d1")
    assert "test-key" not in str(e.value) and "403" in str(e.value)


def test_live_merge_deployment_body(fw):
    fw.create_deployment("d1", hot_reload=False, keep_warm=True, base_model=fw.model_name("glimmer-intent-v1"),
                         shape="default")
    body = fw.calls[-1][3]
    assert body["baseModel"] == "accounts/acct/models/glimmer-intent-v1"
    assert body["deploymentShape"] == "default"
    assert body["minReplicaCount"] == body["maxReplicaCount"] == 1
    assert "enableHotReloadLatestAddon" not in body
