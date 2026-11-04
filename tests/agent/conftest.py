"""Shared fixtures for the agent tests. Run with the ML interpreter:

    cd $MHD_TREE && source env.sh && $MHD_ML_PY -m pytest tests/agent -q

Ports (agent workstream only): fake backend 8400-8409, mock LLM 8480-8484, fake Voice Transcribe
8485-8489. Tests that call the real Meta API are opt-in: MHD_REAL_API=1 (they need MODEL_API_KEY).
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def audio_manifest():
    """Generate (once) and return the `say` test audio manifest."""
    from mhd_ml.agent.make_test_audio import generate

    return generate()


@pytest.fixture(scope="session")
def audio_items(audio_manifest):
    from mhd_ml.agent.make_test_audio import items

    return {g: items(g) for g in ("safety", "neutral", "commands")}


@pytest.fixture(scope="session")
def recognizer():
    from mhd_ml.agent.asr_local import get_shared_recognizer

    return get_shared_recognizer()
