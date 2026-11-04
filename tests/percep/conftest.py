"""Shared fixtures for perception tests. Everything runs offline: no network, no GPU."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(__file__).parent / "data"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A private spend ledger so tests never touch $MHD_HOME/logs/spend.jsonl."""
    path = tmp_path / "spend.jsonl"
    monkeypatch.setenv("MHD_HOME", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)
    return tmp_path / "logs" / "spend.jsonl"
