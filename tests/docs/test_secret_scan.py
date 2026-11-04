"""scripts/secret_scan.py on throwaway git repos: it must find planted secrets without printing them."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCAN = Path(__file__).resolve().parents[2] / "scripts" / "secret_scan.py"
FAKE_KEY = "mk_" + "Q7v" * 14            # 45 chars, built at runtime so this file holds no literal key
FAKE_ID = "acct" + "9z8y7x"


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def make_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    repo.mkdir()
    run(["git", "init", "-q"], repo)
    run(["git", "config", "user.email", "t@example.com"], repo)
    run(["git", "config", "user.name", "t"], repo)
    (repo / "README.md").write_text("hello\n")
    run(["git", "add", "."], repo)
    run(["git", "commit", "-q", "-m", "init"], repo)
    return repo


def scan(repo: Path, env: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCAN), "--tree", str(repo), "--env", str(env), *extra],
                          capture_output=True, text=True, env={**os.environ, "MHD_HOME": str(repo)})


def write_env(tmp: Path) -> Path:
    env = tmp / "dotenv"
    env.write_text(f"MODEL_API_KEY={FAKE_KEY}\nFIREWORKS_ACCOUNT_ID={FAKE_ID}\n"
                   "BASETEN_API_KEY=          # optional fallback for training/serving\n")
    return env


def test_clean_repo_passes(tmp_path):
    repo, env = make_repo(tmp_path), write_env(tmp_path)
    r = scan(repo, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: clean" in r.stdout
    assert "2 values" in r.stdout          # the commented-out Baseten line is not a value


def test_secret_in_file_and_history_fails_without_printing_it(tmp_path):
    repo, env = make_repo(tmp_path), write_env(tmp_path)
    (repo / "config.py").write_text(f'KEY = "{FAKE_KEY}"\n')
    run(["git", "add", "."], repo)
    run(["git", "commit", "-q", "-m", "oops"], repo)
    (repo / "config.py").write_text("KEY = None\n")          # removed again, still in history
    run(["git", "commit", "-q", "-am", "fix"], repo)
    r = scan(repo, env)
    assert r.returncode == 1
    assert "FAIL value of MODEL_API_KEY found in git history" in r.stdout
    assert FAKE_KEY not in r.stdout and FAKE_KEY not in r.stderr
    assert scan(repo, env, "--no-history").returncode == 0   # the working tree alone is clean now


def test_private_value_warns_and_env_file_fails(tmp_path):
    repo, env = make_repo(tmp_path), write_env(tmp_path)
    (repo / "notes.md").write_text(f"account {FAKE_ID}\n")
    (repo / ".env").write_text("X=1\n")
    run(["git", "add", "-f", "."], repo)
    run(["git", "commit", "-q", "-m", "notes"], repo)
    r = scan(repo, env)
    assert "WARN value of FIREWORKS_ACCOUNT_ID found in notes.md:1" in r.stdout
    assert "FAIL .env: an .env file is tracked" in r.stdout
    assert FAKE_ID not in r.stdout
    assert r.returncode == 1


def test_generic_token_pattern_is_masked(tmp_path):
    repo, env = make_repo(tmp_path), write_env(tmp_path)
    token = "sk-" + "a1B2" * 8
    (repo / "client.py").write_text(f'headers = {{"Authorization": "Bearer {token}"}}\n')
    r = scan(repo, env, "--no-history")
    assert r.returncode == 1
    assert "sk- token at client.py:1" in r.stdout
    assert token not in r.stdout
