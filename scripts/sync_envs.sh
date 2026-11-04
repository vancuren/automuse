#!/usr/bin/env bash
# Creates the two virtualenvs inside the project (core at ./.venv, ml at ./ml/.venv).
set -u
source "$(dirname "$0")/../env.sh"
cd "$MHD_HOME" && echo "[$(date +%H:%M:%S)] core sync" && uv sync 2>&1 | tail -5
cd "$MHD_HOME/ml" && echo "[$(date +%H:%M:%S)] ml sync" && uv sync 2>&1 | tail -5
echo "[$(date +%H:%M:%S)] SYNC DONE"
