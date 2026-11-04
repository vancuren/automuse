# Source me: `source env.sh` (bash or zsh). Keeps every cache, model, and tool inside the project.
# MHD_TREE = the checkout this file lives in (main repo or a worktree) -> code on PYTHONPATH.
# MHD_HOME = the main repo -> shared caches, models, tools, and the two virtualenvs.
if [ -n "${BASH_SOURCE:-}" ]; then _mhd_self="${BASH_SOURCE[0]}"; elif [ -n "${ZSH_VERSION:-}" ]; then _mhd_self="${(%):-%x}"; else _mhd_self="$0"; fi
export MHD_TREE="$(cd "$(dirname "$_mhd_self")" && pwd)"
export MHD_HOME="$(cd "$MHD_TREE" && dirname "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$MHD_TREE/.git")")"
unset _mhd_self
export UV_PYTHON=/opt/homebrew/bin/python3.13
export UV_PYTHON_DOWNLOADS=never
export UV_CACHE_DIR="$MHD_HOME/.cache/uv"
export HF_HOME="$MHD_HOME/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$MHD_HOME/.cache/torch"
export XDG_CACHE_HOME="$MHD_HOME/.cache/xdg"
export MPLCONFIGDIR="$MHD_HOME/.cache/matplotlib"
export LLAMA_CACHE="$MHD_HOME/.cache/llama"
export PIP_CACHE_DIR="$MHD_HOME/.cache/pip"
export npm_config_cache="$MHD_HOME/.cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$MHD_HOME/.cache/ms-playwright"
export MHD_MODELS="$MHD_HOME/.cache/models"
export MHD_TOOLS="$MHD_HOME/tools"
export MHD_LOGS="$MHD_TREE/logs"
export TMPDIR="$MHD_HOME/.cache/tmp"   # pytest tmp_path, ffmpeg, tempfile: all inside the project
export MHD_PY="$MHD_HOME/.venv/bin/python"        # core: backend, governor, twin, training data
export MHD_ML_PY="$MHD_HOME/ml/.venv/bin/python"  # ml: perception (SAM 3.1 MLX), agent, Glimmer
export PYTHONPATH="$MHD_TREE/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MHD_HOME/.cache/tmp" "$MHD_HOME/.cache/locks" "$MHD_HOME/media/shared" "$MHD_MODELS" "$MHD_TOOLS" "$MHD_LOGS"
# Secrets (gitignored). Never echo them.
if [ -f "$MHD_HOME/.env" ]; then set -a; . "$MHD_HOME/.env"; set +a; fi
