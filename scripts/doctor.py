"""Environment check. Run: source env.sh && $MHD_PY scripts/doctor.py [--ml]
Fails loudly on anything that would cost hours later. Extend it as components land."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


home, tree = os.environ.get("MHD_HOME"), os.environ.get("MHD_TREE")
check("env.sh sourced", bool(home and tree), f"MHD_HOME={home} MHD_TREE={tree}")
for var in ["UV_CACHE_DIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "npm_config_cache",
            "PLAYWRIGHT_BROWSERS_PATH", "MHD_MODELS", "MHD_TOOLS"]:
    v = os.environ.get(var, "")
    check(f"{var} inside project", bool(home) and v.startswith(home or "\0"), v)
check("python 3.13", sys.version_info[:2] == (3, 13), sys.version.split()[0])
import cv2  # noqa: E402
check("opencv contrib trackers", hasattr(cv2, "TrackerCSRT_create") and hasattr(cv2, "TrackerKCF_create"), cv2.__version__)
ff = shutil.which("ffmpeg")
check("ffmpeg present", bool(ff), ff or "")
if ff:
    out = subprocess.run([ff, "-hide_banner", "-h", "encoder=hevc_videotoolbox"], capture_output=True, text=True).stdout
    check("hevc_videotoolbox encoder", "hevc_videotoolbox" in out)
mtx = os.path.join(os.environ.get("MHD_TOOLS", ""), "mediamtx", "mediamtx")
check("mediamtx binary", os.access(mtx, os.X_OK), mtx)
models = os.environ.get("MHD_MODELS", "")
for rel in ["sam3.1-bf16/config.json", "sherpa/silero_vad.onnx", "opencv/object_tracking_vittrack_2023sep.onnx",
            "mms-tts-eng/config.json"]:
    check(f"model {rel}", os.path.exists(os.path.join(models, rel)))
gl = os.path.join(models, "Muse-Glimmer-30B-4bit")
if os.path.isdir(gl):
    want = [l.strip() for l in open(os.path.join(gl, ".files")) if l.strip()] if os.path.exists(os.path.join(gl, ".files")) else []
    have = [f for f in want if os.path.exists(os.path.join(gl, f))]
    check("Glimmer 4-bit complete", bool(want) and len(have) == len(want), f"{len(have)}/{len(want)} files")
if "--ml" in sys.argv:
    import mlx.core as mx  # noqa: E402
    check("mlx on GPU", "gpu" in str(mx.default_device()), str(mx.default_device()))
    import mlx_vlm.models.sam3_1  # noqa: F401,E402
    check("mlx_vlm sam3_1 importable", True)
    import sherpa_onnx  # noqa: F401,E402
    check("sherpa_onnx importable", True)
print("DOCTOR", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
