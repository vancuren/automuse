"""Generate the agent's test audio with macOS `say` + `afconvert` (no microphone needed).

    $MHD_ML_PY -m mhd_ml.agent.make_test_audio            # writes $MHD_TREE/media/test_audio/
    $MHD_ML_PY -m mhd_ml.agent.make_test_audio --force    # regenerate

Layout (all mono 16-bit WAV, padded with 0.5 s of silence before and 1.0 s after the speech):
    safety/<id>_16k.wav   safety words, alone and mid-sentence (expected action in the manifest)
    neutral/<id>_16k.wav  20 sentences that must NOT trigger the safety listener
    commands/<id>_24k.wav and _16k.wav  push-to-talk mission commands (Voice Transcribe, intent)
    manifest.json         every file with its text, voice, sample rate, and expected labels
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .audio import load_wav, write_wav

# (id, text, expected safety action or None, voice)
SAFETY_CASES = [
    ("land_mid", "okay drone, please land now", "land", "Samantha"),
    ("land_single", "land", "land", "Samantha"),
    ("land_long", "I think we should land the drone now because the battery is getting low", "land", "Samantha"),
    ("land_daniel", "alright, land it right here", "land", "Daniel"),
    ("land_fred", "time to land", "land", "Fred"),
    ("stop_single", "stop", "hover", "Samantha"),
    ("stop_mid", "please stop following him", "hover", "Samantha"),
    ("stop_twice", "stop, stop", "hover", "Daniel"),
    ("hover_mid", "hover right there", "hover", "Samantha"),
    ("abort", "abort, abort", "land", "Samantha"),
    ("emergency", "emergency, land immediately", "land", "Samantha"),
    ("hold", "hold your position", "hover", "Samantha"),
    ("freeze", "freeze", "hover", "Samantha"),
    ("stop_then_land", "stop. now land", "land", "Samantha"),
]

# Must not trigger. Several contain near misses: island, landscape, hand, gold, shop, top, over, cover.
NEUTRAL_CASES = [
    ("n01", "follow the person in the blue shirt", "Samantha"),
    ("n02", "follow the red car", "Samantha"),
    ("n03", "track the brown dog please", "Samantha"),
    ("n04", "the weather is really nice today", "Samantha"),
    ("n05", "can you see the green ball over there", "Samantha"),
    ("n06", "the island looks beautiful from up here", "Samantha"),
    ("n07", "hand me the remote control", "Samantha"),
    ("n08", "she is wearing a gold necklace", "Samantha"),
    ("n09", "the coffee shop opens at nine", "Samantha"),
    ("n10", "we walked to the top of the hill", "Samantha"),
    ("n11", "take a picture of the landscape", "Samantha"),
    ("n12", "the cat jumped over the fence", "Samantha"),
    ("n13", "please cover the camera lens", "Samantha"),
    ("n14", "the children are playing in the park", "Daniel"),
    ("n15", "this is a demo of the follow camera", "Daniel"),
    ("n16", "look at the person with the yellow umbrella", "Daniel"),
    ("n17", "the battery is charged and ready", "Fred"),
    ("n18", "turn on the lights in the kitchen", "Fred"),
    ("n19", "nice shot, that looks great", "Samantha"),
    ("n20", "follow the woman in the green jacket", "Samantha"),
]

# Push-to-talk mission commands: (id, text, expected action, expected subject phrase or "")
COMMAND_CASES = [
    ("c01", "follow the person in the blue shirt", "follow", "person in blue shirt"),
    ("c02", "follow the red car", "follow", "red car"),
    ("c03", "track the brown dog", "follow", "brown dog"),
    ("c04", "stay close to the woman in the green jacket", "follow", "woman in green jacket"),
    ("c05", "follow the kid with the yellow balloon", "follow", "kid with yellow balloon"),
    ("c06", "hover right here", "hover", ""),
    ("c07", "land now", "land", ""),
    ("c08", "follow John", "refuse", ""),
    ("c09", "follow him", "clarify", ""),
    ("c10", "follow the man in the black hoodie from far away", "follow", "man in black hoodie"),
    ("c11", "follow the cyclist in the orange vest", "follow", "cyclist in orange vest"),
    ("c12", "track the white truck", "follow", "white truck"),
    ("c13", "who is that guy in the red hat", "refuse", ""),
    ("c14", "do a backflip", "refuse", ""),
    ("c15", "keep the camera on the girl in the pink dress", "follow", "girl in pink dress"),
    ("c16", "follow the gray cat", "follow", "gray cat"),
    ("c17", "stop", "hover", ""),
    ("c18", "follow the person wearing a yellow raincoat", "follow", "person in yellow raincoat"),
    ("c19", "can you follow the black dog", "follow", "black dog"),
    ("c20", "follow the guy with the green backpack", "follow", "man with green backpack"),
]

PAD_BEFORE_S = 0.5
PAD_AFTER_S = 1.0


def out_root() -> Path:
    tree = os.environ.get("MHD_TREE") or str(Path(__file__).resolve().parents[3])
    return Path(tree) / "media" / "test_audio"


def synth(text: str, voice: str, sample_rate: int, dest: Path, rate_wpm: int | None = None,
          noise_snr_db: float | None = None, seed: int = 0) -> Path:
    """`say` -> AIFF -> `afconvert` -> 16-bit mono WAV at sample_rate, padded with silence."""
    tmp = dest.parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    aiff = tmp / (dest.stem + ".aiff")
    raw = tmp / (dest.stem + ".wav")
    cmd = ["say", "-v", voice, "-o", str(aiff)]
    if rate_wpm:
        cmd += ["-r", str(rate_wpm)]
    subprocess.run(cmd + [text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", "-c", "1", str(aiff), str(raw)],
                   check=True)
    x, sr = load_wav(raw)
    x = np.concatenate([np.zeros(int(PAD_BEFORE_S * sr), np.float32), x,
                        np.zeros(int(PAD_AFTER_S * sr), np.float32)])
    if noise_snr_db is not None:
        rng = np.random.default_rng(seed)
        speech_rms = float(np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))) if np.any(np.abs(x) > 0.01) else 0.1
        x = x + rng.normal(0, speech_rms / (10 ** (noise_snr_db / 20)), len(x)).astype(np.float32)
    write_wav(dest, x, sr)
    aiff.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    return dest


def generate(force: bool = False) -> dict:
    root = out_root()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        m = json.loads(manifest_path.read_text())
        if all((root / it["path"]).exists() for it in m["items"]):
            return m
    items = []

    def add(group, cid, text, voice, sr, suffix="", **labels):
        rel = f"{group}/{cid}{suffix}_{sr // 1000}k.wav"
        synth(text, voice, sr, root / rel, rate_wpm=labels.pop("rate_wpm", None),
              noise_snr_db=labels.get("noise_snr_db"))
        items.append({"group": group, "id": cid + suffix, "text": text, "voice": voice,
                      "sample_rate": sr, "path": rel, **labels})

    for cid, text, action, voice in SAFETY_CASES:
        add("safety", cid, text, voice, 16000, expected=action)
    # Robustness variants: background noise at 20 dB SNR, a 24 kHz file (resampling path), fast speech.
    add("safety", "land_mid", "okay drone, please land now", "Samantha", 16000, "_noisy",
        expected="land", noise_snr_db=20.0)
    add("safety", "stop_mid", "please stop following him", "Samantha", 16000, "_noisy",
        expected="hover", noise_snr_db=20.0)
    add("safety", "land_mid", "okay drone, please land now", "Samantha", 24000, expected="land")
    add("safety", "land_fast", "land the drone right now", "Samantha", 16000, expected="land", rate_wpm=260)
    for cid, text, voice in NEUTRAL_CASES:
        add("neutral", cid, text, voice, 16000, expected=None)
    for cid, text, action, phrase in COMMAND_CASES:
        for sr in (24000, 16000):
            add("commands", cid, text, "Samantha", sr, expected_action=action, expected_subject=phrase)
    m = {"pad_before_s": PAD_BEFORE_S, "pad_after_s": PAD_AFTER_S, "items": items}
    manifest_path.write_text(json.dumps(m, indent=1))
    for d in root.glob("*/_tmp"):
        try:
            d.rmdir()
        except OSError:
            pass
    return m


def items(group: str) -> list[dict]:
    m = generate()
    root = out_root()
    return [dict(it, abspath=str(root / it["path"])) for it in m["items"] if it["group"] == group]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    m = generate(force=args.force)
    print(f"{len(m['items'])} files in {out_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
