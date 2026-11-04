"""Morning microphone check (the macOS Microphone permission cannot be granted overnight).

    $MHD_ML_PY -m mhd_ml.agent.mic_check --list                 # input devices
    $MHD_ML_PY -m mhd_ml.agent.mic_check --seconds 20           # levels + live safety listener
    $MHD_ML_PY -m mhd_ml.agent.mic_check --seconds 20 --rate 16000 --device "AirPods"
    $MHD_ML_PY -m mhd_ml.agent.mic_check --ptt 4                # record 4 s, transcribe locally and with Meta

What to look for:
- "ALL ZEROS": macOS is not giving this terminal app the microphone. System Settings > Privacy &
  Security > Microphone > enable the terminal (or Cursor/VS Code) that runs the agent, then restart it.
- Levels: speech at normal voice should peak around -30 to -15 dBFS. Propeller noise: note the level
  with the drone hovering; the safety listener needs speech well above it (use a headset mic if the
  noise is within 10 dB of your voice).
- Say "stop" and "land" during the check: each should print a SAFETY line within about a second.
Nothing is sent to the backend.
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time

import numpy as np

from .audio import MicHub


def list_devices() -> None:
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "*" if i == sd.default.device[0] else " "
            print(f"{mark} [{i}] {d['name']}  ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)")


def run_levels_and_listener(seconds: float, rate: int, device) -> int:
    from .safety_listener import SafetyListener

    hub = MicHub(sample_rate=rate, device=device)
    lst = SafetyListener(on_transcript=lambda kind, text: print(f"   heard ({kind}): {text}", flush=True))
    q_safety = hub.subscribe(maxsize=500)
    q_level = hub.subscribe(maxsize=500)
    stop = threading.Event()
    th = threading.Thread(target=lst.run_queue, args=(q_safety, rate, stop), daemon=True)
    hub.start()
    th.start()
    print(f"listening {seconds:.0f} s at {rate} Hz; say a sentence, then 'stop', then 'land'", flush=True)
    t0 = time.monotonic()
    buf, peak_db, zeros = [], -120.0, True
    try:
        while time.monotonic() - t0 < seconds:
            try:
                x = q_level.get(timeout=0.5)
            except queue.Empty:
                print("   (no audio callbacks: is the device busy or unplugged?)", flush=True)
                continue
            buf.append(x)
            if sum(len(b) for b in buf) >= rate // 2:
                y = np.concatenate(buf)
                buf = []
                rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
                db = 20 * np.log10(rms)
                peak_db = max(peak_db, db)
                zeros = zeros and not np.any(y)
                bar = "#" * max(0, int((db + 70) / 2))
                print(f"   {db:6.1f} dBFS {bar}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        hub.stop()
    if zeros:
        print("ALL ZEROS: no microphone permission for this terminal app (see the module docstring).")
        return 2
    print(f"peak level {peak_db:.1f} dBFS; safety hits: {[(h.command, h.word) for h in lst.hits]}")
    return 0


def run_ptt(seconds: float, rate: int, device) -> int:
    from .asr_local import LocalASR
    from .asr_meta import MetaRealtimeASR, RealtimeConfig

    hub = MicHub(sample_rate=rate, device=device)
    q = hub.subscribe(maxsize=1000)
    hub.start()
    print(f"recording {seconds:.0f} s: say a mission, e.g. 'follow the person in the blue shirt'", flush=True)
    time.sleep(seconds)
    hub.stop()
    chunks = []
    while not q.empty():
        chunks.append(q.get_nowait())
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if not np.any(x):
        print("ALL ZEROS: no microphone permission for this terminal app.")
        return 2
    print("local Omnilingual:", LocalASR().transcribe(x, rate).text)
    if rate in (16000, 24000):
        try:
            t = asyncio.run(MetaRealtimeASR(RealtimeConfig(sample_rate=rate)).transcribe_array(x, rate, realtime=False))
            print("Muse Voice Transcribe:", t.text)
        except Exception as e:
            print("Muse Voice Transcribe failed:", e)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--device", default=None, help="input device name or index (see --list)")
    ap.add_argument("--ptt", type=float, default=0.0, help="record this many seconds and transcribe")
    args = ap.parse_args(argv)
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.list:
        list_devices()
        return 0
    if args.ptt:
        return run_ptt(args.ptt, args.rate, device)
    return run_levels_and_listener(args.seconds, args.rate, device)


if __name__ == "__main__":
    raise SystemExit(main())
