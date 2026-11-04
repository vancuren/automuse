"""Tiny RC sender for poking the twin by hand: X69 control packets at 25 Hz on loopback.

    $MHD_PY scripts/twin_rc.py --port 23520 takeoff 4 hover 3 yaw=200:2 pitch=170:2 land 3
Each step is NAME[=COUNT][:SECONDS]; flags (takeoff, land, estop) go out in one packet.
"""
import argparse
import socket
import time

from mhd.twin import protocol as P

FLAGS = {"takeoff": P.FLAG_TAKEOFF, "land": P.FLAG_LAND, "estop": P.FLAG_ESTOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23458)
    ap.add_argument("steps", nargs="+")
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    steps = list(a.steps)
    i = 0
    while i < len(steps):
        tok = steps[i]
        dur = 0.04
        if i + 1 < len(steps) and steps[i + 1].replace(".", "", 1).isdigit():
            dur = float(steps[i + 1]); i += 1
        name, _, rest = tok.partition("=")
        count, _, sec = rest.partition(":")
        if sec:
            dur = float(sec)
        sticks = {}
        flags = FLAGS.get(name, 0)
        if name in ("yaw", "pitch", "roll", "throttle"):
            sticks[name] = int(count)
        t_end = time.monotonic() + dur
        first = True
        while time.monotonic() < t_end or first:
            sock.sendto(P.encode_control(flags1=flags if first else 0, **sticks), ("127.0.0.1", a.port))
            first = False
            time.sleep(0.04)
        i += 1


if __name__ == "__main__":
    main()
