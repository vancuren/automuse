"""Pre-flight checks for a real-drone session (PLAN.md sections 5 and 6, Session A steps 1-2).

Run it from the EXACT terminal app that will launch the backend: macOS grants the Local Network
permission per app, and loopback is exempt, so only the real drone shows the problem.

    source env.sh
    $MHD_PY scripts/preflight.py                         # real drone at 172.16.11.1, all checks
    $MHD_PY scripts/preflight.py --only env,wifi,routes,internet,localnet
    $MHD_PY scripts/preflight.py --only video            # ffprobe: fps, GOP, bitrate, second client
    $MHD_PY scripts/preflight.py --board lewei           # our unit: Lewei camera board at 192.168.0.1
    # tonight's test against the twin only (loopback; twin RTSP on 8650):
    $MHD_PY scripts/preflight.py --drone-ip 127.0.0.1 --rtsp-url rtsp://127.0.0.1:8650/live/ch00_1

Groups: env (doctor, both venvs), wifi, routes, internet, localnet (Local Network permission),
video (ffprobe + second client), system (AirDrop/AWDL, stray processes).

What it sends to the drone: one EMPTY UDP datagram to port 9 (discard), TCP connects to 554 and
the RTSP port, and RTSP reads. Never an X69 control packet. With --board lewei: a TCP connect to 7060
and two short video reads (start-video, heartbeat, stop-video only), never a flight-control packet. Run it BEFORE the backend: the video
checks open extra RTSP clients, and the X69 may allow only one.

Exit code 0 when nothing FAILed. Results also go to logs/preflight-<time>.json.
"""
from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

TREE = Path(os.environ.get("MHD_TREE") or Path(__file__).resolve().parents[1])
DRONE_IP = "172.16.11.1"
RTSP_PATH = "/live/ch00_1"
LEWEI_IP = "192.168.0.1"            # our unit's Lewei camera board: video TCP 7060, flight control UDP 50000
LEWEI_VIDEO_PORT = 7060
BOARDS = ("x69_lg", "lewei")
API_HOST = "api.meta.ai"
GROUPS = ("env", "wifi", "routes", "internet", "localnet", "video", "system")

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"


@dataclass
class Result:
    group: str
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- helpers

def run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timed out after {timeout:.0f} s"


def hardware_ports() -> dict[str, str]:
    """{'Wi-Fi': 'en0', 'iPhone USB': 'en7', ...} from networksetup."""
    _, out = run(["networksetup", "-listallhardwareports"])
    ports, name = {}, None
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and name:
            ports[name] = line.split(":", 1)[1].strip()
            name = None
    return ports


def parse_route(out: str) -> dict:
    """`route -n get X` output -> {'interface': 'en0', 'gateway': '...', 'destination': '...'}."""
    d = {}
    for line in out.splitlines():
        m = re.match(r"\s*(destination|gateway|interface|flags):\s*(.+?)\s*$", line)
        if m:
            d[m.group(1)] = m.group(2)
    return d


def route_get(host: str) -> dict:
    code, out = run(["route", "-n", "get", host])
    d = parse_route(out)
    if code != 0 and not d:
        d["error"] = out.strip().splitlines()[-1] if out.strip() else f"exit {code}"
    return d


def iface_ipv4(dev: str) -> Optional[str]:
    code, out = run(["ipconfig", "getifaddr", dev])
    ip = out.strip()
    return ip if code == 0 and re.match(r"^\d+\.\d+\.\d+\.\d+$", ip) else None


def wifi_ssid(dev: str) -> str:
    """SSID if macOS shows it (it is redacted unless the app has Location Services)."""
    _, out = run(["ipconfig", "getsummary", dev])
    m = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else ""


def service_order() -> list[str]:
    _, out = run(["networksetup", "-listnetworkserviceorder"])
    return [m.group(1).strip() for m in re.finditer(r"^\(\d+\)\s+(.+)$", out, re.M)]


def terminal_app() -> str:
    names = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm2", "vscode": "Visual Studio Code",
             "WarpTerminal": "Warp", "ghostty": "Ghostty", "WezTerm": "WezTerm", "tmux": "tmux (its parent app)"}
    tp = os.environ.get("TERM_PROGRAM", "")
    if tp:
        return names.get(tp, tp)
    # walk up the process tree looking for an .app bundle
    pid = os.getppid()
    for _ in range(12):
        code, out = run(["ps", "-o", "ppid=,comm=", "-p", str(pid)])
        if code != 0 or not out.strip():
            break
        ppid, comm = out.strip().split(None, 1)
        m = re.search(r"/([^/]+)\.app/", comm)
        if m:
            return m.group(1)
        pid = int(ppid)
    return "this terminal app"


def in_subnet24(ip: Optional[str], ref: str) -> bool:
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(f"{ref}/24", strict=False)
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------- checks

class Preflight:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.results: list[Result] = []
        self.loopback = ipaddress.ip_address(a.drone_ip).is_loopback
        self.lewei = getattr(a, "board", "x69_lg") == "lewei"
        self.subnet_x = ".".join(a.drone_ip.split(".")[:3]) + ".x"
        self.ports = hardware_ports()
        self.wifi_dev = a.wifi_device or self.ports.get("Wi-Fi", "en0")
        self.iphone_dev = next((d for n, d in self.ports.items() if "iphone" in n.lower()), None)
        self.app = terminal_app()

    def add(self, *args, **kw) -> Result:
        r = Result(*args, **kw)
        self.results.append(r)
        return r

    # ------------------------------------------------------------------ env
    def check_env(self) -> None:
        py = os.environ.get("MHD_PY", sys.executable)
        ml = os.environ.get("MHD_ML_PY", "")
        self.add("env", "terminal app", INFO, f"{self.app} (Local Network and Microphone are granted per app: "
                 f"launch the backend and the agent from {self.app})")
        for label, interp, extra in (("doctor (core venv)", py, []), ("doctor (ml venv)", ml, ["--ml"])):
            if extra and self.a.skip_ml:
                self.add("env", label, SKIP, "--skip-ml")
                continue
            if not interp or not os.access(interp, os.X_OK):
                self.add("env", label, FAIL, f"interpreter missing: {interp!r}",
                         "source env.sh first; if the venv is missing, run scripts/sync_envs.sh")
                continue
            code, out = run([interp, str(TREE / "scripts" / "doctor.py"), *extra], timeout=120)
            fails = [ln.strip() for ln in out.splitlines() if ln.startswith("[FAIL]")]
            summary = next((ln.strip() for ln in reversed(out.splitlines()) if ln.startswith("DOCTOR")), "")
            if code == 0:
                self.add("env", label, PASS, f"{summary or 'ok'} ({sum(ln.startswith('[ok]') for ln in out.splitlines())} checks)")
            else:
                self.add("env", label, FAIL, "; ".join(fails[:4]) or out.strip()[-300:],
                         f"run `{interp} scripts/doctor.py {' '.join(extra)}` and fix each [FAIL] line "
                         "(scripts/sync_envs.sh re-syncs the venvs)", {"fails": fails})

    # ------------------------------------------------------------------ wifi
    def check_wifi(self) -> None:
        dev = self.wifi_dev
        ip = iface_ipv4(dev)
        ssid = wifi_ssid(dev)
        shown = ssid if ssid and "redacted" not in ssid.lower() else "(hidden by macOS: the terminal app has no " \
            "Location Services permission; the address check below is what counts)"
        _, router = run(["ipconfig", "getoption", dev, "router"])
        router = router.strip() if re.match(r"^\d+\.\d+\.\d+\.\d+$", router.strip()) else ""
        data = {"device": dev, "ipv4": ip, "ssid": ssid, "router": router, "hardware_ports": self.ports}
        self.add("wifi", "Wi-Fi network name", INFO, f"{dev}: {shown}", data=data)
        if self.loopback:
            self.add("wifi", "on the drone's Wi-Fi", SKIP,
                     f"drone IP {self.a.drone_ip} is loopback (twin); {dev} has {ip or 'no IPv4 address'}")
            return
        if in_subnet24(ip, self.a.drone_ip):
            self.add("wifi", "on the drone's Wi-Fi", PASS,
                     f"{dev} has {ip} (drone subnet {self.a.drone_ip}/24)"
                     + (f"; DHCP router option {router}: the drone offers a default route, so the "
                        "service order matters" if router else "; no DHCP router option"), data=data)
        else:
            self.add("wifi", "on the drone's Wi-Fi", FAIL, f"{dev} has {ip or 'no IPv4 address'}, not {self.subnet_x}",
                     f"power the drone, then join its Wi-Fi on {dev} (menu bar Wi-Fi icon); turn off auto-join "
                     f"for other networks; wait for a {self.subnet_x} address", data)

    # ------------------------------------------------------------------ routes
    def check_routes(self) -> None:
        d = route_get(self.a.drone_ip)
        want = "lo0" if self.loopback else self.wifi_dev
        iface = d.get("interface")
        if iface == want:
            self.add("routes", f"route to {self.a.drone_ip}", PASS, f"via {iface}", data=d)
        else:
            self.add("routes", f"route to {self.a.drone_ip}", FAIL,
                     f"via {iface or d.get('error', '?')}, expected {want}",
                     f"join the drone's Wi-Fi on {want}; disconnect VPNs; check `route -n get {self.a.drone_ip}`",
                     d)
        t0 = time.monotonic()
        try:
            api_ip = socket.gethostbyname(API_HOST)
            dns_ms = (time.monotonic() - t0) * 1000
        except OSError as exc:
            self.add("routes", f"DNS {API_HOST}", FAIL, f"{exc}",
                     "no DNS: the internet path (iPhone USB tether) is down or below Wi-Fi in the service order")
            api_ip = None
            dns_ms = None
        r = route_get(API_HOST) if api_ip else {}
        iface = r.get("interface")
        order = service_order()
        data = {"route": r, "api_ip": api_ip, "dns_ms": dns_ms, "iphone_device": self.iphone_dev,
                "service_order": order}
        if api_ip is None:
            pass
        elif self.loopback:
            self.add("routes", f"route to {API_HOST}", PASS if iface else FAIL,
                     f"via {iface} (twin run: any interface is fine; real flights need the iPhone USB tether)",
                     "" if iface else "no route to the internet", data)
        elif iface and iface != self.wifi_dev and iface != "lo0":
            via = "iPhone USB" if iface == self.iphone_dev else "a non-Wi-Fi interface"
            self.add("routes", f"route to {API_HOST}", PASS, f"via {iface} ({via}), DNS {dns_ms:.0f} ms", data=data)
        else:
            self.add("routes", f"route to {API_HOST}", FAIL,
                     f"via {iface or r.get('error', '?')}: internet goes through the drone's Wi-Fi (no internet there)",
                     "plug in the iPhone, Settings > Personal Hotspot > Allow Others to Join, Trust this computer; "
                     "then System Settings > Network > (...) menu > Set Service Order: drag iPhone USB ABOVE Wi-Fi, "
                     "click OK; re-run", data)
        if not self.loopback:
            ip_svc = next((i for i, n in enumerate(order) if "iphone" in n.lower()), None)
            wifi_svc = next((i for i, n in enumerate(order) if n.lower().startswith("wi-fi")), None)
            if ip_svc is None:
                self.add("routes", "iPhone USB service", WARN, "no iPhone USB network service",
                         "connect the iPhone by USB with Personal Hotspot on (the service appears in System "
                         "Settings > Network)", {"service_order": order})
            elif wifi_svc is not None and ip_svc > wifi_svc:
                self.add("routes", "service order", WARN, "iPhone USB is below Wi-Fi: " + " > ".join(order),
                         "System Settings > Network > (...) > Set Service Order: drag iPhone USB above Wi-Fi",
                         {"service_order": order})
            else:
                self.add("routes", "service order", PASS, " > ".join(order[:4]), data={"service_order": order})

    # ------------------------------------------------------------------ internet
    def check_internet(self) -> None:
        import httpx
        key = os.environ.get("MODEL_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = f"https://{API_HOST}/v1/models"
        t0 = time.monotonic()
        try:
            r = httpx.get(url, headers=headers, timeout=8.0)
            ms = (time.monotonic() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            self.add("internet", f"HTTPS {API_HOST}", FAIL, f"{type(exc).__name__}: {exc}"[:200],
                     "no internet: is the iPhone hotspot on with cellular data, and iPhone USB above Wi-Fi? "
                     f"Try `curl -sI {url}`")
            return
        code = r.status_code
        data = {"status": code, "ms": round(ms), "with_key": bool(key)}
        if (key and code == 200) or (not key and code == 401):
            self.add("internet", f"HTTPS {API_HOST}", PASS,
                     f"GET /v1/models -> {code} in {ms:.0f} ms ({'key accepted' if key else 'no key loaded: 401 expected'})",
                     data=data)
        elif key and code in (401, 403):
            self.add("internet", f"HTTPS {API_HOST}", FAIL, f"reachable, but the key was refused ({code})",
                     "check MODEL_API_KEY in $MHD_HOME/.env (env.sh loads it); never paste it in a terminal log",
                     data)
        else:
            self.add("internet", f"HTTPS {API_HOST}", WARN, f"GET /v1/models -> {code} in {ms:.0f} ms", data=data)

    # ------------------------------------------------------------------ local network permission
    def check_localnet(self) -> None:
        host = self.a.drone_ip
        udp_err = None
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            payload = b""                           # EMPTY datagram: never an X69 control packet
            assert len(payload) == 0
            s.sendto(payload, (host, 9))            # port 9 = discard
        except OSError as exc:
            udp_err = exc.errno
        finally:
            s.close()
        ports = [LEWEI_VIDEO_PORT] if self.lewei else [554]
        rtsp_port = urlparse(self.rtsp_url()).port or 554
        if not self.lewei and rtsp_port not in ports:
            ports.append(rtsp_port)
        tcp = {}
        for port in ports:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.settimeout(self.a.tcp_timeout)
            t0 = time.monotonic()
            try:
                c.connect((host, port))
                tcp[port] = (0, (time.monotonic() - t0) * 1000)
            except socket.timeout:
                tcp[port] = (errno.ETIMEDOUT, None)
            except OSError as exc:
                tcp[port] = (exc.errno or -1, None)
            finally:
                c.close()
        data = {"udp_errno": udp_err, "tcp": {str(p): {"errno": e, "ms": ms} for p, (e, ms) in tcp.items()}}

        def describe(e):
            return "ok" if not e else f"errno {e} ({errno.errorcode.get(e, '?')})"
        detail = f"UDP {host}:9 {describe(udp_err)}; " + "; ".join(f"TCP {host}:{p} {describe(e)}" for p, (e, _) in tcp.items())
        errs = [udp_err] + [e for e, _ in tcp.values()]
        if errno.EHOSTUNREACH in errs:
            self.add("localnet", "Local Network permission", FAIL, detail + " -> errno 65 = permission missing",
                     f"System Settings > Privacy & Security > Local Network: turn ON {self.app}; then QUIT it "
                     f"(Cmd-Q), reopen, and re-run. Not listed? Run this check once from {self.app} and click "
                     "Allow on the prompt. (Only if the route check passed: otherwise errno 65 means no route.)",
                     data)
            return
        if udp_err is not None:
            self.add("localnet", "Local Network permission", FAIL, detail,
                     "an unexpected socket error: check the Wi-Fi link and re-run", data)
            return
        self.add("localnet", "Local Network permission", PASS, detail, data=data)
        if self.lewei:
            e = tcp[LEWEI_VIDEO_PORT][0]
            if e == 0:
                self.add("localnet", f"TCP {LEWEI_VIDEO_PORT} (Lewei video)", PASS,
                         f"connected in {tcp[LEWEI_VIDEO_PORT][1]:.0f} ms")
            else:
                self.add("localnet", f"TCP {LEWEI_VIDEO_PORT} (Lewei video)", FAIL, describe(e),
                         "the camera board does not answer: is the drone on (it powers off a few minutes after "
                         "video stops)? is the Mac on its Wi-Fi?", data)
            return
        e554 = tcp[554][0]
        if self.loopback:
            self.add("localnet", "TCP 554 (RTSP)", INFO, f"{describe(e554)} (twin: RTSP is on {rtsp_port})")
        elif e554 == 0:
            self.add("localnet", "TCP 554 (RTSP)", PASS, f"connected in {tcp[554][1]:.0f} ms")
        elif e554 == errno.ECONNREFUSED:
            self.add("localnet", "TCP 554 (RTSP)", FAIL, "connection refused: the drone has no RTSP server on 554",
                     "power-cycle the drone; if it persists the twin needs its H.265-over-UDP mode (PLAN 4.2)", data)
        else:
            self.add("localnet", "TCP 554 (RTSP)", FAIL, describe(e554),
                     "the drone does not answer: power, Wi-Fi association, distance", data)
        if rtsp_port != 554:
            e = tcp[rtsp_port][0]
            self.add("localnet", f"TCP {rtsp_port} (RTSP URL port)", PASS if e == 0 else FAIL, describe(e),
                     "" if e == 0 else f"nothing listens on {host}:{rtsp_port}: is the twin running?")

    # ------------------------------------------------------------------ video
    def rtsp_url(self) -> str:
        return self.a.rtsp_url or f"rtsp://{self.a.drone_ip}{RTSP_PATH}"

    def check_video(self) -> None:
        if self.lewei:
            return self.check_video_lewei()
        url = self.rtsp_url()
        if self.backend_running() and not self.loopback:
            self.add("video", "backend not running", WARN, f"something listens on 127.0.0.1:{self.a.backend_port}",
                     "stop the backend first: these checks open extra RTSP clients and the X69 may allow only one "
                     "(a second client could freeze the backend's video, which lands the drone)")
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            self.add("video", "ffprobe", FAIL, "ffprobe not found", "ffmpeg 8+ must be on PATH (brew ffmpeg)")
            return
        info = probe_streams(ffprobe, url, self.a.transport)
        if "error" in info:
            self.add("video", "RTSP stream", FAIL, f"{url}: {info['error']}"[:300],
                     "is the drone's camera streaming? is another client already connected (the backend, a "
                     f"player)? try `ffprobe -rtsp_transport tcp {url}`", info)
            return
        self.add("video", "RTSP stream", PASS, f"{info.get('codec')} {info.get('width')}x{info.get('height')} "
                 f"{info.get('profile') or ''} r_frame_rate {info.get('r_frame_rate')} over {info['transport']}",
                 data=info)
        st = packet_stats(ffprobe, url, info["transport"], self.a.probe_seconds)
        if "error" in st:
            self.add("video", "fps / keyframes / bitrate", FAIL, st["error"][:300], "re-run; check the stream", st)
        else:
            gop = st.get("keyframe_interval_frames")
            detail = (f"{st['fps']:.2f} fps, keyframe every {gop if gop is not None else '?'} frames "
                      f"({st.get('keyframe_interval_s') or '?'} s), {st['bitrate_kbps']:.0f} kb/s, "
                      f"{st['packets']} packets in {st['duration_s']:.1f} s")
            self.add("video", "fps / keyframes / bitrate", PASS if st["fps"] >= 10 else WARN, detail,
                     "" if st["fps"] >= 10 else "low frame rate: check light, distance and Wi-Fi interference", st)
        if self.a.skip_second_client:
            self.add("video", "second simultaneous client", SKIP, "--skip-second-client")
            return
        sc = second_client_test(url, info["transport"], self.a.second_client_seconds)
        verdicts = {
            "both": (PASS, "a second client works and the first keeps its frames"),
            "only_one": (WARN, "only ONE client: the second could not read while the first was connected"),
            "kicks_first": (WARN, "a new client KICKS the first one off"),
            "none": (FAIL, "neither client got frames"),
        }
        status, text = verdicts[sc["verdict"]]
        fix = {"only_one": "while the backend runs, record and measure from the backend (/ws/frames, /mjpeg), "
                           "never with a second player; run ffprobe/ffmpeg -c copy only with the backend stopped",
               "kicks_first": "NEVER open a second player while flying: it would freeze the backend's video "
                              "(autopilot neutral at 0.7 s, landing at 2 s). Record only with the backend stopped",
               "none": "the stream opened for ffprobe but not for OpenCV: check codec support and re-run"}.get(sc["verdict"], "")
        self.add("video", "second simultaneous client", status,
                 f"{text} (A: {sc['a_frames']} frames, {sc['a_frames_during_b']} while B ran; B: {sc['b_frames']})",
                 fix, sc)

    def check_video_lewei(self) -> None:
        """Two short reads through our own Lewei client (start-video, heartbeat, stop-video only).
        Measured on 2026-11-04: a second reader does not disturb the first on this board."""
        if str(TREE / "src") not in sys.path:
            sys.path.insert(0, str(TREE / "src"))
        from mhd.lewei.protocol import LeweiStream
        secs = max(2.0, self.a.probe_seconds)
        stats = {"a": {"frames": 0, "bytes": 0, "err": None, "during_b": 0}, "b": {"frames": 0, "bytes": 0, "err": None}}
        sample = bytearray()
        b_window = [None, None]

        def reader(key: str, delay: float, dur: float) -> None:
            time.sleep(delay)
            if key == "b":
                b_window[0] = time.monotonic()
            end = time.monotonic() + dur
            try:
                with LeweiStream(self.a.drone_ip, LEWEI_VIDEO_PORT, read_timeout=3.0) as cam:
                    for f in cam.frames():
                        st = stats[key]
                        st["frames"] += 1
                        st["bytes"] += len(f.data)
                        if key == "a":
                            if len(sample) < 3_000_000:
                                sample.extend(f.data)
                            if b_window[0] and not b_window[1]:
                                st["during_b"] += 1
                        if time.monotonic() >= end:
                            break
            except Exception as exc:  # noqa: BLE001 - reported below
                stats[key]["err"] = f"{type(exc).__name__}: {exc}"
            if key == "b":
                b_window[1] = time.monotonic()

        ta = threading.Thread(target=reader, args=("a", 0.0, secs), daemon=True)
        ta.start()
        tb = None
        if not self.a.skip_second_client:
            tb = threading.Thread(target=reader, args=("b", secs / 3, secs / 3), daemon=True)
            tb.start()
        ta.join(secs + 20)
        if tb:
            tb.join(secs + 20)
        a_st = stats["a"]
        if a_st["frames"] == 0:
            self.add("video", "Lewei video stream", FAIL, a_st["err"] or "no frames",
                     "power the drone (it switches off a few minutes after video stops) and join its Wi-Fi", stats)
            return
        fps, mbps = a_st["frames"] / secs, a_st["bytes"] * 8 / secs / 1e6
        info = {}
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            tmp = TREE / ".cache" / "tmp" / f"preflight_lewei_{os.getpid()}.h264"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(bytes(sample))
            code, out = run([ffprobe, "-v", "error", "-show_entries", "stream=codec_name,width,height,profile",
                             "-of", "json", str(tmp)])
            try:
                info = (json.loads(out[out.index("{"):]).get("streams") or [{}])[0] if code == 0 else {}
            except ValueError:
                info = {}
            tmp.unlink(missing_ok=True)
        ok = fps >= 10 and info.get("width")
        self.add("video", "Lewei video stream", PASS if ok else WARN,
                 f"{info.get('codec_name', '?')} {info.get('width', '?')}x{info.get('height', '?')} "
                 f"{info.get('profile') or ''}, {fps:.1f} fps, {mbps:.2f} Mbit/s over TCP {LEWEI_VIDEO_PORT}",
                 "" if ok else "low frame rate or undecodable sample: check light, distance and Wi-Fi", {**stats, **info})
        if tb is None:
            self.add("video", "second simultaneous client", SKIP, "--skip-second-client")
        else:
            b_st = stats["b"]
            both = b_st["frames"] > 0 and a_st["during_b"] > 0
            self.add("video", "second simultaneous client", PASS if both else WARN,
                     f"A: {a_st['frames']} frames ({a_st['during_b']} while B ran); B: {b_st['frames']} frames"
                     + (f"; B error {b_st['err']}" if b_st["err"] else ""),
                     "" if both else "a second reader disturbed the first: record only from the backend while flying",
                     stats)

    def backend_running(self) -> bool:
        with socket.socket() as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", self.a.backend_port)) == 0

    # ------------------------------------------------------------------ system
    def check_system(self) -> None:
        _, out = run(["ifconfig", "awdl0"])
        if "status: active" in out:
            self.add("system", "AirDrop / AWDL", WARN, "awdl0 is active (Wi-Fi hops off channel: ~100 ms spikes)",
                     "Control Center > AirDrop: Off; System Settings > General > AirDrop & Handoff: Handoff off. "
                     "Optional, until reboot: sudo ifconfig awdl0 down")
        else:
            self.add("system", "AirDrop / AWDL", PASS, "awdl0 not active")
        procs = {"twin server": r"mhd\.twin\.server", "twin renderer": r"mhd\.twin\.launch_renderer",
                 "Glimmer server (mlx_lm)": r"mlx_lm.*Muse-Glimmer", "backend (uvicorn web_server)": r"uvicorn web_server",
                 "perception": r"mhd_ml\.perception\.service", "agent": r"mhd_ml\.agent\.service"}
        found = {}
        for label, pat in procs.items():
            code, out = run(["pgrep", "-f", pat])
            pids = [p for p in out.split() if p.isdigit() and int(p) != os.getpid()]
            if pids:
                found[label] = pids
        if self.loopback:
            self.add("system", "running processes", INFO, ", ".join(f"{k} ({len(v)})" for k, v in found.items()) or "none",
                     data=found)
        else:
            twin = {k: v for k, v in found.items() if k.startswith("twin")}
            if twin:
                self.add("system", "twin stopped for real flights", WARN, ", ".join(twin),
                         "stop the twin and its Chrome renderer before flying (GPU and CPU load): Ctrl-C its launcher",
                         found)
            else:
                self.add("system", "twin stopped for real flights", PASS, "no twin processes")
            if found:
                self.add("system", "running processes", INFO, ", ".join(f"{k} ({len(v)})" for k, v in found.items()),
                         data=found)

    # ------------------------------------------------------------------ driver
    def run(self) -> int:
        order = [("env", self.check_env), ("wifi", self.check_wifi), ("routes", self.check_routes),
                 ("internet", self.check_internet), ("localnet", self.check_localnet),
                 ("video", self.check_video), ("system", self.check_system)]
        only = set(self.a.only.split(",")) if self.a.only else set(GROUPS)
        skip = set(self.a.skip.split(",")) if self.a.skip else set()
        for g in only | skip:
            if g not in GROUPS:
                raise SystemExit(f"unknown group {g!r}; groups: {', '.join(GROUPS)}")
        for group, fn in order:
            if group not in only or group in skip:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
                self.add(group, f"{group} check", FAIL, f"check crashed: {type(exc).__name__}: {exc}"[:300],
                         "re-run; report the traceback")
        return 1 if any(r.status == FAIL for r in self.results) else 0


# ---------------------------------------------------------------------------------------------- video helpers

def probe_streams(ffprobe: str, url: str, transport: str) -> dict:
    """Codec, size, profile, r_frame_rate. transport: tcp | udp | auto (tcp, then udp)."""
    tried = ["tcp", "udp"] if transport == "auto" else [transport]
    last = ""
    for tr in tried:
        code, out = run([ffprobe, "-v", "error", "-rtsp_transport", tr, "-timeout", "5000000", "-i", url,
                         "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,pix_fmt,bit_rate",
                         "-of", "json"], timeout=20)
        if code == 0:
            try:
                s = json.loads(out[out.index("{"):]).get("streams", [])
            except (ValueError, json.JSONDecodeError):
                s = []
            if s:
                s = s[0]
                return {"transport": tr, "codec": s.get("codec_name"), "profile": s.get("profile"),
                        "width": s.get("width"), "height": s.get("height"), "pix_fmt": s.get("pix_fmt"),
                        "r_frame_rate": s.get("r_frame_rate"), "avg_frame_rate": s.get("avg_frame_rate")}
        last = out.strip().splitlines()[-1] if out.strip() else f"exit {code}"
    return {"error": last, "transports_tried": tried}


def parse_packets(csv_text: str) -> dict:
    """ffprobe `-show_entries packet=pts_time,dts_time,flags,size -of csv=p=0` -> fps, GOP, bitrate."""
    pts, sizes, keys = [], [], []
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        t = parts[0] if parts[0] not in ("", "N/A") else parts[1]
        try:
            tv = float(t)
            size = int(parts[2])
        except ValueError:
            continue
        pts.append(tv)
        sizes.append(size)
        keys.append("K" in parts[3])
    if len(pts) < 3:
        return {"error": f"only {len(pts)} packets"}
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    t = [pts[i] for i in order]
    span = t[-1] - t[0]
    n = len(t)
    fps = (n - 1) / span if span > 0 else 0.0
    key_idx = [i for i, k in enumerate(keys) if k]
    gaps = [b - a for a, b in zip(key_idx, key_idx[1:])]
    key_t = [pts[i] for i in key_idx]
    gaps_s = [b - a for a, b in zip(key_t, key_t[1:])]
    out = {"packets": n, "duration_s": round(span, 3), "fps": round(fps, 3),
           "bitrate_kbps": round(sum(sizes) * 8 / span / 1000, 1) if span > 0 else 0.0,
           "keyframes": len(key_idx), "mean_packet_bytes": round(sum(sizes) / n),
           "keyframe_interval_frames": round(sum(gaps) / len(gaps), 1) if gaps else None,
           "keyframe_interval_s": round(sum(gaps_s) / len(gaps_s), 2) if gaps_s else None}
    if not gaps:
        out["note"] = (f"{len(key_idx)} keyframe(s) in {span:.1f} s: interval is longer than the probe window "
                       "(raise --probe-seconds)")
    return out


def packet_stats(ffprobe: str, url: str, transport: str, seconds: float) -> dict:
    code, out = run([ffprobe, "-v", "error", "-rtsp_transport", transport, "-timeout", "5000000", "-i", url,
                     "-select_streams", "v:0", "-read_intervals", f"%+{seconds:g}",
                     "-show_entries", "packet=pts_time,dts_time,size,flags", "-of", "csv=p=0"],
                    timeout=seconds + 25)
    if code != 0 and not out.strip():
        return {"error": f"ffprobe exit {code}"}
    st = parse_packets(out)
    st["transport"] = transport
    return st


class _Reader(threading.Thread):
    """One OpenCV RTSP client (the backend's reader settings), timestamping every frame."""

    def __init__(self, url: str, transport: str, seconds: float):
        super().__init__(daemon=True)
        self.url, self.transport, self.seconds = url, transport, seconds
        self.times: list[float] = []
        self.opened = False
        self.error = ""

    def run(self) -> None:
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")    # quiet: joining mid-GOP logs HEVC ref errors
        import cv2
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay"
        os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            return
        self.opened = cap.isOpened()
        end = time.monotonic() + self.seconds
        while self.opened and time.monotonic() < end:
            ok = cap.grab()
            if ok:
                self.times.append(time.monotonic())
            elif time.monotonic() < end:
                time.sleep(0.05)
        cap.release()


def second_client_test(url: str, transport: str, seconds: float) -> dict:
    """Client A reads for `seconds`; client B joins after A's first frames and reads for half of it."""
    a = _Reader(url, transport, seconds)
    a.start()
    t_wait = time.monotonic() + 10.0
    while not a.times and time.monotonic() < t_wait and a.is_alive():
        time.sleep(0.05)
    b_secs = max(2.0, seconds / 2)
    b = _Reader(url, transport, b_secs)
    tb0 = time.monotonic()
    b.start()
    b.join(b_secs + 12)
    tb1 = time.monotonic()
    a.join(seconds + 12)
    b_first = b.times[0] if b.times else tb1
    a_during = sum(1 for t in a.times if b_first + 0.5 <= t <= (b.times[-1] if b.times else tb1))
    a_after_b_start = sum(1 for t in a.times if t > tb0 + 1.0)
    if a.times and b.times and a_during > 0:
        verdict = "both"
    elif a.times and not b.times:
        verdict = "only_one"
    elif b.times and a.times and a_after_b_start == 0:
        verdict = "kicks_first"
    elif b.times and not a.times:
        verdict = "kicks_first"
    elif not a.times and not b.times:
        verdict = "none"
    else:
        verdict = "both" if a_during > 0 else "kicks_first"
    return {"verdict": verdict, "a_frames": len(a.times), "b_frames": len(b.times), "a_frames_during_b": a_during,
            "a_opened": a.opened, "b_opened": b.opened, "transport": transport, "seconds": seconds,
            "errors": [e for e in (a.error, b.error) if e]}


# ---------------------------------------------------------------------------------------------- output

def print_table(results: list[Result], color: bool) -> None:
    col = {PASS: "\033[32m", FAIL: "\033[1;31m", WARN: "\033[33m", INFO: "\033[36m", SKIP: "\033[90m"}
    reset = "\033[0m" if color else ""
    w = max(len(f"{r.group}/{r.name}") for r in results) if results else 10
    print()
    print(f"{'CHECK'.ljust(w)}  RESULT  DETAIL")
    print("-" * min(120, w + 60))
    for r in results:
        st = (col.get(r.status, "") if color else "") + r.status.ljust(6) + reset
        print(f"{(r.group + '/' + r.name).ljust(w)}  {st}  {r.detail}")
    fails = [r for r in results if r.status in (FAIL, WARN) and r.fix]
    if fails:
        print("\nHOW TO FIX")
        for r in fails:
            st = (col.get(r.status, "") if color else "") + r.status + reset
            print(f"- [{st}] {r.group}/{r.name}: {r.fix}")
    n_fail = sum(r.status == FAIL for r in results)
    n_warn = sum(r.status == WARN for r in results)
    verdict = "PREFLIGHT FAIL" if n_fail else "PREFLIGHT PASS"
    print(f"\n{(col[FAIL] if n_fail else col[PASS]) if color else ''}{verdict}{reset}: "
          f"{sum(r.status == PASS for r in results)} pass, {n_fail} fail, {n_warn} warn")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--board", choices=BOARDS, default="x69_lg",
                    help=f"x69_lg (172.16.11.1, RTSP) or lewei (our unit: {LEWEI_IP}, video TCP {LEWEI_VIDEO_PORT})")
    ap.add_argument("--drone-ip", default=None, help=f"drone address (default {DRONE_IP}, or {LEWEI_IP} with --board lewei)")
    ap.add_argument("--rtsp-url", default=None, help=f"default rtsp://<drone-ip>{RTSP_PATH}")
    ap.add_argument("--only", default="", help=f"comma list of groups: {','.join(GROUPS)}")
    ap.add_argument("--skip", default="", help="comma list of groups to skip")
    ap.add_argument("--skip-ml", action="store_true", help="skip the ml-venv doctor (imports mlx)")
    ap.add_argument("--wifi-device", default=None, help="default: the 'Wi-Fi' hardware port (en0)")
    ap.add_argument("--transport", default="auto", choices=["auto", "tcp", "udp"], help="RTSP transport")
    ap.add_argument("--probe-seconds", type=float, default=6.0, help="packet window for fps/GOP/bitrate")
    ap.add_argument("--second-client-seconds", type=float, default=6.0)
    ap.add_argument("--skip-second-client", action="store_true")
    ap.add_argument("--tcp-timeout", type=float, default=2.0)
    ap.add_argument("--backend-port", type=int, default=8000, help="warn if a backend already reads the stream")
    ap.add_argument("--json", default=None, help="results file (default logs/preflight-<time>.json)")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    a.drone_ip = a.drone_ip or (LEWEI_IP if a.board == "lewei" else DRONE_IP)
    try:
        ip = ipaddress.ip_address(a.drone_ip)
    except ValueError:
        raise SystemExit(f"--drone-ip {a.drone_ip!r} is not an IPv4 address")
    if os.environ.get("MHD_OPS_LOOPBACK_ONLY") == "1" and not ip.is_loopback:
        raise SystemExit("MHD_OPS_LOOPBACK_ONLY=1: refusing a non-loopback --drone-ip")
    if not ip.is_loopback and a.drone_ip not in (DRONE_IP, LEWEI_IP):
        print(f"note: --drone-ip {a.drone_ip} is neither the X69 ({DRONE_IP}), the Lewei board ({LEWEI_IP}) "
              "nor loopback", file=sys.stderr)
    pf = Preflight(a)
    print(f"preflight: drone {a.drone_ip} ({'loopback/twin' if pf.loopback else 'REAL drone'}, board {a.board}), "
          f"{'video TCP ' + str(LEWEI_VIDEO_PORT) if pf.lewei else 'RTSP ' + pf.rtsp_url()}, Wi-Fi {pf.wifi_dev}, iPhone USB {pf.iphone_dev or 'not present'}, app {pf.app}")
    code = pf.run()
    print_table(pf.results, color=sys.stdout.isatty() and not a.no_color)
    out = Path(a.json) if a.json else TREE / "logs" / f"preflight-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "drone_ip": a.drone_ip, "board": a.board,
                               "rtsp_url": pf.rtsp_url(), "loopback": pf.loopback, "exit_code": code,
                               "results": [asdict(r) for r in pf.results]}, indent=1, default=str))
    print(f"results: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
