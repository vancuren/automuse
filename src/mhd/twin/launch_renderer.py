"""Launch the twin's Three.js renderer in the installed Google Chrome through Playwright.

    caffeinate -i $MHD_PY -m mhd.twin.launch_renderer --twin http://127.0.0.1:8766 [--chase-rec]

- channel="chrome" (never downloads a browser); profile in $MHD_HOME/.cache/chrome-twin.
- Flags keep it rendering when occluded or in the background:
  --use-angle=metal --disable-background-timer-throttling --disable-renderer-backgrounding
  --disable-backgrounding-occluded-windows
- Asserts the WebGL renderer is a GPU (not SwiftShader or another software rasterizer).
Runs until SIGINT/SIGTERM, then closes Chrome.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

SOFTWARE_RENDERERS = ("swiftshader", "llvmpipe", "software", "microsoft basic render")

CHROME_ARGS = [
    "--use-angle=metal",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--no-first-run", "--no-default-browser-check",
    "--use-mock-keychain", "--password-store=basic",
    "--disable-background-networking", "--disable-component-update", "--disable-sync",
    "--disable-features=Translate,MediaRouter,OptimizationHints",
    "--ignore-gpu-blocklist",
    "--window-size=1300,860",
]


def is_software(renderer: str) -> bool:
    r = (renderer or "").lower()
    return any(s in r for s in SOFTWARE_RENDERERS)


async def run(args: argparse.Namespace) -> int:
    home = Path(os.environ.get("MHD_HOME", Path(__file__).resolve().parents[3]))
    profile = Path(args.user_data_dir or home / ".cache" / "chrome-twin")
    profile.mkdir(parents=True, exist_ok=True)
    chrome_args = CHROME_ARGS + [f"--disk-cache-dir={profile / 'cache'}"]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=args.headless, args=chrome_args,
            viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        pages = []

        def wire(page, name):
            page.on("console", lambda m: print(f"[{name} console] {m.type}: {m.text}", flush=True))
            page.on("pageerror", lambda e: print(f"[{name} pageerror] {e}", flush=True))

        urls = [("camera", f"{args.twin}/")]
        if args.chase_rec:
            urls.append(("chase_rec", f"{args.twin}/chase?rec=1"))
        if args.chase:
            urls.append(("chase", f"{args.twin}/chase"))
        first = ctx.pages[0] if ctx.pages else None
        for i, (name, url) in enumerate(urls):
            page = first if (i == 0 and first is not None) else await ctx.new_page()
            wire(page, name)
            await page.goto(url, wait_until="load")
            pages.append((name, page))

        # GPU check on the camera page
        renderer = ""
        for _ in range(50):
            try:
                info = await pages[0][1].evaluate("window.__twin && window.__twin.webgl()")
            except Exception:
                info = None
            if info and info.get("renderer"):
                renderer = info["renderer"]
                break
            await asyncio.sleep(0.2)
        print(json.dumps({"event": "webgl", "renderer": renderer}), flush=True)
        if not renderer or is_software(renderer):
            print(f"ERROR: WebGL is not on the GPU (renderer={renderer!r}); refusing to run", flush=True)
            await ctx.close()
            return 2
        print(json.dumps({"event": "ready", "pages": [n for n, _ in pages]}), flush=True)

        t_end = time.monotonic() + args.duration_s if args.duration_s else None
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            if t_end and time.monotonic() > t_end:
                break
            if not ctx.pages:
                print("ERROR: Chrome closed", flush=True)
                return 3
        await ctx.close()
    return 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the twin renderer in Chrome (Playwright)")
    ap.add_argument("--twin", default="http://127.0.0.1:8766", help="twin server base URL")
    ap.add_argument("--chase-rec", action="store_true", help="also open the chase view for MP4 recording")
    ap.add_argument("--chase", action="store_true", help="also open a passive chase view window")
    ap.add_argument("--headless", action="store_true", help="try headless Chrome (GPU is not guaranteed)")
    ap.add_argument("--user-data-dir", default=None)
    ap.add_argument("--duration-s", type=float, default=0.0, help="exit after this long (0 = run until killed)")
    args = ap.parse_args(argv)
    if not args.twin.startswith(("http://127.0.0.1", "http://localhost")):
        raise SystemExit("the renderer only connects to a loopback twin")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
