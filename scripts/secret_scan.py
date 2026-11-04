"""Pre-publication scan: secrets, private values, raw footage and oversized files.

    source env.sh && $MHD_PY scripts/secret_scan.py            # working tree + full git history
    $MHD_PY scripts/secret_scan.py --no-history                # working tree only (fast)

What it checks, without ever printing a secret:
1. The exact values in $MHD_HOME/.env, in every tracked or untracked-but-not-ignored file and in
   `git log -p --all` (every commit, diff and message on every branch). Variables whose names
   contain KEY, TOKEN, SECRET or PASSWORD are secrets (FAIL); other values are private (WARN).
2. Common credential patterns (private keys, sk-/fw_/hf_/ghp_ tokens, bearer headers,
   api_key = "..." assignments), shown masked as path:line.
3. .env files tracked now or at any point in history.
4. Tracked videos or files under media/, logs/, captures/ (raw footage must stay out).
5. Files over 50 MB (WARN) or 100 MB (FAIL: GitHub refuses them), now or in history.

Exit status: 0 = clean (warnings allowed), 1 = something must be fixed before going public.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
PATTERNS = [
    ("private key block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("sk- token", re.compile(rb"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("fw_ token", re.compile(rb"\bfw_[A-Za-z0-9]{16,}")),
    ("hf_ token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(rb"\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("bearer header", re.compile(rb"(?i)authorization:\s*bearer\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("key assignment", re.compile(rb"(?i)(?:api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_\-\.\|=+/]{20,}[\"']")),
]
FOOTAGE_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".h264", ".h265", ".hevc", ".264", ".265"}
FOOTAGE_DIRS = ("media/", "logs/", "captures/")
MB = 1024 * 1024


def git(*args: str, tree: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=tree, check=True, capture_output=True).stdout


def env_values(env_path: Path) -> list[tuple[str, bytes, bool]]:
    """(name, value, is_secret) for every non-empty value in .env. Values never leave this process."""
    out = []
    if not env_path.exists():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.replace("export ", "").strip()
        value = value.strip()
        if value[:1] in ("'", '"'):             # quoted: up to the closing quote
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:                                    # unquoted, as the shell reads it: up to the first
            value = value.split()[0] if value else ""   # space; "#..." is a comment, not a value
            if value.startswith("#"):
                value = ""
        if len(value) >= 6:
            out.append((name, value.encode(), bool(SECRET_NAME.search(name))))
    return out


def mask(m: bytes) -> str:
    s = m.decode("utf-8", "replace")
    return f"{s[:4]}... ({len(s)} chars)"


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def fail(self, msg: str) -> None:
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def scan_blob(where: str, data: bytes, values, rep: Report, line_of=None) -> None:
    for name, value, is_secret in values:
        if value in data:
            ln = data[: data.index(value)].count(b"\n") + 1
            loc = line_of(ln) if line_of else f"{where}:{ln}"
            (rep.fail if is_secret else rep.warn)(f"value of {name} found in {loc}")
    for label, rx in PATTERNS:
        for m in rx.finditer(data):
            ln = data[: m.start()].count(b"\n") + 1
            loc = line_of(ln) if line_of else f"{where}:{ln}"
            rep.fail(f"{label} at {loc}: {mask(m.group(0))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.environ.get("MHD_TREE", "."), help="repository to scan")
    ap.add_argument("--env", default=None, help="the .env file (default $MHD_HOME/.env)")
    ap.add_argument("--no-history", action="store_true", help="skip git history")
    args = ap.parse_args()
    tree = Path(args.tree).resolve()
    env_path = Path(args.env) if args.env else Path(os.environ.get("MHD_HOME", tree)) / ".env"
    values = env_values(env_path)
    rep = Report()
    print(f"scanning {tree}; {len(values)} values from {env_path} "
          f"({sum(v[2] for v in values)} secret, {sum(not v[2] for v in values)} private)")

    tracked = [p for p in git("ls-files", "-z", tree=tree).decode().split("\0") if p]
    untracked = [p for p in git("ls-files", "-z", "--others", "--exclude-standard", tree=tree).decode().split("\0") if p]
    for rel in tracked + untracked:
        path = tree / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > 100 * MB:
            rep.fail(f"{rel}: {size / MB:.0f} MB (GitHub refuses files over 100 MB)")
        elif size > 50 * MB:
            rep.warn(f"{rel}: {size / MB:.0f} MB (over 50 MB)")
        name = rel.rsplit("/", 1)[-1]
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            rep.fail(f"{rel}: an .env file is {'tracked' if rel in tracked else 'not ignored'}")
        if Path(rel).suffix.lower() in FOOTAGE_EXT or rel.startswith(FOOTAGE_DIRS):
            rep.warn(f"{rel}: video or media/logs file in the repository (raw footage must stay out)")
        if size <= 20 * MB:
            scan_blob(rel, path.read_bytes(), values, rep)
    print(f"working tree: {len(tracked)} tracked and {len(untracked)} untracked files scanned")

    if not args.no_history:
        log = git("log", "-p", "--all", "--full-history", "--no-color", "--text",
                  "--format=commit %H%n%B", tree=tree)
        commits = [m.start() for m in re.finditer(rb"^commit [0-9a-f]{40}$", log, re.M)]

        def line_of(ln: int) -> str:
            # map a line number in the log to its commit
            pos, n = 0, 1
            while n < ln:
                pos = log.index(b"\n", pos) + 1
                n += 1
            starts = [c for c in commits if c <= pos]
            sha = log[starts[-1] + 7: starts[-1] + 47].decode() if starts else "?"
            return f"git history (commit {sha[:10]})"

        scan_blob("history", log, values, rep, line_of=line_of)
        names = git("log", "--all", "--format=", "--name-only", tree=tree).decode().splitlines()
        for rel in sorted(set(n for n in names if n)):
            name = rel.rsplit("/", 1)[-1]
            if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
                rep.fail(f"{rel}: an .env file appears in git history")
        objs = git("rev-list", "--objects", "--all", tree=tree)
        sizes = subprocess.run(["git", "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)"],
                               cwd=tree, input=objs, check=True, capture_output=True).stdout.decode()
        for row in sizes.splitlines():
            parts = row.split(" ", 3)
            if len(parts) >= 3 and parts[0] == "blob" and int(parts[2]) > 50 * MB:
                where = parts[3] if len(parts) > 3 else parts[1]
                (rep.fail if int(parts[2]) > 100 * MB else rep.warn)(
                    f"history blob {where}: {int(parts[2]) / MB:.0f} MB")
        print(f"history: {len(commits)} commits on all branches scanned")

    for w in rep.warns:
        print("WARN", w)
    for f in rep.fails:
        print("FAIL", f)
    print("RESULT:", "FAIL" if rep.fails else "clean", f"({len(rep.fails)} failures, {len(rep.warns)} warnings)")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
