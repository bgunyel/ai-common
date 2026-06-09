#!/usr/bin/env python3
"""GuardDog cache wrapper.

Drop-in replacement for `guarddog pypi verify <requirements-file>` that
caches per-package results in `.guarddog-cache.json` (at the project
root) so unchanged versions are not re-scanned. The cache key includes
the GuardDog version, so a tool upgrade invalidates everything (rules
may have changed).

Cache layout:
{
  "guarddog_version": "2.10.0",
  "entries": {
    "requests==2.33.1": {
      "scanned_at": "2026-05-10T15:50:00+00:00",
      "exit_code": 0,
      "output": "Found 0 potentially malicious indicators scanning requests"
    },
    ...
  }
}
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)")


def find_project_root() -> Path:
    """Walk up from this file until a `pyproject.toml` is found.

    Falls back to the current working directory if no marker is hit
    (unlikely in practice, but keeps the script usable in scratch
    contexts).
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


CACHE_PATH = find_project_root() / ".guarddog-cache.json"


def get_guarddog_version() -> str:
    out = subprocess.run(
        ["guarddog", "--version"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[0]


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = REQ_RE.match(line)
        if m:
            pairs.append((m.group(1).lower(), m.group(2)))
    return pairs


def load_cache(guarddog_version: str) -> dict:
    if not CACHE_PATH.exists():
        return {"guarddog_version": guarddog_version, "entries": {}}
    try:
        data = json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {"guarddog_version": guarddog_version, "entries": {}}
    if data.get("guarddog_version") != guarddog_version:
        return {"guarddog_version": guarddog_version, "entries": {}}
    return data


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def scan_package(name: str, version: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["guarddog", "pypi", "scan", name, "--version", version],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <requirements-file>", file=sys.stderr)
        return 2

    req_path = Path(argv[1])
    if not req_path.exists():
        print(f"requirements file not found: {req_path}", file=sys.stderr)
        return 2

    guarddog_version = get_guarddog_version()
    cache = load_cache(guarddog_version)
    entries = cache["entries"]

    pairs = parse_requirements(req_path)
    print(f"GuardDog v{guarddog_version} — {len(pairs)} packages to evaluate", flush=True)

    overall_exit = 0
    cached = 0
    scanned = 0

    try:
        for name, version in pairs:
            key = f"{name}=={version}"
            if key in entries:
                cached += 1
                entry = entries[key]
                rc = int(entry.get("exit_code", 0))
                output = entry.get("output", "")
                print(f"[cached] {key}", flush=True)
                if output.strip():
                    print(output.rstrip(), flush=True)
                if rc != 0:
                    overall_exit = max(overall_exit, rc)
                continue

            scanned += 1
            print(f"[scanning] {key}", flush=True)
            rc, output = scan_package(name, version)
            if output.strip():
                print(output.rstrip(), flush=True)
            entries[key] = {
                "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "exit_code": rc,
                "output": output,
            }
            if rc != 0:
                overall_exit = max(overall_exit, rc)
            # Persist after every scan so an interrupted run preserves
            # partial progress.
            save_cache(cache)
    except KeyboardInterrupt:
        save_cache(cache)
        print(
            f"\nInterrupted. Cache saved: {cached} cached, {scanned} scanned this run.",
            flush=True,
        )
        return 130

    save_cache(cache)
    print(f"\nSummary: {cached} cached, {scanned} scanned. Exit: {overall_exit}", flush=True)
    return overall_exit


if __name__ == "__main__":
    sys.exit(main(sys.argv))