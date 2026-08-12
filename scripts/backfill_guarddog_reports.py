#!/usr/bin/env python3
"""Import GuardDog reports collected outside the wrapper into `reports/`.

`guarddog-cached` stores a full report for every package version it scans,
but only from the moment that feature existed. Scans completed before it —
notably the 2026-08-11 calibration sweep of 74 known-good dependencies — left
their reports in a directory of their own, and the cache entries those scans
produced are complete, so the wrapper will never re-scan them and never
generate the reports. Without an import, the evidence for those packages is
unreachable to `guarddog-review` even though it exists on disk.

What this may and may not do
----------------------------
It writes **reports only**. It does not create or alter cache entries, and it
cannot change a verdict: the gate reads `cache.json` and `accepted.json`, and
neither is touched here. An imported report makes a finding *reviewable*; it
never makes one pass.

It refuses to overwrite an existing report. A report the wrapper wrote came
from a scan this machine ran and is the better record of the two.

Only completed scans are imported, on the same terms as the wrapper: a report
carrying `errors` describes a scan that did not fully run, and importing it
would put a non-result where a result is expected.

The GuardDog version is not recoverable from a report — it appears nowhere in
GuardDog's own output — so it must be stated on the command line rather than
guessed. It is part of the cache key, and getting it wrong files real evidence
under a key nothing will ever look up.

Usage
-----
    python3 scripts/backfill_guarddog_reports.py <source-dir> --guarddog-version 3.1.0
    python3 scripts/backfill_guarddog_reports.py <source-dir> --guarddog-version 3.1.0 --apply

The first form reports what would happen and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ai_common.security.guarddog_cached import (
    REPORT_PROVENANCE_KEY,
    _write_json_atomically,
    entry_key,
    report_path,
    reports_dir,
)

#: Fields added by the collector that produced the calibration sweep. They are
#: not GuardDog's output, and a stored report is GuardDog's output plus one
#: namespaced provenance block — nothing else.
COLLECTOR_FIELDS = ("_returncode", "_collector_error")


def _readable(report: object) -> str | None:
    """Why this report cannot be imported, or None if it can.

    The checks mirror the wrapper's own shape guard. A report that would be
    unreadable to `guarddog-review` should be rejected here, where the reason
    can be printed, rather than filed and rediscovered later as a broken row.
    """
    if not isinstance(report, dict):
        return "not a JSON object"
    if not isinstance(report.get("errors"), dict):
        return "'errors' is missing or not an object"
    if report["errors"]:
        return f"scan did not complete: {sorted(report['errors'])}"
    if not isinstance(report.get("risks"), list):
        return "'risks' is missing or not a list on a scan that reported no errors"
    if not isinstance(report.get("results"), dict):
        return "'results' is missing or not an object on a scan that reported no errors"
    if not report.get("package") or not report.get("package_version"):
        return "no 'package'/'package_version' to key on"
    return None


def prepare(report: dict, source: Path, guarddog_version: str) -> tuple[str, str, dict]:
    """Turn a collected report into one the wrapper would have written."""
    name = str(report["package"]).lower()
    version = str(report["package_version"])

    stored = {k: v for k, v in report.items() if k not in COLLECTOR_FIELDS}
    stored[REPORT_PROVENANCE_KEY] = {
        "package": name,
        "version": version,
        "guarddog_version": guarddog_version,
        "cache_key": entry_key(name, version, guarddog_version),
        # The collector recorded no timestamp, so the file's own is the best
        # available. Marked as derived by `backfilled_from` sitting beside it:
        # a reader must be able to tell an imported report from a scanned one.
        "scanned_at": datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "backfilled_from": str(source),
    }
    return name, version, stored


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="Import externally collected GuardDog reports into the shared "
                    "reports directory. Writes reports only; never touches the "
                    "cache, the waivers, or any verdict.",
    )
    parser.add_argument("source", type=Path, help="directory of *.json GuardDog reports")
    parser.add_argument("--guarddog-version", required=True,
                        help="the GuardDog version that produced them; part of the "
                             "cache key and not recoverable from the reports")
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without it, nothing is changed.")
    args = parser.parse_args(argv[1:])

    sources = sorted(args.source.glob("*.json"))
    if not sources:
        print(f"no *.json reports under {args.source}", file=sys.stderr)
        return 2

    print(f"{len(sources)} report(s) in {args.source}")
    print(f"target: {reports_dir()}")
    print(f"guarddog version asserted: {args.guarddog_version}")
    print("(dry run — nothing will be written; pass --apply to write)\n"
          if not args.apply else "")

    imported = skipped = rejected = 0
    for source in sources:
        try:
            report = json.loads(source.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ✗ {source.name}: {exc}")
            rejected += 1
            continue

        why = _readable(report)
        if why:
            print(f"  ✗ {source.name}: {why}")
            rejected += 1
            continue

        name, version, stored = prepare(report, source, args.guarddog_version)
        target = report_path(name, version, args.guarddog_version)
        if target.exists():
            print(f"  · {name}=={version}: already stored, left alone")
            skipped += 1
            continue

        risks = len(report["risks"])
        print(f"  {'→' if args.apply else '+'} {name}=={version}"
              f"{f' ({risks} risk(s))' if risks else ''}")
        if args.apply:
            _write_json_atomically(target, stored)
        imported += 1

    verb = "imported" if args.apply else "would import"
    print(f"\n{verb} {imported} · left alone {skipped} · rejected {rejected}")
    if not args.apply and imported:
        print("Re-run with --apply to write them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))