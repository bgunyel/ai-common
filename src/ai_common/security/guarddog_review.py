#!/usr/bin/env python3
"""The review ledger: which stored reports a human has already adjudicated.

`guarddog-cached` keeps a full GuardDog report for every package version it
has scanned. Most carry nothing to decide. A few carry a finding that has to
be read by a person and either waived or acted on, and that reading is the
expensive part — it means opening the rule, finding the matched text, and
judging whether the detection is a defect or a real behaviour.

This module records that the reading happened, so a second pass over a
machine-wide reports directory starts from what is left rather than from the
beginning.

What it is not
--------------
**The ledger never affects a verdict.** Only `accepted.json` waives a
finding, and only `guarddog_cached.verdict_for` decides anything. A ledger
entry saying "reviewed, harmless" changes no gate result on its own; the
reviewer must still write the waiver. Keeping the two apart is deliberate:
a record of *having looked* and a decision to *stop blocking* are different
claims, and a bookkeeping file that could quietly clear a finding would be a
second, weaker waiver mechanism with none of the first one's ceremony.

What makes a review stale
-------------------------
A review is of a *report*, not of a package. The same code scanned by a
newer GuardDog can produce findings nobody has seen, so entries are keyed on
the full cache key — (name, version, guarddog_version) — exactly like the
report they describe.

Within one key, the ledger stores a digest of the findings that were
actually adjudicated: for each risk, its rule, severity and location, plus
the text the rule matched. If a re-scan produces a report whose findings
differ, the digest differs and the entry is pending again. A review answers
for what the reviewer saw and for nothing else.

Reports with no risks at all are not recorded. There is nothing to adjudicate
in them, and writing an entry per clean package would bury the handful of real
decisions in several hundred rows of noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ai_common.security.guarddog_cached import (
    REPORT_PROVENANCE_KEY,
    _write_json_atomically,
    cache_path,
    reports_dir,
    risk_label,
    severity_blocks,
)

REVIEW_SCHEMA = 1

#: What a reviewer can conclude. `waived` means a waiver was written into
#: `accepted.json`; `rejected` means the finding is real and this version is
#: not to be adopted. Both are decisions, and both stop the report being
#: offered for review again — an unreviewed finding and a finding judged bad
#: are different states, and only the first is work outstanding.
OUTCOMES = ("waived", "rejected")


def reviewed_path() -> Path:
    return cache_path().parent / "reviewed.json"


def load_reviewed() -> dict:
    path = reviewed_path()
    empty = {"schema": REVIEW_SCHEMA, "reviewed": {}}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return empty
    if not isinstance(data, dict) or data.get("schema") != REVIEW_SCHEMA:
        return empty
    if not isinstance(data.get("reviewed"), dict):
        return empty
    return data


def save_reviewed(ledger: dict) -> None:
    _write_json_atomically(reviewed_path(), ledger)


# --- what a reviewer is being asked to judge -----------------------------

def review_subject(report: dict) -> list[dict]:
    """The findings in one report, canonically ordered.

    Built from `risks`, because that is what the gate judges and what a
    waiver names, and enriched from `results` with the text each rule
    actually matched — the part that decides whether a finding is a rule
    defect. `google-genai`'s high-severity steganography risk and the string
    `eval(` inside the word `Retrieval(` are the same finding seen from the
    two halves of the report; only together do they support a decision.
    """
    matches = report.get("results")
    matches = matches if isinstance(matches, dict) else {}

    subject = []
    for risk in report.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        rule = risk_label(risk)
        matched = sorted({
            str(m.get("match"))
            for m in (matches.get(rule) or [])
            if isinstance(m, dict) and m.get("match") is not None
        })
        subject.append({
            "rule": rule,
            "severity": str(risk.get("severity")),
            "location": str(risk.get("threat_location") or risk.get("file_path") or ""),
            "matched": matched,
            # Carried for the risks that name no file at all: a metadata risk
            # like `bundled_binary` is about the package as a whole, and its
            # description is the only statement of what was found.
            "detail": str(risk.get("threat_description") or ""),
        })
    return sorted(subject, key=lambda f: (f["rule"], f["location"], f["severity"]))


def subject_digest(subject: list[dict]) -> str:
    """Identifies exactly what was adjudicated, so a changed report re-opens."""
    return hashlib.sha256(
        json.dumps(subject, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def describe(subject: list[dict]) -> list[str]:
    """One legible line per finding, for the ledger and for the terminal."""
    lines = []
    for finding in subject:
        where = f" at {finding['location']}" if finding["location"] else ""
        matched = (f" matched {', '.join(repr(m) for m in finding['matched'])}"
                   if finding["matched"] else "")
        # Only when there is no location: for a located finding the
        # description restates the rule and crowds out the match.
        detail = ""
        if not where and finding["detail"]:
            detail = " — " + " ".join(finding["detail"].split())[:120]
        lines.append(f"{finding['rule']} [{finding['severity']}]{where}{matched}{detail}")
    return lines


def blocks(subject: list[dict]) -> bool:
    return any(severity_blocks(finding["severity"]) for finding in subject)


# --- reading the reports directory ---------------------------------------

class Report:
    """One stored report, with the facts the ledger keys on."""

    def __init__(self, path: Path, key: str, report: dict):
        self.path = path
        self.key = key
        self.subject = review_subject(report)
        self.digest = subject_digest(self.subject)

    @property
    def blocking(self) -> bool:
        return blocks(self.subject)


def stored_reports() -> tuple[list[Report], list[tuple[Path, str]]]:
    """Every readable report, and every one that could not be read.

    Unreadable files are returned rather than skipped. A reviewer sweeping a
    directory needs to know that three files were passed over, not to be
    handed a shorter list that looks complete.
    """
    readable: list[Report] = []
    unreadable: list[tuple[Path, str]] = []

    for path in sorted(reports_dir().glob("*.json")):
        try:
            report = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path, str(exc)))
            continue
        if not isinstance(report, dict):
            unreadable.append((path, "not a JSON object"))
            continue

        provenance = report.get(REPORT_PROVENANCE_KEY)
        key = provenance.get("cache_key") if isinstance(provenance, dict) else None
        if not key:
            # The filename is sanitised and may carry a disambiguating
            # digest, so it is not a reliable key. Say so instead of guessing.
            unreadable.append((path, f"no {REPORT_PROVENANCE_KEY}.cache_key"))
            continue
        readable.append(Report(path, str(key), report))

    return readable, unreadable


def pending(reports: list[Report], ledger: dict) -> list[Report]:
    """Reports carrying findings that no recorded review covers.

    Ordered so that anything failing the gate is dealt with first: those are
    the reviews something is waiting on.
    """
    reviewed = ledger["reviewed"]
    out = []
    for report in reports:
        if not report.subject:
            continue  # nothing to adjudicate
        record = reviewed.get(report.key)
        if isinstance(record, dict) and record.get("digest") == report.digest:
            continue
        out.append(report)
    return sorted(out, key=lambda r: (not r.blocking, r.key))


def record_review(ledger: dict, report: Report, outcome: str, note: str) -> dict:
    """Write one decision into the ledger, in memory."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
    if not note.strip():
        raise ValueError("a review needs a note saying what was decided and why")

    entry = {
        "outcome": outcome,
        "note": note.strip(),
        "digest": report.digest,
        "findings": describe(report.subject),
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ledger["reviewed"][report.key] = entry
    return entry


# --- driver --------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="List the stored GuardDog reports whose findings nobody has "
                    "reviewed yet, and record decisions about them.",
    )
    parser.add_argument("--all", action="store_true",
                        help="list every report with findings, reviewed or not")
    parser.add_argument("--record", metavar="CACHE_KEY",
                        help="record a decision about one report")
    parser.add_argument("--outcome", choices=OUTCOMES,
                        help="with --record: what was decided")
    parser.add_argument("--note", help="with --record: why, in one line")
    return parser.parse_args(argv[1:])


def _report_lines(report: Report, prefix: str = "") -> list[str]:
    lines = [f"{prefix}{report.key}" + ("  ← fails the gate" if report.blocking else "")]
    lines.extend(f"{prefix}    {line}" for line in describe(report.subject))
    lines.append(f"{prefix}    report: {report.path}")
    return lines


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    ledger = load_reviewed()
    reports, unreadable = stored_reports()

    if args.record:
        if not args.outcome or not args.note:
            print("--record needs both --outcome and --note", file=sys.stderr)
            return 2
        match = [r for r in reports if r.key == args.record]
        if not match:
            print(f"no stored report for {args.record}", file=sys.stderr)
            return 2
        entry = record_review(ledger, match[0], args.outcome, args.note)
        save_reviewed(ledger)
        print(f"Recorded {args.record}: {entry['outcome']} — {entry['note']}")
        if entry["outcome"] == "waived":
            # The ledger is bookkeeping; the gate reads accepted.json alone.
            print("Note: this records the review only. The finding still blocks "
                  "until a waiver is written into accepted.json.")
        return 0

    with_findings = [r for r in reports if r.subject]
    shown = with_findings if args.all else pending(reports, ledger)

    print(f"{len(reports)} stored reports · {len(with_findings)} with findings · "
          f"{len(pending(reports, ledger))} awaiting review")

    if unreadable:
        print(f"\n⚠ {len(unreadable)} report(s) could not be read:")
        for path, why in unreadable:
            print(f"    {path.name}: {why}")

    if not shown:
        print("\nNothing awaiting review.")
        return 0

    print("\n" + ("All reports with findings:" if args.all else "Awaiting review:"))
    for report in sorted(shown, key=lambda r: (not r.blocking, r.key)):
        print("\n".join(_report_lines(report, prefix="  ")))
    return 0


def cli() -> None:
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    cli()