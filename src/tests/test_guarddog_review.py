"""Tests for the review ledger over the stored GuardDog reports.

The property that matters most here is a negative one: the ledger records
that a human looked at a finding and must never, on its own, stop that
finding blocking. Only `accepted.json` does that. A bookkeeping file that
could clear a verdict would be a second waiver mechanism without any of the
first one's deliberateness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_common.security import guarddog_cached as gd
from ai_common.security import guarddog_review as gr

#: google-genai 2.11.0 as GuardDog 3.1.0 reported it, trimmed to the parts a
#: review turns on. The risk says a high-severity steganography rule fired at
#: line 217; `results` says the text it fired on was `eval(`, inside the word
#: `Retrieval(`. Neither half supports a decision alone.
STEGO_REPORT = {
    "package": "google-genai",
    "issues": 15,
    "errors": {},
    "results": {
        "threat-runtime-obfuscation-steganography": [{
            "code": "types.Tool(\n    retrieval=types.Retrieval(",
            "location": "google/genai/tests/models/test_generate_content_tools.py:217",
            "match": "eval(",
            "message": "Detects steganography decode followed by code execution",
        }],
    },
    "risks": [{
        "name": "risk.runtime.obfuscation",
        "category": "runtime",
        "severity": "high",
        "threat_rule": "threat-runtime-obfuscation-steganography",
        "threat_location": "google/genai/tests/models/test_generate_content_tools.py:217",
        "file_path": "google/genai/tests/models/test_generate_content_tools.py",
    }],
    "risk_score": {"score": 4.9, "label": "low", "findings_count": 9},
}

ADVISORY_REPORT = {
    "package": "tqdm",
    "issues": 6,
    "errors": {},
    "results": {
        "threat-network-exfiltration": [{
            "location": "tqdm/contrib/telegram.py:26",
            "match": "api.telegram.org",
            "message": "Detects URLs to suspicious domains",
        }],
    },
    "risks": [{
        "name": "risk.network.outbound",
        "category": "network",
        "severity": "medium",
        "threat_rule": "threat-network-exfiltration",
        "threat_location": "tqdm/contrib/telegram.py:26",
        "file_path": "tqdm/contrib/telegram.py",
    }],
    "risk_score": {"score": 7.2, "label": "high_risk", "findings_count": 4},
}

CLEAN_REPORT = {
    "package": "six",
    "issues": 0,
    "errors": {},
    "results": {},
    "risks": [],
    "risk_score": {"score": 0.0, "label": "no_risks_detected"},
}


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg"


def write_report(cache_home: Path, key: str, report: dict) -> Path:
    """Place a stored report exactly as `guarddog-cached` would."""
    path = cache_home / "guarddog-cached" / "reports" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(report)
    stored["_guarddog_cached"] = {"cache_key": key}
    path.write_text(json.dumps(stored))
    return path


def read_ledger(cache_home: Path) -> dict:
    return json.loads((cache_home / "guarddog-cached" / "reviewed.json").read_text())


# --- the ledger must not be a second waiver mechanism ---------------------

def test_a_recorded_review_does_not_clear_the_finding(cache_home):
    """The gate reads accepted.json and nothing else.

    Recording "looked at it, harmless" is a claim about a person's attention.
    Waiving is a claim about the gate. Collapsing them would let a note stop
    a high-severity finding blocking.
    """
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()
    gr.record_review(ledger, reports[0], "waived", "eval( inside Retrieval( — rule defect")
    gr.save_reviewed(ledger)

    entry = {"errors": {}, "risks": STEGO_REPORT["risks"]}
    verdict, rules = gd.verdict_for(entry, gd.waived_rules(gd.load_accepted(),
                                                          "google-genai", "2.11.0"))

    assert verdict == gd.BLOCKED
    assert rules == ["threat-runtime-obfuscation-steganography"]


def test_the_gate_clears_only_once_a_waiver_is_written(cache_home):
    """The other half of the pair: accepted.json does what the ledger cannot."""
    accepted = cache_home / "guarddog-cached" / "accepted.json"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(json.dumps({"schema": 1, "accepted": {
        "google-genai==2.11.0": {"rules": ["threat-runtime-obfuscation-steganography"]}}}))

    entry = {"errors": {}, "risks": STEGO_REPORT["risks"]}
    verdict, _ = gd.verdict_for(entry, gd.waived_rules(gd.load_accepted(),
                                                       "google-genai", "2.11.0"))

    assert verdict == gd.CLEAN


# --- what is offered for review ------------------------------------------

def test_a_report_with_no_findings_is_never_offered(cache_home):
    """Several hundred clean rows would bury the handful of real decisions."""
    write_report(cache_home, "six==1.17.0@3.1.0", CLEAN_REPORT)

    reports, _ = gr.stored_reports()

    assert gr.pending(reports, gr.load_reviewed()) == []


def test_a_report_with_findings_is_pending_until_reviewed(cache_home):
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()

    assert [r.key for r in gr.pending(reports, ledger)] == ["google-genai==2.11.0@3.1.0"]

    gr.record_review(ledger, reports[0], "waived", "rule defect")

    assert gr.pending(reports, ledger) == []


def test_blocking_reviews_come_before_advisory_ones(cache_home):
    """Something is waiting on the blocking ones; nothing is waiting on the rest.

    The advisory package is named so that it sorts *first* alphabetically:
    otherwise ordering by key alone produces the same list and the test
    passes without the severity ever being consulted.
    """
    write_report(cache_home, "anthropic==0.121.0@3.1.0", ADVISORY_REPORT)
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)

    reports, _ = gr.stored_reports()

    assert [r.key for r in gr.pending(reports, gr.load_reviewed())] == [
        "google-genai==2.11.0@3.1.0", "anthropic==0.121.0@3.1.0"]


def test_a_review_does_not_carry_to_a_newer_guarddog(cache_home):
    """Same code, different rules: findings nobody has seen may be waiting."""
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()
    gr.record_review(ledger, reports[0], "waived", "rule defect")

    write_report(cache_home, "google-genai==2.11.0@3.2.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()

    assert [r.key for r in gr.pending(reports, ledger)] == ["google-genai==2.11.0@3.2.0"]


def test_a_changed_finding_reopens_a_reviewed_report(cache_home):
    """A review answers for what the reviewer saw, and for nothing else."""
    key = "google-genai==2.11.0@3.1.0"
    write_report(cache_home, key, STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()
    gr.record_review(ledger, reports[0], "waived", "matched eval( inside Retrieval(")

    moved = json.loads(json.dumps(STEGO_REPORT))
    moved["results"]["threat-runtime-obfuscation-steganography"][0]["match"] = "exec("
    write_report(cache_home, key, moved)
    reports, _ = gr.stored_reports()

    assert [r.key for r in gr.pending(reports, ledger)] == [key], \
        "a review of `eval(` was carried over to a report matching `exec(`"


def test_a_rejected_finding_is_not_offered_again(cache_home):
    """Reviewed-and-bad is a decision, not outstanding work."""
    write_report(cache_home, "evil==1.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()
    gr.record_review(ledger, reports[0], "rejected", "real download-and-execute")

    assert gr.pending(reports, ledger) == []


# --- what a reviewer is shown --------------------------------------------

def test_the_matched_text_is_carried_into_the_review(cache_home):
    """`risks` says a rule fired; only `results` says what it fired on."""
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)

    reports, _ = gr.stored_reports()
    finding = reports[0].subject[0]

    assert finding["rule"] == "threat-runtime-obfuscation-steganography"
    assert finding["severity"] == "high"
    assert finding["matched"] == ["eval("]
    assert "eval(" in gr.describe(reports[0].subject)[0]


def test_a_finding_with_no_location_still_says_what_was_found(cache_home):
    """ruff's `bundled_binary`, as GuardDog 3.1.0 reports it.

    A metadata risk is about the package as a whole and names no file, so
    rendering it by location alone produces a line that says nothing.
    """
    write_report(cache_home, "ruff==0.15.21@3.1.0", {
        "package": "ruff", "issues": 3, "errors": {}, "results": {},
        "risks": [{
            "name": "risk.metadata.bundled-binary",
            "category": "metadata",
            "severity": "medium",
            "threat_rule": "bundled_binary",
            "threat_location": "",
            "file_path": "",
            "threat_description": "Binary file/s detected in package:\n"
                                  "39801cf45e9c9a4b7db3cf7c8347909d91fe7a91e74f6100f22595fc0ecf208f: "
                                  "ruff (elf)",
        }],
    })

    reports, _ = gr.stored_reports()
    line = gr.describe(reports[0].subject)[0]

    assert line.startswith("bundled_binary [medium]")
    assert "Binary file/s detected in package" in line
    assert " at " not in line, "a risk with no location must not claim one"


def test_an_unreadable_report_is_reported_not_skipped(cache_home):
    """A shorter list that looks complete is the failure this repo is built against."""
    reports_path = cache_home / "guarddog-cached" / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)
    (reports_path / "truncated.json").write_text('{"risks": [')
    (reports_path / "no-provenance.json").write_text('{"risks": [], "errors": {}}')
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)

    reports, unreadable = gr.stored_reports()

    assert [r.key for r in reports] == ["google-genai==2.11.0@3.1.0"]
    assert sorted(p.name for p, _ in unreadable) == ["no-provenance.json", "truncated.json"]


def test_the_key_comes_from_the_report_not_the_filename(cache_home):
    """Filenames are sanitised and may carry a digest, so they are not the key."""
    path = cache_home / "guarddog-cached" / "reports" / "evil==1.0_x@3.1.0.abc123def456.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(STEGO_REPORT)
    stored["_guarddog_cached"] = {"cache_key": "evil==1.0/x@3.1.0"}
    path.write_text(json.dumps(stored))

    reports, unreadable = gr.stored_reports()

    assert unreadable == []
    assert [r.key for r in reports] == ["evil==1.0/x@3.1.0"]


# --- recording a decision -------------------------------------------------

def test_a_review_must_say_what_was_decided_and_why(cache_home):
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()

    with pytest.raises(ValueError):
        gr.record_review(ledger, reports[0], "waived", "   ")
    with pytest.raises(ValueError):
        gr.record_review(ledger, reports[0], "probably-fine", "looks ok")

    assert ledger["reviewed"] == {}


def test_the_ledger_keeps_the_findings_it_answered_for(cache_home):
    """A ledger row has to be legible without re-opening the report."""
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    reports, _ = gr.stored_reports()
    ledger = gr.load_reviewed()
    gr.record_review(ledger, reports[0], "waived", "eval( inside Retrieval( — rule defect")
    gr.save_reviewed(ledger)

    entry = read_ledger(cache_home)["reviewed"]["google-genai==2.11.0@3.1.0"]
    assert entry["outcome"] == "waived"
    assert entry["note"] == "eval( inside Retrieval( — rule defect"
    assert "eval(" in entry["findings"][0]
    assert entry["reviewed_at"].startswith("20")


def test_a_corrupt_ledger_loses_reviews_rather_than_hiding_findings(cache_home):
    """Failing closed here means offering work twice, which is the safe way."""
    write_report(cache_home, "google-genai==2.11.0@3.1.0", STEGO_REPORT)
    path = cache_home / "guarddog-cached" / "reviewed.json"
    path.write_text("{ this is not json")

    reports, _ = gr.stored_reports()

    assert len(gr.pending(reports, gr.load_reviewed())) == 1