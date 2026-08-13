"""Tests for the shared GuardDog cache and the verdict it derives.

Two things are pinned here that GuardDog's own exit code cannot express:

* a scan that did not fully run is INCOMPLETE, never a pass — "not
  checked" and "no problems found" must not collapse into each other;
* only complete scans are cached, so a transient failure is retried
  rather than frozen into a shared, machine-wide clean bill.

Multi-project failures are reproduced end-to-end against a fake `guarddog`
on PATH, in an isolated XDG_CACHE_HOME. The real cache is never touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_common.security import guarddog_cached as gd

SHIM = '''#!{python}
import sys, json, os, time
if len(sys.argv) == 2 and sys.argv[1] == "--version":
    print("{version}"); sys.exit(0)
name = sys.argv[3]
version = sys.argv[5]
plan = {{}}
if os.environ.get("FAKE_PLAN"):
    plan = json.load(open(os.environ["FAKE_PLAN"]))
# A plan may key a package by name alone, or by "name==version" when a test
# needs the two version spellings of one release to behave differently.
spec = plan.get(name + "==" + version, plan.get(name, {{}}))
if os.environ.get("FAKE_CALLS"):
    with open(os.environ["FAKE_CALLS"], "a") as fh:
        fh.write(name + "==" + version + "\\n")
time.sleep({delay})
if spec.get("garbage"):
    print("this is not json"); sys.exit(spec.get("exit", 0))
report = {{
    "package": name,
    "issues": spec.get("issues", 0),
    "errors": spec.get("errors", {{}}),
    "results": spec.get("results", {{}}),
    "risks": spec.get("risks", []),
    "risk_score": spec.get("risk_score", {{"score": 0.0, "label": "no_risks_detected"}}),
    "path": "/tmp/ephemeral",
}}
for key in spec.get("drop", []):
    report.pop(key, None)
print(json.dumps(report))
sys.exit(spec.get("exit", 0))
'''

#: A high-severity risk: what the gate exists to stop. Shaped exactly like a
#: GuardDog 3.1.0 risk, from `threat-process-download-exec` (severity high,
#: specificity high, so it stands alone rather than needing a capability).
DOWNLOAD_EXEC = {
    "issues": 1,
    "risks": [{
        "name": "risk.process.spawn",
        "category": "process",
        "severity": "high",
        "mitre_tactics": ["execution"],
        "threat_rule": "threat-process-download-exec",
        "threat_description": "Detects download-and-execute patterns",
        "threat_location": "setup.py:3",
        "file_path": "setup.py",
    }],
    "risk_score": {"score": 9.1, "label": "high_risk", "findings_count": 1},
}

#: tqdm 4.67.1's real GuardDog 3.1.0 report, recorded 2026-08-11. An innocent
#: package that scores 7.2/10 `high_risk` because `contrib/telegram.py`
#: mentions api.telegram.org. Blocking on the score label would block this.
TQDM_NOISE = {
    "issues": 6,
    "risks": [
        {"name": "risk.network.outbound", "category": "network", "severity": "medium",
         "mitre_tactics": ["command-and-control"],
         "threat_rule": "threat-network-outbound-shady-links",
         "threat_description": "Detects URLs to URL shorteners, file sharing, and "
                               "suspicious services",
         "threat_location": "tqdm/contrib/telegram.py:26",
         "file_path": "tqdm/contrib/telegram.py"},
        {"name": "risk.network.outbound", "category": "network", "severity": "low",
         "mitre_tactics": ["exfiltration"],
         "threat_rule": "threat-network-exfiltration",
         "threat_description": "Detects URLs to suspicious domains often used for "
                               "exfiltration or C2",
         "threat_location": "tqdm/contrib/telegram.py:26",
         "file_path": "tqdm/contrib/telegram.py"},
    ],
    "risk_score": {"score": 7.2, "label": "high_risk", "findings_count": 4},
}
BROKEN_RULE = {"errors": {"potentially_compromised_email_domain": "Invalid version: '2013-02-16'"}}

#: The google-genai 2.11.0 finding that made raw reports worth keeping, shaped
#: as GuardDog 3.1.0 reported it. A cache entry can say only that
#: `threat-runtime-obfuscation-steganography` fired at line 217 of a test file.
#: What settles the waiver is in `results`: the matched text is the literal
#: string `eval(`, occurring inside the word `Retrieval(`. Reviewing that
#: finding from a cache entry alone is not possible.
STEGO_FALSE_POSITIVE = {
    "issues": 15,
    "results": {
        "threat-runtime-obfuscation-steganography": [{
            "code": "types.Tool(\n    retrieval=types.Retrieval(\n"
                    "        vertex_ai_search=types.VertexAISearch(",
            "location": "google/genai/tests/models/test_generate_content_tools.py:217",
            "match": "eval(",
            "message": "Detects steganography decode followed by code execution",
        }],
    },
    "risks": [{
        "name": "risk.runtime.obfuscation",
        "category": "runtime",
        "severity": "high",
        "mitre_tactics": ["defense-evasion"],
        "threat_rule": "threat-runtime-obfuscation-steganography",
        "threat_description": "Detects steganography decode followed by code execution",
        "threat_location": "google/genai/tests/models/test_generate_content_tools.py:217",
        "file_path": "google/genai/tests/models/test_generate_content_tools.py",
        "threat_code": "retrieval=types.Retrieval(",
    }],
    "risk_score": {"score": 4.9, "label": "low", "findings_count": 9},
}


@pytest.fixture
def fake_guarddog(tmp_path):
    """A `guarddog` on PATH whose version, timing and per-package report are ours."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "guarddog"

    def install(version: str = "2.10.0", delay: float = 0.0) -> Path:
        shim.write_text(SHIM.format(python=sys.executable, version=version, delay=delay))
        shim.chmod(0o755)
        return bin_dir

    install()
    return install


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    return home


def run_project(project_dir: Path, packages, bin_dir: Path, cache_home: Path,
                plan: dict | None = None, wait: bool = True, extra_args=()):
    """Run the wrapper as a project would, in its own directory."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "req.txt").write_text("".join(f"{p}\n" for p in packages))

    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "XDG_CACHE_HOME": str(cache_home),
           "HOME": str(project_dir),
           "FAKE_CALLS": str(project_dir / "calls.txt")}
    if plan is not None:
        plan_file = project_dir / "plan.json"
        plan_file.write_text(json.dumps(plan))
        env["FAKE_PLAN"] = str(plan_file)

    proc = subprocess.Popen([sys.executable, gd.__file__, *extra_args, "req.txt"],
                            cwd=project_dir,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return proc
    out, _ = proc.communicate(timeout=120)
    return proc.returncode, out


def read_cache(cache_home: Path) -> dict:
    return json.loads((cache_home / "guarddog-cached" / "cache.json").read_text())


def reports_of(cache_home: Path) -> Path:
    """The reports directory, spelled out rather than asked of the module."""
    return cache_home / "guarddog-cached" / "reports"


def read_report(cache_home: Path, filename: str) -> dict:
    return json.loads((reports_of(cache_home) / filename).read_text())


def calls_made(project_dir: Path) -> list[str]:
    """Every `name==version` the fake guarddog was asked to scan, in order."""
    path = project_dir / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def write_accepted(cache_home: Path, accepted: dict) -> None:
    path = cache_home / "guarddog-cached" / "accepted.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "accepted": accepted}))


def _empty() -> dict:
    return {"schema": gd.CACHE_SCHEMA, "entries": {}}


# --- the gate: what passes and what does not ------------------------------

def test_a_clean_package_passes(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["ok==1.0"], fake_guarddog(), cache_home)

    assert rc == 0
    assert "clean 1" in out


def test_a_scan_that_did_not_run_never_counts_as_a_pass(tmp_path, fake_guarddog, cache_home):
    """The defect this wrapper exists for: guarddog exits 0 here."""
    rc, out = run_project(tmp_path / "p", ["broken==1.0"], fake_guarddog(), cache_home,
                          plan={"broken": BROKEN_RULE})

    assert rc == 1, "an unchecked package passed the gate"
    assert "INCOMPLETE" in out
    assert "potentially_compromised_email_domain" in out


def test_a_blocking_rule_fails_the_gate(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["evil==1.0"], fake_guarddog(), cache_home,
                          plan={"evil": DOWNLOAD_EXEC})

    assert rc == 1
    assert "BLOCKED" in out and "threat-process-download-exec" in out


def test_an_advisory_finding_is_reported_but_does_not_fail(tmp_path, fake_guarddog, cache_home):
    """26 of 91 real packages trip a heuristic; blocking on those is unusable."""
    rc, out = run_project(tmp_path / "p", ["noisy==1.0"], fake_guarddog(), cache_home,
                          plan={"noisy": TQDM_NOISE})

    assert rc == 0
    assert "threat-network-outbound-shady-links [medium]" in out
    assert "✗ BLOCKED" not in out          # the detail line, not the "BLOCKED 0" tally
    assert "BLOCKED 0" in out


def test_one_bad_package_fails_a_run_of_many(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["ok==1.0", "evil==1.0", "fine==1.0"],
                          fake_guarddog(), cache_home, plan={"evil": DOWNLOAD_EXEC})

    assert rc == 1
    assert "clean 2" in out and "BLOCKED 1" in out


# --- verdicts, as a pure function ----------------------------------------

def _risk(severity: str, rule: str = "threat-process-download-exec",
          name: str = "risk.process.spawn") -> dict:
    return {"name": name, "severity": severity, "threat_rule": rule}


def test_incomplete_outranks_findings():
    """A partial scan's findings say nothing about what the unrun rules missed."""
    entry = {"errors": {"some-rule": "boom"}, "risks": [_risk("high")]}

    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE


def test_the_severity_vocabulary_is_exactly_these_values():
    """Literal values on purpose.

    The severity-parametrized tests below iterate over the constants they are
    testing, so they cannot notice the vocabulary itself changing — they would
    simply run different cases. This test can. Written the same way, and for
    the same reason, as the blocking-rule test it replaces.
    """
    assert gd.RISK_SEVERITIES == ("low", "medium", "high")
    assert gd.BLOCKING_SEVERITY == "high"


@pytest.mark.parametrize("severity", ["high"])
def test_a_high_severity_risk_blocks(severity):
    assert gd.verdict_for({"risks": [_risk(severity)]}, set())[0] == gd.BLOCKED


@pytest.mark.parametrize("severity", ["low", "medium"])
def test_a_risk_below_the_threshold_is_advisory(severity):
    assert gd.verdict_for({"risks": [_risk(severity)]}, set())[0] == gd.ADVISORY


def test_an_unknown_severity_blocks_rather_than_passing():
    """The lesson of the gate this replaces.

    The old gate asked "is this rule name in my blocking list?", so when
    GuardDog 3 renamed all 61 rules the answer was no for every one of them
    and the gate silently passed everything. A vocabulary this wrapper does
    not recognise must fail loudly instead.
    """
    verdict, rules = gd.verdict_for({"risks": [_risk("catastrophic")]}, set())

    assert verdict == gd.BLOCKED
    assert rules == ["threat-process-download-exec"]


@pytest.mark.parametrize("severity", [None, "", 3, {"level": "high"}])
def test_a_missing_or_malformed_severity_blocks(severity):
    assert gd.verdict_for({"risks": [{"threat_rule": "r", "severity": severity}]},
                          set())[0] == gd.BLOCKED


def test_a_risk_with_no_severity_field_at_all_blocks():
    assert gd.verdict_for({"risks": [{"threat_rule": "r"}]}, set())[0] == gd.BLOCKED


def test_the_headline_score_label_does_not_decide_anything():
    """tqdm scores 7.2/10 `high_risk` for naming api.telegram.org.

    Gating on GuardDog's own label would block two of six ordinary packages
    measured on 2026-08-11 (tqdm 7.2, pyyaml 8.8), which is the same
    unusable-noise failure that made the v2 rule list block nothing at all.
    """
    entry = {"risks": [_risk("medium"), _risk("low")],
             "risk_score": {"score": 7.2, "label": "high_risk"}}

    assert gd.verdict_for(entry, set())[0] == gd.ADVISORY


def test_an_empty_entry_is_clean():
    assert gd.verdict_for({"errors": {}, "risks": []}, set()) == (gd.CLEAN, [])


def test_a_waiver_clears_a_blocking_risk():
    entry = {"risks": [_risk("high")]}

    assert gd.verdict_for(entry, {"threat-process-download-exec"}) == (gd.CLEAN, [])


def test_a_waiver_may_name_the_rolled_up_risk_instead_of_the_rule():
    entry = {"risks": [_risk("high")]}

    assert gd.verdict_for(entry, {"risk.process.spawn"}) == (gd.CLEAN, [])


def test_a_waiver_for_one_rule_does_not_clear_a_second_risk():
    entry = {"risks": [_risk("high"), _risk("high", rule="threat-network-reverse-shell")]}

    verdict, rules = gd.verdict_for(entry, {"threat-process-download-exec"})
    assert verdict == gd.BLOCKED and rules == ["threat-network-reverse-shell"]


def test_a_waiver_clears_an_unrun_rule():
    entry = {"errors": {"potentially_compromised_email_domain": "Invalid version"}}

    assert gd.verdict_for(entry, {"potentially_compromised_email_domain"})[0] == gd.CLEAN


# --- waivers -------------------------------------------------------------

def test_a_waiver_lets_a_blocked_package_through(tmp_path, fake_guarddog, cache_home):
    write_accepted(cache_home, {"evil==1.0": {
        "rules": ["threat-process-download-exec"],
        "reason": "vendored build hook, reviewed",
        "by": "bgunyel", "at": "2026-08-10"}})

    rc, out = run_project(tmp_path / "p", ["evil==1.0"], fake_guarddog(), cache_home,
                          plan={"evil": DOWNLOAD_EXEC})

    assert rc == 0
    assert "✗ BLOCKED" not in out
    assert "BLOCKED 0" in out


def test_a_waiver_does_not_carry_to_a_new_package_version(tmp_path, fake_guarddog, cache_home):
    """A new version is new code; the review does not transfer."""
    write_accepted(cache_home, {"evil==1.0": {"rules": ["threat-process-download-exec"]}})

    rc, out = run_project(tmp_path / "p", ["evil==2.0"], fake_guarddog(), cache_home,
                          plan={"evil": DOWNLOAD_EXEC})

    assert rc == 1
    assert "BLOCKED" in out


def test_a_corrupt_accepted_file_waives_nothing(cache_home):
    path = gd.accepted_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert gd.waived_rules(gd.load_accepted(), "evil", "1.0") == set()


# --- what gets cached, and what does not ---------------------------------

def test_an_incomplete_scan_is_not_cached(tmp_path, fake_guarddog, cache_home):
    """Freezing a non-result into a machine-wide cache is the original bug."""
    run_project(tmp_path / "p", ["broken==1.0"], fake_guarddog(), cache_home,
                plan={"broken": BROKEN_RULE})

    assert read_cache(cache_home)["entries"] == {}


def test_an_incomplete_scan_is_retried_next_run(tmp_path, fake_guarddog, cache_home):
    bin_dir = fake_guarddog()
    run_project(tmp_path / "p", ["flaky==1.0"], bin_dir, cache_home,
                plan={"flaky": {"errors": {"download-package": "network is down"}}})

    # the transient failure clears
    rc, out = run_project(tmp_path / "p", ["flaky==1.0"], bin_dir, cache_home, plan={})

    assert "[scanning] flaky==1.0" in out, "a failed scan was frozen instead of retried"
    assert rc == 0


def test_a_complete_scan_is_cached_even_with_findings(tmp_path, fake_guarddog, cache_home):
    bin_dir = fake_guarddog()
    run_project(tmp_path / "p", ["noisy==1.0"], bin_dir, cache_home, plan={"noisy": TQDM_NOISE})

    _, out = run_project(tmp_path / "p", ["noisy==1.0"], bin_dir, cache_home, plan={"noisy": TQDM_NOISE})
    assert "[cached] noisy==1.0" in out
    assert "threat-network-outbound-shady-links [medium]" in out, \
        "cached findings must still be reported"


# --- the raw report kept beside the cache --------------------------------

def test_the_report_keeps_the_evidence_the_cache_entry_drops(tmp_path, fake_guarddog,
                                                              cache_home):
    """The cache says a rule fired; only the report says what it fired on.

    This is the whole reason the reports exist. `results` and `threat_code`
    are trimmed from the entry on size grounds, and they are exactly what a
    human needs to tell a rule defect from a real finding.
    """
    run_project(tmp_path / "p", ["genai==2.11.0"], fake_guarddog(), cache_home,
                plan={"genai": STEGO_FALSE_POSITIVE})

    report = read_report(cache_home, "genai==2.11.0@2.10.0.json")
    match = report["results"]["threat-runtime-obfuscation-steganography"][0]
    assert match["match"] == "eval("
    assert "Retrieval(" in match["code"]
    assert report["risks"][0]["threat_code"] == "retrieval=types.Retrieval("

    entry = read_cache(cache_home)["entries"]["genai==2.11.0@2.10.0"]
    assert "results" not in entry, "the entry is meant to stay a summary"
    assert "threat_code" not in entry["risks"][0]


def test_the_report_is_named_after_the_cache_key(tmp_path, fake_guarddog, cache_home):
    """A human reading the cache must be able to find the evidence for a row."""
    run_project(tmp_path / "p", ["ok==1.0"], fake_guarddog(version="3.1.0"), cache_home)

    assert "ok==1.0@3.1.0" in read_cache(cache_home)["entries"]
    assert (reports_of(cache_home) / "ok==1.0@3.1.0.json").is_file()


def test_the_report_records_what_was_asked_for(tmp_path, fake_guarddog, cache_home):
    run_project(tmp_path / "p", ["ok==1.0"], fake_guarddog(), cache_home)

    provenance = read_report(cache_home, "ok==1.0@2.10.0.json")["_guarddog_cached"]
    assert provenance["package"] == "ok"
    assert provenance["version"] == "1.0"
    assert provenance["guarddog_version"] == "2.10.0"
    assert provenance["cache_key"] == "ok==1.0@2.10.0"


def test_the_report_and_the_entry_agree_on_when_the_scan_happened(tmp_path, fake_guarddog,
                                                                  cache_home, monkeypatch):
    """One reading of the clock, so the pair cannot drift apart.

    The clock is replaced with one that never returns the same value twice.
    Against the real clock both readings land in the same second and the
    assertion would hold however many times the code looked.
    """
    monkeypatch.setenv("PATH", f"{fake_guarddog()}:/usr/bin:/bin")

    ticks = iter(["2026-08-12T09:00:00+00:00", "2026-08-12T09:00:01+00:00"])

    class _Reading:
        def __init__(self, value):
            self._value = value

        def isoformat(self, timespec="seconds"):
            return self._value

    class _Clock:
        @staticmethod
        def now(tz=None):
            return _Reading(next(ticks))

    monkeypatch.setattr(gd, "datetime", _Clock)

    entry = gd.scan_package("ok", "1.0", "2.10.0")

    report = read_report(cache_home, "ok==1.0@2.10.0.json")
    assert report["_guarddog_cached"]["scanned_at"] == entry["scanned_at"]


def test_an_incomplete_package_is_not_sent_to_a_report_that_is_not_there(tmp_path,
                                                                        fake_guarddog,
                                                                        cache_home):
    """It has no report, and naming the path anyway sends a reader nowhere."""
    rc, out = run_project(tmp_path / "p", ["broken==1.0"], fake_guarddog(), cache_home,
                          plan={"broken": BROKEN_RULE})

    assert rc == 1
    assert "INCOMPLETE" in out
    assert "matched code:" not in out


def test_an_incomplete_scan_leaves_no_report(tmp_path, fake_guarddog, cache_home):
    """Report and entry appear together, so neither can be stale beside the other."""
    run_project(tmp_path / "p", ["broken==1.0"], fake_guarddog(), cache_home,
                plan={"broken": BROKEN_RULE})

    assert read_cache(cache_home)["entries"] == {}
    assert list(reports_of(cache_home).glob("*.json")) == []


def test_a_version_cannot_escape_the_reports_directory(tmp_path, fake_guarddog, cache_home):
    """The filename comes from parsed input, which constrains names but not versions."""
    run_project(tmp_path / "p", ["evil==1.0/../../pwned"], fake_guarddog(), cache_home)

    assert not (cache_home / "pwned@2.10.0.json").exists()
    assert not (cache_home / "guarddog-cached" / "pwned@2.10.0.json").exists()
    written = list(reports_of(cache_home).glob("*.json"))
    assert len(written) == 1
    assert written[0].parent == reports_of(cache_home)


def test_two_versions_that_sanitise_alike_keep_separate_reports(tmp_path, fake_guarddog,
                                                                cache_home):
    """Replacing unsafe characters is many-to-one; the filenames must not be.

    `1.0/x` and `1.0:x` are different packages. If both land on one report,
    the evidence shown for one is the evidence gathered for the other.
    """
    run_project(tmp_path / "p", ["evil==1.0/x", "evil==1.0:x"], fake_guarddog(), cache_home)

    assert len(list(reports_of(cache_home).glob("*.json"))) == 2


def test_a_blocked_package_is_told_where_its_matched_code_is(tmp_path, fake_guarddog,
                                                             cache_home):
    """A waiver decided without reading the match is the rubber stamp to avoid."""
    rc, out = run_project(tmp_path / "p", ["genai==2.11.0"], fake_guarddog(), cache_home,
                          plan={"genai": STEGO_FALSE_POSITIVE})

    assert rc == 1
    assert "genai==2.11.0@2.10.0.json" in out


def test_the_verdict_is_recomputed_from_the_cache_not_stored(tmp_path, fake_guarddog, cache_home):
    """Accepting a finding must re-decide cached packages without re-scanning."""
    bin_dir = fake_guarddog()
    rc, _ = run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": DOWNLOAD_EXEC})
    assert rc == 1

    write_accepted(cache_home, {"evil==1.0": {"rules": ["threat-process-download-exec"]}})
    rc, out = run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": DOWNLOAD_EXEC})

    assert rc == 0
    assert "[cached] evil==1.0" in out, "the entry was re-scanned instead of re-judged"


# --- scan_package: the tool failing must never read as clean --------------

def test_unparseable_output_becomes_an_error_not_a_clean_result(tmp_path, fake_guarddog,
                                                                cache_home, monkeypatch):
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"junk": {"garbage": True}}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    entry = gd.scan_package("junk", "1.0")

    assert entry["errors"], "unparseable output was treated as a clean scan"
    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE


def test_a_nonzero_exit_becomes_an_error_even_with_valid_json(tmp_path, fake_guarddog,
                                                              cache_home, monkeypatch):
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"misused": {"exit": 2}}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    entry = gd.scan_package("misused", "1.0")

    assert "guarddog-cached" in entry["errors"]


def test_the_raw_rule_matches_are_not_cached(tmp_path, fake_guarddog,
                                             cache_home, monkeypatch):
    """`results` is shape-checked and discarded, not stored.

    It is roughly twice the size of `risks` in a machine-wide cache and
    nothing reads it now the verdict comes from `risks`. Keeping a field that
    looks like it feeds the gate but does not is the hazard this rework is
    about.
    """
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"ok": {"results": {
        "capability-network-outbound": [{"location": "a.py:1"}]}}}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    entry = gd.scan_package("ok", "1.0")

    assert "findings" not in entry
    assert "results" not in entry
    assert entry["risks"] == []
    assert gd.verdict_for(entry, set())[0] == gd.CLEAN


def test_the_ephemeral_scan_path_is_not_cached(tmp_path, fake_guarddog, cache_home, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_guarddog()}:/usr/bin:/bin")

    assert "path" not in gd.scan_package("ok", "1.0")


def test_the_risk_fields_the_verdict_needs_are_kept(tmp_path, fake_guarddog,
                                                    cache_home, monkeypatch):
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"evil": DOWNLOAD_EXEC}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    risk = gd.scan_package("evil", "1.0")["risks"][0]

    # Literal keys: a test that asks the module which fields it keeps cannot
    # notice the module dropping one.
    assert risk["severity"] == "high"
    assert risk["threat_rule"] == "threat-process-download-exec"
    assert risk["name"] == "risk.process.spawn"
    assert risk["threat_location"] == "setup.py:3"
    assert risk["mitre_tactics"] == ["execution"]


def test_the_bulky_source_excerpt_is_not_cached(tmp_path, fake_guarddog,
                                                cache_home, monkeypatch):
    """The cache is machine-wide and long-lived; `threat_code` is unbounded."""
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    noisy = {"risks": [dict(DOWNLOAD_EXEC["risks"][0], threat_code="x" * 5000)]}
    plan.write_text(json.dumps({"big": noisy}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    assert "threat_code" not in gd.scan_package("big", "1.0")["risks"][0]


# --- the report shape we depend on ---------------------------------------

@pytest.mark.parametrize("report", [
    '{"package": "x", "issues": 0, "results": {}, "risks": []}',      # errors key gone
    '{"package": "x", "issues": 0, "errors": {}, "risks": []}',       # results key gone
    '{"package": "x", "issues": 0, "errors": [], "results": {}, "risks": []}',   # errors not an object
    '{"package": "x", "issues": 0, "errors": {}, "results": null, "risks": []}',  # results not an object
    '{"package": "x", "issues": 0, "errors": {}, "results": {}}',     # risks key gone
    '{"package": "x", "issues": 0, "errors": {}, "results": {}, "risks": {}}',   # risks not a list
    '{"package": "x", "problems": {}, "matches": {}}',               # renamed wholesale
])
def test_an_unrecognised_report_shape_is_not_read_as_clean(tmp_path, fake_guarddog,
                                                           cache_home, monkeypatch, report):
    """The exit-code trap one level up.

    If a future GuardDog renames these fields, `.get()` returning nothing must
    not look like "no errors, no findings" — that would make every package on
    the machine read as clean.
    """
    bin_dir = tmp_path / "bin2"
    bin_dir.mkdir()
    shim = bin_dir / "guarddog"
    shim.write_text(f'#!{sys.executable}\n'
                    f'import sys\n'
                    f'if len(sys.argv) == 2 and sys.argv[1] == "--version":\n'
                    f'    print("9.9.9"); sys.exit(0)\n'
                    f'print({report!r}); sys.exit(0)\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    entry = gd.scan_package("x", "1.0")

    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE
    assert "unrecognised report shape" in str(entry["errors"])


def test_a_report_with_no_risks_field_cannot_pass_the_gate(tmp_path, fake_guarddog,
                                                           cache_home, monkeypatch):
    """The field the verdict rests on cannot be optional.

    If a later GuardDog drops or renames `risks`, "no risks in the report"
    must not read as "no risks in the package" — that is precisely how the
    rule-name gate went inert.
    """
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"x": {"drop": ["risks"]}}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    entry = gd.scan_package("x", "1.0")

    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE
    assert "unrecognised report shape" in str(entry["errors"])


def test_a_failed_scan_keeps_guarddogs_own_explanation(tmp_path, cache_home, monkeypatch):
    """A failing v3 report has no `results` key at all, and that is not corruption.

    Reported verbatim from GuardDog 3.1.0 on 2026-08-11: the sandbox could not
    start, so no rule ran and the report's keys were exactly
    ['package', 'issues', 'errors']. Requiring `results` unconditionally made
    the wrapper answer "unrecognised report shape" and throw the message below
    away — the verdict stayed right, the diagnosis did not survive.
    """
    report = json.dumps({
        "package": "x",
        "issues": 0,
        "errors": {"download-package": "Sandboxed extraction failed: Fatal Python "
                                       "error: _Py_HashRandomization_Init: failed to "
                                       "get random numbers to initialize Python"},
    })
    bin_dir = tmp_path / "bin3"
    bin_dir.mkdir()
    shim = bin_dir / "guarddog"
    shim.write_text(f'#!{sys.executable}\n'
                    f'import sys\n'
                    f'if len(sys.argv) == 2 and sys.argv[1] == "--version":\n'
                    f'    print("3.1.0"); sys.exit(0)\n'
                    f'print({report!r}); sys.exit(0)\n')
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    entry = gd.scan_package("x", "1.0")

    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE, "a dead scan passed the gate"
    assert "unrecognised report shape" not in str(entry["errors"]), \
        "the wrapper mis-diagnosed a legible failure as a corrupt report"
    assert "Sandboxed extraction failed" in entry["errors"]["download-package"]
    assert "_Py_HashRandomization_Init" in entry["errors"]["download-package"]


def test_a_failed_scan_reports_the_rule_that_failed_by_name(tmp_path, cache_home, monkeypatch):
    """`render` must reach the real message; INCOMPLETE alone is not a diagnosis."""
    entry = {
        "errors": {"download-package": "Sandboxed extraction failed: ..."},
        "risks": [],
    }

    verdict, rules = gd.verdict_for(entry, set())
    detail = gd.render("x==1.0", entry, verdict, rules)

    assert "download-package" in detail
    assert "Sandboxed extraction failed" in detail


# --- the time budget ------------------------------------------------------

def test_a_spent_budget_starts_no_new_scans(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["a==1.0", "b==1.0"], fake_guarddog(), cache_home,
                          extra_args=["--time-budget", "0"])

    assert rc == gd.EXIT_UNFINISHED
    assert "[scanning]" not in out
    assert out.count("[skipped]") == 2
    assert read_cache(cache_home)["entries"] == {}


def test_an_unfinished_run_is_not_a_pass(tmp_path, fake_guarddog, cache_home):
    rc, _ = run_project(tmp_path / "p", ["a==1.0"], fake_guarddog(), cache_home,
                        extra_args=["--time-budget", "0"])

    assert rc != 0, "a run that scanned nothing reported success"


def test_a_spent_budget_still_evaluates_cached_packages(tmp_path, fake_guarddog, cache_home):
    """Cached entries cost nothing, and the report is more use for including them."""
    bin_dir = fake_guarddog()
    run_project(tmp_path / "p", ["known==1.0"], bin_dir, cache_home)

    rc, out = run_project(tmp_path / "p", ["known==1.0", "new==1.0"], bin_dir, cache_home,
                          extra_args=["--time-budget", "0"])

    assert "[cached] known==1.0" in out
    assert "[skipped] new==1.0" in out
    assert rc == gd.EXIT_UNFINISHED


def test_a_definite_failure_outranks_an_unfinished_run(tmp_path, fake_guarddog, cache_home):
    """More scanning will not un-block a blocked package."""
    bin_dir = fake_guarddog()
    run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": DOWNLOAD_EXEC})

    rc, out = run_project(tmp_path / "p", ["evil==1.0", "new==1.0"], bin_dir, cache_home,
                          plan={"evil": DOWNLOAD_EXEC}, extra_args=["--time-budget", "0"])

    assert rc == 1, "an unfinished run masked a blocked package"
    assert "[skipped] new==1.0" in out


def test_budgeted_runs_converge_on_a_full_sweep(tmp_path, fake_guarddog, cache_home):
    """The intended workflow: chip away in slices until one run completes."""
    bin_dir = fake_guarddog()
    packages = ["a==1.0", "b==1.0", "c==1.0"]

    # Nothing cached and no time: everything skipped.
    rc, _ = run_project(tmp_path / "p", packages, bin_dir, cache_home,
                        extra_args=["--time-budget", "0"])
    assert rc == gd.EXIT_UNFINISHED

    # A generous budget finishes and passes.
    rc, out = run_project(tmp_path / "p", packages, bin_dir, cache_home,
                          extra_args=["--time-budget", "600"])
    assert rc == 0
    assert out.count("[scanning]") == 3

    # And now even a zero budget passes, because nothing new needs scanning.
    rc, out = run_project(tmp_path / "p", packages, bin_dir, cache_home,
                          extra_args=["--time-budget", "0"])
    assert rc == 0
    assert "[skipped]" not in out


def test_without_a_budget_everything_is_scanned(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["a==1.0", "b==1.0"], fake_guarddog(), cache_home)

    assert rc == 0
    assert out.count("[scanning]") == 2
    assert "[skipped]" not in out


# --- the multi-project failures -------------------------------------------

def test_concurrent_projects_do_not_clobber_each_other(tmp_path, fake_guarddog, cache_home):
    """Before merge-on-save, whichever finished last erased the other's run."""
    bin_dir = fake_guarddog(delay=0.2)

    procs = [
        run_project(tmp_path / "projA", ["a1==1.0", "a2==1.0", "a3==1.0"], bin_dir, cache_home, wait=False),
        run_project(tmp_path / "projB", ["b1==1.0", "b2==1.0", "b3==1.0"], bin_dir, cache_home, wait=False),
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    keys = set(read_cache(cache_home)["entries"])
    # Literal keys, never gd.entry_key(): a test that builds its expectation
    # with the function under test cannot see that function change.
    expected = {f"{n}==1.0@2.10.0" for n in ("a1", "a2", "a3", "b1", "b2", "b3")}
    assert keys == expected, f"lost entries: {expected - keys}"


def test_upgrading_guarddog_rescans_without_destroying_the_old_entries(tmp_path, fake_guarddog, cache_home):
    bin_dir = fake_guarddog(version="2.10.0")
    _, first = run_project(tmp_path / "projA", ["pkg==1.0"], bin_dir, cache_home)
    assert "[scanning] pkg==1.0" in first

    fake_guarddog(version="2.11.0")
    _, second = run_project(tmp_path / "projB", ["pkg==1.0"], bin_dir, cache_home)
    assert "[scanning] pkg==1.0" in second, "the new GuardDog reused a verdict from the old one"

    assert set(read_cache(cache_home)["entries"]) == {"pkg==1.0@2.10.0", "pkg==1.0@2.11.0"}


def test_a_scan_by_one_project_is_reused_by_another(tmp_path, fake_guarddog, cache_home):
    """The point of the shared cache."""
    bin_dir = fake_guarddog()
    _, first = run_project(tmp_path / "projA", ["shared==1.0"], bin_dir, cache_home)
    assert "[scanning] shared==1.0" in first

    _, second = run_project(tmp_path / "projB", ["shared==1.0"], bin_dir, cache_home)
    assert "[cached] shared==1.0" in second
    assert "1 cached, 0 scanned" in second


def test_rolling_guarddog_back_finds_the_old_entries_intact(tmp_path, fake_guarddog, cache_home):
    bin_dir = fake_guarddog(version="2.10.0")
    run_project(tmp_path / "proj", ["pkg==1.0"], bin_dir, cache_home)
    fake_guarddog(version="2.11.0")
    run_project(tmp_path / "proj", ["pkg==1.0"], bin_dir, cache_home)

    fake_guarddog(version="2.10.0")
    _, out = run_project(tmp_path / "proj", ["pkg==1.0"], bin_dir, cache_home)
    assert "[cached] pkg==1.0" in out


# --- the cache primitives -------------------------------------------------

def test_entry_key_carries_all_three_facts():
    assert gd.entry_key("requests", "2.33.1", "2.10.0") == "requests==2.33.1@2.10.0"


def test_a_waiver_key_deliberately_omits_the_guarddog_version():
    """The review was of the package's code, which a GuardDog upgrade does not change."""
    assert gd.waiver_key("requests", "2.33.1") == "requests==2.33.1"


def test_saving_preserves_entries_written_by_another_project(cache_home):
    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"theirs==1.0@2.10.0": {}}})

    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"ours==1.0@2.10.0": {}}})

    assert set(gd.load_cache()["entries"]) == {"theirs==1.0@2.10.0", "ours==1.0@2.10.0"}


def test_this_runs_result_wins_over_the_one_on_disk(cache_home):
    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"pkg==1.0@2.10.0": {"issues": 1}}})

    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"pkg==1.0@2.10.0": {"issues": 0}}})

    assert gd.load_cache()["entries"]["pkg==1.0@2.10.0"]["issues"] == 0


def test_the_merge_is_visible_through_the_callers_reference(cache_home):
    """`main` holds `cache["entries"]`; rebinding it would strand that alias."""
    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"theirs==1.0@2.10.0": {}}})

    cache = _empty()
    entries = cache["entries"]
    gd.save_cache(cache)

    assert "theirs==1.0@2.10.0" in entries


def test_no_temporary_files_are_left_behind(cache_home):
    gd.save_cache(_empty())

    leftovers = [p.name for p in (cache_home / "guarddog-cached").iterdir() if p.name.endswith(".tmp")]
    assert not leftovers


def test_the_cache_is_replaced_by_rename_not_rewritten_in_place(cache_home):
    """A reader in another project must never observe a half-written file."""
    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"a==1.0@2.10.0": {}}})
    before = gd.cache_path().stat().st_ino

    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"b==1.0@2.10.0": {}}})

    assert gd.cache_path().stat().st_ino != before


def test_a_failed_write_leaves_the_previous_cache_intact(cache_home, monkeypatch):
    gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"good==1.0@2.10.0": {}}})

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(gd.os, "replace", boom)
    with pytest.raises(OSError):
        gd.save_cache({"schema": gd.CACHE_SCHEMA, "entries": {"new==1.0@2.10.0": {}}})

    # No monkeypatch.undo() here: it would also revert the `cache_home`
    # fixture's XDG_CACHE_HOME and point the assertions at the real cache.
    assert set(gd.load_cache()["entries"]) == {"good==1.0@2.10.0"}
    leftovers = [p.name for p in (cache_home / "guarddog-cached").iterdir() if p.name.endswith(".tmp")]
    assert not leftovers, f"a failed write abandoned {leftovers}"


@pytest.mark.parametrize("payload", [
    '{"schema": 3, "entries": {"trunc',                              # corrupt
    '[]',                                                            # not a mapping
    '{"schema": 3, "entries": []}',                                  # entries not a mapping
    '{"guarddog_version": "2.10.0", "entries": {"pkg==1.0": {}}}',   # schema 1
    '{"schema": 2, "entries": {"pkg==1.0@2.10.0": {"output": "x"}}}',  # schema 2: no verdict data
    # schema 3 stored matched rule *names*, which GuardDog 3 renamed wholesale;
    # it has no `risks`, so the current verdict would read it as clean.
    '{"schema": 3, "entries": {"pkg==1.0@2.10.0": '
    '{"errors": {}, "findings": {"code-execution": [{}]}}}}',
])
def test_an_unusable_or_stale_cache_is_discarded(cache_home, payload):
    path = gd.cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)

    assert gd.load_cache() == _empty()


def test_a_legacy_project_root_cache_is_removed_without_being_trusted(tmp_path, cache_home, monkeypatch):
    """Its entries predate structured results, so nothing says whether they completed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / gd.LEGACY_CACHE_NAME).write_text(json.dumps({
        "guarddog_version": "2.9.0", "entries": {"old==1.0": {"exit_code": 0}}}))

    assert gd.discard_legacy_cache() is True
    assert not (tmp_path / gd.LEGACY_CACHE_NAME).exists()


def test_no_legacy_file_is_a_no_op(tmp_path, cache_home, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert gd.discard_legacy_cache() is False


# --- the two spellings of one release ------------------------------------
#
# `uv` takes a version from the wheel filename, PyPI keys its release index
# on the PEP 440 canonical form, and for four of cuda-toolkit's 39 releases
# those disagree: the file is `cuda_toolkit-13.0.3.0-py2.py3-none-any.whl`
# and the release is `13.0.3`. A lock saying `13.0.3.0` therefore asked
# GuardDog for a version that does not exist, and an unscannable package is
# an INCOMPLETE that no adjudication can clear.

#: What GuardDog 3.1.0 really reported for `cuda-toolkit==13.0.3.0`,
#: recorded 2026-08-13.
NO_SUCH_VERSION = {"errors": {
    "download-package": "Version 13.0.3.0 for package cuda-toolkit doesn't exist."}}


def test_a_version_pypi_spells_differently_is_found_on_a_second_attempt(
        tmp_path, fake_guarddog, cache_home):
    project = tmp_path / "p"
    rc, out = run_project(project, ["cuda-toolkit==13.0.3.0"], fake_guarddog(), cache_home,
                          plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                                "cuda-toolkit==13.0.3": {}})

    assert rc == 0
    assert "INCOMPLETE 0" in out
    assert calls_made(project) == ["cuda-toolkit==13.0.3.0", "cuda-toolkit==13.0.3"]


def test_the_substitution_is_announced_rather_than_made_silently(
        tmp_path, fake_guarddog, cache_home):
    _, out = run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(),
                         cache_home, plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                                           "cuda-toolkit==13.0.3": {}})

    assert "13.0.3" in out


def test_the_lock_spelling_stays_the_cache_key(tmp_path, fake_guarddog, cache_home):
    """Keyed on what the lock says, so the next sweep's parse hits the cache."""
    run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(), cache_home,
                plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                      "cuda-toolkit==13.0.3": {}})

    assert set(read_cache(cache_home)["entries"]) == {"cuda-toolkit==13.0.3.0@2.10.0"}


def test_the_report_is_filed_under_the_lock_spelling_too(tmp_path, fake_guarddog, cache_home):
    """A BLOCKED package is pointed at `report_path(name, version)`; it must exist."""
    run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(), cache_home,
                plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                      "cuda-toolkit==13.0.3": {}})

    assert [p.name for p in reports_of(cache_home).iterdir()] == [
        "cuda-toolkit==13.0.3.0@2.10.0.json"]


def test_the_entry_records_which_version_was_actually_scanned(
        tmp_path, fake_guarddog, cache_home):
    run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(), cache_home,
                plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                      "cuda-toolkit==13.0.3": {}})

    entry = read_cache(cache_home)["entries"]["cuda-toolkit==13.0.3.0@2.10.0"]
    assert entry["scanned_version"] == "13.0.3"


def test_the_report_says_which_release_it_describes(tmp_path, fake_guarddog, cache_home):
    """Otherwise the stored report claims to be of a release PyPI does not have."""
    run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(), cache_home,
                plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                      "cuda-toolkit==13.0.3": {}})

    report = read_report(cache_home, "cuda-toolkit==13.0.3.0@2.10.0.json")
    provenance = report["_guarddog_cached"]
    assert provenance["version"] == "13.0.3.0"
    assert provenance["scanned_version"] == "13.0.3"


def test_an_ordinary_scan_is_never_retried(tmp_path, fake_guarddog, cache_home):
    """`1.0` canonicalises to `1`, so only the success suppresses a second call."""
    project = tmp_path / "p"
    run_project(project, ["ok==1.0"], fake_guarddog(), cache_home)

    assert calls_made(project) == ["ok==1.0"]


def test_a_failure_that_is_not_about_spelling_is_not_laundered_into_a_pass(
        tmp_path, fake_guarddog, cache_home):
    """`2.114.0` is a real release key; `2.114` is not, so the retry finds nothing."""
    project = tmp_path / "p"
    rc, out = run_project(project, ["docling==2.114.0"], fake_guarddog(), cache_home,
                          plan={"docling==2.114.0": {
                                    "errors": {"download-package": "connection reset"}},
                                "docling==2.114": {
                                    "errors": {"download-package": "no such version"}}})

    assert rc == 1
    assert "INCOMPLETE 1" in out
    assert calls_made(project) == ["docling==2.114.0", "docling==2.114"]


def test_a_laundered_scan_is_not_cached(tmp_path, fake_guarddog, cache_home):
    run_project(tmp_path / "p", ["docling==2.114.0"], fake_guarddog(), cache_home,
                plan={"docling==2.114.0": {"errors": {"download-package": "connection reset"}},
                      "docling==2.114": {"errors": {"download-package": "no such version"}}})

    assert read_cache(cache_home)["entries"] == {}


def test_the_original_error_survives_a_failed_retry(tmp_path, fake_guarddog, cache_home):
    """The retry's message would name a version the user never asked for."""
    _, out = run_project(tmp_path / "p", ["docling==2.114.0"], fake_guarddog(), cache_home,
                         plan={"docling==2.114.0": {
                                   "errors": {"download-package": "connection reset"}},
                               "docling==2.114": {
                                   "errors": {"download-package": "no such version"}}})

    assert "connection reset" in out


def test_a_waiver_is_keyed_on_the_version_the_lock_shows(tmp_path, fake_guarddog, cache_home):
    """A reviewer waives what they can see in `uv.lock`, not PyPI's spelling."""
    write_accepted(cache_home, {"cuda-toolkit==13.0.3.0": {
        "rules": ["threat-process-download-exec"],
        "reason": "reviewed", "by": "bgunyel", "at": "2026-08-13"}})

    rc, out = run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(),
                          cache_home, plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                                            "cuda-toolkit==13.0.3": DOWNLOAD_EXEC})

    assert rc == 0
    assert "BLOCKED 0" in out


def test_a_waiver_keyed_on_pypis_spelling_does_not_apply(tmp_path, fake_guarddog, cache_home):
    write_accepted(cache_home, {"cuda-toolkit==13.0.3": {
        "rules": ["threat-process-download-exec"]}})

    rc, out = run_project(tmp_path / "p", ["cuda-toolkit==13.0.3.0"], fake_guarddog(),
                          cache_home, plan={"cuda-toolkit==13.0.3.0": NO_SUCH_VERSION,
                                            "cuda-toolkit==13.0.3": DOWNLOAD_EXEC})

    assert rc == 1
    assert "BLOCKED" in out