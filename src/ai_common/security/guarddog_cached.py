#!/usr/bin/env python3
"""GuardDog cache wrapper, with a verdict GuardDog itself does not provide.

Drop-in replacement for `guarddog pypi verify <requirements-file>` that
caches per-package results so unchanged versions are not re-scanned, and
that decides pass/fail on evidence rather than on an exit code.

Why not the exit code
---------------------
`guarddog pypi scan` exits 0 whether it found nothing, found three
malicious indicators, or never managed to download the package at all.
Nonzero means only that GuardDog was *called* wrong. Gating on it lets a
scan that never ran count as a pass, which is the one failure a
supply-chain gate must not have. So this wrapper scans with
`--output-format=json` and derives its own verdict from two fields:

* `errors` — rules that did not run. Non-empty means the package was not
  fully checked, which is reported as INCOMPLETE and never as a pass.
  "Not checked" and "no problems found" are different states.
* `risks` — GuardDog's correlated findings, each carrying a severity.
  A risk at `BLOCKING_SEVERITY` or above fails the gate; the rest are
  advisory, because ordinary packages trip the noisier heuristics all the
  time (tqdm scores 7.2/10 for an api.telegram.org URL in a file called
  `contrib/telegram.py`).

Not on rule names
-----------------
The gate used to block on a list of seven rule names. GuardDog 3 renamed
every rule onto a new `capability-*`/`threat-*` taxonomy, none of the
seven survived, and the gate went inert — matching nothing, blocking
nothing, announcing nothing. Severity is a three-value vocabulary that
GuardDog derives for us and does not churn.

The same failure must not be possible twice, so **anything this wrapper
does not understand blocks rather than passes**: an unrecognised severity
is treated as blocking, and a completed scan whose report has no `risks`
is INCOMPLETE rather than clean. A gate that stops understanding its
input has to say so.

The verdict is computed when an entry is read, not when it is written, so
changing `BLOCKING_SEVERITY` or accepting a finding re-decides every
cached package without re-scanning anything.

The shared cache
----------------
A scan result is a fact about PyPI — package X at version Y, judged by
GuardDog Z — not a fact about any one project, so results are cached in
`$XDG_CACHE_HOME/guarddog-cached/cache.json` and every project on the
machine reuses every other project's scans. This module lives in
ai-common so a project gets the wrapper by depending on the library
rather than by copying a script into its own `scripts/`.

Being shared is what makes the cache worth having and is also the source
of everything careful about it: entries are keyed on all three of
(name, version, guarddog_version); a save re-reads and merges under an
exclusive lock; and the file is replaced by rename.

**Only complete scans are cached.** A scan that reported `errors` is
re-run next time rather than frozen — a transient network failure should
heal itself, and a permanent one should keep saying so. A scan that never
returns at all is one of those transient failures, so each one is given
`SCAN_TIMEOUT_SECONDS` and then killed: a sweep that blocks forever on a
dead socket reports nothing, caches nothing and heals nothing.

Raw reports
-----------
`reports/` beside the cache keeps GuardDog's report in full, one file per
cache entry, named after the same key.

A cache entry is a summary built for the gate: it drops `results` and
`threat_code`, so it can say *that*
`threat-runtime-obfuscation-steganography` matched, and where, but not
*what* it matched. That is the one thing a human waiving a finding needs
to see. The first waiver reviewed on this machine turned entirely on it —
the matched text was the literal string `eval(`, occurring inside the word
`Retrieval(`, which makes the finding a rule defect rather than a
judgement call. From a cache entry alone that is unknowable.

This wrapper writes a report only when the scan completed, mirroring the
rule for the cache: it never stores a report for a scan that failed. It is
replaced by rename like the cache, but needs no lock: it is one file per
key, and two projects scanning the same package version write the same
content. Keys are sanitised before use as filenames — `parse_requirements`
constrains a package name but not a version, and a cache that names files
after its input must not let that input choose the path.

Reports can also arrive from outside, imported from an earlier sweep, so
the directory may hold reports for versions the cache has no entry for.
Nothing requires the two to be in step: the cache drives the gate and the
reports are read by people.

The reports are an audit trail, not an input to the verdict. Nothing here
reads them back, so deleting the directory costs the evidence and never
the gate.

Accepted findings
-----------------
`accepted.json`, beside the cache, waives named rules for a specific
package version machine-wide. Waivers are keyed on (name, version) rather
than including the GuardDog version, since the decision is about the
package's code and that has not changed; a *new* package version is never
covered by an old waiver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from packaging.utils import canonicalize_version

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None

REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)")

LEGACY_CACHE_NAME = ".guarddog-cache.json"

#: Bumped when the on-disk shape changes. Schemas 1 and 2 stored GuardDog's
#: human-readable text and no structured verdict, so they cannot answer the
#: question the gate asks — "did this scan actually complete?". Schema 3
#: stored matched rule *names*, which GuardDog 3 renamed wholesale. All three
#: are discarded rather than guessed at.
CACHE_SCHEMA = 4

ACCEPTED_SCHEMA = 1

#: Risk severities GuardDog can attach to a correlated risk, weakest first.
#:
#: This replaces a hand-curated list of rule names. GuardDog 3 renamed all 61
#: of its rules onto a new `capability-*`/`threat-*` taxonomy and not one of
#: the seven names the gate blocked on still existed, so the gate matched
#: nothing and passed everything — inert, and silent about it. Severity is a
#: three-value vocabulary that GuardDog derives *for* us, and it survives
#: rules being added, renamed or re-tuned.
#:
#: A risk's severity is its threat rule's severity, downgraded one level when
#: the correlating capability is in another file and two when it is in another
#: category. So `high` means a high-severity rule that either stands alone —
#: install-time, or specific enough to be malware-only — or correlates inside
#: a single file.
RISK_SEVERITIES = ("low", "medium", "high")

#: Risks at or above this severity fail the gate; the rest are advisory.
BLOCKING_SEVERITY = "high"

CLEAN = "clean"
ADVISORY = "advisory"
BLOCKED = "blocked"
INCOMPLETE = "incomplete"

#: `errors` is the field the whole verdict rests on: it is what distinguishes
#: "checked and found nothing" from "never checked". Its *absence* is treated
#: as a scan failure rather than as emptiness — otherwise a future GuardDog
#: that renamed it would make every package on the machine read as clean,
#: which is the exit-code trap one level up.
#:
#: `risks` is required on the same terms and for the same reason: a completed
#: scan that reports no `risks` at all is a report this wrapper cannot read,
#: not a clean package. That is what stops a future rename from repeating the
#: silent-inert-gate failure.
#:
#: `results` is deliberately not in either category. GuardDog omits it
#: entirely from a report whose scan failed, so requiring it unconditionally
#: turns a legible failure into "unrecognised report shape" and discards the
#: real message. It is required only when nothing else explains its absence.
ERRORS_KEY = "errors"
RESULTS_KEY = "results"
RISKS_KEY = "risks"

#: Fields kept from each risk. `threat_code` is dropped: it is a multi-line
#: source excerpt, and this cache is machine-wide and long-lived. The rule,
#: the location and the description are enough to *report* a finding; the
#: excerpt needed to *review* one is in the raw report beside the cache.
RISK_FIELDS = (
    "name", "category", "severity", "mitre_tactics", "threat_rule",
    "threat_description", "threat_location", "file_path",
)

#: Ran out of time before scanning everything. Distinct from 1 (something
#: actually failed the gate) so a caller can tell "not finished" from "no".
EXIT_UNFINISHED = 75

#: Wall-clock limit for one `guarddog pypi scan`, in seconds.
#:
#: GuardDog's `repository_integrity_mismatch` rule clones the upstream repo to
#: compare it against the PyPI tarball, and calls `pygit2.clone_repository`
#: with no timeout and no depth limit. When such a clone's connection dies
#: without a FIN or RST — a NAT table dropping a long transfer, a changed
#: route — the call blocks in `recv()` on a socket the kernel has no reason to
#: time out: with nothing queued to send, TCP never retransmits and so never
#: discovers the peer is gone. Seen as an 82-minute stall on `docling`, with
#: four seconds of CPU, a half-written pack file, and an ESTABLISHED socket
#: with no timers armed. The same package scanned in 138 seconds on the retry.
#:
#: `--time-budget` cannot cover this; it bounds when a scan *starts*. Only a
#: per-scan limit bounds when one ends.
#:
#: The default is deliberately loose. Real scans on one machine ranged from
#: 3 seconds (`orjson`) to 210 (`langchain-openrouter`), so this leaves better
#: than 4x headroom for a slow day. Cutting a live scan short costs an
#: INCOMPLETE that blocks the gate, which is the more expensive mistake.
SCAN_TIMEOUT_SECONDS = 900.0


# --- locations -----------------------------------------------------------

def cache_path() -> Path:
    """Location of the shared cache, honouring XDG_CACHE_HOME."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "guarddog-cached" / "cache.json"


def accepted_path() -> Path:
    return cache_path().parent / "accepted.json"


def reports_dir() -> Path:
    return cache_path().parent / "reports"


def entry_key(name: str, version: str, guarddog_version: str) -> str:
    """The three facts a scan result depends on, as one cache key."""
    return f"{name}=={version}@{guarddog_version}"


#: Everything outside this set is replaced before a key becomes a filename.
#: `REQ_RE` pins a package name to `[A-Za-z0-9][A-Za-z0-9._-]*`, but it
#: accepts any run of non-space, non-`;` characters as the *version* — which
#: admits `/` and `..`. The requirements file is untrusted input to this
#: extent: it is generated by `uv export`, but a wrapper that writes files
#: named after whatever it parsed should not depend on that.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._@=+-]")


def report_path(name: str, version: str, guarddog_version: str) -> Path:
    """Where the raw report for one cache entry lives.

    The filename is the cache key, so a human reading the cache can find the
    evidence for any entry without a lookup table.
    """
    key = entry_key(name, version, guarddog_version)
    safe = _UNSAFE_IN_FILENAME.sub("_", key)
    if safe != key:
        # The substitution is many-to-one, so two distinct keys could land on
        # one filename and a report could describe the wrong package. Only
        # ordinary keys keep a filename that reads like the key; anything that
        # had to be rewritten pays for a disambiguator.
        safe = f"{safe}.{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    return reports_dir() / f"{safe}.json"


def waiver_key(name: str, version: str) -> str:
    """Waivers deliberately omit the GuardDog version — see module docstring."""
    return f"{name}=={version}"


# --- the cache -----------------------------------------------------------

def _empty_cache() -> dict:
    return {"schema": CACHE_SCHEMA, "entries": {}}


@contextmanager
def _cache_lock() -> Iterator[None]:
    """Serialise read-merge-write across projects.

    The lock is held on a sidecar file rather than on the cache itself:
    the cache is replaced by rename, so a lock taken on its inode would
    not be seen by whoever opens the path next.
    """
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return

    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / (path.name + ".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_cache() -> dict:
    """Read the shared cache, or start fresh if it is unusable or stale."""
    path = cache_path()
    if not path.exists():
        return _empty_cache()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return _empty_cache()
    if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA:
        return _empty_cache()
    if not isinstance(data.get("entries"), dict):
        return _empty_cache()
    return data


def _write_json_atomically(path: Path, data: dict) -> None:
    """Replace a file by rename, so no reader ever sees a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_cache(cache: dict) -> None:
    """Merge this run's entries into the shared cache and replace it.

    Re-reads from disk under the lock before writing. Another project may
    have added entries since this run loaded the cache, and writing our
    in-memory copy wholesale would silently delete them — which is the
    one thing a cache shared between projects must not do.

    Entries scanned by this run win on conflict, and are merged back into
    `cache` in place so that a long run picks up what other projects
    finish while it is still going. In place, not by rebinding: the
    caller holds a reference to `cache["entries"]`.
    """
    with _cache_lock():
        for key, entry in load_cache()["entries"].items():
            cache["entries"].setdefault(key, entry)
        _write_json_atomically(cache_path(), cache)


def discard_legacy_cache() -> bool:
    """Remove a pre-packaging `.guarddog-cache.json` from the working directory.

    Its entries are not carried over: they predate structured results, so
    nothing in them says whether a scan completed. Re-scanning is the only
    honest way to find out.
    """
    legacy = Path.cwd() / LEGACY_CACHE_NAME
    if not legacy.is_file():
        return False
    legacy.unlink(missing_ok=True)
    return True


# --- raw reports ---------------------------------------------------------

#: Provenance added to a stored report, under one namespaced key so it cannot
#: collide with a field GuardDog adds later. GuardDog's own report names the
#: package it resolved; this records what was *asked for*, which is what the
#: cache key is built from and what a waiver is written against.
REPORT_PROVENANCE_KEY = "_guarddog_cached"


def save_report(name: str, version: str, guarddog_version: str, report: dict,
                scanned_at: str, scanned_version: str | None = None) -> Path:
    """Keep GuardDog's full report beside the cache entry that summarises it.

    Written for completed scans only, so that the presence of a report and
    the presence of a cache entry mean the same thing. No lock is taken:
    unlike the cache this is one file per key, not a shared document, and two
    projects scanning the same package version write the same content.

    Best-effort. The report is evidence for a human, never an input to the
    verdict, so failing to store it must not fail a scan that succeeded.
    """
    path = report_path(name, version, guarddog_version)
    stored = dict(report)
    stored[REPORT_PROVENANCE_KEY] = {
        "package": name,
        "version": version,
        "guarddog_version": guarddog_version,
        "cache_key": entry_key(name, version, guarddog_version),
        # The same string the cache entry carries, not a second reading of
        # the clock, so the report and the entry cannot disagree about when
        # the scan they both describe happened.
        "scanned_at": scanned_at,
    }
    if scanned_version is not None:
        # Filed under the version the lock asked for, but the bytes GuardDog
        # judged were fetched under another string. Say so here, or the
        # report silently claims to describe a release PyPI does not have.
        stored[REPORT_PROVENANCE_KEY]["scanned_version"] = scanned_version
    try:
        _write_json_atomically(path, stored)
    except OSError as exc:
        print(f"  · could not store the raw report ({exc})", file=sys.stderr, flush=True)
    return path


# --- accepted findings ---------------------------------------------------

def load_accepted() -> dict:
    path = accepted_path()
    if not path.exists():
        return {"schema": ACCEPTED_SCHEMA, "accepted": {}}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"schema": ACCEPTED_SCHEMA, "accepted": {}}
    if not isinstance(data, dict) or not isinstance(data.get("accepted"), dict):
        return {"schema": ACCEPTED_SCHEMA, "accepted": {}}
    return data


def waived_rules(accepted: dict, name: str, version: str) -> set[str]:
    """Rules a human has accepted for exactly this package version."""
    record = accepted["accepted"].get(waiver_key(name, version))
    if not isinstance(record, dict):
        return set()
    rules = record.get("rules")
    return set(rules) if isinstance(rules, list) else set()


# --- scanning and verdicts ----------------------------------------------

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


def _unreadable(detail: str) -> dict:
    """A report we cannot interpret is a failed scan, never an empty one."""
    return {
        "issues": 0,
        "errors": {"guarddog-cached": f"unrecognised report shape — {detail}; "
                                      f"refusing to read this as a completed scan"},
        "risks": [],
    }


def _pypi_release_key(version: str) -> str | None:
    """PyPI's own spelling of `version`, when it differs from the one given.

    `uv` reads a version out of the wheel *filename*; PyPI keys its release
    index on the PEP 440 canonical form. Those usually agree and
    occasionally do not. NVIDIA ships
    `cuda_toolkit-13.0.3.0-py2.py3-none-any.whl` under a release PyPI calls
    `13.0.3`, so a lock line saying `13.0.3.0` asks GuardDog for a version
    that, to the JSON API, does not exist — and an unscannable package is an
    INCOMPLETE that never passes. Four of that project's 39 releases are
    spelled this way, so it is a recurring shape and not one bad upload.

    Returns None when the canonical form is the string we already have,
    which is the overwhelmingly common case, and also when `version` is not
    a valid PEP 440 version at all — `canonicalize_version` hands those back
    unchanged, so they produce no second attempt.
    """
    canonical = canonicalize_version(version)
    return canonical if canonical != version else None


def _scan_once(name: str, version: str,
               timeout: float | None = SCAN_TIMEOUT_SECONDS) -> tuple[dict, dict | None]:
    """Run one `guarddog pypi scan` and reduce it to (entry, raw report).

    The raw report is None when the output could not be read as one, which
    lets the caller tell "nothing worth storing" from "a report that has
    errors in it". Storing it is deliberately not done here: which version
    string a report is filed under is decided one level up.

    `timeout` of None waits forever, which is what this call used to do
    unconditionally; see `SCAN_TIMEOUT_SECONDS` for why that is a bug and
    not a default.
    """
    try:
        proc = subprocess.run(
            ["guarddog", "pypi", "scan", name, "--version", version, "--output-format=json"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # `run` kills the child and reaps it before this is raised. GuardDog
        # clones in-process through pygit2 and spawns nothing of its own, so
        # no grandchild is left holding the captured pipes open.
        return {
            "issues": 0,
            "errors": {"guarddog-cached": f"no result after {timeout:g}s; scan killed"},
            "risks": [],
        }, None

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {
            "issues": 0,
            "errors": {"guarddog-cached": f"exit {proc.returncode}, unparseable output: "
                                          f"{detail[-1] if detail else '(no output)'}"},
            "risks": [],
        }, None

    if not isinstance(report, dict):
        return {"issues": 0, "errors": {"guarddog-cached": "output was not an object"},
                "risks": []}, None

    if not isinstance(report.get(ERRORS_KEY), dict):
        return _unreadable(f"{ERRORS_KEY!r} is missing or not an object"), None

    errors = dict(report[ERRORS_KEY])

    # A failing scan reports no `results` at all — the keys are just
    # ('package', 'issues', 'errors'). That is coherent, not corrupt, and the
    # `errors` map already says what went wrong; reporting a shape problem
    # here would replace GuardDog's real message with our own and leave the
    # user with no idea why the scan failed. Absent `results` is only
    # suspicious when nothing explains it.
    results = report.get(RESULTS_KEY)
    if not isinstance(results, dict):
        if not errors:
            return _unreadable(
                f"{RESULTS_KEY!r} is missing or not an object on a scan that "
                f"reported no errors"
            ), None
        results = {}

    # The verdict rests on `risks`, so its absence from a scan that claims to
    # have completed is unreadable rather than reassuring — the gate must not
    # be able to pass a package by failing to find the field it judges on.
    risks = report.get(RISKS_KEY)
    if not isinstance(risks, list):
        if not errors:
            return _unreadable(
                f"{RISKS_KEY!r} is missing or not a list on a scan that "
                f"reported no errors"
            ), None
        risks = []

    if proc.returncode != 0:
        errors.setdefault("guarddog-cached", f"guarddog exited {proc.returncode}")

    scanned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "issues": report.get("issues", 0),
        "scanned_at": scanned_at,
        "errors": errors,
        # `path` is deliberately dropped: it names a temp directory that
        # stopped existing the moment GuardDog returned.
        #
        # `results` is read for the shape check above and then discarded. It
        # is the raw per-rule match list, roughly twice the size of `risks`,
        # and nothing reads it now that the verdict comes from `risks` — a
        # field that looks like it feeds the gate and does not is the hazard
        # this rework exists to remove. Re-scanning regenerates it; the cache
        # is a cache, not an archive.
        "risks": [
            {field: risk[field] for field in RISK_FIELDS if field in risk}
            for risk in risks if isinstance(risk, dict)
        ],
        "risk_score": _kept_score(report.get("risk_score")),
    }, report


def scan_package(name: str, version: str, guarddog_version: str | None = None,
                 timeout: float | None = SCAN_TIMEOUT_SECONDS) -> dict:
    """Scan one package and return the facts, never a judgement.

    A crash, a timeout or unparseable output all become an `errors` entry
    rather than an empty-and-therefore-clean result: the caller must not be
    able to mistake "the tool broke" for "the package is fine".

    Given `guarddog_version`, a completed scan also has its full report
    stored beside the cache — the trimming in `_scan_once` is lossy by
    design, and the part it drops is the part a human needs to review a
    finding. The argument is optional so that a caller exercising the
    trimming need not invent a version; the driver always passes one.
    """
    entry, report = _scan_once(name, version, timeout)

    # A failed scan can mean only that we asked for a version string PyPI
    # does not use as a release key, so a second attempt against the
    # canonical form is worth one round trip.
    #
    # This cannot smuggle in a different release. PyPI refuses two releases
    # whose versions compare equal under PEP 440, so the canonical form
    # resolves either to this same release or to nothing at all: a genuine
    # `2.114.0` that failed for some other reason retries as `2.114`, finds
    # no such release, and keeps the error it already had. The retry is
    # accepted only when it completes, so "not checked" still never becomes
    # a pass.
    scanned_version = None
    if entry["errors"]:
        alternative = _pypi_release_key(version)
        if alternative is not None:
            retried, retried_report = _scan_once(name, alternative, timeout)
            if not retried["errors"]:
                entry, report = retried, retried_report
                scanned_version = alternative
                print(f"  · PyPI has this release as {alternative}; scanned that",
                      flush=True)

    if scanned_version is not None:
        # The lock's spelling stays the identity — cache key, report filename
        # and waiver key all keep `version`, so a reviewer waives the string
        # they can actually see in `uv.lock`. This field is the only record
        # in the entry that the two differ.
        entry["scanned_version"] = scanned_version

    # Only a completed scan is kept, matching the rule for the cache: an
    # entry and its report appear and disappear together, so a stale report
    # can never sit beside a package that is currently failing to scan.
    if guarddog_version is not None and not entry["errors"] and report is not None:
        save_report(name, version, guarddog_version, report, entry["scanned_at"],
                    scanned_version)

    return entry


def _kept_score(score: object) -> dict:
    """GuardDog's own headline score, for the report only — never the verdict.

    tqdm scores 7.2/10 `high_risk` for an api.telegram.org URL in
    `contrib/telegram.py`, and pyyaml 8.8. Gating on the label would block
    two of six ordinary packages, so it is shown to the human and ignored by
    the machine.
    """
    if not isinstance(score, dict):
        return {}
    return {key: score[key] for key in ("score", "label", "findings_count") if key in score}


def severity_blocks(severity: object) -> bool:
    """Whether a risk of this severity fails the gate. Unknown severities do.

    Defaulting this way round is the whole lesson of the previous gate. That
    one asked "is this rule name in my blocking list?", so a vocabulary it no
    longer recognised answered "no" to everything and blocked nothing. Here a
    severity this wrapper has never heard of is treated as blocking: if
    GuardDog's vocabulary moves under us the gate becomes noisy, which is
    survivable, rather than silent, which is not.
    """
    if severity not in RISK_SEVERITIES:
        return True
    return RISK_SEVERITIES.index(severity) >= RISK_SEVERITIES.index(BLOCKING_SEVERITY)


def risk_label(risk: dict) -> str:
    """How a risk is named in reports and in `accepted.json`.

    The threat rule is preferred over the risk name because it is the
    narrower of the two: `threat-network-exfiltration` waives one detection,
    where `risk.network.outbound` would waive every rule that rolls up into
    it.
    """
    return str(risk.get("threat_rule") or risk.get("name") or "unnamed-risk")


def verdict_for(entry: dict, waived: set[str]) -> tuple[str, list[str]]:
    """Decide an entry's verdict. Pure, and recomputed on every read.

    Returns the verdict and the identifiers that caused it, so the caller
    can say *why* rather than only *what*.
    """
    unrun = sorted(set(entry.get("errors") or {}) - waived)
    if unrun:
        return INCOMPLETE, unrun

    risks = [risk for risk in (entry.get("risks") or []) if isinstance(risk, dict)]
    # A waiver names either the threat rule or the rolled-up risk, so that a
    # reviewer can accept one detection or a whole category deliberately.
    live = [
        risk for risk in risks
        if not ({risk_label(risk), str(risk.get("name") or "")} & waived)
    ]

    blocking = sorted({
        risk_label(risk) for risk in live if severity_blocks(risk.get("severity"))
    })
    if blocking:
        return BLOCKED, blocking
    if live:
        return ADVISORY, sorted({risk_label(risk) for risk in live})
    return CLEAN, []


def render(label: str, entry: dict, verdict: str, rules: list[str]) -> str:
    """Human-readable detail for one package, from the stored facts."""
    lines = []
    if verdict == INCOMPLETE:
        lines.append(f"  ⚠ INCOMPLETE — {label} was not fully checked:")
        for rule in rules:
            lines.append(f"      {rule}: {entry['errors'][rule]}")
    elif verdict == BLOCKED:
        lines.append(f"  ✗ BLOCKED — {label} matched {', '.join(rules)}")

    score = entry.get("risk_score") or {}
    if score.get("label") and score.get("label") != "no_risks_detected":
        # Shown, not acted on — see `_kept_score`.
        lines.append(f"  · GuardDog score {score.get('score')}/10 ({score['label']})")

    for risk in sorted((entry.get("risks") or []),
                       key=lambda r: (risk_label(r), str(r.get("threat_location") or ""))):
        name = risk_label(risk)
        marker = "✗" if verdict == BLOCKED and name in rules else "·"
        severity = risk.get("severity", "?")
        tactics = ", ".join(risk.get("mitre_tactics") or [])
        lines.append(f"  {marker} {name} [{severity}]"
                     f"{f' · {tactics}' if tactics else ''}")
        detail = f"      {risk.get('threat_location') or risk.get('file_path') or ''} " \
                 f"{risk.get('threat_description') or ''}"
        if detail.strip():
            lines.append(detail.rstrip())
    return "\n".join(lines)


# --- driver --------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=Path(argv[0]).name,
        description="Scan every pinned package in a requirements file, reusing a "
                    "machine-wide cache of previous scans.",
    )
    parser.add_argument("requirements", help="a requirements file of name==version lines")
    parser.add_argument(
        "--time-budget", type=float, default=None, metavar="SECONDS",
        help="stop starting new scans once SECONDS have elapsed, and exit "
             f"{EXIT_UNFINISHED}. Cached packages are still evaluated, and every "
             "completed scan is saved, so re-running continues where this left off. "
             "The budget bounds when a scan starts, not when it ends.",
    )
    parser.add_argument(
        "--scan-timeout", type=float, default=SCAN_TIMEOUT_SECONDS, metavar="SECONDS",
        help=f"kill any single scan still running after SECONDS (default "
             f"{SCAN_TIMEOUT_SECONDS:g}). A killed scan is INCOMPLETE rather than a "
             f"pass, and is not cached, so the next run tries it again. 0 waits "
             f"forever, which risks a sweep that never returns.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    budget = args.time_budget
    # 0 is the explicit "wait forever" escape hatch, for the slow link where a
    # real scan would outlast any defensible default.
    scan_timeout = args.scan_timeout or None

    req_path = Path(args.requirements)
    print(f"Requirements file: {req_path}")
    if not req_path.exists():
        print(f"requirements file not found: {req_path}", file=sys.stderr)
        return 2

    guarddog_version = get_guarddog_version()
    cache = load_cache()
    accepted = load_accepted()
    entries = cache["entries"]

    if discard_legacy_cache():
        print(f"Removed stale ./{LEGACY_CACHE_NAME} (predates structured results; "
              f"its packages will be re-scanned)", flush=True)

    pairs = parse_requirements(req_path)
    print(f"GuardDog v{guarddog_version} — {len(pairs)} packages to evaluate", flush=True)

    tally = {CLEAN: 0, ADVISORY: 0, BLOCKED: 0, INCOMPLETE: 0}
    problems: list[tuple[str, str, list[str], Path]] = []
    skipped: list[str] = []
    cached = scanned = 0
    started = time.monotonic()

    try:
        for name, version in pairs:
            label = f"{name}=={version}"
            key = entry_key(name, version, guarddog_version)
            waived = waived_rules(accepted, name, version)

            entry = entries.get(key)
            if entry is None and budget is not None and time.monotonic() - started >= budget:
                # Keep going rather than break: cached packages cost nothing
                # and the report is more use when it says what *is* known.
                skipped.append(label)
                print(f"[skipped] {label} — time budget reached", flush=True)
                continue

            if entry is not None:
                cached += 1
                print(f"[cached] {label}", flush=True)
            else:
                scanned += 1
                print(f"[scanning] {label}", flush=True)
                entry = scan_package(name, version, guarddog_version, scan_timeout)
                # Only complete scans are cached. An incomplete one is a
                # non-result; freezing it would be the very bug this
                # wrapper exists to avoid.
                if not entry["errors"]:
                    entries[key] = entry
                    save_cache(cache)

            verdict, rules = verdict_for(entry, waived)
            tally[verdict] += 1
            detail = render(label, entry, verdict, rules)
            if detail:
                print(detail, flush=True)
            if verdict in (BLOCKED, INCOMPLETE):
                problems.append((label, verdict, rules,
                                 report_path(name, version, guarddog_version)))
    except KeyboardInterrupt:
        save_cache(cache)
        print(f"\n⚠ Interrupted. {cached} cached, {scanned} scanned this run; "
              f"completed scans are saved and will be reused.", flush=True)
        return 130

    save_cache(cache)

    print(f"\nSummary: {cached} cached, {scanned} scanned"
          f"{f', {len(skipped)} skipped' if skipped else ''}.")
    print(f"  clean {tally[CLEAN]} · advisory findings {tally[ADVISORY]} · "
          f"BLOCKED {tally[BLOCKED]} · INCOMPLETE {tally[INCOMPLETE]}")

    if problems:
        print("\nNot passed:")
        for label, verdict, rules, report in problems:
            print(f"  {verdict.upper():10s} {label}  ({', '.join(rules)})")
            # Entries cached before reports existed have none, and an
            # INCOMPLETE scan never had one; say nothing rather than name a
            # path that is not there.
            if report.exists():
                print(f"  {'':10s} matched code: {report}")
        print(f"\nAccept a reviewed finding by adding it to {accepted_path()}")
        print("Read the matched code before accepting one: the cache stores "
              "where a rule fired, not what it fired on.")
        # A definite failure outranks an unfinished run: something here is
        # known to be wrong, and more scanning will not change that.
        return 1

    if skipped:
        print(f"\n⏱ Time budget reached with {len(skipped)} package(s) never scanned:")
        for label in skipped[:10]:
            print(f"  {label}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")
        print("\nCompleted scans are cached; re-run to continue where this stopped.")
        return EXIT_UNFINISHED

    return 0


def cli() -> None:
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    cli()
