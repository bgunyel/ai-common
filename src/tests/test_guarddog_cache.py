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
plan = {{}}
if os.environ.get("FAKE_PLAN"):
    plan = json.load(open(os.environ["FAKE_PLAN"]))
spec = plan.get(name, {{}})
time.sleep({delay})
if spec.get("garbage"):
    print("this is not json"); sys.exit(spec.get("exit", 0))
print(json.dumps({{
    "package": name,
    "issues": spec.get("issues", 0),
    "errors": spec.get("errors", {{}}),
    "results": spec.get("results", {{}}),
    "path": "/tmp/ephemeral",
}}))
sys.exit(spec.get("exit", 0))
'''

CODE_EXEC = {"issues": 1, "results": {"code-execution": [
    {"location": "setup.py:3", "code": "os.system('curl evil')", "message": "OS command in setup.py"}]}}
SHADY = {"issues": 2, "results": {"shady-links": [
    {"location": "a.py:1", "message": "suspicious URL"},
    {"location": "b.py:2", "message": "suspicious URL"}]}}
BROKEN_RULE = {"errors": {"potentially_compromised_email_domain": "Invalid version: '2013-02-16'"}}


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
           "HOME": str(project_dir)}
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
                          plan={"evil": CODE_EXEC})

    assert rc == 1
    assert "BLOCKED" in out and "code-execution" in out


def test_an_advisory_finding_is_reported_but_does_not_fail(tmp_path, fake_guarddog, cache_home):
    """26 of 91 real packages trip a heuristic; blocking on those is unusable."""
    rc, out = run_project(tmp_path / "p", ["noisy==1.0"], fake_guarddog(), cache_home,
                          plan={"noisy": SHADY})

    assert rc == 0
    assert "shady-links (2)" in out
    assert "✗ BLOCKED" not in out          # the detail line, not the "BLOCKED 0" tally
    assert "BLOCKED 0" in out


def test_one_bad_package_fails_a_run_of_many(tmp_path, fake_guarddog, cache_home):
    rc, out = run_project(tmp_path / "p", ["ok==1.0", "evil==1.0", "fine==1.0"],
                          fake_guarddog(), cache_home, plan={"evil": CODE_EXEC})

    assert rc == 1
    assert "clean 2" in out and "BLOCKED 1" in out


# --- verdicts, as a pure function ----------------------------------------

def test_incomplete_outranks_findings():
    """A partial scan's findings say nothing about what the unrun rules missed."""
    entry = {"errors": {"some-rule": "boom"}, "findings": {"code-execution": [{}]}}

    assert gd.verdict_for(entry, set())[0] == gd.INCOMPLETE


def test_the_blocking_set_is_exactly_these_rules():
    """Literal names on purpose.

    `test_every_blocking_rule_blocks` below parametrizes over the constant
    it is testing, so it cannot notice the set being emptied or shrunk —
    it would simply run fewer cases. This test can.
    """
    assert gd.BLOCKING_RULES == frozenset({
        "code-execution",
        "exec-base64",
        "download-executable",
        "silent-process-execution",
        "exfiltrate-sensitive-data",
        "cmd-overwrite",
        "steganography",
    })


@pytest.mark.parametrize("rule", sorted(gd.BLOCKING_RULES))
def test_every_blocking_rule_blocks(rule):
    assert gd.verdict_for({"findings": {rule: [{}]}}, set())[0] == gd.BLOCKED


def test_a_rule_outside_the_blocking_set_is_advisory():
    assert gd.verdict_for({"findings": {"shady-links": [{}]}}, set())[0] == gd.ADVISORY


def test_an_empty_entry_is_clean():
    assert gd.verdict_for({"errors": {}, "findings": {}}, set()) == (gd.CLEAN, [])


def test_a_waiver_clears_a_blocking_rule():
    entry = {"findings": {"code-execution": [{}]}}

    assert gd.verdict_for(entry, {"code-execution"})[0] == gd.ADVISORY


def test_a_waiver_clears_an_unrun_rule():
    entry = {"errors": {"potentially_compromised_email_domain": "Invalid version"}}

    assert gd.verdict_for(entry, {"potentially_compromised_email_domain"})[0] == gd.CLEAN


def test_a_waiver_for_one_rule_does_not_clear_another():
    entry = {"findings": {"code-execution": [{}], "exec-base64": [{}]}}

    verdict, rules = gd.verdict_for(entry, {"code-execution"})
    assert verdict == gd.BLOCKED and rules == ["exec-base64"]


# --- waivers -------------------------------------------------------------

def test_a_waiver_lets_a_blocked_package_through(tmp_path, fake_guarddog, cache_home):
    write_accepted(cache_home, {"evil==1.0": {
        "rules": ["code-execution"], "reason": "vendored build hook, reviewed",
        "by": "bgunyel", "at": "2026-08-10"}})

    rc, out = run_project(tmp_path / "p", ["evil==1.0"], fake_guarddog(), cache_home,
                          plan={"evil": CODE_EXEC})

    assert rc == 0
    assert "✗ BLOCKED" not in out
    assert "BLOCKED 0" in out


def test_a_waiver_does_not_carry_to_a_new_package_version(tmp_path, fake_guarddog, cache_home):
    """A new version is new code; the review does not transfer."""
    write_accepted(cache_home, {"evil==1.0": {"rules": ["code-execution"]}})

    rc, out = run_project(tmp_path / "p", ["evil==2.0"], fake_guarddog(), cache_home,
                          plan={"evil": CODE_EXEC})

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
    run_project(tmp_path / "p", ["noisy==1.0"], bin_dir, cache_home, plan={"noisy": SHADY})

    _, out = run_project(tmp_path / "p", ["noisy==1.0"], bin_dir, cache_home, plan={"noisy": SHADY})
    assert "[cached] noisy==1.0" in out
    assert "shady-links (2)" in out, "cached findings must still be reported"


def test_the_verdict_is_recomputed_from_the_cache_not_stored(tmp_path, fake_guarddog, cache_home):
    """Accepting a finding must re-decide cached packages without re-scanning."""
    bin_dir = fake_guarddog()
    rc, _ = run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": CODE_EXEC})
    assert rc == 1

    write_accepted(cache_home, {"evil==1.0": {"rules": ["code-execution"]}})
    rc, out = run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": CODE_EXEC})

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


def test_rules_that_matched_nothing_are_not_recorded(tmp_path, fake_guarddog,
                                                     cache_home, monkeypatch):
    """GuardDog reports unmatched rules as null, not as an empty list."""
    bin_dir = fake_guarddog()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"ok": {"results": {"shady-links": None, "unicode": []}}}))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_PLAN", str(plan))

    assert gd.scan_package("ok", "1.0")["findings"] == {}


def test_the_ephemeral_scan_path_is_not_cached(tmp_path, fake_guarddog, cache_home, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_guarddog()}:/usr/bin:/bin")

    assert "path" not in gd.scan_package("ok", "1.0")


# --- the report shape we depend on ---------------------------------------

@pytest.mark.parametrize("report", [
    '{"package": "x", "issues": 0, "results": {}}',                 # errors key gone
    '{"package": "x", "issues": 0, "errors": {}}',                  # results key gone
    '{"package": "x", "issues": 0, "errors": [], "results": {}}',   # errors not an object
    '{"package": "x", "issues": 0, "errors": {}, "results": null}',  # results not an object
    '{"package": "x", "problems": {}, "matches": {}}',              # renamed wholesale
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
    run_project(tmp_path / "p", ["evil==1.0"], bin_dir, cache_home, plan={"evil": CODE_EXEC})

    rc, out = run_project(tmp_path / "p", ["evil==1.0", "new==1.0"], bin_dir, cache_home,
                          plan={"evil": CODE_EXEC}, extra_args=["--time-budget", "0"])

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