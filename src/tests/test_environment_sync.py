"""The virtual environment must stay coupled to `uv.lock`.

Both supply-chain gate tiers read `uv.lock`: `osv-scanner` parses it directly,
and the GuardDog sweep scans whatever `uv export` flattens out of it. Neither
looks at `.venv`. A package that is installed but not locked is therefore
invisible to both tiers while still being importable by the application and by
these tests — it is scanned by nobody and runs anyway.

That is not hypothetical. Three unrelated drifts were found by hand on
2026-08-16: `olefile` and `python-oxmsg` left behind by an `upgrade-safe`
candidate the gate had rejected, the same again after an interrupted run, and
`httpx2`, `httpcore2` and `truststore` from an extras-installing command.
Nothing reported any of them. Those sources are now closed — the sweep runs
under `uv run --frozen --no-sync` — but this test exists because the next
source will be a different one.

`uv sync --check` is uv's own answer to the question, so the comparison is
deliberately not re-implemented here: it evaluates environment markers, knows
the project installs itself into its own environment, and reports a venv that
is missing packages as readily as one carrying extras. Measured 2026-08-16 —
exit 0 when synchronised, exit 1 in both drift directions, and it modifies
neither the environment nor the lock.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `--frozen` so the check can never rewrite the lock it is checking; without it
# `uv` may re-resolve, which would make this test a mutation of the artifact it
# is meant to be guarding. `--all-groups` because that is the environment the
# Makefile builds when it adopts an upgrade, so it is the project's own
# definition of a complete environment.
UV_CHECK_COMMAND = ["uv", "sync", "--check", "--frozen", "--all-groups"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_installed_packages_match_uv_lock():
    # Drop VIRTUAL_ENV so the check always targets this project's own `.venv`
    # rather than whatever environment the calling shell has activated. An
    # activated venv from another project silently redirects `uv pip list` in
    # exactly this way, which is how one of the 2026-08-16 measurements went
    # wrong before it was caught.
    environment = {
        name: value for name, value in os.environ.items() if name != "VIRTUAL_ENV"
    }

    result = subprocess.run(
        UV_CHECK_COMMAND,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )

    assert result.returncode == 0, (
        "The virtual environment has drifted from uv.lock. Packages that are "
        "installed but not locked are seen by neither gate tier.\n"
        "Repair with:  uv sync --all-groups\n\n"
        f"{result.stdout}{result.stderr}"
    )