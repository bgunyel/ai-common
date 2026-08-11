# Makefile for ai-common

.PHONY: test audit scan verify upgrade upgrade-safe

# The scanning recipes below depend on `trap ... EXIT` firing on signals, so
# pin the shell rather than inherit whatever /bin/sh happens to be.
SHELL := /bin/bash

TEST_DIRECTORY ?= src/tests/

# The lock, flattened to `name==version` lines for GuardDog to read. This is
# NOT the GuardDog cache — that is machine-wide and lives wherever
# `ai_common.security.guarddog_cached.cache_path()` points, normally
# ~/.cache/guarddog-cached/. This file is an input, written before each sweep
# and removed by the EXIT trap after it.
#
# Repo-local on purpose. It used to be a fixed path in /tmp, shared by every
# project on the machine: the wrapper reads the requirements file once at
# startup, so a second repo exporting between this repo's export and its read
# would make this sweep scan the *other* repo's dependencies and report a pass
# on them — silently, since every line in it is a legitimate package.
#
# The remaining hole is deliberate. Two sweeps at once *in the same repo* still
# share this path, and neither the rename nor the move fixes that. Don't do
# that; they would be fighting over uv.lock as well.
FLAT_REQUIREMENTS_FILE := tmp/flat-requirements.txt

# Optional wall-clock budget in seconds for the GuardDog sweep, e.g.
#   make upgrade-safe GUARDDOG_BUDGET=600
# Scanning stops starting new packages once the budget is spent and exits 75.
# Completed scans are cached, so repeated budgeted runs converge on a full
# sweep; only a run that finishes inside its budget can adopt an upgrade.
GUARDDOG_BUDGET ?=
GUARDDOG_BUDGET_FLAG := $(if $(GUARDDOG_BUDGET),--time-budget $(GUARDDOG_BUDGET),)
test:
	@echo "🧪 Running public API test..."
	uv run --group test pytest $(TEST_DIRECTORY)

# Tier 1: scan the committed uv.lock against OSV/GHSA. Cheap, read-only.
audit:
	@command -v osv-scanner >/dev/null 2>&1 || { \
		echo "osv-scanner not installed. Install via 'brew install osv-scanner', 'go install github.com/google/osv-scanner/cmd/osv-scanner@latest', or https://github.com/google/osv-scanner/releases"; \
		exit 1; \
	}
	osv-scanner --lockfile=uv.lock

# Tier 2: GuardDog static analysis on every locked dep. Wrapped by the
# `guarddog-cached` console script (shipped by ai-common itself), which
# caches per-package results in a shared user-level cache keyed on
# (name, version, guarddog_version) so subsequent runs skip unchanged
# packages.
#
# `uv export --frozen` is load-bearing: without it, an export from a lock that
# has drifted from pyproject.toml re-resolves and REWRITES uv.lock, so a
# recipe that reads as read-only would edit the lockfile and then scan a
# resolution nobody chose. It would also let `verify` audit the committed lock
# in tier 1 and scan a different one in tier 2. Verified 2026-08-11: adding a
# dependency to pyproject.toml and running the un-frozen export rewrote
# uv.lock and pulled the new package into the scanned set.
scan:
	@command -v guarddog >/dev/null 2>&1 || { \
		echo "guarddog not installed. Install via 'uv tool install guarddog', 'pip install guarddog', or 'docker pull ghcr.io/datadog/guarddog'"; \
		exit 1; \
	}
	@trap 'rm -f $(FLAT_REQUIREMENTS_FILE)' EXIT; \
	trap 'exit 130' INT; \
	trap 'exit 143' TERM; \
	mkdir -p $(dir $(FLAT_REQUIREMENTS_FILE)); \
	uv export --frozen --no-hashes --all-groups -o $(FLAT_REQUIREMENTS_FILE) >/dev/null; \
	uv run guarddog-cached $(GUARDDOG_BUDGET_FLAG) $(FLAT_REQUIREMENTS_FILE); \
	status=$$?; \
	if [ $$status -eq 75 ]; then \
		echo "UNFINISHED is not a pass. Note that make reports its own exit 2"; \
		echo "for every failure, so the 75 above does not survive this recipe;"; \
		echo "a caller that must tell 'unfinished' from 'blocked' should run"; \
		echo "the wrapper directly: uv run guarddog-cached --time-budget N <file>"; \
	fi; \
	exit $$status

# Combined tier-1 + tier-2 sweep against the committed lock. Use for
# release gates or periodic checks; too slow for every push.
verify: audit scan

# Resolve a candidate upgrade into uv.lock, run BOTH scanners on the
# candidate, and revert if either tier fires. Same scanners as `verify`,
# applied to the post-`uv lock --upgrade` state instead of the
# committed lock.
upgrade-safe:
	@command -v osv-scanner >/dev/null 2>&1 || { \
		echo "osv-scanner not installed (see 'make audit' for install hints)"; \
		exit 1; \
	}
	@command -v guarddog >/dev/null 2>&1 || { \
		echo "guarddog not installed (see 'make scan' for install hints)"; \
		exit 1; \
	}
	@cp uv.lock uv.lock.preupgrade; \
	trap 'rm -f $(FLAT_REQUIREMENTS_FILE); \
	      if [ -f uv.lock.preupgrade ]; then \
	          mv -f uv.lock.preupgrade uv.lock; \
	          echo ""; \
	          echo "↩ uv.lock restored to its pre-upgrade state."; \
	      fi' EXIT; \
	trap 'echo ""; echo "⚠ Interrupted — nothing adopted."; exit 130' INT; \
	trap 'exit 143' TERM; \
	echo "→ Resolving candidate upgrade..."; \
	uv lock --upgrade || exit 1; \
	echo "→ Tier 1 — OSV/GHSA known-advisory scan..."; \
	osv-scanner --lockfile=uv.lock || { \
		echo ""; \
		echo "✗ Candidate fails OSV/GHSA scan."; \
		echo "  Skip an affected package: uv lock --upgrade-package <other> ..."; \
		echo "  Or pin a safe version in pyproject.toml and re-run: make upgrade-safe"; \
		exit 1; \
	}; \
	echo "→ Tier 2 — GuardDog static analysis on candidate deps (cached)..."; \
	mkdir -p $(dir $(FLAT_REQUIREMENTS_FILE)); \
	uv export --frozen --no-hashes --all-groups -o $(FLAT_REQUIREMENTS_FILE) >/dev/null || exit 1; \
	uv run guarddog-cached $(GUARDDOG_BUDGET_FLAG) $(FLAT_REQUIREMENTS_FILE); \
	status=$$?; \
	if [ $$status -eq 130 ]; then exit 130; fi; \
	if [ $$status -eq 75 ]; then \
		echo ""; \
		echo "⏱ Time budget reached before the candidate was fully scanned."; \
		echo "  Completed scans are cached — re-run to continue; the upgrade is"; \
		echo "  adopted only once a run gets all the way through."; \
		exit 75; \
	fi; \
	if [ $$status -ne 0 ]; then \
		echo ""; \
		echo "✗ Candidate fails GuardDog static analysis."; \
		exit 1; \
	fi; \
	rm -f uv.lock.preupgrade; \
	uv sync --all-groups; \
	echo "✓ Clean across both tiers. uv.lock updated and environment synced."

# Blind upgrade with only the 7-day quarantine — bypasses both gates.
# Kept for parity; prefer `upgrade-safe`.
upgrade:
	uv sync --all-groups --upgrade --exclude-newer $$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')
