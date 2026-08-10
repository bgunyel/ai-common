# Makefile for ai-common

.PHONY: test audit scan verify upgrade upgrade-safe

# The scanning recipes below depend on `trap ... EXIT` firing on signals, so
# pin the shell rather than inherit whatever /bin/sh happens to be.
SHELL := /bin/bash

TEST_DIRECTORY ?= src/tests/
GUARDDOG_CACHE := /tmp/flat-requirements-cache.txt

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
scan:
	@command -v guarddog >/dev/null 2>&1 || { \
		echo "guarddog not installed. Install via 'uv tool install guarddog', 'pip install guarddog', or 'docker pull ghcr.io/datadog/guarddog'"; \
		exit 1; \
	}
	@trap 'rm -f $(GUARDDOG_CACHE)' EXIT; \
	trap 'exit 130' INT; \
	trap 'exit 143' TERM; \
	uv export --no-hashes --all-groups -o $(GUARDDOG_CACHE) >/dev/null; \
	uv run guarddog-cached $(GUARDDOG_BUDGET_FLAG) $(GUARDDOG_CACHE)

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
	trap 'rm -f $(GUARDDOG_CACHE); \
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
	uv export --no-hashes --all-groups -o $(GUARDDOG_CACHE) >/dev/null || exit 1; \
	uv run guarddog-cached $(GUARDDOG_BUDGET_FLAG) $(GUARDDOG_CACHE); \
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
