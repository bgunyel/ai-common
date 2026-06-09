# Makefile for ai-common

.PHONY: test audit scan verify upgrade upgrade-safe

TEST_DIRECTORY ?= src/tests/
GUARDDOG_CACHE := /tmp/flat-requirements-cache.txt
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

# Tier 2: GuardDog static analysis on every locked dep. Wrapped by a
# per-package cache (.guarddog-cache.json) keyed on (name, version,
# guarddog_version) so subsequent runs skip unchanged packages.
scan:
	@command -v guarddog >/dev/null 2>&1 || { \
		echo "guarddog not installed. Install via 'uv tool install guarddog', 'pip install guarddog', or 'docker pull ghcr.io/datadog/guarddog'"; \
		exit 1; \
	}
	@uv export --no-hashes -o $(GUARDDOG_CACHE) >/dev/null
	@python3 scripts/guarddog_cached.py $(GUARDDOG_CACHE); \
		status=$$?; \
		rm -f $(GUARDDOG_CACHE); \
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
	@cp uv.lock uv.lock.preupgrade
	@echo "→ Resolving candidate upgrade..."
	@uv lock --upgrade || { mv -f uv.lock.preupgrade uv.lock; exit 1; }
	@echo "→ Tier 1 — OSV/GHSA known-advisory scan..."
	@osv-scanner --lockfile=uv.lock || { \
		mv -f uv.lock.preupgrade uv.lock; \
		echo ""; \
		echo "✗ Candidate fails OSV/GHSA scan. uv.lock restored."; \
		echo "  Skip an affected package: uv lock --upgrade-package <other> ..."; \
		echo "  Or pin a safe version in pyproject.toml and re-run: make upgrade-safe"; \
		exit 1; \
	}
	@echo "→ Tier 2 — GuardDog static analysis on candidate deps (cached)..."
	@uv export --no-hashes -o $(GUARDDOG_CACHE) >/dev/null || { mv -f uv.lock.preupgrade uv.lock; exit 1; }
	@python3 scripts/guarddog_cached.py $(GUARDDOG_CACHE); \
		status=$$?; \
		rm -f $(GUARDDOG_CACHE); \
		if [ $$status -ne 0 ]; then \
			mv -f uv.lock.preupgrade uv.lock; \
			echo ""; \
			echo "✗ Candidate fails GuardDog static analysis. uv.lock restored."; \
			exit 1; \
		fi
	@rm -f uv.lock.preupgrade
	@uv sync
	@echo "✓ Clean across both tiers. uv.lock updated and environment synced."

# Blind upgrade with only the 7-day quarantine — bypasses both gates.
# Kept for parity; prefer `upgrade-safe`.
upgrade:
	uv sync --upgrade --exclude-newer $$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')
