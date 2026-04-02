# Makefile for ai-common

.PHONY: test upgrade

TEST_DIRECTORY ?= src/tests/
test:
	@echo "🧪 Running public API test..."
	uv run --group test pytest $(TEST_DIRECTORY)

upgrade:
	uv sync --upgrade --exclude-newer $$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')
