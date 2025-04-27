# Makefile for ai-common

TEST_DIRECTORY ?= src/tests/
test:
	@echo "🧪 Running public API test..."
	uv run --group test pytest $(TEST_DIRECTORY)
