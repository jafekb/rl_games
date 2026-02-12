#!/usr/bin/env bash
# Pre-merge test runner. Also runs in CI on every PR.
set -e
cd "$(dirname "$0")/.."
uv sync --extra dev
uv run pytest tests/ -v
