#!/usr/bin/env bash
set -euo pipefail

uv run --isolated --with "setuptools<70" --with tensorboard tensorboard --logdir runs
