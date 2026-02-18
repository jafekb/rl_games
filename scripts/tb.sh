#!/usr/bin/env bash
set -euo pipefail
export LOG_DIR="runs/surround"

uv run --isolated --with "setuptools<70" --with tensorboard tensorboard --logdir $LOG_DIR --load_fast=true
