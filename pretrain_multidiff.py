"""Pre-train D3QN from scratch on all 4 difficulties simultaneously.

Goal: produce a multi-difficulty-aware seed checkpoint to replace the round-1
(diff=1-only) checkpoint as the autoresearch round-2 starting point.

Training from scratch on uniform difficulty sampling forces the network to
build representations that generalize across all opponent skill levels, avoiding
the catastrophic forgetting that occurs when fine-tuning a diff=1-specialized policy.

Usage:
    python pretrain_multidiff.py > pretrain.log 2>&1 &

Output:
    runs/surround/pretrain/multidiff_scratch/checkpoints/policy_best.pt
"""

import shutil
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch constants BEFORE importing the trainer
# ---------------------------------------------------------------------------
from surround.conf import constants
from surround.utils.checkpoint import CheckpointPaths

PRETRAIN_LOG_DIR = Path("runs/surround/pretrain/multidiff_scratch")
PRETRAIN_BUDGET_HOURS = 16

constants.D3QN_LOG_DIR = PRETRAIN_LOG_DIR
constants.D3QN_CKPT = CheckpointPaths(PRETRAIN_LOG_DIR)
constants.EPS_START = 1.0  # full exploration from random weights
constants.EPS_END = 0.05  # small floor to retain some exploration
constants.D3QN_LR = 3e-4  # standard LR for scratch training
constants.D3QN_EPS_DECAY_FRACTION = 0.10  # decay over 10% of 50K = 5K episodes
constants.D3QN_FORCE_RESUME_FROM = None  # train from scratch, ignore harness injection
constants.D3QN_CURRICULUM = False
constants.D3QN_MULTIDIFF_DIFFICULTIES = [0, 1, 2, 3]  # uniform sampling across all diffs

import surround.d3qn.train_d3qn as train_mod  # noqa: E402

train_mod.TRAIN_TIME_BUDGET = int(PRETRAIN_BUDGET_HOURS * 3600)

# ---------------------------------------------------------------------------
# Clear any previous attempt
# ---------------------------------------------------------------------------

if PRETRAIN_LOG_DIR.exists():
    shutil.rmtree(PRETRAIN_LOG_DIR)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

t_start = time.time()
trainer = train_mod.D3QNTrainer()
trainer.run()
elapsed_hours = (time.time() - t_start) / 3600

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nPre-training complete: {elapsed_hours:.2f}h")
print(f"Best win rate during training: {trainer.best_win_rate:.3f}")
print(f"Best checkpoint: {constants.D3QN_CKPT.best}")
print(f"Latest checkpoint: {constants.D3QN_CKPT.latest}")
print("\nTo use as autoresearch seed, copy to runs/surround/pretrain/best/checkpoints/")
