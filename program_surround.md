# autoresearch — Surround D3QN

Autonomous RL research loop: modify the training code, run a 20-minute experiment, check if the agent plays Surround better, keep or discard.

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current `feature/autoresearch` branch.
3. **Read the in-scope files**: Read these files for full context:
   - `program_surround.md` — these instructions.
   - `surround/d3qn/train_d3qn.py` — the D3QN trainer: network architecture, optimizer, replay buffer, training loop.
   - `surround/conf/constants.py` — all hyperparameters.
   - `run_experiment.py` — the fixed harness (train + benchmark). Do not modify.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good, then kick off the experiment loop.

## What you can and cannot do

**What you CAN modify:**
- `surround/d3qn/train_d3qn.py` — network architecture (`DuelingDQN`), optimizer, replay buffer, n-step returns, training loop logic, epsilon schedule, etc.
- `surround/conf/constants.py` — any D3QN hyperparameters: LR, batch size, memory capacity, n-step, update frequency, epsilon decay, gamma, tau, etc.

**What you CANNOT modify:**
- `run_experiment.py` — the fixed harness. It defines the time budget, runs the benchmark, and prints the metric.
- `TRAIN_TIME_BUDGET` in `train_d3qn.py` — this constant is fixed at 1200 seconds.
- `surround/utils/` — environment utilities, observation preprocessing, checkpointing.
- `surround/benchmark.py` — separate benchmark script (unused by the harness).
- Do not install new packages.

## The metric

**`point_win_pct` — higher is better.**

Each Surround game goes to 10 points (first to trap the opponent). The benchmark plays 30 episodes under the greedy policy and reports the percentage of total points scored by our agent:

```
point_win_pct = 100 * my_points / (my_points + opp_points)
```

50% = even. Baseline (random init, 20 min of training) will be near 50% or below. Anything above 50% is the agent learning to play.

## Running an experiment

```bash
python run_experiment.py > run.log 2>&1
```

Training runs for 20 minutes (wall clock), then a 30-episode benchmark runs automatically. The script prints a summary:

```
---
point_win_pct:      62.30
mean_return:        1.450
std_return:         8.120
benchmark_episodes: 30
best_win_rate:      0.540
training_seconds:   1201.4
total_seconds:      1340.2
peak_vram_mb:       2048.0
```

Extract the key metric:
```bash
grep "^point_win_pct:" run.log
```

If the grep returns nothing, the run crashed. Check the traceback:
```bash
tail -n 50 run.log
```

## Logging results

When an experiment finishes, log it to `results.tsv` (tab-separated, leave it untracked by git):

```
commit	point_win_pct	mean_return	peak_vram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. point_win_pct (e.g. 62.30) — use 0.00 for crashes
3. mean_return (e.g. 1.450) — use 0.000 for crashes
4. peak_vram_mb (e.g. 2048.0) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

Example:

```
commit	point_win_pct	mean_return	peak_vram_mb	status	description
a1b2c3d	51.20	0.120	2048.0	keep	baseline
b2c3d4e	55.80	0.880	2048.0	keep	reduce learning_starts to 1000
c3d4e5f	49.10	-0.200	2048.0	discard	double hidden size (no improvement)
d4e5f6g	0.00	0.000	0.0	crash	add LSTM layer (shape mismatch)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch and last commit.
2. Make one experimental change to `train_d3qn.py` and/or `constants.py`.
3. `git commit` (commit both files if both changed).
4. Run the experiment: `python run_experiment.py > run.log 2>&1`
5. Read results: `grep "^point_win_pct:\|^peak_vram_mb:" run.log`
6. If the grep is empty, the run crashed. Read `tail -n 50 run.log` for the traceback.
7. Log the result to `results.tsv`.
8. If `point_win_pct` improved (higher), **keep** the commit and build on it.
9. If `point_win_pct` is equal or worse, `git reset --hard HEAD~1` to discard.

**Simplicity criterion**: A small improvement from cleaner/simpler code is worth keeping. A tiny improvement from hacky complexity is not.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the user whether to continue. Do NOT ask "should I keep going?" The user may be away and expects you to run indefinitely until manually stopped. If you run out of obvious ideas, try more radical changes: different architectures (add BatchNorm, attention, larger/smaller CNN), reward shaping, prioritized replay, different optimizers, input representations, etc.

**Crashes**: If a run crashes from something dumb (typo, missing import), fix and re-run. If the idea is fundamentally broken, log `crash` and move on.

**Timeout**: Each experiment takes ~20 min training + ~2 min benchmark. If a run exceeds 30 minutes, kill it (`Ctrl+C`) and treat as crash.

## Ideas to explore

Architecture:
- Wider/deeper `DuelingDQN` (more filters, more FC hidden units)
- BatchNorm after conv layers
- Larger/smaller preprocessing resolution (`DQN_PREPROCESS_HEIGHT/WIDTH`)
- Different conv kernel sizes or strides

Optimization:
- Learning rate (`D3QN_LR`), LR schedule (cosine, step)
- Batch size (`D3QN_BATCH_SIZE`), memory capacity (`D3QN_MEMORY_CAPACITY`)
- Update frequency (`D3QN_UPDATE_EVERY`), target update tau (`TAU`)
- Prioritized experience replay instead of uniform

Exploration:
- Epsilon schedule shape or decay fraction (`D3QN_EPS_DECAY_FRACTION`)
- N-step returns (`D3QN_N_STEP`)

Training dynamics:
- Reduce `D3QN_LEARNING_STARTS` so learning kicks in earlier
- Gradient clipping value (currently 1.0 in `_optimize_model`)
- Different loss function (MSE vs Huber)
