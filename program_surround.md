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

## Round 2 context (autoresearch/mar22) — read this if starting fresh on this branch

This is **round 2** of autoresearch. The branch `autoresearch/mar22` already exists — do not create a new one.

**Key changes from round 1:**
- The benchmark now runs **32 episodes across all 4 difficulties** (8 each at difficulty 0, 1, 2, 3) instead of 30 episodes at difficulty=1 only. This tests generalization, not just peak performance on one difficulty.
- The seed checkpoint is the **round 1 best** (`runs/surround/autoresearch/best/`, 90.91% on the old difficulty=1 benchmark). The new multi-difficulty baseline will be established on the first run.
- Round 1 results are archived at `runs/surround/autoresearch/results.tsv` — **read this before starting** so you know what has already been tried and can avoid re-running dead ends.

**Round 1 summary (46 experiments):**
- Only 3 experiments improved over baseline. The vast majority tied or hurt.
- The two wins that stuck: **N_STEP 10→3** (more stable Q targets) and **EPS=0** (pure greedy fine-tuning, the big win: +7.6pp).
- After EPS=0, the agent hit a hard ceiling at 90.91% on difficulty=1. Every subsequent experiment tied or degraded — LR changes, N_STEP variants, batch size, TAU, cosine schedule, BatchNorm (crashed), optimizer swaps — nothing moved the needle.
- The ceiling appears **intrinsic to the starting checkpoint** on the fixed benchmark seeds at difficulty=1.

**Implication for round 2:** The old benchmark was too narrow (30 fixed seeds, 1 difficulty). The new benchmark should reveal whether the policy genuinely generalizes or was overfit to difficulty=1. Focus on ideas that improve generalization across difficulties, not just memorizing fixed seeds.

## What you can and cannot do

**What you CAN modify:**
- `surround/d3qn/train_d3qn.py` — network architecture (`DuelingDQN`), optimizer, replay buffer, n-step returns, training loop logic, epsilon schedule, etc.
- `surround/conf/constants.py` — any D3QN hyperparameters: LR, batch size, memory capacity, n-step, update frequency, epsilon decay, gamma, tau, etc.

**What you CANNOT modify:**
- `run_experiment.py` — the fixed harness. It manages the best/ checkpoint, defines the time budget, runs the benchmark, and prints the metric.
- `TRAIN_TIME_BUDGET` in `train_d3qn.py` — this constant is fixed at 1200 seconds.
- `D3QN_RESUME_FROM` and `D3QN_LOG_DIR` in `constants.py` — managed by the harness.
- `surround/utils/` — environment utilities, observation preprocessing, checkpointing.
- `surround/benchmark.py` — separate benchmark script (unused by the harness).
- Do not install new packages.

## Checkpoint strategy

Each experiment **resumes from the best saved checkpoint** (`runs/surround/autoresearch2/best/`, seeded from round 1 best on first run). `run_experiment.py` manages this directory — if the new run scores higher than the previous best, the checkpoint is automatically updated. You never touch this logic.

## The metric

**`point_win_pct` — higher is better.** This is the single value you are maximizing.

Each Surround game goes to 10 points (first to trap the opponent). The benchmark plays 32 episodes (8 at each of difficulties 0, 1, 2, 3) under the greedy policy and reports the percentage of total points scored by our agent:

```
point_win_pct = 100 * my_points / (my_points + opp_points)
```

The summary also reports **per-difficulty breakdowns** (`diff0_win_pct` through `diff3_win_pct`). These are diagnostic only — they do not affect checkpoint saving or keep/discard decisions, but they tell you where the policy is strong or weak. Use them to guide hypothesis generation: if diff=2 and diff=3 are low, prioritize ideas that improve generalization to harder difficulties.

The round 2 baseline will be established on the first run.

## Running an experiment

```bash
python run_experiment.py > run.log 2>&1
```

Training runs for 20 minutes (wall clock), then a 32-episode benchmark runs automatically. The script prints a summary:

```
---
point_win_pct:      62.30
prev_best_win_pct:  58.10
diff0_win_pct:      81.25
diff1_win_pct:      68.75
diff2_win_pct:      50.00
diff3_win_pct:      43.75
mean_return:        1.450
std_return:         8.120
benchmark_episodes: 32
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

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the user whether to continue. Do NOT ask "should I keep going?" The user may be away and expects you to run indefinitely until manually stopped. If you run out of obvious ideas, try more radical changes: different architectures, reward shaping, different input representations, etc.

**Crashes**: If a run crashes from something dumb (typo, missing import), fix and re-run. If the idea is fundamentally broken, log `crash` and move on.

**Timeout**: Each experiment takes ~20 min training + ~2 min benchmark. If a run exceeds 30 minutes, kill it (`Ctrl+C`) and treat as crash.

## Ideas to explore

Since the round 1 ceiling was likely an overfitting artifact of the narrow benchmark, round 2 ideas should prioritize **generalization**:

Exploration / robustness:
- Reintroduce a small epsilon (e.g. EPS_START=0.01–0.02) during fine-tuning to expose the policy to more diverse states across difficulties
- Train on a mix of difficulties (modify the env creation in `train_d3qn.py` to sample difficulty randomly each episode)
- Vary seeds during training (currently fixed)

Architecture:
- Wider/deeper `DuelingDQN` (more filters, more FC hidden units)
- Larger preprocessing resolution (`DQN_PREPROCESS_HEIGHT/WIDTH`) — higher-res input may help on harder difficulties
- Different conv kernel sizes or strides

Optimization:
- Learning rate (`D3QN_LR`), LR schedule (cosine, step decay)
- Batch size (`D3QN_BATCH_SIZE`), memory capacity (`D3QN_MEMORY_CAPACITY`)
- Update frequency (`D3QN_UPDATE_EVERY`), target update tau (`TAU`)

Training dynamics:
- Gradient clipping value (currently 1.0 in `_optimize_model`)
- N-step returns (N=3 is established best for difficulty=1, may differ across difficulties)
