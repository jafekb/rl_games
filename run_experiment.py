"""
Autoresearch experiment harness for Surround D3QN.

Resumes training from the best checkpoint (seeded from exp9 on first run),
trains for TRAIN_TIME_BUDGET seconds, benchmarks, then updates the best
checkpoint if the result improved.

Usage:
    python run_experiment.py > run.log 2>&1

Extract the key metric:
    grep "^point_win_pct:" run.log

Do NOT modify this file — it is the fixed evaluation harness.
"""

import json
import math
import shutil
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config (fixed — do not modify)
# ---------------------------------------------------------------------------

BENCHMARK_EPISODES = 32  # 8 per difficulty (0, 1, 2, 3)
BENCHMARK_EPISODES_PER_DIFFICULTY = BENCHMARK_EPISODES // 4
SEED_DIR = Path("runs/surround/autoresearch/best")  # seed from autoresearch round 1 best
BEST_DIR = Path("runs/surround/autoresearch2/best")
BEST_RESULT_PATH = BEST_DIR / "best_result.json"

# ---------------------------------------------------------------------------
# Bootstrap best/ dir from exp9 on first run
# ---------------------------------------------------------------------------


def _measure_seed_baseline(ckpt_dir: Path, episodes_per_difficulty: int) -> float:
    """Benchmark policy_best.pt across all 4 difficulties. Returns point_win_pct."""
    import torch as th

    from surround.conf import constants as cfg
    from surround.d3qn.train_d3qn import (
        DuelingDQN,
        _resize_to_preprocess,
        _step_until_new_frame,
    )
    from surround.utils.checkpoint import load_checkpoint
    from surround.utils.env_state import make_env
    from surround.utils.video_extract_locations import get_location, observation_to_class_map

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    state_dict, _ = load_checkpoint(ckpt_dir / "policy_best.pt", map_location=device)
    net = DuelingDQN(cfg.N_ACTIONS).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    episode_returns = []
    for difficulty in range(4):
        env = make_env(difficulty, cfg.MODE, frameskip=cfg.FRAME_SKIP)
        for ep in range(episodes_per_difficulty):
            obs, _ = env.reset(seed=ep)
            last_pos = get_location(obs)
            total = 0.0
            for _ in range(cfg.MAX_CYCLES):
                cm = _resize_to_preprocess(observation_to_class_map(obs))
                x = th.from_numpy(cm).to(device).float().unsqueeze(0).unsqueeze(0)
                with th.no_grad():
                    action = int(net(x).max(1).indices.item()) + 1
                obs, reward, terminated, truncated, info = _step_until_new_frame(
                    env, last_pos, action
                )
                last_pos = info["location"]
                total += reward
                if terminated or truncated:
                    break
            episode_returns.append(total)
        env.close()
    my_pts = sum(10 if r >= 0 else 10 + r for r in episode_returns)
    opp_pts = sum((10 - r) if r >= 0 else 10 for r in episode_returns)
    return 100.0 * my_pts / (my_pts + opp_pts)


if not BEST_DIR.exists():
    print(f"Initializing best checkpoint from {SEED_DIR} ...")
    ckpt_dir = BEST_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for fname in ["policy_best.pt", "policy_latest.pt", "metadata.json"]:
        src = SEED_DIR / "checkpoints" / fname
        if src.exists():
            shutil.copy(src, ckpt_dir / fname)
    print("Measuring seed baseline across all 4 difficulties ...")
    baseline_pct = _measure_seed_baseline(ckpt_dir, BENCHMARK_EPISODES_PER_DIFFICULTY)
    print(f"Seed baseline: {baseline_pct:.2f}% point_win_pct")
    BEST_RESULT_PATH.write_text(
        json.dumps({"point_win_pct": baseline_pct, "source": "autoresearch1_best"})
    )
    print("Initialized.")

prev_best_win_pct = json.loads(BEST_RESULT_PATH.read_text()).get("point_win_pct", 0.0)

# ---------------------------------------------------------------------------
# Patch constants so the trainer resumes from best/ with fresh epsilon
# (must happen before importing D3QNTrainer)
# ---------------------------------------------------------------------------

from surround.conf import constants  # noqa: E402

constants.D3QN_RESUME_FROM = BEST_DIR
constants.D3QN_FRESH_EPSILON = True  # restart epsilon from eps_start each run
# EPS_START is controlled by constants.py (agent-tunable)

# Clear current run dir
# (D3QNTrainer raises FileExistsError if the log dir already exists)
if constants.D3QN_LOG_DIR.exists():
    shutil.rmtree(constants.D3QN_LOG_DIR)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

from surround.d3qn.train_d3qn import (  # noqa: E402
    D3QNTrainer,
    DuelingDQN,
    _resize_to_preprocess,
    _step_until_new_frame,
)
from surround.utils.checkpoint import load_checkpoint  # noqa: E402
from surround.utils.env_state import make_env  # noqa: E402
from surround.utils.video_extract_locations import (  # noqa: E402
    get_location,
    observation_to_class_map,
)

t_total_start = time.time()

trainer = D3QNTrainer()
trainer.run()

training_seconds = time.time() - t_total_start

# ---------------------------------------------------------------------------
# Benchmark (greedy policy from best checkpoint of this run, no video)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict, _ = load_checkpoint(constants.D3QN_CKPT.best, map_location=device)
net = DuelingDQN(constants.N_ACTIONS).to(device)
net.load_state_dict(state_dict)
net.eval()

returns = []
for difficulty in range(4):
    env = make_env(difficulty, constants.MODE, frameskip=constants.FRAME_SKIP)
    for ep in range(BENCHMARK_EPISODES_PER_DIFFICULTY):
        obs, info = env.reset(seed=ep)
        last_pos = get_location(obs)
        total = 0.0
        last_action = 1
        for _ in range(constants.MAX_CYCLES):
            class_map = observation_to_class_map(obs)
            class_map = _resize_to_preprocess(class_map)
            x = torch.from_numpy(class_map).to(device).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                action_index = int(net(x).max(1).indices.item())
            action = action_index + 1  # env actions are 1..4
            obs, reward, terminated, truncated, info = _step_until_new_frame(env, last_pos, action)
            last_pos = info["location"]
            total += reward
            if terminated or truncated:
                break
            last_action = action
        returns.append(total)
    env.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

n = len(returns)
my_points = sum(10 if r >= 0 else 10 + r for r in returns)
opp_points = sum((10 - r) if r >= 0 else 10 for r in returns)
point_win_pct = 100.0 * my_points / (my_points + opp_points)
mean_ret = sum(returns) / n
std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / n)
total_seconds = time.time() - t_total_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

# Per-difficulty breakdown (returns are ordered: BENCHMARK_EPISODES_PER_DIFFICULTY per difficulty)
epd = BENCHMARK_EPISODES_PER_DIFFICULTY
diff_win_pcts = []
for i in range(4):
    dr = returns[i * epd : (i + 1) * epd]
    my_pts = sum(10 if r >= 0 else 10 + r for r in dr)
    opp_pts = sum((10 - r) if r >= 0 else 10 for r in dr)
    diff_win_pcts.append(100.0 * my_pts / (my_pts + opp_pts))

# Update best checkpoint if this run improved
if point_win_pct > prev_best_win_pct:
    ckpt_dir = BEST_DIR / "checkpoints"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    shutil.copytree(constants.D3QN_CKPT.dir, ckpt_dir)
    BEST_RESULT_PATH.write_text(json.dumps({"point_win_pct": point_win_pct}))
    print(f"New best checkpoint saved ({prev_best_win_pct:.2f}% -> {point_win_pct:.2f}%)")

print("---")
print(f"point_win_pct:      {point_win_pct:.2f}")
print(f"prev_best_win_pct:  {prev_best_win_pct:.2f}")
print(f"diff0_win_pct:      {diff_win_pcts[0]:.2f}")
print(f"diff1_win_pct:      {diff_win_pcts[1]:.2f}")
print(f"diff2_win_pct:      {diff_win_pcts[2]:.2f}")
print(f"diff3_win_pct:      {diff_win_pcts[3]:.2f}")
print(f"mean_return:        {mean_ret:.3f}")
print(f"std_return:         {std_ret:.3f}")
print(f"benchmark_episodes: {n}")
print(f"best_win_rate:      {trainer.best_win_rate:.3f}")
print(f"training_seconds:   {training_seconds:.1f}")
print(f"total_seconds:      {total_seconds:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
