from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import imageio.v2 as imageio
from tqdm import trange

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from surround import constants
from surround.dqn.train_dqn import CHECKPOINT_METADATA, greedy_dqn_policy
from surround.q_learning.train_ql import greedy_q_policy
from surround.utils.env_state import make_env

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
MAX_CYCLES = 100_000
EPISODES = 5
RECORD_VIDEO = True
VIDEO_DIR = Path("video")
VIDEO_FPS = 120
FRAME_STRIDE = 4


POLICIES = {
    # "random": random_policy,
    # "human": get_human_action,
    "dqn": greedy_dqn_policy,
    "q_learning": greedy_q_policy,
    # "snake": snake_policy,
}


def run_episode(env, policy, seed, video_writer, episode_index: int):
    observation, info = env.reset(seed=seed)
    total = 0.0
    last_action = 1
    for cycle_step in trange(
        MAX_CYCLES,
        desc=f"Episode {episode_index + 1}/{EPISODES}",
        leave=False,
    ):
        action = policy(env.action_space, observation, info, last_action)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if video_writer is not None and cycle_step % FRAME_STRIDE == 0:
            frame = env.render()
            if frame is not None:
                video_writer.append_data(frame)
            # imageio.imwrite(f"video/frame_{cycle_step:04d}.png", frame)

        if terminated or truncated:
            break
        last_action = action
    return total


def summarize(returns):
    return {
        "mean": mean(returns),
        "std": pstdev(returns) if len(returns) > 1 else 0.0,
    }


def main() -> None:
    env = make_env(
        constants.DIFFICULTY,
        constants.MODE,
        frameskip=FRAME_STRIDE,
        render_mode="rgb_array" if RECORD_VIDEO else None,
    )
    try:
        results = {}
        q_table_episodes = None
        dqn_episodes = None
        if constants.Q_TABLE_PATH.exists():
            data = json.loads(constants.Q_TABLE_PATH.read_text(encoding="utf-8"))
            analysis = data.get("analysis", {})
            q_table_episodes = analysis.get("episode_index")
        if CHECKPOINT_METADATA.exists():
            dqn_meta = json.loads(CHECKPOINT_METADATA.read_text(encoding="utf-8"))
            dqn_episodes = dqn_meta.get("episode_index")
        for policy_name, policy in POLICIES.items():
            video_writer = None
            if RECORD_VIDEO:
                VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                video_path = VIDEO_DIR / f"{policy_name}.mp4"
                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=VIDEO_FPS,
                    macro_block_size=1,
                )
            returns = []
            for episode in trange(EPISODES, desc=f"Episodes ({policy_name})"):
                total = run_episode(
                    env,
                    policy,
                    seed=constants.SEED + episode,
                    video_writer=video_writer,
                    episode_index=episode,
                )
                returns.append(total)
            results[policy_name] = summarize(returns)
            if video_writer is not None:
                video_writer.close()
    finally:
        env.close()

    print(f"Episodes: {EPISODES}")
    for policy_name, stats in results.items():
        if policy_name == "q_learning" and q_table_episodes is not None:
            name_label = f"{policy_name} ({q_table_episodes} episodes)"
        elif policy_name == "dqn" and dqn_episodes is not None:
            name_label = f"{policy_name} ({dqn_episodes} episodes)"
        else:
            name_label = policy_name
        print(f"{name_label}: mean_return={stats['mean']:.2f} std={stats['std']:.2f}")


if __name__ == "__main__":
    main()
