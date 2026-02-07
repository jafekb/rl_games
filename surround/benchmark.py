from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, pstdev

import ale_py
import gymnasium as gym
import imageio.v2 as imageio
from tqdm import trange

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from surround.human import get_human_action

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
MAX_CYCLES = 10000
EPISODES = 20
SEED = 0
DIFFICULTY = 0
MODE = 0
POLICY = "human"
RECORD_VIDEO = True
VIDEO_DIR = Path("video")
VIDEO_PATH = VIDEO_DIR / "surround_benchmark.mp4"
VIDEO_FPS = 120
FRAME_STRIDE = 4


def random_policy(action_space, observation, info=None) -> int:
    return action_space.sample()


POLICIES = {
    "random": random_policy,
    "human": get_human_action,
}


def make_env(difficulty: int, mode: int):
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="ram",
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
        render_mode="rgb_array" if RECORD_VIDEO else None,
    )


def run_episode(env, policy, seed, video_writer, episode_index: int):
    observation, info = env.reset(seed=seed)
    total = 0.0
    for cycle_step in trange(
        MAX_CYCLES,
        desc=f"Episode {episode_index + 1}/{EPISODES}",
        leave=False,
    ):
        action = policy(env.action_space, observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if video_writer is not None and cycle_step % FRAME_STRIDE == 0:
            frame = env.render()
            if frame is not None:
                video_writer.append_data(frame)
        if terminated or truncated:
            break
    return total


def summarize(returns):
    return {
        "mean": mean(returns),
        "std": pstdev(returns) if len(returns) > 1 else 0.0,
    }


def main() -> None:
    env = make_env(DIFFICULTY, MODE)
    policy = POLICIES[POLICY]
    video_writer = None

    returns = []
    try:
        if RECORD_VIDEO:
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                str(VIDEO_PATH),
                fps=VIDEO_FPS,
                macro_block_size=1,
            )
        for episode in trange(EPISODES, desc="Episodes"):
            total = run_episode(
                env,
                policy,
                seed=SEED + episode,
                video_writer=video_writer,
                episode_index=episode,
            )
            returns.append(total)
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    summary = summarize(returns)
    print(f"Episodes: {EPISODES}")
    print(f"mean_return={summary['mean']:.2f} std={summary['std']:.2f}")


if __name__ == "__main__":
    main()
