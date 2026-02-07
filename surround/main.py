from __future__ import annotations

import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import imageio.v2 as imageio
from tqdm import trange

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from surround.human import get_human_action

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
MAX_CYCLES = 10000
VIDEO_DIR = Path("video")
VIDEO_PATH = VIDEO_DIR / "surround.mp4"
VIDEO_FPS = 120
FRAME_STRIDE = 4
DIFFICULTY = 0
MODE = 0
POLICY = "human"

POLICIES = {
    "human": get_human_action,
    "random": lambda action_space, _observation, _info: action_space.sample(),
}


def make_env(difficulty: int, mode: int):
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="ram",
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
        render_mode="rgb_array",
    )


def main() -> None:
    env = make_env(DIFFICULTY, MODE)
    policy = POLICIES[POLICY]

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    observation, info = env.reset()
    video_writer = imageio.get_writer(str(VIDEO_PATH), fps=VIDEO_FPS, macro_block_size=1)
    try:
        for cycle_step in trange(MAX_CYCLES):
            action = policy(env.action_space, observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            if cycle_step % FRAME_STRIDE == 0:
                frame = env.render()
                if frame is not None:
                    video_writer.append_data(frame)
            if terminated or truncated:
                break
    finally:
        video_writer.close()
        env.close()


if __name__ == "__main__":
    main()
