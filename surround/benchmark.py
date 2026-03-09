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
from surround.d3qn.train_d3qn import _step_until_new_frame, greedy_d3qn_policy
from surround.utils.env_state import make_env
from surround.utils.video_extract_locations import get_location

ROM_PATH = str(constants.ROM_PATH)
MAX_CYCLES = 100_000
EPISODES = 20
RECORD_VIDEO = True
VIDEO_DIR = Path("video")
VIDEO_FPS = 120
FRAME_STRIDE = 4


POLICIES = {
    # "random": random_policy,
    # "human": get_human_action,
    # "dqn": greedy_dqn_policy,
    "d3qn": greedy_d3qn_policy,
    # "q_learning": greedy_q_policy,
    # "snake": snake_policy,
}


def run_episode(env, policy, seed, video_writer, episode_index: int):
    observation, info = env.reset(seed=seed)
    total = 0.0
    last_action = 1
    last_pos = get_location(observation)
    for cycle_step in trange(
        MAX_CYCLES,
        desc=f"Episode {episode_index + 1}/{EPISODES}",
        leave=False,
    ):
        action = policy(env.action_space, observation, info, last_action)
        observation, reward, terminated, truncated, info = _step_until_new_frame(
            env, last_pos, action
        )
        last_pos = info["location"]
        total += reward
        if video_writer is not None and cycle_step % FRAME_STRIDE == 0:
            frame = env.render()
            if frame is not None:
                video_writer.append_data(frame)

        if terminated or truncated:
            break
        last_action = action
    return total


def summarize(returns):
    n = len(returns)
    # Each game goes to 10 points; return is point differential (mine - opp)
    my_points = sum(10 if r >= 0 else 10 + r for r in returns)
    opp_points = sum(10 - r if r >= 0 else 10 for r in returns)
    return {
        "mean": mean(returns),
        "std": pstdev(returns) if n > 1 else 0.0,
        "my_points": my_points,
        "opp_points": opp_points,
        "point_win_pct": 100.0 * my_points / (my_points + opp_points),
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
        d3qn_episodes = None
        if constants.Q_TABLE_PATH.exists():
            data = json.loads(constants.Q_TABLE_PATH.read_text(encoding="utf-8"))
            analysis = data.get("analysis", {})
            q_table_episodes = analysis.get("episode_index")
        if constants.DQN_CKPT.metadata.exists():
            dqn_meta = json.loads(constants.DQN_CKPT.metadata.read_text(encoding="utf-8"))
            dqn_episodes = dqn_meta.get("episodes_completed")
        if constants.D3QN_CKPT.metadata.exists():
            d3qn_meta = json.loads(constants.D3QN_CKPT.metadata.read_text(encoding="utf-8"))
            d3qn_episodes = d3qn_meta.get("episodes_completed")
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
                    seed=None,
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
        elif policy_name == "d3qn" and d3qn_episodes is not None:
            name_label = f"{policy_name} ({d3qn_episodes} episodes)"
        else:
            name_label = policy_name
        s = stats
        print(
            f"{name_label}: "
            f"mean={s['mean']:.2f} std={s['std']:.2f} | "
            f"points W/L={s['my_points']}/{s['opp_points']} ({s['point_win_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
