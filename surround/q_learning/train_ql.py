import ale_py
import gymnasium as gym
import numpy as np
from tqdm import trange

RECORD_VIDEO = True
DIFFICULTY = 1
MODE = 0
SEED = 0
MAX_CYCLES = 10000


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


class QLearning:
    def __init__(self, epsilon: float, episodes: int):
        self.env = make_env(difficulty=0, mode=0)
        self.epsilon = epsilon
        self.episodes = episodes

        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.q_table = np.zeros((n_states, n_actions))
        print(f"Q-Table: {self.q_table.shape}")

    def _get_action(self, observation):
        import pdb

        pdb.set_trace()

    def run_episode(self, episode_index: int):
        observation, info = self.env.reset(seed=SEED + episode_index)
        total = 0.0
        for cycle_step in trange(
            MAX_CYCLES,
            leave=False,
        ):
            action = self._get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if video_writer is not None and cycle_step % FRAME_STRIDE == 0:
                frame = env.render()
                if frame is not None:
                    video_writer.append_data(frame)
            if terminated or truncated:
                break
        return total

    def train(self):
        print("Training Q-Learning...")
        returns = []

        for iternum in trange(self.episodes):
            total = self.run_episode(episode_index=iternum)
            returns.append(total)
        print(returns)


if __name__ == "__main__":
    q_learning = QLearning(
        epsilon=0.1,
        episodes=2000,
    )
    q_learning.train()
