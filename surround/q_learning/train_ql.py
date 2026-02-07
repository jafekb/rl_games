import ale_py
import gymnasium as gym
import numpy as np
from tqdm import trange

RECORD_VIDEO = True
DIFFICULTY = 1
MODE = 0


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

    def train(self):
        print("Training Q-Learning...")

        for iternum in trange(self.episodes):
            pass


if __name__ == "__main__":
    q_learning = QLearning(
        epsilon=0.1,
        episodes=2000,
    )
    q_learning.train()
