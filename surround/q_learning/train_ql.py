import json
from pathlib import Path
from typing import Callable, cast

import ale_py
import gymnasium as gym
import numpy as np
from tqdm import trange

from surround.q_learning.ram_probe import create_extractor

DIFFICULTY = 0
MODE = 0
SEED = 0
MAX_CYCLES = 1000000
ALPHA = 0.1
GAMMA = 0.99
USE_MULTI_AGENT_EXTRACTOR = True
CLIP_MAX = 7
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
EPSILON = 0.1
EPISODES = 40000


def total_possible_states() -> int:
    return (CLIP_MAX + 1) ** 4 * (2 * CLIP_MAX + 1) ** 2


def make_env(difficulty: int, mode: int):
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="ram",
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
    )


class QLearning:
    def __init__(self, epsilon: float, episodes: int):
        print("Initializing Q-Learning...")
        self.env = make_env(difficulty=0, mode=0)
        self.epsilon = epsilon
        self.episodes = episodes
        self.n_actions = self.env.action_space.n
        print("creating extractor...")
        self.extractor = create_extractor(use_multi_agent=USE_MULTI_AGENT_EXTRACTOR)

        print("creating q-table...")
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        print("q-table created")
        self.unique_states: set[tuple[int, ...]] = set()
        self.total_steps = 0
        self.random_steps = 0
        self.greedy_steps = 0
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []

    def _get_state(self, observation: np.ndarray) -> tuple[int, ...]:
        features = self.extractor(observation)
        return _clip_state(tuple(int(value) for value in features))

    def _get_q(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state]

    def _get_action(self, state: tuple[int, ...]) -> tuple[int, bool]:
        if np.random.random() < self.epsilon:
            return int(self.env.action_space.sample()), True
        return int(np.argmax(self._get_q(state))), False

    def run_episode(self, episode_index: int):
        observation, _info = self.env.reset(seed=SEED + episode_index)
        state = self._get_state(observation)
        self.unique_states.add(state)
        episode_steps = 0
        episode_return = 0.0
        for cycle_step in trange(
            MAX_CYCLES,
            leave=False,
        ):
            action, is_random = self._get_action(state)
            observation, reward, terminated, truncated, _info = self.env.step(action)

            next_state = self._get_state(observation)
            q_values = self._get_q(state)
            next_best = float(np.max(self._get_q(next_state)))
            q_values[action] = q_values[action] + ALPHA * (
                reward + GAMMA * next_best - q_values[action]
            )
            state = next_state
            self.unique_states.add(state)
            self.total_steps += 1
            episode_steps += 1
            episode_return += float(reward)
            if is_random:
                self.random_steps += 1
            else:
                self.greedy_steps += 1

            if terminated or truncated:
                break
        self.episode_lengths.append(episode_steps)
        self.episode_returns.append(episode_return)

    def train(self):
        print("Training Q-Learning...")
        for iternum in trange(self.episodes):
            self.run_episode(episode_index=iternum)
            print("q_table size:", len(self.q_table))
            if (iternum + 1) % 1000 == 0:
                self.save_q_table(Q_TABLE_PATH)
        self.save_q_table(Q_TABLE_PATH)

    def save_q_table(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_states = total_possible_states()
        q_table_size = len(self.q_table)
        random_rate = self.random_steps / self.total_steps if self.total_steps else 0.0
        mean_episode_length = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        mean_episode_return = float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
        data = {
            "clip_max": CLIP_MAX,
            "analysis": {
                "episodes": self.episodes,
                "max_cycles": MAX_CYCLES,
                "epsilon": self.epsilon,
                "q_table_size": q_table_size,
                "unique_states": len(self.unique_states),
                "total_states_possible": total_states,
                "state_coverage": q_table_size / total_states if total_states else 0.0,
                "total_steps": self.total_steps,
                "random_steps": self.random_steps,
                "greedy_steps": self.greedy_steps,
                "random_action_rate": random_rate,
                "mean_episode_length": mean_episode_length,
                "mean_episode_return": mean_episode_return,
            },
            "states": {
                ",".join(map(str, state)): values.tolist() for state, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


_Q_TABLE_CACHE: dict[tuple[int, ...], np.ndarray] | None = None
_EXTRACTOR_CACHE: Callable[[np.ndarray], tuple[int, ...]] | None = None


def load_q_table(path: Path) -> dict[tuple[int, ...], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    states: dict[tuple[int, ...], np.ndarray] = {}
    for key, values in data.get("states", {}).items():
        state = tuple(int(part) for part in key.split(","))
        states[state] = np.asarray(values, dtype=np.float32)
    return states


def _clip_state(features: tuple[int, ...]) -> tuple[int, ...]:
    clipped = []
    for idx, value in enumerate(features):
        if idx < 4:
            clipped.append(int(np.clip(value, 0, CLIP_MAX)))
        else:
            clipped.append(int(np.clip(value, -CLIP_MAX, CLIP_MAX)))
    return tuple(clipped)


def greedy_q_policy(action_space, observation, info):
    global _Q_TABLE_CACHE, _EXTRACTOR_CACHE

    if _Q_TABLE_CACHE is None:
        _Q_TABLE_CACHE = load_q_table(Q_TABLE_PATH)
    if _EXTRACTOR_CACHE is None:
        _EXTRACTOR_CACHE = create_extractor(use_multi_agent=USE_MULTI_AGENT_EXTRACTOR)

    extractor = cast(Callable[[np.ndarray], tuple[int, ...]], _EXTRACTOR_CACHE)
    features = extractor(observation)
    state = _clip_state(tuple(int(value) for value in features))
    q_values = _Q_TABLE_CACHE.get(state)
    if q_values is None:
        return int(action_space.sample())
    return int(np.argmax(q_values))


if __name__ == "__main__":
    q_learning = QLearning(
        epsilon=EPSILON,
        episodes=EPISODES,
    )
    q_learning.train()
    print("Training complete")
