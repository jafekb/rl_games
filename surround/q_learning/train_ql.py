import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import trange

from surround import constants
from surround.actions import ACTION_WORD_TO_ID
from surround.utils.callbacks import TensorboardCallback, TrainingCallback
from surround.utils.env_state import (
    build_state_from_observation,
    make_env,
    total_possible_states,
)


class QLearning:
    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_steps: int,
        episodes: int,
        state_mode: str,
        log_dir: Path = constants.LOG_DIR,
        callbacks: list[TrainingCallback] | None = None,
    ):
        self.state_mode = state_mode
        self.env = make_env(
            difficulty=constants.DIFFICULTY, mode=constants.MODE, frameskip=constants.FRAME_SKIP
        )
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.episodes = episodes
        # We removed NOOP from the action space
        self.n_actions = self.env.action_space.n - 1

        self.q_table: dict[tuple[int, ...], dict] = {}
        self.unique_states: set[tuple[int, ...]] = set()
        self.total_steps = 0
        self.random_steps = 0
        self.greedy_steps = 0
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_terminal_rewards: list[float] = []
        self.callbacks = callbacks if callbacks is not None else [TensorboardCallback(log_dir)]

    def _get_state(self, observation: np.ndarray, last_action: int) -> tuple[int, ...]:
        return build_state_from_observation(
            observation,
            last_action,
            state_mode=self.state_mode,
            debug_state=constants.DEBUG_STATE,
        )

    def _get_q(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = {
                "q": np.ones(self.n_actions, dtype=np.float32),
                "visit_count": 0,
            }
        return self.q_table[state]["q"]

    def _get_action(self, state: tuple[int, ...], epsilon: float) -> tuple[int, int, bool]:
        if np.random.random() < epsilon:
            action_index = int(np.random.randint(0, self.n_actions))
            return action_index + 1, action_index, True
        action_index = int(np.argmax(self._get_q(state)))
        return action_index + 1, action_index, False

    def run_episode(self, episode_index: int, epsilon: float):
        observation, _info = self.env.reset(seed=constants.SEED + episode_index)
        last_action = ACTION_WORD_TO_ID["LEFT"]
        state = self._get_state(observation, last_action)
        self.unique_states.add(state)
        episode_steps = 0
        episode_return = 0.0
        terminal_reward = 0.0
        terminated = False
        truncated = False
        for cycle_step in trange(
            constants.MAX_CYCLES,
            leave=False,
        ):
            action_id, action_index, is_random = self._get_action(state, epsilon)
            observation, reward, terminated, truncated, _info = self.env.step(action_id)
            if not (terminated or truncated):
                reward += constants.STEP_REWARD

            next_state = self._get_state(observation, action_id)
            q_values = self._get_q(state)
            next_best = float(np.max(self._get_q(next_state)))
            q_values[action_index] = q_values[action_index] + constants.ALPHA * (
                reward + constants.GAMMA * next_best - q_values[action_index]
            )
            self.q_table[state]["visit_count"] += 1
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
                terminal_reward = float(reward)
                break
        if not (terminated or truncated):
            terminal_reward = 0.0
        self.episode_lengths.append(episode_steps)
        self.episode_returns.append(episode_return)
        self.episode_terminal_rewards.append(terminal_reward)

    def train(self):
        if self.epsilon_start <= 0:
            decay_rate = 0.0
        else:
            decay_rate = np.log(self.epsilon_start / self.epsilon_min) / max(
                self.epsilon_decay_steps, 1
            )
        for cb in self.callbacks:
            cb.on_train_start()
        try:
            for episode_index in trange(self.episodes):
                epsilon = max(
                    self.epsilon_min,
                    self.epsilon_start * np.exp(-decay_rate * episode_index),
                )
                self.run_episode(episode_index=episode_index, epsilon=epsilon)
                episode_steps = self.episode_lengths[-1]
                terminal_reward = self.episode_terminal_rewards[-1]
                for cb in self.callbacks:
                    cb.on_episode_end(episode_index, episode_steps, terminal_reward, epsilon)
                if (episode_index + 1) % 100 == 0:
                    self.save_q_table(
                        constants.Q_TABLE_PATH, episode_index=episode_index, epsilon=epsilon
                    )
            self.save_q_table(constants.Q_TABLE_PATH, episode_index=episode_index, epsilon=epsilon)
        finally:
            for cb in self.callbacks:
                cb.on_train_end()

    def save_q_table(self, path: Path, *, episode_index: int, epsilon: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_states = total_possible_states(self.state_mode)
        q_table_size = len(self.q_table)
        random_rate = self.random_steps / self.total_steps if self.total_steps else 0.0
        mean_episode_length = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        mean_episode_return = float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
        data = {
            "clip_max": constants.CLIP_MAX,
            "analysis": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "episode_index": episode_index,
                "max_episodes": self.episodes,
                "max_cycles": constants.MAX_CYCLES,
                "epsilon": epsilon,
                "state_mode": self.state_mode,
                "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "epsilon_decay_type": "exponential",
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
                ",".join(map(str, state)): {
                    "q": entry["q"].tolist(),
                    "visit_count": entry["visit_count"],
                }
                for state, entry in self.q_table.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


_Q_TABLE_CACHE: dict[tuple[int, ...], dict] | None = None
_STATS = {"valid": []}


def load_q_table(path: Path) -> dict[tuple[int, ...], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    states: dict[tuple[int, ...], dict] = {}
    for key, obj in data.get("states", {}).items():
        state = tuple(int(part) for part in key.split(","))
        states[state] = {
            "q": np.asarray(obj["q"], dtype=np.float32),
            "visit_count": int(obj["visit_count"]),
        }
    return states


def greedy_q_policy(action_space, observation, info, last_action):
    global _Q_TABLE_CACHE, _STATS

    if _Q_TABLE_CACHE is None:
        _Q_TABLE_CACHE = load_q_table(constants.Q_TABLE_PATH)

    state = build_state_from_observation(
        observation,
        last_action,
        state_mode=constants.STATE_MODE,
        debug_state=constants.DEBUG_STATE,
    )
    entry = _Q_TABLE_CACHE.get(state)
    if entry is None:
        _STATS["valid"].append(0)
        with Path("stats.json").open("w") as f:
            json.dump(_STATS, f)
        return int(np.random.randint(1, 5))
    _STATS["valid"].append(1)
    with Path("stats.json").open("w") as f:
        json.dump(_STATS, f)
    return int(np.argmax(entry["q"])) + 1


if __name__ == "__main__":
    q_learning = QLearning(
        epsilon_start=constants.EPSILON_START,
        epsilon_min=constants.EPSILON_MIN,
        epsilon_decay_steps=constants.EPSILON_DECAY_STEPS,
        episodes=constants.EPISODES,
        state_mode=constants.STATE_MODE,
    )
    q_learning.train()
    print("Training complete")
