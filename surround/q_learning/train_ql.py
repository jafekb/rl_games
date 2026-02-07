import json
from pathlib import Path
from typing import Callable

import ale_py
import gymnasium as gym
import numpy as np
from tqdm import trange

from surround.utils.ram_probe import create_extractor
from surround.utils.video_extract_locations import get_location

DIFFICULTY = 0
MODE = 0
SEED = 0
MAX_CYCLES = 10_000
ALPHA = 0.1
GAMMA = 0.99
USE_MULTI_AGENT_EXTRACTOR = True
CLIP_MAX = 7
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1000
EPISODES = 1_000_000
STEP_REWARD = 0.01
STATE_MODE = "surround_window"
WINDOW_SIZE = 7
DEBUG_STATE = False

EMPTY_CELL = 0
WALL_CELL = 1
EGO_CELL = 2


def total_possible_states(state_mode: str) -> int:
    if state_mode == "ram":
        return (CLIP_MAX + 1) ** 4 * (2 * CLIP_MAX + 1) ** 2
    if state_mode == "surround_window":
        window_cells = WINDOW_SIZE * WINDOW_SIZE
        window_states = 3**window_cells
        delta_states = (2 * CLIP_MAX + 1) ** 2
        exit_states = 5 * 5
        return window_states * delta_states * exit_states
    raise ValueError(f"Unknown state mode: {state_mode}")


def make_env(difficulty: int, mode: int, state_mode: str):
    gym.register_envs(ale_py)
    obs_type = "ram" if state_mode == "ram" else "rgb"
    return gym.make(
        "ALE/Surround-v5",
        obs_type=obs_type,
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
    )


def _clip_delta(value: int) -> int:
    return int(np.clip(value, -CLIP_MAX, CLIP_MAX))


def _build_occupancy_grid(
    walls: list[tuple[int, int]],
    ego: tuple[int, int],
    opp: tuple[int, int],
) -> list[list[int]]:
    grid = [[EMPTY_CELL for _ in range(38)] for _ in range(18)]
    for row, col in walls:
        if 0 <= row < 18 and 0 <= col < 38:
            grid[row][col] = WALL_CELL
    erow, ecol = ego
    if 0 <= erow < 18 and 0 <= ecol < 38:
        grid[erow][ecol] = EGO_CELL
    return grid


def _count_exits(
    position: tuple[int, int],
    walls: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> int:
    row, col = position
    exits = 0
    for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nrow = row + drow
        ncol = col + dcol
        if not (0 <= nrow < 18 and 0 <= ncol < 38):
            continue
        if (nrow, ncol) in walls or (nrow, ncol) in blocked:
            continue
        exits += 1
    return exits


def _window_features(
    grid: list[list[int]],
    center: tuple[int, int],
) -> list[int]:
    radius = WINDOW_SIZE // 2
    features: list[int] = []
    center_row, center_col = center
    for drow in range(-radius, radius + 1):
        for dcol in range(-radius, radius + 1):
            row = center_row + drow
            col = center_col + dcol
            if 0 <= row < 18 and 0 <= col < 38:
                features.append(grid[row][col])
            else:
                features.append(WALL_CELL)
    return features


def build_state_from_observation(
    observation: np.ndarray,
    *,
    state_mode: str,
    ram_extractor: Callable[[np.ndarray], tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    if state_mode == "ram":
        if ram_extractor is None:
            raise RuntimeError("RAM extractor not initialized.")
        return _clip_state(ram_extractor(observation))

    locations = get_location(observation, color_space="rgb")
    ego = locations["ego"]
    opp = locations["opp"]
    walls = locations["walls"]
    if ego is None or opp is None or walls is None:
        return (0,) * (WINDOW_SIZE * WINDOW_SIZE + 4)

    walls_set = set(walls)
    ego_set = {ego}
    opp_set = {opp}
    grid = _build_occupancy_grid(walls, ego, opp)
    window = _window_features(grid, opp)
    dx = _clip_delta(opp[1] - ego[1])
    dy = _clip_delta(opp[0] - ego[0])
    ego_exits = _count_exits(ego, walls_set, opp_set)
    opp_exits = _count_exits(opp, walls_set, ego_set)
    state = tuple(*window, dx, dy, ego_exits, opp_exits)
    if DEBUG_STATE:
        print(
            "state_debug",
            {
                "ego": ego,
                "opp": opp,
                "dx": dx,
                "dy": dy,
                "ego_exits": ego_exits,
                "opp_exits": opp_exits,
            },
        )
    return state


class QLearning:
    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_steps: int,
        episodes: int,
        state_mode: str,
    ):
        print("Initializing Q-Learning...")
        self.state_mode = state_mode
        self.env = make_env(difficulty=0, mode=0, state_mode=state_mode)
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.episodes = episodes
        self.n_actions = self.env.action_space.n
        self.ram_extractor: Callable[[np.ndarray], tuple[int, ...]] | None = None
        if self.state_mode == "ram":
            self.ram_extractor = create_extractor(
                use_multi_agent=USE_MULTI_AGENT_EXTRACTOR,
                difficulty=DIFFICULTY,
                mode=MODE,
            )

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
        return build_state_from_observation(
            observation,
            state_mode=self.state_mode,
            ram_extractor=self.ram_extractor,
        )

    def _get_q(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.ones(self.n_actions, dtype=np.float32)
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
            if not (terminated or truncated):
                reward += STEP_REWARD

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
        if self.epsilon_start <= 0:
            decay_rate = 0.0
        else:
            decay_rate = np.log(self.epsilon_start / self.epsilon_min) / max(
                self.epsilon_decay_steps, 1
            )
        for iternum in trange(self.episodes):
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start * np.exp(-decay_rate * iternum),
            )
            self.run_episode(episode_index=iternum)
            print("q_table size:", len(self.q_table))
            if (iternum + 1) % 50 == 0:
                self.save_q_table(Q_TABLE_PATH)
        self.save_q_table(Q_TABLE_PATH)

    def save_q_table(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_states = total_possible_states(self.state_mode)
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
                ",".join(map(str, state)): values.tolist() for state, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


_Q_TABLE_CACHE: dict[tuple[int, ...], np.ndarray] | None = None
_RAM_EXTRACTOR: Callable[[np.ndarray], tuple[int, ...]] | None = None
_STATS = {"valid": []}


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


def greedy_q_policy(action_space, observation, info, last_action):
    global _Q_TABLE_CACHE, _RAM_EXTRACTOR, _STATS

    if _Q_TABLE_CACHE is None:
        _Q_TABLE_CACHE = load_q_table(Q_TABLE_PATH)
    if _RAM_EXTRACTOR is None and STATE_MODE == "ram":
        _RAM_EXTRACTOR = create_extractor(
            use_multi_agent=USE_MULTI_AGENT_EXTRACTOR,
            difficulty=DIFFICULTY,
            mode=MODE,
        )

    state = build_state_from_observation(
        observation,
        state_mode=STATE_MODE,
        ram_extractor=_RAM_EXTRACTOR,
    )
    q_values = _Q_TABLE_CACHE.get(state)
    if q_values is None:
        _STATS["valid"].append(0)
        with Path("stats.json").open("w") as f:
            json.dump(_STATS, f)
        return int(action_space.sample())
    _STATS["valid"].append(1)
    with Path("stats.json").open("w") as f:
        json.dump(_STATS, f)
    return int(np.argmax(q_values))


if __name__ == "__main__":
    q_learning = QLearning(
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        episodes=EPISODES,
        state_mode=STATE_MODE,
    )
    q_learning.train()
    print("Training complete")
