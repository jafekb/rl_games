import json
from datetime import datetime
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.actions import ACTION_WORD_TO_ID
from surround.utils.video_extract_locations import get_location

DIFFICULTY = 0
MODE = 0
SEED = 0
MAX_CYCLES = 10_000
ALPHA = 0.1
GAMMA = 0.99
CLIP_MAX = 7
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
LOG_DIR = Path("runs/surround_q_learning")
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1000
EPISODES = 1_000_000
STEP_REWARD = 0.01
STATE_MODE = "state_tuple"
WINDOW_SIZE = 7
DEBUG_STATE = False
GRID_ROWS = 18
GRID_COLS = 38

EMPTY_CELL = 0
WALL_CELL = 1
EGO_CELL = 2
FRAME_SKIP = 8


def total_possible_states(state_mode: str) -> int:
    if state_mode == "state_tuple":
        return 576  # 2*2*2*2*3*3*4
    raise ValueError(f"Unknown state mode: {state_mode}")


def make_env(difficulty: int, mode: int):
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="rgb",
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
        frameskip=FRAME_SKIP,
    )


def get_state_tuple(locations, last_action: int) -> tuple[int, ...]:
    """
    Builds a state tuple from the locations of the ego, opponent, and walls.

    Args:
        locations: The locations of the ego, opponent, and walls.

    Returns:
        A tuple of the state (d_up, d_down, d_left, d_right, rel_x, rel_y, last_action).
        There are a total of 2*2*2*2*3*3*4 = 576 possible states:
        - d_up: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_right: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_left: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_down: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - rel_x: 0 if the opponent is to the left of the ego, 1 if the opponent is in the
            same column as the ego, 2 if the opponent is to the right of the ego.
        - rel_y: 0 if the opponent is above the ego, 1 if the opponent is in the same row as
            the ego, 2 if the opponent is below the ego
        - last_action: 1..4 for UP/RIGHT/LEFT/DOWN
    """
    if locations["ego"] is None or locations["opp"] is None:
        # Sometimes this happens if the game ends.
        return (1, 1, 1, 1, 1, 1, last_action)
    ego_row, ego_col = locations["ego"]
    opp_row, opp_col = locations["opp"]
    wall_set = locations["walls"]
    collisions = (
        wall_set | {(opp_row, opp_col)} if opp_row is not None and opp_col is not None else wall_set
    )

    # 1. Survival Features (4 Binary Flags)
    # Check adjacent tiles for walls or trails
    d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
    d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= GRID_COLS - 1 else 0
    d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
    d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= GRID_ROWS - 1 else 0

    # 2. Relational Features (Map -1, 0, 1 to 0, 1, 2)
    # We shift the values so they are non-negative for the index math
    if opp_col < ego_col:
        rel_x = 0  # Opponent is Left
    elif opp_col > ego_col:
        rel_x = 2  # Opponent is Right
    else:
        rel_x = 1  # Same Column

    if opp_row < ego_row:
        rel_y = 0  # Opponent is Above
    elif opp_row > ego_row:
        rel_y = 2  # Opponent is Below
    else:
        rel_y = 1  # Same Row

    return (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)


def build_state_from_observation(
    observation: np.ndarray,
    last_action: int,
    *,
    state_mode: str,
) -> tuple[int, ...]:
    """
    Builds a state from an observation.

    Args:
        observation: The observation to build a state from.
        state_mode: The state mode to use.

    Returns:
        A tuple of the state: ()
    """
    if state_mode == "ram":
        raise ValueError("RAM state mode is not supported.")
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    locations = get_location(frame)
    state = get_state_tuple(locations, last_action)
    if DEBUG_STATE:
        print("state_tuple", state)
    return state


class QLearning:
    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_steps: int,
        episodes: int,
        state_mode: str,
        log_dir: Path = LOG_DIR,
    ):
        self.state_mode = state_mode
        self.env = make_env(difficulty=0, mode=0)
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.episodes = episodes
        # We removed NOOP from the action space
        self.n_actions = self.env.action_space.n - 1

        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self.unique_states: set[tuple[int, ...]] = set()
        self.total_steps = 0
        self.random_steps = 0
        self.greedy_steps = 0
        self.episode_lengths: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_terminal_rewards: list[float] = []
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def _get_state(self, observation: np.ndarray, last_action: int) -> tuple[int, ...]:
        return build_state_from_observation(
            observation,
            last_action,
            state_mode=self.state_mode,
        )

    def _get_q(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.ones(self.n_actions, dtype=np.float32)
        return self.q_table[state]

    def _get_action(self, state: tuple[int, ...], epsilon: float) -> tuple[int, int, bool]:
        if np.random.random() < epsilon:
            action_index = int(np.random.randint(0, self.n_actions))
            return action_index + 1, action_index, True
        action_index = int(np.argmax(self._get_q(state)))
        return action_index + 1, action_index, False

    def run_episode(self, episode_index: int, epsilon: float):
        observation, _info = self.env.reset(seed=SEED + episode_index)
        last_action = ACTION_WORD_TO_ID["LEFT"]
        state = self._get_state(observation, last_action)
        self.unique_states.add(state)
        episode_steps = 0
        episode_return = 0.0
        for cycle_step in trange(
            MAX_CYCLES,
            leave=False,
        ):
            action_id, action_index, is_random = self._get_action(state, epsilon)
            observation, reward, terminated, truncated, _info = self.env.step(action_id)
            if not (terminated or truncated):
                reward += STEP_REWARD

            next_state = self._get_state(observation, action_id)
            q_values = self._get_q(state)
            next_best = float(np.max(self._get_q(next_state)))
            q_values[action_index] = q_values[action_index] + ALPHA * (
                reward + GAMMA * next_best - q_values[action_index]
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
        self.episode_terminal_rewards.append(float(reward))

    def train(self):
        if self.epsilon_start <= 0:
            decay_rate = 0.0
        else:
            decay_rate = np.log(self.epsilon_start / self.epsilon_min) / max(
                self.epsilon_decay_steps, 1
            )
        for episode_index in trange(self.episodes):
            epsilon = max(
                self.epsilon_min,
                self.epsilon_start * np.exp(-decay_rate * episode_index),
            )
            self.run_episode(episode_index=episode_index, epsilon=epsilon)
            self.writer.add_scalar(
                "episode/steps_survived",
                self.episode_lengths[-1],
                episode_index,
            )
            self.writer.add_scalar(
                "episode/terminal_reward",
                self.episode_terminal_rewards[-1],
                episode_index,
            )
            self.writer.add_scalar(
                "episode/epsilon",
                epsilon,
                episode_index,
            )
            if (episode_index + 1) % 100 == 0:
                self.save_q_table(Q_TABLE_PATH, episode_index=episode_index, epsilon=epsilon)
        self.save_q_table(Q_TABLE_PATH, episode_index=episode_index, epsilon=epsilon)
        self.writer.close()

    def save_q_table(self, path: Path, *, episode_index: int, epsilon: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        total_states = total_possible_states(self.state_mode)
        q_table_size = len(self.q_table)
        random_rate = self.random_steps / self.total_steps if self.total_steps else 0.0
        mean_episode_length = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        mean_episode_return = float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
        data = {
            "clip_max": CLIP_MAX,
            "analysis": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "episode_index": episode_index,
                "max_episodes": self.episodes,
                "max_cycles": MAX_CYCLES,
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
                ",".join(map(str, state)): values.tolist() for state, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


_Q_TABLE_CACHE: dict[tuple[int, ...], np.ndarray] | None = None
_STATS = {"valid": []}


def load_q_table(path: Path) -> dict[tuple[int, ...], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    states: dict[tuple[int, ...], np.ndarray] = {}
    for key, values in data.get("states", {}).items():
        state = tuple(int(part) for part in key.split(","))
        states[state] = np.asarray(values, dtype=np.float32)
    return states


def greedy_q_policy(action_space, observation, info, last_action):
    global _Q_TABLE_CACHE, _STATS

    if _Q_TABLE_CACHE is None:
        _Q_TABLE_CACHE = load_q_table(Q_TABLE_PATH)

    state = build_state_from_observation(
        observation,
        last_action,
        state_mode=STATE_MODE,
    )
    q_values = _Q_TABLE_CACHE.get(state)
    if q_values is None:
        _STATS["valid"].append(0)
        with Path("stats.json").open("w") as f:
            json.dump(_STATS, f)
        return int(np.random.randint(1, 5))
    _STATS["valid"].append(1)
    with Path("stats.json").open("w") as f:
        json.dump(_STATS, f)
    return int(np.argmax(q_values)) + 1


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
