"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import json
import logging
import math
import random
from collections import deque, namedtuple
from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.actions import ACTION_WORD_TO_ID
from surround.utils.video_extract_locations import get_location

# Environment / game constants
DIFFICULTY = 0
MODE = 0
FRAME_SKIP = 4
GRID_ROWS = 18
GRID_COLS = 38
STATE_MODE = "state_tuple"

# DQN hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
NUM_EPISODES = 600
MAX_CYCLES = 10000
N_OBSERVATIONS = 11  # state tuple size from get_state_tuple (includes clear_* to wall)
LOG_DIR = Path("runs/surround_dqn_save")
CHECKPOINT_DIR = LOG_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 50
POLICY_NET_LATEST = CHECKPOINT_DIR / "policy_net_latest.pt"
CHECKPOINT_METADATA = CHECKPOINT_DIR / "metadata.json"

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def get_state_tuple(locations: dict, last_action: int) -> tuple[int, ...]:
    """Build state tuple: (
        d_up,
        d_right,
        d_left,
        d_down,
        clear_up,
        clear_right,
        clear_left,
        clear_down,
        rel_x,
        rel_y,
        last_action).

    d_*: 1 if adjacent cell blocked.
    clear_*: 1 if path to wall in that direction is fully open.
    """
    if locations["ego"] is None or locations["opp"] is None:
        return (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, last_action)
    ego_row, ego_col = locations["ego"]
    opp_row, opp_col = locations["opp"]
    wall_set = locations["walls"]
    collisions = (
        wall_set | {(opp_row, opp_col)} if opp_row is not None and opp_col is not None else wall_set
    )

    d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
    d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= GRID_COLS - 1 else 0
    d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
    d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= GRID_ROWS - 1 else 0

    # Clear path to wall: 1 if every cell in that direction until the wall is free
    clear_up = 1 if all((r, ego_col) not in collisions for r in range(ego_row - 1, -1, -1)) else 0
    clear_right = (
        1 if all((ego_row, c) not in collisions for c in range(ego_col + 1, GRID_COLS)) else 0
    )
    clear_left = 1 if all((ego_row, c) not in collisions for c in range(ego_col - 1, -1, -1)) else 0
    clear_down = (
        1 if all((r, ego_col) not in collisions for r in range(ego_row + 1, GRID_ROWS)) else 0
    )

    rel_x = 0 if opp_col < ego_col else (2 if opp_col > ego_col else 1)
    rel_y = 0 if opp_row < ego_row else (2 if opp_row > ego_row else 1)

    return (
        d_up,
        d_right,
        d_left,
        d_down,
        clear_up,
        clear_right,
        clear_left,
        clear_down,
        rel_x,
        rel_y,
        last_action,
    )


def get_state_from_observation(observation: np.ndarray, last_action: int) -> tuple[int, ...]:
    """Build state tuple from raw observation (for policy / benchmark)."""
    if STATE_MODE == "ram":
        raise ValueError("RAM state mode is not supported.")
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    locations = get_location(frame)
    return get_state_tuple(locations, last_action)


class DQN(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


class DQNTrainer:
    """Single class owning env, networks, replay memory, and training loop."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Surround-v5",
            obs_type="rgb",
            full_action_space=False,
            difficulty=DIFFICULTY,
            mode=MODE,
            frameskip=FRAME_SKIP,
        )
        self.n_actions = self.env.action_space.n - 1  # ignore NOOP
        self.policy_net = DQN(N_OBSERVATIONS, self.n_actions).to(self.device)
        self.target_net = DQN(N_OBSERVATIONS, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory: deque = deque([], maxlen=MEMORY_CAPACITY)
        self.steps_done = 0
        self.episode_durations: list[int] = []
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        if LOG_DIR.exists():
            raise FileExistsError(f"Log directory already exists: {LOG_DIR}")
        self.writer = SummaryWriter(log_dir=str(LOG_DIR))
        self.writer.add_custom_scalars(
            {
                "episode/steps_survived_by_outcome": {
                    "steps_survived": [
                        "Multiline",
                        [
                            "episode/steps_survived_win",
                            "episode/steps_survived_loss",
                            "episode/steps_survived_trunc",
                        ],
                    ]
                }
            }
        )

    def _get_state(self, observation: np.ndarray, last_action: int) -> tuple[int, ...]:
        """Build state tuple from raw observation (same logic as Q-learning)."""
        return get_state_from_observation(observation, last_action)

    def _state_to_tensor(self, state_tuple: tuple[int, ...]) -> torch.Tensor:
        """Convert state tuple to (1, n_observations) float tensor."""
        return torch.tensor([state_tuple], dtype=torch.float32, device=self.device)

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        return torch.tensor(
            [[random.randrange(self.n_actions)]],
            device=self.device,
            dtype=torch.long,
        )

    def _optimize_model(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _soft_update_target(self) -> None:
        target = self.target_net.state_dict()
        policy = self.policy_net.state_dict()
        for key in policy:
            target[key] = policy[key] * TAU + target[key] * (1 - TAU)
        self.target_net.load_state_dict(target)

    def _save_checkpoint(self, episode_index: int) -> None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ep = episode_index + 1
        torch.save(self.policy_net.state_dict(), POLICY_NET_LATEST)
        metadata = {"episode_index": episode_index, "episodes_completed": ep}
        CHECKPOINT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if ep % CHECKPOINT_INTERVAL == 0:
            path = CHECKPOINT_DIR / f"policy_net_{ep:04d}.pt"
            torch.save(self.policy_net.state_dict(), path)

    def run(self) -> None:
        for episode_index in trange(NUM_EPISODES):
            observation, _info = self.env.reset()
            last_action = ACTION_WORD_TO_ID["LEFT"]
            state = self._state_to_tensor(self._get_state(observation, last_action))
            terminal_reward = 0.0

            for t in trange(MAX_CYCLES, leave=False):
                action = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = self.env.step(action_id)
                reward_t = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = self._state_to_tensor(self._get_state(observation, action_id))

                self.memory.append(Transition(state, action, next_state, reward_t))
                state = next_state

                self._optimize_model()
                self._soft_update_target()

                if done:
                    terminal_reward = float(reward)
                    steps_survived = t + 1
                    self.episode_durations.append(steps_survived)

                    self.writer.add_scalar("episode/steps_survived", steps_survived, episode_index)
                    self.writer.add_scalar(
                        "episode/terminal_reward", terminal_reward, episode_index
                    )
                    self.writer.add_scalar(
                        "episode/steps_survived_win",
                        steps_survived if terminal_reward > 0 else float("nan"),
                        episode_index,
                    )
                    self.writer.add_scalar(
                        "episode/steps_survived_loss",
                        steps_survived if terminal_reward < 0 else float("nan"),
                        episode_index,
                    )
                    self.writer.add_scalar(
                        "episode/steps_survived_trunc",
                        steps_survived if terminal_reward == 0 else float("nan"),
                        episode_index,
                    )
                    eps = EPS_END + (EPS_START - EPS_END) * math.exp(
                        -1.0 * self.steps_done / EPS_DECAY
                    )
                    self.writer.add_scalar("episode/epsilon", eps, episode_index)
                    self._save_checkpoint(episode_index)
                    break
        self.writer.close()
        print("Training complete!")


# Policy for benchmark: lazy-load policy net from latest checkpoint
_POLICY_NET_CACHE: DQN | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4  # Surround without NOOP


def _load_policy_net() -> DQN:
    global _POLICY_NET_CACHE
    if _POLICY_NET_CACHE is None:
        if not POLICY_NET_LATEST.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {POLICY_NET_LATEST}. Run training first."
            )
        _POLICY_NET_CACHE = DQN(N_OBSERVATIONS, N_ACTIONS).to(_DEVICE)
        _POLICY_NET_CACHE.load_state_dict(
            torch.load(POLICY_NET_LATEST, map_location=_DEVICE, weights_only=True)
        )
        _POLICY_NET_CACHE.eval()
    return _POLICY_NET_CACHE


def greedy_dqn_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved DQN weights (same signature as greedy_q_policy)."""
    net = _load_policy_net()
    state_tuple = get_state_from_observation(observation, last_action)
    x = torch.tensor([state_tuple], dtype=torch.float32, device=_DEVICE)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1  # env action 1..4


if __name__ == "__main__":
    DQNTrainer().run()
