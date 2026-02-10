"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import random
from collections import deque, namedtuple

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
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
N_OBSERVATIONS = 7  # state tuple size from get_state_tuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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

    @staticmethod
    def _get_state_tuple(locations: dict, last_action: int) -> tuple[int, ...]:
        """Build state tuple (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)."""
        if locations["ego"] is None or locations["opp"] is None:
            return (1, 1, 1, 1, 1, 1, last_action)
        ego_row, ego_col = locations["ego"]
        opp_row, opp_col = locations["opp"]
        wall_set = locations["walls"]
        collisions = (
            wall_set | {(opp_row, opp_col)}
            if opp_row is not None and opp_col is not None
            else wall_set
        )

        d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
        d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= GRID_COLS - 1 else 0
        d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
        d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= GRID_ROWS - 1 else 0

        rel_x = 0 if opp_col < ego_col else (2 if opp_col > ego_col else 1)
        rel_y = 0 if opp_row < ego_row else (2 if opp_row > ego_row else 1)

        return (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)

    def _get_state(self, observation: np.ndarray, last_action: int) -> tuple[int, ...]:
        """Build state tuple from raw observation (same logic as Q-learning)."""
        if STATE_MODE == "ram":
            raise ValueError("RAM state mode is not supported.")
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        locations = get_location(frame)
        return self._get_state_tuple(locations, last_action)

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

    def run(self) -> None:
        for episode_index in trange(NUM_EPISODES):
            observation, _info = self.env.reset()
            last_action = ACTION_WORD_TO_ID["LEFT"]
            state = self._state_to_tensor(self._get_state(observation, last_action))

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
                    self.episode_durations.append(t + 1)
                    break
        print("Training complete!")


if __name__ == "__main__":
    DQNTrainer().run()
