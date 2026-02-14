"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import json
import logging
import math
import random
from collections import deque, namedtuple

import ale_py
import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.conf import constants
from surround.utils.video_extract_locations import observation_to_class_map

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def _conv_out_size(h: int, w: int, kernel_size: int = 5, stride: int = 2) -> tuple[int, int]:
    h1 = (h - kernel_size) // stride + 1
    w1 = (w - kernel_size) // stride + 1
    h2 = (h1 - kernel_size) // stride + 1
    w2 = (w1 - kernel_size) // stride + 1
    return h2, w2


class DQN(torch.nn.Module):
    """CNN that takes 4-class game map (1, H, W) and outputs Q-values for each action."""

    def __init__(self, n_actions: int):
        super().__init__()
        h, w = constants.DQN_GAME_HEIGHT, constants.DQN_GAME_WIDTH
        h_out, w_out = _conv_out_size(h, w)
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.flat_size = 32 * h_out * w_out
        self.fc1 = torch.nn.Linear(self.flat_size, 128)
        self.fc2 = torch.nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view(-1, self.flat_size)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


class DQNTrainer:
    """Single class owning env, networks, replay memory, and training loop."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Surround-v5",
            obs_type="rgb",
            full_action_space=False,
            difficulty=constants.DIFFICULTY,
            mode=constants.MODE,
            frameskip=constants.DQN_FRAME_SKIP,
        )
        self.n_actions = self.env.action_space.n - 1  # ignore NOOP
        self.policy_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=constants.LR, amsgrad=True
        )
        self.memory: deque = deque([], maxlen=constants.MEMORY_CAPACITY)
        self.steps_done = 0
        self.episode_durations: list[int] = []
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        if constants.DQN_LOG_DIR.exists():
            raise FileExistsError(f"Log directory {constants.DQN_LOG_DIR} already exists.")
        self.writer = SummaryWriter(log_dir=str(constants.DQN_LOG_DIR))
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

    def _preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert RGB observation to 4-class map (H, W) uint8."""
        return observation_to_class_map(observation)

    def _observation_to_tensor(self, class_map: np.ndarray) -> torch.Tensor:
        """Convert (H, W) class map to (1, 1, H, W) float tensor."""
        x = torch.from_numpy(class_map).to(torch.float32).to(self.device)
        return x.unsqueeze(0).unsqueeze(0)

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = constants.EPS_END + (constants.EPS_START - constants.EPS_END) * math.exp(
            -1.0 * self.steps_done / constants.EPS_DECAY
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
        if len(self.memory) < constants.BATCH_SIZE:
            return

        transitions = random.sample(self.memory, constants.BATCH_SIZE)
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

        next_state_values = torch.zeros(constants.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * constants.GAMMA_DQN) + reward_batch

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
            target[key] = policy[key] * constants.TAU + target[key] * (1 - constants.TAU)
        self.target_net.load_state_dict(target)

    def _save_checkpoint(self, episode_index: int) -> None:
        constants.DQN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ep = episode_index + 1
        torch.save(self.policy_net.state_dict(), constants.DQN_POLICY_NET_LATEST)
        metadata = {"episode_index": episode_index, "episodes_completed": ep}
        constants.DQN_CHECKPOINT_METADATA.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if ep % constants.DQN_CHECKPOINT_INTERVAL == 0:
            path = constants.DQN_CHECKPOINT_DIR / f"policy_net_{ep:04d}.pt"
            torch.save(self.policy_net.state_dict(), path)

    def run(self) -> None:
        for episode_index in trange(constants.NUM_EPISODES):
            observation, _info = self.env.reset()
            state = self._observation_to_tensor(self._preprocess_observation(observation))
            terminal_reward = 0.0

            for t in trange(constants.MAX_CYCLES, leave=False):
                action = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = self.env.step(action_id)
                reward_t = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = self._observation_to_tensor(
                        self._preprocess_observation(observation)
                    )

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
                    eps = constants.EPS_END + (constants.EPS_START - constants.EPS_END) * math.exp(
                        -1.0 * self.steps_done / constants.EPS_DECAY
                    )
                    self.writer.add_scalar("episode/epsilon", eps, episode_index)
                    self._save_checkpoint(episode_index)
                    break
        self.writer.close()
        print("Training complete!")


# Policy for benchmark: lazy-load policy net from latest checkpoint
_POLICY_NET_CACHE: DQN | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_policy_net() -> DQN:
    global _POLICY_NET_CACHE
    if _POLICY_NET_CACHE is None:
        if not constants.DQN_POLICY_NET_LATEST.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {constants.DQN_POLICY_NET_LATEST}. Run training first."
            )
        _POLICY_NET_CACHE = DQN(constants.N_ACTIONS).to(_DEVICE)
        _POLICY_NET_CACHE.load_state_dict(
            torch.load(
                constants.DQN_POLICY_NET_LATEST,
                map_location=_DEVICE,
                weights_only=True,
            )
        )
        _POLICY_NET_CACHE.eval()
    return _POLICY_NET_CACHE


def greedy_dqn_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved DQN weights (same signature as greedy_q_policy)."""
    net = _load_policy_net()
    class_map = observation_to_class_map(observation)
    x = torch.from_numpy(class_map).to(torch.float32).to(_DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1  # env action 1..4


if __name__ == "__main__":
    DQNTrainer().run()
