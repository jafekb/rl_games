"""Train a D3QN agent - Experiment 3 (uniform replay + reduced LR + update every 4 steps).

Differences from exp2 (train_d3qn_exp2.py):
  1. Uniform replay buffer: PER removed entirely (alpha=0 equivalent).
     PER was the root cause of Q-value divergence in exp1/exp2; baseline DQN
     achieves 185 steps/ep without it using identical TAU/arch.
  2. LR 3e-4 -> 1e-4: more conservative to prevent early overfit.
  3. Optimize every 4 env steps instead of every step: reduces gradient
     update rate, standard practice in DQN literature.

All other hyperparameters (TAU, gamma, epsilon schedule, arch, learning_starts,
input normalization, terminal fix) are identical to exp2.
Log dir: runs/surround/d3qn/exp3/
"""

import collections
import json
import logging
import math
import random
import time
from collections import namedtuple
from datetime import datetime
from zoneinfo import ZoneInfo

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
from git import Repo
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.conf import constants
from surround.utils.checkpoint import save_checkpoint
from surround.utils.video_extract_locations import get_location, observation_to_class_map

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# ---------------------------------------------------------------------------
# Env stepping helper (unchanged from exp2)
# ---------------------------------------------------------------------------


def _step_until_new_frame(
    env: gym.Env,
    last_pos: dict,
    action_id: int,
    max_substeps: int = 20,
) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Step env until ego or opponent position changes (or episode ends)."""
    total_reward = 0.0
    observation, reward, terminated, truncated, info = None, 0.0, False, False, {}
    locs: dict = {"ego": None, "opp": None}

    for _ in range(max_substeps):
        observation, reward, terminated, truncated, info = env.step(action_id)
        locs = get_location(observation)
        total_reward += reward
        if locs["ego"] is None or locs["opp"] is None:
            continue
        done = terminated or truncated or abs(reward) == 1
        if done:
            break
        if (
            last_pos.get("ego") is None
            or last_pos.get("opp") is None
            or locs["ego"] != last_pos["ego"]
            or locs["opp"] != last_pos["opp"]
        ):
            break

    info = dict(info)
    info["location"] = {"ego": locs["ego"], "opp": locs["opp"]}
    return observation.copy(), total_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Epsilon schedule (unchanged)
# ---------------------------------------------------------------------------


def epsilon_for_episode(
    episode_index: int,
    num_episodes: int,
    decay_fraction: float,
    eps_start: float,
    eps_end: float,
) -> float:
    decay_episodes = max(1, int(num_episodes * decay_fraction))
    return eps_end + (eps_start - eps_end) * math.exp(-episode_index / (decay_episodes / 3))


# ---------------------------------------------------------------------------
# Network helpers (unchanged)
# ---------------------------------------------------------------------------


def _conv_out_size(
    h: int, w: int, n_layers: int = 3, kernel_size: int = 5, stride: int = 2
) -> tuple[int, int]:
    for _ in range(n_layers):
        h = (h - kernel_size) // stride + 1
        w = (w - kernel_size) // stride + 1
    return h, w


def _resize_to_preprocess(arr: np.ndarray) -> np.ndarray:
    h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Dueling DQN network (unchanged)
# ---------------------------------------------------------------------------


class DuelingDQN(torch.nn.Module):
    """Dueling CNN for (1, H, W) class-map input; outputs Q-values per action."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
        h_out, w_out = _conv_out_size(h, w)
        flat_size = 128 * h_out * w_out

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2)

        self.val1 = torch.nn.Linear(flat_size, 256)
        self.val2 = torch.nn.Linear(256, 1)

        self.adv1 = torch.nn.Linear(flat_size, 256)
        self.adv2 = torch.nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        v = torch.nn.functional.relu(self.val1(x))
        v = self.val2(v)

        a = torch.nn.functional.relu(self.adv1(x))
        a = self.adv2(a)

        return v + a - a.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Uniform Replay Memory (replaces PrioritizedReplayMemory)
# ---------------------------------------------------------------------------


class UniformReplayMemory:
    """Simple uniform experience replay buffer using a circular deque."""

    def __init__(self, capacity: int) -> None:
        self._buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Run metadata helper
# ---------------------------------------------------------------------------


def _get_run_metadata() -> dict:
    repo = Repo(".", search_parent_directories=True)
    return {
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": repo.head.commit.hexsha,
        "git_branch": repo.active_branch.name,
    }


# ---------------------------------------------------------------------------
# D3QN Trainer (exp3)
# ---------------------------------------------------------------------------


class D3QNTrainer:
    """D3QN trainer: Dueling + Double DQN, uniform replay, LR=1e-4, update every 4 steps."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Surround-v5",
            obs_type="grayscale",
            full_action_space=False,
            difficulty=constants.DIFFICULTY,
            mode=constants.MODE,
            frameskip=constants.FRAME_SKIP,
        )
        self.n_actions = self.env.action_space.n - 1  # ignore NOOP

        self.policy_net = DuelingDQN(self.n_actions).to(self.device)
        self.target_net = DuelingDQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=constants.D3QN_EXP3_LR, amsgrad=True
        )

        self.memory = UniformReplayMemory(capacity=constants.D3QN_MEMORY_CAPACITY)

        self.best_steps_survived = 0
        self._total_env_steps = 0
        self._run_metadata = _get_run_metadata()

        if constants.D3QN_EXP3_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.D3QN_EXP3_LOG_DIR}."
                " Remove it before a fresh run."
            )
        print(f"Saving checkpoint and run information to {constants.D3QN_EXP3_LOG_DIR}")
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        self.writer = SummaryWriter(log_dir=str(constants.D3QN_EXP3_LOG_DIR))
        self.writer.add_custom_scalars(
            {
                "episode/steps_survived_by_outcome": {
                    "steps_survived": [
                        "Multiline",
                        [
                            "episode/steps_survived_win",
                            "episode/steps_survived_loss",
                        ],
                    ]
                }
            }
        )

    # -- observation preprocessing ------------------------------------------

    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        """Grayscale observation -> (H, W) uint8 4-class map at 80x80."""
        class_map = observation_to_class_map(observation)
        return _resize_to_preprocess(class_map)

    def _to_tensor(self, preprocessed: np.ndarray) -> torch.Tensor:
        """(H, W) uint8 -> (1, 1, H, W) float32 tensor on device, normalized to [0, 1]."""
        arr = np.asarray(preprocessed, dtype=np.float32) / 3.0
        return torch.from_numpy(arr).to(self.device).unsqueeze(0).unsqueeze(0)

    # -- action selection ---------------------------------------------------

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        if random.random() > self._current_epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        return torch.tensor(
            [[random.randrange(self.n_actions)]],
            device=self.device,
            dtype=torch.long,
        )

    # -- Double DQN optimization with uniform replay ------------------------

    def _optimize_model(self) -> dict[str, float] | None:
        if len(self.memory) < constants.D3QN_EXP3_LEARNING_STARTS:
            return None

        transitions = self.memory.sample(constants.D3QN_BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN target: policy net selects action, target net evaluates
        next_state_values = torch.zeros(constants.D3QN_BATCH_SIZE, device=self.device)
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            with torch.no_grad():
                best_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
                )

        expected = (next_state_values * constants.D3QN_GAMMA) + reward_batch

        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected.unsqueeze(1))

        with torch.no_grad():
            td_errors = (expected.unsqueeze(1) - state_action_values).abs().squeeze(1).cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            q_mean = self.policy_net(state_batch).max(1).values.mean().item()

        return {
            "loss": loss.item(),
            "td_error": float(td_errors.mean()),
            "q_mean": q_mean,
            "grad_norm": grad_norm.item(),
        }

    # -- soft target update ------------------------------------------------

    def _soft_update_target(self) -> None:
        target = self.target_net.state_dict()
        policy = self.policy_net.state_dict()
        for key in policy:
            target[key] = policy[key] * constants.D3QN_TAU + target[key] * (1 - constants.D3QN_TAU)
        self.target_net.load_state_dict(target)

    # -- checkpointing -----------------------------------------------------

    def _save_checkpoint(self, episode_index: int, steps_survived: int | None = None) -> None:
        ep = episode_index + 1
        meta = {
            **self._run_metadata,
            "episodes_completed": ep,
            "algorithm": "d3qn_exp3",
            "dqn_state_type": "class_map",
        }
        save_checkpoint(
            constants.D3QN_EXP3_POLICY_NET_LATEST,
            self.policy_net.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_steps_survived": self.best_steps_survived}
            if self.best_steps_survived > 0
            else meta
        )
        constants.D3QN_EXP3_CHECKPOINT_METADATA.write_text(
            json.dumps(json_meta, indent=2), encoding="utf-8"
        )
        if ep % constants.D3QN_EXP3_CHECKPOINT_INTERVAL == 0:
            path = constants.D3QN_EXP3_CHECKPOINT_DIR / f"policy_net_{ep:04d}.pt"
            save_checkpoint(
                path, self.policy_net.state_dict(), steps_survived=steps_survived, **meta
            )

    # -- main training loop ------------------------------------------------

    def run(self) -> None:
        for episode_index in trange(constants.D3QN_NUM_EPISODES):
            self._current_epsilon = epsilon_for_episode(
                episode_index,
                constants.D3QN_NUM_EPISODES,
                constants.D3QN_EPS_DECAY_FRACTION,
                constants.D3QN_EPS_START,
                constants.D3QN_EPS_END,
            )

            observation, _info = self.env.reset()
            last_pos = {
                "ego": get_location(observation)["ego"],
                "opp": get_location(observation)["opp"],
            }
            state = self._to_tensor(self._preprocess(observation))

            terminal_reward = 0.0
            episode_start_time = time.perf_counter()
            episode_losses: list[float] = []
            episode_td_errors: list[float] = []
            episode_q_means: list[float] = []
            episode_grad_norms: list[float] = []

            for t in trange(constants.MAX_CYCLES, leave=False):
                action = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = _step_until_new_frame(
                    self.env, last_pos, action_id
                )
                last_pos = _info["location"]
                reward_t = torch.tensor([reward], device=self.device)
                done = terminated or truncated or abs(reward) == 1

                if terminated or abs(reward) == 1:
                    next_state = None
                else:
                    next_state = self._to_tensor(self._preprocess(observation))

                self.memory.add(Transition(state, action, next_state, reward_t))
                state = next_state
                self._total_env_steps += 1

                # FIX: optimize every UPDATE_EVERY env steps, not every step
                if self._total_env_steps % constants.D3QN_EXP3_UPDATE_EVERY == 0:
                    metrics = self._optimize_model()
                    if metrics is not None:
                        episode_losses.append(metrics["loss"])
                        episode_td_errors.append(metrics["td_error"])
                        episode_q_means.append(metrics["q_mean"])
                        episode_grad_norms.append(metrics["grad_norm"])
                    self._soft_update_target()

                if done:
                    terminal_reward = float(reward)
                    steps_survived = t + 1
                    elapsed = time.perf_counter() - episode_start_time
                    sps = steps_survived / elapsed if elapsed > 0 else 0.0

                    self.writer.add_scalar("episode/steps_per_second", sps, episode_index)
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
                    self.writer.add_scalar("episode/epsilon", self._current_epsilon, episode_index)

                    if episode_losses:
                        n = len(episode_losses)
                        self.writer.add_scalar(
                            "episode/mean_huber_loss",
                            sum(episode_losses) / n,
                            episode_index,
                        )
                        self.writer.add_scalar(
                            "episode/mean_td_error",
                            sum(episode_td_errors) / n,
                            episode_index,
                        )
                        self.writer.add_scalar(
                            "episode/mean_q",
                            sum(episode_q_means) / n,
                            episode_index,
                        )
                        self.writer.add_scalar(
                            "episode/mean_grad_norm",
                            sum(episode_grad_norms) / n,
                            episode_index,
                        )

                    if steps_survived > self.best_steps_survived:
                        self.best_steps_survived = steps_survived
                        save_checkpoint(
                            constants.D3QN_EXP3_POLICY_NET_BEST,
                            self.policy_net.state_dict(),
                            steps_survived=steps_survived,
                            **{
                                **self._run_metadata,
                                "episodes_completed": episode_index + 1,
                                "algorithm": "d3qn_exp3",
                                "dqn_state_type": "class_map",
                            },
                        )
                    self._save_checkpoint(episode_index, steps_survived=steps_survived)
                    break

        self.writer.close()
        print("Training complete!")


if __name__ == "__main__":
    D3QNTrainer().run()
