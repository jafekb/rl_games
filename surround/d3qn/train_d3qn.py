"""Train a D3QN (Dueling Double DQN) agent for the Surround game.

Architecture:
  - DuelingDQN: 3x Conv2d (1->32->64->128, kernel=5, stride=2) + separate value/advantage heads
  - Double DQN target: policy net selects action, target net evaluates
  - Uniform replay buffer with n-step discounted returns (n=10)
  - Class-map input (4-class: empty/wall/opp/ego), soft target updates

Log dir: runs/surround/d3qn/d3qn/
"""

import collections
import json
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
from tqdm import trange

from surround.conf import constants
from surround.utils.callbacks import TBMetricsCallback, make_tb_writer
from surround.utils.checkpoint import CheckpointPaths, load_checkpoint, save_checkpoint
from surround.utils.video_extract_locations import get_location, observation_to_class_map

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# ---------------------------------------------------------------------------
# Env stepping helper
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
# Epsilon schedule
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
# Network helpers
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
# Dueling DQN network
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

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._backbone(x)
        v = self.val2(torch.nn.functional.relu(self.val1(x)))
        a = self.adv2(torch.nn.functional.relu(self.adv1(x)))
        return v + a - a.mean(dim=1, keepdim=True)

    def forward_with_streams(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (q, value_stream, advantage) for metrics logging."""
        x = self._backbone(x)
        v = self.val2(torch.nn.functional.relu(self.val1(x)))
        a = self.adv2(torch.nn.functional.relu(self.adv1(x)))
        return v + a - a.mean(dim=1, keepdim=True), v, a


# ---------------------------------------------------------------------------
# Uniform replay memory
# ---------------------------------------------------------------------------


class UniformReplayMemory:
    """Circular uniform experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# N-step return buffer
# ---------------------------------------------------------------------------


class NStepBuffer:
    """Accumulates transitions and computes n-step discounted returns.

    Each push() appends one transition and returns ready Transition objects.
    A transition is ready when n subsequent steps have been collected, or the
    episode ends (flush). The stored reward is the n-step discounted sum:
        R_n = r_t + gamma*r_{t+1} + ... + gamma^(k-1)*r_{t+k-1}
    where k = min(n, steps until episode end).
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self._buf: collections.deque = collections.deque()

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor | None,
        *,
        flush: bool = False,
    ) -> list[Transition]:
        self._buf.append((state, action, reward, next_state))
        ready: list[Transition] = []

        if len(self._buf) >= self.n:
            ready.append(self._pop_oldest())

        if next_state is None or flush:
            while self._buf:
                ready.append(self._pop_oldest())

        return ready

    def _pop_oldest(self) -> Transition:
        s0, a0, _, _ = self._buf[0]
        discounted_return = 0.0
        final_next: torch.Tensor | None = None
        for i, (_, _, r, ns) in enumerate(self._buf):
            discounted_return += (self.gamma**i) * r
            final_next = ns
            if ns is None:
                break
        self._buf.popleft()
        reward_t = torch.tensor([discounted_return], device=s0.device, dtype=torch.float32)
        return Transition(s0, a0, final_next, reward_t)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def _get_run_metadata() -> dict:
    repo = Repo(".", search_parent_directories=True)
    return {
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": repo.head.commit.hexsha,
        "git_branch": repo.active_branch.name,
    }


# ---------------------------------------------------------------------------
# D3QN Trainer
# ---------------------------------------------------------------------------


class D3QNTrainer:
    """D3QN trainer: Dueling Double DQN with uniform replay and n-step returns."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gym.register_envs(ale_py)
        self._random_difficulty: bool = getattr(constants, "D3QN_RANDOM_DIFFICULTY", False)
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

        self._episode_offset = 0
        resume_log_dir = constants.D3QN_RESUME_FROM
        if resume_log_dir is not None:
            resume_ckpt = CheckpointPaths(resume_log_dir).latest
            state_dict, meta = load_checkpoint(resume_ckpt, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self._episode_offset = int(meta.get("episodes_completed", 0))
            print(f"Resumed from {resume_ckpt} (episodes_completed={self._episode_offset})")
        # Fresh epsilon: restart exploration from ep 0 regardless of how many episodes
        # have already been trained. The loaded weights are kept; only epsilon is reset.
        fresh_eps = getattr(constants, "D3QN_FRESH_EPSILON", False)
        self._eps_offset = 0 if fresh_eps else self._episode_offset

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=constants.D3QN_LR, amsgrad=True
        )
        self.memory = UniformReplayMemory(capacity=constants.D3QN_MEMORY_CAPACITY)
        self.n_step_buf = NStepBuffer(n=constants.D3QN_N_STEP, gamma=constants.GAMMA)

        self._recent_outcomes: collections.deque = collections.deque(maxlen=100)
        self.best_win_rate = 0.0
        self._total_env_steps = 0
        self._run_metadata = _get_run_metadata()

        if constants.D3QN_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.D3QN_LOG_DIR}. Remove it before a fresh run."
            )
        print(f"Saving checkpoint and run information to {constants.D3QN_LOG_DIR}")
        self.writer = make_tb_writer(constants.D3QN_LOG_DIR)
        self.cb = TBMetricsCallback(self.writer, self.n_actions)

    # -- observation preprocessing ------------------------------------------

    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        """Grayscale observation -> (H, W) uint8 4-class map at 80x80."""
        class_map = observation_to_class_map(observation)
        return _resize_to_preprocess(class_map)

    def _to_tensor(self, preprocessed: np.ndarray) -> torch.Tensor:
        """(H, W) float32 -> (1, 1, H, W) float32 tensor on device."""
        arr = np.asarray(preprocessed, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device).unsqueeze(0).unsqueeze(0)

    # -- action selection ---------------------------------------------------

    def _select_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, q_values, value_stream, advantages)."""
        with torch.no_grad():
            q, v, a = self.policy_net.forward_with_streams(state)
        if random.random() > self._current_epsilon:
            return q.max(1).indices.view(1, 1), q, v, a
        return (
            torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
            ),
            q,
            v,
            a,
        )

    # -- Double DQN optimization with n-step bootstrap ----------------------

    def _optimize_model(self) -> dict[str, float] | None:
        if len(self.memory) < constants.D3QN_LEARNING_STARTS:
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

        # Double DQN: policy net selects action, target net evaluates.
        # Bootstrap coefficient is gamma^n (reward_batch already holds n-step return).
        next_state_values = torch.zeros(constants.D3QN_BATCH_SIZE, device=self.device)
        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            with torch.no_grad():
                best_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
                )

        gamma_n = constants.GAMMA**constants.D3QN_N_STEP
        expected = (next_state_values * gamma_n) + reward_batch
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

    # -- soft target update -------------------------------------------------

    def _soft_update_target(self) -> None:
        target = self.target_net.state_dict()
        policy = self.policy_net.state_dict()
        for key in policy:
            target[key] = policy[key] * constants.TAU + target[key] * (1 - constants.TAU)
        self.target_net.load_state_dict(target)

    # -- checkpointing ------------------------------------------------------

    def _save_checkpoint(self, episode_index: int, steps_survived: int | None = None) -> None:
        ep = episode_index + 1
        meta = {
            **self._run_metadata,
            "episodes_completed": ep,
            "algorithm": "d3qn",
            "dqn_state_type": "class_map",
        }
        save_checkpoint(
            constants.D3QN_CKPT.latest,
            self.policy_net.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_win_rate": self.best_win_rate} if self.best_win_rate > 0 else meta
        )
        constants.D3QN_CKPT.metadata.write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
        if ep % constants.CHECKPOINT_INTERVAL == 0:
            path = constants.D3QN_CKPT.dir / f"policy_net_{ep:04d}.pt"
            save_checkpoint(
                path, self.policy_net.state_dict(), steps_survived=steps_survived, **meta
            )

    # -- main training loop -------------------------------------------------

    def run(self) -> None:
        total_episodes = constants.NUM_EPISODES + self._eps_offset
        for episode_index in trange(constants.NUM_EPISODES):
            self._current_epsilon = epsilon_for_episode(
                episode_index + self._eps_offset,
                total_episodes,
                constants.D3QN_EPS_DECAY_FRACTION,
                constants.EPS_START,
                constants.EPS_END,
            )

            if self._random_difficulty:
                self.env.unwrapped.ale.setDifficulty(np.random.randint(0, 4))
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
                action, q_values, v_stream, adv_values = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = _step_until_new_frame(
                    self.env, last_pos, action_id
                )
                last_pos = _info["location"]
                done = terminated or truncated or abs(reward) == 1

                if terminated or abs(reward) == 1:
                    next_state = None
                else:
                    next_state = self._to_tensor(self._preprocess(observation))

                for t_ready in self.n_step_buf.push(
                    state, action, float(reward), next_state, flush=done
                ):
                    self.memory.add(t_ready)

                self.cb.on_step(
                    action.item(), q_values, value_stream=v_stream, adv_values=adv_values
                )
                state = next_state
                self._total_env_steps += 1

                if self._total_env_steps % constants.D3QN_UPDATE_EVERY == 0:
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
                    avg_train_metrics = None
                    if episode_losses:
                        n = len(episode_losses)
                        avg_train_metrics = {
                            "loss": sum(episode_losses) / n,
                            "td_error": sum(episode_td_errors) / n,
                            "q_mean": sum(episode_q_means) / n,
                            "grad_norm": sum(episode_grad_norms) / n,
                        }
                    self.cb.on_episode_end(
                        episode_index + self._episode_offset,
                        steps_survived,
                        terminal_reward,
                        self._current_epsilon,
                        sps,
                        avg_train_metrics,
                    )

                    self._recent_outcomes.append(terminal_reward > 0)
                    win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        save_checkpoint(
                            constants.D3QN_CKPT.best,
                            self.policy_net.state_dict(),
                            steps_survived=steps_survived,
                            **{
                                **self._run_metadata,
                                "episodes_completed": episode_index + self._episode_offset + 1,
                                "algorithm": "d3qn",
                                "dqn_state_type": "class_map",
                            },
                        )
                    self._save_checkpoint(
                        episode_index + self._episode_offset, steps_survived=steps_survived
                    )
                    break

        self.cb.on_train_end()
        print("Training complete!")


# ---------------------------------------------------------------------------
# Greedy policy for benchmark
# ---------------------------------------------------------------------------

_D3QN_POLICY_NET_CACHE: DuelingDQN | None = None
_D3QN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_d3qn_policy_net() -> DuelingDQN:
    global _D3QN_POLICY_NET_CACHE
    if _D3QN_POLICY_NET_CACHE is None:
        ckpt_path = constants.D3QN_CKPT.best
        if not ckpt_path.exists():
            raise FileNotFoundError(f"D3QN checkpoint not found: {ckpt_path}. Run training first.")
        state_dict, _ = load_checkpoint(ckpt_path, map_location=_D3QN_DEVICE)
        net = DuelingDQN(constants.N_ACTIONS).to(_D3QN_DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        _D3QN_POLICY_NET_CACHE = net
    return _D3QN_POLICY_NET_CACHE


def greedy_d3qn_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved D3QN weights."""
    net = _load_d3qn_policy_net()
    class_map = observation_to_class_map(observation)
    class_map = _resize_to_preprocess(class_map)
    x = torch.from_numpy(class_map).to(_D3QN_DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1  # env action 1..4


if __name__ == "__main__":
    D3QNTrainer().run()
