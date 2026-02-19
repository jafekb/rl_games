"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import json
import logging
import math
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from zoneinfo import ZoneInfo

import ale_py
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from git import Repo
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.conf import constants
from surround.conf.constants import GAME_COL_SLICE, GAME_ROW_SLICE
from surround.utils.checkpoint import load_checkpoint, save_checkpoint
from surround.utils.video_extract_locations import get_location

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def epsilon_for_episode(
    episode_index: int,
    num_episodes: int,
    decay_fraction: float,
    eps_start: float,
    eps_end: float,
) -> float:
    """Epsilon for the given episode (episode-fraction-based decay).

    Decay is exponential over the first decay_fraction of num_episodes,
    so the schedule scales with run length. By the end of the decay window
    epsilon is ~95% of the way from eps_start to eps_end.
    """
    decay_episodes = max(1, int(num_episodes * decay_fraction))
    return eps_end + (eps_start - eps_end) * math.exp(-episode_index / (decay_episodes / 3))


def _conv_out_size(
    h: int, w: int, n_layers: int = 3, kernel_size: int = 5, stride: int = 2
) -> tuple[int, int]:
    for _ in range(n_layers):
        h = (h - kernel_size) // stride + 1
        w = (w - kernel_size) // stride + 1
    return h, w


def _get_run_metadata() -> dict:
    """Return dict with git_commit, git_branch, timestamp for run metadata."""
    repo = Repo(".", search_parent_directories=True)
    return {
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": repo.head.commit.hexsha,
        "git_branch": repo.active_branch.name,
    }


def get_state_tuple(locations: dict, last_action: int) -> tuple[int, ...]:
    """Build state tuple (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)."""
    if locations["ego"] is None or locations["opp"] is None:
        return (1, 1, 1, 1, 1, 1, last_action)
    ego_row, ego_col = locations["ego"]
    opp_row, opp_col = locations["opp"]
    wall_set = locations["walls"]
    collisions = (
        wall_set | {(opp_row, opp_col)} if opp_row is not None and opp_col is not None else wall_set
    )

    d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
    d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= constants.GRID_COLS - 1 else 0
    d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
    d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= constants.GRID_ROWS - 1 else 0

    rel_x = 0 if opp_col < ego_col else (2 if opp_col > ego_col else 1)
    rel_y = 0 if opp_row < ego_row else (2 if opp_row > ego_row else 1)

    return (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)


def get_state_from_observation(observation: np.ndarray, last_action: int) -> tuple[int, ...]:
    """Build state tuple from grayscale observation (for policy / benchmark)."""
    if constants.STATE_MODE == "ram":
        raise ValueError("RAM state mode is not supported.")
    locations = get_location(observation)
    return get_state_tuple(locations, last_action)


def _make_dqn_net(n_actions: int, device: torch.device) -> torch.nn.Module:
    """Return policy/target net for current DQN_STATE_TYPE."""
    if constants.DQN_STATE_TYPE == "state_tuple":
        return DqnMlp(n_actions).to(device)
    if constants.DQN_STATE_TYPE == "grayscale":
        return DQN(n_actions).to(device)
    raise ValueError(f"Unknown DQN_STATE_TYPE: {constants.DQN_STATE_TYPE}")


class DqnMlp(torch.nn.Module):
    """MLP that takes 7-tuple state (1, 7) and outputs Q-values for each action."""

    def __init__(self, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = torch.nn.Linear(7, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQN(torch.nn.Module):
    """CNN for (1, H, W) grayscale or 4-class map; outputs Q-values per action."""

    def __init__(self, n_actions: int):
        super().__init__()
        h, w = constants.DQN_GAME_HEIGHT, constants.DQN_GAME_WIDTH
        h_out, w_out = _conv_out_size(h, w)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.flat_size = 128 * h_out * w_out
        self.fc1 = torch.nn.Linear(self.flat_size, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(-1, self.flat_size)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNTrainer:
    """Single class owning env, networks, replay memory, and training loop."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Surround-v5",
            obs_type="grayscale",
            full_action_space=False,
            difficulty=constants.DIFFICULTY,
            mode=constants.MODE,
            frameskip=constants.DQN_FRAME_SKIP,
        )
        self.n_actions = self.env.action_space.n - 1  # ignore NOOP
        self.state_type = constants.DQN_STATE_TYPE
        self.policy_net = _make_dqn_net(self.n_actions, self.device)
        self.target_net = _make_dqn_net(self.n_actions, self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=constants.LR, amsgrad=True
        )
        self.memory: deque = deque(maxlen=constants.MEMORY_CAPACITY)
        self.episode_durations: list[int] = []
        self.best_steps_survived = 0
        if constants.DQN_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.DQN_LOG_DIR}. Remove it before a fresh run."
            )
        self._run_metadata = _get_run_metadata()
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
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

    def _preprocess_observation(
        self, observation: np.ndarray, last_action: int = 1
    ) -> np.ndarray | tuple[int, ...]:
        """Preprocess observation for the current state type.

        - state_tuple: returns 7-tuple (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action).
        - grayscale: returns (H, W) float array in [0, 1] (cropped and resized to DQN_GAME_*).
        """
        if self.state_type == "state_tuple":
            return get_state_from_observation(observation, last_action)
        game = observation[GAME_ROW_SLICE, GAME_COL_SLICE]
        return (game.astype(np.float32) / 255.0).copy()

    def _observation_to_tensor(self, preprocessed: np.ndarray | tuple[int, ...]) -> torch.Tensor:
        """Convert preprocessed state to (1, ...) tensor for the net."""
        if self.state_type == "state_tuple":
            arr = np.array(preprocessed, dtype=np.float32)
            x = torch.from_numpy(arr).to(self.device)
            return x.unsqueeze(0)
        # grayscale: (H, W) -> (1, 1, H, W)
        x = torch.from_numpy(preprocessed).to(torch.float32).to(self.device)
        return x.unsqueeze(0).unsqueeze(0)

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        if sample > self._current_epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        return torch.tensor(
            [[random.randrange(self.n_actions)]],
            device=self.device,
            dtype=torch.long,
        )

    def _optimize_model(self) -> dict[str, float] | None:
        if len(self.memory) < constants.BATCH_SIZE:
            return None

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

        td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).abs()
        mean_td = td_errors.mean().item()
        q_mean = self.policy_net(state_batch).max(1).values.mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), float("inf"))
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "td_error": mean_td,
            "q_mean": q_mean,
            "grad_norm": grad_norm.item(),
        }

    def _soft_update_target(self) -> None:
        target = self.target_net.state_dict()
        policy = self.policy_net.state_dict()
        for key in policy:
            target[key] = policy[key] * constants.TAU + target[key] * (1 - constants.TAU)
        self.target_net.load_state_dict(target)

    def _save_checkpoint(self, episode_index: int, steps_survived: int | None = None) -> None:
        ep = episode_index + 1
        meta = {
            **self._run_metadata,
            "episodes_completed": ep,
            "dqn_state_type": self.state_type,
        }
        save_checkpoint(
            constants.DQN_POLICY_NET_LATEST,
            self.policy_net.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_steps_survived": self.best_steps_survived}
            if self.best_steps_survived > 0
            else meta
        )
        constants.DQN_CHECKPOINT_METADATA.write_text(
            json.dumps(json_meta, indent=2), encoding="utf-8"
        )
        if ep % constants.DQN_CHECKPOINT_INTERVAL == 0:
            path = constants.DQN_CHECKPOINT_DIR / f"policy_net_{ep:04d}.pt"
            save_checkpoint(
                path, self.policy_net.state_dict(), steps_survived=steps_survived, **meta
            )

    def run(self) -> None:
        for episode_index in trange(constants.NUM_EPISODES):
            self._current_epsilon = epsilon_for_episode(
                episode_index,
                constants.NUM_EPISODES,
                constants.EPS_DECAY_FRACTION,
                constants.EPS_START,
                constants.EPS_END,
            )
            observation, _info = self.env.reset()
            video_writer = None
            if constants.VISUALIZE_EPISODES:
                observation = observation.copy()
                episodes_dir = constants.DQN_LOG_DIR / "episodes"
                episodes_dir.mkdir(parents=True, exist_ok=True)
                video_path = episodes_dir / f"episode_{episode_index:04d}.mp4"
                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=constants.DQN_EPISODE_VIDEO_FPS,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
                video_writer.append_data(observation)
            # For state_tuple we need last_action; use 1 as sentinel for first frame
            last_action = 1
            state = self._observation_to_tensor(
                self._preprocess_observation(observation, last_action)
            )
            terminal_reward = 0.0
            episode_start_time = time.perf_counter()
            episode_losses: list[float] = []
            episode_td_errors: list[float] = []
            episode_q_means: list[float] = []
            episode_grad_norms: list[float] = []

            for t in trange(constants.MAX_CYCLES, leave=False):
                action = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = self.env.step(action_id)
                if video_writer is not None:
                    observation = observation.copy()
                    video_writer.append_data(observation)
                reward_t = torch.tensor([reward], device=self.device)
                # Only train for 1 game, not the whole 10-point match.
                done = terminated or truncated or abs(reward) == 1

                if terminated:
                    next_state = None
                else:
                    next_state = self._observation_to_tensor(
                        self._preprocess_observation(observation, action_id)
                    )

                self.memory.append(Transition(state, action, next_state, reward_t))
                state = next_state
                last_action = action_id

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
                    self.episode_durations.append(steps_survived)
                    elapsed = time.perf_counter() - episode_start_time
                    steps_per_second = steps_survived / elapsed if elapsed > 0 else 0.0
                    self.writer.add_scalar(
                        "episode/steps_per_second", steps_per_second, episode_index
                    )
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
                            constants.DQN_POLICY_NET_BEST,
                            self.policy_net.state_dict(),
                            steps_survived=steps_survived,
                            **{
                                **self._run_metadata,
                                "episodes_completed": episode_index + 1,
                                "dqn_state_type": self.state_type,
                            },
                        )
                    self._save_checkpoint(episode_index, steps_survived=steps_survived)
                    break
            if video_writer is not None:
                video_writer.close()
        self.writer.close()
        print("Training complete!")


# Policy for benchmark: lazy-load policy net from latest checkpoint
_POLICY_NET_CACHE: DQN | DqnMlp | None = None
_POLICY_NET_STATE_TYPE: str | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_policy_net() -> torch.nn.Module:
    global _POLICY_NET_CACHE, _POLICY_NET_STATE_TYPE
    if _POLICY_NET_CACHE is None:
        if not constants.DQN_POLICY_NET_LATEST.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {constants.DQN_POLICY_NET_LATEST}. Run training first."
            )
        state_dict, metadata = load_checkpoint(
            constants.DQN_POLICY_NET_LATEST, map_location=_DEVICE
        )
        state_type = metadata.get("dqn_state_type") or constants.DQN_STATE_TYPE
        _POLICY_NET_STATE_TYPE = state_type
        if state_type == "state_tuple":
            _POLICY_NET_CACHE = DqnMlp(constants.N_ACTIONS).to(_DEVICE)
        elif state_type == "grayscale":
            _POLICY_NET_CACHE = DQN(constants.N_ACTIONS).to(_DEVICE)
        else:
            raise ValueError(f"Unknown dqn_state_type in checkpoint: {state_type}")
        _POLICY_NET_CACHE.load_state_dict(state_dict)
        _POLICY_NET_CACHE.eval()
    return _POLICY_NET_CACHE


def greedy_dqn_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved DQN weights (same signature as greedy_q_policy)."""
    net = _load_policy_net()
    state_type = _POLICY_NET_STATE_TYPE or constants.DQN_STATE_TYPE
    if state_type == "state_tuple":
        state_tuple = get_state_from_observation(observation, last_action)
        x = torch.from_numpy(np.array(state_tuple, dtype=np.float32)).to(_DEVICE).unsqueeze(0)
    else:
        game = observation[GAME_ROW_SLICE, GAME_COL_SLICE]
        game = game.astype(np.float32) / 255.0
        x = torch.from_numpy(game).to(torch.float32).to(_DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1  # env action 1..4


if __name__ == "__main__":
    DQNTrainer().run()
