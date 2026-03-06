"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import json
import math
import random
import time
from collections import deque, namedtuple
from datetime import datetime
from zoneinfo import ZoneInfo

import ale_py
import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from git import Repo
from tqdm import trange

from surround.conf import constants
from surround.conf.constants import GAME_COL_SLICE, GAME_ROW_SLICE
from surround.utils.callbacks import TBMetricsCallback, make_tb_writer
from surround.utils.checkpoint import load_checkpoint, save_checkpoint
from surround.utils.env_state import step_until_new_frame
from surround.utils.video_extract_locations import get_location, observation_to_class_map

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


def _resize_to_preprocess(arr: np.ndarray, *, is_class_map: bool) -> np.ndarray:
    """Resize observation to DQN_PREPROCESS_HEIGHT x DQN_PREPROCESS_WIDTH."""
    h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
    if is_class_map:
        resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized
    # grayscale float [0, 1]
    resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


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
    if constants.DQN_STATE_TYPE in ("grayscale", "class_map"):
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
    """CNN for (1, H, W) grayscale or 4-class map; outputs Q-values per action.

    Input is always downsampled to DQN_PREPROCESS_HEIGHT x DQN_PREPROCESS_WIDTH (e.g. 80x80).
    """

    def __init__(self, n_actions: int):
        super().__init__()
        h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
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
            frameskip=constants.FRAME_SKIP,
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
        self._recent_outcomes: deque = deque(maxlen=100)
        self.best_win_rate = 0.0
        if constants.DQN_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.DQN_LOG_DIR}. Remove it before a fresh run."
            )
        print(f"Saving checkpoint and run information to {constants.DQN_LOG_DIR}")
        self._run_metadata = _get_run_metadata()
        self.writer = make_tb_writer(constants.DQN_LOG_DIR)
        self.cb = TBMetricsCallback(self.writer, self.n_actions)

    def _preprocess_observation(
        self, observation: np.ndarray, last_action: int = 1
    ) -> np.ndarray | tuple[int, ...]:
        """Preprocess observation for the current state type.

        - state_tuple: returns 7-tuple (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action).
        - grayscale: returns (H, W) float array in [0, 1] (cropped to game, then resized to 80x80).
        - class_map: returns (H, W) uint8 4-class map resized to 80x80
          (0=empty, 1=wall, 2=opp, 3=ego).
        """
        if self.state_type == "state_tuple":
            return get_state_from_observation(observation, last_action)
        if self.state_type == "class_map":
            class_map = observation_to_class_map(observation)
            return _resize_to_preprocess(class_map, is_class_map=True)
        game = observation[GAME_ROW_SLICE, GAME_COL_SLICE]
        game = (game.astype(np.float32) / 255.0).copy()
        return _resize_to_preprocess(game, is_class_map=False)

    def _observation_to_tensor(self, preprocessed: np.ndarray | tuple[int, ...]) -> torch.Tensor:
        """Convert preprocessed state to (1, ...) tensor for the net."""
        if self.state_type == "state_tuple":
            arr = np.array(preprocessed, dtype=np.float32)
            x = torch.from_numpy(arr).to(self.device)
            return x.unsqueeze(0)
        # grayscale or class_map: (H, W) -> (1, 1, H, W)
        arr = np.asarray(preprocessed, dtype=np.float32)
        x = torch.from_numpy(arr).to(self.device)
        return x.unsqueeze(0).unsqueeze(0)

    def _select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, q_values). Q-values are always computed for metrics."""
        with torch.no_grad():
            q_values = self.policy_net(state)
        if random.random() > self._current_epsilon:
            return q_values.max(1).indices.view(1, 1), q_values
        return (
            torch.tensor(
                [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
            ),
            q_values,
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
        expected_state_action_values = (next_state_values * constants.GAMMA) + reward_batch

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
            constants.DQN_CKPT.latest,
            self.policy_net.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_win_rate": self.best_win_rate} if self.best_win_rate > 0 else meta
        )
        constants.DQN_CKPT.metadata.write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
        if ep % constants.CHECKPOINT_INTERVAL == 0:
            path = constants.DQN_CKPT.dir / f"policy_net_{ep:04d}.pt"
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
            last_pos = {
                "ego": get_location(observation)["ego"],
                "opp": get_location(observation)["opp"],
            }
            video_writer = None
            if constants.DQN_VISUALIZE_EPISODES:
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
                action, q_values = self._select_action(state)
                action_id = action.item() + 1  # env expects 1..4 (no NOOP)
                observation, reward, terminated, truncated, _info = step_until_new_frame(
                    self.env, last_pos, action_id
                )
                last_pos = _info["location"]
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
                self.cb.on_step(action.item(), q_values)
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
                        episode_index,
                        steps_survived,
                        terminal_reward,
                        self._current_epsilon,
                        steps_per_second,
                        avg_train_metrics,
                    )
                    self._recent_outcomes.append(terminal_reward > 0)
                    win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        save_checkpoint(
                            constants.DQN_CKPT.best,
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
        self.cb.on_train_end()
        print("Training complete!")


# Policy for benchmark: lazy-load policy net from latest checkpoint
_POLICY_NET_CACHE: DQN | DqnMlp | None = None
_POLICY_NET_STATE_TYPE: str | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_policy_net() -> torch.nn.Module:
    global _POLICY_NET_CACHE, _POLICY_NET_STATE_TYPE
    if _POLICY_NET_CACHE is None:
        if not constants.DQN_CKPT.best.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {constants.DQN_CKPT.best}. Run training first."
            )
        state_dict, metadata = load_checkpoint(constants.DQN_CKPT.best, map_location=_DEVICE)
        if "dqn_state_type" not in metadata:
            raise ValueError("Checkpoint missing dqn_state_type; re-save from current trainer.")
        state_type = metadata["dqn_state_type"]
        _POLICY_NET_STATE_TYPE = state_type
        if state_type == "state_tuple":
            _POLICY_NET_CACHE = DqnMlp(constants.N_ACTIONS).to(_DEVICE)
        elif state_type in ("grayscale", "class_map"):
            _POLICY_NET_CACHE = DQN(constants.N_ACTIONS).to(_DEVICE)
        else:
            raise ValueError(f"Unknown dqn_state_type in checkpoint: {state_type}")
        _POLICY_NET_CACHE.load_state_dict(state_dict)
        _POLICY_NET_CACHE.eval()
    return _POLICY_NET_CACHE


def greedy_dqn_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved DQN weights (same signature as greedy_q_policy)."""
    net = _load_policy_net()
    state_type = _POLICY_NET_STATE_TYPE
    if state_type == "state_tuple":
        state_tuple = get_state_from_observation(observation, last_action)
        x = torch.from_numpy(np.array(state_tuple, dtype=np.float32)).to(_DEVICE).unsqueeze(0)
    elif state_type == "class_map":
        class_map = observation_to_class_map(observation)
        class_map = _resize_to_preprocess(class_map, is_class_map=True)
        x = torch.from_numpy(class_map.astype(np.float32)).to(_DEVICE).unsqueeze(0).unsqueeze(0)
    else:
        game = observation[GAME_ROW_SLICE, GAME_COL_SLICE]
        game = game.astype(np.float32) / 255.0
        game = _resize_to_preprocess(game, is_class_map=False)
        x = torch.from_numpy(game).to(torch.float32).to(_DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1  # env action 1..4


if __name__ == "__main__":
    DQNTrainer().run()
