"""Train a D3QN agent for Surround via self-play against a pool of past checkpoints.

Architecture:
  - Learner: DuelingDQN (same arch as d3qn/train_d3qn.py) trained as first_0
  - Opponent: frozen copy of a checkpoint sampled from the pool, acting as second_0
  - Fictitious self-play: current policy is snapshotted into the pool every
    D3QN_SELFPLAY_POOL_ADD_EVERY episodes
  - Opponent is resampled from the pool every D3QN_SELFPLAY_OPPONENT_CHANGE_EVERY episodes

Observation preprocessing:
  - Both players receive the same global grayscale frame.
  - first_0 uses observation_to_class_map (ego=EGO_GRAY, opp=OPP_GRAY).
  - second_0's preprocessor swaps the labels so its own trail reads as "ego".

Log dir: runs/surround/d3qn_selfplay/exp1/
"""

import collections
import json
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from git import Repo
from tqdm import trange

from surround.conf import constants
from surround.d3qn.train_d3qn import (
    DuelingDQN,
    NStepBuffer,
    Transition,
    UniformReplayMemory,
    epsilon_for_episode,
)
from surround.d3qn_selfplay.opponent_pool import OpponentPool
from surround.utils.callbacks import TBMetricsCallback, make_tb_writer
from surround.utils.checkpoint import load_checkpoint, save_checkpoint
from surround.utils.video_extract_locations import (
    EGO_GRAY,
    OPP_GRAY,
    WALLS_GRAY,
    get_location,
    observation_to_class_map,
)

# ---------------------------------------------------------------------------
# Second-player preprocessing
# ---------------------------------------------------------------------------


def _observation_to_class_map_second(observation: np.ndarray) -> np.ndarray:
    """Class map for second_0: swap ego and opponent intensity labels.

    The raw ALE frame always uses EGO_GRAY=179 for player 1's trail and
    OPP_GRAY=110 for player 2's trail. When second_0's network processes its
    observation it should see its own trail as class 3 (ego) and player 1's
    trail as class 2 (opponent), so the encoding is consistent with how all
    checkpoints in the pool were trained.
    """
    assert observation.ndim == 2, "Observation must be grayscale (H, W)."
    game = observation[constants.GAME_ROW_SLICE, constants.GAME_COL_SLICE]
    out = np.zeros(game.shape, dtype=np.uint8)
    out[game == WALLS_GRAY] = 1
    out[game == EGO_GRAY] = 2  # player 1's trail -> opponent for second_0
    out[game == OPP_GRAY] = 3  # player 2's trail -> ego for second_0
    return out.astype(np.float32) / 3.0


def _resize_to_preprocess(arr: np.ndarray) -> np.ndarray:
    h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Parallel env stepping helper
# ---------------------------------------------------------------------------


def _step_until_new_frame_parallel(
    env,
    last_pos: dict,
    action_first: int,
    action_second: int,
    max_substeps: int = 20,
) -> tuple[dict, float, dict, dict, dict]:
    """Step parallel env until both player positions change (or episode ends).

    Mirrors _step_until_new_frame from d3qn/train_d3qn.py for the PettingZoo
    parallel API. Both agents repeat their chosen actions on each substep.
    Only the learner's (first_0) reward is accumulated and returned.

    Returns (obs, learner_reward, terminations, truncations, info) where
    info["location"] holds the updated {"ego", "opp"} position dict derived
    from first_0's global frame.
    """
    total_reward = 0.0
    obs: dict = {}
    terminations: dict = {}
    truncations: dict = {}
    locs: dict = {"ego": None, "opp": None}

    for _ in range(max_substeps):
        actions = {}
        if "first_0" in env.agents:
            actions["first_0"] = action_first
        if "second_0" in env.agents:
            actions["second_0"] = action_second

        obs, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += rewards.get("first_0", 0.0)

        # Both players share the same global frame; use first_0's view for positions.
        frame = obs.get("first_0", obs.get("second_0"))
        if frame is not None:
            locs = get_location(frame.squeeze(-1))

        if locs["ego"] is None or locs["opp"] is None:
            continue

        done = (
            terminations.get("first_0", False)
            or truncations.get("first_0", False)
            or abs(total_reward) == 1
            or not env.agents
        )
        if done:
            break

        if (
            last_pos.get("ego") is None
            or last_pos.get("opp") is None
            or locs["ego"] != last_pos["ego"]
            or locs["opp"] != last_pos["opp"]
        ):
            break

    info = {"location": {"ego": locs["ego"], "opp": locs["opp"]}}
    return obs, total_reward, terminations, truncations, info


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
# Self-play trainer
# ---------------------------------------------------------------------------


class SelfPlayTrainer:
    """D3QN self-play trainer.

    Learner (first_0) is trained via D3QN with n-step returns and uniform
    replay. Opponent (second_0) is a frozen network whose weights are drawn
    from OpponentPool and periodically refreshed.
    """

    def __init__(self) -> None:
        try:
            from pettingzoo.atari import surround_v2
        except ImportError as e:
            raise ImportError(
                "pettingzoo[atari] is required for self-play training. "
                "Run: pip install 'pettingzoo[atari]'"
            ) from e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = surround_v2.parallel_env(
            obs_type="grayscale_image",
            full_action_space=False,
            auto_rom_install_path=str(constants.ROM_PATH),
        )
        self.env.reset()
        self.n_actions = constants.N_ACTIONS  # 4, consistent with existing checkpoints

        self.policy_net = DuelingDQN(self.n_actions).to(self.device)
        self.target_net = DuelingDQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.opponent_net = DuelingDQN(self.n_actions).to(self.device)
        self.opponent_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=constants.D3QN_LR, amsgrad=True
        )
        self.memory = UniformReplayMemory(capacity=constants.D3QN_MEMORY_CAPACITY)
        self.n_step_buf = NStepBuffer(n=constants.D3QN_N_STEP, gamma=constants.GAMMA)

        self.pool = OpponentPool(
            scan_dirs=[constants.D3QN_SELFPLAY_POOL_SCAN_DIR],
            min_steps=constants.D3QN_SELFPLAY_POOL_MIN_STEPS,
            device=self.device,
            pool_save_dir=constants.D3QN_SELFPLAY_LOG_DIR / "pool",
        )

        self._recent_outcomes: collections.deque = collections.deque(maxlen=100)
        self.best_win_rate = 0.0
        self._total_env_steps = 0
        self._run_metadata = _get_run_metadata()
        self._opponent_episodes_remaining = 0  # trigger resample on first episode

        if constants.D3QN_SELFPLAY_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.D3QN_SELFPLAY_LOG_DIR}. "
                "Remove it before a fresh run."
            )
        print(f"Saving checkpoint and run information to {constants.D3QN_SELFPLAY_LOG_DIR}")
        self.writer = make_tb_writer(constants.D3QN_SELFPLAY_LOG_DIR)
        self.cb = TBMetricsCallback(self.writer, self.n_actions)

    # -- preprocessing ------------------------------------------------------

    def _to_tensor(self, class_map: np.ndarray) -> torch.Tensor:
        resized = _resize_to_preprocess(class_map)
        arr = np.asarray(resized, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device).unsqueeze(0).unsqueeze(0)

    def _preprocess_first(self, obs: np.ndarray) -> torch.Tensor:
        return self._to_tensor(observation_to_class_map(obs.squeeze(-1)))

    def _preprocess_second(self, obs: np.ndarray) -> torch.Tensor:
        return self._to_tensor(_observation_to_class_map_second(obs.squeeze(-1)))

    # -- action selection ---------------------------------------------------

    def _select_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # -- opponent management ------------------------------------------------

    def _resample_opponent(self) -> None:
        state_dict = self.pool.sample()
        if state_dict is not None:
            self.opponent_net.load_state_dict(state_dict)
        # If pool returns None (shouldn't happen given pool size), keep current
        # opponent_net weights (random at first episode, then whatever was last set).
        self.opponent_net.eval()
        self._opponent_episodes_remaining = constants.D3QN_SELFPLAY_OPPONENT_CHANGE_EVERY

    # -- Double DQN optimization --------------------------------------------

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
            "algorithm": "d3qn_selfplay",
            "dqn_state_type": "class_map",
        }
        save_checkpoint(
            constants.D3QN_SELFPLAY_CKPT.latest,
            self.policy_net.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_win_rate": self.best_win_rate} if self.best_win_rate > 0 else meta
        )
        constants.D3QN_SELFPLAY_CKPT.metadata.write_text(
            json.dumps(json_meta, indent=2), encoding="utf-8"
        )
        if ep % constants.CHECKPOINT_INTERVAL == 0:
            path = constants.D3QN_SELFPLAY_CKPT.dir / f"policy_net_{ep:04d}.pt"
            save_checkpoint(
                path, self.policy_net.state_dict(), steps_survived=steps_survived, **meta
            )

    # -- main training loop -------------------------------------------------

    def run(self) -> None:
        for episode_index in trange(constants.NUM_EPISODES):
            self._current_epsilon = epsilon_for_episode(
                episode_index,
                constants.NUM_EPISODES,
                constants.D3QN_EPS_DECAY_FRACTION,
                constants.EPS_START,
                constants.EPS_END,
            )

            # Resample opponent from pool when due
            if self._opponent_episodes_remaining <= 0:
                self._resample_opponent()
            self._opponent_episodes_remaining -= 1

            obs, _ = self.env.reset()
            first_frame = obs["first_0"].squeeze(-1)
            last_pos = {
                "ego": get_location(first_frame)["ego"],
                "opp": get_location(first_frame)["opp"],
            }
            learner_state = self._preprocess_first(obs["first_0"])
            opp_state = self._preprocess_second(obs["second_0"])

            terminal_reward = 0.0
            episode_start_time = time.perf_counter()
            episode_losses: list[float] = []
            episode_td_errors: list[float] = []
            episode_q_means: list[float] = []
            episode_grad_norms: list[float] = []

            for t in trange(constants.MAX_CYCLES, leave=False):
                # Learner: epsilon-greedy
                action, q_values, v_stream, adv_values = self._select_action(learner_state)
                learner_action_id = action.item() + 1  # skip NOOP (action 0)

                # Opponent: greedy from frozen net
                with torch.no_grad():
                    opp_action_id = self.opponent_net(opp_state).max(1).indices.item() + 1

                obs, learner_reward, terminations, truncations, _info = (
                    _step_until_new_frame_parallel(
                        self.env, last_pos, learner_action_id, opp_action_id
                    )
                )
                last_pos = _info["location"]

                learner_terminated = terminations.get("first_0", False)
                done = (
                    learner_terminated
                    or truncations.get("first_0", False)
                    or abs(learner_reward) == 1
                    or not self.env.agents
                )

                if learner_terminated or abs(learner_reward) == 1 or not self.env.agents:
                    next_learner_state = None
                else:
                    next_learner_state = self._preprocess_first(obs["first_0"])

                for t_ready in self.n_step_buf.push(
                    learner_state, action, float(learner_reward), next_learner_state, flush=done
                ):
                    self.memory.add(t_ready)

                self.cb.on_step(
                    action.item(), q_values, value_stream=v_stream, adv_values=adv_values
                )
                learner_state = next_learner_state
                if "second_0" in obs:
                    opp_state = self._preprocess_second(obs["second_0"])
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
                    terminal_reward = float(learner_reward)
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
                        episode_index,
                        steps_survived,
                        terminal_reward,
                        self._current_epsilon,
                        sps,
                        avg_train_metrics,
                    )

                    self._recent_outcomes.append(terminal_reward > 0)
                    win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
                    self.writer.add_scalar("selfplay/pool_size", self.pool.size, episode_index)

                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        save_checkpoint(
                            constants.D3QN_SELFPLAY_CKPT.best,
                            self.policy_net.state_dict(),
                            steps_survived=steps_survived,
                            **{
                                **self._run_metadata,
                                "episodes_completed": episode_index + 1,
                                "algorithm": "d3qn_selfplay",
                                "dqn_state_type": "class_map",
                            },
                        )
                    self._save_checkpoint(episode_index, steps_survived=steps_survived)

                    # Add current policy snapshot to pool
                    if (episode_index + 1) % constants.D3QN_SELFPLAY_POOL_ADD_EVERY == 0:
                        self.pool.add(
                            self.policy_net.state_dict(), episode_index + 1, steps_survived
                        )

                    break

        self.cb.on_train_end()
        self.env.close()
        print("Training complete!")


# ---------------------------------------------------------------------------
# Greedy policy for benchmark
# ---------------------------------------------------------------------------

_SELFPLAY_POLICY_NET_CACHE: DuelingDQN | None = None
_SELFPLAY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_selfplay_policy_net() -> DuelingDQN:
    global _SELFPLAY_POLICY_NET_CACHE
    if _SELFPLAY_POLICY_NET_CACHE is None:
        ckpt_path = constants.D3QN_SELFPLAY_CKPT.best
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Self-play checkpoint not found: {ckpt_path}. Run training first."
            )
        state_dict, _ = load_checkpoint(ckpt_path, map_location=_SELFPLAY_DEVICE)
        net = DuelingDQN(constants.N_ACTIONS).to(_SELFPLAY_DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        _SELFPLAY_POLICY_NET_CACHE = net
    return _SELFPLAY_POLICY_NET_CACHE


def greedy_selfplay_policy(action_space, observation, info, last_action):
    """Greedy policy using the latest saved self-play D3QN weights."""
    net = _load_selfplay_policy_net()
    class_map = observation_to_class_map(observation)
    class_map = _resize_to_preprocess(class_map)
    x = torch.from_numpy(class_map).to(_SELFPLAY_DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        action_index = int(net(x).max(1).indices.item())
    return action_index + 1


if __name__ == "__main__":
    SelfPlayTrainer().run()
