"""Train a PPO agent for the Surround game."""

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
from git import Repo
from tensorboardX import SummaryWriter
from tqdm import trange

from surround.actions import ACTION_WORD_TO_ID
from surround.conf import constants
from surround.utils.checkpoint import save_checkpoint
from surround.utils.env_state import build_state_from_observation, make_env


def _get_run_metadata() -> dict:
    """Return dict with git_commit, git_branch, timestamp for run metadata."""
    repo = Repo(".", search_parent_directories=True)
    return {
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": repo.head.commit.hexsha,
        "git_branch": repo.active_branch.name,
    }


class ActorCritic(nn.Module):
    """MLP policy (actor) and value (critic) for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        probs = self.actor(state)
        value = self.critic(state)
        return torch.distributions.Categorical(probs), value


class PPO:
    """PPO agent: policy + optimizer and update from rollout memory."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        lr: float = constants.PPO_LR,
        gamma: float = constants.PPO_GAMMA,
        eps_clip: float = constants.PPO_EPS_CLIP,
        epochs: int = constants.PPO_EPOCHS,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.mse_loss = nn.MSELoss()

    def update(self, memory: dict) -> dict[str, float]:
        """Update policy from rollout memory. Returns dict of metrics for logging."""
        states = torch.tensor(np.array(memory["states"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(memory["actions"], device=self.device)
        old_log_probs = torch.stack(memory["log_probs"]).detach().to(self.device)
        rewards = memory["rewards"]
        is_terminals = memory["is_terminals"]

        returns: list[float] = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)

        with torch.no_grad():
            _, state_values_0 = self.policy(states)
            state_values_0 = state_values_0.squeeze()
        advantages = returns_t - state_values_0
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        loss_accum = 0.0
        entropy_accum = 0.0
        ratio_mean_accum = 0.0
        ratio_min_accum = 0.0
        ratio_max_accum = 0.0

        for _ in range(self.epochs):
            dist, state_values = self.policy(states)
            state_values = state_values.squeeze()
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * self.mse_loss(state_values, returns_t)
                - constants.PPO_ENTROPY_COEF * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            if constants.PPO_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), constants.PPO_GRAD_CLIP)
            self.optimizer.step()

            loss_accum += loss.item()
            entropy_accum += entropy.mean().item()
            ratio_mean_accum += ratio.mean().item()
            ratio_min_accum += ratio.min().item()
            ratio_max_accum += ratio.max().item()

        n = self.epochs
        return {
            "ppo/loss": loss_accum / n,
            "ppo/entropy": entropy_accum / n,
            "ppo/ratio_mean": ratio_mean_accum / n,
            "ppo/ratio_min": ratio_min_accum / n,
            "ppo/ratio_max": ratio_max_accum / n,
        }


class PPOTrainer:
    """Owns env, PPO agent, and training loop with checkpointing and TensorBoard."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = make_env(
            constants.DIFFICULTY,
            constants.MODE,
            frameskip=constants.DQN_FRAME_SKIP,
        )
        self.n_actions = self.env.action_space.n - 1
        self.agent = PPO(
            constants.STATE_TUPLE_DIM,
            self.n_actions,
            device=self.device,
        )
        self.best_steps_survived = 0
        self._num_updates = 0
        self._run_metadata = _get_run_metadata()

        if constants.PPO_LOG_DIR.exists():
            raise FileExistsError(
                f"Log dir already exists: {constants.PPO_LOG_DIR}. Remove it before a fresh run."
            )
        constants.PPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        self.writer = SummaryWriter(log_dir=str(constants.PPO_LOG_DIR))
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

    def _get_state(self, observation: np.ndarray, last_action: int) -> np.ndarray:
        return np.array(
            build_state_from_observation(
                observation,
                last_action,
                state_mode=constants.STATE_MODE,
                debug_state=constants.DEBUG_STATE,
            ),
            dtype=np.float32,
        )

    def _save_checkpoint(self, episode_index: int, steps_survived: int | None = None) -> None:
        ep = episode_index + 1
        meta = {**self._run_metadata, "episodes_completed": ep}
        save_checkpoint(
            constants.PPO_POLICY_LATEST,
            self.agent.policy.state_dict(),
            steps_survived=steps_survived,
            **meta,
        )
        json_meta = (
            {**meta, "best_steps_survived": self.best_steps_survived}
            if self.best_steps_survived > 0
            else meta
        )
        constants.PPO_CHECKPOINT_METADATA.write_text(
            json.dumps(json_meta, indent=2), encoding="utf-8"
        )
        if ep % constants.PPO_CHECKPOINT_INTERVAL == 0:
            path = constants.PPO_CHECKPOINT_DIR / f"policy_{ep:04d}.pt"
            save_checkpoint(
                path,
                self.agent.policy.state_dict(),
                steps_survived=steps_survived,
                **meta,
            )

    def run(self) -> None:
        memory: dict = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "is_terminals": [],
        }
        timestep = 0

        for episode_index in trange(constants.PPO_NUM_EPISODES):
            observation, _ = self.env.reset()
            last_action = ACTION_WORD_TO_ID["LEFT"]
            state = self._get_state(observation, last_action)
            ep_reward = 0.0
            terminal_reward = 0.0

            for t in trange(constants.MAX_CYCLES, leave=False):
                timestep += 1
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist, _ = self.agent.policy(state_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                action_id = action.item() + 1
                next_obs, reward, terminated, truncated, _ = self.env.step(action_id)
                next_state = self._get_state(next_obs, action_id)
                done = terminated or truncated or abs(reward) == 1
                # Dense reward for surviving (match Q-learning) so policy doesn't only see ±1 at end
                step_reward = reward + (constants.STEP_REWARD if not done else 0.0)

                memory["states"].append(state.copy())
                memory["actions"].append(action.item())
                memory["log_probs"].append(log_prob.cpu())
                memory["rewards"].append(step_reward)
                memory["is_terminals"].append(done)

                state = next_state
                last_action = action_id
                ep_reward += reward  # raw env reward for logging

                if timestep >= constants.PPO_UPDATE_TIMESTEP:
                    metrics = self.agent.update(memory)
                    for key, value in metrics.items():
                        self.writer.add_scalar(key, value, self._num_updates)
                    self._num_updates += 1
                    memory = {k: [] for k in memory}
                    timestep = 0

                if done:
                    terminal_reward = float(reward)
                    steps_survived = t + 1
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
                    if steps_survived > self.best_steps_survived:
                        self.best_steps_survived = steps_survived
                        save_checkpoint(
                            constants.PPO_POLICY_BEST,
                            self.agent.policy.state_dict(),
                            steps_survived=steps_survived,
                            **{**self._run_metadata, "episodes_completed": episode_index + 1},
                        )
                    self._save_checkpoint(episode_index, steps_survived=steps_survived)
                    break

        self.writer.close()
        self.env.close()
        print("Training complete!")


if __name__ == "__main__":
    PPOTrainer().run()
