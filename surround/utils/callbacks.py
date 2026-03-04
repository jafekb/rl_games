"""Training callbacks for RL agents."""

import collections
import logging
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter


class TrainingCallback:
    """Base class for training callbacks. Override the hooks you need."""

    def on_train_start(self) -> None:
        """Called once at the start of training."""

    def on_step(
        self,
        action_idx: int,
        q_values: torch.Tensor,
        *,
        value_stream: torch.Tensor | None = None,
        adv_values: torch.Tensor | None = None,
    ) -> None:
        """Called once per env step with the action taken and Q-values."""

    def on_episode_end(
        self,
        episode_index: int,
        steps_survived: int,
        terminal_reward: float,
        epsilon: float,
        steps_per_second: float,
        train_metrics: dict[str, float] | None,
    ) -> None:
        """Called after each episode with aggregated metrics."""

    def on_train_end(self) -> None:
        """Called once at the end of training."""


class TBMetricsCallback(TrainingCallback):
    """Rich TensorBoard logging for DQN/D3QN training.

    Metrics logged each episode:
      Standard:
        episode/steps_survived, episode/terminal_reward, episode/epsilon,
        episode/steps_per_second
      1. episode/rolling_win_rate  -- win fraction over last `win_rate_window` eps
      2. episode/steps_survived_win, episode/steps_survived_loss  -- no NaN writes
      3. episode/action_entropy  -- entropy of the action distribution for the episode
      4. episode/mean_q_spread  -- mean(max_Q - min_Q) per step (policy confidence)
      5. episode/mean_value_stream, episode/mean_adv_spread  -- dueling decomposition
         (only written when value_stream/adv_values are passed to on_step)
      Training (when train_metrics is provided):
        episode/mean_huber_loss, episode/mean_td_error, episode/mean_q,
        episode/mean_grad_norm

    For dueling networks, pass value_stream and adv_values to on_step to get
    metrics 5. For plain DQN, omit them.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        n_actions: int,
        win_rate_window: int = 500,
    ) -> None:
        self.writer = writer
        self.n_actions = n_actions
        self._recent_outcomes: collections.deque = collections.deque(maxlen=win_rate_window)
        writer.add_custom_scalars(
            {
                "episode/steps_survived_by_outcome": {
                    "steps_survived": [
                        "Multiline",
                        ["episode/steps_survived_win", "episode/steps_survived_loss"],
                    ],
                },
                "episode/win_rate": {
                    "rolling_win_rate": ["Multiline", ["episode/rolling_win_rate"]],
                },
            }
        )
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._actions: list[int] = []
        self._q_spreads: list[float] = []
        self._v_means: list[float] = []
        self._adv_spreads: list[float] = []

    def on_step(
        self,
        action_idx: int,
        q_values: torch.Tensor,
        *,
        value_stream: torch.Tensor | None = None,
        adv_values: torch.Tensor | None = None,
    ) -> None:
        """Record per-step data. Call once per env step during action selection."""
        self._actions.append(action_idx)
        q = q_values.detach().squeeze().cpu().numpy()
        self._q_spreads.append(float(q.max() - q.min()))
        if value_stream is not None:
            self._v_means.append(float(value_stream.detach().squeeze().cpu().item()))
        if adv_values is not None:
            adv = adv_values.detach().squeeze().cpu().numpy()
            self._adv_spreads.append(float(adv.max() - adv.min()))

    def on_episode_end(
        self,
        episode_index: int,
        steps_survived: int,
        terminal_reward: float,
        epsilon: float,
        steps_per_second: float,
        train_metrics: dict[str, float] | None,
    ) -> None:
        w = self.writer

        # Standard metrics
        w.add_scalar("episode/steps_survived", steps_survived, episode_index)
        w.add_scalar("episode/terminal_reward", terminal_reward, episode_index)
        w.add_scalar("episode/epsilon", epsilon, episode_index)
        w.add_scalar("episode/steps_per_second", steps_per_second, episode_index)

        # 1. Rolling win rate
        self._recent_outcomes.append(terminal_reward)
        win_rate = sum(1 for r in self._recent_outcomes if r > 0) / len(self._recent_outcomes)
        w.add_scalar("episode/rolling_win_rate", win_rate, episode_index)

        # 2. Steps survived by outcome -- skip the write entirely rather than log NaN
        if terminal_reward > 0:
            w.add_scalar("episode/steps_survived_win", steps_survived, episode_index)
        elif terminal_reward < 0:
            w.add_scalar("episode/steps_survived_loss", steps_survived, episode_index)

        # 3. Action distribution entropy
        if self._actions:
            counts = np.bincount(self._actions, minlength=self.n_actions).astype(float)
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log(np.where(probs > 0, probs, 1.0))))
            w.add_scalar("episode/action_entropy", entropy, episode_index)

        # 4. Q-value spread (policy confidence -- higher = more decisive)
        if self._q_spreads:
            w.add_scalar("episode/mean_q_spread", float(np.mean(self._q_spreads)), episode_index)

        # 5. Dueling stream decomposition (only logged when on_step received these)
        if self._v_means:
            w.add_scalar("episode/mean_value_stream", float(np.mean(self._v_means)), episode_index)
        if self._adv_spreads:
            w.add_scalar(
                "episode/mean_adv_spread", float(np.mean(self._adv_spreads)), episode_index
            )

        # Training metrics
        if train_metrics:
            w.add_scalar("episode/mean_huber_loss", train_metrics["loss"], episode_index)
            w.add_scalar("episode/mean_td_error", train_metrics["td_error"], episode_index)
            w.add_scalar("episode/mean_q", train_metrics["q_mean"], episode_index)
            w.add_scalar("episode/mean_grad_norm", train_metrics["grad_norm"], episode_index)

        self._reset_episode()

    def on_train_end(self) -> None:
        self.writer.close()


def make_tb_writer(log_dir: Path) -> SummaryWriter:
    """Create a SummaryWriter with tensorboardX logging suppressed."""
    logging.getLogger("tensorboardX").setLevel(logging.ERROR)
    return SummaryWriter(log_dir=str(log_dir))
