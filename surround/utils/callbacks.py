"""Training callbacks for Q-learning."""

import logging
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter


class TrainingCallback:
    """Base class for training callbacks. Override the hooks you need."""

    def on_train_start(self) -> None:
        """Called once at the start of training."""

    def on_episode_end(
        self,
        episode_index: int,
        episode_steps: int,
        terminal_reward: float,
        epsilon: float,
    ) -> None:
        """Called after each episode."""

    def on_train_end(self) -> None:
        """Called once at the end of training."""


class TensorboardCallback(TrainingCallback):
    """Logs episode metrics to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._writer: SummaryWriter | None = None

    def on_train_start(self) -> None:
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._writer.add_custom_scalars(
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

    def on_episode_end(
        self,
        episode_index: int,
        episode_steps: int,
        terminal_reward: float,
        epsilon: float,
    ) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("episode/steps_survived", episode_steps, episode_index)
        self._writer.add_scalar("episode/terminal_reward", terminal_reward, episode_index)
        self._writer.add_scalar(
            "episode/steps_survived_win",
            episode_steps if terminal_reward > 0 else np.nan,
            episode_index,
        )
        self._writer.add_scalar(
            "episode/steps_survived_loss",
            episode_steps if terminal_reward < 0 else np.nan,
            episode_index,
        )
        self._writer.add_scalar(
            "episode/steps_survived_trunc",
            episode_steps if terminal_reward == 0 else np.nan,
            episode_index,
        )
        self._writer.add_scalar("episode/epsilon", epsilon, episode_index)

    def on_train_end(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
