"""Opponent pool for self-play training.

Scans a set of directories for past policy checkpoints and provides
steps_survived-weighted random sampling. Better opponents (higher
steps_survived) are sampled more often, so opponent quality naturally
increases as the learner improves and adds stronger snapshots to the pool.
"""

import random
from pathlib import Path

import torch
from tqdm import tqdm

from surround.utils.checkpoint import load_checkpoint, save_checkpoint


class OpponentPool:
    """Pool of past policy checkpoints for self-play opponent sampling.

    At init, all checkpoint files in the scan dirs are loaded (metadata only
    kept in memory) to build a steps_survived-weighted sampling distribution.
    On each sample() call a path is drawn proportional to its steps_survived,
    then the state dict is loaded from disk.
    """

    def __init__(
        self,
        scan_dirs: list[Path],
        min_steps: int,
        device: torch.device,
        pool_save_dir: Path,
    ) -> None:
        self._min_steps = min_steps
        self._device = device
        self._pool_save_dir = pool_save_dir

        all_paths = [p for d in scan_dirs for p in Path(d).rglob("*.pt")]
        print(f"OpponentPool: scanning {len(all_paths)} checkpoints...")

        self._pool: list[tuple[Path, int]] = []  # (path, steps_survived)
        for path in tqdm(all_paths, desc="OpponentPool scan", leave=False):
            _, meta = load_checkpoint(path, map_location="cpu")
            steps = meta.get("steps_survived", 0)
            if steps >= min_steps:
                self._pool.append((path, steps))

        print(f"OpponentPool: {len(self._pool)} qualifying checkpoints (min_steps={min_steps})")

    def sample(self) -> tuple[dict, int] | None:
        """Return (state_dict, steps_survived) sampled proportional to steps_survived.

        Returns None if the pool is empty.
        """
        if not self._pool:
            return None
        weights = [s for _, s in self._pool]
        ((path, steps_survived),) = random.choices(self._pool, weights=weights, k=1)
        state_dict, _ = load_checkpoint(path, map_location=self._device)
        return state_dict, steps_survived

    def add(self, state_dict: dict, episode: int, steps_survived: int) -> None:
        """Save a snapshot of the current policy and register it in the pool."""
        self._pool_save_dir.mkdir(parents=True, exist_ok=True)
        path = self._pool_save_dir / f"pool_{episode:06d}.pt"
        save_checkpoint(path, state_dict, steps_survived=steps_survived)
        self._pool.append((path, steps_survived))

    @property
    def size(self) -> int:
        return len(self._pool)
