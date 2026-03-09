"""Opponent pool for self-play training.

Scans a set of directories for past policy checkpoints and provides uniform
random sampling. New snapshots from the current training run are added to the
pool as training progresses (fictitious self-play).
"""

import random
from pathlib import Path

import torch

from surround.utils.checkpoint import load_checkpoint, save_checkpoint


class OpponentPool:
    """Pool of past policy checkpoints for self-play opponent sampling.

    Checkpoints are discovered lazily: file paths are collected at init, but
    state dicts are only loaded when sample() is called. This keeps startup fast
    even with thousands of checkpoints on disk.

    Only checkpoints whose metadata reports steps_survived >= min_steps are
    returned by sample(); others are silently skipped.
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

        self._paths: list[Path] = []
        for d in scan_dirs:
            self._paths.extend(Path(d).rglob("*.pt"))
        random.shuffle(self._paths)
        print(f"OpponentPool: {len(self._paths)} checkpoint files found in scan dirs")

    def sample(self, max_tries: int = 30) -> dict | None:
        """Return a random qualifying state dict, or None if none can be found."""
        if not self._paths:
            return None
        candidates = random.sample(self._paths, min(max_tries, len(self._paths)))
        for path in candidates:
            state_dict, meta = load_checkpoint(path, map_location=self._device)
            if meta.get("steps_survived", 0) >= self._min_steps:
                return state_dict
        return None

    def add(self, state_dict: dict, episode: int, steps_survived: int) -> None:
        """Save a snapshot of the current policy and register it in the pool."""
        self._pool_save_dir.mkdir(parents=True, exist_ok=True)
        path = self._pool_save_dir / f"pool_{episode:06d}.pt"
        save_checkpoint(path, state_dict, steps_survived=steps_survived)
        self._paths.append(path)

    @property
    def size(self) -> int:
        return len(self._paths)
