"""Save/load PyTorch checkpoints with optional metadata."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CheckpointPaths:
    """Checkpoint file paths derived from a training log directory."""

    log_dir: Path

    @property
    def dir(self) -> Path:
        return self.log_dir / "checkpoints"

    @property
    def latest(self) -> Path:
        return self.dir / "policy_latest.pt"

    @property
    def best(self) -> Path:
        return self.dir / "policy_best.pt"

    @property
    def metadata(self) -> Path:
        return self.dir / "metadata.json"


def save_checkpoint(
    path: Path | str,
    state_dict: dict,
    *,
    steps_survived: int | None = None,
    **metadata: object,
) -> None:
    """Save model state_dict to path with optional metadata.

    The file can be loaded with load_checkpoint(); legacy loaders that expect
    only a state_dict are not supported (use load_checkpoint()[0] for state_dict).
    """
    payload = {
        "state_dict": state_dict,
        "metadata": {
            **metadata,
            **({"steps_survived": steps_survived} if steps_survived is not None else {}),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    map_location: torch.device | str | None = None,
) -> tuple[dict, dict]:
    """Load checkpoint from path. Returns (state_dict, metadata).

    Supports both the metadata format (dict with "state_dict" and "metadata")
    and legacy checkpoints that are just a state_dict (metadata will be {}).
    """
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"], loaded.get("metadata", {})
    return loaded, {}
