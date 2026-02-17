"""Tests for checkpoint save/load with metadata."""

import torch

from surround.utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_roundtrip_metadata(tmp_path):
    """Save checkpoint with metadata; load returns same state_dict and metadata."""
    state_dict = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor(3.0)}
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        state_dict,
        steps_survived=1234,
        episode_index=56,
        episodes_completed=57,
        git_commit="abc",
        git_branch="main",
    )
    assert path.exists()

    loaded_state, meta = load_checkpoint(path)
    assert list(loaded_state.keys()) == ["weight", "bias"]
    assert torch.equal(loaded_state["weight"], state_dict["weight"])
    assert torch.equal(loaded_state["bias"], state_dict["bias"])
    assert meta["steps_survived"] == 1234
    assert meta["episode_index"] == 56
    assert meta["episodes_completed"] == 57
    assert meta["git_commit"] == "abc"
    assert meta["git_branch"] == "main"


def test_save_without_steps_survived(tmp_path):
    """steps_survived omitted when not passed; metadata still round-trips."""
    state_dict = {"x": torch.tensor(1.0)}
    path = tmp_path / "model.pt"
    save_checkpoint(path, state_dict, episode_index=0)
    loaded_state, meta = load_checkpoint(path)
    assert "steps_survived" not in meta
    assert meta["episode_index"] == 0
    assert torch.equal(loaded_state["x"], state_dict["x"])


def test_load_legacy_raw_state_dict(tmp_path):
    """Legacy checkpoint (raw state_dict only) loads as (state_dict, {})."""
    state_dict = {"weight": torch.tensor([4.0, 5.0])}
    path = tmp_path / "legacy.pt"
    torch.save(state_dict, path)

    loaded_state, meta = load_checkpoint(path)
    assert torch.equal(loaded_state["weight"], state_dict["weight"])
    assert meta == {}
