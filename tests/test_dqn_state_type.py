"""Tests for DQN state type support (7-tuple vs grayscale image)."""

import numpy as np
import pytest
import torch

from surround.conf import constants
from surround.dqn.train_dqn import (
    DQN,
    DqnMlp,
    _make_dqn_net,
    get_state_from_observation,
)


def test_make_dqn_net_state_tuple_returns_mlp(monkeypatch):
    """_make_dqn_net with DQN_STATE_TYPE=state_tuple returns DqnMlp."""
    monkeypatch.setattr(constants, "DQN_STATE_TYPE", "state_tuple")
    device = torch.device("cpu")
    net = _make_dqn_net(4, device)
    assert isinstance(net, DqnMlp)
    assert next(net.parameters()).device == device


def test_make_dqn_net_grayscale_returns_cnn(monkeypatch):
    """_make_dqn_net with DQN_STATE_TYPE=grayscale returns DQN (CNN)."""
    monkeypatch.setattr(constants, "DQN_STATE_TYPE", "grayscale")
    device = torch.device("cpu")
    net = _make_dqn_net(4, device)
    assert isinstance(net, DQN)
    assert next(net.parameters()).device == device


def test_make_dqn_net_class_map_returns_cnn(monkeypatch):
    """_make_dqn_net with DQN_STATE_TYPE=class_map returns DQN (CNN), same as exp11."""
    monkeypatch.setattr(constants, "DQN_STATE_TYPE", "class_map")
    device = torch.device("cpu")
    net = _make_dqn_net(4, device)
    assert isinstance(net, DQN)
    assert next(net.parameters()).device == device


def test_make_dqn_net_unknown_raises(monkeypatch):
    """_make_dqn_net with unknown DQN_STATE_TYPE raises ValueError."""
    monkeypatch.setattr(constants, "DQN_STATE_TYPE", "invalid")
    with pytest.raises(ValueError, match="Unknown DQN_STATE_TYPE"):
        _make_dqn_net(4, torch.device("cpu"))


def test_dqn_mlp_forward_shape():
    """DqnMlp(4): input (batch, 7) -> output (batch, 4)."""
    net = DqnMlp(n_actions=4)
    x = torch.randn(3, 7)
    out = net(x)
    assert out.shape == (3, 4)


def test_dqn_cnn_forward_shape():
    """DQN(4): input (batch, 1, H, W) -> output (batch, 4). Uses DQN_PREPROCESS spatial size."""
    net = DQN(n_actions=4)
    h, w = constants.DQN_PREPROCESS_HEIGHT, constants.DQN_PREPROCESS_WIDTH
    x = torch.randn(2, 1, h, w)
    out = net(x)
    assert out.shape == (2, 4)


def test_get_state_from_observation_returns_7_tuple():
    """get_state_from_observation returns a 7-tuple (works with synthetic obs)."""
    # Minimal grayscale-like shape; get_location will see no ego/opp -> (1,1,1,1,1,1,last_action)
    obs = np.zeros((210, 160), dtype=np.uint8)
    state = get_state_from_observation(obs, last_action=2)
    assert isinstance(state, tuple)
    assert len(state) == 7
    assert state[-1] == 2  # last_action
    assert all(isinstance(x, int | np.integer) for x in state)


def _fake_run_metadata():
    """Stub for run metadata so tests don't need git (e.g. CI detached HEAD)."""
    return {
        "timestamp": "2020-01-01 00:00:00 PST",
        "git_commit": "test",
        "git_branch": "test",
    }


@pytest.mark.parametrize("state_type", ["state_tuple", "grayscale", "class_map"])
def test_trainer_one_step_per_state_type(state_type, monkeypatch, tmp_path):
    """Run one episode step for each DQN_STATE_TYPE (integration; needs ALE)."""
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        env = gym.make(
            "ALE/Surround-v5",
            obs_type="grayscale",
            full_action_space=False,
            difficulty=0,
            mode=0,
            frameskip=4,
        )
        env.close()
    except Exception:
        pytest.skip("ALE/Surround-v5 grayscale env not available")

    monkeypatch.setattr(constants, "DQN_STATE_TYPE", state_type)
    monkeypatch.setattr(constants, "DQN_LOG_DIR", tmp_path / "dqn_test_log")
    monkeypatch.setattr(constants, "NUM_EPISODES", 1)
    monkeypatch.setattr(constants, "MAX_CYCLES", 5)
    monkeypatch.setattr(constants, "MEMORY_CAPACITY", 500)
    monkeypatch.setattr(constants, "BATCH_SIZE", 32)

    from surround.dqn import train_dqn

    monkeypatch.setattr(train_dqn, "_get_run_metadata", _fake_run_metadata)
    trainer = train_dqn.DQNTrainer()
    obs, _ = trainer.env.reset()
    last_action = 1
    preprocessed = trainer._preprocess_observation(obs, last_action)
    state_tensor = trainer._observation_to_tensor(preprocessed)

    if state_type == "state_tuple":
        assert state_tensor.shape == (1, 7)
        assert isinstance(preprocessed, tuple)
    else:
        assert state_tensor.dim() == 4
        assert state_tensor.shape[0] == 1
        assert state_tensor.shape[1] in (1, 4)  # 1=single frame, 4=frame stack
        assert state_tensor.shape[2] == constants.DQN_PREPROCESS_HEIGHT
        assert state_tensor.shape[3] == constants.DQN_PREPROCESS_WIDTH
        assert isinstance(preprocessed, np.ndarray)

    # One forward pass
    with torch.no_grad():
        q = trainer.policy_net(state_tensor)
    assert q.shape == (1, trainer.n_actions)
    trainer.writer.close()
    trainer.env.close()
