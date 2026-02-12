"""Tests for surround.q_learning.train_ql."""

import json
from pathlib import Path

import numpy as np
import pytest

from surround import constants
from surround.q_learning.train_ql import load_q_table


def test_load_q_table_roundtrip(tmp_path):
    """load_q_table reads JSON written in save_q_table format."""
    data = {
        "states": {
            "1,1,1,1,1,1,1": [0.1, 0.2, 0.3, 0.4],
            "0,0,0,0,1,1,2": [0.5, 0.5, 0.5, 0.5],
        },
        "analysis": {},
    }
    path = tmp_path / "q_table.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    table = load_q_table(path)
    assert len(table) == 2
    assert (1, 1, 1, 1, 1, 1, 1) in table
    assert (0, 0, 0, 0, 1, 1, 2) in table
    np.testing.assert_array_almost_equal(table[(1, 1, 1, 1, 1, 1, 1)], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_almost_equal(table[(0, 0, 0, 0, 1, 1, 2)], [0.5, 0.5, 0.5, 0.5])


def test_load_q_table_empty_states(tmp_path):
    """load_q_table handles missing or empty states key."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"analysis": {}}, indent=2), encoding="utf-8")
    table = load_q_table(path)
    assert table == {}


def test_qlearning_train_with_empty_callbacks(ale_available, tmp_path, monkeypatch):
    """QLearning.train() runs without error with callbacks=[] and one episode (smoke)."""
    if not ale_available:
        pytest.skip("ALE Surround ROM not available")
    monkeypatch.setattr(constants, "Q_TABLE_PATH", tmp_path / "q_table.json")
    from surround.q_learning.train_ql import QLearning

    ql = QLearning(
        epsilon_start=0.1,
        epsilon_min=0.05,
        epsilon_decay_steps=10,
        episodes=1,
        state_mode=constants.STATE_MODE,
        callbacks=[],
    )
    ql.train()


def test_qlearning_default_callbacks_is_tensorboard(ale_available):
    """QLearning with default callbacks uses one TensorboardCallback."""
    if not ale_available:
        pytest.skip("ALE Surround ROM not available")
    from surround.q_learning.train_ql import QLearning
    from surround.utils.callbacks import TensorboardCallback

    log_dir = Path("/tmp/surround_test_log")
    ql = QLearning(
        epsilon_start=0.1,
        epsilon_min=0.05,
        epsilon_decay_steps=10,
        episodes=0,
        state_mode=constants.STATE_MODE,
        log_dir=log_dir,
    )
    assert len(ql.callbacks) == 1
    assert isinstance(ql.callbacks[0], TensorboardCallback)
    assert ql.callbacks[0].log_dir == log_dir
