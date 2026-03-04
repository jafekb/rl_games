"""Tests for surround/utils/callbacks.py."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from surround.utils.callbacks import TBMetricsCallback, TensorboardCallback, TrainingCallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tb_cb(n_actions=4, win_rate_window=10):
    writer = MagicMock()
    cb = TBMetricsCallback(writer, n_actions=n_actions, win_rate_window=win_rate_window)
    return cb, writer


def scalar_calls(writer):
    """Return {tag: last_value} from all add_scalar mock calls."""
    result = {}
    for c in writer.add_scalar.call_args_list:
        result[c.args[0]] = c.args[1]
    return result


def q(values):
    return torch.tensor([values], dtype=torch.float32)


# ===========================================================================
# TrainingCallback base class
# ===========================================================================


def test_training_callback_no_ops():
    """TrainingCallback base-class hooks are all no-ops."""
    cb = TrainingCallback()
    cb.on_train_start()
    cb.on_step(0, q([1.0, 0.0, 0.0, 0.0]))
    cb.on_episode_end(0, 10, 0.5, 0.1, 5.0, None)
    cb.on_train_end()


# ===========================================================================
# TensorboardCallback (Q-learning, backward-compat)
# ===========================================================================


def test_tensorboard_callback_lifecycle(tmp_path):
    cb = TensorboardCallback(log_dir=tmp_path)
    assert cb._writer is None
    cb.on_train_start()
    assert cb._writer is not None
    cb.on_episode_end(episode_index=0, episode_steps=5, terminal_reward=1.0, epsilon=0.5)
    cb.on_episode_end(episode_index=1, episode_steps=3, terminal_reward=-1.0, epsilon=0.4)
    cb.on_train_end()
    assert cb._writer is None
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) >= 1


def test_tensorboard_callback_safe_without_start():
    cb = TensorboardCallback(log_dir=__import__("pathlib").Path("/tmp"))
    cb.on_episode_end(0, 1, 0.0, 0.1)  # no exception
    cb.on_train_end()
    assert cb._writer is None


def test_tensorboard_callback_logs_outcomes(tmp_path):
    cb = TensorboardCallback(log_dir=tmp_path)
    cb.on_train_start()
    cb.on_episode_end(episode_index=0, episode_steps=10, terminal_reward=1.0, epsilon=0.5)
    cb.on_episode_end(episode_index=1, episode_steps=5, terminal_reward=-1.0, epsilon=0.4)
    cb.on_episode_end(episode_index=2, episode_steps=7, terminal_reward=0.0, epsilon=0.3)
    cb.on_train_end()
    assert not cb._writer


# ===========================================================================
# TBMetricsCallback — rolling win rate
# ===========================================================================


class TestRollingWinRate:
    def test_all_wins(self):
        cb, writer = make_tb_cb(win_rate_window=5)
        for i in range(5):
            cb.on_episode_end(i, 50, +1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/rolling_win_rate"] == pytest.approx(1.0)

    def test_all_losses(self):
        cb, writer = make_tb_cb(win_rate_window=5)
        for i in range(5):
            cb.on_episode_end(i, 50, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/rolling_win_rate"] == pytest.approx(0.0)

    def test_window_slides_out_old_results(self):
        cb, writer = make_tb_cb(win_rate_window=3)
        # 2 losses then 3 wins — window should contain only the 3 wins
        for i in range(2):
            cb.on_episode_end(i, 50, -1.0, 0.1, 10.0, None)
        for i in range(2, 5):
            cb.on_episode_end(i, 50, +1.0, 0.1, 10.0, None)
        win_rate_vals = [
            c.args[1]
            for c in writer.add_scalar.call_args_list
            if c.args[0] == "episode/rolling_win_rate"
        ]
        assert win_rate_vals[-1] == pytest.approx(1.0)

    def test_mixed(self):
        cb, writer = make_tb_cb(win_rate_window=4)
        for reward in [+1.0, -1.0, +1.0, -1.0]:
            cb.on_episode_end(0, 50, reward, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/rolling_win_rate"] == pytest.approx(0.5)


# ===========================================================================
# TBMetricsCallback — steps survived by outcome (no NaN)
# ===========================================================================


class TestStepsSurvivedByOutcome:
    def test_win_writes_win_tag_only(self):
        cb, writer = make_tb_cb()
        cb.on_episode_end(0, 42, +1.0, 0.1, 10.0, None)
        tags = {c.args[0] for c in writer.add_scalar.call_args_list}
        assert "episode/steps_survived_win" in tags
        assert "episode/steps_survived_loss" not in tags

    def test_loss_writes_loss_tag_only(self):
        cb, writer = make_tb_cb()
        cb.on_episode_end(0, 42, -1.0, 0.1, 10.0, None)
        tags = {c.args[0] for c in writer.add_scalar.call_args_list}
        assert "episode/steps_survived_loss" in tags
        assert "episode/steps_survived_win" not in tags

    def test_correct_step_count_logged(self):
        cb, writer = make_tb_cb()
        cb.on_episode_end(0, 77, +1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/steps_survived_win"] == 77

    def test_no_nan_values_written(self):
        cb, writer = make_tb_cb()
        for reward in [+1.0, -1.0, +1.0, -1.0]:
            cb.on_episode_end(0, 50, reward, 0.1, 10.0, None)
        for c in writer.add_scalar.call_args_list:
            v = c.args[1]
            assert not (isinstance(v, float) and math.isnan(v)), f"NaN written for {c.args[0]}"


# ===========================================================================
# TBMetricsCallback — action entropy
# ===========================================================================


class TestActionEntropy:
    def test_uniform_policy_is_max_entropy(self):
        cb, writer = make_tb_cb(n_actions=4)
        for a in range(4):
            cb.on_step(a, q([0.1, 0.2, 0.3, 0.4]))
        cb.on_episode_end(0, 4, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/action_entropy"] == pytest.approx(
            math.log(4), abs=1e-5
        )

    def test_deterministic_policy_is_zero_entropy(self):
        cb, writer = make_tb_cb(n_actions=4)
        for _ in range(10):
            cb.on_step(0, q([1.0, 0.0, 0.0, 0.0]))
        cb.on_episode_end(0, 10, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/action_entropy"] == pytest.approx(0.0, abs=1e-5)

    def test_entropy_resets_between_episodes(self):
        cb, writer = make_tb_cb(n_actions=4)
        for _ in range(4):
            cb.on_step(0, q([1.0, 0.0, 0.0, 0.0]))
        cb.on_episode_end(0, 4, -1.0, 0.1, 10.0, None)
        writer.reset_mock()
        for a in range(4):
            cb.on_step(a, q([0.25, 0.25, 0.25, 0.25]))
        cb.on_episode_end(1, 4, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/action_entropy"] == pytest.approx(
            math.log(4), abs=1e-5
        )


# ===========================================================================
# TBMetricsCallback — Q-value spread
# ===========================================================================


class TestQSpread:
    def test_spread_is_mean_of_max_minus_min(self):
        cb, writer = make_tb_cb(n_actions=4)
        cb.on_step(0, q([1.0, 2.0, 3.0, 4.0]))  # spread 3.0
        cb.on_step(1, q([0.0, 0.0, 0.0, 1.0]))  # spread 1.0
        cb.on_episode_end(0, 2, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/mean_q_spread"] == pytest.approx(2.0)

    def test_tag_absent_when_no_steps(self):
        cb, writer = make_tb_cb()
        cb.on_episode_end(0, 50, -1.0, 0.1, 10.0, None)
        assert "episode/mean_q_spread" not in scalar_calls(writer)


# ===========================================================================
# TBMetricsCallback — dueling stream decomposition
# ===========================================================================


class TestDuelingStreams:
    def test_value_stream_logged(self):
        cb, writer = make_tb_cb(n_actions=4)
        cb.on_step(0, q([0.6, 0.4, 0.7, 0.3]), value_stream=torch.tensor([[0.5]]))
        cb.on_episode_end(0, 1, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/mean_value_stream"] == pytest.approx(0.5)

    def test_adv_spread_is_max_minus_min(self):
        cb, writer = make_tb_cb(n_actions=4)
        a = torch.tensor([[1.0, -1.0, 0.5, -0.5]])  # spread = 2.0
        cb.on_step(0, q([0.0, 0.0, 0.0, 0.0]), adv_values=a)
        cb.on_episode_end(0, 1, -1.0, 0.1, 10.0, None)
        assert scalar_calls(writer)["episode/mean_adv_spread"] == pytest.approx(2.0)

    def test_dueling_tags_absent_without_streams(self):
        cb, writer = make_tb_cb(n_actions=4)
        cb.on_step(0, q([1.0, 0.0, 0.0, 0.0]))
        cb.on_episode_end(0, 1, -1.0, 0.1, 10.0, None)
        tags = scalar_calls(writer)
        assert "episode/mean_value_stream" not in tags
        assert "episode/mean_adv_spread" not in tags


# ===========================================================================
# TBMetricsCallback — training metrics passthrough
# ===========================================================================


class TestTrainingMetrics:
    def test_all_metrics_written(self):
        cb, writer = make_tb_cb()
        tm = {"loss": 0.1, "td_error": 0.05, "q_mean": -0.3, "grad_norm": 0.8}
        cb.on_episode_end(0, 50, -1.0, 0.1, 10.0, tm)
        calls = scalar_calls(writer)
        assert calls["episode/mean_huber_loss"] == pytest.approx(0.1)
        assert calls["episode/mean_td_error"] == pytest.approx(0.05)
        assert calls["episode/mean_q"] == pytest.approx(-0.3)
        assert calls["episode/mean_grad_norm"] == pytest.approx(0.8)

    def test_no_train_tags_when_none(self):
        cb, writer = make_tb_cb()
        cb.on_episode_end(0, 50, -1.0, 0.1, 10.0, None)
        tags = scalar_calls(writer)
        assert "episode/mean_huber_loss" not in tags


# ===========================================================================
# DuelingDQN.forward_with_streams
# ===========================================================================


class TestDuelingDQNStreams:
    @pytest.fixture
    def net(self):
        from surround.dqn.train_d3qn import DuelingDQN

        return DuelingDQN(n_actions=4)

    def test_forward_and_streams_give_same_q(self, net):
        x = torch.zeros(1, 1, 80, 80)
        q_forward = net(x)
        q_streams, _v, _a = net.forward_with_streams(x)
        assert torch.allclose(q_forward, q_streams, atol=1e-6)

    def test_output_shapes(self, net):
        x = torch.zeros(1, 1, 80, 80)
        q, v, a = net.forward_with_streams(x)
        assert q.shape == (1, 4)
        assert v.shape == (1, 1)
        assert a.shape == (1, 4)

    def test_decomposition_identity(self, net):
        """Q == V + A - mean(A) for all inputs."""
        x = torch.randn(4, 1, 80, 80)
        q, v, a = net.forward_with_streams(x)
        expected = v + a - a.mean(dim=1, keepdim=True)
        assert torch.allclose(q, expected, atol=1e-6)

    def test_batch_consistency(self, net):
        x = torch.randn(8, 1, 80, 80)
        assert torch.allclose(net(x), net.forward_with_streams(x)[0], atol=1e-6)
