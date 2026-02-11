"""Tests for surround.utils.callbacks."""

from surround.utils.callbacks import TensorboardCallback, TrainingCallback


def test_training_callback_no_ops():
    """TrainingCallback base class methods are no-ops."""
    cb = TrainingCallback()
    cb.on_train_start()
    cb.on_episode_end(0, 10, 0.5, 0.1)
    cb.on_train_end()


def test_tensorboard_callback_lifecycle(tmp_path):
    """TensorboardCallback creates writer, logs, and closes."""
    cb = TensorboardCallback(log_dir=tmp_path)
    assert cb._writer is None

    cb.on_train_start()
    assert cb._writer is not None

    cb.on_episode_end(episode_index=0, episode_steps=5, terminal_reward=1.0, epsilon=0.5)
    cb.on_episode_end(episode_index=1, episode_steps=3, terminal_reward=-1.0, epsilon=0.4)

    cb.on_train_end()
    assert cb._writer is None

    # Log dir should contain events (tensorboard writes them)
    assert tmp_path.exists()
    events = list(tmp_path.glob("events.out.tfevents.*"))
    assert len(events) >= 1


def test_tensorboard_callback_on_episode_end_without_start():
    """on_episode_end is safe when writer was never created."""
    cb = TensorboardCallback(log_dir=__import__("pathlib").Path("/tmp"))
    cb.on_episode_end(0, 1, 0.0, 0.1)
    # No exception; writer is None so it returns early


def test_tensorboard_callback_on_train_end_without_start():
    """on_train_end is safe when writer was never created."""
    cb = TensorboardCallback(log_dir=__import__("pathlib").Path("/tmp"))
    cb.on_train_end()
    assert cb._writer is None


def test_tensorboard_callback_logs_nan_for_outcomes(tmp_path):
    """TensorboardCallback logs episode_steps with np.nan for non-matching outcomes."""
    cb = TensorboardCallback(log_dir=tmp_path)
    cb.on_train_start()
    # Win: terminal_reward > 0 -> steps_survived_win = steps, others nan
    cb.on_episode_end(episode_index=0, episode_steps=10, terminal_reward=1.0, epsilon=0.5)
    # Loss: terminal_reward < 0
    cb.on_episode_end(episode_index=1, episode_steps=5, terminal_reward=-1.0, epsilon=0.4)
    # Trunc: terminal_reward == 0
    cb.on_episode_end(episode_index=2, episode_steps=7, terminal_reward=0.0, epsilon=0.3)
    cb.on_train_end()
    # Just ensure no exception; we don't assert on internal scalar values
    assert not cb._writer
