"""Tests for DQN episode-fraction epsilon decay (epsilon_for_episode)."""

import math

import pytest

from surround.dqn.train_dqn import epsilon_for_episode


def test_epsilon_episode_zero_is_start():
    """First episode uses eps_start."""
    eps = epsilon_for_episode(
        episode_index=0,
        num_episodes=10_000,
        decay_fraction=0.4,
        eps_start=0.9,
        eps_end=0.01,
    )
    assert eps == pytest.approx(0.9)


def test_epsilon_at_decay_window_near_end():
    """At end of decay window (40% of 10k = 4000), epsilon is ~95% toward eps_end."""
    num_episodes = 10_000
    decay_fraction = 0.4
    decay_episodes = int(num_episodes * decay_fraction)  # 4000
    eps = epsilon_for_episode(
        episode_index=decay_episodes,
        num_episodes=num_episodes,
        decay_fraction=decay_fraction,
        eps_start=0.9,
        eps_end=0.01,
    )
    # exp(-3) ≈ 0.0498 => epsilon ≈ 0.01 + 0.0498 * 0.89 ≈ 0.054
    expected = 0.01 + (0.9 - 0.01) * math.exp(-3)
    assert eps == pytest.approx(expected, rel=1e-5)
    assert 0.05 < eps < 0.06


def test_epsilon_monotonic_decrease():
    """Epsilon decreases as episode_index increases."""
    num_episodes = 1000
    decay_fraction = 0.4
    prev = epsilon_for_episode(0, num_episodes, decay_fraction, 0.9, 0.01)
    for episode_index in [1, 10, 100, 200, 400, 500, 1000]:
        curr = epsilon_for_episode(episode_index, num_episodes, decay_fraction, 0.9, 0.01)
        assert curr < prev
        prev = curr


def test_epsilon_late_episodes_near_end():
    """Well past the decay window, epsilon is close to eps_end."""
    eps = epsilon_for_episode(
        episode_index=50_000,
        num_episodes=10_000,
        decay_fraction=0.4,
        eps_start=0.9,
        eps_end=0.01,
    )
    assert eps == pytest.approx(0.01, abs=0.001)


def test_epsilon_scale_invariant_with_num_episodes():
    """Same fraction of decay window gives same epsilon regardless of num_episodes."""
    # At 20% through a 40% decay window, episode_index = 0.2 * 0.4 * num_episodes
    eps_10k = epsilon_for_episode(
        episode_index=800,  # 0.2 * 4000 for num_episodes=10k, fraction=0.4
        num_episodes=10_000,
        decay_fraction=0.4,
        eps_start=0.9,
        eps_end=0.01,
    )
    eps_20k = epsilon_for_episode(
        episode_index=1600,  # 0.2 * 8000 for num_episodes=20k
        num_episodes=20_000,
        decay_fraction=0.4,
        eps_start=0.9,
        eps_end=0.01,
    )
    assert eps_10k == pytest.approx(eps_20k)


def test_epsilon_decay_fraction_one_uses_single_episode_window():
    """decay_fraction=1 with small num_episodes: decay_episodes = num_episodes."""
    eps_mid = epsilon_for_episode(2, 5, 1.0, 0.9, 0.01)
    eps_end_window = epsilon_for_episode(5, 5, 1.0, 0.9, 0.01)
    assert eps_mid > eps_end_window
    assert eps_end_window == pytest.approx(0.01 + (0.9 - 0.01) * math.exp(-3), rel=1e-5)
