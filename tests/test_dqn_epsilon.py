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


def _epsilon_step_based(
    steps_done: int,
    eps_decay: int | float,
    eps_start: float = 0.9,
    eps_end: float = 0.01,
) -> float:
    """Step-based epsilon (exp11 style): same curve as EPS_DECAY in train_dqn."""
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)


def _fraction_to_match_step_decay(
    eps_decay_steps: int | float,
    steps_per_episode: float,
    num_episodes: int,
) -> float:
    """EPS_DECAY_FRACTION such that episode-based decay matches step-based EPS_DECAY.

    With constant steps_per_episode, matching exponents gives:
      steps / eps_decay_steps  =  episode_index / (decay_episodes / 3)
      => decay_episodes = 3 * eps_decay_steps / steps_per_episode
      => fraction = decay_episodes / num_episodes
    """
    decay_episodes = 3 * eps_decay_steps / steps_per_episode
    return decay_episodes / num_episodes


def test_epsilon_fraction_matches_step_decay_over_first_10k_steps():
    """For constant steps per episode, we can pick a fraction so the episode-based
    curve matches the step-based curve (e.g. EPS_DECAY=100_000) over the first 10k steps.
    That fraction is 0.06 (not the 0.01 we use in production for TensorBoard matching).
    """
    eps_start = 0.9
    eps_end = 0.01
    eps_decay_steps = 100_000
    num_episodes = 50_000
    steps_per_episode = 100.0  # constant for this test
    max_steps = 10_000

    fraction = _fraction_to_match_step_decay(eps_decay_steps, steps_per_episode, num_episodes)
    assert 0 < fraction <= 1.0
    # decay_episodes = 3 * 100_000 / 100 = 3000, fraction = 3000/50000 = 0.06
    assert fraction == pytest.approx(0.06)

    # Sample at step boundaries: 0, 100, 200, ..., 10_000
    for step in range(0, max_steps + 1, int(steps_per_episode)):
        episode_index = int(step // steps_per_episode)
        eps_step = _epsilon_step_based(step, eps_decay_steps, eps_start, eps_end)
        eps_episode = epsilon_for_episode(episode_index, num_episodes, fraction, eps_start, eps_end)
        assert eps_step == pytest.approx(eps_episode), (
            f"At step={step}, episode_index={episode_index}: "
            f"step_eps={eps_step} vs episode_eps={eps_episode}"
        )


def test_epsilon_fraction_001_reaches_near_min_by_episode_1000():
    """EPS_DECAY_FRACTION=0.01 is chosen so epsilon is ~0.01 by ~episode 1000, matching
    the original exp11 TensorBoard curve (x-axis = episode index)."""
    from surround.conf import constants

    num_episodes = constants.NUM_EPISODES
    fraction = constants.EPS_DECAY_FRACTION
    eps_at_1000 = epsilon_for_episode(
        1000, num_episodes, fraction, constants.EPS_START, constants.EPS_END
    )
    assert eps_at_1000 == pytest.approx(0.01, abs=0.01), (
        f"Expected epsilon ~0.01 by episode 1000 for TensorBoard match; got {eps_at_1000}"
    )
