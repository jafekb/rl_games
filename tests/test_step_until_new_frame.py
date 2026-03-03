"""Tests for surround.utils.env_state.step_until_new_frame."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from surround.utils.env_state import step_until_new_frame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBS = np.zeros((210, 160), dtype=np.uint8)  # arbitrary grayscale frame


def _make_env(*step_returns):
    """Mock env whose step() yields the given (obs, reward, term, trunc, info) tuples."""
    env = MagicMock()
    env.step.side_effect = [
        (_OBS.copy(), reward, term, trunc, {}) for (reward, term, trunc) in step_returns
    ]
    return env


def _locs(ego, opp):
    """Build a get_location return value."""
    return {"ego": ego, "opp": opp, "walls": set()}


# ---------------------------------------------------------------------------
# Core behaviour: stepping until position changes
# ---------------------------------------------------------------------------


def test_returns_after_one_substep_when_position_changes():
    """If the position changes on the very first step, only one env.step is called."""
    env = _make_env((0.0, False, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    new_locs = _locs((1, 0), (5, 5))  # ego moved

    with patch("surround.utils.env_state.get_location", return_value=new_locs):
        obs, reward, terminated, truncated, info = step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_keeps_stepping_until_position_changes():
    """Substeps are repeated until the board state actually changes."""
    env = _make_env(
        (0.0, False, False),  # substep 1 — same position
        (0.0, False, False),  # substep 2 — same position
        (0.0, False, False),  # substep 3 — position changes
    )
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))
    moved = _locs((1, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", side_effect=[same, same, moved]):
        step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 3


def test_rewards_accumulated_across_substeps():
    """Total reward is the sum of all substep rewards."""
    env = _make_env(
        (0.1, False, False),  # substep 1 — same position
        (0.2, False, False),  # substep 2 — same position
        (0.3, False, False),  # substep 3 — position changes
    )
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))
    moved = _locs((1, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", side_effect=[same, same, moved]):
        _, total_reward, *_ = step_until_new_frame(env, last_pos, 1)

    assert total_reward == pytest.approx(0.1 + 0.2 + 0.3)


# ---------------------------------------------------------------------------
# Stopping conditions
# ---------------------------------------------------------------------------


def test_stops_immediately_on_terminated():
    """A terminated=True step exits the loop regardless of position."""
    env = _make_env((0.0, True, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))  # position did NOT change

    with patch("surround.utils.env_state.get_location", return_value=same):
        _, _, terminated, truncated, _ = step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
    assert terminated
    assert not truncated


def test_stops_immediately_on_truncated():
    """A truncated=True step exits the loop regardless of position."""
    env = _make_env((0.0, False, True))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=same):
        _, _, terminated, truncated, _ = step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
    assert truncated


def test_stops_on_terminal_reward_positive():
    """A reward of +1 (win) terminates the step loop."""
    env = _make_env((1.0, False, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=same):
        _, reward, *_ = step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
    assert reward == pytest.approx(1.0)


def test_stops_on_terminal_reward_negative():
    """A reward of -1 (loss) terminates the step loop."""
    env = _make_env((-1.0, False, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=same):
        _, reward, *_ = step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
    assert reward == pytest.approx(-1.0)


def test_respects_max_substeps():
    """Never calls env.step more than max_substeps times, even if position never changes."""
    max_sub = 5
    env = _make_env(*[(0.0, False, False)] * max_sub)
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    same = _locs((0, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=same):
        step_until_new_frame(env, last_pos, 1, max_substeps=max_sub)

    assert env.step.call_count == max_sub


# ---------------------------------------------------------------------------
# None / sentinel positions
# ---------------------------------------------------------------------------


def test_skips_none_ego_locations_and_keeps_stepping():
    """Frames where get_location returns ego=None are skipped (position not yet visible)."""
    env = _make_env(
        (0.0, False, False),  # substep 1 — ego None
        (0.0, False, False),  # substep 2 — valid new position
    )
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    no_ego = _locs(None, (5, 5))
    moved = _locs((1, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", side_effect=[no_ego, moved]):
        step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 2


def test_none_last_pos_breaks_on_first_valid_frame():
    """last_pos with ego=None means any valid frame is accepted immediately."""
    env = _make_env((0.0, False, False))
    last_pos = {"ego": None, "opp": None}
    valid = _locs((3, 7), (10, 20))

    with patch("surround.utils.env_state.get_location", return_value=valid):
        step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1


# ---------------------------------------------------------------------------
# Return value structure
# ---------------------------------------------------------------------------


def test_info_contains_location_key():
    """Returned info dict always has a 'location' key with ego/opp dicts."""
    env = _make_env((0.0, False, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    new_locs = _locs((1, 0), (6, 5))

    with patch("surround.utils.env_state.get_location", return_value=new_locs):
        _, _, _, _, info = step_until_new_frame(env, last_pos, 1)

    assert "location" in info
    assert info["location"]["ego"] == (1, 0)
    assert info["location"]["opp"] == (6, 5)


def test_returned_observation_is_a_copy():
    """The observation returned is a copy, not the original array."""
    original = np.ones((210, 160), dtype=np.uint8) * 42
    env = MagicMock()
    env.step.return_value = (original, 0.0, False, False, {})
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    new_locs = _locs((1, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=new_locs):
        obs, *_ = step_until_new_frame(env, last_pos, 1)

    assert obs is not original
    np.testing.assert_array_equal(obs, original)


def test_action_id_forwarded_to_env_step():
    """The action_id passed in is forwarded unchanged to env.step."""
    env = _make_env((0.0, False, False))
    last_pos = {"ego": None, "opp": None}
    valid = _locs((1, 0), (5, 5))

    with patch("surround.utils.env_state.get_location", return_value=valid):
        step_until_new_frame(env, last_pos, action_id=3)

    env.step.assert_called_once_with(3)


def test_opponent_move_also_triggers_new_frame():
    """A change in opponent position (not just ego) counts as a new frame."""
    env = _make_env((0.0, False, False))
    last_pos = {"ego": (0, 0), "opp": (5, 5)}
    opp_moved = _locs((0, 0), (6, 5))  # ego same, opp moved

    with patch("surround.utils.env_state.get_location", return_value=opp_moved):
        step_until_new_frame(env, last_pos, 1)

    assert env.step.call_count == 1
