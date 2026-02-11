"""Tests for surround.utils.env_state."""

import pytest

from surround.utils.env_state import (
    get_state_tuple,
    total_possible_states,
)


def test_total_possible_states_state_tuple():
    """state_tuple mode returns 576 (2*2*2*2*3*3*4)."""
    assert total_possible_states("state_tuple") == 576


def test_total_possible_states_unknown_raises():
    """Unknown state mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown state mode"):
        total_possible_states("unknown")


def test_get_state_tuple_none_ego_returns_sentinel():
    """When ego or opp is None, returns (1,1,1,1,1,1, last_action)."""
    locations = {"ego": None, "opp": (1, 1), "walls": set()}
    assert get_state_tuple(locations, 3) == (1, 1, 1, 1, 1, 1, 3)

    locations = {"ego": (0, 0), "opp": None, "walls": set()}
    assert get_state_tuple(locations, 1) == (1, 1, 1, 1, 1, 1, 1)


def test_get_state_tuple_known_locations():
    """get_state_tuple with known locations returns correct structure."""
    # Ego at (1, 1), opp at (0, 0) (above and left), no walls adjacent
    locations = {
        "ego": (1, 1),
        "opp": (0, 0),
        "walls": set(),
    }
    state = get_state_tuple(locations, 2)
    assert len(state) == 7
    d_up, d_right, d_left, d_down, rel_x, rel_y, last_action = state
    assert last_action == 2
    assert rel_x == 0  # opp left of ego
    assert rel_y == 0  # opp above ego
    assert d_up in (0, 1)
    assert d_right in (0, 1)
    assert d_left in (0, 1)
    assert d_down in (0, 1)


def test_get_state_tuple_with_custom_grid():
    """get_state_tuple accepts grid_rows/grid_cols."""
    locations = {"ego": (1, 1), "opp": (1, 2), "walls": set()}
    state = get_state_tuple(locations, 1, grid_rows=18, grid_cols=38)
    assert state[4] == 2  # rel_x: opp right of ego
    assert state[5] == 1  # rel_y: same row
