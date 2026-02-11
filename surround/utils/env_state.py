"""Environment creation and state extraction utilities for Surround."""

from __future__ import annotations

import ale_py
import cv2
import gymnasium as gym
import numpy as np

from surround import constants
from surround.utils.video_extract_locations import get_location


def total_possible_states(state_mode: str) -> int:
    if state_mode == "state_tuple":
        return 576  # 2*2*2*2*3*3*4
    raise ValueError(f"Unknown state mode: {state_mode}")


def make_env(
    difficulty: int,
    mode: int,
    *,
    obs_type: str = "rgb",
    frameskip: int | None = 8,
    render_mode: str | None = None,
):
    gym.register_envs(ale_py)
    kwargs = {
        "obs_type": obs_type,
        "full_action_space": False,
        "difficulty": difficulty,
        "mode": mode,
    }
    if frameskip is not None:
        kwargs["frameskip"] = frameskip
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make("ALE/Surround-v5", **kwargs)


def get_state_tuple(
    locations,
    last_action: int,
    *,
    grid_rows: int = constants.GRID_ROWS,
    grid_cols: int = constants.GRID_COLS,
) -> tuple[int, ...]:
    """
    Builds a state tuple from the locations of the ego, opponent, and walls.

    Args:
        locations: The locations of the ego, opponent, and walls.
        last_action: Last action id (1..4 for UP/RIGHT/LEFT/DOWN).
        grid_rows: Number of grid rows (default from conf).
        grid_cols: Number of grid columns (default from conf).

    Returns:
        A tuple of the state (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action).
        There are a total of 2*2*2*2*3*3*4 = 576 possible states:
        - d_up: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_right: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_left: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - d_down: 1 if the ego is adjacent to a wall or out-of-bounds, 0 otherwise
        - rel_x: 0 if the opponent is to the left of the ego, 1 if the opponent is in the
            same column as the ego, 2 if the opponent is to the right of the ego.
        - rel_y: 0 if the opponent is above the ego, 1 if the opponent is in the same row as
            the ego, 2 if the opponent is below the ego
        - last_action: 1..4 for UP/RIGHT/LEFT/DOWN
    """
    if locations["ego"] is None or locations["opp"] is None:
        # Sometimes this happens if the game ends.
        return (1, 1, 1, 1, 1, 1, last_action)
    ego_row, ego_col = locations["ego"]
    opp_row, opp_col = locations["opp"]
    wall_set = locations["walls"]
    collisions = (
        wall_set | {(opp_row, opp_col)} if opp_row is not None and opp_col is not None else wall_set
    )

    # 1. Survival Features (4 Binary Flags)
    # Check adjacent tiles for walls or trails
    d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
    d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= grid_cols - 1 else 0
    d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
    d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= grid_rows - 1 else 0

    # 2. Relational Features (Map -1, 0, 1 to 0, 1, 2)
    # We shift the values so they are non-negative for the index math
    if opp_col < ego_col:
        rel_x = 0  # Opponent is Left
    elif opp_col > ego_col:
        rel_x = 2  # Opponent is Right
    else:
        rel_x = 1  # Same Column

    if opp_row < ego_row:
        rel_y = 0  # Opponent is Above
    elif opp_row > ego_row:
        rel_y = 2  # Opponent is Below
    else:
        rel_y = 1  # Same Row

    return (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)


def build_state_from_observation(
    observation: np.ndarray,
    last_action: int,
    *,
    state_mode: str,
    debug_state: bool = False,
) -> tuple[int, ...]:
    """
    Builds a state from an observation.

    Args:
        observation: The observation to build a state from.
        last_action: Last action id (1..4).
        state_mode: The state mode to use.
        debug_state: If True, print the state tuple to stdout.

    Returns:
        A tuple of the state (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action).
    """
    if state_mode == "ram":
        raise ValueError("RAM state mode is not supported.")
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    locations = get_location(frame)
    state = get_state_tuple(locations, last_action)
    if debug_state:
        print("state_tuple", state)
    return state
