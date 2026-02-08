from typing import Set

import cv2

from surround.actions import ACTION_WORD_TO_ID, ACTION_WORDS_5
from surround.utils.video_extract_locations import get_location

GRID_ROWS = 18
GRID_COLS = 38


def get_action(
    locations: dict[str, tuple[int, int] | Set[tuple[int, int]] | None], last_action: str
) -> tuple[int, int]:
    collisions = locations["walls"] | {locations["opp"]} if locations["opp"] is not None else set()
    ego = locations["ego"]
    if ego is None:
        return "NOOP"
    collision_up = (ego[0] - 1, ego[1]) in collisions or ego[0] <= 0
    collision_down = (ego[0] + 1, ego[1]) in collisions or ego[0] >= GRID_ROWS - 1
    collision_left = (ego[0], ego[1] - 1) in collisions or ego[1] <= 0
    collision_right = (ego[0], ego[1] + 1) in collisions or ego[1] >= GRID_COLS - 1
    # This order is really important: there is more left-right space
    # than up-down space, so we want to move towards the opponent, then
    # snake back and forth covering the entire space.
    # This strategy wins every time on difficulty 0.
    if not collision_left and last_action != "RIGHT":
        return "LEFT"
    if not collision_right and last_action != "LEFT":
        return "RIGHT"
    if not collision_down and last_action != "UP":
        return "DOWN"
    if not collision_up and last_action != "DOWN":
        return "UP"
    return "NOOP"


def snake_policy(action_space, observation, info, last_action) -> int:
    # my extractor function inherits cv2's BGR convention.
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    locations = get_location(frame)
    action = get_action(locations, ACTION_WORDS_5[last_action])
    return ACTION_WORD_TO_ID[action]
