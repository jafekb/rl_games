from pathlib import Path
from typing import Set

import cv2
import numpy as np

from surround import constants

VIDEO_DIR = Path("video")
EXTRACT_DIR = Path("extract")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

X_SIZE = 9
Y_SIZE = 4

# Crop bounds for the game region
GAME_ROW_SLICE = slice(35, 197)
GAME_COL_SLICE = slice(4, 156)

VISUALIZE = False


def mask_to_grid_locations(mask: np.ndarray) -> Set[tuple[int, int]]:
    locations: Set[tuple[int, int]] = set()
    for row in range(constants.GRID_ROWS):
        row_start = row * X_SIZE
        row_end = row_start + X_SIZE
        for col in range(constants.GRID_COLS):
            col_start = col * Y_SIZE
            col_end = col_start + Y_SIZE
            cell = mask[row_start:row_end, col_start:col_end]
            if np.any(cell):
                locations.add((row, col))
    return locations


def get_location(image: np.ndarray) -> dict:
    """
    Extracts the locations of the ego, opponent, and walls from an image.

    Args:
        image: The input game image in BGR format.

    Returns:
        A dictionary containing the locations of the ego, opponent, and walls.
    """
    locations = {
        "ego": None,
        "opp": None,
        "walls": set(),
    }
    game = image[GAME_ROW_SLICE, GAME_COL_SLICE, :]
    ego = cv2.inRange(game, (90, 192, 180), (100, 197, 185))
    opponent = cv2.inRange(game, (70, 70, 195), (75, 75, 205))
    walls = cv2.inRange(game, (190, 100, 210), (200, 110, 220))
    x, y = np.where(ego)
    if x.size > 0 and y.size > 0:
        locations["ego"] = (int(x.min() // X_SIZE), int(y.min() // Y_SIZE))
    x, y = np.where(opponent)
    if x.size > 0 and y.size > 0:
        locations["opp"] = (int(x.min() // X_SIZE), int(y.min() // Y_SIZE))
    locations["walls"] = mask_to_grid_locations(walls)

    if VISUALIZE:
        cv2.imwrite(EXTRACT_DIR / "1_orig.png", image)
        cv2.imwrite(EXTRACT_DIR / "2_game.png", game)
        cv2.imwrite(EXTRACT_DIR / "3_ego.png", ego)
        cv2.imwrite(EXTRACT_DIR / "4_opponent.png", opponent)
        cv2.imwrite(EXTRACT_DIR / "5_walls.png", walls)

    return locations


def observation_to_class_map(observation: np.ndarray) -> np.ndarray:
    """
    Convert an RGB observation to a single (H, W) array with 4 pixel classes.

    Uses the same game crop and BGR inRange logic as get_location. Each pixel is
    assigned one of: 0=empty, 1=wall, 2=opponent, 3=ego (priority: ego > opponent > wall > empty).

    Args:
        observation: RGB image from the env, shape (height, width, 3).

    Returns:
        (H, W) array, dtype uint8, values in {0, 1, 2, 3}.
    """
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    game = frame[GAME_ROW_SLICE, GAME_COL_SLICE, :]
    walls = cv2.inRange(game, (190, 100, 210), (200, 110, 220))
    opponent = cv2.inRange(game, (70, 70, 195), (75, 75, 205))
    ego = cv2.inRange(game, (90, 192, 180), (100, 197, 185))
    out = np.zeros(game.shape[:2], dtype=np.uint8)
    out[walls.astype(bool)] = 1
    out[opponent.astype(bool)] = 2
    out[ego.astype(bool)] = 3
    return out


def main(images: list[Path]) -> None:
    for im_fn in images:
        image = cv2.imread(im_fn)
        locs = get_location(image)
        print(im_fn.stem, locs)

    print("Done!")


if __name__ == "__main__":
    IMAGES = list(VIDEO_DIR.glob("frame_*.png"))
    IMAGES = [i for i in IMAGES if int(i.stem.split("_")[-1]) < 115]
    IMAGES = sorted(IMAGES, key=lambda x: int(x.stem.split("_")[-1]))
    main(IMAGES)
