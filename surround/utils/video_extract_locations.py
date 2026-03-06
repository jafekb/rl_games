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


# Grayscale intensity ranges for ego/opponent/walls/(background)
EGO_GRAY = 179
OPP_GRAY = 110
WALLS_GRAY = 149
BACKGROUND_GRAY = 104


def get_location(image: np.ndarray) -> dict:
    """
    Extracts the locations of the ego, opponent, and walls from a grayscale image.

    Args:
        image: Grayscale game image (H, W).

    Returns:
        A dictionary containing the locations of the ego, opponent, and walls.
    """
    assert image.ndim == 2, "Image must be grayscale (H, W)."
    locations = {
        "ego": None,
        "opp": None,
        "walls": set(),
    }
    game = image[constants.GAME_ROW_SLICE, constants.GAME_COL_SLICE]
    ego = (game == EGO_GRAY).astype(np.uint8) * 255
    opponent = (game == OPP_GRAY).astype(np.uint8) * 255
    walls = (game == WALLS_GRAY).astype(np.uint8) * 255
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
    Convert a grayscale observation to a single (H, W) array with 4 pixel classes.

    Uses the same game crop and grayscale intensity constants as get_location.
    Each pixel is assigned one of: 0=empty, 1=wall, 2=opponent, 3=ego
    (priority: ego > opponent > wall > empty).

    Args:
        observation: Grayscale image from the env, shape (height, width).

    Returns:
        (H, W) array, dtype float32, values in {0, 1/3, 2/3, 1} for
        {empty, wall, opponent, ego}.
    """
    assert observation.ndim == 2, "Observation must be grayscale (H, W)."
    game = observation[constants.GAME_ROW_SLICE, constants.GAME_COL_SLICE]
    out = np.zeros(game.shape, dtype=np.uint8)
    out[game == WALLS_GRAY] = 1
    out[game == OPP_GRAY] = 2
    out[game == EGO_GRAY] = 3
    return out.astype(np.float32) / 3.0


def main(images: list[Path]) -> None:
    """Run location extraction on image files.
    Converts BGR from disk to grayscale for get_location."""
    for im_fn in images:
        image = cv2.imread(im_fn)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        locs = get_location(gray)
        print(im_fn.stem, locs)

    print("Done!")


if __name__ == "__main__":
    IMAGES = list(VIDEO_DIR.glob("frame_*.png"))
    IMAGES = [i for i in IMAGES if int(i.stem.split("_")[-1]) < 115]
    IMAGES = sorted(IMAGES, key=lambda x: int(x.stem.split("_")[-1]))
    main(IMAGES)
