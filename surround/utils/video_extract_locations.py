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
    locations = {
        "ego": None,
        "opp": None,
        "walls": set(),
    }
    assert image.ndim == 2, "Image must be grayscale (H, W)."
    game = image[35:198, 4:156]
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


def main(images: list[Path]) -> None:
    """Run location extraction on image files (BGR from disk). Live env gives grayscale."""
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
