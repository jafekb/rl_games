from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path("video")
EXTRACT_DIR = Path("extract")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

X_SIZE = 9
Y_SIZE = 4
GRID_ROWS = 18
GRID_COLS = 38

VISUALIZE = False

COLOR_TOLERANCE = 5
EGO_HEAD_BGR = (95, 194, 183)
OPP_HEAD_BGR = (72, 72, 200)
WALL_BGR = (195, 108, 212)


def _mask_color(image: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    lower = np.clip(np.array(color_bgr) - COLOR_TOLERANCE, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(color_bgr) + COLOR_TOLERANCE, 0, 255).astype(np.uint8)
    return cv2.inRange(image, lower, upper)


def _to_bgr(image: np.ndarray, color_space: str) -> np.ndarray:
    if color_space.lower() == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def mask_to_grid_locations(mask: np.ndarray) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for row in range(GRID_ROWS):
        row_start = row * X_SIZE
        row_end = row_start + X_SIZE
        for col in range(GRID_COLS):
            col_start = col * Y_SIZE
            col_end = col_start + Y_SIZE
            cell = mask[row_start:row_end, col_start:col_end]
            if np.any(cell):
                locations.append((row, col))
    return locations


def get_location(
    image: np.ndarray,
    *,
    color_space: str = "bgr",
) -> dict[str, tuple[int, int] | list[tuple[int, int]] | None]:
    locations: dict[str, tuple[int, int] | list[tuple[int, int]] | None] = {
        "ego": None,
        "opp": None,
        "walls": None,
    }
    image_bgr = _to_bgr(image, color_space)
    game = image_bgr[35:197, 4:156, :]
    ego = _mask_color(game, EGO_HEAD_BGR)
    opponent = _mask_color(game, OPP_HEAD_BGR)
    walls = _mask_color(game, WALL_BGR)
    x, y = np.where(ego)
    if x.size > 0 and y.size > 0:
        locations["ego"] = (int(x.min() // X_SIZE), int(y.min() // Y_SIZE))
    x, y = np.where(opponent)
    if x.size > 0 and y.size > 0:
        locations["opp"] = (int(x.min() // X_SIZE), int(y.min() // Y_SIZE))
    locations["walls"] = mask_to_grid_locations(walls)

    if VISUALIZE:
        cv2.imwrite(EXTRACT_DIR / "1_orig.png", image_bgr)
        cv2.imwrite(EXTRACT_DIR / "2_game.png", game)
        cv2.imwrite(EXTRACT_DIR / "3_ego.png", ego)
        cv2.imwrite(EXTRACT_DIR / "4_opponent.png", opponent)
        cv2.imwrite(EXTRACT_DIR / "5_walls.png", walls)

    return locations


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
