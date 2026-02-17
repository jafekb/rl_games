"""
Test that state extraction is equivalent for grayscale and RGB observation modes.

get_state_from_observation() is grayscale-only (get_location asserts ndim == 2);
we dropped backwards compatibility for RGB. We therefore cannot call it on RGB
directly. This test proves equivalence by converting RGB to grayscale outside the
function and asserting both inputs yield the same state.

Analytical basis: get_location() uses only grayscale intensity thresholds
(EGO_GRAY=179, OPP_GRAY=110, WALLS_GRAY=149). So for the same frame, grayscale
obs and cv2.COLOR_RGB2GRAY(rgb_obs) yield the same state.
"""

import cv2
import numpy as np
import pytest

from surround import constants
from surround.dqn.train_dqn import get_state_from_observation
from surround.utils.env_state import make_env


def _rgb_to_grayscale(rgb_obs: np.ndarray) -> np.ndarray:
    """Convert RGB observation to grayscale (same convention as get_location expects)."""
    return cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2GRAY)


def test_state_extraction_grayscale_and_rgb_equivalent(ale_available):
    """
    Same seed and same actions → same trajectory. At each timestep, state from
    grayscale observation equals state from RGB observation converted to grayscale.
    """
    if not ale_available:
        pytest.skip("ALE Surround ROM not available")
    seed = 42
    num_steps = 10
    difficulty = constants.DIFFICULTY
    mode = constants.MODE
    frameskip = constants.DQN_FRAME_SKIP

    env_grayscale = make_env(difficulty, mode, obs_type="grayscale", frameskip=frameskip)
    env_rgb = make_env(difficulty, mode, obs_type="rgb", frameskip=frameskip)
    try:
        obs_g, _ = env_grayscale.reset(seed=seed)
        obs_rgb, _ = env_rgb.reset(seed=seed)

        # Initial last_action (e.g. LEFT = 1) to match typical training
        last_action = 1

        for t in range(num_steps):
            state_from_grayscale = get_state_from_observation(obs_g, last_action)
            gray_from_rgb = _rgb_to_grayscale(obs_rgb)
            state_from_rgb = get_state_from_observation(gray_from_rgb, last_action)

            assert state_from_grayscale == state_from_rgb, (
                f"At step {t}: grayscale state {state_from_grayscale} != "
                f"state from RGB {state_from_rgb}"
            )

            # Same action for both envs so trajectories stay in sync
            action_id = (t % 4) + 1  # 1..4
            obs_g, _, _, _, _ = env_grayscale.step(action_id)
            obs_rgb, _, _, _, _ = env_rgb.step(action_id)
            last_action = action_id
    finally:
        env_grayscale.close()
        env_rgb.close()
