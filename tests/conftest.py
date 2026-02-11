"""Pytest configuration and shared fixtures."""

import pytest


def _ale_env_available() -> bool:
    """Return True if ALE Surround env can be created (ROM installed)."""
    try:
        from surround.utils.env_state import make_env

        env = make_env(0, 0, obs_type="ram", frameskip=None)
        env.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def ale_available() -> bool:
    """Whether ALE/Surround env is available (for integration tests)."""
    return _ale_env_available()
