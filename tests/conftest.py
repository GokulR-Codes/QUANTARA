"""Shared fixtures for unit and integration tests."""
import pytest

from tests.helpers.bundles import build_synthetic_bundle


@pytest.fixture(scope="session")
def settings():
    from src.config.settings import get_settings
    return get_settings()


@pytest.fixture(scope="session")
def synthetic_bundle_dir(tmp_root, settings):
    """A SYNTHETIC-TEST pipeline bundle (plumbing only; see tests/helpers/bundles.py)."""
    return build_synthetic_bundle(tmp_root / "bundles" / "synthetic", settings)
