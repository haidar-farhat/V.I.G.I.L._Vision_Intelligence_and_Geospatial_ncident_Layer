"""Shared fixtures.

The reference video is encoded once per test session rather than committed. It is
about 2.5 MB of generated content, it is fully determined by ``scene.py``, and a
binary in version control that can be regenerated exactly is a binary that will
eventually disagree with the code that generates it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import scene
from sentinel.core import CameraPose, LatLon

#: An arbitrary but fixed site, so map assertions have concrete numbers.
SITE = LatLon(33.8938, 35.5018)


@pytest.fixture(scope="session")
def reference_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real video file containing the synthetic reference scene."""
    path = tmp_path_factory.mktemp("media") / "reference.mp4"
    return scene.write_scene(path)


@pytest.fixture(scope="session")
def reference_pose() -> CameraPose:
    """A plausible pose for the camera that shot the reference scene.

    A 6 m mast tilted 22 degrees down, looking south. Chosen so the scene's
    objects land between roughly 8 m and 40 m from the camera — the range where
    projection is meaningful and its uncertainty is neither negligible nor
    absurd.
    """
    return CameraPose(
        position=SITE,
        mount_height=6.0,
        heading=180.0,
        pitch=-22.0,
        horizontal_fov=62.0,
        vertical_fov=36.0,
        range_meters=90.0,
    )


@pytest.fixture(autouse=True, scope="session")
def isolated_data_directory(tmp_path_factory: pytest.TempPathFactory):
    """Keep the suite out of the operator's real application data directory.

    Without this, running the tests writes a log — and, for anything that
    touches the default database path, a database — into the same folder a real
    installation uses. A test suite that pollutes the thing it is testing is a
    test suite that eventually destroys somebody's evidence.
    """
    directory = tmp_path_factory.mktemp("appdata")
    previous = os.environ.get("SENTINEL_DATA_DIR")
    os.environ["SENTINEL_DATA_DIR"] = str(directory)
    yield directory
    if previous is None:
        os.environ.pop("SENTINEL_DATA_DIR", None)
    else:
        os.environ["SENTINEL_DATA_DIR"] = previous


@pytest.fixture(autouse=True)
def isolated_keychain():
    """Keep every test out of the developer's real Credential Manager.

    Adding a network camera files its password in the operating system's
    keychain; a suite that did that for real would leave hundreds of entries
    behind and, worse, read one back into a test. An in-memory stand-in per
    test; `test_secrets.py` opts into the real one for exactly one round trip.
    """
    from sentinel import secrets

    secrets.use(secrets.InMemoryBackend())
    yield
    secrets.use(secrets.InMemoryBackend())
