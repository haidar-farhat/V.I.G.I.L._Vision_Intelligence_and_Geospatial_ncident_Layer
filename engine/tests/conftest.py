"""Shared fixtures.

The reference video is encoded once per test session rather than committed. It is
about 2.5 MB of generated content, it is fully determined by ``scene.py``, and a
binary in version control that can be regenerated exactly is a binary that will
eventually disagree with the code that generates it.
"""

from __future__ import annotations

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
