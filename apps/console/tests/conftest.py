"""Fixtures for the console tests.

The reference video is the same one the engine tests use: a console that renders
correctly on footage the engine has never seen is not evidence of much.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Set before any Qt import, or Qt binds to a display that is not there.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import scene  # noqa: E402


@pytest.fixture(scope="session")
def reference_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "reference.mp4"
    return scene.write_scene(path)
