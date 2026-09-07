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
