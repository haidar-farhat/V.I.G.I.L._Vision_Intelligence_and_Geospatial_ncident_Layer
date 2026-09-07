from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from vigil.adapters.keychain import InMemoryBackend, Keychain
from vigil.domain.geo import CameraPose, LatLon

# The suites never touch the machine's keychain or its data folder.
os.environ.setdefault("VIGIL_ALERT_FILE", "")


@pytest.fixture(scope="session")
def qt_app():
    """One offscreen QApplication for the session. Qt allows exactly one."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


@pytest.fixture
def keychain() -> Keychain:
    return Keychain(InMemoryBackend())


@pytest.fixture
def pose() -> CameraPose:
    """A camera on a 4 m mast looking north and down, 62° wide."""
    return CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0, 0.0, 62.0, 36.0, 60.0)


def make_video(path: Path, *, frames: int = 90, fps: float = 15.0, size=(320, 240), walk=True) -> Path:
    """A block walks from the left edge to the centre and stays. Enough for motion to see."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    w, h = size
    for i in range(frames):
        image = np.full((h, w, 3), 40, dtype=np.uint8)
        noise = (np.random.default_rng(i).random((h, w, 1)) * 12).astype(np.uint8)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        x = min(w // 2, 10 + i * 4) if walk else w // 2
        cv2.rectangle(image, (x, h - 90), (x + 28, h - 20), (230, 230, 230), -1)
        writer.write(image)
    writer.release()
    return path


@pytest.fixture(scope="session")
def reference_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_video(tmp_path_factory.mktemp("video") / "walk.mp4")
