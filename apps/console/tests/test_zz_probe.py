from __future__ import annotations
import os, time
from pathlib import Path
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication
from sentinel_console.app import ConsoleWindow


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app):
    win = ConsoleWindow(":memory:")
    win.resize(1280, 800)
    win.show()
    yield win
    win.close()


def pump(app, window, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline and window._running:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def test_probe(qt_app, window, reference_video: Path):
    from sentinel.core import CameraPose, LatLon

    for camera_id in ("cam-07", "cam-08"):
        session = window.add_camera(reference_video, camera_id)
        session.pose = CameraPose(
            position=LatLon(33.8938, 35.5018),
            mount_height=6.0,
            heading=180.0,
            pitch=-22.0,
            horizontal_fov=62.0,
            vertical_fov=36.0,
            range_meters=90.0,
        )
    window._refresh_placement()
    window.zone_radius.setValue(10.0)
    window._add_zone()

    window._start()
    pump(qt_app, window, 16.0)
    window._stop()

    per_cam = {cid: len(s.events) for cid, s in window._sessions.items()}
    print("EVENTS PER CAMERA:", per_cam)
    print("TOTAL EVENTS:", sum(per_cam.values()))
    print("ROWS (correct impl):", window.incidents.topLevelItemCount())
    print("INCIDENTS:", [(i.id, i.cameras, i.opened_at_millis, i.closed_at_millis, len(i.events)) for i in window._incidents])

    # now simulate the regression: correlate per session
    from sentinel.incidents import Correlator
    regressed = []
    for s in window._sessions.values():
        if s.events:
            c = Correlator(zone_kinds={z.id: z.kind for z in window._zones})
            regressed.extend(c.correlate(s.events))
    print("REGRESSED COUNT:", len(regressed))
    print("REGRESSED:", [(i.id, i.cameras, len(i.events)) for i in regressed])
    window.incidents.show_incidents(regressed)
    print("REGRESSED ROWS:", window.incidents.topLevelItemCount())
