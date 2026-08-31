"""Tests for the console.

Run headless with the offscreen Qt platform, so they work in CI and over SSH.
What is checked here is not "does it look right" — that needs eyes — but the
things a screenshot cannot tell you and a user would only discover in the worst
moment:

- An unplaced camera reports **no** position, rather than a plausible-looking one.
- A track box is never labelled with a class the detector cannot produce.
- The analysis does not run on the UI thread.
- A password never reaches the interface, by any route.
- Closing the window while a camera is running does not leave a thread behind.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import scene  # noqa: E402
from sentinel.core import CameraPose, LatLon  # noqa: E402
from sentinel.decode import VideoSource  # noqa: E402
from sentinel.detect import MotionDetector  # noqa: E402
from sentinel_console import theme  # noqa: E402
from sentinel_console.app import ConsoleWindow  # noqa: E402
from sentinel_console.map_view import MapView  # noqa: E402
from sentinel_console.placement import PlacementDialog  # noqa: E402
from sentinel_console.video_view import VideoView  # noqa: E402
from sentinel_console.worker import AnalysisWorker  # noqa: E402

POSITION_COLUMN = 7
UNCERTAINTY_COLUMN = 8
SOURCE_COLUMN = 9


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app):
    win = ConsoleWindow()
    win.resize(1280, 800)
    # Shown, because a child widget's isVisible() is False while its top-level
    # window is hidden — a test against an unshown window cannot tell a widget
    # that is correctly displayed from one that is not.
    win.show()
    yield win
    win.close()


def pump(app, window, seconds: float) -> None:
    """Run the event loop as a real session would."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline and window._worker is not None:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


# ------------------------------------------------------------------- placement


def test_a_new_console_has_no_camera_placement(window):
    # No nominal origin, no "0, 0". A position on a map gets someone sent to it.
    assert window._pose is None


def test_an_unplaced_camera_reports_no_position(qt_app, window, reference_video: Path):
    window._source_path = reference_video
    window.start_button.setEnabled(True)
    window._start()
    pump(qt_app, window, 4.0)

    assert window.tracks.topLevelItemCount() > 0, "nothing was tracked, so nothing was checked"
    for index in range(window.tracks.topLevelItemCount()):
        row = window.tracks.topLevelItem(index)
        assert row.text(POSITION_COLUMN) == "not placed"
        assert row.text(SOURCE_COLUMN) == "no pose"

    window._stop()


def test_placing_a_camera_mid_run_produces_positions(qt_app, window, reference_video: Path):
    window._source_path = reference_video
    window.start_button.setEnabled(True)
    window._start()
    pump(qt_app, window, 3.0)

    window._pose = CameraPose(
        position=LatLon(33.8938, 35.5018),
        mount_height=6.0,
        heading=180.0,
        pitch=-22.0,
        horizontal_fov=62.0,
        vertical_fov=36.0,
        range_meters=90.0,
    )
    window.map.set_pose(window._pose)
    window._worker.set_pose(window._pose)
    pump(qt_app, window, 4.0)

    row = window.tracks.topLevelItem(0)
    assert row is not None
    assert row.text(POSITION_COLUMN) != "not placed"
    assert row.text(SOURCE_COLUMN) == "projected"
    assert row.text(UNCERTAINTY_COLUMN).startswith("±")

    window._stop()


def test_the_placement_dialog_reports_what_the_camera_can_actually_see(qt_app):
    # A 6 m mast tilted 22 degrees down with a 36-degree vertical field sees the
    # ground from 6/tan(40°) = 7.2 m to 6/tan(4°) = 86 m. The near limit is the
    # part installers are surprised by: the camera is blind at its own feet, and
    # nothing within 7 m of the mast is covered at all.
    dialog = PlacementDialog()
    dialog.mount_height.setValue(6.0)
    dialog.pitch.setValue(-22.0)
    dialog.vertical_fov.setValue(36.0)
    dialog.range_meters.setValue(90.0)

    text = dialog._warning.text()
    assert "blind closer than 7 m" in text
    assert "86 m" in text


def test_a_steeper_camera_covers_a_much_shallower_band(qt_app):
    # The same mast at 45 degrees covers 4 m to 12 m — a band a seventh as deep.
    # Pitch is the single most consequential number in a placement, and this is
    # where an installer should discover that.
    dialog = PlacementDialog()
    dialog.mount_height.setValue(6.0)
    dialog.pitch.setValue(-45.0)
    dialog.vertical_fov.setValue(36.0)
    dialog.range_meters.setValue(90.0)

    assert "3 m to 12 m" in dialog._warning.text()


def test_a_camera_pointed_at_the_horizon_is_called_out(qt_app):
    dialog = PlacementDialog()
    dialog.pitch.setValue(0.0)
    dialog.vertical_fov.setValue(36.0)

    assert "above the horizon" in dialog._warning.text()


def test_the_dialog_returns_the_pose_it_was_given(qt_app):
    original = CameraPose(
        position=LatLon(1.25, -3.5),
        mount_height=9.5,
        heading=145.0,
        pitch=-31.0,
        horizontal_fov=48.0,
        vertical_fov=27.0,
        range_meters=140.0,
    )
    restored = PlacementDialog(original).pose()

    assert restored.position.lat == pytest.approx(original.position.lat)
    assert restored.position.lon == pytest.approx(original.position.lon)
    assert restored.mount_height == pytest.approx(original.mount_height)
    assert restored.heading == pytest.approx(original.heading)
    assert restored.pitch == pytest.approx(original.pitch)
    assert restored.range_meters == pytest.approx(original.range_meters)


# ------------------------------------------------------------------ honesty


def test_a_track_is_never_labelled_with_a_class_the_detector_cannot_produce(qt_app):
    view = VideoView()
    view.set_detector_info(MotionDetector().info)

    class FakeTrack:
        id = 4
        class_id = 9999
        speed_mps = 1.4
        position = None

    label = view._label_for(FakeTrack())

    assert label.startswith("#4")
    assert "person" not in label
    assert "unclassified" not in label, (
        "repeating 'unclassified' on every box is noise; the identity alone "
        "claims nothing, and the detector is named in the toolbar"
    )


def test_an_unplaced_map_says_so_rather_than_drawing_an_empty_grid(qt_app):
    view = MapView()
    view.resize(300, 300)

    assert view._pose is None
    # Rendering must not raise with no pose, and must not invent a footprint.
    view.grab()
    assert view._footprint == []


def test_the_map_fetches_nothing(qt_app):
    # The strongest form this can take without a network sandbox: the module
    # references no URL, no tile server, and no HTTP client at all.
    source = Path(MapView.__module__.replace(".", "/"))
    text = (Path(__file__).parents[1] / "sentinel_console" / "map_view.py").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()

    for forbidden in ("http://", "https://", "requests.", "urllib", "qnetwork", "tile.", "{z}/{x}/{y}"):
        assert forbidden not in lowered, f"the map view references {forbidden}"


def test_no_part_of_the_console_embeds_a_browser(qt_app):
    package = Path(__file__).parents[1] / "sentinel_console"
    for module in package.glob("*.py"):
        text = module.read_text(encoding="utf-8").lower()
        assert "qtwebengine" not in text, f"{module.name} pulls in a browser engine"
        assert "qwebview" not in text, f"{module.name} pulls in a browser engine"


# ------------------------------------------------------------------- threading


def test_the_analysis_does_not_run_on_the_ui_thread(qt_app, reference_video: Path):
    # A decode loop on the UI thread freezes the interface, including the button
    # that stops it. Subclassed rather than monkeypatched: QThread dispatches to
    # the class's run(), so assigning an instance attribute would silently not
    # be called and the test would pass while proving nothing.
    seen: list[object] = []

    class Observed(AnalysisWorker):
        def run(self):
            seen.append(QThread.currentThread())
            super().run()

    worker = Observed(VideoSource(reference_video), MotionDetector(), realtime=False)
    worker.start()

    deadline = time.perf_counter() + 5.0
    while not seen and time.perf_counter() < deadline:
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    worker.stop()
    worker.wait(5000)

    assert seen, "the worker never started"
    assert seen[0] is not QThread.currentThread()
    assert seen[0] is worker, "run() must execute on the worker's own thread"


def test_the_newest_result_wins_rather_than_a_backlog_building(qt_app, reference_video: Path):
    # Never reading from the worker must not accumulate frames.
    worker = AnalysisWorker(VideoSource(reference_video), MotionDetector(), realtime=False)
    worker.start()

    deadline = time.perf_counter() + 5.0
    while worker._skipped < 5 and time.perf_counter() < deadline:
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)

    skipped = worker._skipped
    worker.stop()
    worker.wait(5000)

    assert skipped > 0, "nothing was skipped, so the drop path never ran"
    # Exactly one result is retained no matter how many were produced.
    assert worker.take_latest() is not None
    assert worker.take_latest() is None


def test_closing_the_window_stops_the_analysis(qt_app, reference_video: Path):
    win = ConsoleWindow()
    win._source_path = reference_video
    win.start_button.setEnabled(True)
    win._start()
    pump(qt_app, win, 2.0)

    worker = win._worker
    assert worker is not None and worker.isRunning()

    win.close()
    assert not worker.isRunning(), "a thread outlived the window that owned it"


# -------------------------------------------------------------------- redaction


def test_no_credential_reaches_the_interface(qt_app):
    secret = "hunter2-not-a-real-password"
    url = f"rtsp://admin:{secret}@10.20.30.40:554/Streaming/Channels/101"

    worker = AnalysisWorker(VideoSource(url, source_id="cam-07"), MotionDetector())

    for text in (worker.display_url, worker.source_id, repr(worker._source)):
        assert secret not in text


def test_a_failed_source_reports_in_place_rather_than_in_a_modal(qt_app, window):
    # A modal here would block the operator from looking at the cameras that are
    # still working — and twenty cameras drop together when a switch loses power.
    # This test would hang forever if a modal were opened, which is exactly what
    # an operator would experience.
    message = "rtsp://admin:***@203.0.113.99:554/none is not reachable"
    window._on_failed(message)

    assert window.fault_label.isVisible()
    assert message in window.fault_label.text()
    assert "***" in window.status.currentMessage()


def test_a_new_run_clears_a_previous_fault(qt_app, window, reference_video: Path):
    # A stale error beside a healthy camera is worse than no error at all.
    window._on_failed("something went wrong earlier")
    assert window.fault_label.isVisible()

    window._source_path = reference_video
    window.start_button.setEnabled(True)
    window._start()
    assert not window.fault_label.isVisible()
    window._stop()


# ----------------------------------------------------------------------- theme


def test_state_colours_are_distinct():
    # An operator reads colour before text. A colour meaning two things is a bug.
    states = [theme.LIVE, theme.STALE, theme.FAULT, theme.IDLE]
    assert len({colour.name() for colour in states}) == len(states)


def test_evidence_and_inference_are_drawn_differently():
    assert theme.TRACK.name() != theme.TRACK_COASTING.name()
    assert theme.DETECTION.name() != theme.TRACK.name()


# ----------------------------------------------------------------- incidents


def _incident_from_events(count: int = 3):
    """A real incident, built by the real correlator from real events."""
    from datetime import datetime, timezone

    from sentinel.core import LatLon
    from sentinel.events import Event, Evidence, EventType, Severity
    from sentinel.incidents import Correlator

    site = LatLon(33.8938, 35.5018)
    events = []
    for index in range(count):
        evidence = Evidence(
            camera_id="cam-07",
            track_id=index + 1,
            first_seen_millis=index * 500,
            last_seen_millis=index * 500 + 2000,
            observations=12,
            detector="MOG2 background subtraction",
            detector_classifies=False,
            model_digest=None,
            class_label="unclassified",
            latitude=site.lat,
            longitude=site.lon,
            position_uncertainty_meters=1.5,
            position_source="GROUND_PROJECTION",
            speed_mps=1.1,
            heading_degrees=90.0,
            frame_indices=(index,),
        )
        events.append(
            Event(
                id=f"ev_{index}",
                type=EventType.ZONE_ENTRY,
                severity=Severity.HIGH,
                summary="An object entered Restricted Area A",
                occurred_at_millis=index * 500,
                occurred_at=datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc),
                zone_id="zone-a",
                zone_name="Restricted Area A",
                rule_id="zone-entry",
                evidence=evidence,
                triggering_conditions=("membership held for 600 ms",),
                confidence=1.0,
            )
        )
    return Correlator().correlate(events)


def test_the_incident_panel_shows_one_row_for_many_events(qt_app):
    # The product. Three events describing one situation must not be three rows.
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    incidents = _incident_from_events(3)
    view.show_incidents(incidents)

    assert len(incidents) == 1
    assert view.topLevelItemCount() == 1


def test_an_incident_row_shows_its_object_count_and_risk(qt_app):
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    view.show_incidents(_incident_from_events(3))
    row = view.topLevelItem(0)

    assert row.text(2) == "3", "the object count is what an operator triages on"
    assert float(row.text(4)) > 0.0


def test_an_incident_carries_its_reasoning_as_children(qt_app):
    # A risk score with no reasons attached is a number an operator learns to
    # ignore. Every factor must be reachable without leaving the panel.
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    incidents = _incident_from_events(3)
    view.show_incidents(incidents)
    row = view.topLevelItem(0)

    children = [row.child(i).text(2) for i in range(row.childCount())]
    assert any("severity" in text for text in children)
    assert any("entered Restricted Area A" in text for text in children)
    assert row.childCount() >= len(incidents[0].events)


def test_the_panel_never_claims_people_from_motion_blobs(qt_app):
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    view.show_incidents(_incident_from_events(3))
    row = view.topLevelItem(0)

    assert "object" in row.text(0)
    assert "person" not in row.text(0) and "people" not in row.text(0)


def test_an_expanded_incident_stays_expanded_across_a_refresh(qt_app):
    # An operator reading an incident must not have it collapse under them when
    # a new event arrives.
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    incidents = _incident_from_events(3)
    view.show_incidents(incidents)
    view.topLevelItem(0).setExpanded(True)

    view.show_incidents(incidents)
    assert view.topLevelItem(0).isExpanded()


def test_incidents_are_listed_most_serious_first(qt_app):
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    first = _incident_from_events(1)
    second = _incident_from_events(3)
    view.show_incidents(first + second)

    severities = [view.topLevelItem(i).text(1) for i in range(view.topLevelItemCount())]
    ranked = sorted(severities, key=lambda s: ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s), reverse=True)
    assert severities == ranked


def test_a_zone_needs_a_placed_camera(qt_app, window):
    # A zone is an area on the ground. With no camera placed there is nothing to
    # measure it against, and creating one anyway would produce alerts nobody
    # can act on.
    assert window._pose is None
    assert window._zones == []
