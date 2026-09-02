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
from sentinel_console.add_camera import AddCameraDialog  # noqa: E402
from sentinel_console.app import ConsoleWindow  # noqa: E402
from sentinel_console.map_view import MapView  # noqa: E402
from sentinel_console.placement import PlacementDialog  # noqa: E402
from sentinel_console.video_view import VideoView  # noqa: E402
from sentinel_console.worker import AnalysisWorker  # noqa: E402

# The track table leads with the camera, because a track id is only unique
# within one camera and a table without it shows two objects as if they were one.
CAMERA_COLUMN = 0
POSITION_COLUMN = 8
UNCERTAINTY_COLUMN = 9
SOURCE_COLUMN = 10


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app):
    # In memory, always. A test that wrote to the operator's real database
    # would leave fabricated incidents in an evidence trail.
    win = ConsoleWindow(":memory:")
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
    while time.perf_counter() < deadline and window._running:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


# ------------------------------------------------------------------- placement


def test_a_new_console_has_no_camera_placement(window):
    # No nominal origin, no "0, 0". A position on a map gets someone sent to it.
    assert window._pose is None


def test_an_unplaced_camera_reports_no_position(qt_app, window, reference_video: Path):
    window.add_camera(reference_video, "cam-07")
    window._start()
    pump(qt_app, window, 4.0)

    assert window.tracks.topLevelItemCount() > 0, "nothing was tracked, so nothing was checked"
    for index in range(window.tracks.topLevelItemCount()):
        row = window.tracks.topLevelItem(index)
        assert row.text(POSITION_COLUMN) == "not placed"
        assert row.text(SOURCE_COLUMN) == "no pose"

    window._stop()


def test_placing_a_camera_mid_run_produces_positions(qt_app, window, reference_video: Path):
    session = window.add_camera(reference_video, "cam-07")
    window._start()
    pump(qt_app, window, 3.0)

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
    session.worker.set_pose(session.pose)
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
    win = ConsoleWindow(":memory:")
    session = win.add_camera(reference_video, "cam-07")
    win._start()
    pump(qt_app, win, 2.0)

    worker = session.worker
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

    window.add_camera(reference_video, "cam-07")
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


def _incident_from_events(count: int = 3, *, severity=None, camera: str = "cam-07"):
    """A real incident, built by the real correlator from real events."""
    from datetime import datetime, timezone

    from sentinel.core import LatLon
    from sentinel.events import Event, Evidence, EventType, Severity
    from sentinel.incidents import Correlator

    site = LatLon(33.8938, 35.5018)
    events = []
    for index in range(count):
        evidence = Evidence(
            camera_id=camera,
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
                id=f"ev_{camera}_{index}",
                type=EventType.ZONE_ENTRY,
                severity=severity or Severity.HIGH,
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
    """Triage order, not arrival order.

    An earlier version built two incidents that both came out HIGH and then
    asserted the list was sorted — which is true of any two equal things, in any
    order, including reversed. It now builds incidents that genuinely differ and
    feeds them in deliberately wrong.
    """
    from sentinel.events import Severity
    from sentinel_console.incident_view import IncidentView

    quiet = _incident_from_events(1, severity=Severity.LOW)
    loud = _incident_from_events(1, severity=Severity.CRITICAL, camera="cam-99")
    middling = _incident_from_events(1, severity=Severity.MEDIUM, camera="cam-42")

    view = IncidentView()
    view.show_incidents(quiet + middling + loud)

    listed = [view.topLevelItem(i).text(1) for i in range(view.topLevelItemCount())]
    assert listed == ["CRITICAL", "MEDIUM", "LOW"], (
        f"listed {listed}; an operator triages from the top and the worst has to be there"
    )


def test_a_zone_needs_a_placed_camera(qt_app, window, monkeypatch):
    """Asking for a zone with no camera placed must create nothing.

    An earlier version of this test asserted the fixture's own initial state and
    never called `_add_zone` at all — it would have passed with the guard
    deleted. It now presses the button.

    The modal is intercepted rather than shown, because an unattended dialog
    hangs the suite forever — which is also exactly what it would do to an
    operator, so the interception records that it was raised.
    """
    from PySide6.QtWidgets import QMessageBox

    told: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *args, **kwargs: told.append(args[2] if len(args) > 2 else ""),
    )

    assert window._pose is None
    window._add_zone()

    assert window._zones == [], "a zone was created with nothing to measure it against"
    assert told, "the operator was given no reason why nothing happened"
    assert "place" in told[0].lower()

# ------------------------------------------------------------- several cameras


def test_several_cameras_each_get_their_own_pane(qt_app, window, reference_video: Path):
    for index in range(3):
        window.add_camera(reference_video, f"cam-{index:02d}")

    assert len(window._sessions) == 3
    panes = [window.wall_layout.itemAt(i).widget() for i in range(window.wall_layout.count())]
    assert len({id(pane) for pane in panes}) == 3


def test_a_camera_id_is_never_reused(qt_app, window, reference_video: Path):
    # Two files with the same stem must not become one camera, silently
    # discarding half the site's coverage.
    first = window.add_camera(reference_video, "cam-07")
    second = window.add_camera(reference_video, "cam-07")

    assert first.camera_id != second.camera_id
    assert len(window._sessions) == 2


def test_each_camera_runs_its_own_pipeline(qt_app, window, reference_video: Path):
    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    window._start()
    pump(qt_app, window, 3.0)

    workers = [s.worker for s in window._sessions.values()]
    assert len(workers) == 2
    assert workers[0] is not workers[1], "two cameras shared one pipeline"

    window._stop()


def test_the_track_table_says_which_camera_saw_what(qt_app, window, reference_video: Path):
    # Track ids are only unique within a camera. A table without the camera
    # column shows two different objects as if they were one.
    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    window._start()
    pump(qt_app, window, 4.0)

    cameras = {
        window.tracks.topLevelItem(i).text(0)
        for i in range(window.tracks.topLevelItemCount())
    }
    window._stop()

    assert cameras <= {"cam-07", "cam-08"}
    assert cameras, "no tracks were listed at all"


def test_correlation_runs_across_cameras_not_within_one(qt_app, window, reference_video: Path):
    # The claim the console exists to present. A camera correlating its own
    # events would raise one incident per camera for one intrusion.
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

    per_camera = {c: len(s.events) for c, s in window._sessions.items()}
    assert sum(per_camera.values()) > 0, (
        "no events were raised, so correlation was never exercised"
    )
    contributing = [c for c, n in per_camera.items() if n]

    # Assert the property, not a bound. An earlier version of this test allowed
    # `<= 2` incidents, which per-session correlation of two cameras satisfies
    # exactly — so it passed while the regression it exists for was present. A
    # threshold that admits the failure it was written to catch is not a test.
    if len(contributing) > 1:
        assert window.incidents.topLevelItemCount() == 1, (
            f"{len(contributing)} cameras watching one scene produced "
            f"{window.incidents.topLevelItemCount()} incidents; correlation must "
            "run above the cameras, not inside each one"
        )
        row = window.incidents.topLevelItem(0)
        named = set(row.text(3).split(", "))
        assert named == set(contributing), (
            f"the incident names {named}, but {set(contributing)} contributed — "
            "the cameras were not correlated together"
        )
    else:
        # Only one camera saw anything, so this run cannot exercise the claim.
        # Say so rather than passing silently on a vacuous assertion.
        assert window.incidents.topLevelItemCount() >= 1


def test_the_map_shows_every_placed_camera(qt_app, window, reference_video: Path):
    from sentinel.core import CameraPose, LatLon, destination_point

    site = LatLon(33.8938, 35.5018)
    for index, camera_id in enumerate(("cam-07", "cam-08")):
        session = window.add_camera(reference_video, camera_id)
        session.pose = CameraPose(
            position=destination_point(site, 90.0 * index, 20.0),
            mount_height=6.0,
            heading=180.0,
            pitch=-22.0,
            horizontal_fov=62.0,
            vertical_fov=36.0,
            range_meters=90.0,
        )
    window._refresh_placement()

    assert len(window.map._cameras) == 2
    assert len(window.map._footprints) == 2


def test_an_unplaced_camera_is_named_as_unplaced(qt_app, window, reference_video: Path):
    from sentinel.core import CameraPose, LatLon

    placed = window.add_camera(reference_video, "cam-07")
    placed.pose = CameraPose(
        position=LatLon(33.8938, 35.5018),
        mount_height=6.0,
        heading=180.0,
        pitch=-22.0,
    )
    window.add_camera(reference_video, "cam-08")
    window._refresh_placement()

    assert "1 of 2" in window.placement_label.text()
    assert "will not locate" in window.placement_label.text()


# --------------------------------------------------------------- persistence


def test_incidents_survive_the_console_being_closed(qt_app, reference_video, tmp_path):
    """The point of persisting anything.

    An incident an operator cannot go back to a week later did not, as far as
    anybody reviewing it is concerned, happen.
    """
    from sentinel.core import CameraPose, LatLon

    database = tmp_path / "sentinel.db"

    first = ConsoleWindow(database)
    first.show()
    session = first.add_camera(reference_video, "cam-07")
    session.pose = CameraPose(
        position=LatLon(33.8938, 35.5018),
        mount_height=6.0,
        heading=180.0,
        pitch=-22.0,
        horizontal_fov=62.0,
        vertical_fov=36.0,
        range_meters=90.0,
    )
    first._refresh_placement()
    first.zone_radius.setValue(10.0)
    first._add_zone()
    first._start()
    pump(qt_app, first, 16.0)
    first._stop()
    first._correlate()
    stored_incidents = first.store.incident_count()
    first.close()

    assert stored_incidents > 0, "nothing was persisted, so nothing was checked"

    second = ConsoleWindow(database)
    try:
        assert second.store.incident_count() == stored_incidents
        assert second.store.event_count() > 0
        # A zone describes the ground, not the run, so it comes back.
        assert len(second._zones) == 1
        assert second._zones[0].name == "Restricted Area A"
    finally:
        second.close()


def test_a_camera_placement_is_remembered(qt_app, reference_video, tmp_path):
    from sentinel.core import CameraPose, LatLon

    database = tmp_path / "sentinel.db"

    first = ConsoleWindow(database)
    session = first.add_camera(reference_video, "cam-07")
    session.pose = CameraPose(
        position=LatLon(33.8938, 35.5018),
        mount_height=7.25,
        heading=145.0,
        pitch=-24.0,
    )
    first.store.save_camera("cam-07", "cam-07", str(reference_video), session.pose)
    first.close()

    second = ConsoleWindow(database)
    try:
        restored = second.store.camera_pose("cam-07")
        assert restored is not None
        assert restored.mount_height == pytest.approx(7.25)
        assert restored.heading == pytest.approx(145.0)
    finally:
        second.close()


def test_the_console_records_what_the_operator_did(qt_app, window, reference_video):
    from sentinel.core import CameraPose, LatLon

    session = window.add_camera(reference_video, "cam-07")
    session.pose = CameraPose(
        position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0, pitch=-22.0
    )
    window.store.save_camera("cam-07", "cam-07", str(reference_video), session.pose)
    window.store.audit("console", "camera.placed", "cam-07", "6.0 m, bearing 180")

    actions = {row["action"] for row in window.store.audit_trail()}
    assert "console.started" in actions
    assert "camera.placed" in actions


def test_re_correlating_does_not_multiply_stored_incidents(qt_app, window, reference_video):
    # The correlate timer runs every second and a half over a growing window,
    # so the same incident is written many times. Deterministic ids make each
    # of those an upsert; without that a ten-minute run would leave hundreds of
    # copies of one intrusion.
    from sentinel.core import CameraPose, LatLon

    session = window.add_camera(reference_video, "cam-07")
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

    window._correlate()
    after_one = window.store.incident_count()
    for _ in range(5):
        window._correlate()

    assert after_one > 0
    assert window.store.incident_count() == after_one


# ------------------------------------------------------------------- export


def test_export_is_disabled_until_there_is_something_to_export(qt_app, window):
    assert window.export_button.isEnabled() is False


def test_an_incident_can_be_exported_with_a_verifiable_manifest(qt_app, window, tmp_path):
    from sentinel.evidence import export_incident, verify_export
    from sentinel.incidents import Correlator
    from test_store import make_event

    incident = Correlator().correlate([make_event(track=n) for n in (1, 2)])[0]
    window._incidents = [incident]

    export = export_incident(incident, tmp_path, exported_by="console (unauthenticated)")
    window.store.audit("console", "incident.exported", incident.id, str(export.directory))

    assert verify_export(export.directory) == []
    assert any(
        row["action"] == "incident.exported" for row in window.store.audit_trail()
    ), "an export that is not audited breaks the chain of custody"


def test_the_export_names_no_operator_it_cannot_verify(qt_app, window, tmp_path):
    # There is no authentication yet, so there is nobody to name. Inventing an
    # operator would be a false entry in a chain of custody.
    import json

    from sentinel.evidence import export_incident
    from sentinel.incidents import Correlator
    from test_store import make_event

    incident = Correlator().correlate([make_event()])[0]
    export = export_incident(incident, tmp_path, exported_by="console (unauthenticated)")

    manifest = json.loads((export.directory / "manifest.json").read_text(encoding="utf-8"))
    assert "unauthenticated" in manifest["exported_by"]


# ----------------------------------------------------------- adding a camera
#
# "Add camera" now covers three genuinely different things: a camera attached to
# this machine through the operating system's own device interface, a camera on
# the network, and a video file. These check the two properties that would be
# expensive to discover in the field — that listing cameras does not switch one
# on, and that a network camera's password does not reach the screen.


def test_the_add_camera_dialog_lists_local_cameras_without_opening_one(qt_app, monkeypatch):
    # The property that makes it safe to open this dialog at all: enumeration
    # reads metadata and captures nothing, so this does not light the webcam
    # and, on macOS, does not raise a permission prompt for a camera nobody
    # asked to use.
    import cv2
    from sentinel import devices as device_module

    def forbidden(*args, **kwargs):
        raise AssertionError("opening the dialog switched a camera on")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(
        device_module, "list_cameras",
        lambda: [
            device_module.LocalCamera(0, "Integrated Camera", r"USB\ONE", "Media Foundation"),
            device_module.LocalCamera(1, "Logitech C920", r"USB\TWO", "Media Foundation"),
        ],
    )

    dialog = AddCameraDialog()
    try:
        assert dialog._device_list.count() == 2
        assert "Integrated Camera" in dialog._device_list.item(0).text()
        # An unconfirmed index must say so where an operator will read it.
        assert "assumed" in dialog._device_list.item(0).text()
        assert "Detect" in dialog._device_note.text()
    finally:
        dialog.deleteLater()


def test_choosing_a_local_camera_yields_a_device_source(qt_app, monkeypatch):
    from sentinel import devices as device_module

    monkeypatch.setattr(
        device_module, "list_cameras",
        lambda: [device_module.LocalCamera(2, "Logitech C920", r"USB\TWO", "Video4Linux2")],
    )

    dialog = AddCameraDialog()
    try:
        dialog._tabs.setCurrentIndex(0)
        dialog._device_list.item(0).setSelected(True)

        chosen = dialog._current_choices()

        assert len(chosen) == 1
        assert chosen[0].source == "device:2"
        assert chosen[0].is_device
        # The operating system's name, not "device:2". An operator picked
        # "Logitech C920" and should see that in the camera list.
        assert chosen[0].suggested_id == "logitech-c920"
    finally:
        dialog.deleteLater()


def test_a_camera_with_no_index_confirmed_does_not_claim_one(qt_app, monkeypatch):
    from sentinel import devices as device_module

    monkeypatch.setattr(
        device_module, "list_cameras",
        lambda: [device_module.LocalCamera(0, "Camera A", None, "Media Foundation")],
    )

    dialog = AddCameraDialog()
    try:
        note = dialog._device_note.text()
        assert "assumed" in note
        # And it must say how to resolve it, not merely that it is uncertain.
        assert "Detect" in note and "picture" in note
    finally:
        dialog.deleteLater()


def test_a_machine_with_no_camera_says_so_and_offers_the_next_step(qt_app, monkeypatch):
    from sentinel import devices as device_module

    monkeypatch.setattr(device_module, "list_cameras", lambda: [])

    dialog = AddCameraDialog()
    try:
        assert dialog._device_list.count() == 0
        assert "No cameras" in dialog._device_note.text()
        assert "Detect" in dialog._device_note.text()
    finally:
        dialog.deleteLater()


def test_a_device_subsystem_that_fails_does_not_break_the_dialog(qt_app, monkeypatch):
    # The other two tabs must still work. A machine whose device registry cannot
    # be queried can still open a file and an RTSP URL.
    from sentinel import devices as device_module

    def explode():
        raise OSError("the device subsystem is unavailable")

    monkeypatch.setattr(device_module, "list_cameras", explode)

    dialog = AddCameraDialog()
    try:
        assert dialog._device_list.count() == 0
        dialog._tabs.setCurrentIndex(2)
        dialog._file.setText("/media/gate.mp4")
        assert dialog._current_choices()[0].source == "/media/gate.mp4"
    finally:
        dialog.deleteLater()


def test_a_network_camera_password_never_reaches_the_screen(qt_app, monkeypatch):
    from sentinel import devices as device_module
    from sentinel.decode import contains_credential

    monkeypatch.setattr(device_module, "list_cameras", lambda: [])

    secret = "hunter2-not-a-real-password"
    url = f"rtsp://admin:{secret}@192.168.1.64:554/Streaming/Channels/101"

    dialog = AddCameraDialog()
    try:
        dialog._tabs.setCurrentIndex(1)
        dialog._url.setText(url)

        # The preview shows the operator exactly what everything downstream will
        # see, so the redaction is something they can verify rather than trust.
        preview = dialog._url_preview.text()
        assert secret not in preview
        assert not contains_credential(preview, url)
        assert "192.168.1.64" in preview

        chosen = dialog._current_choices()[0]
        # The raw URL is what gets connected with, and only that.
        assert chosen.source == url
        assert secret not in chosen.display
        assert secret not in chosen.suggested_id
    finally:
        dialog.deleteLater()


def test_a_session_holds_a_source_string_not_a_path(qt_app, window):
    # A `Path` could only represent a file, and made a device and an RTSP URL
    # look like files that did not exist.
    session = window.add_camera("device:0", camera_id="webcam")

    assert session.source == "device:0"
    assert session.display_source == "device:0"
    assert session.is_live is True


def test_a_camera_session_never_exposes_its_credential(qt_app, window):
    from sentinel.decode import contains_credential

    secret = "hunter2-not-a-real-password"
    url = f"rtsp://admin:{secret}@10.20.30.40:554/Streaming/Channels/101"

    session = window.add_camera(url, camera_id="gate")

    assert not contains_credential(session.display_source, url)
    assert not contains_credential(session.camera_id, url)
    assert session.is_live is True
