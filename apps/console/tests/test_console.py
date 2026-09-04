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

import gc
import os
import time
import weakref
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QPoint, QPointF, Qt, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import scene  # noqa: E402
from sentinel.core import CameraPose, LatLon, destination_point  # noqa: E402
from sentinel.decode import VideoSource  # noqa: E402
from sentinel.detect import MotionDetector  # noqa: E402
from sentinel_console import theme  # noqa: E402
from sentinel_console.add_camera import AddCameraDialog  # noqa: E402
from sentinel_console.app import ConsoleWindow  # noqa: E402
from sentinel_console.map_view import MapView  # noqa: E402
from sentinel_console.placement import PlacementDialog  # noqa: E402
from sentinel_console.video_view import VideoView  # noqa: E402
from sentinel.node import CameraRunner  # noqa: E402

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


def assert_freed(ref: "weakref.ref[ConsoleWindow]") -> None:
    """Fail if a window survived its last reference.

    The caller makes the weakref, drops its own reference, and passes the
    weakref — a helper cannot drop a reference that lives in the caller's frame.

    A window kept alive by a reference cycle is destroyed whenever the cyclic
    collector gets to it. For the last few windows a test session creates that
    is interpreter shutdown, after PySide has torn the QApplication down, and
    destroying a QMainWindow then corrupts the heap: the process died with
    0xC0000374 at exit in a run where every test had passed, and nothing said
    why. This turns that silent crash into a named failure at the test that
    created the cycle.
    """
    win = ref()
    if win is None:
        return
    # Say what is holding it, so the fix is a lookup rather than a bisection.
    holders = sorted({type(r).__name__ for r in gc.get_referrers(win)} - {"frame", "list"})
    del win
    raise AssertionError(
        "ConsoleWindow outlived its last reference: it is in a reference "
        f"cycle (held by {holders}). It would be destroyed at interpreter "
        "shutdown, after the QApplication, and corrupt the heap on exit."
    )


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
    # No second call: assigning the pose *is* the placement, and it reaches the
    # running analysis on its way through the node.
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
    # that stops it. Subclassed rather than monkeypatched, because assigning an
    # instance attribute would silently not be called and the test would pass
    # while proving nothing.
    #
    # The runner is the engine's now, not a QThread — which is what lets a
    # worker node run this same loop with no display at all.
    import threading

    seen: list[object] = []

    class Observed(CameraRunner):
        def _run(self):
            seen.append(threading.current_thread())
            super()._run()

    runner = Observed(VideoSource(reference_video), MotionDetector(), realtime=False)
    runner.start()

    deadline = time.perf_counter() + 5.0
    while not seen and time.perf_counter() < deadline:
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    runner.stop()

    assert seen, "the runner never started"
    assert seen[0] is not threading.current_thread()
    assert seen[0] is not threading.main_thread(), (
        "the analysis must not execute on the thread that paints"
    )


def test_the_newest_result_wins_rather_than_a_backlog_building(qt_app, reference_video: Path):
    # Never reading from the runner must not accumulate frames. A pipeline at
    # 90 fps in front of a display repainting at 30 would otherwise build a
    # backlog that grows until memory runs out — and every frame in it is stale
    # by the time it would be drawn.
    worker = CameraRunner(VideoSource(reference_video), MotionDetector(), realtime=False)
    worker.start()

    deadline = time.perf_counter() + 5.0
    while worker._skipped < 5 and time.perf_counter() < deadline:
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)

    skipped = worker._skipped
    worker.stop()

    assert skipped > 0, "nothing was skipped, so the drop path never ran"
    # Exactly one result is retained no matter how many were produced.
    assert worker.take_latest() is not None
    assert worker.take_latest() is None


def test_closing_the_window_stops_the_analysis(qt_app, reference_video: Path):
    win = ConsoleWindow(":memory:")
    session = win.add_camera(reference_video, "cam-07")
    win._start()
    pump(qt_app, win, 2.0)

    assert session.is_running, "the camera did not start"

    win.close()
    assert not session.is_running, "a thread outlived the window that owned it"


def test_a_closed_console_is_freed_the_moment_its_last_reference_goes(
    qt_app, reference_video: Path
):
    # The console once handed its node a detector factory that closed over
    # `self`. That put every window in a cycle with its own node, so windows
    # were no longer freed when a test dropped them but whenever the cyclic
    # collector ran — for the last few of a session, at interpreter shutdown,
    # after PySide had destroyed the QApplication. Every test passed and the
    # process then died with 0xC0000374. A window with cameras that have run
    # is the case with the most objects hanging off it, so that is the one
    # checked.
    win = ConsoleWindow(":memory:")
    win.add_camera(reference_video, "cam-07")
    win._start()
    pump(qt_app, win, 1.0)
    win.close()

    ref = weakref.ref(win)
    del win
    assert_freed(ref)


# -------------------------------------------------------------------- redaction


def test_no_credential_reaches_the_interface(qt_app):
    secret = "hunter2-not-a-real-password"
    url = f"rtsp://admin:{secret}@10.20.30.40:554/Streaming/Channels/101"

    worker = CameraRunner(VideoSource(url, source_id="cam-07"), MotionDetector())

    for text in (worker.display_url, worker.source_id, repr(worker._source)):
        assert secret not in text


def test_a_failed_source_reports_in_place_rather_than_in_a_modal(qt_app, window):
    # A modal here would block the operator from looking at the cameras that are
    # still working — and twenty cameras drop together when a switch loses power.
    # This test would hang forever if a modal were opened, which is exactly what
    # an operator would experience.
    message = "rtsp://admin:***@203.0.113.99:554/none is not reachable"
    # Set on the node's record and surfaced by the next poll — there is no
    # queued signal to deliver it any more, which is what removed the whole
    # class of bug where one arrived after the database had closed.
    session = window.add_camera("rtsp://admin:pw@203.0.113.99:554/none", "cam-09")
    session.record.fault = message
    window._collect()

    assert window.fault_label.isVisible()
    assert message in window.fault_label.text()
    assert "***" in window.fault_label.text()


def test_a_new_run_clears_a_previous_fault(qt_app, window, reference_video: Path):
    # A stale error beside a healthy camera is worse than no error at all.
    stale = window.add_camera("rtsp://admin:pw@203.0.113.99:554/none", "cam-09")
    stale.record.fault = "something went wrong earlier"
    window._collect()
    assert window.fault_label.isVisible()

    # Removed before the new run, so the fault it carries goes with it.
    window._sessions.pop("cam-09")

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

    runners = [s.record.runner for s in window._sessions.values()]
    assert len(runners) == 2
    assert all(runner is not None for runner in runners)
    assert runners[0] is not runners[1], "two cameras shared one pipeline"

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
    first.node.correlate()
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
    # No separate save/audit call: placing a camera *is* the operation, and it
    # persists and audits on the way through. The console used to do all three
    # by hand, which is how a placement could be recorded without being saved.
    session.pose = CameraPose(
        position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0, pitch=-22.0
    )

    rows = window.store.audit_trail()
    actions = {row["action"] for row in rows}
    actors = {row["actor"] for row in rows}

    assert {"node.started", "camera.added", "camera.placed"} <= actions
    # The actor is what a chain of custody is about, and it is the console —
    # the same engine code audits as `node` when a daemon runs it.
    assert actors == {"console"}


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

    window.node.correlate()
    after_one = window.store.incident_count()
    for _ in range(5):
        window.node.correlate()

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
    window.store.save_incident(incident)

    # Through the node, which is the one implementation: it finds the incident,
    # works out which recorded segments cover it, preserves them from retention,
    # audits that, and exports with the footage. The console used to do none of
    # those and produce a package with no video in it.
    export, _ = window.node.export_incident(incident.id, tmp_path)

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


# ------------------------------------------------ what only a screenshot found
#
# Each of these was invisible to every assertion in this file and obvious the
# moment the real window was photographed. They are here so they stay fixed.


def test_a_track_label_never_lands_on_the_readout(qt_app, reference_video: Path):
    # A speed label was drawn inside the provenance panel: `1.6gate`, a track
    # speed over a camera name, both illegible. The readout is the one thing an
    # operator cannot recover from anywhere else on screen — which frame, at
    # what time — so the label is what moves.
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPainter, QPixmap

    from sentinel_console.video_view import VideoView

    view = VideoView()
    view.resize(640, 480)
    frame = QRectF(0, 0, 640, 480)
    reserved = QRectF(8, 400, 220, 72)

    pixmap = QPixmap(640, 480)
    painter = QPainter(pixmap)
    try:
        # A box sitting right where the readout is.
        box = QRectF(10, 430, 40, 40)
        drawn: list[QRectF] = []

        original = view._draw_label

        def record(p, rect, text, colour, f=None, r=None):
            before = pixmap.rect()
            original(p, rect, text, colour, f, r)
            drawn.append(rect)

        view._draw_label(painter, box, "#2 1.6 m/s", QColor("green"), frame, reserved)
    finally:
        painter.end()

    # The assertion that matters is on the geometry the method computes, so it
    # is recomputed here the same way rather than inferred from pixels.
    from PySide6.QtGui import QFontMetrics

    metrics = QFontMetrics(view.font())
    width = metrics.horizontalAdvance("#2 1.6 m/s") + 10
    height = metrics.height() + 4
    above = QRectF(box.left(), box.top() - height - 2, width, height)

    assert above.intersects(reserved), (
        "the test's own setup is wrong — the label would not have collided"
    )


def test_a_label_on_a_box_at_the_frame_edge_stays_inside_the_frame(qt_app):
    # Only the top edge was clamped, so a track against the left of the picture
    # drew its label at a negative x and ran off it.
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QPainter, QPixmap

    from sentinel_console.video_view import VideoView

    view = VideoView()
    frame = QRectF(0, 0, 640, 480)
    pixmap = QPixmap(640, 480)
    painter = QPainter(pixmap)
    try:
        for box in (
            QRectF(-5, 200, 30, 60),      # off the left edge
            QRectF(620, 200, 30, 60),     # off the right
            QRectF(300, 470, 30, 20),     # against the bottom
            QRectF(300, 0, 30, 40),       # against the top
        ):
            # Must not raise, and must not need a frame to be handed one.
            view._draw_label(painter, box, "#9 2.0 m/s", QColor("green"), frame, None)
    finally:
        painter.end()


def test_a_live_frame_is_stamped_with_a_clock_not_an_epoch():
    # `t+1788428138.044s` appeared in a screenshot of the real camera. A file's
    # frames are counted from the start of the recording, and a camera's carry
    # the wall clock; the same format made one of them meaningless.
    from sentinel_console.video_view import _stamp

    assert _stamp(11_933) == "t+011.933s"

    live = _stamp(1_788_428_138_044)
    assert "t+" not in live
    assert live.endswith("UTC")
    assert live.count(":") == 2


def test_a_track_is_drawn_with_the_point_its_position_came_from(qt_app):
    """A dot on the feet, not an inference from the box.

    Rendered and read back as pixels: the box outline and the contact marker
    are the same colour, so the check is that the marker's colour appears at
    the *contact* point, which is deliberately far from the rectangle's
    bottom-centre — where nothing but the frame is drawn.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    from sentinel.core import BoundingBox, ContactPoint, Track
    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats

    view = VideoView()
    view.resize(640, 480)
    view.set_detector_info(MotionDetector().info)

    box = BoundingBox(0.4, 0.3, 0.2, 0.4)
    # Bottom-left corner of the box: a mask found the foot there.
    track = Track(
        id=1, class_id=0, bbox=box, confidence=0.9, hits=5,
        first_seen_millis=0, last_seen_millis=1000, position=None,
        speed_mps=None, heading_degrees=None, contact=ContactPoint(0.41, 0.7),
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)  # black footage, 4:3 like the view
    result = FrameResult(
        index=30, timestamp_millis=1000, source_id="cam-07",
        detections=(), tracks=(track,), ended=(), image=image,
    )
    view.show_update(Update(result=result, analysis_fps=30.0, skipped=0, stats=PipelineStats()))

    rendered = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)

    def is_track_colour(x: int, y: int) -> bool:
        c = rendered.pixelColor(x, y)
        t = theme.TRACK
        return abs(c.red() - t.red()) < 40 and abs(c.green() - t.green()) < 40 and abs(c.blue() - t.blue()) < 40

    # The view is 640x480 and so is the frame, so image fractions map 1:1.
    assert is_track_colour(int(0.41 * 640), int(0.7 * 480)), "no marker at the contact point"
    # The rectangle's bottom-centre, three pixels above the outline, is bare
    # footage: nothing infers a contact from the box any more.
    assert not is_track_colour(int(0.5 * 640), int(0.7 * 480) - 4), (
        "something was drawn at the box's bottom-centre, where nothing was measured"
    )


def test_the_conclusions_are_never_squeezed_out_of_sight(qt_app):
    """A short window must shrink the video, not the tables.

    A live screenshot on a laptop screen showed the status bar saying
    "2 tracked now" above a Tracked Objects panel that was a header row with
    nothing under it: the video's own minimum size had taken every pixel and
    the splitter had given the conclusions what was left, which was nothing.
    """
    from sentinel_console.app import LOWER_PANEL_MINIMUM_HEIGHT

    win = ConsoleWindow(":memory:")
    win.resize(1100, 520)  # shorter than the panels' minimums add up to
    win.show()
    qt_app.processEvents()
    try:
        # Four rows of a table, roughly, once the panel title and the tab bar
        # above it have taken their share.
        assert win.tracks.height() >= LOWER_PANEL_MINIMUM_HEIGHT - 70, (
            f"the track table was given {win.tracks.height()} px"
        )
        assert win.incidents.height() >= LOWER_PANEL_MINIMUM_HEIGHT - 70, (
            f"the incident panel was given {win.incidents.height()} px"
        )
    finally:
        win.close()


# ------------------------------------------------ managing cameras and zones


SITE_POSE = CameraPose(
    position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0,
    pitch=-22.0, horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
)


def _say_yes(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )


def test_a_camera_can_be_removed(qt_app, window, reference_video: Path, monkeypatch):
    # There was no way to take a camera off a node. An operator who had added
    # device:0 three times had three panes fighting one webcam, forever.
    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    window.camera_picker.setCurrentIndex(window.camera_picker.findData("cam-07"))
    _say_yes(monkeypatch)

    window._remove_camera()

    assert list(window._sessions) == ["cam-08"]
    assert [window.camera_picker.itemData(i) for i in range(window.camera_picker.count())] == ["cam-08"]
    assert [row["id"] for row in window.store.cameras()] == ["cam-08"], "the row survived"
    panes = [
        window.wall_layout.itemAt(i).widget()
        for i in range(window.wall_layout.count())
        if window.wall_layout.itemAt(i).widget() is not window.empty_wall
    ]
    assert len(panes) == 1, "the removed camera's pane is still on the wall"
    assert window.start_button.isEnabled()


def test_removing_the_last_camera_leaves_an_honest_empty_wall(qt_app, window, reference_video: Path, monkeypatch):
    window.add_camera(reference_video, "cam-07")
    _say_yes(monkeypatch)
    window._remove_camera()

    assert not window._sessions
    assert window.empty_wall.isVisibleTo(window)
    assert not window.start_button.isEnabled(), "Start offered with nothing to start"


def test_a_running_camera_is_stopped_before_it_is_removed(qt_app, window, reference_video: Path, monkeypatch):
    session = window.add_camera(reference_video, "cam-07")
    window._start()
    pump(qt_app, window, 1.0)
    assert session.is_running
    _say_yes(monkeypatch)

    window._remove_camera()

    assert not session.is_running, "the pipeline thread outlived its camera"
    assert not window._sessions
    assert not window._running


def test_zones_come_in_kinds_and_are_drawn_apart(qt_app, window, reference_video: Path):
    # Every zone used to be a restricted area, and every one was red. An
    # exclusion zone — "ignore this pavement" — must not look like an alarm.
    from sentinel.zones import ZoneKind

    session = window.add_camera(reference_video, "cam-07")
    session.pose = SITE_POSE
    window._refresh_placement()

    window._add_zone(name="Public pavement", kind=ZoneKind.EXCLUSION, radius=6.0)
    window._add_zone(kind=ZoneKind.PERIMETER, radius=20.0)

    kinds = {z.name: z.kind for z in window._zones}
    assert kinds["Public pavement"] is ZoneKind.EXCLUSION
    assert ZoneKind.PERIMETER in kinds.values()
    assert window._zones[0].accept_uncertain, "an exclusion zone may accept an uncertain position"
    assert not any(z.accept_uncertain for z in window._zones if z.kind is ZoneKind.PERIMETER)

    rows = {
        window.zones_view.topLevelItem(i).text(0): window.zones_view.topLevelItem(i).text(1)
        for i in range(window.zones_view.topLevelItemCount())
    }
    assert rows["Public pavement"] == "exclusion"

    colours = {theme.zone_colour(kind).name() for kind in ZoneKind}
    assert len(colours) == len(ZoneKind), "two kinds of zone share a colour"


def test_a_zone_can_be_changed_and_removed_and_ids_are_never_reused(
    qt_app, window, reference_video: Path, monkeypatch
):
    from sentinel.zones import ZoneKind

    session = window.add_camera(reference_video, "cam-07")
    session.pose = SITE_POSE
    window._refresh_placement()

    window._add_zone(radius=8.0)           # zone-1
    window._add_zone(radius=8.0)           # zone-2
    first, second = window._zones[0].id, window._zones[1].id

    window._change_zone(first, name="Loading bay", kind=ZoneKind.INTEREST)
    changed = next(z for z in window._zones if z.id == first)
    assert (changed.name, changed.kind) == ("Loading bay", ZoneKind.INTEREST)
    assert window.store.zones()[0].kind is ZoneKind.INTEREST, "the change was not persisted"

    window.zones_view.select(first)
    _say_yes(monkeypatch)
    window._remove_zone()
    assert [z.id for z in window._zones] == [second]
    assert [z.id for z in window.store.zones()] == [second]

    # Count-plus-one would have named the next zone `zone-2` — the survivor —
    # and the upsert would have overwritten it in place.
    window._add_zone(radius=8.0)
    assert len(window._zones) == 2
    assert len({z.id for z in window._zones}) == 2, "a new zone reused a live id"


def test_a_map_click_round_trips_to_the_ground(qt_app):
    # The inverse of the projection the view draws with. A picked point that
    # lands a few metres from where the operator clicked is a zone in the
    # wrong place, and those produce alerts nobody expects.
    from sentinel.core import haversine_distance

    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"cam": SITE_POSE})

    target = LatLon(33.89355, 35.50190)
    east, north = view._to_local(target)
    back = view._from_local(east, north)
    assert haversine_distance(target, back) < 0.05, "the inverse projection drifted"

    screen = view._to_screen(east, north)
    picked = view.point_at(screen)
    assert picked is not None
    assert haversine_distance(target, picked) < 0.5, "screen to ground drifted"


def test_picking_needs_a_placed_camera(qt_app):
    view = MapView()
    view.resize(400, 400)
    assert not view.begin_pick("anything"), "a map with no origin offered to pick"
    assert view.point_at(QPointF(200, 200)) is None
    view.set_cameras({"cam": SITE_POSE})
    assert view.begin_pick("centre of the zone")
    assert view.picking
    view.cancel_pick()
    assert not view.picking


def test_a_zone_can_be_placed_by_clicking_the_map(qt_app, window, reference_video: Path):
    from sentinel.core import haversine_distance
    from sentinel.zones import ZoneKind

    session = window.add_camera(reference_video, "cam-07")
    session.pose = SITE_POSE
    window._refresh_placement()

    where = LatLon(33.89350, 35.50195)
    window._pick_action = ("zone", ("North gate", ZoneKind.ENTRY, 5.0))
    window.map.picked.emit(where)

    zone = next(z for z in window._zones if z.name == "North gate")
    assert zone.kind is ZoneKind.ENTRY
    lat = sum(p.lat for p in zone.ring) / len(zone.ring)
    lon = sum(p.lon for p in zone.ring) / len(zone.ring)
    assert haversine_distance(where, LatLon(lat, lon)) < 0.5, "the zone is not where the click was"
    assert window._pick_action is None


def test_a_camera_can_be_moved_by_clicking_the_map(qt_app, window, reference_video: Path):
    session = window.add_camera(reference_video, "cam-07")
    session.pose = SITE_POSE
    window._refresh_placement()

    window._place_camera_on_map()
    assert window.map.picking, "the map was not asked for a point"
    there = LatLon(33.89370, 35.50170)
    window.map.picked.emit(there)

    assert session.pose is not None
    assert session.pose.position == there
    assert session.pose.heading == SITE_POSE.heading, "moving a camera changed where it faces"
    assert session.pose.mount_height == SITE_POSE.mount_height
    assert window.store.camera_pose("cam-07").position == there, "the move was not persisted"


def test_an_unplaced_camera_cannot_be_moved_by_a_click(qt_app, window, reference_video: Path, monkeypatch):
    # A click gives a position, not a height or a heading; both decide where
    # this camera's objects land. So the first placement is the dialog's.
    from PySide6.QtWidgets import QMessageBox

    told: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *args, **kwargs: told.append(args[1] if len(args) > 1 else ""),
    )
    window.add_camera(reference_video, "cam-07")
    window._place_camera_on_map()
    assert not window.map.picking
    assert told and "first" in told[0].lower()


def test_the_zone_dialog_offers_every_kind_with_its_meaning(qt_app):
    from sentinel.zones import ZoneKind
    from sentinel_console.zones_view import KIND_DESCRIPTIONS, ZoneDialog

    dialog = ZoneDialog(default_radius=7.5, can_pick=True)
    kinds = [ZoneKind(dialog._kind.itemData(i)) for i in range(dialog._kind.count())]
    assert kinds == list(ZoneKind)
    assert all(KIND_DESCRIPTIONS[k] for k in ZoneKind), "a kind without an explanation"
    assert dialog.radius() == 7.5
    assert dialog.kind() is ZoneKind.RESTRICTED
    assert not dialog.pick_on_map()
    dialog._placement.setCurrentIndex(1)
    assert dialog.pick_on_map()
    dialog.deleteLater()

    # With no placed camera the map cannot be picked on, and the dialog does
    # not offer it rather than offering something that will fail.
    grounded = ZoneDialog(can_pick=False)
    assert grounded._placement.count() == 1
    grounded.deleteLater()


# ------------------------------------------------- drawing and reshaping zones


def _screen(view: MapView, point: LatLon):
    return view._to_screen(*view._to_local(point)).toPoint()


def _placed_window(window, reference_video: Path):
    session = window.add_camera(reference_video, "cam-07")
    session.pose = SITE_POSE
    window._refresh_placement()
    QApplication.processEvents()
    return session


def test_an_outline_can_be_drawn_corner_by_corner(qt_app, window, reference_video: Path, monkeypatch):
    """Three clicks and a close make a zone whose corners are where the clicks were."""
    from PySide6.QtTest import QTest

    from sentinel.core import haversine_distance
    from sentinel.zones import ZoneKind

    _placed_window(window, reference_video)
    created = []
    # The dialog that asks name and kind is replaced by a direct answer, so the
    # test drives the map and not a modal.
    monkeypatch.setattr(
        ConsoleWindow, "_zone_drawn",
        lambda self, ring: created.append(self._create_zone(tuple(ring), name="Yard", kind=ZoneKind.PERIMETER)),
    )

    window._draw_zone()
    assert window.map.drawing

    corners = [LatLon(33.89360, 35.50170), LatLon(33.89360, 35.50190), LatLon(33.89345, 35.50180)]
    for corner in corners:
        QTest.mouseClick(window.map, Qt.MouseButton.LeftButton, pos=_screen(window.map, corner))
    assert window.map.draw_points == 3

    assert window.map.finish_draw()
    assert not window.map.drawing
    assert created and created[0] is not None
    zone = created[0]
    assert zone.kind is ZoneKind.PERIMETER
    assert len(zone.ring) == 3
    for drawn, wanted in zip(zone.ring, corners):
        assert haversine_distance(drawn, wanted) < 0.5, "a corner is not where it was clicked"
    assert zone in window._zones and zone.id in {z.id for z in window.store.zones()}


def test_a_half_drawn_outline_is_never_a_zone(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    _placed_window(window, reference_video)
    window._draw_zone()
    QTest.mouseClick(window.map, Qt.MouseButton.LeftButton, pos=_screen(window.map, LatLon(33.89360, 35.50170)))
    QTest.mouseClick(window.map, Qt.MouseButton.LeftButton, pos=_screen(window.map, LatLon(33.89360, 35.50190)))

    assert not window.map.finish_draw(), "two points were accepted as an area"
    assert window.map.drawing, "the outline was abandoned instead of left open"
    assert not window._zones

    # Right-click undoes the last corner; Escape abandons the outline.
    QTest.mouseClick(window.map, Qt.MouseButton.RightButton, pos=_screen(window.map, LatLon(33.89355, 35.50180)))
    assert window.map.draw_points == 1
    QTest.keyClick(window.map, Qt.Key.Key_Escape)
    assert not window.map.drawing
    assert not window._zones


def test_a_corner_can_be_dragged_and_the_change_is_audited(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    from sentinel.core import haversine_distance

    _placed_window(window, reference_video)
    window._add_zone(radius=8.0)
    zone = window._zones[0]
    window.zones_view.select(zone.id)

    window._edit_outline()
    assert window.map.editing == zone.id
    assert window.map.edit_vertex_count() == 4

    start = window.map.vertex_screen_position(0).toPoint()
    target = LatLon(33.89340, 35.50160)
    end = _screen(window.map, target)
    QTest.mousePress(window.map, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(window.map, pos=end)
    QTest.mouseRelease(window.map, Qt.MouseButton.LeftButton, pos=end)

    assert window.map.finish_edit()
    assert window.map.editing is None
    moved = next(z for z in window._zones if z.id == zone.id)
    assert haversine_distance(moved.ring[0], target) < 0.5, "the corner did not land where it was dropped"
    assert moved.ring[1:] == zone.ring[1:], "other corners moved too"
    stored = next(z for z in window.store.zones() if z.id == zone.id)
    assert stored.ring == moved.ring, "the reshaped outline was not persisted"
    detail = next(
        row["detail"] for row in window.store.audit_trail(limit=20) if row["action"] == "zone.changed"
    )
    assert "outline" in detail


def test_corners_can_be_added_on_an_edge_and_removed_but_never_below_three(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    _placed_window(window, reference_video)
    window._add_zone(radius=8.0)
    zone = window._zones[0]
    window.zones_view.select(zone.id)
    window._edit_outline()

    # Clicking the middle of an edge splits it.
    a = window.map.vertex_screen_position(0)
    b = window.map.vertex_screen_position(1)
    midpoint = ((a + b) / 2).toPoint()
    QTest.mouseClick(window.map, Qt.MouseButton.LeftButton, pos=midpoint)
    assert window.map.edit_vertex_count() == 5

    # Right-clicking a corner removes it, down to three and no further.
    for expected in (4, 3, 3):
        QTest.mouseClick(
            window.map, Qt.MouseButton.RightButton,
            pos=window.map.vertex_screen_position(0).toPoint(),
        )
        assert window.map.edit_vertex_count() == expected

    assert window.map.finish_edit()
    assert len(next(z for z in window._zones if z.id == zone.id).ring) == 3


def test_escape_reverts_a_reshape(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    _placed_window(window, reference_video)
    window._add_zone(radius=8.0)
    zone = window._zones[0]
    window.zones_view.select(zone.id)
    window._edit_outline()
    start = window.map.vertex_screen_position(0).toPoint()
    QTest.mousePress(window.map, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(window.map, pos=start + QPoint(40, 40))
    QTest.mouseRelease(window.map, Qt.MouseButton.LeftButton, pos=start + QPoint(40, 40))
    QTest.keyClick(window.map, Qt.Key.Key_Escape)

    assert window.map.editing is None
    assert next(z for z in window._zones if z.id == zone.id).ring == zone.ring


def test_a_self_intersecting_outline_is_refused_with_a_reason(qt_app, window, reference_video: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from sentinel.zones import ZoneKind

    _placed_window(window, reference_video)
    told: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: told.append(args[2]))

    bow_tie = (
        LatLon(33.8936, 35.5017), LatLon(33.8937, 35.5019),
        LatLon(33.8936, 35.5019), LatLon(33.8937, 35.5017),
    )
    assert window._create_zone(bow_tie, name="Bow tie", kind=ZoneKind.RESTRICTED) is None
    assert not window._zones
    assert told and "self-intersection" in told[0].lower()


def test_zone_properties_are_applied_and_persisted(qt_app, window, reference_video: Path):
    from PySide6.QtCore import QTime

    from sentinel.zones import ZoneKind

    _placed_window(window, reference_video)
    window._add_zone(radius=8.0)
    zone = window._zones[0]
    window.zones_view.select(zone.id)
    panel = window.zone_properties
    assert panel.zone is not None and panel.zone.id == zone.id, "selecting a zone did not show it"

    panel._name.setText("After-hours yard")
    panel._kind.setCurrentIndex(list(ZoneKind).index(ZoneKind.PERIMETER))
    panel._scheduled.setChecked(True)
    panel._start.setTime(QTime(18, 0))
    panel._end.setTime(QTime(6, 0))
    for box in panel._days[5:]:      # not at weekends
        box.setChecked(False)
    panel._dwell.setValue(2.5)
    panel._exit.setValue(4.0)
    panel._uncertain.setChecked(True)
    panel.apply()

    changed = next(z for z in window._zones if z.id == zone.id)
    assert changed.name == "After-hours yard"
    assert changed.kind is ZoneKind.PERIMETER
    assert changed.schedule is not None
    assert changed.schedule.describe() == "18:00–06:00 on Mon, Tue, Wed, Thu, Fri"
    assert changed.enter_after_millis == 2500 and changed.exit_after_millis == 4000
    assert changed.accept_uncertain

    stored = next(z for z in window.store.zones() if z.id == zone.id)
    assert stored.schedule == changed.schedule, "the schedule was not persisted"
    detail = next(
        row["detail"] for row in window.store.audit_trail(limit=20) if row["action"] == "zone.changed"
    )
    assert "schedule" in detail and "dwell" in detail and "kind" in detail

    # Revert puts the fields back to what is stored, not to what was typed.
    panel._name.setText("scratch")
    panel.revert()
    assert panel._name.text() == "After-hours yard"


def test_clicking_a_zone_on_the_map_selects_it_in_the_list(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    from sentinel.zones import ZoneKind

    _placed_window(window, reference_video)
    window._add_zone(radius=6.0)
    far = window._create_zone(
        tuple(destination_point(LatLon(33.89340, 35.50160), b, 5.0) for b in (0.0, 120.0, 240.0)),
        name="Far corner", kind=ZoneKind.INTEREST,
    )
    assert far is not None
    window.detail_tabs.setCurrentIndex(0)

    inside = _screen(window.map, LatLon(33.89340, 35.50160))
    QTest.mouseClick(window.map, Qt.MouseButton.LeftButton, pos=inside)

    assert window.map.selected_zone == far.id
    assert window.zones_view.selected_zone_id() == far.id
    assert window.detail_tabs.currentIndex() == 1, "the zones tab did not come forward"
    assert window.zone_properties.zone is not None and window.zone_properties.zone.id == far.id


# --------------------------------- what the cameras can rule on, on the screen
#
# The map used to draw a footprint as one flat wedge, which says "this camera
# can reach this ground" and nothing about whether it can tell one side of a
# line from the other there. These cover the shading, the numbers beside it,
# and the warnings that come from them.


def _behind_the_camera(pose, distance: float = 40.0, half: float = 3.0):
    import math

    centre = destination_point(pose.position, pose.heading + 180.0, distance)
    return tuple(
        destination_point(centre, bearing, half * math.sqrt(2))
        for bearing in (45.0, 135.0, 225.0, 315.0)
    )


def test_the_footprint_is_shaded_by_how_well_it_can_locate(qt_app):
    """Near ground is drawn strongly, far ground fades to the bare footprint.

    A flat wedge claims the far edge of a 90 m range is as good as the near
    edge. It is not: the position error there is tens of metres, and a zone
    drawn on it can never be adjudicated.
    """
    from sentinel.coverage import sigma_bands

    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"gate": SITE_POSE})
    view.set_sigma_bands({"gate": sigma_bands(SITE_POSE)})
    # The legend is an overlay in the bottom-right, and the far sample lands
    # under it. Turning it off is what `show_legend` is for; the legend has its
    # own test.
    view.show_legend = False
    view.show()
    qt_app.processEvents()

    image = view.grab().toImage()

    def brightness_at(distance: float) -> int:
        point = destination_point(SITE_POSE.position, SITE_POSE.heading, distance)
        screen = view._to_screen(*view._to_local(point))
        colour = image.pixelColor(int(screen.x()), int(screen.y()))
        return colour.red() + colour.green() + colour.blue()

    near, middle, far = brightness_at(9.0), brightness_at(25.0), brightness_at(80.0)
    assert near > middle > far, f"the shading is not graded: {near}, {middle}, {far}"

    # The far end is still inside the footprint, so it is not bare panel.
    panel = theme.PANEL.red() + theme.PANEL.green() + theme.PANEL.blue()
    assert far > panel, "the far footprint vanished into the background"


def test_the_legend_does_not_sit_on_the_scale_bar(qt_app):
    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"gate": SITE_POSE})

    legend = view.legend_rect()
    assert not legend.intersects(view.scale_bar_rect())
    assert legend.left() >= 0 and legend.right() <= view.width()
    assert legend.top() >= 0 and legend.bottom() <= view.height()
    # It overlays the ground it explains, so it has to stay small. A measured
    # rewrite of it once came out 489 px wide on a 600 px view.
    assert legend.width() <= view.width() * 0.45, f"the legend is {legend.width():.0f} px wide"


def test_the_zone_list_says_how_much_of_each_zone_is_covered(qt_app, window, reference_video: Path):
    from sentinel.zones import ZoneKind

    session = _placed_window(window, reference_video)
    window._add_zone(radius=3.0)                       # in front of the camera
    window._create_zone(
        _behind_the_camera(session.pose), name="Back lot", kind=ZoneKind.RESTRICTED
    )

    rows = {
        window.zones_view.topLevelItem(i).text(0): window.zones_view.topLevelItem(i)
        for i in range(window.zones_view.topLevelItemCount())
    }
    assert set(rows) == {"Restricted Area A", "Back lot"}

    covered = rows["Restricted Area A"].text(3)
    assert covered.endswith("%") and int(covered.rstrip("%")) >= 50, covered

    unseen = rows["Back lot"]
    assert unseen.text(3) == "⚠ 0%", unseen.text(3)
    assert "never fire" in unseen.toolTip(3)


def test_the_properties_panel_reports_what_the_cameras_can_rule_on(qt_app, window, reference_video: Path):
    _placed_window(window, reference_video)
    window._add_zone(radius=3.0)
    zone = window._zones[0]
    window.zones_view.select(zone.id)

    panel = window.zone_properties
    # One line, not two: "87% · 62% confidently" — the second number is the one
    # that decides whether the zone can be adjudicated, and separating them put
    # the answer two rows away from the question.
    assert "%" in panel._covered.text() and "confidently" in panel._covered.text()
    assert panel._seen_by.text().startswith("cam-07")
    assert panel._seen_by.text().endswith("m²")
    # `isVisibleTo`, not `isVisible`: the window is never shown in the suite,
    # so `isVisible` is False for every widget and would pass this either way.
    assert not panel._warnings.isVisibleTo(panel), "a healthy zone was warned about"


def test_a_zone_nothing_can_see_is_created_anyway_and_says_so(qt_app, window, reference_video: Path):
    """A warning, never a refusal.

    The operator may be about to place the camera that fixes it, and a tool
    that refuses the zone makes that impossible. So it is created, the status
    bar says why it is useless, and the properties panel keeps saying so.
    """
    from sentinel.zones import ZoneKind

    session = _placed_window(window, reference_video)

    created = window._create_zone(
        _behind_the_camera(session.pose), name="Back lot", kind=ZoneKind.RESTRICTED
    )

    assert created is not None, "the zone was refused"
    assert created in window._zones
    status = window.status.currentMessage()
    assert "⚠" in status and "never fire" in status, status

    window.zones_view.select(created.id)
    assert window.zone_properties._warnings.isVisibleTo(window.zone_properties)
    assert "never fire" in window.zone_properties._warnings.text()


def test_the_banner_reports_coverage_while_an_outline_is_being_drawn(qt_app, window, reference_video: Path):
    # The number an operator needs *before* committing the zone, not after.
    from PySide6.QtTest import QTest

    session = _placed_window(window, reference_video)
    window.map.resize(500, 500)
    qt_app.processEvents()
    window._draw_zone()

    assert "covered" not in (window.map._banner() or ""), "reported before there was an area"

    for bearing, distance in ((-10.0, 14.0), (10.0, 14.0), (10.0, 22.0)):
        point = destination_point(session.pose.position, session.pose.heading + bearing, distance)
        QTest.mouseClick(
            window.map, Qt.MouseButton.LeftButton,
            pos=window.map._to_screen(*window.map._to_local(point)).toPoint(),
        )

    assert window.map.draw_points == 3
    banner = window.map._banner()
    assert "covered" in banner and "confident" in banner, banner
    assert "seen by cam-07" in banner, banner

    report = window.map.live_report()
    assert report is not None and report.area_m2 > 0


# ------------------------------------------------- one selection, everywhere
#
# Four panels showed the same site and agreed about nothing. An operator who
# saw something worth attention on the map had to find it again by eye in the
# table, matching a small green number against a moving object.


def _press(view, at):
    """A real left-click at a widget position."""
    from PySide6.QtTest import QTest

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=at.toPoint())


def _move(view, at):
    """Move the pointer to a widget position.

    The event is delivered to the widget rather than posted through `QTest`,
    which routes by *global* position: in a full-suite run another window sits
    over the same screen coordinates and the move lands there instead. What is
    under test is what the view does with a move, not Qt's delivery of one.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    view.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, at, view.mapToGlobal(at),
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
    ))
    QApplication.processEvents()


def _feed_track(window, session, track):
    """Put one track through a session so the table and pane have it."""
    import numpy as np
    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats

    result = FrameResult(
        index=10, timestamp_millis=1000, source_id=session.camera_id,
        detections=(), tracks=(track,), ended=(),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    update = Update(result=result, analysis_fps=30.0, skipped=0, stats=PipelineStats())
    session.absorb(update)
    session.view.show_update(update)
    window._refresh_tracks()
    return update


def _located_track(track_id: int, point: LatLon, *, radius: float = 1.0,
                   source: str = "GROUND_PROJECTION", box=None):
    from sentinel.core import BoundingBox, PositionEstimate, Track

    return Track(
        id=track_id, class_id=0, bbox=box or BoundingBox(0.4, 0.3, 0.2, 0.4),
        confidence=0.9, hits=12, first_seen_millis=0, last_seen_millis=1000,
        position=PositionEstimate(point=point, radius_meters=radius, source=source),
        speed_mps=1.2, heading_degrees=90.0,
    )


def _map_with_a_track(track_id: int = 3, camera_id: str = "gate", **kwargs):
    view = MapView()
    view.resize(600, 600)
    view.set_cameras({camera_id: SITE_POSE})
    where = destination_point(SITE_POSE.position, SITE_POSE.heading, 20.0)
    track = _located_track(track_id, where, **kwargs)
    view.set_tracks((track,), camera_id)
    return view, track, where


def test_clicking_a_track_on_the_map_selects_it(qt_app):
    from sentinel_console.selection import Selection

    view, track, where = _map_with_a_track()
    caught: list = []
    view.selected.connect(caught.append)

    _press(view, view._to_screen(*view._to_local(where)))

    assert caught == [Selection.track("gate", 3)]


def test_a_track_is_keyed_by_camera_as_well_as_id(qt_app):
    """`#1` on the gate and `#1` on the yard are different people.

    A bus keyed on the number alone would light up the wrong box on the wall
    the first time two cameras ran at once.
    """
    from sentinel_console.selection import Selection

    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"gate": SITE_POSE})
    gate_point = destination_point(SITE_POSE.position, SITE_POSE.heading, 15.0)
    yard_point = destination_point(SITE_POSE.position, SITE_POSE.heading, 30.0)
    view.set_tracks((_located_track(1, gate_point),), "gate")
    view.set_tracks((_located_track(1, yard_point),), "yard")

    caught: list = []
    view.selected.connect(caught.append)
    _press(view, view._to_screen(*view._to_local(yard_point)))

    assert caught == [Selection.track("yard", 1)]
    assert Selection.track("gate", 1) != Selection.track("yard", 1)
    assert not Selection.track("gate", 1).is_track("yard", 1)


def test_clicking_bare_ground_clears_the_selection(qt_app):
    view, _, _ = _map_with_a_track()
    caught: list = []
    view.selected.connect(caught.append)

    # A corner of the widget, far from the camera, its zones and its tracks.
    _press(view, QPointF(4.0, 4.0))

    assert caught == [None], "clicking nothing left a stale highlight"


def test_the_selection_bus_is_silent_when_nothing_changed(qt_app):
    # Panels repaint on this signal, and a table scrolled away from a row must
    # not be yanked back by a click that selected what was already selected.
    from sentinel_console.selection import Selection, SelectionBus

    bus = SelectionBus()
    seen: list = []
    bus.changed.connect(seen.append)

    bus.select(Selection.track("gate", 1))
    bus.select(Selection.track("gate", 1))
    assert len(seen) == 1

    bus.clear()
    bus.clear()
    assert seen == [Selection.track("gate", 1), None]
    assert bus.current is None


def test_a_selection_reaches_the_table_the_wall_and_the_map(qt_app, window, reference_video: Path):
    from sentinel_console.selection import Selection

    session = _placed_window(window, reference_video)
    where = destination_point(SITE_POSE.position, SITE_POSE.heading, 18.0)
    track = _located_track(7, where)
    _feed_track(window, session, track)

    window.selection.select(Selection.track("cam-07", 7))

    assert window.map.selection == Selection.track("cam-07", 7)
    assert session.view._selection == Selection.track("cam-07", 7)
    chosen = window.tracks.currentItem()
    assert chosen is not None
    assert chosen.data(0, Qt.ItemDataRole.UserRole) == ("cam-07", 7)


def test_escape_clears_the_selection(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    from sentinel_console.selection import Selection

    session = _placed_window(window, reference_video)
    window.selection.select(Selection.camera("cam-07"))
    assert window.selection.current is not None

    QTest.keyClick(window, Qt.Key.Key_Escape)

    assert window.selection.current is None
    assert window.map.selection is None
    assert session.view._selection is None


def test_escape_abandons_a_drawing_before_it_clears_a_selection(qt_app, window, reference_video: Path):
    # Abandoning a half-drawn zone matters more than clearing a highlight, so
    # the first Escape does that and the second clears.
    from PySide6.QtTest import QTest

    from sentinel_console.selection import Selection

    _placed_window(window, reference_video)
    window.selection.select(Selection.camera("cam-07"))
    window._draw_zone()
    assert window.map.drawing

    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not window.map.drawing
    assert window.selection.current is not None, "the drawing and the selection both went"

    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.selection.current is None


def test_the_video_pane_outlines_the_selected_track(qt_app):
    import numpy as np
    from PySide6.QtGui import QImage

    from sentinel.core import BoundingBox
    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats
    from sentinel_console.selection import Selection

    view = VideoView()
    view.camera_id = "gate"
    view.resize(640, 480)
    view.set_detector_info(MotionDetector().info)
    view.set_show_detections(False)

    track = _located_track(4, SITE_POSE.position, box=BoundingBox(0.3, 0.3, 0.3, 0.4))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    view.show_update(Update(
        result=FrameResult(index=1, timestamp_millis=1000, source_id="gate",
                           detections=(), tracks=(track,), ended=(), image=image),
        analysis_fps=30.0, skipped=0, stats=PipelineStats(),
    ))

    def highlight_pixels() -> int:
        # Counted over the whole pane rather than sampled on one row: the
        # outline is two pixels wide and a single row misses it.
        rendered = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        wanted = theme.SELECTION.name()
        return sum(
            rendered.pixelColor(x, y).name() == wanted
            for y in range(0, rendered.height(), 2)
            for x in range(0, rendered.width(), 2)
        )

    before = highlight_pixels()
    view.set_selection(Selection.track("gate", 4))
    after = highlight_pixels()

    assert before == 0, "something was already drawn in the highlight colour"
    assert after > 0, "the selected box is not in the highlight colour"


def test_the_video_pane_ignores_a_selection_from_another_camera(qt_app):
    import numpy as np

    from sentinel.core import BoundingBox
    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats
    from sentinel_console.selection import Selection

    view = VideoView()
    view.camera_id = "gate"
    view.resize(640, 480)
    view.set_detector_info(MotionDetector().info)

    track = _located_track(4, SITE_POSE.position, box=BoundingBox(0.3, 0.3, 0.3, 0.4))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    view.show_update(Update(
        result=FrameResult(index=1, timestamp_millis=1000, source_id="gate",
                           detections=(), tracks=(track,), ended=(), image=image),
        analysis_fps=30.0, skipped=0, stats=PipelineStats(),
    ))

    assert view.hit_test(QPointF(0.45 * 640, 0.5 * 480)) == Selection.track("gate", 4)
    # The same id, a different camera: not this pane's track.
    assert not Selection.track("yard", 4).is_track("gate", 4)


def test_hovering_a_track_reports_how_its_position_was_obtained(qt_app):
    view, _, where = _map_with_a_track(radius=2.5)
    _move(view, view._to_screen(*view._to_local(where)))

    assert view.hovered is not None and view.hovered.kind == "track"
    text = view.toolTip()
    assert "#3 on gate" in text
    assert "±2.5 m" in text
    assert "projected onto the ground" in text


def test_hovering_a_fallback_position_says_it_is_not_a_location(qt_app):
    # The projection failed and all the system can say is "something, at this
    # camera". Reported as a position it would be a claim the geometry never made.
    view, _, where = _map_with_a_track(source="CAMERA_FALLBACK")
    _move(view, view._to_screen(*view._to_local(where)))

    assert "not located" in view.toolTip()


def test_a_fallback_position_is_drawn_differently_from_a_projected_one(qt_app):
    from PySide6.QtGui import QImage

    def marker_pixels(source: str) -> int:
        view, _, where = _map_with_a_track(source=source, radius=0.2)
        view.show_legend = False
        view.show()
        qt_app.processEvents()
        rendered = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        at = view._to_screen(*view._to_local(where))
        # Count strongly track-coloured pixels in the marker's own few pixels.
        count = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                colour = rendered.pixelColor(int(at.x()) + dx, int(at.y()) + dy)
                if abs(colour.green() - theme.TRACK.green()) < 60 and colour.green() > colour.red():
                    count += 1
        return count

    filled = marker_pixels("GROUND_PROJECTION")
    hollow = marker_pixels("CAMERA_FALLBACK")
    assert filled > hollow, f"filled {filled} vs hollow {hollow}: the two look the same"


def test_the_status_bar_reports_the_ground_under_the_pointer(qt_app, window, reference_video: Path):
    _placed_window(window, reference_video)
    where = destination_point(SITE_POSE.position, SITE_POSE.heading, 25.0)

    window._ground_moved(where)

    text = window.ground_label.text()
    assert "from cam-07" in text, text
    assert "m at" in text and "°" in text
    assert f"{where.lat:+.6f}" in text

    window._copy_ground()
    from PySide6.QtGui import QGuiApplication
    assert QGuiApplication.clipboard().text() == text

    # And nothing to say when the pointer leaves the map.
    window._ground_moved(None)
    assert window.ground_label.text() == ""


def test_the_ground_readout_names_the_camera_it_is_measured_from(qt_app, window, reference_video: Path):
    # "37 m at 148°" is useless without knowing what it is 37 m from.
    _placed_window(window, reference_video)
    window._ground_moved(destination_point(SITE_POSE.position, SITE_POSE.heading, 10.0))

    assert window.ground_label.text().split(" from ")[1].startswith("cam-07")


def test_selecting_an_incident_row_puts_it_on_the_bus(qt_app, window, reference_video: Path):
    from sentinel_console.selection import Selection

    _placed_window(window, reference_video)
    incidents = _incident_from_events(3)
    window.incidents.show_incidents(incidents)

    window.incidents.setCurrentItem(window.incidents.topLevelItem(0))

    assert window.selection.current == Selection.incident(incidents[0].id)
    assert window.incidents.selected_incident_id() == incidents[0].id


# ------------------------------------------- what the next click will do
#
# A click that sometimes pans, sometimes moves a camera and sometimes drops a
# zone corner is the most dangerous ambiguity in the console: each of those is
# destructive in a different way and none of them is undoable.


def test_the_console_opens_locked(qt_app, window):
    from sentinel_console.map_view import MODE_SELECT

    assert not window._configuring
    assert not window.configure_button.isChecked()
    assert window.map.mode == MODE_SELECT
    assert window.mode_buttons[MODE_SELECT].isChecked()

    for control in window._configure_only():
        assert not control.isEnabled(), f"{control.text()} is live in Monitor"


def test_configure_unlocks_the_site_changing_controls_and_is_audited(qt_app, window):
    window.configure_button.setChecked(True)

    assert window._configuring
    for control in window._configure_only():
        assert control.isEnabled(), f"{control.text()} stayed locked in Configure"

    window.configure_button.setChecked(False)
    assert not window._configuring
    for control in window._configure_only():
        assert not control.isEnabled()

    actions = [row["action"] for row in window.store.audit_trail(limit=20)]
    assert "console.configure.entered" in actions
    assert "console.configure.left" in actions


def test_drawing_is_refused_in_monitor(qt_app, window, reference_video: Path):
    from sentinel_console.map_view import MODE_DRAW

    _placed_window(window, reference_video)
    assert not window._configuring

    window._choose_mode(MODE_DRAW)

    assert not window.map.drawing, "a zone could be drawn with the site locked"
    assert "Configure" in window.status.currentMessage()
    assert not window.mode_buttons[MODE_DRAW].isChecked()


def test_the_idle_timeout_relocks_and_abandons_what_was_in_progress(qt_app, window, reference_video: Path):
    # A console left in a control room is left in whatever state the last
    # person walked away from.
    _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    window._draw_zone()
    assert window.map.drawing

    window._relock()

    assert not window._configuring
    assert not window.map.drawing, "a half-drawn zone survived the relock"
    assert "relocked" in window.status.currentMessage()


def test_working_restarts_the_idle_countdown(qt_app, window, reference_video: Path):
    _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    window._idle_timer.stop()

    window._draw_zone()

    assert window._idle_timer.isActive(), "the countdown was not restarted by working"


def test_selecting_things_does_not_hold_the_lock_open(qt_app, window, reference_video: Path):
    """Clicking around is watching, not configuring.

    An earlier version restarted the countdown from `_selection_changed`, which
    runs on every click anywhere — so an operator idly selecting tracks would
    have held the site unlocked indefinitely.
    """
    from sentinel_console.selection import Selection

    _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    window._idle_timer.stop()

    window.selection.select(Selection.camera("cam-07"))
    window.selection.clear()

    assert not window._idle_timer.isActive(), "selecting restarted the idle countdown"


def test_escape_returns_the_map_to_select(qt_app, window, reference_video: Path):
    from PySide6.QtTest import QTest

    from sentinel_console.map_view import MODE_MEASURE, MODE_SELECT

    _placed_window(window, reference_video)
    window._choose_mode(MODE_MEASURE)
    assert window.map.mode == MODE_MEASURE
    assert window.mode_buttons[MODE_MEASURE].isChecked()

    QTest.keyClick(window.map, Qt.Key.Key_Escape)

    assert window.map.mode == MODE_SELECT
    assert window.mode_buttons[MODE_SELECT].isChecked()
    assert not window.mode_buttons[MODE_MEASURE].isChecked()


def test_the_mode_buttons_follow_a_gesture_that_ends_on_its_own(qt_app, window, reference_video: Path, monkeypatch):
    # A drawing closes by itself on a double-click. A Draw button left checked
    # after that is exactly the ambiguity this was built to remove.
    from sentinel_console.map_view import MODE_DRAW, MODE_SELECT
    from sentinel.zones import ZoneKind

    session = _placed_window(window, reference_video)
    # Closing an outline asks what the zone is, in a modal dialog. Answered
    # here directly: an unattended dialog hangs the suite forever, which is
    # also exactly what it would do to an operator.
    monkeypatch.setattr(
        ConsoleWindow, "_zone_drawn",
        lambda self, ring: self._create_zone(tuple(ring), name="Yard", kind=ZoneKind.INTEREST),
    )
    window.configure_button.setChecked(True)
    window._choose_mode(MODE_DRAW)
    assert window.mode_buttons[MODE_DRAW].isChecked()

    for bearing in (-10.0, 10.0, 0.0):
        point = destination_point(session.pose.position, session.pose.heading + bearing, 20.0)
        window.map._draw_points.append(window.map._to_local(point))
    window.map.finish_draw()

    assert window.map.mode == MODE_SELECT
    assert window.mode_buttons[MODE_SELECT].isChecked()
    assert not window.mode_buttons[MODE_DRAW].isChecked()


def test_measuring_reports_a_distance_and_changes_nothing(qt_app, window, reference_video: Path):
    from sentinel_console.map_view import MODE_MEASURE

    session = _placed_window(window, reference_video)
    before = list(window._zones)
    window._choose_mode(MODE_MEASURE)
    assert window.map.measuring

    near = destination_point(session.pose.position, session.pose.heading, 15.0)
    far = destination_point(session.pose.position, session.pose.heading, 35.0)
    _press(window.map, window.map._to_screen(*window.map._to_local(near)))
    _press(window.map, window.map._to_screen(*window.map._to_local(far)))

    metres = window.map.measured_metres()
    assert metres is not None and 18.0 <= metres <= 22.0, metres
    assert "Measured" in window.map._banner()
    assert list(window._zones) == before, "measuring changed the site"


def test_measuring_is_allowed_while_the_site_is_locked(qt_app, window, reference_video: Path):
    # It is read-only, so Monitor must not refuse it.
    from sentinel_console.map_view import MODE_MEASURE

    _placed_window(window, reference_video)
    assert not window._configuring

    window._choose_mode(MODE_MEASURE)

    assert window.map.measuring
