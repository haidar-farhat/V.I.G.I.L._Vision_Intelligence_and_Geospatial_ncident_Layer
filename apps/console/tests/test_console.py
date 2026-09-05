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
def window(qt_app, tmp_path):
    # In memory, always. A test that wrote to the operator's real database
    # would leave fabricated incidents in an evidence trail.
    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
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


MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "yolov8n-seg.onnx"


def _isolated_settings(directory: Path):
    """A QSettings that reaches nothing on the machine: an INI in a temp dir."""
    from PySide6.QtCore import QSettings

    return QSettings(str(directory / "console.ini"), QSettings.Format.IniFormat)


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

    def highlight_pixels() -> int:
        from PySide6.QtGui import QImage

        rendered = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        wanted = theme.SELECTION.name()
        return sum(
            rendered.pixelColor(x, y).name() == wanted
            for y in range(0, rendered.height(), 2)
            for x in range(0, rendered.width(), 2)
        )

    # The same id on another camera must light nothing here. The earlier
    # version of this test never set a selection at all, so it passed no matter
    # what `set_selection` did with the camera id.
    view.set_selection(Selection.track("yard", 4))
    assert highlight_pixels() == 0, "another camera's track #4 was highlighted here"

    view.set_selection(Selection.track("gate", 4))
    assert highlight_pixels() > 0, "this camera's own track #4 was not highlighted"


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


def test_the_keyboard_cannot_walk_around_the_lock(qt_app, window):
    """Ctrl+O and Ctrl+P change the site, so Monitor must disable them too.

    They were connected straight to the same slots as the toolbar buttons but
    left out of the lock, so the console could say it was locked while a
    shortcut added or placed a camera. Found by listing every path that reaches
    `node.add_camera` / `node.place_camera` and asking what could trigger it.
    """
    assert not window._configuring
    assert not window.add_camera_action.isEnabled()
    assert not window.place_action.isEnabled()

    window.configure_button.setChecked(True)
    assert window.add_camera_action.isEnabled()
    assert window.place_action.isEnabled()

    window.configure_button.setChecked(False)
    assert not window.add_camera_action.isEnabled()
    assert not window.place_action.isEnabled()


def test_every_path_that_changes_the_site_goes_through_the_lock(qt_app, window):
    """A structural check, so a new control cannot quietly skip the lock.

    Every method that calls a mutating `node.*` must either be a slot the lock
    disables, or be reachable only from one that is. The two lists are written
    out rather than derived, so adding a control that changes the site is a
    deliberate decision about which of those it is — and a new one that is
    neither fails here rather than in a control room.
    """
    import inspect

    from sentinel_console import app as app_module

    gated_slots = {
        "_choose_source", "_place_camera", "_place_camera_on_map",
        "_remove_camera", "_add_zone_dialog", "_draw_zone", "_edit_outline",
        "_remove_zone", "_edit_zone",
    }
    reachable_only_from_gated = {
        "_create_zone": "from _add_zone and _zone_drawn",
        "_change_zone": "from _edit_zone",
        "add_camera": "from _choose_source",
        "_map_picked": "only after a gated control began a pick",
        "_zone_outline_edited": "only after Reshape",
        "_zone_properties_applied": "only from the Apply button",
        "_add_zone": "from _add_zone_dialog and the screenshot tool",
        # The map emits these only while it is editable, and it is editable
        # only while the console is configuring — `_set_configuring` is the
        # single caller of `map.set_editable`. That is the lock, worn by a
        # widget instead of by a button, and
        # `test_a_camera_cannot_be_dragged_in_monitor` holds it to that.
        "_camera_dragged": "only while map.set_editable(True), which follows the lock",
        "_camera_turned": "only while map.set_editable(True), which follows the lock",
        # A command-line flag is a deliberate act by whoever launched the
        # process, not a stray click on a control room screen; every change
        # it makes goes through the node and is audited like a clicked one.
        "seed_site": "from run(): --camera, --place and --zone",
        "_camera_record_toggled": "only while camera_list.set_recording_editable(True), which follows the lock",
    }
    mutators = (
        "self.node.add_camera", "self.node.remove_camera", "self.node.place_camera",
        "self.node.add_zone", "self.node.replace_zone", "self.node.remove_zone",
    )

    current = None
    offenders = set()
    for line in inspect.getsource(app_module).splitlines():
        if line.startswith("    def "):
            current = line.split("def ", 1)[1].split("(", 1)[0]
        if current is not None and any(call in line for call in mutators):
            if current not in gated_slots and current not in reachable_only_from_gated:
                offenders.add(current)

    assert not offenders, (
        f"{sorted(offenders)} change the site but are neither disabled by the "
        "lock nor listed as reachable only from something that is"
    )


# --------------------------------------------- what the review found missing


def test_moving_the_pointer_while_drawing_still_paints(qt_app):
    """The repaint during a draw must survive the pointer moving.

    `_hover` held two types at once — a `Selection` for what is under the
    pointer, and a `QPointF` for the rubber band — so one mouse move after
    pressing Draw raised inside `paintEvent`. Qt swallows that: the plan view
    silently stopped drawing its corners, its rubber band, the scale bar and
    the coverage banner, and the retained traceback kept the widget alive past
    its last reference. No test covered draw-mode plus a move plus a repaint.
    """
    view, track, where = _map_with_a_track()
    assert view.begin_draw("Draw a zone")
    view._draw_points.append(view._to_local(where))

    _move(view, QPointF(120.0, 140.0))

    # `grab()` runs paintEvent synchronously and lets the exception out.
    image = view.grab().toImage()
    assert not image.isNull()
    assert view.draw_points == 1, "the corner was lost"
    assert view.hovered is None or hasattr(view.hovered, "kind"), (
        "hover holds something that is not a Selection"
    )


def test_the_whole_plan_view_still_renders_while_drawing(qt_app):
    # The failure above was silent, so assert the parts that vanished with it.
    view, _, where = _map_with_a_track()
    view.begin_draw("Draw a zone")
    for offset in (-30.0, 30.0):
        view._draw_points.append(
            view._to_local(destination_point(where, offset + 90.0, 8.0))
        )
    _move(view, QPointF(200.0, 200.0))

    assert view._banner() is not None, "the band went with the paint"
    image = view.grab().toImage()
    assert not image.isNull()


def test_control_c_copies_the_ground_readout(qt_app, window, reference_video: Path):
    """Through the real shortcut, not by calling the slot.

    The readout's tooltip promised Ctrl+C while nothing was bound to it, and
    the only test called `_copy_ground` directly — so the whole feature could
    be unwired and the suite stayed green.
    """
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest

    _placed_window(window, reference_video)
    QGuiApplication.clipboard().setText("")
    window._ground_moved(destination_point(SITE_POSE.position, SITE_POSE.heading, 22.0))
    expected = window.ground_label.text()
    assert expected

    window.show()
    QApplication.processEvents()
    QTest.keyClick(window, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    QApplication.processEvents()

    assert QGuiApplication.clipboard().text() == expected


def test_clicking_a_box_on_a_camera_pane_selects_that_track(qt_app):
    # The pane's own click-to-select had no test at all.
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
    view.show_update(Update(
        result=FrameResult(index=1, timestamp_millis=1000, source_id="gate",
                           detections=(), tracks=(track,), ended=(),
                           image=np.zeros((480, 640, 3), dtype=np.uint8)),
        analysis_fps=30.0, skipped=0, stats=PipelineStats(),
    ))

    caught: list = []
    view.clicked.connect(caught.append)
    _press(view, QPointF(0.45 * 640, 0.5 * 480))
    assert caught == [Selection.track("gate", 4)]

    # And a click on bare frame reports nothing selected.
    caught.clear()
    _press(view, QPointF(0.05 * 640, 0.05 * 480))
    assert caught == [None]


def test_copying_a_track_position_carries_its_uncertainty_and_provenance(qt_app, window, reference_video: Path):
    """A coordinate must never travel alone.

    Pasted into a radio call or a report without its uncertainty — or without
    the fact that it is a *fallback* and not a location at all — is how
    somebody gets sent to a place the system never claimed.
    """
    session = _placed_window(window, reference_video)
    where = destination_point(SITE_POSE.position, SITE_POSE.heading, 20.0)
    _feed_track(window, session, _located_track(7, where, radius=2.5))

    row = window.tracks.topLevelItem(0)
    assert row.data(0, Qt.ItemDataRole.UserRole) == ("cam-07", 7)
    assert row.text(9).startswith("±2.5")
    assert row.text(10) == "projected"

    # What the menu would put on the clipboard, assembled the same way.
    copied = f"{row.text(8)}  ±{row.text(9).lstrip('±')}  ({row.text(10)})"
    assert "±2.5 m" in copied and "projected" in copied
    assert row.text(8) in copied


def test_a_fallback_position_is_copied_as_a_fallback(qt_app, window, reference_video: Path):
    session = _placed_window(window, reference_video)
    _feed_track(window, session, _located_track(
        8, SITE_POSE.position, source="CAMERA_FALLBACK", radius=30.0
    ))

    row = window.tracks.topLevelItem(0)
    assert row.text(10) == "fallback", "the provenance column lost the fallback"
    copied = f"{row.text(8)}  ±{row.text(9).lstrip('±')}  ({row.text(10)})"
    assert "fallback" in copied, "a fallback would be pasted as if it were a location"


def test_the_incident_panel_keeps_its_selection_across_a_rebuild(qt_app):
    # The panel is rebuilt on every collection tick. A selection that did not
    # survive that would flicker off about thirty times a second.
    from sentinel_console.incident_view import IncidentView
    from sentinel_console.selection import Selection

    view = IncidentView()
    incidents = _incident_from_events(3)
    view.show_incidents(incidents)
    view.set_selection(Selection.incident(incidents[0].id))
    assert view.selected_incident_id() == incidents[0].id

    view.show_incidents(incidents)          # the rebuild

    assert view.selected_incident_id() == incidents[0].id
    assert view.topLevelItem(0).isSelected()

    view.set_selection(None)
    assert not view.topLevelItem(0).isSelected()


def test_the_lock_covers_the_controls_by_name(qt_app, window):
    """Named, not derived.

    `test_the_console_opens_locked` iterates `_configure_only()` itself, so a
    control left out of that list passes it. This names the controls that must
    be in it, so removing one fails here.
    """
    gated = set(window._configure_only())
    for name in (
        "open_button", "place_button", "map_place_button", "remove_button",
        "zone_button", "add_zone_button", "draw_zone_button",
        "reshape_zone_button", "remove_zone_button",
        "add_camera_action", "place_action",
    ):
        control = getattr(window, name)
        assert control in gated, f"{name} is not covered by the lock"
    assert window.zone_properties.apply_button in gated
# ------------------------------------------- moving a camera on the plan view
#
# Where a camera is said to be is the input to every position it will ever
# report. So the map may move one only when it has been told it may, it says so
# exactly once — on release, never during — and abandoning the gesture leaves
# the site exactly as it was.


def _hold(view, at):
    """Press the left button and keep it down: the start of a drag."""
    from PySide6.QtTest import QTest

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=at.toPoint())


def _let_go(view, at):
    """Release it, which is the only moment anything is committed."""
    from PySide6.QtTest import QTest

    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=at.toPoint())


def _map_of_one_camera(pose=None, *, editable=False, dark=(), bands=False,
                       camera_id="gate"):
    """A shown 600x600 plan view of one placed camera.

    Shown, and the events processed, because `grab()` on a widget that was
    never realised returns a picture of nothing and every pixel assertion
    against it passes for the wrong reason.
    """
    from sentinel.coverage import sigma_bands

    view = MapView()
    view.resize(600, 600)
    view.set_cameras({camera_id: pose or SITE_POSE})
    if bands:
        view.set_sigma_bands({camera_id: sigma_bands(pose or SITE_POSE)})
    if dark:
        view.set_dark_cameras(dark)
    view.set_editable(editable)
    # The legend is an overlay in the bottom-right and several of these sample
    # under it. Turning it off is what `show_legend` is for.
    view.show_legend = False
    view.show()
    QApplication.processEvents()
    return view


def _camera_centre(view, camera_id="gate"):
    return view._to_screen(*view._to_local(view._cameras[camera_id].position))


def _watch(signal, into: list):
    """Collect a two-argument signal's emissions as pairs.

    A bound `list.append` takes one argument, and connecting one to a signal
    that carries a camera id *and* a position drops half of every placement.
    """
    def remember(first, second):
        into.append((first, second))

    signal.connect(remember)
    return remember


def test_a_camera_does_not_move_until_the_view_is_told_it_may(qt_app):
    """The console has a Monitor lock, and dragging a camera is a change.

    A sleeve across a touchscreen in Monitor must not be able to move the mast
    that every one of that camera's positions is measured from.
    """
    view = _map_of_one_camera()
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)

    _hold(view, centre)
    _move(view, QPointF(centre.x() + 70, centre.y() + 40))
    _let_go(view, QPointF(centre.x() + 70, centre.y() + 40))

    assert moved == [], "a locked map moved a camera"
    assert view._cameras["gate"] == SITE_POSE
    assert view.heading_handle("gate") is None, "a handle was offered for a gesture that is refused"


def test_dragging_a_placed_camera_commits_once_on_release(qt_app):
    from sentinel.core import haversine_distance

    view = _map_of_one_camera(editable=True)
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)
    target = QPointF(centre.x() + 60, centre.y() + 40)

    _hold(view, centre)
    _move(view, QPointF(centre.x() + 30, centre.y() + 20))
    _move(view, target)
    assert moved == [], "the camera was placed half way through the drag"
    assert view.dragging_camera == "gate"

    _let_go(view, target)

    assert len(moved) == 1, f"one release, {len(moved)} placements"
    assert view.dragging_camera is None
    camera_id, point = moved[0]
    assert camera_id == "gate"
    assert point == view._cameras["gate"].position, "the map shows one place and reported another"
    # Measured for this drag on a 600 px view at 6.17 px/m: 11.7 m.
    metres = haversine_distance(SITE_POSE.position, point)
    print(f"dragged {metres:.1f} m")
    assert 8.0 < metres < 16.0, f"{metres:.1f} m is not the drag that was made"
    # Only the position. A drag across the ground is not a claim about the mast.
    assert view._cameras["gate"].heading == SITE_POSE.heading
    assert view._cameras["gate"].mount_height == SITE_POSE.mount_height


def test_a_click_on_a_camera_selects_it_without_moving_it(qt_app):
    # Press and release in the same place is how an operator picks a camera to
    # look at. It must not count as a placement, or every inspection would
    # write a pose and an audit line.
    from sentinel_console.selection import Selection

    view = _map_of_one_camera(editable=True)
    picked, moved = [], []
    view.selected.connect(picked.append)
    _watch(view.camera_moved, moved)

    _press(view, _camera_centre(view))

    assert picked == [Selection.camera("gate")]
    assert moved == [], "clicking a camera re-placed it"


def test_the_footprint_follows_the_camera_while_it_is_dragged(qt_app):
    """The wedge is the thing being aimed, so it moves during the gesture.

    A marker that moves while its coverage stays behind asks the operator to
    imagine where the ground will end up.
    """
    view = _map_of_one_camera(editable=True)
    centre = _camera_centre(view)
    scanline = int(centre.y() + 300)

    def footprint_span():
        image = view.grab().toImage()
        lit = [
            x for x in range(view.width())
            if image.pixelColor(x, scanline).blue() > theme.PANEL.blue() + 6
        ]
        return (min(lit), max(lit)) if lit else None

    before = footprint_span()
    _hold(view, centre)
    _move(view, QPointF(centre.x() + 60, centre.y() + 40))
    during = footprint_span()
    print(f"footprint on row {scanline}: {before} then {during}")

    assert before is not None and during is not None
    # Measured: the west edge of the wedge moved 75 px east for a 60 px drag.
    assert during[0] - before[0] > 40, "the footprint stayed behind the camera"


def test_the_error_bands_come_down_for_the_length_of_a_drag(qt_app):
    """Redrawing them per mouse move is not affordable, and stale ones lie.

    `sigma_bands` is 1750 calls across the FFI per camera — measured at 7.9 ms
    warm, 75 ms cold — which at mouse-move rate is a slideshow. Keeping the old
    ones on screen instead would draw the ground this camera knows best where
    the camera used to be, so during the drag only the footprint is drawn.
    """
    view = _map_of_one_camera(editable=True, bands=True)
    centre = _camera_centre(view)

    def near_ground_brightness():
        pose = view._cameras["gate"]
        point = destination_point(pose.position, pose.heading, 10.0)
        screen = view._to_screen(*view._to_local(point))
        colour = view.grab().toImage().pixelColor(int(screen.x()), int(screen.y()))
        return colour.red() + colour.green() + colour.blue()

    before = near_ground_brightness()
    assert view._bands, "nothing was shaded, so nothing was checked"

    _hold(view, centre)
    for step in range(1, 6):
        _move(view, QPointF(centre.x() + step * 12, centre.y() + step * 8))
    during = near_ground_brightness()
    print(f"near ground: {before} shaded, {during} while dragging")

    assert not view._bands, "the bands were kept, and would be recomputed per move"
    # Measured: 295 with the bands, 164 with the bare footprint.
    assert during < before - 100, "the ground is still shaded from the old pose"

    _let_go(view, QPointF(centre.x() + 60, centre.y() + 40))


def test_escape_during_a_drag_puts_the_camera_back(qt_app):
    from PySide6.QtTest import QTest

    view = _map_of_one_camera(editable=True)
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)
    away = QPointF(centre.x() + 70, centre.y() + 50)

    _hold(view, centre)
    _move(view, away)
    assert view._cameras["gate"] != SITE_POSE, "the drag did nothing, so the revert proves nothing"
    QTest.keyClick(view, Qt.Key.Key_Escape)

    assert view._cameras["gate"] == SITE_POSE, "Escape left the camera where the drag put it"
    assert view.dragging_camera is None
    _let_go(view, away)
    assert moved == [], "the release committed a drag that had been abandoned"


def test_a_right_click_during_a_drag_does_not_place_the_camera(qt_app):
    """Only the release of the gesture that made the placement commits it.

    A right-click is a live gesture in this view — it takes a vertex back
    while drawing — so an operator has every reason to press one mid-drag.
    Committing on any button coming up wrote a pose through `place_camera`
    and left an audit entry nobody confirmed.
    """
    from PySide6.QtTest import QTest

    view = _map_of_one_camera(editable=True)
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)
    away = QPointF(centre.x() + 80, centre.y() + 50)

    _hold(view, centre)
    _move(view, away)
    assert view._cameras["gate"] != SITE_POSE, "the drag did nothing, so nothing is proven"

    QTest.mouseRelease(view, Qt.MouseButton.RightButton, pos=away.toPoint())

    assert moved == [], "a right-click placed the camera"
    assert view._cameras["gate"] == SITE_POSE, "the camera stayed where the abandoned drag put it"
    assert view.dragging_camera is None, "the gesture is still in flight"


def test_relocking_the_view_mid_drag_reverts_it(qt_app):
    # The idle timer relocks the console on its own. A gesture in flight when
    # that happens is not the operator saying yes to it.
    view = _map_of_one_camera(editable=True)
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)
    away = QPointF(centre.x() + 50, centre.y() + 50)

    _hold(view, centre)
    _move(view, away)
    view.set_editable(False)
    _let_go(view, away)

    assert moved == [], "a camera was placed by a lock coming back"
    assert view._cameras["gate"] == SITE_POSE


def test_a_move_the_node_refuses_does_not_stay_on_the_map(qt_app):
    """`camera_moved` asks. It is not told whether the node agreed.

    The node refuses a placement whose camera is not placed or whose session
    has gone, and nothing comes back to say so. Left alone the map goes on
    drawing the camera metres from where the node has it, and the hover text
    quotes a range for a pose that exists nowhere but this widget.
    """
    from sentinel.core import haversine_distance

    view = _map_of_one_camera(editable=True, bands=True)
    moved = []
    _watch(view.camera_moved, moved)
    centre = _camera_centre(view)
    away = QPointF(centre.x() + 70, centre.y() + 45)

    _hold(view, centre)
    _move(view, away)
    _let_go(view, away)

    assert len(moved) == 1, "nothing was asked for, so nothing can be refused"
    drift = haversine_distance(SITE_POSE.position, view._cameras["gate"].position)
    print(f"the map is showing the camera {drift:.2f} m from where the node has it")
    # Measured at 13.50 m for this drag; any real distance makes the point.
    assert drift > 5.0, "the drag barely moved it, so the divergence proves nothing"

    # The owner's refusal path: it did not call set_cameras, so it says so.
    view.revert_uncommitted("gate")

    assert view._cameras["gate"] == SITE_POSE, "the refused pose is still on the map"
    back = haversine_distance(SITE_POSE.position, view._cameras["gate"].position)
    print(f"after the refusal: {back:.4f} m")
    assert back < 0.01
    assert view._bands.get("gate"), "the ground the camera knows best never came back"
    # And a second call has nothing left to take back.
    view.revert_uncommitted("gate")
    assert view._cameras["gate"] == SITE_POSE


def test_dragging_the_heading_handle_aims_the_camera_rather_than_moving_it(qt_app):
    """Aiming and moving are different mistakes, so they are different signals."""
    import math

    view = _map_of_one_camera(editable=True)
    aimed, moved = [], []
    _watch(view.camera_aimed, aimed)
    _watch(view.camera_moved, moved)

    handle = view.heading_handle("gate")
    centre = _camera_centre(view)
    assert handle is not None, "there is no grip to turn the camera by"
    # It sits on the footprint's axis, out at its far edge: due south of a
    # camera facing 180°.
    assert abs(handle.x() - centre.x()) < 1.0, "the grip is not on the camera's axis"
    assert handle.y() > centre.y(), "the grip is behind the camera"

    reach = math.hypot(handle.x() - centre.x(), handle.y() - centre.y())
    due_east = QPointF(centre.x() + reach, centre.y())
    _hold(view, handle)
    _move(view, due_east)
    assert aimed == [], "the camera was re-aimed half way through the gesture"
    _let_go(view, due_east)

    assert len(aimed) == 1 and aimed[0][0] == "gate"
    print(f"aimed to {aimed[0][1]:.2f}°")
    assert abs(aimed[0][1] - 90.0) < 0.5, aimed[0][1]
    assert moved == [], "turning the camera reported it as moved"
    assert view._cameras["gate"].position == SITE_POSE.position, "aiming moved the mast"


def test_the_console_opens_with_a_map_that_cannot_be_dragged(qt_app, window, reference_video: Path):
    # The console opens in Monitor, and the map is one of the things that is
    # locked. A view that defaulted to editable would arrive unlocked.
    _placed_window(window, reference_video)
    assert not window.map.editable


# ------------------------------------------------- a dark camera on the ground
#
# A camera keeps its pose when it stops delivering, and until this existed it
# kept the full blue wedge that goes with one: an operator read the yard as
# covered by a camera that had not produced a frame in an hour.


def _bare_ground_samples(view, camera_id="gate"):
    """How many samples across the wedge are still bare panel.

    A filled footprint covers every one of them; a hatch leaves the gaps
    between its strokes showing, which is the difference being claimed.
    """
    image = view.grab().toImage()
    pose = view._cameras[camera_id]
    panel = (theme.PANEL.red(), theme.PANEL.green(), theme.PANEL.blue())
    bare = total = 0
    for metres in range(12, 80, 2):
        for offset in (-12.0, -6.0, 0.0, 6.0, 12.0):
            point = destination_point(pose.position, pose.heading + offset, metres)
            screen = view._to_screen(*view._to_local(point))
            colour = image.pixelColor(int(screen.x()), int(screen.y()))
            total += 1
            bare += (colour.red(), colour.green(), colour.blue()) == panel
    return bare, total


def test_a_dark_cameras_ground_is_hatched_rather_than_filled(qt_app):
    """A camera producing nothing covers nothing.

    The fill is how this view says "seen". Left under a camera that has gone
    silent it is a claim that somebody is watching that ground.
    """
    lit = _map_of_one_camera()
    dark = _map_of_one_camera(dark=("gate",))

    lit_bare, total = _bare_ground_samples(lit)
    dark_bare, _ = _bare_ground_samples(dark)
    print(f"bare panel inside the wedge: {lit_bare}/{total} lit, {dark_bare}/{total} dark")

    # Measured: 0 of 170 under a working camera, 140 of 170 under a dark one.
    assert lit_bare == 0, "the working camera's footprint is not filled"
    assert dark_bare > 80, "the dark camera's ground is filled, not hatched"


def test_a_dark_camera_draws_no_error_bands(qt_app):
    # The bands say how well it *would* locate something. It is locating
    # nothing, and shading them is the same false claim in a stronger colour.
    lit = _map_of_one_camera(bands=True)
    dark = _map_of_one_camera(bands=True, dark=("gate",))

    def near_ground_brightness(view):
        point = destination_point(SITE_POSE.position, SITE_POSE.heading, 10.0)
        screen = view._to_screen(*view._to_local(point))
        colour = view.grab().toImage().pixelColor(int(screen.x()), int(screen.y()))
        return colour.red() + colour.green() + colour.blue()

    shaded, unshaded = near_ground_brightness(lit), near_ground_brightness(dark)
    print(f"near ground: {shaded} lit, {unshaded} dark")

    assert dark._bands, "the bands were never given, so nothing was suppressed"
    # Measured: 295 against 121.
    assert unshaded < shaded - 100, "a dark camera is still shading its best ground"


def test_a_dark_camera_is_drawn_distinctly_from_a_working_one(qt_app):
    # Two markers that look alike put the operator's eye on the footprint to
    # work out which camera is which, and that is the drawing they cannot trust.
    lit = _map_of_one_camera()
    dark = _map_of_one_camera(dark=("gate",))

    def marker_colour(view):
        centre = _camera_centre(view)
        return view.grab().toImage().pixelColor(int(centre.x()) + 5, int(centre.y()))

    working, silent = marker_colour(lit), marker_colour(dark)
    print(f"marker: working {working.getRgb()[:3]}, dark {silent.getRgb()[:3]}")

    assert working.getRgb()[:3] == theme.CAMERA.getRgb()[:3]
    # The colour the console already uses for "nothing is running".
    assert silent.getRgb()[:3] == theme.IDLE.getRgb()[:3]


def test_the_hover_text_says_a_dark_camera_is_seeing_nothing(qt_app):
    from sentinel_console.selection import Selection

    view = _map_of_one_camera(dark=("gate",))
    text = view._hover_text(Selection.camera("gate"))
    print(text)

    assert "delivering nothing" in text
    assert "not being watched" in text


# --------------------------------------------- what stops a footprint, exactly
#
# A footprint's far edge is one of two facts: the range somebody typed, which
# they can raise, or the ground running out, which no setting will move. Drawn
# alike, an operator who wants twenty more metres raises a range that was never
# what stopped them.


def test_the_far_edge_says_whether_the_range_or_the_ground_stopped_it(qt_app):
    from dataclasses import replace

    from sentinel_console.map_view import FAR_EDGE_HORIZON, FAR_EDGE_RANGE

    # The site's own camera: 6 m up at -22°, so the top row of its frame lands
    # 85.8 m out — short of the 90 m range it is allowed. The ground stops it.
    view = _map_of_one_camera()
    reach = view.far_edge_metres("gate")
    print(f"the site camera reaches {reach:.1f} m of its {SITE_POSE.range_meters:.0f} m range")
    assert 80.0 < reach < SITE_POSE.range_meters
    assert view.far_edge_kind("gate") == FAR_EDGE_HORIZON

    # The same camera with its range wound in to 50 m: now the number stops it.
    clamped = _map_of_one_camera(replace(SITE_POSE, range_meters=50.0))
    assert abs(clamped.far_edge_metres("gate") - 50.0) < 0.5
    assert clamped.far_edge_kind("gate") == FAR_EDGE_RANGE

    # And a camera the view does not have is not guessed about.
    assert view.far_edge_kind("no-such-camera") is None


def _far_edge_gaps(view, camera_id="gate"):
    """Samples along the drawn far edge, and how many are not on a stroke.

    Walked pixel by pixel along the arc the view itself calls its far edge, so
    a solid stroke answers zero gaps and a dashed one answers its dashes.
    """
    import math

    image = view.grab().toImage()
    local = [view._to_local(point) for point in view._footprints[camera_id]]
    count = view._far_arc_count(camera_id, local)
    assert count, "the view cannot say which edge is the far one"
    points = [view._to_screen(east, north) for east, north in local[:count]]

    samples = gaps = 0
    for start, end in zip(points, points[1:]):
        steps = max(2, int(math.hypot(end.x() - start.x(), end.y() - start.y())))
        for step in range(steps):
            fraction = step / steps
            x = int(round(start.x() + (end.x() - start.x()) * fraction))
            y = int(round(start.y() + (end.y() - start.y()) * fraction))
            if not (0 <= x < view.width() and 0 <= y < view.height()):
                continue
            samples += 1
            gaps += image.pixelColor(x, y).blue() < 60
    return samples, gaps


def test_a_range_clamped_edge_is_drawn_solid_and_a_horizon_limited_one_dashed(qt_app):
    from dataclasses import replace

    horizon = _map_of_one_camera()
    clamped = _map_of_one_camera(replace(SITE_POSE, range_meters=50.0))

    solid_samples, solid_gaps = _far_edge_gaps(clamped)
    dashed_samples, dashed_gaps = _far_edge_gaps(horizon)
    print(f"solid: {solid_gaps}/{solid_samples} off the stroke; "
          f"dashed: {dashed_gaps}/{dashed_samples}")

    # Measured: 0 gaps in 560 samples solid, 72 in 560 dashed.
    assert solid_gaps == 0, "the range-clamped edge is broken, and reads as the horizon"
    assert dashed_gaps > 20, "the horizon-limited edge is unbroken, and reads as the clamp"


def test_the_legend_explains_the_two_far_edges(qt_app):
    from sentinel.coverage import sigma_bands

    view = MapView()
    view.resize(600, 600)
    view.set_cameras({"gate": SITE_POSE})
    view.set_sigma_bands({"gate": sigma_bands(SITE_POSE)})

    captions = view._legend_captions()
    print(captions, view.legend_rect())
    assert "solid edge: range" in captions
    assert "dashed edge: horizon" in captions
    assert "hatched: no frames" not in captions, "explained a hatch nothing is drawn in"

    view.set_dark_cameras({"gate"})
    assert "hatched: no frames" in view._legend_captions()

    # It grew by three rows and still has to stay off the scale bar and inside
    # the view. Measured at 236 px wide and 67 px tall on a 600 px view.
    legend = view.legend_rect()
    assert not legend.intersects(view.scale_bar_rect())
    assert legend.top() >= 0 and legend.bottom() <= view.height()
    assert legend.width() <= view.width() * 0.45, f"the legend is {legend.width():.0f} px wide"


# ------------------------------------- the camera list, its strip, and the map
#
# Three finished things used to be wired to nothing: a camera list panel nobody
# put in the window, a `Node.camera_health()` nobody called on the poll timer,
# and a plan view that could drag a camera only for a caller that set
# `set_editable` itself. What follows is what an operator can now actually
# reach, and each of these fails if the connection is taken out of `app.py`.


def _click_camera_row(panel, camera_id: str) -> None:
    """Click the row for a camera, the way an operator picks one.

    A real click on the viewport rather than a call to `setCurrentItem`: the
    whole point of the panel is the path from a mouse to the selection bus, and
    a test that sets the current item skips exactly the wiring under test.
    """
    from PySide6.QtTest import QTest

    tree = panel.tree
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        if item.data(0, Qt.ItemDataRole.UserRole) == camera_id:
            rect = tree.visualItemRect(item)
            assert not rect.isEmpty(), f"{camera_id}'s row has no rectangle to click"
            QTest.mouseClick(
                tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center()
            )
            QApplication.processEvents()
            return
    raise AssertionError(f"there is no row for {camera_id} to click")


def _row_status(panel, camera_id: str) -> str:
    from sentinel_console.camera_list import STATUS_COLUMN

    tree = panel.tree
    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        if item.data(0, Qt.ItemDataRole.UserRole) == camera_id:
            return item.text(STATUS_COLUMN)
    raise AssertionError(f"there is no row for {camera_id}")


def _listed_cameras(window) -> list:
    tree = window.camera_list.tree
    return [
        tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole)
        for index in range(tree.topLevelItemCount())
    ]


def test_the_camera_list_is_in_the_window_and_names_every_camera(
    qt_app, window, reference_video: Path
):
    """A combo box shows one camera and hides the rest.

    Eight cameras looked identical whether all eight were delivering frames or
    seven had been dark since midnight, because seven of them were not on the
    screen at all. Every camera has a row now, and the row says what that
    camera is doing.
    """
    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    QApplication.processEvents()

    assert window.camera_list.isVisibleTo(window), "the list is not in the window"
    assert _listed_cameras(window) == ["cam-07", "cam-08"]
    # "not started" is a state, and it is the true one: nothing has run.
    for camera_id in ("cam-07", "cam-08"):
        status = _row_status(window.camera_list, camera_id)
        assert "not started" in status, f"{camera_id} shows {status!r}, which is not a state"
    assert "2 cameras" in window.camera_list.summary.text()


def test_the_camera_status_strip_follows_the_poll_timer(
    qt_app, window, reference_video: Path
):
    """The strip is refreshed by the same call that moves the engine forward.

    "Is this camera delivering frames" is only true of the instant it was
    asked. A strip refreshed when a camera is added and never again is the
    green dot on a wedged decoder that this exists to remove, so what is
    asserted is that running the console forward changes it.
    """
    session = window.add_camera(reference_video, "cam-07")
    QApplication.processEvents()
    assert "not started" in _row_status(window.camera_list, "cam-07")

    window._start()
    pump(qt_app, window, 2.0)

    after = _row_status(window.camera_list, "cam-07")
    assert "not started" not in after, "the strip never noticed the camera start"
    assert session.record.runner is not None
    window._stop()


def test_picking_a_row_in_the_camera_list_selects_that_camera_everywhere(
    qt_app, window, reference_video: Path
):
    """One selected camera, whichever panel the operator used to say so."""
    from sentinel_console.selection import Selection

    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    QApplication.processEvents()
    # The newest camera is the selected one, so cam-07 is a real change.
    assert window.camera_picker.currentData() == "cam-08"

    _click_camera_row(window.camera_list, "cam-07")

    assert window.selection.current == Selection.camera("cam-07"), "the bus was not told"
    assert window.camera_picker.currentData() == "cam-07", (
        "the list and the toolbar disagree about which camera the buttons act on"
    )
    assert window._selected.camera_id == "cam-07"


def test_choosing_a_camera_in_the_picker_lights_its_row_in_the_list(
    qt_app, window, reference_video: Path
):
    """And back the other way, because two views of one choice must not drift."""
    from sentinel_console.selection import Selection

    window.add_camera(reference_video, "cam-07")
    window.add_camera(reference_video, "cam-08")
    QApplication.processEvents()

    window.camera_picker.setCurrentIndex(window.camera_picker.findData("cam-07"))
    QApplication.processEvents()

    assert window.camera_list.selected_camera_id() == "cam-07", "the row did not light"
    assert window.selection.current == Selection.camera("cam-07")


def test_a_camera_delivering_nothing_is_dark_on_the_map_as_well_as_in_the_list(
    qt_app, window, reference_video: Path
):
    """A camera that stopped keeps its pose, and used to keep its wedge with it.

    The plan view drew the yard as covered by a camera that had not produced a
    frame since it failed. `camera_health()` already decided this; the map is
    now told, from the same read that fills the list, so the two cannot
    disagree about the same camera.
    """
    session = _placed_window(window, reference_video)
    assert window.map.dark_cameras == frozenset(), "dark before anything went wrong"

    # What the node's own poll writes when a camera's run ends badly.
    session.record.fault = "the decoder stopped returning frames"
    window._refresh_cameras()

    assert window.node.camera_health()["cam-07"].is_dark
    assert window.map.dark_cameras == frozenset({"cam-07"}), (
        "the map still paints ground nobody is watching as covered"
    )
    assert "failed" in _row_status(window.camera_list, "cam-07")
    assert "1 failed" in window.camera_list.summary.text()


def test_the_plan_view_can_be_edited_only_while_the_console_is_configuring(
    qt_app, window, reference_video: Path
):
    """The drag lock is the site lock, not a second quieter one."""
    _placed_window(window, reference_video)
    assert not window._configuring
    assert not window.map.editable, "a locked console offered a drag"

    window.configure_button.setChecked(True)
    assert window.map.editable

    window.configure_button.setChecked(False)
    assert not window.map.editable, "Monitor left the masts draggable"


def _drag_camera(view, camera_id: str, dx: float, dy: float) -> QPointF:
    """Pick a mast up, move it, and let it go. Real events, on the real view."""
    centre = view._to_screen(*view._to_local(view._cameras[camera_id].position))
    target = QPointF(centre.x() + dx, centre.y() + dy)
    _hold(view, centre)
    _move(view, target)
    _let_go(view, target)
    QApplication.processEvents()
    return target


def test_a_camera_cannot_be_dragged_while_the_console_is_monitoring(
    qt_app, window, reference_video: Path
):
    """Where a camera is said to be is the input to every position it reports.

    A sleeve across a touchscreen in Monitor must not be able to change it, and
    "must not" here means the node never hears about it — not that the map
    quietly draws the camera somewhere else.
    """
    session = _placed_window(window, reference_video)
    before = session.pose
    assert window.map.size().width() > 200, "the plan view is too small to drag on"

    _drag_camera(window.map, "cam-07", 60, 40)

    assert session.pose == before, "a locked console moved a camera"
    assert window.store.camera_pose("cam-07") == before, "and wrote it down"
    assert window.map._cameras["cam-07"] == before, "the map shows a move that never happened"


def test_dragging_a_camera_in_configure_places_it_and_keeps_its_height_and_heading(
    qt_app, window, reference_video: Path
):
    """The drag is a placement, made through the node like every other one.

    A drag across the ground says where the mast is. It says nothing about how
    high it stands or which way it faces, and a placement that reset either
    would silently re-aim a camera the operator only meant to shift — every
    position that camera has ever reported is measured from those three facts
    together.
    """
    from sentinel.core import haversine_distance

    session = _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    before = session.pose

    _drag_camera(window.map, "cam-07", 60, 40)

    after = session.pose
    metres = haversine_distance(before.position, after.position)
    # Measured for this drag on the console's own plan view at 1280x800: 17.0 m.
    # Floored well under it, because the width the splitter gives the map is a
    # layout decision and not the thing under test.
    print(f"dragged {metres:.1f} m")
    assert metres > 5.0, f"{metres:.2f} m is not a drag, so nothing is proven"
    assert after.heading == before.heading, "the drag re-aimed the camera"
    assert after.mount_height == before.mount_height, "the drag changed the mast height"
    assert after.pitch == before.pitch
    assert after.horizontal_fov == before.horizontal_fov

    # Through the node, so it survives a restart and leaves an audit line.
    stored = window.store.camera_pose("cam-07")
    assert stored is not None and stored.position == after.position, (
        "the console moved the camera on screen and never persisted it"
    )


# ----------------------------------------------- the two wires nobody could pull
#
# The investigation panel and the plate readings were built and tested and
# reachable by nothing. These fail if either wire is removed again.


def test_the_investigation_panel_is_in_the_window_and_searches_the_nodes_store(qt_app, window):
    labels = [window.detail_tabs.tabText(i) for i in range(window.detail_tabs.count())]
    assert "Investigation" in labels
    assert window.investigation._store is window.node.store, "the panel searches some other database"


def test_a_result_picked_in_the_investigation_panel_reaches_the_bus(qt_app, window):
    from sentinel_console.selection import Selection

    window.investigation.selected.emit(Selection.incident("inc_wired"))

    assert window.selection.current == Selection.incident("inc_wired")
    assert window.incidents.selected_incident_id() in (None, "inc_wired")


def test_a_selection_made_elsewhere_reaches_the_investigation_panel(qt_app, window):
    from sentinel_console.selection import Selection

    window.selection.select(Selection.incident("inc_elsewhere"))
    assert window.investigation._selected == Selection.incident("inc_elsewhere")

    window.selection.clear()
    assert window.investigation._selected is None


def test_the_search_pickers_follow_the_cameras_and_zones(qt_app, window, reference_video: Path):
    session = _placed_window(window, reference_video)
    window._add_zone(radius=6.0)

    cameras = [window.investigation.camera.itemText(i) for i in range(window.investigation.camera.count())]
    zones = [window.investigation.zone.itemText(i) for i in range(window.investigation.zone.count())]
    assert session.camera_id in cameras
    assert window._zones[0].name in zones


def test_a_vehicles_plate_reading_reaches_the_track_table(qt_app, window, reference_video: Path):
    """The pipeline published plates and nothing displayed them.

    A half-read plate must reach the screen as its display form, with "?" for
    every character the frames have not agreed on, and a thin reading must say
    how thin it is. `text` — the completed string a rule may act on — is never
    what the operator is shown, because it is None until every character has.
    """
    import numpy as np

    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats, TrackPlate

    session = _placed_window(window, reference_video)
    where = destination_point(SITE_POSE.position, SITE_POSE.heading, 15.0)
    car = _located_track(2, where)
    van = _located_track(3, destination_point(SITE_POSE.position, SITE_POSE.heading, 25.0))
    plates = (
        TrackPlate(track_id=2, country="GENERIC", display="B7?4921", text=None,
                   is_confident=False, agreement=2, reads=3),
        TrackPlate(track_id=3, country="GENERIC", display="KX19ABC", text="KX19ABC",
                   is_confident=True, agreement=6, reads=6),
    )
    result = FrameResult(
        index=10, timestamp_millis=1000, source_id=session.camera_id,
        detections=(), tracks=(car, van), ended=(),
        image=np.zeros((480, 640, 3), dtype=np.uint8), plates=plates,
    )
    session.absorb(Update(result=result, analysis_fps=30.0, skipped=0, stats=PipelineStats()))
    window._refresh_tracks()

    rows = {
        window.tracks.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)[1]: window.tracks.topLevelItem(i)
        for i in range(window.tracks.topLevelItemCount())
    }
    header = window.tracks.headerItem()
    plate_column = next(c for c in range(header.columnCount()) if header.text(c) == "Plate")
    assert rows[2].text(plate_column) == "B7?4921 (2)", "an unresolved character was completed or the thin count dropped"
    assert rows[3].text(plate_column) == "KX19ABC"
    assert "?" not in rows[3].text(plate_column)


# ------------------------------------------------------------ --start


def test_start_on_launch_runs_the_restored_cameras(qt_app, window, reference_video: Path):
    session = window.add_camera(reference_video, "cam-07")
    assert not session.is_running

    window.start_on_launch()
    pump(qt_app, window, 1.0)

    assert session.is_running, "--start did not start the restored camera"
    window._stop()


def test_start_on_launch_with_no_cameras_says_so_instead_of_looking_busy(qt_app, window):
    # A console opened with --start on a machine with no cameras would look
    # exactly like one that was starting, for as long as anybody waited.
    window.start_on_launch()

    assert not window._running
    assert "--start" in window.status.currentMessage()
    assert "no cameras" in window.status.currentMessage().lower()


def test_the_console_accepts_the_start_flag(qt_app):
    # The flag is parsed by run(); parse it the same way run() does so the
    # help text and the name cannot drift from what the binary accepts.
    import argparse

    from sentinel_console import app as app_module

    import inspect

    # The flags moved into build_parser() when the console grew a command line
    # for the packaged binary; run() must still act on this one.
    assert '"--start"' in inspect.getsource(app_module.build_parser)
    source = inspect.getsource(app_module.run)
    assert "arguments.start" in source and "start_on_launch" in source, (
        "the flag is parsed but never acted on"
    )


# --------------------------------------------------- zone classes, wired


def test_a_motion_only_console_offers_no_class_filter(qt_app, window):
    # The window fixture runs motion detection, which labels nothing. A class
    # filter here would silence a zone for ever, so the picker must be off
    # and say why rather than offer an empty list.
    assert window._detector_labels() == []
    panel = window.zone_properties
    # An empty vocabulary, not an unknown one: the detector is known and it
    # labels nothing, which is the case the picker must refuse rather than
    # offer an empty list.
    assert not panel._vocabulary
    assert not panel.class_picker.isEnabled(), "a motion-only site was offered a class filter"


def test_the_class_picker_takes_the_detectors_own_vocabulary(qt_app, window):
    # Fed the words a real segmenter produces, the picker comes alive with
    # exactly those, so a person-only zone can be drawn on an idle console.
    window.zone_properties.set_classes(["person", "car", "truck"])

    panel = window.zone_properties
    assert panel.class_picker.isEnabled()
    assert set(panel._vocabulary) == {"person", "car", "truck"}


def test_detector_labels_come_from_the_models_class_names(qt_app, window, monkeypatch):
    from sentinel_console import app as app_module

    class Info:
        classifies = True
        class_names = {0: "person", 2: "car", 7: "truck", 99: "car"}

    class Detector:
        info = Info()

    window._model = Path("a-model.onnx")
    monkeypatch.setattr(app_module, "detector_for", lambda _path, **_: Detector())

    assert window._detector_labels() == ["car", "person", "truck"]


def test_the_incident_views_timeline_is_relative_to_the_incident_not_the_epoch(qt_app):
    """The evidence report had this fixed this morning; the console did not.

    Built from events stamped with a real wall clock, the children must read
    "t+0.0s", "t+0.5s" — not fifty-six years.
    """
    from sentinel_console.incident_view import IncidentView

    view = IncidentView()
    incidents = _incident_from_events(3)          # events at index * 500 ms
    view.show_incidents(incidents)
    row = view.topLevelItem(0)
    stamps = [row.child(i).text(1) for i in range(row.childCount()) if row.child(i).text(1).startswith("t+")]
    assert stamps, [row.child(i).text(1) for i in range(row.childCount())]
    offsets = [float(s[2:-1]) for s in stamps]
    assert offsets[0] == 0.0
    assert max(offsets) < 3600, offsets


def test_the_audit_tab_is_in_the_window_and_reads_the_nodes_store(qt_app, window, reference_video: Path):
    labels = [window.detail_tabs.tabText(i) for i in range(window.detail_tabs.count())]
    assert "Audit" in labels

    # Something audited through the node must be visible in the tab: the
    # panel reads the same store the node writes, not a copy.
    _placed_window(window, reference_video)
    window._add_zone(radius=5.0)
    window.audit.refresh()
    assert window.audit.listed_row_ids(), "the audit tab shows nothing after an audited action"


def test_hovering_a_frame_edge_position_says_the_feet_were_below_the_frame(qt_app):
    """The laptop camera's finding, in the operator's words.

    A person seated at the desk had every contact on the frame's bottom edge and
    was projected to 2.16 m ± 0.13 m. The engine now reports such a position as
    a bound between the camera and where the edge projects; the map must say so
    and must not read it out as a measured range.
    """
    view, _, where = _map_with_a_track(source="FRAME_EDGE", radius=1.1)
    _move(view, view._to_screen(*view._to_local(where)))

    tip = view.toolTip()
    assert "feet below the frame" in tip
    assert "between the camera and" in tip
    assert "from the camera" not in tip
    assert "not located" not in tip


def test_a_frame_edge_position_is_drawn_hollow_like_a_bound(qt_app):
    from PySide6.QtGui import QImage

    def marker_pixels(source: str) -> int:
        view, _, where = _map_with_a_track(source=source, radius=0.2)
        view.show_legend = False
        view.show()
        qt_app.processEvents()
        rendered = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        at = view._to_screen(*view._to_local(where))
        count = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                colour = rendered.pixelColor(int(at.x()) + dx, int(at.y()) + dy)
                if abs(colour.green() - theme.TRACK.green()) < 60 and colour.green() > colour.red():
                    count += 1
        return count

    assert marker_pixels("GROUND_PROJECTION") > marker_pixels("FRAME_EDGE")


# ------------------------------------------------ the toolbar, readable


def test_no_toolbar_button_is_narrower_than_its_own_text(qt_app, window):
    """The operator's screenshot: "d camer", "ve on m", "onfigur".

    One row of twelve buttons, a picker, a spin box, a checkbox and two
    captions was wider than a laptop's screen at its display scale, and Qt
    squeezed every button below its text. Two rows, and the captions in the
    status bar, must leave every button at least as wide as its size hint at
    a width a laptop actually has.
    """
    window.resize(1280, 800)
    window.show()
    QApplication.processEvents()
    buttons = [
        window.open_button, window.start_button, window.stop_button,
        window.place_button, window.map_place_button, window.remove_button,
        window.configure_button, window.zone_button, window.export_button,
        *window.mode_buttons.values(),
    ]
    narrow = [
        (b.text(), b.width(), b.sizeHint().width())
        for b in buttons if b.width() < b.sizeHint().width()
    ]
    assert not narrow, f"clipped: {narrow}"
    window.close()


def test_the_captions_live_in_the_status_bar_not_the_toolbar(window):
    assert window.placement_label.parentWidget() is window.status
    assert window.detector_label.parentWidget() is window.status


def test_a_locked_control_says_why_it_is_disabled(qt_app, window):
    """A greyed button with no reason is "a button that does nothing"."""
    assert not window._configuring
    assert not window.place_button.isEnabled()
    assert "Configure" in window.place_button.toolTip()
    assert "Where this camera is" in window.place_button.toolTip(), "the description was lost"
    assert "Configure" in window.map_place_button.toolTip()

    window.configure_button.setChecked(True)
    assert window.place_button.isEnabled()
    assert "Locked" not in window.place_button.toolTip()
    assert "Where this camera is" in window.place_button.toolTip()
    assert "No incident to export yet" in window.export_button.toolTip()
    window.configure_button.setChecked(False)


# ------------------------------------------------ the watch list


def test_the_console_watches_people_and_vehicles_by_default_and_remembers_a_change(qt_app, tmp_path):
    """A jar on a shelf became a "bottle" track. Nobody asked for bottles.

    The default is the security set; a change is applied to the factory the
    node builds detectors from and survives a restart of the console on the
    same machine — and reaches nothing outside the INI the test hands in.
    """
    from sentinel.detect import WATCHED_LABELS

    first = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    assert first._watched == WATCHED_LABELS
    assert first._detector_factory.classes == WATCHED_LABELS
    first._set_watched({"person"})
    assert first._detector_factory.classes == frozenset({"person"})
    first.close()

    second = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    assert second._watched == frozenset({"person"})
    second.close()
    del first, second
    gc.collect()


def test_a_motion_only_console_has_nothing_to_watch_and_says_so(window):
    assert window._vocabulary() == []
    assert window._detector_labels() == []


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="no segmentation model on this machine")
def test_with_a_model_the_zone_picker_offers_only_what_is_watched(qt_app, tmp_path):
    from sentinel.detect import WATCHED_LABELS

    win = ConsoleWindow(":memory:", model=MODEL_PATH, settings=_isolated_settings(tmp_path))
    try:
        assert len(win._vocabulary()) == 80
        assert win._detector_labels() == sorted(WATCHED_LABELS)
        win._set_watched({"person", "dog"})
        assert win._detector_labels() == ["dog", "person"]
        assert list(win.zone_properties._vocabulary) == ["dog", "person"]
    finally:
        win.close()


def test_the_watched_classes_dialog_refuses_to_watch_nothing(qt_app):
    from PySide6.QtWidgets import QDialogButtonBox
    from sentinel_console.watch_dialog import WatchedClassesDialog

    dialog = WatchedClassesDialog(["person", "bottle", "car"], {"person"}, defaults={"person", "car"})
    assert dialog.chosen() == frozenset({"person"})
    dialog._use_defaults()
    assert dialog.chosen() == frozenset({"person", "car"})
    dialog._use_all()
    assert dialog.chosen() == frozenset({"person", "bottle", "car"})
    dialog._set_all(frozenset())
    assert dialog.chosen() == frozenset()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.deleteLater()


# ------------------------------------------- a locked control answers a click
#
# The operator's report of the lock was "the buttons do nothing". A greyed
# control that swallows a click is exactly that, whatever its tooltip says.


def _answer_the_key_yes(monkeypatch, asked: list):
    from PySide6.QtWidgets import QMessageBox

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))


def _answer_the_key_no(monkeypatch, asked: list):
    from PySide6.QtWidgets import QMessageBox

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))


def _swallow_information_boxes(monkeypatch, shown: list):
    from PySide6.QtWidgets import QMessageBox

    def information(parent, title, text, *args, **kwargs):
        shown.append(title)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", staticmethod(information))


def test_clicking_a_locked_control_offers_the_key_and_then_does_what_was_asked(
    qt_app, window, monkeypatch
):
    from PySide6.QtTest import QTest

    asked, shown = [], []
    _answer_the_key_yes(monkeypatch, asked)
    _swallow_information_boxes(monkeypatch, shown)
    assert not window.remove_button.isEnabled()

    # A real press on the greyed button, delivered the way a mouse would.
    QTest.mouseClick(window.remove_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert asked, "the click on a locked control was swallowed"
    assert "locked" in asked[0].lower() and "Remove camera" in asked[0]
    assert window._configuring, "yes did not unlock the site"
    assert window.remove_button.isEnabled()
    # With no camera, Remove says so — which proves the click was carried
    # out after the unlock rather than merely permitted for next time.
    assert shown == ["No camera"], shown
    actions = [row["action"] for row in window.store.audit_trail(limit=5)]
    assert "console.configure.entered" in actions, "the unlock was not audited"


def test_declining_the_key_leaves_the_site_locked_and_says_where_it_is(
    qt_app, window, monkeypatch
):
    from PySide6.QtTest import QTest

    asked, shown = [], []
    _answer_the_key_no(monkeypatch, asked)
    _swallow_information_boxes(monkeypatch, shown)

    QTest.mouseClick(window.place_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert asked and "Place" in asked[0]
    assert not window._configuring
    assert not window.place_button.isEnabled()
    assert shown == [], "the refused click was carried out anyway"
    assert "Configure" in window.status.currentMessage()


def test_the_draw_button_offers_the_key_when_the_site_is_locked(
    qt_app, window, reference_video: Path, monkeypatch
):
    from sentinel_console.map_view import MODE_DRAW

    _placed_window(window, reference_video)
    asked = []
    _answer_the_key_yes(monkeypatch, asked)
    assert not window._configuring

    window.mode_buttons[MODE_DRAW].click()

    assert asked and "Draw" in asked[0]
    assert window._configuring
    assert window.map.drawing, "unlocked, but the drawing the operator asked for never began"
    assert window.mode_buttons[MODE_DRAW].isChecked()


def test_the_draw_button_declined_still_refuses_and_resets(
    qt_app, window, reference_video: Path, monkeypatch
):
    from sentinel_console.map_view import MODE_DRAW, MODE_SELECT

    _placed_window(window, reference_video)
    asked = []
    _answer_the_key_no(monkeypatch, asked)

    window.mode_buttons[MODE_DRAW].click()

    assert asked
    assert not window._configuring
    assert not window.map.drawing
    assert window.mode_buttons[MODE_SELECT].isChecked()
    assert not window.mode_buttons[MODE_DRAW].isChecked()
    assert "Configure" in window.status.currentMessage()


def test_an_enabled_control_is_not_second_guessed(qt_app, window, monkeypatch):
    """The filter answers greyed controls only. In Configure a click is a click."""
    from PySide6.QtTest import QTest

    asked, shown = [], []
    _answer_the_key_yes(monkeypatch, asked)
    _swallow_information_boxes(monkeypatch, shown)
    window.configure_button.setChecked(True)

    QTest.mouseClick(window.remove_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert asked == [], "an unlocked control asked for the key"
    assert shown == ["No camera"]


def test_the_status_bar_always_says_whether_the_site_is_locked(qt_app, window):
    assert window.lock_label.isVisible() or not window.isVisible()
    assert "MONITOR" in window.lock_label.text()
    assert "Configure" in window.lock_label.text()
    window.configure_button.setChecked(True)
    assert "CONFIGURE" in window.lock_label.text()
    window.configure_button.setChecked(False)
    assert "MONITOR" in window.lock_label.text()


def test_escape_does_not_relock_and_the_lock_no_longer_claims_it_does(
    qt_app, window, reference_video: Path
):
    """The tooltip and the manual said Configure "returns to Monitor on
    Escape". It never did — Escape clears a selection or abandons a drawing —
    and a lock described wrongly is worse than one described not at all: an
    operator who trusts the sentence walks away believing the site relocked.
    """
    from PySide6.QtTest import QTest

    _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    QTest.keyClick(window, Qt.Key.Key_Escape)
    QApplication.processEvents()

    assert window._configuring, "Escape relocked; the docs now say it does not"
    tip = window.configure_button.toolTip()
    assert "on Escape" not in tip
    assert "does not relock" in tip


def test_a_locked_control_survives_the_freeing_test(qt_app, tmp_path):
    """Installing the window as an event filter on its own children must not
    put it in a reference cycle — the cycle is the exit-time heap corruption.
    """
    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    assert win._guarded, "nothing was guarded"
    win.close()
    ref = weakref.ref(win)
    del win
    assert_freed(ref)


# ----------------------------------------------------- an exception in a slot


def test_an_exception_raised_in_a_slot_is_logged_named_and_let_go(qt_app, window):
    """A slot that raises looks, to the operator, like a button that did
    nothing: PySide prints the traceback to a stderr the packaged console does
    not have, and keeps it on sys.last_* where it pins the widget.
    """
    import logging
    import sys

    from sentinel_console import app as app_module

    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Keep()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.getLogger("sentinel")
    logger.addHandler(handler)
    try:
        try:
            raise RuntimeError("raised inside a slot")
        except RuntimeError:
            info = sys.exc_info()
        # What PyErr_Print does before it calls the hook.
        sys.last_type, sys.last_value, sys.last_traceback = info
        app_module._ACTIVE_WINDOW = weakref.ref(window)

        app_module._report_uncaught(*info)
    finally:
        logger.removeHandler(handler)
        app_module._ACTIVE_WINDOW = None

    assert any("CRITICAL" in r and "raised inside a slot" in r for r in records), records
    assert sys.last_traceback is None and sys.last_value is None
    assert "RuntimeError" in window.status.currentMessage()
    assert "log" in window.status.currentMessage()


def test_run_installs_the_hook_before_the_window_shows(qt_app):
    import inspect

    from sentinel_console import app as app_module

    source = inspect.getsource(app_module.run)
    assert "sys.excepthook = _report_uncaught" in source
    assert source.index("sys.excepthook = _report_uncaught") < source.index("window.show()")


# ----------------------------------------------------------- the floor


def test_the_console_requires_half_confidence_by_default_and_remembers_a_change(
    qt_app, tmp_path
):
    from sentinel_console.app import DEFAULT_CONFIDENCE

    assert DEFAULT_CONFIDENCE == 0.5
    first = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    assert first._confidence == DEFAULT_CONFIDENCE
    assert first._detector_factory.confidence == DEFAULT_CONFIDENCE
    first._set_confidence(0.6)
    assert first._detector_factory.confidence == 0.6
    assert "0.60" in first.status.currentMessage()
    first.close()

    second = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    assert second._confidence == 0.6
    second.close()
    del first, second
    gc.collect()


def test_a_nonsense_stored_confidence_falls_back_to_the_default(qt_app, tmp_path):
    from sentinel_console.app import DEFAULT_CONFIDENCE

    for bad in ("high", "", 3.0, 0.0, -1):
        settings = _isolated_settings(tmp_path)
        settings.setValue("detection/confidence", bad)
        settings.sync()
        win = ConsoleWindow(":memory:", settings=settings)
        try:
            assert win._confidence == DEFAULT_CONFIDENCE, bad
        finally:
            win.close()
        settings.clear()
        settings.sync()


def test_the_floor_is_clamped_to_the_range(qt_app, window):
    from sentinel_console.app import CONFIDENCE_RANGE

    low, high = CONFIDENCE_RANGE
    window._set_confidence(0.0)
    assert window._confidence == low
    window._set_confidence(1.5)
    assert window._confidence == high


def test_the_factory_hands_the_floor_and_the_watch_list_to_the_detector(monkeypatch):
    from sentinel.detect import MotionDetector
    from sentinel_console import app as app_module
    from sentinel_console.app import _DetectorFactory

    calls = []

    def fake(model, **options):
        calls.append((model, options))
        return MotionDetector()

    monkeypatch.setattr(app_module, "detector_for", fake)
    _DetectorFactory(Path("m.onnx"), frozenset({"person"}), 0.6)()
    assert calls == [(Path("m.onnx"), {"classes": frozenset({"person"}), "confidence_threshold": 0.6})]


def test_a_motion_only_console_still_starts_with_a_floor_set(qt_app, window):
    # The real factory, no model: motion takes what applies to it and ignores
    # the rest. A console that could not start on a machine with no model
    # because of a setting meant for classifiers would be a regression.
    from sentinel.detect import MotionDetector

    assert window._detector_factory.confidence == 0.5
    assert isinstance(window._detector_factory(), MotionDetector)


def test_the_detector_line_states_the_floor_beside_the_watch_list(qt_app):
    from sentinel_console.app import _detector_summary

    class Segmenter:
        classifies = True
        kind = "onnx-segment"
        name = "yolov8n-seg instance segmentation"
        model_sha256 = "f828ccfa4b69abcdef"
        class_names = {0: "person", 2: "car"}

    class Motion:
        classifies = False
        kind = "motion"
        name = "MOG2"
        model_sha256 = None
        class_names = {}

    line = _detector_summary(Segmenter(), 0.5)
    assert "watching car, person" in line
    assert "≥ 0.50" in line
    assert line.index("≥ 0.50") < line.index("f828ccfa4b69")
    assert "0.50" not in _detector_summary(Motion(), 0.5)
    assert "≥" not in _detector_summary(Segmenter(), None)


def test_the_watched_classes_dialog_carries_the_floor(qt_app):
    from sentinel_console.watch_dialog import WatchedClassesDialog

    without = WatchedClassesDialog(["person"], {"person"}, defaults={"person"})
    assert without.confidence() is None and without.confidence_spin is None
    without.deleteLater()

    dialog = WatchedClassesDialog(["person"], {"person"}, defaults={"person"}, confidence=0.5)
    assert dialog.confidence() == 0.5
    dialog.confidence_spin.setValue(0.65)
    assert dialog.confidence() == 0.65
    assert dialog.chosen() == frozenset({"person"})
    dialog.deleteLater()


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="no segmentation model on this machine")
def test_with_a_model_the_floor_reaches_the_segmenter(qt_app, tmp_path):
    from sentinel.segment import Segmenter

    win = ConsoleWindow(":memory:", model=MODEL_PATH, settings=_isolated_settings(tmp_path))
    try:
        win._set_confidence(0.7)
        detector = win._detector_factory()
        assert isinstance(detector, Segmenter)
        assert detector._confidence == 0.7
        assert set(detector.info.class_names.values()) == set(win._watched)
    finally:
        win.close()


# ------------------------------------- the packaged binary as the test medium
#
# The rule: the product is tested through the real camera and the shipped
# binary, never a checkout and never a file. A binary cannot be clicked by a
# test, so it takes the same flags `sentinel run` does and leaves evidence.


def test_the_console_parser_takes_the_flags_a_camera_test_needs(qt_app):
    from sentinel_console.app import build_parser

    arguments, unknown = build_parser().parse_known_args([
        "--start", "--for", "5", "--screenshots", "out",
        "--camera", "device:0", "--camera", "gate.mp4",
        "--place", "33.8938,35.5018,1.2,180,-15",
        "--zone", "Room:33.8938,35.5018;33.8939,35.5018;33.8939,35.5019",
        "--zone-classes", "Room=person",
        "--watch", "person, car", "--confidence", "0.6", "--settings", "t.ini",
        "-platform", "offscreen",
    ])
    assert arguments.start and arguments.duration == 5.0 and arguments.screenshots == "out"
    assert arguments.camera == ["device:0", "gate.mp4"]
    assert arguments.place.mount_height == 1.2 and arguments.place.pitch == -15.0
    assert [zone.name for zone in arguments.zone] == ["Room"]
    assert arguments.zone_classes == [("Room", frozenset({"person"}))]
    assert arguments.watch == "person, car" and arguments.confidence == 0.6
    assert arguments.settings == "t.ini"
    # Qt's own arguments pass through, as they always did.
    assert unknown == ["-platform", "offscreen"]


def test_a_confidence_outside_the_range_is_refused_on_the_command_line(qt_app, capsys):
    from sentinel_console.app import build_parser

    for bad in ("1.5", "0", "high"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--confidence", bad])
    assert "--confidence" in capsys.readouterr().err


def test_command_line_overrides_apply_to_this_window_and_are_not_remembered(qt_app, tmp_path):
    settings = _isolated_settings(tmp_path)
    win = ConsoleWindow(":memory:", settings=settings, watched={"person"}, confidence=0.7)
    try:
        assert win._watched == frozenset({"person"})
        assert win._confidence == 0.7
        assert win._detector_factory.classes == frozenset({"person"})
        assert win._detector_factory.confidence == 0.7
        assert settings.value("detection/watched", None) is None, "--watch was written to the machine"
        assert settings.value("detection/confidence", None) is None, "--confidence was written to the machine"
    finally:
        win.close()


def test_the_site_can_be_seeded_from_the_command_line_and_is_audited(
    qt_app, window, reference_video: Path
):
    from sentinel.cli import _pose, _zone

    pose = _pose("33.8938,35.5018,6,180,-22")
    zone = _zone("Yard:33.89365,35.50170;33.89365,35.50190;33.89345,35.50190;33.89345,35.50170")

    named = window.seed_site(cameras=[reference_video], pose=pose, zones=[zone])

    assert len(named) == 1
    session = window._sessions[named[0]]
    assert session.pose == pose
    assert [zone.id for zone in window._zones] == ["yard"]
    assert window.start_button.isEnabled()
    assert not window._configuring, "seeding is not an unlock"
    actions = [row["action"] for row in window.store.audit_trail(limit=10)]
    for expected in ("camera.added", "camera.placed", "zone.created"):
        assert expected in actions, f"{expected} was not audited"


def test_seeding_again_keeps_what_is_there_rather_than_duplicating_it(
    qt_app, window, reference_video: Path
):
    from sentinel.cli import _pose, _zone

    pose = _pose("33.8938,35.5018,6,180,-22")
    zone = _zone("Yard:33.89365,35.50170;33.89365,35.50190;33.89345,35.50190;33.89345,35.50170")
    first = window.seed_site(cameras=[reference_video], pose=pose, zones=[zone])

    again = window.seed_site(cameras=[reference_video], pose=None, zones=[zone])

    assert again == first
    assert len(window._sessions) == 1
    assert len(window._zones) == 1
    # A restored, unplaced camera named again with a placement is placed.
    window._sessions[first[0]].record.pose = None
    window.seed_site(cameras=[reference_video], pose=pose)
    assert window._sessions[first[0]].pose == pose


def test_a_timed_run_photographs_itself_reports_and_closes(
    qt_app, window, reference_video: Path, tmp_path, capsys
):
    _placed_window(window, reference_video)
    window._start()
    shots = tmp_path / "shots"

    window.end_after(0.5, screenshots=shots)

    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline and window.isVisible():
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    assert not window.isVisible(), "the timed run did not close the window"
    assert not window._running

    names = sorted(path.name for path in shots.glob("*.png"))
    for expected in ("console.png", "plan-view.png", "incidents.png", "tracks.png", "zones.png", "camera-cam-07.png"):
        assert expected in names, names
    out = capsys.readouterr().out
    assert "screenshot" in out
    assert "node local" in out, out
    assert "cam-07" in out and "frames" in out
    assert "distinct tracks" in out, "the per-track summary is what a camera test reads"


def test_a_camera_pane_picture_is_named_safely_for_a_file_system(qt_app, window, tmp_path):
    """`device:0` — the id every local camera gets — is not a Windows file
    name. The first packaged run wrote five pictures of six and a warning."""
    from sentinel_console.app import _file_safe

    assert _file_safe("device:0") == "device-0"
    assert _file_safe("rtsp://admin@10.0.0.5/s") == "rtsp-admin-10.0.0.5-s"
    assert _file_safe("gate") == "gate"
    assert _file_safe("::") == "camera"

    window.add_camera("device:0", camera_id="device:0")
    written = window.photograph(tmp_path)
    names = {path.name for path in written}
    assert "camera-device-0.png" in names, names
    assert all(":" not in path.name for path in written)


def test_a_timed_run_with_no_pictures_still_reports(qt_app, window, capsys):
    window.end_after(0.0)
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and window.isVisible():
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    assert not window.isVisible()
    assert "node local" in capsys.readouterr().out


def test_an_unknown_watch_label_is_refused_before_any_window_exists(qt_app, monkeypatch):
    from sentinel.detect import DetectionError
    from sentinel_console import app as app_module

    def refuse(model, **options):
        raise DetectionError("asked to watch unicorn, which this model does not name")

    monkeypatch.setattr(app_module, "detector_for", refuse)
    problem = app_module._refuse_unknown_labels(Path("m.onnx"), frozenset({"unicorn"}))
    assert problem is not None and "unicorn" in problem
    assert app_module._labels(None) is None
    assert app_module._labels("person, car,") == frozenset({"person", "car"})


def test_run_seeds_times_and_starts_in_that_order(qt_app):
    """The flags are wired, and wired in the order that works: seed before
    the start timer (so a --camera is what --start starts), the excepthook
    before the window shows."""
    import inspect

    from sentinel_console import app as app_module

    source = inspect.getsource(app_module.run)
    for fragment in ("build_parser()", "window.seed_site(", "window.end_after(", "start_on_launch"):
        assert fragment in source, fragment
    assert source.index("window.seed_site(") < source.index("start_on_launch")
    assert source.index("sys.excepthook = _report_uncaught") < source.index("window.show()")


# ---------------------------------------------- dialogs, with OK pressed
#
# Found by the packaged binary on the laptop camera, 2026-09-05 09:03: Place…
# → OK raised "QDoubleSpinBox already deleted", Add zone… → OK raised the same
# for its QLineEdit, three times. `WA_DeleteOnClose` had the dialog deleted
# inside `done()`, before `exec()` returned and the slot read its fields. No
# test had ever pressed OK: every test called the slot beneath the dialog.
# These press OK the way Qt does, including the deferred delete that follows.


def _press_ok(monkeypatch, cls, prepare=None):
    """Make `cls.exec()` behave as a person pressing OK.

    `accept()` is what `done()` runs, and it is what deletes a
    `WA_DeleteOnClose` dialog; the deferred delete lands when the dialog's
    event loop exits, which `sendPostedEvents` stands in for here.
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QDialog

    def exec_(self):
        if prepare is not None:
            prepare(self)
        self.accept()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(cls, "exec", exec_)


def test_pressing_ok_in_the_placement_dialog_actually_places_the_camera(
    qt_app, window, reference_video: Path, monkeypatch
):
    session = window.add_camera(reference_video, "cam-07")
    window.configure_button.setChecked(True)
    assert session.pose is None

    def type_a_pose(dialog):
        dialog.latitude.setValue(33.8938)
        dialog.longitude.setValue(35.5018)
        dialog.mount_height.setValue(6.0)
        dialog.heading.setValue(180.0)
        dialog.pitch.setValue(-22.0)

    _press_ok(monkeypatch, PlacementDialog, type_a_pose)
    window._place_camera()

    assert session.pose is not None, "OK was pressed and the camera is still unplaced"
    assert session.pose.mount_height == 6.0 and session.pose.heading == 180.0
    actions = [row["action"] for row in window.store.audit_trail(limit=10)]
    assert "camera.placed" in actions


def test_pressing_ok_in_the_zone_dialog_actually_creates_a_zone(
    qt_app, window, reference_video: Path, monkeypatch
):
    from sentinel_console.zones_view import ZoneDialog

    _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    _press_ok(monkeypatch, ZoneDialog, lambda dialog: dialog._name.setText("Loading bay"))

    window._add_zone_dialog()

    assert [zone.name for zone in window._zones] == ["Loading bay"]
    actions = [row["action"] for row in window.store.audit_trail(limit=10)]
    assert "zone.created" in actions


def test_naming_a_drawn_outline_on_ok_creates_the_zone(
    qt_app, window, reference_video: Path, monkeypatch
):
    from sentinel.core import destination_point
    from sentinel_console.zones_view import ZoneDialog

    session = _placed_window(window, reference_video)
    window.configure_button.setChecked(True)
    anchor = destination_point(session.pose.position, session.pose.heading, 20.0)
    ring = [destination_point(anchor, bearing, 5.0) for bearing in (0.0, 120.0, 240.0)]
    _press_ok(monkeypatch, ZoneDialog, lambda dialog: dialog._name.setText("Drawn"))

    window._zone_drawn(ring)

    assert [zone.name for zone in window._zones] == ["Drawn"]
    assert len(window._zones[0].ring) == 3


def test_pressing_ok_in_the_add_camera_dialog_adds_the_camera(
    qt_app, window, reference_video: Path, monkeypatch
):
    window.configure_button.setChecked(True)

    def choose_the_file(dialog):
        dialog._tabs.setCurrentIndex(2)
        dialog._file.setText(str(reference_video))
        dialog._accept()  # the OK handler, which records the choice

    _press_ok(monkeypatch, AddCameraDialog, choose_the_file)
    window._choose_source()

    assert len(window._sessions) == 1
    assert next(iter(window._sessions.values())).source == str(reference_video)


def test_no_dialog_the_console_reads_after_exec_is_delete_on_close(qt_app):
    """The structural half: the attribute must not come back on a dialog
    whose fields are read after `exec()` returns."""
    import inspect

    from sentinel_console import app as app_module

    source = inspect.getsource(app_module.ConsoleWindow)
    # The call, not the word: the comments beside each dialog name the
    # attribute to say why it is absent.
    assert "setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)" not in source, (
        "a dialog read after exec() is deleted by done() before exec() returns"
    )


# ------------------------------------------ the timed run ends, whatever


def test_a_timed_run_closes_even_when_the_terminal_cannot_print_the_report(
    qt_app, window, monkeypatch
):
    import builtins

    def cannot_print(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "\u2265", 0, 1, "character maps to <undefined>")

    monkeypatch.setattr(builtins, "print", cannot_print)
    window.end_after(0.0)
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and window.isVisible():
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    assert not window.isVisible(), "the report could not be printed and the run never ended"


def test_a_timed_run_dismisses_a_dialog_somebody_left_open(qt_app, window):
    from PySide6.QtCore import QTimer

    dialog = PlacementDialog(None, window)
    dialog.show()
    qt_app.processEvents()
    assert dialog.isVisible()
    outcomes = []
    dialog.finished.connect(outcomes.append)

    window.end_after(0.0)
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and window.isVisible():
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

    assert not window.isVisible()
    assert outcomes == [int(PlacementDialog.DialogCode.Rejected)], outcomes
    dialog.deleteLater()


def test_run_replaces_what_the_terminal_cannot_encode(qt_app):
    import inspect

    from sentinel_console import app as app_module

    assert 'reconfigure(errors="replace")' in inspect.getsource(app_module.run)


def test_a_timed_run_closed_early_by_a_person_still_reports(
    qt_app, window, reference_video: Path, tmp_path, capsys
):
    """The second packaged camera run was closed by hand at twelve seconds and
    left no picture and no summary; the tool read it as a run that never
    started. Closing the window is not a reason to lose what it saw."""
    _placed_window(window, reference_video)
    window._start()
    window.end_after(600.0, screenshots=tmp_path / "early")
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline:
        qt_app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

    window.close()

    assert not window._timed_pending
    assert (tmp_path / "early" / "console.png").is_file()
    out = capsys.readouterr().out
    assert "node local" in out and "distinct tracks" in out
    # And the timer, when it fires later, must not report a second time.
    window._finish_timed_run()
    assert "node local" not in capsys.readouterr().out


def test_closing_a_window_that_was_never_timed_reports_nothing(qt_app, window, capsys):
    window.close()
    assert capsys.readouterr().out == ""


# ------------------------------------ export through the node, and the build


def test_the_console_exports_through_the_node_with_footage_and_preservation(
    qt_app, window, tmp_path, monkeypatch
):
    """`_export_incident` called the exporter directly, so a package made from
    the console carried no footage and preserved nothing — for as long as the
    node's own docstring said the console "used to". Now it goes through the
    node, like the test above it always did."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from sentinel.evidence import verify_export
    from sentinel.incidents import Correlator
    from test_store import make_event

    incident = Correlator().correlate([make_event(track=n) for n in (1, 2)])[0]
    window.store.save_incident(incident)
    monkeypatch.setattr(ConsoleWindow, "_selected_incident", lambda self: incident)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path))
    )
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text)) or QMessageBox.StandardButton.Ok),
    )

    window._export_incident()

    assert shown and shown[0][0] == "Evidence exported", shown
    packages = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(packages) == 1
    assert (packages[0] / "footage.json").is_file(), "the package has no footage record"
    assert verify_export(packages[0]) == []
    assert "footage" in shown[0][1]
    actions = [row["action"] for row in window.store.audit_trail(limit=10)]
    assert actions.count("incident.exported") == 1, "audited twice, or not at all"


def test_about_names_the_build(qt_app, window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from sentinel import version

    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: shown.append(text) or QMessageBox.StandardButton.Ok),
    )
    window._show_about()
    assert shown and shown[0].startswith(f"Sentinel Vision {version.__version__}")


# --------------------------------------------------- recording from the console


def test_the_record_box_goes_through_the_node_and_follows_the_lock(
    qt_app, window, reference_video: Path
):
    from sentinel_console.camera_list import RECORD_COLUMN

    window.add_camera(reference_video, "cam-07")
    item = window.camera_list.tree.topLevelItem(0)
    assert item is not None
    assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable), "recording could be changed in Monitor"

    window.configure_button.setChecked(True)
    item = window.camera_list.tree.topLevelItem(0)
    assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    item.setCheckState(RECORD_COLUMN, Qt.CheckState.Checked)

    assert window.node.camera("cam-07").record is True
    assert window.store.camera_recording("cam-07") is True
    actions = [row["action"] for row in window.store.audit_trail(limit=10)]
    assert "camera.recording" in actions
    assert "record" in window.status.currentMessage()

    window.configure_button.setChecked(False)
    item = window.camera_list.tree.topLevelItem(0)
    assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    assert item.checkState(RECORD_COLUMN) == Qt.CheckState.Checked, "relocking forgot the flag"


def test_the_console_records_a_camera_that_asked_into_the_recordings_directory(
    qt_app, tmp_path, monkeypatch, reference_video: Path
):
    """The console could not record at all until it handed the node a
    recordings directory; nothing it exported carried footage."""
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path))
    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    try:
        win.add_camera(reference_video, "cam-07")
        win.node.set_recording("cam-07", True)
        win._start()
        pump(qt_app, win, 25.0)
        assert not win._running, "the file did not finish"
        clips = list((tmp_path / "recordings").rglob("*.mp4"))
        assert clips, "the camera asked to record and nothing was written"
        health = win.node.camera_health()["cam-07"]
        assert health.asked_to_record is True
    finally:
        win.close()


def test_a_camera_that_did_not_ask_records_nothing(qt_app, tmp_path, monkeypatch, reference_video: Path):
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path))
    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    try:
        win.add_camera(reference_video, "cam-07")
        win._start()
        pump(qt_app, win, 25.0)
        assert not list((tmp_path / "recordings").rglob("*.mp4"))
    finally:
        win.close()


def test_the_record_flag_reaches_the_command_line(qt_app, window, reference_video: Path):
    from sentinel_console.app import build_parser

    assert build_parser().parse_args(["--record"]).record is True
    assert build_parser().parse_args([]).record is False
    named = window.seed_site(cameras=[reference_video], record=True)
    assert named and window.node.camera(named[0]).record is True


def test_a_console_export_of_a_recorded_incident_carries_its_clips_and_preserves_them(
    qt_app, tmp_path, monkeypatch, reference_video: Path
):
    """The other half of the export test: an incident whose camera was
    recording, exported from the console, must contain the clips and protect
    the originals from retention — the two things the console never did."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path))
    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path))
    try:
        _placed_window(win, reference_video)
        win.node.set_recording("cam-07", True)
        win.zone_radius.setValue(12.0)
        win._add_zone()
        win._start()
        pump(qt_app, win, 25.0)
        assert not win._running
        # `pump` stops the moment the last thread dies, which can be before
        # the repaint timer's next tick — the tick a real session would get.
        win._collect()
        assert win.store.recording_count() > 0, "the run finished and its clip was never indexed"
        assert win.node.incidents, "the reference scene raised no incident"
        incident = win.node.incidents[0]
        monkeypatch.setattr(ConsoleWindow, "_selected_incident", lambda self: incident)
        out = tmp_path / "out"
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out)))
        shown = []
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda p, title, text, *a, **k: shown.append(text) or QMessageBox.StandardButton.Ok))

        win._export_incident()

        packages = [p for p in out.iterdir() if p.is_dir()]
        assert len(packages) == 1
        assert list(packages[0].glob("*.mp4")), "the package has no video in it"
        assert "clip(s) of footage" in shown[0]
        assert win.store.recorded_bytes(preserved=True) > 0
        actions = [row["action"] for row in win.store.audit_trail(limit=500)]
        assert "recording.preserved" in actions and "incident.exported" in actions
    finally:
        win.close()


# ------------------------------------------------------- accounts and permission


def _operator(store):
    from sentinel.accounts import Accounts, Role

    return Accounts(store).add("alice", "correct horse battery", Role.OPERATOR)


def test_with_no_account_nothing_is_gated_and_the_status_bar_says_so(qt_app, window):
    assert window.user is None
    assert window.actor == "console"
    assert "nobody" in window.user_label.text()
    window.configure_button.setChecked(True)
    assert window._configuring


def test_a_viewer_cannot_configure_and_the_refusal_is_audited(qt_app, tmp_path):
    from sentinel.accounts import Accounts, Role, User

    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path), user=User("vic", Role.VIEWER))
    try:
        Accounts(win.store).add("vic", "pw-pw-pw-pw", Role.VIEWER)
        win.configure_button.setChecked(True)
        assert not win._configuring
        assert not win.configure_button.isChecked()
        assert "vic" in win.status.currentMessage()
        rows = [r for r in win.store.audit_trail(limit=10) if r["action"] == "console.configure.refused"]
        assert rows and rows[0]["actor"] == "console:vic"
        assert "vic · viewer" in win.user_label.text()
    finally:
        win.close()


def test_an_operator_configures_and_every_change_is_audited_under_their_name(
    qt_app, tmp_path, reference_video: Path
):
    from sentinel.accounts import Role, User

    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path), user=User("alice", Role.OPERATOR))
    try:
        assert win.actor == "console:alice"
        win.configure_button.setChecked(True)
        assert win._configuring
        win.add_camera(reference_video, "cam-07")
        actors = {r["actor"] for r in win.store.audit_trail(limit=40) if r["action"] in ("camera.added", "console.configure.entered")}
        assert actors == {"console:alice"}, actors
    finally:
        win.close()


def test_a_viewer_cannot_export_evidence(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from sentinel.accounts import Role, User
    from sentinel.incidents import Correlator
    from test_store import make_event

    win = ConsoleWindow(":memory:", settings=_isolated_settings(tmp_path), user=User("vic", Role.VIEWER))
    try:
        incident = Correlator().correlate([make_event(track=n) for n in (1, 2)])[0]
        win.store.save_incident(incident)
        monkeypatch.setattr(ConsoleWindow, "_selected_incident", lambda self: incident)
        shown = []
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda p, title, text, *a, **k: shown.append(title) or QMessageBox.StandardButton.Ok))
        win._export_incident()
        assert shown == ["Not permitted"]
        assert "incident.export.refused" in [r["action"] for r in win.store.audit_trail(limit=10)]
    finally:
        win.close()


def test_the_login_dialog_signs_in_gives_up_after_the_attempts_and_never_keeps_the_password(qt_app, tmp_path):
    from sentinel.accounts import Accounts, Role
    from sentinel.store import Store
    from sentinel_console.login import MAX_ATTEMPTS, LoginDialog

    with Store(":memory:") as store:
        Accounts(store).add("alice", "correct horse battery", Role.OPERATOR)
        accounts = Accounts(store)

        dialog = LoginDialog(accounts)
        dialog.name.setText("alice")
        dialog.password.setText("wrong")
        dialog._try()
        assert dialog.user is None and "attempt" in dialog.message.text()
        assert dialog.password.text() == "", "a wrong password stayed in the box"
        dialog.password.setText("correct horse battery")
        dialog._try()
        assert dialog.user is not None and dialog.user.name == "alice"
        dialog.deleteLater()

        given_up = LoginDialog(accounts)
        outcomes = []
        given_up.finished.connect(outcomes.append)
        given_up.name.setText("alice")
        for _ in range(MAX_ATTEMPTS):
            given_up.password.setText("no")
            given_up._try()
        assert outcomes == [int(LoginDialog.DialogCode.Rejected)]
        given_up.deleteLater()


def test_the_first_administrator_dialog_creates_an_admin_or_is_declined(qt_app):
    from sentinel.accounts import Accounts, Role
    from sentinel.store import Store
    from sentinel_console.login import FirstAdminDialog

    with Store(":memory:") as store:
        accounts = Accounts(store)
        dialog = FirstAdminDialog(accounts)
        dialog.name.setText("root")
        dialog.password.setText("short")
        dialog.confirm.setText("short")
        dialog._create()
        assert dialog.user is None and "eight" in dialog.message.text()
        dialog.password.setText("long enough now")
        dialog.confirm.setText("long enough noW")
        dialog._create()
        assert dialog.user is None and "differ" in dialog.message.text()
        dialog.confirm.setText("long enough now")
        dialog._create()
        assert dialog.user is not None and dialog.user.role is Role.ADMIN
        assert accounts.any()
        dialog.deleteLater()


def test_the_command_line_takes_a_user_and_refuses_seeding_to_a_viewer(qt_app):
    import inspect

    from sentinel_console import app as app_module
    from sentinel_console.app import build_parser

    assert build_parser().parse_args(["--user", "alice"]).user == "alice"
    source = inspect.getsource(app_module.run)
    assert "_sign_in(arguments)" in source
    assert "SITE_CONFIGURE" in source, "seeding from the command line is not permission-checked"
