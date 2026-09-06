"""The console: a thin view over the service, and the v1 defects it must not repeat."""

from __future__ import annotations

import ast
import gc
import inspect
import time
import weakref
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog

from vigil.adapters.detectors import MotionDetector
from vigil.domain.geo import CameraPose, LatLon
from vigil.service.alerts import CAMERA_DARK, Alerts
from vigil.service.auth import Principal, Role
from vigil.service.runtime import Runtime
from vigil.service.site import SiteService
from vigil.storage.store import Store
from vigil.interfaces.console import dialogs
from vigil.interfaces.console.commands import Commands
from vigil.interfaces.console.window import LOCKED_REASON, ConsoleWindow

CONSOLE = Path(__file__).resolve().parent.parent / "vigil" / "interfaces" / "console"
OPERATOR = Principal("alice", Role.OPERATOR, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


@pytest.fixture
def console(qt_app, tmp_path, keychain, request):
    """A window over an in-memory site. Closed and freed after every test."""
    principal = getattr(request, "param", OPERATOR)
    store = Store(":memory:")
    site = SiteService(store, keychain)
    runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True), keep_images=True)
    window = ConsoleWindow(Commands(site, runtime, principal, evidence_dir=tmp_path / "evidence"))
    window.resize(1280, 800)
    window.show()
    qt_app.processEvents()
    yield window
    window.close()
    store.close()


def _press_ok(dialog: QDialog) -> None:
    """Press OK the way Qt does, so the accept path itself is exercised."""
    from PySide6.QtWidgets import QDialogButtonBox

    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()


# ------------------------------------------------------------- structural


def test_no_connection_closes_over_the_window():
    """A lambda over `self` in a connection is a cycle holding a QWidget — v1's shutdown crash."""
    for path in CONSOLE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"):
                continue
            for argument in node.args:
                assert not isinstance(argument, ast.Lambda), f"{path.name}:{node.lineno} connects a lambda"


def test_no_dialog_deletes_itself_on_close_and_every_one_is_opened_through_ask():
    for path in CONSOLE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "setAttribute":
                # The prose above the dialogs names the attribute; only a
                # *call* is the defect, and v1's first version of this test
                # matched its own comment.
                assert "WA_DeleteOnClose" not in ast.unparse(node), f"{path.name}:{node.lineno}"
    window = (CONSOLE / "window.py").read_text(encoding="utf-8")
    tree = ast.parse(window)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exec":
            raise AssertionError(f"window.py:{node.lineno} calls exec() directly instead of dialogs.ask")


def test_the_window_never_reaches_past_commands():
    """Every mutation goes through one object; the window holds no service call of its own."""
    tree = ast.parse((CONSOLE / "window.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("store", "_site", "add_camera", "place_camera", "add_zone"):
            source = ast.unparse(node)
            assert source.startswith("self.commands.") or source.startswith("self._"), source


def test_the_window_is_freed_when_it_is_closed(qt_app, tmp_path, keychain):
    """A window kept alive by a cycle is destroyed after the QApplication, which crashes."""
    store = Store(":memory:")
    site = SiteService(store, keychain)
    runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
    window = ConsoleWindow(Commands(site, runtime, OPERATOR, evidence_dir=tmp_path))
    reference = weakref.ref(window)
    window.show()
    qt_app.processEvents()
    window.close()
    del window
    gc.collect()
    qt_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    store.close()
    assert reference() is None, "the console is in a reference cycle"


# -------------------------------------------------------------- the lock


def test_the_site_is_locked_until_configure_and_the_lock_says_so(console):
    assert not console._configuring
    assert "MONITOR" in console.lock_label.text()
    for control in console._configure_only():
        assert not control.isEnabled() and control.toolTip() == LOCKED_REASON
    console.configure_button.setChecked(True)
    assert console._configuring and "CONFIGURE" in console.lock_label.text()
    assert all(c.isEnabled() for c in console._configure_only())
    actions = [r["action"] for r in console.commands.audit_rows()]
    assert "console.configure.entered" in actions
    console.configure_button.setChecked(False)
    assert "console.configure.left" in [r["action"] for r in console.commands.audit_rows()]


def test_a_greyed_button_still_answers_a_click(console, qt_app):
    button = console.add_button
    assert not button.isEnabled()
    event = QMouseEvent(QEvent.Type.MouseButtonPress, button.rect().center().toPointF(), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    assert console.eventFilter(button, event), "the click was not answered"
    assert console.status.currentMessage() == LOCKED_REASON


@pytest.mark.parametrize("console", [VIEWER], indirect=True)
def test_a_viewer_cannot_unlock_and_the_refusal_names_them(console):
    console.configure_button.setChecked(True)
    assert not console._configuring and not console.configure_button.isChecked()
    assert "vic" in console.status.currentMessage()
    assert "vic · viewer" in console.user_label.text()


def test_with_no_account_nothing_is_gated_and_the_bar_says_nobody_is_named(qt_app, tmp_path, keychain):
    store = Store(":memory:")
    site = SiteService(store, keychain)
    runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
    window = ConsoleWindow(Commands(site, runtime, Principal.open_site("hayda"), evidence_dir=tmp_path))
    try:
        assert "nobody" in window.user_label.text()
        window.configure_button.setChecked(True)
        assert window._configuring
    finally:
        window.close()
        store.close()


# --------------------------------------------------------------- dialogs


def test_the_add_camera_dialog_ok_path_adds_a_camera(console, qt_app, reference_video, monkeypatch):
    console.configure_button.setChecked(True)
    opened = {}

    def instead(dialog):
        opened["dialog"] = dialog
        dialog.identifier.setText("gate")
        dialog.source.setText(str(reference_video))
        dialog.record.setChecked(True)
        _press_ok(dialog)
        value = dialog.value() if dialog.result() == QDialog.DialogCode.Accepted else None
        dialog.deleteLater()
        return dialog.result() == QDialog.DialogCode.Accepted, value

    monkeypatch.setattr(dialogs, "ask", instead)
    console._add_camera()
    assert [c.id for c in console.commands.cameras()] == ["gate"]
    assert console.commands.cameras()[0].record
    assert "Added gate" in console.status.currentMessage()
    assert console.camera_list.tree.topLevelItemCount() == 1


def test_the_place_dialog_validates_and_places(console, qt_app, reference_video, monkeypatch):
    console.configure_button.setChecked(True)
    console.commands.add_camera("gate", str(reference_video))
    console.refresh_site()
    console.camera_list.tree.topLevelItem(0).setSelected(True)

    dialog = dialogs.PlaceCameraDialog("gate")
    dialog.set_pose(CameraPose(LatLon(33.8938, 35.5018), 4.0, 10.0, 12.0))
    assert "horizon" in dialog.check(), "a camera pointed up sees no ground"
    dialog.set_pose(CameraPose(LatLon(33.8938, 35.5018), 4.0, 10.0, -22.0))
    assert dialog.check() is None
    pose = dialog.value()
    dialog.deleteLater()

    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (True, pose))[1])
    console._place_camera()
    assert console.commands.cameras()[0].placed
    assert console.placement_label.text() == "1 of 1 camera(s) placed"
    assert console.plan._cameras, "the plan did not learn about the placement"


def test_a_camera_can_be_renamed_and_moved_to_a_new_address_from_the_window(console, qt_app, reference_video,
                                                                            monkeypatch):
    """Both existed only on the command line, so the two interfaces disagreed."""
    console.configure_button.setChecked(True)
    console.commands.add_camera("gate", str(reference_video), pose=CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0))
    console.refresh_site()
    console.camera_list.tree.topLevelItem(0).setSelected(True)

    dialog = dialogs.EditCameraDialog(console.commands.cameras()[0])
    assert dialog.identifier.text() == "gate", "the identifier is shown, not editable"
    dialog.display_name.setText("  ")
    assert "needs a name" in dialog.check()
    dialog.display_name.setText("North gate")
    dialog.source.setText("device:1")
    assert dialog.check() is None
    value = dialog.value()
    dialog.deleteLater()

    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (True, value))[1])
    console._edit_camera()
    camera = console.commands.cameras()[0]
    assert camera.id == "gate" and camera.name == "North gate" and camera.source == "device:1"
    assert camera.placed, "moving a camera must keep its placement, not throw it away"
    actions = [r["action"] for r in console.commands.audit_rows()]
    assert "camera.renamed" in actions and "camera.source_changed" in actions

    # Saving the dialog unchanged writes nothing: an audit row for a change
    # nobody made is a row somebody has to explain later.
    before = len(console.commands.audit_rows())
    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (True, {"name": "North gate",
                                                                            "source": "device:1"}))[1])
    console._edit_camera()
    assert len(console.commands.audit_rows()) == before
    assert "unchanged" in console.status.currentMessage()


def test_drawing_a_zone_on_the_plan_creates_it(console, qt_app, monkeypatch):
    console.configure_button.setChecked(True)
    console.commands.add_camera("gate", "clip.mp4", pose=CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0))
    console.refresh_site()
    console.zone_button.setChecked(True)
    assert console.plan.drawing
    for x, y in ((100, 100), (150, 100), (150, 150)):
        console.plan.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPoint(x, y).toPointF(),
                                                 Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                                 Qt.KeyboardModifier.NoModifier))
    assert len(console.plan.draft) == 3

    def instead(dialog):
        dialog.identifier.setText("yard")
        dialog.name.setText("Yard")
        _press_ok(dialog)
        value = dialog.value()
        dialog.deleteLater()
        return True, value

    monkeypatch.setattr(dialogs, "ask", instead)
    console.zone_button.setChecked(False)
    zones = console.commands.zones()
    assert [z.id for z in zones] == ["yard"] and len(zones[0].ring) == 3
    assert not console.plan.drawing and console.plan._zones


def test_two_points_are_refused_and_nothing_is_added(console, qt_app):
    console.configure_button.setChecked(True)
    console.zone_button.setChecked(True)
    console.plan._draft = [LatLon(0, 0), LatLon(0, 0.001)]
    console.zone_button.setChecked(False)
    assert console.commands.zones() == []
    assert "three" in console.status.currentMessage()


# ----------------------------------------------------------- the whole run


def test_a_file_runs_through_the_console_and_produces_an_incident(console, qt_app, reference_video):
    from test_runtime import _zone_where_the_block_stops

    pose = CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0)
    console.configure_button.setChecked(True)
    console.commands.add_camera("gate", str(reference_video), pose=pose)
    console.commands.add_zone("yard", "Yard", "RESTRICTED", _zone_where_the_block_stops(pose))
    console.refresh_site()
    console._start()
    assert console.commands.runtime.running
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline and console.commands.runtime.running:
        console._collect()
        qt_app.processEvents()
        time.sleep(0.05)
    console._collect(final=True)
    assert console.incidents.tree.topLevelItemCount() >= 1, "the block entered the yard and the console showed nothing"
    assert console.tracks.tree.topLevelItemCount() >= 0
    assert console._views and console._views["gate"]._pixmap is not None, "no frame reached the wall"
    assert "Finished" in console.status.currentMessage() or not console.commands.runtime.running


def test_an_alert_shows_once_and_leaves_when_it_clears(console, qt_app, monkeypatch):
    beeps = []
    monkeypatch.setattr(QApplication, "beep", staticmethod(lambda: beeps.append(1)))
    assert console.alert_label.text() == ""
    console.commands.alerts().raise_(CAMERA_DARK, "gate", "no frame for 30 s")
    console._collect()
    assert "camera.dark" in console.alert_label.text() and beeps == [1]
    console._collect()
    assert beeps == [1], "an alarm that repeats every poll is one somebody mutes"
    console.commands.alerts().clear(CAMERA_DARK, "gate")
    console._collect()
    assert console.alert_label.text() == ""


@pytest.mark.parametrize("console", [VIEWER], indirect=True)
def test_a_viewer_cannot_export(console, qt_app, monkeypatch):
    from test_incidents import event
    from vigil.domain.incidents import Correlator

    events = [event("a", 1, 10_000), event("a", 2, 12_000)]
    console.commands._site.store.save_events(events)
    incident = Correlator().correlate(events)[0]
    console.commands._site.store.save_incidents([incident])
    console.incidents.show_incidents([incident])
    console.incidents.tree.topLevelItem(0).setSelected(True)
    shown = []
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", staticmethod(lambda *a, **k: shown.append(a)))
    console._export()
    assert "may not" in console.status.currentMessage() and not shown
    assert "console.refused" in [r["action"] for r in console.commands.audit_rows()]


# ------------------------------------------------------------- appearance


def test_the_status_message_keeps_its_room_and_the_labels_elide(console, qt_app):
    long = "yolov8n-seg — watching bicycle, bus, car, motorcycle, person, truck with masks, 0123456789ab"
    console.detector_label.setText(long)
    qt_app.processEvents()
    assert console.detector_label.text() == long, "text() must return the whole text"
    assert console.detector_label.elided and console.detector_label.toolTip() == long
    taken = sum(l.sizeHint().width() for l in (console.user_label, console.lock_label, console.detector_label, console.alert_label))
    assert taken <= console.status.width() * 0.66, f"the permanent labels took {taken} of {console.status.width()}"


def test_the_camera_list_shows_all_five_columns_without_a_scrollbar(console, qt_app, reference_video):
    console.configure_button.setChecked(True)
    console.commands.add_camera("north-gate-camera-07", str(reference_video))
    console.refresh_site()
    qt_app.processEvents()
    tree = console.camera_list.tree
    widths = [tree.columnWidth(i) for i in range(5)]
    assert sum(widths) <= tree.viewport().width() + 2, (widths, tree.viewport().width())
    assert not tree.horizontalScrollBar().isVisible()
    assert widths[3] >= 70 and widths[4] >= 40
    item = tree.topLevelItem(0)
    assert item.toolTip(3), "an elided status must still be readable on hover"
    assert item.toolTip(1) == str(reference_video)


def test_the_record_box_follows_the_lock_and_a_tick_is_audited(console, qt_app, reference_video):
    console.configure_button.setChecked(True)
    console.commands.add_camera("gate", str(reference_video))
    console.refresh_site()
    item = console.camera_list.tree.topLevelItem(0)
    assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    item.setCheckState(4, Qt.CheckState.Checked)
    assert console.commands.cameras()[0].record
    assert "camera.recording" in [r["action"] for r in console.commands.audit_rows()]
    console.configure_button.setChecked(False)
    assert not (console.camera_list.tree.topLevelItem(0).flags() & Qt.ItemFlag.ItemIsUserCheckable)


def test_the_console_entry_point_signs_in_before_the_window_and_never_blocks_a_timed_run():
    from vigil.interfaces.console import main as console_main

    source = inspect.getsource(console_main.run)
    assert source.index("_sign_in(") < source.index("ConsoleWindow(commands)")
    sign_in = inspect.getsource(console_main._sign_in)
    assert "arguments.seconds is not None" in sign_in, "a timed run must not stop on a dialog"
    assert "--for" in console_main.build_parser().format_help()


def test_the_plan_maps_latitude_to_pixels_and_back_and_refuses_to_place_the_unprojected(qt_app, pose):
    from vigil.domain.geo import PositionEstimate, PositionSource, destination_point
    from vigil.interfaces.console.plan import PlanView

    plan = PlanView()
    plan.resize(400, 400)
    plan.set_cameras({"gate": pose})
    screen = plan._to_screen(pose.position)
    back = plan._to_lat_lon(screen.x(), screen.y())
    assert abs(back.lat - pose.position.lat) < 1e-7 and abs(back.lon - pose.position.lon) < 1e-7
    north = plan._to_screen(destination_point(pose.position, 0.0, 10.0))
    assert north.y() < screen.y(), "north must be up"

    class _Track:
        id = 1
        position = None
        heading_degrees = None
        speed_mps = None

    plan.set_tracks("gate", [_Track()])
    plan.grab()  # an unprojected track must not be drawn, and must not crash the paint
    _Track.position = PositionEstimate(pose.position, 1.0, PositionSource.CAMERA_FALLBACK)
    plan.set_tracks("gate", [_Track()])
    plan.grab()
    plan.deleteLater()


def test_the_photographer_is_owned_by_the_window_and_writes_every_panel(console, qt_app, tmp_path):
    """A slot whose object nobody owns is collected and never runs — silently."""
    import ast

    from vigil.interfaces.console.main import _Photographer

    source = ast.parse((CONSOLE / "main.py").read_text(encoding="utf-8"))
    connects = [ast.unparse(n) for n in ast.walk(source)
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "connect"]
    assert any("window._photographer.shoot" in c for c in connects), "the photographer must be owned by the window"
    assert not any("_Photographer(" in c for c in connects), "a temporary object was connected"

    into = tmp_path / "shots"
    _Photographer(console, into).shoot()
    names = sorted(p.name for p in into.glob("*.png"))
    assert names == ["cameras.png", "console.png", "incidents.png", "plan.png", "tracks.png", "wall.png"]
    assert all(p.stat().st_size > 0 for p in into.glob("*.png"))


def test_the_bar_says_what_is_drawing_the_conclusions_and_what_it_cannot_do(console, qt_app, tmp_path):
    """A console that silently fell back to motion is what the first photograph caught."""
    from vigil.domain.detection import DetectorInfo
    from vigil.interfaces.console.commands import Commands

    assert "motion only" in console.detector_label.text() and "cannot classify" in console.detector_label.text()
    console.commands.model = tmp_path / "yolov8n-seg.onnx"
    console.refresh_site()
    assert console.detector_label.text() == "Will use yolov8n-seg.onnx"

    motion = DetectorInfo("motion", "MOG2 background subtraction", classifies=False)
    classifier = DetectorInfo("onnx-segment", "yolov8n-seg", model_sha256="ab" * 32,
                              class_names={0: "person", 2: "car"}, classifies=True)
    seen = {"info": motion}
    console._detector_info = lambda camera_id: seen["info"]
    console._views["gate"] = console._views.get("gate") or object()
    console._show_detector()
    assert "does not classify" in console.detector_label.text()
    assert "cannot see a stationary object" in console.detector_label.text()
    seen["info"] = classifier
    console._show_detector()
    text = console.detector_label.text()
    assert text.startswith("yolov8n-seg — watching car, person") and "with masks" in text and "abababababab" in text
    console._views.pop("gate", None)


def test_selecting_an_incident_shows_why_it_was_raised(console, qt_app):
    from test_incidents import event
    from vigil.domain.incidents import Correlator

    assert "Select an incident" in console.detail.text.toPlainText()
    events = [event("a", 1, 10_000), event("a", 2, 11_000)]
    incident = Correlator().correlate(events)[0]
    console.incidents.show_incidents([incident])
    console.incidents.tree.topLevelItem(0).setSelected(True)
    shown = console.detail.text.toPlainText()
    assert console.tabs.currentWidget() is console.detail
    assert incident.summary in shown and incident.id in shown
    assert "zone-entry" in shown and "confidence" in shown
    for condition in incident.events[0].evidence.conditions:
        assert condition in shown, "an event must show the conditions its rule checked"
    for factor in incident.risk.factors:
        assert factor.reason in shown, "a risk score must show what it is made of"
    assert "same object" in shown or "No association" in shown


def test_the_why_panel_says_how_far_each_event_was_from_its_camera(console, qt_app, pose):
    """The exported report has said it for months; the person acting on it reads the screen."""
    from test_incidents import event
    from vigil.domain.incidents import Correlator
    from vigil.interfaces.console.widgets import IncidentDetail

    incident = Correlator().correlate([event("a", 1, 10_000), event("a", 2, 11_000)])[0]
    placed = incident.events[0]
    assert placed.evidence.distance_from(None) is None, "no pose, no distance — not a bare number"
    away = placed.evidence.distance_from(pose)

    detail = IncidentDetail()
    detail.show_incident(incident, {"a": pose})
    shown = detail.text.toPlainText()
    assert away is not None and away.describe() in shown and "±" in shown
    assert f"from {placed.evidence.camera_id}" in shown

    # A camera nobody placed reports no distance rather than inventing one.
    detail.show_incident(incident, {})
    assert away.describe() not in detail.text.toPlainText()
    detail.deleteLater()


def test_a_track_the_geometry_could_not_place_is_shown_as_unplaced_never_as_a_fix(qt_app, pose):
    """`CameraFallback` exists so an operator still learns "something is at this camera"."""
    from vigil.domain.geo import PositionEstimate, PositionSource
    from vigil.interfaces.console.plan import PlanView

    class _Track:
        id = 3
        heading_degrees = None
        speed_mps = None
        position = PositionEstimate(pose.position, pose.range_meters, PositionSource.CAMERA_FALLBACK)

    plan = PlanView()
    plan.resize(400, 400)
    plan.set_cameras({"gate": pose})
    plan.set_tracks("gate", [_Track()])
    drawn = []
    plan._draw_unprojected = lambda painter, camera_id, count: drawn.append((camera_id, count))
    plan.grab()
    assert drawn == [("gate", 1)], "an unplaced track was drawn as nothing at all"
    plan.deleteLater()


def test_a_selected_row_keeps_the_colour_that_says_what_the_camera_is_doing(qt_app):
    """Green LIVE text on a saturated blue selection was unreadable."""
    from vigil.interfaces.console import theme

    sheet = theme.stylesheet()
    tree_rule = sheet[sheet.index("QTreeWidget::item:selected"):]
    tree_rule = tree_rule[:tree_rule.index("}")]
    assert "selection-color" not in tree_rule and "color:" not in tree_rule.replace("background", "")
    assert "rgba(" in tree_rule, "the selection must be a wash, not a solid fill"


def test_two_camera_labels_a_few_metres_apart_do_not_print_over_each_other(qt_app, pose):
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QPainter
    from vigil.domain.geo import destination_point
    from vigil.interfaces.console.plan import PlanView

    close = CameraPose(destination_point(pose.position, 90, 3.0), 4.0, 90.0, -25.0)
    plan = PlanView()
    plan.resize(500, 500)
    plan.set_cameras({"laptop": pose, "street": close})
    picture = plan.grab()
    painter = QPainter(picture)
    taken: list[QRectF] = []
    first = plan._free_label_spot(painter, QPointF(100, 100), "laptop", taken)
    second = plan._free_label_spot(painter, QPointF(100, 100), "street", taken)
    painter.end()
    assert first.y() < second.y(), "the second label sat on top of the first"
    assert len(taken) == 2
    plan.deleteLater()


def _one_incident(console):
    from test_incidents import event
    from vigil.domain.incidents import Correlator

    events = [event("a", 1, 10_000), event("a", 2, 40_000)]
    console.commands._site.store.save_events(events)
    incidents = Correlator().correlate(events)
    console.commands._site.store.save_incidents(incidents)
    console._refresh_incidents()
    console.incidents.tree.topLevelItem(0).setSelected(True)
    return console.incidents.selected_incident()


def test_an_operator_works_the_queue_from_the_console(console, qt_app, monkeypatch):
    incident = _one_incident(console)
    assert incident is not None
    assert console.incidents.tree.topLevelItem(0).text(5) == "new"
    assert console.acknowledge_button.isEnabled() and console.dismiss_button.isEnabled()

    console._acknowledge()
    assert "Acknowledged" in console.status.currentMessage()
    assert console.incidents.tree.topLevelItem(0).text(5) == "acknowledged · alice"
    assert "acknowledged by user:alice" in console.detail.text.toPlainText()

    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (True, "the cat again"))[1])
    console._dismiss()
    assert "the cat again" in console.status.currentMessage()
    assert console.incidents.tree.topLevelItemCount() == 0, "the queue must not show what was dismissed"
    console.dismissed_box.setChecked(True)
    assert console.incidents.tree.topLevelItemCount() == 1
    assert console.incidents.tree.topLevelItem(0).text(5) == "dismissed · alice"
    actions = [r["action"] for r in console.commands.audit_rows()]
    assert "incident.acknowledged" in actions and "incident.dismissed" in actions


def test_a_dismissal_needs_a_reason_and_a_cancelled_dialog_changes_nothing(console, qt_app, monkeypatch):
    incident = _one_incident(console)
    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (False, None))[1])
    console._dismiss()
    assert console.incidents.tree.topLevelItemCount() == 1, "cancelling changed something"

    from vigil.interfaces.console.dialogs import NoteDialog

    dialog = NoteDialog("Dismiss", "why?")
    assert dialog.check() == "A reason is required."
    dialog.note.setText("  a delivery  ")
    assert dialog.check() is None and dialog.value() == "a delivery"
    dialog.deleteLater()


@pytest.mark.parametrize("console", [VIEWER], indirect=True)
def test_a_viewer_may_not_judge_an_incident(console, qt_app):
    _one_incident(console)
    assert not console.acknowledge_button.isEnabled() and not console.dismiss_button.isEnabled()
    console._acknowledge()
    assert "may not" in console.status.currentMessage()
    assert "console.refused" in [r["action"] for r in console.commands.audit_rows()]


def test_the_bar_names_the_site_and_the_clock_its_schedules_are_read_in(console, qt_app):
    """A zone that closes at 22:00 closes in this clock; an operator has to be able to check that."""
    assert console.site_label.text() == "Unnamed site · UTC"
    console.commands._site.name_site("Depot", "Asia/Beirut", by=OPERATOR)
    console._show_site()
    assert console.site_label.text() == "Depot · Asia/Beirut"
    assert "Depot" in console.windowTitle()
    assert "clock" in console.site_label.toolTip()


def test_a_zone_is_edited_through_the_same_dialog_that_drew_it(console, qt_app, monkeypatch):
    from vigil.domain.zones import Schedule, ZoneKind
    from vigil.interfaces.console.dialogs import ZoneDialog

    console.configure_button.setChecked(True)
    ring = [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0.001)]
    console.commands.add_zone("yard", "Yrad", "INTEREST", ring, watch=["person"], schedule=Schedule(22, 6))
    console.refresh_site()
    existing = console.commands.zones()[0]

    dialog = ZoneDialog((), ["person", "car"], existing=existing)
    assert dialog.identifier.text() == "yard" and dialog.identifier.isReadOnly()
    assert dialog.name.text() == "Yrad" and dialog.kind.currentData() == "INTEREST"
    assert dialog.closed.isChecked() and dialog.closed_from.value() == 22
    checked = [dialog.watch.item(i).text() for i in range(dialog.watch.count())
               if dialog.watch.item(i).checkState() == Qt.CheckState.Checked]
    assert checked == ["person"], "the existing watch list was not shown"
    dialog.deleteLater()

    monkeypatch.setattr(console, "_pick_zone", lambda title: existing)

    def instead(shown):
        shown.name.setText("Yard")
        shown.kind.setCurrentIndex(shown.kind.findData(ZoneKind.RESTRICTED.value))
        value = shown.value()
        shown.deleteLater()
        return True, value

    monkeypatch.setattr(dialogs, "ask", instead)
    console._edit_zone()
    saved = console.commands.zones()[0]
    assert saved.name == "Yard" and saved.kind is ZoneKind.RESTRICTED
    assert saved.ring == tuple(ring), "the ring somebody drew was lost"
    assert "zone.changed" in [r["action"] for r in console.commands.audit_rows()]


def test_the_incident_filters_narrow_the_list_and_say_what_was_asked(console, qt_app):
    from test_incidents import event
    from vigil.domain.events import Severity
    from vigil.domain.incidents import Correlator

    events = [event("north-gate", 1, 10_000, severity=Severity.HIGH),
              event("loading-bay", 2, 500_000, severity=Severity.LOW)]
    console.commands._site.store.save_events(events)
    console.commands._site.store.save_incidents(Correlator().correlate(events))
    console.commands.add_camera("north-gate", "one.mp4")
    console.commands.add_camera("loading-bay", "two.mp4")
    console.refresh_site()
    console._refresh_incidents()
    assert console.incidents.tree.topLevelItemCount() == 2

    picker = console.incidents.camera_filter
    assert [picker.itemData(i) for i in range(picker.count())] == ["", "loading-bay", "north-gate"]
    picker.setCurrentIndex(picker.findData("north-gate"))
    assert console.incidents.tree.topLevelItemCount() == 1
    assert "camera north-gate" in console.status.currentMessage()

    picker.setCurrentIndex(0)
    console.incidents.severity_filter.setCurrentIndex(console.incidents.severity_filter.findData("HIGH"))
    assert console.incidents.tree.topLevelItemCount() == 1
    console.incidents.severity_filter.setCurrentIndex(0)
    console.incidents.text_filter.setText("nothing like this")
    assert console.incidents.tree.topLevelItemCount() == 0
    console.incidents.text_filter.setText("")
    assert console.incidents.tree.topLevelItemCount() == 2

    # The picker keeps the operator's choice when the site is refreshed.
    picker.setCurrentIndex(picker.findData("loading-bay"))
    console.refresh_site()
    assert picker.currentData() == "loading-bay"


def test_the_track_table_says_how_far_away_and_what_the_track_is_doing(console, qt_app, pose):
    from vigil.domain.detection import BoundingBox, DetectorInfo
    from vigil.domain.geo import PositionEstimate, PositionSource, Vec2, destination_point
    from vigil.domain.relations import Relation, RelationKind
    from vigil.domain.tracking import Track

    info = DetectorInfo("onnx-detect", "w", class_names={0: "person", 2: "backpack"}, classifies=True)
    where = destination_point(pose.position, 0.0, 12.0)

    def made(track_id, class_id):
        return Track.observing(track_id, class_id, BoundingBox(0.4, 0.5, 0.1, 0.2), contact=Vec2(0.45, 0.7),
                               confidence=0.8,
                               position=PositionEstimate(where, 1.5, PositionSource.GROUND_PROJECTION))

    carried = Relation(RelationKind.CARRIED, 1, 2, confidence=0.7,
                       conditions=("58% of the backpack's box lay within the person's",))
    rows = [("gate", made(1, 0), info, pose, (carried,)), ("gate", made(2, 2), info, pose, ())]
    console.tracks.show_tracks(rows)

    first = console.tracks.tree.topLevelItem(0)
    assert first.text(5) == "12.0 ± 1.5 m", first.text(5)
    assert first.text(6) == "person appears to be carrying backpack"
    assert "58% of the backpack's box" in first.toolTip(6), "the reasons must be one hover away"
    assert "±1.5 m" in first.toolTip(5)
    assert console.tracks.tree.topLevelItem(1).text(6) == ""

    # A track with no ground position says so rather than printing a number.
    nowhere = Track.observing(3, 0, BoundingBox(0.1, 0.1, 0.1, 0.1), contact=Vec2(0.15, 0.2), confidence=0.5)
    console.tracks.show_tracks([("gate", nowhere, info, pose, ())])
    assert console.tracks.tree.topLevelItem(0).text(5) == "—"
    assert "not placed on the ground" in console.tracks.tree.topLevelItem(0).toolTip(5)


def test_the_track_table_counts_who_is_apparently_in_the_car(qt_app, pose):
    """The car does not know it is occupied; the count comes from the people."""
    from vigil.adapters.detectors import DetectorInfo
    from vigil.domain.detection import BoundingBox
    from vigil.domain.geo import PositionEstimate, PositionSource, Vec2, destination_point
    from vigil.domain.relations import Relation, RelationKind
    from vigil.domain.tracking import Track
    from vigil.interfaces.console.widgets import TrackTable

    info = DetectorInfo("onnx-detect", "w", class_names={0: "person", 2: "car"}, classifies=True)
    where = destination_point(pose.position, 0.0, 12.0)

    def made(track_id, class_id):
        return Track.observing(track_id, class_id, BoundingBox(0.4, 0.5, 0.1, 0.2), contact=Vec2(0.45, 0.7),
                               confidence=0.8,
                               position=PositionEstimate(where, 1.5, PositionSource.GROUND_PROJECTION))

    inside = tuple(Relation(RelationKind.INSIDE, subject, 9, confidence=0.6,
                            conditions=("72% of the person's box lay within the car's",))
                   for subject in (1, 2, 3))
    table = TrackTable()
    rows = [("gate", made(9, 2), info, pose, inside)]
    rows.extend(("gate", made(subject, 0), info, pose, (inside[subject - 1],)) for subject in (1, 2, 3))
    table.show_tracks(rows)

    car = table.tree.topLevelItem(0)
    assert car.text(2) == "car"
    assert car.text(6) == "3 people apparently inside it", car.text(6)
    assert "72% of the person's box" in car.toolTip(6), "an inference must show its working"
    person = table.tree.topLevelItem(1)
    assert "probably in" in person.text(6), "the person's side of the same relation stays hedged"


def test_the_window_changes_what_the_site_watches_for(console, qt_app, monkeypatch):
    """The setting existed only on the command line, so the two interfaces disagreed."""
    from vigil.service.detection import DetectionSettings

    console.configure_button.setChecked(True)
    assert console.commands.detection() == DetectionSettings(), "a fresh site watches the built-in list"

    dialog = dialogs.DetectionDialog(DetectionSettings(frozenset({"person"}), 0.6), ("person", "car"))
    assert dialog.watch.text() == "person" and dialog.confidence.value() == 0.6
    dialog.watch.setText("person, car")
    dialog.confidence.setValue(0.75)
    value = dialog.value()
    dialog.deleteLater()

    monkeypatch.setattr(dialogs, "ask", lambda d: (d.deleteLater(), (True, value))[1])
    console._set_detection()
    stored = console.commands.detection()
    assert stored.labels == frozenset({"person", "car"}) and stored.confidence == 0.75
    assert "watching car, person" in console.status.currentMessage()
    assert "site.detection_changed" in [r["action"] for r in console.commands.audit_rows()]

    # Nothing chosen means the built-in list, and the spin box says so rather
    # than showing a zero somebody would read as a threshold.
    cleared = dialogs.DetectionDialog(DetectionSettings())
    assert cleared.value() == {"labels": [], "confidence": None}
    assert cleared.confidence.specialValueText()
    cleared.deleteLater()

    # A setting the detector cannot satisfy is refused, and nothing is stored.
    refused = console.commands.set_detection(["person"], 1.5)
    assert not refused and "outside" in refused.message
    assert console.commands.detection() == stored, "a refused setting must not be half-applied"


def test_the_toolbar_stays_readable_on_the_narrowest_window_it_claims(console, qt_app):
    """Qt answers a toolbar that will not fit by cutting the words in half.

    The shipped window read "dd camera.", "elete zone" and "ort evidenc"
    after one more button was added to a row that had just fitted. Nothing
    failed, nothing was logged, and only a photograph showed it.
    """
    from PySide6.QtWidgets import QPushButton

    console.resize(console.NARROWEST_WINDOW, 800)
    console.show()
    qt_app.processEvents()
    toolbar = console.toolbar
    for index in range(toolbar.count()):
        widget = toolbar.itemAt(index).widget()
        assert widget.width() >= widget.sizeHint().width(),             f"{widget.text()!r} is narrower than its own label and would be cut"
        assert widget.x() + widget.width() <= console.NARROWEST_WINDOW, f"{widget.text()!r} is off the window"
    assert toolbar.rows() >= 1

    # Every button an operator can press is in the toolbar: a control that
    # exists but sits in no layout is worse than one that is missing.
    placed = {toolbar.itemAt(i).widget() for i in range(toolbar.count())}
    buttons = [w for w in console.findChildren(QPushButton) if w.parent() is console]
    assert set(buttons) <= placed, "a button was built and never added to the toolbar"
    assert set(console._configure_only()) <= placed


def test_the_plan_links_two_tracks_a_relation_joins(qt_app, pose):
    from vigil.domain.detection import BoundingBox
    from vigil.domain.geo import PositionEstimate, PositionSource, Vec2, destination_point
    from vigil.domain.relations import Relation, RelationKind
    from vigil.domain.tracking import Track
    from vigil.interfaces.console.plan import PlanView

    def somewhere(track_id, bearing):
        point = destination_point(pose.position, bearing, 8.0)
        return Track.observing(track_id, 0, BoundingBox(0.4, 0.5, 0.1, 0.2), contact=Vec2(0.45, 0.7),
                               confidence=0.8,
                               position=PositionEstimate(point, 0.6, PositionSource.GROUND_PROJECTION))

    plan = PlanView()
    plan.resize(400, 400)
    plan.set_cameras({"gate": pose})
    together = Relation(RelationKind.NEAR, 1, 2, confidence=0.8)
    plan.set_tracks("gate", [somewhere(1, 0.0), somewhere(2, 20.0)], [together])
    drawn = []
    plan._draw_links = lambda painter, camera_id: drawn.append(camera_id)
    plan.grab()
    assert drawn == ["gate"]

    # A relation whose other end is not on the ground draws nothing and does
    # not crash the paint.
    plan.set_tracks("gate", [somewhere(1, 0.0)], [together])
    plan.grab()
    plan.clear_tracks()
    plan.grab()
    plan.deleteLater()
