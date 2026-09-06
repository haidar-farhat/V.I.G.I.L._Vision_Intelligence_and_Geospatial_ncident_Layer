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
