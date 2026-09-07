"""What happens when the operator presses something.

Split from `window` at the line budget, and the seam is a real one: this file
is every verb, and `window` is the layout, the lock and the repaint. Nothing
here touches a widget's geometry and nothing there decides anything.

A mixin rather than a helper object holding a window, because every one of
these is a Qt slot connected to a bound method of the window itself — and a
slot connected to a method of *another* object that nothing owns is a slot
that silently never runs, which is this codebase's most expensive class of
bug and the reason `_Photographer` exists as a named attribute.

Each method is the same shape: read the selection, open a dialog through
`dialogs.ask`, hand the answer to `Commands`, say what happened. The service
decides; this asks and reports.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from . import dialogs


class ConsoleActions:
    """The command handlers. Mixed into `ConsoleWindow`, never used alone."""


    def _add_camera(self) -> None:
        ok, value = dialogs.ask(dialogs.AddCameraDialog(self))
        if not ok or value is None:
            return
        source = value["source"]
        outcome = self.commands.add_camera(value["id"], source, record=value["record"])
        if outcome and value["password"]:
            outcome = self.commands.set_password(value["id"], value["password"])
        self._say(outcome.message)
        if outcome:
            self.refresh_site()

    def _place_camera(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to place.")
            return
        ok, pose = dialogs.ask(dialogs.PlaceCameraDialog(camera.id, camera.pose, self))
        if not ok or pose is None:
            return
        self._say(self.commands.place_camera(camera.id, pose).message)
        self.refresh_site()

    def _calibrate_camera(self) -> None:
        """Measure the selected camera's pose against points on the plan."""
        from .calibrate import CalibrateDialog

        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to measure.")
            return
        if camera.pose is None:
            self._say(f"Place {camera.id} first, roughly. This measures a placement; it cannot "
                      f"invent one.")
            return
        view = self._views.get(camera.id)
        still = view.still() if view is not None else None
        if still is None:
            self._say(f"No frame from {camera.id} yet. Start the analysis, wait for the picture, "
                      f"then measure — the points are marked on a frame.")
            return
        ok, value = dialogs.ask(CalibrateDialog(camera, self.commands, still, self))
        if not ok or not value:
            return
        points, solve_position = value
        outcome = self.commands.calibrate_camera(camera.id, points, solve_position=solve_position)
        self._say(outcome.message)
        if outcome:
            self.refresh_site()

    def _edit_camera(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to edit.")
            return
        ok, value = dialogs.ask(dialogs.EditCameraDialog(camera, self))
        if not ok or value is None:
            return
        said = []
        # Two service calls, each audited on its own, and neither attempted
        # when the field was left alone: an audit row for a change nobody
        # made is a row somebody has to explain later.
        if value["name"] != camera.name:
            said.append(self.commands.rename_camera(camera.id, value["name"]).message)
        if value["source"] != camera.source:
            said.append(self.commands.set_source(camera.id, value["source"]).message)
        self._say(" ".join(said) if said else f"{camera.id} is unchanged.")
        self.refresh_site()

    def _set_password(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera.")
            return
        ok, password = dialogs.ask(dialogs.PasswordDialog(camera.id, self))
        if ok and password:
            self._say(self.commands.set_password(camera.id, password).message)

    def _remove_camera(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._say("Select a camera to remove.")
            return
        answer = QMessageBox.question(self, f"Remove {camera.id}?",
                                      "The camera is forgotten and its password removed from the keychain. "
                                      "What it saw — its events and incidents — is kept.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._say(self.commands.remove_camera(camera.id).message)
        self.refresh_site()

    def _set_detection(self) -> None:
        info = self.commands.detector()
        names = tuple(info.class_names.values()) if info is not None and info.classifies else ()
        ok, value = dialogs.ask(dialogs.DetectionDialog(self.commands.detection(), names, self))
        if not ok or value is None:
            return
        self._say(self.commands.set_detection(value["labels"], value["confidence"]).message)
        self._show_detector()

    def _identity(self) -> None:
        """Faces and plates. A dialog would imply it is a setting like the
        others; it is not, so this says what it is and where to change it."""
        QMessageBox.information(
            self, "Faces and plates",
            "Off by default, and off on this site unless somebody turned it on.\n\n"
            "It is changed from the command line on purpose: turning it on needs a written "
            "reason and a retention limit, both recorded against your name, and a dialog "
            "with two boxes invites treating that as a preference.\n\n"
            "    vigil identity show\n"
            "    vigil identity enable --retention-days 30 --reason \"...\"\n\n"
            "`vigil doctor` fails if it is ever on with no retention limit set.")

    def _draw_zone(self, on: bool) -> None:
        if on:
            self.plan.begin_zone()
            self.zone_button.setText("Finish zone")
            self._say("Click the plan to place the corners; three or more, then Finish zone.")
            return
        self.zone_button.setText("Draw zone")
        ring = self.plan.end_zone()
        if len(ring) < 3:
            if ring:
                self._say("A zone needs at least three points; nothing was added.")
            return
        ok, value = dialogs.ask(dialogs.ZoneDialog(ring, self.commands.labels(), self))
        if not ok or value is None:
            return
        outcome = self.commands.add_zone(value["id"], value["name"], value["kind"], value["ring"],
                                         watch=value["watch"], schedule=value["schedule"])
        self._say(outcome.message)
        self.refresh_site()

    def _edit_zone(self) -> None:
        zone = self._pick_zone("Edit a zone")
        if zone is None:
            return
        ok, value = dialogs.ask(dialogs.ZoneDialog((), self.commands.labels(), self, existing=zone))
        if not ok or value is None:
            return
        self._say(self.commands.edit_zone(zone.id, name=value["name"], kind=value["kind"],
                                          watch=value["watch"], schedule=value["schedule"]).message)
        self.refresh_site()

    def _pick_zone(self, title: str):
        zones = self.commands.zones()
        if not zones:
            self._say("There is no zone yet. Draw one first.")
            return None
        from PySide6.QtWidgets import QInputDialog

        names = [f"{z.id} — {z.name}" for z in zones]
        chosen, ok = QInputDialog.getItem(self, title, "Zone", names, 0, False)
        if not ok or not chosen:
            return None
        return next((z for z in zones if z.id == chosen.split(" — ")[0]), None)

    def _remove_zone(self) -> None:
        zone = self._pick_zone("Delete a zone")
        if zone is None:
            return
        self._say(self.commands.remove_zone(zone.id).message)
        self.refresh_site()

    def _start(self) -> None:
        outcome = self.commands.start()
        self._say(outcome.message)
        if not outcome:
            return
        self._rebuild_wall()
        self._timer.start()
        self._show_running(True)
        self.add_button.setEnabled(False)

    def _stop(self) -> None:
        self._timer.stop()
        outcome = self.commands.stop()
        self._collect(final=True)
        self._say(outcome.message)
        self._show_running(False)
        self.add_button.setEnabled(self._configuring)
        for view in self._views.values():
            view.set_caption("stopped")

    def _show_running(self, running: bool, site=None) -> None:
        """Which of the two loud buttons is live.

        Both are always present; the dead one is greyed rather than hidden, so
        the pair does not move under the cursor between runs.

        `site` is passed by every caller that already has one, so a repaint
        does not ask the service for the cameras a second time.
        """
        cameras = site.cameras if site is not None else self.commands.cameras()
        self.start_button.setEnabled(not running and bool(cameras))
        self.stop_button.setEnabled(running)

    def _acknowledge(self) -> None:
        incident = self.incidents.selected_incident()
        if incident is None:
            self._say("Select an incident to acknowledge.")
            return
        self._say(self.commands.acknowledge(incident).message)
        self._refresh_incidents()

    def _dismiss(self) -> None:
        incident = self.incidents.selected_incident()
        if incident is None:
            self._say("Select an incident to dismiss.")
            return
        ok, note = dialogs.ask(dialogs.NoteDialog(
            "Dismiss this incident", "Why is it not worth acting on? Without a reason a dismissal cannot be "
            "told from nobody having looked.", self))
        if not ok or not note:
            return
        self._say(self.commands.dismiss(incident, note).message)
        self._refresh_incidents()

    def _show_dismissed(self, on: bool) -> None:
        self.commands.show_dismissed = bool(on)
        self._refresh_incidents()

    def _filter_incidents(self, camera: str, severity: str, contains: str) -> None:
        self.commands.set_filter(camera, severity, contains)
        self._refresh_incidents()
        found = self.incidents.tree.topLevelItemCount()
        self._say(f"{found} incident(s) matching {self.commands.query.describe()}.")

    def _refresh_incidents(self) -> None:
        self.incidents.show_incidents(self.commands.incidents())
        self._incident_selected(self.incidents.selected_incident())

    def _export(self) -> None:
        incident = self.incidents.selected_incident()
        if incident is None:
            self._say("Select an incident to export.")
            return
        outcome = self.commands.export(incident)
        self._say(outcome.message)
        if outcome:
            QMessageBox.information(self, "Evidence exported", outcome.message)

    def _shortcuts(self) -> None:
        QMessageBox.information(
            self, "Keyboard",
            "F5 / F6\tstart and stop the analysis\n"
            "Ctrl+K\tunlock or re-lock the site controls\n"
            "Ctrl+N\tadd a camera\n"
            "Ctrl+A\tacknowledge the selected incident\n"
            "Ctrl+D\tdismiss it, with a reason\n"
            "Ctrl+E\texport its evidence package\n"
            "F1\tthis list")

    def _about(self) -> None:
        from ...version import describe

        QMessageBox.information(self, "About this build",
                                f"{describe()}\n\nNative Qt widgets — no embedded browser.\n"
                                "No Internet access at any point: no tiles, no telemetry, no model downloads.\n\n"
                                f"Signed in as {self.commands.principal.actor}.")
