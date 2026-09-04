"""Tests for the People and Vehicles register panel.

The panel is the console's only screen over an identity register, so the
failures it must not have are the ones that make holding that register
indefensible:

- **It must list what is enrolled, of its kind, and nothing else.** A people
  panel that showed a van, or a vehicles panel that showed nobody, is a
  review screen that cannot be used for a review.
- **Enrolment must be an act on a track the operator picked**, and for a person
  only on a track the node holds face templates for. The button is disabled
  otherwise, with the reason as its tooltip — never silently.
- **The dialog must refuse a blank name and pass the basis and notes through
  unchanged.** A template held under a basis the operator did not choose is
  the failure the whole register is built to prevent.
- **Forgetting must name the subject and the counts before, and report the
  register's own counts after.** An erasure nobody could review at the moment
  of taking it is not an erasure anybody can defend.
- **The lock must hold.** In Monitor the three buttons are disabled and the
  direct methods refuse; the list and the history stay readable.
- **Nothing reaches a Qt slot's caller.** A refusal from the node is a sentence
  on the status line — and a *refusal* is worded as one, not as a failure.
- **The affordances the panel advertises must be the ones under test.** The
  buttons are clicked here, through the widgets; the dialog is driven by
  patching its ``exec`` on the class, the way the console's own tests answer
  a ``QMessageBox``, so a button wired to nothing fails.
- **The surface the panel asks of the node must be the surface the node has.**
  The first version of this panel required a `register` attribute the node
  never had, every fake here supplied one, and forty tests passed over a wire
  that would have failed on the first real refresh. So the surface is now
  checked against a real `sentinel.node.Node` — built over a temporary
  database, no camera started, no model anywhere — and the panel reads and
  writes through it, audit rows and all.

The register underneath is the real one — `Store(":memory:").register`, the
same object the node hands the console — and the node is stood in for by
`FakeActions`, which implements `IdentityActions` over that register the way
`engine/tests/test_registry.py` enrols directly. No model is loaded anywhere
in this file and none exists on the machine that runs it: a face template is
128 floats some encoder produced, and `template()` below produces them.
"""

from __future__ import annotations

import gc
import os
import re
import typing
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from sentinel.core import LatLon  # noqa: E402
from sentinel.faces import FaceTemplate as LiveTemplate  # noqa: E402
from sentinel.registry import (  # noqa: E402
    Confidence,
    FaceTemplate,
    Plate,
    RegistryError,
    RetentionPolicy,
    SubjectKind,
)
from sentinel.node import Node  # noqa: E402
from sentinel.site import DEFAULT_SITE_ID, Site  # noqa: E402
from sentinel.store import Store, StoreError  # noqa: E402

from sentinel_console.register_view import (  # noqa: E402
    HISTORY_CAMERA_COLUMN,
    HISTORY_CONFIDENCE_COLUMN,
    HISTORY_FIRST_COLUMN,
    HISTORY_LAST_COLUMN,
    HISTORY_SCORE_COLUMN,
    HISTORY_TRACK_COLUMN,
    IDENTIFIERS_COLUMN,
    IDENTITY_SURFACE,
    LAST_SEEN_COLUMN,
    NAME_COLUMN,
    NOTHING,
    PINNED_COLUMN,
    SIGHTINGS_COLUMN,
    EnrolDialog,
    IdentityActions,
    RegisterPanel,
    missing_from,
)
from sentinel_console.selection import Selection, SelectionBus  # noqa: E402

#: An arbitrary but fixed instant, in whole seconds, so times shown are exact.
NOW = 1_700_000_000_000
MINUTE = 60_000
MODEL = "pretend-sface-v1"

#: The one live track the fake node holds face templates for.
LIVE_CAMERA, LIVE_TRACK = "cam-01", 7


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def vector(who: str) -> np.ndarray:
    """128 unit-length floats, deterministic per name. Not a face; the shape of one."""
    rng = np.random.default_rng(abs(hash((MODEL, who))) % (2**32))
    values = rng.standard_normal(128).astype(np.float32)
    return values / np.linalg.norm(values)


def stored_template(who: str, quality: float = 0.87) -> FaceTemplate:
    """A template as the register stores it: packed bytes, tagged with its model."""
    return FaceTemplate(vector=vector(who).tobytes(), model=MODEL, quality=quality)


def live_template(who: str, quality: float) -> LiveTemplate:
    """A template as the node holds it for a live track: `sentinel.faces.FaceTemplate`."""
    return LiveTemplate(
        vector=tuple(float(v) for v in vector(who)),
        quality=quality,
        model=MODEL,
        source=f"{LIVE_CAMERA}#{LIVE_TRACK}",
        created_unix_millis=NOW,
    )


def utc(millis: int) -> str:
    return datetime.fromtimestamp(millis / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class FakeActions:
    """The node's identity surface, over the real register.

    Implements `IdentityActions` exactly — the register reachable only as
    `store.register`, the way `Node` keeps it, so the fake cannot offer the
    panel a member the node lacks — and records every write it is asked for,
    so a test can assert what the panel passed through: the name, the basis,
    the notes, the track. Refuses the way the node does: a blank name, a track
    with no templates.
    """

    def __init__(self, store: Store, *, status: str = "faces: on (stand-in)"):
        self.store = store
        self.identity_status = status
        self.templates: dict[tuple[str, int], tuple[LiveTemplate, ...]] = {
            (LIVE_CAMERA, LIVE_TRACK): (
                live_template("ali-blurred", 0.61),
                live_template("ali-clear", 0.93),
                live_template("ali-profile", 0.72),
            )
        }
        self.site_record = Site(
            id=DEFAULT_SITE_ID, name="Site", origin=LatLon(33.8938, 35.5018)
        )
        self.calls: list[tuple] = []

    def site(self):
        return self.site_record

    def set_identity(self, identity, *, reason: str):
        self.calls.append(("set_identity", identity, reason))
        return self.site_record

    def identity_of(self, camera_id: str, track_id: int):
        return None

    def templates_for(self, camera_id: str, track_id: int):
        return self.templates.get((camera_id, track_id), ())

    def enrol_person(self, name, camera_id, track_id, *, basis, notes=None):
        self.calls.append(("enrol_person", name, camera_id, track_id, basis, notes))
        if not name.strip():
            raise ValueError("a name is required")
        templates = self.templates_for(camera_id, track_id)
        if not templates:
            raise RegistryError(f"no face templates for {camera_id} #{track_id}")
        best = max(templates, key=lambda t: t.quality)
        subject_id = "person-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        self.store.register.enrol(
            subject_id=subject_id,
            display_name=name,
            identifier=FaceTemplate(
                vector=np.asarray(best.vector, np.float32).tobytes(),
                model=best.model,
                quality=best.quality,
            ),
            actor="operator:test",
            basis=basis,
            notes=notes,
            source_camera=camera_id,
            source_track=track_id,
            now_millis=NOW + 10 * MINUTE,
        )
        return subject_id

    def enrol_vehicle(self, name, plate, *, basis, notes=None, camera_id=None, track_id=None):
        self.calls.append(("enrol_vehicle", name, plate, basis, notes, camera_id, track_id))
        if not name.strip():
            raise ValueError("a name is required")
        subject_id = "vehicle-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        self.store.register.enrol(
            subject_id=subject_id,
            display_name=name,
            identifier=Plate(plate),
            actor="operator:test",
            basis=basis,
            notes=notes,
            source_camera=camera_id,
            source_track=track_id,
            now_millis=NOW + 10 * MINUTE,
        )
        return subject_id

    def forget_subject(self, subject_id):
        self.calls.append(("forget_subject", subject_id))
        return self.store.register.forget(subject_id, now_millis=NOW + 20 * MINUTE)

    def pin_subject(self, subject_id, pinned):
        self.calls.append(("pin_subject", subject_id, pinned))
        self.store.register.set_pinned(subject_id, pinned, actor="operator:test")


class RefusingActions(FakeActions):
    """A node that refuses every write, the way a node with faces off would."""

    def enrol_person(self, *args, **kwargs):
        raise RegistryError("faces are off on this node; nothing can be enrolled")

    def enrol_vehicle(self, *args, **kwargs):
        raise RegistryError("plates are off on this node; nothing can be enrolled")

    def forget_subject(self, subject_id):
        raise RegistryError("the register is read-only while a sweep runs")

    def pin_subject(self, subject_id, pinned):
        raise RegistryError("the register is read-only while a sweep runs")


@pytest.fixture
def store() -> Store:
    """Two people and one vehicle, enrolled the way the engine's tests do.

    Ali has two templates and three sightings across three cameras — one of
    them POSSIBLE, so the hedge has a row to be drawn on. Rana has one
    template, no sightings, and is pinned. The van has one plate and one
    DECLARED sighting. Enough that every column has a value to be wrong.
    """
    with Store(":memory:") as db:
        register = db.register
        register.enrol(
            subject_id="person-ali",
            display_name="Ali Hassan",
            identifier=stored_template("ali"),
            actor="operator:nadia",
            basis="employment contract, staff register",
            notes="night shift",
            source_camera="cam-01",
            source_track=3,
            now_millis=NOW,
        )
        register.enrol(
            subject_id="person-ali",
            display_name="Ali Hassan",
            identifier=stored_template("ali-in-a-hat", 0.7),
            actor="operator:nadia",
            basis="employment contract, staff register",
            now_millis=NOW + MINUTE,
        )
        register.enrol(
            subject_id="person-rana",
            display_name="Rana",
            identifier=stored_template("rana"),
            actor="operator:nadia",
            basis="consent",
            now_millis=NOW,
        )
        register.set_pinned("person-rana", True, actor="operator:nadia")
        register.enrol(
            subject_id="vehicle-van",
            display_name="Contractor van",
            identifier=Plate("B 7421", frames_agreeing=9),
            actor="operator:nadia",
            basis="site access list",
            now_millis=NOW,
        )
        for camera, track, offset, confidence, score in (
            ("cam-03", 31, 9 * MINUTE, Confidence.MATCH, 0.91),
            ("cam-01", 11, 1 * MINUTE, Confidence.MATCH, 0.88),
            ("cam-02", 21, 5 * MINUTE, Confidence.POSSIBLE, 0.42),
        ):
            register.record_sighting(
                subject_id="person-ali",
                camera_id=camera,
                track_id=track,
                first_seen_millis=NOW + offset,
                last_seen_millis=NOW + offset + 20_000,
                confidence=confidence,
                score=score,
            )
        register.record_sighting(
            subject_id="vehicle-van",
            camera_id="cam-01",
            track_id=99,
            first_seen_millis=NOW,
            last_seen_millis=NOW + 30_000,
            confidence=Confidence.DECLARED,
        )
        assert len(register.subjects()) == 3
        yield db


@pytest.fixture
def actions(store: Store) -> FakeActions:
    return FakeActions(store)


@pytest.fixture
def people(qt_app, actions: FakeActions) -> RegisterPanel:
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(actions)
    widget.set_editable(True)
    return widget


@pytest.fixture
def vehicles(qt_app, actions: FakeActions) -> RegisterPanel:
    widget = RegisterPanel(SubjectKind.VEHICLE)
    widget.set_actions(actions)
    widget.set_editable(True)
    return widget


def accept_dialog_with(monkeypatch, *, name: str, basis: str, notes: str = "", plate: str | None = None):
    """Drive the enrol dialog the way a person would, without a modal loop.

    Patched on the class, not the instance: the panel builds its own dialog
    inside the click, so an instance patch would never be reached. What the
    fake does is fill the fields and accept through the dialog's own
    `accept`, so the refusal rules still apply.
    """

    def filled_in(dialog):
        dialog.name_field.setText(name)
        dialog.basis_box.setCurrentText(basis)
        dialog.notes_field.setText(notes)
        if plate is not None and dialog.plate_field is not None:
            dialog.plate_field.setText(plate)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(EnrolDialog, "exec", filled_in)


def cancel_dialog(monkeypatch):
    monkeypatch.setattr(EnrolDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)


def answer_question(monkeypatch, answer, asked: list) -> None:
    """Answer the confirmation the way the console's own tests do, keeping its text."""

    def question(*args, **kwargs):
        asked.append(args[2])
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)


# ------------------------------------------------------------------ the list


def test_a_people_panel_lists_only_people_and_a_vehicles_panel_only_vehicles(people, vehicles):
    print("people:", people.listed_subject_ids(), "vehicles:", vehicles.listed_subject_ids())
    assert people.listed_subject_ids() == ["person-ali", "person-rana"], "by name, as the register orders"
    assert vehicles.listed_subject_ids() == ["vehicle-van"]


def test_a_panel_with_no_node_says_so_instead_of_showing_nothing(qt_app):
    widget = RegisterPanel(SubjectKind.PERSON)
    assert widget.listed_subject_ids() == []
    assert "no node is connected" in widget.capability_text().lower(), widget.capability_text()
    assert not widget.enrol_button.isEnabled()
    assert "no node" in widget.enrol_button.toolTip().lower()


def test_a_panel_built_before_the_node_reads_the_register_it_is_handed(qt_app, actions):
    widget = RegisterPanel(SubjectKind.VEHICLE)
    assert widget.listed_subject_ids() == []
    widget.set_actions(actions)
    assert widget.listed_subject_ids() == ["vehicle-van"]


def test_the_top_line_carries_the_nodes_own_identity_status(qt_app, store):
    # "faces: on but no models" has to be read beside the button it explains,
    # not in a log after an afternoon of wondering why it never enables.
    missing = "faces: on but no models in C:/Sentinel/models — expecting yunet.onnx and sface.onnx"
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(FakeActions(store, status=missing))
    print("capability:", widget.capability_text())
    assert missing in widget.capability_text()

    widget.set_actions(FakeActions(store, status="off"))
    assert "Identity: off." in widget.capability_text()
    assert missing not in widget.capability_text(), "a status about another node survived"


def test_a_subject_row_shows_identifiers_last_seen_sightings_and_the_pin(people, vehicles, store):
    ali = people.subject_row_texts("person-ali")
    rana = people.subject_row_texts("person-rana")
    van = vehicles.subject_row_texts("vehicle-van")
    print("ali:", ali, "rana:", rana, "van:", van)

    assert ali[NAME_COLUMN] == "Ali Hassan"
    assert ali[IDENTIFIERS_COLUMN] == "2 face template(s)"
    assert ali[LAST_SEEN_COLUMN] == utc(NOW + 9 * MINUTE + 20_000), "the latest last_seen, not the first"
    assert ali[SIGHTINGS_COLUMN] == "3"
    assert ali[PINNED_COLUMN] == NOTHING

    assert rana[LAST_SEEN_COLUMN] == NOTHING, "never seen must be said, not blanked"
    assert rana[SIGHTINGS_COLUMN] == "0"
    assert rana[PINNED_COLUMN] == "pinned"

    assert van[IDENTIFIERS_COLUMN] == "B 7421", "a plate is shown as read; a template is only counted"
    assert van[SIGHTINGS_COLUMN] == "1"


def test_a_subject_whose_templates_were_swept_is_shown_as_such(people, store):
    # The subject row survives a sweep that empties it — that is the register's
    # rule — and the panel must say so rather than show a person with nothing
    # behind the name as though the name were still matchable.
    store.register.sweep_expired(NOW + 40 * 86_400_000, RetentionPolicy(face_template_days=30.0))
    people.refresh_button.click()
    assert people.listed_subject_ids() == ["person-ali", "person-rana"]
    assert people.subject_row_texts("person-ali")[IDENTIFIERS_COLUMN].startswith("none")
    assert people.subject_row_texts("person-rana")[IDENTIFIERS_COLUMN] == "1 face template(s)", (
        "the pinned subject's template was reported swept"
    )


def test_the_refresh_button_reads_subjects_enrolled_since(people, store):
    store.register.enrol(
        subject_id="person-sami",
        display_name="Sami",
        identifier=stored_template("sami"),
        actor="operator:nadia",
        basis="consent",
    )
    assert people.listed_subject_ids() == ["person-ali", "person-rana"], "the panel read the store unasked"
    people.refresh_button.click()
    assert people.listed_subject_ids() == ["person-ali", "person-rana", "person-sami"]


def test_the_selected_subject_survives_a_refresh(people):
    assert people.select_subject("person-rana")
    people.refresh_button.click()
    assert people.selected_subject_id() == "person-rana"
    assert "Rana" in people.history_title.text()


# --------------------------------------------------------------- the history


def test_selecting_a_subject_shows_their_history_in_the_registers_order(people, store):
    people.subjects.topLevelItem(0).setSelected(True)  # Ali, through the widget
    rows = people.history_rows()
    history = store.register.history("person-ali")
    print("history rows:", rows)

    assert len(rows) == len(history) == 3
    assert [row[HISTORY_CAMERA_COLUMN] for row in rows] == [s.camera_id for s in history]
    assert [row[HISTORY_TRACK_COLUMN] for row in rows] == [f"#{s.track_id}" for s in history]
    assert [row[HISTORY_FIRST_COLUMN] for row in rows] == [utc(s.first_seen_millis) for s in history]
    assert [row[HISTORY_LAST_COLUMN] for row in rows] == [utc(s.last_seen_millis) for s in history]
    assert [row[HISTORY_CONFIDENCE_COLUMN] for row in rows] == ["match", "possible match", "match"]
    assert [row[HISTORY_SCORE_COLUMN] for row in rows] == ["0.88", "0.42", "0.91"]
    assert rows[0][HISTORY_CAMERA_COLUMN] == "cam-01", "oldest first: a trail has a direction"
    assert people.history_selections() == [
        Selection.track(s.camera_id, s.track_id) for s in history
    ]
    assert "3 sighting(s)" in people.history_title.text()


def test_a_declared_sighting_shows_no_score_because_there_is_none(vehicles):
    vehicles.select_subject("vehicle-van")
    (row,) = vehicles.history_rows()
    assert row[HISTORY_CONFIDENCE_COLUMN] == "declared"
    assert row[HISTORY_SCORE_COLUMN] == NOTHING


def test_a_subject_never_sighted_has_an_empty_history_that_says_so(people):
    people.select_subject("person-rana")
    assert people.history_rows() == []
    assert "never recognised" in people.history_title.text()


def test_selecting_a_subject_emits_nothing(people):
    # A subject is not a thing on the ground. Pushing one onto the bus would
    # clear the track every other panel is pointing at.
    seen: list = []
    people.selected.connect(seen.append)
    people.subjects.topLevelItem(0).setSelected(True)
    assert seen == []
    assert people.history_rows(), "the history was not filled either"


def test_clicking_a_sighting_emits_the_track_it_names(people):
    people.select_subject("person-ali")
    seen: list = []
    people.selected.connect(seen.append)
    people.history.topLevelItem(1).setSelected(True)
    print("clicked row 1, emitted", seen)
    assert seen == [Selection.track("cam-02", 21)]


def test_a_clicked_sighting_reaches_a_real_selection_bus(people):
    bus = SelectionBus()
    people.selected.connect(bus.select)
    people.select_subject("person-ali")
    people.history.topLevelItem(2).setSelected(True)
    assert bus.current == Selection.track("cam-03", 31)


def test_set_selection_highlights_the_sighting_without_emitting(people):
    people.select_subject("person-ali")
    seen: list = []
    people.selected.connect(seen.append)
    people.set_selection(Selection.track("cam-03", 31))
    assert seen == [], "the bus's own call was pushed back into the bus"
    assert people.selected_sighting() == Selection.track("cam-03", 31)
    people.set_selection(Selection.camera("cam-03"))
    assert people.selected_sighting() is None
    assert seen == []


# -------------------------------------------------------------- the enrol button


def test_the_enrol_button_needs_a_track_selection(people):
    assert not people.enrol_button.isEnabled()
    assert "select a track" in people.enrol_button.toolTip().lower()

    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    assert people.enrol_button.isEnabled()
    assert "cam-01" in people.enrol_button.toolTip()

    # A camera picked after the track is no longer the track being picked.
    people.set_selection(Selection.camera("cam-01"))
    assert not people.enrol_button.isEnabled()
    people.set_selection(None)
    assert not people.enrol_button.isEnabled()


def test_a_person_cannot_be_enrolled_from_a_track_with_no_templates_but_a_vehicle_can(
    people, vehicles
):
    # The node holds templates for cam-01 #7 only. A person panel must refuse
    # any other track — there is nothing to enrol — and say why; a vehicle
    # needs a plate typed in, not a template, so the same track is enough.
    people.set_selection(Selection.track("cam-02", 3))
    print("tooltip:", people.enrol_button.toolTip())
    assert not people.enrol_button.isEnabled()
    assert "no face templates" in people.enrol_button.toolTip().lower()

    vehicles.set_selection(Selection.track("cam-02", 3))
    assert vehicles.enrol_button.isEnabled()


def test_a_node_that_cannot_be_asked_for_templates_is_reported_not_raised(people, actions):
    def explode(camera_id, track_id):
        raise RuntimeError("the node is shutting down")

    actions.templates_for = explode
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    assert not people.enrol_button.isEnabled()
    assert "shutting down" in people.status_text()


# --------------------------------------------------------------- the dialog


def test_the_dialog_refuses_a_blank_name_and_keeps_the_typing(qt_app):
    dialog = EnrolDialog(SubjectKind.PERSON, templates_count=3)
    dialog.basis_box.setCurrentText("legal obligation")
    dialog.notes_field.setText("visiting engineer")
    dialog.accept()
    print("refusal:", dialog.refusal.text())
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "name is required" in dialog.refusal.text().lower()
    assert dialog.basis() == "legal obligation", "the refusal threw the basis away"
    assert dialog.notes() == "visiting engineer"

    dialog.name_field.setText("   ")
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted, "whitespace is not a name"

    dialog.name_field.setText("Ali Hassan")
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.name() == "Ali Hassan"
    assert dialog.plate() is None


def test_the_dialog_offers_the_lawful_bases_and_accepts_free_text(qt_app):
    dialog = EnrolDialog(SubjectKind.PERSON, templates_count=1)
    offered = [dialog.basis_box.itemText(i) for i in range(dialog.basis_box.count())]
    assert offered == ["consent", "employment", "contract", "legal obligation", "legitimate interest"]
    assert dialog.basis() == "consent", "the first choice is the default, never blank"
    dialog.basis_box.setCurrentText("site access policy §4")
    assert dialog.basis() == "site access policy §4"

    dialog.name_field.setText("Somebody")
    dialog.basis_box.setCurrentText("  ")
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted, "a blank basis was accepted"
    assert "basis is required" in dialog.refusal.text().lower()


def test_the_vehicle_dialog_carries_the_plate_hint_and_refuses_a_blank_plate(qt_app):
    dialog = EnrolDialog(SubjectKind.VEHICLE, plate_hint="B 7421")
    assert dialog.plate_field is not None
    assert dialog.plate() == "B 7421", "the recogniser's read did not reach the field"
    dialog.name_field.setText("Contractor van")
    dialog.plate_field.setText("")
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "plate is required" in dialog.refusal.text().lower()

    dialog.plate_field.setText("C 1234")  # corrected by hand: a hint is not a fact
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.plate() == "C 1234"


def test_the_person_dialog_says_how_many_templates_the_act_rests_on(qt_app):
    dialog = EnrolDialog(SubjectKind.PERSON, templates_count=12)
    captions = [w.text() for w in dialog.findChildren(type(dialog.refusal))]
    assert any("best of 12 face template(s)" in text for text in captions), captions


# ------------------------------------------------------------------ enrolling


def test_the_enrol_button_runs_the_dialog_and_passes_name_basis_and_notes_through(
    people, actions, store, monkeypatch
):
    """The whole hop: button, dialog, node, register, list, status.

    Not `panel.enrol()`: this asserts the connection from the button through
    the dialog to the node, which is the part that could be wired to nothing
    while every direct call still passed.
    """
    accept_dialog_with(monkeypatch, name="Ali Hassan", basis="employment", notes="night shift")
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    people.enrol_button.click()
    print("calls:", actions.calls, "status:", people.status_text())

    assert actions.calls == [
        ("enrol_person", "Ali Hassan", LIVE_CAMERA, LIVE_TRACK, "employment", "night shift")
    ]
    assert "Enrolled 'Ali Hassan'" in people.status_text()
    assert f"track #{LIVE_TRACK} on {LIVE_CAMERA}" in people.status_text()
    # The best-quality template was stored under the basis the operator chose,
    # from the track they picked — checked against the register, not the fake.
    (identifier,) = store.register.identifiers("person-ali-hassan")
    assert identifier.basis == "employment"
    assert identifier.quality == 0.93, "the best template of the three was not the one stored"
    assert (identifier.source_camera, identifier.source_track) == (LIVE_CAMERA, LIVE_TRACK)
    assert store.register.subject("person-ali-hassan").notes == "night shift"
    assert people.listed_subject_ids() == ["person-ali", "person-ali-hassan", "person-rana"]
    assert people.selected_subject_id() == "person-ali-hassan", "the enrolled subject was not selected"
    assert people.subject_row_texts("person-ali-hassan")[IDENTIFIERS_COLUMN] == "1 face template(s)"


def test_notes_left_empty_reach_the_node_as_none_not_an_empty_string(people, actions, monkeypatch):
    accept_dialog_with(monkeypatch, name="Newcomer", basis="consent", notes="")
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    people.enrol_button.click()
    assert actions.calls[-1] == ("enrol_person", "Newcomer", LIVE_CAMERA, LIVE_TRACK, "consent", None)
    assert people.listed_subject_ids() == ["person-ali", "person-newcomer", "person-rana"]


def test_a_cancelled_dialog_stores_nothing(people, actions, store, monkeypatch):
    cancel_dialog(monkeypatch)
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    before = len(store.register.identifiers("person-ali"))
    people.enrol_button.click()
    assert actions.calls == []
    assert len(store.register.identifiers("person-ali")) == before
    assert "cancelled" in people.status_text().lower()


def test_enrolling_a_vehicle_passes_the_plate_and_the_track_through(
    vehicles, actions, store, monkeypatch
):
    accept_dialog_with(
        monkeypatch, name="Site manager's car", basis="contract", notes="", plate="C 1234"
    )
    vehicles.set_plate_hint("C 1234")
    vehicles.set_selection(Selection.track("cam-02", 3))
    vehicles.enrol_button.click()
    print("calls:", actions.calls, "status:", vehicles.status_text())

    assert actions.calls == [
        ("enrol_vehicle", "Site manager's car", "C 1234", "contract", None, "cam-02", 3)
    ]
    found = store.register.find_plate("C-1234")
    assert found is not None and found.display_name == "Site manager's car"
    assert vehicles.listed_subject_ids() == ["vehicle-van", "vehicle-site-manager-s-car"]
    assert vehicles.subject_row_texts("vehicle-site-manager-s-car")[IDENTIFIERS_COLUMN] == "C 1234"


def test_the_plate_hint_pre_fills_the_dialog_the_button_opens(vehicles, monkeypatch):
    seen: list[str | None] = []

    def record_hint(dialog):
        seen.append(dialog.plate())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(EnrolDialog, "exec", record_hint)
    vehicles.set_plate_hint("B 9999")
    vehicles.set_selection(Selection.track("cam-02", 3))
    vehicles.enrol_button.click()
    assert seen == ["B 9999"]


def test_a_plate_already_somebody_elses_is_refused_and_the_refusal_is_shown(
    vehicles, store, monkeypatch
):
    # The register refuses to move a plate between subjects. That refusal has
    # to reach the status line as a refusal — the operator's decision to make.
    accept_dialog_with(monkeypatch, name="Another van", basis="contract", plate="B-7421")
    vehicles.set_selection(Selection.track("cam-02", 3))
    vehicles.enrol_button.click()
    print("status:", vehicles.status_text())
    assert vehicles.status_text().startswith("Not enrolled:")
    assert "already enrolled" in vehicles.status_text()
    assert vehicles.listed_subject_ids() == ["vehicle-van"]
    assert store.register.subject("vehicle-another-van") is None, "a refused enrolment left a subject behind"


# ----------------------------------------------------------------- forgetting


def test_forget_asks_naming_the_subject_and_the_counts_then_removes_the_row(
    people, store, monkeypatch
):
    asked: list[str] = []
    answer_question(monkeypatch, QMessageBox.StandardButton.Yes, asked)
    people.select_subject("person-ali")
    people.forget_button.click()
    print("asked:", asked, "status:", people.status_text())

    assert len(asked) == 1
    assert "'Ali Hassan'" in asked[0]
    assert "2 identifier(s)" in asked[0] and "3 sighting(s)" in asked[0]
    assert "never the name" in asked[0]

    assert people.listed_subject_ids() == ["person-rana"]
    assert store.register.subject("person-ali") is None
    assert store.register.history("person-ali") == ()
    # The register's own counts, on the line: what was actually deleted.
    assert "Forgot 'Ali Hassan'" in people.status_text()
    assert "2 identifier(s) deleted" in people.status_text()
    assert "3 sighting(s) removed" in people.status_text()
    assert people.history_rows() == [], "the forgotten person's history is still on screen"


def test_answering_no_to_forget_leaves_the_subject_enrolled(people, store, monkeypatch):
    asked: list[str] = []
    answer_question(monkeypatch, QMessageBox.StandardButton.No, asked)
    people.select_subject("person-ali")
    people.forget_button.click()
    assert len(asked) == 1
    assert people.listed_subject_ids() == ["person-ali", "person-rana"]
    assert store.register.subject("person-ali") is not None
    assert "still enrolled" in people.status_text()


def test_forgetting_somebody_already_forgotten_says_so_rather_than_failing(people, store):
    store.register.forget("person-rana")
    forgotten = people.forget("person-rana")
    assert forgotten is not None and forgotten.found is False
    assert "Nothing to forget" in people.status_text()
    assert people.listed_subject_ids() == ["person-ali"]


# -------------------------------------------------------------------- pinning


def test_pin_and_unpin_go_through_the_node_and_the_button_says_which_it_is(
    people, actions, store
):
    people.select_subject("person-ali")
    assert people.pin_button.text() == "Pin"
    people.pin_button.click()
    assert actions.calls[-1] == ("pin_subject", "person-ali", True)
    assert store.register.subject("person-ali").pinned is True
    assert people.subject_row_texts("person-ali")[PINNED_COLUMN] == "pinned"
    assert people.pin_button.text() == "Unpin"
    assert "Pinned 'Ali Hassan'" in people.status_text()

    people.pin_button.click()
    assert actions.calls[-1] == ("pin_subject", "person-ali", False)
    assert store.register.subject("person-ali").pinned is False
    assert people.pin_button.text() == "Pin"

    people.select_subject("person-rana")
    assert people.pin_button.text() == "Unpin", "the button did not read the selected subject's pin"


# ------------------------------------------------------------------- the lock


def test_locking_disables_the_three_buttons_but_the_list_and_history_stay_readable(
    people, actions, store
):
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    people.select_subject("person-ali")
    assert people.enrol_button.isEnabled()
    assert people.forget_button.isEnabled()
    assert people.pin_button.isEnabled()

    people.set_editable(False)
    for button in (people.enrol_button, people.forget_button, people.pin_button):
        assert not button.isEnabled()
        assert "configure is locked" in button.toolTip().lower(), button.toolTip()
    assert people.listed_subject_ids() == ["person-ali", "person-rana"]
    assert len(people.history_rows()) == 3

    # The direct methods refuse too: a button is not the only way in.
    before = len(store.register.identifiers("person-ali"))
    assert people.enrol("Ali Hassan", "employment") is None
    assert "locked" in people.status_text().lower()
    assert people.forget("person-ali") is None
    assert people.set_pinned("person-ali", True) is False
    assert actions.calls == [], "a locked panel reached the node"
    assert len(store.register.identifiers("person-ali")) == before
    assert store.register.subject("person-ali").pinned is False

    people.set_editable(True)
    assert people.enrol_button.isEnabled()


def test_a_fresh_panel_is_locked_until_the_console_unlocks_it(qt_app, actions):
    # Off by default, like dragging a camera on the map: a console left in a
    # control room is left in whatever state the last person walked away from.
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(actions)
    widget.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    widget.select_subject("person-ali")
    assert not widget.enrol_button.isEnabled()
    assert not widget.forget_button.isEnabled()
    assert not widget.pin_button.isEnabled()


# ------------------------------------------------------------------- refusals


def test_a_refusal_from_the_node_is_shown_on_the_status_line_not_raised(
    qt_app, store, monkeypatch
):
    """A refusal is worded as a refusal, and nothing leaves the slot.

    `RegistryError` and `ValueError` are the node's plain-message refusals —
    faces off, no templates, a blank name — and they read "Not enrolled: …".
    Anything else is a failure and reads as one. The distinction is on the
    prefix, so a panel that lumped refusals in with failures fails here.
    """
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(RefusingActions(store))
    widget.set_editable(True)
    widget.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    widget.select_subject("person-ali")

    accept_dialog_with(monkeypatch, name="Ali Hassan", basis="employment")
    widget.enrol_button.click()  # through the slot: an exception here would be retained
    print("enrol:", widget.status_text())
    assert widget.status_text().startswith("Not enrolled: faces are off")
    assert isinstance(widget.last_failure, RegistryError)

    answer_question(monkeypatch, QMessageBox.StandardButton.Yes, [])
    widget.forget_button.click()
    print("forget:", widget.status_text())
    assert widget.status_text().startswith("Not forgotten: the register is read-only")
    assert widget.listed_subject_ids() == ["person-ali", "person-rana"]

    widget.pin_button.click()
    print("pin:", widget.status_text())
    assert widget.status_text().startswith("Pin not changed:")
    assert store.register.subject("person-ali").pinned is False


def test_a_blank_name_refused_by_the_node_reads_as_a_refusal(people, actions):
    # The dialog refuses first; the node refuses too, and a caller of the
    # direct method must see the node's sentence rather than a traceback.
    people.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    assert people.enrol("", "consent") is None
    assert people.status_text() == "Not enrolled: a name is required"
    assert isinstance(people.last_failure, ValueError)


def test_a_register_whose_store_has_closed_is_reported_rather_than_raised(qt_app):
    # The real failure, not a stand-in for it: `Store.register` raises after
    # `close()`, naming the store, and a panel holding a node whose store has
    # gone is the case that docstring was written for. The node still has the
    # right shape — `store` is there, it just cannot be read — so this must
    # reach the status line as a read failure, not as a refused node.
    closed = Store(":memory:")
    closed.close()
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(FakeActions(closed))
    print("status:", widget.status_text())
    assert widget.listed_subject_ids() == []
    assert widget.status_text().startswith("The register could not be read:")
    assert "is closed; its register closed with it" in widget.status_text()
    assert isinstance(widget.last_failure, StoreError)
    widget.refresh_button.click()  # and again through the slot
    assert "could not be read" in widget.status_text()


# ---------------------------------------------------------- the node's surface


def test_the_surface_the_panel_checks_is_the_protocol_it_declares():
    # `IDENTITY_SURFACE` is spelled out so the check cannot pass vacuously on
    # an interpreter without `get_protocol_members`; this is what keeps the
    # spelling honest. A member added to the protocol and not the tuple — or
    # the other way round — is a member the panel would never check for.
    assert set(IDENTITY_SURFACE) == typing.get_protocol_members(IdentityActions)
    assert len(set(IDENTITY_SURFACE)) == len(IDENTITY_SURFACE)


def test_a_node_shaped_the_old_way_is_refused_by_name_not_reported_on_every_refresh(
    qt_app, store
):
    # The shape the first version of this panel demanded: a `register`
    # attribute and no `store`. The real node has never had it. Such an object
    # must be refused at `set_actions`, with the missing member named, rather
    # than accepted and then reported as "the register could not be read" on
    # every refresh — which is what the fake-only tests let through.
    old_shape = FakeActions(store)
    old_shape.register = store.register
    del old_shape.store
    assert missing_from(old_shape) == ("store",)

    # Connected to a good node first, so a refusal that kept the old node
    # would be visible: the list must empty, not stay as it was.
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(FakeActions(store))
    assert widget.listed_subject_ids() == ["person-ali", "person-rana"]
    widget.set_actions(old_shape)
    print("status:", widget.status_text(), "| capability:", widget.capability_text())
    assert widget.status_text().startswith("That node cannot be used: it has no 'store'")
    assert isinstance(widget.last_failure, TypeError)
    assert "no node is connected" in widget.capability_text().lower()
    assert widget.listed_subject_ids() == []
    assert not widget.enrol_button.isEnabled()

    # Refresh does not quietly reconnect: the panel is not holding the object.
    widget.refresh_button.click()
    assert widget.listed_subject_ids() == []
    assert "could not be read" not in widget.status_text()

    # The check reads the shape without touching it. A node whose `store` is
    # a property that raises — one mid-shutdown, say — has the right shape;
    # `missing_from` must not evaluate the property (a `getattr` would, and
    # would raise out of the check), and the panel must accept the node and
    # report the read failing in the guarded place, with the node's reason.
    class Shutting(FakeActions):
        @property
        def store(self):
            raise RuntimeError("the node is shutting down")

    shutting = Shutting.__new__(Shutting)  # no __init__: it would assign over the property
    shutting.identity_status = "off"
    assert missing_from(shutting) == ()
    widget.set_actions(shutting)
    print("shutting:", widget.status_text())
    assert widget.status_text() == "The register could not be read: the node is shutting down"
    assert "Identity: off." in widget.capability_text()


def test_the_real_node_satisfies_the_surface_and_the_panel_reads_and_writes_through_it(
    qt_app, tmp_path: Path, monkeypatch
):
    """The wire to the real node, end to end, with nothing stood in.

    `FakeActions` proves what the panel passes through; it cannot prove the
    node has the members the panel calls, and once it did not. So: a real
    `Node` over a temporary database, no camera started, identity off, no
    model anywhere (the models directory is an empty temporary one, so "no
    models" is a fact about this test and not about the checkout). The
    vehicles panel enrols, pins and forgets through the node — the acts that
    do not need a face engine — and the audit rows the node writes are read
    back, by id and never by name. The people panel asks the real
    `templates_for` for a track on a camera the node has, and the real
    `identity_status` is the line at the top. Then the node closes, and the
    panel reports the closed store rather than raising from the slot.
    """
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("SENTINEL_MODELS_DIR", str(models))
    accept_dialog_with(
        monkeypatch, name="Contractor van", basis="site access list", plate="B 7421"
    )
    asked: list[str] = []
    answer_question(monkeypatch, QMessageBox.StandardButton.Yes, asked)

    with Node(tmp_path / "console.db", node_id="console-test", actor="operator:test") as node:
        node.add_camera(tmp_path / "gate.mp4", camera_id="gate")  # added, never opened
        assert missing_from(node) == (), "the panel's surface is not the node's"
        assert isinstance(node, IdentityActions)

        people = RegisterPanel(SubjectKind.PERSON)
        people.set_actions(node)
        assert people.last_failure is None, people.last_failure
        assert people.status_text() == "", people.status_text()
        assert node.identity_status in people.capability_text()
        assert "Identity: off." in people.capability_text()
        assert people.listed_subject_ids() == []
        people.set_editable(True)
        people.set_selection(Selection.track("gate", 3))
        # The real `templates_for`: no engine, no templates, and the button
        # says so; a person cannot be enrolled through this node as it stands.
        assert not people.enrol_button.isEnabled()
        assert "no face templates" in people.enrol_button.toolTip().lower()
        assert people.last_failure is None, people.last_failure

        vehicles = RegisterPanel(SubjectKind.VEHICLE)
        vehicles.set_actions(node)
        vehicles.set_editable(True)
        vehicles.set_selection(Selection.track("gate", 3))
        assert vehicles.enrol_button.isEnabled()
        vehicles.enrol_button.click()
        print("enrol:", vehicles.status_text())
        assert vehicles.status_text().startswith("Enrolled 'Contractor van' as vehicle vehicle-")
        assert "from track #3 on gate" in vehicles.status_text()
        (subject_id,) = vehicles.listed_subject_ids()
        assert vehicles.selected_subject_id() == subject_id
        assert vehicles.subject_row_texts(subject_id)[IDENTIFIERS_COLUMN] == "B 7421"
        register = node.store.register
        found = register.find_plate("B-7421")
        assert found is not None and found.id == subject_id
        (identifier,) = register.identifiers(subject_id)
        assert identifier.basis == "site access list"
        assert (identifier.source_camera, identifier.source_track) == ("gate", 3)

        vehicles.pin_button.click()
        assert register.subject(subject_id).pinned is True
        assert vehicles.pin_button.text() == "Unpin"

        vehicles.forget_button.click()
        print("asked:", asked, "| forget:", vehicles.status_text())
        assert len(asked) == 1 and "'Contractor van'" in asked[0]
        assert vehicles.status_text().startswith("Forgot 'Contractor van'")
        assert register.subject(subject_id) is None
        assert vehicles.listed_subject_ids() == []

        # The node's audit rows, one per act, by id and never by name or plate.
        actions = [
            (row["action"], row["detail"])
            for row in node.store.audit_trail(limit=50)
            if row["action"] in ("vehicle.enrolled", "subject.pinned", "subject.forgotten")
        ]
        print("audit:", actions)
        assert [action for action, _ in actions] == [
            "subject.forgotten", "subject.pinned", "vehicle.enrolled"
        ], "newest first: forget, pin, enrol"
        for _action, detail in actions:
            assert "Contractor van" not in detail
            assert "7421" not in detail
        assert "from camera gate, track 3" in actions[-1][1]

    # The node has closed; its store says so, and the panel repeats it.
    vehicles.refresh_button.click()
    print("after close:", vehicles.status_text())
    assert vehicles.status_text().startswith("The register could not be read:")
    assert "is closed; its register closed with it" in vehicles.status_text()


# ------------------------------------------------------------------ the clock


def test_times_are_shown_in_the_sites_clock_and_the_caption_says_which(qt_app, actions):
    # A movement history read against the wrong zone puts somebody at the gate
    # an hour before they were there. The site's clock is used and named.
    actions.site_record = SimpleNamespace(
        timezone="Asia/Beirut", clock=lambda: timezone(timedelta(hours=3))
    )
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(actions)
    widget.select_subject("person-ali")
    first = widget.history_rows()[0][HISTORY_FIRST_COLUMN]
    print("first seen:", first, "caption:", widget.clock_caption.text())
    shifted = datetime.fromtimestamp((NOW + MINUTE) / 1000, timezone(timedelta(hours=3)))
    assert first == shifted.strftime("%Y-%m-%d %H:%M:%S")
    assert first != utc(NOW + MINUTE)
    assert "Asia/Beirut" in widget.clock_caption.text()


def test_a_refresh_reads_the_sites_clock_once_for_the_rows_and_the_history(qt_app, store):
    # Each read of the site is a row from the store. A refresh was measured
    # reading it twice — once for the subject rows, once again inside the
    # history — and the second read is the one this pins down.
    class CountingActions(FakeActions):
        def __init__(self, store):
            super().__init__(store)
            self.site_reads = 0

        def site(self):
            self.site_reads += 1
            return self.site_record

    counting = CountingActions(store)
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(counting)
    widget.select_subject("person-ali")
    assert len(widget.history_rows()) == 3, "the history must be filled for this to mean anything"
    counting.site_reads = 0
    widget.refresh_button.click()
    print("site reads during one refresh:", counting.site_reads)
    assert len(widget.history_rows()) == 3
    assert counting.site_reads == 1


def test_a_site_clock_that_cannot_be_resolved_falls_back_to_utc_and_says_so(qt_app, actions):
    actions.site_record = Site(
        id=DEFAULT_SITE_ID, name="Site", origin=LatLon(0, 0), timezone="Mars/Olympus_Mons"
    )
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(actions)
    widget.select_subject("person-ali")
    assert widget.history_rows()[0][HISTORY_FIRST_COLUMN] == utc(NOW + MINUTE)
    caption = widget.clock_caption.text()
    print("caption:", caption)
    assert caption.startswith("Times are UTC")
    assert "could not be read" in caption


# --------------------------------------------------------------- lifetime


def test_the_panel_is_freed_when_its_last_reference_goes(qt_app, actions, store, monkeypatch):
    # A reference cycle holding a QWidget means the widget is destroyed at
    # interpreter shutdown, after PySide has torn the QApplication down, which
    # corrupts the heap and kills the process with 0xC0000374 at exit — after
    # every test has passed. A lambda closing over `self` in a signal connection
    # is how that happens, so every connection inside the panel is a bound
    # method. Every path is walked first: the dialog, the confirmation, a pin.
    accept_dialog_with(monkeypatch, name="Ali Hassan", basis="employment")
    answer_question(monkeypatch, QMessageBox.StandardButton.Yes, [])
    widget = RegisterPanel(SubjectKind.PERSON)
    widget.set_actions(actions)
    widget.set_editable(True)
    widget.set_selection(Selection.track(LIVE_CAMERA, LIVE_TRACK))
    widget.select_subject("person-ali")
    widget.enrol_button.click()
    widget.pin_button.click()
    widget.forget_button.click()
    widget.refresh_button.click()
    qt_app.processEvents()
    ref = weakref.ref(widget)
    del widget
    gc.collect()
    survivor = ref()
    if survivor is not None:
        holders = sorted(
            {type(r).__name__ for r in gc.get_referrers(survivor)} - {"frame", "list"}
        )
        del survivor
        raise AssertionError(f"RegisterPanel outlived its last reference (held by {holders}).")
