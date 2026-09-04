"""What the operator is looking at, in one place.

Four panels show the same site from four angles: the camera wall shows a box
around a person, the plan view shows a disc where that person is standing, the
track table shows the row that says how fast they are moving, and the incident
list says why any of it matters. Until now those were four unrelated pictures.
An operator who saw something worth attention in one of them had to find it
again by eye in the others — matching a track id in a table against a small
green number on a video pane, while it moved.

So there is exactly one selected thing, and every panel agrees about it. Click
the disc on the map and the box lights up on the video and the row lights up in
the table. Click the incident and the object it is about is selected. Escape
clears it everywhere.

**A track is keyed by camera *and* id.** Track ids are only unique within one
camera: `#3` on the gate and `#3` on the yard are different people, and a bus
that keyed on the number alone would light up the wrong box on the wall the
first time two cameras ran at once.

Nothing here reaches the node or the store. A selection is a fact about a
screen, not about the site, and it is deliberately not persisted: an operator
coming back to a console should be shown what is happening, not where somebody
last clicked.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

#: The kinds of thing that can be selected. Strings rather than an enum so a
#: panel can compare without importing this module's namespace into its own.
CAMERA = "camera"
ZONE = "zone"
TRACK = "track"
INCIDENT = "incident"


@dataclass(frozen=True, slots=True)
class Selection:
    """One selected thing. Frozen, so a panel cannot edit what it was handed."""

    kind: str
    camera_id: str | None = None
    zone_id: str | None = None
    track_id: int | None = None
    incident_id: str | None = None

    @classmethod
    def camera(cls, camera_id: str) -> "Selection":
        return cls(kind=CAMERA, camera_id=camera_id)

    @classmethod
    def zone(cls, zone_id: str) -> "Selection":
        return cls(kind=ZONE, zone_id=zone_id)

    @classmethod
    def track(cls, camera_id: str, track_id: int) -> "Selection":
        """A track, keyed by the camera it belongs to as well as its id."""
        return cls(kind=TRACK, camera_id=camera_id, track_id=track_id)

    @classmethod
    def incident(cls, incident_id: str) -> "Selection":
        return cls(kind=INCIDENT, incident_id=incident_id)

    def is_track(self, camera_id: str, track_id: int) -> bool:
        """Whether this is that track. Both halves of the key, always."""
        return (
            self.kind == TRACK
            and self.camera_id == camera_id
            and self.track_id == track_id
        )

    def describe(self) -> str:
        """For the status bar. Says which camera a track belongs to, because
        "#3" alone is ambiguous the moment a second camera is running."""
        if self.kind == TRACK:
            return f"track #{self.track_id} on {self.camera_id}"
        if self.kind == CAMERA:
            return f"camera {self.camera_id}"
        if self.kind == ZONE:
            return f"zone {self.zone_id}"
        return f"incident {self.incident_id}"


class SelectionBus(QObject):
    """Holds the current selection and tells every panel when it changes.

    A bus rather than each panel wiring to each other panel: with four panels
    that would be twelve connections, and the thirteenth — added when a fifth
    panel arrives — is the one somebody forgets.
    """

    #: The new selection, or ``None`` when it was cleared.
    changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._current: Selection | None = None

    @property
    def current(self) -> Selection | None:
        return self._current

    def select(self, selection: Selection | None) -> None:
        """Set the selection. Silent when nothing actually changed.

        The silence matters: panels repaint on this signal, and a click that
        re-selects what was already selected must not cost a repaint of every
        panel — nor, worse, scroll a table back to a row the operator had
        deliberately scrolled away from.
        """
        if selection == self._current:
            return
        self._current = selection
        self.changed.emit(selection)

    def clear(self) -> None:
        self.select(None)

    def is_selected(self, selection: Selection) -> bool:
        return self._current == selection
