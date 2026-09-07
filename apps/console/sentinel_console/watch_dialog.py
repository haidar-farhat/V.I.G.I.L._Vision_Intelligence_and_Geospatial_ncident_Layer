"""Which of the model's classes the site watches, and how sure the model must be.

A security console that tracks every one of COCO's eighty classes tracks jars,
phones and cushions with the same green box as a person. This is where an
operator says what matters on their site. The list is the model's own
vocabulary — nothing is offered that the detector cannot say — and the two
buttons are the two honest presets: the security default (people and the
vehicles they arrive in) and everything.

The confidence floor sits in the same dialog because it answers the same
complaint from the other side: a class that *is* watched can still be claimed
on a weak score — a coat on a chair as a person at 0.4 — and the floor is what
drops that before it becomes a track. The two are one decision about what the
site is prepared to be told.
"""
from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class WatchedClassesDialog(QDialog):
    def __init__(
        self,
        vocabulary: Iterable[str],
        watched: Iterable[str],
        *,
        defaults: Iterable[str] = (),
        confidence: float | None = None,
        parent: QWidget | None = None,
    ):
        """
        ``confidence`` is the floor a detection must clear to be tracked at
        all. ``None`` leaves it out of the dialog entirely, for a caller that
        has no such setting; a number puts a spin box in, seeded with it.
        """
        super().__init__(parent)
        self.setWindowTitle("Watched classes and confidence")
        self.setModal(True)
        self._defaults = frozenset(defaults)
        watched = frozenset(watched)

        layout = QVBoxLayout(self)
        caption = QLabel(
            "Only the classes ticked here are tracked at all. The rest are "
            "dropped at the detector, before they can become a track, a zone "
            "event or an incident. Takes effect at the next Start."
        )
        caption.setWordWrap(True)
        layout.addWidget(caption)

        self.list = QListWidget()
        for label in sorted(set(vocabulary)):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if label in watched else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        layout.addWidget(self.list, 1)

        presets = QHBoxLayout()
        self.default_button = QPushButton("Security default")
        self.default_button.setToolTip(
            "People and the vehicles they arrive in: " + ", ".join(sorted(self._defaults))
        )
        self.default_button.clicked.connect(self._use_defaults)
        presets.addWidget(self.default_button)
        self.all_button = QPushButton("Everything")
        self.all_button.clicked.connect(self._use_all)
        presets.addWidget(self.all_button)
        presets.addStretch(1)
        layout.addLayout(presets)

        #: The floor, or ``None`` when the caller has no such setting.
        self.confidence_spin: QDoubleSpinBox | None = None
        if confidence is not None:
            floor = QHBoxLayout()
            floor.addWidget(QLabel("Minimum confidence"))
            self.confidence_spin = QDoubleSpinBox()
            self.confidence_spin.setRange(0.10, 0.95)
            self.confidence_spin.setSingleStep(0.05)
            self.confidence_spin.setDecimals(2)
            self.confidence_spin.setValue(float(confidence))
            self.confidence_spin.setToolTip(
                "A detection scoring below this is dropped at the detector, "
                "before it can become a track. Higher means fewer false objects "
                "and a later first sighting of a real one. 0.50 is the security "
                "default; the model's own convention is 0.35. Motion detection "
                "ignores it — its confidence is how much of a box moved."
            )
            floor.addWidget(self.confidence_spin)
            floor.addStretch(1)
            layout.addLayout(floor)

        self.count_label = QLabel("")
        layout.addWidget(self.count_label)
        self.list.itemChanged.connect(self._recount)
        self._recount()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.buttons = buttons
        layout.addWidget(buttons)

    # ----------------------------------------------------------------- state
    def chosen(self) -> frozenset[str]:
        """The ticked classes. Empty means the operator ticked nothing, which
        the caller refuses rather than reading as 'watch nothing'."""
        return frozenset(
            self.list.item(i).text()
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        )

    def confidence(self) -> float | None:
        """The floor as set, or ``None`` if this dialog was built without one."""
        if self.confidence_spin is None:
            return None
        return round(float(self.confidence_spin.value()), 2)

    def _set_all(self, labels: frozenset[str] | None) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            ticked = labels is None or item.text() in labels
            item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)

    def _use_defaults(self) -> None:
        self._set_all(self._defaults)

    def _use_all(self) -> None:
        self._set_all(None)

    def _recount(self, *_) -> None:
        chosen = len(self.chosen())
        total = self.list.count()
        self.count_label.setText(f"{chosen} of {total} classes watched")
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok) if hasattr(self, "buttons") else None
        if ok is not None:
            ok.setEnabled(chosen > 0)
            ok.setToolTip("" if chosen else "Tick at least one class; watching nothing is never what anybody meant.")
