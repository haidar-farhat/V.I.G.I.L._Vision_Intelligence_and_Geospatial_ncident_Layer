"""Which of the model's classes the site watches.

A security console that tracks every one of COCO's eighty classes tracks jars,
phones and cushions with the same green box as a person. This is where an
operator says what matters on their site. The list is the model's own
vocabulary — nothing is offered that the detector cannot say — and the two
buttons are the two honest presets: the security default (people and the
vehicles they arrive in) and everything.
"""
from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Watched classes")
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
