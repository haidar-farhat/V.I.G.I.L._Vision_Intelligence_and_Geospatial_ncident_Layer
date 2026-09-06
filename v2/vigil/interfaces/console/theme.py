"""One palette, one stylesheet. Colours mean things and are named for them."""

from __future__ import annotations

from PySide6.QtGui import QColor

BACKGROUND = QColor(18, 20, 24)
PANEL = QColor(26, 29, 35)
PANEL_RAISED = QColor(34, 38, 46)
BORDER = QColor(52, 58, 70)

TEXT = QColor(226, 232, 240)
TEXT_MUTED = QColor(148, 160, 178)
TEXT_FAINT = QColor(100, 112, 130)

#: A source is delivering frames.
LIVE = QColor(74, 222, 128)
#: A source is late, reconnecting, or a state is provisional.
STALE = QColor(250, 204, 21)
#: A source has failed, or something needs a person now.
FAULT = QColor(248, 113, 113)
#: Selected, and the accent for anything the operator is acting on.
ACCENT = QColor(96, 165, 250)

#: Track colours, cycled by track id. Distinguishable on this background and
#: from each other; none of them is any of the state colours above, so a box
#: can never be mistaken for a status.
TRACKS = (
    QColor(129, 212, 250), QColor(244, 143, 177), QColor(255, 213, 79), QColor(174, 213, 129),
    QColor(179, 157, 219), QColor(255, 171, 145), QColor(128, 222, 234), QColor(240, 244, 195),
)


def track_colour(track_id: int) -> QColor:
    return TRACKS[track_id % len(TRACKS)]


def severity_colour(severity: str) -> QColor:
    return {"CRITICAL": FAULT, "HIGH": FAULT, "MEDIUM": STALE, "LOW": ACCENT}.get(str(severity), TEXT_MUTED)


def stylesheet() -> str:
    return f"""
    QWidget {{ background: {BACKGROUND.name()}; color: {TEXT.name()}; font-size: 13px; }}
    QFrame#Panel {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; border-radius: 4px; }}
    QLabel#PanelTitle {{ color: {TEXT_FAINT.name()}; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 6px 8px 2px 8px; }}
    QLabel#Caption {{ color: {TEXT_MUTED.name()}; font-size: 11px; }}
    QPushButton {{ background: {PANEL_RAISED.name()}; border: 1px solid {BORDER.name()}; border-radius: 3px; padding: 5px 12px; }}
    QPushButton:hover:enabled {{ border-color: {ACCENT.name()}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT.name()}; background: {PANEL.name()}; }}
    QPushButton:checked {{ background: {ACCENT.name()}; color: {BACKGROUND.name()}; font-weight: 600; }}
    QTreeWidget, QTableWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {PANEL.name()}; border: 1px solid {BORDER.name()}; border-radius: 3px; selection-background-color: {ACCENT.name()};
        selection-color: {BACKGROUND.name()}; }}
    QHeaderView::section {{ background: {PANEL_RAISED.name()}; color: {TEXT_MUTED.name()}; border: 0; border-bottom: 1px solid {BORDER.name()}; padding: 4px; }}
    QTabBar::tab {{ background: {PANEL.name()}; padding: 5px 12px; border: 1px solid {BORDER.name()}; border-bottom: 0; }}
    QTabBar::tab:selected {{ background: {PANEL_RAISED.name()}; color: {TEXT.name()}; }}
    QStatusBar {{ background: {PANEL.name()}; border-top: 1px solid {BORDER.name()}; }}
    QSplitter::handle {{ background: {BORDER.name()}; }}
    QToolTip {{ background: {PANEL_RAISED.name()}; color: {TEXT.name()}; border: 1px solid {ACCENT.name()}; }}
    """
