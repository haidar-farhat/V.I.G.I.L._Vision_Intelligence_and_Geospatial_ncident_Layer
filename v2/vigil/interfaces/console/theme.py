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
    /* The one control an operator presses most, and the only one that is
       loud. Everything else on screen is grey on grey on purpose: a window
       where three things shout is a window where nothing does. */
    QPushButton#Primary {{ background: {LIVE.name()}; color: {BACKGROUND.name()}; font-weight: 700;
        border: 1px solid {LIVE.name()}; padding: 6px 20px; }}
    QPushButton#Primary:hover:enabled {{ background: {LIVE.lighter(112).name()}; }}
    QPushButton#Primary:disabled {{ background: {PANEL.name()}; color: {TEXT_FAINT.name()};
        border-color: {BORDER.name()}; font-weight: 600; }}
    QPushButton#Stop {{ background: {FAULT.name()}; color: {BACKGROUND.name()}; font-weight: 700;
        border: 1px solid {FAULT.name()}; padding: 6px 20px; }}
    QPushButton#Stop:hover:enabled {{ background: {FAULT.lighter(112).name()}; }}
    QPushButton#Stop:disabled {{ background: {PANEL.name()}; color: {TEXT_FAINT.name()};
        border-color: {BORDER.name()}; font-weight: 600; }}
    /* The verbs that sit under the thing they act on. Quieter than a normal
       button, because a panel with six loud buttons under it reads as six
       decisions rather than as a list with some things you can do to it. */
    QWidget#ActionBar {{ background: transparent; }}
    QWidget#ActionBar QPushButton {{ background: transparent; border: 1px solid transparent;
        padding: 4px 9px; color: {TEXT_MUTED.name()}; }}
    QWidget#ActionBar QPushButton:hover:enabled {{ background: {PANEL_RAISED.name()};
        border-color: {BORDER.name()}; color: {TEXT.name()}; }}
    QWidget#ActionBar QPushButton:disabled {{ color: {TEXT_FAINT.name()}; background: transparent; }}
    QWidget#ActionBar QPushButton:checked {{ background: {ACCENT.name()}; color: {BACKGROUND.name()}; }}
    QMenuBar {{ background: {BACKGROUND.name()}; color: {TEXT_MUTED.name()}; }}
    QMenuBar::item:selected {{ background: {PANEL_RAISED.name()}; color: {TEXT.name()}; }}
    QMenu {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; }}
    QMenu::item:selected {{ background: {ACCENT.name()}; color: {BACKGROUND.name()}; }}
    QMenu::item:disabled {{ color: {TEXT_FAINT.name()}; }}
    QMenu::separator {{ height: 1px; background: {BORDER.name()}; margin: 4px 8px; }}
    QLabel#PanelDetail {{ color: {TEXT_MUTED.name()}; font-size: 11px; padding: 6px 8px 2px 8px; }}
    QTreeWidget, QTableWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {PANEL.name()}; border: 1px solid {BORDER.name()}; border-radius: 3px; }}
    /* A muted wash, and deliberately no `selection-color`: a row carries the
       state colours — LIVE green, FAULTED red — and a selection that repaints
       the text kills exactly the information the row exists to give. Green on
       a saturated blue row was unreadable in the first two-camera photograph. */
    QTreeWidget::item:selected, QTableWidget::item:selected {{
        background: rgba(96, 165, 250, 64); border-left: 2px solid {ACCENT.name()}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        selection-background-color: {ACCENT.name()}; selection-color: {BACKGROUND.name()}; }}
    QHeaderView::section {{ background: {PANEL_RAISED.name()}; color: {TEXT_MUTED.name()}; border: 0; border-bottom: 1px solid {BORDER.name()}; padding: 4px; }}
    QTabBar::tab {{ background: {PANEL.name()}; padding: 5px 12px; border: 1px solid {BORDER.name()}; border-bottom: 0; }}
    QTabBar::tab:selected {{ background: {PANEL_RAISED.name()}; color: {TEXT.name()}; }}
    QStatusBar {{ background: {PANEL.name()}; border-top: 1px solid {BORDER.name()}; }}
    QSplitter::handle {{ background: {BORDER.name()}; }}
    QToolTip {{ background: {PANEL_RAISED.name()}; color: {TEXT.name()}; border: 1px solid {ACCENT.name()}; }}
    """
