"""Colours and metrics for the console.

Kept in one place because the palette carries meaning rather than taste. An
operator watching a wall of cameras at three in the morning reads colour before
they read text, so a colour used for two different things is a defect.

The scheme is dark because control rooms are dark, and because a bright interface
behind a monitor wall destroys night vision.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# Surfaces, from furthest back to nearest.
BACKGROUND = QColor(18, 20, 24)
PANEL = QColor(26, 29, 35)
PANEL_RAISED = QColor(34, 38, 46)
BORDER = QColor(52, 58, 70)

TEXT = QColor(226, 232, 240)
TEXT_MUTED = QColor(148, 160, 178)
TEXT_FAINT = QColor(100, 112, 130)

# State. Reserved meanings — do not reuse for decoration.
LIVE = QColor(74, 222, 128)          # a source is delivering frames
STALE = QColor(250, 204, 21)         # a source is late or reconnecting
FAULT = QColor(248, 113, 113)        # a source has failed
IDLE = QColor(100, 112, 130)         # nothing is running

# Evidence.
DETECTION = QColor(96, 165, 250)     # this frame's raw detections
TRACK = QColor(74, 222, 128)         # a confirmed, persisting object
TRACK_COASTING = QColor(250, 204, 21)  # held open with no detection this frame
UNCERTAINTY = QColor(96, 165, 250, 40)  # the 1-sigma position disc

# Zones. Deliberately distinct from both evidence and footprint colours: a zone
# boundary is a rule an operator wrote, not something the system observed.
ZONE_FILL = QColor(248, 113, 113, 26)
ZONE_EDGE = QColor(248, 113, 113, 150)

#: One colour per kind of zone, keyed by the kind's value so this module stays
#: free of engine imports. A restricted area and an exclusion zone drawn alike
#: would let an operator read "ignore this" as "nobody should be here".
ZONE_COLOURS = {
    "RESTRICTED": QColor(248, 113, 113),   # nobody should be here
    "PERIMETER": QColor(251, 146, 60),     # the site boundary
    "ENTRY": QColor(96, 165, 250),         # a door, gate or lane
    "EXCLUSION": QColor(148, 160, 178),    # deliberately ignored
    "INTEREST": QColor(74, 222, 128),      # worth recording, nothing implied
}


def zone_colour(kind) -> QColor:
    """The colour for a zone kind (an enum or its value). Unknown kinds fall
    back to the restricted colour, which is the alarming one on purpose."""
    return ZONE_COLOURS.get(getattr(kind, "value", str(kind)), QColor(ZONE_EDGE))

# The map.
CAMERA = QColor(226, 232, 240)
FOOTPRINT = QColor(96, 165, 250, 28)
FOOTPRINT_EDGE = QColor(96, 165, 250, 110)
#: Alphas of the footprint blue for the ≤ 5 / ≤ 2 / ≤ 1 / ≤ 0.5 m position-error
#: bands, loosest first. The bands nest and are painted widest first, so the
#: alphas accumulate and the ground the camera knows best ends up the most
#: strongly coloured — brightest where its answer can be trusted, fading to the
#: bare footprint where the error passes five metres.
SIGMA_BANDS = (22, 36, 52, 70)
#: A zone's part that no camera can see, hatched in the fault colour.
OUTSIDE_HATCH = QColor(248, 113, 113, 60)
WARNING = STALE
GRID = QColor(44, 50, 60)
GRID_MAJOR = QColor(60, 68, 82)

#: A track that has not been corroborated by a detection for this long is drawn
#: as coasting, so an operator can see the difference between an object being
#: watched and an object being remembered.
COASTING_AFTER_MILLIS = 400

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND.name()};
    color: {TEXT.name()};
    font-family: "Segoe UI", "Inter", system-ui, sans-serif;
    font-size: 13px;
}}
QFrame#Panel {{
    background: {PANEL.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 6px;
}}
QLabel#PanelTitle {{
    color: {TEXT_MUTED.name()};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 8px 10px 4px 10px;
}}
QLabel#Caption {{
    color: {TEXT_FAINT.name()};
    font-size: 11px;
}}
QPushButton {{
    background: {PANEL_RAISED.name()};
    border: 1px solid {BORDER.name()};
    border-radius: 5px;
    padding: 6px 14px;
    color: {TEXT.name()};
}}
QPushButton:hover {{ background: {BORDER.name()}; }}
QPushButton:disabled {{ color: {TEXT_FAINT.name()}; }}
QTreeWidget, QTableWidget {{
    background: {PANEL.name()};
    border: none;
    alternate-background-color: {PANEL_RAISED.name()};
    gridline-color: {BORDER.name()};
}}
QHeaderView::section {{
    background: {PANEL.name()};
    color: {TEXT_MUTED.name()};
    border: none;
    border-bottom: 1px solid {BORDER.name()};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
}}
QSplitter::handle {{ background: {BORDER.name()}; }}
QStatusBar {{ color: {TEXT_MUTED.name()}; border-top: 1px solid {BORDER.name()}; }}
QScrollBar:vertical {{ background: {PANEL.name()}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER.name()}; border-radius: 5px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
