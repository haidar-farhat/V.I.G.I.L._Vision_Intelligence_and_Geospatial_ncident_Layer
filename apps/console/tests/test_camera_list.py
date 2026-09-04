"""Tests for the camera list panel.

The panel replaced a combo box, and the two failures it exists to prevent are
the ones checked hardest here:

- A camera that is nominally running and delivering nothing must not look like a
  camera that is running. Colour, glyph and words all have to differ, because an
  operator reads whichever of the three their screen and their eyes give them.
- A password must never reach a widget, by any route — cell, tooltip or
  accessibility text — including for a record that hands over a raw source.

The live-versus-dark assertions are driven by the engine's own
`sentinel.node.CameraHealth`, not by a stub. The first version of this file
built every health object out of field names the panel had invented, so all
twenty-one tests passed while the shipped panel rendered LIVE, DARK and STOPPED
byte-identically as "not started" — a green suite for an absent feature. Stubs
are kept only where degradation itself is the subject.

Plus the loop that froze the first version: `set_selection` is what the console's
selection bus calls, so it must not emit.
"""

from __future__ import annotations

import gc
import os
import weakref
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sentinel.node import CameraHealth, CameraState  # noqa: E402

from sentinel_console import theme  # noqa: E402
from sentinel_console.camera_list import (  # noqa: E402
    CAMERA_COLUMN,
    DARK_AFTER_SECONDS,
    HEALTH_FACTS,
    PLACED_COLUMN,
    SOURCE_COLUMN,
    STATE_DARK,
    STATE_FAULT,
    STATE_LIVE,
    STATE_OFF,
    STATE_STARTING,
    STATUS_COLUMN,
    CameraListPanel,
    camera_state,
    health_tooltip,
)
from sentinel_console.selection import Selection  # noqa: E402

#: A camera URL with a real-looking credential in it. The password is checked for
#: by this exact string everywhere below.
SECRET_SOURCE = "rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101"
PASSWORD = "hunter2"


@pytest.fixture(scope="session")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def record(camera_id: str, source: str = "/site/gate.mp4", placed: bool = False):
    """A stand-in for `CameraRecord` carrying only what the panel reads.

    Deliberately not the engine's dataclass: a panel that can only be built with
    a node behind it cannot be tested for the thing that matters, which is what
    it shows when the node is in trouble.
    """
    pose = SimpleNamespace(lat=51.5, lon=-0.1) if placed else None
    return SimpleNamespace(camera_id=camera_id, source=source, pose=pose)


def health(**facts):
    """A partial health object, for the tests about what happens when a fact is
    missing or unreadable. Never used for a live-versus-dark assertion: that is
    what `engine_health` is for, because a stub can only prove the panel agrees
    with the test that wrote it."""
    return SimpleNamespace(**facts)


def engine_health(camera_id: str = "cam-01", **overrides) -> CameraHealth:
    """The real `CameraHealth` the node hands the console, with real defaults.

    Built from the engine's own dataclass so a renamed or removed field breaks
    these tests loudly at construction, instead of degrading to "unknown" in
    silence the way `getattr` does inside the panel.
    """
    facts = dict(
        camera_id=camera_id,
        state=CameraState.LIVE,
        is_running=True,
        is_placed=True,
        analysis_fps=24.6,
        frames=1480,
        frames_dropped=0,
        frames_not_drawn=0,
        reconnects=0,
        fault=None,
        seconds_since_frame=0.04,
        seconds_since_started=61.0,
    )
    facts.update(overrides)
    return CameraHealth(**facts)


#: A camera delivering frames, and one that is nominally running and delivering
#: nothing. The engine decided both words; the panel only translates them.
LIVE = engine_health()
DARK = engine_health(
    state=CameraState.DARK, analysis_fps=0.0, frames=0,
    seconds_since_frame=None, seconds_since_started=45.0, reconnects=3,
)


def readable(text: str) -> str:
    """A status string a Windows console can actually print.

    The strip's glyphs and its separator are not in cp1252, and a test that dies
    inside its own diagnostic print under ``-s`` teaches nothing about the panel.
    """
    return text.encode("ascii", "backslashreplace").decode("ascii")


def row_texts(panel: CameraListPanel, index: int) -> list[str]:
    item = panel.tree.topLevelItem(index)
    return [item.text(column) for column in range(panel.tree.columnCount())]


def row_texts_for(facts) -> str:
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": facts})
    return row_texts(panel, 0)[STATUS_COLUMN]


def every_string(panel: CameraListPanel) -> list[str]:
    """Every piece of text the panel could put in front of a person: cell text,
    tooltips and the summary line. What the redaction test scans."""
    found = [panel.summary.text()]
    for index in range(panel.tree.topLevelItemCount()):
        item = panel.tree.topLevelItem(index)
        for column in range(panel.tree.columnCount()):
            found.append(item.text(column))
            found.append(item.toolTip(column))
            found.append(str(item.data(column, Qt.ItemDataRole.UserRole)))
    return found


# ------------------------------------------------------------------- the rows


def test_every_camera_gets_a_row_in_the_order_it_was_given(qt_app):
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-01"), record("cam-02", placed=True), record("cam-03")],
        {"cam-01": LIVE},
    )
    assert panel.tree.topLevelItemCount() == 3
    assert [row_texts(panel, i)[CAMERA_COLUMN] for i in range(3)] == [
        "cam-01", "cam-02", "cam-03"
    ]


def test_an_unplaced_camera_is_named_as_unplaced(qt_app):
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01"), record("cam-02", placed=True)], {})
    assert row_texts(panel, 0)[PLACED_COLUMN] == "not placed"
    assert row_texts(panel, 1)[PLACED_COLUMN] == "placed"
    assert "Not placed" in panel.tree.topLevelItem(0).toolTip(PLACED_COLUMN)


def test_a_camera_nobody_started_says_so_rather_than_showing_a_zero(qt_app):
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("status with no health:", readable(status))
    assert "not started" in status
    assert "fps" not in status
    assert "never been started" in panel.tree.topLevelItem(0).toolTip(STATUS_COLUMN)


# ------------------------------------------------------------------ redaction


def test_a_password_never_reaches_any_cell_or_tooltip(qt_app):
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-01", source=SECRET_SOURCE)],
        {"cam-01": LIVE},
    )
    shown = row_texts(panel, 0)[SOURCE_COLUMN]
    print("source cell:", readable(shown))
    assert PASSWORD not in shown
    assert "***" in shown
    # The username is not a secret and operators identify cameras by it.
    assert "admin" in shown and "10.0.0.5" in shown
    for text in every_string(panel):
        assert PASSWORD not in text


def test_a_record_that_redacts_its_own_source_is_trusted_over_the_raw_one(qt_app):
    # `CameraRecord.display_source` is the engine's redaction, and using it keeps
    # one definition of what a credential looks like rather than two that drift.
    already = SimpleNamespace(
        camera_id="cam-01", source=SECRET_SOURCE, pose=None,
        display_source="rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101",
    )
    panel = CameraListPanel()
    panel.show_cameras([already], {})
    assert row_texts(panel, 0)[SOURCE_COLUMN] == already.display_source
    for text in every_string(panel):
        assert PASSWORD not in text


# --------------------------------------------------------------- dark vs live


def test_the_panel_reads_only_facts_the_engine_actually_ships(qt_app):
    # The failure this catches has already happened once: the panel read
    # `running`, `last_frame_age_seconds` and `objects_now`, none of which exist
    # on `CameraHealth`, and `getattr`-with-default turned every one of them into
    # "unknown" without a word in a log. A LIVE camera and a camera nobody had
    # started rendered the same string. Nothing about that is visible in a test
    # that builds its own health objects, so the field names are checked directly.
    shipped = set(CameraHealth.__dataclass_fields__)
    print("engine fields:", sorted(shipped))
    print("panel reads  :", sorted(HEALTH_FACTS))
    assert set(HEALTH_FACTS) <= shipped
    # And the states, which are read by their string value rather than imported.
    assert {state.value for state in CameraState} == {
        "LIVE", "DARK", "STARTING", "STOPPED", "FAULTED",
    }


def test_a_running_camera_with_no_frames_is_not_shown_as_running(qt_app):
    # Driven by real `CameraHealth` records, not stubs: this assertion is the
    # feature, and against a stub it only proves the panel agrees with itself.
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-live"), record("cam-dark")],
        {"cam-live": LIVE, "cam-dark": DARK},
    )
    live_row = panel.tree.topLevelItem(0)
    dark_row = panel.tree.topLevelItem(1)
    live_status = live_row.text(STATUS_COLUMN)
    dark_status = dark_row.text(STATUS_COLUMN)
    print("live:", readable(live_status))
    print("dark:", readable(dark_status))

    # Words differ.
    assert "live" in live_status and "dark" not in live_status
    assert "dark" in dark_status and "live" not in dark_status
    # Glyph differs: filled means frames are arriving, hollow means they are not.
    assert live_status[0] != dark_status[0]
    # Colour differs.
    assert live_row.foreground(STATUS_COLUMN).color() != dark_row.foreground(
        STATUS_COLUMN
    ).color()
    # And the id itself is marked, so the left-hand column carries the warning
    # for an operator scanning a long list.
    assert dark_row.foreground(CAMERA_COLUMN).color() != live_row.foreground(
        CAMERA_COLUMN
    ).color()
    assert dark_row.font(CAMERA_COLUMN).bold()
    # Byte-identical strips are how the first version shipped: both rows read
    # "·  not started", in the same grey, and the suite was green.
    assert live_status != dark_status


def test_the_live_and_dark_colours_are_far_apart_on_a_dim_screen(qt_app):
    # Measured, then floored. A control-room monitor at a glancing angle
    # compresses everything, so "two different QColors" is not the bar — they
    # have to be separated by a real distance in RGB.
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-live"), record("cam-dark")],
        {"cam-live": LIVE, "cam-dark": DARK},
    )
    live = panel.tree.topLevelItem(0).foreground(STATUS_COLUMN).color()
    dark = panel.tree.topLevelItem(1).foreground(STATUS_COLUMN).color()
    distance = sum(
        (a - b) ** 2 for a, b in zip(live.getRgb()[:3], dark.getRgb()[:3])
    ) ** 0.5
    print("live", live.name(), "dark", dark.name(), "distance", round(distance, 1))
    assert distance > 150.0


def test_a_camera_seeing_nothing_happen_is_still_live(qt_app):
    # An empty yard at 4 a.m. produces no detections and no tracks all night and
    # is working perfectly. Dark is about frames, never about detections — and
    # structurally so: no fact this panel reads counts an object, so no amount of
    # quiet can move a row towards dark.
    assert not [
        name for name in HEALTH_FACTS
        if "object" in name or "detect" in name or "track" in name
    ]
    quiet = engine_health(state=CameraState.LIVE, analysis_fps=24.6)
    assert camera_state(quiet) == STATE_LIVE
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": quiet})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("quiet but live:", readable(status))
    assert "live" in status and "dark" not in status


def test_a_camera_still_opening_its_stream_is_not_called_dark(qt_app):
    # The engine's STARTING exists because opening an RTSP stream takes seconds,
    # and it is not `is_dark` — the map paints such a camera's ground as covered.
    # A list calling it dark at the same moment would have the two windows
    # contradicting each other about the same camera in front of the operator.
    opening = engine_health(
        state=CameraState.STARTING, analysis_fps=0.0, frames=0,
        seconds_since_frame=None, seconds_since_started=3.0,
    )
    assert camera_state(opening) == STATE_STARTING
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-opening"), record("cam-dark")],
        {"cam-opening": opening, "cam-dark": DARK},
    )
    starting_status = row_texts(panel, 0)[STATUS_COLUMN]
    dark_status = row_texts(panel, 1)[STATUS_COLUMN]
    print("starting:", readable(starting_status))
    print("dark    :", readable(dark_status))
    assert "starting" in starting_status and "dark" not in starting_status
    assert starting_status[0] != dark_status[0]
    assert panel.tree.topLevelItem(0).foreground(STATUS_COLUMN).color() == theme.STALE


def test_a_stopped_camera_is_not_reported_as_a_failed_one(qt_app):
    # STOPPED and FAULTED are different facts — nobody started it, versus it died
    # overnight — and the engine keeps them apart, so the strip must too.
    stopped = engine_health(
        state=CameraState.STOPPED, is_running=False, analysis_fps=0.0,
        seconds_since_frame=None, seconds_since_started=None,
    )
    assert camera_state(stopped) == STATE_OFF
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": stopped})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("stopped:", readable(status))
    assert "not started" in status
    assert "fail" not in status and "dark" not in status


def test_a_health_object_with_no_state_still_gets_the_engines_threshold(qt_app):
    # The frame-clock fallback, for a health object that carries no state at all.
    # It uses `sentinel.node.DARK_AFTER_SECONDS` rather than a second copy of the
    # number, because two thresholds are two answers to "is this camera dark".
    from sentinel import node

    assert DARK_AFTER_SECONDS == node.DARK_AFTER_SECONDS
    print("dark after", DARK_AFTER_SECONDS, "seconds")
    recent = health(is_running=True, seconds_since_frame=1.0)
    gone = health(is_running=True, seconds_since_frame=DARK_AFTER_SECONDS + 1)
    opening = health(is_running=True, seconds_since_frame=None,
                     seconds_since_started=2.0)
    never = health(is_running=True, seconds_since_frame=None,
                   seconds_since_started=DARK_AFTER_SECONDS + 1)
    assert camera_state(recent) == STATE_LIVE
    assert camera_state(gone) == STATE_DARK
    assert camera_state(opening) == STATE_STARTING
    # A camera that never connects must not sit at "starting" all night.
    assert camera_state(never) == STATE_DARK


def test_a_failed_camera_reports_the_reason_it_failed(qt_app):
    facts = engine_health(
        state=CameraState.FAULTED, is_running=False, analysis_fps=0.0,
        fault="10.0.0.5 is not reachable: timed out",
        seconds_since_frame=None,
    )
    assert camera_state(facts) == STATE_FAULT
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": facts})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("failed:", readable(status))
    assert "not reachable" in status
    assert panel.tree.topLevelItem(0).foreground(STATUS_COLUMN).color() == theme.FAULT


def test_a_live_camera_whose_frames_slowed_says_how_stale_it_is(qt_app):
    # The engine quotes no frame rate unless a frame arrived in the last second,
    # so a live camera can carry a measured 0.0 fps. Printing "live · 0.0 fps"
    # would read as a stalled camera that is not stalled; the age is the fact
    # that is actually known.
    slowing = engine_health(analysis_fps=0.0, seconds_since_frame=4.0)
    assert camera_state(slowing) == STATE_LIVE
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": slowing})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("slowed:", readable(status))
    assert "live" in status and "0.0 fps" not in status
    assert "4s" in status


def test_a_camera_that_stopped_delivering_says_when_it_stopped(qt_app):
    # A camera that ran all day and went dark at 3 a.m. is a different fact from
    # one that never delivered anything, and the strip must not flatten them.
    went_dark = engine_health(
        state=CameraState.DARK, analysis_fps=0.0, seconds_since_frame=45.0,
    )
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-went-dark"), record("cam-never")],
        {"cam-went-dark": went_dark, "cam-never": DARK},
    )
    stopped_status = row_texts(panel, 0)[STATUS_COLUMN]
    never_status = row_texts(panel, 1)[STATUS_COLUMN]
    print("went dark:", readable(stopped_status))
    print("never    :", readable(never_status))
    assert "dark" in stopped_status and "45s" in stopped_status
    assert "no frame yet" in never_status
    assert stopped_status != never_status


def test_a_fault_with_no_reason_does_not_read_as_a_cut_off_message(qt_app):
    nameless = health(state="FAULTED", fault=None)
    text = row_texts_for(nameless)
    print("faulted, no reason:", readable(text))
    assert not text.endswith("— ")
    assert "no reason reported" in text


def test_the_summary_counts_the_cameras_nobody_would_go_looking_at(qt_app):
    panel = CameraListPanel()
    panel.show_cameras(
        [record("cam-01", placed=True), record("cam-02", placed=True), record("cam-03")],
        {"cam-01": LIVE, "cam-02": DARK},
    )
    summary = panel.summary.text()
    print("summary:", readable(summary))
    assert "3 cameras" in summary
    assert "1 dark" in summary
    assert "1 not placed" in summary


def test_an_empty_list_says_so_instead_of_showing_nothing(qt_app):
    panel = CameraListPanel()
    panel.show_cameras([], {})
    assert panel.tree.topLevelItemCount() == 0
    assert "No cameras" in panel.summary.text()


# ----------------------------------------------------------------- duck typing


def test_a_health_object_carrying_only_some_facts_is_accepted(qt_app):
    # The engine owns this type and it will gain and lose fields. Anything with
    # the attributes actually read must work, and a missing fact is unknown
    # rather than a plausible zero.
    minimal = SimpleNamespace(state="DARK")
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": minimal})
    status = row_texts(panel, 0)[STATUS_COLUMN]
    tip = panel.tree.topLevelItem(0).toolTip(STATUS_COLUMN)
    print("minimal:", readable(status), "|", readable(tip.replace(chr(10), " / ")))
    assert "dark" in status
    assert "unknown" in tip


def test_a_health_fact_that_raises_does_not_take_the_panel_down(qt_app):
    class Exploding:
        state = "LIVE"
        is_running = True
        seconds_since_frame = 0.1

        @property
        def analysis_fps(self):
            raise RuntimeError("the runner died mid-read")

    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": Exploding()})
    assert panel.tree.topLevelItemCount() == 1
    status = row_texts(panel, 0)[STATUS_COLUMN]
    print("exploding:", readable(status))
    assert camera_state(Exploding()) == STATE_LIVE
    # A frame rate nobody could read is reported as unknown, not as 0.0 fps —
    # a fabricated zero is indistinguishable from a camera that has stalled.
    assert "fps unknown" in status


def test_a_camera_with_no_health_entry_at_all_is_off(qt_app):
    assert camera_state(None) == STATE_OFF
    assert camera_state(health(is_running=False)) == STATE_OFF
    # An unrecognised state word is not trusted either: it falls through to the
    # frame clock rather than being shown to an operator as a colour.
    assert camera_state(health(state="SOMETHING_NEW", is_running=False)) == STATE_OFF


def test_no_tooltip_claims_never_while_counting_frames(qt_app):
    # The contradiction a misspelled field name produced: "frames 1480" one line
    # above "newest never". An operator cannot tell which line to believe, and
    # both were printed with the same confidence. Ignorance says "unknown";
    # "never" is a claim about the whole past and needs a frame count that agrees.
    counted = health(state="DARK", frames=1480, seconds_since_frame=None)
    tip = health_tooltip("cam-01", counted)
    print("tooltip:", readable(tip.replace(chr(10), " / ")))
    assert "unknown" in tip and "never" not in tip

    fresh = health(state="STARTING", frames=0, seconds_since_frame=None)
    print("tooltip:", readable(health_tooltip("cam-01", fresh).replace(chr(10), " / ")))
    assert "never" in health_tooltip("cam-01", fresh)

    # And nothing the panel can render out of a real record does it either.
    for facts in (LIVE, DARK, engine_health(state=CameraState.STARTING, frames=0,
                                            seconds_since_frame=None)):
        tip = health_tooltip(facts.camera_id, facts)
        frames = facts.frames
        assert not (frames > 0 and "never" in tip), tip


# ------------------------------------------------------------------ selection


def test_selecting_a_row_emits_a_selection_for_that_camera(qt_app):
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01"), record("cam-02")], {})
    seen: list = []
    panel.selected.connect(seen.append)
    panel.tree.topLevelItem(1).setSelected(True)
    assert seen == [Selection.camera("cam-02")]
    assert panel.selected_camera_id() == "cam-02"


def test_set_selection_highlights_the_row_without_emitting(qt_app):
    # The console's selection bus calls this from its own `changed` signal. A
    # re-emit here goes straight back into the bus, and the window stops
    # answering.
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01"), record("cam-02")], {})
    seen: list = []
    panel.selected.connect(seen.append)
    panel.set_selection(Selection.camera("cam-02"))
    assert seen == []
    assert panel.selected_camera_id() == "cam-02"


def test_selecting_something_that_is_not_a_camera_clears_the_list(qt_app):
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01"), record("cam-02")], {})
    panel.set_selection(Selection.camera("cam-01"))
    seen: list = []
    panel.selected.connect(seen.append)
    panel.set_selection(Selection.zone("zone-9"))
    assert panel.selected_camera_id() is None
    panel.set_selection(None)
    assert panel.selected_camera_id() is None
    assert seen == []


def test_a_track_does_not_highlight_the_camera_it_belongs_to(qt_app):
    # A track carries a camera id, but selecting a track is not selecting the
    # camera: a highlighted row is a claim that the operator picked that camera,
    # and the toolbar acts on the row that is highlighted.
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {})
    panel.set_selection(Selection.track("cam-01", 3))
    assert panel.selected_camera_id() is None


def test_the_selection_survives_the_list_being_rebuilt(qt_app):
    # The panel is rebuilt on every collection tick; an operator's selection
    # disappearing under them once a second is unusable.
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01"), record("cam-02")], {})
    panel.set_selection(Selection.camera("cam-02"))
    seen: list = []
    panel.selected.connect(seen.append)
    panel.show_cameras(
        [record("cam-01"), record("cam-02")], {"cam-02": DARK}
    )
    assert panel.selected_camera_id() == "cam-02"
    assert seen == []


def test_the_panel_is_freed_when_its_last_reference_goes(qt_app):
    # A reference cycle holding a QWidget means the widget is destroyed at
    # interpreter shutdown, after PySide has torn the QApplication down, which
    # corrupts the heap and kills the process with 0xC0000374 at exit — after
    # every test has passed. A lambda closing over `self` in a signal connection
    # is how that happens, so the connection inside the panel is a bound method.
    panel = CameraListPanel()
    panel.show_cameras([record("cam-01")], {"cam-01": LIVE})
    ref = weakref.ref(panel)
    del panel
    gc.collect()
    survivor = ref()
    if survivor is not None:
        holders = sorted(
            {type(r).__name__ for r in gc.get_referrers(survivor)} - {"frame", "list"}
        )
        del survivor
        raise AssertionError(
            f"CameraListPanel outlived its last reference (held by {holders})."
        )
