#!/usr/bin/env python3
"""Drive the real console and photograph it.

A test asserts that a value is right. It does not notice that the value is drawn
in dark grey on a slightly darker grey, that two panels overlap, that a label is
truncated to `Restricted Are…`, or that the track table is empty because the
splitter gave it four pixels. Every one of those passes a suite and fails an
operator.

So this starts the actual `ConsoleWindow`, feeds it the actual reference scene
through the actual node, and writes PNGs of the actual widgets. The images are
the output: they are meant to be *looked at*, and reading one renders it.

**It runs on the native platform, deliberately, and this matters more than it
sounds.** The obvious thing is `QT_QPA_PLATFORM=offscreen`, and the first
version did that. Every label in every screenshot came out as tofu boxes —
`□□□□` — and it looked exactly like a serious rendering defect. It was not:
Qt's offscreen platform reports **zero** font families on this machine, against
290 on the native one. Measured, both ways.

So an offscreen harness cannot verify text at all, which is most of what an
interface is. It renders geometry faithfully and every string as a row of boxes.
This prefers the real platform and only falls back to offscreen with a warning
saying that the images that follow prove nothing about any label.

What it still cannot catch, on either platform: window decorations, DPI scaling
on a real monitor, and a genuinely stuck event loop. Those need a person at a
screen.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine" / "tests"), str(ROOT / "apps" / "console")]

OUT = ROOT / "dist" / "screenshots"

#: How long to let the analysis run before photographing it. Long enough for the
#: reference scene to produce tracks, events and an incident — a screenshot of
#: an empty interface proves only that it starts.
SETTLE_SECONDS = 12.0


def pump(app, window, seconds: float) -> None:
    """Run the event loop and the console's own collection timer.

    Both, deliberately. Processing events alone leaves the interface repainting
    an empty node, because the timer that calls `poll` is what moves any work
    forward.
    """
    from PySide6.QtCore import QEventLoop

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        window._collect()


def shoot(widget, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    pixmap = widget.grab()
    if pixmap.isNull() or pixmap.width() < 8 or pixmap.height() < 8:
        raise SystemExit(
            f"{name}: grab produced nothing usable ({pixmap.width()}x{pixmap.height()}). "
            "A widget with no size is a layout that collapsed."
        )
    pixmap.save(str(path))
    print(f"  {path.relative_to(ROOT)}  {pixmap.width()}x{pixmap.height()}")
    return path


def choose_platform() -> bool:
    """Use the native platform when there is one. Returns whether text is real.

    Set before QApplication exists, because the platform plugin is chosen once
    and cannot be changed afterwards.
    """
    import os

    forced = os.environ.get("QT_QPA_PLATFORM")
    if forced and forced != "offscreen":
        return True

    if sys.platform == "win32":
        # Windows always has a window station available to this process, and
        # `grab()` needs no visible window, so the native platform costs
        # nothing and is the only one with a font database.
        os.environ["QT_QPA_PLATFORM"] = "windows"
        return True

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        os.environ.pop("QT_QPA_PLATFORM", None)
        return True

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    return False


def live_camera() -> str | None:
    """The first camera on this machine that actually opens, or None.

    Probing opens each device briefly, which is a deliberate act — so this is
    only called when the caller asked for a live run.
    """
    from sentinel import devices

    for camera in devices.discover(probe_indices=True):
        if camera.index_confirmed:
            return camera.source
    return None


def main() -> int:
    live = "--live" in sys.argv
    # Off by default: the reference scene is a drawn walker, and a COCO model
    # correctly finds no person in it. Shots of the segmenter have to be taken
    # against a real camera or they show an empty track table and prove nothing.
    segment = "--segment" in sys.argv
    text_is_real = choose_platform()

    from PySide6.QtWidgets import QApplication

    import scene
    from sentinel import logs
    from sentinel.core import CameraPose, LatLon, destination_point
    from sentinel_console.app import ConsoleWindow

    logs.configure(level="WARNING", file="")

    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="sentinel-shots-"))
    media = scene.write_scene(workspace / "reference.mp4")

    app = QApplication.instance() or QApplication([])
    model = None
    if segment:
        from sentinel.paths import default_model_path

        model = default_model_path()
        if model is None:
            print("--segment: no *-seg.onnx in the models directory", file=sys.stderr)
            return 2
        print(f"detector: {model}")

    window = ConsoleWindow(workspace / "console.db", model=model)
    window.resize(1500, 920)
    window.show()

    from PySide6.QtGui import QFontDatabase

    families = len(QFontDatabase.families())
    if not text_is_real or families == 0:
        print(file=sys.stderr)
        print(
            "  WARNING: no display, so these are offscreen renders and Qt has "
            f"{families} font families.",
            file=sys.stderr,
        )
        print(
            "  Every label will appear as boxes. The geometry is real; nothing "
            "here says anything about text.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
    else:
        print(f"  platform: native, {families} font families")
        print()

    print("capturing:")
    shoot(window, ("live-" if live else "") + "01-empty")

    if live:
        source = live_camera()
        if source is None:
            print("  no camera on this machine would open", file=sys.stderr)
            return 1
        print(f"  live camera: {source}")
        session = window.add_camera(source, "webcam")
    else:
        session = window.add_camera(media, "gate")
    session.pose = CameraPose(
        position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0,
        pitch=-22.0, horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
    )
    window._refresh_placement()

    # A zone the camera can actually see, placed the way the console places one
    # — and a second of another kind, at a picked point, because a plan view
    # with one red square says nothing about whether kinds are told apart.
    from sentinel.zones import ZoneKind

    window.zone_radius.setValue(12.0)
    window._add_zone()
    window._add_zone(
        name="Public pavement", kind=ZoneKind.EXCLUSION, radius=6.0,
        centre=destination_point(session.pose.position, session.pose.heading + 28.0, 30.0),
    )

    shoot(window, ("live-" if live else "") + "02-configured")

    window._start()
    pump(app, window, SETTLE_SECONDS)

    prefix = "live-" if live else ""
    shoot(window, f"{prefix}03-running")
    shoot(window.map, f"{prefix}04-plan-view")
    shoot(window.tracks, f"{prefix}05-track-table")
    window.detail_tabs.setCurrentIndex(1)
    shoot(window.detail_tabs, f"{prefix}08-zones")
    window.detail_tabs.setCurrentIndex(0)
    shoot(window.incidents, f"{prefix}06-incidents")
    if window._sessions:
        shoot(next(iter(window._sessions.values())).view, f"{prefix}07-camera-view")

    # What the interface actually concluded, printed beside the images so a
    # blank-looking screenshot can be told apart from a blank one that is right.
    print()
    print(f"  cameras     {len(window._sessions)}")
    print(f"  zones       {len(window._zones)}")
    print(f"  tracks      {window.tracks.topLevelItemCount()} row(s)")
    print(f"  incidents   {len(window._incidents)}")
    print(f"  status      {window.status.currentMessage()}")

    window._stop()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
