"""The console's half of a camera.

A camera used to be four things bundled here — a source, a placement, a running
analysis and a widget — because the console *was* the application and there was
nowhere else for the first three to live. There is now: `sentinel.node.Node`
owns the source, the pose, the events, the faults and the thread, and it does so
with no Qt anywhere near it.

What is left here is the part that genuinely belongs to an interface: the widget
that draws this camera, and the most recent frame drawn on it. Everything else
is read through `record`, which is the node's, so there is exactly one copy of
each fact rather than two that can disagree.

That mattered more than it sounds. The old console kept its own event list, fed
from the *latest-wins* update slot — so events on frames the interface never
collected were never correlated and never persisted. Correlation ran on a
frame-rate-dependent sample of the evidence, and a busy interface meant a
quieter incident log. Reading from the node instead removes that by
construction: the node drains events on their own path, independent of whatever
the display managed to keep up with.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.core import CameraPose
from sentinel.node import CameraRecord, Node, Update

from .video_view import VideoView


@dataclass
class CameraSession:
    """One camera as the console sees it: a pane, and what was last drawn on it."""

    #: The node's record. The single copy of every fact about this camera that
    #: is not about drawing it.
    record: CameraRecord
    view: VideoView
    #: The node that owns `record`. Held so that assigning a pose is the same
    #: operation the placement dialog performs — persisted and audited — rather
    #: than a second, quieter way of doing it that forgets both.
    node: "Node | None" = None
    #: The most recent update collected for this camera. Used for painting and
    #: for the track table; never for correlation, which reads the node.
    last: Update | None = None

    # ---- everything below is the node's, exposed here so the interface reads
    # ---- one place rather than reaching through `session.record` everywhere.

    @property
    def camera_id(self) -> str:
        return self.record.camera_id

    @property
    def source(self) -> str:
        """May carry a credential. Use `display_source` for anything visible."""
        return self.record.source

    @property
    def display_source(self) -> str:
        return self.record.display_source

    @property
    def pose(self) -> CameraPose | None:
        return self.record.pose

    @pose.setter
    def pose(self, pose: CameraPose | None) -> None:
        """Place this camera. Not merely a field assignment.

        Routed through the node so it persists, audits, and reaches a running
        analysis — which is what placing a camera has to mean. A plain
        attribute would have been a second way to do it that did none of those,
        and the two would have disagreed the first time anybody used the
        quieter one.
        """
        if self.node is None:
            self.record.pose = pose
            return
        self.node.place_camera(self.camera_id, pose)

    @property
    def is_live(self) -> bool:
        """Whether this source has no end.

        A file is replayed to completion; a camera runs until it is stopped.
        The difference decides whether the interface can ever show "finished".
        """
        from sentinel.decode import _looks_live

        return _looks_live(self.record.source)

    @property
    def fault(self) -> str | None:
        return self.record.fault

    @property
    def is_running(self) -> bool:
        return self.record.is_running

    @property
    def is_placed(self) -> bool:
        return self.record.pose is not None

    @property
    def events(self) -> list:
        """This camera's events, as the node has them — not as the display saw them."""
        return self.record.events

    def absorb(self, update: Update) -> None:
        """Keep the newest frame for painting.

        It no longer accumulates events. It used to, and that was the bug: the
        update slot is latest-wins, so anything raised on a frame the interface
        was too busy to collect never reached correlation at all.
        """
        self.last = update
