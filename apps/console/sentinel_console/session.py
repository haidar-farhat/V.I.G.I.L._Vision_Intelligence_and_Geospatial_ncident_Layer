"""One camera's worth of console state.

A camera is a source, a placement, a running analysis, and a view — and those
four have to stay together or the interface starts showing one camera's tracks
over another's frame. Bundling them is what makes a second camera a matter of
adding to a list rather than a matter of rewriting the window.

Each session owns its own pipeline. Nothing is shared between them except the
zones, which belong to the ground rather than to any camera, and the correlation
that runs above them. That independence is the same one a distributed deployment
needs: a session is what a worker node runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sentinel.core import CameraPose
from sentinel.events import Event

from .video_view import VideoView
from .worker import AnalysisWorker, Update


@dataclass
class CameraSession:
    """A camera in the console: where it is, what it is watching, what it found."""

    camera_id: str
    source_path: Path
    view: VideoView
    pose: CameraPose | None = None
    worker: AnalysisWorker | None = None
    #: The most recent update drawn for this camera.
    last: Update | None = None
    #: Events this camera has raised, retained for correlation across cameras.
    events: list[Event] = field(default_factory=list)
    #: Set when this camera's own run ends or fails, so the window can show
    #: which camera is in trouble rather than only that something is.
    fault: str | None = None

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @property
    def is_placed(self) -> bool:
        return self.pose is not None

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None

    def absorb(self, update: Update) -> None:
        """Take an update from this camera's worker.

        Events accumulate here rather than in the worker, because correlation
        happens across cameras and a worker that correlated its own events in
        isolation would produce one incident per camera — which is exactly the
        alert duplication the system exists to prevent.
        """
        self.last = update
        self.events.extend(update.result.events)

        # Bounded. A console left running for a week must not accumulate every
        # event it ever saw; persistence is where the full history belongs.
        if len(self.events) > 4000:
            del self.events[: len(self.events) - 4000]

    def describe(self) -> str:
        if self.fault:
            return f"{self.camera_id}: {self.fault}"
        if not self.is_running:
            return f"{self.camera_id}: stopped"
        if self.last is None:
            return f"{self.camera_id}: starting"
        where = "placed" if self.is_placed else "not placed"
        return (
            f"{self.camera_id}: {self.last.analysis_fps:.0f} fps, "
            f"{len(self.last.result.tracks)} tracked, {where}"
        )
