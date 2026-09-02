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
from sentinel.decode import _looks_live, redact_url
from sentinel.events import Event

from .video_view import VideoView
from .worker import AnalysisWorker, Update


@dataclass
class CameraSession:
    """A camera in the console: where it is, what it is watching, what it found."""

    camera_id: str
    #: What the source *is*, as a string rather than a path: `device:0` for a
    #: camera attached to this machine, an RTSP URL for one on the network, a
    #: file path for footage. A `Path` could only represent the last of the
    #: three, and made the other two look like files that did not exist.
    #:
    #: For a network camera this holds the credential. It is read in exactly one
    #: place — the moment `VideoSource` is constructed — and everything else
    #: uses `display_source`.
    source: str
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
    def display_source(self) -> str:
        """The source with any credential removed. Safe to log, show and store."""
        return redact_url(self.source)

    @property
    def is_live(self) -> bool:
        """Whether this source has no end.

        A file is replayed to completion; a camera runs until it is stopped.
        The difference decides whether the window can ever show "finished".
        """
        return _looks_live(self.source)

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @property
    def is_placed(self) -> bool:
        return self.pose is not None

    def stop(self, timeout_millis: int = 3000) -> bool:
        """Ask this camera's analysis to end, and wait for it to actually end.

        Returns whether the thread finished. The return value is the point: an
        earlier version called ``wait()`` and discarded the result, then dropped
        the reference regardless. If the thread had not finished — a decode
        blocked on a stalled camera is the ordinary way that happens — Python
        would garbage-collect a running QThread, and Qt aborts the process for
        that with "QThread: Destroyed while thread is still running".

        A thread that will not stop is therefore kept referenced rather than
        released. It is left running and marked as faulted, because leaking one
        thread is recoverable and killing the process in front of an operator is
        not. `terminate()` is deliberately not called: it stops the thread at an
        arbitrary instruction, which for one holding a decoder and a database
        handle risks far worse than a leak.
        """
        worker = self.worker
        if worker is None:
            return True

        worker.stop()
        if not worker.wait(timeout_millis):
            self.fault = (
                f"{self.camera_id}: the analysis thread did not stop within "
                f"{timeout_millis / 1000:.0f}s and is still running"
            )
            # Deliberately keeps `self.worker` set. Dropping it here is what
            # destroys a running QThread and aborts the process.
            return False

        # Unparented rather than deleteLater()'d. The worker is parented to the
        # window so it cannot outlive it, but leaving it parented once it has
        # finished means Qt owns a QThread nobody will ever start again — one
        # per Start/Stop cycle, for the life of the console. Detaching hands
        # ownership back to Python, whose refcount drops to zero the moment this
        # assignment lands. `deleteLater()` was tried and is wrong here: it
        # destroys the C++ object while queued signals from this worker may
        # still be in flight, and Qt aborts the process for that.
        worker.setParent(None)
        self.worker = None
        return True

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
