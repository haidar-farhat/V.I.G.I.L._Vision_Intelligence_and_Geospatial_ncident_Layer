"""The analysis thread.

The pipeline decodes, detects and tracks at whatever rate it can manage. None of
that may happen on the UI thread: Qt repaints between events, so a decode loop on
the same thread makes the interface stop responding — including the button that
stops it.

So the pipeline runs here and the interface collects results from it. Note the
direction: the interface **pulls**, on a repaint timer, rather than the worker
pushing a signal per frame. Qt's queued-signal delivery is unbounded, so a
pipeline running at 90 fps in front of a display repainting at 30 accumulates a
backlog that grows until memory runs out — and every frame in that backlog is
already stale by the time it is drawn.

Latest-wins is not a compromise here, it is what an operator actually wants: a
control room needs to see now, not a slow replay of the last minute. What was
skipped is counted rather than hidden, because a viewer showing a third of the
frames while reporting nothing unusual is worse than one that says so.

Nothing here is dropped from the *analysis*. Every frame is still decoded,
detected and tracked; it is only the drawing that skips.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from sentinel.core import CameraPose
from sentinel.decode import DecodeError, VideoSource
from sentinel.detect import Detector, DetectorInfo
from sentinel.events import Rule
from sentinel.zones import Zone
from sentinel.incidents import Incident
from sentinel.pipeline import FrameResult, Pipeline, PipelineStats


@dataclass(frozen=True, slots=True)
class Update:
    """One frame's worth of everything the interface needs.

    The image and the conclusions travel together, because drawing a track box
    over a different frame than the one it was computed from misrepresents what
    the system saw.
    """

    result: FrameResult
    #: Frames the analysis completed per second, measured over the last second
    #: rather than averaged since start-up, which would hide a stall.
    analysis_fps: float
    #: Results the interface never saw because it was busy.
    skipped: int
    stats: PipelineStats
    #: Incidents as of this update. Recomputed on a slow timer rather than per
    #: frame: correlation is a batch operation over a window, and running it at
    #: frame rate would cost far more than it tells anyone.
    incidents: tuple[Incident, ...] = ()

    @property
    def image(self) -> np.ndarray | None:
        return self.result.image


class AnalysisWorker(QThread):
    """Runs one camera's pipeline and holds the most recent result."""

    #: Emitted once, when the run ends normally. Carries a displayable reason.
    finished_run = Signal(str)
    #: Emitted once, on failure. The message is already redacted and safe to
    #: display: it comes from DecodeError, which never carries a credential.
    failed = Signal(str)

    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        pose: CameraPose | None = None,
        *,
        realtime: bool = True,
        zones: Sequence[Zone] = (),
        rules: Sequence[Rule] = (),
        node_id: str = "local",
        wall_clock_epoch_millis: int | None = None,
        correlate_every_millis: int = 2000,
        parent=None,
    ):
        """
        ``realtime`` paces a recording to its own timeline. Without it a file is
        analysed as fast as the machine allows, which is right for batch review
        and useless for watching: twelve seconds of footage flashes past in two.
        """
        super().__init__(parent)
        self._source = source
        self._detector = detector
        self._pose = pose
        self._realtime = realtime
        self._zones = list(zones)
        self._rules = list(rules)
        self._node_id = node_id
        self._epoch_millis = wall_clock_epoch_millis
        self._correlate_every = correlate_every_millis
        self._last_correlated = 0
        self._incidents: tuple[Incident, ...] = ()

        self._mutex = QMutex()
        self._stopping = False
        self._latest: Update | None = None
        self._skipped = 0
        self._pending_pose: CameraPose | None = None
        self._pose_changed = False

    @property
    def detector_info(self) -> DetectorInfo:
        return self._detector.info

    @property
    def source_id(self) -> str:
        return self._source.source_id

    @property
    def display_url(self) -> str:
        """Safe to show: any credential has already been removed."""
        return self._source.display_url

    def stop(self) -> None:
        """Ask the run to end. Safe to call from any thread."""
        with QMutexLocker(self._mutex):
            self._stopping = True

    def set_pose(self, pose: CameraPose | None) -> None:
        """Place or move the camera while it is running.

        Applied by the analysis thread between frames rather than here, so the
        tracker is never mutated from under a call that is using it. Existing
        tracks keep their identity: placing a camera does not turn the people it
        was already following into different people.
        """
        with QMutexLocker(self._mutex):
            self._pending_pose = pose
            self._pose_changed = True

    def take_latest(self) -> Update | None:
        """The newest result, or ``None`` if nothing new has arrived.

        Called from the UI thread on a repaint timer. Taking clears the slot, so
        a slow interface skips frames rather than falling behind.
        """
        with QMutexLocker(self._mutex):
            update, self._latest = self._latest, None
            return update

    def _should_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stopping

    def _publish(self, update: Update) -> None:
        with QMutexLocker(self._mutex):
            if self._latest is not None:
                self._skipped += 1
            self._latest = update

    def run(self) -> None:  # noqa: D102 - QThread entry point
        pipeline = Pipeline(
            self._source,
            self._detector,
            pose=self._pose,
            keep_images=True,
            zones=self._zones,
            rules=self._rules,
            node_id=self._node_id,
            wall_clock_epoch_millis=self._epoch_millis,
        )
        started_wall = time.perf_counter()
        first_stamp: int | None = None
        recent: list[float] = []

        try:
            for result in pipeline.run():
                if self._should_stop():
                    self.finished_run.emit("Stopped.")
                    return

                self._apply_pending_pose(pipeline)

                if first_stamp is None:
                    first_stamp = result.timestamp_millis
                if self._realtime and not self._source.is_live:
                    self._pace(started_wall, first_stamp, result.timestamp_millis)

                now = time.perf_counter()
                recent.append(now)
                while recent and now - recent[0] > 1.0:
                    recent.pop(0)
                measured = (
                    (len(recent) - 1) / (now - recent[0]) if len(recent) > 1 else 0.0
                )

                if result.timestamp_millis - self._last_correlated >= self._correlate_every:
                    self._last_correlated = result.timestamp_millis
                    self._incidents = tuple(pipeline.incidents())

                with QMutexLocker(self._mutex):
                    skipped = self._skipped
                self._publish(
                    Update(
                        result=result,
                        analysis_fps=measured,
                        skipped=skipped,
                        stats=pipeline.stats,
                        incidents=self._incidents,
                    )
                )

            self.finished_run.emit("Source ended.")

        except DecodeError as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001 - a dead thread must still explain itself
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            pipeline.close()

    def _apply_pending_pose(self, pipeline: Pipeline) -> None:
        with QMutexLocker(self._mutex):
            if not self._pose_changed:
                return
            pose, self._pose_changed = self._pending_pose, False
        pipeline.set_pose(pose)

    def _pace(self, started_wall: float, first_stamp: int, stamp: int) -> None:
        """Hold a recording to its own timeline."""
        target = (stamp - first_stamp) / 1000.0
        drift = target - (time.perf_counter() - started_wall)
        if drift > 0:
            # Bounded, so a container with one bad timestamp cannot stall the run.
            self.msleep(int(min(drift, 0.5) * 1000))
