"""One machine's worth of Sentinel Vision, with no interface attached.

Until this existed the console *was* the application: `ConsoleWindow` owned the
database, the zones, the rule set, the camera lifecycle and the correlation loop
in a thousand lines of `QMainWindow`. That is fine for a demonstration and wrong
for a deployment, because it makes three things impossible at once — running
unattended on a machine with no display, running analysis on a worker node while
an operator watches from somewhere else, and restarting the interface without
stopping the cameras.

A `Node` is what a worker runs. It owns cameras, zones, rules, correlation,
persistence and recording, and it imports no Qt at all — asserted by test,
because the way that guarantee dies is one convenient import.

**The interface pulls; nothing is pushed at it.** Each camera runs on its own
thread and publishes its most recent result into a single slot, latest-wins. A
viewer that is busy skips frames rather than accumulating a backlog of images
that are already stale by the time they are drawn — and what was skipped is
counted, because a display showing a third of the frames while reporting nothing
unusual is worse than one that says so. Nothing is dropped from the *analysis*:
every frame is decoded, detected and tracked. Only the drawing skips.

**Correlation runs on a cadence, not per frame.** It is a batch operation over a
window of events across every camera, and running it at frame rate would cost far
more than it tells anyone. Doing it across all cameras rather than within each
is the whole point: a camera correlating only its own events raises one incident
per camera for one intrusion, which is the duplication this system exists to
remove.

What is deliberately *not* here: any network listener. A node that other machines
can talk to needs a control plane, mTLS and pairing, and none of that exists yet
— see ROADMAP 2.1. This is the half that had to come first, because there was no
object for an API to be an API *of*.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from pathlib import Path
from typing import Iterator, Sequence

from .core import CameraPose
from .decode import REDACTED, DecodeError, VideoSource
from .detect import Detector, DetectorInfo, MotionDetector
from .evidence import (
    DEFAULT_LEAD_SECONDS,
    DEFAULT_TRAIL_SECONDS,
    coverage_for,
    export_incident,
)
from .events import Event, Rule, default_rules
from .incidents import Correlator, Incident
from .logs import get as _get_logger
from .pipeline import FrameResult, Pipeline, PipelineStats
from .store import Store, default_database_path
from .zones import Zone

_log = _get_logger(__name__)

#: How often correlation runs across every camera. Two seconds is far slower
#: than frame rate and far faster than a person notices.
DEFAULT_CORRELATE_MILLIS = 2000

#: How long `stop` waits for a camera thread to end before reporting that it did
#: not. A decode blocked on a stalled camera is the ordinary way that happens.
STOP_TIMEOUT_SECONDS = 10.0

#: Recorded in the audit log. There is no authentication yet, so there is nobody
#: to name; recording the truth beats inventing an operator, because an audit
#: trail with a false entry is worse than no audit trail.
ACTOR = "node"


class NodeError(RuntimeError):
    """The node was asked for something it cannot do."""


@dataclass(frozen=True, slots=True)
class Update:
    """One frame's worth of everything a viewer needs.

    The image and the conclusions travel together, because drawing a track box
    over a different frame than the one it was computed from misrepresents what
    the system saw.
    """

    result: FrameResult
    #: Frames the analysis completed per second, measured over the last second
    #: rather than averaged since start-up, which would hide a stall.
    analysis_fps: float
    #: Results nobody collected, because the collector was busy.
    skipped: int
    stats: PipelineStats

    @property
    def image(self) -> "np.ndarray | None":
        """The frame these conclusions were drawn from, when one was kept.

        `None` unless the runner was built with `keep_images` — a node with
        nobody watching has no use for a full-resolution image per result.
        """
        return self.result.image


@dataclass
class CameraRecord:
    """A camera this node knows about, running or not."""

    camera_id: str
    #: A file path, an `rtsp://` URL, or `device:N`. May carry a credential, so
    #: it is never logged, displayed or stored — `display_source` is.
    source: str
    pose: CameraPose | None = None
    runner: "CameraRunner | None" = None
    #: Events this camera has raised, kept so correlation can run across the
    #: whole node rather than within one camera.
    events: list[Event] = field(default_factory=list)
    #: Set when this camera's own run ends or fails, so an interface can show
    #: *which* camera is in trouble rather than only that something is.
    fault: str | None = None

    @property
    def display_source(self) -> str:
        from .decode import redact_url

        return redact_url(self.source)

    @property
    def is_running(self) -> bool:
        return self.runner is not None and self.runner.is_running


class CameraRunner:
    """One camera's pipeline, on its own thread.

    The non-Qt twin of the console's `AnalysisWorker`, and the reason a worker
    node needs no display. Threading here is `threading`, locking is `Lock`, and
    publishing is a single slot rather than a signal — Qt's queued delivery is
    unbounded, and a pipeline running at 90 fps in front of a collector running
    at 30 accumulates until memory runs out.
    """

    __slots__ = (
        "_source", "_detector", "_pose", "_zones", "_rules", "_node_id",
        "_epoch_millis", "_keep_images", "_realtime", "_record_to",
        "_segment_seconds", "_thread", "_lock", "_stopping",
        "_latest", "_skipped", "_pending_pose", "_pose_changed", "_fault",
        "_pipeline", "_new_events", "_new_segments",
    )

    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        pose: CameraPose | None = None,
        *,
        zones: Sequence[Zone] = (),
        rules: Sequence[Rule] = (),
        node_id: str = "local",
        keep_images: bool = False,
        realtime: bool = False,
        record_to: Path | None = None,
        segment_seconds: float = 60.0,
    ):
        """
        ``keep_images`` is off by default. A node with nobody watching has no use
        for a full-resolution image per result, and it is tens of megabytes over
        a short clip.

        ``realtime`` paces a file to its own timeline. Off by default, because a
        node analysing recorded footage should go as fast as it can; a *viewer*
        wants it paced, and turns this on.
        """
        self._source = source
        self._detector = detector
        self._pose = pose
        self._zones = list(zones)
        self._rules = list(rules)
        self._node_id = node_id
        self._keep_images = keep_images
        self._realtime = realtime
        self._record_to = record_to
        self._segment_seconds = segment_seconds

        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stopping = False
        self._latest: Update | None = None
        self._skipped = 0
        self._pending_pose: CameraPose | None = None
        self._pose_changed = False
        self._fault: str | None = None
        self._pipeline: Pipeline | None = None
        self._new_events: list[Event] = []
        self._new_segments: list = []

    # ------------------------------------------------------------------ state

    @property
    def source_id(self) -> str:
        return self._source.source_id

    @property
    def display_url(self) -> str:
        """Safe to show: any credential has already been removed."""
        return self._source.display_url

    @property
    def detector_info(self) -> DetectorInfo:
        return self._detector.info

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault

    @property
    def stats(self) -> PipelineStats | None:
        return self._pipeline.stats if self._pipeline is not None else None

    # ---------------------------------------------------------------- control

    def start(self) -> None:
        if self._thread is not None:
            raise NodeError(
                f"{self.source_id} is already running. A runner is used once; "
                "construct another rather than restarting this one, so a "
                "half-stopped thread can never be revived underneath a new run."
            )
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name=f"camera:{self.source_id}", daemon=True
        )
        self._thread.start()

    def ask_to_stop(self) -> None:
        """Set the flag and return immediately. Blocks nothing.

        Separate from :meth:`stop` so a node with sixteen cameras can signal all
        sixteen and then wait once, rather than waiting up to the timeout for
        each in turn — which is the difference between a ten-second shutdown and
        a three-minute one.
        """
        with self._lock:
            self._stopping = True

    def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> bool:
        """Ask the run to end and wait for it. Returns whether it actually ended.

        The return value is the point. Dropping the reference to a thread that
        has not finished leaves it decoding into a pipeline whose store may
        already be closed.
        """
        self.ask_to_stop()
        if self._thread is None:
            return True

        self._thread.join(timeout=timeout)
        ended = not self._thread.is_alive()
        if not ended:
            _log.error(
                "%s: the analysis thread did not stop within %.0fs; it is "
                "probably blocked in a decode", self.source_id, timeout,
            )
        return ended

    def set_pose(self, pose: CameraPose | None) -> None:
        """Place or move the camera while it is running.

        Applied by the analysis thread between frames rather than here, so the
        tracker is never mutated from under a call that is using it. Existing
        tracks keep their identity: placing a camera does not turn the people it
        was already following into different people.
        """
        with self._lock:
            self._pending_pose = pose
            self._pose_changed = True

    # -------------------------------------------------------------- collection

    def take_latest(self) -> Update | None:
        """The newest result, or ``None`` if nothing new has arrived.

        Taking clears the slot, so a slow collector skips rather than falling
        behind.
        """
        with self._lock:
            update, self._latest = self._latest, None
            return update

    def take_segments(self) -> list:
        """Recorded segments closed since the last call.

        Handed across exactly like events, and for a harder reason: indexing a
        segment is a database write, and **an SQLite connection belongs to the
        thread that created it**. Passing `store.save_segment` down into the
        pipeline put that write on this thread, where every one of them failed
        with a `ProgrammingError` that `recording.py` caught and logged — so the
        clips existed on disk and none of them were indexed, which means nothing
        could find them and retention would have deleted them as unreferenced.

        That failure has now happened twice, in two different callers. The rule
        it establishes: **only the thread that owns the node touches the store.**
        Everything else queues.
        """
        with self._lock:
            segments, self._new_segments = self._new_segments, []
            return segments

    def take_events(self) -> list[Event]:
        """Events raised since the last call.

        Drained rather than accumulated here: the node keeps the history,
        because correlation is across cameras and no single camera can do it.
        """
        with self._lock:
            events, self._new_events = self._new_events, []
            return events

    # ------------------------------------------------------------------- work

    def _collect_segment(self, segment) -> None:
        """Queue a finished segment for the node's thread. Never writes."""
        with self._lock:
            self._new_segments.append(segment)

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stopping

    def _publish(self, update: Update) -> None:
        with self._lock:
            if self._latest is not None:
                self._skipped += 1
            self._latest = update

    def _run(self) -> None:
        pipeline = Pipeline(
            self._source,
            self._detector,
            pose=self._pose,
            keep_images=self._keep_images,
            zones=self._zones,
            rules=self._rules,
            node_id=self._node_id,
            record_to=self._record_to,
            segment_seconds=self._segment_seconds,
            on_segment=self._collect_segment,
        )
        self._pipeline = pipeline

        started_wall = time.perf_counter()
        first_stamp: int | None = None
        recent: list[float] = []

        try:
            with pipeline:
                for result in pipeline.run():
                    if self._should_stop():
                        break

                    with self._lock:
                        if self._pose_changed:
                            pipeline.set_pose(self._pending_pose)
                            self._pose_changed = False
                        if result.events:
                            self._new_events.extend(result.events)
                        skipped = self._skipped

                    if self._realtime:
                        # Pace a file to its own timeline. Without this, twelve
                        # seconds of footage flashes past in two, which is right
                        # for batch review and useless for watching.
                        if first_stamp is None:
                            first_stamp = result.timestamp_millis
                        target = (result.timestamp_millis - first_stamp) / 1000.0
                        drift = target - (time.perf_counter() - started_wall)
                        if drift > 0:
                            time.sleep(min(drift, 1.0))

                    now = time.perf_counter()
                    recent.append(now)
                    # A rate over the last second, not an average since start-up
                    # — an average hides a stall behind a healthy beginning.
                    recent = [stamp for stamp in recent if now - stamp <= 1.0]

                    self._publish(
                        Update(
                            result=result,
                            analysis_fps=float(len(recent)),
                            skipped=skipped,
                            stats=pipeline.stats,
                        )
                    )
        except DecodeError as error:
            # Redacted by construction: DecodeError never carries a URL that
            # still has its credential in it.
            with self._lock:
                self._fault = str(error)
            _log.warning("%s: analysis stopped — %s", self.source_id, error)
        except Exception as error:  # noqa: BLE001
            # The type, never the message: an arbitrary exception's text may
            # have been built from a URL that carries a password.
            with self._lock:
                self._fault = (
                    f"{self.source_id} stopped unexpectedly "
                    f"({type(error).__name__})"
                )
            _log.error("%s: analysis raised", self.source_id, exc_info=True)


class Node:
    """Everything a machine runs, with nothing on screen.

    Use as a context manager. Adding a camera does not start it; `start` does,
    and `poll` is what moves work forward — draining events, persisting them and
    correlating on a cadence. A caller with an interface polls on a repaint
    timer; a daemon calls `run_forever`.
    """

    def __init__(
        self,
        database: str | Path | None = None,
        *,
        node_id: str = "local",
        zones: Sequence[Zone] = (),
        rules: Sequence[Rule] | None = None,
        detector_factory=None,
        record_to: str | Path | None = None,
        segment_seconds: float = 60.0,
        keep_images: bool = False,
        realtime: bool = False,
        restore_cameras: bool = True,
        correlate_every_millis: int = DEFAULT_CORRELATE_MILLIS,
        event_retention: int = 5000,
        actor: str = ACTOR,
    ):
        """
        ``detector_factory`` is called once per camera. One detector per camera,
        never shared: MOG2 carries a per-pixel model of *its* scene, and feeding
        it two cameras corrupts both models and every detection that comes out
        of them.

        ``rules`` defaults to :func:`~sentinel.events.default_rules` over the
        zones given, so a node configured with no rules still does something
        sensible rather than nothing silently.

        ``zones`` left empty means *load whatever this node already had*, not
        *watch nothing* — a restarted node must come back watching what it was
        watching. Passing zones explicitly replaces that.

        ``restore_cameras`` brings back the cameras and, crucially, their poses.
        Placements were written to the database and never read, so a console
        restart silently lost every one of them while zones survived: the
        cameras came back, unplaced, and reported objects as *not placed* with
        no indication that anything had been forgotten.
        """
        self.store = Store(database if database is not None else default_database_path())
        self._node_id = node_id
        self._zones: list[Zone] = list(zones)
        self._rules = list(rules) if rules is not None else default_rules(self._zones)
        self._detector_factory = detector_factory or (lambda: MotionDetector())
        self._record_to = Path(record_to) if record_to else None
        self._segment_seconds = segment_seconds
        self._keep_images = keep_images
        self._realtime = realtime
        self._correlate_every = correlate_every_millis
        self._event_retention = event_retention
        # The same code is a daemon, a console and a CLI, and an audit row that
        # says "node" when an operator pressed a button is a false entry in a
        # chain of custody. Whoever owns this node names themselves.
        self._actor = actor

        self._cameras: dict[str, CameraRecord] = {}
        self._restored_cameras = 0
        self._incidents: tuple[Incident, ...] = ()
        self._persisted: set[str] = set()
        self._last_correlated = 0.0
        self._running = False
        self._stop = threading.Event()

        if zones:
            for zone in self._zones:
                self.store.save_zone(zone)
        else:
            # Nothing was passed, so take what this node already had. A daemon
            # restarted at 03:00 must come back watching the same zones it was
            # watching at 02:59, and asking the caller to re-supply them makes
            # every caller responsible for remembering.
            self._zones = list(self.store.zones())
            if self._zones:
                self._rules = default_rules(self._zones)

        if restore_cameras:
            self._restore_cameras()

        self.store.audit(self._actor, "node.started", node_id)
        _log.info(
            "node %s: %d zone(s), %d rule(s), recording %s",
            node_id, len(self._zones), len(self._rules),
            self._record_to or "off",
        )

    # ------------------------------------------------------------------ state

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def cameras(self) -> tuple[CameraRecord, ...]:
        return tuple(self._cameras.values())

    @property
    def zones(self) -> tuple[Zone, ...]:
        return tuple(self._zones)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    @property
    def incidents(self) -> tuple[Incident, ...]:
        return self._incidents

    @property
    def is_running(self) -> bool:
        return self._running

    def camera(self, camera_id: str) -> CameraRecord:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise NodeError(f"no camera {camera_id!r} on this node") from None

    def _restore_cameras(self) -> None:
        """Bring back the cameras this node had, with their placements."""
        for row in self.store.cameras():
            camera_id = row["id"]
            if camera_id in self._cameras:
                continue
            # `source` is the *redacted* form — the raw one, credential and all,
            # was never persisted and must not be. A network camera therefore
            # comes back needing its password again, and says so rather than
            # failing at connect time with something unhelpful.
            self._cameras[camera_id] = CameraRecord(
                camera_id=camera_id,
                source=row["source"],
                pose=self.store.camera_pose(camera_id),
            )
            self._restored_cameras += 1

        if self._restored_cameras:
            placed = sum(1 for r in self._cameras.values() if r.pose is not None)
            _log.info(
                "node %s: restored %d camera(s), %d placed",
                self._node_id, self._restored_cameras, placed,
            )

    @property
    def restored_cameras(self) -> int:
        """How many cameras came back from the database rather than being added."""
        return self._restored_cameras

    def needs_credentials(self, camera_id: str) -> bool:
        """Whether this camera cannot be started without its password again.

        A restored network camera carries the redacted source, because the real
        one was never stored — that is the point of the credential rule. It has
        to be re-entered, and an interface should say so *before* the operator
        presses Start rather than after the connection fails.
        """
        record = self.camera(camera_id)
        return REDACTED in record.source

    # ---------------------------------------------------------------- cameras

    def add_camera(
        self, source: str | Path, *, camera_id: str | None = None,
        pose: CameraPose | None = None,
    ) -> CameraRecord:
        """Register a camera. Does not start it, and opens nothing."""
        text = str(source)
        identifier = camera_id or f"cam-{len(self._cameras) + 1:02d}"
        if identifier in self._cameras:
            raise NodeError(
                f"there is already a camera called {identifier!r} on this node"
            )

        record = CameraRecord(camera_id=identifier, source=text, pose=pose)
        self._cameras[identifier] = record

        # The redacted form, never the raw one: this row is read by anything
        # that lists cameras, and a password in it is a password on a screen.
        self.store.save_camera(identifier, identifier, record.display_source, pose=pose)
        self.store.audit(self._actor, "camera.added", identifier, record.display_source)
        _log.info("node %s: added camera %s (%s)", self._node_id, identifier,
                  record.display_source)
        return record

    def place_camera(self, camera_id: str, pose: CameraPose | None) -> None:
        """Set where a camera is and where it points, running or not."""
        record = self.camera(camera_id)
        record.pose = pose
        if record.runner is not None:
            record.runner.set_pose(pose)

        self.store.save_camera(
            camera_id, camera_id, record.display_source, pose=pose
        )
        self.store.audit(
            self._actor, "camera.placed", camera_id,
            f"{pose.position.lat:.6f},{pose.position.lon:.6f} "
            f"h={pose.mount_height} hdg={pose.heading} pitch={pose.pitch}"
            if pose else "unplaced",
        )

    def add_zone(self, zone: Zone) -> None:
        """Add a zone, and rebuild the rule set if it was the first one.

        A node that had no zones was given the rules that need none. Adding one
        without revisiting that would leave the zone unwatched — configured,
        visible, and silently doing nothing, which is the worst state for a
        security control to be in.
        """
        had_none = not self._zones
        self._zones.append(zone)
        self.store.save_zone(zone)
        self.store.audit(self._actor, "zone.created", zone.id, zone.name)

        if had_none:
            self._rules = default_rules(self._zones)
            _log.info(
                "node %s: first zone added; rule set is now %d rule(s)",
                self._node_id, len(self._rules),
            )
        if self._running:
            _log.warning(
                "node %s: zone %s applies to cameras started from now on; "
                "restart a camera for it to take effect there",
                self._node_id, zone.id,
            )

    def probe(self, camera_id: str) -> None:
        """Open a camera and close it again, or raise.

        A daemon must not block its start-up on a camera that is not there — an
        unreachable RTSP host costs seconds, and twenty of them cost minutes.
        An *interface* wants the opposite: the operator asked for this, just
        now, and is waiting, so a bad source should say so immediately rather
        than appear as a fault banner a moment later.

        Both are right, so this is a separate call and neither `start` nor
        `run_forever` makes it. The console probes; the daemon does not.
        """
        record = self.camera(camera_id)
        source = VideoSource(record.source, source_id=camera_id)
        try:
            source.open()
        finally:
            source.close()

    # --------------------------------------------------------------- lifecycle

    def __enter__(self) -> "Node":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> int:
        """Start every camera that is not already running. Returns how many."""
        if not self._cameras:
            raise NodeError("this node has no cameras to start")

        started = 0
        for record in self._cameras.values():
            if record.is_running:
                continue
            record.fault = None
            source = VideoSource(record.source, source_id=record.camera_id)
            record.runner = CameraRunner(
                source,
                # One detector per camera, never shared.
                self._detector_factory(),
                record.pose,
                zones=self._zones,
                rules=self._rules,
                node_id=self._node_id,
                keep_images=self._keep_images,
                realtime=self._realtime,
                record_to=self._record_to,
                segment_seconds=self._segment_seconds,
            )
            record.runner.start()
            started += 1

        self._running = started > 0
        self._stop.clear()
        self.store.audit(self._actor, "analysis.started", detail=f"{started} camera(s)")
        _log.info("node %s: started %d camera(s)", self._node_id, started)
        return started

    def stop(self) -> bool:
        """Stop every camera. Returns whether all of them actually ended.

        **Two passes, deliberately.** Signalling everything before waiting on
        anything bounds the shutdown at roughly one timeout rather than one per
        camera; sixteen cameras behind a switch that has just lost power would
        otherwise take three minutes.

        **The runner is not released.** An earlier version set
        ``record.runner = None`` here, and the final drain in `close` and
        `run_forever` then found nothing to drain — so every event raised in the
        last poll window was discarded, along with any incident that would have
        been correlated from it. A file ending on an intrusion recorded nothing.
        The thread has ended; keeping the object costs nothing and keeps its
        stats and its fault readable afterwards, which is what `summary` needs.
        """
        self._stop.set()

        for record in self._cameras.values():
            if record.runner is not None:
                record.runner.ask_to_stop()

        ended = True
        stubborn: list[str] = []
        for record in self._cameras.values():
            if record.runner is None:
                continue
            if not record.runner.stop():
                ended = False
                stubborn.append(record.camera_id)

        if stubborn:
            # Named, and left running rather than killed: the thread holds a
            # decoder and a database handle, and terminating it takes both.
            self.store.audit(
                self._actor, "analysis.thread_stuck", ", ".join(stubborn),
                "left running rather than killed; it holds a decoder",
            )

        self._running = False
        self.store.audit(self._actor, "analysis.stopped")
        _log.info("node %s: stopped", self._node_id)
        return ended

    def close(self) -> None:
        """Stop, correlate one last time, and close the database.

        The final correlation matters: events raised in the last seconds of a
        run would otherwise be persisted with no incident referring to them,
        which reads afterwards as a period where nothing was concluded rather
        than one where the process ended.
        """
        if self._running:
            self.stop()
        try:
            self.poll(force_correlate=True)
        finally:
            self.store.audit(self._actor, "node.stopped", self._node_id)
            self.store.close()

    # -------------------------------------------------------------------- work

    def poll(self, *, force_correlate: bool = False) -> tuple[Update, ...]:
        """Move everything forward one step, and return the newest frames.

        Drains each camera's events, persists them, notices faults, and
        correlates when due. Called on a timer by an interface, and in a loop by
        a daemon — the same work either way, which is the point of the class.
        """
        updates: list[Update] = []
        fresh: list[Event] = []

        for record in self._cameras.values():
            if record.runner is None:
                continue

            update = record.runner.take_latest()
            if update is not None:
                updates.append(update)

            for segment in record.runner.take_segments():
                # On this thread, which is the only one allowed to write.
                self.store.save_segment(segment)

            events = record.runner.take_events()
            if events:
                record.events.extend(events)
                fresh.extend(events)
                # Bounded: a node running for a month must not hold every event
                # it ever raised in memory. Persistence is the full history.
                if len(record.events) > self._event_retention:
                    record.events = record.events[-self._event_retention:]

            fault = record.runner.fault
            if fault is not None and record.fault != fault:
                record.fault = fault
                _log.warning("node %s: %s", self._node_id, fault)

        if fresh:
            self.store.save_events(fresh)

        due = (time.monotonic() - self._last_correlated) * 1000 >= self._correlate_every
        if fresh or force_correlate or due:
            self.correlate()

        return tuple(updates)

    def correlate(self) -> tuple[Incident, ...]:
        """Group every camera's events into incidents, across the whole node.

        Across all cameras, deliberately. A camera correlating only its own
        events raises one incident per camera for one intrusion, which is
        exactly the duplication this stage exists to remove.
        """
        self._last_correlated = time.monotonic()

        events = [event for record in self._cameras.values() for event in record.events]
        if not events:
            self._incidents = ()
            return self._incidents

        correlator = Correlator(zone_kinds={zone.id: zone.kind for zone in self._zones})
        self._incidents = tuple(correlator.correlate(events))

        # Idempotent every time: ids are deterministic, so re-correlating a
        # growing window upserts the same incident rather than accumulating a
        # new one each pass.
        for incident in self._incidents:
            self.store.save_incident(incident)
            if incident.id not in self._persisted:
                self._persisted.add(incident.id)
                self.store.audit(
                    self._actor, "incident.opened", incident.id, incident.summary
                )
                _log.info(
                    "node %s: incident %s — %s (%s, risk %.0f)",
                    self._node_id, incident.id, incident.summary,
                    incident.severity.value, incident.risk.score,
                )

        return self._incidents

    def run_forever(self, *, poll_millis: int = 200, until=None) -> None:
        """Run until stopped. What a daemon calls.

        ``until`` is an optional predicate checked each pass, so a caller can
        bound a run without a signal — which matters because on Windows an
        external SIGINT does not reach a Python process at all.
        """
        if not self._running:
            self.start()

        _log.info("node %s: running", self._node_id)
        try:
            while not self._stop.is_set():
                self.poll()

                if until is not None and until(self):
                    _log.info("node %s: stop condition met", self._node_id)
                    break
                if not any(record.is_running for record in self._cameras.values()):
                    # Every camera has ended. For files that is completion; for
                    # cameras it is total failure, and either way there is
                    # nothing left to poll.
                    _log.info("node %s: every camera has ended", self._node_id)
                    break

                self._stop.wait(poll_millis / 1000.0)
        except KeyboardInterrupt:
            _log.info("node %s: interrupted", self._node_id)
        finally:
            self.stop()
            self.poll(force_correlate=True)

    def export_incident(
        self, incident_id: str, destination: str | Path,
        *, lead_seconds: float = DEFAULT_LEAD_SECONDS,
        trail_seconds: float = DEFAULT_TRAIL_SECONDS,
    ):
        """Write an evidence package, with the footage that shows it.

        One implementation for every caller. The console's own export had
        neither half — no `footage=`, no `preserve_segments` — so a package
        produced from the interface contained no video, and the clips it was
        built from stayed deletable by the next retention pass.

        Preservation happens before the copy and stands even if the copy then
        fails. Over-preserving costs disk; under-preserving destroys evidence.
        """
        incident = next(
            (found for found in self._incidents if found.id == incident_id), None
        ) or self.store.incident(incident_id)
        if incident is None:
            raise NodeError(f"no incident {incident_id!r}")

        coverage = coverage_for(
            self.store, incident,
            lead_seconds=lead_seconds, trail_seconds=trail_seconds,
        )

        clips = [segment.path for cover in coverage for segment in cover.segments]
        if clips:
            preserved = self.store.preserve_segments(clips)
            self.store.audit(
                self._actor, "recording.preserved", incident.id,
                f"{preserved} segment(s) held as evidence and exempted from retention",
            )

        export = export_incident(
            incident, Path(destination), exported_by=self._actor, footage=coverage
        )
        self.store.audit(
            self._actor, "incident.exported", incident.id, str(export.directory)
        )
        return export, coverage

    def summary(self) -> str:
        """What happened, for a person to read."""
        lines = [f"node {self._node_id}"]
        for record in self._cameras.values():
            stats = record.runner.stats if record.runner else None
            state = "running" if record.is_running else "stopped"
            if record.fault:
                state = f"FAULT — {record.fault}"
            lines.append(f"  {record.camera_id:<16} {state}")
            lines.append(f"    source        {record.display_source}")
            lines.append(f"    placed        {'yes' if record.pose else 'no'}")
            if stats is not None:
                lines.append(
                    f"    frames        {stats.frames}, "
                    f"{stats.detections} detection(s), "
                    f"{stats.distinct_objects} object(s)"
                )
            lines.append(f"    events        {len(record.events)}")
        lines.append(f"  incidents       {len(self._incidents)}")
        return "\n".join(lines)
