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

**Identity is a site switch, and the node is where it is enforced.** Faces and
plates each existed as tested modules that nothing called: `sentinel.faces` had
an engine with a switch, `sentinel.plates` had a reader the pipeline would take,
and no production path built either — so the register could be enrolled into
and matched against nobody. The switch lives on the site row (`Site.identity`),
survives a restart, and every flip is an audit row with a before and an after.
Off means nothing runs, and the tests prove it the only way that counts: the
stand-in models were shown no pixels. On means the node hands a plate reader to
each pipeline it starts — behind the switch, so that off reaches a running
camera at its next read — and examines person tracks for faces itself, on its
own thread, because the register is a database and the database belongs to this
thread. Face work is confined to the box of a track the detector labelled a
person, at most once every :data:`FACE_STRIDE_FRAMES` frames per track, and the
templates it produces live only as long as the track does unless an operator
names it. What a running camera does about a switch turned *on* is what it was
started with, and :attr:`Node.identity_status` names every running camera the
switch has not reached rather than printing "on" over a site examining nobody.

What is deliberately *not* here: any network listener. A node that other machines
can talk to needs a control plane, mTLS and pairing, and none of that exists yet
— see ROADMAP 2.1. This is the half that had to come first, because there was no
object for an API to be an API *of*.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
from pathlib import Path
from typing import Iterator, Sequence

from . import paths
from .auditing import MISSING, AuditRecord
from .core import CameraPose, LatLon, Track
from .decode import REDACTED, DecodeError, VideoSource, is_live_source
from .detect import Detector, DetectorInfo, MotionDetector
from .evidence import (
    DEFAULT_LEAD_SECONDS,
    DEFAULT_TRAIL_SECONDS,
    coverage_for,
    export_incident,
)
from .events import Event, Rule, default_rules
from .faces import (
    EnrolledPerson,
    FaceBackend,
    FaceEngine,
    FaceError,
    FaceTemplate,
    TrackIdentity,
    Verdict,
    similarity,
)
from .incidents import Correlator, Incident
from .logs import get as _get_logger
from .pipeline import FrameResult, Pipeline, PipelineStats, TrackPlate
from .plates import PlateModels, PlateReader
from .registry import Confidence, Forgotten, Plate, RegistryError, Subject, SubjectKind
from .registry import FaceTemplate as StoredFaceTemplate
from .site import DEFAULT_SITE_ID, Identity, Site
from .store import Store, default_database_path
from .zones import Zone

_log = _get_logger(__name__)

#: How often correlation runs across every camera. Two seconds is far slower
#: than frame rate and far faster than a person notices.
DEFAULT_CORRELATE_MILLIS = 2000

#: How long `stop` waits for a camera thread to end before reporting that it did
#: not. A decode blocked on a stalled camera is the ordinary way that happens.
STOP_TIMEOUT_SECONDS = 10.0

#: How long a nominally running camera may produce nothing before it is called
#: dark rather than live. Thirty seconds is far longer than any gap a working
#: camera leaves — a stalled RTSP stream, a decoder wedged inside a driver — and
#: short enough that an operator finds out while it still matters.
DARK_AFTER_SECONDS = 30.0

#: How often a node that records sweeps its own recordings. Ten minutes is
#: far more often than a disk fills and far less often than a poll, and it is
#: the difference between retention that happens and retention that somebody
#: was supposed to run: `apply_retention` existed for three days as a command
#: nothing scheduled, with its own shortfall message reading "Recording will
#: continue until the disk is full and then stop."
RETENTION_EVERY_SECONDS = 600.0

#: Recorded in the audit log. There is no authentication yet, so there is nobody
#: to name; recording the truth beats inventing an operator, because an audit
#: trail with a false entry is worse than no audit trail.
ACTOR = "node"

#: Frames between face examinations of one person track.
#:
#: Face work is a detector and an embedder per person per frame, and at thirty
#: frames a second that is sixty model calls a second for one person standing
#: still — paid on the node's thread, which is also the thread that persists
#: events and correlates. Adjacent frames are not independent evidence either:
#: two frames a thirtieth of a second apart are two samples of one instant's
#: pose and lighting, and `faces.identify_track` wants corroboration across the
#: track, not agreement between neighbours. Every fifth frame keeps the cost
#: fixed per person and spreads the frames that do get examined across the
#: time the person is actually in shot. Per track rather than per frame, so
#: twelve people do not all come due at once and turn a rate limit into a
#: periodic stall.
FACE_STRIDE_FRAMES = 5

#: Face templates held per live person track, for enrolment and identification.
#:
#: Held in memory only, dropped when the track ends, and never written unless
#: an operator names the track — this is the whole difference between a
#: register and a gallery of strangers. Twelve is enough that the operator
#: naming a track gets its best-quality face rather than its first, and that
#: identification has :data:`~sentinel.faces.FRAMES_FOR_A_MATCH` several times
#: over; it is few enough that a hundred people in shot cost a few hundred
#: kilobytes rather than the frames they were seen in.
FRAMES_KEPT_FOR_ENROLMENT = 12

#: The two files the face path expects in the models directory. Both are the
#: operator's to supply; nothing here fetches them.
FACE_MODEL_FILES = ("yunet.onnx", "sface.onnx")

#: The three files the plate path expects in the models directory.
PLATE_MODEL_FILES = ("plate-detector.onnx", "plate-recogniser.onnx", "plate-charset.txt")


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


class CameraState(StrEnum):
    """The one word an operator gets for a camera, and what it is allowed to mean.

    Five states rather than two, because "running" was the lie this exists to
    correct: a camera whose thread is alive and whose decoder has produced
    nothing for a minute was reported exactly like one delivering thirty frames
    a second, and the console's own strip showed it green. The distinction that
    matters is not running versus stopped — it is *producing frames* versus
    *not*, and only the last-frame clock knows which.

    :class:`~enum.StrEnum`, not ``(str, Enum)``, and the difference is visible
    to an operator: with the mixin, ``f"{CameraState.LIVE}"`` renders
    ``"CameraState.LIVE"`` on this interpreter — measured — so the first
    f-string an interface writes around the state puts a Python repr on the
    status strip. Comparison against ``"LIVE"`` is unaffected.
    """

    #: The thread is alive and frames have arrived recently. The only state that
    #: entitles a camera to claim ground on the map.
    LIVE = "LIVE"
    #: Alive, and nothing has arrived for :data:`DARK_AFTER_SECONDS`. This is the
    #: failure an operator must see, and the reason this enum has five members.
    DARK = "DARK"
    #: Alive, no frame yet, still inside the grace window. Opening an RTSP
    #: stream takes seconds, and calling that camera dark for those seconds
    #: would teach an operator to ignore the word.
    STARTING = "STARTING"
    #: Not running, and it did not fail: never started, or asked to stop, or a
    #: file that reached its end. Says nothing about whether it once worked.
    STOPPED = "STOPPED"
    #: The run ended in, or was refused because of, a named fault. Distinct from
    #: STOPPED because a camera nobody started and a camera that died overnight
    #: are not the same fact.
    FAULTED = "FAULTED"


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """Whether one camera is actually working, in the terms an operator needs.

    Assembled rather than stored, so it cannot go stale: every field is read
    from the runner and the pipeline at the moment it is asked for. Frozen
    because a status strip that can write back into the node is a status strip
    that will eventually place a camera by accident.

    Nothing here is inferred. Each number is one the pipeline or the runner
    already measured — the alternative, an interface computing a plausible-looking
    frame rate of its own, is how a display ends up disagreeing with the log
    about the same camera.
    """

    camera_id: str
    state: CameraState
    #: Its thread is alive. Deliberately *not* the same as working — see `state`.
    is_running: bool
    #: It has a pose, so its detections are locations rather than sightings.
    is_placed: bool
    #: Frames analysed per second over the last second, or 0.0 when there is no
    #: measurement current enough to quote. A rate from a minute ago is a claim
    #: about now that the data does not support.
    analysis_fps: float
    #: Frames the pipeline actually analysed, for the whole run.
    frames: int
    #: Frames the live decoder discarded to stay current, because the analytic
    #: was behind. Always 0 for a file, which drops nothing.
    frames_dropped: int
    #: Results that were published and never collected, because the viewer was
    #: busy. Nothing was lost from the analysis — only from the screen — but a
    #: display showing a third of the frames must say so.
    frames_not_drawn: int
    #: How many times the live reader had to re-open the stream. A camera
    #: reconnecting every few seconds is failing even while it looks alive.
    reconnects: int
    #: Why it stopped, or why it was never started. Redacted by construction.
    fault: str | None
    #: Since the last frame arrived. ``None`` when none ever has — which is a
    #: different fact from "a long time ago", and the interface must not print
    #: it as a number.
    seconds_since_frame: float | None
    #: Since its thread was started, or ``None`` if it never was. This is what
    #: makes DARK defensible for a camera that has produced nothing at all.
    seconds_since_started: float | None
    #: The stored flag: this camera records when it runs. Distinct from
    #: `recording`, which is whether a recorder is open right now — the flag
    #: can be on for a camera that has not been restarted since it was set.
    asked_to_record: bool = False
    #: A recorder is attached to the running pipeline.
    recording: bool = False
    clips_written: int = 0
    bytes_recorded: int = 0
    #: Why the recorder stopped early, or ``None``. A camera whose writer died
    #: in the first minute must not report the same line as one recording all
    #: night.
    recording_fault: str | None = None

    @property
    def is_dark(self) -> bool:
        """Whether the map must hatch it and drop it from coverage.

        A camera that is nominally running but silent still has a pose, and
        painting its footprint as covered ground is the specific dishonesty this
        answers: the ground is not being watched, and an operator reading that
        map would believe it was.
        """
        return self.state in (CameraState.DARK, CameraState.FAULTED)

    @property
    def covers_ground(self) -> bool:
        """Whether ground is being watched *right now*. Not "draw this camera".

        Placed *and* producing frames. Either half alone is a footprint filled
        in over ground nobody is watching.

        **It is deliberately not the draw-at-all test**, and reading it as one
        erases the entire coverage map whenever the node is stopped — including
        the moment an operator is placing cameras and most needs to see what
        they would cover. The map draws the *planned* footprint from the pose,
        as it does today, and uses this only to decide how: filled where this is
        True, hatched where :attr:`is_dark`, dimmed or outlined where the camera
        is placed and merely STOPPED.

        A camera silent for twenty-nine seconds still claims its ground, on
        purpose. :data:`DARK_AFTER_SECONDS` is the one threshold in this module;
        a second, quieter one for the map would have the map and the status
        strip disagreeing about the same camera, and an operator cannot tell
        which of the two is lying.
        """
        return self.is_placed and self.state is CameraState.LIVE

    def describe(self) -> str:
        """One line for a status strip, with the reason attached.

        The state word alone sends an operator to the camera to find out why;
        the seconds and the fault are what stop that trip. No number appears
        here that the last-frame clock contradicts — a rate is only quoted while
        the measurement window it came from is still current, and otherwise the
        silence is quoted instead.
        """
        parts = [self.state.value]
        if self.state is CameraState.FAULTED and self.fault:
            parts.append(self.fault)
        elif self.state in (CameraState.LIVE, CameraState.DARK):
            silent = (
                self.seconds_since_frame
                if self.seconds_since_frame is not None
                else self.seconds_since_started
            )
            if silent is None:
                parts.append("no frame ever")
            elif silent > 1.0:
                # The rate window is one second wide, so a camera that last
                # delivered twenty seconds ago has a *measured* rate of zero and
                # is still inside the grace period before DARK. Quoting it read
                # "LIVE — 0 fps", a line that contradicts itself in four words.
                # The silence is the fact that is actually known, so say that.
                parts.append(f"no frame for {silent:.0f}s")
            else:
                parts.append(f"{self.analysis_fps:.0f} fps")
        if self.reconnects:
            parts.append(f"{self.reconnects} reconnect(s)")
        if not self.is_placed:
            parts.append("not placed")
        if self.recording_fault:
            parts.append(f"recording stopped: {self.recording_fault}")
        elif self.recording:
            parts.append(f"recording · {self.clips_written} clip(s)")
        elif self.asked_to_record and self.is_running:
            # Set after the camera started: the flag reaches a pipeline when it
            # is built, and saying "recording" here would be the one lie a
            # status line about evidence must not tell.
            parts.append("will record when restarted")
        return " — ".join(parts)


@dataclass
class CameraRecord:
    """A camera this node knows about, running or not."""

    camera_id: str
    #: A file path, an `rtsp://` URL, or `device:N`. May carry a credential, so
    #: it is never logged, displayed or stored — `display_source` is.
    source: str
    pose: CameraPose | None = None
    #: Records continuously when it runs. Stored with the camera (migration
    #: 11), read by `Node.start`, set by `Node.set_recording`.
    record: bool = False
    #: The opaque handle the camera's password is filed under in the operating
    #: system's keychain, or ``None`` when there is no password or no keychain.
    #: Never the password. See `sentinel.secrets`.
    credentials_ref: str | None = None
    #: How many runners this record has been given. Track ids and frame
    #: indices start again with each one, so anything the node keeps keyed on
    #: a track id — a refusal, an encounter already audited — is keyed on this
    #: as well. Without it a camera restarted in the same process handed its
    #: new track 3 whatever the old track 3 had earned: a known van read
    #: confidently on a track id that once carried a refused read was never
    #: sighted, and nothing said so.
    run: int = field(default=0, init=False)
    runner: "CameraRunner | None" = None
    #: The identity switch as it stood when this camera's runner was built.
    #: Snapshotted at `Node.start` because the plate reader is handed to the
    #: pipeline then and cannot be added to a running one, so what a running
    #: camera does about a switch turned *on* is what it was started with —
    #: and `Node.identity_status` names every running camera the switch has
    #: not reached. *Off* is different, deliberately, in both directions:
    #: faces off stops face work on every camera at the next poll, because
    #: that work runs on the node's thread; plates off stops the reader every
    #: pipeline holds at its next read, because the reader is handed over
    #: behind the switch (`_SwitchedPlateReader`). A switch-off that waited
    #: for a restart would leave a model running on a site that had just said
    #: no, with a status line reading "off" above it.
    identity: Identity = Identity()
    #: Events this camera has raised, kept so correlation can run across the
    #: whole node rather than within one camera.
    events: list[Event] = field(default_factory=list)
    #: Set when this camera's own run ends or fails, so an interface can show
    #: *which* camera is in trouble rather than only that something is.
    fault: str | None = None

    def __setattr__(self, name: str, value: object) -> None:
        # Counted on assignment rather than in `Node.start`, so that a runner
        # swapped in by any other route — a test's stand-in included — is a
        # new run too. Clearing the runner, or setting the same one again, is
        # not a run.
        if (
            name == "runner"
            and value is not None
            and value is not self.__dict__.get("runner")
        ):
            object.__setattr__(self, "run", self.__dict__.get("run", 0) + 1)
        object.__setattr__(self, name, value)

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
        "_pipeline", "_new_events", "_new_segments", "_site_tz",
        "_started_at", "_last_frame_at", "_analysis_fps", "_plate_reader",
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
        site_tz=None,
        plate_reader: PlateReader | None = None,
    ):
        """
        ``keep_images`` is off by default. A node with nobody watching has no use
        for a full-resolution image per result, and it is tens of megabytes over
        a short clip.

        ``realtime`` paces a file to its own timeline. Off by default, because a
        node analysing recorded footage should go as fast as it can; a *viewer*
        wants it paced, and turns this on.

        ``plate_reader`` goes to the pipeline as it is. `Pipeline` has accepted
        one since the plate path was built, and until the node passed it
        nothing did, so plate reading was dead in every production run while
        every test of it passed. ``None`` is the off state and costs nothing.
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
        self._site_tz = site_tz
        self._plate_reader = plate_reader

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
        # Monotonic, never wall clock: health is measured in elapsed seconds,
        # and a machine that syncs its clock mid-run would otherwise report a
        # camera as silent for two hours or as having produced a frame in the
        # future.
        self._started_at: float | None = None
        self._last_frame_at: float | None = None
        self._analysis_fps = 0.0

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

    @property
    def recorder_stats(self):
        """What the recorder did, or ``None`` when this run has no recorder."""
        pipeline = self._pipeline
        recorder = pipeline.recorder if pipeline is not None else None
        return recorder.stats if recorder is not None else None

    @property
    def recording_fault(self) -> str | None:
        """Why a recorder that was asked for never started, or ``None``."""
        pipeline = self._pipeline
        return pipeline.recording_fault if pipeline is not None else None

    @property
    def skipped(self) -> int:
        """Results published that nobody ever collected.

        Read separately from `take_latest` because the count belongs to the
        camera, not to whichever `Update` happened to be in the slot: a viewer
        that never polls would otherwise see a skip count of zero forever.
        """
        with self._lock:
            return self._skipped

    @property
    def analysis_fps(self) -> float:
        """The last measured rate, or 0.0 if nothing has been measured yet.

        Stale by construction once frames stop arriving — a rate is a
        measurement over a window, and there is no window without frames. Whoever
        quotes it must check `seconds_since_frame` first; `Node.camera_health`
        does.
        """
        with self._lock:
            return self._analysis_fps

    @property
    def seconds_since_frame(self) -> float | None:
        """How long since a frame was analysed, or ``None`` if none ever was.

        The one number that separates a camera that is working from a camera
        whose thread is merely alive. `None` is deliberately not `inf` and not a
        large number: "it has never produced a frame" and "it stopped producing
        frames" are different faults with different first questions.
        """
        with self._lock:
            if self._last_frame_at is None:
                return None
            return max(0.0, time.monotonic() - self._last_frame_at)

    @property
    def seconds_since_started(self) -> float | None:
        """How long the thread has been up, or ``None`` if it never started.

        Opening a stream takes seconds, so this is what keeps a camera three
        seconds into its first connection from being reported as dark.
        """
        with self._lock:
            if self._started_at is None:
                return None
            return max(0.0, time.monotonic() - self._started_at)

    @property
    def dropped_frames(self) -> int:
        """Frames the live reader discarded to stay current. 0 for a file.

        A file is iterated frame by frame and drops nothing; a camera drops
        whatever arrived while the analytic was busy. Reporting the file's zero
        as "unknown" would make every replay look suspect, and reporting a
        camera's drops as zero would hide the failure the counter exists for.
        """
        stream = self._pipeline.stream if self._pipeline is not None else None
        return stream.dropped_frames if stream is not None else 0

    @property
    def reconnects(self) -> int:
        """How many times the live reader re-opened the stream. 0 for a file.

        A camera that reconnects every few seconds looks alive in every other
        measure — the thread runs, frames arrive, the fault stays `None` — and
        is losing most of what it sees.
        """
        stream = self._pipeline.stream if self._pipeline is not None else None
        return stream.reconnects if stream is not None else 0

    # ---------------------------------------------------------------- control

    def start(self) -> None:
        if self._thread is not None:
            raise NodeError(
                f"{self.source_id} is already running. A runner is used once; "
                "construct another rather than restarting this one, so a "
                "half-stopped thread can never be revived underneath a new run."
            )
        self._stopping = False
        # Before the thread, not inside it: a camera whose first `open()` blocks
        # for thirty seconds has been running for thirty seconds, and health has
        # to be able to say so while it is happening.
        with self._lock:
            self._started_at = time.monotonic()
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
            pipeline = self._pipeline

        # The flag alone only stops the loop *between* frames, and a camera that
        # has gone quiet delivers no frames to be between. The pipeline's own
        # read wait has to be interrupted or shutdown blocks for the full frame
        # timeout on every silent camera.
        if pipeline is not None:
            pipeline.ask_to_stop()

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
            # Stamped on publication rather than on collection, because health
            # must describe the camera and not the viewer: a node with nobody
            # polling it is still producing frames, and reading the clock in
            # `take_latest` would have called every headless camera dark.
            self._last_frame_at = time.monotonic()
            self._analysis_fps = update.analysis_fps

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
            site_tz=self._site_tz,
            plate_reader=self._plate_reader,
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


def _health_for(record: CameraRecord) -> CameraHealth:
    """One camera's facts, read at the moment of asking. Never cached.

    A cached health record is a status strip that keeps saying LIVE after the
    camera has gone — which is the failure the whole type exists to prevent, so
    the assembly is deliberately cheap enough to run on a repaint timer.
    """
    runner = record.runner
    running = record.is_running
    # The runner's fault is the thread's own account and wins; the record's is
    # what the last poll saw, plus the faults the node sets itself for a camera
    # it refused to start at all.
    fault = (runner.fault if runner is not None else None) or record.fault

    since_frame = runner.seconds_since_frame if runner is not None else None
    since_start = runner.seconds_since_started if runner is not None else None

    if fault is not None:
        state = CameraState.FAULTED
    elif not running:
        state = CameraState.STOPPED
    else:
        # For a camera that has produced nothing at all, silence is measured
        # from the moment its thread started — otherwise a camera that never
        # connects would sit at STARTING forever, which is precisely the
        # green-light-on-a-dead-camera this replaces.
        silent_for = since_frame if since_frame is not None else since_start
        if silent_for is not None and silent_for >= DARK_AFTER_SECONDS:
            state = CameraState.DARK
        elif since_frame is None:
            state = CameraState.STARTING
        else:
            state = CameraState.LIVE

    # The rate is measured over the last second and means nothing outside it. No
    # frame in the last second is not a missing measurement — it is a measured
    # zero, and quoting the old number instead is how a stalled camera keeps
    # showing thirty frames a second on a strip.
    fps = 0.0
    if running and since_frame is not None and since_frame <= 1.0:
        assert runner is not None  # `is_running` is false without one
        fps = runner.analysis_fps

    stats = runner.stats if runner is not None else None
    recorder = getattr(runner, "recorder_stats", None) if runner is not None else None
    return CameraHealth(
        camera_id=record.camera_id,
        state=state,
        is_running=running,
        is_placed=record.pose is not None,
        analysis_fps=fps,
        frames=stats.frames if stats is not None else 0,
        frames_dropped=runner.dropped_frames if runner is not None else 0,
        frames_not_drawn=runner.skipped if runner is not None else 0,
        reconnects=runner.reconnects if runner is not None else 0,
        fault=fault,
        seconds_since_frame=since_frame,
        seconds_since_started=since_start,
        asked_to_record=record.record,
        recording=recorder is not None and running and recorder.fault is None,
        clips_written=recorder.segments_written if recorder is not None else 0,
        bytes_recorded=recorder.bytes_written if recorder is not None else 0,
        recording_fault=(
            (recorder.fault if recorder is not None else None)
            or (getattr(runner, "recording_fault", None) if runner is not None else None)
        ),
    )


def _describe_zone_change(before: Zone, after: Zone) -> str:
    """What a zone edit used to read as, kept as the sentence to match.

    `Node.replace_zone` no longer calls this: it writes the audit row through
    `auditing.AuditRecord`, which diffs the two zones and renders the same line
    from the changes, so the prose and the structured record cannot drift into
    disagreeing about what happened.

    This stays because it is the baseline that comparison is measured against —
    `test_auditing` runs both spellings over the same pair of zones and asserts
    they are character-identical, and names the three cases where they are not.
    Deleting it would leave that comparison with nothing to compare to, and the
    prose could then change voice on operators without a single test failing.

    Every field, because a zone quietly shrinking to exclude the door it was
    drawn around is exactly the edit an audit log exists to record — and
    "name -> name" would have hidden it.
    """
    parts: list[str] = []
    if before.name != after.name:
        parts.append(f"name {before.name!r} -> {after.name!r}")
    if before.kind != after.kind:
        parts.append(f"kind {before.kind.value} -> {after.kind.value}")
    if before.ring != after.ring:
        parts.append(f"outline {len(before.ring)} -> {len(after.ring)} points, moved")
    if before.schedule != after.schedule:
        parts.append(
            "schedule "
            + (before.schedule.describe() if before.schedule else "always")
            + " -> "
            + (after.schedule.describe() if after.schedule else "always")
        )
    if before.enter_after_millis != after.enter_after_millis:
        parts.append(f"dwell {before.enter_after_millis} -> {after.enter_after_millis} ms")
    if before.exit_after_millis != after.exit_after_millis:
        parts.append(f"exit {before.exit_after_millis} -> {after.exit_after_millis} ms")
    if before.accept_uncertain != after.accept_uncertain:
        parts.append(f"accept uncertain {before.accept_uncertain} -> {after.accept_uncertain}")
    if getattr(before, "classes", frozenset()) != getattr(after, "classes", frozenset()):
        # A filter edit is the one that decides what a zone will ignore, and it
        # was being audited in prose as "no change" while the structured
        # before/after carried it. The prose is what a person reads.
        was = ", ".join(sorted(before.classes)) or "any"
        now = ", ".join(sorted(after.classes)) or "any"
        parts.append(f"watches {was} -> {now}")
    return "; ".join(parts) if parts else "no change"


@dataclass
class _TrackFaces:
    """What the node holds about one live person track's face, and only that.

    Templates, not crops: the frame is examined and dropped, and what remains
    is the last :data:`FRAMES_KEPT_FOR_ENROLMENT` vectors, which cannot be
    rendered or inverted. The whole record dies with the track — `Node.poll`
    pops it on the frame that ends the track, and on any change of runner —
    so a stranger who walked through leaves nothing behind. That is not a
    cache policy; it is the difference between a register and a gallery.
    """

    templates: deque = field(
        default_factory=lambda: deque(maxlen=FRAMES_KEPT_FOR_ENROLMENT)
    )
    #: The frame index at which this track may next be examined. Per track,
    #: so a crowd does not all come due on one frame.
    next_index: int = 0
    #: The last verdict over everything held, or ``None`` before the first.
    identity: TrackIdentity | None = None


def _new_subject_id(kind: str) -> str:
    """An id for a subject the operator is about to name.

    Random rather than derived from the name, for the reason
    :func:`registry.new_identifier_id` gives for identifiers: the id goes into
    audit rows that outlive a `forget`, and an id built from a name would put
    the name back into the one log that is supposed to have lost it. The same
    source of randomness as that function, so the register holds one
    convention for the ids it is handed rather than two. The kind is kept as a
    prefix because an audit row reading ``person-…`` or ``vehicle-…`` says
    what was enrolled without saying who.
    """
    return f"{kind}-{secrets.token_hex(12)}"


class _SwitchedPlateReader:
    """The reader a pipeline is handed: the operator's, behind the site's switch.

    A pipeline takes its reader when it is built and holds it for the run;
    nothing on the camera thread re-reads the site row. Handed the reader
    bare, a switch turned *off* would have left every running camera reading
    plates until somebody restarted it — with the status line reporting "off"
    above a model still running, which is the one lie this feature must not
    tell. So the pipeline gets this instead: each ``read`` asks the node's
    switch and returns nothing once the site has said no. Off reaches every
    running camera at its next scheduled read, on the frame it would have
    cropped, and the pipeline's own bookkeeping — the accumulator per track,
    the reads-per-second cadence — carries on over an empty answer.

    The switch is a `threading.Event` because that is the one primitive whose
    read from the camera thread and write from the node's needs no lock and
    no reasoning about ordering; a bare attribute would have worked in
    CPython and been a question in every review.

    Only *off* travels this way. *On* still needs the camera restarted, since
    a pipeline started with plates off was given no reader at all — it has no
    accumulators and warned about nothing — and `Node.identity_status` says
    which running cameras that leaves behind.
    """

    __slots__ = ("reader", "_on")

    def __init__(self, reader: PlateReader, on: threading.Event):
        #: The operator's reader, kept readable for the status line and for a
        #: test that needs to know which one was handed over.
        self.reader = reader
        self._on = on

    @property
    def country(self) -> str:
        return self.reader.country

    @property
    def info(self):
        return getattr(self.reader, "info", None)

    def read(self, frame, vehicle_box, *, frame_index: int):
        if not self._on.is_set():
            return ()
        return self.reader.read(frame, vehicle_box, frame_index=frame_index)


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
        site_tz=None,
        face_backend: FaceBackend | None = None,
        plate_reader: PlateReader | None = None,
        record_every_camera: bool = True,
        retention=None,
        retention_every_seconds: float = RETENTION_EVERY_SECONDS,
    ):
        """
        ``record_to`` is where clips go; without it nothing records, whatever
        the cameras ask. With it, every camera records — what `sentinel node
        --record` has always meant — unless ``record_every_camera`` is False,
        in which case a camera records only when its stored flag says so
        (`CameraRecord.record`, the console's Record box). The console is the
        caller that opts out: it always has somewhere to write, and recording
        is a per-camera decision there.

        ``retention`` is the policy swept every ``retention_every_seconds``
        from :meth:`poll`, by this node, while it runs. It defaults to
        `RetentionPolicy()` when the node records and to nothing when it does
        not; a node that never writes video has nothing of its own to sweep.

        ``detector_factory`` is called once per camera. One detector per camera,
        never shared: MOG2 carries a per-pixel model of *its* scene, and feeding
        it two cameras corrupts both models and every detection that comes out
        of them.

        ``face_backend`` and ``plate_reader`` are the seams the tests inject
        through, and they change nothing about the switch: with a backend given,
        faces run when — and only when — the site's identity has faces on, and
        the model files are simply not consulted. Without them the node looks in
        :func:`paths.models_directory` for the operator's files, and when they
        are absent it runs nothing and says so in :attr:`identity_status`
        rather than raising: a camera must start whether or not the site's face
        models have arrived.

        ``rules`` defaults to :func:`~sentinel.events.default_rules` over the
        zones given, so a node configured with no rules still does something
        sensible rather than nothing silently.

        ``zones`` left empty means *load whatever this node already had*, not
        *watch nothing* — a restarted node must come back watching what it was
        watching. Passing zones explicitly replaces that.

        ``site_tz`` is the clock zone schedules are written in. Defaults to this
        machine's own zone — the honest interim until a site record declares
        one — and is labelled as such wherever a schedule is shown. Schedules
        used to be evaluated in UTC while the interface implied local time,
        which armed an 18:00 zone at 21:00 in Beirut.

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
        self._record_every_camera = bool(record_every_camera)
        from .recording import RetentionPolicy

        self._retention = (
            retention if retention is not None
            else (RetentionPolicy() if self._record_to else None)
        )
        self._retention_every = float(retention_every_seconds)
        #: Monotonic time of the last sweep; ``None`` until the first, so a
        #: node restarted after a month sweeps on its first poll.
        self._last_retention: float | None = None
        self._retention_shortfall: str | None = None
        self._keep_images = keep_images
        self._realtime = realtime
        self._correlate_every = correlate_every_millis
        self._event_retention = event_retention
        # The same code is a daemon, a console and a CLI, and an audit row that
        # says "node" when an operator pressed a button is a false entry in a
        # chain of custody. Whoever owns this node names themselves.
        self._actor = actor
        self._site_tz = site_tz if site_tz is not None else datetime.now().astimezone().tzinfo

        self._cameras: dict[str, CameraRecord] = {}
        self._restored_cameras = 0
        self._incidents: tuple[Incident, ...] = ()
        self._persisted: set[str] = set()
        # Encounters already written to the audit log, keyed the way the
        # register keys a sighting — one row per (subject, camera, track) —
        # plus the camera's run, because track ids start again with every
        # runner. Without the set the audit log would gain a line per poll
        # for a van parked in shot, and a log that says the same thing five
        # times a second is a log nobody reads; without the run a camera
        # restarted in the same process would take the new run's track 3 for
        # the old one's and never audit it. In memory only, so a restarted
        # node — same process or not — audits a still-live encounter once
        # more; that is one extra line, and the alternative is a query per
        # poll per vehicle.
        self._sighted: set[tuple[str, str, int, int]] = set()
        # Tracks whose confident text the register refused to spell, so the
        # refusal is logged once and not looked up again every poll. Keyed on
        # the run as well: a refusal has to die with the runner that earned
        # it, or the next run's track 3 — a different vehicle — inherits it,
        # and a known van's sighting is lost without a line in any log.
        self._plate_refused: set[tuple[str, int, int]] = set()
        self._last_correlated = 0.0
        self._running = False
        self._stop = threading.Event()

        # The identity machinery. `_identity` mirrors the site row and is the
        # switch every decision below reads; it is set from the store once the
        # cameras are back, because the default site's origin is the first
        # placed camera. `_faces` is the one face engine on this node — built
        # only while faces are on, used only on this thread — and `_face_tracks`
        # is everything held about live person tracks, keyed like a sighting
        # plus the run, for the same reason `_sighted` is.
        self._face_backend = face_backend
        self._injected_plate_reader = plate_reader
        # Set while plates are on; every reader handed to a pipeline consults
        # it on each read, which is how plates *off* reaches a running camera
        # without a restart. See `_SwitchedPlateReader`.
        self._plates_on = threading.Event()
        self._identity = Identity()
        self._faces: FaceEngine | None = None
        self._face_status = ""
        self._plate_status = ""
        self._face_fault: str | None = None
        self._face_tracks: dict[tuple[str, int, int], _TrackFaces] = {}
        # Frames that reached face work with no image attached. Counted and
        # reported rather than silently skipped, because a node built without
        # `keep_images` would otherwise run a switched-on face path that never
        # looked at anything, and nothing on any screen would say so.
        self._frames_without_image = 0
        # Person encounters audited, per live track, each with its confidence:
        # a possible match that later becomes a match is two claims, and the
        # log records both once rather than the first forever or the second
        # every poll. Keyed like `_face_tracks` and pruned with it — on track
        # end, on a change of runner, on stop and on faces off — because a set
        # that only ever grew held a subject id for every encounter for the
        # life of the process, forgotten subjects included.
        self._face_sighted: dict[tuple[str, int, int], set[tuple[str, str]]] = {}

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

        # After the cameras, because the default site's origin is the first
        # placed one. The switch comes from the row, never from a default here:
        # a site that turned faces on yesterday must come back with them on, or
        # the register it enrolled into matches nobody and nothing says so.
        self._identity = self.site().identity
        self._rebuild_identity()

        self.store.audit(self._actor, "node.started", node_id)
        import os

        if os.environ.get("SENTINEL_ALLOW_PUBLIC_SOURCES", "").strip().lower() not in ("", "0", "false", "no"):
            # The egress guard's one override, on the record: the log says it
            # at every start, and the audit trail — the one place a later
            # reader looks for "was this machine allowed off the site" — must
            # say it too. Written by every node, so no process reaches routable
            # addresses without a row that survives log rotation.
            self.store.audit(
                self._actor, "egress.override", node_id,
                "SENTINEL_ALLOW_PUBLIC_SOURCES is set: camera addresses outside "
                "the local network are allowed for this process",
            )
        _log.info(
            "node %s: %d zone(s), %d rule(s), recording %s, identity %s",
            node_id, len(self._zones), len(self._rules),
            self._record_to or "off", self.identity_status,
        )

    # ------------------------------------------------------------------ state

    @property
    def site_tz(self):
        """The clock zone schedules are evaluated in."""
        return self._site_tz

    @property
    def site_clock_label(self) -> str:
        """For the interface: which clock a schedule is read against.

        The UTC offset, never the zone's name: on Windows the machine zone's
        name is localised and can run to forty characters.
        """
        offset = datetime.now(self._site_tz).strftime("%z")
        return f"this machine's clock, UTC{offset[:3]}:{offset[3:]}"

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

    # ------------------------------------------------------------------- site

    def site(self) -> Site:
        """The site this node watches: the declared row, or a default it derives.

        Every caller — the identity switch, the interface, an export — has a
        site to read without first asking whether one was ever declared. A row
        an operator declared is returned as it is. Otherwise the origin is
        derived, here and on every call: the first placed camera, or nowhere.
        That is the honest interim `site.py` argues against for anything that
        lasts, and it lasts until a site editor writes a real one.

        Looking does not write. A row that appeared because somebody read it
        would be a site nobody declared, with an origin nobody chose. The one
        row this node writes on its own is :meth:`set_identity`'s, because the
        switch has nowhere else to live; it is marked undeclared, and this
        method treats it as the switch and nothing more. An earlier version
        returned that row whole, and a node that turned plates on before its
        first camera was placed kept the site at (0, 0) for good, with every
        camera placed afterwards anchored on the Gulf of Guinea.
        """
        stored = self.store.site()
        if stored is not None and stored.declared:
            return stored
        placed = next(
            (record.pose.position for record in self._cameras.values()
             if record.pose is not None),
            None,
        )
        origin = placed if placed is not None else LatLon(0.0, 0.0)
        if stored is not None:
            # The node's own row: authoritative for the switch, a snapshot
            # for everything else.
            return replace(stored, origin=origin)
        return Site(id=DEFAULT_SITE_ID, name="Site", origin=origin, declared=False)

    def set_identity(self, identity: Identity, *, reason: str) -> Site:
        """Flip the site's identity switch, audited, with the reason recorded.

        The one path by which a site starts or stops reading plates or looking
        at faces. It writes the site row — so the decision survives a restart —
        and an audit row with both states, so "who turned faces on, when, and
        why" is answerable from the log rather than from memory. ``reason`` is
        required and refused when blank for the same reason the register
        refuses a blank lawful basis: a switch that starts biometric processing
        for no recorded reason is the entry an auditor asks about first.

        **What takes effect when.** *On* applies to cameras started from now
        on, for plates and faces alike: a plate reader is handed to a pipeline
        when it is built and cannot be added to a running one, and faces keep
        the same rule so there is one rule. :attr:`identity_status` names
        every running camera the switch has not reached, until each is
        restarted, so "faces: on" is never printed over a site examining
        nobody without the line saying so. *Off* reaches every running camera
        without a restart: faces at the next poll, because that work runs here
        on the node's thread, and the held templates go with it; plates at the
        next read, because every reader was handed over behind the switch. A
        switch-off that waited for a restart would leave a model running on a
        site that had just said no. The warning logged when cameras are
        running says exactly this.

        **The row it writes.** The switch lives on the site row, and a
        deployment nobody has declared a site for has no row. One is written
        here, marked undeclared: it carries the switch, and :meth:`site` keeps
        deriving its origin from the cameras rather than freezing whatever
        could be derived at this moment — nowhere, on a node with no camera
        placed yet — as the origin every plan would later hang from.

        Returns the site as it now stands; the caller holds nothing stale.
        """
        if not reason or not reason.strip():
            raise NodeError(
                "changing what a site identifies needs a reason to record; a "
                "switch flipped for no stated reason is the audit row nobody "
                "can defend"
            )
        before = self.site()
        after = replace(before, identity=identity)
        self.store.save_site(after)
        # Structured before/after of the switch alone — three booleans — with
        # the reason in the prose. Chained, so the row that turned faces on
        # cannot be edited afterwards without the chain saying so.
        self._audit(
            "site.identity", DEFAULT_SITE_ID,
            before=before.identity, after=identity,
            detail=(
                f"identity {before.identity.describe()} -> {identity.describe()}: "
                f"{reason.strip()}"
            ),
        )
        self._identity = identity
        self._rebuild_identity()

        running = [r.camera_id for r in self._cameras.values() if r.is_running]
        if running:
            _log.warning(
                "node %s: identity is now %s. On applies to cameras started from "
                "now on; %s keep what they were started with until restarted, "
                "and the status line names them. Off reaches every running "
                "camera now: faces at the next poll, plates at the next read.",
                self._node_id, identity.describe(), ", ".join(running),
            )
        _log.info("node %s: identity %s", self._node_id, self.identity_status)
        return after

    @property
    def identity(self) -> Identity:
        """The switch as this node currently reads it."""
        return self._identity

    @property
    def identity_status(self) -> str:
        """One honest line about what the identity switch is actually doing.

        Not what it is set to — what it is *doing*. A site with faces on and no
        face models is running no face model, and an interface that printed
        "faces: on" for it would be describing a configuration rather than the
        system. So the line names the models directory and the files expected
        in it, the backend when one is loaded, the fault when one stopped it,
        the frames that arrived with no image to examine — and every running
        camera the switch has not reached. A camera keeps what it was started
        with (`CameraRecord.identity`), so "faces: on" over cameras started
        with faces off was a site examining nobody, and this line said "on";
        now it names those cameras, until each is restarted. "off" is the
        whole line when nothing is switched on, and it means nothing runs: off
        reaches a running camera without a restart, so there is nothing to
        qualify it with.
        """
        parts: list[str] = []
        if self._identity.plates:
            parts.append(
                self._plate_status + self._not_yet_reached("plates", "reads no plate")
            )
        if self._identity.faces:
            line = self._face_status
            if self._face_fault is not None:
                line += f" — stopped: {self._face_fault}"
            if self._frames_without_image:
                line += (
                    f" — {self._frames_without_image} frame(s) arrived without "
                    "an image and were not examined"
                )
            line += self._not_yet_reached("faces", "examines nobody")
            parts.append(line)
        if self._identity.face_crops:
            parts.append(
                "face crops: recorded as on, but nothing in this build keeps a "
                "crop; templates only"
            )
        return "; ".join(parts) if parts else "off"

    def _not_yet_reached(self, feature: str, consequence: str) -> str:
        """The clause naming running cameras started before ``feature`` was on.

        Empty when there are none, so the line reads as before on a node
        whose cameras were all started under the current switch. Running
        cameras only: a stopped camera takes the switch as it stands when it
        next starts, and naming it here would tell the operator to restart
        something that is not running.
        """
        behind = [
            record.camera_id for record in self._cameras.values()
            if record.is_running and not getattr(record.identity, feature)
        ]
        if not behind:
            return ""
        return (
            f" — {len(behind)} running camera(s) started with {feature} off "
            f"({', '.join(behind)}): each {consequence} until restarted"
        )

    def _rebuild_identity(self) -> None:
        """Build, or tear down, what the switch says should exist.

        Called at start-up and on every flip. The face engine is rebuilt rather
        than toggled because `FaceEngine` has no ``enable()`` on purpose: an
        engine is on or off from construction, so there is never a reference
        that holds a frame and the wrong answer about the switch. Turning faces
        off drops every held template with the engine — nothing biometric
        outlives the decision to stop.

        A missing model is a status line and a warning, never an exception. A
        node has to start whether or not the operator's files have arrived,
        and it has to say which files it is waiting for and where.
        """
        self._face_fault = None
        if not self._identity.faces:
            self._faces = None
            self._face_tracks.clear()
            self._face_sighted.clear()
            self._face_status = ""
        else:
            self._faces, self._face_status = self._build_face_engine()

        # The event every handed-out reader consults on each read. Cleared
        # first, so a reader mid-frame on a camera thread sees "off" no later
        # than this method returns.
        if not self._identity.plates:
            self._plates_on.clear()
            self._plate_status = ""
        else:
            self._plates_on.set()
            self._plate_status = self._describe_plate_readiness()

    def _build_face_engine(self) -> tuple[FaceEngine | None, str]:
        """The engine for a site with faces on, or ``None`` and the reason."""
        backend = self._face_backend
        if backend is not None:
            engine = FaceEngine(enabled=True, backend=backend)
            return engine, f"faces: on ({backend.name})"

        directory = paths.models_directory()
        detector, embedder = (directory / name for name in FACE_MODEL_FILES)
        if not (detector.is_file() and embedder.is_file()):
            expected = " and ".join(FACE_MODEL_FILES)
            _log.warning(
                "node %s: faces are on for this site but no models are in %s "
                "(expecting %s); no face will be examined until they are there. "
                "Nothing is downloaded.",
                self._node_id, directory, expected,
            )
            return None, (
                f"faces: on but no models in {directory} — expecting {expected}; "
                "no face is examined until they are there"
            )
        try:
            engine = FaceEngine(
                enabled=True, detector_model=detector, embedder_model=embedder
            )
        except Exception as error:  # noqa: BLE001 - third-party model files
            # The files are the operator's and their failure modes are not
            # ours. The camera still runs; the status says what happened.
            _log.error(
                "node %s: the face models in %s could not be loaded (%s); no "
                "face will be examined",
                self._node_id, directory, type(error).__name__, exc_info=True,
            )
            return None, (
                f"faces: on but the models in {directory} could not be loaded "
                f"({type(error).__name__}); no face is examined"
            )
        return engine, f"faces: on ({engine.info.backend})"

    def _plate_models(self) -> PlateModels:
        directory = paths.models_directory()
        detector, recogniser, charset = (directory / name for name in PLATE_MODEL_FILES)
        return PlateModels(detector, recogniser, charset)

    def _describe_plate_readiness(self) -> str:
        """What the plate path will do for the next camera started."""
        reader = self._injected_plate_reader
        if reader is not None:
            info = getattr(reader, "info", None)
            if info is not None:
                return (
                    f"plates: on ({Path(info.detector_path).name} + "
                    f"{Path(info.recogniser_path).name}, {info.country})"
                )
            return f"plates: on ({type(reader).__name__})"

        models = self._plate_models()
        missing = models.missing()
        if missing:
            directory = paths.models_directory()
            expected = ", ".join(PLATE_MODEL_FILES)
            _log.warning(
                "node %s: plates are on for this site but no models are in %s "
                "(expecting %s); no plate will be read until they are there. "
                "Nothing is downloaded.",
                self._node_id, directory, expected,
            )
            return (
                f"plates: on but no models in {directory} — expecting {expected}; "
                "no plate is read until they are there"
            )
        return (
            f"plates: on ({models.detector_path.name} + {models.recogniser_path.name} "
            f"from {models.detector_path.parent}); a reader is built for each "
            "camera when it starts"
        )

    def _plate_reader_for(self, record: CameraRecord) -> "_SwitchedPlateReader | None":
        """The reader a camera about to start gets, or ``None`` and no plates.

        One reader per camera, never shared, for the reason detectors are not
        shared: the reader holds two OpenCV networks, and a network driven from
        two threads at once is undefined. The injected reader is the exception
        and is a test's to share. A reader that cannot be built — the model
        files are the operator's and may be anything — is logged and becomes
        a camera without plates, not a camera that failed to start.

        Whatever is handed over goes behind the switch — see
        `_SwitchedPlateReader` — so that plates *off* stops it at its next read
        rather than at the camera's next restart.
        """
        if not record.identity.plates:
            return None
        if self._injected_plate_reader is not None:
            return _SwitchedPlateReader(self._injected_plate_reader, self._plates_on)
        models = self._plate_models()
        if models.missing():
            return None
        try:
            reader = PlateReader(models)
        except Exception as error:  # noqa: BLE001 - third-party model files
            _log.error(
                "node %s: camera %s: the plate reader could not be built (%s); "
                "the camera runs without plates",
                self._node_id, record.camera_id, type(error).__name__,
                exc_info=True,
            )
            self._plate_status = (
                f"plates: on but the reader could not be built "
                f"({type(error).__name__}); no plate is read"
            )
            return None
        return _SwitchedPlateReader(reader, self._plates_on)

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
            # The stored source is the redacted one. The password, if the
            # keychain kept it, goes back in here — and nowhere else — so a
            # site with network cameras comes back from a restart able to
            # open them, which an unattended node has to.
            ref = row["credentials_ref"] if "credentials_ref" in row.keys() else None
            source_text = row["source"]
            if ref:
                from . import secrets as keychain
                from .redact import with_password

                password = keychain.load(ref)
                if password:
                    source_text = with_password(source_text, password)
                else:
                    _log.warning(
                        "node %s: camera %s has a credential handle but the keychain "
                        "holds nothing for it; the password must be given again",
                        self._node_id, camera_id,
                    )
            record = CameraRecord(
                camera_id=camera_id,
                source=source_text,
                pose=self.store.camera_pose(camera_id),
                record=bool(row["record"]) if "record" in row.keys() else False,
                credentials_ref=ref,
            )
            # Rows written before one-source-one-camera was enforced. Kept, so
            # the operator can see them and nothing silently disappears from a
            # list they made — but faulted from the start, and `start` leaves a
            # faulted duplicate alone rather than fighting the first for the
            # device.
            twin = next(
                (r for r in self._cameras.values() if r.source == record.source), None
            )
            if twin is not None and is_live_source(record.source):
                record.fault = (
                    f"same source as {twin.camera_id}; not started. One source is "
                    "one camera."
                )
                _log.warning(
                    "node %s: camera %s duplicates %s (%s) and will not be started",
                    self._node_id, camera_id, twin.camera_id, record.display_source,
                )
            self._cameras[camera_id] = record
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

    # ------------------------------------------------------------------ audit

    def _audit(
        self,
        action: str,
        subject: str | None,
        *,
        before=MISSING,
        after=MISSING,
        detail: str | None = None,
    ) -> None:
        """Record an edit as both the sentence and the two states behind it.

        The prose is what an operator reads and it does not change; the states
        are what makes the row answerable to a question nobody asked at the
        time — "which corner moved, and by how far" — and they come from the
        same comparison as the sentence, so the two halves cannot disagree.

        Pass :data:`MISSING` for a side that does not exist: a removal has no
        after, and reporting every field of a deleted thing as "changed to
        nothing" would make it indistinguishable from an edit that blanked it.

        ``detail`` overrides the rendered line for a call whose existing wording
        says more than a generic diff would — a camera's placement, which reads
        as a coordinate and a bearing rather than as JSON.
        """
        self.store.audit_record(
            AuditRecord.of(
                actor=self._actor,
                action=action,
                subject=subject,
                node_id=self._node_id,
                # Aware, and UTC. A naive timestamp is read back in whatever
                # zone the reader is in, which puts an audit row hours from
                # where it belongs on the one machine that is not in UTC.
                at=datetime.now(timezone.utc),
                before=before,
                after=after,
            ),
            detail=detail,
        )

    # ---------------------------------------------------------------- cameras

    def add_camera(
        self, source: str | Path, *, camera_id: str | None = None,
        pose: CameraPose | None = None, recording: bool = False,
    ) -> CameraRecord:
        """Register a camera. Does not start it, and opens nothing.

        ``recording`` asks it to record whenever it runs; stored with it.
        """
        text = str(source)
        identifier = camera_id or f"cam-{len(self._cameras) + 1:02d}"
        if identifier in self._cameras:
            raise NodeError(
                f"there is already a camera called {identifier!r} on this node"
            )
        # One live source, one camera. An operator's log showed `device:0`
        # added three times across sessions, all three restored and all three
        # started against one webcam: the driver refused two of them, every
        # pane reconnected in a loop, and the one that worked was down to a
        # frame a second. A file is exempt — it is a replay, and any number of
        # cameras may read it. The display form is compared and reported,
        # never the raw one.
        if is_live_source(text):
            display = CameraRecord(camera_id=identifier, source=text).display_source
            for existing in self._cameras.values():
                if existing.source == text or existing.display_source == display:
                    raise NodeError(
                        f"{display} is already camera {existing.camera_id!r} on "
                        "this node. One camera per device; start that one."
                    )

        record = CameraRecord(
            camera_id=identifier, source=text, pose=pose, record=bool(recording)
        )
        # A password travels to the operating system's keychain under a random
        # handle, and the handle to the database. With no keychain on the
        # machine, nothing is kept and the camera needs its password again
        # after a restart — the behaviour before the keychain existed, and
        # said so in the log rather than stored somewhere worse than nowhere.
        record.credentials_ref = self._keep_password(identifier, text)
        self._cameras[identifier] = record

        # The redacted form, never the raw one: this row is read by anything
        # that lists cameras, and a password in it is a password on a screen.
        self.store.save_camera(
            identifier, identifier, record.display_source, pose=pose,
            record=record.record, credentials_ref=record.credentials_ref,
        )
        self.store.audit(
            self._actor, "camera.added", identifier,
            record.display_source + (" (recording)" if record.record else ""),
        )
        _log.info("node %s: added camera %s (%s)", self._node_id, identifier,
                  record.display_source)
        return record

    def remove_camera(self, camera_id: str) -> None:
        """Forget a camera. Stops it first if it is running.

        Its events and incidents are kept — a camera taken down does not unmake
        what it saw — so only the camera row, its pane and its runner go. A
        runner that will not stop is not removed: dropping the reference to a
        thread that is still decoding is how a decoder ends up writing into a
        closed store.
        """
        record = self.camera(camera_id)
        if record.runner is not None and record.runner.is_running:
            if not record.runner.stop():
                raise NodeError(
                    f"{camera_id} did not stop within {STOP_TIMEOUT_SECONDS:.0f}s "
                    "and was not removed; it still holds its decoder."
                )
            # The last events it raised are drained and correlated before the
            # camera goes, or a run that ended on an intrusion would lose it.
            self.poll(force_correlate=True)

        del self._cameras[camera_id]
        self.store.delete_camera(camera_id)
        if record.credentials_ref:
            from . import secrets as keychain

            keychain.forget(record.credentials_ref)
        # The redacted source and the placement, never the raw source: this row
        # outlives the camera, and it is the only surviving record of where the
        # thing that produced the evidence was pointing. A removal has no after.
        self._audit(
            "camera.removed",
            camera_id,
            before={
                "camera_id": record.camera_id,
                "source": record.display_source,
                "pose": record.pose,
            },
            detail=record.display_source,
        )
        self._running = any(r.is_running for r in self._cameras.values())
        _log.info("node %s: removed camera %s (%s)", self._node_id, camera_id,
                  record.display_source)

    def place_camera(self, camera_id: str, pose: CameraPose | None) -> None:
        """Set where a camera is and where it points, running or not.

        Audited with the pose it had as well as the one it was given. A camera
        nudged three degrees is the difference between a track that lands in the
        zone and one that does not, and until both poses were recorded the log
        could say where it ended up and never where it had been.
        """
        record = self.camera(camera_id)
        was = record.pose
        record.pose = pose
        if record.runner is not None:
            record.runner.set_pose(pose)

        # `record=record.record`, or a placement would switch recording off.
        self.store.save_camera(
            camera_id, camera_id, record.display_source, pose=pose, record=record.record
        )
        # The line is the one this log has always carried: a coordinate to six
        # places and the angles, which is what a person checking a placement
        # reads. A diff of two poses would replace it with JSON, and that is a
        # worse sentence for the reader, so the structured pair goes beside it
        # rather than over it.
        self._audit(
            "camera.placed", camera_id,
            before=was,
            after=pose,
            detail=(
                f"{pose.position.lat:.6f},{pose.position.lon:.6f} "
                f"h={pose.mount_height} hdg={pose.heading} pitch={pose.pitch}"
                if pose else "unplaced"
            ),
        )

    def _keep_password(self, camera_id: str, source: str) -> str | None:
        """File a source's password in the keychain; return the handle, or None."""
        from . import secrets as keychain
        from .redact import split_password

        _, password = split_password(source)
        if not password:
            return None
        if not keychain.available():
            _log.warning(
                "node %s: camera %s has a password and this machine has no keychain "
                "to keep it in; it must be given again after a restart",
                self._node_id, camera_id,
            )
            return None
        ref = keychain.new_ref()
        if not keychain.store(ref, password):
            return None
        return ref

    def set_password(self, camera_id: str, password: str) -> None:
        """Give a stored camera its password, or a new one, without it ever
        touching the command line or the database.

        Rebuilds the in-memory source from the redacted form plus the secret,
        files the secret in the keychain under a fresh handle (the old one is
        forgotten), and audits that a credential changed — the handle, never
        the value.
        """
        from . import secrets as keychain
        from .redact import redact_url, with_password

        if not password:
            raise NodeError("a password cannot be empty")
        record = self.camera(camera_id)
        display = record.display_source
        if "://" not in display:
            raise NodeError(f"{camera_id} is not a network camera; it has no password")
        placeholder = display if REDACTED in display else display.replace("@", f":{REDACTED}@", 1)
        rebuilt = with_password(placeholder, password)
        if rebuilt == placeholder:
            raise NodeError(f"{camera_id}: could not place a password into {display}")
        if not keychain.available():
            raise NodeError("this machine has no keychain to keep the password in")
        ref = keychain.new_ref()
        if not keychain.store(ref, password):
            raise NodeError("the keychain refused the password")
        keychain.forget(record.credentials_ref)
        record.credentials_ref = ref
        record.source = rebuilt
        self.store.save_camera(
            camera_id, camera_id, redact_url(rebuilt), pose=record.pose,
            record=record.record, credentials_ref=ref,
        )
        self.store.audit(self._actor, "camera.credential", camera_id, "password stored in the keychain")

    def set_recording(self, camera_id: str, on: bool) -> None:
        """Ask a camera to record whenever it runs, or stop asking.

        Stored with the camera, so it survives a restart, and audited with
        both states: whether a camera was recording on the night is the first
        question asked of any footage that is missing. Takes effect when the
        camera is next started — a recorder is handed to a pipeline when it is
        built — and the health line says "will record when restarted" until
        then, so the flag is never mistaken for the fact.
        """
        record = self.camera(camera_id)
        was, record.record = record.record, bool(on)
        if was == record.record:
            return
        self.store.save_camera(
            camera_id, camera_id, record.display_source, pose=record.pose, record=record.record
        )
        self._audit(
            "camera.recording", camera_id,
            before={"record": was}, after={"record": record.record},
            detail="recording on" if record.record else "recording off",
        )
        if record.is_running:
            _log.warning(
                "node %s: camera %s will %s when it is next started; the run in "
                "progress keeps what it was started with",
                self._node_id, camera_id, "record" if record.record else "stop recording",
            )
        if record.record and self._record_to is None:
            _log.warning(
                "node %s: camera %s asks to record, but this node has no "
                "recordings directory; nothing will be written",
                self._node_id, camera_id,
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

    def replace_zone(self, zone: Zone) -> None:
        """Change a zone that already exists — its name, kind, or shape.

        Same id, new definition. Persisted as an upsert so the row keeps its
        creation time, and audited with what changed, because a restricted area
        quietly becoming an exclusion zone is exactly the edit an audit log
        exists to record.
        """
        index = next((i for i, z in enumerate(self._zones) if z.id == zone.id), None)
        if index is None:
            raise NodeError(f"no zone {zone.id!r} on this node")
        before = self._zones[index]
        self._zones[index] = zone
        self.store.save_zone(zone)
        # The sentence is rendered from the same comparison that is stored, so
        # the log cannot say the kind changed while the record says it did not.
        self._audit("zone.changed", zone.id, before=before, after=zone)
        if self._running:
            _log.warning(
                "node %s: zone %s changed; cameras already running keep the old "
                "definition until restarted", self._node_id, zone.id,
            )

    def remove_zone(self, zone_id: str) -> None:
        """Forget a zone, and drop the rules that need one if it was the last.

        The mirror of `add_zone`: a node whose last zone has gone must not keep
        rules that watch zones, or it carries dead weight that reports "0 events"
        for a reason unrelated to the footage. Events already raised inside the
        zone are kept; they name it in their own text.
        """
        zone = next((z for z in self._zones if z.id == zone_id), None)
        if zone is None:
            raise NodeError(f"no zone {zone_id!r} on this node")
        self._zones.remove(zone)
        self.store.delete_zone(zone_id)
        self.store.audit(self._actor, "zone.removed", zone_id, zone.name)

        if not self._zones:
            self._rules = default_rules(self._zones)
            _log.info(
                "node %s: last zone removed; rule set is now %d rule(s)",
                self._node_id, len(self._rules),
            )
        if self._running:
            _log.warning(
                "node %s: zone %s removed; cameras already running keep it until "
                "restarted", self._node_id, zone_id,
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
            if record.fault is not None and record.fault.startswith("same source as"):
                continue
            record.fault = None
            # The switch as it stands now is what this run gets. See
            # `CameraRecord.identity` for what a later flip does and does not
            # change about a running camera.
            record.identity = self._identity
            source = VideoSource(record.source, source_id=record.camera_id)
            record.runner = CameraRunner(
                source,
                # One detector per camera, never shared.
                self._detector_factory(),
                record.pose,
                zones=self._zones,
                rules=self._rules,
                node_id=self._node_id,
                # Face work needs the frame. A daemon keeps no images because
                # nobody is watching, and with faces on it would otherwise run
                # a switched-on face path over results that carried nothing
                # to look at. The image travels in the latest-wins slot and is
                # dropped after the poll; nothing here stores it. Forced only
                # while an engine exists to examine it: faces on with no
                # models examines nothing, and a headless camera carrying a
                # full-resolution frame per result for nothing would be the
                # cost of the feature without the feature.
                keep_images=self._keep_images
                or (record.identity.faces and self._faces is not None),
                realtime=self._realtime,
                # The flag reaches the pipeline here and nowhere else: a
                # camera records when it asked to, or when the whole node was
                # told to (the CLI's --record), and a node with nowhere to
                # write records nothing whatever was asked.
                record_to=(
                    self._record_to
                    if self._record_to is not None
                    and (self._record_every_camera or record.record)
                    else None
                ),
                segment_seconds=self._segment_seconds,
                site_tz=self._site_tz,
                plate_reader=self._plate_reader_for(record),
            )
            record.runner.start()
            started += 1

        self._running = started > 0
        self._stop.clear()
        self.store.audit(self._actor, "analysis.started", detail=f"{started} camera(s)")
        _log.info(
            "node %s: started %d camera(s); zone schedules read against %s",
            self._node_id, started, self.site_clock_label,
        )
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
        # No track is live once its camera has stopped, and the templates were
        # only ever the last few faces of a live track. They go now rather
        # than when the next runner replaces them, and the audited claims
        # about those tracks go with them.
        self._face_tracks.clear()
        self._face_sighted.clear()
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
                if update.result.plates:
                    # On this thread, for the same reason as the segments.
                    self._note_plates(record, update.result)
                if record.identity.faces and self._faces is not None:
                    # Gated twice on purpose: the run's own snapshot says
                    # faces were on when it started, and the live engine says
                    # they still are. Either alone is a way to examine a face
                    # on a site that said no.
                    self._note_faces(record, update)

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

        self._sweep_retention_if_due()

        due = (time.monotonic() - self._last_correlated) * 1000 >= self._correlate_every
        if fresh or force_correlate or due:
            self.correlate()

        return tuple(updates)

    def _note_plates(self, record: CameraRecord, result: FrameResult) -> None:
        """Keep what the plate reader concluded, and match what it is sure of.

        Until this existed a reading reached the screen and nothing else:
        `FrameResult.plates` was drawn beside the box and dropped with the
        frame, so a reader that worked all night left no record that it had.
        Each reading is upserted on (camera, track) — one row per vehicle,
        refreshed as agreement grows, never one per frame — and the whole
        frame's readings land in one unit of work, so a poll that fails
        halfway leaves the table as it was rather than with half a frame.

        Only a reading that is *confident* and fully resolved is looked up in
        the register. ``display`` carries ``?`` where a character is unread and
        ``text`` is ``None`` until every character resolved, and neither is
        ever handed to the register: completing a half-read plate against a
        list of known plates is how somebody's car is sighted at a gate it
        never came through. The pipeline draws that line; this method keeps it.
        """
        with self.store.transaction():
            for plate in result.plates:
                self.store.save_plate_read(
                    record.camera_id, plate,
                    frame_index=result.index,
                    seen_at_millis=result.timestamp_millis,
                )
                if plate.is_confident and plate.text and plate.reads > 0:
                    self._note_sighting(record, plate, plate.text, result)

    def _note_sighting(
        self, record: CameraRecord, plate: TrackPlate, text: str, result: FrameResult
    ) -> None:
        """Record a known vehicle's track as a sighting of its subject.

        A ``MATCH``, because the register compares the whole normalised string
        for equality and the reading cleared the pipeline's bar for acting on
        it. The score is the share of this track's reads that agreed on the
        weakest character — the evidence the reading actually rests on, and a
        number an operator can argue with, unlike a similarity nobody
        measured. The enrolment matched is cited, so "why does it think this
        is that van" has an answer in the row.

        The window is the track's own first and last observation, on the
        pipeline's clock, so a sighting lines up with the track's events. It
        widens on every poll: `Register.record_sighting` folds the window and
        keeps one row per track, which is what makes calling this per poll
        safe.

        A reading the register cannot spell — characters outside the plate
        alphabet, which the pipeline's own normalisation keeps — is logged
        once and never becomes a sighting. It is not a fault: it is a read of
        something that is not a registration, and the poll goes on. The
        refusal is remembered per run, not per camera: a runner rebuilt in
        the same process starts its track ids again, and a refusal that
        outlived its runner once cost a known van its sighting — the new
        run's track 3 inherited the old run's silence.

        The first sighting of an encounter is audited by subject id and camera,
        never by plate or name. The audit log is read by more people than the
        register is, and a registration in it is a registration the register
        can no longer take back. The node log gets no plate either: the
        register's refusal quotes the read, and a log travels in every
        support bundle, so the reason is given here in the node's own words.
        """
        camera_id = record.camera_id
        key = (camera_id, record.run, plate.track_id)
        if key in self._plate_refused:
            return
        try:
            subject = self.store.register.find_plate(text)
        except RegistryError:
            self._plate_refused.add(key)
            _log.warning(
                "node %s: camera %s track %d: the register cannot spell this "
                "read — no character in it survives the plate alphabet — so "
                "it is recorded as a reading and nothing more",
                self._node_id, camera_id, plate.track_id,
            )
            return
        if subject is None:
            return

        track = next((t for t in result.tracks if t.id == plate.track_id), None)
        first_seen = track.first_seen_millis if track else result.timestamp_millis
        last_seen = track.last_seen_millis if track else result.timestamp_millis
        self.store.register.record_sighting(
            subject_id=subject.id,
            camera_id=camera_id,
            track_id=plate.track_id,
            first_seen_millis=min(first_seen, last_seen),
            last_seen_millis=max(first_seen, last_seen),
            confidence=Confidence.MATCH,
            score=min(1.0, plate.agreement / plate.reads),
            identifier_id=self._enrolment_matching(subject, text),
        )

        encounter = (subject.id, camera_id, record.run, plate.track_id)
        if encounter not in self._sighted:
            self._sighted.add(encounter)
            self.store.audit(
                self._actor, "vehicle.sighted", subject.id,
                f"camera {camera_id}, track {plate.track_id}",
            )
            _log.info(
                "node %s: subject %s sighted on camera %s track %d "
                "(agreement %d/%d)",
                self._node_id, subject.id, camera_id, plate.track_id,
                plate.agreement, plate.reads,
            )

    def _note_faces(self, record: CameraRecord, update: Update) -> None:
        """Examine the person tracks on one frame for faces, and match them.

        Everything the People feature promises is kept or broken here, so the
        promises are the structure of the method:

        **Only inside a person's box.** The engine is handed the frame and one
        track's box, and crops for itself; a track the detector did not label
        ``person`` is never handed over at all. A motion detector labels
        nothing, so a node running on motion alone examines nobody however the
        switch is set. The test that proves the stand-in models saw no pixels
        with the switch off is stronger than that: it runs a detector that
        labels a person on every frame, so the switch alone is what kept the
        pixels away.

        **At most once every :data:`FACE_STRIDE_FRAMES` frames per track.** The
        frame index carries the cadence rather than the poll, so a viewer that
        polls slowly does not examine every frame it happens to collect and a
        fast one does not examine every frame there is.

        **Held for the track, and dropped with it.** Templates go into the
        track's deque and the deque goes when the track ends, or when the run
        does. A person who walked through and was named by nobody has left
        nothing on this node by the time their box closes.

        **A verdict is over the track, never a frame.** Everything held is
        re-identified together, so a match rests on the median of several
        frames and a single flattering frame cannot make it. A MATCH or a
        POSSIBLE becomes a sighting on the register, with its confidence and
        its score, and the first of each is audited by subject id. NONE writes
        nothing anywhere.

        **A broken model stops the faces, not the camera.** A backend raising
        on a crop is logged with its type, the engine is dropped, and the
        status line says so; the poll and the camera carry on.
        """
        result = update.result
        camera_id = record.camera_id
        run = record.run

        # A new runner starts its track ids again, so anything held under an
        # older run for this camera describes tracks that no longer exist.
        self._drop_face_state(
            key for key in set(self._face_tracks) | set(self._face_sighted)
            if key[0] == camera_id and key[1] != run
        )
        self._drop_face_state((camera_id, run, track_id) for track_id in result.ended)

        image = update.image
        if image is None:
            self._frames_without_image += 1
            return

        engine = self._faces
        assert engine is not None  # the caller gated on it
        info = record.runner.detector_info if record.runner is not None else None
        if info is None:
            return
        now = int(time.time() * 1000)
        enrolled: list[tuple[EnrolledPerson, tuple[str, ...]]] | None = None

        for track in result.tracks:
            if info.label_for(track.class_id).strip().lower() != "person":
                continue
            key = (camera_id, run, track.id)
            state = self._face_tracks.get(key)
            if state is None:
                state = _TrackFaces()
                self._face_tracks[key] = state
            if result.index < state.next_index:
                continue
            state.next_index = result.index + FACE_STRIDE_FRAMES

            try:
                templates = engine.templates_in(
                    image, track.bbox,
                    source=f"{camera_id}/{track.id}",
                    timestamp_unix_millis=now,
                )
            except FaceError as error:
                # A box the engine refuses — no area, or not normalised — is a
                # fact about this track on this frame, not about the model.
                _log.debug("node %s: camera %s track %d: %s",
                           self._node_id, camera_id, track.id, error)
                continue
            except Exception as error:  # noqa: BLE001 - the operator's models
                self._face_fault = type(error).__name__
                self._faces = None
                self._face_tracks.clear()
                _log.error(
                    "node %s: FACE WORK STOPPED — %s. Cameras keep running; no "
                    "face is examined until identity is set again.",
                    self._node_id, self._face_fault, exc_info=True,
                )
                return
            if not templates:
                continue

            state.templates.extend(templates)
            if enrolled is None:
                enrolled = self._enrolled_people(engine)
            identity = engine.identify_track(
                tuple(state.templates), [person for person, _ in enrolled]
            )
            state.identity = identity
            if identity.person_id is not None and identity.verdict in (
                Verdict.MATCH, Verdict.POSSIBLE
            ):
                self._note_face_sighting(record, track, state, identity, enrolled)

    def _drop_face_state(self, keys) -> None:
        """Forget everything held about these tracks: templates, verdict, claims.

        One method for both dicts, because the pair fell out of step once:
        the templates died with the track and the audited claims did not.
        """
        for key in list(keys):
            self._face_tracks.pop(key, None)
            self._face_sighted.pop(key, None)

    def _enrolled_people(
        self, engine: FaceEngine
    ) -> list[tuple[EnrolledPerson, tuple[str, ...]]]:
        """Everybody enrolled for this backend, rebuilt from the register.

        Rebuilt per poll rather than cached, for the reason `faces._Register`
        gives: a cached copy of the site's biometrics is a second register
        that outlives the `forget` that was supposed to delete them. One query
        per poll that examined a face is the price, and it is small.

        Filtered to the engine's own model by the register itself — a template
        from another encoder is a number that looks like a score — and each
        person's enrolment ids travel beside their templates in the same
        order, so a sighting can cite the enrolment it actually matched.
        """
        register = self.store.register
        backend = engine.info.backend or ""
        held: dict[str, list[tuple[str, FaceTemplate]]] = {}
        for identifier in register.templates(model=backend):
            if identifier.template is None:
                continue
            vector = np.frombuffer(identifier.template, dtype=np.float32)
            try:
                template = FaceTemplate(
                    vector=tuple(float(v) for v in vector),
                    quality=float(np.clip(identifier.quality or 0.0, 0.0, 1.0)),
                    model=identifier.model or backend,
                    source=(
                        f"{identifier.source_camera}/{identifier.source_track}"
                        if identifier.source_camera is not None else ""
                    ),
                    created_unix_millis=identifier.enrolled_at_millis,
                )
            except FaceError as error:
                # A stored template this engine cannot read is skipped, not
                # matched: comparing against it would score nobody honestly.
                _log.warning(
                    "node %s: enrolment %s is unusable and was not compared: %s",
                    self._node_id, identifier.id, error,
                )
                continue
            held.setdefault(identifier.subject_id, []).append((identifier.id, template))

        people: list[tuple[EnrolledPerson, tuple[str, ...]]] = []
        for subject_id, enrolments in held.items():
            subject = register.subject(subject_id)
            if subject is None:
                continue
            people.append((
                EnrolledPerson(
                    person_id=subject_id,
                    name=subject.display_name,
                    templates=tuple(template for _, template in enrolments),
                ),
                tuple(identifier_id for identifier_id, _ in enrolments),
            ))
        return people

    def _note_face_sighting(
        self,
        record: CameraRecord,
        track: Track,
        state: _TrackFaces,
        identity: TrackIdentity,
        enrolled: Sequence[tuple[EnrolledPerson, tuple[str, ...]]],
    ) -> None:
        """Record a matched or possibly-matched person track on the register.

        Mirrors `_note_sighting` for plates: the window is the track's own
        first and last observation and widens on every poll, the register
        keeps one row per track, and the confidence is the verdict's — MATCH
        for MATCH, POSSIBLE for POSSIBLE, so the *possible match* rendering
        survives into the movement history and a later panel cannot show the
        certain form of a name this one hedged. The score is the track median
        `faces.identify_track` reached, not the best frame.

        The enrolment cited is the one of this person's that the track's held
        templates resemble most, so "why does it think this is her" has a row
        to point at. Audited once per encounter per confidence, by subject id
        and camera, never by name: the audit log outlives a forget.
        """
        camera_id = record.camera_id
        person = next(
            ((p, ids) for p, ids in enrolled if p.person_id == identity.person_id), None
        )
        if person is None:
            return
        enrolled_person, identifier_ids = person

        # Which of their enrolments this track resembles most, scored with
        # `faces.similarity` rather than a product written here. The pair loop
        # `faces` warns against at register scale is, for one person's few
        # enrolments against at most FRAMES_KEPT_FOR_ENROLMENT held faces, a
        # few dozen dot products — and one scoring rule in the system is worth
        # more than the microseconds a second copy of it would save.
        cited, _ = max(
            zip(identifier_ids, enrolled_person.templates),
            key=lambda pair: max(similarity(held, pair[1]) for held in state.templates),
        )

        confidence = (
            Confidence.MATCH if identity.verdict is Verdict.MATCH else Confidence.POSSIBLE
        )
        first_seen, last_seen = track.first_seen_millis, track.last_seen_millis
        try:
            self.store.register.record_sighting(
                subject_id=identity.person_id,
                camera_id=camera_id,
                track_id=track.id,
                first_seen_millis=min(first_seen, last_seen),
                last_seen_millis=max(first_seen, last_seen),
                confidence=confidence,
                score=identity.score,
                identifier_id=cited,
            )
        except RegistryError as error:
            # The subject was forgotten between the rebuild and now, or the
            # enrolment was swept. A sighting of nobody is not recorded.
            _log.warning("node %s: sighting not recorded: %s", self._node_id, error)
            return

        claims = self._face_sighted.setdefault((camera_id, record.run, track.id), set())
        claim = (identity.person_id, confidence.value)
        if claim not in claims:
            claims.add(claim)
            word = "match" if confidence is Confidence.MATCH else "possible match"
            self.store.audit(
                self._actor, "person.sighted", identity.person_id,
                f"camera {camera_id}, track {track.id}: {word}, "
                f"{identity.score:.2f} over {identity.frames} frame(s)",
            )
            _log.info(
                "node %s: subject %s sighted on camera %s track %d (%s, %.2f over "
                "%d frame(s))",
                self._node_id, identity.person_id, camera_id, track.id, word,
                identity.score, identity.frames,
            )

    # --------------------------------------------------------- the register

    def identity_of(self, camera_id: str, track_id: int) -> TrackIdentity | None:
        """What the node currently says about who a live person track is.

        ``None`` when faces are off, when the track is not one this node holds
        a face for, or before its first examination — those are all "nothing
        has been decided", and an interface must draw nothing for them. A
        :class:`~sentinel.faces.TrackIdentity` otherwise, verdict and all;
        ``verdict is Verdict.NONE`` is a decision that this is nobody enrolled.
        """
        record = self.camera(camera_id)
        state = self._face_tracks.get((camera_id, record.run, track_id))
        return state.identity if state is not None else None

    def templates_for(self, camera_id: str, track_id: int) -> tuple[FaceTemplate, ...]:
        """The templates held for a live person track, newest last.

        At most :data:`FRAMES_KEPT_FOR_ENROLMENT`, empty when faces are off or
        the track has ended, and never persisted by anything but
        :meth:`enrol_person`. An interface offers "name this person" exactly
        when this is non-empty.
        """
        record = self.camera(camera_id)
        state = self._face_tracks.get((camera_id, record.run, track_id))
        return tuple(state.templates) if state is not None else ()

    def enrol_person(
        self,
        name: str,
        camera_id: str,
        track_id: int,
        *,
        basis: str,
        notes: str | None = None,
        subject_id: str | None = None,
    ) -> str:
        """Name a live person track, storing its best face template. Returns the subject id.

        The only path from a face to the register, and it is an operator's
        act on a track they can see: nothing is enrolled by being observed.
        The best-quality template held for the track — the detector's own
        confidence that this was a face — is stored as the enrolment, packed
        as float32 with the backend that produced it named beside it, so the
        register can refuse to compare it against another model's output.

        Refused, with a plain message, when faces are off for the site or no
        engine is running, when the track has no templates held, and when the
        name is blank; the register refuses a blank basis itself. Each is a
        `RegistryError`, because each is the register declining a write.

        ``subject_id`` names an existing person to add a second template to —
        the register adds to a subject and does not rename it — and must be a
        person already enrolled; without it a new subject is created.

        Audited by subject id, camera and track. The name and the vector are
        deliberately absent from the audit detail: the log is append-only and
        read by more people than the register, and a name in it would outlive
        the forget this register exists to make possible.
        """
        if not self._identity.faces or self._faces is None:
            raise RegistryError(
                "faces are off for this site, or no face engine is running "
                f"({self.identity_status}), so there is no template to enrol. "
                "Turn faces on for the site and let the person be seen first."
            )
        if not name or not name.strip():
            raise RegistryError(
                "a person needs a display name: an entry nobody can recognise "
                "cannot be reviewed, renamed or forgotten on purpose"
            )
        templates = self.templates_for(camera_id, track_id)
        if not templates:
            raise RegistryError(
                f"camera {camera_id} track {track_id} has no face held: the track "
                "must be live and its face seen before it can be named. A track "
                "that has ended cannot be enrolled from memory, because nothing "
                "of it is kept."
            )
        register = self.store.register
        if subject_id is not None:
            existing = register.subject(subject_id)
            if existing is None or existing.kind is not SubjectKind.PERSON:
                raise RegistryError(
                    f"no person {subject_id!r} to add a template to; enrol "
                    "without a subject id to create one"
                )

        best = max(templates, key=lambda template: template.quality)
        stored = StoredFaceTemplate(
            vector=np.asarray(best.vector, dtype=np.float32).tobytes(),
            model=best.model,
            quality=best.quality,
        )
        subject = subject_id if subject_id is not None else _new_subject_id("person")
        enrolment = register.enrol(
            subject_id=subject,
            display_name=name.strip(),
            identifier=stored,
            actor=self._actor,
            basis=basis,
            notes=notes,
            source_camera=camera_id,
            source_track=track_id,
        )
        self.store.audit(
            self._actor, "person.enrolled", subject,
            f"camera {camera_id}, track {track_id}: enrolment "
            f"{enrolment.identifier.id}, best of {len(templates)} template(s) "
            f"held (quality {best.quality:.2f}), "
            f"{'new subject' if enrolment.created_subject else 'added to subject'}; "
            "basis recorded",
        )
        _log.info(
            "node %s: subject %s enrolled from camera %s track %d (%d template(s) "
            "held, best quality %.2f)",
            self._node_id, subject, camera_id, track_id, len(templates), best.quality,
        )
        return subject

    def enrol_vehicle(
        self,
        name: str,
        plate: str,
        *,
        basis: str,
        notes: str | None = None,
        camera_id: str | None = None,
        track_id: int | None = None,
    ) -> str:
        """Name a vehicle by its plate. Returns the subject id.

        Does not need plates to be on: a contractor's van is listed from the
        paperwork before any model reads it, and the register normalises the
        text to the site's plate format so ``B 7421`` and ``B-7421`` are one
        vehicle. The register refuses a blank name or basis, an unresolved
        read, and a plate already enrolled to somebody else.

        Audited by subject id only — never the plate text, for the reason
        `_note_sighting` gives.
        """
        subject = _new_subject_id("vehicle")
        enrolment = self.store.register.enrol(
            subject_id=subject,
            display_name=name,
            identifier=Plate(plate),
            actor=self._actor,
            basis=basis,
            notes=notes,
            source_camera=camera_id,
            source_track=track_id,
        )
        self.store.audit(
            self._actor, "vehicle.enrolled", subject,
            f"enrolment {enrolment.identifier.id}"
            + (f", from camera {camera_id}, track {track_id}"
               if camera_id is not None else ", typed in")
            + "; basis recorded",
        )
        _log.info("node %s: subject %s enrolled (vehicle)", self._node_id, subject)
        return subject

    def forget_subject(self, subject_id: str) -> Forgotten:
        """Erase a subject: every identifier, every sighting, the row itself.

        The register does the deleting and reports what went; this audits it
        with counts and ids only — `Forgotten.detail` carries no name — and
        drops any live verdict that named the subject, so a track on screen
        does not keep saying a name the register no longer holds.
        """
        forgotten = self.store.register.forget(subject_id)
        self.store.audit(self._actor, "subject.forgotten", subject_id, forgotten.detail())
        for state in self._face_tracks.values():
            if state.identity is not None and state.identity.person_id == subject_id:
                state.identity = None
        # The audited claims about a forgotten subject can never fire again —
        # there is nobody left to match — so the memory of them goes too,
        # rather than keeping the id of a person the register no longer holds.
        for claims in self._face_sighted.values():
            claims.difference_update({claim for claim in claims if claim[0] == subject_id})
        _log.info(
            "node %s: subject %s forgotten (%d identifier(s), %d sighting(s))",
            self._node_id, subject_id, forgotten.identifiers_deleted,
            forgotten.sightings_unlinked,
        )
        return forgotten

    def pin_subject(self, subject_id: str, pinned: bool) -> None:
        """Exempt a subject from the retention sweep, or stop exempting it. Audited."""
        pinning = self.store.register.set_pinned(subject_id, pinned, actor=self._actor)
        self.store.audit(
            self._actor, "subject.pinned" if pinned else "subject.unpinned",
            subject_id, pinning.detail(),
        )

    def _enrolment_matching(self, subject: Subject, text: str) -> str | None:
        """The id of the plate enrolment this read matched, to cite as evidence.

        Looked up rather than assumed. `Register.find_plate` returns the
        subject and not the row, and a sighting citing nothing would answer
        "why this van" with silence while a subject with two plates enrolled
        would leave the question genuinely open.
        """
        register = self.store.register
        normalised = register.plate_format.normalise(text)
        for identifier in register.identifiers(subject.id):
            if identifier.plate == normalised:
                return identifier.id
        return None

    @property
    def retention_shortfall(self) -> str | None:
        """Why the last sweep could not reach its target, or ``None``.

        Set when everything left is preserved evidence, which the sweep will
        never delete; the disk then fills, and the interface must say so.
        """
        return self._retention_shortfall

    def _sweep_retention_if_due(self) -> None:
        """Retention, from the node's own loop, on a cadence.

        On this thread, which is the store's. A sweep is a query and a few
        unlinks; hundreds of them on a GUI thread are felt, and moving the
        node's persistence off that thread is a separate item — but a sweep
        that nobody runs is a disk that fills, and that is the worse trade.
        """
        if self._retention is None:
            return
        now = time.monotonic()
        if self._last_retention is not None and now - self._last_retention < self._retention_every:
            return
        self._last_retention = now
        from .recording import apply_retention

        try:
            result = apply_retention(self.store, self._retention, actor=self._actor)
        except Exception:  # noqa: BLE001 - a failed sweep must not stop the poll
            _log.exception("node %s: retention sweep failed", self._node_id)
            return
        self._retention_shortfall = result.shortfall
        if result.deleted:
            _log.info(
                "node %s: retention removed %d clip(s), %.2f GiB",
                self._node_id, len(result.deleted), result.freed_gib,
            )

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

    def camera_health(self) -> dict[str, CameraHealth]:
        """Per camera, whether it is actually working. Keyed by camera id.

        `summary()` is prose for a person reading a log after the fact; this is
        the same question asked while it matters, in fields an interface can put
        on a strip and a map can hatch from. The distinction they both need and
        `is_running` cannot make: a camera whose thread is alive and whose
        decoder has produced nothing for thirty seconds is not watching anything,
        and until this existed it looked identical to one that was.

        Safe to call on a repaint timer, and safe while cameras are running: it
        reads each runner's published counters under that runner's own lock and
        takes nothing from the analysis threads.
        """
        return {
            camera_id: _health_for(record)
            for camera_id, record in self._cameras.items()
        }

    def summary(self) -> str:
        """What happened, for a person to read — including what was recorded.

        The audit block is here because nothing else reads the log. Every zone
        edit, every placement and every removal has been written with its two
        states and its chain hash since those columns existed, and an operator
        had no way to see that any of it happened: the writing was end to end
        and the reading stopped at the database. Three numbers close that, and
        they are the three a person checking a log actually needs — how much is
        there, how much of it is chained, and what the head is.

        The head is printed in full, never shortened. Its whole purpose is to be
        copied somewhere this process cannot reach, and a prefix copied into a
        logbook is not the head: it would verify nothing and look as though it
        had. The rows the chain does not cover are stated as a count rather than
        passed over, because a chain over part of a log is worth having and
        worth being honest about.
        """
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
                    f"{stats.distinct_objects} track(s)"
                )
            lines.append(f"    events        {len(record.events)}")
            recorder = record.runner.recorder_stats if record.runner is not None else None
            if recorder is not None:
                lines.append(
                    f"    recording     {recorder.segments_written} clip(s), "
                    f"{recorder.bytes_written / 1024 / 1024:.1f} MiB"
                    + (f", STOPPED EARLY: {recorder.fault}" if recorder.fault else "")
                )
            elif record.runner is not None and getattr(record.runner, "recording_fault", None):
                lines.append(f"    recording     UNAVAILABLE: {record.runner.recording_fault}")
            elif record.record:
                lines.append("    recording     asked for; begins when the camera starts")
        lines.append(f"  incidents       {len(self._incidents)}")
        # What the switch is doing, in the run's own summary: a site that turned
        # faces on and has no models must read it here, not discover it.
        lines.append(f"  identity        {self.identity_status}")

        written, chained = self.store.audit_totals()
        lines.append(
            f"  audit log       {written} row(s), {chained} chained, "
            f"{written - chained} prose-only"
        )
        head = self.store.audit_chain_head()
        lines.append(
            f"  chain head      {head}" if head
            else "  chain head      none — nothing chained has been written yet"
        )
        return "\n".join(lines)
