"""The per-camera pipeline: decode, detect, track, place on the map.

VIDEO -> DETECTION -> TRACKING -> SPATIAL CONTEXT -> TEMPORAL CONTEXT
      -> EVENT ANALYSIS -> CORRELATION -> INCIDENT

Each stage answers a different question, and the value is in the sequence rather
than any one of them:

- **Decode** asks *what did the camera see, and when*. The timestamp is the part
  everything downstream depends on.
- **Detect** asks *what is in this frame*. It is frame-local and has no memory,
  so it is wrong intermittently — a person is found in one frame and missed in
  the next, and background subtraction loses an object entirely once it stops
  moving.
- **Track** asks *is this the same thing as before*. This is the stage that
  turns an unreliable per-frame detector into a continuous object, by holding
  identity across the gaps the detector leaves. Without it the system reports a
  new intruder every frame.
- **Place** asks *where on the ground is this*. Projected through the camera
  pose, with an uncertainty that is part of the answer rather than a footnote.
- **Read**, where the operator has supplied plate models, asks *which vehicle is
  this*. It runs inside vehicle track boxes only — never over the frame, which
  would read the street outside the site boundary — and it answers with the
  characters several frames of one track agreed on and a ``?`` for the rest.
- **Zones and rules** ask *does any of this mean anything*. This is where
  observation becomes assertion, and it is the first stage whose output is
  intended to interrupt a person — so it is the first that has to justify
  itself. Every event carries the evidence for it.

The pipeline records what it did as well as what it found. A track's identity is
only meaningful alongside how often the detector actually saw it, so the
statistics are collected here rather than inferred later.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .core import CameraPose, Detection, Track, Tracker
from .decode import Frame, LiveStream, VideoSource
from .detect import Detector, DetectorInfo
from .events import Event, EventEngine, Rule, utc_from_millis
from .incidents import Correlator, Incident
from .plates import (
    MIN_AGREEMENT,
    MIN_CHARACTER_CONFIDENCE,
    PlateAccumulator,
    PlateRead,
    PlateReader,
    Reading,
)
from .zones import Zone, ZoneEvaluator

from .logs import get as _get_logger
from .recording import Recorder, file_safe

_log = _get_logger(__name__)


#: How many tracks' per-track detail the statistics keep. Everything that grows
#: per track — the id set, the observation counts, the spans — is trimmed to
#: this, because a node that runs for a month sees an unbounded number of
#: objects and none of that detail is read once the track has ended. The count
#: of distinct objects is NOT derived from those, so trimming cannot change it.
_MAX_TRACKED_DETAIL = 4096

#: How often one track's plate is read, in reads per second of source time.
#: The reader was called on every frame of every vehicle track until this
#: existed, and that cost is quadratic rather than linear: each read is appended
#: to that track's accumulator and every read already in it is re-tallied on the
#: next frame. Measured on this machine, injected models, one parked vehicle:
#: 3200 frames cost 11.64s with no bound and 0.10s with these, the accumulator
#: holding one read per frame — 3199 of them — in the first case; longer runs
#: went as far as 40.3s for 6400 frames. A car parked in front of a gate camera
#: for four minutes is the ordinary case, not the adversarial one.
#:
#: Three a second is more evidence per second than a reading needs — a character
#: resolves on :data:`~sentinel.plates.MIN_AGREEMENT` reads in total — and it is
#: bounded by the clock rather than by the frame rate, so a 60 fps camera costs
#: what a 25 fps one does instead of two and a half times as much.
PLATE_READS_PER_SECOND = 3.0

#: Reads one track's accumulator may hold before this stage stops reading that
#: track at all. The stride above bounds the rate; this bounds the total, which
#: is what a vehicle that parks in shot for a shift needs — at three a second
#: and 387 bytes a read, an unbounded accumulator is 4 MB per hour per vehicle
#: and a resolve() over it that grows without end.
#:
#: The cost of the ceiling, stated because it is real: a vehicle whose plate is
#: unreadable for the first twenty seconds of its track — approaching from far
#: enough away that every read is noise — is not read afterwards either. That is
#: the wrong trade for a long approach road and the right one for everything
#: else; the fix if footage shows it mattering is a sliding window inside
#: :class:`~sentinel.plates.PlateAccumulator`, not a bigger number here, because
#: a bigger number only moves where the growth stops.
MAX_PLATE_READS_PER_TRACK = 64

#: Frame rate assumed for the stride when the source reports none. A source that
#: cannot say how fast it runs must still be rate-limited: falling back to
#: "every frame" would restore the quadratic cost above on exactly the sources
#: least able to afford it.
_ASSUMED_FPS = 25.0

#: Detector labels whose boxes a plate may be read inside. A reader pointed at
#: anything else is a reader pointed at whatever text is in shot — a sign, a
#: hoarding, a delivery driver's shirt — and every string lifted from those
#: arrives looking exactly like a plate. `plates.py` calls reading from a
#: vehicle box rather than from a frame its central promise; this set is where
#: the pipeline, which is the only stage holding the whole image, keeps it.
#:
#: Matched by label rather than by class id because the ids are the model's, and
#: a site that changes model must not silently start reading plates off people.
VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})


@dataclass(frozen=True, slots=True)
class TrackPlate:
    """What one vehicle track's plate has been read as, so far.

    Three separate facts rather than one string, because keeping them apart is
    what stops a half-read plate reaching a watchlist. :attr:`display` is for an
    operator and carries ``?`` wherever a character has not resolved.
    :attr:`text` is ``None`` until every character has, so there is no completed
    string for a rule or an export to match by accident. :attr:`is_confident`
    says whether the *weakest* character has enough agreement behind it to act
    on, which is a higher bar than being resolved.

    The counts travel with them for the same reason `plates.py` keeps them: a
    plate shown without its evidence is a conclusion presented as a fact, and a
    reading from four reads is a different thing from one from forty.
    """

    track_id: int
    country: str
    #: What an operator is shown. Contains ``?`` where nothing resolved.
    display: str
    #: The plate, or ``None`` while any character is unresolved.
    text: str | None
    #: Whether this may be matched against a register or a watchlist.
    is_confident: bool
    #: Reads behind the least-agreed character; ``0`` if any is unresolved.
    agreement: int
    #: Reads that voted, out of everything this track offered.
    reads: int

    @classmethod
    def of(cls, track_id: int, reading: Reading) -> "TrackPlate":
        """Flatten a :class:`~sentinel.plates.Reading` for one track.

        Flattened rather than passed whole so that nothing downstream can reach
        past ``text`` for the characters behind it and reassemble the completed
        string the reading refuses to produce.
        """
        return cls(
            track_id=track_id,
            country=reading.country,
            display=reading.display,
            text=reading.text,
            is_confident=reading.is_confident,
            agreement=reading.weakest_agreement,
            reads=reading.contributing_reads,
        )


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything the pipeline concluded about one frame.

    Detections and tracks are both present on purpose. A track is a claim that
    persists across frames; a detection is the evidence for it in this one. An
    operator reviewing an incident needs to see both, because a track held open
    through eight frames of nothing is a weaker claim than one confirmed every
    frame — and the interface should be able to show the difference.
    """

    index: int
    timestamp_millis: int
    source_id: str
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    #: Track ids that closed on this frame.
    ended: tuple[int, ...]
    #: Events raised on this frame. Usually empty — that is the point.
    events: tuple[Event, ...] = ()
    #: One entry per vehicle track something has been read on, so a caller can
    #: show the plate beside the box it came from. Always empty unless the
    #: pipeline was given a plate reader.
    plates: tuple[TrackPlate, ...] = ()
    #: The frame these conclusions were drawn from, when the caller asked for
    #: it. Opt-in because a full-resolution image per result is tens of
    #: megabytes over a short clip, and most callers want the conclusions only.
    #: A viewer needs it, though: drawing a track box over a different frame
    #: than the one it was computed from misrepresents what the system saw.
    image: np.ndarray | None = None


@dataclass
class PipelineStats:
    """What actually happened, measured rather than assumed."""

    frames: int = 0
    frames_with_detections: int = 0
    detections: int = 0
    #: Track ids seen recently, trimmed to ``_MAX_TRACKED_DETAIL``. This is a
    #: window for reporting, not a census: read ``distinct_objects`` for the
    #: count, which is counted on arrival and never trimmed.
    track_ids: set[int] = field(default_factory=set)
    #: Distinct objects the system believes it saw — the number that matters,
    #: and the number a fragmenting tracker inflates. Counted as ids arrive so
    #: that bounding the sets above cannot quietly deflate it.
    objects_seen: int = 0
    #: Per track: how many frames it was confirmed by a detection.
    observations: dict[int, int] = field(default_factory=dict)
    #: Per track: first and last timestamp.
    spans: dict[int, tuple[int, int]] = field(default_factory=dict)
    #: Per track: what the detector called it. Without this the summary read
    #: "track 1 observed in 344 frames" and an operator could not tell a
    #: person from the sofa behind them — nor whether a silent run had
    #: ignored a person or never seen one.
    labels: dict[int, str] = field(default_factory=dict)
    #: Frames where a track was held open with no detection supporting it.
    held_without_detection: int = 0
    #: Zone presences opened and closed.
    presences_started: int = 0
    presences_ended: int = 0
    #: Events raised. The number that matters most, and the one that should stay
    #: small: this system is measured by how little it says.
    events: int = 0
    #: Plate reads that *voted*, across every vehicle track — not reads the
    #: reader returned. A read whose text normalises to nothing is dropped by
    #: the accumulator without a vote, and counting those here would show a
    #: reader lifting only garbage as a reader finding plates. Counted because a
    #: reader that is running and finding nothing looks exactly like a reader
    #: that is switched off, and the two want different remedies.
    plate_reads: int = 0

    @property
    def distinct_objects(self) -> int:
        return self.objects_seen

    @property
    def mean_detections_per_frame(self) -> float:
        return self.detections / self.frames if self.frames else 0.0

    def duration_millis(self, track_id: int) -> int:
        span = self.spans.get(track_id)
        return span[1] - span[0] if span else 0

    def summary(self) -> str:
        lines = [
            f"frames                {self.frames}",
            f"frames with detections{self.frames_with_detections:>6}",
            f"detections            {self.detections}",
            # Ids the tracker issued, which is not a count of objects: one
            # object can hold several when the tracker loses and re-finds it.
            # It was labelled "distinct objects" for a long time, and a person
            # reading "12 distinct objects" about one colleague stops
            # believing counts.
            f"distinct tracks       {self.distinct_objects}"
            "   (ids issued; one object can hold several)",
            f"zone presences        {self.presences_started}",
            f"events                {self.events}",
        ]
        if self.plate_reads:
            lines.append(f"plate reads           {self.plate_reads}")
        for track_id in sorted(self.track_ids):
            seen = self.observations.get(track_id, 0)
            lines.append(
                f"  track {track_id:<3} {self.labels.get(track_id, ''):<12} "
                f"observed in {seen:>4} frames, "
                f"spanning {self.duration_millis(track_id) / 1000:.1f}s"
            )
        return "\n".join(lines)


class Pipeline:
    """One camera's worth of processing.

    Deliberately single-camera and stateless between runs. Multi-camera
    correlation is a separate stage operating on the events this produces,
    because a camera that cannot be reasoned about alone cannot be reasoned about
    in a group either.
    """

    __slots__ = ("_source", "_detector", "_pose", "_tracker", "stats", "_config",
                 "_keep_images", "_zones", "_evaluator", "_engine", "_epoch_millis",
                 "_correlator", "_recent_events", "_event_retention",
                 "_resolved_epoch", "_epoch_basis",
                 "_record_to", "_segment_seconds", "_on_segment", "_recorder",
                 "_recording_fault",
                 "_stopping", "_stream",
                 "_plate_reader", "_plate_settings", "_plates", "_vehicle_class_ids",
                 "_plate_last", "_plate_next_read", "_plate_stride", "_plate_fault")

    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        *,
        pose: CameraPose | None = None,
        iou_threshold: float = 0.2,
        gate_factor: float = 2.5,
        max_gap_millis: int = 2000,
        min_hits_to_confirm: int = 2,
        keep_images: bool = False,
        zones: Sequence[Zone] = (),
        rules: Sequence[Rule] = (),
        node_id: str = "local",
        wall_clock_epoch_millis: int | None = None,
        event_retention: int = 5000,
        record_to: str | Path | None = None,
        segment_seconds: float = 60.0,
        on_segment=None,
        site_tz=None,
        plate_reader: PlateReader | None = None,
        plate_min_agreement: int = MIN_AGREEMENT,
        plate_min_character_confidence: float = MIN_CHARACTER_CONFIDENCE,
    ):
        """
        ``max_gap_millis`` is how long a track survives without a detection. It
        is the pipeline's single most consequential number: too short and a
        detector dropout splits one person into two incidents; too long and two
        people who passed the same spot a second apart become one. The default
        of two seconds assumes a detector that misses intermittently, which the
        motion detector demonstrably does.

        ``plate_reader`` is off unless the operator has supplied one, and with
        none supplied nothing about a run changes and nothing is paid for the
        ability: no crop is taken, no accumulator is made, no inference is run.
        The two thresholds beside it are the bars a character has to clear to
        vote and to resolve; they default to the ones `plates.py` argues for,
        and lowering them lowers the standard of every plate this camera
        reports. What a supplied reader costs per second is fixed by
        :data:`PLATE_READS_PER_SECOND` and :data:`MAX_PLATE_READS_PER_TRACK`
        rather than by the frame rate — see :meth:`_read_plates`.
        """
        self._source = source
        self._detector = detector
        self._pose = pose
        self._keep_images = keep_images
        self._config = dict(
            iou_threshold=iou_threshold,
            gate_factor=gate_factor,
            max_gap_millis=max_gap_millis,
            min_hits_to_confirm=min_hits_to_confirm,
        )
        self._tracker: Tracker | None = None
        self.stats = PipelineStats()

        # A live run has no end of its own, so it needs to be told. Checking
        # this between frames is not enough: a camera that has gone quiet
        # produces no frames to check between, and the caller's own loop never
        # gets a turn. The read loop below waits on this directly.
        self._stopping = threading.Event()
        self._stream: LiveStream | None = None

        # Recording is opt-in, and off by default. Writing video is the single
        # most expensive thing this system can do to a disk — roughly 17.5 GB
        # per camera per day — so it happens because somebody asked for it.
        self._record_to = Path(record_to) if record_to else None
        self._segment_seconds = segment_seconds
        self._on_segment = on_segment
        self._recorder: Recorder | None = None
        self._recording_fault: str | None = None

        # Plate reading is opt-in for the same reason recording is: it costs a
        # crop and two model calls per vehicle per frame, and a camera watching
        # a footpath has no use for either. With no reader the whole stage is
        # skipped rather than run over nothing.
        self._plate_reader = plate_reader
        self._plate_settings = (plate_min_agreement, plate_min_character_confidence)
        #: One accumulator per live vehicle track, dropped as the track ends.
        self._plates: dict[int, PlateAccumulator] = {}
        #: The last reading published for each live track, republished on the
        #: frames between reads so a plate does not blink out between them.
        self._plate_last: dict[int, TrackPlate] = {}
        #: Frame index at which each live track may next be read. Per track
        #: rather than per frame so twenty parked cars do not all come due on
        #: the same frame and turn a rate limit into a periodic stall.
        self._plate_next_read: dict[int, int] = {}
        #: Frames between reads of one track. Replaced from the source's own
        #: frame rate in `run()`; the value here is what an unopened pipeline
        #: would use.
        self._plate_stride = max(1, round(_ASSUMED_FPS / PLATE_READS_PER_SECOND))
        self._plate_fault: str | None = None
        self._vehicle_class_ids: frozenset[int] = frozenset()
        if plate_reader is not None:
            info = detector.info
            self._vehicle_class_ids = frozenset(
                class_id
                for class_id, label in info.class_names.items()
                if label.strip().lower() in VEHICLE_LABELS
            )
            if not self._vehicle_class_ids:
                # Silence here would look identical to a car park where no
                # vehicle ever came: models loaded, reader running, not one
                # plate ever read, and nothing said why. A detector that cannot
                # name a vehicle cannot hand this reader a vehicle box.
                _log.warning(
                    "%s: a plate reader was supplied, but the detector (%s) "
                    "labels no vehicle class of %s — no plate will be read. "
                    "Plates are read inside vehicle boxes only.",
                    source.source_id,
                    detector.info.name,
                    ", ".join(sorted(VEHICLE_LABELS)),
                )

        self._zones = {zone.id: zone for zone in zones}
        # The clock zone schedules are written in. `None` keeps UTC, which is
        # what every test that hands the evaluator a UTC moment expects; the
        # node passes the machine's zone.
        self._evaluator = ZoneEvaluator(zones, site_tz=site_tz) if zones else None
        self._engine = (
            EventEngine(rules, node_id=node_id, camera_id=source.source_id, site_tz=site_tz)
            if rules
            else None
        )
        # Media time is relative to the start of the recording; rules that depend
        # on the time of day need wall-clock time. Supplying the epoch explicitly
        # keeps a replay reproducible: reading the system clock here would make
        # the same footage produce different events on different days, which is
        # exactly what an evidence trail must not do.
        self._epoch_millis = wall_clock_epoch_millis

        self._event_retention = event_retention
        self._resolved_epoch: int | None = None
        self._epoch_basis = "not yet determined"
        self._correlator = Correlator(
            zone_kinds={zone.id: zone.kind for zone in zones}
        )
        # Retained so correlation can be re-run over a window. Bounded, because a
        # node running for a month must not accumulate every event it ever saw in
        # memory; persistence is where the full history belongs.
        self._recent_events: list[Event] = []

    @property
    def detector_info(self) -> DetectorInfo:
        return self._detector.info

    @property
    def pose(self) -> CameraPose | None:
        return self._pose

    def set_pose(self, pose: CameraPose | None) -> None:
        """Move the camera. Existing tracks keep their identity.

        A PTZ camera that moves does not make its tracked objects new objects;
        it only changes where they land on the map.
        """
        self._pose = pose
        if self._tracker is not None:
            self._tracker.set_pose(pose)

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        # Each accumulator holds the reads of one vehicle, crops included, and
        # each published reading holds that vehicle's plate. Keeping either past
        # the run would hold the plates of every vehicle in the last camera this
        # object watched.
        self._plates.clear()
        self._plate_last.clear()
        self._plate_next_read.clear()
        self._source.close()

    @property
    def plate_fault(self) -> str | None:
        """Why plate reading stopped, or ``None`` if it did not.

        Read it the way `RecorderStats.fault` is read: a camera whose plate
        models died in the first minute must not report the same summary as one
        that read plates all night. The run continues either way — see
        :meth:`_read_plates` for why a plate model's failure is not treated as
        the analysis failing.
        """
        return self._plate_fault

    @property
    def recorder(self) -> Recorder | None:
        """The recorder, once :meth:`run` has started one."""
        return self._recorder

    @property
    def recording_fault(self) -> str | None:
        """Why no recorder is running although one was asked for, or ``None``.

        Distinct from `RecorderStats.fault`, which is a recorder that started
        and then died. This is one that never started — a directory that
        could not be made, a codec the build lacks — and the analysis went on
        without it. Read the way `plate_fault` is read: a camera that was
        asked to record and is not must never report the same line as one
        that is.
        """
        return self._recording_fault

    def ask_to_stop(self) -> None:
        """End a live run. Safe from any thread, and returns immediately."""
        self._stopping.set()

    @property
    def stream(self) -> LiveStream | None:
        """The live reader, once :meth:`run` has started one. ``None`` for a file."""
        return self._stream

    def run(self) -> Iterator[FrameResult]:
        """Process the source, yielding one result per frame."""
        info = self._source.open()
        self._tracker = Tracker(self._pose, **self._config)

        # Reads are spaced by the clock, not by the frame: the cost of reading
        # a stationary vehicle must not double because the camera was swapped
        # for a faster one.
        self._plate_stride = max(
            1,
            round((info.fps if info.fps and info.fps > 0 else _ASSUMED_FPS)
                  / PLATE_READS_PER_SECOND),
        )

        if self._record_to is not None:
            # Guarded, because the first packaged camera run died here: the
            # camera id was `device:0`, the directory `recordings/device:0`
            # cannot exist on Windows, `start()` raised in `mkdir`, and the
            # whole camera stopped before a frame was analysed. Recording that
            # cannot begin is a fault to report, not a reason to stop watching
            # — the recorder is fed *before* analysis precisely so that the two
            # cannot take each other down.
            try:
                recorder = Recorder(
                    self._source.source_id,
                    # A camera id is not a directory name: see `file_safe`.
                    self._record_to / file_safe(self._source.source_id),
                    fps=info.fps,
                    # A file must not lose frames; a camera must not build a
                    # backlog. The whole difference is this argument.
                    live=info.is_live,
                    # A file's frames are stamped from the start of the
                    # recording, and retention works in days.
                    epoch_millis=self._wall_clock_epoch(),
                    segment_seconds=self._segment_seconds,
                    on_segment=self._on_segment,
                )
                recorder.start()
            except Exception as error:  # noqa: BLE001 - the disk is not the analysis
                self._recording_fault = f"{type(error).__name__}: {error}"
                _log.error(
                    "%s: RECORDING UNAVAILABLE — %s. Analysis continues without it; "
                    "nothing from this run will be on disk.",
                    self._source.source_id, self._recording_fault,
                )
            else:
                self._recorder = recorder

        _log.info(
            "%s: analysis started (detector %s, %s, %d zone(s), %d rule(s)%s)",
            self._source.source_id,
            self._detector.info.name,
            "placed" if self._pose else "not placed",
            len(self._zones),
            len(self._engine.rules) if self._engine else 0,
            # Named in the same line as the detector because a plate is
            # personal data almost everywhere, and a run that reads them should
            # be visibly a run that reads them.
            f", plates in {self._plate_reader.country} every "
            f"{self._plate_stride} frame(s)"
            if self._plate_reader is not None
            else "",
        )

        try:
            if info.is_live:
                yield from self._run_live()
            else:
                # A file has an end, and every frame of it matters. Iterating
                # the source directly is what makes replay deterministic.
                for frame in self._source:
                    yield self._process(frame)
        finally:
            if self._recorder is not None:
                segments = self._recorder.close()
                # The last segment closes during `close()`, on the writer
                # thread, so it is still waiting to be indexed here.
                self._recorder.finished()
                stats = self._recorder.stats
                _log.info(
                    "%s: recorded %d segment(s), %.1f MiB%s",
                    self._source.source_id, len(segments),
                    stats.bytes_written / 1024 / 1024,
                    f", {stats.frames_dropped} frame(s) dropped"
                    if stats.frames_dropped else "",
                )
                if stats.fault is not None:
                    # `RecorderStats.fault` says the pipeline surfaces it, and
                    # for a while the pipeline did not: a writer that died in
                    # the first minute of an overnight run ended with the same
                    # cheerful summary as a healthy one. Recording that is not
                    # recording must not look like recording.
                    _log.error(
                        "%s: RECORDING STOPPED EARLY — %s. Footage after that "
                        "point does not exist.",
                        self._source.source_id, stats.fault,
                    )
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None
            # At INFO because this is the line an operator sends when asked what
            # the system saw, and it is one line per run rather than per frame.
            _log.info(
                "%s: analysis finished: %d frames, %d detections, %d track(s), "
                "%d event(s)%s",
                self._source.source_id,
                self.stats.frames,
                self.stats.detections,
                self.stats.distinct_objects,
                self.stats.events,
                # Silence about dropped frames would let a camera that lost half
                # its input report the same line as one that lost none.
                (
                    f", {self._stream.dropped_frames} dropped, "
                    f"{self._stream.reconnects} reconnect(s)"
                    if self._stream is not None
                    and (self._stream.dropped_frames or self._stream.reconnects)
                    else ""
                ),
            )

    def _run_live(self) -> Iterator[FrameResult]:
        """Read a camera until told to stop.

        A file ends; a camera does not. Until this existed the pipeline iterated
        a live source the same way it iterated a file, and `VideoSource.read`
        returns ``None`` for *both* the end of a file and a single failed read —
        so one dropped frame ended the run, and the log said "analysis finished"
        as though that were the normal conclusion. A security camera that stops
        watching must never look like a camera that finished.

        `LiveStream` already had the right behaviour — reconnect with bounded
        backoff, keep only the newest frame, count what it drops — and nothing
        in the product called it.
        """
        with LiveStream(self._source) as stream:
            self._stream = stream
            while not self._stopping.is_set():
                # Raises if the reader has given up; returns `None` merely
                # because nothing arrived in time, which on a camera is a gap
                # and not an ending.
                frame = stream.read()
                if frame is None:
                    continue
                yield self._process(frame)

    def _process(self, frame: Frame) -> FrameResult:
        assert self._tracker is not None

        if self._recorder is not None:
            # Before analysis, not after. What gets recorded must not depend on
            # what the analytic concludes, or on whether it concludes anything
            # at all — an exception in a rule would otherwise take the footage
            # with it.
            self._recorder.offer(frame)
            # Indexed here, on this thread, because `on_segment` is normally a
            # database write and SQLite connections belong to the thread that
            # created them.
            self._recorder.finished()

        detections = self._detector.detect(frame.image)
        tracks = self._tracker.update(detections, frame.timestamp_millis)
        ended = self._tracker.ended()

        self._record(frame, detections, tracks)
        plates = self._read_plates(frame, tracks, ended)
        events = self._evaluate(frame, tracks)

        return FrameResult(
            index=frame.index,
            timestamp_millis=frame.timestamp_millis,
            source_id=frame.source_id,
            detections=tuple(detections),
            tracks=tuple(tracks),
            ended=tuple(ended),
            events=tuple(events),
            plates=plates,
            image=frame.image if self._keep_images else None,
        )

    def _read_plates(
        self, frame: Frame, tracks: Sequence[Track], ended: Sequence[int]
    ) -> tuple[TrackPlate, ...]:
        """Read the plate inside each vehicle track, and forget the ones that left.

        Three things this does not do, each of which is a way a plate reader
        manufactures a wrong answer:

        **It never sees the frame as a frame.** The reader is handed one track's
        box at a time, so what it reads is a vehicle this camera is tracking and
        not the traffic on the road behind the fence. The reader crops for
        itself, but only this stage knows which boxes are vehicles, so this is
        where the promise is kept or broken.

        **It reads only inside a vehicle.** A track's class must be one of
        :data:`VEHICLE_LABELS` as the detector's own labels name it. Anything
        else — a person, a bag, a blob a motion detector cannot classify at
        all — is never cropped, so there is no text lifted off a shirt or a sign
        for the accumulator to vote on.

        **It forgets a vehicle when the vehicle goes.** One accumulator per live
        track is bounded by what is on screen; one per vehicle ever seen is the
        same unbounded growth the per-track statistics were trimmed to stop,
        and each of these holds a track's reads and a crop with them.

        **It stops reading a vehicle it has already read.** Bounding the number
        of accumulators bounds nothing on its own — the *contents* grew too.
        Reading every vehicle on every frame appended a read per frame to a list
        this stage then re-tallied per frame, which is quadratic in how long a
        vehicle stays in shot and is worst for the vehicle that is easiest to
        read: the parked one. Three bounds replace it, and the reading published
        between reads is the last one, so a plate does not blink out —

        * a track whose reading is already confident is never read again: the
          bar `plates.py` sets for acting on a plate has been cleared, and the
          fortieth read of a stationary plate buys nothing the fourth did not;
        * a track due no sooner than :data:`PLATE_READS_PER_SECOND` a second is
          not read on the frames between;
        * a track holding :data:`MAX_PLATE_READS_PER_TRACK` reads is not read
          again at all, so a vehicle parked in shot for a shift costs a fixed
          amount rather than an accumulating one.

        :meth:`~sentinel.plates.PlateAccumulator.resolve` runs only on the
        frames where a read was actually taken *into* the accumulator, for the
        same reason: it re-tallies everything, and re-tallying an unchanged list
        cannot change the answer.

        **A failing plate model stops the plates, not the camera.** The models
        are third-party files the operator supplied, and their failure modes are
        not the detector's. Anything raised out of a read is caught, logged once
        with the camera it happened on, and ends plate reading for the rest of
        the run — the zone, rule and recording stages, which have nothing to do
        with plates, keep running, exactly as recording is deliberately isolated
        from an analysis failure. Once rather than per track because a model
        that throws on one crop is a model, not a crop, and per-track
        suppression would produce one error line per vehicle forever;
        :attr:`plate_fault` is how a caller finds out it happened.

        Returns nothing at all, having done nothing at all, when no reader was
        supplied.
        """
        if self._plate_reader is None:
            return ()

        for track_id in ended:
            self._plates.pop(track_id, None)
            self._plate_last.pop(track_id, None)
            self._plate_next_read.pop(track_id, None)

        min_agreement, min_confidence = self._plate_settings
        plates: list[TrackPlate] = []
        for track in tracks:
            if track.class_id not in self._vehicle_class_ids:
                continue
            plate = self._plate_last.get(track.id)
            if self._is_due_a_read(track.id, frame.index, plate):
                accumulator = self._plates.get(track.id)
                if accumulator is None:
                    accumulator = PlateAccumulator(
                        country=self._plate_reader.country,
                        min_agreement=min_agreement,
                        min_character_confidence=min_confidence,
                    )
                    self._plates[track.id] = accumulator
                self._plate_next_read[track.id] = frame.index + self._plate_stride

                held = len(accumulator)
                accumulator.add_all(self._read_one(frame, track))
                # What the accumulator took, not what the reader returned. A
                # read whose text normalises to nothing is dropped without a
                # vote, and counting it here would show a reader finding only
                # garbage as a reader finding plates — the exact confusion this
                # statistic was added to remove.
                voted = len(accumulator) - held
                self.stats.plate_reads += voted

                if voted:
                    plate = TrackPlate.of(track.id, accumulator.resolve())
                    self._plate_last[track.id] = plate

            if plate is None:
                # Nothing has been read on this vehicle yet, which is the normal
                # state of a car that is still too far away. An empty reading
                # published beside it every frame would read as "no plate"
                # rather than "not yet", and those are different claims.
                continue
            plates.append(plate)
        return tuple(plates)

    def _is_due_a_read(
        self, track_id: int, frame_index: int, plate: TrackPlate | None
    ) -> bool:
        """Whether reading this track again on this frame could tell us anything.

        The three bounds :meth:`_read_plates` describes, in the order that costs
        least to test. A track never read before is due immediately: the first
        read is the one that puts a plate on the operator's screen, and making
        it wait a third of a second would be a delay bought for nothing.
        """
        if self._plate_fault is not None:
            return False
        if plate is not None and plate.is_confident:
            return False
        accumulator = self._plates.get(track_id)
        if accumulator is not None and len(accumulator) >= MAX_PLATE_READS_PER_TRACK:
            return False
        return frame_index >= self._plate_next_read.get(track_id, frame_index)

    def _read_one(self, frame: Frame, track: Track) -> Sequence[PlateRead]:
        """One track's reads on one frame, or none at all if the model failed.

        The whole read is inside the guard rather than the model call alone,
        because a plate reader is a crop, a detector, a decode and a
        normalisation, and any of the four can raise on input the operator's
        models were not built for.
        """
        assert self._plate_reader is not None
        try:
            return self._plate_reader.read(
                frame.image, track.bbox, frame_index=frame.index
            )
        except Exception as error:  # noqa: BLE001 - see the docstring above
            self._plate_fault = f"{type(error).__name__}: {error}"
            _log.error(
                "%s: PLATE READING STOPPED — %s. The camera is still running "
                "and everything else about it is unaffected; no plate will be "
                "read on it until it is restarted.",
                self._source.source_id, self._plate_fault,
                exc_info=True,
            )
            return ()

    def _wall_clock_epoch(self) -> int:
        """When media time zero happened, in real-world milliseconds.

        This exists because the obvious fallback — treat a missing epoch as
        zero — silently dated every event, incident and evidence package in the
        shipping console to January 1970, and made `clock_skew()` report
        fifty-six years of camera drift. A wrong absolute time is worse than an
        absent one: an after-hours rule evaluates against it, and an evidence
        package carries it in front of somebody who will believe it.

        Three cases, each a real fact rather than a default:

        - **An explicit epoch** wins. A replay must reproduce the original
          wall-clock reasoning exactly, and only the caller knows when the
          footage was taken.
        - **A live source** already timestamps frames with the wall clock, so
          the epoch is zero — adding anything would double-count.
        - **A file** is dated from its own modification time, less its duration:
          the file was last written when the recording ended, so the recording
          began that much earlier. Derived, and derived from something real.

        Computed once and cached, so every frame in a run shares one basis and
        the run stays internally consistent.
        """
        if self._resolved_epoch is not None:
            return self._resolved_epoch

        if self._epoch_millis is not None:
            self._resolved_epoch = self._epoch_millis
            self._epoch_basis = "supplied by the caller"
            return self._resolved_epoch

        if self._source.is_live:
            # decode.py stamps live frames with time.time(); they are already
            # absolute.
            self._resolved_epoch = 0
            self._epoch_basis = "live source, frames are already wall-clock"
            return 0

        try:
            path = Path(self._source.display_url)
            ended_at = int(path.stat().st_mtime * 1000)
        except (OSError, ValueError):
            # Nothing real to derive from. Zero would date the run to 1970, so
            # fall back to now: an approximate time that is at least in the
            # right century, and recorded as approximate.
            self._resolved_epoch = int(time.time() * 1000)
            self._epoch_basis = "unknown; defaulted to the time of analysis"
            return self._resolved_epoch

        info = self._source.info
        duration = 0
        if info.frame_count and info.fps:
            duration = int(1000 * info.frame_count / info.fps)

        self._resolved_epoch = ended_at - duration
        self._epoch_basis = "derived from the file's modification time"
        return self._resolved_epoch

    @property
    def wall_clock_basis(self) -> str:
        """How the wall-clock time of this run was established.

        Provenance, not decoration. An operator reading an incident dated three
        weeks ago should be able to find out whether that date was measured,
        supplied, or inferred from a file's metadata.
        """
        self._wall_clock_epoch()
        return self._epoch_basis

    def _evaluate(self, frame: Frame, tracks: Sequence[Track]) -> list[Event]:
        """Zones and rules, if any are configured."""
        if self._evaluator is None:
            return []

        moment = utc_from_millis(self._wall_clock_epoch() + frame.timestamp_millis)
        changes = self._evaluator.update(tracks, frame.timestamp_millis, moment)

        self.stats.presences_started += sum(1 for c in changes if c.kind == "ENTERED")
        self.stats.presences_ended += sum(1 for c in changes if c.kind == "LEFT")

        if self._engine is None:
            return []

        by_id = {track.id: track for track in tracks}
        detector = self._detector.info

        events = self._engine.on_presence_changes(
            changes, self._zones, by_id,
            at_millis=frame.timestamp_millis, moment=moment,
            detector=detector, frame_index=frame.index,
        )
        events += self._engine.on_frame(
            self._evaluator.open_presences(), self._zones, by_id,
            at_millis=frame.timestamp_millis, moment=moment,
            detector=detector, frame_index=frame.index,
        )

        self.stats.events += len(events)

        self._recent_events.extend(events)
        if len(self._recent_events) > self._event_retention:
            del self._recent_events[: len(self._recent_events) - self._event_retention]

        return events

    def incidents(self) -> list[Incident]:
        """Correlate the retained events into incidents.

        Batch, over what is currently retained. An incident is a statement about
        a span of time, and deciding it is closed means knowing nothing more is
        coming — which a per-frame correlator can only guess at. A live console
        calls this on a timer; the semantics are the same either way, and the
        ids are deterministic so re-correlating produces the same incidents
        rather than new ones beside them.
        """
        return Correlator(
            zone_kinds={zone.id: zone.kind for zone in self._zones.values()}
        ).correlate(self._recent_events)

    @property
    def recent_events(self) -> tuple[Event, ...]:
        return tuple(self._recent_events)

    def _record(
        self, frame: Frame, detections: Sequence[Detection], tracks: Sequence[Track]
    ) -> None:
        stats = self.stats
        stats.frames += 1
        stats.detections += len(detections)
        if detections:
            stats.frames_with_detections += 1
        elif tracks:
            # A track alive on a frame with no evidence for it. Counting these
            # keeps the system honest about how much of what it reports is
            # observation and how much is inference.
            stats.held_without_detection += 1

        # Bounded. These three held an entry for every track ever seen, so a
        # node running for a month accumulated one per object that ever crossed
        # the frame and released none of them. Half the oldest detail is dropped
        # at the ceiling, skipping ids still live this frame: an id the tracker
        # has issued is never issued again, so a dropped one cannot come back
        # and be miscounted as a new object.
        if len(stats.track_ids) > _MAX_TRACKED_DETAIL:
            live = {track.id for track in tracks}
            stale = [id for id in sorted(stats.track_ids) if id not in live]
            for id in stale[: len(stale) // 2]:
                stats.track_ids.discard(id)
                stats.observations.pop(id, None)
                stats.spans.pop(id, None)
                stats.labels.pop(id, None)

        for track in tracks:
            if track.id not in stats.track_ids:
                stats.objects_seen += 1
                stats.track_ids.add(track.id)
                stats.labels[track.id] = self._detector.info.label_for(track.class_id)
            stats.observations[track.id] = stats.observations.get(track.id, 0) + 1
            span = stats.spans.get(track.id)
            if span is None:
                stats.spans[track.id] = (track.first_seen_millis, track.last_seen_millis)
            else:
                stats.spans[track.id] = (span[0], max(span[1], track.last_seen_millis))
