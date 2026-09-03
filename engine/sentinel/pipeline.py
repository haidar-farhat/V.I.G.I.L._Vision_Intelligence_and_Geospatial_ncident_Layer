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
from .zones import Zone, ZoneEvaluator

from .logs import get as _get_logger
from .recording import Recorder

_log = _get_logger(__name__)


#: How many tracks' per-track detail the statistics keep. Everything that grows
#: per track — the id set, the observation counts, the spans — is trimmed to
#: this, because a node that runs for a month sees an unbounded number of
#: objects and none of that detail is read once the track has ended. The count
#: of distinct objects is NOT derived from those, so trimming cannot change it.
_MAX_TRACKED_DETAIL = 4096


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
    #: Frames where a track was held open with no detection supporting it.
    held_without_detection: int = 0
    #: Zone presences opened and closed.
    presences_started: int = 0
    presences_ended: int = 0
    #: Events raised. The number that matters most, and the one that should stay
    #: small: this system is measured by how little it says.
    events: int = 0

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
            f"distinct objects      {self.distinct_objects}",
            f"zone presences        {self.presences_started}",
            f"events                {self.events}",
        ]
        for track_id in sorted(self.track_ids):
            seen = self.observations.get(track_id, 0)
            lines.append(
                f"  track {track_id:<3} observed in {seen:>4} frames, "
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
                 "_stopping", "_stream")

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
    ):
        """
        ``max_gap_millis`` is how long a track survives without a detection. It
        is the pipeline's single most consequential number: too short and a
        detector dropout splits one person into two incidents; too long and two
        people who passed the same spot a second apart become one. The default
        of two seconds assumes a detector that misses intermittently, which the
        motion detector demonstrably does.
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
        self._source.close()

    @property
    def recorder(self) -> Recorder | None:
        """The recorder, once :meth:`run` has started one."""
        return self._recorder

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

        if self._record_to is not None:
            self._recorder = Recorder(
                self._source.source_id,
                self._record_to / self._source.source_id,
                fps=info.fps,
                # A file must not lose frames; a camera must not build a
                # backlog. The whole difference is this argument.
                live=info.is_live,
                # A file's frames are stamped from the start of the recording,
                # and retention works in days.
                epoch_millis=self._wall_clock_epoch(),
                segment_seconds=self._segment_seconds,
                on_segment=self._on_segment,
            )
            self._recorder.start()

        _log.info(
            "%s: analysis started (detector %s, %s, %d zone(s), %d rule(s))",
            self._source.source_id,
            self._detector.info.name,
            "placed" if self._pose else "not placed",
            len(self._zones),
            len(self._engine.rules) if self._engine else 0,
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
                "%s: analysis finished: %d frames, %d detections, %d object(s), "
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
        events = self._evaluate(frame, tracks)

        return FrameResult(
            index=frame.index,
            timestamp_millis=frame.timestamp_millis,
            source_id=frame.source_id,
            detections=tuple(detections),
            tracks=tuple(tracks),
            ended=tuple(ended),
            events=tuple(events),
            image=frame.image if self._keep_images else None,
        )

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

        for track in tracks:
            if track.id not in stats.track_ids:
                stats.objects_seen += 1
                stats.track_ids.add(track.id)
            stats.observations[track.id] = stats.observations.get(track.id, 0) + 1
            span = stats.spans.get(track.id)
            if span is None:
                stats.spans[track.id] = (track.first_seen_millis, track.last_seen_millis)
            else:
                stats.spans[track.id] = (span[0], max(span[1], track.last_seen_millis))
