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

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .core import CameraPose, Detection, Track, Tracker
from .decode import Frame, VideoSource
from .detect import Detector, DetectorInfo
from .events import Event, EventEngine, Rule, utc_from_millis
from .incidents import Correlator, Incident
from .zones import Zone, ZoneEvaluator


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
    #: Track ids ever seen. Its size is how many distinct objects the system
    #: believes it saw — the number that matters, and the number a fragmenting
    #: tracker inflates.
    track_ids: set[int] = field(default_factory=set)
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
        return len(self.track_ids)

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
                 "_correlator", "_recent_events", "_event_retention")

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

        self._zones = {zone.id: zone for zone in zones}
        self._evaluator = ZoneEvaluator(zones) if zones else None
        self._engine = (
            EventEngine(rules, node_id=node_id, camera_id=source.source_id)
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

    def run(self) -> Iterator[FrameResult]:
        """Process the source, yielding one result per frame."""
        self._source.open()
        self._tracker = Tracker(self._pose, **self._config)

        try:
            for frame in self._source:
                yield self._process(frame)
        finally:
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None

    def _process(self, frame: Frame) -> FrameResult:
        assert self._tracker is not None

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

    def _evaluate(self, frame: Frame, tracks: Sequence[Track]) -> list[Event]:
        """Zones and rules, if any are configured."""
        if self._evaluator is None:
            return []

        moment = utc_from_millis((self._epoch_millis or 0) + frame.timestamp_millis)
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

        for track in tracks:
            stats.track_ids.add(track.id)
            stats.observations[track.id] = stats.observations.get(track.id, 0) + 1
            span = stats.spans.get(track.id)
            if span is None:
                stats.spans[track.id] = (track.first_seen_millis, track.last_seen_millis)
            else:
                stats.spans[track.id] = (span[0], max(span[1], track.last_seen_millis))
