"""The per-camera pipeline: decode, detect, track, place on the map.

VIDEO -> DETECTION -> TRACKING -> SPATIAL CONTEXT

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

The pipeline records what it did as well as what it found. A track's identity is
only meaningful alongside how often the detector actually saw it, so the
statistics are collected here rather than inferred later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .core import CameraPose, Detection, Track, Tracker
from .decode import Frame, VideoSource
from .detect import Detector, DetectorInfo


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

    __slots__ = ("_source", "_detector", "_pose", "_tracker", "stats", "_config")

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
        self._config = dict(
            iou_threshold=iou_threshold,
            gate_factor=gate_factor,
            max_gap_millis=max_gap_millis,
            min_hits_to_confirm=min_hits_to_confirm,
        )
        self._tracker: Tracker | None = None
        self.stats = PipelineStats()

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

        return FrameResult(
            index=frame.index,
            timestamp_millis=frame.timestamp_millis,
            source_id=frame.source_id,
            detections=tuple(detections),
            tracks=tuple(tracks),
            ended=tuple(ended),
        )

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
