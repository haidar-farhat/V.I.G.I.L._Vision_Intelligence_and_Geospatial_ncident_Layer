"""Multi-object tracking, ported from v1's Rust core with its measurements.

Greedy association on IoU with a size-scaled distance gate for small fast
objects; cumulative confirmation (a track seen twice has been seen twice,
whether or not a miss came between); coasting along the smoothed velocity
through an occlusion, never recorded as a measurement of speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .detection import BoundingBox, Detection
from .geo import CameraPose, LatLon, PositionEstimate, Vec2, bearing_degrees, haversine_distance, project_point

#: The most the distance gate widens for a track that has been missed.
MAX_GATE_WIDENING = 3.0


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    iou_threshold: float = 0.2
    gate_factor: float = 2.5
    max_gap_millis: int = 2000
    #: Cumulative, not consecutive. Measured in v1: consecutive took three
    #: people to eight tracks at 0.69 recall.
    min_hits_to_confirm: int = 2
    motion_window_millis: int = 3000
    #: Below this span a speed is ``None``, not a number: ±1.3 m over 200 ms
    #: is ±9 m/s of noise.
    min_motion_span_millis: int = 1200


@dataclass(slots=True)
class Track:
    id: int
    class_id: int
    first_seen_millis: int
    last_seen_millis: int
    last_detected_millis: int
    bbox: BoundingBox
    contact: Vec2
    velocity: Vec2 = Vec2(0.0, 0.0)
    confidence: float = 0.0
    hits: int = 1
    confirmed: bool = False
    position: PositionEstimate | None = None
    ground_history: list[tuple[int, LatLon]] = field(default_factory=list)
    speed_mps: float | None = None
    heading_degrees: float | None = None

    @property
    def age_millis(self) -> int:
        return self.last_seen_millis - self.first_seen_millis

    @property
    def coasting(self) -> bool:
        return self.last_seen_millis > self.last_detected_millis


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    ended: tuple[int, ...]


class TrackerProtocol(Protocol):
    def update(self, detections: Sequence[Detection], at_millis: int) -> TrackerUpdate: ...
    def tracks(self) -> list[Track]: ...
    def reset(self) -> list[int]: ...
    def set_pose(self, pose: CameraPose | None) -> None: ...


class Tracker:
    def __init__(self, config: TrackerConfig | None = None, pose: CameraPose | None = None):
        self.config = config or TrackerConfig()
        self._pose = pose
        self._tracks: list[Track] = []
        self._next_id = 1
        self._last_update: int | None = None

    def set_pose(self, pose: CameraPose | None) -> None:
        self._pose = pose

    @property
    def pose(self) -> CameraPose | None:
        return self._pose

    def tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.confirmed]

    def all_tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> list[int]:
        ended = [t.id for t in self._tracks]
        self._tracks.clear()
        return ended

    def update(self, detections: Sequence[Detection], at_millis: int) -> TrackerUpdate:
        predicted = [self._predict(t, at_millis) for t in self._tracks]
        interval = float(at_millis - self._last_update) if self._last_update is not None and at_millis > self._last_update else 0.0
        self._last_update = at_millis

        candidates: list[tuple[int, int, float]] = []
        for ti, track in enumerate(self._tracks):
            for di, detection in enumerate(detections):
                if detection.class_id != track.class_id:
                    continue
                gap = float(max(0, at_millis - track.last_detected_millis))
                elapsed_ratio = gap / interval if interval > 0 else 1.0
                score = self._association_score(predicted[ti], detection.bbox, elapsed_ratio)
                if score is not None:
                    candidates.append((ti, di, score))
        candidates.sort(key=lambda c: (-c[2], self._tracks[c[0]].id))

        claimed_tracks = [False] * len(self._tracks)
        claimed_detections = [False] * len(detections)
        for ti, di, _ in candidates:
            if claimed_tracks[ti] or claimed_detections[di]:
                continue
            claimed_tracks[ti] = claimed_detections[di] = True
            self._apply(self._tracks[ti], detections[di], at_millis)

        for di, detection in enumerate(detections):
            if claimed_detections[di]:
                continue
            track = Track(
                id=self._next_id, class_id=detection.class_id,
                first_seen_millis=at_millis, last_seen_millis=at_millis, last_detected_millis=at_millis,
                bbox=detection.bbox, contact=detection.ground_contact, confidence=detection.confidence,
                hits=1, confirmed=self.config.min_hits_to_confirm <= 1,
                position=project_point(self._pose, detection.ground_contact) if self._pose else None,
            )
            self._next_id += 1
            self._tracks.append(track)

        ended: list[int] = []
        for index, track in enumerate(self._tracks):
            if index >= len(claimed_tracks) or claimed_tracks[index]:
                continue
            if at_millis - track.last_detected_millis > self.config.max_gap_millis:
                ended.append(track.id)
                continue
            if track.confirmed:
                coasted = self._predict(track, at_millis).clamped()
                track.contact = Vec2(track.contact.x + (coasted.x - track.bbox.x), track.contact.y + (coasted.y - track.bbox.y))
                track.bbox = coasted
                track.last_seen_millis = at_millis
                track.position = project_point(self._pose, track.contact) if self._pose else None
        if ended:
            gone = set(ended)
            self._tracks = [t for t in self._tracks if t.id not in gone]
        return TrackerUpdate(tuple(ended))

    # ------------------------------------------------------------ internals

    @staticmethod
    def _predict(track: Track, at_millis: int) -> BoundingBox:
        dt = float(at_millis - track.last_seen_millis)
        if dt <= 0:
            return track.bbox
        return BoundingBox(track.bbox.x + track.velocity.x * dt, track.bbox.y + track.velocity.y * dt, track.bbox.width, track.bbox.height)

    def _association_score(self, predicted: BoundingBox, observed: BoundingBox, elapsed_ratio: float) -> float | None:
        iou = predicted.iou(observed)
        if iou >= self.config.iou_threshold:
            return 1.0 + iou
        size = max(predicted.width, predicted.height, observed.width, observed.height, 1e-6)
        widening = min(MAX_GATE_WIDENING, max(1.0, elapsed_ratio))
        gate = self.config.gate_factor * size * widening
        pc, oc = predicted.center, observed.center
        distance = math.hypot(pc.x - oc.x, pc.y - oc.y)
        if distance <= gate:
            return 1.0 - distance / gate
        return None

    def _apply(self, track: Track, detection: Detection, at_millis: int) -> None:
        dt = float(at_millis - track.last_seen_millis)
        if dt > 0:
            previous, following = track.bbox.center, detection.bbox.center
            instant = Vec2((following.x - previous.x) / dt, (following.y - previous.y) / dt)
            track.velocity = Vec2(track.velocity.x * 0.6 + instant.x * 0.4, track.velocity.y * 0.6 + instant.y * 0.4)
        track.bbox = detection.bbox
        track.contact = detection.ground_contact
        track.last_seen_millis = at_millis
        track.last_detected_millis = at_millis
        track.confidence = track.confidence * 0.7 + detection.confidence * 0.3
        track.hits += 1
        if track.hits >= self.config.min_hits_to_confirm:
            track.confirmed = True
        if self._pose is not None:
            track.position = project_point(self._pose, detection.ground_contact)
            self._record_ground(track, at_millis)

    def _record_ground(self, track: Track, at_millis: int) -> None:
        if track.position is None or not track.position.is_projected:
            return
        track.ground_history.append((at_millis, track.position.point))
        oldest = at_millis - self.config.motion_window_millis
        track.ground_history = [(t, p) for t, p in track.ground_history if t >= oldest]
        first_t, first_p = track.ground_history[0]
        span = at_millis - first_t
        if span < self.config.min_motion_span_millis:
            track.speed_mps = None
            track.heading_degrees = None
            return
        distance = haversine_distance(first_p, track.position.point)
        track.speed_mps = distance / (span / 1000.0)
        track.heading_degrees = bearing_degrees(first_p, track.position.point) if distance > 0.25 else None
