"""Multi-object tracking.

# What this replaced, and why none of it could be patched

v1's Rust tracker and v2's Python port of it shared one design: velocity as an
exponential moving average of box displacement, association by a **greedy**
pass over IoU with a size-scaled distance gate, and a two-second hold before a
track was thrown away. Four things follow from that, and all four were visible
in the product rather than theoretical.

**Greedy association swaps identities.** Sort every pair by score and take
each in turn: the single best-scoring pair locks in, and whoever is left is
matched to whatever remains. Two people crossing is the case that breaks it,
and it is not rare — it is what people do. The fix is a globally optimal
matching, which minimises the *total* cost and therefore rejects a locally
attractive pair that forces an expensive remainder. See `core/src/assign.rs`
for the worked two-by-two example.

**Position alone cannot re-identify.** v1 measured this: twenty seconds of one
person counted 3, 10, 4 and 11 distinct objects across four runs, because a
detector miss longer than the gap budget produced a new id. v1 wrote
`reid.py` to reconcile the count afterwards and said in its own docstring
that it could not un-split a track mid-life. v2 dropped `reid.py` and kept the
tracker. Appearance is consulted *during* association here, so the fragment is
prevented rather than reconciled.

**A discarded low-confidence detection is a lost track.** A detector that is
70% sure drops to 30% when somebody walks behind a post — and the object is
still there, in exactly the place the track predicts. Throwing those away and
then coasting blindly is backwards. The second association pass takes the
detections below the reporting threshold and offers them to tracks that have
nothing else, which is what recovers an object through an occlusion.

**A moving camera moves every box.** Nothing in v1 or v2 estimated camera
motion, so a gust on a mast or a PTZ nudge translated every detection at once
and the tracker read it as everything accelerating together. The warp is
applied to the filters before prediction, covariance included.

# The pipeline, in order

1. Camera motion, if measured, is applied to every track.
2. Every track is predicted forward to this frame's timestamp.
3. Detections are split at `confidence_high`.
4. **Confirmed tracks against strong detections**, cost combining appearance
   and IoU, gated by the filter's own Mahalanobis distance.
5. **Confirmed tracks against weak detections**, IoU only — the occlusion
   recovery.
6. **Tentative tracks against what is left**, IoU only and strictly: a track
   nobody has confirmed yet does not get to claim things on appearance.
7. **Lost tracks against what is still left**, appearance-dominant with a
   spatial gate that widens with how long they have been gone — the
   re-identification, in the only place it can prevent a fragment.
8. Whatever is still unmatched and strong enough starts a new track.
9. Tracks that matched nothing age: confirmed to coasting, coasting to lost,
   lost to gone.

# What is deliberately unchanged

Confirmation is *cumulative*, not consecutive: a track seen twice has been
seen twice whether or not a miss came between. v1 measured the alternative —
consecutive confirmation took three people to eight tracks at 0.69 recall.

A coasted box is never recorded as a measurement of speed or position on the
ground. Extrapolation is not observation, and a map that cannot tell them
apart is a map that invents movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Sequence

import numpy as np

from ..kernel import native
from ..kernel.filtering import (
    CHI2_GATE_2DOF, CHI2_GATE_4DOF, STATE_VALUES, box_xywh, to_measurement,
)
from .appearance import (
    MAX_APPEARANCE_DISTANCE, MAX_REIDENTIFY_DISTANCE, Appearance, Gallery, SceneSeparation,
)
from .detection import BoundingBox, Detection
from .geo import (
    Bearing, CameraPose, LatLon, PositionEstimate, Speed, Vec2, motion_of, project_point,
)


class TrackState(StrEnum):
    """Where a track is in its life.

    `LOST` is the state v1 and v2 had no room for: not visible, not yet
    disproved, and still worth recognising if it comes back. Without it the
    only options are to keep coasting a box nobody can see — which puts an
    invented position on a map — or to throw the identity away, which is the
    fragmentation.
    """

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    # -- association
    #: Detections at or above this are strong enough to start a track and to
    #: take part in the first association pass.
    confidence_high: float = 0.5
    #: Below this a detection is noise and is not offered to anything.
    confidence_low: float = 0.1
    #: Minimum IoU for the first pass. Loose, because the Mahalanobis gate is
    #: doing the real work and IoU is only shaping the cost.
    min_iou: float = 0.1
    #: Minimum IoU for the weak-detection pass, which has no appearance term
    #: to protect it and so needs the overlap to be convincing.
    min_iou_recovery: float = 0.3
    #: Weight of appearance in the first pass's cost. Half: neither signal is
    #: trusted alone, and a track whose gallery is empty falls back to IoU
    #: with no special case.
    appearance_weight: float = 0.5
    #: Share of a detection's box that another detection may cover before its
    #: appearance is discarded.
    #:
    #: Measured, and the measurement is why this exists. Four people milling
    #: within a couple of body widths produced *more* identity switches with
    #: appearance than without — 107 against 95 over eight runs — because a
    #: crop of a partly occluded person is a crop of two people, and a gallery
    #: that stores it goes on matching the wrong one for seconds afterwards.
    #: Containment rather than IoU: a large occluder in front of a small
    #: object has low IoU and covers all of it.
    max_occlusion: float = 0.25

    # -- lifecycle, in milliseconds
    #: Cumulative sightings before a track is reported.
    min_hits_to_confirm: int = 2
    #: A tentative track that has not been confirmed within this is dropped.
    tentative_window_millis: int = 2000
    #: How long a confirmed track keeps being reported while unmatched. Its
    #: box is extrapolated and marked as such.
    coast_millis: int = 1000
    #: How long a lost track stays in the re-identification gallery. Fifteen
    #: seconds is about how long somebody is behind a building, and it is
    #: bounded because an identity kept forever eventually matches a stranger.
    lost_millis: int = 15000
    #: Spatial gate for re-identification, in multiples of the track's own
    #: height per second lost. A person does about two of their own heights
    #: per second walking; three allows for running and for the gate to be
    #: worth having at all.
    reidentify_speed: float = 3.0

    # -- motion on the ground
    motion_window_millis: int = 3000
    #: Below this span a speed is `None`, not a number: ±1.3 m over 200 ms is
    #: ±9 m/s of noise.
    min_motion_span_millis: int = 1200


@dataclass(slots=True)
class Track:
    id: int
    class_id: int
    first_seen_millis: int
    last_seen_millis: int
    last_detected_millis: int
    #: The Kalman state: 8 mean values then a flattened 8x8 covariance.
    filter_state: np.ndarray
    state: TrackState = TrackState.TENTATIVE
    confidence: float = 0.0
    hits: int = 1
    gallery: Gallery = field(default_factory=Gallery)
    contact: Vec2 = Vec2(0.5, 1.0)
    position: PositionEstimate | None = None
    #: `(millis, point, sigma)` per observation. The sigma travels with the
    #: point because a speed computed from two positions inherits their error,
    #: and looking it up later would mean using the *current* uncertainty for a
    #: position measured three seconds ago at a different range.
    ground_history: list[tuple[int, LatLon, float]] = field(default_factory=list)
    #: Speed over the ground, with its error. `None` until enough span.
    speed: Speed | None = None
    #: Direction of travel, with its error.
    heading: Bearing | None = None
    #: Ids this track absorbed by re-identification. Kept because an incident
    #: exported for evidence has to be able to say that "track 3" and "track
    #: 11" were judged the same object, and on what basis.
    absorbed: tuple[int, ...] = ()

    @classmethod
    def observing(cls, track_id: int, class_id: int, bbox: BoundingBox, at_millis: int = 0,
                  *, confidence: float = 0.0, contact: Vec2 | None = None,
                  state: TrackState = TrackState.CONFIRMED,
                  position: PositionEstimate | None = None, speed_mps: float | None = None,
                  heading_degrees: float | None = None) -> "Track":
        """A track holding one observed box.

        The filter state is eight numbers and a covariance, and nothing
        outside this module should have to know that to make a track — for
        a test, for a replayed recording, or for a view that wants to draw
        one. `bbox` reads back exactly what is passed here.
        """
        return cls(
            id=track_id, class_id=class_id, first_seen_millis=at_millis,
            last_seen_millis=at_millis, last_detected_millis=at_millis,
            filter_state=native.kalman_initiate(
                *to_measurement(bbox.x, bbox.y, bbox.width, bbox.height)),
            state=state, confidence=confidence,
            contact=contact if contact is not None else bbox.bottom_center,
            position=position,
            # A caller handing over a bare number is stating a measurement it
            # has no error for; zero is the only honest reading of that, and
            # it keeps `observing` usable from a test or a replay.
            speed=None if speed_mps is None else Speed(speed_mps, 0.0),
            heading=None if heading_degrees is None else Bearing(heading_degrees, 0.0),
        )

    @property
    def speed_mps(self) -> float | None:
        """The plain number, or `None` when the measurement does not support one.

        `None` rather than a figure whose error exceeds it: 0.4 +/- 0.9 m/s is
        not a slow walk, it is no measurement, and printing it invites somebody
        to read "walking pace" off noise.
        """
        return self.speed.mps if self.speed is not None and self.speed.meaningful else None

    @property
    def heading_degrees(self) -> float | None:
        """The plain bearing, or `None` when it could be any of a quadrant."""
        return self.heading.degrees if self.heading is not None and self.heading.meaningful else None

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(*box_xywh(self.filter_state))

    @property
    def velocity(self) -> Vec2:
        return Vec2(float(self.filter_state[4]), float(self.filter_state[5]))

    @property
    def age_millis(self) -> int:
        return self.last_seen_millis - self.first_seen_millis

    @property
    def coasting(self) -> bool:
        return self.last_seen_millis > self.last_detected_millis

    @property
    def confirmed(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    def missing_millis(self, at_millis: int) -> int:
        return max(0, at_millis - self.last_detected_millis)


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    """What changed this frame.

    `ended` is final: the track will not come back and downstream must forget
    it. A track that merely stopped being visible is *not* ended — it goes
    `LOST` and stays recognisable, and the zone's own exit hold is what
    decides that somebody has left. Ending a track the moment it is occluded
    is how one person becomes two visits.

    `recovered` names the tracks that came back from `LOST` this frame. It is
    the evidence of a join: an incident that says "one person" where a naive
    tracker said "three" has to be able to show why.
    """

    ended: tuple[int, ...] = ()
    recovered: tuple[int, ...] = ()


class TrackerProtocol(Protocol):
    def update(self, detections: Sequence[Detection], at_millis: int) -> TrackerUpdate: ...
    def tracks(self) -> list[Track]: ...
    def reset(self) -> list[int]: ...
    def set_pose(self, pose: CameraPose | None) -> None: ...


#: What the assignment solver is given for a pair that must not be matched.
FORBIDDEN = 1.0e9
_FORBIDDEN_THRESHOLD = 1.0e8

#: Appearance distance standing in for "this pair cannot be judged on looks".
#:
#: Every cell of a cost matrix has to be on one scale, and this is what makes
#: it so. The first version of this tracker blended appearance into the cost
#: when it had one and used bare geometry when it did not — so a pair with a
#: good appearance match scored `0.5*0 + 0.5*(1-IoU)`, half of what the same
#: geometry scored for a pair whose appearance was unknown. The solver then
#: systematically preferred whichever candidate happened to have a usable
#: crop, which in a crowd is whichever one was not occluded, which is the
#: wrong one. Measured: it cost 8 identity switches in 8 runs of four people
#: milling.
#:
#: Half the association gate, because that is the expected distance of a pair
#: drawn from the range it admits: neither evidence for nor against.
NEUTRAL_APPEARANCE = MAX_APPEARANCE_DISTANCE / 2


class Tracker:
    """One camera's tracks. Not thread-safe; one worker owns one tracker."""

    def __init__(self, config: TrackerConfig | None = None, pose: CameraPose | None = None):
        self.config = config or TrackerConfig()
        self._pose = pose
        self._tracks: list[Track] = []
        self._next_id = 1
        self._last_update: int | None = None
        #: What a *different* object looks like in this scene, measured
        #: live from pairs that are different by construction.
        self.separation = SceneSeparation()

    # ------------------------------------------------------------- accessors

    def set_pose(self, pose: CameraPose | None) -> None:
        self._pose = pose

    @property
    def pose(self) -> CameraPose | None:
        return self._pose

    def tracks(self) -> list[Track]:
        """The tracks worth reporting: confirmed, and visible or briefly
        coasting. A lost track is not reported — it has no position anybody
        observed."""
        return [t for t in self._tracks if t.confirmed]

    def all_tracks(self) -> list[Track]:
        return list(self._tracks)

    def lost_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state is TrackState.LOST]

    def reset(self) -> list[int]:
        ended = [t.id for t in self._tracks if t.confirmed]
        self._tracks.clear()
        self._last_update = None
        return ended

    # ---------------------------------------------------------------- update

    def update(self, detections: Sequence[Detection], at_millis: int, *,
               appearances: Sequence[Appearance | None] | None = None,
               warp: np.ndarray | None = None) -> TrackerUpdate:
        """Advance every track to `at_millis` against this frame's detections.

        `appearances` is one descriptor per detection, from
        `vigil.perception.appearance.describe`. Without them association
        falls back to geometry alone — which is exactly v2's old behaviour,
        including its fragmentation, and is stated here rather than hidden.
        The tracker takes measurements rather than pixels so that it can be
        tested without an image and so that the domain never sees OpenCV.

        `warp` is a 2x3 affine describing how the *camera* moved since the
        last call, in normalised image coordinates. See
        `vigil.perception.motion`.
        """
        dt = 0.0
        if self._last_update is not None and at_millis > self._last_update:
            dt = (at_millis - self._last_update) / 1000.0
        self._last_update = at_millis

        if warp is not None:
            for track in self._tracks:
                native.kalman_warp(track.filter_state, warp)
        for track in self._tracks:
            native.kalman_predict(track.filter_state, dt)

        measurements = np.array(
            [to_measurement(d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height) for d in detections],
            dtype=np.float64,
        ).reshape(-1, 4)
        looks: list[Appearance | None] = list(appearances) if appearances is not None \
            else [None] * len(detections)
        if len(looks) != len(detections):
            raise ValueError("one appearance per detection, or none at all")
        looks = [look if look is not None and look.usable else None for look in looks]
        self._discard_occluded_appearances(detections, looks)

        strong = [i for i, d in enumerate(detections) if d.confidence >= self.config.confidence_high]
        weak = [i for i, d in enumerate(detections)
                if self.config.confidence_low <= d.confidence < self.config.confidence_high]

        claimed: set[int] = set()
        recovered: list[int] = []

        # 1. Confirmed tracks against strong detections: appearance and IoU.
        confirmed = [t for t in self._tracks if t.confirmed]
        matched = self._associate(confirmed, strong, detections, measurements, looks,
                                  use_appearance=True, min_iou=self.config.min_iou,
                                  position_only=False)
        for track, index in matched:
            self._apply(track, detections[index], looks[index], at_millis)
            claimed.add(index)

        # 2. Confirmed tracks that still have nothing, against weak detections.
        #    Position-only gating, because a detector that has become unsure is
        #    usually unsure about the box's extent — a half-occluded person has
        #    a right centre and a wrong height.
        pending = [t for t in confirmed if t.last_detected_millis != at_millis]
        remaining = [i for i in weak if i not in claimed]
        matched = self._associate(pending, remaining, detections, measurements, looks,
                                  use_appearance=False, min_iou=self.config.min_iou_recovery,
                                  position_only=True)
        for track, index in matched:
            self._apply(track, detections[index], looks[index], at_millis)
            claimed.add(index)

        # 3. Tentative tracks, on geometry alone and strictly.
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]
        remaining = [i for i in strong if i not in claimed]
        matched = self._associate(tentative, remaining, detections, measurements, looks,
                                  use_appearance=False, min_iou=self.config.min_iou_recovery,
                                  position_only=False)
        for track, index in matched:
            self._apply(track, detections[index], looks[index], at_millis)
            claimed.add(index)

        # Two objects detected in the same frame are certainly different
        # objects, so every such pair measures what a stranger scores in
        # this scene. That is the ground truth re-identification is
        # calibrated against, and nobody had to label it.
        self._measure_separation(detections, looks)

        # 4. Lost tracks: the re-identification.
        lost = [t for t in self._tracks if t.state is TrackState.LOST]
        remaining = [i for i in strong if i not in claimed]
        for track, index in self._reidentify(lost, remaining, detections, looks, at_millis):
            self._apply(track, detections[index], looks[index], at_millis)
            track.state = TrackState.CONFIRMED
            claimed.add(index)
            recovered.append(track.id)

        # 5. Anything strong and unclaimed is something new.
        for index in strong:
            if index in claimed:
                continue
            self._tracks.append(self._begin(detections[index], looks[index], at_millis))

        # 6. Age whatever matched nothing.
        ended = self._age(at_millis)
        return TrackerUpdate(tuple(ended), tuple(recovered))

    def _discard_occluded_appearances(self, detections: Sequence[Detection],
                                      looks: list[Appearance | None]) -> None:
        """Throw away the descriptor of anything another detection is standing
        in front of.

        A crop of a partly occluded person is a crop of two people, and it is
        worse than having no descriptor at all: no descriptor falls back to
        geometry, while a contaminated one is *confidently* wrong and is then
        remembered. See `TrackerConfig.max_occlusion` for the measurement.

        A detection carrying a segmentation mask is exempt: the mask has
        already excluded whoever is in front, which is the whole reason the
        detector keeps it.

        Which of two overlapping boxes is in front cannot be read off the
        boxes — but it can be read off the *ground*. For objects standing on
        a plane, the one whose box bottom is lower in the frame is nearer the
        camera, because image row maps monotonically to ground distance. That
        is the same projection the rest of this system runs on, used here to
        answer a question a 2-D overlap cannot.

        So a detection loses its descriptor to anything at least as near as it
        is. "At least as near" rather than "nearer" so that two objects side
        by side at the same distance lose both — they genuinely do contaminate
        each other — while something clearly further away, whose pixels fall
        in the top of the box where nothing is sampled anyway, does not.
        """
        limit = self.config.max_occlusion
        for i, detection in enumerate(detections):
            if looks[i] is None or detection.mask is not None:
                continue
            box = detection.bbox
            area = box.area
            if area <= 0:
                looks[i] = None
                continue
            # A twentieth of the box's height: below that the two contact
            # points are indistinguishable given how well a detector places a
            # bottom edge.
            tolerance = 0.05 * box.height
            for j, other in enumerate(detections):
                if i == j:
                    continue
                obox = other.bbox
                if obox.bottom < box.bottom - tolerance:
                    continue  # further away; its pixels are above, not over
                overlap = (max(0.0, min(box.right, obox.right) - max(box.x, obox.x))
                           * max(0.0, min(box.bottom, obox.bottom) - max(box.y, obox.y)))
                if overlap / area > limit:
                    looks[i] = None
                    break

    # ----------------------------------------------------------- association

    def _associate(self, tracks: list[Track], candidates: list[int], detections: Sequence[Detection],
                   measurements: np.ndarray, looks: list[Appearance | None], *,
                   use_appearance: bool, min_iou: float,
                   position_only: bool) -> list[tuple[Track, int]]:
        """One optimal assignment pass over a cost matrix with hard gates.

        A gate is a `FORBIDDEN` cost rather than a filtered-out row, so the
        solver still sees a rectangular problem and the caller drops the pairs
        it was forced into. That is exact, not approximate: with every real
        cost bounded by 1, no sum of real costs can reach the threshold.
        """
        if not tracks or not candidates:
            return []
        rows, cols = len(tracks), len(candidates)
        cost = np.full((rows, cols), FORBIDDEN, dtype=np.float64)
        subset = measurements[candidates]
        gate_limit = CHI2_GATE_2DOF if position_only else CHI2_GATE_4DOF
        for r, track in enumerate(tracks):
            gates = native.kalman_gate(track.filter_state, subset, position_only)
            predicted = track.bbox
            for c, index in enumerate(candidates):
                detection = detections[index]
                if detection.class_id != track.class_id:
                    continue
                if not math.isfinite(gates[c]) or gates[c] > gate_limit:
                    continue
                overlap = predicted.iou(detection.bbox)
                if overlap < min_iou:
                    continue
                geometry = 1.0 - overlap
                if not use_appearance:
                    cost[r, c] = geometry
                    continue
                # One scale for the whole matrix: a pair nobody can judge on
                # looks is scored as neither evidence for nor against, never
                # as free. See `NEUTRAL_APPEARANCE`.
                appearance = NEUTRAL_APPEARANCE
                if track.gallery.usable and looks[index] is not None:
                    appearance = track.gallery.distance(looks[index])
                    if appearance > MAX_APPEARANCE_DISTANCE:
                        continue
                weight = self.config.appearance_weight
                cost[r, c] = weight * appearance + (1.0 - weight) * geometry
        assignment = native.assign(cost)
        out: list[tuple[Track, int]] = []
        for r, c in enumerate(assignment):
            if c < 0 or cost[r, c] >= _FORBIDDEN_THRESHOLD:
                continue
            out.append((tracks[r], candidates[c]))
        return out

    def _reidentify(self, lost: list[Track], candidates: list[int], detections: Sequence[Detection],
                    looks: list[Appearance | None], at_millis: int) -> list[tuple[Track, int]]:
        """Match a returning object to the track it left as.

        Appearance leads and geometry gates, which is the reverse of every
        pass above and is the whole point: the box has moved, that is why the
        track was lost. The spatial gate widens with the time gone at a walking
        pace, so a two-second gap allows a few metres of image and a
        fourteen-second one allows most of the frame — and a track that has
        been gone for its whole budget is still not allowed to match something
        on the other side of a wall.

        A track with no usable gallery is not re-identified at all. Position
        alone over a gap of seconds is not evidence, and matching on it is how
        two different people become one — which is worse than the fragment,
        because a fragment is visible and a merge is not.

        # How close is close enough

        Not a constant. `MAX_REIDENTIFY_DISTANCE` was calibrated on synthetic
        colour blocks and `tools/calibrate.py` found it far too generous on
        real video — different objects there sat at a median of 0.105 against
        a gate of 0.35. `SceneSeparation` measures what a stranger actually
        scores *in this scene*, and a candidate has to beat that.

        Two conditions, and both are needed:

        - **Below the ceiling.** Closer than 95% of the pairs this scene has
          proved are different objects. This is what protects the case where
          there is only one candidate and nothing to compare it against.
        - **Ahead by a margin.** Better than the runner-up by enough that the
          choice is not a coin toss. Two objects that look equally like a
          lost track mean the descriptor cannot tell, and picking one is
          guessing with an operator's incident report.
        """
        if not lost or not candidates:
            return []
        ceiling = self.separation.ceiling()
        margin = self.separation.margin()
        rows, cols = len(lost), len(candidates)
        cost = np.full((rows, cols), FORBIDDEN, dtype=np.float64)
        for r, track in enumerate(lost):
            if not track.gallery.usable:
                continue
            gone = track.missing_millis(at_millis) / 1000.0
            reach = self.config.reidentify_speed * max(track.bbox.height, 0.02) * max(gone, 0.2)
            last = track.bbox.center
            for c, index in enumerate(candidates):
                detection = detections[index]
                if detection.class_id != track.class_id:
                    continue
                look = looks[index]
                if look is None:
                    continue
                appearance = track.gallery.distance(look)
                # This scene's own ceiling, not the shipped constant.
                if appearance > ceiling:
                    continue
                centre = detection.bbox.center
                if math.hypot(centre.x - last.x, centre.y - last.y) > reach:
                    continue
                cost[r, c] = appearance
        # A row or a column with two plausible answers has no answer. Both
        # directions matter: a lost track that two detections both fit, and
        # a detection that two lost tracks both claim, are the same failure
        # seen from opposite sides.
        self._require_margin(cost, margin)
        assignment = native.assign(cost)
        out: list[tuple[Track, int]] = []
        for r, c in enumerate(assignment):
            if c < 0 or cost[r, c] >= _FORBIDDEN_THRESHOLD:
                continue
            out.append((lost[r], candidates[c]))
        return out

    @staticmethod
    def _require_margin(cost: np.ndarray, margin: float) -> None:
        """Forbid any row or column whose best answer is not clearly best.

        In place, and symmetric. A margin of zero disables it, which is what
        a scene with no measured separation gets — there, the ceiling is the
        shipped constant and it is doing all the work.
        """
        if margin <= 0 or cost.size == 0:
            return
        for axis in (1, 0):
            ordered = np.sort(cost, axis=axis)
            if cost.shape[axis] < 2:
                continue
            best = np.take(ordered, 0, axis=axis)
            second = np.take(ordered, 1, axis=axis)
            # Only a *real* runner-up counts. A single feasible candidate is
            # decided by the ceiling, not by a comparison with nothing.
            ambiguous = (second < _FORBIDDEN_THRESHOLD) & ((second - best) < margin)
            if axis == 1:
                cost[ambiguous, :] = FORBIDDEN
            else:
                cost[:, ambiguous] = FORBIDDEN

    def _measure_separation(self, detections: Sequence[Detection],
                            looks: list[Appearance | None]) -> None:
        """Record how far apart the objects in this frame look.

        Every pair of detections in one frame is a pair of different objects,
        because one object cannot be in two places. Same class only: a person
        and a van being far apart says nothing about telling two people
        apart, and re-identification never crosses classes anyway.
        """
        for i, look in enumerate(looks):
            if look is None:
                continue
            for j in range(i + 1, len(looks)):
                other = looks[j]
                if other is None or detections[i].class_id != detections[j].class_id:
                    continue
                distance = look.distance(other)
                if math.isfinite(distance):
                    self.separation.observe(distance)

    # ------------------------------------------------------------- lifecycle

    def _begin(self, detection: Detection, look: Appearance | None, at_millis: int) -> Track:
        measurement = to_measurement(detection.bbox.x, detection.bbox.y,
                                     detection.bbox.width, detection.bbox.height)
        track = Track(
            id=self._next_id,
            class_id=detection.class_id,
            first_seen_millis=at_millis,
            last_seen_millis=at_millis,
            last_detected_millis=at_millis,
            filter_state=native.kalman_initiate(*measurement),
            confidence=detection.confidence,
            contact=detection.ground_contact,
        )
        self._next_id += 1
        if look is not None:
            track.gallery.observe(look)
        if self.config.min_hits_to_confirm <= 1:
            track.state = TrackState.CONFIRMED
            self._locate(track, at_millis)
        return track

    def _apply(self, track: Track, detection: Detection, look: Appearance | None, at_millis: int) -> None:
        measurement = to_measurement(detection.bbox.x, detection.bbox.y,
                                     detection.bbox.width, detection.bbox.height)
        if not native.kalman_update(track.filter_state, *measurement):
            # A filter that has lost definiteness is restarted on this
            # measurement rather than carried forward. The identity survives;
            # the numbers behind it do not deserve to.
            track.filter_state = native.kalman_initiate(*measurement)
        track.contact = detection.ground_contact
        track.last_seen_millis = at_millis
        track.last_detected_millis = at_millis
        track.confidence = track.confidence * 0.7 + detection.confidence * 0.3
        track.hits += 1
        if look is not None:
            track.gallery.observe(look)
        if track.state is TrackState.TENTATIVE and track.hits >= self.config.min_hits_to_confirm:
            track.state = TrackState.CONFIRMED
        elif track.state is TrackState.COASTING:
            track.state = TrackState.CONFIRMED
        if track.confirmed:
            self._locate(track, at_millis)

    def _age(self, at_millis: int) -> list[int]:
        ended: list[int] = []
        surviving: list[Track] = []
        for track in self._tracks:
            if track.last_detected_millis == at_millis:
                surviving.append(track)
                continue
            missing = track.missing_millis(at_millis)
            if track.state is TrackState.TENTATIVE:
                if at_millis - track.first_seen_millis > self.config.tentative_window_millis:
                    continue  # never confirmed, never reported, so never `ended`
                surviving.append(track)
            elif track.confirmed:
                if missing <= self.config.coast_millis:
                    track.state = TrackState.COASTING
                    track.last_seen_millis = at_millis
                    # The box is extrapolated by the filter's own prediction,
                    # which already ran. Nothing about the ground is recorded:
                    # `_locate` is not called, so `position` and
                    # `ground_history` keep their last observed values.
                    surviving.append(track)
                elif track.gallery.usable:
                    # Out of sight but recognisable. Not ended: the zone's exit
                    # hold decides that somebody has left, and this track can
                    # still be matched to the person who walked back out from
                    # behind the van.
                    track.state = TrackState.LOST
                    surviving.append(track)
                else:
                    # Nothing to recognise it by — no frame was given, or every
                    # crop was too small to describe. Parking it in a gallery it
                    # can never be found in is memory spent on nothing, so it
                    # ends here, which is exactly v2's old behaviour for the
                    # case where v2's information is all there is.
                    ended.append(track.id)
            elif missing <= self.config.lost_millis:
                surviving.append(track)
            else:
                # The gallery has run out of time. This is where a track that
                # was once reported is finally ended, and it has to be said:
                # dropping it silently leaks every rule that keyed on its id.
                ended.append(track.id)
        self._tracks = surviving
        return ended

    # ------------------------------------------------------------ the ground

    def _locate(self, track: Track, at_millis: int) -> None:
        if self._pose is None:
            return
        track.position = project_point(self._pose, track.contact)
        if track.position is None or not track.position.is_projected:
            return
        track.ground_history.append(
            (at_millis, track.position.point, track.position.radius_meters))
        oldest = at_millis - self.config.motion_window_millis
        track.ground_history = [e for e in track.ground_history if e[0] >= oldest]
        first_t, first_p, first_sigma = track.ground_history[0]
        span = at_millis - first_t
        if span < self.config.min_motion_span_millis:
            track.speed = None
            track.heading = None
            return
        # The two endpoint errors in quadrature — the same rule `separation`
        # uses, because it is the same question asked of the same two points.
        sigma = math.hypot(first_sigma, track.position.radius_meters)
        track.speed, track.heading = motion_of(
            first_p, track.position.point, span / 1000.0, sigma)


__all__ = [
    "Track", "TrackState", "Tracker", "TrackerConfig", "TrackerProtocol", "TrackerUpdate",
    "STATE_VALUES",
]
