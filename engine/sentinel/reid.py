"""Re-identification after the fact: linking the fragments a tracker leaves.

VIDEO -> DETECTION -> TRACKING -> [RE-IDENTIFICATION] -> SPATIAL CONTEXT -> ...

**Why this exists.** Twenty seconds of one person and some furniture on the
laptop camera counted 3, 10, 4 and 11 "distinct objects" across four runs. The
tracker associates on position alone: when the detector loses an object for
longer than the tracker's gap budget, the object comes back as a new id, and
every rule downstream — the object count on an incident, the "group" risk
factor, the loitering clock — treats the new id as a new person. An operator
who reads "11 distinct objects" about one colleague stops believing counts.

The tracker lives in the Rust core and cannot be changed from here. So the
honest first step is to **link fragments after the fact**, the way the
correlator links tracks across cameras: two tracks on one camera are the same
object when one ends about where and when the other begins *and* they look
alike. The association has the same shape as :func:`sentinel.incidents.associate`
— a judgement with its separation, its allowance, its gap and its reasons — and
the same union-find gives the transitive closure, so a chain of three fragments
is one object rather than two pairs.

**Why a colour histogram and not a learned embedding.** Nothing here may be
downloaded, and a re-identification network is a download. More to the point,
nobody has yet measured how much of the fragmentation a descriptor of any kind
would reconcile: `tools/measure_fragmentation.py` is that measurement, and it
runs on this descriptor first. If a masked HSV histogram closes most of the
gap, a learned embedding is a cost without a case. If it does not, the
measurement says why — and that, not a hunch, is the argument for carrying
appearance into the Rust tracker (ABI 7) and for what it should carry.

**What this cannot do, stated plainly.** Post-hoc linking reconciles the
*count*. It cannot un-split a track mid-life: an event already raised against
fragment #7 was raised, the loitering clock already restarted, and the
console already drew a fresh box. Only the tracker can prevent that, which is
why linking is a step towards ABI 7 and not a substitute for it. It also cannot
tell two people in the same dark coat apart — colour is all it has — which is
why similarity alone never links, and time and place are conditions rather
than tie-breakers: a red coat leaving and a red coat arriving are two people.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .core import BoundingBox, Detection, PositionEstimate, Track, haversine_distance
from .incidents import MAX_UNCERTAINTY_ALLOWANCE_METERS, ObjectIdentity

# ------------------------------------------------------------------ descriptor

#: Hue and saturation together, value separately. A full 3-D cube spreads a
#: distant person's few hundred pixels across a thousand bins and the cosine
#: between two frames of the *same* object becomes noise; two dense marginals
#: are enough to tell one coat from another.
HUE_BINS = 16
SATURATION_BINS = 8
#: Value is kept at all because most clothing — and every walker in the
#: synthetic scene — is near grey, where hue is undefined and saturation is
#: nil. Value is the only channel that separates a black coat from a white
#: shirt. It carries half the histogram's mass rather than all of it because
#: illumination drift moves it and moves nothing else.
VALUE_BINS = 8

#: Fewer pixels than this is not an appearance, it is a colour sample of the
#: noise floor, and comparing two of them links anything to anything.
MINIMUM_PIXELS = 16

#: Weight of the newest sample in the running descriptor. A five-frame time
#: constant: a track genuinely changing appearance (turning, walking into
#: shade) is followed within a second at 15 fps, while one bad frame — a lamp
#: post across the box — moves the descriptor a fifth of the way and no further.
EMA_ALPHA = 0.2

# -------------------------------------------------------------------- linking

#: The longest a fragment may be missing and still be joined to its successor.
#: Longer than the tracker's own 2 s hold — inside that the tracker already
#: coasts, and there is nothing for this to reconcile — and short enough that a
#: walking person cannot have gone far, which the spatial gate then checks.
#: Beyond it, "the same person came back" is a claim colour cannot support.
DEFAULT_MAX_GAP_MILLIS = 5000

#: Cosine between two running descriptors. Measured on the reference walkers
#: over their ground-truth boxes: one walker's own descriptor a second later
#: scores 0.90 or better, three seconds later 0.73 at worst (the approaching
#: walker's box quadruples and fills with ground); a *different* walker
#: scores 0.69–0.86, because all three are near-grey on the same ground and
#: a box is a third background. 0.85 keeps every one-second reconnection and
#: rejects nearly every stranger, at the cost of some reconnections across a
#: longer gap. That cost is the finding, not a tuning problem: a box
#: histogram on grey clothing is a weak witness, a mask removes the shared
#: background, and what remains is the case for a learned descriptor — which
#: `tools/measure_fragmentation.py` is there to make or refuse.
DEFAULT_MIN_SIMILARITY = 0.85

#: A running person. Anything covering ground faster than this across a gap —
#: a vehicle — is not linked, which is stated here rather than hidden: the
#: allowance grows with the gap, and a gate that grows fast enough for a car
#: would join every pedestrian on the same pavement.
PLAUSIBLE_SPEED_MPS = 4.0

#: What two ground-contact estimates of one *standing* object disagree by
#: frame to frame: the box bottom jitters, and the projection amplifies it.
BASE_ALLOWANCE_METERS = 2.0

#: The same two numbers for a camera with no pose, in normalised frame units.
#: Crude, because a frame unit is a metre near the camera and ten at the
#: horizon; the placed gate is the honest one and this is the fallback.
BASE_ALLOWANCE_FRAME = 0.05
PLAUSIBLE_SPEED_FRAME_PER_SECOND = 0.25


@dataclass(frozen=True, slots=True)
class Appearance:
    """What one observation of an object looked like: a masked colour histogram.

    Masked because a box around a person is mostly not the person — a third of
    it is the wall behind them — and a histogram over the box describes the
    wall as much as the coat. When the detector produced a mask the histogram
    is taken over the mask; otherwise over the box, and ``source`` records
    which, because the two are not equally trustworthy and a caller comparing
    them should know.

    Normalised to unit mass so a near object and a far one compare as colours
    rather than as pixel counts; ``pixels`` keeps the count so a descriptor
    built from a dozen pixels can be treated with the suspicion it deserves.
    """

    histogram: np.ndarray
    #: ``"mask"`` when the detector's silhouette bounded the sample, ``"box"``
    #: when only the rectangle did.
    source: str
    pixels: int

    @classmethod
    def of(
        cls,
        image: np.ndarray,
        bbox: BoundingBox,
        mask: np.ndarray | None = None,
    ) -> "Appearance | None":
        """Describe the pixels of ``image`` inside ``bbox`` — inside ``mask`` if given.

        Returns ``None`` for a box that lies outside the frame or covers too
        few pixels to say anything: an empty descriptor compared against
        another would report perfect similarity and link the two on nothing.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return None
        frame_h, frame_w = image.shape[:2]

        # The box in pixels *before* clipping, because a mask is cropped to the
        # unclipped box and must be resized to it, then clipped the same way.
        px = int(round(bbox.x * frame_w))
        py = int(round(bbox.y * frame_h))
        pw = max(1, int(round(bbox.w * frame_w)))
        ph = max(1, int(round(bbox.h * frame_h)))

        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(frame_w, px + pw), min(frame_h, py + ph)
        if x1 <= x0 or y1 <= y0:
            return None

        crop = image[y0:y1, x0:x1]
        weights = np.full(crop.shape[:2], 255, dtype=np.uint8)
        source = "box"

        if mask is not None and mask.size:
            sized = cv2.resize(
                (mask > 0).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST
            )
            clipped = sized[y0 - py : y1 - py, x0 - px : x1 - px]
            if int(cv2.countNonZero(clipped)) >= MINIMUM_PIXELS:
                weights = clipped * 255
                source = "mask"
            # A mask with almost nothing lit is a speck of activation, not a
            # silhouette; the box is the better evidence then, and saying so
            # in ``source`` is what keeps that honest.

        pixels = int(cv2.countNonZero(weights))
        if pixels < MINIMUM_PIXELS:
            return None

        hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV)
        hue_sat = cv2.calcHist(
            [hsv], [0, 1], weights, [HUE_BINS, SATURATION_BINS], [0, 180, 0, 256]
        ).flatten()
        value = cv2.calcHist([hsv], [2], weights, [VALUE_BINS], [0, 256]).flatten()

        histogram = np.concatenate(
            [_unit_mass(hue_sat) * 0.5, _unit_mass(value) * 0.5]
        ).astype(np.float32)
        return cls(histogram=histogram, source=source, pixels=pixels)

    def similarity(self, other: "Appearance") -> float:
        return cosine_similarity(self.histogram, other.histogram)


def _unit_mass(histogram: np.ndarray) -> np.ndarray:
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two descriptors, ``0.0`` when either is empty.

    Zero rather than one for an empty descriptor: a track nothing was ever
    sampled from must not look identical to everything.
    """
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0.0:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / norm, -1.0, 1.0))


# ---------------------------------------------------------------- per track


@dataclass(slots=True)
class TrackAppearance:
    """One track's running appearance, and the ends of its life.

    The descriptor is an exponential moving average rather than the last
    sample, because the last sample of a fragment is the frame the detector was
    already losing — half a person behind a pillar — and it is the worst frame
    to describe them by. ``samples`` counts what went into the average so a
    caller can tell a descriptor built from one frame from one built from a
    hundred.

    The ends matter as much as the colour. ``last_confirmed_millis`` is the
    last frame a detection actually confirmed this track, as opposed to
    ``last_seen_millis``, which the tracker keeps advancing while it coasts a
    predicted box towards the frame edge. A fragment's gap to its successor
    is measured from the confirmed end, and its last known place is where it
    was last *seen*, not where the prediction drifted to.
    """

    camera_id: str
    track_id: int
    first_seen_millis: int
    last_seen_millis: int
    last_confirmed_millis: int
    first_box: BoundingBox
    last_box: BoundingBox
    first_position: PositionEstimate | None
    last_position: PositionEstimate | None
    histogram: np.ndarray | None = None
    samples: int = 0
    #: How many of the samples were bounded by a silhouette rather than a box.
    mask_samples: int = 0

    @classmethod
    def begin(cls, camera_id: str, track: Track, at_millis: int) -> "TrackAppearance":
        return cls(
            camera_id=camera_id,
            track_id=track.id,
            first_seen_millis=track.first_seen_millis,
            last_seen_millis=max(track.last_seen_millis, at_millis),
            last_confirmed_millis=at_millis,
            first_box=track.bbox,
            last_box=track.bbox,
            first_position=track.position,
            last_position=track.position,
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera_id, self.track_id)

    @property
    def has_appearance(self) -> bool:
        return self.histogram is not None and self.samples > 0

    def observe(
        self,
        track: Track,
        at_millis: int,
        appearance: Appearance | None,
        *,
        confirmed: bool,
        alpha: float = EMA_ALPHA,
    ) -> None:
        """Fold one more frame in.

        ``confirmed`` says whether a detection backed the track on this frame.
        A coasted frame extends the track's life but not its confirmed end and
        not its last known place — the tracker's prediction is extrapolation,
        and the gate downstream must not measure distance from a guess.
        """
        self.last_seen_millis = max(self.last_seen_millis, track.last_seen_millis, at_millis)
        if confirmed:
            self.last_confirmed_millis = max(self.last_confirmed_millis, at_millis)
            self.last_box = track.bbox
            self.last_position = track.position

        if appearance is None:
            return
        if self.histogram is None:
            self.histogram = appearance.histogram.astype(np.float32).copy()
        else:
            self.histogram = (
                (1.0 - alpha) * self.histogram + alpha * appearance.histogram
            ).astype(np.float32)
        self.samples += 1
        if appearance.source == "mask":
            self.mask_samples += 1

    def similarity(self, other: "TrackAppearance") -> float | None:
        """Cosine between the two running descriptors; ``None`` if either has none."""
        if not self.has_appearance or not other.has_appearance:
            return None
        assert self.histogram is not None and other.histogram is not None
        return cosine_similarity(self.histogram, other.histogram)


# --------------------------------------------------------- frame bookkeeping

#: A track's box is the detection's box on the frame it was matched, and a
#: prediction on the frames it was coasted. Overlap this high with a detection
#: means the tracker matched them; a coasted box that overlapped a detection
#: this much would have been matched to it instead of coasted.
_CONFIRMING_IOU = 0.9


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.w, b.x + b.w)
    bottom = min(a.y + a.h, b.y + b.h)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = a.w * a.h + b.w * b.h - overlap
    return overlap / union if union > 0 else 0.0


def confirming_detection(track: Track, detections: Sequence[Detection]) -> Detection | None:
    """The detection that backed ``track`` on this frame, if one did.

    The pipeline reports detections and tracks side by side without saying
    which fed which; the box is the join. ``None`` means the track was coasted
    this frame, and a coasted frame has no pixels that are evidence of anything.
    """
    best, best_iou = None, _CONFIRMING_IOU
    for detection in detections:
        overlap = _iou(track.bbox, detection.bbox)
        if overlap >= best_iou:
            best, best_iou = detection, overlap
    return best


class AppearanceLedger:
    """Turns a run's frame results into one :class:`TrackAppearance` per track.

    Fed one frame at a time, in order, from whatever produced the results —
    the pipeline, a replay, a test. It never sees the tracker's internals; it
    reads the same results an operator's console reads, which is the point:
    what it links is what was reported.
    """

    __slots__ = ("_tracks", "_alpha")

    def __init__(self, *, alpha: float = EMA_ALPHA) -> None:
        self._tracks: dict[tuple[str, int], TrackAppearance] = {}
        self._alpha = alpha

    def observe(
        self,
        camera_id: str,
        at_millis: int,
        tracks: Sequence[Track],
        detections: Sequence[Detection] = (),
        image: np.ndarray | None = None,
    ) -> None:
        """One frame's worth of tracks, with the detections and pixels behind them.

        Without ``image`` the ledger still learns each track's span and ends —
        enough for the time and place gates — but no appearance, and without
        appearance nothing links. That is deliberate: see the module docstring.
        """
        for track in tracks:
            record = self._tracks.get((camera_id, track.id))
            if record is None:
                record = TrackAppearance.begin(camera_id, track, at_millis)
                self._tracks[record.key] = record
                confirmed = True
                detection = confirming_detection(track, detections)
            else:
                detection = confirming_detection(track, detections)
                confirmed = detection is not None

            appearance = None
            if image is not None and (detection is not None or not detections):
                # No detections at all this frame means the caller supplied
                # tracks alone; take the box as the evidence rather than
                # refusing to learn anything.
                appearance = Appearance.of(
                    image, track.bbox, detection.mask if detection is not None else None
                )
            record.observe(track, at_millis, appearance, confirmed=confirmed, alpha=self._alpha)

    def observe_result(self, result) -> None:
        """Convenience for a pipeline ``FrameResult`` (typed loosely to avoid the import)."""
        self.observe(
            result.source_id, result.timestamp_millis, result.tracks,
            result.detections, getattr(result, "image", None),
        )

    @property
    def tracks(self) -> tuple[TrackAppearance, ...]:
        return tuple(self._tracks.values())

    def link(
        self,
        *,
        max_gap_millis: int = DEFAULT_MAX_GAP_MILLIS,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> tuple["FragmentGroup", ...]:
        return link_fragments(
            self.tracks, max_gap_millis=max_gap_millis, min_similarity=min_similarity
        )


# ----------------------------------------------------------------- association


@dataclass(frozen=True, slots=True)
class FragmentLink:
    """A judgement that ``later`` continues ``earlier`` on one camera.

    Same shape as the correlator's :class:`~sentinel.incidents.Association`,
    with the appearance similarity added, and for the same reason: an operator
    shown "#7 = #3" must be able to see the gap, the distance and the colour
    match that justified it, and disagree.
    """

    earlier: tuple[str, int]
    later: tuple[str, int]
    score: float
    gap_millis: int
    separation: float
    allowance: float
    #: ``"m"`` when both ends were placed on the ground, ``"frame"`` when the
    #: gate had to fall back to normalised image distance.
    separation_unit: str
    similarity: float
    reasons: tuple[str, ...]

    def describe(self) -> str:
        return f"#{self.later[1]} = #{self.earlier[1]}"


@dataclass(frozen=True, slots=True)
class FragmentGroup:
    """The track ids that are, after linking, one object.

    What a caller counts: ``len(groups)`` is the distinct-object count with
    fragmentation reconciled. Every track is in exactly one group; a track
    nothing linked to is a group of one.
    """

    camera_id: str
    #: In order of first appearance, so ``members[0]`` is the id the object
    #: first carried and the natural one to display.
    members: tuple[int, ...]
    links: tuple[FragmentLink, ...]

    @property
    def canonical(self) -> int:
        return self.members[0]

    def describe(self) -> str:
        """``"#7 = #3"``: every later id read as the first one."""
        return " = ".join(f"#{member}" for member in reversed(self.members))


def _bottom_centre(box: BoundingBox) -> tuple[float, float]:
    return (box.x + box.w / 2.0, box.y + box.h)


def _spatial_gate(
    earlier: TrackAppearance, later: TrackAppearance, gap_seconds: float
) -> tuple[float, float, str]:
    """Separation, allowance and unit between where one ended and the other began.

    Placed positions when both ends have them, widened by their uncertainties
    the way the correlator widens its gate and capped the same way. Otherwise
    normalised image distance, which is a poorer instrument and is labelled as
    such in the unit.
    """
    if earlier.last_position is not None and later.first_position is not None:
        separation = haversine_distance(earlier.last_position.point, later.first_position.point)
        slack = min(
            MAX_UNCERTAINTY_ALLOWANCE_METERS,
            earlier.last_position.radius_meters + later.first_position.radius_meters,
        )
        allowance = BASE_ALLOWANCE_METERS + PLAUSIBLE_SPEED_MPS * gap_seconds + slack
        return separation, allowance, "m"

    ax, ay = _bottom_centre(earlier.last_box)
    bx, by = _bottom_centre(later.first_box)
    separation = float(np.hypot(ax - bx, ay - by))
    allowance = BASE_ALLOWANCE_FRAME + PLAUSIBLE_SPEED_FRAME_PER_SECOND * gap_seconds
    return separation, allowance, "frame"


def link_fragments(
    tracks_with_appearance: Sequence[TrackAppearance],
    *,
    max_gap_millis: int = DEFAULT_MAX_GAP_MILLIS,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> tuple[FragmentGroup, ...]:
    """Decide which tracks on one camera are fragments of one object.

    Three conditions, all required, none a tie-breaker for another:

    1. **Time.** The later track begins no earlier than the earlier track's
       last confirmed frame and within ``max_gap_millis`` of it. Two tracks
       confirmed at the same moment are two objects, whatever else is true.
    2. **Place.** Where the earlier track was last seen and where the later
       one first appeared are within what a person could have walked in the
       gap, plus the jitter of the two position estimates.
    3. **Look.** The two running descriptors agree to at least
       ``min_similarity``. A track never sampled cannot satisfy this, so it
       never links — position and time alone would join a red coat leaving
       and a blue coat arriving.

    Each fragment continues at most one other and is continued by at most one:
    when two tracks appear after one vanishes, only the better-supported one
    can be its continuation, and the other is somebody else. Never across
    cameras — that is the correlator's job, with different evidence.

    Returns one group per object, singletons included, so the caller's count
    is ``len(result)``.
    """
    by_camera: dict[str, list[TrackAppearance]] = {}
    for track in tracks_with_appearance:
        by_camera.setdefault(track.camera_id, []).append(track)

    identity = ObjectIdentity()
    accepted: list[FragmentLink] = []

    for camera_id, tracks in by_camera.items():
        candidates: list[FragmentLink] = []
        for earlier in tracks:
            identity.add(*earlier.key)
            for later in tracks:
                if later.key == earlier.key:
                    continue
                link = _consider(earlier, later, max_gap_millis, min_similarity)
                if link is not None:
                    candidates.append(link)

        # Best-supported first, one continuation per fragment in each direction.
        candidates.sort(key=lambda link: (-link.score, link.gap_millis, link.later[1]))
        continued: set[tuple[str, int]] = set()
        continues: set[tuple[str, int]] = set()
        for link in candidates:
            if link.earlier in continued or link.later in continues:
                continue
            continued.add(link.earlier)
            continues.add(link.later)
            identity.union(link.earlier, link.later)
            accepted.append(link)

    first_seen = {track.key: track.first_seen_millis for track in tracks_with_appearance}
    groups: list[FragmentGroup] = []
    for keys in identity.groups():
        camera_id = next(iter(keys))[0]
        members = tuple(
            key[1] for key in sorted(keys, key=lambda key: (first_seen[key], key[1]))
        )
        links = tuple(
            link for link in accepted if link.earlier in keys and link.later in keys
        )
        groups.append(FragmentGroup(camera_id=camera_id, members=members, links=links))

    groups.sort(key=lambda group: (group.camera_id, first_seen[(group.camera_id, group.members[0])]))
    return tuple(groups)


def _consider(
    earlier: TrackAppearance,
    later: TrackAppearance,
    max_gap_millis: int,
    min_similarity: float,
) -> FragmentLink | None:
    gap = later.first_seen_millis - earlier.last_confirmed_millis
    if gap < 0 or gap > max_gap_millis:
        return None
    gap_seconds = gap / 1000.0

    separation, allowance, unit = _spatial_gate(earlier, later, gap_seconds)
    if separation > allowance:
        return None

    similarity = earlier.similarity(later)
    if similarity is None or similarity < min_similarity:
        return None

    proximity = 1.0 - separation / allowance if allowance > 0 else 1.0
    recency = 1.0 - gap / max_gap_millis if max_gap_millis > 0 else 1.0
    # Appearance carries the most weight because it is the one condition the
    # tracker itself never had; time and place it already used, and lost.
    score = round(0.5 * similarity + 0.25 * proximity + 0.25 * recency, 4)

    return FragmentLink(
        earlier=earlier.key,
        later=later.key,
        score=score,
        gap_millis=gap,
        separation=round(separation, 3),
        allowance=round(allowance, 3),
        separation_unit=unit,
        similarity=round(similarity, 4),
        reasons=(
            f"#{later.track_id} began {gap_seconds:.1f} s after #{earlier.track_id} "
            f"was last seen, within {max_gap_millis / 1000:.0f} s",
            f"{separation:.2f} {unit} apart, within {allowance:.2f} {unit} "
            + ("allowed for the gap and the two position uncertainties"
               if unit == "m" else "allowed in the frame (camera not placed)"),
            f"appearance {similarity:.2f} alike, at least {min_similarity:.2f} required "
            f"({earlier.samples} and {later.samples} samples)",
        ),
    )
