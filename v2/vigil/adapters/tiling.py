"""Running the detector over parts of the frame as well as the whole of it.

# The recall this recovers

A 1080p frame letterboxed into 640x640 is scaled by 0.59. A person 40 m from
a 4 m mast with a 62-degree lens is about 40 px tall in the original and 24 px
after letterboxing — near the floor of what a nano-scale YOLO can find, and
below it once they turn side-on. The model never had a chance, and no
threshold recovers a detection that was never proposed.

Running the same model over a *crop* at native scale puts that person back at
their full height in the tile, which is the whole trick. It costs one extra
inference per tile.

# Where to tile, and why it is not everywhere

Uniform tiling is the usual approach and it is wasteful here, because this
product knows something a general detector does not: **where the far ground
is**. `geo.project_to_ground` says which image rows land beyond a given
distance, so the tiles can be put exactly where objects will be small and the
near foreground — where a person is already 300 px tall and the whole-frame
pass finds them easily — can be left alone.

Without a pose there is no such knowledge and the band falls back to the upper
middle of the frame, which is where the distance is in nearly every fixed
security view. Stated as a fallback rather than presented as geometry.

# Merging, and the failure it avoids

An object straddling two tiles is detected twice, at slightly different boxes.
Merging with class-aware NMS across the union of whole-frame and tile
detections resolves that. Class-aware, not class-agnostic, for the same reason
`detectors._nms_per_class` is: a person in front of a car at 70% overlap is a
car park, not a duplicate.

The subtler failure is the opposite one. A *large* object — a lorry crossing
the far half — appears whole in the frame pass and cropped in a tile, and the
cropped piece can score higher. Its box would then be the piece rather than
the lorry. So a tile detection touching the tile's own edge is dropped when
the whole-frame pass already has something overlapping it: the frame pass saw
all of it, and the tile did not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..domain.detection import BoundingBox, Detection, DetectorInfo
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: How much neighbouring tiles overlap, as a fraction of a tile. A person is
#: about a twelfth of a 640 px tile at these ranges, so an eighth guarantees
#: that anything person-sized is wholly inside at least one tile rather than
#: split across two and found in neither.
TILE_OVERLAP = 0.125

#: Ground distance past which objects are small enough to be worth tiling.
#: Below it the whole-frame pass already resolves a person to over a hundred
#: pixels and a tile adds cost for nothing.
FAR_GROUND_METERS = 18.0

#: Fraction of a tile's edge within which a detection counts as touching it.
EDGE_FRACTION = 0.02

#: Overlap above which a tile detection is considered the same object as a
#: whole-frame one, for the edge rule above.
EDGE_IOU = 0.3


@dataclass(frozen=True, slots=True)
class Tile:
    """A crop of the frame, in normalised coordinates."""

    x: float
    y: float
    w: float
    h: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(round(self.x * width))))
        y1 = max(0, min(height - 1, int(round(self.y * height))))
        x2 = max(x1 + 1, min(width, int(round((self.x + self.w) * width))))
        y2 = max(y1 + 1, min(height, int(round((self.y + self.h) * height))))
        return x1, y1, x2, y2


def far_band(pose, *, beyond_meters: float = FAR_GROUND_METERS) -> tuple[float, float]:
    """The rows of the frame that land beyond `beyond_meters`, as `(top, bottom)`.

    `(0.0, 0.0)` when nothing in this view is that far — a camera looking down
    at a doorway — which means no tiles and no cost.

    Derived rather than assumed. Without a pose, see `DEFAULT_BAND`.
    """
    from ..domain.geo import project_to_ground

    if pose is None:
        return DEFAULT_BAND
    top = None
    bottom = 0.0
    for step in range(101):
        v = step / 100.0
        ground = project_to_ground(pose, 0.5, v, enforce_range=False)
        if ground is None:
            continue
        if ground.ground_distance_meters >= beyond_meters:
            if top is None:
                top = v
            bottom = v
    if top is None:
        return 0.0, 0.0
    return top, bottom


#: Where the far ground is in a fixed security view when nothing has said.
#: The upper middle: the sky is above it and the foreground below.
DEFAULT_BAND = (0.25, 0.65)


def tiles_for(pose, width: int, height: int, tile_size: tuple[int, int],
              *, beyond_meters: float = FAR_GROUND_METERS) -> list[Tile]:
    """Tiles covering the part of this frame where objects will be small.

    Empty when the frame is already at or below the model's input size — there
    is nothing to gain from cropping an image the model sees whole — or when
    no part of the view is far enough away to need it.
    """
    tw, th = tile_size
    if width <= tw and height <= th:
        return []
    top, bottom = far_band(pose, beyond_meters=beyond_meters)
    if bottom <= top:
        return []
    # The band in pixels, grown to at least a tile high so a shallow band
    # still gets a tile rather than a sliver stretched to fit.
    y1, y2 = top * height, bottom * height
    if y2 - y1 < th:
        centre = (y1 + y2) / 2
        y1, y2 = centre - th / 2, centre + th / 2
        y1 = max(0.0, min(y1, height - th))
        y2 = y1 + th
    y1, y2 = max(0.0, y1), min(float(height), y2)

    step_x = tw * (1 - TILE_OVERLAP)
    step_y = th * (1 - TILE_OVERLAP)
    out: list[Tile] = []
    ys = _starts(y1, y2, th, step_y)
    xs = _starts(0.0, float(width), tw, step_x)
    for sy in ys:
        for sx in xs:
            out.append(Tile(sx / width, sy / height, tw / width, th / height))
    return out


def _starts(low: float, high: float, size: float, step: float) -> list[float]:
    span = high - low
    if span <= size:
        return [max(0.0, low)]
    count = int(math.ceil((span - size) / step)) + 1
    return [low + min(i * step, span - size) for i in range(count)]


def merge(whole: list[Detection], tiled, iou: float = 0.55) -> list[Detection]:
    """One list from the whole-frame pass and the tile passes.

    `tiled` is `(detection, touched_a_tile_edge)` pairs. The flag travels
    beside the detection rather than on it, because `Detection` is frozen with
    slots — deliberately, so that nothing downstream can quietly annotate what
    the detector said — and this is a fact about *where it was found*, not
    about the object.

    Class-aware suppression, plus the edge rule: a tile detection that touches
    its tile's own boundary and overlaps something the whole-frame pass
    already found is dropped, because the frame pass saw the whole object and
    the tile saw a piece of it.
    """
    from .detectors import _nms_per_class

    kept_tiles = [d for d, edge in tiled if not (edge and _overlaps_one_of(d, whole))]
    everything = list(whole) + kept_tiles
    if not everything:
        return []
    xyxy = np.array([[d.bbox.x, d.bbox.y, d.bbox.right, d.bbox.bottom]
                     for d in everything], dtype=np.float64)
    scores = np.array([d.confidence for d in everything], dtype=np.float64)
    classes = np.array([d.class_id for d in everything], dtype=np.int64)
    return [everything[i] for i in _nms_per_class(xyxy, scores, classes, iou)]


def _overlaps_one_of(detection: Detection, whole: list[Detection]) -> bool:
    return any(other.class_id == detection.class_id
               and _iou(detection.bbox, other.bbox) >= EDGE_IOU for other in whole)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def touches_edge(bbox: BoundingBox, tile: Tile) -> bool:
    """Whether a detection, already in frame coordinates, reaches the edge of
    the tile it came from — and that edge is not also the frame's."""
    margin_x, margin_y = tile.w * EDGE_FRACTION, tile.h * EDGE_FRACTION
    left, top = tile.x, tile.y
    right, bottom = tile.x + tile.w, tile.y + tile.h
    if left > 1e-6 and bbox.x <= left + margin_x:
        return True
    if top > 1e-6 and bbox.y <= top + margin_y:
        return True
    if right < 1 - 1e-6 and bbox.right >= right - margin_x:
        return True
    if bottom < 1 - 1e-6 and bbox.bottom >= bottom - margin_y:
        return True
    return False


class TiledDetector:
    """A detector, plus the same detector over the far part of the frame.

    Wraps any `Detector`. The cost is `1 + len(tiles)` inferences per frame,
    which is what the GPU provider bought: measured at 4.5 ms against 38.5 ms
    for the session alone, so four tiles on the GPU still cost less than one
    whole frame did on the CPU.
    """

    def __init__(self, inner, pose=None, *, beyond_meters: float = FAR_GROUND_METERS,
                 max_tiles: int = 6):
        self._inner = inner
        self._pose = pose
        self._beyond = beyond_meters
        self._max_tiles = max_tiles
        self._tiles: list[Tile] | None = None
        self._shape: tuple[int, int] | None = None

    @property
    def info(self) -> DetectorInfo:
        return self._inner.info

    @property
    def inner(self):
        return self._inner

    def set_pose(self, pose) -> None:
        """A pose changed, so where the far ground is has changed with it."""
        self._pose = pose
        self._tiles = None

    def tiles_for_shape(self, height: int, width: int) -> list[Tile]:
        if self._tiles is None or self._shape != (height, width):
            size = self.info.input_size or (640, 640)
            tiles = tiles_for(self._pose, width, height, size, beyond_meters=self._beyond)
            if len(tiles) > self._max_tiles:
                # Keep the widest coverage rather than the first N: an
                # arbitrary truncation would tile the left of the frame and
                # leave the right permanently unexamined.
                step = len(tiles) / self._max_tiles
                tiles = [tiles[int(i * step)] for i in range(self._max_tiles)]
            self._tiles = tiles
            self._shape = (height, width)
            _log.info("%s: %d tile(s) over the far ground", self.info.name, len(tiles))
        return self._tiles

    def detect(self, image: np.ndarray) -> list[Detection]:
        whole = self._inner.detect(image)
        if image is None or image.ndim != 3:
            return whole
        height, width = image.shape[:2]
        tiles = self.tiles_for_shape(height, width)
        if not tiles:
            return whole
        found = []
        for tile in tiles:
            x1, y1, x2, y2 = tile.pixels(width, height)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            for detection in self._inner.detect(crop):
                moved = _into_frame(detection, tile)
                if moved is not None:
                    found.append(moved)
        return merge(whole, found)


def _into_frame(detection: Detection, tile: Tile):
    """A tile-relative detection in whole-frame coordinates, with whether it
    reached the tile's own edge.

    The contact point moves with the box. It is the one number the projection
    uses, so a contact left in tile coordinates would place every tiled
    detection at the wrong end of the yard — silently, because the box on
    screen would be right.
    """
    from dataclasses import replace

    from ..domain.geo import Vec2

    b = detection.bbox
    bbox = BoundingBox(tile.x + b.x * tile.w, tile.y + b.y * tile.h,
                       b.width * tile.w, b.height * tile.h).clamped()
    if bbox.area <= 0:
        return None
    contact = detection.contact
    if contact is not None:
        contact = Vec2(tile.x + contact.x * tile.w, tile.y + contact.y * tile.h)
    # The flag goes to `merge`, which needs to know this box may be a piece
    # of something the whole-frame pass saw entire.
    return replace(detection, bbox=bbox, contact=contact), touches_edge(bbox, tile)
