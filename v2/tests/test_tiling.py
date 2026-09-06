"""Finding more, and being able to say what "more" cost.

Every claim in here is a number this test computes, because the phase it
belongs to has no labelled data to measure precision against and will not
pretend otherwise. What can be shown without labels is shown: that a small
object the whole-frame pass misses is found, that a large one is not turned
into a piece of itself, that a crowded scene keeps its second object, and that
the extra passes are counted rather than hidden.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from vigil.adapters.detectors import _nms_per_class
from vigil.adapters.tiling import (
    DEFAULT_BAND, Tile, TiledDetector, far_band, merge, tiles_for, touches_edge,
)
from vigil.domain.detection import BoundingBox, Detection, DetectorInfo
from vigil.domain.geo import CameraPose, LatLon, Vec2

POSE = CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -20.0, 0.0, 62.0, 36.0, 80.0)
INFO = DetectorInfo("onnx-detect", "test", "/x", "sha", (640, 640), {0: "person", 1: "car"}, True, "CPU")


@dataclass
class _Fake:
    """A detector that finds boxes above a size floor, as a real one does.

    The floor is the whole point: a model has a smallest object it can find,
    and tiling works by making a small object large rather than by lowering
    that floor.
    """

    minimum_height: float = 0.06
    calls: int = 0
    #: `(x, y, w, h)` in whole-frame coordinates, and what class.
    objects: tuple = ()
    _crop: tuple = (0.0, 0.0, 1.0, 1.0)

    @property
    def info(self):
        return INFO

    def detect(self, image):
        self.calls += 1
        ox, oy, ow, oh = self._crop
        out = []
        for x, y, w, h, class_id in self.objects:
            # Into this crop's coordinates.
            cx, cy = (x - ox) / ow, (y - oy) / oh
            cw, ch = w / ow, h / oh
            if cx < 0 or cy < 0 or cx + cw > 1 or cy + ch > 1:
                continue
            if ch < self.minimum_height:
                continue
            out.append(Detection(BoundingBox(cx, cy, cw, ch).clamped(), 0.8, class_id,
                                 Vec2(cx + cw / 2, cy + ch)))
        return out


class _CroppingDetector(_Fake):
    """Wraps `_Fake` so a crop is reported in the crop's own coordinates,
    which is what a real detector handed a cropped array does."""

    def __init__(self, objects, frame_shape, minimum_height=0.06):
        super().__init__(minimum_height=minimum_height, objects=objects)
        self._shape = frame_shape

    def detect(self, image):
        h, w = image.shape[:2]
        fh, fw = self._shape
        # Recover which crop this is from its size and content marker.
        ox = float(image[0, 0, 0]) / 255.0
        oy = float(image[0, 0, 1]) / 255.0
        self._crop = (ox, oy, w / fw, h / fh)
        return super().detect(image)


def _frame(height=1080, width=1920):
    """A frame whose top-left pixel encodes where the crop starts, so the fake
    detector can place its objects without being told."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        image[y, :, 1] = int(round(y / height * 255))
    for x in range(width):
        image[:, x, 0] = int(round(x / width * 255))
    return image


def test_the_far_band_is_derived_from_the_pose_not_guessed():
    top, bottom = far_band(POSE, beyond_meters=18.0)
    assert bottom > top
    from vigil.domain.geo import project_to_ground

    at_bottom = project_to_ground(POSE, 0.5, bottom, enforce_range=False)
    just_below = project_to_ground(POSE, 0.5, min(1.0, bottom + 0.05), enforce_range=False)
    assert at_bottom.ground_distance_meters >= 18.0
    assert just_below is None or just_below.ground_distance_meters < 18.0


def test_a_camera_looking_at_a_doorway_gets_no_tiles_and_costs_nothing():
    """Nothing in this view is far away, so there is nothing to gain."""
    close = replace(POSE, pitch=-70.0, range_meters=8.0)
    assert far_band(close, beyond_meters=18.0) == (0.0, 0.0)
    assert tiles_for(close, 1920, 1080, (640, 640), beyond_meters=18.0) == []


def test_a_frame_no_larger_than_the_model_input_is_never_tiled():
    """Cropping an image the model already sees whole buys nothing at all."""
    assert tiles_for(POSE, 640, 480, (640, 640)) == []


def test_without_a_pose_the_band_is_a_stated_fallback():
    assert far_band(None) == DEFAULT_BAND
    assert tiles_for(None, 1920, 1080, (640, 640))


def test_a_distant_object_the_whole_frame_pass_misses_is_found_in_a_tile():
    """The recall this exists to recover, as a number.

    A person 0.045 of the frame high is under the fake model's floor of 0.06.
    In a tile a third of the frame wide they are 0.135 of the tile, and found.
    """
    small = (0.42, 0.16, 0.012, 0.045, 0)
    inner = _CroppingDetector([small], (1080, 1920))
    image = _frame()

    whole_only = inner.detect(image)
    assert whole_only == [], "this test needs an object the plain pass cannot find"

    tiled = TiledDetector(_CroppingDetector([small], (1080, 1920)), POSE)
    found = tiled.detect(image)
    assert len(found) == 1, f"the tile pass should have found it: {found}"
    box = found[0].bbox
    assert abs(box.x - small[0]) < 0.02 and abs(box.y - small[1]) < 0.02, (
        f"found at {box.x:.3f},{box.y:.3f} but it is at {small[0]},{small[1]} — "
        f"the tile coordinates were not mapped back")


def test_a_tiled_detection_carries_its_contact_point_back_into_the_frame():
    """The contact is the one number the projection uses. Left in tile
    coordinates it would place the object at the wrong end of the yard,
    silently, because the box on screen would be right."""
    small = (0.42, 0.16, 0.012, 0.045, 0)
    tiled = TiledDetector(_CroppingDetector([small], (1080, 1920)), POSE)
    found = tiled.detect(_frame())
    contact = found[0].contact
    assert contact is not None
    assert abs(contact.x - (small[0] + small[2] / 2)) < 0.02
    assert abs(contact.y - (small[1] + small[3])) < 0.02


def test_the_cost_is_one_inference_per_tile_and_is_counted():
    inner = _CroppingDetector([], (1080, 1920))
    tiled = TiledDetector(inner, POSE)
    image = _frame()
    tiles = tiled.tiles_for_shape(1080, 1920)
    tiled.detect(image)
    assert inner.calls == 1 + len(tiles), (
        f"{inner.calls} inferences for 1 whole frame and {len(tiles)} tiles")
    assert 1 <= len(tiles) <= 6, f"{len(tiles)} tiles is not a sane number"


def test_a_large_object_is_not_replaced_by_the_piece_a_tile_saw():
    """A lorry crossing the far half appears whole in the frame pass and
    cropped in a tile. The crop can score higher, and the box an operator sees
    would then be a piece of the lorry."""
    lorry = Detection(BoundingBox(0.10, 0.20, 0.55, 0.22), 0.72, 1, Vec2(0.375, 0.42))
    piece = Detection(BoundingBox(0.10, 0.20, 0.23, 0.22), 0.91, 1, Vec2(0.21, 0.42))
    merged = merge([lorry], [(piece, True)])
    assert len(merged) == 1
    assert merged[0] is lorry, "the higher-scoring crop won and the lorry became a cab"


def test_a_second_object_at_the_edge_of_a_tile_is_kept_when_nothing_saw_it_whole():
    """The edge rule must not become a blanket ban on edge detections: an
    object only in a tile has nothing in the frame pass to defer to."""
    piece = Detection(BoundingBox(0.10, 0.20, 0.05, 0.05), 0.66, 0, Vec2(0.125, 0.25))
    assert merge([], [(piece, True)]) == [piece]


def test_touching_the_frames_own_edge_does_not_count_as_touching_a_tiles():
    """A person half out of the left of the *frame* is genuinely half out of
    the frame; no pass could have seen more of them."""
    whole_left = Tile(0.0, 0.0, 0.33, 0.6)
    assert not touches_edge(BoundingBox(0.0, 0.2, 0.04, 0.1), whole_left)
    assert touches_edge(BoundingBox(0.30, 0.2, 0.03, 0.1), whole_left)


# ---------------------------------------------------------------- soft NMS


def _boxes(*rects):
    xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h, _s in rects], dtype=np.float64)
    scores = np.array([s for *_r, s in rects], dtype=np.float64)
    return xyxy, scores, np.zeros(len(rects), dtype=np.int64)


def test_hard_nms_deletes_the_second_car_in_a_row_and_soft_nms_keeps_it():
    """The measurement that justifies the change. Two parked cars overlapping
    0.55 are two cars; hard NMS at 0.45 calls them one."""
    xyxy, scores, classes = _boxes((0.10, 0.50, 0.20, 0.16, 0.86),
                                   (0.163, 0.50, 0.20, 0.16, 0.71))
    inter = (0.30 - 0.163) * 0.16
    union = 2 * 0.20 * 0.16 - inter
    assert 0.4 < inter / union < 0.7, f"overlap {inter / union:.2f} is not the case under test"

    hard = _nms_per_class(xyxy, scores, classes, 0.45, soft=False)
    soft = _nms_per_class(xyxy, scores, classes, 0.45, soft=True)
    assert len(hard) == 1, "this test needs hard NMS to actually delete one"
    assert len(soft) == 2, "soft NMS decayed the second car out of existence"


def test_soft_nms_still_removes_a_genuine_duplicate():
    """Two proposals for one object at 0.9 overlap. The decay must push the
    weaker under the floor, or soft NMS is just "no NMS"."""
    xyxy, scores, classes = _boxes((0.10, 0.50, 0.20, 0.16, 0.90),
                                   (0.105, 0.505, 0.20, 0.16, 0.34))
    assert len(_nms_per_class(xyxy, scores, classes, 0.45, soft=True)) == 1


def test_soft_nms_never_reorders_two_boxes_that_do_not_overlap():
    xyxy, scores, classes = _boxes((0.02, 0.50, 0.10, 0.10, 0.55),
                                   (0.60, 0.20, 0.10, 0.10, 0.51))
    keep = _nms_per_class(xyxy, scores, classes, 0.45, soft=True)
    assert keep == [0, 1]


def test_the_detection_floor_came_down_and_says_why():
    from vigil.adapters.detectors import DEFAULT_CONFIDENCE, LEGACY_CONFIDENCE

    assert DEFAULT_CONFIDENCE < LEGACY_CONFIDENCE
    assert DEFAULT_CONFIDENCE >= 0.2, "below this a frame is mostly noise proposals"
