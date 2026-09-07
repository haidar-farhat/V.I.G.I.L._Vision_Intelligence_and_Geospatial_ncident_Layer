"""Tracking, against the failures v1 measured and v2 inherited.

Every test here names a way the old tracker was wrong. The two that matter
most are `test_two_people_crossing_do_not_swap_identities` — greedy
association's characteristic failure — and
`test_one_person_with_a_blinking_detector_stays_one_object`, which is v1's own
measurement: twenty seconds of one person counted 3, 10, 4 and 11 distinct
objects across four runs.
"""

import math

import numpy as np
import pytest

from vigil.domain.detection import BoundingBox, Detection
from vigil.domain.tracking import FORBIDDEN, Tracker, TrackerConfig, TrackState
from vigil.perception.appearance import describe

FPS = 15
STEP = 1000 // FPS


def _box(x, y=0.6, w=0.1, h=0.2):
    return BoundingBox(x, y, w, h)


def _frame(*people, size=(240, 320)):
    """A frame with a solid rectangle per person, each its own colour.

    Deliberately crude: what the appearance descriptor has to do here is tell
    a red coat from a blue one over a few seconds, and a synthetic frame
    measures that honestly. It cannot measure whether the descriptor copes
    with real lighting, and no synthetic frame can — `tools/` has the harness
    that runs this on a camera.
    """
    image = np.full((*size, 3), 30, dtype=np.uint8)
    height, width = size
    for box, colour in people:
        x0, y0 = int(box.x * width), int(box.y * height)
        x1, y1 = int(box.right * width), int(box.bottom * height)
        image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = colour
    return image


def _looks(image, detections):
    """One appearance per detection, the way a camera worker produces them."""
    if image is None:
        return None
    return [describe(image, (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height), d.mask)
            for d in detections]


def _walk(tracker, boxes_at, frames=20, start=0, colours=None, images=True):
    """Run `frames` steps, `boxes_at(i)` giving that frame's boxes."""
    seen = []
    for i in range(frames):
        boxes = boxes_at(i)
        detections = [Detection(b, 0.9, c) for b, c in boxes]
        image = None
        if images:
            palette = colours or [(200, 60, 60), (60, 60, 200), (60, 200, 60)]
            image = _frame(*[(b, palette[n % len(palette)]) for n, (b, _) in enumerate(boxes)])
        tracker.update(detections, start + i * STEP, appearances=_looks(image, detections))
        seen.append({t.id for t in tracker.tracks()})
    return seen


# --------------------------------------------------------------- confirmation


def test_a_track_is_confirmed_after_cumulative_hits_even_with_a_miss_between():
    # Cumulative, not consecutive. v1 measured consecutive confirmation taking
    # three people to eight tracks at 0.69 recall.
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=2))
    tracker.update([Detection(_box(0.1), 0.9, 1)], 0)
    assert tracker.tracks() == [], "one sighting is not a track"
    tracker.update([], STEP)
    tracker.update([Detection(_box(0.12), 0.9, 1)], 2 * STEP)
    tracks = tracker.tracks()
    assert len(tracks) == 1 and tracks[0].hits == 2 and tracks[0].confirmed


def test_a_tentative_track_that_never_confirms_is_dropped_without_being_reported():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=3, tentative_window_millis=500))
    update = tracker.update([Detection(_box(0.1), 0.9, 1)], 0)
    assert update.ended == ()
    update = tracker.update([], 1000)
    assert tracker.all_tracks() == []
    assert update.ended == (), "a track nobody was told about does not need ending"


def test_one_object_keeps_one_id_and_classes_never_merge():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    ids = set()
    for i in range(10):
        tracker.update(
            [Detection(_box(0.1 + i * 0.02), 0.8, 1), Detection(_box(0.6, y=0.3), 0.8, 2)],
            i * STEP,
        )
        ids.update(t.id for t in tracker.tracks())
    assert ids == {1, 2}


# ---------------------------------------------------------------- the filter


def test_a_stationary_object_is_not_given_a_velocity_by_detector_jitter():
    # The EMA this replaced turned box jitter into motion, so a parked car
    # drifted and its coasted box walked off it.
    rng = np.random.default_rng(4)
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    for i in range(60):
        jitter = rng.normal(0, 0.004, 2)
        tracker.update([Detection(_box(0.5 + jitter[0], 0.5 + jitter[1]), 0.9, 1)], i * STEP)
    track = tracker.tracks()[0]
    speed = math.hypot(track.velocity.x, track.velocity.y)
    assert speed < 0.06, f"a stationary object drifted at {speed:.3f} of a frame per second"
    assert abs(track.bbox.center.x - 0.55) < 0.02


def test_the_filter_converges_on_a_constant_velocity_and_coasts_on_it():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=400))
    speed = 0.3  # frames per second
    for i in range(45):
        tracker.update([Detection(_box(0.1 + speed * i * STEP / 1000), 0.9, 1)], i * STEP)
    track = tracker.tracks()[0]
    assert abs(track.velocity.x - speed) < 0.03, f"got {track.velocity.x:.3f}"
    where = track.bbox.x
    tracker.update([], 45 * STEP + 200)
    coasted = tracker.tracks()[0]
    assert coasted.coasting
    assert abs(coasted.bbox.x - (where + speed * 0.2)) < 0.02, "a coast follows the velocity"


# ------------------------------------------------------- the greedy failure


def test_two_people_crossing_do_not_swap_identities():
    """Greedy association's characteristic failure, as a scenario.

    Two people walk towards each other, pass, and continue. At the crossing
    frame both detections overlap both predictions. A greedy pass takes the
    single best pair and is then forced into whatever is left, which is
    frequently the other person's track; the optimal matching considers the
    total and keeps both.
    """
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    left_colour, right_colour = (220, 40, 40), (40, 40, 220)
    identities: list[tuple[int | None, int | None]] = []
    for i in range(40):
        t = i / 39
        left = _box(0.15 + 0.5 * t, y=0.55, w=0.09, h=0.22)
        right = _box(0.65 - 0.5 * t, y=0.57, w=0.09, h=0.22)
        image = _frame((left, left_colour), (right, right_colour))
        detections = [Detection(left, 0.9, 1), Detection(right, 0.9, 1)]
        tracker.update(detections, i * STEP, appearances=_looks(image, detections))
        found = {t_.id: t_ for t_ in tracker.tracks()}
        # Which track is nearest each true position?
        def nearest(box):
            best, best_d = None, 1e9
            for track in found.values():
                d = abs(track.bbox.center.x - box.center.x)
                if d < best_d:
                    best, best_d = track.id, d
            return best
        identities.append((nearest(left), nearest(right)))

    # Before and after the crossing, each person must be the same track.
    before_left, before_right = identities[3]
    after_left, after_right = identities[-1]
    assert before_left is not None and before_right is not None
    assert before_left != before_right
    assert (after_left, after_right) == (before_left, before_right), (
        f"identities swapped across the crossing: {identities[3]} -> {identities[-1]}"
    )
    assert len(tracker.tracks()) == 2, "and nobody was duplicated"


# ------------------------------------------------------- the fragmentation


def test_one_person_with_a_blinking_detector_stays_one_object():
    """v1's measurement, as a test.

    One person walks across the frame while the detector drops out for
    stretches longer than any coast budget. The old design produced a new id
    after every gap — v1 counted 3, 10, 4 and 11 objects for one person across
    four runs of twenty seconds. Appearance during association is what makes
    it one.
    """
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=2, coast_millis=400))
    colour = (200, 50, 50)
    ids: set[int] = set()
    # Twenty seconds at 15 fps, with the detector blind for 1.5 s out of every
    # 4 s — far past the coast budget, which is the whole point.
    for i in range(20 * FPS):
        millis = i * STEP
        x = 0.05 + 0.85 * (i / (20 * FPS))
        box = _box(x, y=0.55, w=0.08, h=0.24)
        blind = (millis % 4000) > 2500
        image = _frame((box, colour))
        detections = [] if blind else [Detection(box, 0.9, 1)]
        tracker.update(detections, millis, appearances=_looks(image, detections))
        ids.update(t.id for t in tracker.tracks())
    assert len(ids) == 1, f"one person became {len(ids)} objects: {sorted(ids)}"


def test_re_identification_will_not_join_two_people_who_look_different():
    """The error that matters more than the fragment.

    A red coat leaves the left of the frame and a blue coat arrives at the
    right. Joining them would report one person where there were two, and a
    merge is worse than a split because a split is visible on screen and a
    merge is not.
    """
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=200))
    for i in range(10):
        box = _box(0.1 + 0.01 * i, y=0.55, w=0.08, h=0.24)
        detections = [Detection(box, 0.9, 1)]
        tracker.update(detections, i * STEP, appearances=_looks(_frame((box, (220, 40, 40))), detections))
    first = {t.id for t in tracker.tracks()}
    for i in range(10, 25):  # gone
        tracker.update([], i * STEP, appearances=_looks(_frame(), []))
    for i in range(25, 35):
        box = _box(0.30 + 0.01 * (i - 25), y=0.55, w=0.08, h=0.24)
        detections = [Detection(box, 0.9, 1)]
        tracker.update(detections, i * STEP, appearances=_looks(_frame((box, (40, 40, 220))), detections))
    second = {t.id for t in tracker.tracks()}
    assert first and second and not (first & second), (
        f"a red coat and a blue coat were judged the same object: {first} vs {second}"
    )


def test_a_track_with_no_appearance_to_recognise_it_by_ends_rather_than_lingering():
    # No frame is given, so no gallery is built. Parking such a track in a
    # gallery it can never be found in is memory spent on nothing.
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=300))
    tracker.update([Detection(_box(0.4, y=0.7), 0.9, 1)], 0)
    update = tracker.update([], 100)
    assert update.ended == () and tracker.tracks()[0].coasting
    update = tracker.update([], 1000)
    assert update.ended == (1,)
    assert tracker.all_tracks() == []


def test_a_lost_track_is_kept_recognisable_and_then_expires():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=200, lost_millis=2000))
    box = _box(0.4, y=0.7, w=0.08, h=0.24)
    for i in range(5):
        detections = [Detection(box, 0.9, 1)]
        tracker.update(detections, i * STEP, appearances=_looks(_frame((box, (200, 50, 50))), detections))
    tracker.update([], 1000)
    assert [t.state for t in tracker.all_tracks()] == [TrackState.LOST]
    assert tracker.tracks() == [], "a lost track has no position anybody observed"
    update = tracker.update([], 4000)
    assert update.ended == (1,) and tracker.all_tracks() == []


# --------------------------------------------------------------- occlusion


def test_a_weak_detection_recovers_a_track_the_strong_pass_would_have_dropped():
    """A detector that is 70% sure drops to 30% behind a post, and the object
    is still exactly where the track predicts. The old design discarded those
    and coasted blind."""
    config = TrackerConfig(min_hits_to_confirm=1, confidence_high=0.5, confidence_low=0.1)
    tracker = Tracker(config)
    for i in range(10):
        tracker.update([Detection(_box(0.1 + 0.02 * i), 0.9, 1)], i * STEP)
    before = tracker.tracks()[0]
    # Ten frames where the detector is barely sure. The track must stay put
    # and keep being *detected* rather than coasting.
    for i in range(10, 20):
        tracker.update([Detection(_box(0.1 + 0.02 * i), 0.25, 1)], i * STEP)
        track = tracker.tracks()[0]
        assert track.id == before.id
        assert not track.coasting, "a weak detection is still a detection"
    assert len(tracker.tracks()) == 1, "and it did not also start a new track"


def test_a_detection_below_the_noise_floor_is_offered_to_nothing():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, confidence_low=0.1))
    tracker.update([Detection(_box(0.5), 0.02, 1)], 0)
    assert tracker.all_tracks() == []


# ------------------------------------------------------------ camera motion


def test_a_camera_pan_does_not_look_like_everything_accelerating():
    """The whole scene shifts by 8% of the frame between two frames. Without
    the warp the tracker sees every object jump at once; with it, the tracks
    move with the camera and their velocities stay near zero."""
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    boxes = [_box(0.3, y=0.5), _box(0.6, y=0.4)]
    for i in range(20):
        tracker.update([Detection(b, 0.9, n + 1) for n, b in enumerate(boxes)], i * STEP)
    shift = 0.08
    warp = np.array([[1.0, 0.0, shift], [0.0, 1.0, 0.0]])
    moved = [BoundingBox(b.x + shift, b.y, b.width, b.height) for b in boxes]
    tracker.update([Detection(b, 0.9, n + 1) for n, b in enumerate(moved)], 20 * STEP, warp=warp)
    tracks = {t.class_id: t for t in tracker.tracks()}
    assert len(tracks) == 2, "the pan must not have started new tracks"
    for n, box in enumerate(moved):
        track = tracks[n + 1]
        assert abs(track.bbox.center.x - box.center.x) < 0.02
        assert abs(track.velocity.x) < 0.5, (
            f"the pan was read as motion: {track.velocity.x:.2f} frames/s"
        )


# ------------------------------------------------------------- on the ground


def test_speed_is_none_until_enough_span_and_then_measured(pose):
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, min_motion_span_millis=1000), pose)
    track = None
    for i in range(20):
        tracker.update([Detection(_box(0.3 + i * 0.01, y=0.7), 0.9, 1)], i * 100)
        track = tracker.tracks()[0]
        if i * 100 < 1000:
            assert track.speed_mps is None
    assert track is not None
    assert track.speed_mps is not None and track.speed_mps > 0
    assert track.position is not None and track.position.is_projected


def test_a_coasted_box_is_never_recorded_as_a_measurement(pose):
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=1000), pose)
    tracker.update([Detection(_box(0.4, y=0.7), 0.9, 1)], 0)
    tracker.update([Detection(_box(0.42, y=0.7), 0.9, 1)], 100)
    history = len(tracker.tracks()[0].ground_history)
    where = tracker.tracks()[0].position
    tracker.update([], 400)
    coasting = tracker.tracks()[0]
    assert coasting.coasting
    assert len(coasting.ground_history) == history, (
        "extrapolation is not observation, and a map that cannot tell them apart invents movement"
    )
    assert coasting.position == where


def test_a_position_carries_its_error_ellipse(pose):
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1), pose)
    tracker.update([Detection(_box(0.5, y=0.85), 0.9, 1)], 0)
    position = tracker.tracks()[0].position
    assert position is not None and position.is_projected
    assert position.ellipse is not None
    assert position.ellipse.along_meters > 0 and position.ellipse.across_meters > 0
    assert position.radius_meters == pytest.approx(position.ellipse.radius_meters)


# ------------------------------------------------------------------ hygiene


def test_reset_ends_every_reported_track():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    tracker.update([Detection(_box(0.1), 0.9, 1), Detection(_box(0.5), 0.9, 1)], 0)
    assert sorted(tracker.reset()) == [1, 2] and tracker.tracks() == []


def test_repeated_timestamps_and_backwards_time_do_not_corrupt_the_filter():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    for millis in (0, 0, 100, 100, 50, 200):
        tracker.update([Detection(_box(0.5), 0.9, 1)], millis)
    track = tracker.tracks()[0]
    assert np.isfinite(track.filter_state).all()
    assert 0.0 <= track.bbox.x <= 1.0


def test_a_degenerate_box_does_not_produce_a_nan_track():
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    tracker.update([Detection(BoundingBox(0.5, 0.5, 0.0, 0.0), 0.9, 1)], 0)
    tracker.update([Detection(BoundingBox(0.5, 0.5, 0.0, 0.0), 0.9, 1)], STEP)
    for track in tracker.all_tracks():
        assert np.isfinite(track.filter_state).all()
        assert track.bbox.height > 0


# ------------------------------------------------------- the cost matrix


def test_every_cell_of_the_cost_matrix_is_on_one_scale():
    """A pair nobody can judge on looks must score as neutral, never as free.

    The first version of this tracker blended appearance into the cost when it
    had one and used bare geometry when it did not, so a good appearance match
    scored half what the same geometry scored for a pair whose appearance was
    unknown. In a crowd the unknown one is whoever is occluded, so the solver
    systematically preferred the wrong candidate. Measured at the time: 105
    identity switches with appearance against 95 without, over eight runs of
    four people milling.
    """
    from vigil.domain.appearance import Appearance
    from vigil.domain.tracking import NEUTRAL_APPEARANCE

    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, appearance_weight=0.5))
    look = Appearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), 500)
    box = _box(0.5, y=0.5)
    tracker.update([Detection(box, 0.9, 1)], 0, appearances=[look])
    tracker.update([Detection(box, 0.9, 1)], STEP, appearances=[look])
    track = tracker.tracks()[0]

    matched = tracker._associate([track], [0], [Detection(box, 0.9, 1)],
                                 np.array([[0.55, 0.6, 0.5, 0.2]]), [None],
                                 use_appearance=True, min_iou=0.1, position_only=False)
    assert matched, "a detection with no appearance must still be matchable"
    # And the neutral value is what a matrix of mixed evidence is levelled on.
    assert 0 < NEUTRAL_APPEARANCE < 1


def test_the_two_appearance_gates_do_different_jobs():
    """Loose inside a frame, tight across a gap — and the tight one is tighter.

    Measured: a single gate at 0.45 vetoed correct pairs whose crop was
    momentarily contaminated and cost ten identity switches; loosening it to
    0.70 recovered them. But 0.70 is past the separation between two different
    people (0.56 on the synthetic scenes), so re-identification — where
    appearance is the only evidence — keeps its own tighter threshold.
    """
    from vigil.domain.appearance import MAX_APPEARANCE_DISTANCE, MAX_REIDENTIFY_DISTANCE

    assert MAX_REIDENTIFY_DISTANCE < MAX_APPEARANCE_DISTANCE


def test_an_occluded_crop_is_not_remembered_as_what_somebody_looks_like():
    """A crop of a partly occluded person is a crop of two people, and a
    gallery that stores it goes on matching the wrong one for seconds."""
    from vigil.domain.appearance import Appearance

    # Which is in front is read off the ground: the lower box bottom is the
    # nearer object, because image row maps monotonically to ground distance.
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, max_occlusion=0.25))
    behind = BoundingBox(0.40, 0.46, 0.10, 0.20)    # bottom 0.66: further
    infront = BoundingBox(0.44, 0.50, 0.10, 0.20)   # bottom 0.70: nearer
    look = Appearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), 500)
    tracker.update([Detection(behind, 0.9, 1), Detection(infront, 0.9, 2)], 0,
                   appearances=[look, look])
    galleries = {t.class_id: len(t.gallery) for t in tracker.tracks()}
    assert galleries[1] == 0, "the occluded object's contaminated crop was remembered"
    assert galleries[2] == 1, "the object in front was seen cleanly and must keep its look"

    # Two objects at the same distance contaminate each other both ways.
    side_by_side = Tracker(TrackerConfig(min_hits_to_confirm=1, max_occlusion=0.25))
    a = BoundingBox(0.40, 0.50, 0.10, 0.20)
    b = BoundingBox(0.44, 0.50, 0.10, 0.20)
    side_by_side.update([Detection(a, 0.9, 1), Detection(b, 0.9, 2)], 0, appearances=[look, look])
    assert all(len(t.gallery) == 0 for t in side_by_side.tracks())

    # A mask exempts it: the detector has already excluded whoever is in front.
    masked = Tracker(TrackerConfig(min_hits_to_confirm=1, max_occlusion=0.25))
    mask = np.ones((8, 8), dtype=np.uint8)
    masked.update([Detection(behind, 0.9, 1, mask=mask), Detection(infront, 0.9, 2, mask=mask)], 0,
                  appearances=[look, look])
    assert all(len(t.gallery) == 1 for t in masked.tracks())


# ------------------------------------------------- self-calibrating re-id


def test_the_scene_measures_what_a_stranger_looks_like_without_being_told():
    """Two tracks in one frame are different objects by construction, so the
    between-object distribution is ground truth nobody had to label."""
    from vigil.domain.appearance import MIN_SEPARATION_SAMPLES, Appearance

    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    red = Appearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), 500)
    blue = Appearance(np.array([0.0, 1.0, 0.0], dtype=np.float32), 500)
    assert not tracker.separation.measured
    for i in range(MIN_SEPARATION_SAMPLES + 5):
        tracker.update(
            [Detection(_box(0.2, y=0.5), 0.9, 1), Detection(_box(0.7, y=0.5), 0.9, 1)],
            i * STEP, appearances=[red, blue],
        )
    assert tracker.separation.measured
    # Orthogonal descriptors are a cosine distance of 1 apart.
    assert abs(tracker.separation.ceiling() - min(1.0, 0.35)) < 1e-6, (
        "a scene where strangers are obviously different must not raise the ceiling above "
        "the shipped one"
    )
    assert "different objects in this scene" in tracker.separation.describe()


def test_a_scene_where_everything_looks_alike_tightens_its_own_gate():
    """The finding `tools/calibrate.py` produced on real video: different
    objects at a median of 0.105 against a shipped gate of 0.35. A scene that
    proves its strangers look similar must lower its own ceiling."""
    from vigil.domain.appearance import MAX_REIDENTIFY_DISTANCE, SceneSeparation

    scene = SceneSeparation()
    for _ in range(60):
        scene.observe(0.10)
    assert scene.measured
    assert scene.ceiling() < MAX_REIDENTIFY_DISTANCE
    assert scene.ceiling() <= 0.11
    assert scene.margin() > 0


def test_an_unmeasured_scene_falls_back_to_the_shipped_ceiling_and_no_margin():
    from vigil.domain.appearance import MAX_REIDENTIFY_DISTANCE, SceneSeparation

    scene = SceneSeparation()
    scene.observe(0.5)
    assert not scene.measured
    assert scene.ceiling() == MAX_REIDENTIFY_DISTANCE
    assert "not yet measured" in scene.describe()


def test_two_equally_good_candidates_are_refused_rather_than_guessed_between():
    """A margin, symmetric. Two objects that look equally like a lost track
    mean the descriptor cannot tell, and picking one is guessing with an
    operator's incident report."""
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    cost = np.array([[0.10, 0.12], [0.90, 0.95]])
    tracker._require_margin(cost, margin=0.10)
    assert cost[0, 0] > 1e8 and cost[0, 1] > 1e8, "an ambiguous row must be refused whole"

    clear = np.array([[0.10, 0.60], [0.90, 0.95]])
    tracker._require_margin(clear, margin=0.10)
    assert clear[0, 0] == 0.10, "a clear winner survives"

    # A single feasible candidate is decided by the ceiling, not by a
    # comparison with nothing.
    alone = np.array([[0.10, FORBIDDEN]])
    tracker._require_margin(alone, margin=0.10)
    assert alone[0, 0] == 0.10


def test_a_scene_that_cannot_tell_two_objects_apart_declines_to_merge_them():
    """End to end: two identical-looking objects, one of which disappears and
    the other stays. The tracker must not hand the survivor the lost one's
    identity — a fragment is visible on screen and a merge is not."""
    from vigil.domain.appearance import Appearance

    same = Appearance(np.array([1.0, 0.0, 0.0], dtype=np.float32), 500)
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1, coast_millis=200))
    # Long enough for the scene to learn that its two objects are identical.
    for i in range(60):
        tracker.update(
            [Detection(_box(0.20, y=0.5), 0.9, 1), Detection(_box(0.60, y=0.5), 0.9, 1)],
            i * STEP, appearances=[same, same],
        )
    assert tracker.separation.measured
    assert tracker.separation.ceiling() < 0.05, "identical objects must collapse the ceiling"
    left = {t.id for t in tracker.tracks() if t.bbox.center.x < 0.4}
    assert left

    # The left one leaves; the right one drifts towards where it was.
    for i in range(60, 90):
        x = 0.60 - 0.01 * (i - 60)
        tracker.update([Detection(_box(x, y=0.5), 0.9, 1)], i * STEP, appearances=[same])
    survivors = {t.id for t in tracker.tracks()}
    assert not (survivors & left), (
        "the survivor was handed the departed object's identity, which is a merge"
    )
