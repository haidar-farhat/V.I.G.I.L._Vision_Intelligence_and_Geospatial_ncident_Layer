"""Tests for the Rust boundary.

The geometry and tracking mathematics is tested in Rust, where it lives. These
tests cover what only a caller can break: struct layouts, ownership, buffer
limits, and whether a value that is correct in Rust is still correct after it
crosses into Python.

Read them as "does the boundary lie?" rather than "is the maths right?".
"""

from __future__ import annotations

import ctypes
import math

import pytest

from sentinel.core import (
    ABI_VERSION,
    BoundingBox,
    CameraPose,
    CDetection,
    CoreError,
    CPoint,
    CPose,
    CProjection,
    CTrack,
    Detection,
    LatLon,
    Tracker,
    bearing_degrees,
    camera_sees,
    destination_point,
    field_of_view,
    haversine_distance,
    load_core,
    point_in_zone,
    project_to_ground,
)

SITE = LatLon(33.8938, 35.5018)


def pose(**overrides: float) -> CameraPose:
    """A camera on a 10 m mast, looking north, tilted 45 degrees down."""
    defaults = dict(
        position=SITE,
        mount_height=10.0,
        heading=0.0,
        pitch=-45.0,
        horizontal_fov=60.0,
        vertical_fov=34.0,
        range_meters=120.0,
    )
    defaults.update(overrides)
    return CameraPose(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the loader


def test_the_core_loads_and_agrees_on_the_abi_version():
    assert load_core().sentinel_abi_version() == ABI_VERSION


def test_every_struct_layout_matches_the_rust_definition():
    # The load itself already asserts this, but stating it as a test names the
    # failure: a drifted layout does not crash, it misreads every field after
    # the drift and produces geometry that looks plausible.
    lib = load_core()
    structs = (CPose, CDetection, CTrack, CProjection, CPoint)
    buffer = (ctypes.c_uint32 * len(structs))()

    assert lib.sentinel_struct_sizes(buffer, len(structs)) == len(structs)
    for index, struct in enumerate(structs):
        assert buffer[index] == ctypes.sizeof(struct), struct.__name__


def test_the_core_is_loaded_once_and_reused():
    assert load_core() is load_core()


# -------------------------------------------------------------- floats survive


def test_doubles_are_not_truncated_crossing_the_boundary():
    # Without declared argtypes ctypes assumes int, and every float silently
    # becomes garbage. This is the canary for that whole class of mistake.
    a = LatLon(33.8938, 35.5018)
    b = LatLon(33.8940, 35.5018)

    distance = haversine_distance(a, b)
    assert 20.0 < distance < 25.0
    assert distance != int(distance), "a truncated double would land on an integer"


def test_a_round_trip_through_destination_and_back_returns_the_origin():
    moved = destination_point(SITE, 37.5, 250.0)

    assert haversine_distance(SITE, moved) == pytest.approx(250.0, abs=0.01)
    assert bearing_degrees(SITE, moved) == pytest.approx(37.5, abs=0.001)


# ------------------------------------------------------------------ projection


def test_a_forty_five_degree_ray_lands_at_one_mount_height():
    result = project_to_ground(pose(), 0.5, 0.5)

    assert result is not None
    assert result.ground_distance_meters == pytest.approx(10.0, abs=1e-9)
    assert result.bearing_deg == pytest.approx(0.0, abs=1e-9)
    assert result.uncertainty_meters > 0.0


def test_a_ray_above_the_horizon_yields_nothing_rather_than_a_guess():
    # A camera looking nearly level: the top of its frame is sky. Placing that
    # detection anywhere on the map would be an invention.
    assert project_to_ground(pose(pitch=-2.0), 0.5, 0.05) is None


def test_uncertainty_grows_toward_the_horizon():
    near = project_to_ground(pose(), 0.5, 0.9)
    far = project_to_ground(pose(), 0.5, 0.1)

    assert near is not None and far is not None
    assert far.ground_distance_meters > near.ground_distance_meters
    # Super-linear: the error grows faster than the distance does, which is why
    # a horizon detection must never be drawn like a nearby one.
    assert (far.uncertainty_meters / near.uncertainty_meters) > (
        far.ground_distance_meters / near.ground_distance_meters
    )


def test_range_is_enforced_unless_the_caller_opts_out():
    short = pose(range_meters=12.0)

    assert project_to_ground(short, 0.5, 0.1) is None
    unbounded = project_to_ground(short, 0.5, 0.1, enforce_range=False)
    assert unbounded is not None and unbounded.ground_distance_meters > 12.0


# ---------------------------------------------------------------- field of view


def test_the_footprint_is_an_open_ring_traversing_the_far_arc_then_the_near():
    # Open by contract: the closing point is a serialisation concern, and
    # point-in-polygon handles the wrap itself. A renderer that needs it closed
    # appends the first point.
    footprint = field_of_view(pose(), arc_segments=12)

    assert len(footprint) == 26
    assert footprint[0] != footprint[-1]

    far = haversine_distance(SITE, footprint[0])
    near = haversine_distance(SITE, footprint[-1])
    assert far > near, "the far arc comes first, then the near one back"


def test_the_footprint_excludes_the_ground_at_the_camera_mast():
    # A downward-tilted camera is blind at its own feet. Drawing a pie slice
    # from the mast claims coverage the optics do not have.
    footprint = field_of_view(pose(), arc_segments=16)
    nearest = min(haversine_distance(SITE, point) for point in footprint)

    assert nearest > 1.0


def test_a_point_behind_the_camera_is_not_seen():
    ahead = destination_point(SITE, 0.0, 12.0)
    behind = destination_point(SITE, 180.0, 12.0)

    assert camera_sees(pose(), ahead) is True
    assert camera_sees(pose(), behind) is False


def test_the_vertical_field_of_view_bounds_coverage_regardless_of_stated_range():
    # This camera claims 120 m of range, but a 34-degree vertical FOV tilted 45
    # degrees down sees the ground only between about 5.3 m and 18.8 m. Trusting
    # the stated range would paint 100 m of coverage that does not exist.
    camera = pose(range_meters=120.0)

    assert camera_sees(camera, destination_point(SITE, 0.0, 2.0)) is False
    assert camera_sees(camera, destination_point(SITE, 0.0, 12.0)) is True
    assert camera_sees(camera, destination_point(SITE, 0.0, 40.0)) is False


def test_a_point_beyond_range_is_not_seen():
    assert camera_sees(pose(range_meters=30.0), destination_point(SITE, 0.0, 200.0)) is False


# ----------------------------------------------------------------------- zones


def test_a_zone_contains_its_interior_and_excludes_its_exterior():
    ring = [
        destination_point(SITE, bearing, 30.0)
        for bearing in (0.0, 90.0, 180.0, 270.0)
    ]

    assert point_in_zone(ring, SITE) is True
    assert point_in_zone(ring, destination_point(SITE, 45.0, 100.0)) is False


def test_a_degenerate_ring_contains_nothing():
    # Two points are a line, not an area. Reporting containment would let a
    # half-drawn zone start producing intrusion events.
    assert point_in_zone([SITE, destination_point(SITE, 0.0, 10.0)], SITE) is False


# -------------------------------------------------------------------- tracking


def walk(tracker: Tracker, steps: int, *, start: float = 0.30, step: float = 0.02):
    """Walk one box down the frame, returning the tracks from the last update."""
    tracks = []
    for index in range(steps):
        box = BoundingBox(0.45, start + index * step, 0.06, 0.12)
        tracks = tracker.update([Detection(box, 0.9, 0)], index * 200)
    return tracks


def test_a_single_object_keeps_one_identity():
    with Tracker(pose()) as tracker:
        seen = set()
        for index in range(20):
            box = BoundingBox(0.45, 0.30 + index * 0.02, 0.06, 0.12)
            for track in tracker.update([Detection(box, 0.9, 0)], index * 200):
                seen.add(track.id)

        assert seen == {1}


def test_small_fast_boxes_do_not_shatter_into_many_tracks():
    # A distant person is a small box moving further per frame than its own
    # width. Pure IoU association produces a new identity every frame, which
    # turns one person into a crowd.
    with Tracker(pose()) as tracker:
        seen = set()
        for index in range(15):
            box = BoundingBox(0.20 + index * 0.009, 0.55, 0.018, 0.040)
            for track in tracker.update([Detection(box, 0.85, 0)], index * 200):
                seen.add(track.id)

        assert len(seen) == 1


def test_a_track_carries_a_map_position_with_its_uncertainty():
    with Tracker(pose()) as tracker:
        tracks = walk(tracker, 10)

    assert len(tracks) == 1
    position = tracks[0].position
    assert position is not None
    assert position.source == "GROUND_PROJECTION"
    assert position.radius_meters > 0.0
    assert haversine_distance(SITE, position.point) < 120.0


def test_a_tracker_without_a_pose_reports_no_position():
    # An unplaced camera still tracks. It simply cannot say where on the map,
    # and says so rather than defaulting to the origin.
    with Tracker() as tracker:
        tracks = walk(tracker, 6)

    assert len(tracks) == 1
    assert tracks[0].position is None


def test_motion_appears_only_once_there_is_motion_to_report():
    with Tracker(pose()) as tracker:
        first = tracker.update([Detection(BoundingBox(0.45, 0.3, 0.06, 0.12), 0.9, 0)], 0)
        assert all(track.speed_mps is None for track in first)

        later = walk(tracker, 12)

    assert later[0].speed_mps is not None
    assert later[0].speed_mps > 0.0
    assert later[0].heading_degrees is not None


def test_standing_still_is_distinguishable_from_not_knowing():
    # Three states, not two. A person standing in one place for four minutes is
    # the loitering signal; reporting it as "motion unknown" throws that away.
    with Tracker(pose()) as tracker:
        first = tracker.update([Detection(BoundingBox(0.45, 0.55, 0.06, 0.12), 0.9, 0)], 0)
        assert all(track.speed_mps is None for track in first), "nothing known yet"

        tracks = []
        for index in range(1, 14):
            box = BoundingBox(0.45, 0.55, 0.06, 0.12)
            tracks = tracker.update([Detection(box, 0.9, 0)], index * 200)

    assert tracks[0].speed_mps == pytest.approx(0.0, abs=0.05), "the speed is known"
    assert tracks[0].heading_degrees is None, "a heading from jitter is worse than none"


def test_an_empty_frame_is_a_legitimate_update():
    with Tracker(pose(), max_gap_millis=100_000) as tracker:
        walk(tracker, 6)
        held = tracker.update([], 5_000)

        # The object is not gone, only unobserved: the track is held open until
        # the gap budget expires, so a one-frame miss is not a new identity.
        assert len(held) == 1


def test_a_track_ends_after_the_gap_budget_expires():
    with Tracker(pose(), max_gap_millis=1_000) as tracker:
        walk(tracker, 6)
        tracker.update([], 60_000)

        assert tracker.ended() == [1]
        assert tracker.update([], 60_200) == []


def test_reset_clears_the_tracker():
    with Tracker(pose()) as tracker:
        walk(tracker, 6)
        assert tracker.reset() >= 1
        assert tracker.update([], 10_000) == []


def test_tracking_is_deterministic():
    # Replaying evidence must reproduce it exactly, or an incident review cannot
    # be trusted to show what the operator saw.
    def run() -> list[tuple[int, float, float]]:
        with Tracker(pose()) as tracker:
            tracks = walk(tracker, 14)
            return [(t.id, t.bbox.x, t.bbox.y) for t in tracks]

    assert run() == run()


# ------------------------------------------------------------------- ownership


def test_a_closed_tracker_refuses_further_use():
    tracker = Tracker(pose())
    tracker.close()

    with pytest.raises(CoreError):
        tracker.update([], 0)


def test_closing_twice_is_safe():
    # Freeing a Rust allocation twice is undefined behaviour; the guard is here
    # rather than trusted to caller discipline.
    tracker = Tracker(pose())
    tracker.close()
    tracker.close()


def test_many_trackers_can_be_created_and_destroyed():
    # A worker cycling cameras creates and drops these constantly. A leak here
    # is invisible until a long-running node runs out of memory.
    for _ in range(200):
        with Tracker(pose()) as tracker:
            tracker.update([Detection(BoundingBox(0.4, 0.4, 0.1, 0.1), 0.9, 0)], 0)


def test_the_pose_can_change_under_a_live_tracker():
    # A PTZ camera moves. Its existing tracks keep their identity; only where
    # they land on the map changes.
    with Tracker(pose()) as tracker:
        before = walk(tracker, 8)
        tracker.set_pose(pose(heading=90.0))
        after = walk(tracker, 8, start=0.46)

        assert before[0].id == after[0].id
        assert before[0].position is not None and after[0].position is not None
        assert bearing_degrees(SITE, after[0].position.point) > 45.0


def test_a_track_survives_more_detections_than_the_read_buffer_expects():
    # 256 tracks is the read buffer. Feeding more must truncate honestly rather
    # than overflow, so the cap is exercised rather than assumed.
    with Tracker(pose(), min_hits_to_confirm=1) as tracker:
        detections = [
            Detection(BoundingBox(0.001 * i, 0.5, 0.002, 0.004), 0.9, 0)
            for i in range(300)
        ]
        tracks = tracker.update(detections, 0)

    assert len(tracks) <= 256
    assert len(tracks) > 0
    assert len({track.id for track in tracks}) == len(tracks)


# ---------------------------------------------------------- geometry sanity net


def test_the_projection_agrees_with_hand_arithmetic():
    # d = h / tan(theta), computed here independently of the core.
    camera = pose(mount_height=6.0, pitch=-30.0, vertical_fov=0.0)
    result = project_to_ground(camera, 0.5, 0.5)

    assert result is not None
    assert result.ground_distance_meters == pytest.approx(
        6.0 / math.tan(math.radians(30.0)), abs=1e-9
    )
