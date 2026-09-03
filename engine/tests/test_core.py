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

import numpy as np

import pytest

from sentinel.core import (
    ABI_VERSION,
    BoundingBox,
    CameraPose,
    CDetection,
    ContactPoint,
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


# ------------------------------------------------- the loader's refusal to guess
#
# Both of these are about the same failure: a library that is present but is not
# the core this build expects. Loading it and calling into it produces corrupted
# geometry — plausible numbers, silently wrong — which is far worse than a
# refusal at start-up.


class _NotTheCore:
    """A library that loads but exports none of the symbols wanted."""

    def __getattr__(self, name: str):
        raise AttributeError(name)


class _WrongVersionCore(_NotTheCore):
    """A real core, from an older build."""

    def __getattr__(self, name: str):
        if name == "sentinel_abi_version":
            function = lambda: ABI_VERSION + 1  # noqa: E731
            function.restype = None
            return function
        raise AttributeError(name)


def _load_with(monkeypatch, library, tmp_path):
    """Run `load_core` against a stand-in library, with the cache cleared."""
    from sentinel import core as core_module

    stand_in = tmp_path / "sentinel_core.dll"
    stand_in.write_bytes(b"not really a library")

    monkeypatch.setattr(core_module, "_lib", None)
    monkeypatch.setattr(core_module, "_candidate_paths", lambda: [stand_in])
    monkeypatch.setattr(core_module.ctypes, "CDLL", lambda _: library)
    return core_module.load_core


def test_a_library_without_the_version_symbol_is_refused_by_name(monkeypatch, tmp_path):
    # `_bind` touches every exported symbol, so binding before checking the
    # version meant a stale core died on a bare AttributeError naming whichever
    # symbol happened to be looked up first — a message that says nothing about
    # what is wrong or what to do about it.
    load = _load_with(monkeypatch, _NotTheCore(), tmp_path)

    with pytest.raises(CoreError, match="sentinel_abi_version"):
        load()


def test_a_core_from_an_older_build_is_refused_before_it_is_called(monkeypatch, tmp_path):
    load = _load_with(monkeypatch, _WrongVersionCore(), tmp_path)

    with pytest.raises(CoreError, match=f"expects {ABI_VERSION}"):
        load()


# --------------------------------------------------- the footprint is not partial


def test_a_tiny_segment_count_still_returns_a_whole_footprint():
    # The core clamps the arc to a minimum of two segments. Sizing the buffer
    # from the *requested* count instead under-allocated, the core truncated the
    # ring to fit, and the truncation status was discarded — so an open,
    # incomplete polygon came back and would have been drawn on the map as real
    # coverage.
    camera = pose()

    clamped = field_of_view(camera, arc_segments=0)
    explicit = field_of_view(camera, arc_segments=2)

    assert len(clamped) == len(explicit)
    assert [(p.lat, p.lon) for p in clamped] == [(p.lat, p.lon) for p in explicit]


def test_every_segment_count_returns_a_closed_ring():
    camera = pose()

    for segments in (0, 1, 2, 3, 8, 24, 64):
        ring = field_of_view(camera, arc_segments=segments)
        # far arc + near arc, both of `max(2, segments) + 1` points.
        assert len(ring) == 2 * (max(2, segments) + 1)


# ------------------------------------------------------------ ground contact


def test_the_map_position_is_projected_from_the_mask_not_the_box():
    """The whole point of carrying the mask across the boundary.

    Two detections with the identical box. One has a mask whose lowest lit row
    is far to the left of the box's bottom-centre — a person leaning out from
    behind something. Same rectangle, different place on the ground, and the
    map must say so.
    """
    box = BoundingBox(0.4, 0.5, 0.2, 0.3)
    mask = np.zeros((30, 20), dtype=np.uint8)
    mask[:, 0:4] = 1  # a bar down the left edge: the foot is at the left

    plain = Detection(bbox=box, confidence=0.9, class_id=0)
    shaped = Detection(bbox=box, confidence=0.9, class_id=0, mask=mask)

    with Tracker(pose(), min_hits_to_confirm=1) as a, Tracker(pose(), min_hits_to_confirm=1) as b:
        for step in range(3):
            from_box = a.update([plain], step * 200)
            from_mask = b.update([shaped], step * 200)

    assert from_box[0].contact == ContactPoint(0.5, 0.8), (
        "without a mask the contact must be exactly the box's bottom-centre"
    )
    assert from_mask[0].contact is not None
    assert from_mask[0].contact.x < 0.45, "the mask's foot was on the left"
    assert from_mask[0].contact == ground_contact_of(shaped)

    assert from_box[0].position is not None and from_mask[0].position is not None
    moved = haversine_distance(from_box[0].position.point, from_mask[0].position.point)
    assert moved > 0.5, f"the map position moved only {moved:.2f} m for a foot 8% of the frame away"


def ground_contact_of(detection: Detection) -> ContactPoint:
    from sentinel.core import ground_contact

    return ground_contact(detection)


def test_a_contact_point_survives_the_round_trip_through_the_core():
    # Written into CDetection by the tracker, read back out of CTrack. If the
    # two struct layouts disagree the value comes back as plausible garbage,
    # which is exactly the failure the struct-size guard cannot see.
    box = BoundingBox(0.4, 0.5, 0.2, 0.3)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[9, 7:9] = 1  # one foot, bottom right
    detection = Detection(bbox=box, confidence=0.9, class_id=0, mask=mask)

    with Tracker(None, min_hits_to_confirm=1) as tracker:
        (track,) = tracker.update([detection], 0)

    assert track.contact is not None
    assert track.contact.x == pytest.approx(0.4 + 0.2 * 8.0 / 10.0, abs=1e-9)
    assert track.contact.y == pytest.approx(0.8, abs=1e-9)
