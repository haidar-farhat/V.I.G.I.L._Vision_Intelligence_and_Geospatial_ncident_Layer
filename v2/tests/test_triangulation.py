"""Two cameras instead of one assumed plane.

The claim under test is not "the arithmetic is right" — `tests/test_native.py`
holds the Rust and the NumPy to each other for that. It is that the result is
*better than the assumption it replaces*, and that it refuses when it is not.
"""

from dataclasses import dataclass, replace

import pytest

from vigil.domain.geo import (
    CameraPose, LatLon, LocalFrame, PoseUncertainty, PositionSource, Vec2, distance_meters,
    image_coordinates, project_to_ground,
)
from vigil.domain.tracking import TrackState
from vigil.domain.triangulation import (
    GroundPlane, Refusal, Sighting, TriangulatedPoint, fit_ground, triangulate,
)
from vigil.service.triangulation import (
    MIN_GROUND_SAMPLES, Geometry, WORTH_HAVING_M, tilted,
)

#: Two cameras across a yard, both calibrated — which is the case where
#: triangulating is worth doing at all.
MEASURED = PoseUncertainty(0.10, 0.25, 0.11, 0.04)
WEST = CameraPose(LatLon(33.89380, 35.50180), 5.0, 70.0, -18.0, 0.0, 62.0, 36.0, 80.0, MEASURED)
EAST = CameraPose(LatLon(33.89380, 35.50245), 4.6, 290.0, -17.0, 0.0, 62.0, 36.0, 80.0, MEASURED)
FRAME = LocalFrame(WEST.position)


@dataclass
class _Track:
    """What `Geometry.observe` reads off a track, and nothing else."""

    id: int
    class_id: int
    contact: Vec2
    state: TrackState = TrackState.CONFIRMED
    last_seen_millis: int = 1000
    last_detected_millis: int = 1000


def _at(east: float, north: float, up: float) -> LatLon:
    return FRAME.to_lat_lon(Vec2(east, north))


def _sees(pose: CameraPose, east: float, north: float, up: float) -> Vec2 | None:
    """Where a point `up` metres above the ground lands in this camera.

    `image_coordinates` places points *on* the ground, so the ray has to be
    built by hand for anything standing above it — which is the whole case
    this module exists for.
    """
    import numpy as np

    from vigil.domain.geo import basis_for

    basis = basis_for(pose)
    origin = FRAME.to_local(pose.position)
    offset = np.array([east - origin.x, north - origin.y, up - pose.mount_height])
    # Solve for the (u, v) whose ray is parallel to `offset`.
    right = np.array(basis.ray(1.0, 0.5)) - np.array(basis.ray(0.0, 0.5))
    down = np.array(basis.ray(0.5, 1.0)) - np.array(basis.ray(0.5, 0.0))
    centre = np.array(basis.ray(0.5, 0.5))
    matrix = np.column_stack([right, down, -offset])
    try:
        du, dv, scale = np.linalg.solve(matrix, -centre)
    except np.linalg.LinAlgError:
        return None
    if scale <= 0:
        return None
    return Vec2(0.5 + du, 0.5 + dv)


def test_two_cameras_recover_a_known_point_on_the_ground():
    truth = project_to_ground(WEST, 0.5, 0.85, enforce_range=False).position
    result = triangulate(Sighting(WEST, image_coordinates(WEST, truth)),
                         Sighting(EAST, image_coordinates(EAST, truth)), frame=FRAME)
    assert not isinstance(result, Refusal), result
    assert distance_meters(result.position, truth) < 0.01
    assert abs(result.height_m) < 0.01, "a point on the ground is at the ground"
    assert result.gap_m < 0.01
    assert result.estimate().source is PositionSource.TRIANGULATED
    assert result.estimate().is_projected, "a triangulated point is a place, not a fallback"


def test_a_person_standing_above_the_ground_is_placed_where_they_are():
    """The defect this whole phase exists to remove. A flat-ground projection
    follows the ray past the person and lands metres long; two rays do not."""
    east, north, height = 8.0, 26.0, 1.6
    truth = _at(east, north, height)
    west_point, east_point = _sees(WEST, east, north, height), _sees(EAST, east, north, height)
    assert west_point is not None and east_point is not None

    result = triangulate(Sighting(WEST, west_point), Sighting(EAST, east_point), frame=FRAME)
    assert not isinstance(result, Refusal), result
    assert distance_meters(result.position, truth) < 0.05
    assert abs(result.height_m - height) < 0.05, (
        "the height above the ground is the measurement that says they are not on it")

    # What one camera and the flat-ground assumption do with the same pixel.
    flat = project_to_ground(WEST, west_point.x, west_point.y, enforce_range=False)
    assert flat is not None
    error = distance_meters(flat.position, truth)
    assert error > 2.0, f"the flat-ground error here is only {error:.2f} m — a weak example"


def test_a_pair_too_close_together_is_refused_rather_than_averaged():
    """Two cameras on one mast see one ray between them."""
    near = replace(EAST, position=LatLon(33.89380, 35.501815), heading=70.0, pitch=-18.0)
    truth = project_to_ground(WEST, 0.5, 0.7, enforce_range=False).position
    result = triangulate(Sighting(WEST, image_coordinates(WEST, truth)),
                         Sighting(near, image_coordinates(near, truth)), frame=FRAME)
    assert result is Refusal.TOO_LITTLE_PARALLAX
    assert "worse than projecting" in result.describe()


def test_rays_at_two_different_people_are_refused_as_the_association_test():
    one = project_to_ground(WEST, 0.35, 0.85, enforce_range=False).position
    other = project_to_ground(WEST, 0.75, 0.85, enforce_range=False).position
    assert distance_meters(one, other) > 4.0, "the two need to be genuinely apart"
    result = triangulate(Sighting(WEST, image_coordinates(WEST, one)),
                         Sighting(EAST, image_coordinates(EAST, other)), frame=FRAME)
    assert result is Refusal.TOO_FAR_APART


def test_the_parallax_a_pair_needs_comes_from_how_well_it_is_known():
    """A constant floor would let an uncalibrated pair produce confident
    nonsense, or throw a calibrated one away."""
    geometry = Geometry(WEST.position)
    assumed = replace(WEST, uncertainty=PoseUncertainty())
    needed_assumed = geometry._parallax_for(assumed, assumed)
    needed_measured = geometry._parallax_for(WEST, EAST)
    assert needed_measured < 10.0, f"a calibrated pair should be easy to satisfy: {needed_measured}"
    assert needed_assumed > 40.0, f"an assumed pose at 80 m needs a wide baseline: {needed_assumed}"
    # And the derivation holds: at the returned angle, the error is the
    # tolerance the constant was defined from.
    import math

    sigma = math.radians(PoseUncertainty().heading_deg)
    at_limit = assumed.range_meters * sigma / math.sin(math.radians(needed_assumed))
    assert abs(at_limit - WORTH_HAVING_M) < 0.1


def test_a_yard_with_a_fall_is_measured_rather_than_assumed():
    points = []
    for i in range(300):
        east = (i % 20) * 2.0 - 20.0
        north = (i // 20) * 3.0
        points.append((east, north, 0.03 * east - 0.015 * north))
    plane = fit_ground(points, WEST.position, threshold_m=0.3)
    assert isinstance(plane, GroundPlane)
    assert abs(plane.tilt_east - 0.03) < 1e-4 and abs(plane.tilt_north + 0.015) < 1e-4
    assert plane.inliers == 300
    assert "falls 3.4%" in plane.describe()

    # And a camera stops carrying an error bar for a slope that is now known.
    before = replace(WEST, uncertainty=PoseUncertainty())
    after = tilted(before, plane)
    assert after.uncertainty.terrain_slope < before.uncertainty.terrain_slope


def test_a_plane_from_too_little_evidence_is_not_written_back():
    points = [(float(i), 0.0, 0.0) for i in range(10)] + [(0.0, float(i), 0.0) for i in range(10)]
    plane = fit_ground(points, WEST.position)
    assert plane is not None and plane.inliers < MIN_GROUND_SAMPLES
    pose = replace(WEST, uncertainty=PoseUncertainty())
    assert tilted(pose, plane) == pose, "twenty points do not describe a site"
    assert tilted(pose, None) == pose


def test_people_standing_on_a_dock_do_not_tilt_the_yard():
    """RANSAC's reason for being here. A quarter of the contacts are 1.2 m up
    at one end; a least-squares plane would tilt to split the difference and
    every position on the yard would move."""
    points = [((i % 20) * 2.0 - 20.0, (i // 20) * 3.0, 0.0) for i in range(300)]
    points += [(18.0, float(i) * 0.5, 1.2) for i in range(90)]
    plane = fit_ground(points, WEST.position, threshold_m=0.3, seed=17)
    assert plane is not None
    assert plane.inliers == 300, "the dock must be excluded, not averaged in"
    assert abs(plane.tilt_east) < 1e-6 and abs(plane.tilt_north) < 1e-6
    assert abs(plane.height_above(18.0, 10.0, 1.2) - 1.2) < 1e-6


def test_two_cameras_watching_one_yard_pair_their_tracks_and_solve_the_ground():
    geometry = Geometry(WEST.position)
    walk = [(-6.0 + i * 0.6, 20.0 + i * 0.25) for i in range(60)]
    paired = 0
    for step, (east, north) in enumerate(walk):
        at = 1000 + step * 100
        west_point, east_point = _sees(WEST, east, north, 0.0), _sees(EAST, east, north, 0.0)
        if west_point is None or east_point is None:
            continue
        geometry.observe("west", WEST, at, [_Track(1, 0, west_point, last_seen_millis=at,
                                                   last_detected_millis=at)])
        geometry.observe("east", EAST, at, [_Track(7, 0, east_point, last_seen_millis=at,
                                                   last_detected_millis=at)])
        found = geometry.resolve()
        for pairing in found:
            paired += 1
            assert {pairing.track_a, pairing.track_b} == {1, 7}
            assert distance_meters(pairing.point.position, _at(east, north, 0.0)) < 0.05
    assert paired > 30, f"only {paired} of {len(walk)} steps paired"
    assert geometry.samples == paired


def test_a_coasted_track_is_never_offered_to_the_triangulator():
    """A coasted box is the filter's prediction. Two predictions triangulate
    into a confident position that nothing observed."""
    geometry = Geometry(WEST.position)
    point = _sees(WEST, 0.0, 25.0, 0.0)
    other = _sees(EAST, 0.0, 25.0, 0.0)
    geometry.observe("west", WEST, 1000, [_Track(1, 0, point, last_detected_millis=600)])
    geometry.observe("east", EAST, 1000, [_Track(2, 0, other)])
    assert geometry.resolve() == ()

    geometry.observe("west", WEST, 1000, [_Track(1, 0, point, state=TrackState.TENTATIVE)])
    geometry.observe("east", EAST, 1000, [_Track(2, 0, other)])
    assert geometry.resolve() == ()


def test_sightings_from_different_moments_are_not_paired():
    """Two cameras are not synchronised. Pairing a person against where they
    were a second ago converges perfectly well and is wrong."""
    geometry = Geometry(WEST.position)
    point, other = _sees(WEST, 0.0, 25.0, 0.0), _sees(EAST, 0.0, 25.0, 0.0)
    geometry.observe("west", WEST, 1000, [_Track(1, 0, point, last_seen_millis=1000,
                                                 last_detected_millis=1000)])
    late = 1000 + 5000
    geometry.observe("east", EAST, late, [_Track(2, 0, other, last_seen_millis=late,
                                                 last_detected_millis=late)])
    assert geometry.resolve() == ()


def test_one_track_cannot_be_paired_with_two_objects_at_once():
    """A crowd otherwise produces a pairing per combination and a position
    per pairing."""
    geometry = Geometry(WEST.position, min_parallax_deg=1.0)
    places = [(-4.0, 24.0), (4.0, 26.0)]
    west_tracks, east_tracks = [], []
    for i, (east, north) in enumerate(places):
        west_tracks.append(_Track(i + 1, 0, _sees(WEST, east, north, 0.0)))
        east_tracks.append(_Track(i + 11, 0, _sees(EAST, east, north, 0.0)))
    geometry.observe("west", WEST, 1000, west_tracks)
    geometry.observe("east", EAST, 1000, east_tracks)
    found = geometry.resolve()
    assert len(found) == 2
    assert len({p.track_a for p in found}) == 2 and len({p.track_b for p in found}) == 2


def test_an_unplaced_camera_contributes_nothing_rather_than_a_guess():
    geometry = Geometry(WEST.position)
    geometry.observe("west", None, 1000, [_Track(1, 0, Vec2(0.5, 0.8))])
    assert geometry.resolve() == ()
    assert triangulate(Sighting(None, Vec2(0.5, 0.8)),
                       Sighting(EAST, Vec2(0.5, 0.8))) is Refusal.NOT_PLACED


def test_height_above_the_fitted_ground_tells_a_dock_from_a_yard():
    geometry = Geometry(WEST.position)
    for i in range(400):
        east = (i % 20) * 2.0 - 20.0
        north = (i // 20) * 1.5 + 10.0
        geometry._ground.append((east, north, 0.02 * east))
    assert geometry.solve_ground(threshold_m=0.3) is not None
    assert geometry.ground is not None and abs(geometry.ground.tilt_east - 0.02) < 1e-4

    from vigil.domain.triangulation import TriangulatedPoint

    on_the_slope = TriangulatedPoint(_at(10.0, 20.0, 0.0), 0.2, 0.1, 40.0, 0.02, 30.0, 30.0)
    assert abs(geometry.height_above_ground(on_the_slope)) < 0.05
    assert geometry.stands_on_the_ground(on_the_slope)

    on_a_dock = TriangulatedPoint(_at(10.0, 20.0, 0.0), 1.4, 0.1, 40.0, 0.02, 30.0, 30.0)
    assert abs(geometry.height_above_ground(on_a_dock) - 1.2) < 0.05
    assert not geometry.stands_on_the_ground(on_a_dock)


def test_a_solved_ground_reaches_every_camera_including_the_one_that_cannot_see_it(tmp_path):
    """The write-back, which is what turns a measurement into an improvement.

    It reaches the lone camera watching the far corner too — the one whose
    positions were worst and which could never have measured the ground
    itself. A site's ground is one surface.
    """
    from vigil.service.auth import Principal, Role
    from vigil.service.runtime import Runtime
    from vigil.service.site import SiteService
    from vigil.storage.store import Store

    admin = Principal("root", Role.ADMIN, "user")
    with Store(":memory:") as store:
        site = SiteService(store)
        site.add_camera("west", "file:///a", pose=WEST, by=admin)
        site.add_camera("east", "file:///b", pose=EAST, by=admin)
        # No overlap with the other two, and no way to measure anything.
        alone = replace(WEST, position=LatLon(33.8960, 35.5100), heading=180.0)
        site.add_camera("far", "file:///c", pose=alone, by=admin)
        runtime = Runtime(site)
        assert runtime.ground_plane() is None, "nothing is solved before anything is seen"

        runtime._geometry = Geometry(WEST.position)
        for i in range(500):
            east = (i % 25) * 2.0 - 25.0
            north = (i // 25) * 2.5 + 8.0
            runtime._geometry._ground.append((east, north, 0.025 * east - 0.01 * north))
        plane = runtime._geometry.solve_ground(threshold_m=0.3)
        assert plane is not None and plane.inliers >= MIN_GROUND_SAMPLES
        assert runtime.ground_plane() is plane, "and the runtime publishes what was solved"

        poses = {c["id"]: c["pose"] for c in store.cameras()}
        assert all(p.ground_tilt_east == 0.0 for p in poses.values()), "level to begin with"
        runtime._apply_ground(plane, poses)

        for camera in store.cameras():
            pose = camera["pose"]
            assert abs(pose.ground_tilt_east - 0.025) < 1e-4, camera["id"]
            assert abs(pose.ground_tilt_north + 0.01) < 1e-4, camera["id"]
            assert camera["ground_observations"] == plane.inliers
        assert any(r["action"] == "site.ground_solved" for r in store.audit_trail())

        # And it changes where that lone camera puts things, which is the
        # whole point of writing it to a camera that measured nothing.
        after = next(c["pose"] for c in store.cameras() if c["id"] == "far")
        level = project_to_ground(alone, 0.5, 0.75, enforce_range=False)
        solved = project_to_ground(after, 0.5, 0.75, enforce_range=False)
        assert distance_meters(level.position, solved.position) > 0.05, (
            "a solved ground that moves nothing has not been applied")
