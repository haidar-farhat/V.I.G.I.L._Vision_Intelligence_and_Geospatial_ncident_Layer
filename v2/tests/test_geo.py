import math

import pytest

from vigil.domain.geo import (
    CameraPose, LatLon, PositionSource, Vec2, bearing_degrees, camera_sees, destination_point, field_of_view,
    haversine_distance, image_coordinates, point_in_ring, project_point, project_to_ground, ray_angles,
)


def test_geodesy_round_trips_across_a_site():
    origin = LatLon(33.8938, 35.5018)
    there = destination_point(origin, 45.0, 100.0)
    assert abs(haversine_distance(origin, there) - 100.0) < 0.01
    assert abs(bearing_degrees(origin, there) - 45.0) < 0.01


def test_the_frame_centre_ray_follows_the_pose(pose):
    bearing, elevation = ray_angles(pose, 0.5, 0.5)
    assert abs(bearing - pose.heading) < 1e-9 and abs(elevation - pose.pitch) < 1e-9
    left, _ = ray_angles(pose, 0.0, 0.5)
    assert abs(((left - pose.heading + 180) % 360 - 180) + pose.horizontal_fov / 2) < 1e-6


def test_a_point_projects_to_the_ground_with_a_stated_uncertainty(pose):
    projection = project_to_ground(pose, 0.5, 0.8)
    assert projection is not None
    expected = pose.mount_height / math.tan(math.radians(-ray_angles(pose, 0.5, 0.8)[1]))
    assert abs(projection.ground_distance_meters - expected) < 1e-6
    assert projection.uncertainty_meters > 0
    assert abs(haversine_distance(pose.position, projection.position) - expected) < 0.01


def test_a_ray_at_the_horizon_projects_nowhere_and_falls_back_to_the_camera():
    level = CameraPose(LatLon(0, 0), 4.0, 90.0, 0.0)
    assert project_to_ground(level, 0.5, 0.5) is None
    estimate = project_point(level, Vec2(0.5, 0.5))
    assert estimate.source is PositionSource.CAMERA_FALLBACK and estimate.radius_meters == level.range_meters
    assert not estimate.is_projected


def test_beyond_range_is_refused_not_clamped():
    far = CameraPose(LatLon(0, 0), 4.0, 0.0, -3.0, range_meters=30.0)
    assert project_to_ground(far, 0.5, 0.5) is None
    assert project_to_ground(far, 0.5, 0.5, enforce_range=False) is not None


def test_the_field_of_view_is_an_annular_sector_and_empty_when_the_camera_sees_no_ground(pose):
    ring = field_of_view(pose, 8)
    assert len(ring) == 18, "far arc, then near arc back"
    distances = [haversine_distance(pose.position, p) for p in ring]
    assert min(distances) > 1.0, "a tilted camera does not see its own feet"
    sky = CameraPose(LatLon(0, 0), 4.0, 0.0, 20.0)
    assert field_of_view(sky) == []


def test_image_coordinates_invert_the_projection(pose):
    for u, v in ((0.5, 0.7), (0.2, 0.9), (0.8, 0.6)):
        projection = project_to_ground(pose, u, v)
        assert projection is not None
        back = image_coordinates(pose, projection.position)
        assert back is not None and abs(back.x - u) < 1e-3 and abs(back.y - v) < 1e-3
    assert camera_sees(pose, project_to_ground(pose, 0.5, 0.7).position)
    behind = destination_point(pose.position, 180.0, 10.0)
    assert not camera_sees(pose, behind)


def test_point_in_ring_uses_metres_not_degrees():
    ring = [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0.001), LatLon(0.001, 0)]
    assert point_in_ring(ring, LatLon(0.0005, 0.0005))
    assert not point_in_ring(ring, LatLon(0.002, 0.0005))
    assert not point_in_ring(ring[:2], LatLon(0, 0))


def test_a_pose_validates_itself():
    with pytest.raises(ValueError):
        CameraPose(LatLon(0, 0), 0.0, 0.0, -10.0).validate()
    with pytest.raises(ValueError):
        CameraPose(LatLon(95, 0), 4.0, 0.0, -10.0).validate()


def test_a_distance_carries_its_error_and_answers_only_what_it_can(pose):
    from vigil.domain.geo import (
        Distance, PositionEstimate, PositionSource, distance_from_camera, separation,
    )

    a = PositionEstimate(pose.position, 1.5, PositionSource.GROUND_PROJECTION)
    b = PositionEstimate(destination_point(pose.position, 90.0, 10.0), 2.0, PositionSource.GROUND_PROJECTION)
    gap = separation(a, b)
    assert abs(gap.meters - 10.0) < 0.01
    # In quadrature: adding would claim the errors always conspire.
    assert abs(gap.error_meters - 2.5) < 0.01
    assert gap.describe() == "10.0 ± 2.5 m"
    assert gap.at_least == pytest.approx(7.5) and gap.at_most == pytest.approx(12.5)

    # `within` and `beyond` are deliberately not each other's negation: between
    # 7.5 and 12.5 metres, a ten-metre question has no answer this can give.
    assert gap.within(13.0) and not gap.within(10.0)
    assert gap.beyond(7.0) and not gap.beyond(10.0)
    assert not gap.within(10.0) and not gap.beyond(10.0), "an unanswerable question must not be answered either way"

    from_mast = distance_from_camera(pose, b)
    assert abs(from_mast.meters - 10.0) < 0.01 and from_mast.error_meters == 2.0
    assert Distance(0.4, 0.1).describe() == "0.4 ± 0.1 m"


def test_a_zone_says_how_far_a_position_is_from_its_edge_and_which_side(pose):
    from vigil.domain.geo import PositionEstimate, PositionSource
    from vigil.domain.zones import Zone, ZoneKind

    centre = pose.position
    ring = tuple(destination_point(centre, b, 7.07) for b in (45, 135, 225, 315))
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, ring)
    inside = PositionEstimate(centre, 0.5, PositionSource.GROUND_PROJECTION)
    outside = PositionEstimate(destination_point(centre, 0.0, 20.0), 0.5, PositionSource.GROUND_PROJECTION)
    assert zone.distance_from(inside).meters < 0, "inside must read negative"
    assert abs(zone.distance_from(inside).meters + 5.0) < 0.5
    assert zone.distance_from(outside).meters > 0
    assert abs(zone.distance_from(outside).meters - 15.0) < 0.5
    assert zone.distance_from(outside).error_meters == 0.5
