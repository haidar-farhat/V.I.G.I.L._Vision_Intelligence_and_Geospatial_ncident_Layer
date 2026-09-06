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
