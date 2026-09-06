"""Geometry, against answers worked out independently of the code.

Where a closed form exists the test uses it. Where one does not, the test
states the geometric property that must hold — orthonormality, exact
invertibility, monotonicity of error with range — because a test that just
records what the code printed last time cannot tell a rewrite from a
regression.
"""

import math

import pytest

from vigil.domain.geo import (
    CameraPose, Distance, LatLon, LocalFrame, PoseUncertainty, PositionEstimate, PositionSource,
    ProjectionFailure, Vec2, basis_for, bearing_degrees, camera_sees, destination_point,
    distance_from_camera, distance_meters, far_ground_distance, field_of_view, image_coordinates,
    near_ground_distance, point_in_ring, project_point, project_to_ground,
    project_to_ground_detail, ray_angles, separation, spherical_distance,
)

SITE = LatLon(33.8938, 35.5018)


def _exact(pose: CameraPose) -> CameraPose:
    """The same pose with every input known perfectly, so a test can measure
    one error at a time."""
    from dataclasses import replace

    return replace(pose, uncertainty=PoseUncertainty.exact())


# ------------------------------------------------------------------ geodesy


def test_distance_bearing_and_destination_are_exact_inverses():
    for bearing in (0.0, 45.0, 137.0, 271.5, 359.0):
        for distance in (0.5, 50.0, 500.0):
            there = destination_point(SITE, bearing, distance)
            assert abs(distance_meters(SITE, there) - distance) < 1e-6
            off = (bearing_degrees(SITE, there) - bearing + 180) % 360 - 180
            # Expressed as sideways metres: a bearing tolerance means nothing
            # without a range.
            assert abs(math.radians(off)) * distance < 1e-6


def test_the_product_uses_one_earth_and_the_other_one_is_a_quarter_percent_away():
    # The defect this module was rewritten to remove, held as a measurement so
    # it cannot come back quietly. If a haversine reappears in a distance
    # path, this is what it costs.
    there = destination_point(SITE, 0.0, 1000.0)
    plane, sphere = distance_meters(SITE, there), spherical_distance(SITE, there)
    assert 0.002 < abs(sphere / plane - 1) < 0.003
    assert sphere - plane > 2.0, "over a kilometre the two Earths differ by metres"


def test_a_site_on_the_date_line_measures_itself_correctly():
    west, east = LatLon(0.0, 179.9995), LatLon(0.0, -179.9995)
    assert 100.0 < distance_meters(west, east) < 200.0, "not most of the way round the world"


def test_a_point_at_its_own_position_answers_rather_than_returning_a_nan():
    assert bearing_degrees(SITE, SITE) == 0.0
    assert distance_meters(SITE, SITE) == 0.0


def test_the_local_frame_round_trips_exactly():
    frame = LocalFrame(SITE)
    for bearing in (0.0, 30.0, 210.0):
        point = destination_point(SITE, bearing, 300.0)
        local = frame.to_local(point)
        assert abs(math.hypot(local.x, local.y) - 300.0) < 1e-6
        assert distance_meters(point, frame.to_lat_lon(local)) < 1e-6


# ------------------------------------------------------------- camera model


def test_the_camera_basis_is_orthonormal_and_left_handed_for_every_orientation():
    # (right, up, forward) is left-handed because a compass heading turns
    # clockwise while ENU turns anticlockwise; `up x right = forward` is the
    # identity that holds. Getting this backwards mirrors the image.
    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    for heading in (0.0, 37.0, 180.0, 300.0):
        for pitch in (-60.0, -25.0, 0.0, 15.0):
            for roll in (-30.0, 0.0, 12.0):
                basis = basis_for(CameraPose(SITE, 4.0, heading, pitch, roll))
                axes = (basis.right, basis.up, basis.forward)
                for axis in axes:
                    assert abs(math.sqrt(sum(c * c for c in axis)) - 1.0) < 1e-12
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    assert abs(sum(a * b for a, b in zip(axes[i], axes[j]))) < 1e-12
                for got, want in zip(cross(basis.up, basis.right), basis.forward):
                    assert abs(got - want) < 1e-12


def test_the_centre_ray_is_the_pose_and_roll_never_moves_it(pose):
    from dataclasses import replace

    for heading in (0.0, 91.0, 271.0):
        for pitch in (-45.0, -10.0):
            for roll in (-20.0, 0.0, 20.0):
                turned = replace(pose, heading=heading, pitch=pitch, roll=roll)
                bearing, elevation = ray_angles(turned, 0.5, 0.5)
                assert abs((bearing - heading + 180) % 360 - 180) < 1e-9
                assert abs(elevation - pitch) < 1e-9


def test_a_level_camera_reduces_to_the_closed_form():
    # With no pitch the pinhole model and the decoupled formula it replaced
    # agree exactly. That is the case the old model was checked against, which
    # is why its error survived two versions.
    level = CameraPose(SITE, 4.0, 0.0, 0.0, 0.0, 62.0, 36.0)
    half = math.tan(math.radians(31.0))
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        bearing, _ = ray_angles(level, u, 0.5)
        expected = math.degrees(math.atan((2 * u - 1) * half))
        assert abs((bearing - expected + 180) % 360 - 180) < 1e-9


def test_the_pinhole_model_differs_from_the_decoupled_one_by_a_fifth_of_the_range(pose):
    # The regression this rewrite exists for, as a number. At the bottom
    # corner of the reference pose the decoupled formula is degrees out, and
    # degrees of depression are metres of range.
    _, elevation = ray_angles(pose, 1.0, 1.0)
    decoupled = pose.pitch + math.degrees(math.atan(-math.tan(math.radians(pose.vertical_fov / 2))))
    assert abs(elevation - decoupled) > 4.0, f"{elevation} vs {decoupled}"
    correct = pose.mount_height / math.tan(math.radians(-elevation))
    wrong = pose.mount_height / math.tan(math.radians(-decoupled))
    assert abs(correct - wrong) / correct > 0.15, f"{correct:.2f} m vs {wrong:.2f} m"


def test_a_point_projects_to_the_ground_at_the_distance_the_closed_form_gives(pose):
    # Down the centre column there is a closed form for the whole pipeline:
    # h / tan(depression), and the position must be that far away.
    for v in (0.55, 0.7, 0.9, 1.0):
        projection = project_to_ground(_exact(pose), 0.5, v, 0.0)
        assert projection is not None
        _, elevation = ray_angles(pose, 0.5, v)
        expected = pose.mount_height / math.tan(math.radians(-elevation))
        assert abs(projection.ground_distance_meters - expected) < 1e-9
        assert abs(distance_meters(pose.position, projection.position) - expected) < 1e-6


def test_image_coordinates_invert_the_projection_exactly_including_under_roll(pose):
    from dataclasses import replace

    for roll in (-15.0, 0.0, 8.0):
        for heading in (0.0, 47.0, 300.0):
            turned = replace(pose, roll=roll, heading=heading)
            for u, v in ((0.5, 0.7), (0.05, 0.95), (0.95, 0.6), (0.2, 0.55), (0.8, 0.99)):
                projection = project_to_ground(turned, u, v, enforce_range=False)
                assert projection is not None
                back = image_coordinates(turned, projection.position)
                assert back is not None
                assert abs(back.x - u) < 1e-6 and abs(back.y - v) < 1e-6, f"roll {roll} at ({u},{v})"


def test_roll_moves_a_corner_by_about_a_metre_and_is_no_longer_ignored(pose):
    from dataclasses import replace

    level = project_to_ground(_exact(pose), 0.15, 0.9, 0.0)
    rolled = project_to_ground(_exact(replace(pose, roll=10.0)), 0.15, 0.9, 0.0)
    assert level is not None and rolled is not None
    moved = distance_meters(level.position, rolled.position)
    # Measured at 0.98 m for this pose. The threshold states "roll is not
    # negligible" without pinning a number a change of default FOV would move.
    assert moved > 0.5, f"10° of roll moved a corner by {moved:.2f} m"


def test_camera_sees_accepts_what_it_projected_and_refuses_what_is_behind(pose):
    projection = project_to_ground(pose, 0.5, 0.7)
    assert projection is not None and camera_sees(pose, projection.position)
    assert not camera_sees(pose, destination_point(pose.position, 180.0, 10.0))


# ------------------------------------------------------------- uncertainty


def test_the_height_column_of_the_jacobian_matches_its_closed_form(pose):
    # Ground offset is exactly proportional to mount height, so dg/dh = g/h.
    # If the finite difference is right here it is right in general — which is
    # the argument for taking the Jacobian numerically at all.
    from dataclasses import replace

    only_height = replace(pose, uncertainty=PoseUncertainty(0.0, 0.0, 0.0, 0.15, 0.0))
    projection = project_to_ground(only_height, 0.5, 0.8, 0.0)
    assert projection is not None
    expected = projection.ground_distance_meters / pose.mount_height * 0.15
    assert abs(projection.uncertainty.along_meters - expected) < 1e-6
    assert projection.uncertainty.across_meters < 1e-9, "height is a range error only"


def test_a_perfectly_known_pose_and_contact_point_has_no_error_at_all(pose):
    projection = project_to_ground(_exact(pose), 0.5, 0.8, 0.0)
    assert projection is not None
    assert projection.uncertainty.radius_meters < 1e-9
    # and the contact point's own error alone still produces one.
    with_contact = project_to_ground(_exact(pose), 0.5, 0.8, 0.75)
    assert with_contact is not None and with_contact.uncertainty.radius_meters > 0


def test_error_grows_with_range_and_is_an_ellipse_not_a_circle(pose):
    near = project_to_ground(pose, 0.5, 0.95)
    far = project_to_ground(pose, 0.5, 0.55)
    assert near is not None and far is not None
    assert far.ground_distance_meters > near.ground_distance_meters
    assert far.uncertainty.radius_meters > near.uncertainty.radius_meters
    # A shallow ray is vague about range and sharp about direction. Reporting
    # this as one circle is what the rewrite stopped doing.
    assert far.uncertainty.along_meters > 2 * far.uncertainty.across_meters
    assert far.uncertainty.radius_meters == far.uncertainty.along_meters
    assert far.uncertainty.rms_meters < far.uncertainty.radius_meters


def test_the_stated_pose_error_dominates_and_a_survey_would_shrink_it(pose):
    from dataclasses import replace

    typed = project_to_ground(pose, 0.5, 0.7)
    surveyed = replace(pose, uncertainty=PoseUncertainty(0.2, 0.2, 0.2, 0.02, 0.005))
    measured = project_to_ground(surveyed, 0.5, 0.7)
    assert typed is not None and measured is not None
    assert typed.uncertainty.radius_meters > 2 * measured.uncertainty.radius_meters, (
        "an operator's click and a survey must not produce the same claim"
    )


# ------------------------------------------------------------------ refusal


def test_each_way_a_projection_can_fail_is_named_rather_than_collapsed(pose):
    from dataclasses import replace

    level = replace(pose, pitch=0.0)
    assert project_to_ground_detail(level, 0.5, 0.5)[1] is ProjectionFailure.ABOVE_HORIZON
    shallow = replace(pose, pitch=-1.0, vertical_fov=1.0)
    assert project_to_ground_detail(shallow, 0.5, 0.5)[1] is ProjectionFailure.TOO_SHALLOW
    short = replace(pose, range_meters=3.0)
    assert project_to_ground_detail(short, 0.5, 0.6)[1] is ProjectionFailure.OUT_OF_RANGE
    assert project_to_ground_detail(short, 0.5, 0.6, enforce_range=False)[1] is None
    broken = replace(pose, mount_height=0.0)
    assert project_to_ground_detail(broken, 0.5, 0.6)[1] is ProjectionFailure.BAD_POSE


def test_a_ray_that_cannot_be_projected_falls_back_to_the_camera_and_says_so():
    level = CameraPose(LatLon(0, 0), 4.0, 90.0, 0.0)
    assert project_to_ground(level, 0.5, 0.5) is None
    estimate = project_point(level, Vec2(0.5, 0.5))
    assert estimate.source is PositionSource.CAMERA_FALLBACK
    assert estimate.radius_meters == level.range_meters
    assert not estimate.is_projected and estimate.ellipse is None


def test_a_pose_validates_itself():
    with pytest.raises(ValueError):
        CameraPose(LatLon(0, 0), 0.0, 0.0, -10.0).validate()
    with pytest.raises(ValueError):
        CameraPose(LatLon(95, 0), 4.0, 0.0, -10.0).validate()
    with pytest.raises(ValueError):
        CameraPose(LatLon(0, 0), 4.0, 0.0, -200.0).validate()
    with pytest.raises(ValueError):
        CameraPose(LatLon(0, 0), 4.0, 0.0, -10.0, roll=400.0).validate()
    CameraPose(LatLon(0, 0), 4.0, 0.0, -10.0).validate()


# ---------------------------------------------------------------- footprint


def test_the_footprint_is_an_annulus_that_excludes_the_mast(pose):
    ring = field_of_view(pose, 8)
    assert len(ring) == 18, "far edge, then near edge back"
    distances = [distance_meters(pose.position, p) for p in ring]
    assert min(distances) > 1.0, "a tilted camera does not see its own feet"
    assert max(distances) <= pose.range_meters + 1e-6
    assert field_of_view(CameraPose(LatLon(0, 0), 4.0, 0.0, 20.0)) == []


def test_the_near_edge_is_measured_across_the_frame_not_down_its_middle(pose):
    # With roll the nearest ground in view is at a corner. v1 read the centre
    # column and drew a footprint over ground the camera could see.
    from dataclasses import replace

    rolled = replace(pose, roll=20.0)
    across = near_ground_distance(rolled)
    centre = project_to_ground(rolled, 0.5, 1.0, 0.0, enforce_range=False)
    assert across is not None and centre is not None
    assert across <= centre.ground_distance_meters + 1e-9
    assert centre.ground_distance_meters - across > 0.1


def test_the_ground_span_brackets_every_projected_point(pose):
    near, far = near_ground_distance(pose), far_ground_distance(pose)
    assert near is not None and far is not None and near < far
    for u in (0.0, 0.5, 1.0):
        for v in (0.0, 0.5, 1.0):
            projection = project_to_ground(pose, u, v, enforce_range=False)
            if projection is not None:
                assert near - 1e-6 <= projection.ground_distance_meters <= far + 1e-6


# ----------------------------------------------------------------- distance


def test_a_distance_carries_its_error_and_answers_only_what_it_can(pose):
    a = PositionEstimate(pose.position, 1.5, PositionSource.GROUND_PROJECTION)
    b = PositionEstimate(destination_point(pose.position, 90.0, 10.0), 2.0,
                         PositionSource.GROUND_PROJECTION)
    gap = separation(a, b)
    assert abs(gap.meters - 10.0) < 0.01
    # In quadrature: adding would claim the errors always conspire.
    assert abs(gap.error_meters - 2.5) < 0.01
    assert gap.describe() == "10.0 ± 2.5 m"
    assert gap.at_least == pytest.approx(7.5) and gap.at_most == pytest.approx(12.5)

    # `within` and `beyond` are deliberately not each other's negation:
    # between 7.5 and 12.5 m, a ten-metre question has no answer this can give.
    assert gap.within(13.0) and not gap.within(10.0)
    assert gap.beyond(7.0) and not gap.beyond(10.0)
    assert not gap.within(10.0) and not gap.beyond(10.0)

    from_mast = distance_from_camera(pose, b)
    assert abs(from_mast.meters - 10.0) < 0.01 and from_mast.error_meters == 2.0
    assert Distance(0.4, 0.1).describe() == "0.4 ± 0.1 m"


# ----------------------------------------------------------------- polygons


def test_point_in_ring_uses_metres_not_degrees():
    ring = [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0.001), LatLon(0.001, 0)]
    assert point_in_ring(ring, LatLon(0.0005, 0.0005))
    assert not point_in_ring(ring, LatLon(0.002, 0.0005))
    assert not point_in_ring(ring[:2], LatLon(0, 0))


def test_a_zone_says_how_far_a_position_is_from_its_edge_and_which_side(pose):
    from vigil.domain.zones import Zone, ZoneKind

    centre = pose.position
    ring = tuple(destination_point(centre, b, 7.07) for b in (45, 135, 225, 315))
    zone = Zone("z", "Yard", ZoneKind.RESTRICTED, ring)
    inside = PositionEstimate(centre, 0.5, PositionSource.GROUND_PROJECTION)
    outside = PositionEstimate(destination_point(centre, 0.0, 20.0), 0.5,
                               PositionSource.GROUND_PROJECTION)
    assert zone.distance_from(inside).meters < 0, "inside must read negative"
    assert abs(zone.distance_from(inside).meters + 5.0) < 0.5
    assert zone.distance_from(outside).meters > 0
    assert abs(zone.distance_from(outside).meters - 15.0) < 0.5
    assert zone.distance_from(outside).error_meters == 0.5
