"""Two cameras, one world.

The specification's central claim is that three cameras seeing one person is one
incident, not three alerts. ``test_incidents.py`` checks that against events built
by hand, which proves the correlator's logic and nothing about the system.

This file proves the system. Two videos are rendered from one world through two
different camera poses, using the exact inverse of the projection the pipeline
uses to interpret them. Each is decoded, detected, tracked and projected
independently — neither pipeline knows the other exists — and only then are their
events correlated.

If the geometry is wrong anywhere in that loop, the two cameras disagree about
where the same person is, the association fails, and one person is reported as
two. That is a failure this test can actually detect, which is what separates it
from one built on events written to agree.

Ground truth here is a **world position in metres**, not a box in a picture, so
spatial accuracy is measured against where the person actually was.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

import world
from sentinel.core import CameraPose, destination_point, haversine_distance
from sentinel.decode import VideoSource
from sentinel.detect import MotionDetector
from sentinel.events import LoiteringRule, ZoneEntryRule
from sentinel.incidents import Correlator
from sentinel.pipeline import Pipeline
from sentinel.zones import Zone, ZoneKind


@pytest.fixture(scope="module")
def two_views(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One world, encoded from two camera poses."""
    directory = tmp_path_factory.mktemp("world")
    return {
        "cam-07": world.write_view(
            directory / "west.mp4", world.CAMERA_WEST, world.LONE_WALKER
        ),
        "cam-08": world.write_view(
            directory / "east.mp4", world.CAMERA_EAST, world.LONE_WALKER
        ),
    }


POSES = {"cam-07": world.CAMERA_WEST, "cam-08": world.CAMERA_EAST}


def restricted_zone() -> Zone:
    centre = world._offset(world.ORIGIN, 0.0, 18.0)
    return Zone(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=tuple(destination_point(centre, b, 10.0) for b in (0.0, 90.0, 180.0, 270.0)),
        enter_after_millis=600,
    )


def run_camera(camera_id: str, path: Path, zone: Zone | None = None):
    rules = [ZoneEntryRule(), LoiteringRule(dwell_millis=5000)] if zone else []
    with Pipeline(
        VideoSource(path, source_id=camera_id),
        MotionDetector(),
        pose=POSES[camera_id],
        zones=[zone] if zone else [],
        rules=rules,
        node_id="nd_site",
    ) as pipeline:
        results = list(pipeline.run())
        return results, pipeline.stats


@pytest.fixture(scope="module")
def both_cameras(two_views: dict[str, Path]):
    zone = restricted_zone()
    per_camera = {
        camera_id: run_camera(camera_id, path, zone)
        for camera_id, path in two_views.items()
    }
    events = [
        event
        for results, _ in per_camera.values()
        for result in results
        for event in result.events
    ]
    correlator = Correlator(zone_kinds={"zone-a": ZoneKind.RESTRICTED})
    return per_camera, events, correlator.correlate(events), correlator


# ------------------------------------------------------- each camera alone


def test_each_camera_tracks_the_person_as_one_object(both_cameras):
    per_camera, _, _, _ = both_cameras

    for camera_id, (_, stats) in per_camera.items():
        assert stats.distinct_objects == 1, (
            f"{camera_id} reported {stats.distinct_objects} objects for one person"
        )


def test_positions_agree_with_where_the_person_actually_was(both_cameras):
    """Spatial accuracy against a world position, not against a picture.

    The renderer projects world to image; the pipeline projects image back to
    world. Any error in either shows up here as distance from the truth.
    """
    per_camera, _, _, _ = both_cameras

    for camera_id, (results, _) in per_camera.items():
        errors = []
        for result in results:
            truth = world.truth_at(world.LONE_WALKER, result.index)
            if "walker" not in truth:
                continue
            for track in result.tracks:
                if track.position is not None:
                    errors.append(haversine_distance(truth["walker"], track.position.point))

        assert len(errors) > 100, f"{camera_id} placed almost nothing"
        # Measured at 0.32 m median on both cameras. The floor is generous
        # because this is a regression guard, not a specification of accuracy.
        assert statistics.median(errors) < 1.5, (
            f"{camera_id} median error {statistics.median(errors):.2f} m"
        )


def test_reported_uncertainty_actually_covers_the_error(both_cameras):
    """The claim that makes an uncertainty worth printing.

    A radius nobody checks is decoration. If the true position falls outside the
    stated 2-sigma disc far more often than it should, the number is not an
    uncertainty — it is a decoration that invites false confidence.
    """
    per_camera, _, _, _ = both_cameras

    covered = total = 0
    for results, _ in per_camera.values():
        for result in results:
            truth = world.truth_at(world.LONE_WALKER, result.index)
            if "walker" not in truth:
                continue
            for track in result.tracks:
                if track.position is None:
                    continue
                error = haversine_distance(truth["walker"], track.position.point)
                total += 1
                covered += error <= 2.0 * track.position.radius_meters

    assert total > 100
    assert covered / total > 0.8, (
        f"only {covered / total:.0%} of positions fell within their stated 2-sigma"
    )


# ------------------------------------------------------- the central claim


def test_two_cameras_seeing_one_person_produce_one_incident(both_cameras):
    """The claim the system exists to make, through the real pipeline.

    Neither camera knows the other exists. Each decoded its own video, ran its
    own detector, kept its own track identities, and projected onto the ground
    through its own pose. Only their events meet.
    """
    _, events, incidents, _ = both_cameras

    assert len(events) >= 2, "at least one camera raised nothing"
    cameras_that_saw = {e.evidence.camera_id for e in events}
    assert cameras_that_saw == {"cam-07", "cam-08"}, (
        f"only {cameras_that_saw} raised events; the hand-off was never exercised"
    )

    assert len(incidents) == 1, f"{len(incidents)} alerts reached the operator"
    assert incidents[0].distinct_objects == 1, (
        f"reported {incidents[0].distinct_objects} objects for one person"
    )


def test_the_incident_names_both_cameras(both_cameras):
    _, _, incidents, _ = both_cameras

    assert incidents[0].cameras == ("cam-07", "cam-08")
    assert "2 cameras" in incidents[0].summary


def test_corroboration_from_a_second_camera_raises_the_risk(both_cameras):
    # Two cameras agreeing is harder to explain away as one camera's error, and
    # the score must reflect that with a stated reason.
    _, _, incidents, _ = both_cameras

    factors = {factor.name for factor in incidents[0].risk.factors}
    assert "corroboration" in factors


def test_the_association_shows_its_reasoning(both_cameras):
    # An operator must be able to see why two cameras were treated as one
    # object, and disagree with it.
    _, _, incidents, _ = both_cameras
    incident = incidents[0]

    assert incident.associations, "no association was recorded to justify the merge"
    link = incident.associations[0]
    assert {link.a[0], link.b[0]} == {"cam-07", "cam-08"}
    assert link.separation_meters <= link.allowance_meters
    assert link.reasons


def test_the_hand_off_appears_in_the_timeline(both_cameras):
    # One person walking from one camera's view into the other's. The timeline
    # is what an operator reads to understand that this was one journey.
    _, _, incidents, _ = both_cameras
    timeline = incidents[0].timeline()

    assert len({entry.camera_id for entry in timeline}) == 2
    assert [entry.at_millis for entry in timeline] == sorted(
        entry.at_millis for entry in timeline
    )


def test_correlation_reduces_the_operator_s_load(both_cameras):
    _, events, incidents, correlator = both_cameras

    assert len(incidents) < len(events)
    assert correlator.stats.reduction > 0.5


# ------------------------------------------------------------ not over-merging


def test_two_people_are_two_objects_in_one_incident(tmp_path: Path):
    """The opposite error, which a correlator that merges too eagerly would make.

    Two people crossing the same ground at the same time is one incident — it is
    one situation — but it contains two objects, and reporting one would hide
    somebody.
    """
    path = world.write_view(tmp_path / "two.mp4", world.CAMERA_WEST, world.TWO_WALKERS)
    zone = restricted_zone()

    with Pipeline(
        VideoSource(path, source_id="cam-07"),
        MotionDetector(),
        pose=world.CAMERA_WEST,
        zones=[zone],
        rules=[ZoneEntryRule()],
        node_id="nd_site",
    ) as pipeline:
        events = [event for result in pipeline.run() for event in result.events]

    incidents = Correlator().correlate(events)

    assert incidents, "two people crossing a restricted zone raised nothing"
    assert incidents[0].distinct_objects >= 2, (
        f"two people were reported as {incidents[0].distinct_objects} object(s)"
    )


def test_a_camera_that_saw_nothing_contributes_nothing(two_views: dict[str, Path]):
    # A camera pointed away from the action must not be listed on an incident it
    # played no part in, however close it is.
    away = CameraPose(
        position=world.CAMERA_WEST.position,
        mount_height=world.CAMERA_WEST.mount_height,
        heading=200.0,
        pitch=world.CAMERA_WEST.pitch,
        horizontal_fov=world.CAMERA_WEST.horizontal_fov,
        vertical_fov=world.CAMERA_WEST.vertical_fov,
        range_meters=world.CAMERA_WEST.range_meters,
    )
    zone = restricted_zone()

    with Pipeline(
        VideoSource(two_views["cam-07"], source_id="cam-99"),
        MotionDetector(),
        pose=away,
        zones=[zone],
        rules=[ZoneEntryRule()],
        node_id="nd_site",
    ) as pipeline:
        events = [event for result in pipeline.run() for event in result.events]

    # It may still detect movement, but nothing it saw is inside the zone, so it
    # has nothing to say about this incident.
    assert events == []
