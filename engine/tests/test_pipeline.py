"""End-to-end tests: a real video file through decode, detect, track and project.

This is the vertical slice. It runs the actual decoder over an actual encoded
file, the actual background-subtraction detector over the decoded pixels, the
actual Rust tracker over the detections, and the actual ground projection over
the tracks — no mocks anywhere in the path.

**What this does and does not establish.** It establishes that the stages fit
together, that identity persists across frames, that positions land on the map
with honest uncertainty, and that the numbers reported are the numbers measured.
It does not establish that any of it works on real footage, because the scene is
generated. That distinction is recorded in STATUS.md and must not be quietly
dropped.

The thresholds below are floors under measurements, not aspirations. Where the
system currently does something imperfectly, the test asserts the imperfect truth
and names it, rather than asserting what it ought to do and being skipped.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

import scene
from sentinel.core import CameraPose, haversine_distance
from sentinel.decode import VideoSource
from sentinel.detect import UNCLASSIFIED, MotionDetector
from sentinel.pipeline import Pipeline

#: Objects genuinely present in the reference scene.
TRUE_OBJECT_COUNT = len(scene.WALKERS)


@pytest.fixture(scope="module")
def run_over_reference(reference_video: Path, reference_pose: CameraPose):
    """One full pass, kept for the whole module because it takes a few seconds."""
    detector = MotionDetector()
    source = VideoSource(reference_video, source_id="cam-01")

    results = []
    with Pipeline(source, detector, pose=reference_pose) as pipeline:
        results = list(pipeline.run())
        stats = pipeline.stats
    return results, stats


# ---------------------------------------------------------------- it runs at all


def test_every_frame_produces_a_result(run_over_reference):
    results, _ = run_over_reference

    assert len(results) == scene.FRAME_COUNT
    assert [r.index for r in results] == list(range(scene.FRAME_COUNT))
    assert all(r.source_id == "cam-01" for r in results)


def test_results_carry_the_frame_timestamp_not_a_wall_clock(run_over_reference):
    results, _ = run_over_reference

    assert results[0].timestamp_millis == 0
    stamps = [r.timestamp_millis for r in results]
    assert stamps == sorted(stamps)


def test_the_pipeline_finds_the_objects(run_over_reference):
    _, stats = run_over_reference

    assert stats.detections > 300
    assert stats.frames_with_detections > scene.FRAME_COUNT * 0.8


# ------------------------------------------------------------------- identity


def test_tracks_persist_across_many_frames(run_over_reference):
    # The point of tracking. A detector that finds a person in 69% of frames
    # would, without this stage, report a new intruder in each of them.
    _, stats = run_over_reference

    longest = max(stats.observations.values())
    assert longest > scene.FRAME_COUNT * 0.7


def test_the_object_count_is_not_wildly_inflated(run_over_reference):
    # Three people walked past. The specification is explicit that reporting one
    # alert per detection is a failure mode, so the number that matters is how
    # many distinct objects the system believes it saw.
    #
    # It currently reports 5 for 3. That is a real, measured over-count, caused
    # by identity being lost when the detector drops an object for longer than
    # the tracker's gap budget, and it is recorded rather than hidden. The bound
    # here is what keeps it from silently getting worse.
    _, stats = run_over_reference

    assert stats.distinct_objects >= TRUE_OBJECT_COUNT
    assert stats.distinct_objects <= TRUE_OBJECT_COUNT + 3, (
        f"{stats.distinct_objects} tracks for {TRUE_OBJECT_COUNT} objects — "
        "fragmentation has regressed"
    )


def test_track_identities_mostly_stay_with_one_object(run_over_reference):
    """How often a track follows the object it started on.

    Measured by nearest-centre attribution on frames where the walkers are far
    enough apart for the attribution itself to be unambiguous — otherwise this
    would be measuring the metric's confusion rather than the tracker's.
    """
    results, _ = run_over_reference

    assignments: dict[int, list[str]] = defaultdict(list)
    for result in results:
        truth = scene.ground_truth(result.index)
        centres = {
            name: (box[0] + box[2] / 2, box[1] + box[3] / 2) for name, box in truth.items()
        }
        names = list(centres)
        separation = min(
            (
                ((centres[a][0] - centres[b][0]) ** 2 + (centres[a][1] - centres[b][1]) ** 2)
                ** 0.5
                for index, a in enumerate(names)
                for b in names[index + 1 :]
            ),
            default=999.0,
        )
        if separation < 50:
            continue

        for track in result.tracks:
            cx = (track.bbox.x + track.bbox.w / 2) * scene.WIDTH
            cy = (track.bbox.y + track.bbox.h / 2) * scene.HEIGHT
            best, distance = None, 1e9
            for name, (gx, gy) in centres.items():
                d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                if d < distance:
                    distance, best = d, name
            if distance < 50 and best is not None:
                assignments[track.id].append(best)

    observations = sum(len(v) for v in assignments.values())
    switches = sum(
        1 for v in assignments.values() for a, b in zip(v, v[1:]) if a != b
    )

    assert observations > 200, "too few unambiguous observations to judge"
    # Measured at 8 switches over ~410 observations. Appearance-free association
    # cannot do better than this when two people cross; adding an appearance
    # model is the identified next step, not a claim made here.
    assert switches / observations < 0.05, f"{switches} identity switches"


# ---------------------------------------------------------------- the map


def test_tracks_land_on_the_map_with_uncertainty(run_over_reference):
    results, _ = run_over_reference

    placed = [t for r in results for t in r.tracks if t.position is not None]
    assert placed, "nothing was projected onto the ground"

    for track in placed:
        assert track.position.radius_meters > 0.0, "a position without uncertainty is a lie"
        assert track.position.source in ("GROUND_PROJECTION", "CAMERA_FALLBACK")


def test_projected_positions_are_within_the_camera_range(
    run_over_reference, reference_pose: CameraPose
):
    results, _ = run_over_reference

    for result in results:
        for track in result.tracks:
            if track.position is None:
                continue
            distance = haversine_distance(reference_pose.position, track.position.point)
            assert distance <= reference_pose.range_meters


def test_uncertainty_grows_with_distance_from_the_camera(
    run_over_reference, reference_pose: CameraPose
):
    # The property that stops a horizon detection being drawn like one at the
    # camera's feet.
    results, _ = run_over_reference

    samples = [
        (haversine_distance(reference_pose.position, t.position.point), t.position.radius_meters)
        for r in results
        for t in r.tracks
        if t.position is not None
    ]
    assert len(samples) > 100

    # Asserted as the physical relationship rather than as buckets at chosen
    # distances, so the test does not quietly become vacuous when the scene or
    # the pose changes and every sample lands in one bucket.
    distances = [d for d, _ in samples]
    uncertainties = [u for _, u in samples]
    mean_d = sum(distances) / len(distances)
    mean_u = sum(uncertainties) / len(uncertainties)
    covariance = sum(
        (d - mean_d) * (u - mean_u) for d, u in samples
    )
    spread_d = sum((d - mean_d) ** 2 for d in distances) ** 0.5
    spread_u = sum((u - mean_u) ** 2 for u in uncertainties) ** 0.5
    correlation = covariance / (spread_d * spread_u)

    assert correlation > 0.9, f"uncertainty barely tracks distance (r={correlation:.2f})"

    # And it must grow *faster* than distance does. A linear relationship would
    # still let a detection at the far edge be drawn almost as confidently as one
    # nearby, which the geometry does not support.
    ordered = sorted(samples)
    nearest = ordered[: len(ordered) // 4]
    furthest = ordered[-len(ordered) // 4 :]
    distance_ratio = (sum(d for d, _ in furthest) / len(furthest)) / (
        sum(d for d, _ in nearest) / len(nearest)
    )
    uncertainty_ratio = (sum(u for _, u in furthest) / len(furthest)) / (
        sum(u for _, u in nearest) / len(nearest)
    )
    assert uncertainty_ratio > distance_ratio


def test_an_unplaced_camera_still_tracks_but_reports_no_position(reference_video: Path):
    # A camera nobody has placed on the map is still worth running. It simply
    # cannot say where, and says so rather than defaulting to a plausible point.
    with Pipeline(VideoSource(reference_video), MotionDetector(), pose=None) as pipeline:
        results = list(pipeline.run())
        assert pipeline.stats.distinct_objects > 0

    assert all(t.position is None for r in results for t in r.tracks)


# ------------------------------------------------------------------ provenance


def test_every_track_is_attributable(run_over_reference):
    # An AI conclusion without provenance cannot be reviewed. Each track carries
    # when it was first and last seen and how much evidence supports it.
    results, stats = run_over_reference

    for result in results:
        for track in result.tracks:
            assert track.first_seen_millis <= track.last_seen_millis
            assert track.hits >= 1
            assert 0.0 <= track.confidence <= 1.0

    for track_id in stats.track_ids:
        assert stats.observations[track_id] >= 1
        assert stats.duration_millis(track_id) >= 0


def test_nothing_is_classified_when_the_detector_cannot_classify(run_over_reference):
    results, _ = run_over_reference

    for result in results:
        assert all(d.class_id == UNCLASSIFIED for d in result.detections)
        assert all(t.class_id == UNCLASSIFIED for t in result.tracks)


def test_the_pipeline_reports_which_detector_produced_its_results(reference_video: Path):
    detector = MotionDetector()
    with Pipeline(VideoSource(reference_video), detector) as pipeline:
        info = pipeline.detector_info

    assert info.kind == "motion"
    assert info.classifies is False


# ---------------------------------------------------------------- determinism


def test_the_same_video_produces_the_same_result_twice(
    reference_video: Path, reference_pose: CameraPose
):
    # Replaying evidence must reproduce it. Without this an incident review shows
    # something other than what the operator saw.
    def run() -> list[tuple[int, int, float, float]]:
        with Pipeline(
            VideoSource(reference_video), MotionDetector(), pose=reference_pose
        ) as pipeline:
            return [
                (r.index, t.id, round(t.bbox.x, 9), round(t.bbox.y, 9))
                for r in pipeline.run()
                for t in r.tracks
            ]

    assert run() == run()


def test_moving_the_camera_moves_the_map_positions_not_the_identities(
    reference_video: Path, reference_pose: CameraPose
):
    # A PTZ camera that pans does not turn the people it was watching into
    # different people.
    detector = MotionDetector()
    with Pipeline(VideoSource(reference_video), detector, pose=reference_pose) as pipeline:
        before = None
        after = None
        for result in pipeline.run():
            if result.index == 100 and result.tracks:
                before = result.tracks[0]
                pipeline.set_pose(
                    CameraPose(
                        position=reference_pose.position,
                        mount_height=reference_pose.mount_height,
                        heading=90.0,
                        pitch=reference_pose.pitch,
                        horizontal_fov=reference_pose.horizontal_fov,
                        vertical_fov=reference_pose.vertical_fov,
                        range_meters=reference_pose.range_meters,
                    )
                )
            elif before is not None and result.index == 104:
                after = next((t for t in result.tracks if t.id == before.id), None)
                break

    assert before is not None and after is not None, "the track did not survive the pan"
    assert before.position is not None and after.position is not None
    assert haversine_distance(before.position.point, after.position.point) > 5.0
