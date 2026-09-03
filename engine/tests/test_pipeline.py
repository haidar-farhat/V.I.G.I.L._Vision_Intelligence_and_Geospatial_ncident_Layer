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

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import onnx_fixture
import scene
from sentinel.core import CameraPose, haversine_distance
from sentinel.decode import Frame, SourceInfo, VideoSource
from sentinel.detect import UNCLASSIFIED, MotionDetector, OnnxDetector
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
    # Measured at 4 for 3 people. The bound was +3 while the detector ran at
    # full resolution; detecting at 0.75 scale improved both recall and
    # fragmentation, so the bound tightens with it — a threshold left loose
    # after the thing it measures improved stops being a test.
    assert stats.distinct_objects <= TRUE_OBJECT_COUNT + 2, (
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


# ------------------------------------------------------ the pipeline on a model


def test_the_whole_pipeline_runs_on_a_real_onnx_model(
    tmp_path: Path, reference_video: Path, reference_pose: CameraPose
):
    """Decode, ONNX inference, tracking and projection, with no mocks anywhere.

    The detector is a locally built brightness model rather than trained weights,
    so this says nothing about detection quality. What it does establish is that
    the two detectors are genuinely interchangeable — that everything downstream
    of `detect()` is indifferent to which produced the detections, which is the
    property that lets an operator supply a model without the rest of the system
    changing.
    """
    model = onnx_fixture.build_model(tmp_path / "brightness.onnx")
    detector = OnnxDetector(model, confidence_threshold=0.45)

    with Pipeline(
        VideoSource(reference_video, source_id="cam-onnx"),
        detector,
        pose=reference_pose,
    ) as pipeline:
        results = list(pipeline.run())
        stats = pipeline.stats

    assert len(results) == scene.FRAME_COUNT
    assert stats.detections > 0, "the model produced nothing across the whole clip"
    assert stats.distinct_objects > 0, "detections never became tracks"

    # A classifying detector must label its tracks with the model's own class
    # names, where the motion detector leaves them UNCLASSIFIED.
    labelled = {t.class_id for r in results for t in r.tracks}
    assert labelled and UNCLASSIFIED not in labelled
    assert pipeline.detector_info.classifies is True
    assert pipeline.detector_info.class_names == {0: "bright_region"}


def test_provenance_names_the_exact_weights(tmp_path: Path, reference_video: Path):
    # Months later, with three model versions through the site, "which model said
    # that" has to be answerable from the record alone.
    model = onnx_fixture.build_model(tmp_path / "brightness.onnx")

    with Pipeline(VideoSource(reference_video), OnnxDetector(model)) as pipeline:
        info = pipeline.detector_info

    assert info.model_path == str(model.resolve())
    assert info.model_sha256 is not None and len(info.model_sha256) == 64


# ------------------------------------------------- the spine, end to end


def _restricted_zone(pose: CameraPose):
    """A restricted area on the ground the reference scene's walkers cross."""
    from sentinel.core import destination_point
    from sentinel.zones import Zone, ZoneKind

    centre = destination_point(pose.position, 180.0, 14.0)
    return Zone(
        id="zone-a",
        name="Restricted Area A",
        kind=ZoneKind.RESTRICTED,
        ring=tuple(destination_point(centre, b, 9.0) for b in (0.0, 90.0, 180.0, 270.0)),
        enter_after_millis=600,
    )


@pytest.fixture(scope="module")
def spine(reference_video: Path, reference_pose: CameraPose):
    """The whole chain: video to incidents, on a real file."""
    from datetime import datetime, time, timezone

    from sentinel.events import AfterHoursRule, LoiteringRule, ZoneEntryRule
    from sentinel.zones import Schedule

    from dataclasses import replace

    zone = replace(_restricted_zone(reference_pose), schedule=Schedule(time(18, 0), time(6, 0)))

    # 02:00, so the after-hours schedule is active. Supplied explicitly rather
    # than read from the clock, so the same footage produces the same events on
    # any day — which an evidence trail requires.
    epoch = int(datetime(2026, 8, 30, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
    rules = [ZoneEntryRule(), AfterHoursRule(), LoiteringRule(dwell_millis=4000)]

    with Pipeline(
        VideoSource(reference_video, source_id="cam-07"),
        MotionDetector(),
        pose=reference_pose,
        zones=[zone],
        rules=rules,
        node_id="nd_test",
        wall_clock_epoch_millis=epoch,
    ) as pipeline:
        events = [event for result in pipeline.run() for event in result.events]
        incidents = pipeline.incidents()
        stats = pipeline.stats

    return events, incidents, stats


def test_the_whole_spine_runs_on_real_video(spine):
    events, incidents, stats = spine

    assert stats.presences_started > 0, "nothing ever entered the zone"
    assert events, "presences never became events"
    assert incidents, "events never became an incident"


def test_many_events_collapse_into_one_incident(spine):
    # The measure of this system is how little it says. Twelve seconds of people
    # walking through a restricted area is one thing that happened.
    events, incidents, _ = spine

    assert len(events) > 8
    assert len(incidents) == 1, f"{len(incidents)} incidents reached the operator"


def test_the_incident_carries_its_evidence(spine):
    _, incidents, _ = spine
    incident = incidents[0]

    assert incident.events, "an incident with no events cannot be reviewed"
    assert incident.timeline()
    assert incident.risk.factors, "a risk score with no reasoning is a number to ignore"
    assert incident.zones == ("Restricted Area A",)
    assert incident.cameras == ("cam-07",)

    for event in incident.events:
        assert event.evidence.detector
        assert event.evidence.observations > 0


def test_the_incident_does_not_claim_people(spine):
    # The detector is motion. A blob is not a person.
    _, incidents, _ = spine

    assert "object" in incidents[0].summary
    assert "person" not in incidents[0].summary


def test_correlating_twice_produces_the_same_incident(spine):
    _, incidents, _ = spine
    from sentinel.incidents import Correlator
    from sentinel.zones import ZoneKind

    again = Correlator(zone_kinds={"zone-a": ZoneKind.RESTRICTED}).correlate(
        [event for incident in incidents for event in incident.events]
    )

    assert [i.id for i in again] == [i.id for i in incidents]


def test_the_object_count_is_the_tracker_s_count_not_a_segment_count(spine):
    # It currently reports 5 objects where 3 people walked past. That is the
    # tracker's over-count surfacing at the top of the stack, which is exactly
    # where it hurts most — and it is reported rather than hidden. The bound
    # here stops it silently getting worse.
    _, incidents, _ = spine
    incident = incidents[0]

    assert incident.distinct_objects >= 3
    # Measured at 4. See the note on the same bound above.
    assert incident.distinct_objects <= 5, (
        f"{incident.distinct_objects} objects for 3 people — fragmentation "
        "has regressed"
    )


# ---------------------------------------------- the statistics do not grow forever


def test_per_track_detail_is_bounded_but_the_object_count_is_not(
    reference_video: Path, reference_pose: CameraPose
):
    # These three collections held one entry per track ever seen. A node left
    # running accumulates one per object that ever crossed the frame and
    # releases none of them, so the process grows without limit for as long as
    # it is useful. Trimming them must not cost the one number that matters.
    from sentinel.decode import Frame
    from sentinel.pipeline import _MAX_TRACKED_DETAIL

    import numpy as np

    from sentinel.core import BoundingBox, PositionEstimate, Track

    source = VideoSource(reference_video, source_id="cam-01")
    pipeline = Pipeline(source, MotionDetector(), pose=reference_pose)

    blank = Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp_millis=0,
        index=0,
        source_id="cam-01",
    )

    def synthetic(track_id: int, millis: int) -> Track:
        return Track(
            id=track_id,
            class_id=0,
            bbox=BoundingBox(0.1, 0.1, 0.1, 0.1),
            confidence=0.9,
            hits=1,
            first_seen_millis=millis,
            last_seen_millis=millis,
            position=None,
            speed_mps=None,
            heading_degrees=None,
        )

    total = _MAX_TRACKED_DETAIL * 3
    for track_id in range(1, total + 1):
        pipeline._record(blank, (), (synthetic(track_id, track_id * 40),))

    stats = pipeline.stats
    assert stats.distinct_objects == total, "the object count must survive trimming"
    assert len(stats.track_ids) <= _MAX_TRACKED_DETAIL + 1
    assert len(stats.observations) <= _MAX_TRACKED_DETAIL + 1
    assert len(stats.spans) <= _MAX_TRACKED_DETAIL + 1

    # What is retained is the most recent, and it is retained consistently:
    # `summary()` walks the ids and looks up the other two by them.
    assert total in stats.track_ids
    for track_id in stats.track_ids:
        assert track_id in stats.observations
        assert track_id in stats.spans
    assert str(total) in stats.summary()


def test_a_live_track_is_never_trimmed_out_from_under_itself(
    reference_video: Path, reference_pose: CameraPose
):
    # Trimming the *oldest* ids would drop a long-lived track that is still on
    # screen, and the next frame would then count it as a new object — a
    # stationary loiterer inflating the object count once per frame forever.
    from sentinel.decode import Frame
    from sentinel.pipeline import _MAX_TRACKED_DETAIL

    import numpy as np

    from sentinel.core import BoundingBox, Track

    source = VideoSource(reference_video, source_id="cam-01")
    pipeline = Pipeline(source, MotionDetector(), pose=reference_pose)
    blank = Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0, 0, "cam-01")

    def synthetic(track_id: int, millis: int) -> Track:
        return Track(track_id, 0, BoundingBox(0.1, 0.1, 0.1, 0.1), 0.9, 1,
                     0, millis, None, None, None)

    loiterer = 1
    for track_id in range(2, _MAX_TRACKED_DETAIL * 2 + 2):
        pipeline._record(
            blank, (), (synthetic(loiterer, track_id * 40), synthetic(track_id, track_id * 40))
        )

    stats = pipeline.stats
    assert loiterer in stats.track_ids, "a track still on screen was forgotten"
    assert stats.observations[loiterer] == _MAX_TRACKED_DETAIL * 2
    # The loiterer plus one new object per frame, each counted exactly once.
    assert stats.distinct_objects == _MAX_TRACKED_DETAIL * 2 + 1


# ------------------------------------------- a camera does not end, it stops


class _FlakyLiveSource:
    """A live camera: paced frames, one failed read in the middle, no end.

    `VideoSource.read` returns ``None`` both at the end of a file *and* on a
    single failed read, and the pipeline used to iterate a live source exactly
    the way it iterated a file — so one dropped frame ended the run, and the log
    reported "analysis finished", which is what a file does when it runs out.
    A security camera that stops watching must never be reported the same way as
    a file that finished.

    Paced deliberately: `LiveStream` keeps only the newest frame and drops the
    rest, so a source that returns instantly would have most of its frames
    discarded and the test would be measuring the queue rather than the
    reconnect.
    """

    is_live = True
    source_id = "flaky"
    display_url = "device:test"

    def __init__(self, fail_at: int | None = 5, quiet_after: int | None = None):
        self._fail_at = fail_at
        self._quiet_after = quiet_after
        self._index = 0
        self._failed = False
        self.opens = 0

    def open(self):
        self.opens += 1
        return SourceInfo(
            width=64, height=48, fps=15.0, frame_count=None, is_live=True,
            display_url=self.display_url,
        )

    @property
    def info(self):
        return self.open()

    def read(self):
        if self._index == self._fail_at and not self._failed:
            self._failed = True
            return None  # the glitch, indistinguishable from an ending
        if self._quiet_after is not None and self._index >= self._quiet_after:
            # A camera that has gone quiet without dying: it neither returns a
            # frame nor reports an error, which is the case that makes shutdown
            # hard.
            time.sleep(0.05)
            return None
        time.sleep(0.02)
        frame = Frame(
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            timestamp_millis=int(time.time() * 1000),
            index=self._index,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self):
        pass


def test_a_live_camera_survives_a_dropped_frame():
    source = _FlakyLiveSource(fail_at=5)
    pipeline = Pipeline(source, MotionDetector())

    seen = 0
    for _ in pipeline.run():
        seen += 1
        if seen >= 10:
            pipeline.ask_to_stop()
    pipeline.close()

    # Before the fix the run ended at the glitch, having called itself finished.
    assert seen >= 10, f"the run ended after {seen} frame(s)"
    assert source.opens >= 2, "the stream was never reconnected"


def test_a_live_run_stops_when_asked_even_while_the_camera_is_silent():
    # The stop flag alone is only checked between frames, and a silent camera
    # delivers no frames to be between. A shutdown that waits for the next frame
    # from a camera that has stopped sending is a shutdown that never happens —
    # and the console signals every camera and then waits.
    source = _FlakyLiveSource(fail_at=None, quiet_after=2)
    pipeline = Pipeline(source, MotionDetector())

    results = pipeline.run()
    next(results)

    pipeline.ask_to_stop()
    started = time.perf_counter()
    with pytest.raises(StopIteration):
        next(results)
    elapsed = time.perf_counter() - started
    pipeline.close()

    assert elapsed < 5.0, f"shutdown took {elapsed:.1f}s waiting on a dead camera"
