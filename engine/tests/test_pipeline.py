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

import contextlib
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import onnx_fixture
import scene
from sentinel.core import BoundingBox, CameraPose, Detection, haversine_distance
from sentinel.decode import Frame, SourceInfo, VideoSource
from sentinel.detect import UNCLASSIFIED, DetectorInfo, MotionDetector, OnnxDetector
from sentinel.pipeline import (
    MAX_PLATE_READS_PER_TRACK,
    PLATE_READS_PER_SECOND,
    VEHICLE_LABELS,
    Pipeline,
    TrackPlate,
)
from sentinel.plates import CONFIDENT_AGREEMENT, PlateModels, PlateReader

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


# --------------------------------------------- plates, and only inside vehicles

#: Where the vehicle sits in every frame below: 100x50 pixels of a 400x200
#: frame. Big enough that the reader's own minimum-pixel refusals are not what
#: these tests are measuring.
VEHICLE_BOX = BoundingBox(0.25, 0.5, 0.25, 0.25)
VEHICLE_CROP_SHAPE = (50, 100, 3)
WHOLE_FRAME_SHAPE = (200, 400, 3)


class _StillVehicleSource:
    """A short file with one thing parked in the same place in every frame.

    Synthetic on purpose: what is under test is which boxes reach the plate
    reader, not whether a detector can find a car. Frames are 40 ms apart so a
    track's gap budget can be reasoned about in whole frames.
    """

    is_live = False
    source_id = "cam-plate"
    display_url = "memory:one-vehicle"

    def __init__(self, frames: int = 8):
        self.frames = frames

    def open(self) -> SourceInfo:
        return SourceInfo(
            width=400, height=200, fps=25.0, frame_count=self.frames,
            display_url=self.display_url, is_live=False,
        )

    @property
    def info(self) -> SourceInfo:
        return self.open()

    def __iter__(self):
        for index in range(self.frames):
            image = np.zeros((200, 400, 3), dtype=np.uint8)
            # Only the vehicle is lit, so a crop of it is distinguishable from
            # the frame it came out of by its contents as well as its shape.
            image[100:150, 100:200] = 200
            yield Frame(image, index * 40, index, self.source_id)

    def close(self) -> None:
        pass


class _LabellingDetector:
    """A detector that classifies, and puts one labelled box in the same place.

    ``present_for`` is how many frames it reports the box at all; after that it
    finds nothing, which is how a vehicle leaves.
    """

    def __init__(
        self, label: str = "car", class_id: int = 2, present_for: int | None = None
    ):
        self._class_id = class_id
        self._present_for = present_for
        self._frames = 0
        self.info = DetectorInfo(
            kind="fake", name="one-labelled-box",
            class_names={class_id: label}, classifies=True,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._frames += 1
        if self._present_for is not None and self._frames > self._present_for:
            return []
        return [Detection(bbox=VEHICLE_BOX, confidence=0.9, class_id=self._class_id)]


class _AlwaysAPlate:
    """A plate detector that finds one plate, and records every image it saw.

    The recording is the whole point: it is what proves the reader was handed a
    vehicle crop and never the frame the vehicle was in.
    """

    def __init__(self, box: BoundingBox = BoundingBox(0.1, 0.4, 0.7, 0.4)):
        self._box = box
        self.seen: list[tuple[int, ...]] = []

    def find(self, image: np.ndarray):
        self.seen.append(tuple(image.shape))
        return [(self._box, 0.9)]


class _DictatedText:
    """A recogniser that reads what a test dictated, with dictated confidences."""

    def __init__(self, text: str, confidences: tuple[float, ...] | None = None):
        self.text = text
        self.confidences = (
            confidences if confidences is not None else (0.9,) * len(text)
        )
        self.calls = 0

    def read_text(self, image: np.ndarray):
        self.calls += 1
        return self.text, self.confidences


@pytest.fixture()
def plate_models(tmp_path: Path) -> PlateModels:
    """Three files standing where the operator's plate models would be.

    Nothing reads their contents but the digest and the charset loader, because
    both model calls are injected. They exist so the presence check runs exactly
    as it would in an installation: a fake must never stand in for a model an
    installation is missing.
    """
    detector = tmp_path / "plate-detector.onnx"
    recogniser = tmp_path / "plate-crnn.onnx"
    charset = tmp_path / "charset-uk.txt"
    detector.write_bytes(b"not a model, and never fetched")
    recogniser.write_bytes(b"not a model either")
    charset.write_text(
        "\n".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), encoding="utf-8"
    )
    return PlateModels(detector, recogniser, charset)


def _reader(models: PlateModels, finder, text_reader) -> PlateReader:
    return PlateReader(models, country="UK", box_finder=finder, text_reader=text_reader)


def test_without_a_plate_reader_nothing_changes_and_nothing_is_read(plate_models):
    # Off unless the operator supplied models. A camera watching a footpath must
    # not pay a crop and two model calls per vehicle per frame for a feature
    # aimed at a car park, and it must produce exactly what it produced before.
    def run(reader):
        pipeline = Pipeline(
            _StillVehicleSource(8), _LabellingDetector(), plate_reader=reader
        )
        return list(pipeline.run()), pipeline

    finder = _AlwaysAPlate()
    with_reader, on = run(_reader(plate_models, finder, _DictatedText("AB12CDE")))
    without, off = run(None)

    assert off._plates == {}, "an accumulator was made for a pipeline with no reader"
    assert finder.seen, "the reader that was supplied was never used either"
    assert all(r.plates == () for r in without)
    assert off.stats.plate_reads == 0

    # And the tracking is untouched: reading a plate must not move a box or
    # renumber an object.
    def spine(results):
        return [(r.index, t.id, t.bbox) for r in results for t in r.tracks]

    assert spine(without) == spine(with_reader)
    on.close()
    off.close()


def test_a_vehicle_track_accumulates_across_frames_into_one_confident_reading(
    plate_models,
):
    """One reading per vehicle, built from several frames of its track.

    The failure this is against is the one `plates.py` was written against: a
    single frame's seven characters presented as a plate. The first frame that
    reads anything must not be confident, and the confidence must arrive from
    agreement across frames rather than from any one of them.

    Sixty-four frames rather than eight because reads are spaced by the clock:
    at 25 fps this is two and a half seconds of vehicle, and the reading becomes
    confident inside it.
    """
    finder = _AlwaysAPlate()
    text = _DictatedText("AB12CDE")
    pipeline = Pipeline(
        _StillVehicleSource(64),
        _LabellingDetector(),
        plate_reader=_reader(plate_models, finder, text),
    )
    results = list(pipeline.run())
    accumulators = len(pipeline._plates)
    stats = pipeline.stats
    pipeline.close()

    read = [r for r in results if r.plates]
    print([(r.index, r.plates[0].display, r.plates[0].is_confident) for r in read])

    frames_with_the_vehicle = sum(1 for r in results if r.tracks)
    assert frames_with_the_vehicle == 63, "the synthetic track did not persist"
    # A plate is published on every frame the vehicle is on, including the ones
    # between reads: an operator watching a box must not see the plate under it
    # flicker off three frames in four.
    assert len(read) == frames_with_the_vehicle
    assert accumulators == 1, "one accumulator per track, not one per frame"
    assert stats.plate_reads == text.calls, "a read that never voted was counted"

    first, last = read[0].plates[0], read[-1].plates[0]
    assert isinstance(last, TrackPlate)
    assert first.is_confident is False, "one frame was called a confident plate"
    assert first.reads == 1

    assert last.track_id == results[-1].tracks[0].id
    assert last.display == "AB12CDE"
    assert last.text == "AB12CDE"
    assert last.is_confident is True
    assert last.country == "UK"
    assert last.reads >= CONFIDENT_AGREEMENT
    assert last.agreement >= CONFIDENT_AGREEMENT


def test_the_reader_is_shown_a_vehicle_crop_and_never_the_whole_frame(plate_models):
    # A plate reader pointed at the frame is a plate reader pointed at the
    # street outside the site boundary. Only this stage knows which boxes are
    # vehicles, so this is where that promise is kept or broken.
    finder = _AlwaysAPlate()
    pipeline = Pipeline(
        _StillVehicleSource(8),
        _LabellingDetector(),
        plate_reader=_reader(plate_models, finder, _DictatedText("AB12CDE")),
    )
    list(pipeline.run())
    pipeline.close()

    print(set(finder.seen))
    assert finder.seen, "the plate detector was never called at all"
    assert set(finder.seen) == {VEHICLE_CROP_SHAPE}
    assert WHOLE_FRAME_SHAPE not in finder.seen


def test_a_track_that_is_not_a_vehicle_is_never_read(plate_models):
    # The detector finds a person, in the same place, with the same box. Nothing
    # about it is cropped, so there is no text lifted off a shirt or a sign for
    # an accumulator to vote on.
    finder = _AlwaysAPlate()
    text = _DictatedText("AB12CDE")
    pipeline = Pipeline(
        _StillVehicleSource(8),
        _LabellingDetector(label="person", class_id=0),
        plate_reader=_reader(plate_models, finder, text),
    )
    results = list(pipeline.run())
    accumulators = len(pipeline._plates)
    stats = pipeline.stats
    pipeline.close()

    assert any(r.tracks for r in results), "there was no track to decline to read"
    assert finder.seen == [], "a person was cropped and handed to a plate detector"
    assert text.calls == 0
    assert accumulators == 0
    assert stats.plate_reads == 0
    assert all(r.plates == () for r in results)


def test_the_accumulator_for_an_ended_track_is_dropped(plate_models):
    # One accumulator per live vehicle is bounded by what is on screen. One per
    # vehicle ever seen is the growth the per-track statistics above were
    # trimmed to stop, and each of these holds a track's reads with it, so a
    # gate camera would end the week holding every plate that passed it.
    pipeline = Pipeline(
        _StillVehicleSource(8),
        _LabellingDetector(present_for=5),
        max_gap_millis=100,
        plate_reader=_reader(plate_models, _AlwaysAPlate(), _DictatedText("AB12CDE")),
    )

    held: list[tuple[int, int, tuple[int, ...]]] = []
    for result in pipeline.run():
        held.append((result.index, len(pipeline._plates), result.ended))
    print(held)

    kept_while_present = [count for _, count, _ in held if count]
    ended_on = [index for index, _, ended in held if ended]

    assert kept_while_present == [1] * len(kept_while_present)
    assert kept_while_present, "the vehicle's plate was never accumulated"
    assert ended_on, "the track never ended, so nothing was there to drop"
    # Dropped on the frame the track closed on, not merely by the end of the
    # run: a live camera has no end of run to be tidied up at.
    assert [count for index, count, _ in held if index >= ended_on[0]] == [0] * (
        len(held) - ended_on[0]
    )
    assert pipeline._plates == {}, "the accumulator outlived the track"
    pipeline.close()


def test_an_unresolved_character_reaches_the_operator_as_a_question_mark(plate_models):
    """A character nobody could read stays unread all the way to the result.

    The last character comes back at a confidence below the bar to vote at all,
    so no amount of agreement resolves it. What must survive to the top is the
    ``?``, and the absence of any completed string beside it: `AB12CD?` matched
    or exported as `AB12CDE` is a watchlist hit against somebody else's car.
    """
    weak_last_character = (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2)
    pipeline = Pipeline(
        _StillVehicleSource(40),
        _LabellingDetector(),
        plate_reader=_reader(
            plate_models,
            _AlwaysAPlate(),
            _DictatedText("AB12CDE", weak_last_character),
        ),
    )
    results = list(pipeline.run())
    pipeline.close()

    plate = results[-1].plates[0]
    print(plate)

    assert plate.display == "AB12CD?"
    assert plate.text is None, "a half-read plate was completed into a whole one"
    assert plate.is_confident is False
    assert plate.agreement == 0, "an unresolved character claimed agreement"
    assert plate.reads == 5


# ------------------------------------ what a vehicle that stays in shot costs


#: A plate too blurred for any character to be believed: every read is taken by
#: the accumulator and none of them votes, because each character is under
#: `plates.MIN_CHARACTER_CONFIDENCE`. This is the track that never becomes
#: confident and so never stops being asked — the one the ceiling exists for.
NOTHING_BELIEVABLE = (0.2,) * 7


class _ARecogniserThatBreaks:
    """A model that raises, the way an operator's own ONNX file can."""

    def __init__(self):
        self.calls = 0

    def read_text(self, image: np.ndarray):
        self.calls += 1
        raise RuntimeError("the recogniser fell over on this crop")


@contextlib.contextmanager
def _listening_to_the_pipeline(level: int = logging.WARNING):
    """Everything `pipeline.py` logs, at or above ``level``.

    `caplog` cannot be used: `logs.configure` sets ``propagate=False`` on the
    `sentinel` tree, deliberately, so the handler has to go on the logger that
    actually emits.
    """
    said: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            said.append(record)

    logger = logging.getLogger("sentinel.pipeline")
    handler = Collect()
    handler.setLevel(level)
    was = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(was, level) if was else level)
    try:
        yield said
    finally:
        logger.removeHandler(handler)
        logger.setLevel(was)


def test_a_vehicle_that_stays_in_shot_is_not_read_on_every_frame(plate_models):
    """A parked car costs a fixed amount, not an accumulating one.

    Reading every vehicle on every frame was quadratic, not linear: each read
    was appended to that track's accumulator and every read already in it was
    re-tallied on the next frame. Measured here before this bound existed, one
    stationary vehicle and no-op models: 200 frames cost 0.03s, 1600 cost 1.58s,
    6400 cost 40.3s, and the accumulator held one read per frame throughout. A
    car parked in front of a gate camera for four minutes is the ordinary case.

    Two floors under it, and the second is the one that matters: the reader is
    not called on most frames, and it stops being called at all once the reading
    has cleared the bar `plates.py` sets for acting on a plate. The fortieth
    read of a stationary plate buys nothing the fourth did not.
    """
    frames = 400
    finder = _AlwaysAPlate()
    text = _DictatedText("AB12CDE")
    pipeline = Pipeline(
        _StillVehicleSource(frames),
        _LabellingDetector(),
        plate_reader=_reader(plate_models, finder, text),
    )

    started = time.perf_counter()
    results = list(pipeline.run())
    elapsed = time.perf_counter() - started
    held = [len(accumulator) for accumulator in pipeline._plates.values()]
    print(f"{frames} frames in {elapsed:.3f}s, {text.calls} read(s), held {held}")
    pipeline.close()

    frames_with_the_vehicle = sum(1 for r in results if r.tracks)
    assert frames_with_the_vehicle > frames * 0.9, "the vehicle did not stay in shot"

    # Confident on the fourth read, and never read again: measured at 4 calls
    # over 400 frames, which is 1% of the frames the vehicle was in shot for.
    assert text.calls == CONFIDENT_AGREEMENT, "a confident plate was read again"
    assert len(finder.seen) == text.calls, "a crop was taken and then not read"
    assert held == [CONFIDENT_AGREEMENT]
    assert pipeline.stats.plate_reads == CONFIDENT_AGREEMENT

    # And the plate is still published under the box on every one of those
    # frames. Reading less often must not make the plate flicker.
    published = [r.plates[0] for r in results if r.plates]
    assert len(published) == frames_with_the_vehicle
    assert published[-1].display == "AB12CDE"
    assert published[-1].is_confident is True
    # No wall-clock floor here: 400 frames is short enough that the old
    # per-frame cost (about 0.1s) and this one (0.011s measured) are both fast,
    # and a bound either would pass tests nothing. The call count above is what
    # holds, and the length at which the difference is unmistakable is measured
    # in the next test.


def test_a_plate_that_never_agrees_stops_at_a_ceiling_of_reads(plate_models):
    """The track that never resolves is the one that would grow forever.

    Stopping when a reading is confident bounds nothing here: no character in
    these reads is believed enough to vote, so the reading never becomes
    confident and the vehicle would be read for as long as it sits there. The
    accumulator holds what it is allowed to hold and no more — at 387 bytes a
    read and three reads a second, the alternative is 4 MB an hour for one
    parked car, and a ``resolve()`` over every byte of it on every frame.
    """
    frames = 3200
    finder = _AlwaysAPlate()
    text = _DictatedText("AB12CDE", NOTHING_BELIEVABLE)
    pipeline = Pipeline(
        _StillVehicleSource(frames),
        _LabellingDetector(),
        plate_reader=_reader(plate_models, finder, text),
    )

    # Which frames were read on, taken from the statistic rather than from the
    # fakes, because the spacing is the behaviour under test and the statistic
    # is what an operator would see it through.
    results: list = []
    read_on: list[int] = []
    counted = 0
    started = time.perf_counter()
    for result in pipeline.run():
        results.append(result)
        if pipeline.stats.plate_reads > counted:
            counted = pipeline.stats.plate_reads
            read_on.append(result.index)
    elapsed = time.perf_counter() - started
    held = [len(accumulator) for accumulator in pipeline._plates.values()]
    print(f"{frames} frames in {elapsed:.3f}s, {text.calls} read(s), held {held}")
    print(f"read on {read_on[:8]} ... {read_on[-2:]}")
    pipeline.close()

    frames_with_the_vehicle = sum(1 for r in results if r.tracks)
    assert frames_with_the_vehicle > frames * 0.9, "the vehicle did not stay in shot"

    # Rate-limited first: at the source's 25 fps and three reads a second that
    # is one frame in eight, evenly, and not eight reads in a burst.
    stride = round(25.0 / PLATE_READS_PER_SECOND)
    assert stride > 1, "the source's frame rate no longer needs limiting"
    assert {b - a for a, b in zip(read_on, read_on[1:])} == {stride}, read_on[:12]

    # And then the ceiling, which is reached rather than approached: without it
    # this vehicle would be read once every eight frames for as long as it sat
    # there, which at this length is 400 reads and rising.
    rate_limited = int(frames_with_the_vehicle / stride) + 1
    assert text.calls == MAX_PLATE_READS_PER_TRACK, "the ceiling did not hold"
    assert text.calls < rate_limited / 4, "the ceiling was never reached to be tested"
    assert held == [MAX_PLATE_READS_PER_TRACK]
    assert pipeline.stats.plate_reads == MAX_PLATE_READS_PER_TRACK

    # Nothing resolved, and the last reading is still published every frame:
    # "we are looking and cannot tell" is a different claim from "no plate".
    last = results[-1].plates[0]
    assert last.display == "?" * 7
    assert last.text is None
    assert last.is_confident is False
    # The one wall-clock floor in these tests, and a real discriminator at this
    # length. Measured on this machine, this test: 0.10s as it stands, and
    # 11.64s with the three bounds above removed so that every frame is read and
    # every read re-tallied — the quadratic cost, on one parked car. The ceiling
    # sits between the two and nearer this one, so it fails on a return to
    # per-frame reading even on a machine several times slower than this.
    assert elapsed < 3.0, f"{frames} frames of one parked car took {elapsed:.1f}s"


def test_only_the_reads_that_voted_are_counted(plate_models):
    """A reader lifting nothing but punctuation is not a reader finding plates.

    `PlateAccumulator.add` drops a read whose text normalises to nothing, so
    counting what the reader returned rather than what the accumulator took
    reported votes that were never cast — the precise confusion this statistic
    exists to remove.
    """
    finder = _AlwaysAPlate()
    text = _DictatedText("---")
    reader = _reader(plate_models, finder, text)

    # The reader does return a read: what is under test is a statistic that
    # disagreed with its source, not a reader that found nothing.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    returned = reader.read(frame, VEHICLE_BOX, frame_index=0)
    print(returned)
    assert len(returned) == 1
    assert returned[0].raw_text == "---"
    assert returned[0].text == ""

    pipeline = Pipeline(
        _StillVehicleSource(40), _LabellingDetector(), plate_reader=reader
    )
    results = list(pipeline.run())
    stats = pipeline.stats
    held = [len(accumulator) for accumulator in pipeline._plates.values()]
    pipeline.close()

    assert text.calls > 1, "the recogniser was never asked"
    assert stats.plate_reads == 0, "a read that never voted was counted as one"
    assert held == [0]
    # And nothing is published: an empty reading beside the box would read as
    # "no plate" when the truth is "nothing readable yet".
    assert all(r.plates == () for r in results)


def test_a_detector_that_names_no_vehicle_says_so_once(plate_models):
    """The warning is the only thing that distinguishes this from a quiet car park.

    Models loaded, reader running, not one plate ever read, and nothing said
    why. It is said once, when the pipeline is built, rather than per frame:
    guidance repeated 25 times a second is not guidance.
    """
    with _listening_to_the_pipeline(logging.WARNING) as said:
        pipeline = Pipeline(
            _StillVehicleSource(8),
            _LabellingDetector(label="person", class_id=0),
            plate_reader=_reader(
                plate_models, _AlwaysAPlate(), _DictatedText("AB12CDE")
            ),
        )
        results = list(pipeline.run())
        pipeline.close()

    warnings = [r for r in said if r.levelno == logging.WARNING]
    print([r.getMessage() for r in warnings])

    assert len(warnings) == 1, "the operator was told nothing, or told every frame"
    message = warnings[0].getMessage()
    assert any(label in message for label in VEHICLE_LABELS)
    assert "one-labelled-box" in message, "the detector that cannot was not named"
    assert all(r.plates == () for r in results)


def test_a_plate_model_that_fails_does_not_stop_the_camera(plate_models):
    """A third-party plate model takes the plates down with it, and nothing else.

    The models are files the operator supplied and their failure modes are not
    the detector's. A gate camera whose recogniser throws on one malformed crop
    must not lose its zones, its rules and its recording with it — recording is
    deliberately isolated from an analysis failure for the same reason.

    Logged once for the run, not once per track and never per frame: a model
    that throws on one crop is a model, not a crop.
    """
    frames = 40
    text = _ARecogniserThatBreaks()
    pipeline = Pipeline(
        _StillVehicleSource(frames),
        _LabellingDetector(),
        plate_reader=_reader(plate_models, _AlwaysAPlate(), text),
    )

    with _listening_to_the_pipeline(logging.ERROR) as said:
        results = list(pipeline.run())
    fault = pipeline.plate_fault
    pipeline.close()

    errors = [r for r in said if r.levelno == logging.ERROR]
    print(fault, [r.getMessage() for r in errors])

    # The camera ran to the end of the source and kept tracking throughout.
    assert len(results) == frames
    assert sum(1 for r in results if r.tracks) == frames - 1

    assert text.calls == 1, "a model that had already failed was called again"
    assert all(r.plates == () for r in results)
    assert fault is not None and fault.startswith("RuntimeError:")
    assert len(errors) == 1, "one fault produced no line, or a line per frame"
    assert "the recogniser fell over" in errors[0].getMessage()
