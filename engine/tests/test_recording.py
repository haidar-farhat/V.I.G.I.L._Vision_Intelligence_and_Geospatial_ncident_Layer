"""Tests for continuous recording, retention, and footage as evidence.

Before this, `evidence.py` wrote a SHA-256 manifest for video that did not
exist. These cover the three things that make recorded video *evidence* rather
than merely files:

- **A file loses no frames.** A camera may drop, because it cannot be slowed
  down. A file may not, because a replay that does not reproduce the original
  result is not evidence — and the first version of this dropped 130 frames
  out of 180.
- **Retention never deletes evidence.** Losing the footage of the one thing
  that happened, in order to keep the footage of everything that did not, is
  the failure the whole mechanism exists to prevent.
- **A package says what it does not have.** A package containing forty seconds
  of a ninety-second incident plays, verifies clean, and misleads whoever is
  relying on it.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel import logs
from sentinel.decode import Frame
from sentinel.evidence import coverage_for, export_incident
from sentinel.recording import (
    DEFAULT_CODEC,
    Recorder,
    RecordingError,
    RetentionPolicy,
    Segment,
    apply_retention,
    sha256_of,
)
from sentinel.store import Store


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


def frames(count: int, *, width: int = 160, height: int = 120, start_millis: int = 0,
           interval: int = 66) -> list[Frame]:
    """Frames with content that changes, so a codec cannot collapse them."""
    made = []
    for index in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # A moving block: distinguishable frame to frame, which is what makes a
        # readback assertion capable of failing.
        x = (index * 7) % max(1, width - 20)
        image[40:80, x:x + 20] = 255
        made.append(Frame(image, start_millis + index * interval, index, "cam"))
    return made


def segment_row(path: Path, camera_id: str = "cam", *, start: int, seconds: int = 60,
                size: int = 1_000_000) -> Segment:
    return Segment(
        camera_id=camera_id, path=path,
        started_millis=start, ended_millis=start + seconds * 1000,
        frames=seconds * 15, width=640, height=480,
        nominal_fps=15.0, measured_fps=15.0, codec=DEFAULT_CODEC,
        size_bytes=size, sha256=f"{abs(hash(str(path))):064x}"[:64],
    )


# --------------------------------------------------------------- the recorder


def test_frames_become_a_playable_clip(tmp_path: Path):
    with Recorder("cam", tmp_path, fps=15.0, live=False) as recorder:
        for frame in frames(45):
            recorder.offer(frame)
        segments = recorder.close()

    assert len(segments) == 1
    segment = segments[0]
    assert segment.frames == 45
    assert segment.path.is_file()

    # Written is not the same as readable. A writer that reports success and
    # produces a file nothing can open is the failure worth catching here.
    capture = cv2.VideoCapture(str(segment.path))
    read = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        read += 1
    capture.release()

    assert read == 45


def test_a_file_source_loses_no_frames(tmp_path: Path):
    # The measured failure this argument exists for: replaying a 180-frame clip
    # into a recorder that dropped when busy wrote 50 frames and dropped 130.
    # A file can wait. A file that is not recorded whole is not evidence.
    with Recorder("cam", tmp_path, fps=15.0, live=False, queue_frames=2) as recorder:
        for frame in frames(120):
            recorder.offer(frame)
        recorder.close()

    assert recorder.stats.frames_written == 120
    assert recorder.stats.frames_dropped == 0


def test_a_live_source_drops_rather_than_building_a_backlog(tmp_path: Path):
    # A camera cannot be slowed down, so a full queue drops — and counts it.
    # An unbounded backlog is what kills the process.
    recorder = Recorder("cam", tmp_path, fps=15.0, live=True, queue_frames=1)
    recorder.start()
    try:
        for frame in frames(400):
            recorder.offer(frame)
    finally:
        recorder.close()

    stats = recorder.stats
    assert stats.frames_dropped > 0, "a backlog was allowed to build"
    assert stats.frames_written + stats.frames_dropped == 400
    assert 0.0 < stats.dropped_fraction < 1.0


def test_segments_rotate_on_their_boundary(tmp_path: Path):
    # 300 frames at 66 ms is ~19.8 s; two-second segments make ten of them.
    with Recorder("cam", tmp_path, fps=15.0, live=False, segment_seconds=2.0) as recorder:
        for frame in frames(300):
            recorder.offer(frame)
        segments = recorder.close()

    assert len(segments) >= 9
    assert sum(segment.frames for segment in segments) == 300
    # Contiguous and ordered: a gap between consecutive segments would be
    # footage nobody can account for.
    for earlier, later in zip(segments, segments[1:]):
        assert later.started_millis >= earlier.started_millis
    assert all(segment.duration_millis <= 2500 for segment in segments)


def test_segment_times_are_wall_clock_not_media_time(tmp_path: Path):
    # A file's frames are stamped from the start of the recording. Retention
    # works in days and an operator works in clock time, so a segment called
    # `19700101-000000` is no use to either.
    epoch = int(datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc).timestamp() * 1000)

    with Recorder("cam", tmp_path, fps=15.0, live=False, epoch_millis=epoch) as recorder:
        for frame in frames(30):
            recorder.offer(frame)
        segments = recorder.close()

    assert segments[0].started_millis >= epoch
    assert "20260304-050607" in segments[0].path.name


def test_a_segment_is_hashed_when_it_closes(tmp_path: Path):
    with Recorder("cam", tmp_path, fps=15.0, live=False) as recorder:
        for frame in frames(30):
            recorder.offer(frame)
        segments = recorder.close()

    segment = segments[0]
    assert len(segment.sha256) == 64
    # The hash is of what is actually on the disk, so an evidence package can
    # be checked against it later by somebody who has only the folder.
    assert segment.sha256 == sha256_of(segment.path)


def test_the_measured_rate_is_recorded_beside_the_claimed_one(tmp_path: Path):
    # A live camera does not report a usable rate, so the container header
    # carries an assumption. The index carries the measurement, and that is the
    # one anything downstream should believe.
    with Recorder("cam", tmp_path, fps=30.0, live=False) as recorder:
        for frame in frames(60, interval=100):  # 10 fps in truth
            recorder.offer(frame)
        segments = recorder.close()

    segment = segments[0]
    assert segment.nominal_fps == 30.0
    assert 9.0 < segment.measured_fps < 11.0


def test_an_empty_segment_is_not_left_behind(tmp_path: Path):
    with Recorder("cam", tmp_path, fps=15.0, live=False) as recorder:
        segments = recorder.close()

    assert segments == ()
    assert list(tmp_path.glob("*.mp4")) == []


def test_a_resolution_change_starts_a_new_segment(tmp_path: Path):
    # A reconnect can come back on a different stream profile. Writing frames
    # the container will not accept loses them silently.
    with Recorder("cam", tmp_path, fps=15.0, live=False) as recorder:
        for frame in frames(20, width=160, height=120):
            recorder.offer(frame)
        for frame in frames(20, width=320, height=240, start_millis=2000):
            recorder.offer(frame)
        segments = recorder.close()

    assert len(segments) == 2
    assert (segments[0].width, segments[0].height) == (160, 120)
    assert (segments[1].width, segments[1].height) == (320, 240)


def test_an_impossible_segment_length_is_refused():
    with pytest.raises(RecordingError):
        Recorder("cam", "/tmp", segment_seconds=0)


def test_an_unavailable_codec_says_so_and_does_not_reach_for_one(tmp_path: Path):
    # Nothing is ever downloaded to obtain a codec — which is exactly why H.264
    # is not the default, since asking OpenCV for it prints a download link.
    recorder = Recorder("cam", tmp_path, fps=15.0, live=False, codec="ZZZZ")
    recorder.start()
    for frame in frames(5):
        recorder.offer(frame)
    recorder.close()

    assert recorder.stats.fault is not None
    assert "downloaded" in recorder.stats.fault or "codec" in recorder.stats.fault


def test_indexing_happens_on_the_caller_s_thread(tmp_path: Path):
    # The obvious implementation calls `on_segment` from the writer thread, and
    # the obvious thing to pass is a database write, and SQLite connections
    # belong to the thread that made them. Every segment of the first recording
    # this module made was written and then failed to be indexed.
    import threading

    seen: list[tuple[int, Segment]] = []
    recorder = Recorder(
        "cam", tmp_path, fps=15.0, live=False, segment_seconds=1.0,
        on_segment=lambda s: seen.append((threading.get_ident(), s)),
    )
    caller = threading.get_ident()

    with recorder:
        for frame in frames(60):
            recorder.offer(frame)
            recorder.finished()
        recorder.close()
        recorder.finished()

    assert seen, "nothing was indexed"
    assert all(thread == caller for thread, _ in seen)


# ------------------------------------------------------------------ the index


def test_a_segment_round_trips_through_the_store(tmp_path: Path):
    with Store(tmp_path / "t.db") as store:
        original = segment_row(tmp_path / "a.mp4", start=1_000_000)
        store.save_segment(original)

        back = store.segments()[0]

    assert back.camera_id == original.camera_id
    assert back.started_millis == original.started_millis
    assert back.frames == original.frames
    assert back.sha256 == original.sha256
    assert back.measured_fps == original.measured_fps


def test_the_index_finds_a_segment_that_merely_overlaps(tmp_path: Path):
    # Overlap, not containment. A ten-second incident inside a sixty-second
    # segment is contained by nothing, and asking for containment returns
    # nothing for the commonest case there is.
    with Store(tmp_path / "t.db") as store:
        store.save_segment(segment_row(tmp_path / "a.mp4", start=0, seconds=60))
        store.save_segment(segment_row(tmp_path / "b.mp4", start=60_000, seconds=60))

        inside = store.segments(camera_id="cam", start_millis=20_000, end_millis=30_000)
        across = store.segments(camera_id="cam", start_millis=55_000, end_millis=65_000)
        outside = store.segments(camera_id="cam", start_millis=500_000, end_millis=600_000)

    assert [s.path.name for s in inside] == ["a.mp4"]
    assert [s.path.name for s in across] == ["a.mp4", "b.mp4"]
    assert outside == []


def test_preserving_by_path_or_string_both_work(tmp_path: Path):
    # `str(Path("/rec/a.mp4"))` is `\\rec\\a.mp4` on Windows, so a path written
    # by one call and looked up by another that skipped the normalisation did
    # not match — and failed *silently*, reporting nothing preserved. The next
    # retention pass would then have deleted the evidence.
    with Store(tmp_path / "t.db") as store:
        first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
        store.save_segment(segment_row(first, start=0))
        store.save_segment(segment_row(second, start=60_000))

        assert store.preserve_segments([first]) == 1
        assert store.preserve_segments([str(second)]) == 1
        assert store.preserve_segments([tmp_path / "absent.mp4"]) == 0
        assert store.recorded_bytes(preserved=True) == 2_000_000


def test_reindexing_does_not_unpreserve_evidence(tmp_path: Path):
    # Re-indexing a directory after a crash must replace the metadata and leave
    # the preservation alone: un-preserving is how evidence gets deleted by a
    # process that was only trying to be tidy.
    with Store(tmp_path / "t.db") as store:
        path = tmp_path / "a.mp4"
        store.save_segment(segment_row(path, start=0))
        store.preserve_segments([path])

        store.save_segment(segment_row(path, start=0, size=2_000_000))

        assert store.recording_count() == 1
        assert store.recorded_bytes(preserved=True) == 2_000_000


# ------------------------------------------------------------------ retention


def make_days(store: Store, root: Path, count: int) -> list[Path]:
    """`count` segments, one per day going back from now, on disk and indexed."""
    now = int(time.time() * 1000)
    day = 86_400_000
    paths = []
    for index in range(count):
        path = root / f"seg_{index}.mp4"
        path.write_bytes(b"x" * 1_000_000)
        paths.append(path)
        start = now - (count - index) * day
        store.save_segment(
            Segment("cam", path, start, start + 60_000, 900, 640, 480,
                    15.0, 15.0, DEFAULT_CODEC, 1_000_000, f"{index:064x}")
        )
    return paths


def test_retention_deletes_by_age_oldest_first(tmp_path: Path):
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 10)

        result = apply_retention(
            store, RetentionPolicy(max_age_days=5, min_free_bytes=None)
        )

    assert len(result.deleted) == 5
    assert not any(path.exists() for path in paths[:5])
    assert all(path.exists() for path in paths[5:])


def test_retention_never_deletes_preserved_evidence(tmp_path: Path):
    # The rule the whole mechanism exists for. Losing the footage of the one
    # thing that happened, to keep the footage of everything that did not, is
    # the failure being prevented.
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 10)
        store.preserve_segments([paths[0]])  # the very oldest

        result = apply_retention(
            store, RetentionPolicy(max_age_days=1, min_free_bytes=None)
        )

    assert paths[0].exists(), "evidence was deleted by retention"
    assert result.kept_preserved == 1
    assert paths[0] not in [segment.path for segment in result.deleted]


def test_retention_deletes_by_size_when_age_allows_everything(tmp_path: Path):
    with Store(tmp_path / "t.db") as store:
        make_days(store, tmp_path, 10)

        result = apply_retention(
            store,
            RetentionPolicy(max_age_days=None, max_bytes=6_000_000, min_free_bytes=None),
        )

        assert store.recorded_bytes() <= 6_000_000
    assert len(result.deleted) == 4


def test_a_dry_run_touches_nothing(tmp_path: Path):
    # The first thing anybody should do with a retention policy is find out
    # what it would have eaten.
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 10)

        result = apply_retention(
            store, RetentionPolicy(max_age_days=5, min_free_bytes=None), dry_run=True
        )

        assert store.recording_count() == 10

    assert len(result.deleted) == 5
    assert all(path.exists() for path in paths)


def test_every_deletion_is_audited(tmp_path: Path):
    # What was removed and when is part of the chain of custody. "Deleted by
    # retention on the 17th" is an answer; silence is not.
    with Store(tmp_path / "t.db") as store:
        make_days(store, tmp_path, 4)
        apply_retention(store, RetentionPolicy(max_age_days=1, min_free_bytes=None))

        actions = [row["action"] for row in store.audit_trail()]

    assert actions.count("recording.deleted") == 3


def test_an_indexed_file_that_is_already_gone_is_forgotten(tmp_path: Path):
    # An index entry for a file that does not exist is worse than no entry:
    # evidence will offer it and then fail to copy it.
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 3)
        paths[0].unlink()

        result = apply_retention(
            store, RetentionPolicy(max_age_days=1, min_free_bytes=None)
        )

        assert store.recording_count() == 1

    assert result.already_missing == 1


def test_a_policy_that_cannot_be_met_says_so(tmp_path: Path):
    # The disk will keep filling and somebody has to act, so this is never
    # swallowed.
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 5)
        store.preserve_segments(paths)

        result = apply_retention(
            store,
            RetentionPolicy(max_age_days=None, max_bytes=1_000, min_free_bytes=None),
        )

    assert result.deleted == []
    assert result.shortfall is not None
    assert "preserved" in result.shortfall


def test_an_unbounded_policy_deletes_nothing(tmp_path: Path):
    with Store(tmp_path / "t.db") as store:
        paths = make_days(store, tmp_path, 5)

        result = apply_retention(
            store, RetentionPolicy(max_age_days=None, max_bytes=None, min_free_bytes=None)
        )

    assert result.deleted == []
    assert all(path.exists() for path in paths)


# ------------------------------------------------------- footage as evidence


class FakeIncident:
    """Just enough of an incident for coverage, without running a pipeline."""

    def __init__(self, opened: datetime, duration_seconds: float, cameras=("cam",)):
        self.id = "inc_test"
        self.opened_at = opened
        self.opened_at_millis = 0
        self.closed_at_millis = int(duration_seconds * 1000)
        self.cameras = tuple(cameras)

    @property
    def duration_millis(self) -> int:
        return self.closed_at_millis - self.opened_at_millis


def test_coverage_uses_the_wall_clock_not_media_time(tmp_path: Path):
    # An incident's millis are media time, counted from the start of the
    # footage; recordings are indexed by when they actually happened. Using the
    # wrong clock asked for footage from 1970 and reported, correctly and
    # uselessly, that none existed.
    opened = datetime.now(timezone.utc) - timedelta(minutes=5)
    opened_millis = int(opened.timestamp() * 1000)

    with Store(tmp_path / "t.db") as store:
        store.save_segment(
            segment_row(tmp_path / "a.mp4", start=opened_millis - 5_000, seconds=60)
        )

        coverage = coverage_for(store, FakeIncident(opened, 10.0), lead_seconds=2,
                                trail_seconds=2)

    assert len(coverage[0].segments) == 1
    assert coverage[0].is_complete


def test_a_missing_window_is_reported_as_a_gap(tmp_path: Path):
    # A package containing forty seconds of a ninety-second incident plays,
    # verifies clean, and misleads. Every gap is measured and named.
    opened = datetime.now(timezone.utc) - timedelta(minutes=5)
    opened_millis = int(opened.timestamp() * 1000)

    with Store(tmp_path / "t.db") as store:
        # Covers only the first ten seconds of a sixty-second incident.
        store.save_segment(
            segment_row(tmp_path / "a.mp4", start=opened_millis, seconds=10)
        )

        coverage = coverage_for(store, FakeIncident(opened, 60.0), lead_seconds=0,
                                trail_seconds=0)[0]

    assert not coverage.is_complete
    assert len(coverage.gaps) == 1
    assert 0.1 < coverage.covered_fraction < 0.3


def test_a_camera_that_recorded_nothing_is_still_reported(tmp_path: Path):
    # An empty result would read as "nothing to attach" when the truth is "this
    # camera recorded nothing", which is a finding rather than an absence.
    opened = datetime.now(timezone.utc)

    with Store(tmp_path / "t.db") as store:
        coverage = coverage_for(store, FakeIncident(opened, 10.0, cameras=("north",)))

    assert len(coverage) == 1
    assert coverage[0].camera_id == "north"
    assert coverage[0].segments == ()
    assert coverage[0].covered_fraction == 0.0
    assert coverage[0].gaps


def test_overlapping_segments_do_not_manufacture_a_gap(tmp_path: Path):
    # Two segments overlap across a resolution change, where one is closed and
    # another opened on the same instant. Subtracting them naively has each
    # punch a hole in the other.
    opened = datetime.now(timezone.utc)
    opened_millis = int(opened.timestamp() * 1000)

    with Store(tmp_path / "t.db") as store:
        store.save_segment(segment_row(tmp_path / "a.mp4", start=opened_millis - 5_000,
                                       seconds=30))
        store.save_segment(segment_row(tmp_path / "b.mp4", start=opened_millis + 10_000,
                                       seconds=30))

        coverage = coverage_for(store, FakeIncident(opened, 20.0), lead_seconds=0,
                                trail_seconds=0)[0]

    assert coverage.is_complete
    assert coverage.gaps == ()


# ------------------------------------------- what an adversarial review found
#
# Three defects that survived three independent refuters each. All three shared
# a shape: the code was *documented* as doing the safe thing and did not, so
# nothing looked wrong from the outside.


def test_the_frame_handed_to_the_writer_is_a_real_copy(tmp_path: Path):
    # `np.ascontiguousarray` returns the SAME object for an already-contiguous
    # array, which every `cv2.VideoCapture.read()` frame is — so the docstring's
    # "the image is copied" was false and the queued frame aliased the caller's.
    # A viewer drawing track boxes onto `FrameResult.image` would have baked its
    # overlay into the recorded evidence.
    source = frames(30)

    with Recorder("cam", tmp_path, fps=15.0, live=False) as recorder:
        for frame in source:
            recorder.offer(frame)
            # Exactly what a viewer does to the array it was handed, and what a
            # reused capture buffer does on its own.
            frame.image[:] = 255
        segments = recorder.close()

    capture = cv2.VideoCapture(str(segments[0].path))
    ok, first = capture.read()
    capture.release()

    assert ok
    # The frames offered were mostly black with a small white block. If the
    # writer had seen the caller's mutation, every pixel would be white.
    assert first.mean() < 200, "the caller's mutation reached the recording"


def test_a_second_run_does_not_overwrite_the_first(tmp_path: Path):
    # `cv2.VideoWriter` truncates an existing file, and for a file source every
    # part of a segment's name is deterministic — so re-analysing the same clip
    # into the same directory reproduced the first run's filenames exactly.
    # Worse than losing a recording: `save_segment` upserts on the path and
    # keeps `preserved=1`, so evidence would have had its bytes replaced while
    # the index went on vouching for it.
    epoch = int(datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc).timestamp() * 1000)

    def record_once() -> list[Segment]:
        with Recorder("cam", tmp_path, fps=15.0, live=False, epoch_millis=epoch) as rec:
            for frame in frames(30):
                rec.offer(frame)
            return list(rec.close())

    first = record_once()
    second = record_once()

    assert first[0].path != second[0].path, "the second run reused the first's name"
    assert first[0].path.is_file(), "the first run's evidence was destroyed"
    assert second[0].path.is_file()
    assert len(list(tmp_path.glob("*.mp4"))) == 2


def test_a_dead_writer_is_reported_rather_than_accepted_from(tmp_path: Path):
    # `offer` checked only whether it had ever started a thread, so it went on
    # returning True for a writer that had died — telling the caller a frame was
    # recorded when nothing was going to record it.
    recorder = Recorder("cam", tmp_path, fps=15.0, live=True, codec="ZZZZ")
    recorder.start()
    try:
        for frame in frames(20):
            recorder.offer(frame)
            time.sleep(0.01)

        # The writer is dead by now; every further frame must be refused.
        assert recorder.offer(frames(1)[0]) is False
    finally:
        recorder.close()

    assert recorder.stats.fault is not None
    assert recorder.stats.frames_dropped > 0


def test_a_recording_that_stopped_early_is_surfaced_by_the_pipeline(
    tmp_path: Path, reference_video: Path
):
    # `RecorderStats.fault` documented that "the pipeline surfaces it", and the
    # pipeline did not read the field at all. A writer that died in the first
    # minute of an overnight run ended with the same cheerful summary as a
    # healthy one.
    #
    # The recorder is replaced with one that reports a fault, because what is
    # under test is the *surfacing* — not the many ways a writer can die, which
    # are covered above. `caplog` cannot be used: `logs.configure` sets
    # propagate=False on the `sentinel` tree, deliberately, so the handler goes
    # on the logger that actually emits.
    import logging

    from sentinel import pipeline as pipeline_module
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector
    from sentinel.pipeline import Pipeline

    class FaultingRecorder(Recorder):
        def start(self) -> None:
            super().start()
            with self._lock:
                self._stats.fault = "OSError: the disk went away"

    said: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            said.append(record.getMessage())

    handler = Collect(level=logging.ERROR)
    logger = logging.getLogger("sentinel.pipeline")
    logger.addHandler(handler)
    original = pipeline_module.Recorder
    pipeline_module.Recorder = FaultingRecorder
    try:
        with Pipeline(
            VideoSource(reference_video, source_id="cam"),
            MotionDetector(),
            record_to=tmp_path,
        ) as pipeline:
            for _ in pipeline.run():
                pass
    finally:
        pipeline_module.Recorder = original
        logger.removeHandler(handler)

    assert any("RECORDING STOPPED EARLY" in message for message in said), (
        "a recorder that stopped early ended the run with no error anywhere"
    )
    assert any("the disk went away" in message for message in said), (
        "the reason the writer stopped was not reported"
    )


# ------------------------------------------------- the wiring, not the parts
#
# Recording worked. Coverage worked. Preservation worked. And nothing in the
# shipped code called any of them, so every exported package came out with no
# video and no segment was ever preserved — leaving retention free to delete
# the exact footage an incident depended on. Every part was tested; the wire
# between them was not. These test the wire.


def cli_run(*arguments: str) -> int:
    from sentinel.cli import main

    return main(list(arguments))


@pytest.fixture
def recorded_incident(tmp_path: Path, reference_video: Path):
    """A real run with recording on, leaving an incident and its footage."""
    database = tmp_path / "sentinel.db"
    zone = (
        "Yard:33.893736,35.501800;33.893628,35.501930;"
        "33.893520,35.501800;33.893628,35.501670"
    )
    code = cli_run(
        "-q", "--database", str(database), "run", str(reference_video),
        "--id", "gate",
        "--place", "33.8938,35.5018,6,180,-22,62,36,90",
        "--zone", zone,
        "--record", str(tmp_path / "recordings"),
        "--segment-seconds", "5",
    )
    assert code == 0

    with Store(database) as store:
        rows = store.incidents()
        assert rows, "the reference scene raised no incident to export"
        return database, rows[0]["id"], tmp_path


def test_an_exported_package_actually_contains_the_footage(recorded_incident):
    database, incident_id, tmp_path = recorded_incident
    destination = tmp_path / "export"

    assert cli_run(
        "-q", "--database", str(database), "export", incident_id, "--to", str(destination)
    ) == 0

    package = destination / incident_id
    clips = sorted(package.glob("*.mp4"))

    assert clips, "the package has no video in it"
    assert (package / "footage.json").is_file()
    assert sum(clip.stat().st_size for clip in clips) > 100_000


def test_exporting_preserves_the_footage_from_retention(recorded_incident):
    # The failure this prevents is delayed and invisible: the package is fine,
    # and a retention pass weeks later deletes the originals it came from.
    database, incident_id, tmp_path = recorded_incident

    with Store(database) as store:
        assert store.recorded_bytes(preserved=True) == 0

    cli_run(
        "-q", "--database", str(database), "export", incident_id,
        "--to", str(tmp_path / "export"),
    )

    with Store(database) as store:
        assert store.recorded_bytes(preserved=True) > 0, (
            "exporting an incident left its footage deletable"
        )

        # And the strongest form: a policy that would delete everything does not.
        result = apply_retention(
            store, RetentionPolicy(max_age_days=0, min_free_bytes=None)
        )

    assert result.deleted == [], "retention deleted footage an incident depends on"
    assert result.kept_preserved > 0


def test_preserving_evidence_is_audited(recorded_incident):
    # "Why can this segment not be deleted?" deserves an answer on the record.
    database, incident_id, tmp_path = recorded_incident

    cli_run(
        "-q", "--database", str(database), "export", incident_id,
        "--to", str(tmp_path / "export"),
    )

    with Store(database) as store:
        actions = [row["action"] for row in store.audit_trail()]

    assert "recording.preserved" in actions
    assert "incident.exported" in actions
