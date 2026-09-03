"""Continuous recording.

Until this module existed, `evidence.py` wrote a SHA-256 manifest for video that
did not exist. A security system that does not record is not a security system:
the incident is the index, and the footage is the thing anybody actually wants
to look at.

**Segments, not files.** Each camera writes a sequence of bounded clips rather
than one growing file. Three reasons, all of them operational:

- Retention can delete a whole unit. Truncating a file in place is not something
  a container format survives.
- An evidence package copies the segments that overlap the incident, and nothing
  else — no re-encode, no seeking, no partial reads.
- **A power cut loses at most one segment.** OpenCV writes an MP4
  progressively and the container index is finalised on close, so a file killed
  mid-write may not play at all. Sixty seconds is the default because sixty
  seconds is what that costs.

**The pre-event problem is solved by recording continuously.** An intrusion
event fires *after* somebody is already inside the zone, so the useful footage
starts before the trigger. With continuous recording the earlier segments are
simply already on disk, and export asks for a window that starts before the
incident did. That is why continuous recording came first and event-triggered
recording did not: the simpler mechanism is also the one that gets the evidence
right.

**Why not H.264.** Because OpenCV's H.264 encoder is not present, and what it
does when asked for it is print the name of a DLL, a release page to download it
from, and fail:

    Failed to load OpenH264 library: openh264-2.5.0-win64.dll
    Please check environment and/or download library: <a release page>

That is the zero-WAN guarantee being broken by a dependency, which §132 forbids
outright, so the encoder is refused rather than the guarantee. (The URL is
deliberately not written here. The offline audit refused this file when it was,
which is the guard working: "it is only in a docstring" is exactly how a
destination gets into shipped source.) `mp4v` — MPEG-4
Part 2 — ships inside the OpenCV wheel, needs nothing, plays in any player an
operator already has, and was measured at the same size as XVID and a quarter
the size of MJPG on real content. It costs roughly **4× the bytes H.264 would**.
That is the honest price of the promise, and it is written down in
`docs/USAGE.md` next to the storage numbers rather than discovered when a disk
fills.

**Measured, on 640×480 at 15 fps:** ~12.7 MiB/minute, **~17.5 GB per day per
camera**, encoding at ~800 fps — so the writer is not what limits throughput
(the detector, at ~433 fps, is). Sixteen cameras is roughly 280 GB/day, which is
the number retention exists to bound.
"""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .decode import Frame
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: MPEG-4 Part 2, in an .mp4 container. See the module docstring for why this
#: and not H.264.
DEFAULT_CODEC = "mp4v"
CONTAINER = ".mp4"

#: How long one segment covers. The upper bound on what a power cut costs.
DEFAULT_SEGMENT_SECONDS = 60.0

#: Frames held between the decoder and the writer. Two seconds at 15 fps, which
#: absorbs a slow disk without letting a stalled one grow into the memory that
#: kills the process.
DEFAULT_QUEUE_FRAMES = 30

#: Assumed rate when a source will not say. A live camera reports nothing
#: useful, and the container header has to carry *some* number — so the index
#: records the rate that was actually measured, and that is the one anything
#: downstream should believe.
FALLBACK_FPS = 15.0


class RecordingError(RuntimeError):
    """A segment could not be opened or written."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One recorded clip, and what is true of it.

    ``measured_fps`` exists because the container header cannot be trusted for a
    live source: the rate has to be chosen when the file is opened and the
    camera only reveals its real rate by delivering frames. The header carries
    the assumption; this carries the measurement, and evidence quotes this one.
    """

    camera_id: str
    path: Path
    started_millis: int
    ended_millis: int
    frames: int
    width: int
    height: int
    #: What the container header claims.
    nominal_fps: float
    #: Frame intervals divided into the wall-clock span the frames actually
    #: covered — first frame to last frame, never derived from the nominal rate.
    #: A one-frame segment has nothing to measure and carries the nominal value.
    measured_fps: float
    codec: str
    size_bytes: int
    sha256: str
    #: False when the writer was closed by a failure rather than by rotation, so
    #: the clip may be short or unplayable. Recorded rather than hidden: a gap an
    #: operator knows about is a different thing from one they do not.
    complete: bool = True

    @property
    def duration_millis(self) -> int:
        return self.ended_millis - self.started_millis

    def overlaps(self, start_millis: int, end_millis: int) -> bool:
        """Whether this segment covers any part of a window."""
        return self.started_millis <= end_millis and self.ended_millis >= start_millis


@dataclass
class RecorderStats:
    """What the recorder actually did, as opposed to what it was asked to do."""

    frames_written: int = 0
    #: Frames the writer could not keep up with. Counted rather than swallowed:
    #: a recorder silently dropping half its input is the worst kind of failure,
    #: because the footage looks fine until the moment it matters.
    frames_dropped: int = 0
    segments_written: int = 0
    bytes_written: int = 0
    #: Segments closed by a failure rather than by rotation.
    segments_incomplete: int = 0
    #: Set when the writer thread has stopped for a reason other than being asked
    #: to. The pipeline surfaces it, because "recording" that is not recording
    #: must not look like recording.
    fault: str | None = None

    @property
    def dropped_fraction(self) -> float:
        total = self.frames_written + self.frames_dropped
        return self.frames_dropped / total if total else 0.0


def _unique_path(path: Path) -> Path:
    """A path nothing is already using.

    `cv2.VideoWriter` truncates an existing file — verified — and for a file
    source every part of a segment's name is deterministic: the epoch comes
    from the source's modification time, and the frame timestamps and indices
    replay identically. So analysing the same clip twice into the same
    directory reproduced the first run's filenames exactly and overwrote it.

    That was worse than losing a recording. `save_segment` upserts on the path
    and deliberately preserves `preserved=1`, so a segment held as incident
    evidence would have had its bytes replaced while the index went on
    vouching for it with a freshly computed hash — and the SHA-256 already
    quoted in an exported `footage.json` would match nothing at all.

    Re-analysing footage is a legitimate thing to do, so the second run gets a
    suffix rather than a refusal, and both recordings survive.
    """
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            _log.info(
                "%s already exists; writing %s instead", path.name, candidate.name
            )
            return candidate
        index += 1


def sha256_of(path: Path) -> str:
    """Streamed, because a segment is tens of megabytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Recorder:
    """Writes one camera's frames to segmented clips, on its own thread.

    ``offer`` never blocks. A disk that stalls must not stop the analytic, and
    an analytic that stalls must not stop the disk — so the two are separated by
    a bounded queue and the drops are counted.

    **The limit of that separation, stated plainly.** This build pulls frames
    through one loop: decode, then analyse, then offer. So the writer surviving
    a slow analytic is real, and recording surviving a *dead* decode loop is not
    — there would be nothing to record. Genuine independence needs the decode
    thread to fan out to a recorder and an analytic queue separately, which is
    the headless daemon's job (ROADMAP 1.2) and not this module's.
    """

    __slots__ = (
        "_camera_id", "_directory", "_codec", "_segment_millis", "_nominal_fps",
        "_queue", "_thread", "_stop", "_stats", "_lock", "_segments", "_on_segment",
        "_writer", "_open_path", "_open_started", "_open_frames", "_open_first_millis",
        "_open_last_millis", "_size", "_live", "_epoch_millis", "_finished",
    )

    def __init__(
        self,
        camera_id: str,
        directory: str | Path,
        *,
        fps: float | None = None,
        live: bool = True,
        epoch_millis: int = 0,
        segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
        codec: str = DEFAULT_CODEC,
        queue_frames: int = DEFAULT_QUEUE_FRAMES,
        on_segment=None,
    ):
        """
        ``fps`` is what the container header will claim. Pass the source's
        reported rate when it has one; a live camera does not, and the fallback
        is used with the measured rate recorded per segment.

        ``live`` decides what happens when the writer falls behind, and it is
        the most consequential argument here. **A live camera cannot be slowed
        down**, so a full queue drops the frame and counts it: a backlog is
        worse than a gap, because an operator needs the present. **A file can
        wait, and must** - a file is evidence, every frame is processed in order
        so that a replay reproduces the original result, and a recording that
        silently discarded three frames in four would not be evidence of
        anything. Measured before this argument existed: replaying a 180-frame
        clip recorded 50 frames and dropped 130.

        ``epoch_millis`` is when media time zero happened. A file's frames are
        stamped from the start of the recording, while retention works in days
        and evidence works in incident times - both need the wall clock. The
        pipeline resolves it; a live source needs nothing, because its frames
        already carry it.

        ``on_segment`` is called with each finished :class:`Segment` — **on the
        caller's thread, from** :meth:`finished`, never on the writer's. That is
        not a detail: the obvious implementation calls it from the writer, and
        the obvious thing to pass is a database write, and SQLite connections
        belong to the thread that made them. Every segment of the first
        recording this module ever made was written to disk and then failed to
        be indexed, with the failure visible only in the log.
        """
        if segment_seconds <= 0:
            raise RecordingError("segment_seconds must be greater than zero")

        self._camera_id = camera_id
        self._directory = Path(directory)
        self._codec = codec
        self._segment_millis = int(segment_seconds * 1000)
        self._nominal_fps = float(fps) if fps and fps > 0 else FALLBACK_FPS
        self._live = live
        self._epoch_millis = int(epoch_millis)
        self._on_segment = on_segment

        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=max(1, queue_frames))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stats = RecorderStats()
        self._segments: list[Segment] = []
        # Finished segments waiting to be handed to `on_segment` on a thread
        # that is allowed to touch a database.
        self._finished: queue.SimpleQueue[Segment] = queue.SimpleQueue()

        self._writer: cv2.VideoWriter | None = None
        self._open_path: Path | None = None
        self._open_started = 0
        self._open_frames = 0
        self._open_first_millis = 0
        self._open_last_millis = 0
        self._size = (0, 0)

    # ------------------------------------------------------------------ state

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def stats(self) -> RecorderStats:
        with self._lock:
            return RecorderStats(**vars(self._stats))

    @property
    def segments(self) -> tuple[Segment, ...]:
        with self._lock:
            return tuple(self._segments)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def finished(self) -> list[Segment]:
        """Segments closed since the last call, and notify ``on_segment``.

        Called by whoever owns the recorder, on their own thread. Draining it
        regularly is what makes an interrupted run still leave findable
        footage — the alternative, indexing everything at the end, loses the
        index precisely when the run ended badly.
        """
        drained: list[Segment] = []
        while True:
            try:
                drained.append(self._finished.get_nowait())
            except queue.Empty:
                break

        for segment in drained:
            if self._on_segment is None:
                continue
            try:
                self._on_segment(segment)
            except Exception:  # noqa: BLE001
                # A failure to index must not destroy the segment. The file is
                # on the disk and is still evidence; it is merely harder to find.
                _log.error(
                    "%s: could not index %s", self._camera_id, segment.path.name,
                    exc_info=True,
                )
        return drained

    # ---------------------------------------------------------------- control

    def __enter__(self) -> "Recorder":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"record:{self._camera_id}", daemon=True
        )
        self._thread.start()
        _log.info(
            "%s: recording to %s (%s, %.0fs segments)",
            self._camera_id, self._directory, self._codec,
            self._segment_millis / 1000,
        )

    def offer(self, frame: Frame) -> bool:
        """Hand a frame to the writer. Returns whether it was taken.

        For a **live** source this never blocks: a camera cannot be slowed down,
        so a full queue drops the frame and counts it. For a **file** it waits,
        because a file is evidence and a recording missing three frames in four
        is not evidence of anything.

        **The image is copied, with `.copy()` and not with
        `np.ascontiguousarray`.** That distinction was a real defect: a frame
        from `cv2.VideoCapture.read()` is already C-contiguous, so
        `ascontiguousarray` returns *the same object* — verified — and the
        queued frame aliased the array the caller went on using. Two ways that
        loses evidence. A viewer draws track boxes onto `FrameResult.image`,
        which is the same array, so the overlay gets baked into the recording;
        and `decode.py` warns that a live capture may reuse its buffer, which
        would make every queued frame mutate into a later moment while its
        timestamp, hash and index all went on describing the earlier one.

        The copy costs about 0.9 MB per frame at 640×480 — a memcpy, far below
        the encode it feeds.
        """
        if self._thread is None or self._stop.is_set():
            return False

        copied = Frame(
            image=frame.image.copy(),
            timestamp_millis=frame.timestamp_millis,
            index=frame.index,
            source_id=frame.source_id,
        )

        if not self._live:
            # Bounded waiting, never infinite. A writer that has died must not
            # deadlock the caller into waiting for it forever.
            while not self._stop.is_set():
                try:
                    self._queue.put(copied, timeout=1.0)
                    return True
                except queue.Full:
                    if not self.is_running:
                        _log.error(
                            "%s: the writer has stopped; the rest of this file "
                            "will not be recorded", self._camera_id,
                        )
                        return False
            return False

        if not self.is_running:
            # The writer is gone, so this frame is not going to be recorded.
            # Reporting that as a drop is the truth; returning True would have
            # the caller believe it was written.
            with self._lock:
                self._stats.frames_dropped += 1
            return False

        try:
            self._queue.put_nowait(copied)
            return True
        except queue.Full:
            with self._lock:
                self._stats.frames_dropped += 1
                dropped = self._stats.frames_dropped
            if dropped in (1, 100) or dropped % 1000 == 0:
                # Logged on a curve rather than every frame: a stalled disk
                # would otherwise produce a log line per frame and fill the disk
                # it is complaining about.
                _log.warning(
                    "%s: the writer is behind — %d frame(s) dropped", self._camera_id, dropped
                )
            return False

    def close(self) -> tuple[Segment, ...]:
        """Finish the open segment and stop. Returns everything written."""
        if self._thread is None:
            return self.segments

        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        self._thread.join(timeout=15.0)
        if self._thread.is_alive():
            # Reported, not ignored. A writer that will not stop is holding a
            # file open, and the segment it holds is the one being lost.
            _log.error("%s: the writer did not stop within 15s", self._camera_id)
        self._thread = None
        return self.segments

    # ----------------------------------------------------------------- writing

    def _run(self) -> None:
        # A segment closed because the loop ended is complete; one closed
        # because the loop *failed* is not, and the difference is the whole
        # value of the flag.
        ended_cleanly = False
        try:
            while True:
                try:
                    frame = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    # A rotation boundary must be honoured even when no frames
                    # are arriving, or a camera that goes quiet leaves its last
                    # segment open and unfinalised for as long as it stays quiet.
                    self._rotate_if_due(int(time.time() * 1000))
                    continue

                if frame is None:
                    break
                self._write(frame)
            ended_cleanly = True
        except Exception as error:  # noqa: BLE001
            with self._lock:
                self._stats.fault = f"{type(error).__name__}: {error}"
            _log.error("%s: recording stopped", self._camera_id, exc_info=True)
        finally:
            self._finalise(complete=ended_cleanly)

    def _write(self, frame: Frame) -> None:
        height, width = frame.image.shape[:2]

        if self._writer is not None and (width, height) != self._size:
            # A stream that changes resolution mid-flight is a real thing —
            # a reconnect can come back on a different profile. The segment is
            # closed and a new one opened rather than writing frames the
            # container will not accept.
            _log.info(
                "%s: resolution changed %dx%d -> %dx%d; starting a new segment",
                self._camera_id, self._size[0], self._size[1], width, height,
            )
            self._finalise(complete=True)

        if self._writer is None:
            self._open(frame, width, height)

        self._rotate_if_due(self._epoch_millis + frame.timestamp_millis)
        if self._writer is None:
            self._open(frame, width, height)

        assert self._writer is not None
        self._writer.write(frame.image)
        self._open_frames += 1
        # The last frame's wall-clock time is what truthfully ends the segment.
        self._open_last_millis = self._epoch_millis + frame.timestamp_millis
        with self._lock:
            self._stats.frames_written += 1

    def _open(self, frame: Frame, width: int, height: int) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        # Sortable, unambiguous, and it carries the camera — so a folder of
        # segments from several cameras is still readable by a person.
        # Wall clock, not media time. A file's frames are stamped from the start
        # of the recording, and a segment called `19700101-000000` tells an
        # operator nothing about when it happened.
        wall = self._epoch_millis + frame.timestamp_millis
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(wall / 1000))
        path = _unique_path(
            self._directory / f"{self._camera_id}_{stamp}_{frame.index:08d}{CONTAINER}"
        )

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*self._codec),
            self._nominal_fps, (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RecordingError(
                f"Could not open {path} for writing with codec {self._codec!r}. "
                "The codec may not be available in this OpenCV build. Nothing "
                "is ever downloaded to obtain one."
            )

        self._writer = writer
        self._open_path = path
        self._open_started = wall
        self._open_first_millis = wall
        self._open_last_millis = wall
        self._open_frames = 0
        self._size = (width, height)

    def _rotate_if_due(self, now_millis: int) -> None:
        if self._writer is None:
            return
        if now_millis - self._open_started >= self._segment_millis:
            self._finalise(complete=True)

    def _finalise(self, *, complete: bool) -> None:
        """Close the open segment, hash it, and record what it is."""
        if self._writer is None:
            return

        writer, path = self._writer, self._open_path
        frames, started = self._open_frames, self._open_started
        width, height = self._size

        self._writer = None
        self._open_path = None

        writer.release()

        if path is None:
            return
        if frames == 0 or not path.exists():
            # An empty segment is not evidence of anything and would only be a
            # file somebody has to explain.
            path.unlink(missing_ok=True)
            return

        # From the frames themselves, never from the nominal rate. An earlier
        # version computed `ended` as `first + (frames-1)/nominal`, which made
        # `measured_fps` echo the assumption it exists to check — the test that
        # fed it 10 fps footage under a 30 fps header read back "30 measured".
        # It also falsified `ended_millis` in the index, and the coverage maths
        # for evidence believes `ended_millis`: a wrong one fabricates footage
        # coverage, or fabricates a gap, depending on which way it is wrong.
        ended = max(started, self._open_last_millis)
        span_millis = max(1, ended - started)
        size = path.stat().st_size

        # N frames span N-1 intervals. A one-frame segment measures nothing,
        # and says so by carrying the assumption rather than an invented rate.
        measured = (
            (frames - 1) / (span_millis / 1000.0) if frames > 1 and span_millis > 0
            else self._nominal_fps
        )

        segment = Segment(
            camera_id=self._camera_id,
            path=path,
            started_millis=started,
            ended_millis=ended,
            frames=frames,
            width=width,
            height=height,
            nominal_fps=self._nominal_fps,
            measured_fps=measured,
            codec=self._codec,
            size_bytes=size,
            sha256=sha256_of(path),
            complete=complete,
        )

        with self._lock:
            self._segments.append(segment)
            self._stats.segments_written += 1
            self._stats.bytes_written += size
            if not complete:
                self._stats.segments_incomplete += 1

        _log.info(
            "%s: wrote %s — %d frames, %.1f MiB, %.1f fps measured%s",
            self._camera_id, path.name, frames, size / 1024 / 1024,
            segment.measured_fps, "" if complete else " (INCOMPLETE)",
        )

        # Handed over rather than called: see `finished`.
        self._finished.put(segment)


# ------------------------------------------------------------------- retention


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How much recorded video to keep.

    Two independent bounds, because they fail differently. **Age** is what an
    operator reasons about and what a policy or a regulator states. **Size** is
    what the disk actually enforces, and it is the one that matters at three in
    the morning when a camera has spent a week staring at a swaying tree.

    Any of them may be ``None``, meaning unbounded. All three being ``None`` is
    legal and means "keep everything" — a real choice on a machine with a large
    disk, and one a caller has to make explicitly rather than acquire by
    omission.
    """

    max_age_days: float | None = 14.0
    max_bytes: int | None = None
    #: Stop deleting once this much of the volume is free. A disk at 100% does
    #: not merely stop recording — SQLite cannot write either, so the events and
    #: the audit log stop with it. Keeping that from happening is the job.
    min_free_bytes: int | None = 5 * 1024 * 1024 * 1024

    def describe(self) -> str:
        parts = []
        if self.max_age_days is not None:
            parts.append(f"{self.max_age_days:g} days")
        if self.max_bytes is not None:
            parts.append(f"{self.max_bytes / 1024**3:.0f} GiB stored")
        if self.min_free_bytes is not None:
            parts.append(f"{self.min_free_bytes / 1024**3:.0f} GiB free")
        return ", ".join(parts) if parts else "unbounded"


@dataclass
class RetentionResult:
    """What a retention pass did, and what it could not do."""

    deleted: list[Segment] = field(default_factory=list)
    freed_bytes: int = 0
    #: Segments an incident depends on. Skipped however old they are.
    kept_preserved: int = 0
    #: Files the index knew about that were already gone. Their rows are
    #: dropped: an index entry for a file that does not exist is worse than no
    #: entry, because evidence will offer it and then fail to copy it.
    already_missing: int = 0
    #: Files that could not be deleted — in use, or a permission problem. The
    #: index row is kept, because the file is still on the disk and forgetting
    #: it would leave video nothing will ever clean up.
    failed: list[Path] = field(default_factory=list)
    #: Set when the policy could not be met. The disk will keep filling and
    #: somebody has to act, so this is never swallowed.
    shortfall: str | None = None

    @property
    def freed_gib(self) -> float:
        return self.freed_bytes / 1024**3


def apply_retention(
    store,
    policy: RetentionPolicy,
    *,
    actor: str = "retention",
    dry_run: bool = False,
) -> RetentionResult:
    """Delete recorded video the policy no longer covers. Oldest first.

    **A preserved segment is never deleted**, however old it is and however full
    the disk is. Losing the footage of the one thing that happened in order to
    keep the footage of everything that did not is the failure this mechanism
    exists to prevent — so when the only way to satisfy the policy would be to
    delete evidence, the policy goes unmet and says so loudly.

    Every deletion is audited. What was removed and when is part of the chain of
    custody: an operator who asks "where is the footage from the 3rd" deserves
    "deleted by retention on the 17th", not silence.

    ``dry_run`` reports what would go without touching anything, because the
    first thing anybody should do with a retention policy is find out what it
    would have eaten.
    """
    import shutil

    result = RetentionResult()
    now_millis = int(time.time() * 1000)

    everything = store.segments(limit=1_000_000)
    preserved = store.preserved_paths()
    candidates = [
        segment for segment in everything
        if store.segment_key(segment.path) not in preserved
    ]
    result.kept_preserved = len(everything) - len(candidates)
    candidates.sort(key=lambda segment: segment.started_millis)

    total_bytes = store.recorded_bytes()
    free_bytes = _free_bytes(shutil, candidates)

    def over_budget() -> bool:
        if policy.max_bytes is not None and total_bytes > policy.max_bytes:
            return True
        if policy.min_free_bytes is not None and free_bytes < policy.min_free_bytes:
            return True
        return False

    for segment in candidates:
        too_old = (
            policy.max_age_days is not None
            and now_millis - segment.ended_millis > policy.max_age_days * 86_400_000
        )
        # Sorted oldest first, so once the oldest remaining segment is neither
        # too old nor needed for space, nothing after it is either.
        if not too_old and not over_budget():
            break

        if dry_run:
            result.deleted.append(segment)
            result.freed_bytes += segment.size_bytes
            total_bytes -= segment.size_bytes
            free_bytes += segment.size_bytes
            continue

        if not segment.path.exists():
            store.forget_segment(segment.path)
            result.already_missing += 1
            total_bytes -= segment.size_bytes
            continue

        try:
            segment.path.unlink()
        except OSError as error:
            _log.warning("could not delete %s: %s", segment.path.name, error)
            result.failed.append(segment.path)
            continue

        store.forget_segment(segment.path)
        store.audit(
            actor, "recording.deleted", str(segment.path),
            f"{segment.camera_id}, {segment.frames} frames, "
            f"{segment.size_bytes / 1024**2:.1f} MiB",
        )
        result.deleted.append(segment)
        result.freed_bytes += segment.size_bytes
        total_bytes -= segment.size_bytes
        free_bytes += segment.size_bytes

    if over_budget():
        result.shortfall = (
            f"retention could not reach its target: {total_bytes / 1024**3:.1f} GiB "
            f"recorded, {free_bytes / 1024**3:.1f} GiB free, "
            f"{result.kept_preserved} segment(s) preserved as evidence and not "
            "eligible for deletion. Recording will continue until the disk is "
            "full and then stop."
        )
        _log.error("%s", result.shortfall)

    if result.deleted and not dry_run:
        _log.info(
            "retention removed %d segment(s), %.2f GiB (%s)",
            len(result.deleted), result.freed_gib, policy.describe(),
        )
    return result


def _free_bytes(shutil_module, candidates: list[Segment]) -> float:
    """Free space on the volume the recordings are actually on.

    Measured from a segment rather than from the data directory, because the
    normal deployment puts video on its own disk and the two answers differ.
    An unreadable volume reports infinity: a retention pass must never delete
    everything because it could not stat a disk.
    """
    for segment in candidates:
        try:
            return float(shutil_module.disk_usage(segment.path.parent).free)
        except OSError:
            continue
    return float("inf")
