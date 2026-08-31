"""Video decode.

Turns a file or a live stream into timestamped frames. Everything downstream —
detection, tracking, projection — depends on two properties this module is
responsible for:

**Timestamps must be real.** The tracker's gap budget, an object's speed, and
whether two cameras saw the same person within three seconds are all differences
between timestamps. Inventing them from a nominal frame rate produces a system
that is subtly wrong whenever a stream stutters, which is exactly when it
matters.

**A file and a live stream are not the same thing.** A file is evidence: every
frame is processed, in order, so replaying it reproduces the original result. A
live stream is a firehose: when the analytic is slower than the camera, old
frames are dropped, because an operator needs to know what is happening now, not
what happened forty seconds ago. Treating a file like a stream breaks replay
determinism; treating a stream like a file builds an unbounded backlog until the
process dies.

No credential ever leaves this module. An RTSP URL carrying a password is held
in one place, redacted for every other purpose, and never logged, formatted, or
raised in an exception.
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

#: How long to wait for a live frame before treating the stream as stalled.
LIVE_FRAME_TIMEOUT_SECONDS = 10.0

#: How long to wait for a network source to answer before giving up.
#:
#: Applied by a socket probe before the decoder is involved at all, because
#: OpenCV's own connect timeout is a hard-coded 30 seconds that its documented
#: FFmpeg options do not change. Thirty seconds per camera means a node with
#: twenty cameras behind a switch that has just lost power takes ten minutes to
#: work out that none of them are there. Five is longer than any healthy camera
#: on a LAN needs, and the reconnect loop retries anyway.
OPEN_TIMEOUT_SECONDS = 5.0

#: FFmpeg options are passed through a process-wide environment variable, so
#: setting them has to be serialised against other threads opening sources.
_FFMPEG_OPTIONS_LOCK = threading.Lock()

#: Default ports, so a URL that omits one can still be probed.
_DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}

#: Reconnect backoff for a live source, in seconds. Bounded and jittered, so an
#: outage that takes out twenty cameras does not produce a synchronised
#: reconnect storm against the switch.
_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


class DecodeError(RuntimeError):
    """A source could not be opened or read.

    Messages from this class are shown to operators and written to logs, so they
    must never carry a credential. Construct them from redacted values only.
    """


def redact_url(url: str) -> str:
    """Strip credentials from a URL so it is safe to log or display.

    ``rtsp://admin:hunter2@10.0.0.5/stream`` becomes
    ``rtsp://admin:***@10.0.0.5/stream``. The username survives because
    operators identify cameras by it and it is not a secret; the password never
    appears in any form, including its length.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"

    if not parts.hostname:
        return url

    if parts.password is None:
        return url

    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"

    userinfo = f"{parts.username}:***@" if parts.username else "***@"
    return urlunsplit((parts.scheme, userinfo + host, parts.path, parts.query, parts.fragment))


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame and when it happened.

    ``image`` is BGR uint8, as OpenCV produces. It is not copied — treat it as
    read-only, because for a live source the decoder may reuse the buffer.
    """

    image: np.ndarray
    #: Milliseconds since the epoch for a live source; milliseconds since the
    #: start of the media for a file. Monotonic within a source either way.
    timestamp_millis: int
    index: int
    source_id: str

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What the decoder found when it opened a source.

    ``frame_count`` and ``fps`` are ``None`` for a live stream, and are reported
    as the container claims them for a file — a claim, not a measurement.
    Containers lie about both.
    """

    width: int
    height: int
    fps: float | None
    frame_count: int | None
    #: Always safe to log or display.
    display_url: str
    is_live: bool


class VideoSource:
    """A file or live stream, decoded to frames.

    Use as a context manager. Iterating yields :class:`Frame`; the iteration ends
    when a file is exhausted, and for a live stream only when the source is
    closed or reconnection is abandoned.
    """

    __slots__ = ("_url", "_display", "_id", "_capture", "_info", "_index", "_is_live", "_opened")

    def __init__(self, url: str | Path, *, source_id: str | None = None, live: bool | None = None):
        """
        ``live`` overrides the guess drawn from the URL scheme. A file being
        replayed as if it were live is a legitimate thing to want for testing, so
        the choice is explicit rather than inferred and stuck.
        """
        raw = str(url)
        self._url = raw
        self._display = redact_url(raw)
        self._id = source_id or self._display
        self._is_live = _looks_live(raw) if live is None else live
        self._capture: cv2.VideoCapture | None = None
        self._info: SourceInfo | None = None
        self._index = 0
        self._opened = False

    # The password lives in self._url and nowhere else. These two methods exist
    # so that printing a source anywhere — a log line, a traceback, a debugger —
    # cannot leak it.
    def __repr__(self) -> str:
        return f"VideoSource({self._display!r}, live={self._is_live})"

    __str__ = __repr__

    @property
    def source_id(self) -> str:
        return self._id

    @property
    def display_url(self) -> str:
        """The URL with any credential removed. Safe to log."""
        return self._display

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            raise DecodeError("the source is not open")
        return self._info

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> SourceInfo:
        if self._opened:
            return self.info

        if not self._is_live:
            path = Path(self._url)
            if not path.exists():
                raise DecodeError(f"No such video file: {path}")
            if not path.is_file():
                raise DecodeError(f"Not a file: {path}")

        if self._is_live:
            self._require_reachable()

        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            # Deliberately says nothing about why. OpenCV's reasons are in its
            # own log, and the URL here may carry a credential.
            raise DecodeError(f"Could not open {self._display}")

        if self._is_live:
            # One frame of slack. Without this the buffer fills while the
            # analytic thinks, and the operator watches the past.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise DecodeError(f"{self._display} opened but reports no video dimensions")

        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        self._capture = capture
        self._info = SourceInfo(
            width=width,
            height=height,
            fps=float(fps) if fps and fps > 0 else None,
            frame_count=int(count) if not self._is_live and count and count > 0 else None,
            display_url=self._display,
            is_live=self._is_live,
        )
        self._opened = True
        return self._info

    def _require_reachable(self) -> None:
        """Fail fast when nothing is listening.

        FFmpeg's connect timeout is set through a process-wide environment
        variable whose option names differ between builds, so it cannot be relied
        on. A plain TCP connect can be, and it turns a thirty-second stall into a
        sub-second answer — which is the difference between a node discovering
        that a switch has lost power in a moment and taking ten minutes over
        twenty cameras.

        It also produces a far better message. "Connection refused" and "no route
        to host" are different problems with different fixes, and "could not
        open" is neither.
        """
        parts = urlsplit(self._url)
        host = parts.hostname
        if not host:
            return

        port = parts.port or _DEFAULT_PORTS.get(parts.scheme.lower())
        if port is None:
            return

        try:
            with socket.create_connection((host, port), timeout=OPEN_TIMEOUT_SECONDS):
                return
        except socket.timeout:
            raise DecodeError(
                f"{self._display} did not answer within {OPEN_TIMEOUT_SECONDS:.0f}s. "
                "The camera may be off, or a firewall may be dropping the connection."
            ) from None
        except OSError as error:
            # Only errno and strerror, never the exception's own repr: on some
            # platforms that includes the address it was given, and the address
            # came from a URL that carries a credential.
            reason = error.strerror or type(error).__name__
            raise DecodeError(f"{self._display} is not reachable: {reason}") from None

    def _open_capture(self) -> cv2.VideoCapture:
        """Hand the URL to OpenCV.

        There is no connect timeout here, and that is not an oversight. FFmpeg's
        timeout options are passed through the process-wide
        ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` variable, and all four spellings —
        ``stimeout``, ``timeout``, ``open_timeout``, and the combination with
        ``rtsp_transport`` — were measured against an unroutable address on this
        build and every one of them still took the full 30 seconds. OpenCV's own
        hard-coded interrupt callback is what fires, and it does not read those
        options. Code that set them would look like a protection while providing
        none, so :meth:`_require_reachable` does the job with a plain socket
        instead.

        ``rtsp_transport=tcp`` is still worth setting: RTSP over UDP loses frames
        on a congested link, and TCP is what almost every deployment wants. It is
        a preference passed to the demuxer rather than a claim about behaviour,
        and it has not been verified against physical hardware.
        """
        if not self._url.lower().startswith(("rtsp://", "rtsps://")):
            return cv2.VideoCapture(self._url)

        with _FFMPEG_OPTIONS_LOCK:
            previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            try:
                return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            finally:
                if previous is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._opened = False

    def read(self) -> Frame | None:
        """One frame, or ``None`` at the end of a file / on a read failure."""
        if self._capture is None:
            raise DecodeError("the source is not open")

        ok, image = self._capture.read()
        if not ok or image is None:
            return None

        frame = Frame(
            image=image,
            timestamp_millis=self._timestamp(),
            index=self._index,
            source_id=self._id,
        )
        self._index += 1
        return frame

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def _timestamp(self) -> int:
        """When this frame happened.

        For a file, the container's presentation timestamp, so a stream that was
        recorded with gaps replays with those gaps intact. For a live source, the
        wall clock at the moment of decode — the closest thing available, and
        honest about being an arrival time rather than a capture time.
        """
        if self._is_live:
            return int(time.time() * 1000)

        assert self._capture is not None
        position = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        if position and position > 0:
            return int(position)

        # Some containers report nothing until the second frame. Falling back to
        # the declared frame rate is a guess, and is only ever used for a file
        # whose timestamps are absent — never to override one that is present.
        fps = self._info.fps if self._info else None
        if fps:
            return int(self._index * 1000.0 / fps)
        return self._index


def _looks_live(url: str) -> bool:
    lowered = url.lower()
    return lowered.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://"))


@dataclass
class _StreamState:
    stop: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class LiveStream:
    """A live source decoded on its own thread, with the newest frame winning.

    Decode runs ahead of the analytic or behind it, never in lockstep. When it
    runs ahead, the queue holds one frame and older ones are discarded: an
    operator needs the present, and a backlog is worse than a gap. When the
    stream drops, it reconnects with bounded backoff rather than a tight loop
    that turns one camera outage into a broadcast storm.

    Frames dropped this way are counted, because a system silently discarding
    half its input while reporting healthy is worse than one that says so.
    """

    __slots__ = ("_source", "_queue", "_thread", "_state", "_dropped", "_reconnects", "_lock")

    def __init__(self, source: VideoSource):
        if not source.is_live:
            raise DecodeError(
                "LiveStream drops frames to stay current, which would make replay "
                "non-deterministic. Iterate a file source directly instead."
            )
        self._source = source
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._state = _StreamState()
        self._dropped = 0
        self._reconnects = 0
        self._lock = threading.Lock()

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def reconnects(self) -> int:
        with self._lock:
            return self._reconnects

    def __enter__(self) -> "LiveStream":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._state.stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"decode:{self._source.source_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._state.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._source.close()

    def read(self, timeout: float = LIVE_FRAME_TIMEOUT_SECONDS) -> Frame | None:
        """The newest frame, or ``None`` if none arrived within ``timeout``."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._state.error is not None:
                raise DecodeError(str(self._state.error)) from self._state.error
            return None

    def __iter__(self) -> Iterator[Frame]:
        while not self._state.stop.is_set():
            frame = self.read()
            if frame is not None:
                yield frame

    def _run(self) -> None:
        attempt = 0
        while not self._state.stop.is_set():
            try:
                self._source.open()
                attempt = 0
                self._pump()
            except DecodeError as error:
                # Redacted by construction: DecodeError never carries a URL that
                # still has its credential in it.
                self._state.error = error

            if self._state.stop.is_set():
                return

            self._source.close()
            with self._lock:
                self._reconnects += 1

            delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
            attempt += 1
            self._state.stop.wait(delay)

    def _pump(self) -> None:
        while not self._state.stop.is_set():
            frame = self._source.read()
            if frame is None:
                return

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # The analytic has not caught up. Throw away what it has not
                # taken and give it the newer frame.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                with self._lock:
                    self._dropped += 1
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass
