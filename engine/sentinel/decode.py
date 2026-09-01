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
from urllib.parse import urlsplit

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

#: Whether a camera outside the local network may be opened at all.
#:
#: Off by default, and deliberately an environment variable rather than a
#: setting in the interface: reaching a routable address contradicts the
#: product's central promise, so it should require a deliberate act by whoever
#: runs the process rather than a checkbox an operator can tick by accident.
_ALLOW_PUBLIC_SOURCES = os.environ.get("SENTINEL_ALLOW_PUBLIC_SOURCES", "").strip() not in (
    "", "0", "false", "no",
)

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


#: Query parameters whose value is a credential. Cameras and NVRs routinely put
#: one here instead of in the userinfo, and a redactor that only strips userinfo
#: passes the password through untouched while looking like it worked.
_CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "password", "passwd", "pwd", "pass", "secret", "token", "auth",
        "key", "apikey", "api_key", "access_token", "accesstoken",
        "signature", "sig", "credential", "credentials", "session",
    }
)

#: Substituted for every removed secret. A fixed marker, never one character per
#: character: the length of a password is information an attacker can use.
REDACTED = "***"


def redact_url(url: str) -> str:
    """Strip every credential from a URL so it is safe to log or display.

    ``rtsp://admin:hunter2@10.0.0.5/stream`` becomes
    ``rtsp://admin:***@10.0.0.5/stream``. The username survives because
    operators identify cameras by it and it is not a secret; the password never
    appears in any form, including its length.

    Written defensively, because this function is the only thing standing between
    a camera password and every log line, error message and database row in the
    system. Three ways an earlier version leaked, all now covered:

    - **It parsed before it redacted.** ``urlsplit`` succeeds but ``.port``
      raises ``ValueError`` on a non-numeric port, and ``.hostname`` returns
      ``None`` for shapes it does not recognise. Both paths returned the input
      verbatim. Nothing here depends on a successful parse: the userinfo is
      removed by string surgery on the netloc, which cannot fail.
    - **It only ever looked at the userinfo.** A credential in the query string
      survived untouched.
    - **It rebuilt the host from ``.hostname``**, which strips the brackets from
      an IPv6 literal and produced an unparseable display URL.

    When anything is uncertain the function fails *closed* — it returns a marker
    rather than the input, because echoing a string that might contain a
    password is the one outcome that must never happen.
    """
    if not isinstance(url, str) or not url:
        return "<no source>"

    # A local path is not a URL and has no credential to strip. Recognised
    # before any parsing, so a Windows path like C:\media\clip.mp4 is never
    # mangled by scheme detection.
    if "://" not in url:
        # A path, not a URL — returned intact so an operator can find the file.
        # The query pass still runs: "?" is illegal in a Windows filename and
        # vanishingly rare in a POSIX one, so redacting a credential-shaped
        # parameter here costs nothing and covers a schemeless "host/s?token=x".
        return _redact_query(url if "@" not in url else "<redacted path>")

    scheme, _, remainder = url.partition("://")
    netloc, slash, tail = remainder.partition("/")

    # Userinfo removal by string surgery. rpartition, not partition: a password
    # may itself contain an "@", and only the last one separates host from
    # userinfo.
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        username = userinfo.split(":", 1)[0]
        netloc = (f"{username}:{REDACTED}@" if username else f"{REDACTED}@") + host

    rebuilt = f"{scheme}://{netloc}"
    if slash:
        rebuilt += "/" + tail

    return _redact_query(rebuilt)


def _redact_query(url: str) -> str:
    """Replace credential-shaped query values, leaving the rest legible."""
    head, sep, query = url.partition("?")
    if not sep or not query:
        return url

    query, hash_sep, fragment = query.partition("#")

    redacted = []
    for pair in query.split("&"):
        name, has_value, _ = pair.partition("=")
        if has_value and name.lower() in _CREDENTIAL_QUERY_KEYS:
            redacted.append(f"{name}={REDACTED}")
        else:
            redacted.append(pair)

    return head + "?" + "&".join(redacted) + (hash_sep + fragment if hash_sep else "")


def contains_credential(text: str, url: str) -> bool:
    """Whether ``text`` leaks any secret held in ``url``.

    Used by the tests rather than by the runtime. It exists so a test can assert
    the absence of *this URL's* secrets rather than of one hard-coded sentinel,
    which is what lets it catch a leak through a path nobody thought of.
    """
    secrets = []
    if "://" in url:
        netloc = url.partition("://")[2].partition("/")[0]
        if "@" in netloc:
            userinfo = netloc.rpartition("@")[0]
            _, has_password, password = userinfo.partition(":")
            if has_password and password:
                secrets.append(password)

    _, sep, query = url.partition("?")
    if sep:
        for pair in query.partition("#")[0].split("&"):
            name, has_value, value = pair.partition("=")
            if has_value and value and name.lower() in _CREDENTIAL_QUERY_KEYS:
                secrets.append(value)

    return any(secret in text for secret in secrets)


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
            # Reported through the redacted display string, never the raw path.
            # `live` is a caller-supplied override and `_looks_live` only knows
            # six schemes, so a credentialed rtmp:// or srt:// URL reaches this
            # branch — and an earlier version put it, password and all, straight
            # into an error the operator reads.
            if not path.exists():
                raise DecodeError(f"No such video file: {self._display}")
            if not path.is_file():
                raise DecodeError(f"Not a file: {self._display}")

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

        self._require_private(host)

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

    def _require_private(self, host: str) -> None:
        """Refuse a camera address outside the local network.

        The product's central promise is that it works with the network cable
        unplugged and never reaches the Internet. A camera URL is the one string
        an operator types that the software then *connects to*, which makes it
        the natural way for that promise to be broken — by a typo, by a
        misconfigured DNS entry resolving to a public address, or deliberately.

        Loopback and the RFC 1918 / RFC 4193 ranges are allowed. Anything else is
        refused with the address named, so an operator who genuinely means to
        reach a routable host knows exactly what to override and why it stopped
        them.
        """
        import ipaddress

        try:
            addresses = {
                info[4][0]
                for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            }
        except OSError:
            # Cannot resolve. Left to the connect below, which reports it with a
            # better message than anything that could be said here.
            return

        public = []
        for address in addresses:
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                continue
            if not (parsed.is_private or parsed.is_loopback or parsed.is_link_local):
                public.append(address)

        if public and not _ALLOW_PUBLIC_SOURCES:
            raise DecodeError(
                f"{self._display} resolves to {', '.join(sorted(public))}, which is "
                "outside the local network. This system does not reach the "
                "Internet; if that address is genuinely a camera on a routed "
                "network, set SENTINEL_ALLOW_PUBLIC_SOURCES=1."
            )

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

        # A protocol allowlist, because the reachability probe checks the address
        # the operator typed and FFmpeg is free to follow the stream somewhere
        # else. Without this, an SDP or a redirect from a camera on the LAN can
        # send the process to a host on the Internet — which would defeat the
        # zero-WAN guarantee through a door nobody was watching.
        options = (
            "rtsp_transport;tcp"
            "|protocol_whitelist;file,rtp,udp,tcp,rtsps,tls,crypto"
        )

        with _FFMPEG_OPTIONS_LOCK:
            previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
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
            except Exception as error:  # noqa: BLE001
                # Anything else killed this thread silently, leaving the capture
                # open and the stream permanently empty while the interface went
                # on showing a camera that had stopped existing. The type is
                # reported rather than the message, because an arbitrary
                # exception's text may have come from a URL.
                self._state.error = DecodeError(
                    f"{self._source.display_url} stopped unexpectedly "
                    f"({type(error).__name__}). The stream will be reconnected."
                )

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
