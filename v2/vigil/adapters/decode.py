"""Video in: files, local devices and RTSP, with the egress guard in front.

The one string an operator types that the software then *connects to* is a
camera address, so this is where the zero-WAN promise is enforced for
sources: a host that resolves to a public address is refused unless the
operator sets `VIGIL_ALLOW_PUBLIC_SOURCES=1`, and every such connection is
logged at WARNING.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from ..logs import get as _get_logger

_log = _get_logger(__name__)

PUBLIC_SOURCES_VARIABLE = "VIGIL_ALLOW_PUBLIC_SOURCES"
LIVE_SCHEMES = ("rtsp", "rtsps", "http", "https")
REDACTED = "***"
OPEN_TIMEOUT_MILLIS = 5000
READ_TIMEOUT_MILLIS = 5000
BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


class DecodeError(RuntimeError):
    pass


def public_sources_allowed(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return environ.get(PUBLIC_SOURCES_VARIABLE, "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray
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
    source_id: str
    display: str
    width: int
    height: int
    nominal_fps: float
    live: bool


def is_live_source(url: str | Path) -> bool:
    text = str(url)
    if text.startswith("device:"):
        return True
    return urlsplit(text).scheme.lower() in LIVE_SCHEMES


def redacted(url: str | Path) -> str:
    """The address without its password, for logs, screens and the database."""
    text = str(url)
    parts = urlsplit(text)
    if not parts.scheme or "@" not in parts.netloc:
        return text
    userinfo, host = parts.netloc.rsplit("@", 1)
    user = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{user}:{REDACTED}@{host}" if ":" in userinfo else f"{user}@{host}", parts.path, parts.query, parts.fragment))


def split_password(url: str) -> tuple[str, str | None]:
    """(url without password, password) — the password never reaches the store."""
    parts = urlsplit(url)
    if not parts.scheme or "@" not in parts.netloc:
        return url, None
    userinfo, host = parts.netloc.rsplit("@", 1)
    if ":" not in userinfo:
        return url, None
    user, password = userinfo.split(":", 1)
    return urlunsplit((parts.scheme, f"{user}@{host}", parts.path, parts.query, parts.fragment)), password


def with_password(url: str, password: str | None) -> str:
    if password is None:
        return url
    parts = urlsplit(url)
    userinfo, host = parts.netloc.rsplit("@", 1) if "@" in parts.netloc else ("", parts.netloc)
    user = userinfo.split(":", 1)[0]
    return urlunsplit((parts.scheme, f"{user}:{password}@{host}", parts.path, parts.query, parts.fragment))


def require_private(host: str, *, allow_public: bool | None = None) -> None:
    """Refuse a host outside RFC1918/loopback/link-local unless overridden."""
    allow = public_sources_allowed() if allow_public is None else allow_public
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)}
    except OSError:
        return  # cannot resolve; the connect will say so with a better message
    public = []
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not (parsed.is_private or parsed.is_loopback or parsed.is_link_local):
            public.append(address)
    if not public:
        return
    if allow:
        _log.warning("%s allows a public source: %s resolves to %s", PUBLIC_SOURCES_VARIABLE, host, ", ".join(sorted(public)))
        return
    raise DecodeError(
        f"{host} resolves to {', '.join(sorted(public))}, which is outside the local network. "
        f"This system does not reach the Internet. For a camera on a routed private WAN set {PUBLIC_SOURCES_VARIABLE}=1."
    )


class VideoSource:
    """One source, opened lazily, read frame by frame. Not thread-safe; one owner."""

    def __init__(self, url: str | Path, *, source_id: str | None = None):
        self._url = str(url)
        self.source_id = source_id or _default_id(self._url)
        self.display = redacted(self._url)
        self.live = is_live_source(self._url)
        self._capture: cv2.VideoCapture | None = None
        self._index = 0
        self._started_at: float | None = None
        self._info: SourceInfo | None = None

    def __repr__(self) -> str:
        return f"VideoSource({self.display!r})"

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def info(self) -> SourceInfo | None:
        return self._info

    def open(self) -> SourceInfo:
        if self._capture is not None and self._info is not None:
            return self._info
        parts = urlsplit(self._url)
        if parts.scheme.lower() in LIVE_SCHEMES and parts.hostname:
            require_private(parts.hostname)
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise DecodeError(f"could not open {self.display}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._capture = capture
        self._started_at = time.monotonic()
        self._info = SourceInfo(self.source_id, self.display, width, height, fps if fps > 0 else 0.0, self.live)
        _log.info("opened %s: %dx%d @ %.1f fps%s", self.display, width, height, fps, " (live)" if self.live else "")
        return self._info

    def _open_capture(self) -> cv2.VideoCapture:
        if self._url.startswith("device:"):
            index = int(self._url.split(":", 1)[1] or 0)
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened() and os.name == "nt":
                capture.release()
                capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
            return capture
        if self.live:
            capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            try:
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MILLIS)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MILLIS)
            except cv2.error:
                pass
            return capture
        path = Path(self._url)
        if not path.is_file():
            raise DecodeError(f"no file at {path}")
        return cv2.VideoCapture(str(path))

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(image, self._timestamp(), self._index, self.source_id)
        self._index += 1
        return frame

    def _timestamp(self) -> int:
        if self.live or self._capture is None:
            return int(time.time() * 1000)
        position = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        if position and position > 0:
            return int(position)
        fps = self._info.nominal_fps if self._info and self._info.nominal_fps > 0 else 25.0
        return int(self._index * 1000 / fps)

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _default_id(url: str) -> str:
    if url.startswith("device:"):
        return url
    parts = urlsplit(url)
    if parts.scheme in LIVE_SCHEMES:
        return parts.hostname or "camera"
    return Path(url).stem


class LiveReader:
    """Reads a live source on its own thread, keeping only the newest frame.

    A live camera does not wait: a consumer slower than the stream must
    drop frames, and it must drop the *oldest*. Reconnects with backoff when
    the stream ends or fails; counts both.
    """

    def __init__(self, source: VideoSource):
        self._source = source
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.reconnects = 0
        self.fault: str | None = None
        self.last_frame_at: float | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"vigil-read-{self._source.source_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        with self._lock:
            self._ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def read(self, timeout: float = 5.0) -> Frame | None:
        with self._lock:
            if self._latest is None:
                self._ready.wait(timeout)
            frame, self._latest = self._latest, None
            return frame

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self._source.open()
                attempt = 0
                self.fault = None
                while not self._stop.is_set():
                    frame = self._source.read()
                    if frame is None:
                        raise DecodeError("stream ended")
                    self.last_frame_at = time.monotonic()
                    with self._lock:
                        if self._latest is not None:
                            self.dropped += 1
                        self._latest = frame
                        self._ready.notify()
            except DecodeError as error:
                self.fault = str(error)
                _log.warning("%s: %s; reconnecting", self._source.display, error)
            except Exception as error:  # noqa: BLE001 - the thread must not die silently
                self.fault = f"{type(error).__name__}: {error}"
                _log.exception("%s: reader failed", self._source.display)
            finally:
                self._source.close()
            if self._stop.is_set():
                break
            self.reconnects += 1
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            attempt += 1
            self._stop.wait(delay)
