"""Clips on disk, named safely, hashed as they close."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..logs import get as _get_logger

_log = _get_logger(__name__)

DEFAULT_CODEC = "mp4v"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def file_safe(name: str) -> str:
    """`device:0` would be an NTFS alternate data stream. Learned in v1."""
    cleaned = _UNSAFE.sub("-", name).strip("-.")
    return cleaned or "camera"


@dataclass(frozen=True, slots=True)
class Segment:
    camera_id: str
    path: Path
    started_millis: int
    ended_millis: int
    frames: int
    width: int
    height: int
    nominal_fps: float
    size_bytes: int
    sha256: str


@dataclass
class RecorderStats:
    segments_written: int = 0
    bytes_written: int = 0
    frames_written: int = 0
    frames_dropped: int = 0
    fault: str | None = None


class Recorder:
    """Writes frames into segments of `segment_seconds`. One owner thread."""

    def __init__(self, camera_id: str, directory: Path, *, fps: float = 15.0, segment_seconds: float = 60.0,
                 codec: str = DEFAULT_CODEC):
        self.camera_id = camera_id
        self.directory = Path(directory) / file_safe(camera_id)
        self.fps = fps if fps > 0 else 15.0
        self.segment_millis = int(segment_seconds * 1000)
        self.codec = codec
        self.stats = RecorderStats()
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._started: int | None = None
        self._last: int | None = None
        self._frames = 0
        self._size: tuple[int, int] | None = None
        self._closed: list[Segment] = []

    def start(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.stats.fault = f"cannot create {self.directory}: {error}"
            raise

    def write(self, image: np.ndarray, at_millis: int) -> None:
        if self.stats.fault is not None:
            self.stats.frames_dropped += 1
            return
        h, w = image.shape[:2]
        started = self._started if self._started is not None else at_millis
        if self._writer is not None and (self._size != (w, h) or at_millis - started >= self.segment_millis):
            self._close_segment(at_millis)
        if self._writer is None:
            self._open(at_millis, (w, h))
        assert self._writer is not None
        self._writer.write(image)
        self._frames += 1
        self._last = at_millis
        self.stats.frames_written += 1

    def _open(self, at_millis: int, size: tuple[int, int]) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(at_millis / 1000))
        path = self.directory / f"{file_safe(self.camera_id)}-{stamp}-{at_millis % 1000:03d}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
        if not writer.isOpened():
            self.stats.fault = f"cannot open a writer at {path}"
            self.stats.frames_dropped += 1
            _log.error("%s: RECORDING STOPPED — %s", self.camera_id, self.stats.fault)
            raise OSError(self.stats.fault)
        self._writer, self._path, self._started, self._size, self._frames = writer, path, at_millis, size, 0

    def _close_segment(self, ended_millis: int) -> None:
        if self._writer is None or self._path is None or self._started is None or self._size is None:
            return
        self._writer.release()
        self._writer = None
        if self._frames == 0:
            self._path.unlink(missing_ok=True)
            return
        size = self._path.stat().st_size
        digest = hashlib.sha256(self._path.read_bytes()).hexdigest()
        segment = Segment(self.camera_id, self._path, self._started, ended_millis, self._frames,
                          self._size[0], self._size[1], self.fps, size, digest)
        self._closed.append(segment)
        self.stats.segments_written += 1
        self.stats.bytes_written += size
        _log.info("%s: clip %s (%d frames, %.1f MiB)", self.camera_id, self._path.name, self._frames, size / 1048576)

    def take_closed(self) -> list[Segment]:
        out, self._closed = self._closed, []
        return out

    def close(self) -> list[Segment]:
        self._close_segment(self._last if self._last is not None else (self._started or 0))
        return self.take_closed()
