"""The site's ground, built while the site runs rather than in a separate pass.

# Why this exists

`vigil map build` opens every camera a second time, records for a minute and
writes a map. It works, and it is the wrong shape for the product: the cameras
are *already open*, already decoding, already projecting. Every frame the
analysis looks at is a free sample of the ground, and throwing it away meant a
site had a map of one minute of one afternoon or no map at all.

Fed from the analysis, the map improves for as long as the site runs, and the
plan an operator is looking at fills in while they watch.

# Threads, and why there is a lock

The frames arrive on each camera's own worker thread; the composite is built
on the runtime's. The per-camera accumulators are separate — no two workers
touch the same one — but `MapBuilder.build` reads all of them at once, and the
accumulator is a Rust object being written to by another thread while it is
read. That is a data race, not a stale read: the median accumulator resizes
its per-cell sample buffers.

So one lock around both. Contention is negligible by construction — a camera
offers a frame every two seconds and a build happens every thirty — and the
alternative, a per-camera lock, buys nothing while making the invariant harder
to state.

# What a pose change does

`MapBuilder.observe` already discards a camera's accumulator when its pose
changes, because samples projected through the old pose describe ground that
is somewhere else. What this adds is that the composite is rebuilt promptly
afterwards rather than at the next scheduled time, so the plan stops showing
ground computed from a pose that is no longer true. Those cells go back to
empty and fill in again — empty being the honest state for ground nobody has
looked at through the current pose.

This matters more than it used to: `service.triangulation` writes a solved
ground tilt back to every camera, which is a pose change, arriving
automatically a minute or so into a run.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from ..domain.geo import CameraPose, LatLon
from ..logs import get as _get_logger
from .mapping import DEFAULT_CELL_SIZE_M, GroundMap, MapBuilder, MappingError, save_map

_log = _get_logger(__name__)

#: How often the per-camera accumulators are composited into one map.
#:
#: Thirty seconds. The composite is the expensive half — it allocates the
#: union grid and asks every accumulator for its median — while folding one
#: frame in is microseconds. Rebuilding per frame would spend most of a core
#: re-deriving something that changes slowly by construction, since the whole
#: point of the median is that one more sample barely moves it.
REBUILD_EVERY_SECONDS = 30.0

#: How often the map is written to disk. Longer than the rebuild: a write is
#: a PNG encode, a compressed array and a fingerprint over both, and the file
#: only has to be good enough that a restart does not begin from nothing.
PERSIST_EVERY_SECONDS = 300.0


class LiveMap:
    """One ground map per site, fed by the running analysis.

    Every method is safe to call from any thread. `observe` is called from the
    camera workers and everything else from the runtime.
    """

    def __init__(self, origin: LatLon, directory: Path | None = None, *,
                 cell_size_m: float = DEFAULT_CELL_SIZE_M,
                 rebuild_every_seconds: float = REBUILD_EVERY_SECONDS,
                 persist_every_seconds: float = PERSIST_EVERY_SECONDS):
        self.origin = origin
        self.directory = Path(directory) if directory is not None else None
        self._builder = MapBuilder(origin, cell_size_m=cell_size_m)
        self._lock = threading.Lock()
        self._ground: GroundMap | None = None
        self._rebuild_every = rebuild_every_seconds
        self._persist_every = persist_every_seconds
        # Both start at "now" so the first rebuild waits a full interval.
        # Left at zero, the first tick always rebuilt — monotonic() is a large
        # number and the difference from zero always exceeds any interval —
        # and it composited a map from one sample per cell, which is below
        # `MIN_SAMPLES` and therefore an empty map. The console then said "no
        # ground was mapped" for the first half minute of every run, which
        # reads as a broken feature rather than as one that has not finished.
        self._last_build = time.monotonic()
        self._last_persist = time.monotonic()
        self._dirty = False
        #: Set when a pose changed under us, so the next tick rebuilds
        #: immediately instead of leaving the plan showing ground computed
        #: through a pose that is no longer true.
        self._stale = False
        self._samples = 0

    # --------------------------------------------------------- from a worker

    def observe(self, camera_id: str, pose: CameraPose, image: np.ndarray,
                at_seconds: float | None = None) -> bool:
        """Offer one frame. False when the rate limit skipped it.

        Never raises into a worker. A camera whose frame cannot be projected
        must go on being analysed — the map is the least important thing this
        product does with a frame, and it is not allowed to stop the rest.
        """
        try:
            with self._lock:
                known = self._builder._cameras.get(camera_id)
                if known is not None and known.pose != pose:
                    self._stale = True
                taken = self._builder.observe(camera_id, pose, image, at_seconds)
                if taken:
                    self._samples += 1
                    self._dirty = True
                return taken
        except Exception:  # noqa: BLE001 - the map must not take a camera down
            _log.exception("%s: folding a frame into the live map failed", camera_id)
            return False

    # -------------------------------------------------------- from the runtime

    def tick(self) -> GroundMap | None:
        """Rebuild and persist if either is due. Returns the current map."""
        now = time.monotonic()
        due = self._stale or (self._dirty and now - self._last_build >= self._rebuild_every)
        if due:
            self._rebuild(now)
        if (self.directory is not None and self._ground is not None
                and now - self._last_persist >= self._persist_every):
            self._last_persist = now
            self.save()
        return self._ground

    def _rebuild(self, now: float) -> None:
        was_stale, self._stale = self._stale, False
        self._last_build = now
        self._dirty = False
        try:
            with self._lock:
                ground = self._builder.build()
        except (MappingError, ValueError) as error:
            _log.error("the live map could not be built: %s", error)
            return
        self._ground = ground
        if was_stale:
            _log.info("a camera moved, so the map was rebuilt without its old samples: %s",
                      ground.describe())

    def save(self) -> Path | None:
        """Write the map where a restart will find it, or `None`.

        Best effort by design: a disk that is full must not stop a site
        analysing, and the map is reconstructible from the next few minutes of
        frames in a way that an event is not.
        """
        if self.directory is None or self._ground is None:
            return None
        try:
            return save_map(self._ground, self.directory)
        except OSError as error:
            _log.error("the live map could not be written to %s: %s", self.directory, error)
            return None

    @property
    def ground(self) -> GroundMap | None:
        """The last composite, or `None` before the first one."""
        return self._ground

    @property
    def samples(self) -> int:
        """Frames folded in, across every camera."""
        return self._samples

    def describe(self) -> str:
        if self._ground is None:
            return f"no map yet — {self._samples} frame(s) folded in"
        return self._ground.describe()

    def close(self) -> None:
        with self._lock:
            self._builder.close()
