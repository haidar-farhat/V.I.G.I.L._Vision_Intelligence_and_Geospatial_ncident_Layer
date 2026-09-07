"""The analysis, running. One worker thread per camera, one owner of the store.

A worker does decode → detect → track → presence → rules → record and hands
results over a bounded outbox. The runtime, on the owning thread, drains the
outboxes, persists, correlates, watches health and raises alerts. Workers
never touch the store.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..adapters.decode import DecodeError, Frame, LiveReader, VideoSource
from ..adapters.detectors import Detector
from ..adapters.recorder import Recorder, Segment
from ..domain.detection import DetectorInfo
from ..domain.events import Event, Rule, RuleContext, default_rules
from ..domain.incidents import Correlator, Incident
from ..domain.relations import RelationTracker
from ..domain.tracking import Track, Tracker, TrackerConfig
from ..domain.zones import PresenceTracker, Zone
from ..logs import get as _get_logger
from ..perception.appearance import describe
from ..perception.motion import CameraMotion, CameraMotionEstimator
from ..perception.quality import FrameQuality, FrameQualityMonitor
from ..storage.store import Store
from .alerts import (
    CAMERA_DARK, CAMERA_DEGRADED, DISK_LOW, RECORDING_STOPPED, RETENTION_SHORTFALL,
    THREAD_STUCK, Alerts,
)
from .auth import ANALYSIS_CONTROL, Principal
from .site import Camera, SiteService
from .livemap import LiveMap
from .mapping import MappingError
from .triangulation import MIN_GROUND_SAMPLES, Geometry, tilted
# Re-exported: `FrameResult` is this module's published shape as far as every
# interface is concerned, and moving the class must not move its import.
from .worker import CameraWorker, FrameResult, WorkerStats  # noqa: F401

_log = _get_logger(__name__)

DARK_AFTER_SECONDS = 30.0
#: Consecutive unusable frames before a camera is called degraded. Ten
#: seconds at 15 fps: one bad frame is a bad frame, a hundred and fifty is
#: a lens somebody has to go and clean.
DEGRADED_AFTER_FRAMES = 150
STOP_TIMEOUT_SECONDS = 8.0
DISK_WATERMARK_BYTES = 2 * 1024**3
OUTBOX_EVENTS = 1000
#: How often an unattended run says what it is doing, in seconds. Nobody is
#: reading the screen; the log is the only place this can be seen afterwards.
METRICS_EVERY_SECONDS = 60.0
#: How far back a correlation looks. An incident is a statement about a span,
#: and re-reading a month of events every two seconds is how a node that has
#: been up for a month stops keeping up with its cameras. Well beyond the
#: association window, so nothing that could be joined is cut off.
CORRELATION_SPAN_MILLIS = 60 * 60 * 1000

#: How often the ground plane is re-fitted from accumulated observations.
#:
#: A minute, not a frame. The fit is over thousands of points and the ground
#: does not move; running it per frame would spend real time re-deriving a
#: constant. The accumulator is fed every cycle, so nothing is missed — only
#: the conclusion is drawn less often than the evidence arrives.
GROUND_SOLVE_EVERY_SECONDS = 60.0



@dataclass(frozen=True, slots=True)
class CameraHealth:
    camera_id: str
    state: str  # STOPPED | STARTING | LIVE | DARK | FAULTED
    running: bool
    placed: bool
    analysis_fps: float
    frames: int
    fault: str | None
    seconds_since_frame: float | None
    recording: bool
    recording_fault: str | None
    clips: int
    #: Rolling frame quality, 0..1, or `None` before the first frame.
    quality: float | None = None
    #: Why the camera's frames are unusable, when they have been for long
    #: enough to be a fault rather than a moment.
    degraded: str | None = None

    def describe(self) -> str:
        parts = [self.state]
        if self.state == "LIVE":
            parts.append(f"{self.analysis_fps:.0f} fps")
        if self.state == "DARK" and self.seconds_since_frame is not None:
            parts.append(f"no frame for {self.seconds_since_frame:.0f} s")
        if self.degraded:
            parts.append(self.degraded)
        if self.fault:
            parts.append(self.fault)
        if self.recording:
            parts.append(f"recording ({self.clips} clip(s))")
        if self.recording_fault:
            parts.append(f"recording stopped: {self.recording_fault}")
        return " | ".join(parts)


@dataclass
class RetentionPolicy:
    max_age_days: float | None = 14.0
    max_bytes: int | None = None
    min_free_bytes: int | None = 5 * 1024**3


class Runtime:
    """Start, stop, poll. Owns the store's thread. Every control action takes a principal."""

    def __init__(self, site: SiteService, *, node_id: str = "local", detector_factory: Callable[[], Detector] | None = None,
                 record_to: Path | None = None, retention: RetentionPolicy | None = None, alerts: Alerts | None = None,
                 realtime: bool = False, keep_images: bool = False, rules_factory: Callable[[], list[Rule]] | None = None,
                 correlate_every_millis: int = 2000, retention_every_seconds: float = 600.0,
                 record_every_camera: bool = False, map_dir: Path | None = None,
                 build_map: bool = True):
        self.site = site
        self.store: Store = site.store
        self.node_id = node_id
        self._detector_factory = detector_factory
        self._record_to = record_to
        #: ``--record``: this run records every camera, whatever the site says.
        self._record_every_camera = record_every_camera
        self._retention = retention if retention is not None else (RetentionPolicy() if record_to else None)
        self._retention_every = retention_every_seconds
        self._last_retention: float | None = None
        self.retention_shortfall: str | None = None
        self._realtime = realtime
        self._keep_images = keep_images
        self._rules_factory = rules_factory
        self._correlate_every = correlate_every_millis
        self._last_correlated = 0.0
        self._workers: dict[str, CameraWorker] = {}
        self._events: list[Event] = []
        self._incidents: tuple[Incident, ...] = ()
        self.alerts = alerts if alerts is not None else Alerts()
        self.alerts.bind(self.store, f"node:{node_id}")
        self._running = False
        self._last_metrics: float | None = None
        #: Cross-camera positions and the site's own ground. Built on the first
        #: poll that has two placed cameras, because it needs an origin and
        #: there is nothing to triangulate before then.
        self._geometry: Geometry | None = None
        self._last_ground_solve = 0.0
        #: Where the live map is kept between runs. `None` means it is built
        #: and drawn but never written, which is what an ad-hoc `vigil run`
        #: over a file wants.
        self._map_dir = Path(map_dir) if map_dir is not None else None
        #: Off for a run that has no business building one — a test, or a
        #: pass over a recording whose poses describe a different day.
        self._build_map = build_map
        self._map: LiveMap | None = None

    # ------------------------------------------------------------ control

    @property
    def running(self) -> bool:
        return self._running

    @property
    def incidents(self) -> tuple[Incident, ...]:
        return self._incidents

    def site_timezone(self):
        """The clock the site's schedules are written in, or UTC.

        Read here rather than assumed, because a schedule that says "closed
        22:00 to 06:00" means the site's night, not the meridian's — and an
        after-hours rule evaluated in the wrong clock fires at the wrong
        hours, which is worse than not firing at all.
        """
        from datetime import timezone
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        name = (self.store.site().get("timezone") or "UTC").strip()
        if name.upper() == "UTC":
            return timezone.utc
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            _log.error("the site's timezone %r is not one this machine knows; using UTC, so any schedule "
                       "written in local hours will be wrong", name)
            return timezone.utc

    def start(self, by: Principal, *, cameras: Sequence[str] | None = None) -> int:
        by.require(ANALYSIS_CONTROL)
        if self._running:
            return len(self._workers)
        zones = self.site.zones(by)
        site_tz = self.site_timezone()
        rules = self._rules_factory() if self._rules_factory else self._default_rules()
        chosen = [c for c in self.site.cameras(by) if cameras is None or c.id in cameras]
        started = 0
        for camera in chosen:
            worker = CameraWorker(
                camera, self.site.source_with_credentials(camera), self._factory(), zones,
                list(rules), node_id=self.node_id,
                record_to=self._record_to, realtime=self._realtime, keep_images=self._keep_images,
                record_anyway=self._record_every_camera, site_tz=site_tz,
                detect_every=self._detect_every(), watch=self._watch(),
            )
            worker.live_map = self._ensure_map(chosen)
            worker.start()
            self._workers[camera.id] = worker
            started += 1
        self._running = started > 0
        self.store.audit(by.actor, "analysis.started", self.node_id, f"{started} camera(s)")
        return started

    def stop(self, by: Principal) -> bool:
        by.require(ANALYSIS_CONTROL)
        if self._map is not None:
            # Built over the whole run and written once here, so a site that
            # is stopped between the five-minute persists does not throw away
            # everything since the last one.
            self._map.tick()
            self._map.save()
        for worker in self._workers.values():
            worker.ask_to_stop()
        stubborn = [cid for cid, w in self._workers.items() if not w.stop()]
        if stubborn:
            self.store.audit(by.actor, "analysis.thread_stuck", ", ".join(stubborn), "left running rather than killed")
            self.alerts.raise_(THREAD_STUCK, ", ".join(stubborn), "analysis thread did not stop; left running rather than killed")
        self.poll(force_correlate=True)
        self._running = False
        self.store.audit(by.actor, "analysis.stopped", self.node_id)
        return not stubborn

    def close(self, by: Principal | None = None) -> None:
        if self._running:
            self.stop(by or Principal.system())
        self.store.close()

    def _default_rules(self) -> list[Rule]:
        """The standard rules, plus the threat rule this site's vocabulary makes.

        Read here rather than baked into `default_rules` because the
        vocabulary is a site setting, and a rule that cannot see it would
        quietly treat every site the same.
        """
        from ..domain.threats import ThreatRule

        vocabulary = self.site.threats()
        if vocabulary:
            _log.info("threat labels for this site: %s", vocabulary.describe())
        return [*default_rules(), ThreatRule(vocabulary)]

    def use_detector(self, factory: Callable[[], Detector], *, by: Principal) -> None:
        """Analyse with this from the next start. Running workers keep theirs.

        A detector is made once per analysis thread and lives as long as it
        does, so swapping one mid-run would mean two cameras drawing
        conclusions from different settings within one incident. The change
        is deliberately visible only at the next start, and the caller says
        so where it is typed.
        """
        by.require(ANALYSIS_CONTROL)
        self._detector_factory = factory

    def _factory(self) -> Callable[[], Detector]:
        if self._detector_factory is not None:
            return self._detector_factory
        from ..adapters.detectors import MotionDetector

        return MotionDetector

    # --------------------------------------------------------------- poll

    def poll(self, *, force_correlate: bool = False) -> list[FrameResult]:
        results: list[FrameResult] = []
        fresh: list[Event] = []
        for worker in self._workers.values():
            result = worker.take_latest()
            if result is not None:
                results.append(result)
            for segment in worker.take_segments():
                self.store.save_segment(segment)
            events = worker.take_events()
            if events:
                fresh.extend(events)
        if fresh:
            self.store.save_events(fresh)
            self._events.extend(fresh)
            self._events = self._events[-5000:]
        self._triangulate(results)
        if self._map is not None:
            self._map.tick()
        self._sweep_retention_if_due()
        self._watch_for_alerts()
        if self._running:
            self._log_metrics_if_due()
        due = (time.monotonic() - self._last_correlated) * 1000 >= self._correlate_every
        if fresh or force_correlate or due:
            self.correlate()
        if self._running and self._workers and not any(w.alive for w in self._workers.values()):
            self._running = False
        return results

    def _watch(self) -> frozenset[str] | None:
        """Labels this site raises events for, from the detector factory when
        there is one. `None` leaves the worker on the built-in list."""
        factory = self._detector_factory
        watch = getattr(factory, "watch", None)
        return watch() if callable(watch) else None

    def _ensure_map(self, cameras: Sequence[Camera]) -> "LiveMap | None":
        """The site's live map, built on the first placed camera.

        `None` when nothing is placed — there is no origin to hang a lattice
        on, and a map of a site whose cameras have no positions would be a
        picture of nothing. Also `None` without the engine core: `MapBuilder`
        refuses rather than falling back, because v1 measured the NumPy path
        at 79 ms per frame per camera and quietly running eighty times slower
        is not a fallback.
        """
        if not self._build_map:
            return None
        if self._map is None:
            placed = [c.pose for c in cameras if c.pose is not None]
            if not placed:
                return None
            try:
                self._map = LiveMap(placed[0].position, self._map_dir)
            except MappingError as error:
                _log.info("no live map on this machine: %s", error)
                self._build_map = False
                return None
            _log.info("building the site's map from the running analysis")
        return self._map

    def ground(self):
        """The site's ground map as it stands, or `None`.

        The live one when this node is analysing, which is fresher than
        anything on disk by definition. A caller wanting the stored map — the
        console before a run has started — reads it with `mapping.load_map`.
        """
        return None if self._map is None else self._map.ground

    def map_state(self) -> str:
        return "no map" if self._map is None else self._map.describe()

    # ------------------------------------------------------- cross-camera

    def _triangulate(self, results: Sequence[FrameResult]) -> None:
        """Offer this cycle's frames to the cross-camera geometry.

        Positions on the tracks are **not** rewritten here. A `FrameResult` is
        what one camera concluded from one frame, and quietly replacing a
        track's position with one derived from another camera would make that
        no longer true — the console draws boxes on the frame they came from
        for the same reason. The triangulated positions are published
        alongside, through `pairings()`, and the ground plane feeds back into
        the poses, which is where it belongs.
        """
        if not results:
            return
        poses = {c["id"]: c["pose"] for c in self.store.cameras() if c["pose"] is not None}
        if len(poses) < 2:
            return
        if self._geometry is None:
            origin = poses[sorted(poses)[0]].position
            self._geometry = Geometry(origin)
        for result in results:
            pose = poses.get(result.camera_id)
            if pose is not None:
                self._geometry.observe(result.camera_id, pose, result.at_millis, result.tracks,
                                       getattr(result, "appearance", None))
        self._geometry.resolve()
        now = time.monotonic()
        if now - self._last_ground_solve >= GROUND_SOLVE_EVERY_SECONDS:
            self._last_ground_solve = now
            self._apply_ground(self._geometry.solve_ground(), poses)

    def _apply_ground(self, plane, poses: dict) -> None:
        """Write a solved ground back to the cameras that will project onto it.

        To the **store** and to the running workers both. The store, because
        the next run should start from what this one measured rather than
        re-deriving it from nothing; the workers, because a camera projecting
        onto a level plane it has been shown is not level goes on producing
        biased positions until somebody restarts it.

        This reaches every placed camera, not only the overlapping pair that
        produced the evidence. That is the point: a site's ground is one
        surface, and the camera watching the far corner alone is the one whose
        positions were worst and which could never have measured it itself.
        """
        if plane is None or plane.inliers < MIN_GROUND_SAMPLES:
            return
        if all(abs(p.ground_tilt_east - plane.tilt_east) < 1e-6
               and abs(p.ground_tilt_north - plane.tilt_north) < 1e-6 for p in poses.values()):
            return
        solved = (int(time.time()), plane.inliers)
        for camera in self.store.cameras():
            pose = camera["pose"]
            if pose is None:
                continue
            revised = tilted(replace(pose, ground_tilt_east=plane.tilt_east,
                                     ground_tilt_north=plane.tilt_north), plane)
            self.store.save_camera(camera["id"], camera["name"], camera["source"],
                                   credentials_ref=camera["credentials_ref"], pose=revised,
                                   record=camera["record"], ground=solved)
            worker = self._workers.get(camera["id"])
            if worker is not None:
                worker.revise_pose(revised)
        self.store.audit("node:" + self.node_id, "site.ground_solved", None, plane.describe())
        _log.info("ground written back to %d camera(s): %s", len(poses), plane.describe())

    def pairings(self) -> tuple:
        """What two cameras agreed on, most recently. Empty when nothing
        overlaps, which is the common case for a single-camera site."""
        return () if self._geometry is None else self._geometry.pairings()

    def ground_plane(self):
        """The site's own ground, once enough of it has been observed, or
        `None` while every projection is still assuming a level yard."""
        return None if self._geometry is None else self._geometry.ground

    def correlate(self) -> tuple[Incident, ...]:
        """Group the recent past into incidents. Bounded, and measured from the events.

        Older incidents are not re-derived; they are already in the store and
        `Store.incidents` reads them. What this returns is what is live.
        """
        self._last_correlated = time.monotonic()
        zones = {z.id: z.kind for z in self.store.zones()}
        newest = self.store.newest_event_millis()
        since = None if newest is None else newest - CORRELATION_SPAN_MILLIS
        ceiling = None if self._geometry is None else self._geometry.appearance_ceiling()
        correlator = Correlator(zone_kinds=zones, appearance_ceiling=ceiling)
        self._incidents = tuple(correlator.correlate(self.store.events(since=since)))
        if self._incidents:
            self.store.save_incidents(self._incidents)
        return self._incidents

    # ------------------------------------------------------------- health

    def detector_info(self, camera_id: str) -> DetectorInfo | None:
        """What is drawing one camera's conclusions, or ``None`` when it is not running."""
        worker = self._workers.get(camera_id)
        return getattr(worker, "detector_info", None) if worker is not None else None

    def metrics(self) -> dict:
        """One flat reading of everything worth watching, for a log or a probe."""
        health = self.health()
        return {
            "node": self.node_id,
            "running": self._running,
            "cameras": len(health),
            "live": sum(1 for h in health.values() if h.state == "LIVE"),
            "dark": sum(1 for h in health.values() if h.state == "DARK"),
            "faulted": sum(1 for h in health.values() if h.state == "FAULTED"),
            "recording": sum(1 for h in health.values() if h.recording),
            "frames": sum(h.frames for h in health.values()),
            "fps": round(sum(h.analysis_fps for h in health.values()), 1),
            "dropped": sum(w.stats.dropped_results for w in self._workers.values()),
            "events": self.store.event_count(),
            "incidents": len(self._incidents),
            "alerts_open": len(self.alerts.active()),
        }

    def _log_metrics_if_due(self) -> None:
        now = time.monotonic()
        if self._last_metrics is not None and now - self._last_metrics < METRICS_EVERY_SECONDS:
            return
        self._last_metrics = now
        reading = self.metrics()
        _log.info("metrics: %d camera(s), %d live, %.0f fps, %d frames, %d event(s), %d incident(s), %d alert(s)",
                  reading["cameras"], reading["live"], reading["fps"], reading["frames"],
                  reading["events"], reading["incidents"], reading["alerts_open"], extra=reading)

    def health(self) -> dict[str, CameraHealth]:
        out = {}
        for camera in self.store.cameras():
            worker = self._workers.get(camera["id"])
            out[camera["id"]] = self._health_for(camera, worker)
        return out

    def _detect_every(self) -> int:
        """How often this site looks, read at start rather than at build time.

        Off the site's own settings, like the watch list and the threshold —
        a service started at boot has nobody to type a flag at it, which is
        the lesson migration 4 was written for.
        """
        factory = self._detector_factory
        settings = getattr(factory, "settings", None)
        return max(1, int(getattr(settings, "detect_every", 1) or 1))

    @staticmethod
    def _health_for(camera: dict, worker: CameraWorker | None) -> CameraHealth:
        if worker is None:
            return CameraHealth(camera["id"], "STOPPED", False, camera["pose"] is not None,
                                0.0, 0, None, None, False, None, 0)
        running = worker.alive
        since_frame = worker.seconds_since_frame()
        since_start = worker.seconds_since_started()
        fault = worker.stats.fault
        if fault:
            state = "FAULTED"
        elif not running:
            state = "STOPPED"
        else:
            silent = since_frame if since_frame is not None else since_start
            if silent is not None and silent >= DARK_AFTER_SECONDS:
                state = "DARK"
            elif since_frame is None:
                state = "STARTING"
            else:
                state = "LIVE"
        fps = worker.stats.analysis_fps if running and since_frame is not None and since_frame <= 1.0 else 0.0
        recording = ((camera["record"] or worker._record_anyway) and running
                     and worker.stats.recording_fault is None and worker._record_to is not None)
        # One bad frame is a bad frame; a hundred and fifty is a lens somebody
        # has to go and clean. A camera in this state is producing frames at a
        # healthy rate, which is why nothing before this noticed.
        degraded = (worker.stats.quality_fault
                    if worker.stats.unusable_frames >= DEGRADED_AFTER_FRAMES else None)
        return CameraHealth(camera["id"], state, running, camera["pose"] is not None, fps, worker.stats.frames, fault,
                            since_frame, bool(recording), worker.stats.recording_fault, worker.stats.clips,
                            worker.stats.quality, degraded)

    def _watch_for_alerts(self) -> None:
        for camera_id, health in self.health().items():
            if health.state == "DARK":
                self.alerts.raise_(CAMERA_DARK, camera_id, health.describe())
            elif health.state in ("LIVE", "STOPPED"):
                self.alerts.clear(CAMERA_DARK, camera_id)
            if health.degraded:
                self.alerts.raise_(CAMERA_DEGRADED, camera_id, health.degraded)
            elif health.state in ("LIVE", "STOPPED"):
                self.alerts.clear(CAMERA_DEGRADED, camera_id)
            if health.recording_fault:
                self.alerts.raise_(RECORDING_STOPPED, camera_id, health.recording_fault)
            elif health.recording:
                self.alerts.clear(RECORDING_STOPPED, camera_id)
        if self.retention_shortfall:
            self.alerts.raise_(RETENTION_SHORTFALL, self.node_id, self.retention_shortfall)
        else:
            self.alerts.clear(RETENTION_SHORTFALL, self.node_id)
        free = self._free_recording_bytes()
        if free is not None and free < DISK_WATERMARK_BYTES:
            self.alerts.raise_(DISK_LOW, self.node_id, f"{free / 1024**3:.1f} GiB free where recordings go, below the {DISK_WATERMARK_BYTES // 1024**3} GiB watermark")
        elif free is not None:
            self.alerts.clear(DISK_LOW, self.node_id)

    def _free_recording_bytes(self) -> float | None:
        if self._record_to is None:
            return None
        probe = Path(self._record_to)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            return float(shutil.disk_usage(probe).free)
        except OSError:
            return None

    # ---------------------------------------------------------- retention

    def _sweep_retention_if_due(self) -> None:
        if self._retention is None:
            return
        now = time.monotonic()
        if self._last_retention is not None and now - self._last_retention < self._retention_every:
            return
        self._last_retention = now
        try:
            self.retention_shortfall = apply_retention(self.store, self._retention, principal=f"node:{self.node_id}")
        except Exception:  # noqa: BLE001
            _log.exception("retention sweep failed")



def apply_retention(store: Store, policy: RetentionPolicy, *, principal: str = "retention",
                    now_millis: int | None = None) -> str | None:
    """Delete the oldest unpreserved clips until the policy is met; the shortfall if it cannot be."""
    now = now_millis if now_millis is not None else int(time.time() * 1000)
    everything = store.segments()
    preserved = store.preserved_paths()
    candidates = sorted((s for s in everything if str(s.path) not in preserved), key=lambda s: s.started_millis)
    total = store.recorded_bytes()
    free = _free_bytes(everything)

    def over_budget() -> bool:
        if policy.max_bytes is not None and total > policy.max_bytes:
            return True
        return policy.min_free_bytes is not None and free < policy.min_free_bytes

    for segment in candidates:
        too_old = policy.max_age_days is not None and now - segment.ended_millis > policy.max_age_days * 86_400_000
        if not too_old and not over_budget():
            continue
        try:
            segment.path.unlink(missing_ok=True)
        except OSError as error:
            _log.warning("could not delete %s: %s", segment.path, error)
            continue
        store.forget_segment(segment.path)
        store.audit(principal, "recording.deleted", str(segment.path), f"{segment.camera_id}, {segment.size_bytes / 1048576:.1f} MiB")
        total -= segment.size_bytes
        free += segment.size_bytes
    if over_budget():
        kept = len(everything) - len(candidates)
        shortfall = (f"retention could not reach its target: {total / 1024**3:.1f} GiB recorded, {free / 1024**3:.1f} GiB free, "
                     f"{kept} clip(s) preserved as evidence and not eligible for deletion")
        _log.error("%s", shortfall)
        return shortfall
    return None


def _free_bytes(segments: Sequence[Segment]) -> float:
    for segment in segments:
        try:
            return float(shutil.disk_usage(segment.path.parent).free)
        except OSError:
            continue
    return float("inf")
