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
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class FrameResult:
    camera_id: str
    frame_index: int
    at_millis: int
    tracks: tuple[Track, ...]
    detections: int
    image: object | None = None
    #: What those tracks are doing with each other, as far as one camera can
    #: tell. Inferred, never observed; see `vigil.domain.relations`.
    relations: tuple = ()
    #: What the frame itself was worth. A console that draws boxes over a
    #: frame nothing could be detected in should say so.
    quality: object = None
    #: How the camera moved into this frame, when it could be measured.
    camera_motion: object = None


@dataclass
class WorkerStats:
    frames: int = 0
    #: Frames the detector actually ran on. Below `frames` when the site is
    #: detecting on a subset and tracking through the rest.
    detected_frames: int = 0
    detections: int = 0
    events: int = 0
    dropped_results: int = 0
    analysis_fps: float = 0.0
    #: Rolling frame quality, 0..1, or `None` before the first frame.
    quality: float | None = None
    #: Why the most recent frame was unusable, or `None`.
    quality_fault: str | None = None
    #: Consecutive unusable frames.
    unusable_frames: int = 0
    #: Frames on which the camera itself measurably moved.
    moved_frames: int = 0
    last_frame_at: float | None = None
    started_at: float | None = None
    fault: str | None = None
    recording_fault: str | None = None
    clips: int = 0


class CameraWorker:
    def __init__(self, camera: Camera, source_url: str, detector_factory: Callable[[], Detector], zones: Sequence[Zone],
                 rules: Sequence[Rule] | None = None, *, node_id: str = "local", record_to: Path | None = None,
                 realtime: bool = False, keep_images: bool = False, site_tz=None, segment_seconds: float = 60.0,
                 record_anyway: bool = False, detect_every: int = 1):
        self.camera = camera
        self._url = source_url
        self._detector_factory = detector_factory
        self._zones = list(zones)
        self._rules = list(rules) if rules is not None else default_rules()
        self._node_id = node_id
        self._record_to = record_to
        self._realtime = realtime
        self._keep_images = keep_images
        self._site_tz = site_tz
        self._segment_seconds = segment_seconds
        #: This run records whatever the camera's stored flag says. Set by
        #: `--record`, which would otherwise set a destination and record
        #: nothing — which is what happened, and what nobody was told.
        self._record_anyway = record_anyway
        #: Run the detector on one frame in this many and track through the
        #: rest. See `vigil.service.detection.MAX_DETECT_EVERY` for the
        #: measurement; 1 is every frame.
        self._detect_every = max(1, int(detect_every))
        self.stats = WorkerStats()
        self.detector_info: DetectorInfo | None = None
        self._latest: FrameResult | None = None
        self._latest_lock = threading.Lock()
        self._events: queue.Queue[Event] = queue.Queue(maxsize=OUTBOX_EVENTS)
        self._segments: queue.Queue[Segment] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recent: list[float] = []

    # ------------------------------------------------------------ control

    def start(self) -> None:
        self._stop.clear()
        self.stats.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"vigil-camera-{self.camera.id}", daemon=True)
        self._thread.start()

    def ask_to_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------- outbox

    def take_latest(self) -> FrameResult | None:
        with self._latest_lock:
            result, self._latest = self._latest, None
            return result

    def take_events(self) -> list[Event]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def take_segments(self) -> list[Segment]:
        out = []
        while True:
            try:
                out.append(self._segments.get_nowait())
            except queue.Empty:
                return out

    def seconds_since_frame(self) -> float | None:
        return None if self.stats.last_frame_at is None else time.monotonic() - self.stats.last_frame_at

    def seconds_since_started(self) -> float | None:
        return None if self.stats.started_at is None else time.monotonic() - self.stats.started_at

    # --------------------------------------------------------------- work

    def _run(self) -> None:
        source = VideoSource(self._url, source_id=self.camera.id)
        reader: LiveReader | None = None
        recorder: Recorder | None = None
        try:
            detector = self._detector_factory()
            self.detector_info = detector.info
            tracker = Tracker(TrackerConfig(), self.camera.pose)
            presence = PresenceTracker(self._zones)
            relations = RelationTracker()
            quality = FrameQualityMonitor()
            camera_motion = CameraMotionEstimator()
            info = source.open()
            if self._record_to is not None and (self.camera.record or self._record_anyway):
                recorder = Recorder(self.camera.id, self._record_to, fps=info.nominal_fps or 15.0, segment_seconds=self._segment_seconds)
                try:
                    recorder.start()
                except OSError as error:
                    self.stats.recording_fault = str(error)
                    recorder = None
            if source.live:
                reader = LiveReader(source)
                reader.start()
            frame_interval = 1.0 / info.nominal_fps if (self._realtime and info.nominal_fps > 0) else 0.0
            while not self._stop.is_set():
                started = time.monotonic()
                frame = reader.read(timeout=1.0) if reader is not None else source.read()
                if frame is None:
                    if reader is not None:
                        if reader.fault and not reader.alive:
                            raise DecodeError(reader.fault)
                        continue
                    break  # a file ended
                self._process(frame, detector, tracker, presence, relations, recorder,
                              quality, camera_motion)
                if frame_interval:
                    remaining = frame_interval - (time.monotonic() - started)
                    if remaining > 0:
                        self._stop.wait(remaining)
            # Every live track ends with the camera. The departures this
            # produces are dropped on purpose: no rule acts on leaving, and
            # inventing an event for "the camera stopped" would put a
            # conclusion in the trail that nothing observed.
            for track_id in tracker.reset():
                presence.forget_track(track_id, int(time.time() * 1000))
        except DecodeError as error:
            self.stats.fault = str(error)
            _log.error("%s: %s", self.camera.id, error)
        except Exception as error:  # noqa: BLE001 - a worker must not die silently
            self.stats.fault = f"{type(error).__name__}: {error}"
            _log.exception("%s: analysis failed", self.camera.id)
        finally:
            if reader is not None:
                reader.stop()
            if recorder is not None:
                for segment in recorder.close():
                    self._segments.put(segment)
                if recorder.stats.fault:
                    self.stats.recording_fault = recorder.stats.fault
            source.close()
            _log.info("%s: analysis finished: %d frames, %d detections, %d events", self.camera.id,
                      self.stats.frames, self.stats.detections, self.stats.events)

    def _process(self, frame: Frame, detector: Detector, tracker: Tracker, presence: PresenceTracker,
                 relations: RelationTracker, recorder: Recorder | None,
                 quality: FrameQualityMonitor, camera_motion: CameraMotionEstimator) -> None:
        now = time.monotonic()
        self.stats.frames += 1
        self.stats.last_frame_at = now
        self._recent = [t for t in self._recent if now - t <= 1.0] + [now]
        self.stats.analysis_fps = float(len(self._recent))

        # What this frame is worth, before anything is asked of it. A
        # detector reports how sure it is *given the pixels it was shown*
        # and has no way to say the lens is dirty.
        measured = quality.measure(frame.image)
        self.stats.quality = quality.recent_score
        # Either kind of degradation counts here: an unusable image and a
        # frozen stream are both cameras that will never report anything, and
        # both were invisible to a frame counter.
        self.stats.quality_fault = measured.degraded
        self.stats.unusable_frames = 0 if measured.degraded is None else self.stats.unusable_frames + 1

        # How the camera moved. Not attempted on a frame nothing can be
        # measured in: optical flow over a blown-out frame returns a
        # confident transform built from points that matched nothing.
        motion: CameraMotion | None = None
        if measured.usable:
            motion = camera_motion.estimate(frame.image)
            if motion.measured and not motion.still:
                self.stats.moved_frames += 1
        else:
            camera_motion.reset()
        warp = motion.warp if (motion is not None and motion.measured and not motion.still) else None

        # Detect on one frame in `detect_every` and track through the rest.
        # The tracker carries the gap: it predicts with a Kalman filter rather
        # than extrapolating an average, so a skipped frame widens the gate by
        # the right amount instead of by whatever the frame counter did.
        detections: list = []
        looks: list = []
        if self.stats.frames % self._detect_every == 0:
            detections = detector.detect(frame.image)
            self.stats.detected_frames += 1
            self.stats.detections += len(detections)
            # An appearance per detection: what stops one person becoming
            # eleven objects the moment the detector blinks.
            looks = [describe(frame.image, (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height), d.mask)
                     for d in detections]
        update = tracker.update(detections, frame.timestamp_millis, appearances=looks, warp=warp)
        tracks = tracker.tracks()
        moment = datetime.fromtimestamp(frame.timestamp_millis / 1000, tz=timezone.utc)
        events: list[Event] = []
        info = detector.info
        for ended in update.ended:
            # A track that ended has left every zone it was in; see above for
            # why the departures are not turned into events.
            presence.forget_track(ended, frame.timestamp_millis)
            relations.forget_track(ended)
            for rule in self._rules:
                forget = getattr(rule, "forget", None)
                if forget:
                    forget(ended)
        by_id = {t.id: t for t in tracks}
        zones = presence.zones
        # What the tracks are doing with each other, before any rule looks at
        # them: a rule may say "carrying" only if this measured it.
        found = tuple(relations.update(tracks, frame.timestamp_millis, label_of=info.label_for,
                                       zones=list(zones.values())))
        names = {t.id: info.label_for(t.class_id) for t in tracks}
        for change in presence.update(tracks, frame.timestamp_millis, label_for=info.label_for):
            zone = zones[change.presence.zone_id]
            track = by_id.get(change.presence.track_id)
            context = RuleContext(self._node_id, self.camera.id, zone, track, change.presence, frame.timestamp_millis,
                                  moment, info, frame.index, self._site_tz,
                                  _for(found, change.presence.track_id), names)
            for rule in self._rules:
                events.extend(rule.on_presence_change(change, context))
        # A relation is the only way a rule hears about something that has
        # not arrived yet, so it is offered before presence is considered.
        for relation in found:
            track = by_id.get(relation.subject)
            if track is None:
                continue
            zone = zones.get(relation.zone_id) if relation.zone_id else None
            context = RuleContext(self._node_id, self.camera.id, zone, track, None, frame.timestamp_millis,
                                  moment, info, frame.index, self._site_tz, _for(found, relation.subject), names)
            for rule in self._rules:
                events.extend(rule.on_relation(relation, context))

        for p in presence.presences():
            track = by_id.get(p.track_id)
            if track is None:
                continue
            context = RuleContext(self._node_id, self.camera.id, zones[p.zone_id], track, p, frame.timestamp_millis,
                                  moment, info, frame.index, self._site_tz, _for(found, p.track_id), names)
            for rule in self._rules:
                events.extend(rule.on_frame(context))
        for event in events:
            try:
                self._events.put_nowait(event)
                self.stats.events += 1
            except queue.Full:
                self.stats.dropped_results += 1
        if recorder is not None:
            try:
                recorder.write(frame.image, frame.timestamp_millis)
                for segment in recorder.take_closed():
                    self._segments.put(segment)
                    self.stats.clips += 1
            except OSError as error:
                self.stats.recording_fault = str(error)
                _log.error("%s: RECORDING STOPPED EARLY - %s", self.camera.id, error)
        result = FrameResult(self.camera.id, frame.index, frame.timestamp_millis, tuple(tracks), len(detections),
                             frame.image if self._keep_images else None, found, measured, motion)
        with self._latest_lock:
            if self._latest is not None:
                self.stats.dropped_results += 1
            self._latest = result


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
                 record_every_camera: bool = False):
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
                detect_every=self._detect_every(),
            )
            worker.start()
            self._workers[camera.id] = worker
            started += 1
        self._running = started > 0
        self.store.audit(by.actor, "analysis.started", self.node_id, f"{started} camera(s)")
        return started

    def stop(self, by: Principal) -> bool:
        by.require(ANALYSIS_CONTROL)
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

    def correlate(self) -> tuple[Incident, ...]:
        """Group the recent past into incidents. Bounded, and measured from the events.

        Older incidents are not re-derived; they are already in the store and
        `Store.incidents` reads them. What this returns is what is live.
        """
        self._last_correlated = time.monotonic()
        zones = {z.id: z.kind for z in self.store.zones()}
        newest = self.store.newest_event_millis()
        since = None if newest is None else newest - CORRELATION_SPAN_MILLIS
        correlator = Correlator(zone_kinds=zones)
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


def _for(relations: tuple, track_id: int) -> tuple:
    """The relations one track takes part in, either end."""
    return tuple(r for r in relations if r.subject == track_id or r.object == track_id)


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
