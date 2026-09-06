"""One camera's thread: decode, detect, track, judge, record.

Split from `runtime` at the line budget, and the seam is a real one. A worker
owns exactly one camera and **never touches the store**; the runtime owns the
store's thread and drains the workers' outboxes. Everything below runs on a
thread that has no database handle, which is what makes that rule enforceable
rather than merely stated.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..adapters.decode import DecodeError, Frame, LiveReader, VideoSource
from ..adapters.detectors import Detector
from ..adapters.recorder import Recorder, Segment
from ..domain.detection import DetectorInfo
from ..domain.events import Event, Rule, RuleContext, default_rules
from ..domain.relations import RelationTracker
from ..domain.tracking import Track, Tracker, TrackerConfig
from ..domain.zones import PresenceTracker, Zone
from ..logs import get as _get_logger
from ..adapters.detectors import WATCHED_LABELS
from ..domain.appearance import ColourBalance
from ..perception.appearance import describe
from ..perception.motion import CameraMotion, CameraMotionEstimator
from ..perception.quality import FrameQuality, FrameQualityMonitor
from .site import Camera

_log = _get_logger(__name__)

#: Consecutive unusable frames before a camera is called degraded. Ten
#: seconds at 15 fps: one bad frame is a bad frame, a hundred and fifty is
#: a lens somebody has to go and clean.
DEGRADED_AFTER_FRAMES = 150
OUTBOX_EVENTS = 1000
STOP_TIMEOUT_SECONDS = 8.0


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
    #: Track id to that track's colour descriptor, normalised for this camera
    #: so another camera's is comparable with it. Carried on the result rather
    #: than recomputed downstream: the camera's colour statistics are measured
    #: here, and a second running average of the same pictures is a second
    #: answer waiting to differ from this one.
    appearance: dict = field(default_factory=dict)


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
                 record_anyway: bool = False, detect_every: int = 1,
                 watch: frozenset[str] | None = None):
        self.camera = camera
        #: This camera's own colour bias, so its objects can be compared with
        #: another camera's. Owned here because it is a property of the camera
        #: and this is the one thing per camera; `service.triangulation` and
        #: the correlator both read what this produces rather than each
        #: keeping their own running average of the same pictures.
        self._balance = ColourBalance()
        #: The site's live map, when one is being built. Set by the runtime
        #: after construction, because a worker may exist before a site has
        #: two placed cameras and an origin to hang a lattice on.
        self.live_map = None
        #: A pose the runtime has revised — today, the ground tilt it solved.
        #: Read and cleared at the top of the loop rather than applied where
        #: it is set: a bare assignment between threads is safe under the GIL,
        #: and reaching into another thread's tracker is not.
        self._revised_pose = None
        self._url = source_url
        self._detector_factory = detector_factory
        #: Labels that may **raise an event**. Not what the detector looks
        #: for: it looks for everything its model knows, and everything it
        #: finds is tracked, drawn and mapped. This governs only what reaches
        #: a rule.
        #:
        #: The two were one frozen set of six, and that cost recall on both
        #: sides. A trailer, a dog or a ladder against a fence was invisible
        #: to the tracker, to the plan and to the map, because the only thing
        #: that could have seen it was told not to look. Splitting them is
        #: what "detect more" means without reintroducing the defect the six
        #: were chosen to prevent — v1's operator screenshot of a bottle
        #: raising an alarm.
        self._watch = frozenset(watch) if watch else None
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

    def _normalised(self, tracks) -> dict:
        """Each track's newest look, in the shared cross-camera colour space.

        The camera's own bias is measured from the same descriptors it is
        applied to, which is safe because the bias is what *every* object here
        has in common and no single object moves it far. It is applied only
        once enough has been seen; until then the descriptor is passed through
        unchanged rather than corrected by a guess.
        """
        out: dict[int, tuple[float, ...]] = {}
        for track in tracks:
            gallery = getattr(track, "gallery", None)
            look = gallery.newest() if gallery is not None else None
            if look is None:
                continue
            self._balance.observe(look)
            out[track.id] = tuple(float(v) for v in self._balance.normalise(look).vector)
        return out

    def _watched(self, tracks, info) -> list:
        """The tracks whose label this site alerts on.

        An empty or absent watch list means the built-in one, not "everything":
        a site that has never been configured must not raise an event for a
        chair, and the difference between "not set" and "set to nothing" is
        not one an operator can see on a screen at three in the morning.

        A detector that does not classify is exempt. Motion has no label to
        watch, and filtering its tracks against a list of words it can never
        produce silences the camera completely — which is what happened the
        first time this gate was written.
        """
        if not info.classifies:
            return list(tracks)
        watch = self._watch or WATCHED_LABELS
        return [t for t in tracks if (info.label_for(t.class_id) or "").lower() in watch]

    def revise_pose(self, pose) -> None:
        """Hand the worker a corrected pose. Picked up on its next frame."""
        self._revised_pose = pose

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
            # A tiled detector needs to know where the far ground is, and only
            # the pose can say. Without one it falls back to the upper middle
            # of the frame, which is where the distance is in nearly every
            # fixed view — a fallback, not geometry, and it says so.
            aim = getattr(detector, "set_pose", None)
            if callable(aim):
                aim(self.camera.pose)
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
                revised, self._revised_pose = self._revised_pose, None
                if revised is not None:
                    # Applied here rather than where it was set, because the
                    # tracker belongs to this thread and the runtime that
                    # solved the ground does not run on it.
                    tracker.set_pose(revised)
                    aim = getattr(detector, "set_pose", None)
                    if callable(aim):
                        aim(revised)
                    _log.info("%s: ground tilt applied to the pose (%.1f%% east, %.1f%% north)",
                              self.camera.id, revised.ground_tilt_east * 100,
                              revised.ground_tilt_north * 100)
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
        looks = self._normalised(tracks)
        # Everything above is about every object the model found. From here
        # down only the watched ones are considered, because from here down
        # things raise alarms.
        watched = self._watched(tracks, info)
        for change in presence.update(watched, frame.timestamp_millis, label_for=info.label_for):
            zone = zones[change.presence.zone_id]
            track = by_id.get(change.presence.track_id)
            context = RuleContext(self._node_id, self.camera.id, zone, track, change.presence, frame.timestamp_millis,
                                  moment, info, frame.index, self._site_tz,
                                  _for(found, change.presence.track_id), names, looks)
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
                                  moment, info, frame.index, self._site_tz, _for(found, relation.subject),
                                  names, looks)
            for rule in self._rules:
                events.extend(rule.on_relation(relation, context))

        for p in presence.presences():
            track = by_id.get(p.track_id)
            if track is None:
                continue
            context = RuleContext(self._node_id, self.camera.id, zones[p.zone_id], track, p, frame.timestamp_millis,
                                  moment, info, frame.index, self._site_tz, _for(found, p.track_id),
                                  names, looks)
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
        if self.live_map is not None and self.camera.pose is not None:
            # The frame is here, decoded, and about to be dropped. Offering it
            # to the map costs a rate-limit check on all but one frame in
            # `DEFAULT_SAMPLE_INTERVAL_S`, and the alternative is opening
            # every camera a second time to build a map of one afternoon.
            self.live_map.observe(self.camera.id, tracker.pose or self.camera.pose,
                                  frame.image, frame.timestamp_millis / 1000.0)
        result = FrameResult(self.camera.id, frame.index, frame.timestamp_millis, tuple(tracks), len(detections),
                             frame.image if self._keep_images else None, found, measured, motion, looks)
        with self._latest_lock:
            if self._latest is not None:
                self.stats.dropped_results += 1
            self._latest = result

def _for(relations: tuple, track_id: int) -> tuple:
    """The relations one track takes part in, either end."""
    return tuple(r for r in relations if r.subject == track_id or r.object == track_id)
