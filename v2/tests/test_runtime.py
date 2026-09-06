import time
from collections import namedtuple
from pathlib import Path

import pytest

from vigil.adapters.detectors import MotionDetector
from vigil.domain.events import EventType
from vigil.domain.geo import CameraPose, LatLon, destination_point, project_to_ground
from vigil.domain.zones import ZoneKind
from vigil.service.alerts import CAMERA_DARK, DISK_LOW, RECORDING_STOPPED, RETENTION_SHORTFALL, THREAD_STUCK, Alerts
from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.runtime import DARK_AFTER_SECONDS, CameraWorker, RetentionPolicy, Runtime
from vigil.service.site import SiteService
from vigil.storage.store import Store

OPERATOR = Principal("alice", Role.OPERATOR, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


def _zone_where_the_block_stops(pose: CameraPose):
    """The block ends at the frame centre, bottom around v ≈ 0.92 for a 240-px frame with the box bottom at 220."""
    landing = project_to_ground(pose, 0.55, 0.92, enforce_range=False)
    assert landing is not None
    centre = landing.position
    return [destination_point(centre, b, 4.0) for b in (45, 135, 225, 315)]


def _pump(runtime: Runtime, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while runtime.running and time.monotonic() < deadline:
        runtime.poll()
        time.sleep(0.05)


def test_a_file_through_the_whole_spine_produces_an_event_and_an_incident(tmp_path, reference_video, keychain, pose):
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", str(reference_video), pose=pose, by=OPERATOR)
        site.add_zone("yard", "Yard", ZoneKind.RESTRICTED, _zone_where_the_block_stops(pose), enter_after_millis=300, by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
        assert runtime.start(OPERATOR) == 1
        _pump(runtime, 30)
        runtime.stop(OPERATOR)
        events = store.events()
        assert events, "the block walked into the yard and nothing was said"
        assert events[0].type is EventType.ZONE_ENTRY and events[0].summary.startswith("An object")
        assert events[0].evidence.latitude is not None
        incidents = runtime.incidents
        assert len(incidents) >= 1 and incidents[0].summary.startswith("1 object")
        assert store.incidents()[0].id == incidents[0].id
        actions = [r["action"] for r in store.audit_trail()]
        assert "analysis.started" in actions and "analysis.stopped" in actions
        health = runtime.health()["gate"]
        assert health.state == "STOPPED" and health.frames >= 80


def test_a_viewer_cannot_start_the_analysis(tmp_path, reference_video, keychain):
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", str(reference_video), by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector)
        with pytest.raises(Forbidden):
            runtime.start(VIEWER)


def test_recording_writes_clips_the_store_indexes(tmp_path, reference_video, keychain):
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", str(reference_video), record=True, by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector, record_to=tmp_path / "rec", alerts=Alerts(synchronous=True))
        runtime.start(OPERATOR)
        _pump(runtime, 30)
        runtime.stop(OPERATOR)
        segments = store.segments(camera_id="gate")
        assert segments and all(s.path.is_file() for s in segments)
        assert runtime.health()["gate"].clips + 1 >= len(segments)


class _Stalled:
    """Stands in for a worker whose thread is alive and whose decoder has died."""

    def __init__(self, silent_for, since_frame=None, recording_fault=None, stops=True):
        from vigil.service.runtime import WorkerStats

        self.stats = WorkerStats(started_at=time.monotonic() - silent_for, last_frame_at=None if since_frame is None else time.monotonic() - since_frame)
        self.stats.recording_fault = recording_fault
        self.alive = True
        self._record_to = Path(".")
        self._stops = stops

    def seconds_since_frame(self):
        return None if self.stats.last_frame_at is None else time.monotonic() - self.stats.last_frame_at

    def seconds_since_started(self):
        return time.monotonic() - self.stats.started_at

    def take_latest(self):
        return None

    def take_events(self):
        return []

    def take_segments(self):
        return []

    def ask_to_stop(self):
        pass

    def stop(self, timeout=None):
        return self._stops


def test_dark_cameras_stopped_recordings_stuck_threads_and_a_full_disk_alert_and_clear(tmp_path, keychain, monkeypatch):
    import shutil

    usage = namedtuple("usage", "total used free")
    free = {"value": 100 * 1024**2}
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 0, free["value"]))
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", "rtsp" + "://10.0.0.9/s", record=True, by=OPERATOR)
        runtime = Runtime(site, record_to=tmp_path / "rec", retention=RetentionPolicy(max_age_days=None, max_bytes=None, min_free_bytes=1024**3),
                          retention_every_seconds=0.0, alerts=Alerts(synchronous=True))
        runtime._workers["gate"] = _Stalled(DARK_AFTER_SECONDS * 2, recording_fault="disk full", stops=False)
        runtime._running = True
        runtime.poll()
        keys = {a.key for a in runtime.alerts.active()}
        assert (CAMERA_DARK, "gate") in keys and (RECORDING_STOPPED, "gate") in keys and (DISK_LOW, "local") in keys
        assert runtime.alerts.raised_count == 3
        runtime.poll()
        assert runtime.alerts.raised_count == 3, "one condition, one alert"
        runtime._workers["gate"] = _Stalled(DARK_AFTER_SECONDS * 2, since_frame=0.1, stops=False)
        free["value"] = 50 * 1024**3
        runtime.poll()
        keys = {a.key for a in runtime.alerts.active()}
        assert (CAMERA_DARK, "gate") not in keys and (DISK_LOW, "local") not in keys
        assert runtime.stop(OPERATOR) is False
        assert (THREAD_STUCK, "gate") in {a.key for a in runtime.alerts.active()}
        assert {"alert.raised", "alert.cleared", "analysis.thread_stuck"} <= {r["action"] for r in store.audit_trail()}
        assert len(store.open_alerts()) >= 1


def test_a_worker_reports_a_missing_file_as_a_fault_not_a_crash(tmp_path, keychain):
    from vigil.service.site import Camera

    camera = Camera("ghost", "Ghost", str(tmp_path / "none.mp4"), None, None, False)
    worker = CameraWorker(camera, camera.source, MotionDetector, [])
    worker.start()
    assert worker.stop()
    assert worker.stats.fault and "no file" in worker.stats.fault
