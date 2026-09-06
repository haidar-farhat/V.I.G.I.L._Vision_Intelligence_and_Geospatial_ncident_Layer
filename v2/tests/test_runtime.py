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


def test_the_record_flag_on_a_run_records_a_camera_the_site_has_not_flagged(tmp_path, reference_video, keychain):
    """`--record` set a destination and then recorded nothing, silently. Found by exporting a live incident."""
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", str(reference_video), record=False, by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector, record_to=tmp_path / "rec",
                          record_every_camera=True, alerts=Alerts(synchronous=True))
        runtime.start(OPERATOR)
        # Read while it is running: "is this recording" is only true of the
        # instant it is asked, and the file ends part way through the pump.
        said_recording = False
        deadline = time.monotonic() + 30
        while runtime.running and time.monotonic() < deadline:
            runtime.poll()
            said_recording = said_recording or runtime.health()["gate"].recording
            time.sleep(0.05)
        assert said_recording, "the health never said it was recording"
        runtime.stop(OPERATOR)
        segments = store.segments(camera_id="gate")
        assert segments and all(s.path.is_file() for s in segments)
        assert not site.camera("gate", OPERATOR).record, "an override for one run must not change the site"


def test_an_unattended_run_says_what_it_is_doing_in_the_log_and_as_json(tmp_path, reference_video, keychain, caplog, monkeypatch):
    """Nobody is reading the screen; the log is the only place this can be seen afterwards."""
    import json
    import logging

    from vigil import logs
    from vigil.service import runtime as runtime_module

    caplog.set_level(logging.INFO, logger="vigil")
    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", str(reference_video), by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
        assert runtime.metrics()["cameras"] == 1 and runtime.metrics()["running"] is False
        monkeypatch.setattr(runtime_module, "METRICS_EVERY_SECONDS", 0.0)
        runtime.start(OPERATOR)
        _pump(runtime, 10)
        runtime.stop(OPERATOR)
        reading = runtime.metrics()
        assert reading["frames"] >= 0 and set(reading) >= {"node", "live", "dark", "fps", "events", "alerts_open"}
        assert any("metrics:" in r.getMessage() for r in caplog.records), "an unattended run said nothing"

    # The real path: `configure(json=True)` writes one object per line to the
    # log file, and restoring prose afterwards leaves the suite as it was.
    written = logs.configure(tmp_path / "jsonlogs", json=True)
    logging.getLogger("vigil.test").info("a line", extra={"frames": 7})
    for handler in logging.getLogger("vigil").handlers:
        handler.flush()
    first = [l for l in written.read_text(encoding="utf-8").splitlines() if l.strip()][0]
    assert json.loads(first)["message"] == "a line" and json.loads(first)["frames"] == 7
    logs.configure(None)

    line = logs._Json().format(logging.LogRecord("vigil.t", logging.INFO, "f", 1, "metrics: %d", (2,), None))
    assert json.loads(line)["message"] == "metrics: 2" and json.loads(line)["level"] == "INFO"
    secret = logging.LogRecord("vigil.t", logging.INFO, "f", 1, "opened rtsp" + "://u:hunter2@10.0.0.9/s", (), None)
    assert "hunter2" not in logs._Json().format(secret), "the json form must redact what the prose form does"


def test_correlation_is_bounded_so_a_node_that_has_been_up_for_a_month_keeps_up(tmp_path, keychain, monkeypatch):
    """Re-reading every event ever stored, every two seconds, does not scale."""
    from test_incidents import event as make_event
    from vigil.service.runtime import CORRELATION_SPAN_MILLIS

    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
        old = [make_event("a", 1, 1_000), make_event("a", 2, 2_000)]
        recent = [make_event("b", 3, CORRELATION_SPAN_MILLIS * 3), make_event("b", 4, CORRELATION_SPAN_MILLIS * 3 + 1000)]
        store.save_events(old + recent)

        read = []
        real = Store.events
        monkeypatch.setattr(Store, "events", lambda self, **kw: read.append(kw) or real(self, **kw))
        live = runtime.correlate()
        assert read and read[0]["since"] is not None, "the whole history was read"
        assert all(e.evidence.camera_id == "b" for i in live for e in i.events), "old events were re-correlated"
        # The old ones are still there; they are simply not re-derived.
        assert len(store.events()) == 4
        assert store.incidents(), "the live incidents were persisted"


def test_one_incident_is_read_without_loading_every_incident(tmp_path, keychain):
    from test_incidents import event as make_event
    from vigil.domain.incidents import Correlator

    with Store(tmp_path / "r.db") as store:
        events = [make_event("a", 1, 10_000), make_event("a", 2, 11_000)]
        store.save_events(events)
        incident = Correlator().correlate(events)[0]
        store.save_incidents([incident])
        found = store.incident(incident.id)
        assert found is not None and found.id == incident.id and len(found.events) == len(incident.events)
        assert store.incident("inc-nothing") is None


def test_a_schedule_is_read_in_the_sites_own_clock_not_the_meridians(tmp_path, keychain):
    """"Closed 22:00 to 06:00" means the site's night. Evaluated in UTC it fires at the wrong hours."""
    from datetime import timezone
    from zoneinfo import ZoneInfo

    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
        assert runtime.site_timezone() is timezone.utc, "an unnamed site is UTC"

        site.name_site("Depot", "Asia/Beirut", by=OPERATOR)
        assert runtime.site_timezone() == ZoneInfo("Asia/Beirut")

        # An unknown zone is refused where it is typed. Only a database that
        # went missing after the fact can reach the fall-back below.
        from vigil.service.site import SiteError

        with pytest.raises(SiteError, match="does not know the time zone"):
            site.name_site("Depot", "Mars/Olympus", by=OPERATOR)
        store.save_site("Depot", "Mars/Olympus")
        assert runtime.site_timezone() is timezone.utc, "a zone that went missing must fall back, loudly"


def test_the_workers_are_given_the_sites_clock(tmp_path, reference_video, keychain):
    from zoneinfo import ZoneInfo

    with Store(tmp_path / "r.db") as store:
        site = SiteService(store, keychain)
        site.name_site("Depot", "Asia/Beirut", by=OPERATOR)
        site.add_camera("gate", str(reference_video), by=OPERATOR)
        runtime = Runtime(site, detector_factory=MotionDetector, alerts=Alerts(synchronous=True))
        runtime.start(OPERATOR)
        try:
            assert runtime._workers["gate"]._site_tz == ZoneInfo("Asia/Beirut"), "the rules would read the wrong clock"
        finally:
            runtime.stop(OPERATOR)
