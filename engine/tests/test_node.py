"""Tests for the headless node.

Until `node.py` existed the console *was* the application, and three things were
impossible at once: running unattended on a machine with no display, running
analysis on a worker while an operator watches from elsewhere, and restarting
the interface without stopping the cameras.

The load-bearing assertions here are not about analysis — that is tested to
death elsewhere, and the node runs the same pipeline. They are about the
properties that make it a *node*:

- **It imports no Qt.** The way that guarantee dies is one convenient import,
  and it dies silently: everything still works on the developer's machine.
- **Only the node's thread touches the store.** An SQLite connection belongs to
  the thread that made it. This has now failed twice in two different callers,
  so it is pinned here.
- **Stopping actually stops.** Dropping a reference to a thread that has not
  finished leaves it decoding into a pipeline whose store may already be closed.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sentinel import logs
from sentinel.core import CameraPose, LatLon
from sentinel.node import (
    ACTOR,
    DARK_AFTER_SECONDS,
    CameraRunner,
    CameraState,
    Node,
    NodeError,
)
from sentinel.store import Store
from sentinel.zones import Zone, ZoneKind


@pytest.fixture(autouse=True)
def clean_logging():
    logs.reset()
    yield
    logs.reset()


@pytest.fixture
def yard() -> Zone:
    return Zone(
        id="yard", name="Yard", kind=ZoneKind.RESTRICTED,
        ring=(
            LatLon(33.893736, 35.501800), LatLon(33.893628, 35.501930),
            LatLon(33.893520, 35.501800), LatLon(33.893628, 35.501670),
        ),
        enter_after_millis=600,
    )


@pytest.fixture
def site() -> CameraPose:
    return CameraPose(
        position=LatLon(33.8938, 35.5018), mount_height=6.0, heading=180.0,
        pitch=-22.0, horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
    )


# ------------------------------------------------------- the headless guarantee


def test_the_node_imports_no_qt():
    # In a subprocess, because this test process has almost certainly imported
    # Qt already for the console suite — asserting on `sys.modules` here would
    # pass whatever the node does.
    #
    # The failure this prevents is silent: one convenient import, everything
    # still works on the developer's machine, and the worker node in a cupboard
    # with no display fails to start.
    code = (
        "import sys; import sentinel.node; "
        "bad = sorted(m for m in sys.modules "
        "if m.startswith('PySide') or m.startswith('shiboken')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, f"the node dragged in Qt: {result.stdout.strip()}"


def test_the_whole_engine_imports_no_qt():
    # The same guarantee, one level up. A worker node runs the engine, not the
    # console, and any engine module reaching for Qt breaks that for all of them.
    code = (
        "import sys, pkgutil, importlib, sentinel; "
        # `__main__` IS the CLI — importing it runs it, and it exits 2 asking
        # for a subcommand. Everything else is a library.
        "[importlib.import_module('sentinel.' + m.name) "
        " for m in pkgutil.iter_modules(sentinel.__path__) "
        " if m.name != '__main__']; "
        "bad = sorted(m for m in sys.modules "
        "if m.startswith('PySide') or m.startswith('shiboken')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, f"the engine dragged in Qt: {result.stdout.strip()}"


# --------------------------------------------------------------- configuration


def test_a_node_with_no_zones_still_gets_rules_that_can_fire(tmp_path: Path):
    # Without a zone there is nothing to be inside, so the zone rules would be
    # dead weight — and a run would report "0 events" for a reason that has
    # nothing to do with the footage.
    with Node(tmp_path / "n.db") as node:
        assert node.rules
        assert all("zone" not in type(rule).__name__.lower() for rule in node.rules)


def test_adding_the_first_zone_rebuilds_the_rule_set(tmp_path: Path, yard: Zone):
    # A zone added to a node that had none would otherwise be configured,
    # visible, and silently watched by nothing — the worst state for a security
    # control to be in.
    with Node(tmp_path / "n.db") as node:
        before = len(node.rules)
        node.add_zone(yard)

        assert len(node.rules) > before
        assert any("zone" in type(rule).__name__.lower() for rule in node.rules)


def test_the_rule_set_is_the_one_shared_definition(tmp_path: Path, yard: Zone):
    # There were two copies of this list, in the console and the CLI, and a
    # third was about to appear here. Rule sets that drift produce two
    # deployments that disagree about what an incident is.
    from sentinel.events import default_rules

    with Node(tmp_path / "n.db", zones=[yard]) as node:
        assert [type(r).__name__ for r in node.rules] == [
            type(r).__name__ for r in default_rules([yard])
        ]


def test_two_cameras_cannot_share_an_id(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        node.add_camera("a.mp4", camera_id="gate")

        with pytest.raises(NodeError, match="already a camera"):
            node.add_camera("b.mp4", camera_id="gate")


def test_a_camera_is_stored_without_its_credential(tmp_path: Path):
    # The row is read by anything that lists cameras. A password in it is a
    # password on a screen.
    secret = "hunter2-not-a-real-password"
    url = f"rtsp://admin:{secret}@10.20.30.40:554/Streaming/Channels/101"

    with Node(tmp_path / "n.db") as node:
        record = node.add_camera(url, camera_id="gate")

        assert secret not in record.display_source
        assert secret not in node.summary()

    with Store(tmp_path / "n.db") as store:
        for row in store._connection.execute("SELECT * FROM cameras").fetchall():
            assert secret not in str(tuple(row))
        for row in store.audit_trail():
            assert secret not in str(tuple(row))


def test_starting_with_no_cameras_says_so(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        with pytest.raises(NodeError, match="no cameras"):
            node.start()


# ------------------------------------------------------------------- the work


def test_a_node_runs_a_camera_to_an_incident(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    with Node(tmp_path / "n.db", node_id="site-a", zones=[yard]) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        assert node.incidents, "the reference scene produced no incident"
        incident = node.incidents[0]
        assert incident.cameras == ("gate",)
        assert node.camera("gate").events

    with Store(tmp_path / "n.db") as store:
        assert store.incident_count() >= 1
        assert store.event_count() >= 1


def test_run_forever_ends_when_every_camera_ends(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # A file completes. Without this the daemon would sit polling a node with
    # nothing running, forever.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)

        started = time.monotonic()
        node.run_forever()
        elapsed = time.monotonic() - started

    assert elapsed < 60, "run_forever did not notice the camera had ended"
    assert not node.camera("gate").is_running


def test_a_stop_condition_bounds_a_run(tmp_path: Path, reference_video: Path):
    # On Windows an external SIGINT does not reach a Python process at all, so
    # a scheduled job or a container has no other way to stop one.
    with Node(tmp_path / "n.db", realtime=True) as node:
        node.add_camera(reference_video, camera_id="gate")

        deadline = time.monotonic() + 2.0
        started = time.monotonic()
        node.run_forever(until=lambda _: time.monotonic() >= deadline)
        elapsed = time.monotonic() - started

    assert elapsed < 30, "the stop condition was not honoured"


def test_events_and_incidents_are_persisted_as_the_run_goes(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # Not only at the end. A node killed mid-run must leave what it had already
    # concluded, or an interrupted night is a night with no record.
    with Node(tmp_path / "n.db", zones=[yard]) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.start()

        persisted = 0
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            node.poll()
            with Store(tmp_path / "n.db", auto_migrate=False) as reader:
                persisted = reader.event_count()
            if persisted:
                break
            time.sleep(0.05)

        node.stop()

    assert persisted > 0, "nothing was persisted until the run ended"


def test_correlation_is_idempotent_across_polls(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # Ids are deterministic, so re-correlating a growing window upserts the same
    # incident rather than accumulating a new one every two seconds.
    with Node(tmp_path / "n.db", zones=[yard]) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        first = {incident.id for incident in node.incidents}
        node.correlate()
        node.correlate()
        again = {incident.id for incident in node.incidents}

    assert first == again

    with Store(tmp_path / "n.db") as store:
        assert store.incident_count() == len(first)
        opened = [
            row for row in store.audit_trail(limit=500)
            if row["action"] == "incident.opened"
        ]
        # Audited once each, however many times correlation runs.
        assert len(opened) == len(first)


# --------------------------------------------------------- threads and the store


def test_recorded_segments_are_indexed_from_the_node_s_thread(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # This has failed twice, in two different callers. An SQLite connection
    # belongs to the thread that made it, and handing `store.save_segment` down
    # into a pipeline running on a camera thread makes every index write fail
    # with a ProgrammingError that recording.py catches and logs — so the clips
    # exist on disk, none of them are indexed, nothing can find them, and
    # retention deletes them as unreferenced.
    with Node(
        tmp_path / "n.db", record_to=tmp_path / "rec", segment_seconds=2.0
    ) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

    with Store(tmp_path / "n.db") as store:
        assert store.recording_count() > 0, "recorded segments were never indexed"
        assert store.recorded_bytes() > 0
        for segment in store.segments():
            assert segment.path.is_file()


def test_a_runner_is_used_once(tmp_path: Path, reference_video: Path):
    # Restarting a runner whose thread has not fully stopped would revive it
    # underneath a new run. Construct another instead.
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector

    runner = CameraRunner(
        VideoSource(reference_video, source_id="gate"), MotionDetector()
    )
    runner.start()
    try:
        with pytest.raises(NodeError, match="already running"):
            runner.start()
    finally:
        runner.stop()


def test_stopping_reports_whether_the_thread_actually_ended(
    tmp_path: Path, reference_video: Path
):
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector

    runner = CameraRunner(
        VideoSource(reference_video, source_id="gate"), MotionDetector()
    )
    runner.start()

    assert runner.stop() is True
    assert not runner.is_running
    # Stopping something already stopped is not an error.
    assert runner.stop() is True


def test_a_camera_that_cannot_be_opened_becomes_a_named_fault(tmp_path: Path):
    # Which camera is in trouble, not merely that something is. Twenty cameras
    # drop together when a switch loses power.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(tmp_path / "absent.mp4", camera_id="gate")
        node.start()

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and node.camera("gate").fault is None:
            node.poll()
            time.sleep(0.05)

        fault = node.camera("gate").fault
        node.stop()

    assert fault is not None
    assert "gate" in fault or "absent" in fault


def test_one_failing_camera_does_not_stop_the_others(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    with Node(tmp_path / "n.db") as node:
        node.add_camera(tmp_path / "absent.mp4", camera_id="broken")
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        assert node.camera("broken").fault is not None
        assert node.camera("gate").runner is not None
        assert node.camera("gate").runner.stats.frames > 0


def test_per_camera_event_history_is_bounded(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # A node running for a month must not hold every event it ever raised in
    # memory. Persistence is the full history; this is a working window.
    with Node(tmp_path / "n.db", zones=[yard], event_retention=3) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        assert len(node.camera("gate").events) <= 3

    with Store(tmp_path / "n.db") as store:
        # Bounding memory must not bound the record.
        assert store.event_count() > 3


def test_closing_drains_what_was_never_polled(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # The precise regression. `stop()` used to release each runner before the
    # final drain in `close()` could reach it, so every event raised since the
    # last poll was discarded along with any incident that would have been
    # correlated from it — a file ending on an intrusion recorded nothing. It is
    # the same ordering the console had to fix in `_on_finished`: collect, then
    # let go.
    #
    # So this deliberately never polls. Everything the run produced is still
    # inside the runner when `close()` is called, and `close()` is the only
    # chance it will ever get.
    node = Node(tmp_path / "n.db", zones=[yard])
    node.add_camera(reference_video, camera_id="gate", pose=site)
    node.start()

    deadline = time.monotonic() + 60
    while node.camera("gate").is_running and time.monotonic() < deadline:
        time.sleep(0.05)

    node.close()

    with Store(tmp_path / "n.db") as store:
        assert store.event_count() > 0, "the last window of events was thrown away"
        assert store.incident_count() > 0, "nothing was correlated on the way out"
        actions = [row["action"] for row in store.audit_trail(limit=500)]
        assert "node.stopped" in actions


def test_a_stopped_camera_can_still_say_what_it_did(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # Releasing the runner on stop also threw away the statistics, so a node
    # could finish a run and be unable to report a single thing about it — the
    # summary printed a camera with no frame count at all.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        assert not node.camera("gate").is_running
        assert node.camera("gate").runner is not None
        assert node.camera("gate").runner.stats.frames > 0
        assert "frames" in node.summary()


def test_stopping_signals_every_camera_before_waiting_on_any(tmp_path: Path):
    # Sixteen cameras behind a switch that has just lost power would otherwise
    # take sixteen timeouts to shut down, one after another.
    order: list[tuple[str, str]] = []

    class Watched(CameraRunner):
        def ask_to_stop(self):
            order.append(("ask", self.source_id))
            super().ask_to_stop()

        def stop(self, timeout: float = 1.0):
            order.append(("wait", self.source_id))
            return super().stop(timeout=timeout)

    from sentinel import node as node_module

    original = node_module.CameraRunner
    node_module.CameraRunner = Watched
    try:
        with Node(tmp_path / "n.db") as node:
            for index in range(3):
                node.add_camera(tmp_path / f"absent-{index}.mp4", camera_id=f"c{index}")
            node.start()
            node.stop()
    finally:
        node_module.CameraRunner = original

    # `stop()` asks again on its way in, so counting "ask" events would count
    # those too. The property is about ordering: by the time the first camera
    # is waited on, every camera has already been asked.
    first_wait = next(
        index for index, (kind, _) in enumerate(order) if kind == "wait"
    )
    asked_first = {name for kind, name in order[:first_wait] if kind == "ask"}

    assert asked_first == {"c0", "c1", "c2"}, (
        "a camera was waited on before the others had been asked to stop, so "
        "shutdown costs one timeout per camera instead of one in total"
    )


def test_the_node_audits_as_itself(tmp_path: Path, reference_video: Path):
    # Not as "console". There is no authentication yet, so there is nobody to
    # name; recording the truth beats inventing an operator.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate")

    with Store(tmp_path / "n.db") as store:
        actors = {row["actor"] for row in store.audit_trail(limit=500)}

    assert actors == {ACTOR}


# ------------------------------------------------------------- remembering


def test_a_restarted_node_comes_back_watching_the_same_zones(
    tmp_path: Path, yard: Zone
):
    # A daemon restarted at 03:00 must come back watching what it was watching
    # at 02:59. Asking the caller to re-supply the zones makes every caller
    # responsible for remembering, and one of them will forget.
    with Node(tmp_path / "n.db", zones=[yard]) as node:
        assert len(node.zones) == 1

    with Node(tmp_path / "n.db") as restarted:
        assert [zone.id for zone in restarted.zones] == ["yard"]
        # And the rule set that goes with having a zone, not the one for none.
        assert any("zone" in type(rule).__name__.lower() for rule in restarted.rules)


def test_a_restarted_node_remembers_where_its_cameras_are(
    tmp_path: Path, site: CameraPose
):
    # Placements were written to the database and never read back, so a restart
    # lost every one of them while zones survived — the cameras returned
    # unplaced, and reported objects as "not placed" with nothing saying that
    # anything had been forgotten.
    with Node(tmp_path / "n.db") as node:
        node.add_camera("gate.mp4", camera_id="gate", pose=site)
        node.add_camera("yard.mp4", camera_id="yard")

    with Node(tmp_path / "n.db") as restarted:
        assert restarted.restored_cameras == 2
        assert {record.camera_id for record in restarted.cameras} == {"gate", "yard"}

        gate = restarted.camera("gate")
        assert gate.pose is not None
        assert gate.pose.position.lat == pytest.approx(site.position.lat)
        assert gate.pose.heading == pytest.approx(site.heading)
        assert gate.pose.pitch == pytest.approx(site.pitch)
        # An unplaced camera stays unplaced. There is deliberately no default.
        assert restarted.camera("yard").pose is None


def test_a_restored_network_camera_says_it_needs_its_password_again(tmp_path: Path):
    # The raw URL was never persisted — that is the whole point of the
    # credential rule — so a restored network camera cannot connect. An
    # interface should be able to say so *before* the operator presses Start,
    # rather than after the connection fails with something unhelpful.
    secret = "hunter2-not-a-real-password"
    with Node(tmp_path / "n.db") as node:
        node.add_camera(
            f"rtsp://admin:{secret}@10.20.30.40:554/s", camera_id="net"
        )
        node.add_camera("gate.mp4", camera_id="file")

    with Node(tmp_path / "n.db") as restarted:
        assert restarted.needs_credentials("net") is True
        assert restarted.needs_credentials("file") is False
        assert secret not in restarted.camera("net").source


def test_explicit_zones_replace_what_was_stored(tmp_path: Path, yard: Zone):
    # Loading is the default, not the law. A caller that names its zones means it.
    other = Zone(
        id="dock", name="Dock", kind=ZoneKind.RESTRICTED,
        ring=yard.ring, enter_after_millis=600,
    )
    with Node(tmp_path / "n.db", zones=[yard]):
        pass

    with Node(tmp_path / "n.db", zones=[other]) as node:
        assert [zone.id for zone in node.zones] == ["dock"]


# ------------------------------------------------------- export, with footage


def test_the_node_exports_an_incident_with_its_footage(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # The console's own export had neither half — no footage, no preservation —
    # so a package produced from the interface contained no video and the clips
    # it was built from stayed deletable by the next retention pass. One
    # implementation now, for every caller.
    with Node(
        tmp_path / "n.db", zones=[yard],
        record_to=tmp_path / "rec", segment_seconds=2.0,
    ) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        assert node.incidents
        export, coverage = node.export_incident(
            node.incidents[0].id, tmp_path / "out"
        )

        clips = sorted(export.directory.glob("*.mp4"))
        assert clips, "the package has no video in it"
        assert (export.directory / "footage.json").is_file()
        assert coverage and coverage[0].camera_id == "gate"

        # And the originals are now protected from retention.
        assert node.store.recorded_bytes(preserved=True) > 0
        actions = [row["action"] for row in node.store.audit_trail(limit=500)]
        assert "recording.preserved" in actions
        assert "incident.exported" in actions


def test_exporting_an_unknown_incident_says_so(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        with pytest.raises(NodeError, match="no incident"):
            node.export_incident("inc_nothing", tmp_path / "out")


def test_an_incident_can_be_exported_long_after_the_run(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # An evidence package is a thing you produce when somebody asks, not a thing
    # you must remember to produce at the time.
    with Node(tmp_path / "n.db", zones=[yard]) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()
        incident_id = node.incidents[0].id

    with Node(tmp_path / "n.db") as later:
        export, _ = later.export_incident(incident_id, tmp_path / "out")

    assert (export.directory / "incident.json").is_file()
    assert (export.directory / "report.txt").is_file()


# --------------------------------------------------- one live source, one camera


def test_the_same_device_cannot_be_added_twice(tmp_path: Path):
    # An operator's log: device:0 added three times across sessions, all three
    # started against one webcam, two refused by the driver and every pane
    # reconnecting in a loop. Nothing is opened here — add_camera never opens.
    with Node(tmp_path / "n.db") as node:
        node.add_camera("device:0", camera_id="integrated-camera")
        with pytest.raises(NodeError, match="already camera 'integrated-camera'"):
            node.add_camera("device:0", camera_id="integrated-camera-2")
        with pytest.raises(NodeError, match="already camera"):
            node.add_camera("rtsp://user:secret@10.0.0.9/live", camera_id="gate")
            node.add_camera("rtsp://user:secret@10.0.0.9/live", camera_id="gate-2")
        assert [c.camera_id for c in node.cameras] == ["integrated-camera", "gate"]


def test_the_refusal_never_carries_the_credential(tmp_path: Path):
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://user:secret@10.0.0.9/live", camera_id="gate")
        with pytest.raises(NodeError) as caught:
            node.add_camera("rtsp://user:secret@10.0.0.9/live", camera_id="gate-2")
        assert "secret" not in str(caught.value)


def test_a_file_may_back_as_many_cameras_as_asked(tmp_path: Path, reference_video: Path):
    # A file is a replay; two cameras on one clip is how multi-camera
    # correlation is tested and demonstrated.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="cam-07")
        node.add_camera(reference_video, camera_id="cam-08")
        assert len(node.cameras) == 2


def test_a_duplicate_device_already_stored_is_restored_but_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Rows written before the rule existed. They must not vanish from a list
    # the operator made, and they must not fight the first camera for the
    # device either.
    database = tmp_path / "n.db"
    with Store(database) as store:
        store.save_camera("integrated-camera", "integrated-camera", "device:0")
        store.save_camera("integrated-camera-2", "integrated-camera-2", "device:0")

    # Nothing may touch a real webcam from a unit test: the runner's thread is
    # never started, so what is checked is exactly which runners were built.
    monkeypatch.setattr(CameraRunner, "start", lambda self: None)

    with Node(database) as node:
        assert node.restored_cameras == 2
        twin = node.camera("integrated-camera-2")
        assert twin.fault is not None and "same source as integrated-camera" in twin.fault

        assert node.start() == 1, "only one camera may open the device"
        assert node.camera("integrated-camera").runner is not None
        assert twin.runner is None
        assert "same source as integrated-camera" in (twin.fault or "")


# ------------------------------------------------------ removing what was added


def test_a_removed_camera_is_gone_from_the_node_and_the_database(tmp_path: Path, reference_video: Path):
    database = tmp_path / "n.db"
    with Node(database) as node:
        node.add_camera(reference_video, camera_id="gate")
        node.add_camera(reference_video, camera_id="yard")
        node.remove_camera("gate")
        assert [c.camera_id for c in node.cameras] == ["yard"]
        with pytest.raises(NodeError, match="no camera 'gate'"):
            node.remove_camera("gate")

    with Node(database) as again:
        assert [c.camera_id for c in again.cameras] == ["yard"], "the row came back"

    with Store(database) as store:
        actions = [(row["action"], row["subject"]) for row in store.audit_trail(limit=50)]
        assert ("camera.removed", "gate") in actions


def test_removing_a_running_camera_stops_it_and_keeps_its_evidence(
    tmp_path: Path, reference_video: Path, yard: Zone, site: CameraPose
):
    # What it saw stays. A camera being taken down does not unmake an incident.
    database = tmp_path / "n.db"
    with Node(database, zones=[yard]) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.start()
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline and not node.incidents:
            node.poll(force_correlate=True)
            time.sleep(0.05)
        assert node.incidents, "the reference scene produced no incident to keep"

        node.remove_camera("gate")
        assert not node.cameras
        assert not node.is_running
        assert node.incidents, "removing the camera discarded its incidents"

    with Store(database) as store:
        assert store.incident_count() > 0
        assert store.event_count() > 0


def test_a_zone_can_be_changed_and_removed(tmp_path: Path, yard: Zone):
    database = tmp_path / "n.db"
    with Node(database, zones=[yard]) as node:
        assert len(node.rules) == 4, "a node with a zone has the zone rules"

        renamed = Zone(
            id=yard.id, name="Loading bay", kind=ZoneKind.EXCLUSION, ring=yard.ring,
            enter_after_millis=yard.enter_after_millis,
        )
        node.replace_zone(renamed)
        assert node.zones[0].name == "Loading bay"
        assert node.zones[0].kind is ZoneKind.EXCLUSION

        node.remove_zone(yard.id)
        assert not node.zones
        assert len(node.rules) == 1, "the last zone went, and the zone rules with it"
        with pytest.raises(NodeError, match="no zone"):
            node.remove_zone(yard.id)

    with Node(database) as again:
        assert not again.zones, "the zone row came back"

    with Store(database) as store:
        actions = [row["action"] for row in store.audit_trail(limit=50)]
        assert "zone.changed" in actions and "zone.removed" in actions
        detail = next(row["detail"] for row in store.audit_trail(limit=50) if row["action"] == "zone.changed")
        assert "RESTRICTED" in detail and "EXCLUSION" in detail
        assert "Loading bay" in detail, "the audit row did not say what the name became"


def test_a_zone_outline_change_is_audited_as_such(tmp_path: Path, yard: Zone):
    # "name -> name" would hide a zone quietly shrinking to exclude the door
    # it was drawn around. The outline change is named in the audit row.
    from sentinel.core import destination_point

    with Node(tmp_path / "n.db", zones=[yard]) as node:
        centre = yard.ring[0]
        bigger = tuple(destination_point(centre, b, 25.0) for b in (0.0, 72.0, 144.0, 216.0, 288.0))
        node.replace_zone(Zone(id=yard.id, name=yard.name, kind=yard.kind, ring=bigger))
        assert len(node.zones[0].ring) == 5

    with Store(tmp_path / "n.db") as store:
        detail = next(row["detail"] for row in store.audit_trail(limit=50) if row["action"] == "zone.changed")
        assert "outline" in detail and "4 -> 5 points" in detail
        assert "name" not in detail, "an unchanged field was reported as changed"


# ---------------------------------------------------- structured audit records


def audit_row(store: Store, action: str):
    """The most recent row for one action, or a failure that names what was there.

    A bare ``next(...)`` raises `StopIteration` with nothing in it, and the
    question when this fails is always "then what did get written".
    """
    for row in store.audit_trail(limit=100):
        if row["action"] == action:
            return row
    written = [row["action"] for row in store.audit_trail(limit=100)]
    raise AssertionError(f"no {action!r} row was written; the log holds {written}")


def test_a_zone_change_stores_both_states_and_still_reads_as_it_did(
    tmp_path: Path, yard: Zone
):
    """The prose is unchanged for a person; the states are there for everything else.

    A rename that reached the log as "name 'Yard' -> 'Loading bay'" could be
    read and nothing else: not filtered, not replayed, not checked. Now the two
    zones are stored beside the sentence, and the sentence is rendered from the
    same comparison — so the row cannot say the kind changed while the record
    says it did not.

    Both halves are asserted, because either alone passes while the wiring is
    only half there: the prose alone passes with `store.audit`, and the JSON
    alone passes with a line nobody can read.
    """
    import json

    from sentinel.node import _describe_zone_change

    renamed = Zone(
        id=yard.id, name="Loading bay", kind=yard.kind, ring=yard.ring,
        schedule=yard.schedule, enter_after_millis=yard.enter_after_millis,
        exit_after_millis=yard.exit_after_millis,
        accept_uncertain=yard.accept_uncertain,
    )

    with Node(tmp_path / "n.db", zones=[yard], node_id="gatehouse") as node:
        node.replace_zone(renamed)

    with Store(tmp_path / "n.db") as store:
        row = audit_row(store, "zone.changed")
        print(row["detail"])
        print(row["before_json"])

        assert row["detail"] == _describe_zone_change(yard, renamed), (
            "the line operators have been reading changed voice"
        )
        assert row["before_json"] is not None, "the row kept only the sentence"
        before = json.loads(row["before_json"])
        after = json.loads(row["after_json"])
        differing = {key for key in before if before[key] != after[key]}

        assert differing == {"name"}, "the states differ somewhere nothing changed"
        assert before["name"] == "Yard" and after["name"] == "Loading bay"
        assert row["node_id"] == "gatehouse", "the row cannot say which node wrote it"
        assert row["chain_hash"], "the row is not in the chain"


def test_a_camera_placement_records_where_it_was_as_well_as_where_it_is(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    """A camera nudged three degrees is a track that lands outside the zone.

    The log could always say where a camera ended up and never where it had
    been, so a placement that quietly moved a boundary out of view left nothing
    to compare against. The prose stays the coordinate line a person reads —
    a diff of two poses would put JSON there instead, which is worse for the
    reader — and the two poses go beside it.
    """
    import json

    nudged = CameraPose(
        position=site.position, mount_height=site.mount_height,
        heading=site.heading + 3.0, pitch=site.pitch,
        horizontal_fov=site.horizontal_fov, vertical_fov=site.vertical_fov,
        range_meters=site.range_meters,
    )

    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate")
        node.place_camera("gate", site)
        node.place_camera("gate", nudged)

    with Store(tmp_path / "n.db") as store:
        row = audit_row(store, "camera.placed")
        print(row["detail"])

        assert row["detail"].startswith(f"{site.position.lat:.6f},"), (
            "the placement line stopped being a coordinate"
        )
        assert "hdg=183.0" in row["detail"]
        before = json.loads(row["before_json"])
        after = json.loads(row["after_json"])
        differing = {key for key in before if before[key] != after[key]}

        assert differing == {"heading"}, "a pose changed where nothing was moved"
        assert before["heading"] == site.heading
        assert after["heading"] == nudged.heading


def test_a_removed_camera_leaves_the_placement_it_took_with_it(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    """The row is the last surviving record of where the evidence came from.

    Its events keep the camera id and nothing else; once the camera row is
    deleted, where it was pointing exists only here. The source recorded is the
    redacted one, because this row outlives the camera and a password in it is a
    password in an append-only log.
    """
    import json

    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        record = node.camera("gate")
        display = record.display_source
        node.remove_camera("gate")

    with Store(tmp_path / "n.db") as store:
        row = audit_row(store, "camera.removed")
        print(row["before_json"])

        assert row["detail"] == display, "the removal line changed voice"
        assert row["after_json"] is None, "a deletion was given an after state"
        before = json.loads(row["before_json"])

        assert before["source"] == display
        assert before["pose"]["heading"] == site.heading, (
            "where it was pointing went with it"
        )


def test_a_dragged_corner_reads_as_the_corner_that_moved(tmp_path: Path, yard: Zone):
    """The one operator-facing line this change rewrote, pinned so it holds.

    `_describe_zone_change` said "outline 4 -> 4 points, moved" for a ring whose
    corners moved without changing count, which names the one number that did
    not change and hides the one that did. `auditing.describe` says which corner,
    and that is an improvement rather than a drift — but until now nothing held
    the new wording, so the only three lines this change altered were the three
    with no test on them.

    Both spellings are asserted, so the day somebody makes the two agree again
    this test says which one moved.
    """
    from sentinel.node import _describe_zone_change

    moved = Zone(
        id=yard.id, name=yard.name, kind=yard.kind,
        ring=(
            yard.ring[0], yard.ring[1],
            LatLon(33.893500, 35.501850), yard.ring[3],
        ),
        schedule=yard.schedule, enter_after_millis=yard.enter_after_millis,
        exit_after_millis=yard.exit_after_millis,
        accept_uncertain=yard.accept_uncertain,
    )
    assert len(moved.ring) == len(yard.ring), "the count changed; that is the other case"

    with Node(tmp_path / "n.db", zones=[yard], node_id="gatehouse") as node:
        node.replace_zone(moved)

    with Store(tmp_path / "n.db") as store:
        row = audit_row(store, "zone.changed")
        print(f"now: {row['detail']!r}; was: {_describe_zone_change(yard, moved)!r}")

        assert row["detail"] == "outline corner 2 moved"
        assert _describe_zone_change(yard, moved) == "outline 4 -> 4 points, moved", (
            "the line this replaced changed, so the comparison is against nothing"
        )


def test_the_summary_carries_the_chain_head_an_operator_has_to_write_down(
    tmp_path: Path, yard: Zone
):
    """Somebody has to be able to read the log, or writing it proved nothing.

    Every zone edit has been stored with both its states and a chain hash, and
    an operator had no way to see that any of it happened: `audit_trail` had no
    reader anywhere. `summary` is what `sentinel run` prints when a run ends, so
    the three numbers that matter go there — how much was recorded, how much of
    it is chained, and the head.

    The head is asserted in full. Its only use is to be copied somewhere this
    process cannot reach, and a prefix copied into a logbook verifies nothing
    while looking as though it would.

    Delete the audit block from `summary` and this fails, which is the point of
    it: the write path already had tests, and none of them needed a reader.
    """
    renamed = Zone(
        id=yard.id, name="Loading bay", kind=yard.kind, ring=yard.ring,
        schedule=yard.schedule, enter_after_millis=yard.enter_after_millis,
        exit_after_millis=yard.exit_after_millis,
        accept_uncertain=yard.accept_uncertain,
    )

    with Node(tmp_path / "n.db", zones=[yard], node_id="gatehouse") as node:
        before_any_edit = node.summary()
        assert "chain head      none" in before_any_edit, (
            "a node that has changed nothing claimed a chain it does not have"
        )

        node.replace_zone(renamed)
        head = node.store.audit_chain_head()
        written, chained = node.store.audit_totals()
        summary = node.summary()
        print(summary)

    assert head is not None and len(head) == 64, "the head stopped being a SHA-256"
    assert head in summary, (
        "the head an operator has to record is not on the only screen that shows it"
    )
    assert f"{written} row(s)" in summary and f"{chained} chained" in summary
    assert chained == 1 and written > chained, (
        "the fixture stopped exercising both kinds of row, so the counts prove nothing"
    )
    assert f"{written - chained} prose-only" in summary, (
        "the rows the chain does not cover were passed over in silence"
    )


def test_a_node_names_the_clock_its_schedules_are_read_against(tmp_path: Path):
    from datetime import timedelta, timezone

    with Node(tmp_path / "n.db", site_tz=timezone(timedelta(hours=3))) as node:
        assert node.site_tz == timezone(timedelta(hours=3))
        assert node.site_clock_label.startswith("this machine's clock, UTC+0")
    with Node(tmp_path / "m.db") as node:
        assert node.site_tz is not None, "a node with no clock declared falls back to the machine's"
        assert "UTC" in node.site_clock_label


# --------------------------------------------------------------- camera health


class _StalledRunner:
    """Stands in for a camera whose thread is alive and whose decoder has died.

    There is no way to produce the real thing inside a test in under thirty
    seconds — that is the nature of the failure — so the clocks are the only
    part faked. Everything else is what a real runner would report.

    ``silent_for`` is the time since the thread started; ``silent_since_frame``
    is the time since the last frame, and leaving it ``None`` means *this camera
    has never produced one*. Both are needed, because they are the two different
    faults health exists to tell apart: a camera that never connected, and a
    camera that was working and stopped.
    """

    def __init__(
        self,
        silent_for: float,
        *,
        silent_since_frame: float | None = None,
        running: bool = True,
        skipped: int = 0,
        dropped_frames: int = 0,
        reconnects: int = 0,
    ):
        self._silent_for = silent_for
        self._silent_since_frame = silent_since_frame
        self.is_running = running
        self.fault = None
        self.stats = None
        self.skipped = skipped
        self.analysis_fps = 30.0  # the last rate it ever measured, now stale
        self.dropped_frames = dropped_frames
        self.reconnects = reconnects

    @property
    def seconds_since_frame(self):
        return self._silent_since_frame

    @property
    def seconds_since_started(self):
        return self._silent_for


def test_a_camera_that_was_never_started_is_stopped_and_claims_no_rate(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # A stopped camera used to be indistinguishable from a working one in
    # everything except `is_running`, and the numbers beside it were whatever
    # the last run had left there.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)

        health = node.camera_health()["gate"]

    assert health.state is CameraState.STOPPED
    assert health.is_running is False
    assert health.is_placed is True
    assert health.fault is None
    # Never ran, so there is no rate and no last frame — and neither is dressed
    # up as a number that reads like a measurement.
    assert health.analysis_fps == 0.0
    assert health.frames == 0
    assert health.seconds_since_frame is None
    assert health.seconds_since_started is None
    # Stopped is not dark: nothing failed. It covers no ground either way.
    assert health.is_dark is False
    assert health.covers_ground is False


def test_a_running_camera_over_the_reference_video_reports_a_measured_rate(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # Paced to the file's own timeline so the run lasts long enough to be
    # observed while it is happening — which is the only moment health is worth
    # anything.
    live = None
    with Node(tmp_path / "n.db", realtime=True) as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.start()

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            node.poll()
            health = node.camera_health()["gate"]
            if health.state is CameraState.LIVE:
                live = health
                break
            time.sleep(0.05)

        node.stop()

    assert live is not None, "a camera analysing a file never reported itself live"
    print(
        f"live: {live.analysis_fps} fps, {live.frames} frames, "
        f"{live.seconds_since_frame:.3f}s since the last, "
        f"{live.frames_not_drawn} not drawn, {live.describe()}"
    )

    assert live.is_running is True
    assert live.is_placed is True
    assert live.fault is None
    # Measured above, floored here: any positive rate proves frames are being
    # analysed now, and the exact number depends on the machine.
    assert live.analysis_fps > 0.0
    assert live.frames > 0
    assert live.seconds_since_frame is not None
    assert live.seconds_since_frame < 1.0
    # A file drops nothing and reconnects to nothing. These zeros are the
    # file's real measured values — `Pipeline.stream` is None for a file, so
    # there is no live reader to have dropped anything — and NOT a default: the
    # stub test below drives the same fields to 12 and 3 and watches them
    # arrive, so replacing the counters with `return 0` fails there.
    assert live.frames_dropped == 0
    assert live.reconnects == 0
    # Placed and producing frames is the only combination that may claim ground.
    assert live.covers_ground is True
    assert live.is_dark is False


def test_a_camera_that_faulted_is_dark_and_says_why(tmp_path: Path, site: CameraPose):
    # The map hatches this one and drops it from coverage. A placed camera that
    # cannot open its source still has a footprint, and painting that footprint
    # as watched ground is the dishonesty health exists to stop.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(tmp_path / "absent.mp4", camera_id="gate", pose=site)
        node.start()

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and node.camera("gate").fault is None:
            node.poll()
            time.sleep(0.05)

        health = node.camera_health()["gate"]
        node.stop()

    print(f"faulted: {health.describe()}")

    assert health.state is CameraState.FAULTED
    assert health.fault is not None
    assert "gate" in health.fault or "absent" in health.fault
    assert health.is_dark is True, "a faulted camera must be excluded from coverage"
    # Placed, and still covering nothing.
    assert health.is_placed is True
    assert health.covers_ground is False
    assert health.analysis_fps == 0.0
    assert health.fault in health.describe()


def test_the_seconds_since_the_last_frame_grow_once_the_frames_stop(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # The number the whole state machine rests on. If it does not grow, a
    # stalled camera stays LIVE forever and nobody is told.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        first = node.camera_health()["gate"]
        assert first.seconds_since_frame is not None
        time.sleep(0.5)
        later = node.camera_health()["gate"]

    grew = later.seconds_since_frame - first.seconds_since_frame
    print(f"since last frame: {first.seconds_since_frame:.3f}s -> "
          f"{later.seconds_since_frame:.3f}s (grew {grew:.3f}s)")

    # Measured above; floored a little under the sleep to allow for timer
    # granularity, and bounded above so a clock jump would be caught.
    assert grew >= 0.4
    assert grew < 5.0
    # The run ended by itself, so this is completion and not a failure.
    assert later.state is CameraState.STOPPED
    assert later.frames > 0
    # The rate is measured over the last second, and no frame arrived in it.
    assert later.analysis_fps == 0.0


def test_a_camera_running_and_producing_nothing_is_dark_rather_than_live(
    tmp_path: Path, site: CameraPose
):
    # The failure an operator must see: the thread is alive, nothing is faulted,
    # and no frame has arrived in a minute. This used to read as "running".
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://10.0.0.9:554/stream", camera_id="gate", pose=site)
        node.camera("gate").runner = _StalledRunner(DARK_AFTER_SECONDS * 2)

        health = node.camera_health()["gate"]
        node.camera("gate").runner = None

    print(f"stalled: {health.describe()}")

    assert health.state is CameraState.DARK
    assert health.is_running is True, "the thread really is alive; that is the trap"
    assert health.is_dark is True
    assert health.covers_ground is False, "silent ground must not count as covered"
    # The last rate it ever measured is not a rate now.
    assert health.analysis_fps == 0.0
    assert "no frame" in health.describe()


def test_a_camera_still_opening_its_stream_is_not_yet_called_dark(
    tmp_path: Path, site: CameraPose
):
    # Opening an RTSP stream takes seconds. Calling those seconds dark teaches
    # an operator to ignore the word, which costs more than it saves.
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://10.0.0.9:554/stream", camera_id="gate", pose=site)
        node.camera("gate").runner = _StalledRunner(DARK_AFTER_SECONDS / 10)

        health = node.camera_health()["gate"]
        node.camera("gate").runner = None

    assert health.state is CameraState.STARTING
    assert health.is_dark is False
    # It is not watching anything yet either, so it claims no ground.
    assert health.covers_ground is False


def test_health_covers_every_camera_and_survives_one_of_them_failing(
    tmp_path: Path, reference_video: Path, site: CameraPose
):
    # A status strip is read when something is wrong, so the one broken camera
    # must not be able to take the other fifteen off the screen with it.
    with Node(tmp_path / "n.db") as node:
        node.add_camera(tmp_path / "absent.mp4", camera_id="broken", pose=site)
        node.add_camera(reference_video, camera_id="gate", pose=site)
        node.run_forever()

        health = node.camera_health()

    assert set(health) == {"broken", "gate"}
    assert health["broken"].state is CameraState.FAULTED
    assert health["gate"].state is CameraState.STOPPED
    assert health["gate"].frames > 0
    # Exactly the split the map needs: one hatched, one merely finished.
    assert [c for c, h in health.items() if h.is_dark] == ["broken"]


def test_a_camera_that_stopped_delivering_frames_goes_dark(
    tmp_path: Path, site: CameraPose
):
    # The wedged-RTSP case this whole type exists for, and the one arm of the
    # DARK decision the other stub tests never reach: this camera *was* working.
    # Its thread is alive, it has no fault, it delivered frames for ten minutes
    # and then delivered nothing for a minute. Measuring from the start clock
    # here would say "no frame for 600s" and send an operator looking for an
    # outage that began at the wrong end of the shift.
    silent = DARK_AFTER_SECONDS * 2
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://10.0.0.11:554/stream", camera_id="gate", pose=site)
        node.camera("gate").runner = _StalledRunner(
            600.0, silent_since_frame=silent
        )

        health = node.camera_health()["gate"]
        node.camera("gate").runner = None

    print(f"stopped delivering: {health.describe()}")

    assert health.state is CameraState.DARK
    assert health.is_running is True, "the thread really is alive; that is the trap"
    assert health.is_dark is True
    assert health.covers_ground is False, "silent ground must not count as covered"
    assert health.seconds_since_frame == silent
    # The silence since the last *frame*, not since the thread started.
    assert f"{silent:.0f}s" in health.describe()
    assert "600s" not in health.describe(), (
        "the line dated the outage from start-up rather than from the last frame"
    )
    # A rate from before the stall is a claim about now that nothing supports.
    assert health.analysis_fps == 0.0


def test_a_camera_quiet_for_seconds_says_so_rather_than_claiming_zero_fps(
    tmp_path: Path, site: CameraPose
):
    # Between one second and thirty, a camera is still LIVE — the grace window
    # is one threshold, deliberately — but the rate window is only a second
    # wide, so the measured rate is zero. The line used to read "LIVE — 0 fps",
    # which contradicts itself in four words and tells an operator nothing about
    # which half to believe.
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://10.0.0.12:554/stream", camera_id="gate", pose=site)
        node.camera("gate").runner = _StalledRunner(
            120.0, silent_since_frame=5.0
        )

        health = node.camera_health()["gate"]
        node.camera("gate").runner = None

    line = health.describe()
    print(f"quiet but not yet dark: {line}")

    assert health.state is CameraState.LIVE
    assert health.analysis_fps == 0.0, "no frame in the last second is a measured zero"
    assert "no frame for 5s" in line
    assert "fps" not in line, "a zero rate was quoted beside a five-second silence"
    # Still inside the grace window, so it still claims its ground — one
    # threshold for the strip and the map, or they disagree about one camera.
    assert health.is_dark is False
    assert health.covers_ground is True


def test_the_counters_that_reveal_a_flapping_camera_reach_the_strip(
    tmp_path: Path, site: CameraPose
):
    # A camera reconnecting every few seconds looks alive in every other measure
    # — the thread runs, frames arrive, the fault stays None — and is losing most
    # of what it sees. These three numbers are the only evidence of it, and a
    # file source reports zero for all of them, so nothing else in this suite
    # would notice them being dropped on the floor between runner and health.
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://10.0.0.13:554/stream", camera_id="gate", pose=site)
        node.camera("gate").runner = _StalledRunner(
            120.0, silent_since_frame=0.2,
            dropped_frames=12, reconnects=3, skipped=7,
        )

        health = node.camera_health()["gate"]
        node.camera("gate").runner = None

    print(f"flapping: {health.describe()}")

    assert health.state is CameraState.LIVE
    assert health.frames_dropped == 12
    assert health.reconnects == 3
    assert health.frames_not_drawn == 7, (
        "results the viewer never collected are a different failure from frames "
        "the decoder discarded, and both must survive the trip"
    )
    # The one an operator has to be told without being asked.
    assert "3 reconnect(s)" in health.describe()


def test_the_state_word_is_what_an_interface_actually_prints():
    # `class CameraState(str, Enum)` renders as "CameraState.LIVE" under str()
    # and f-strings on this interpreter — measured, 3.14.7 — so the first
    # f-string a UI writes around the state puts a Python repr on an operator's
    # status strip. StrEnum is what makes the obvious code correct.
    print(f"str: {str(CameraState.LIVE)!r}, format: {f'{CameraState.LIVE}'!r}")

    assert f"{CameraState.LIVE}" == "LIVE"
    assert str(CameraState.DARK) == "DARK"
    assert "the gate is " + CameraState.FAULTED == "the gate is FAULTED"
    # And the comparison every caller was told it could make still holds.
    assert CameraState.LIVE == "LIVE"
    assert CameraState.LIVE.value == "LIVE"


# ---------------------------------------------------------------- plate reads


class _PlateRunner:
    """Stands in for a camera whose pipeline has a plate reader.

    A real one needs a plate model this suite does not ship, and the property
    under test is not reading — `test_plates.py` and `test_pipeline.py` own
    that — but what the node does with a reading once the pipeline has
    published it. Hands out the updates it was given, one per poll, in order;
    everything else is what a live runner would report.
    """

    def __init__(self, updates):
        self._updates = list(updates)
        self.is_running = True
        self.fault = None
        self.stats = None
        self.skipped = 0
        self.analysis_fps = 30.0
        self.dropped_frames = 0
        self.reconnects = 0

    def take_latest(self):
        return self._updates.pop(0) if self._updates else None

    def take_segments(self):
        return []

    def take_events(self):
        return []

    def ask_to_stop(self):
        self.is_running = False

    def stop(self, timeout=None):
        self.is_running = False
        return True

    @property
    def seconds_since_frame(self):
        return 0.0

    @property
    def seconds_since_started(self):
        return 1.0


def _reading(
    track: int = 3,
    *,
    display: str,
    text: str | None = None,
    confident: bool = False,
    agreement: int = 0,
    reads: int = 3,
):
    from sentinel.pipeline import TrackPlate

    return TrackPlate(
        track_id=track, country="LB", display=display, text=text,
        is_confident=confident, agreement=agreement, reads=reads,
    )


def _frame(index: int, *, plates, first_seen: int, last_seen: int):
    """One published frame carrying a vehicle track and its readings."""
    from sentinel.core import BoundingBox, Track
    from sentinel.node import Update
    from sentinel.pipeline import FrameResult, PipelineStats

    tracks = tuple(
        Track(
            id=plate.track_id, class_id=2, bbox=BoundingBox(0.4, 0.5, 0.2, 0.2),
            confidence=0.9, hits=index, first_seen_millis=first_seen,
            last_seen_millis=last_seen, position=None, speed_mps=None,
            heading_degrees=None,
        )
        for plate in plates
    )
    result = FrameResult(
        index=index, timestamp_millis=last_seen, source_id="gate",
        detections=(), tracks=tracks, ended=(), events=(), plates=tuple(plates),
    )
    return Update(result=result, analysis_fps=30.0, skipped=0, stats=PipelineStats())


def _enrol_van(node: Node):
    from sentinel.registry import Plate

    return node.store.register.enrol(
        subject_id="veh-1", display_name="Contractor van",
        identifier=Plate("B 7421", frames_agreeing=6),
        actor="operator:alice", basis="site access list",
    )


def _sightings_of_anyone(node: Node):
    register = node.store.register
    return [
        sighting
        for subject in register.subjects()
        for sighting in register.history(subject.id)
    ]


def _sighting_audits(node: Node):
    return [
        row for row in node.store.audit_trail(limit=500)
        if row["action"] == "vehicle.sighted"
    ]


@contextlib.contextmanager
def _listening_to_the_node(level: int = logging.WARNING):
    """Everything `node.py` logs, at or above ``level``.

    `caplog` cannot be used: `logs.configure` sets ``propagate=False`` on the
    `sentinel` tree, deliberately, so the handler has to go on the logger that
    actually emits.
    """
    said: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            said.append(record)

    logger = logging.getLogger("sentinel.node")
    handler = Collect()
    handler.setLevel(level)
    was = logger.level
    logger.addHandler(handler)
    logger.setLevel(min(was, level) if was else level)
    try:
        yield said
    finally:
        logger.removeHandler(handler)
        logger.setLevel(was)


def test_a_plate_reading_is_persisted_once_however_many_frames_carry_it(
    tmp_path: Path, reference_video: Path
):
    """Until this was wired, `FrameResult.plates` reached the screen and nothing else.

    Three polls, three frames of the same van, one row — holding the last
    tally and the whole window. A row per frame would make one van three.
    """
    frames = [
        _frame(10, plates=[_reading(display="B74?1", reads=3)], first_seen=1_000, last_seen=1_400),
        _frame(14, plates=[_reading(display="B7421", text="B7421", agreement=3, reads=5)], first_seen=1_000, last_seen=1_560),
        _frame(22, plates=[_reading(display="B7421", text="B7421", agreement=3, reads=6)], first_seen=1_000, last_seen=1_880),
    ]
    with Node(tmp_path / "n.db") as node:
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(frames)

        polled = sum(len(node.poll()) for _ in range(3))
        rows = node.store.plate_reads()

    assert polled == 3, "the fake fed fewer frames than the test believes"
    assert len(rows) == 1
    assert (rows[0].camera_id, rows[0].track_id) == ("gate", 3)
    assert (rows[0].first_frame, rows[0].last_frame) == (10, 22)
    assert (rows[0].agreement, rows[0].reads) == (3, 6)
    assert rows[0].seen_at_millis == 1_880


def test_an_unconfident_reading_is_kept_but_never_becomes_a_sighting(
    tmp_path: Path, reference_video: Path
):
    """A half-read plate must not become a sighting of somebody's car.

    Two readings of an enrolled plate that fall short in the two ways a
    reading can: one character unread, and every character read but the
    weakest under the bar. Both are recorded — an operator may look — and
    neither reaches the register.
    """
    frames = [
        _frame(10, plates=[_reading(1, display="B74?1", reads=3)], first_seen=1_000, last_seen=1_400),
        _frame(11, plates=[_reading(2, display="B7421", text="B7421", agreement=3, reads=3)], first_seen=1_000, last_seen=1_440),
    ]
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(frames)

        node.poll()
        node.poll()

        readings = node.store.plate_reads()
        history = node.store.register.history("veh-1")
        audits = _sighting_audits(node)

    assert {r.track_id for r in readings} == {1, 2}, "an unconfident reading was not kept"
    assert all(r.is_confident is False for r in readings)
    assert history == (), "a reading under the bar was matched against the register"
    assert audits == []


def test_a_confident_reading_of_an_enrolled_plate_records_a_sighting_that_advances(
    tmp_path: Path, reference_video: Path
):
    """This is the movement history the Vehicles panel will show.

    Two polls of one track: the sighting is one row, its first observation
    stays where the track began and its last advances with the track. The
    claim carries its evidence — a MATCH with the share of reads that agreed
    on the weakest character — and cites the enrolment it matched.
    """
    from sentinel.registry import Confidence

    confident = _reading(display="B7421", text="B7421", confident=True, agreement=4, reads=6)
    frames = [
        _frame(30, plates=[confident], first_seen=1_000, last_seen=2_000),
        _frame(80, plates=[confident], first_seen=1_000, last_seen=4_000),
    ]
    with Node(tmp_path / "n.db") as node:
        enrolment = _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(frames)

        node.poll()
        after_one = node.store.register.history("veh-1")
        node.poll()
        after_two = node.store.register.history("veh-1")
        readings = node.store.plate_reads()

    assert len(after_one) == 1 and len(after_two) == 1, "one track became two sightings"
    assert (after_one[0].first_seen_millis, after_one[0].last_seen_millis) == (1_000, 2_000)
    assert (after_two[0].first_seen_millis, after_two[0].last_seen_millis) == (1_000, 4_000)
    sighting = after_two[0]
    assert (sighting.camera_id, sighting.track_id) == ("gate", 3)
    assert sighting.confidence is Confidence.MATCH
    assert sighting.score == pytest.approx(4 / 6)
    assert sighting.identifier_id == enrolment.identifier.id, "the sighting does not say why"
    assert len(readings) == 1 and readings[0].is_confident is True


def test_the_first_sighting_is_audited_once_by_id_and_never_by_plate(
    tmp_path: Path, reference_video: Path
):
    # The audit log is read by more people than the register is. It gets the
    # subject id and the camera; the plate and the name stay where a forget
    # can reach them.
    confident = _reading(display="B7421", text="B7421", confident=True, agreement=4, reads=6)
    frames = [
        _frame(30, plates=[confident], first_seen=1_000, last_seen=2_000),
        _frame(80, plates=[confident], first_seen=1_000, last_seen=4_000),
    ]
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(frames)

        node.poll()
        node.poll()
        audits = _sighting_audits(node)

    assert len(audits) == 1, "one encounter was audited once per poll"
    row = audits[0]
    assert row["subject"] == "veh-1"
    assert row["actor"] == ACTOR
    assert "gate" in row["detail"] and "track 3" in row["detail"]
    for column in ("subject", "detail"):
        assert "7421" not in (row[column] or ""), f"the plate reached the audit log in {column}"
        assert "Contractor" not in (row[column] or ""), f"the name reached the audit log in {column}"


def test_a_confident_reading_of_a_stranger_stays_a_reading(
    tmp_path: Path, reference_video: Path
):
    # Nothing is ever enrolled by being observed. A plate nobody named is
    # stored as a reading, and the register is not touched.
    confident = _reading(display="C9988", text="C9988", confident=True, agreement=5, reads=7)
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(
            [_frame(30, plates=[confident], first_seen=1_000, last_seen=2_000)]
        )

        node.poll()

        readings = node.store.plate_reads()
        sightings = _sightings_of_anyone(node)
        subjects = node.store.register.subjects()
        audits = _sighting_audits(node)

    assert [r.text for r in readings] == ["C9988"] and readings[0].is_confident is True
    assert sightings == []
    assert [s.id for s in subjects] == ["veh-1"], "observation enrolled somebody"
    assert audits == []


def test_a_read_the_register_cannot_spell_does_not_stop_the_poll(
    tmp_path: Path, reference_video: Path
):
    """The pipeline keeps any alphanumeric; the register keeps A–Z and 0–9.

    Greek capitals are both alphanumeric and outside the plate alphabet, so a
    confident read of them normalises to nothing and the register refuses it.
    That is a read of something that is not a registration, not a fault: the
    reading is kept, no sighting is made, and the next poll still runs.
    """
    odd = _reading(display="\u0391\u0392\u0393", text="\u0391\u0392\u0393", confident=True, agreement=4, reads=4)
    frames = [
        _frame(30, plates=[odd], first_seen=1_000, last_seen=2_000),
        _frame(31, plates=[odd], first_seen=1_000, last_seen=2_040),
    ]
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        node.camera("gate").runner = _PlateRunner(frames)

        with _listening_to_the_node() as said:
            polled = len(node.poll()) + len(node.poll())

        readings = node.store.plate_reads()
        sightings = _sightings_of_anyone(node)

    assert polled == 2, "the refusal stopped the poll"
    assert len(readings) == 1 and readings[0].last_frame == 31
    assert sightings == []
    # Logged once, not once per poll \u2014 and without the read. The register's
    # own refusal quotes it, and the node log travels in support bundles.
    refusals = [r.getMessage() for r in said if "track 3" in r.getMessage()]
    assert len(refusals) == 1, "the refusal was logged once per poll, or not at all"
    assert "\u0391\u0392\u0393" not in refusals[0], "the plate reached the node log"


def test_a_refusal_dies_with_the_runner_that_earned_it(
    tmp_path: Path, reference_video: Path
):
    """A camera restarted in the same process starts its track ids again.

    Found by a skeptic: the refusal was remembered per (camera, track) and
    never let go, so a Greek read on track 3 in one run silenced the enrolled
    van's confident read on track 3 in the next. No sighting, no audit row,
    and no line in any log saying why.
    """
    odd = _reading(display="\u0391\u0392\u0393", text="\u0391\u0392\u0393", confident=True, agreement=4, reads=4)
    van = _reading(display="B7421", text="B7421", confident=True, agreement=4, reads=6)
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        gate = node.camera("gate")

        gate.runner = _PlateRunner([_frame(30, plates=[odd], first_seen=1_000, last_seen=2_000)])
        node.poll()
        gate.runner = _PlateRunner([_frame(3, plates=[van], first_seen=5_000, last_seen=5_200)])
        node.poll()

        history = node.store.register.history("veh-1")
        audits = _sighting_audits(node)

    assert len(history) == 1, "the new run's track 3 inherited the old run's refusal"
    assert (history[0].camera_id, history[0].track_id) == ("gate", 3)
    assert (history[0].first_seen_millis, history[0].last_seen_millis) == (5_000, 5_200)
    assert len(audits) == 1 and audits[0]["subject"] == "veh-1"


def test_a_second_run_of_the_same_track_id_is_a_second_encounter(
    tmp_path: Path, reference_video: Path
):
    """The mirror of the refusal: an audit in one run must not silence the next.

    The van on track 3, the runner rebuilt, the van on the new run's track 3:
    two encounters, two audit lines. The limit written down was that a
    restarted node audits a still-live encounter *once more*; keyed without
    the run it audited it never again.
    """
    van = _reading(display="B7421", text="B7421", confident=True, agreement=4, reads=6)
    with Node(tmp_path / "n.db") as node:
        _enrol_van(node)
        node.add_camera(reference_video, camera_id="gate")
        gate = node.camera("gate")

        gate.runner = _PlateRunner([_frame(30, plates=[van], first_seen=1_000, last_seen=2_000)])
        node.poll()
        first_run = len(_sighting_audits(node))
        gate.runner = _PlateRunner([
            _frame(3, plates=[van], first_seen=5_000, last_seen=5_200),
            _frame(4, plates=[van], first_seen=5_000, last_seen=5_240),
        ])
        node.poll()
        node.poll()

        audits = _sighting_audits(node)
        runs = gate.run

    assert runs == 2, "the record did not count its runners"
    assert first_run == 1
    assert len(audits) == 2, "the second run's encounter was taken for the first's"
    assert all(row["subject"] == "veh-1" for row in audits)
