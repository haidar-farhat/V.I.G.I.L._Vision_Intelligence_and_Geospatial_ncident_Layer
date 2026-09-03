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

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sentinel import logs
from sentinel.core import CameraPose, LatLon
from sentinel.node import ACTOR, CameraRunner, Node, NodeError
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
