"""Alerts: the four conditions that must leave the process, and where they go."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sentinel.alerts import (
    CAMERA_DARK, RECORDING_STOPPED, RETENTION_SHORTFALL, THREAD_STUCK,
    Alerts, CommandSink, FileSink, WebhookSink,
)
from sentinel.store import Store


class _Memory:
    def __init__(self):
        self.seen = []

    def deliver(self, alert, *, cleared=False):
        self.seen.append((alert.kind, alert.subject, cleared))


class _Broken:
    def deliver(self, alert, *, cleared=False):
        raise OSError("the pager is off")


def test_an_alert_is_raised_once_until_cleared_and_every_step_is_audited():
    memory, clock = _Memory(), [1_700_000_000.0]
    with Store(":memory:") as store:
        alerts = Alerts([_Broken(), memory], store=store, actor="node", clock=lambda: clock[0], synchronous=True)
        assert alerts.raise_(CAMERA_DARK, "gate", "no frame for 30 s")
        assert not alerts.raise_(CAMERA_DARK, "gate", "no frame for 31 s"), "the same condition raised twice"
        assert alerts.raise_(CAMERA_DARK, "yard", "no frame for 30 s")
        assert [a.subject for a in alerts.active()] == ["gate", "yard"]
        assert [a.subject for a in alerts.take_new()] == ["gate", "yard"]
        assert alerts.take_new() == ()
        assert alerts.clear(CAMERA_DARK, "gate")
        assert not alerts.clear(CAMERA_DARK, "gate")
        assert [a.subject for a in alerts.active()] == ["yard"]
        assert alerts.raised_count == 2
        # The broken sink did not stop the working one.
        assert memory.seen == [(CAMERA_DARK, "gate", False), (CAMERA_DARK, "yard", False), (CAMERA_DARK, "gate", True)]
        actions = [(r["action"], r["subject"]) for r in store.audit_trail(limit=10)]
        assert ("alert.raised", "gate") in actions and ("alert.cleared", "gate") in actions
        assert ("alert.raised", "yard") in actions


def test_the_file_sink_appends_one_readable_line_per_change(tmp_path: Path):
    path = tmp_path / "deep" / "alerts.log"
    alerts = Alerts([FileSink(path)], clock=lambda: 1_700_000_000.0, synchronous=True)
    alerts.raise_(RECORDING_STOPPED, "gate", "disk full")
    alerts.clear(RECORDING_STOPPED, "gate")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("2023-11-14T22:13:20+00:00 RAISED recording.stopped gate — disk full")
    assert " CLEARED recording.stopped gate " in lines[1]


def test_the_command_sink_runs_the_operators_command_with_the_alert_in_its_environment(tmp_path: Path):
    out = tmp_path / "seen.json"
    script = tmp_path / "pager.py"
    script.write_text(
        "import json, os, sys\n"
        "json.dump({'argv': sys.argv[1:], 'env': {k: v for k, v in os.environ.items() if k.startswith('SENTINEL_ALERT_')}}, "
        f"open({str(out)!r}, 'w'))\n",
        encoding="utf-8",
    )
    sink = CommandSink([sys.executable, str(script)])
    alerts = Alerts([sink], synchronous=True)
    alerts.raise_(THREAD_STUCK, "gate, yard", "left running")
    seen = json.loads(out.read_text(encoding="utf-8"))
    assert seen["argv"] == ["analysis.thread_stuck", "gate, yard", "left running", "raised"]
    assert seen["env"]["SENTINEL_ALERT_KIND"] == "analysis.thread_stuck"
    assert seen["env"]["SENTINEL_ALERT_STATE"] == "raised"
    with pytest.raises(ValueError):
        CommandSink("")


class _Collector(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def do_POST(self):  # noqa: N802 - the library's name
        length = int(self.headers.get("Content-Length", "0"))
        _Collector.bodies.append(json.loads(self.rfile.read(length)))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_the_webhook_sink_posts_to_the_local_network_and_refuses_anywhere_else():
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        sink = WebhookSink("http" + f"://127.0.0.1:{port}/alerts")
        alerts = Alerts([sink], clock=lambda: 5.0, synchronous=True)
        alerts.raise_(RETENTION_SHORTFALL, "local", "everything left is preserved")
        assert _Collector.bodies == [{
            "kind": "retention.shortfall", "subject": "local",
            "detail": "everything left is preserved", "raised_at": 5.0, "state": "raised",
        }]
    finally:
        server.shutdown()
        server.server_close()

    # A public address is refused before anything is sent. (The documentation
    # ranges count as private since Python 3.13, so a plainly global one.)
    with pytest.raises(ValueError, match="outside the local network"):
        WebhookSink("http" + "://1.1.1.1/hook")
    with pytest.raises(ValueError, match="web address"):
        WebhookSink("not a url")


def test_the_environment_chooses_the_sinks_and_a_bad_value_is_not_fatal(tmp_path: Path, caplog):
    alerts = Alerts.from_environment(environ={
        "SENTINEL_ALERT_FILE": str(tmp_path / "a.log"),
        "SENTINEL_ALERT_COMMAND": "",
        "SENTINEL_ALERT_WEBHOOK": "http" + "://1.1.1.1/hook",
    })
    assert [type(s).__name__ for s in alerts.sinks] == ["FileSink"]
    assert "SENTINEL_ALERT_WEBHOOK ignored" in caplog.text
    assert alerts.describe() == "alerts go to FileSink"

    none = Alerts.from_environment(environ={"SENTINEL_ALERT_FILE": ""})
    assert none.sinks == () and none.describe() == "alerts go to the log only"

    by_default = Alerts.from_environment(environ={})
    assert [type(s).__name__ for s in by_default.sinks] == ["FileSink"]
    assert by_default.sinks[0].path.name == "alerts.log"
