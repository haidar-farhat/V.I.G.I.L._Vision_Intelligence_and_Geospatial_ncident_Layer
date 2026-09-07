import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from vigil.config import Settings
from vigil.service.alerts import CAMERA_DARK, RETENTION_SHORTFALL, THREAD_STUCK, Alerts, CommandSink, FileSink, WebhookSink
from vigil.storage.store import Store


class _Memory:
    def __init__(self):
        self.seen = []

    def deliver(self, alert, *, cleared=False):
        self.seen.append((alert.kind, alert.subject, cleared))


class _Broken:
    def deliver(self, alert, *, cleared=False):
        raise OSError("pager off")


def test_raised_once_until_cleared_persisted_and_a_broken_sink_does_not_stop_the_rest():
    memory = _Memory()
    with Store(":memory:") as store:
        alerts = Alerts([_Broken(), memory], store=store, principal="node:t", clock=lambda: 1_700_000_000.0, synchronous=True)
        assert alerts.raise_(CAMERA_DARK, "gate", "dark") and not alerts.raise_(CAMERA_DARK, "gate", "still dark")
        assert alerts.raise_(CAMERA_DARK, "yard", "dark")
        assert [a.subject for a in alerts.take_new()] == ["gate", "yard"] and alerts.take_new() == ()
        assert alerts.clear(CAMERA_DARK, "gate") and not alerts.clear(CAMERA_DARK, "gate")
        assert [a.subject for a in alerts.active()] == ["yard"]
        assert memory.seen == [(CAMERA_DARK, "gate", False), (CAMERA_DARK, "yard", False), (CAMERA_DARK, "gate", True)]
        assert [r["subject"] for r in store.open_alerts()] == ["yard"]
        assert {"alert.raised", "alert.cleared"} <= {r["action"] for r in store.audit_trail()}


def test_the_file_sink_appends_readable_lines(tmp_path):
    path = tmp_path / "deep" / "alerts.log"
    alerts = Alerts([FileSink(path)], clock=lambda: 1_700_000_000.0, synchronous=True)
    alerts.raise_(RETENTION_SHORTFALL, "local", "preserved")
    alerts.clear(RETENTION_SHORTFALL, "local")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("2023-11-14T22:13:20+00:00 RAISED retention.shortfall local") and "CLEARED" in lines[1]


def test_the_command_sink_runs_the_operators_command(tmp_path):
    out = tmp_path / "seen.json"
    script = tmp_path / "pager.py"
    script.write_text("import json, os, sys\njson.dump({'argv': sys.argv[1:], 'kind': os.environ.get('VIGIL_ALERT_KIND')}, open(%r, 'w'))\n" % str(out), encoding="utf-8")
    Alerts([CommandSink([sys.executable, str(script)])], synchronous=True).raise_(THREAD_STUCK, "gate", "stuck")
    seen = json.loads(out.read_text(encoding="utf-8"))
    assert seen == {"argv": ["analysis.thread_stuck", "gate", "stuck", "raised"], "kind": "analysis.thread_stuck"}
    with pytest.raises(ValueError):
        CommandSink("")


class _Collector(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802
        _Collector.bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")))))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_the_webhook_posts_locally_and_refuses_the_internet():
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sink = WebhookSink("http" + f"://127.0.0.1:{server.server_address[1]}/hook")
        Alerts([sink], clock=lambda: 5.0, synchronous=True).raise_(CAMERA_DARK, "gate", "dark")
        assert _Collector.bodies == [{"kind": "camera.dark", "subject": "gate", "detail": "dark", "raised_at": 5.0, "state": "raised"}]
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(ValueError, match="outside the local network"):
        WebhookSink("http" + "://1.1.1.1/hook")
    with pytest.raises(ValueError, match="web address"):
        WebhookSink("nope")


def test_settings_choose_the_sinks_and_a_bad_value_is_not_fatal(tmp_path, caplog):
    settings = Settings(tmp_path, str(tmp_path / "a.log"), "", "http" + "://1.1.1.1/hook", False)
    alerts = Alerts.from_settings(settings)
    assert [type(s).__name__ for s in alerts.sinks] == ["FileSink"] and "ignored" in caplog.text
    assert Alerts.from_settings(Settings(tmp_path, "", None, None, False)).describe() == "alerts go to the log only"
    default = Alerts.from_settings(Settings(tmp_path, None, None, None, False))
    assert default.sinks[0].path == tmp_path / "logs" / "alerts.log"
