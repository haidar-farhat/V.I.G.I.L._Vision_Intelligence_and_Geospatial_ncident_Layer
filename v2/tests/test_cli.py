import io
import re
from pathlib import Path

import pytest

from vigil.interfaces import cli
from vigil.interfaces.cli import build_parser, main


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("VIGIL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIGIL_ALERT_FILE", "")
    from vigil.adapters.keychain import InMemoryBackend, Keychain

    monkeypatch.setattr(cli, "_keychain", lambda: Keychain(InMemoryBackend()))
    return tmp_path / "data"


def test_every_service_method_has_a_command():
    """Reachability: the parser knows a command for each thing the service can do."""
    commands = build_parser()._subparsers._group_actions[0].choices
    assert {"where", "users", "cameras", "zones", "run", "incidents", "export", "verify", "audit", "alerts", "backup", "restore", "health"} <= set(commands)


def test_an_open_site_then_accounts_then_sign_in(data, monkeypatch, capsys):
    assert main(["where"]) == 0
    assert "no account exists" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", io.StringIO("a strong one\n"))
    assert main(["users", "add", "root", "--role", "ADMIN", "--stdin"]) == 0
    assert main(["users", "list"]) != 0, "accounts exist; an anonymous command is refused"
    monkeypatch.setattr("sys.stdin", io.StringIO("a strong one\n"))
    assert main(["--as", "root", "--password-stdin", "users", "list"]) == 0
    assert "root" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO("wrong\n"))
    assert main(["--as", "root", "--password-stdin", "users", "list"]) == 1


def test_cameras_zones_run_incidents_export_and_audit_end_to_end(data, reference_video, capsys):
    assert main(["cameras", "add", "gate", str(reference_video), "--place", "33.8938,35.5018,4,0,-25"]) == 0
    assert main(["cameras", "list"]) == 0 and "placed" in capsys.readouterr().out
    from test_runtime import _zone_where_the_block_stops
    from vigil.domain.geo import CameraPose, LatLon

    ring = ";".join(f"{p.lat},{p.lon}" for p in _zone_where_the_block_stops(CameraPose(LatLon(33.8938, 35.5018), 4.0, 0.0, -25.0)))
    assert main(["zones", "add", "yard", ring, "--enter-after", "300"]) == 0
    assert main(["run", "--no-model", "--for", "40"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"\d+ event\(s\) -> \d+ incident\(s\)", out)
    assert main(["incidents"]) == 0
    listing = capsys.readouterr().out
    match = re.search(r"(inc-[0-9a-f]{16})", listing)
    assert match, listing
    assert main(["export", match.group(1), "--to", str(data / "ev")]) == 0
    folder = next((data / "ev").iterdir())
    assert main(["verify", str(folder)]) == 0
    assert main(["audit"]) == 0
    trail = capsys.readouterr().out
    assert "camera.added" in trail and "incident.exported" in trail
    assert main(["health"]) == 0
    assert main(["alerts", "--test"]) == 0
    assert main(["backup", "--to", str(data / "b.db")]) == 0
    assert main(["restore", str(data / "b.db")]) == 0
    assert main(["site", "name", "Depot", "--timezone", "Asia/Beirut"]) == 0
    assert main(["site", "show"]) == 0 and "Depot" in capsys.readouterr().out
    assert main(["retention", "--max-age-days", "0", "--min-free-gib", "0"]) == 0


def test_a_password_in_argv_is_warned_about_and_never_stored(data, capsys):
    assert main(["cameras", "add", "cam", "rtsp" + "://u:secret@10.0.0.9/s"]) == 0
    err = capsys.readouterr().err
    assert "readable by every process" in err
    assert main(["cameras", "list"]) == 0
    assert "secret" not in capsys.readouterr().out
    log = (data / "logs" / "vigil.log").read_text(encoding="utf-8")
    assert "secret" not in log


def test_a_stop_can_be_asked_for_and_the_service_definition_printed(data, capsys, monkeypatch):
    from vigil.service.supervise import stop_file

    assert main(["run", "--stop"]) == 0
    assert stop_file(data).is_file()
    assert main(["service", "print"]) == 0
    out = capsys.readouterr().out
    assert "supervise" in out and ("install:" in out)
    # A stale request must not stop the next run before it starts.
    assert main(["run", "--no-model", "--for", "1"]) == 2, "no camera yet"
    assert not stop_file(data).is_file()


def test_supervise_runs_the_child_and_a_stop_ends_it(data, monkeypatch, capsys):
    from vigil.service import supervise as supervise_module

    attempts = []
    monkeypatch.setattr(supervise_module, "_run_child", lambda command: attempts.append(command) or 3)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert main(["supervise", "--max-restarts", "2", "--", "run", "--for", "1"]) == 3
    assert len(attempts) == 3 and "run" in attempts[0]
    assert "--data-dir" in attempts[0], "the child inherits the data directory it was supervised from"


def test_the_console_is_dispatched_however_the_global_flags_are_ordered(monkeypatch):
    """`vigil --data-dir X console --for 20` reached argparse, which refused it."""
    from vigil.interfaces.cli import _command_of

    handed = []
    monkeypatch.setattr("vigil.interfaces.console.main.run", lambda argv: handed.append(list(argv)) or 0)
    assert _command_of(["console", "--for", "5"]) == ("console", 0)
    assert _command_of(["--data-dir", "X", "console", "--for", "5"]) == ("console", 2)
    assert _command_of(["--verbose", "--as", "alice", "console"]) == ("console", 3)
    assert _command_of(["cameras", "add", "console", "device:0"]) == ("cameras", 0), "a camera called console is a value"
    assert _command_of([]) == (None, 0)

    assert main(["--data-dir", "X", "console", "--for", "5", "--start"]) == 0
    assert handed == [["--data-dir", "X", "--for", "5", "--start"]], handed
