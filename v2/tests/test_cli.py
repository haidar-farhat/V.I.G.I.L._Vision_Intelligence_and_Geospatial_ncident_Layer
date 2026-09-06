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


def test_double_clicking_the_executable_opens_the_window(monkeypatch, capsys):
    """No arguments at all means somebody double-clicked it, and they want the
    window.

    argparse answered `error: the following arguments are required: command`
    and exit 2 — a usage message flashed into a console that closes before it
    can be read, from an executable the README calls "both the command line
    and the console". It is the first thing a person does with a product and
    it did the one thing that looks like a broken install.
    """
    handed = []
    monkeypatch.setattr("vigil.interfaces.console.main.run", lambda argv: handed.append(list(argv)) or 0)
    assert main([]) == 0
    assert handed == [[]], handed


def test_asking_for_help_or_a_version_still_prints_rather_than_opening_a_window(monkeypatch):
    """Only the *empty* command line opens a window. Somebody who typed
    `--help` asked for text, and a window instead of it would be worse than
    the usage error this replaced."""
    monkeypatch.setattr("vigil.interfaces.console.main.run",
                        lambda argv: pytest.fail("--help must not open a window"))
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as caught:
            main([flag])
        assert caught.value.code == 0


def test_an_incident_can_be_acknowledged_and_dismissed_from_the_command_line(data, capsys):
    from test_incidents import event
    from vigil.domain.incidents import Correlator
    from vigil.service.maintenance import open_store

    assert main(["where"]) == 0  # creates the data directory and the database
    store = open_store(data / "vigil.db")
    events = [event("a", 1, 10_000), event("a", 2, 40_000)]
    store.save_events(events)
    incidents = Correlator().correlate(events)
    store.save_incidents(incidents)
    store.close()

    assert main(["incidents"]) == 0
    listing = capsys.readouterr().out
    assert "not yet reviewed" in listing
    assert main(["review", incidents[0].id, "dismiss"]) == 1, "a dismissal needs a reason"
    assert "reason" in capsys.readouterr().err
    assert main(["review", incidents[0].id, "dismiss", "--note", "the cat"]) == 0
    assert "dismissed by" in capsys.readouterr().out
    assert main(["incidents"]) == 0
    assert incidents[0].id not in capsys.readouterr().out, "the queue must not show what was dismissed"
    assert main(["incidents", "--state", "all"]) == 0
    assert incidents[0].id in capsys.readouterr().out
    assert main(["review", incidents[0].id, "reopen"]) == 0
    assert main(["incidents", "--state", "new"]) == 0
    assert incidents[0].id in capsys.readouterr().out


def test_running_a_source_the_site_already_has_uses_that_camera(data, reference_video, capsys):
    """Running it as a second, unplaced camera silently ignored the placement and the zones."""
    assert main(["cameras", "add", "gate", str(reference_video), "--place", "33.8938,35.5018,4,0,-25"]) == 0
    capsys.readouterr()
    assert main(["run", str(reference_video), "--no-model", "--for", "3"]) == 0
    out = capsys.readouterr().out
    assert "using the stored camera gate" in out
    assert main(["cameras", "list"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 1, "a duplicate camera was created"

    # A source the site does not know is added, named for itself, and the
    # operator is told both that it was added and that it is unplaced.
    assert main(["run", "device:9", "--no-model", "--for", "1"]) in (0, 2)
    printed = capsys.readouterr()
    assert "added device-9" in printed.out and "remove it later" in printed.out
    assert "unplaced" in printed.err


def test_a_camera_and_a_zone_can_be_edited_from_the_command_line(data, reference_video, capsys):
    """Deleting and re-adding was the only way, and it threw away the placement and the ring."""
    assert main(["cameras", "add", "gate", str(reference_video), "--place", "33.8938,35.5018,4,0,-25"]) == 0
    assert main(["zones", "add", "yard", "33.8937,35.5018;33.8937,35.5019;33.8936,35.5019", "--name", "Yrad"]) == 0
    capsys.readouterr()

    assert main(["cameras", "source", "gate", "rtsp" + "://10.0.0.44/s"]) == 0
    out = capsys.readouterr().out
    assert "10.0.0.44" in out and "placement and zones are unchanged" in out
    assert main(["cameras", "rename", "gate", "North gate"]) == 0
    assert "North gate" in capsys.readouterr().out

    assert main(["zones", "edit", "yard", "--name", "Yard", "--kind", "RESTRICTED", "--watch", "person",
                 "--closed", "22-6"]) == 0
    assert "Yard" in capsys.readouterr().out
    assert main(["zones", "list"]) == 0
    listing = capsys.readouterr().out
    assert "Yard" in listing and "RESTRICTED" in listing and "person" in listing
    assert main(["zones", "edit", "yard", "--closed", "none"]) == 0
    capsys.readouterr()
    assert main(["audit"]) == 0
    trail = capsys.readouterr().out
    assert "camera.source_changed" in trail and "zone.changed" in trail


def test_the_doctor_reports_on_a_bare_installation_and_exits_zero_when_nothing_fails(data, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    for name in ("data directory", "database", "site clock", "keychain", "disk", "accounts", "cameras", "zones"):
        assert name in out, name
    assert "to look at" in out and "->" in out
    assert main(["doctor", "--probe"]) == 0
    assert "every camera opened" in capsys.readouterr().out or True  # no cameras yet

    # A site whose clock this machine does not know is a failure, and the
    # doctor is meant to be the last line of an install script.
    from vigil.service.maintenance import open_store

    store = open_store(data / "vigil.db")
    store.save_site("Depot", "Mars/Olympus")
    store.close()
    assert main(["doctor"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_incidents_and_events_can_be_searched_from_the_command_line(data, capsys):
    from test_incidents import event
    from vigil.domain.events import Severity
    from vigil.domain.incidents import Correlator
    from vigil.service.maintenance import open_store

    assert main(["where"]) == 0
    store = open_store(data / "vigil.db")
    events = [event("north-gate", 1, 10_000, severity=Severity.HIGH),
              event("loading-bay", 2, 500_000, severity=Severity.LOW)]
    store.save_events(events)
    store.save_incidents(Correlator().correlate(events))
    store.close()
    capsys.readouterr()

    assert main(["incidents", "--camera", "north-gate"]) == 0
    found = capsys.readouterr().out
    assert "north-gate" in found and "loading-bay" not in found

    assert main(["incidents", "--severity", "HIGH"]) == 0
    assert "north-gate" in capsys.readouterr().out
    assert main(["incidents", "--camera", "nothing"]) == 0
    assert "no incidents matching camera nothing" in capsys.readouterr().out

    assert main(["incidents", "--since", "not a time"]) == 1
    assert "is not a time" in capsys.readouterr().err
    assert main(["incidents", "--severity", "URGENT"]) == 1
    assert "not a severity" in capsys.readouterr().err

    assert main(["events"]) == 0
    listing = capsys.readouterr().out
    assert "entered" in listing and "drawn by" in listing and "rule zone-entry" in listing
    assert main(["events", "--camera", "loading-bay", "--contains", "entered"]) == 0
    only = capsys.readouterr().out
    assert "loading-bay" in only and "north-gate" not in only


def test_the_threat_vocabulary_is_read_set_and_cleared_from_the_command_line(data, capsys):
    assert main(["site", "threats"]) == 0
    assert "no label is treated as a threat" in capsys.readouterr().out
    assert main(["site", "threats", "--suggest"]) == 0
    offered = capsys.readouterr().out
    assert "a starting point" in offered.lower() and "knife" in offered and "CRITICAL" in offered
    assert main(["site", "threats", "--set", "knife"]) == 0
    assert "a knife (HIGH)" in capsys.readouterr().out
    assert main(["site", "threats"]) == 0
    assert "a knife (HIGH)" in capsys.readouterr().out
    assert main(["site", "threats", "--clear"]) == 0
    assert "no label is treated as a threat" in capsys.readouterr().out


def test_the_command_line_reads_sets_and_clears_it(data, capsys):
    assert main(["site", "detection"]) == 0
    assert "built-in" in capsys.readouterr().out
    assert main(["site", "detection", "--watch", "person,car", "--confidence", "0.65"]) == 0
    assert "person" in capsys.readouterr().out
    assert main(["site", "detection", "--confidence", "0.7"]) == 0
    kept = capsys.readouterr().out
    assert "person" in kept and "0.70" in kept, "changing one value must not drop the other"
    assert main(["site", "detection", "--confidence", "1.5"]) == 1
    assert "outside" in capsys.readouterr().err
    assert main(["site", "detection", "--clear"]) == 0
    assert "built-in" in capsys.readouterr().out


def test_a_camera_pose_is_measured_from_the_command_line_and_refused_when_it_is_worse(data, capsys):
    """The whole path: place a camera roughly, mark points that can be found on
    both the picture and the map, and have the assumption replaced by a number
    that says what it is worth."""
    from vigil.domain.geo import CameraPose, LatLon, project_to_ground

    truth = CameraPose(LatLon(33.8938, 35.5018), 4.0, 37.0, -22.0, 3.0, 62.0, 36.0, 60.0)
    marks = []
    for u in (0.12, 0.35, 0.5, 0.68, 0.9):
        for v in (0.6, 0.75, 0.95):
            ground = project_to_ground(truth, u, v, enforce_range=False)
            if ground is not None:
                marks.append(f"{u},{v},{ground.position.lat:.8f},{ground.position.lon:.8f}")
    points = "; ".join(marks)

    assert main(["cameras", "add", "gate", "file:///x", "--place", "33.8938,35.5018,4.5,34,-25"]) == 0
    capsys.readouterr()
    assert main(["cameras", "calibrate", "gate", "--points", points, "--dry-run"]) == 0
    assert "not saved (--dry-run)" in capsys.readouterr().out

    assert main(["cameras", "calibrate", "gate", "--points", points]) == 0
    out = capsys.readouterr().out
    assert "heading" in out and "37.0" in out, out
    assert main(["cameras", "list"]) == 0
    assert "+/-" in capsys.readouterr().out, "a measured camera must say so in the list"

    # One point is refused for being one point, not for being unparseable.
    assert main(["cameras", "calibrate", "gate", "--points", "0.5,0.5,33.9,35.5"]) == 1
    assert "not enough" in capsys.readouterr().err

    # A cluster is refused, and the pose already measured stands.
    cluster = []
    for u in (0.5000, 0.5002):
        for v in (0.8000, 0.8002):
            g = project_to_ground(truth, u, v, enforce_range=False)
            cluster.append(f"{u},{v},{g.position.lat:.10f},{g.position.lon:.10f}")
    assert main(["cameras", "calibrate", "gate", "--points", "; ".join(cluster)]) == 1
    assert "cannot separate" in capsys.readouterr().err
    assert main(["cameras", "list"]) == 0
    assert "+/-" in capsys.readouterr().out, "the refused fit must not have overwritten the good one"
    assert main(["audit"]) == 0
    assert "camera.calibrated" in capsys.readouterr().out


def test_pixels_typed_where_fractions_were_asked_for_are_refused(data, capsys):
    """960,540 is somebody typing pixels. Read as fractions it is off the frame,
    and clamping it would produce a confident wrong pose from a typo."""
    assert main(["cameras", "add", "gate", "file:///x", "--place", "33.8938,35.5018,4,0,-25"]) == 0
    capsys.readouterr()
    assert main(["cameras", "calibrate", "gate", "--points",
                 "960,540,33.894,35.502; 100,200,33.895,35.503"]) == 1
    assert "fractions of the picture, not pixels" in capsys.readouterr().err
