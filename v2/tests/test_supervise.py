import sys
from pathlib import Path

import pytest

from vigil.service.supervise import (
    BACKOFF_SECONDS, SERVICE_NAME, clear_stop, install_service, request_stop, service_definition, stop_file,
    supervise, supervisor_command, uninstall_service,
)


def test_a_clean_exit_is_not_restarted(tmp_path):
    calls = []
    assert supervise(["x"], data_dir=tmp_path, run=lambda c: calls.append(c) or 0, sleep=lambda s: None) == 0
    assert len(calls) == 1, "a clean exit is a decision, not a failure"


def test_a_crash_restarts_with_a_growing_pause_until_the_limit(tmp_path):
    pauses, attempts = [], []
    code = supervise(["x"], data_dir=tmp_path, max_restarts=3,
                     run=lambda c: attempts.append(c) or 7, sleep=pauses.append)
    assert code == 7 and len(attempts) == 4, "the first run plus three restarts"
    assert pauses == list(BACKOFF_SECONDS[:3])


def test_a_stop_request_ends_the_supervisor_and_the_running_process_clears_it(tmp_path):
    path = request_stop(tmp_path)
    assert path == stop_file(tmp_path) and path.is_file()
    assert supervise(["x"], data_dir=tmp_path, run=lambda c: 1, sleep=lambda s: None) == 0
    assert clear_stop(tmp_path) and not clear_stop(tmp_path)


def test_every_platform_has_a_definition_that_says_what_it_writes_and_runs():
    for platform, kind in (("win32", "scheduled task"), ("darwin", "launch agent"), ("linux", "systemd user unit")):
        definition = service_definition(platform, ["run", "--record"])
        assert definition["kind"] == kind and definition["note"]
        assert definition["install"] and definition["uninstall"]
        line = " ".join(str(p) for p in definition["install"]) + (definition["content"] or "")
        assert "supervise" in line and "run" in line
        if definition["path"] is not None:
            assert SERVICE_NAME.lower() in definition["path"].name
    assert "supervise" in supervisor_command(["run"]) and supervisor_command(["run"])[-1] == "run"


def test_install_writes_the_file_and_uninstall_removes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ran = []
    code, definition = install_service(["run"], platform="linux", run=lambda c: ran.append(c) or 0)
    assert code == 0 and definition["path"].is_file() and "ExecStart" in definition["path"].read_text(encoding="utf-8")
    code, definition = uninstall_service(platform="linux", run=lambda c: ran.append(c) or 0)
    assert code == 0 and not definition["path"].exists()
    assert [c[0] for c in ran] == ["systemctl", "systemctl"]
