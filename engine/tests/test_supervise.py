"""The node restarts when it dies, stops when asked, and registers with the OS.

Nothing here runs a real node or a real `schtasks`: the process runner and
the clock are injected, so what is tested is the decision — restart, wait,
give up, stop — and the definitions each platform is handed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sentinel import supervise


def test_a_failing_node_is_restarted_with_a_growing_pause_then_stops_on_a_clean_exit(tmp_path: Path):
    codes = iter([1, 1, 1, 0])
    runs, waited = [], []

    def run(command):
        runs.append(list(command))
        return next(codes)

    result = supervise.supervise(
        ["node", "x"], stop=tmp_path / "node.stop", run=run, sleep=waited.append
    )

    assert result == 0
    assert len(runs) == 4 and all(r == ["node", "x"] for r in runs)
    assert waited == [1.0, 2.0, 5.0], "the pause between attempts must grow"


def test_the_restart_limit_is_honoured_and_the_last_code_returned(tmp_path: Path):
    result = supervise.supervise(
        ["node"], max_restarts=2, stop=tmp_path / "node.stop",
        run=lambda command: 3, sleep=lambda seconds: None,
    )
    assert result == 3


def test_a_requested_stop_ends_supervision_even_after_a_bad_exit(tmp_path: Path):
    stop = tmp_path / "node.stop"
    runs = []

    def run(command):
        runs.append(1)
        stop.write_text("stop", encoding="utf-8")
        return 1

    assert supervise.supervise(["node"], stop=stop, run=run, sleep=lambda s: None) == 0
    assert runs == [1], "the supervisor restarted a node that was asked to stop"


def test_the_backoff_is_capped_rather_than_unbounded(tmp_path: Path):
    codes = iter([1] * 9 + [0])
    waited = []
    supervise.supervise(["node"], stop=tmp_path / "s", run=lambda c: next(codes), sleep=waited.append)
    assert waited[-1] == supervise.BACKOFF_SECONDS[-1]
    assert max(waited) == supervise.BACKOFF_SECONDS[-1]


def test_the_stop_file_lives_in_the_data_directory_and_round_trips(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path))
    assert supervise.stop_file() == tmp_path / "node.stop"
    written = supervise.request_stop()
    assert written.is_file()
    assert supervise.clear_stop() is True
    assert not written.exists()
    assert supervise.clear_stop() is False


def test_the_child_command_names_the_node_from_a_checkout_and_from_the_bundle(monkeypatch):
    assert supervise.child_command(["--record"]) == [sys.executable, "-m", "sentinel", "node", "--record"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert supervise.child_command(["--record"]) == [sys.executable, "node", "--record"]


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_every_platform_gets_a_definition_that_names_the_supervisor(platform):
    definition = supervise.service_definition(platform, ["--record"])
    joined = " ".join(definition["install"]) + (definition["content"] or "")
    assert "supervise" in joined and "--record" in joined
    assert definition["uninstall"]
    assert definition["note"]
    if platform == "win32":
        assert definition["path"] is None and "schtasks" in definition["install"][0]
        assert "ONLOGON" in definition["install"]
    else:
        assert definition["path"] is not None and definition["content"]
        # The supervisor restarts; the platform must not fight it.
        assert "Restart=no" in definition["content"] or "<false/>" in definition["content"]


def test_install_writes_the_unit_and_runs_the_platforms_command(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    ran = []
    code, definition = supervise.install_service(["--record"], platform="linux", run=lambda c: ran.append(list(c)) or 0)
    assert code == 0
    assert definition["path"].is_file() and "ExecStart=" in definition["path"].read_text(encoding="utf-8")
    assert ran == [definition["install"]]

    code, definition = supervise.uninstall_service(platform="linux", run=lambda c: ran.append(list(c)) or 0)
    assert code == 0 and not definition["path"].exists()
    assert ran[-1] == definition["uninstall"]


def test_windows_install_runs_schtasks_and_writes_no_file(tmp_path: Path):
    ran = []
    code, definition = supervise.install_service(["--record"], platform="win32", run=lambda c: ran.append(list(c)) or 0)
    assert code == 0 and definition["path"] is None
    assert ran[0][:4] == ["schtasks", "/Create", "/TN", "SentinelVision"]
