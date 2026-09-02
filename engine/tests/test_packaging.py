"""Tests for the packaged build's paths and entry point.

Three executables come out of one PyInstaller analysis, and which application
starts is decided from the name the operator launched. That dispatch is the
whole difference between them, it runs before anything else, and in a packaged
build a mistake in it produces a window that opens and closes with no trace —
so it is tested here from a checkout, where a failure is legible.

The path tests matter for a different reason. A packaged install lives in
Program Files or /usr/lib, which the account running it cannot write to, and a
security system that silently fails to record because its install directory is
read-only is worse than one that refuses to start.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from sentinel import paths

ROOT = Path(__file__).resolve().parents[2]


def load_entry():
    """Load `packaging/entry.py` by path — it is not an installed module."""
    spec = importlib.util.spec_from_file_location(
        "sentinel_packaging_entry", ROOT / "packaging" / "entry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    "executable, expected",
    [
        ("SentinelVision.exe", "console"),
        ("SentinelVision-dev.exe", "console-verbose"),
        ("sentinel.exe", "cli"),
        # Linux and macOS have no extension.
        ("SentinelVision", "console"),
        ("SentinelVision-dev", "console-verbose"),
        ("sentinel", "cli"),
        # Case does not decide anything: Windows filesystems are not case
        # sensitive and a rename must not change which application starts.
        ("SENTINELVISION.EXE", "console"),
        ("Sentinel.EXE", "cli"),
    ],
)
def test_the_executable_name_decides_which_application_starts(
    executable: str, expected: str, monkeypatch
):
    entry = load_entry()
    started: list[tuple[str, object]] = []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", f"/opt/sentinel/{executable}")
    monkeypatch.setattr(sys, "argv", [f"/opt/sentinel/{executable}"])

    class FakeCli:
        @staticmethod
        def main(argv=None):
            started.append(("cli", argv))
            return 0

    class FakeConsole:
        @staticmethod
        def run(argv=None):
            started.append(("console", argv))
            return 0

    monkeypatch.setitem(sys.modules, "sentinel.cli", FakeCli)
    monkeypatch.setitem(sys.modules, "sentinel_console.app", FakeConsole)

    assert entry.main() == 0
    assert len(started) == 1

    kind, argv = started[0]
    if expected == "cli":
        assert kind == "cli"
    else:
        assert kind == "console"
        verbose = "--verbose" in (argv or [])
        assert verbose is (expected == "console-verbose")


def test_the_developer_build_does_not_duplicate_a_verbose_the_operator_typed(
    monkeypatch,
):
    entry = load_entry()
    seen: list[list[str]] = []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/sentinel/SentinelVision-dev")
    monkeypatch.setattr(sys, "argv", ["/opt/sentinel/SentinelVision-dev", "--verbose"])

    class FakeConsole:
        @staticmethod
        def run(argv=None):
            seen.append(list(argv or []))
            return 0

    monkeypatch.setitem(sys.modules, "sentinel_console.app", FakeConsole)

    entry.main()

    assert seen == [["--verbose"]]


def test_arguments_reach_the_application(monkeypatch, tmp_path: Path):
    entry = load_entry()
    seen: list[list[str]] = []

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/sentinel/SentinelVision")
    monkeypatch.setattr(
        sys, "argv", ["/opt/sentinel/SentinelVision", "--database", str(tmp_path / "x.db")]
    )

    class FakeConsole:
        @staticmethod
        def run(argv=None):
            seen.append(list(argv or []))
            return 0

    monkeypatch.setitem(sys.modules, "sentinel_console.app", FakeConsole)

    entry.main()

    assert seen == [["--database", str(tmp_path / "x.db")]]


# ---------------------------------------------------------------------- paths


def test_every_path_lives_under_one_directory(monkeypatch, tmp_path: Path):
    # "Where does this thing keep my files" has one answer, so an operator asked
    # to send the logs finds one folder with everything in it.
    monkeypatch.setenv(paths.DATA_DIR_VARIABLE, str(tmp_path))

    root = paths.data_directory()

    assert root == tmp_path
    for path in (paths.database_path(), paths.log_directory(), paths.evidence_directory()):
        assert root in path.parents


def test_the_override_wins_over_the_operating_system_default(monkeypatch, tmp_path: Path):
    # The normal case for a security appliance: a dedicated disk, not the
    # system volume, because continuous video does not belong there.
    monkeypatch.setenv(paths.DATA_DIR_VARIABLE, str(tmp_path / "evidence-volume"))

    assert paths.data_directory() == tmp_path / "evidence-volume"


def test_the_default_is_never_beside_the_code(monkeypatch):
    # A packaged install lives somewhere the running account cannot write.
    monkeypatch.delenv(paths.DATA_DIR_VARIABLE, raising=False)

    default = paths.data_directory()

    assert ROOT not in default.parents and default != ROOT
    assert default.name == "SentinelVision"


def test_reading_a_path_does_not_create_it(monkeypatch, tmp_path: Path):
    # Asking where something lives must not have the side effect of making it.
    target = tmp_path / "not-yet"
    monkeypatch.setenv(paths.DATA_DIR_VARIABLE, str(target))

    paths.data_directory()
    paths.log_directory()
    paths.database_path()

    assert not target.exists()

    assert paths.ensure_data_directory() == target
    assert target.is_dir()


def test_a_checkout_is_not_a_frozen_build():
    assert paths.is_frozen() is False
    assert paths.bundle_directory() is None


def test_the_database_default_agrees_with_the_paths_module(monkeypatch, tmp_path: Path):
    # Two definitions of "the data directory" is one too many: the log would go
    # one place and the database another, and neither would be where the
    # operator was told to look.
    from sentinel.store import default_database_path

    monkeypatch.setenv(paths.DATA_DIR_VARIABLE, str(tmp_path))

    assert default_database_path() == paths.database_path()


# ------------------------------------------------------------- the spec itself


def test_the_spec_refuses_to_package_without_a_built_core():
    # Packaging a stale or absent library is how a green test run ships broken
    # geometry: the tests loaded one core and the bundle carries another.
    spec = (ROOT / "packaging" / "sentinel.spec").read_text(encoding="utf-8")

    assert "is not built" in spec
    assert "raise SystemExit" in spec


def test_the_bundle_excludes_the_browser_engine():
    # A control-room console must not ship a browser: doing so inherits its
    # update cadence, its memory profile and its network assumptions. The
    # console tests assert no module imports it; this asserts the packaged
    # bundle cannot even carry it.
    spec = (ROOT / "packaging" / "sentinel.spec").read_text(encoding="utf-8")

    for forbidden in ("QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick"):
        assert forbidden in spec, f"{forbidden} must be excluded from the bundle"


def test_the_bundle_is_not_compressed_with_upx():
    # A packed binary looks exactly like malware to every endpoint product an
    # operator runs, and a security appliance that trips the antivirus is a
    # security appliance that gets uninstalled.
    spec = (ROOT / "packaging" / "sentinel.spec").read_text(encoding="utf-8")

    assert "upx=True" not in spec
    assert spec.count("upx=False") >= 2
