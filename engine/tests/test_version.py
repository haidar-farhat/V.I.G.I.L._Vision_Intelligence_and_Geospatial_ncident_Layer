"""The build identifies itself, everywhere a person might ask.

Until this existed the version lived in one constant in the evidence exporter
and nowhere on any screen: two bundles a week apart both said 0.1.0 and a bug
report could not say which one it was about.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sentinel import version

ROOT = Path(__file__).resolve().parents[2]


def test_the_version_here_is_the_version_the_package_declares():
    pyproject = (ROOT / "engine" / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert declared is not None
    assert declared.group(1) == version.__version__, (
        "pyproject.toml and sentinel/version.py disagree about the version"
    )


def test_a_checkout_names_its_commit_or_says_it_cannot():
    info = version.build_info()
    assert info.version == version.__version__
    assert info.source in ("checkout", "unknown")
    line = info.describe()
    assert line.startswith(f"Sentinel Vision {version.__version__}")
    if info.commit:
        assert info.commit in line
        assert re.fullmatch(r"[0-9a-f]{7,}(\+dirty)?", info.commit), info.commit


def test_a_bundle_stamp_round_trips(tmp_path: Path):
    written = version.write_build_file(tmp_path, commit="abc1234", built_at="2026-09-05 10:05 UTC")
    assert written.name == version.BUILD_FILE
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document == {"version": version.__version__, "commit": "abc1234", "built_at": "2026-09-05 10:05 UTC"}

    info = version.build_info(written)
    assert info.source == "packaged"
    assert info.commit == "abc1234"
    assert info.describe() == f"Sentinel Vision {version.__version__} (abc1234, built 2026-09-05 10:05 UTC)"


def test_a_missing_or_broken_stamp_is_an_honest_unknown_not_a_crash(tmp_path: Path):
    info = version.build_info(tmp_path / "absent.json")
    assert info.source == "unknown" and info.commit is None
    assert "unidentified" in info.describe()

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert version.build_info(broken).source == "unknown"

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    info = version.build_info(empty)
    assert info.version == version.__version__ and info.commit is None


def test_the_version_reaches_the_evidence_report_and_the_where_command():
    from sentinel import evidence
    from sentinel.cli import _where
    import argparse
    import io
    import contextlib

    assert evidence.APPLICATION_VERSION == version.__version__

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _where(argparse.Namespace(database=None))
    assert f"Sentinel Vision {version.__version__}" in out.getvalue()
