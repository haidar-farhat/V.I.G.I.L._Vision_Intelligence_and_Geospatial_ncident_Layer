"""The task runner itself, where a green result can be evidence for nothing."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tasks():
    spec = importlib.util.spec_from_file_location("vigil_tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_an_unknown_task_is_an_error_and_not_a_printed_menu():
    """`tasks.py build` is not a task; it must not look like a build that worked."""
    tasks = _tasks()
    assert "build" not in tasks.TASKS
    assert {"test", "check", "capabilities", "package", "exetest"} <= set(tasks.TASKS)


def test_a_bundle_built_from_other_sources_is_reported_as_stale(tmp_path, monkeypatch):
    """An old executable runs perfectly, which is exactly why it has to be caught.

    Timestamps cannot answer this: PyInstaller reads each source at its own
    moment, so a bundle built across an edit mixes two versions while its
    executable, written last, looks newer than everything. One did — the
    packaged window imported a function the packaged domain module lacked —
    and the run still said PASS.
    """
    tasks = _tasks()
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    (tmp_path / "vigil").mkdir()
    (tmp_path / "packaging").mkdir()
    source = tmp_path / "vigil" / "thing.py"
    source.write_text("x = 1", encoding="utf-8")
    exe = tmp_path / "dist" / "vigil" / "vigil.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"binary")

    digests = tasks.source_digests()
    assert list(digests) == ["vigil/thing.py"]
    (exe.parent / "build.json").write_text(json.dumps({"sources": digests}), encoding="utf-8")
    assert tasks.stale_sources(exe) == [], "a bundle built from this tree is current"

    source.write_text("x = 2", encoding="utf-8")
    assert tasks.stale_sources(exe) == ["vigil/thing.py"]

    (tmp_path / "vigil" / "new.py").write_text("y = 1", encoding="utf-8")
    assert tasks.stale_sources(exe) == ["vigil/new.py", "vigil/thing.py"], "a file the bundle never saw counts too"

    (exe.parent / "build.json").write_text(json.dumps({"commit": "abc"}), encoding="utf-8")
    assert tasks.stale_sources(exe), "a bundle from before the check is not trusted either"
    assert tasks.stale_sources(tmp_path / "gone.exe") == [], "a missing executable is a different complaint"
