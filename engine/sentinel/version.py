"""What build this is, for every place a person might ask.

Until this module existed the version lived in one constant inside the evidence
exporter and nowhere a person could see it: the About dialog had none, `sentinel
where` printed none, the log started without one, and a bug report could not
say which build it was about. A security system's evidence names the software
that produced it, and "0.1.0" alone does not name a build — two bundles a week
apart both carried it.

So the answer has three parts and one source: the **version** here, the
**commit** the code came from, and **when** the bundle was built. A checkout
asks git; a packaged bundle cannot, so `python tasks.py package` writes
``build.json`` beside the executables and this reads it back. Nothing here
reaches the network, imports anything heavy, or fails: a build that cannot be
identified says so rather than refusing to start.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: The one place the version is written. `pyproject.toml` must agree, and a
#: test holds the two together.
__version__ = "0.1.0"

#: Written beside the executables by `tasks.py package`; read here in a frozen
#: build. Beside them rather than inside `_internal`, like `models/`, so a
#: person can open it.
BUILD_FILE = "build.json"

#: How long a checkout waits for git before deciding it cannot say.
_GIT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """Version, commit and build time, each ``None`` when genuinely unknown."""

    version: str
    commit: str | None
    built_at: str | None
    #: ``"packaged"`` when read from `build.json`, ``"checkout"`` when asked of
    #: git, ``"unknown"`` when neither could answer.
    source: str

    def describe(self) -> str:
        """One line: ``Sentinel Vision 0.1.0 (d38c64e, built 2026-09-05 10:05 UTC)``."""
        parts = []
        if self.commit:
            parts.append(self.commit)
        if self.built_at:
            parts.append(f"built {self.built_at}")
        if self.source == "checkout":
            parts.append("checkout")
        elif self.source == "unknown":
            parts.append("build unidentified")
        detail = f" ({', '.join(parts)})" if parts else ""
        return f"Sentinel Vision {self.version}{detail}"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def build_file_path() -> Path | None:
    """Where a packaged build keeps its stamp, or ``None`` in a checkout."""
    if not _is_frozen():
        return None
    return Path(sys.executable).resolve().parent / BUILD_FILE


def _from_git(root: Path) -> str | None:
    """The short commit, with ``+dirty`` when the tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS,
        )
        if commit.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    short = commit.stdout.strip() or None
    if short and status.returncode == 0 and status.stdout.strip():
        short += "+dirty"
    return short


def read_build_file(path: Path) -> BuildInfo:
    """A stamp written by `write_build_file`, or an honest unknown."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return BuildInfo(
            version=str(document.get("version") or __version__),
            commit=document.get("commit") or None,
            built_at=document.get("built_at") or None,
            source="packaged",
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return BuildInfo(version=__version__, commit=None, built_at=None, source="unknown")


def write_build_file(directory: Path, *, commit: str | None, built_at: str | None = None) -> Path:
    """Stamp a bundle. Called by `tasks.py package` after the executables exist."""
    if built_at is None:
        built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path = Path(directory) / BUILD_FILE
    path.write_text(
        json.dumps(
            {"version": __version__, "commit": commit, "built_at": built_at},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


_cached: BuildInfo | None = None


def build_info(build_file: Path | None = None) -> BuildInfo:
    """What this process is running. Cached: git is asked once, not per line.

    ``build_file`` overrides the automatic choice, for a test — or for a tool
    that wants to identify a bundle it has not launched.
    """
    global _cached
    if build_file is not None:
        return read_build_file(Path(build_file))
    if _cached is None:
        stamped = build_file_path()
        if stamped is not None:
            _cached = read_build_file(stamped)
        else:
            root = Path(__file__).resolve().parents[2]
            commit = _from_git(root)
            _cached = BuildInfo(
                version=__version__, commit=commit, built_at=None,
                source="checkout" if commit else "unknown",
            )
    return _cached


def describe() -> str:
    """The one-line identity of this build."""
    return build_info().describe()
