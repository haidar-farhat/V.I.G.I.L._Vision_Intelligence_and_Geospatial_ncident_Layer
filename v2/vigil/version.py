"""The one place the version lives. Everything else reads it from here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

__version__ = "2.0.0a1"


def build_info() -> dict:
    """What this build is: version, commit if stamped, frozen or not."""
    info = {"version": __version__, "frozen": bool(getattr(sys, "frozen", False)), "commit": None}
    stamp = _build_file()
    if stamp is not None and stamp.is_file():
        try:
            info.update(json.loads(stamp.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return info


def describe() -> str:
    info = build_info()
    commit = info.get("commit") or "unstamped"
    kind = "packaged" if info["frozen"] else "checkout"
    return f"Sentinel Vision {info['version']} ({commit}, {kind})"


def _build_file() -> Path | None:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "build.json"
    return Path(__file__).resolve().parent.parent / "build.json"
