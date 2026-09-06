"""Logging: one file, redaction of anything that looks like a password, a crash log."""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import sys
from pathlib import Path

LEVEL_VARIABLE = "VIGIL_LOG_LEVEL"
#: One JSON object per line instead of prose. For a node nobody is watching,
#: where the reader is a monitoring agent rather than a person.
JSON_VARIABLE = "VIGIL_LOG_JSON"
_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")
_crash_handle = None


class _Redact(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _PASSWORD.sub(r"\1***@", super().format(record))


class _Json(logging.Formatter):
    """One object per line. Redacted by the same rule as the prose form.

    Anything a caller passes as ``extra`` is carried through, so a metrics
    line is machine-readable without a second logging path to keep in step.
    """

    _BUILT_IN = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))

    def format(self, record: logging.LogRecord) -> str:
        import json as _json

        payload = {
            "at": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": _PASSWORD.sub(r"\1***@", record.getMessage()),
        }
        for key, value in vars(record).items():
            if key not in self._BUILT_IN and key not in ("message", "asctime"):
                payload[key] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        if record.exc_info:
            payload["exception"] = _PASSWORD.sub(r"\1***@", self.formatException(record.exc_info))
        return _json.dumps(payload, default=str)


def configure(directory: Path | None = None, *, level: str | None = None, json: bool | None = None) -> Path | None:
    """Set up logging. ``json`` overrides `JSON_VARIABLE` when it is given."""
    global _crash_handle
    root = logging.getLogger("vigil")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    chosen = (level or os.environ.get(LEVEL_VARIABLE) or "INFO").upper()
    root.setLevel(getattr(logging, chosen, logging.INFO))
    as_json = json if json is not None else os.environ.get(JSON_VARIABLE, "").strip().lower() in ("1", "true", "yes")
    formatter = _Json() if as_json else _Redact("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    # A windowed build has no standard error at all — on Windows a
    # GUI-subsystem executable is given none, so `sys.stderr` is `None`.
    # `StreamHandler(None)` does not fail on construction; it fails on the
    # first log line, inside logging's own error handling, which then tries to
    # report the failure to the stream that is not there. The file handler
    # below is the one that matters for a windowed run anyway.
    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    path = None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "vigil.log"
        file = logging.FileHandler(path, encoding="utf-8")
        file.setFormatter(formatter)
        root.addHandler(file)
        try:
            _crash_handle = open(directory / "crash.log", "a", encoding="utf-8")  # noqa: SIM115
            faulthandler.enable(_crash_handle)
        except OSError:
            _crash_handle = None
    root.propagate = False
    return path


def get(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("vigil") else f"vigil.{name}")


def redact(text: str) -> str:
    return _PASSWORD.sub(r"\1***@", text)
