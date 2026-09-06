"""Logging: one file, redaction of anything that looks like a password, a crash log."""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import sys
from pathlib import Path

LEVEL_VARIABLE = "VIGIL_LOG_LEVEL"
_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")
_configured = False
_crash_handle = None


class _Redact(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _PASSWORD.sub(r"\1***@", super().format(record))


def configure(directory: Path | None = None, *, level: str | None = None) -> Path | None:
    global _configured, _crash_handle
    root = logging.getLogger("vigil")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    chosen = (level or os.environ.get(LEVEL_VARIABLE) or "INFO").upper()
    root.setLevel(getattr(logging, chosen, logging.INFO))
    formatter = _Redact("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
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
    _configured = True
    return path


def get(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("vigil") else f"vigil.{name}")


def redact(text: str) -> str:
    return _PASSWORD.sub(r"\1***@", text)
