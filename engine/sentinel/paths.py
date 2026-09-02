"""Where this system keeps its files.

One definition, used by the database, the log, the evidence export and the
packaged application, so that "where is my data" has a single answer an operator
can be told.

Nothing here is beside the code. A packaged install lives in Program Files or
/usr/lib, which the account running the application cannot write to — and a
security system that silently fails to record because its install directory is
read-only is worse than one that refuses to start.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Overrides everything below. The normal case for a security appliance, which
#: has a dedicated disk and should not be putting continuous video on the system
#: volume.
DATA_DIR_VARIABLE = "SENTINEL_DATA_DIR"


def data_directory() -> Path:
    """The per-user application data directory, or the override.

    Not created here: reading where something lives must not have the side
    effect of making it. Use :func:`ensure_data_directory` at the point a file is
    actually about to be written.
    """
    override = os.environ.get(DATA_DIR_VARIABLE)
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / "SentinelVision"


def ensure_data_directory() -> Path:
    directory = data_directory()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_directory() -> Path:
    """Where log files go.

    Under the data directory rather than a system log location, because the
    files belong to this deployment and an operator asked to "send the logs"
    should find one folder with everything in it.
    """
    return data_directory() / "logs"


def database_path() -> Path:
    return data_directory() / "sentinel.db"


def evidence_directory() -> Path:
    return data_directory() / "evidence"


def recordings_directory() -> Path:
    """Where recorded video goes.

    Under the data directory like everything else, and overridable with it —
    which matters more here than anywhere else, because continuous video is the
    one thing that genuinely wants its own disk: roughly 17.5 GB per camera per
    day at 640×480/15fps.
    """
    return data_directory() / "recordings"


def is_frozen() -> bool:
    """Whether this is running from a packaged build rather than a checkout.

    PyInstaller sets both of these. Worth knowing in exactly two places: finding
    the bundled engine core, and deciding whether an unhandled exception should
    print a traceback at somebody who cannot act on it.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_directory() -> Path | None:
    """The unpacked bundle root, or ``None`` when running from a checkout."""
    return Path(sys._MEIPASS) if is_frozen() else None  # type: ignore[attr-defined]
