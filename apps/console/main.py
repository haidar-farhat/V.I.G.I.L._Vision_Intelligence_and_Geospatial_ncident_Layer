"""Entry point for the console.

Run with ``python apps/console/main.py``, or ``sentinel-console`` once installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a checkout without an install step, which is what
# a developer does far more often than not.
_ROOT = Path(__file__).resolve().parents[2]
for candidate in (_ROOT / "engine", Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


if __name__ == "__main__":
    from sentinel_console.app import run

    raise SystemExit(run())
