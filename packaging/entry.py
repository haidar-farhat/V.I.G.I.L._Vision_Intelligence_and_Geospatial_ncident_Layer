"""The packaged entry point for all three executables.

One script, three names. PyInstaller analyses a bundle once — analysing PySide6
three times costs minutes and produces the same answer — so which application
starts is decided here, from the name the operator double-clicked:

    SentinelVision.exe        the console, no terminal
    SentinelVision-dev.exe    the console, with a terminal and --verbose forced
    sentinel.exe              the headless analyser

The developer executable is not a debug build. It is the same code with its
output visible, and it exists because a packaged Qt application on Windows has
nowhere to print: an exception raised before the window appears leaves no trace
at all, and "it just closes" is the least actionable bug report there is.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _name() -> str:
    """What this executable is called, without the extension.

    `sys.executable` when frozen; `sys.argv[0]` otherwise, so the same dispatch
    can be exercised from a checkout.
    """
    frozen = getattr(sys, "frozen", False)
    return Path(sys.executable if frozen else sys.argv[0]).stem.lower()


def main() -> int:
    name = _name()

    if name.startswith("sentinel") and "vision" not in name:
        from sentinel.cli import main as cli_main

        return cli_main()

    # The console. `--verbose` is forced for the developer build rather than
    # offered, because somebody who launched the developer executable has
    # already asked for the detail.
    arguments = sys.argv[1:]
    if name.endswith("-dev") and "--verbose" not in arguments and "-v" not in arguments:
        arguments = ["--verbose", *arguments]

    from sentinel_console.app import run

    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
