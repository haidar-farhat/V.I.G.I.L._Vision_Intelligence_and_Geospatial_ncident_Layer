"""PyInstaller entry point for the **windowed** build.

Two executables come out of one codebase, and the difference is not cosmetic.

`vigil.exe` is a console-subsystem binary: it has a standard output, so every
command prints, pipes and redirects the way a command line is expected to.
Running the console from it works, and leaves a black terminal window sitting
behind the operator's window for the length of the session.

`vigil-console.exe` is a GUI-subsystem binary: Windows gives it no console, so
nothing flashes and nothing lingers. What it also gives it is **no standard
output and no standard error at all** — `sys.stdout` and `sys.stderr` are
`None`, and anything that writes to them without checking raises inside
whatever was doing the writing. `vigil.logs.configure` skips its stream
handler when there is no stream; the log file is the record for a windowed
run, and `vigil where` will say where it is.

The two are built from the same tree by `python tasks.py package`, so they
cannot drift.
"""

import sys

# A GUI-subsystem process on Windows has no standard streams. Give it inert
# ones before anything else imports: a stray `print` in a dependency is
# otherwise an `AttributeError` on `None` at an arbitrary depth, and the
# window never appears with no indication of why.
#
# The null device rather than a `StringIO`: an in-memory buffer would hold
# every line the process ever wrote, and this window is meant to be left open
# for a shift. Discarding is also honest — there is no console to read it in,
# and the log file has the same lines.
if sys.stdout is None or sys.stderr is None:
    import os

    _null = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - lives as long as the process
    if sys.stdout is None:
        sys.stdout = _null
    if sys.stderr is None:
        sys.stderr = _null

from vigil.interfaces.console.main import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
