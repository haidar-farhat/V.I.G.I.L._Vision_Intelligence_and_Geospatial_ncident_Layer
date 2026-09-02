#!/usr/bin/env python3
"""Task runner.

    python tasks.py build      build the Rust engine core
    python tasks.py test       everything: Rust, engine, console
    python tasks.py lint       rustfmt and clippy
    python tasks.py audit      prove the source has no route off the site,
                               and that every diagram in the docs parses
    python tasks.py console    run the operator console
    python tasks.py cli        run the headless analyser (pass arguments after it)
    python tasks.py package    build the standalone executables
    python tasks.py db         report the database's migration state
    python tasks.py db-migrate apply pending migrations
    python tasks.py db-rollback undo the most recent migration
    python tasks.py check      audit, lint, build and test — what CI runs

Python rather than a Makefile or a shell script, because the product ships on
Windows, macOS and Linux and the developer commands should not be the one part
that only works on one of them.

Nothing here reaches the network except ``cargo`` resolving crates on a first
build. The test suites themselves do not, which is the point of the offline job
in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
ENGINE = ROOT / "engine"
CONSOLE = ROOT / "apps" / "console"


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a command, or stop the whole task on failure.

    Failing loudly and immediately matters more than continuing: a runner that
    presses on after a failed build reports test results for the previous
    binary, which is worse than no result at all.
    """
    where = cwd.relative_to(ROOT) if cwd != ROOT else Path(".")
    print(f"\n\033[1m$ {' '.join(command)}\033[0m  ({where})", flush=True)

    merged = {**os.environ, **(env or {})}
    result = subprocess.run(command, cwd=cwd, env=merged)
    if result.returncode != 0:
        print(f"\n\033[31mfailed: {' '.join(command)}\033[0m", file=sys.stderr)
        raise SystemExit(result.returncode)


def python_path() -> dict[str, str]:
    """Let the engine and the console import without an install step."""
    parts = [str(ENGINE), str(ENGINE / "tests"), str(CONSOLE)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return {"PYTHONPATH": os.pathsep.join(parts)}


def build() -> None:
    run(["cargo", "build", "--release"], CORE)


def lint() -> None:
    run(["cargo", "fmt", "--check"], CORE)
    run(["cargo", "clippy", "--all-targets", "--", "-D", "warnings"], CORE)


def audit() -> None:
    """The two claims that fail silently.

    The offline CI job proves the *tests* need no network. `offline_audit`
    proves the *source* has nowhere to go, which is the stronger claim and the
    one an operator is actually relying on: an untested code path can still call
    home.

    `docs_lint` is here for the same reason. A mermaid diagram that fails to
    parse renders as raw text or as nothing, with no error anywhere — and every
    architectural claim in this repository is carried by one.
    """
    run([sys.executable, str(ROOT / "tools" / "offline_audit.py")], ROOT)
    run([sys.executable, str(ROOT / "tools" / "docs_lint.py")], ROOT)


def test() -> None:
    # The core is built first, because the Python suites load it. Testing
    # against a stale library is how a green run hides a broken change.
    run(["cargo", "test"], CORE)
    build()
    run([sys.executable, "-m", "pytest"], ENGINE, python_path())
    run(
        [sys.executable, "-m", "pytest"],
        CONSOLE,
        {**python_path(), "QT_QPA_PLATFORM": "offscreen"},
    )


def check() -> None:
    audit()
    lint()
    test()


def db() -> None:
    """Report the database's migration state.

    Read-only, and deliberately so: applying a migration is something an
    operator does knowingly, so `db-migrate` is a separate word.
    """
    import sys as _sys

    _sys.path.insert(0, str(ENGINE))
    from sentinel.store import MIGRATIONS, Store, default_database_path

    path = default_database_path()
    existed = path.exists()

    print()
    print(f"database   {path}")
    print(f"existed    {existed}")

    # Opened without auto-migrating, so `db` reports what is actually there
    # rather than what it would be after silently fixing it.
    with Store(path, auto_migrate=False) as store:
        applied = store.applied_versions()
        pending = store.pending()
        print(f"applied    {applied or 'none'}")
        print(f"pending    {[m.version for m in pending] or 'none'}")
        print(f"known      {[m.version for m in MIGRATIONS]}")

        # Counted only when the tables exist. `db` is most useful on a database
        # that is *not* migrated, so it must not fall over on one.
        if pending:
            print("events     — schema not applied")
            print("incidents  — schema not applied")
        else:
            print(f"events     {store.event_count()}")
            print(f"incidents  {store.incident_count()}")


def db_migrate() -> None:
    import sys as _sys

    _sys.path.insert(0, str(ENGINE))
    from sentinel.store import Store, default_database_path

    with Store(default_database_path(), auto_migrate=False) as store:
        applied = store.migrate()
    if applied:
        for migration in applied:
            print(f"applied {migration.version}: {migration.name}")
    else:
        print("nothing to apply")


def db_rollback() -> None:
    import sys as _sys

    _sys.path.insert(0, str(ENGINE))
    from sentinel.store import Store, default_database_path

    with Store(default_database_path(), auto_migrate=False) as store:
        undone = store.rollback()
    print(f"undid {undone.version}: {undone.name}" if undone else "nothing to undo")


def console() -> None:
    build()
    run([sys.executable, str(CONSOLE / "main.py")], ROOT, python_path())


def cli() -> None:
    """The headless analyser, with everything after `cli` passed through."""
    build()
    run([sys.executable, "-m", "sentinel", *sys.argv[2:]], ROOT, python_path())


#: What the bundle contains. Named once, because the summary, the cleanup and
#: the notes written into the folder must not be able to disagree.
EXECUTABLES = ("SentinelVision", "SentinelVision-dev", "sentinel")

RUN_NOTES = """Sentinel Vision
===============

Run one of these, from THIS folder:

  SentinelVision{suffix}       the operator console
  SentinelVision-dev{suffix}   the same console with a terminal and verbose
                          logging. Run this one if the console will not start:
                          a packaged Qt application has nowhere to print, so
                          without a terminal an error at start-up is invisible.
  sentinel{suffix}             the headless analyser. Try `sentinel{suffix} where`.

Keep the whole folder together. The executables need `_internal` beside them,
which is what every Qt application ships. Moving an executable on its own gives:

    Failed to load Python DLL '...\\_internal\\python3xx.dll'

Nothing here is installed, written to the registry, or downloaded. Copy the
folder, run it, delete the folder.

Your data — the database, the log and any exported evidence — is NOT in here.
`sentinel{suffix} where` prints exactly where it is.

Documentation: docs/USAGE.md in the source repository.
"""


def executable_names() -> tuple[str, ...]:
    suffix = ".exe" if sys.platform == "win32" else ""
    return tuple(f"{name}{suffix}" for name in EXECUTABLES)


def strip_work_executables(work: Path) -> list[Path]:
    """Delete the runnable-looking stubs PyInstaller leaves in its scratch dir.

    This exists because of a real hour lost to it. PyInstaller writes each
    executable into the *work* directory first and then copies it into `dist`
    alongside the libraries. What is left behind in `build/` is a bootloader
    with no `_internal` next to it: it has the right name, the right icon and
    the right size, it is the first thing you find if you go looking for a file
    called `sentinel.exe`, and running it produces

        Failed to load Python DLL '...\\build\\sentinel\\_internal\\python314.dll'

    which reads like a broken build rather than the wrong file.

    Only the executables go. The `.toc`, `.pkg` and `.pyz` files beside them are
    the incremental-rebuild cache and deleting those would make every rebuild a
    full one.
    """
    removed: list[Path] = []
    if not work.is_dir():
        return removed

    for name in executable_names():
        for stub in work.rglob(name):
            if stub.is_file():
                stub.unlink()
                removed.append(stub)
    return removed


def package() -> None:
    """Build the standalone executables.

    The core is built first and unconditionally. Packaging a stale library is
    how a green test run ships broken geometry: the tests loaded one core and
    the bundle carries another.

    PyInstaller is a build-time dependency and is not installed by default,
    because a machine that only runs the tests should not have to carry it.
    """
    build()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed. It is a build tool, not a runtime "
            "dependency:\n\n"
            "    python -m pip install pyinstaller\n\n"
            "Nothing it produces reaches the network; the bundle is assembled "
            "from what is already on this machine."
        ) from None

    spec = ROOT / "packaging" / "sentinel.spec"
    work = ROOT / "build"
    run(
        [
            sys.executable, "-m", "PyInstaller", str(spec),
            "--noconfirm",
            "--distpath", str(ROOT / "dist"),
            "--workpath", str(work),
        ],
        ROOT,
        python_path(),
    )

    produced = ROOT / "dist" / "SentinelVision"
    missing = [name for name in executable_names() if not (produced / name).is_file()]
    if missing:
        raise SystemExit(
            f"The build finished but {', '.join(missing)} is not in {produced}. "
            "Nothing here is shippable; look at the PyInstaller output above."
        )

    # Only after the dist copies are confirmed present: the stubs are the
    # fallback if something went wrong, right up until there is a real bundle.
    stripped = strip_work_executables(work)

    (produced / "HOW TO RUN.txt").write_text(
        RUN_NOTES.format(suffix=".exe" if sys.platform == "win32" else ""),
        encoding="utf-8",
    )

    print()
    print("  Run it from here, and nowhere else:")
    print()
    print(f"      {produced}")
    print()
    for name in executable_names():
        print(f"        {name}")
    print()
    print("  Ship the whole folder. The executables need `_internal` beside")
    print("  them, which is what every Qt application ships.")
    if stripped:
        print()
        print(f"  ({len(stripped)} unrunnable stub(s) removed from {work.name}/ —")
        print("   PyInstaller leaves copies there with no libraries next to them,")
        print("   and running one reports a missing Python DLL.)")


TASKS = {
    "build": build,
    "lint": lint,
    "audit": audit,
    "test": test,
    "check": check,
    "console": console,
    "cli": cli,
    "package": package,
    "db": db,
    "db-migrate": db_migrate,
    "db-rollback": db_rollback,
}


#: Tasks that take arguments of their own, passed through untouched.
PASSTHROUGH = {"cli"}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in TASKS:
        print(__doc__)
        return 1 if len(argv) > 1 else 0
    if len(argv) > 2 and argv[1] not in PASSTHROUGH:
        print(f"{argv[1]} takes no arguments")
        return 1

    TASKS[argv[1]]()
    print("\n\033[32mok\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
