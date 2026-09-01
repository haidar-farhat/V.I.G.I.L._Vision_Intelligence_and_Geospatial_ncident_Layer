#!/usr/bin/env python3
"""Task runner.

    python tasks.py build      build the Rust engine core
    python tasks.py test       everything: Rust, engine, console
    python tasks.py lint       rustfmt and clippy
    python tasks.py audit      prove the source contains no route off the site
    python tasks.py console    run the operator console
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
    """The zero-WAN guarantee, checked statically.

    The offline CI job proves the *tests* need no network. This proves the
    *source* has nowhere to go, which is the stronger claim and the one an
    operator is actually relying on: an untested code path can still call home.
    """
    run([sys.executable, str(ROOT / "tools" / "offline_audit.py")], ROOT)


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


TASKS = {
    "build": build,
    "lint": lint,
    "audit": audit,
    "test": test,
    "check": check,
    "console": console,
    "db": db,
    "db-migrate": db_migrate,
    "db-rollback": db_rollback,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in TASKS:
        print(__doc__)
        return 1 if len(argv) > 1 else 0

    TASKS[argv[1]]()
    print("\n\033[32mok\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
