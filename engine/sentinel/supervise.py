"""Keeping the node running when nobody is watching it.

A security system that stops at 03:00 and stays stopped until somebody notices
is not a security system. Until this module existed nothing restarted the
node: a crash — a decoder wedged, a model file gone, a bug — ended monitoring
silently, and the audit's first blocker (REL-01) said so.

Three things live here, all deliberately small and all offline:

**`supervise`** runs the node as a child process and starts it again when it
dies, with a bounded, growing pause between attempts so a machine with a
broken camera does not spin. It stops when the node exits cleanly — a `--for`
that ran out, or a stop that was asked for — because a clean exit is a
decision, not a failure.

**The stop file** is how a node is asked to stop on every platform. On Windows
an external Ctrl-C does not reach a Python process at all (measured), a
service has no terminal, and a signal is not a thing a scheduled task can
send. A file in the data directory is: `sentinel node --stop` writes it, the
running node sees it on its next poll, removes it, and exits 0; the supervisor
sees the same file and does not restart.

**`service`** registers the supervisor with the operating system's own
mechanism for starting things at logon or boot — a scheduled task on Windows,
a user unit on systemd, a launch agent on macOS — by writing the definition
and running the platform's command. Nothing is downloaded; nothing is
installed beyond one task, one unit file or one plist, and `uninstall` removes
exactly that.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from . import paths
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Seconds between restart attempts, growing and capped. A camera that refuses
#: to open should not be asked sixty times a minute.
BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

#: The file whose presence asks a running node to stop.
STOP_FILE = "node.stop"

#: The name every platform's registration uses, so `uninstall` finds it.
SERVICE_NAME = "SentinelVision"


def stop_file() -> Path:
    """Where a stop request is written, in this build's data directory."""
    return paths.data_directory() / STOP_FILE


def request_stop() -> Path:
    """Ask the node running against this data directory to stop."""
    path = stop_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stop\n", encoding="utf-8")
    return path


def clear_stop() -> bool:
    """Remove a stop request. Returns whether there was one."""
    path = stop_file()
    if path.exists():
        path.unlink()
        return True
    return False


def child_command(node_arguments: Sequence[str]) -> list[str]:
    """The command that runs the node, from a checkout or from the bundle.

    Frozen, `sys.executable` *is* `sentinel.exe`; from a checkout it is the
    interpreter and the module is named. Either way the node's own arguments
    follow unchanged.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "node", *node_arguments]
    return [sys.executable, "-m", "sentinel", "node", *node_arguments]


def supervise(
    command: Sequence[str],
    *,
    max_restarts: int | None = None,
    stop: Path | None = None,
    run: Callable[[Sequence[str]], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run ``command`` and restart it when it fails.

    Returns 0 when the child exited cleanly or a stop was requested, and the
    child's last exit code when ``max_restarts`` ran out. ``run`` and ``sleep``
    are the seams a test injects through; the defaults run a real process and
    really wait.
    """
    stop = stop if stop is not None else stop_file()
    run = run if run is not None else _run_child
    restarts = 0
    while True:
        _log.info("supervisor: starting %s", " ".join(str(part) for part in command))
        code = run(command)
        if code == 0:
            _log.info("supervisor: the node exited cleanly; not restarting")
            return 0
        if stop.exists():
            _log.info("supervisor: a stop was requested; not restarting")
            return 0
        if max_restarts is not None and restarts >= max_restarts:
            _log.error(
                "supervisor: the node exited %d and the restart limit (%d) is spent",
                code, max_restarts,
            )
            return int(code)
        pause = BACKOFF_SECONDS[min(restarts, len(BACKOFF_SECONDS) - 1)]
        restarts += 1
        _log.warning(
            "supervisor: the node exited %d; restart %d in %.0fs", code, restarts, pause
        )
        sleep(pause)


def _run_child(command: Sequence[str]) -> int:
    try:
        return subprocess.run(list(command), check=False).returncode
    except KeyboardInterrupt:
        # The person at the terminal, if there is one, means the whole thing.
        return 0
    except OSError as error:
        _log.error("supervisor: could not start the node: %s", type(error).__name__)
        return 1


# ---------------------------------------------------------------- service


def supervisor_command(node_arguments: Sequence[str]) -> list[str]:
    """What the service runs: this executable, supervising the node."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "supervise", "--", *node_arguments]
    return [sys.executable, "-m", "sentinel", "supervise", "--", *node_arguments]


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part or not part else part


def service_definition(platform: str, node_arguments: Sequence[str]) -> dict:
    """The registration for one platform: what is written, where, and the
    commands that install and remove it. Pure — nothing is run here — so a
    test can read every platform's answer on one machine.
    """
    command = supervisor_command(node_arguments)
    line = " ".join(_quote(str(part)) for part in command)
    if platform == "win32":
        return {
            "kind": "scheduled task",
            "path": None,
            "content": None,
            "install": [
                "schtasks", "/Create", "/TN", SERVICE_NAME, "/SC", "ONLOGON",
                "/TR", line, "/RL", "LIMITED", "/F",
            ],
            "uninstall": ["schtasks", "/Delete", "/TN", SERVICE_NAME, "/F"],
            "note": (
                "Runs at logon of the account that installed it, as that "
                "account. To start before anyone logs on, change the trigger "
                "to ONSTART in Task Scheduler and give it a service account."
            ),
        }
    if platform == "darwin":
        path = Path.home() / "Library" / "LaunchAgents" / f"com.sentinelvision.{SERVICE_NAME.lower()}.plist"
        arguments = "\n".join(f"        <string>{part}</string>" for part in command)
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<plist version="1.0"><dict>\n'
            f"    <key>Label</key><string>com.sentinelvision.{SERVICE_NAME.lower()}</string>\n"
            "    <key>ProgramArguments</key><array>\n"
            f"{arguments}\n"
            "    </array>\n"
            "    <key>RunAtLoad</key><true/>\n"
            "    <key>KeepAlive</key><false/>\n"
            "</dict></plist>\n"
        )
        return {
            "kind": "launch agent",
            "path": path,
            "content": content,
            "install": ["launchctl", "load", "-w", str(path)],
            "uninstall": ["launchctl", "unload", "-w", str(path)],
            "note": "A launch agent runs when its user logs in. KeepAlive is off: the supervisor does the restarting.",
        }
    path = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME.lower()}.service"
    content = (
        "[Unit]\n"
        "Description=Sentinel Vision node, supervised\n"
        "After=network.target\n\n"
        "[Service]\n"
        f"ExecStart={line}\n"
        "Restart=no\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    return {
        "kind": "systemd user unit",
        "path": path,
        "content": content,
        "install": ["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME.lower()}.service"],
        "uninstall": ["systemctl", "--user", "disable", "--now", f"{SERVICE_NAME.lower()}.service"],
        "note": (
            "A user unit runs while that user has a session; `loginctl "
            "enable-linger` keeps it running without one. Restart=no because "
            "the supervisor does the restarting."
        ),
    }


def install_service(
    node_arguments: Sequence[str],
    *,
    platform: str = sys.platform,
    run: Callable[[Sequence[str]], int] | None = None,
) -> tuple[int, dict]:
    """Write the definition (where one is a file) and run the install command."""
    definition = service_definition(platform, node_arguments)
    if definition["path"] is not None:
        definition["path"].parent.mkdir(parents=True, exist_ok=True)
        definition["path"].write_text(definition["content"], encoding="utf-8")
    runner = run if run is not None else _run_quietly
    code = runner(definition["install"])
    return code, definition


def uninstall_service(
    *, platform: str = sys.platform, run: Callable[[Sequence[str]], int] | None = None
) -> tuple[int, dict]:
    definition = service_definition(platform, [])
    runner = run if run is not None else _run_quietly
    code = runner(definition["uninstall"])
    if definition["path"] is not None and definition["path"].exists():
        definition["path"].unlink()
    return code, definition


def _run_quietly(command: Sequence[str]) -> int:
    try:
        return subprocess.run(list(command), check=False).returncode
    except OSError as error:
        _log.error("could not run %s: %s", command[0], type(error).__name__)
        return 1
