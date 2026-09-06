"""Keeping the analysis running when nobody is watching it.

A security system that stops at 03:00 and stays stopped until somebody
notices is not a security system. Three small, offline things live here.

**`supervise`** runs the product as a child process and starts it again when
it dies, with a bounded, growing pause so a machine with a broken camera does
not spin. A clean exit is a decision, not a failure, and is not restarted.

**The stop file** is how a run is asked to stop on every platform. On Windows
an external Ctrl-C does not reach a Python process, a scheduled task has no
terminal, and a signal is not something a task can send. A file in the data
directory is: `vigil run --stop` writes it, the running process sees it on
its next poll, removes it and exits 0, and the supervisor sees the same file
and does not restart.

**`service`** registers the supervisor with the operating system's own
mechanism for starting things at logon — a scheduled task, a launch agent, a
systemd user unit. Nothing is downloaded; `uninstall` removes exactly what
`install` wrote.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Seconds between restart attempts, growing and capped. A camera that refuses
#: to open should not be asked sixty times a minute.
BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

#: The file whose presence asks a running process to stop.
STOP_FILE = "vigil.stop"

#: The name every platform's registration uses, so `uninstall` finds it.
SERVICE_NAME = "SentinelVision"


def stop_file(data_dir: Path) -> Path:
    return Path(data_dir) / STOP_FILE


def request_stop(data_dir: Path) -> Path:
    path = stop_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"stop requested at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")
    return path


def clear_stop(data_dir: Path) -> bool:
    """Remove the request. True when one was there — the running process's answer."""
    path = stop_file(data_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def supervise(command: Sequence[str], *, data_dir: Path, max_restarts: int | None = None,
              run: Callable[[Sequence[str]], int] | None = None,
              sleep: Callable[[float], None] = time.sleep) -> int:
    """Run ``command`` and start it again when it fails.

    0 when the child exited cleanly or a stop was asked for; the child's last
    exit code when ``max_restarts`` runs out. ``run`` and ``sleep`` are the
    seams a test injects through.
    """
    runner = run if run is not None else _run_child
    stop = stop_file(data_dir)
    restarts = 0
    while True:
        _log.info("supervisor: starting %s", " ".join(str(p) for p in command))
        code = runner(command)
        if code == 0:
            _log.info("supervisor: the child exited cleanly; not restarting")
            return 0
        if stop.exists():
            _log.info("supervisor: a stop was requested; not restarting")
            return 0
        if max_restarts is not None and restarts >= max_restarts:
            _log.error("supervisor: the child exited %d and the restart limit (%d) is spent", code, max_restarts)
            return int(code)
        pause = BACKOFF_SECONDS[min(restarts, len(BACKOFF_SECONDS) - 1)]
        restarts += 1
        _log.warning("supervisor: the child exited %d; restart %d in %.0f s", code, restarts, pause)
        sleep(pause)


def _run_child(command: Sequence[str]) -> int:
    try:
        return subprocess.run(list(command), check=False).returncode
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        _log.error("supervisor: could not start the child: %s", type(error).__name__)
        return 1


# ------------------------------------------------------------------ service


def child_command(arguments: Sequence[str]) -> list[str]:
    """The product itself, with the arguments a service was asked to run."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "vigil", *arguments]


def supervisor_command(arguments: Sequence[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "supervise", "--", *arguments]
    return [sys.executable, "-m", "vigil", "supervise", "--", *arguments]


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part or not part else part


def service_definition(platform: str, arguments: Sequence[str]) -> dict:
    """One platform's registration: what is written, where, and the commands.

    Pure — nothing is run here — so a test reads every platform's answer on
    one machine.
    """
    command = supervisor_command(arguments)
    line = " ".join(_quote(str(p)) for p in command)
    if platform == "win32":
        return {
            "kind": "scheduled task", "path": None, "content": None,
            "install": ["schtasks", "/Create", "/TN", SERVICE_NAME, "/SC", "ONLOGON", "/TR", line, "/RL", "LIMITED", "/F"],
            "uninstall": ["schtasks", "/Delete", "/TN", SERVICE_NAME, "/F"],
            "note": ("Runs at logon of the account that installed it, as that account. To start before "
                     "anyone logs on, change the trigger to ONSTART in Task Scheduler and give it a service account."),
        }
    if platform == "darwin":
        path = Path.home() / "Library" / "LaunchAgents" / f"com.sentinelvision.{SERVICE_NAME.lower()}.plist"
        arguments_xml = "\n".join(f"        <string>{p}</string>" for p in command)
        content = ('<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>\n'
                   f"    <key>Label</key><string>com.sentinelvision.{SERVICE_NAME.lower()}</string>\n"
                   f"    <key>ProgramArguments</key><array>\n{arguments_xml}\n    </array>\n"
                   "    <key>RunAtLoad</key><true/>\n    <key>KeepAlive</key><false/>\n</dict></plist>\n")
        return {"kind": "launch agent", "path": path, "content": content,
                "install": ["launchctl", "load", "-w", str(path)],
                "uninstall": ["launchctl", "unload", "-w", str(path)],
                "note": "A launch agent runs when its user logs in. KeepAlive is off: the supervisor does the restarting."}
    path = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME.lower()}.service"
    content = ("[Unit]\nDescription=Sentinel Vision, supervised\nAfter=network.target\n\n"
               f"[Service]\nExecStart={line}\nRestart=no\n\n[Install]\nWantedBy=default.target\n")
    return {"kind": "systemd user unit", "path": path, "content": content,
            "install": ["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME.lower()}.service"],
            "uninstall": ["systemctl", "--user", "disable", "--now", f"{SERVICE_NAME.lower()}.service"],
            "note": ("A user unit runs while that user has a session; `loginctl enable-linger` keeps it "
                     "running without one. Restart=no because the supervisor does the restarting.")}


def install_service(arguments: Sequence[str], *, platform: str = sys.platform,
                    run: Callable[[Sequence[str]], int] | None = None) -> tuple[int, dict]:
    definition = service_definition(platform, arguments)
    if definition["path"] is not None:
        definition["path"].parent.mkdir(parents=True, exist_ok=True)
        definition["path"].write_text(definition["content"], encoding="utf-8")
    code = (run or _run_quietly)(definition["install"])
    return code, definition


def uninstall_service(*, platform: str = sys.platform, run: Callable[[Sequence[str]], int] | None = None) -> tuple[int, dict]:
    definition = service_definition(platform, [])
    code = (run or _run_quietly)(definition["uninstall"])
    if definition["path"] is not None and definition["path"].exists():
        definition["path"].unlink()
    return code, definition


def _run_quietly(command: Sequence[str]) -> int:
    try:
        return subprocess.run(list(command), check=False).returncode
    except OSError as error:
        _log.error("could not run %s: %s", command[0], type(error).__name__)
        return 1
