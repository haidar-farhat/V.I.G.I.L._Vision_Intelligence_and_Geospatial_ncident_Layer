"""Starting the console: who is here, then the window.

The sign-in happens before the window exists, so a console that nobody may
open never opens. A timed run (``--for``) is unattended by definition and so
never stops on a dialog — it is how the packaged build is tested on a real
camera.
"""

from __future__ import annotations

import argparse
import sys
import weakref
from pathlib import Path

from ... import logs
from ...config import Settings
from ...service.alerts import Alerts
from ...service.auth import Accounts, AuthError, Principal
from ...service.maintenance import StoreError, open_store
from ...service.runtime import Runtime
from ...service.site import SiteService
from ...version import describe

_log = logs.get(__name__)
_WINDOW = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vigil-console", description="Sentinel Vision — operator console.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--as", dest="as_user", default=None, metavar="NAME",
                        help="sign in as this account without the dialog; the password is read from standard input")
    parser.add_argument("--for", dest="seconds", type=float, default=None, metavar="SECONDS",
                        help="close after this long. Unattended: no dialog is shown and the analysis starts itself")
    parser.add_argument("--start", action="store_true", help="start the analysis as soon as the window opens")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--watch", default=None, help="labels to track, comma-separated")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--record", action="store_true",
                        help="record every camera for this run, whatever each camera's stored Record flag says")
    parser.add_argument("--screenshots", default=None, metavar="DIR",
                        help="with --for: photograph the window into DIR before closing")
    parser.add_argument("--verbose", action="store_true")
    # Accepted because `vigil --password-stdin --as NAME console` is a
    # reasonable thing to type; the console always reads the password from
    # standard input when `--as` is given.
    parser.add_argument("--password-stdin", action="store_true", help=argparse.SUPPRESS)
    return parser


def _sign_in(accounts: Accounts, arguments) -> Principal | None:
    """The principal, or ``None`` when the console must not open."""
    if not accounts.any():
        if arguments.seconds is not None or arguments.as_user:
            return Principal.open_site()
        from .dialogs import FirstAdminDialog, ask

        _ok, principal = ask(FirstAdminDialog(accounts))
        return principal or Principal.open_site()
    if arguments.as_user:
        secret = sys.stdin.readline().rstrip("\r\n") if sys.stdin is not None else ""
        try:
            return accounts.authenticate(arguments.as_user, secret)
        except AuthError as error:
            print(f"sign-in failed: {error}", file=sys.stderr)
            return None
    from .dialogs import SignInDialog, ask

    _ok, principal = ask(SignInDialog(accounts))
    return principal


def _report_uncaught(kind, value, traceback) -> None:
    """A packaged build has no terminal; an exception must still be visible."""
    _log.critical("uncaught exception", exc_info=(kind, value, traceback))
    window = _WINDOW() if _WINDOW is not None else None
    try:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(window, "Something failed",
                             f"{kind.__name__}: {value}\n\nIt is in the log. The console is still running.")
    except Exception:  # noqa: BLE001 - never fail inside the handler
        pass


def run(argv: list[str] | None = None) -> int:
    global _WINDOW

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    arguments, unknown = build_parser().parse_known_args(argv)

    settings = Settings.from_environment()
    if arguments.data_dir:
        settings = Settings(Path(arguments.data_dir), settings.alert_file, settings.alert_command,
                            settings.alert_webhook, settings.allow_public_sources)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    logs.configure(settings.logs, level="DEBUG" if arguments.verbose else None)
    _log.info("console starting: %s", describe())

    from PySide6.QtWidgets import QApplication

    from .commands import Commands
    from .window import ConsoleWindow

    application = QApplication([sys.argv[0], *unknown])
    application.setApplicationName("Sentinel Vision")
    application.setOrganizationName("Sentinel Vision")

    try:
        store = open_store(settings.database)
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    from ...adapters.keychain import Keychain

    site = SiteService(store, Keychain.system())
    principal = _sign_in(Accounts(store), arguments)
    if principal is None:
        _log.warning("console: sign-in cancelled; not opening")
        store.close()
        return 3

    model = Path(arguments.model) if arguments.model else (None if arguments.no_model else settings.default_model())
    try:
        factory = _detector_factory(model, arguments.watch, arguments.confidence)
    except Exception as error:  # noqa: BLE001 - a bad watch list must not open a window
        print(f"error: {error}", file=sys.stderr)
        store.close()
        return 2
    runtime = Runtime(site, detector_factory=factory, keep_images=True, realtime=True,
                      record_to=settings.recordings if arguments.record else None,
                      record_every_camera=arguments.record,
                      alerts=Alerts.from_settings(settings, store=store))
    commands = Commands(site, runtime, principal, evidence_dir=settings.evidence, model=model,
                        model_places=settings.model_directories())
    if model is None and not arguments.no_model:
        _log.warning("no model found in %s; motion detection only, which cannot classify",
                     ", ".join(str(p) for p in settings.model_directories()))

    window = ConsoleWindow(commands)
    _WINDOW = weakref.ref(window)
    sys.excepthook = _report_uncaught
    window.show()

    if arguments.start or arguments.seconds is not None:
        window._start()  # noqa: SLF001 - the same button, pressed by the flag
    if arguments.seconds is not None:
        from PySide6.QtCore import QTimer

        closer = QTimer(window)
        closer.setSingleShot(True)
        closer.setInterval(int(arguments.seconds * 1000))
        closer.timeout.connect(window.close)
        closer.start()
        if arguments.screenshots:
            # Held on the window on purpose. Connecting a bound method of a
            # temporary plain object leaves nothing owning it, and the slot
            # then never runs — silently, which is the worst way to fail.
            window._photographer = _Photographer(window, Path(arguments.screenshots))
            shots = QTimer(window)
            shots.setSingleShot(True)
            shots.setInterval(max(500, int(arguments.seconds * 1000) - 800))
            shots.timeout.connect(window._photographer.shoot)
            shots.start()

    code = application.exec()
    try:
        runtime.close(principal)
    except Exception:  # noqa: BLE001
        _log.exception("console: closing the runtime failed")
    _log.info("console exited with %d", code)
    return code


class _Photographer:
    """Photographs the window. A plain object, not a lambda closing over it."""

    def __init__(self, window, directory: Path):
        self._window = window
        self._directory = directory

    def shoot(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._window.grab().save(str(self._directory / "console.png"))
        for name, widget in (("cameras", self._window.camera_list), ("wall", self._window.wall),
                             ("plan", self._window.plan), ("incidents", self._window.incidents),
                             ("tracks", self._window.tracks)):
            widget.grab().save(str(self._directory / f"{name}.png"))
        _log.info("console: photographed into %s", self._directory)


def _detector_factory(model, watch, confidence):
    from ...adapters.detectors import WATCHED_LABELS, detector_for, model_info

    classes = frozenset(w.strip().lower() for w in watch.split(",") if w.strip()) if watch else WATCHED_LABELS
    if model is not None:
        model_info(model, classes=classes)

    class _Factory:
        def __call__(self):
            return detector_for(model, classes=classes if model is not None else None, confidence=confidence)

    return _Factory()


def main(argv: list[str] | None = None) -> int:
    return run(argv)
