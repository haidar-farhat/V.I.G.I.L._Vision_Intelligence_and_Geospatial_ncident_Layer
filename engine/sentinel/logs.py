"""Logging.

Until this module existed the system produced **no output at all** — no log, no
diagnostic, nothing. That is untenable for something meant to run unattended on
somebody else's site: when a camera stops at 03:00 the only evidence that it
happened is what was written down at the time.

Three rules shape everything here.

**A log line must never carry a credential.** Every record passes through
:class:`RedactingFilter` before it is formatted. That is a *backstop*, not the
defence — messages are built from `display_url`, never from the raw one — but a
backstop is exactly what is wanted for a rule that must hold even when a library
raises an exception containing a URL nobody sanitised.

**A log must not become the thing that fills the disk.** Files rotate at 5 MB
and keep five, so the ceiling is 30 MB per stream and is reached rather than
approached.

**The developer and the operator want different things.** An operator wants a
file, at INFO, that says what happened. A developer wants a console, at DEBUG,
with the module and line that said it. `configure()` serves both, and the
packaged build ships as two executables for exactly this reason.

Nothing here writes to the network. There is no syslog handler, no HTTP handler,
and no `logging.config` file loading — the last of which can be told to import
arbitrary modules, which is not a capability a log configuration needs.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .paths import log_directory
from .redact import redact_text

#: The root of this system's logger tree. Every module logs under it, so a
#: caller embedding the engine can silence or redirect all of it in one call
#: without touching the root logger it does not own.
ROOT = "sentinel"

#: 5 MB × 5 files = 30 MB per stream, reached rather than approached.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

#: Reads the level when no argument is given, so a packaged operator build can
#: be turned up in the field without a rebuild.
LEVEL_VARIABLE = "SENTINEL_LOG_LEVEL"

#: Set to a path to put the log somewhere specific; set to "" to disable the
#: file handler entirely, which is what a container wants — its log is stdout.
FILE_VARIABLE = "SENTINEL_LOG_FILE"

#: The egress guard's one override (see `decode.py`). Read here only to say,
#: at start-up, that it is set.
PUBLIC_SOURCES_VARIABLE = "SENTINEL_ALLOW_PUBLIC_SOURCES"

#: Where a hard crash leaves its trace. A segfault, an abort, a heap corruption
#: — the 0xC0000374 that once ended a green test run with nothing but an exit
#: code — never reaches the Python log, because the interpreter is not there
#: to write it. `faulthandler` writes every thread's Python stack to this file
#: at the moment of the crash, from C, with no allocation.
CRASH_FILE = "crash.log"
_crash_handle = None
_faulthandler_was_enabled = False

OPERATOR_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s"
DEVELOPER_FORMAT = (
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-26s  "
    "%(threadName)-14s  %(filename)s:%(lineno)d  %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class RedactingFilter(logging.Filter):
    """Strip credentials from every record, whatever built it.

    Applied to the *record*, not to the formatter, so it covers the message, the
    interpolation arguments and the exception text — all three of which have
    leaked a password in some system somewhere.

    It runs on every record, so it is written to be cheap on the common case:
    :func:`redact_text` returns immediately when there is no ``://`` in the
    string, which is true of almost every line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact_text(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    redact_text(value) if isinstance(value, str) else value
                    for value in record.args
                )

        # `exc_text` is the cached, already-formatted traceback. Clearing it is
        # not enough — it would simply be rebuilt from the exception — so it is
        # redacted in place if a formatter has already produced it.
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)

        return True


class _RedactingFormatter(logging.Formatter):
    """Redacts the finished line as well.

    Belt and braces: the filter covers the parts, this covers the whole, and the
    whole is what reaches the disk. A traceback formatted during `format` has
    not passed the filter.
    """

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        level = os.environ.get(LEVEL_VARIABLE, "INFO")
    if isinstance(level, str):
        resolved = logging.getLevelNamesMapping().get(level.strip().upper())
        # An unrecognised level must not silence the log. INFO is the safe
        # answer: too much output is a nuisance, none is a blind spot.
        return resolved if resolved is not None else logging.INFO
    return level


def configure(
    *,
    level: int | str | None = None,
    console: bool = True,
    developer: bool = False,
    file: Path | str | None = None,
) -> logging.Logger:
    """Set up logging once, and return this system's root logger.

    ``developer`` switches the console format to one with thread, file and line,
    and is what the ``-dev`` executable and ``--verbose`` turn on.

    ``file`` defaults to a rotating file under the data directory. Pass ``""``
    to suppress it — a container's log is its stdout, and a second copy inside a
    layer that is thrown away when it stops is worse than useless.

    Calling this more than once is a no-op rather than a duplicated handler,
    because the console imports the engine and both would otherwise configure.
    """
    global _configured

    logger = logging.getLogger(ROOT)
    if _configured:
        return logger

    resolved = _resolve_level(level)

    # The file is never quieter than INFO, whatever the console is set to.
    # `--quiet` is a statement about the terminal, not about the record: an
    # operator who silences the console and then has an outage still needs the
    # log to say what happened. The logger sits at the lower of the two, because
    # a record the logger rejects never reaches any handler.
    recorded = min(resolved, logging.INFO)
    logger.setLevel(recorded)
    # This tree is configured here and nowhere else, so it must not also reach
    # whatever the embedding application did to the root logger.
    logger.propagate = False

    redactor = RedactingFilter()
    fmt = DEVELOPER_FORMAT if developer else OPERATOR_FORMAT

    if console:
        # A character the terminal's code page cannot represent must not become
        # an exception inside a log handler. Windows pipes a redirected stderr
        # through the locale code page, so a degree sign or an em dash in a
        # message would raise UnicodeEncodeError and logging would swallow the
        # line entirely — losing the record to make the terminal look tidy.
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass

        # stderr, not stdout: the CLI writes its report to stdout, and a log
        # interleaved with it would make the report unusable in a pipe.
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(resolved)
        stream.setFormatter(_RedactingFormatter(fmt, DATE_FORMAT))
        stream.addFilter(redactor)
        logger.addHandler(stream)

    target = _log_file(file)
    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                target, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            rotating.setLevel(recorded)
            rotating.setFormatter(_RedactingFormatter(DEVELOPER_FORMAT, DATE_FORMAT))
            rotating.addFilter(redactor)
            logger.addHandler(rotating)
        except OSError as error:
            # A read-only or full disk must not stop the application starting.
            # It is reported on the console handler that is already attached.
            logger.warning("could not open the log file at %s: %s", target, error)

    if not logger.handlers:
        # Everything was disabled. Without this, every call site pays the cost
        # of "no handlers could be found" on the first record.
        logger.addHandler(logging.NullHandler())

    _configured = True
    logger.info(
        "logging configured: console %s, file %s at %s",
        logging.getLevelName(resolved),
        target or "disabled",
        logging.getLevelName(recorded),
    )
    # The build, on the first line of every log. A log that cannot say which
    # build wrote it is a log that has to be matched to one by its dates.
    from .version import describe

    logger.info("%s", describe())
    if os.environ.get(PUBLIC_SOURCES_VARIABLE, "").strip().lower() not in ("", "0", "false", "no"):
        # The one override of the egress guard, said out loud once per process
        # so that a machine reaching routable addresses never does so quietly.
        logger.warning(
            "%s is set: the egress guard will let camera addresses outside the "
            "local network through. Every connection it allows is logged.",
            PUBLIC_SOURCES_VARIABLE,
        )
    if target is not None:
        _enable_crash_log(target.parent)
    return logger


def _enable_crash_log(directory: Path) -> Path | None:
    """Point `faulthandler` at ``crash.log`` beside the log. Never fails."""
    global _crash_handle, _faulthandler_was_enabled

    import faulthandler

    _faulthandler_was_enabled = faulthandler.is_enabled()
    path = directory / CRASH_FILE
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 - held for the process's life
    except OSError:
        return None
    # `enable` again replaces the file. Held open on purpose: at the moment it
    # is needed there is no later in which to open one.
    faulthandler.enable(file=handle, all_threads=True)
    if _crash_handle is not None and _crash_handle is not handle:
        try:
            _crash_handle.close()
        except OSError:
            pass
    _crash_handle = handle
    return path


def crash_log_path() -> Path | None:
    """Where the crash trace goes, or ``None`` when no file log is configured."""
    return Path(_crash_handle.name) if _crash_handle is not None else None


def _log_file(file: Path | str | None) -> Path | None:
    if file is None:
        file = os.environ.get(FILE_VARIABLE)
    if file is None:
        return log_directory() / "sentinel.log"
    if str(file) == "":
        return None
    return Path(file)


def get(name: str) -> logging.Logger:
    """A logger for one module, under this system's root.

    ``get(__name__)`` from inside the package gives the module's dotted name
    unchanged; from outside it is prefixed, so an embedding application's own
    modules still land under ``sentinel`` and are silenced with it.
    """
    if name == ROOT or name.startswith(ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT}.{name}")


def reset() -> None:
    """Undo :func:`configure`. Tests only.

    A test that configures logging must not leave handlers attached to a
    process-wide logger for every test that follows.
    """
    global _configured, _crash_handle

    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    _configured = False
    if _crash_handle is not None:
        import faulthandler

        # Back to whatever was there before — pytest's own handler on stderr,
        # usually — rather than off, so a native crash in the rest of a test
        # session still says where it happened.
        if _faulthandler_was_enabled:
            faulthandler.enable()
        else:
            faulthandler.disable()
        try:
            _crash_handle.close()
        except OSError:
            pass
        _crash_handle = None
