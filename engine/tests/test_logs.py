"""Tests for logging.

Before this module existed the system produced no output at all, so these tests
are not protecting behaviour that used to work — they are protecting a new
surface that has the single most dangerous property in this codebase: it takes
arbitrary text from anywhere and writes it to a file.

**A log line must never carry a camera password.** Every test below that mentions
a URL is about that. The primary defence is that messages are built from
`display_url` — the filter here is the backstop, for the case that actually
happens: a library raises an exception containing a URL nobody sanitised, and
somebody logs it with `exc_info=True`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sentinel import logs
from sentinel.decode import contains_credential
from sentinel.redact import redact_text

SECRET = "hunter2-not-a-real-password"
CAMERA_URL = f"rtsp://admin:{SECRET}@10.20.30.40:554/Streaming/Channels/101"


@pytest.fixture(autouse=True)
def clean_logging():
    """No test may leave handlers attached to a process-wide logger."""
    logs.reset()
    yield
    logs.reset()


def records_from(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# ----------------------------------------------------------------- redaction


def test_a_password_in_a_message_is_removed():
    filtered = logs.RedactingFilter()
    record = logging.LogRecord(
        "sentinel.test", logging.INFO, __file__, 1,
        f"opening {CAMERA_URL}", None, None,
    )

    filtered.filter(record)

    assert not contains_credential(record.getMessage(), CAMERA_URL)
    assert SECRET not in record.getMessage()
    # The host must survive: a redacted line nobody can act on is not useful.
    assert "10.20.30.40" in record.getMessage()


def test_a_password_in_an_interpolation_argument_is_removed():
    # The common shape. `_log.info("opening %s", url)` with the wrong url.
    filtered = logs.RedactingFilter()
    record = logging.LogRecord(
        "sentinel.test", logging.INFO, __file__, 1, "opening %s", (CAMERA_URL,), None,
    )

    filtered.filter(record)

    assert not contains_credential(record.getMessage(), CAMERA_URL)


def test_a_password_in_a_dict_argument_is_removed():
    filtered = logs.RedactingFilter()
    # Wrapped in a tuple: LogRecord unwraps a single mapping itself, and
    # passing the bare dict makes it subscript with 0.
    record = logging.LogRecord(
        "sentinel.test", logging.INFO, __file__, 1,
        "opening %(url)s", ({"url": CAMERA_URL},), None,
    )

    filtered.filter(record)

    assert not contains_credential(record.getMessage(), CAMERA_URL)


def test_a_password_in_a_traceback_is_removed(tmp_path: Path):
    # This is the case the filter exists for. A library raises an exception
    # built from the URL it was handed, and somebody logs it with exc_info.
    target = tmp_path / "sentinel.log"
    logs.configure(level="DEBUG", console=False, file=target)

    try:
        raise ValueError(f"could not connect to {CAMERA_URL}")
    except ValueError:
        logs.get("test").error("camera failed", exc_info=True)

    written = target.read_text(encoding="utf-8")

    assert "ValueError" in written, "the traceback should still be there"
    assert not contains_credential(written, CAMERA_URL)
    assert SECRET not in written


def test_a_credential_in_a_query_string_is_removed():
    url = "http://cam.local/stream?user=admin&password=abc123xyz&x=1"
    filtered = logs.RedactingFilter()
    record = logging.LogRecord(
        "sentinel.test", logging.INFO, __file__, 1, "opening %s", (url,), None,
    )

    filtered.filter(record)

    assert not contains_credential(record.getMessage(), url)
    assert "x=1" in record.getMessage(), "the harmless parameters should survive"


def test_several_urls_in_one_line_are_all_redacted():
    other = "rtsp://user:secondsecret@10.0.0.9/s"
    line = f"failing over from {CAMERA_URL} to {other}"

    redacted = redact_text(line)

    assert not contains_credential(redacted, CAMERA_URL)
    assert not contains_credential(redacted, other)


def test_ordinary_text_is_left_alone():
    # A redactor that mangles normal messages is a redactor somebody removes.
    for line in (
        "gate: analysis finished — 180 frames, 4 objects, 6 events",
        "zone presence started for track 3 in Restricted Area A",
        "C:\\Users\\operator\\footage\\gate.mp4",
        "no colons or slashes at all",
    ):
        assert redact_text(line) == line, line


# ------------------------------------------------------------------ handlers


def test_the_file_records_at_info_even_when_the_console_is_quiet(tmp_path: Path):
    # `--quiet` is a statement about the terminal, not about the record. An
    # operator who silences the console and then has an outage still needs the
    # log to say what happened.
    target = tmp_path / "sentinel.log"
    logs.configure(level="WARNING", console=False, file=target)

    logs.get("test").info("a camera reconnected")

    assert "a camera reconnected" in target.read_text(encoding="utf-8")


def test_the_file_can_be_switched_off_for_a_container(tmp_path: Path):
    # A container's log is its stdout. A second copy inside a layer that is
    # discarded when it stops is worse than useless.
    logs.configure(level="INFO", console=False, file="")

    logger = logging.getLogger(logs.ROOT)

    assert not any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    )


def test_configuring_twice_does_not_duplicate_handlers(tmp_path: Path):
    # The console imports the engine; both would otherwise configure, and every
    # line would appear twice.
    logs.configure(level="INFO", console=True, file=tmp_path / "a.log")
    before = len(logging.getLogger(logs.ROOT).handlers)

    logs.configure(level="DEBUG", console=True, file=tmp_path / "b.log")

    assert len(logging.getLogger(logs.ROOT).handlers) == before


def test_a_log_file_that_cannot_be_opened_does_not_stop_the_application(tmp_path: Path):
    # A read-only install directory or a full disk must not be fatal. It is a
    # security system: it starting matters more than it logging.
    blocked = tmp_path / "a-file-not-a-directory"
    blocked.write_text("", encoding="utf-8")

    logs.configure(level="INFO", console=False, file=blocked / "nested" / "sentinel.log")

    assert logging.getLogger(logs.ROOT).handlers, "logging was left unusable"


def test_an_unrecognised_level_falls_back_to_info_rather_than_silence():
    # Too much output is a nuisance. None is a blind spot.
    assert logs._resolve_level("not-a-level") == logging.INFO
    assert logs._resolve_level("debug") == logging.DEBUG
    assert logs._resolve_level(None) == logging.INFO


def test_the_level_can_be_raised_in_the_field_without_a_rebuild(monkeypatch):
    # A packaged operator build has no command line to change.
    monkeypatch.setenv(logs.LEVEL_VARIABLE, "DEBUG")

    assert logs._resolve_level(None) == logging.DEBUG


def test_the_tree_does_not_leak_into_the_root_logger(tmp_path: Path):
    # An embedding application owns the root logger, and this system flooding it
    # would be this system's fault.
    logs.configure(level="INFO", console=False, file=tmp_path / "sentinel.log")

    assert logging.getLogger(logs.ROOT).propagate is False


def test_a_module_outside_the_package_still_lands_under_sentinel():
    assert logs.get("sentinel_console.app").name == "sentinel.sentinel_console.app"
    assert logs.get("sentinel.decode").name == "sentinel.decode"


# ----------------------------------------------------------- nothing networked


#: `logging.handlers` ships HTTP, syslog and datagram handlers, and any of them
#: turns a log line into an outbound packet. `logging.config` is worse: its file
#: and dict loaders can be told to import arbitrary modules, which is not a
#: capability a log configuration needs.
FORBIDDEN = {
    "HTTPHandler", "SysLogHandler", "DatagramHandler", "SocketHandler",
    "QueueHandler", "fileConfig", "dictConfig", "listen",
}


def test_no_handler_can_reach_the_network():
    # §132: no hidden telemetry. Asserted against the module's parsed syntax
    # rather than its text, so the docstring can name the things it refuses
    # without the test firing on its own explanation.
    import ast

    tree = ast.parse(Path(logs.__file__).read_text(encoding="utf-8"))

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Import):
            used.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module)

    leaked = used & FORBIDDEN

    assert not leaked, f"logging must not be able to reach {sorted(leaked)}"
    assert "logging.config" not in used
