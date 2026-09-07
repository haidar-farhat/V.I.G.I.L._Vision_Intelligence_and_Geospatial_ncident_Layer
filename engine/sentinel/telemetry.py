"""Switching off telemetry in dependencies, before they can emit any.

The zero-WAN promise has to hold for every dependency, not only for the code
written here — and a dependency does not ask permission. This module is the one
place that says so, and it must run **before** the libraries it disarms are
imported, because several of them read their configuration once at load time and
several emit an initialisation event before any API is reachable.

**Why this is not paranoia.** ONNX Runtime is a dependency this project already
ships. Microsoft's own privacy documentation states that telemetry is **on by
default in the official builds**, and that on Linux, macOS, Android and iOS
those builds use the cross-platform 1DS SDK with statically linked curl and
mbedTLS. The Windows wheels installed here contain no such endpoint — checked,
by scanning the shipped `.dll` and `.pyd` for the collector host, for
`OneCollector` and for the ETW provider name, and finding none — but the
platform this developer happens to use is not the platform the product runs on.

`disable_telemetry_events()` is not sufficient on its own for two reasons the
documentation is explicit about: it suppresses non-essential events only, and an
initialisation event may already have been emitted before any Python call can
reach it. The environment variable is read earlier and is the one that closes
that window, so both are used.

**What this module cannot do.** It cannot make a dependency honest, and it
cannot prove one is. It reduces a known, documented, default-on behaviour to an
off one. The controls that catch what it misses are elsewhere: the offline CI
job runs the whole suite with outbound traffic dropped, and
`tools/offline_audit.py` now scans compiled dependencies for external hosts
rather than only this project's own source — because a hostname inside a `.so`
was, until it did, completely invisible to it.
"""

from __future__ import annotations

import os

from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Read by ONNX Runtime when the native library initialises, which is earlier
#: than any Python API can be called. This is the one that closes the
#: initialisation-event window.
ORT_DISABLE = "ORT_DISABLE_TELEMETRY"

#: Set by anything that has already run. Making this idempotent matters because
#: the console, the CLI and the node each want to call it and none of them knows
#: about the others.
_SILENCED = "SENTINEL_TELEMETRY_SILENCED"

#: Variables set to disarm dependencies before they load. Each is a documented
#: opt-out of that library's own, not a guess.
ENVIRONMENT = {
    ORT_DISABLE: "1",
    # ONNX Runtime's ETW/TraceLogging path on Windows, which routes into the
    # operating system's diagnostics pipeline rather than over a socket.
    "ORT_LOGGING_LEVEL": os.environ.get("ORT_LOGGING_LEVEL", "3"),
    # OpenCV ships an opt-in usage-statistics hook; off, explicitly.
    "OPENCV_VIDEOIO_DEBUG": os.environ.get("OPENCV_VIDEOIO_DEBUG", "0"),
}


def silence() -> None:
    """Disarm dependency telemetry. Call before importing anything heavy.

    Safe to call repeatedly and from anywhere; the work happens once. An
    operator's explicit setting is never overwritten — if somebody has
    deliberately set one of these, that is a decision, and quietly reversing it
    would be exactly the kind of thing this module exists to prevent.
    """
    if os.environ.get(_SILENCED) == "1":
        return

    for name, value in ENVIRONMENT.items():
        os.environ.setdefault(name, value)

    os.environ[_SILENCED] = "1"
    _log.debug("dependency telemetry disarmed: %s", ", ".join(sorted(ENVIRONMENT)))


def silence_runtime_apis() -> None:
    """The second half, for libraries already imported.

    The environment variables above are read at load time; these calls are for
    anything that exposes a runtime switch as well. Both are used because
    neither is sufficient: the variable cannot reach a library that is already
    loaded, and the API cannot reach an event emitted before it existed.
    """
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - not installed, or failed to load
        return

    disable = getattr(ort, "disable_telemetry_events", None)
    if callable(disable):
        try:
            disable()
        except Exception:  # noqa: BLE001
            # A build where the call exists and does nothing useful is not a
            # reason to fail to start; the environment variable is the control
            # that matters and it has already been set.
            _log.debug("onnxruntime.disable_telemetry_events() was not accepted")
