"""Is this installation actually going to work? Asked before anybody leaves site.

Every check answers one question a deployment can fail on quietly: a model
that is not there, a keychain that cannot hold a password, a disk that is
already nearly full, a site clock this machine does not know, a camera nobody
placed. Each returns a state, a sentence saying what is true, and — when
something is wrong — the one thing to do about it.

Nothing here changes anything. `vigil doctor` runs it, and exits non-zero
when any check fails, so it can be the last line of an install script.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable

from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Free space below which a recording deployment is already in trouble.
LOW_DISK_BYTES = 5 * 1024**3


class State(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    state: State
    detail: str
    remedy: str | None = None

    def describe(self) -> str:
        line = f"{self.state:<4} {self.name:<22} {self.detail}"
        # `->`, not an arrow: a Windows console is cp1252 and prints one as
        # `?`, which reads as a defect in the tool doing the checking.
        return line if self.remedy is None else f"{line}\n     -> {self.remedy}"


def _writable(directory: Path) -> Check:
    name = "data directory"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".vigil-write-test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return Check(name, State.FAIL, f"{directory} cannot be written: {error}",
                     "Choose another with --data-dir or VIGIL_DATA_DIR, or fix the permissions.")
    return Check(name, State.OK, str(directory))


def _database(store) -> Check:
    from ..storage.schema import SCHEMA_VERSION

    applied = store.applied_versions()
    if not applied:
        return Check("database", State.FAIL, "no migration has been applied", "Run any command once to create it.")
    if applied[-1] != SCHEMA_VERSION:
        return Check("database", State.WARN, f"schema {applied[-1]}, this build knows {SCHEMA_VERSION}",
                     "Run any command once; migrations apply themselves.")
    row = store._connection.execute("PRAGMA quick_check").fetchone()  # noqa: SLF001 - the check is the point
    if row is None or row[0] != "ok":
        return Check("database", State.FAIL, f"integrity check says {row[0] if row else 'nothing'}",
                     "Restore a backup: `vigil restore <file>`.")
    return Check("database", State.OK, f"schema {applied[-1]}, integrity ok")


def _model(settings) -> Check:
    model = settings.default_model()
    if model is None:
        return Check("detection model", State.WARN,
                     "none found, so this site runs motion only, which cannot classify anything",
                     "Put an ONNX model in " + str(settings.models))
    try:
        from ..adapters.detectors import model_info

        info = model_info(model)
    except Exception as error:  # noqa: BLE001 - any failure here is the operator's to hear
        return Check("detection model", State.FAIL, f"{model.name} will not load: {error}",
                     "Replace the file, or remove it to fall back to motion detection.")
    names = len(info.class_names)
    return Check("detection model", State.OK, f"{model.name}, {names} class(es), {'masks' if info.kind.endswith('segment') else 'boxes'}")


def _keychain(keychain) -> Check:
    if not keychain.available:
        return Check("keychain", State.WARN, "none usable, so a camera password cannot be kept across restarts",
                     "On Linux install a Secret Service provider; elsewhere check the OS credential store.")
    return Check("keychain", State.OK, "available; passwords are stored under a random handle")


def _disk(settings) -> Check:
    probe = settings.recordings
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError as error:
        return Check("disk", State.WARN, f"cannot be measured: {error}")
    state = State.OK if free >= LOW_DISK_BYTES else State.WARN
    remedy = None if state is State.OK else "Free space, or lower the retention policy: `vigil retention`."
    return Check("disk", state, f"{free / 1024**3:.1f} GiB free where recordings go", remedy)


def _clock(store) -> Check:
    from .site import SiteError, SiteService

    name = store.site().get("timezone") or "UTC"
    try:
        SiteService.known_timezone(name)
    except SiteError as error:
        return Check("site clock", State.FAIL, str(error), "Set a known zone: `vigil site name <name> --timezone <zone>`.")
    return Check("site clock", State.OK, f"schedules are read in {name}")


def _accounts(store) -> Check:
    if not store.users():
        return Check("accounts", State.WARN, "none, so nothing is gated and the audit trail names the OS account",
                     "Create one: `vigil users add NAME --role ADMIN`.")
    active = [u for u in store.users() if u["active"]]
    if not active:
        return Check("accounts", State.FAIL, "every account is disabled; nobody can sign in",
                     "Enable one: `vigil users enable NAME`.")
    return Check("accounts", State.OK, f"{len(active)} active of {len(store.users())}")


def _identity(store) -> Check:
    """Whether this site processes biometrics, and whether it says for how long.

    **A FAIL, not a warning, when it is on with no retention limit.** An
    unbounded biometric store is the worst thing this feature can become, and
    the check has to be loud because the failure is entirely silent: nothing
    looks wrong about a `face_observations` table with four million rows in
    it, and nobody goes looking.

    OK when it is off, which is where every site starts and most stay.
    """
    from .identity import IdentityService

    state = IdentityService(store).state()
    if not state.enabled:
        return Check("identity", State.OK, "faces and plates are off")
    if state.retention_days is None:
        return Check("identity", State.FAIL,
                     "faces and plates are ON with no retention limit, so biometric data is being "
                     "kept for ever",
                     "Set one now: `vigil identity enable --retention-days N --reason ...`, or "
                     "turn it off with `vigil identity disable`.")
    counted = store.biometric_counts()
    return Check("identity", State.OK,
                 f"faces and plates are on; {counted['subjects']} subject(s), observations "
                 f"deleted after {state.retention_days} day(s)")


def _site(store) -> Check:
    cameras = store.cameras()
    if not cameras:
        return Check("cameras", State.WARN, "none", "Add one: `vigil cameras add NAME SOURCE --place …`.")
    unplaced = [c["id"] for c in cameras if c["pose"] is None]
    if unplaced:
        return Check("cameras", State.WARN,
                     f"{len(cameras)} camera(s); {', '.join(unplaced)} unplaced, so nothing they see can be located",
                     "Place them: `vigil cameras place ID lat,lon,height,heading,pitch`.")
    return Check("cameras", State.OK, f"{len(cameras)} camera(s), all placed")


def _zones(store) -> Check:
    zones = store.zones()
    if not zones:
        return Check("zones", State.WARN, "none, so no rule can fire and no incident can be raised",
                     "Draw one in the console, or `vigil zones add …`.")
    return Check("zones", State.OK, f"{len(zones)} zone(s)")


def _threats(settings, store) -> Check:
    """Whether the site's threat labels mean anything against the model it has."""
    from ..domain.threats import ThreatVocabulary

    vocabulary = ThreatVocabulary.from_labels(store.site().get("threat_labels") or [])
    if not vocabulary:
        return Check("threat labels", State.OK, "none, so nothing is called a threat here",
                     "If this site needs them: `vigil site threats --set knife,gun` (a model that names them is required).")
    model = settings.default_model()
    if model is None:
        return Check("threat labels", State.FAIL, f"{len(vocabulary)} configured but there is no model to name them",
                     "Install a model, or clear them with `vigil site threats --clear`.")
    try:
        from ..adapters.detectors import model_info

        names = list(model_info(model).class_names.values())
    except Exception as error:  # noqa: BLE001
        return Check("threat labels", State.WARN, f"cannot be checked: {error}")
    missing = vocabulary.unknown_to(names)
    if missing:
        return Check("threat labels", State.FAIL,
                     f"{', '.join(missing)} cannot be produced by {model.name}, so nothing will ever raise them",
                     "Use a model trained on those classes, or drop them from the list.")
    return Check("threat labels", State.OK, vocabulary.describe())


def _detection(settings, store) -> Check:
    """Whether this site's watch list is one the installed model can satisfy."""
    from .detection import DetectionSettings

    chosen = DetectionSettings.from_site(store.site())
    if chosen.labels is None and chosen.confidence is None:
        return Check("detection", State.OK, "built-in watch list, the detector's own threshold",
                     "To change it for every run: `vigil site detection --watch person,car`.")
    model = settings.default_model()
    if model is None:
        return Check("detection", State.WARN, f"{chosen.describe()}, but motion detection cannot watch a class",
                     "Install a model, or the watch list has no effect.")
    try:
        from ..adapters.detectors import model_info

        names = list(model_info(model).class_names.values())
    except Exception as error:  # noqa: BLE001
        return Check("detection", State.WARN, f"cannot be checked: {error}")
    missing = sorted(l for l in (chosen.labels or ()) if l not in names)
    if missing:
        return Check("detection", State.FAIL,
                     f"{', '.join(missing)} cannot be produced by {model.name}, so this site watches for nothing",
                     "Use a model trained on those classes, or `vigil site detection --clear`.")
    return Check("detection", State.OK, chosen.describe())


def _engine_core() -> Check:
    """Is the Rust core loaded, and is it the right one?

    A WARN rather than a FAIL, and the distinction is the point: without the
    core the product still analyses, tracks and raises events — the NumPy
    paths are the same algorithms and `tests/test_native.py` holds them to the
    same answers — but it cannot build a map at all, and it does the
    association arithmetic an order of magnitude more slowly. That is a
    degraded installation, not a broken one, and calling it either of the
    other two things would be wrong.
    """
    from ..kernel import native

    name = "engine core"
    if native.available():
        return Check(name, State.OK,
                     f"loaded from {native.loaded_from()}, ABI {native.ABI_VERSION}")
    return Check(name, State.WARN,
                 f"not loaded, so `vigil map` is unavailable and tracking runs on the slower "
                 f"NumPy path: {native.fault()}",
                 "build it with `python tasks.py core`, or set VIGIL_CORE_PATH")


def _inference(settings) -> Check:
    """Which execution provider inference will actually get.

    "Why is this slow" is answered here more often than by anything in the
    model. onnxruntime falls back silently when a provider will not
    initialise, so a machine with a GPU, a CUDA build and the wrong driver
    runs on the CPU and says nothing at all.
    """
    from ..adapters.detectors import available_providers

    name = "inference"
    providers = available_providers()
    if not providers:
        return Check(name, State.FAIL, "onnxruntime is not importable, so no model can run",
                     "install onnxruntime, or onnxruntime-gpu for a machine with a GPU")
    if providers[0] == "CPUExecutionProvider":
        return Check(name, State.WARN,
                     "the installed onnxruntime offers only the CPU, so inference will use it",
                     "on a machine with a GPU, install onnxruntime-gpu (or onnxruntime-directml "
                     "on Windows) for a large speed-up; on a CPU-only appliance this is expected")
    return Check(name, State.OK, f"{providers[0]} available (then {', '.join(providers[1:])})")


def _alerts(settings) -> Check:
    from .alerts import Alerts

    sinks = Alerts.from_settings(settings).sinks
    if not sinks:
        return Check("alerts", State.WARN, "go to the log only, which nobody is watching",
                     "Set VIGIL_ALERT_FILE, VIGIL_ALERT_COMMAND or VIGIL_ALERT_WEBHOOK.")
    return Check("alerts", State.OK, ", ".join(type(s).__name__ for s in sinks))


def _sources(store, probe: bool) -> Check:
    """Whether every camera can actually be opened. Slow, so it is asked for."""
    if not probe:
        return Check("camera sources", State.OK, "not probed (pass --probe to open each one)")
    from ..adapters.decode import DecodeError, VideoSource

    broken = []
    for camera in store.cameras():
        source = VideoSource(camera["source"], source_id=camera["id"])
        try:
            source.open()
        except DecodeError as error:
            broken.append(f"{camera['id']}: {error}")
        finally:
            source.close()
    if broken:
        return Check("camera sources", State.FAIL, "; ".join(broken),
                     "Check the address, the network, and the password (`vigil cameras password ID`).")
    return Check("camera sources", State.OK, "every camera opened")


def run_checks(settings, store, keychain, *, probe: bool = False) -> list[Check]:
    """Every check, in the order a deployment fails in."""
    checks: list[Callable[[], Check]] = [
        lambda: _writable(settings.data_dir),
        lambda: _database(store),
        lambda: _clock(store),
        lambda: _model(settings),
        lambda: _engine_core(),
        lambda: _inference(settings),
        lambda: _keychain(keychain),
        lambda: _disk(settings),
        lambda: _accounts(store),
        lambda: _site(store),
        lambda: _identity(store),
        lambda: _zones(store),
        lambda: _detection(settings, store),
        lambda: _threats(settings, store),
        lambda: _alerts(settings),
        lambda: _sources(store, probe),
    ]
    out = []
    for check in checks:
        try:
            out.append(check())
        except Exception as error:  # noqa: BLE001 - a check that throws is itself a finding
            _log.exception("a check failed to run")
            out.append(Check("check", State.FAIL, f"this check could not run: {type(error).__name__}: {error}"))
    return out


def worst(checks: list[Check]) -> State:
    if any(c.state is State.FAIL for c in checks):
        return State.FAIL
    if any(c.state is State.WARN for c in checks):
        return State.WARN
    return State.OK
