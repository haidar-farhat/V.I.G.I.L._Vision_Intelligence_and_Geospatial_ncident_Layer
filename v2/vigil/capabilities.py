"""The capability manifest: what the product does, as data a test can check.

Each capability names the symbols that implement it and the test files that
exercise it. `tests/test_capabilities.py` fails when a symbol does not exist,
when a TESTED capability's tests do not reference its symbols, and when a
public service method is called by no interface. `CAPABILITIES.md` is
generated from this file by `python tasks.py capabilities`; it is never
edited by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    TESTED = "TESTED"   # implemented, tested, reachable from an interface
    IMPL = "IMPL"       # implemented and reachable; tests incomplete
    PLAN = "PLAN"       # decided, not built


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    title: str
    state: State
    symbols: tuple[str, ...]
    tests: tuple[str, ...]
    note: str = ""


MANIFEST: tuple[Capability, ...] = (
    Capability("geo", "Ground projection with honest uncertainty and fallback", State.TESTED,
               ("vigil.domain.geo.project_to_ground", "vigil.domain.geo.project_point", "vigil.domain.geo.field_of_view", "vigil.domain.geo.image_coordinates"),
               ("tests/test_geo.py",)),
    Capability("tracking", "Multi-object tracking with cumulative confirmation and coasting", State.TESTED,
               ("vigil.domain.tracking.Tracker",), ("tests/test_tracking.py",)),
    Capability("distance", "Distances that carry their own error, and never one without the other", State.TESTED,
               ("vigil.domain.geo.Distance", "vigil.domain.geo.separation", "vigil.domain.geo.distance_from_camera",
                "vigil.domain.zones.Zone.distance_from"),
               ("tests/test_geo.py",),
               "Errors combine in quadrature; `within` and `beyond` are not each other's negation, so a question the "
               "measurement cannot answer is answered neither way. A zone's distance is signed: inside reads negative"),
    Capability("relations", "What tracked things are doing together: inside, carried, with, approaching", State.TESTED,
               ("vigil.domain.relations.RelationTracker", "vigil.domain.relations.Relation",
                "vigil.domain.relations.occupants_of", "vigil.domain.relations.describe_group"),
               ("tests/test_relations.py", "tests/test_runtime.py", "tests/test_zones_events.py"),
               "Every relation is inferred and worded as such — one camera cannot tell inside from in front of — and "
               "carries the overlap, the distance and the frames it held. A zone entry says what the person was "
               "carrying, quoting those conditions; a vehicle entering one says how many people appear to be in "
               "it, which is the difference between a delivery and a problem; the plan links two joined tracks "
               "with a dotted line, because a solid one would read as a fact about the ground"),
    Capability("detection-settings", "What a site watches for and how sure it must be, kept with the site", State.TESTED,
               ("vigil.service.detection.DetectionSettings", "vigil.service.detection.detector_factory"),
               ("tests/test_detection.py", "tests/test_cli.py", "tests/test_console.py"),
               "They were flags, so a service started at boot analysed with the built-in list while the operator "
               "believed what they had typed once still applied. A flag now overrides the setting for that one run, "
               "the model is opened before any thread exists so an impossible watch list fails at start-up, and "
               "`vigil doctor` fails when a watched label is one the model cannot produce. The console's "
               "*Watch for…* edits the same setting and says a change lands at the next start, because a "
               "detector is made once per camera thread and lives as long as it does"),
    Capability("threats", "Labels a site treats as dangerous, and what it refuses to claim", State.TESTED,
               ("vigil.domain.threats.ThreatVocabulary", "vigil.domain.threats.ThreatRule"),
               ("tests/test_threats.py", "tests/test_site.py", "tests/test_diagnostics.py", "tests/test_cli.py"),
               "Empty by default: the shipped model names `knife` and `scissors`, and a kitchen raising a critical "
               "alert nightly teaches an operator to ignore the word. A claim needs a higher confidence and several "
               "frames, and `vigil doctor` fails when a configured label is one the model cannot produce"),
    Capability("zones", "Zones with membership hysteresis and a watch list", State.TESTED,
               ("vigil.domain.zones.Zone", "vigil.domain.zones.PresenceTracker"), ("tests/test_zones_events.py",)),
    Capability("rules", "Zone entry, loitering, after-hours and approach, each with its evidence", State.TESTED,
               ("vigil.domain.events.ZoneEntryRule", "vigil.domain.events.LoiteringRule", "vigil.domain.events.AfterHoursRule",
                "vigil.domain.events.ApproachRule", "vigil.service.runtime.Runtime.site_timezone"),
               ("tests/test_zones_events.py", "tests/test_runtime.py"),
               "A schedule is read in the site's own clock, which the workers are given at start; the IANA database "
               "ships with the product because Windows has none, and an unknown zone is refused where it is typed. "
               "`ApproachRule` acts on a relation rather than on presence, so a warning comes before the breach"),
    Capability("incidents", "Time-and-place correlation into incidents with risk", State.TESTED,
               ("vigil.domain.incidents.Correlator", "vigil.domain.incidents.associate", "vigil.domain.incidents.score_risk",
                "vigil.domain.incidents.link_same_camera_fragments"),
               ("tests/test_incidents.py",),
               "Cross-camera identity rests on time and place alone and each link says so; same-camera fragments are "
               "rejoined within 2 s and half the allowance, so a blinking detector does not report one person as two"),
    Capability("decode", "Files, local devices and RTSP behind the egress guard", State.TESTED,
               ("vigil.adapters.decode.VideoSource", "vigil.adapters.decode.LiveReader", "vigil.adapters.decode.require_private"),
               ("tests/test_decode.py",)),
    Capability("motion", "Motion detection that says it does not classify", State.TESTED,
               ("vigil.adapters.detectors.MotionDetector",), ("tests/test_detectors.py",)),
    Capability("onnx", "ONNX detection and segmentation, model read once per process", State.TESTED,
               ("vigil.adapters.detectors.OnnxDetector", "vigil.adapters.detectors.model_info"), ("tests/test_detectors.py",),
               "The inference path is exercised with a synthetic ONNX model built in the test; real weights are the operator's"),
    Capability("recording", "Clips on disk with digests, retention with preservation", State.TESTED,
               ("vigil.adapters.recorder.Recorder", "vigil.service.runtime.apply_retention"), ("tests/test_recording.py",)),
    Capability("store", "SQLite with migrations that have a way back, integrity check, backup/restore", State.TESTED,
               ("vigil.storage.store.Store", "vigil.storage.store.verify_backup", "vigil.storage.store.restore_backup"),
               ("tests/test_store.py",)),
    Capability("auth", "Accounts, scrypt, lockout, permission-based roles, principals", State.TESTED,
               ("vigil.service.auth.Accounts", "vigil.service.auth.Principal"), ("tests/test_auth.py",)),
    Capability("site", "Every site change through one service with a principal and an audit row", State.TESTED,
               ("vigil.service.site.SiteService", "vigil.service.site.SiteService.edit_zone",
                "vigil.service.site.SiteService.set_source"),
               ("tests/test_site.py", "tests/test_cli.py", "tests/test_console.py"),
               "Cameras and zones are edited in place: a camera that moved keeps its placement and its password "
               "follows; a zone's meaning changes while the ring somebody drew stays"),
    Capability("runtime", "Camera workers with bounded outboxes; poll persists, correlates, watches health", State.TESTED,
               ("vigil.service.runtime.Runtime", "vigil.service.runtime.CameraWorker"), ("tests/test_runtime.py",)),
    Capability("alerts", "Dark camera, stopped recording, retention shortfall, stuck thread, low disk leave the process", State.TESTED,
               ("vigil.service.alerts.Alerts",), ("tests/test_alerts.py", "tests/test_runtime.py")),
    Capability("review", "Working the queue: acknowledge, or dismiss with a reason", State.TESTED,
               ("vigil.service.review.IncidentReview",), ("tests/test_review.py", "tests/test_console.py", "tests/test_cli.py"),
               "Migration 2. A judgement names the person and is audited with what it was before; a dismissal without "
               "a reason is refused; re-correlation refines the conclusion and never undoes the judgement"),
    Capability("search", "Finding what happened: by camera, zone, severity, time, state or text", State.TESTED,
               ("vigil.service.search.Search", "vigil.service.search.moment"),
               ("tests/test_search.py", "tests/test_cli.py", "tests/test_console.py"),
               "Every filter is applied in SQL, so a search does not read a month of history into memory. Times may "
               "be typed as 2h, 3d, a date, or a full ISO moment; anything else is refused rather than widened to "
               "everything. `vigil incidents --camera … --since …`, `vigil events …`, and a filter bar in the console"),
    Capability("evidence", "Incident export with clips, hashes and a verifiable manifest", State.TESTED,
               ("vigil.service.evidence.export_incident", "vigil.service.evidence.verify_package"), ("tests/test_evidence.py",)),
    Capability("keychain", "Camera passwords in the OS keychain under a random handle", State.TESTED,
               ("vigil.adapters.keychain.Keychain",), ("tests/test_site.py",)),
    Capability("cli", "One command for every service method", State.TESTED,
               ("vigil.interfaces.cli.main",), ("tests/test_cli.py",)),
    Capability("console", "Desktop console: a thin view over the service, with the lock and the plan", State.TESTED,
               ("vigil.interfaces.console.window.ConsoleWindow", "vigil.interfaces.console.commands.Commands",
                "vigil.interfaces.console.plan.PlanView", "vigil.interfaces.console.dialogs.ask"),
               ("tests/test_console.py",),
               "`vigil console`. No domain state in the view; every change through Commands with the principal. "
               "Structural tests hold the three v1 Qt rules: no lambda over self in a connection, no WA_DeleteOnClose, "
               "a greyed control answers a click"),
    Capability("faces-plates", "Faces, plates and a subject register behind an identity switch", State.PLAN, (), (), "DECISIONS.md D-08"),
    Capability("packaging", "One executable, built and camera-tested by the task runner", State.IMPL,
               (), (), "`tasks.py package` builds dist/vigil/vigil.exe; `exetest` ran it on device:0 with recording and passed on 2026-09-05 (README.md); no installer, no signing"),
    Capability("observability", "An unattended run is watchable: metrics in the log, JSON on demand, a crash file", State.TESTED,
               ("vigil.service.runtime.Runtime.metrics", "vigil.logs.configure"),
               ("tests/test_runtime.py",),
               "A metrics line every minute while running; `VIGIL_LOG_JSON=1` makes every line one JSON object, redacted "
               "the same way; `faulthandler` writes a native crash to crash.log beside the log"),
    Capability("doctor", "An installation check: every way a deployment fails quietly, asked out loud", State.TESTED,
               ("vigil.service.diagnostics.run_checks",), ("tests/test_diagnostics.py", "tests/test_cli.py"),
               "`vigil doctor` checks the data directory, the database's integrity and schema, the site clock, the "
               "model, the keychain, free disk, accounts, cameras, zones and alert sinks; `--probe` opens every "
               "camera. Non-zero exit on any failure, so it can end an install script"),
    Capability("ci", "Every suite on three platforms, and again with the network taken away", State.IMPL,
               (), (),
               "`.github/workflows/ci.yml` jobs `v2` (ubuntu, windows, macos) and `v2-offline` (outbound traffic dropped, "
               "and the drop proved before anything runs). Written, never executed: it runs on the first push"),
    Capability("supervise", "Run unattended: restart on crash, a stop file, an OS service registration", State.TESTED,
               ("vigil.service.supervise.supervise", "vigil.service.supervise.service_definition", "vigil.service.supervise.install_service"),
               ("tests/test_supervise.py", "tests/test_cli.py"),
               "`vigil supervise -- run`, `vigil service install|print|uninstall`, `vigil run --stop`. The scheduled task runs at logon, not at boot"),
)
