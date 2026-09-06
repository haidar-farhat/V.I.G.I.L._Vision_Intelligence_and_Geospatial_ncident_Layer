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
    Capability("geo", "A pinhole camera on one Earth, with the error propagated through it", State.TESTED,
               ("vigil.domain.geo.project_to_ground", "vigil.domain.geo.project_point",
                "vigil.domain.geo.field_of_view", "vigil.domain.geo.image_coordinates",
                "vigil.domain.geo.PoseUncertainty", "vigil.domain.geo.ProjectionFailure"),
               ("tests/test_geo.py", "tests/test_native.py"),
               "Two defects fixed. v1 and v2 both decoupled yaw from elevation and called it "
               "rectilinear: that is a cylindrical sensor, and at the corner of the reference pose it "
               "was 4.8 degrees out in elevation, which is 21% of the distance to whatever stood there. "
               "`roll` had existed since v1 and was read by nothing. And distances between tracks went "
               "through a spherical haversine while zones went through a WGS84 tangent plane — two "
               "Earths 0.248% apart. Uncertainty is now a Jacobian over the pose, the mount height, "
               "the contact point and the terrain, reported as an ellipse because a shallow ray is "
               "vague along its own direction and sharp across it"),
    Capability("tracking", "Kalman tracking with optimal association and appearance re-identification", State.TESTED,
               ("vigil.domain.tracking.Tracker", "vigil.domain.tracking.TrackState",
                "vigil.kernel.filtering.predict", "vigil.kernel.native.assign"),
               ("tests/test_tracking.py", "tests/test_native.py"),
               "Replaces the greedy IoU pass v1 and v2 shared, which swapped two people's identities "
               "whenever they crossed, and the EMA velocity, which had no uncertainty to gate on and "
               "gave stationary objects a drift. Association is a globally optimal assignment over a "
               "cost that combines appearance and overlap, gated by the filter's own Mahalanobis "
               "distance; a second pass offers weak detections to tracks that have nothing, which is "
               "what recovers an object through an occlusion; a third re-identifies a lost track by "
               "appearance, which is what stops one person becoming eleven objects. Confirmation stays "
               "cumulative and a coasted box is still never recorded as a measurement"),
    Capability("appearance", "What a tracked thing looks like, during association rather than after it", State.TESTED,
               ("vigil.domain.appearance.Appearance", "vigil.domain.appearance.Gallery",
                "vigil.domain.appearance.SceneSeparation", "vigil.perception.appearance.describe"),
               ("tests/test_perception.py", "tests/test_tracking.py"),
               "A masked HSV histogram, migrated from v1's `reid.py` and moved to where it can prevent "
               "a fragment instead of reconciling one an hour later. A gallery of recent looks rather "
               "than v1's single moving average, because the mean of a person's front and their back "
               "is a person who does not exist. Colour is all it has, so similarity never links on its "
               "own: time and place are conditions, not tie-breakers. The re-identification "
               "threshold is **measured, not assumed**: two tracks visible in one frame are "
               "different objects by construction, so `SceneSeparation` learns what a stranger "
               "scores in this particular scene and a candidate has to beat that by a margin. "
               "`tools/calibrate.py` found the shipped constant far too generous on real video — "
               "different objects at a median of 0.105 against a gate of 0.35 — and a scene that "
               "proves its strangers look alike now refuses to re-identify rather than merging "
               "two people, which is the invisible error"),
    Capability("camera-motion", "Whether the camera moved, told apart from whether the scene did", State.TESTED,
               ("vigil.perception.motion.CameraMotionEstimator", "vigil.perception.motion.CameraMotion"),
               ("tests/test_perception.py", "tests/test_tracking.py"),
               "Sparse optical flow with a forward-backward check and a RANSAC partial affine, applied "
               "to every track's filter — covariance included — before prediction. Neither v1 nor v2 "
               "measured this, so a gust on a mast read as every object accelerating at once. A fit "
               "explaining a minority of the frame is a lorry crossing it and is refused; so is a shift "
               "larger than shake produces"),
    Capability("frame-quality", "A camera that is producing frames nobody could detect anything in", State.TESTED,
               ("vigil.perception.quality.FrameQualityMonitor", "vigil.service.alerts.CAMERA_DEGRADED"),
               ("tests/test_perception.py", "tests/test_runtime.py"),
               "v1 and v2 both had `camera.dark` — no frames at all — and nothing between that and "
               "working, so an unfocused lens, a blown-out frame and a decoder repeating its last frame "
               "all failed silently while the frame counter climbed. Measured per frame, alerted on when "
               "sustained, and reported in `vigil health`"),
    Capability("mapping", "The site's ground map and texture, built by the cameras that watch it", State.TESTED,
               ("vigil.service.mapping.MapBuilder", "vigil.service.mapping.build_from_cameras",
                "vigil.service.mapping.save_map", "vigil.service.mapping.load_map",
                "vigil.service.mapping.confidence_of"),
               ("tests/test_mapping.py", "tests/test_native.py"),
               "Migrated from v1's `orthophoto.py` and `basemap.py`, which v2 dropped entirely. A "
               "per-cell median over many frames removes whoever walked through; a confidence layer "
               "made of sample count, disturbance, ground resolution and projection error says which "
               "cells are a measurement of the ground and which are a picture of something over it. "
               "Ground nobody looked at stays empty and is drawn as empty. The raster is in Rust: v1 "
               "measured the NumPy version at 79 ms per frame per camera and sampled four frames a "
               "second because of it"),
    Capability("coverage", "Which ground the cameras reach, and — the useful half — which they do not", State.TESTED,
               ("vigil.service.coverage.analyse", "vigil.service.coverage.Gap",
                "vigil.service.coverage.Band", "vigil.service.coverage.boundary_from_cameras"),
               ("tests/test_coverage.py",),
               "Migrated from v1, which v2 dropped, and rebuilt on the corrected footprint: v1's "
               "numbers inherited the decoupled camera model's error and ignored roll entirely. "
               "`vigil coverage`. Gaps are invisible on a plan view — six wedges look thorough and "
               "the four-metre corridor between two of them looks like nothing until somebody walks "
               "down it. Reported in error bands as well as area, because \"covered\" and \"covered "
               "well enough to say which side of a line somebody was on\" are different questions. "
               "Every figure is an upper bound and says so: nothing here models occlusion or "
               "resolution, and both would make coverage smaller"),
    Capability("dataset", "The corpus running this product already writes, made readable", State.TESTED,
               ("vigil.service.dataset.collect", "vigil.service.dataset.write",
                "vigil.service.dataset.Sample"),
               ("tests/test_dataset.py",),
               "`vigil dataset export`. The clips, the events and the *reasons a person typed when "
               "dismissing a false positive* were all already on disk and nothing joined them. "
               "Frames are cut from the clip covering each incident, pre-labelled with the "
               "detector's own output, and written with the model's digest so a corrected set "
               "cannot be confused about whose mistakes it was correcting. The split is by DAY and "
               "a single day gets no validation set at all: consecutive video frames are "
               "near-duplicates, a random split leaks almost perfectly, and inventing one is how a "
               "meaningless number comes to be believed"),
    Capability("detect-every", "Detecting on a subset of frames and tracking through the rest", State.TESTED,
               ("vigil.service.detection.MAX_DETECT_EVERY",),
               ("tests/test_detection.py",),
               "Measured at 3.0x less detection for 0.008 box heights of position lag at N=3, "
               "against a projection error over a metre at range. Only safe because the tracker "
               "was rebuilt around a Kalman filter, a weak-detection recovery pass and "
               "re-identification across a gap — on the exponential-average tracker it replaced "
               "this would have been reckless. A per-site setting rather than a flag, because a "
               "service started at boot has nobody to type a flag at it"),
    Capability("engine-core", "The arithmetic that runs per pixel, in Rust behind a C ABI", State.TESTED,
               ("vigil.kernel.native.load", "vigil.kernel.native.ortho_sample",
                "vigil.kernel.native.assign", "vigil.kernel.native.kalman_predict"),
               ("tests/test_native.py", "tests/test_diagnostics.py"),
               "Ground rasterisation, optimal assignment and the box filter. ABI-versioned and "
               "layout-checked on load, so a signature that moved produces a refusal rather than "
               "plausible wrong geometry. `vigil doctor` reports whether it is loaded and from where; "
               "`tests/test_native.py` holds it to the NumPy mirrors it can fall back to, and to the "
               "independent exact inverse where it has none"),
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
