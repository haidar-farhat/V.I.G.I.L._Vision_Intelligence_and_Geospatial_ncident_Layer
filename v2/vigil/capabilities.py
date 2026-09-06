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
    Capability("suppression", "Deciding which boxes to throw away, which was a third of detection",
               State.TESTED,
               ("vigil.kernel.native.suppress", "vigil.kernel.fallbacks._suppress_numpy"),
               ("tests/test_native.py", "tests/test_tiling.py"),
               "Measured before it was written, which is this repository's rule for a Rust kernel — and the "
               "measurement also said what *not* to write. NumPy: hard NMS 3.14 ms and soft-NMS 4.92 ms on "
               "300 proposals over 8 classes, against about 12.5 ms for the detection itself, so suppression "
               "was a quarter to a third of detection and tiling runs it once per tile. In Rust: **0.044 ms, "
               "138x**, with both implementations returning identical indices over randomised inputs "
               "including ties — the ordering is defined as descending score then ascending index, because "
               "`argsort` is not stable and a quantised model emits equal scores constantly. Two candidates "
               "on the same list were measured and **left in Python**: the assignment cost matrix at 5 "
               "microseconds and mask decode at 0.9 ms per frame, where a second implementation would cost "
               "more to keep in step than it saves. The appearance descriptor was the third, and the answer "
               "there was neither: `cv2.calcHist` with a mask does the same arithmetic 3.5x faster than "
               "indexing the pixels out and counting them in NumPy — identical results over 300 randomised "
               "crops — so 12 detections went from 3.61 ms to 1.49 ms with no new implementation of a colour "
               "space to maintain. The honest limit: none of this shows up on a near-empty frame, because "
               "suppression of one proposal is free. It bites on the busy frame, which is the frame you "
               "cannot afford to drop"),
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
    Capability("calibration", "A pose that has been measured, and says what the measurement is worth", State.TESTED,
               ("vigil.service.calibration.calibrate_pose", "vigil.service.calibration.calibrate_lens",
                "vigil.service.calibration.Calibration", "vigil.service.calibration.Correspondence",
                "vigil.service.site.SiteService.calibrate_camera"),
               ("tests/test_calibration.py", "tests/test_store.py", "tests/test_cli.py", "tests/test_console.py"),
               "`PoseUncertainty`'s defaults are what a compass and a tape are worth, and they dominated every "
               "position at range: at 40 m, two degrees of heading is 1.4 m of sideways error before the detector "
               "has contributed anything. Levenberg-Marquardt over heading, pitch, roll and mount height against "
               "reprojection error, with the covariance from sigma^2 (J^T J)^-1 -- and the covariance, not the "
               "refined pose, is the product: a fit from points in a cluster returns a *large* uncertainty, which "
               "is the correct answer. Measured on synthetic correspondences with 4 px of click noise, the reported "
               "1 sigma bracketed the true error 62-68% of the time against the 68% a correctly scaled covariance "
               "gives, and heading came out at +/-0.064 deg against the +/-2 assumed. Two things the intuition got "
               "wrong and measurement corrected: points down one image column are *not* degenerate (1.6e4), a tight "
               "cluster is (9.5e9); and position must be solved in metres east/north, because in degrees the "
               "condition number measures the units and refuses every good fit. Refuses a fit worse than the "
               "assumption it would replace rather than storing it"),
    Capability("triangulation", "Two cameras instead of one assumed plane, and the ground solved from what they see",
               State.TESTED,
               ("vigil.domain.triangulation.triangulate", "vigil.domain.triangulation.fit_ground",
                "vigil.domain.triangulation.TriangulatedPoint", "vigil.domain.triangulation.GroundPlane",
                "vigil.service.triangulation.Geometry", "vigil.service.runtime.Runtime.ground_plane"),
               ("tests/test_triangulation.py", "tests/test_native.py"),
               "Every position this product produced came from intersecting one ray with an *assumed* level "
               "plane at the camera's mount height, and it was wrong in three specific ways: anything not "
               "standing on the ground was placed long (a person 1.6 m up on a dock, seen at 26 m, by over 2 m), "
               "a sloped yard biased every position along the line of sight, and nothing could tell a wall from "
               "a floor. Two rays need none of it. The midpoint of their common perpendicular is the position, "
               "the gap between them at closest approach *is* the association test -- two rays at one object "
               "pass within centimetres, two at different people do not come within three metres -- and the "
               "height above the fitted plane is what says an object is not standing on the ground. RANSAC for "
               "the plane, because the contacts are contaminated by construction and a least-squares fit tilts "
               "towards every person on a dock. The minimum parallax is solved per pair from how well the two "
               "poses are known rather than fixed: 5 degrees for a calibrated pair, 44 for an assumed one at "
               "80 m, because below that the two-view error is worse than the projection it would replace. The "
               "solved slope is written back to **every** placed camera, including the lone one that could "
               "never have measured it -- which is the camera whose positions were worst",
               ),
    Capability("cross-camera", "One object followed between two cameras, with the threshold measured on this site",
               State.TESTED,
               ("vigil.domain.appearance.ColourBalance", "vigil.domain.appearance.CrossCameraSeparation",
                "vigil.domain.incidents.associate", "vigil.domain.events.Evidence"),
               ("tests/test_crosscamera.py", "tests/test_incidents.py"),
               "ROADMAP section 4, and the part that was missing was never the matcher. Two cameras render the "
               "same coat differently -- white balance, exposure, a sodium lamp over one of them -- so a cosine "
               "distance between raw histograms measures *which camera took the picture* at least as strongly as "
               "what was in it. Measured on synthetic tints: one coat through two cameras came out 0.117 apart "
               "while two different coats through one camera came out 0.044, so the camera outweighed the object "
               "by nearly three to one. Each camera's descriptors are now divided by that camera's own running "
               "mean before they are compared, and what survives is how an object differs from its camera's "
               "average. The threshold is then measured rather than assumed, and both halves of the evidence are "
               "free: two tracks in one frame are certainly different objects, and two tracks whose rays converge "
               "to within three metres are certainly the same one -- geometry decided that pair without "
               "consulting appearance, which makes it genuine ground truth for appearance. When what the same "
               "object scores overlaps what different objects score, no threshold does both jobs and the "
               "correlator is told so: it falls back to time and place alone rather than adding noise. Appearance "
               "only ever rejects a link, never raises a score, because colour is weak evidence for sameness and "
               "strong evidence for difference"),
    Capability("detection-recall", "Finding more, and being able to say what more cost", State.TESTED,
               ("vigil.adapters.tiling.TiledDetector", "vigil.adapters.tiling.tiles_for",
                "vigil.adapters.tiling.far_band", "vigil.adapters.detectors._soft_nms"),
               ("tests/test_tiling.py", "tests/test_detectors.py"),
               "Four changes, each measured on this machine with the shipped yolov8n-seg on DirectML over "
               "1080p frames. **The vocabulary split**: the classes the detector runs with and the classes the "
               "site alerts on were one frozen set of six, so a trailer, a dog or a ladder against a fence was "
               "invisible to the tracker, the plan and the map. The detector now reports everything its model "
               "knows and `watch_labels` governs only what reaches a rule. **The floor came down** from 0.50 to "
               "0.25: measured at 0.00 detections per frame at 0.50 against 1.08 at 0.25 on the same twelve "
               "frames -- the shipped threshold found nothing at all in an ordinary room -- and it is safe "
               "because the tracker was rebuilt around cumulative confirmation, so a thing seen once at 0.3 and "
               "never again is never confirmed. **Soft-NMS** decays an overlapping box instead of deleting it, "
               "because a queue of people or a row of parked cars at 0.55 overlap is not a duplicate and a "
               "suppressed detection leaves no trace for anybody to notice. **Tiling** runs the model over "
               "crops of the far ground at native scale, where a person 40 m away is 24 px after letterboxing "
               "and below what a nano model can find; the tiles are placed from the pose rather than uniformly, "
               "because `project_to_ground` says which rows are far. Measured: 1.08 to 1.33 detections per "
               "frame for 11.0 ms to 55.3 ms, which is exactly the 1+4 inferences it costs. That cost is why "
               "it is off on CPU and why the honest gain is unproven -- the laptop webcam has no far ground, "
               "so the far-half recall proxy this was built to measure came out zero on both sides and will "
               "stay unmeasured until it runs on a camera that can see something distant"),
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
    Capability("console", "Desktop console: every verb under the thing it acts on", State.TESTED,
               ("vigil.interfaces.console.window.ConsoleWindow", "vigil.interfaces.console.commands.Commands",
                "vigil.interfaces.console.commands.Site", "vigil.interfaces.console.widgets.Panel",
                "vigil.interfaces.console.plan.PlanView", "vigil.interfaces.console.dialogs.ask"),
               ("tests/test_console.py",),
               "`vigil console`, and `VIGIL.exe` at the repository root for the window on its own. No domain state "
               "in the view; every change through Commands with the principal. Structural tests hold the three v1 "
               "Qt rules: no lambda over self in a connection, no WA_DeleteOnClose, a greyed control answers a "
               "click. **Reworked**, and each change fixes something a photograph showed. Seventeen buttons sat in "
               "one row across the top at equal weight, five of them acting on a selection at the far side of the "
               "window -- so the commonest outcome of pressing one was the sentence \"Select a camera to place\", "
               "the interface asking for something it could see. The verbs now sit under their nouns and that error "
               "is unreachable; a control greyed for want of a selection says **that**, not \"Locked\", because a "
               "reason that does not match the cause sends somebody to press a button already pressed. The status "
               "bar carried seven permanent labels at eleven per cent of its width each and rendered them "
               "\"MONITOR - site locked; press...\" and \"yolov8n-seg - watching 80 cl...\"; it carries three now "
               "and the other four moved into the heading of the panel they describe. One control is loud -- Start, "
               "or Stop while running -- because a window where three things shout is one where nothing does. The "
               "window reads the service **once** per repaint through `Commands.snapshot()` rather than asking it "
               "five separate questions, so every panel is drawn from one moment: a camera removed between two of "
               "those calls used to appear in one panel and not the next"),
    Capability("identity", "Faces, plates and a subject register behind one identity switch",
               State.TESTED,
               ("vigil.domain.identity.Register", "vigil.domain.identity.Verdict",
                "vigil.perception.faces.FaceReader", "vigil.perception.plates.Accumulator",
                "vigil.service.identity.IdentityService"),
               ("tests/test_identity.py",),
               "DECISIONS.md D-08, whose gate -- an hour on a physical IP camera -- was **waived**, not met; "
               "that entry records who waived it and what is therefore unproven. **No face or plate model "
               "ships and none has been run, so every threshold here is a stated assumption and says so in "
               "its own docstring.** What is built and tested is the discipline. OFF by default: one switch, "
               "not two, and `enable` refuses without both a retention limit and a written reason, while "
               "`vigil doctor` FAILS on a database found on with no limit -- an unbounded biometric store is "
               "the worst thing this can become and the failure is otherwise entirely silent. A single frame "
               "is **structurally incapable** of asserting a name: `compare` has no path to MATCH and "
               "`Match.label` raises unless it earned one, where v1 stated the same rule in its headline and "
               "left a public method that broke it. Two encoders' embeddings are never compared -- the check "
               "v1 asserted at length and never wrote. A plate has **no text at all** while any character is "
               "unresolved, so a half-read plate cannot be logged, exported or searched; a character needs "
               "three votes *and* more than the runner-up, because the count alone resolves a position four "
               "frames called 8 and four called B. Four v1 bugs are fixed here with tests naming them: the "
               "unconditional softmax that silently flattened every confidence under the threshold so no "
               "plate ever resolved; a CTC blank index off by one for any blank not at the end; a read with "
               "no per-character confidence getting a free vote, which is exactly what the fabricated 1.0 it "
               "claimed to avoid would have done; and a person-box check on x and y but not width, so a box "
               "of width 200.0 clamped to the whole frame and pointed the face pipeline at the street. "
               "Erasure vacuums with `secure_delete`, because a DELETE alone leaves the template readable in "
               "freed pages -- v1 promised 'a delete that really deletes' and did not. The audit trail holds "
               "identifiers and never names, since it is append-only and a name in it outlives the erasure "
               "it records"),
    Capability("evaluation", "Precision and recall against corrected labels, and the split it refuses",
               State.TESTED,
               ("vigil.service.evaluation.evaluate", "vigil.service.evaluation.check_split",
                "vigil.service.evaluation.average_precision"),
               ("tests/test_evaluation.py",),
               "The harness, built before there is data to put in it, so that on the day labels appear there "
               "is no temptation to write a quick script that scores the training set. Precision, recall and "
               "all-points average precision per class at IoU 0.5, greedy matching by descending confidence "
               "with each label claimed once. **It refuses a validation set that shares a day with training** "
               "-- and stops rather than warning, because a warning printed above a 0.98 is read as a 0.98. "
               "Consecutive frames of security footage are near-duplicates, so a random split tests the model "
               "on the frame after the one it trained on. Classes the validation set never contained are "
               "excluded from mAP rather than scored zero, since averaging in a zero for each of COCO's "
               "eighty would bury every real figure. Frames missing an image or a label are counted and "
               "reported, because a corpus quietly missing half its labels scores beautifully on recall. "
               "**Nothing has been scored: no labelled dataset exists for this site**, and this module says "
               "so rather than producing a number",
               ),
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
