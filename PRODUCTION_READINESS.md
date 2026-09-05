# Production Readiness — Brutal, Exhaustive TODO Audit

**Date:** 2026-09-05 · **Branch:** `Phase2` at `d38c64e` plus the uncommitted work of
2026-09-05 · **Auditor's stance:** hostile production review. Nothing below is
marked done because code exists, a test passes, or a document says so. Every
finding names the evidence it rests on; every TODO says what *done* means.

This document **adds to** [FEATURES.md](FEATURES.md) (the product definition,
469 capabilities with states), [ROADMAP.md](ROADMAP.md) (the build order),
[STATUS.md](STATUS.md) (what is measured) and [docs/SECURITY.md](docs/SECURITY.md).
It removes nothing from them. Where it disagrees with a document, it says which
line of code disagrees.

**Verdict in one line:** the analytical core is real, measured and unusually
well tested; the *product around it* — accounts, secrets, service supervision,
installers, alerting, backup, a release process, a single run on a real IP
camera, a single run of CI — does not exist, and the repository already says so.
It is not production-ready, and it cannot be called so until the P0 list is empty.

---

## 1. Executive Production Readiness Assessment

| Dimension | State | Blocking? |
|---|---|---|
| Core analytics (decode → detect → track → project → zones → events → correlation) | Built, measured on a rendered scene and one webcam; 1,500+ tests | no |
| Detection quality on real footage | Unmeasured. Tonight's false objects ("couch", "bottle") were the 80-class model at a 0.35 floor; mitigated by a watch list and a 0.50 floor, **both unmeasured** | **yes (P1)** |
| Operator console | Native Qt, 374 tests, drivable from the command line since tonight; runs on the GUI thread with the node's persistence. **Until tonight, Place… → OK and Add zone… → OK raised on a deleted dialog and did nothing** — found by the packaged binary on the camera, never by a test | partly |
| Authentication / authorization | **None.** Every action is attributed to the literal string `console` | **yes (P0)** |
| Secrets | **Nothing is persisted**; RTSP passwords must be retyped after every restart and travel on the command line | **yes (P0)** |
| Recording & retention | **Closed 2026-09-05 (REL-02):** the console records per camera through a Record box behind the lock, the flag survives restarts (migration 11), and retention sweeps itself from the node's poll every ten minutes. Still open: an alert that leaves the process when the sweep cannot reach its target | partly (P1 remainder → OBS-01) |
| Crash recovery / supervision | None. No service, no watchdog, no restart | **yes (P0)** |
| Installer / signing | None. A folder, unsigned; SmartScreen warns, Gatekeeper refuses | **yes (P0)** |
| CI | Written for three platforms; **has never executed remotely** (ROADMAP 0.4) | **yes (P0)** |
| IP camera (RTSP) | Code path exists; **no physical IP camera has ever been contacted** | **yes (P1)** |
| Backup / restore / DR | None | **yes (P1)** |
| Observability & alerting | Log file + in-window status; no alert of any kind leaves the process | **yes (P1)** |
| Documentation truth | Several documents describe a deleted TypeScript design; the egress-override contradiction is closed (SEC-03, 2026-09-05); DATABASE.md's claims remain (DATA-02) | **yes (P1)** |
| Zero-WAN / credential redaction / audit chain | Enforced three ways, tested against deliberately bad input; the strongest part of the system | no |

**What is genuinely good, and should not be re-litigated:** the C-ABI boundary
with ABI and struct-size guards; deterministic event ids; three-state semantics
(unknown / stationary / moving; inside / outside / uncertain); the credential
redaction that tests even a password's *length*; the offline audit that reads
compiled dependency bytes; the append-only audit with a hash chain; the honesty
culture (every measured failure is bounded by a test rather than hidden). A
production programme should build on this rather than replace it.

---

## 2. Current State / Existing Implementation

### What was verified tonight, by running it

| Check | Result | Evidence |
|---|---|---|
| Rust core suite | 60 passed | `python tasks.py test`, 2026-09-04 23:14 |
| Engine suite (HEAD `7a1b5df`) | 1096 passed, 2 skipped, 1 xfailed | same run |
| Console suite (HEAD `7a1b5df`) | **2 failed**, 330 passed — two tests added in that commit did not match the code they tested; fixed in the working tree by a concurrent editor at 23:18 | same run; `git diff` at 23:20 |
| Console suite after tonight's changes | 374 passed (363 before the camera runs' fixes) | 2026-09-05, offscreen |
| Static guards | offline audit 58 files clean; docs lint 41 diagrams parse | tonight |
| Operator's log | Every run on 2026-09-04 loaded the model with **80 classes** (`segmenter ready … 80 class name(s)`) — the watch-list commit at 23:10 post-dates the packaged build at 22:00 | `%LOCALAPPDATA%\SentinelVision\logs\sentinel.log` lines 465–516 |
| Operator's database | 1 camera (`device:0`), unplaced; 0 zones; 0 events; 0 incidents; 74 audit rows. The audit trail shows unlock → start → stop → remove camera → re-add → relock → start → stop, and **never** `camera.placed` or `zone.created` | `sentinel.db`, read-only query |
| Packaged bundle | `dist/SentinelVision/*.exe` built 2026-09-04 22:00, models copied; rebuilt tonight by `tasks.py ci --package` (result recorded in §2.3) | `dist/` listing |

### What the repository says about itself

- `STATUS.md`: *"Nothing in this repository is PRODUCTION-READY."*
- `FEATURES.md`: 469 capabilities — 188 `TESTED`, 23 `IMPL`, 32 `SKEL`, 226 `PLAN`.
- `ROADMAP.md`: *"a very well-tested analysis library with a demo application on top of it, and the gap to a product is recording, a headless runtime, and having pointed it at something real."* Tier 0.4: *"CI … has never executed."*
- `docs/DEPLOYMENT.md`, `docs/DATABASE.md`, `docs/PROTOCOL.md`, `docs/CAMERAS.md`, `docs/AI.md`: each carries a banner saying it describes a design that no longer exists in code.

### What changed tonight (uncommitted, on `Phase2`)

Driven by the operator's two complaints — *"a lot of hallucinations"* and *"the
buttons do nothing"* — and the new standing rule that the packaged binary on the
real camera is the test medium.

| Change | Where | Proof |
|---|---|---|
| Confidence floor for classifiers: **0.50** default, per-machine, 0.10–0.95, in the Detection dialog, shown in the status line (`≥ 0.50`); motion ignores it | `app.py`, `watch_dialog.py`, `detect.py` | 9 console tests, 1 engine test |
| A greyed (locked) control answers a click: names itself, says the site is locked, offers to unlock and then does what was asked; Draw the same; the status bar reads MONITOR / CONFIGURE at all times | `app.py` (`eventFilter`, `_offer_unlock`, `lock_label`) | 8 console tests incl. the freeing test |
| Exceptions raised inside Qt slots are logged (redacted), named in the status bar, and released from `sys.last_*` | `app.py` (`_report_uncaught`) | 2 tests; PySide6 probe confirmed `sys.excepthook` is reached |
| The Configure tooltip and USAGE/FEATURES no longer claim Escape relocks (it never did) | `app.py`, docs | behavioural test |
| Console command line: `--camera --place --zone --zone-classes --watch --confidence --settings --start --for --screenshots`; timed runs print the node summary and per-track classes and photograph every panel | `app.py` (`build_parser`, `seed_site`, `end_after`, `photograph`, `report`) | 9 tests + a real process run (6 PNGs, exit 0) |
| `python tasks.py exetest` — runs `dist/SentinelVision/SentinelVision-dev.exe` on `device:0` in an isolated data directory, keeps pictures, stdout, stderr and the log, reads the summary back into a PASS/FAIL table; fails on any exception the console's hook logged or a run that outlives its `--for` | `tools/exe_camera_test.py`, `tasks.py` | runs recorded in §2.3 |
| **The actual "buttons do nothing" defect, found by the first packaged camera run:** `PlacementDialog`, `ZoneDialog` (both routes) and `AddCameraDialog` carried `WA_DeleteOnClose`, so `QDialog.done()` deleted them the instant OK was pressed, before `exec()` returned; the slot then read a deleted spin box or line edit and raised, Qt swallowed it, and Place…/Add zone…/Draw→name did nothing. The operator's audit trail has no `camera.placed` and no `zone.created` for exactly this reason. Fixed: read, then `deleteLater()`; a structural test forbids the attribute on any dialog read after `exec()` | `app.py` `_place_camera`, `_add_zone_dialog`, `_zone_drawn`, `_choose_source` | 5 tests that press OK the way Qt does (`accept()` plus the deferred delete); the packaged run's traceback at 09:03:54 |
| The timed run's report crashed printing `≥` to a cp1252 terminal and the window never closed; stdout/stderr now replace what they cannot encode and `_finish_timed_run` closes in a `finally`, dismissing any dialog left open | `app.py` `run()`, `_finish_timed_run` | 3 tests; the packaged run's traceback at 09:03:55 |
| A camera id is not a file name: `camera-device:0.png` became an NTFS alternate data stream on an empty `camera-device` file (five pictures of six) | `app.py` `_file_safe` | 1 test |
| **On `continue` (to-dos):** the console exports through the node with footage and preservation (REL-03); `sentinel.version` with a `build.json` stamp reaches About, `where`, the log, the report and the run notes (UI-01); the egress override is announced and logged and the docs stop denying it (SEC-03); `faulthandler` writes `crash.log` beside the log (OBS-03); **recording from the console**: a Record box per camera behind the lock, migration 11, `Node.set_recording`, the node records only asked cameras while `sentinel node --record` keeps recording all, retention sweeps from `poll` every ten minutes, recorder facts in the status strip, `--record` on the command line (REL-02) | `app.py`, `camera_list.py`, `store.py`, `node.py`, `cli.py`, `logs.py`, `decode.py`, `evidence.py`, `version.py`, `tasks.py`, docs | 33 tests across the engine and console suites; CI and a camera run recorded in §2.3 |
| Standing rule 8 (camera + shipped binary, never a prerecorded file) | `HANDOFF.md` | — |

### 2.3 Verification record for tonight's build

| Step | Result | Evidence |
|---|---|---|
| `python tasks.py ci --package` at HEAD `d38c64e` | **red** — 3 engine tests in `test_identity.py` (`migration_nine_is_the_newest…`, `migration_nine_arrives_and_leaves…`, `plate_reader_is_built_…`) fail against the `site_declared` migration 10 and `_SwitchedPlateReader` that arrived in that commit; reproduced in a clean worktree at HEAD; none of those symbols appear in this session's patches. Static guards, Rust, format and clippy green | `scratchpad/ci.log`, `ci-worktree.log` |
| `python tasks.py package` from HEAD + this session's fixes | built three times (09:02, 09:16, 09:24), models copied, stubs removed | `package*.log` |
| `python tasks.py exetest` run 1 (09:03, exe 09:02) | Pipeline real: 1,335 frames, 1,259 detections, one `person` at 0.86 held 28.2 s, **no couch, no bottle**, status line `watching bicycle, bus, car, motorcycle, person, truck with masks · ≥ 0.50`. The operator clicked Place… and Add zone… during the run: **both raised on a deleted dialog** (the defect since fixed); the report died printing `≥` to a cp1252 console; the camera picture went into an NTFS alternate stream. Tool verdict FAIL | `dist/exetest/20260905-090322/` — `console.png`, `stderr.txt` |
| exetest run 2 (09:17, exe 09:16) | Dialog fixes in; watch list active (`6 class name(s)`); a person tracked; the window was closed by hand at 12 s, before the timer, so no picture and no summary were left. Tool verdict FAIL — which led to `_conclude_timed_run` on close | `dist/exetest/20260905-091709/` |
| exetest run 3 (09:25, exe 09:24) | **PASS with a caveat**: 32.9 s wall for `--for 30`, 570 frames analysed, six pictures including `camera-device-0.png`, summary printed, no exception logged, closed itself; nobody was in front of the camera, so 0 detections | `dist/exetest/20260905-092516/` |
| Console suite after all fixes | 374 passed | offscreen, 2026-09-05 |
| `test_identity.py` brought up to date with migration 10 and the switched reader, plus 4 tests for what they promise | 40 passed | 2026-09-05, after `continue` |
| `python tasks.py ci --package` on the green tree | **green**, all 13 stages: source audit (after it caught an `rtsp://…` in a new docstring, reworded), binary audit, docs lint, rustfmt, clippy, `cargo test` (60), release build, engine (162.9 s), console (91.9 s), engine with the network poisoned (advisory, passed), package (265.7 s), `sentinel.exe where` and `coverage` launch checks | `scratchpad/ci3.log`, executables 10:05 |
| exetest run 4 (10:06, the CI-built exe) | **PASS with a caveat**: 34.3 s wall, 427 frames, six pictures, summary, no exception, closed itself; nobody in frame | `dist/exetest/20260905-100625/` |

**Read together:** run 1 proves detection, tracking, the watch list and the
floor on the shipped binary with a real person; runs 3 and 4 prove the timed
run, the pictures and the summary; the defects between them were found only
because a person was at the keyboard while the binary ran — which is the whole
argument for rule 8 in HANDOFF.md. Local CI is green on the working tree
(uncommitted). Remote CI has still never run (TEST-01), and no run with a
person in frame has yet been made on the final binary — the next `exetest`
with somebody in the chair is the first thing to do.

---

## 3. P0 — Absolute Production Blockers

The application must not be deployed to any site until every row is closed.

| ID | Task | Done when |
|---|---|---|
| SEC-01 | Authentication and permission-based authorization for every site-changing and evidence-touching action | Local accounts exist; every mutating `Node` method and CLI command checks a permission; audit rows carry the user; denied attempts are audited; tests cover allowed, denied and escalation |
| SEC-02 | Secret storage: camera credentials in the OS keychain; never on the command line; restarts do not lose them | `cameras.credentials_ref` populated by a keychain binding on Windows/macOS/Linux; a restored RTSP camera starts unattended; no argv path can carry a password without a warning; tests |
| REL-01 | Process supervision: service wrappers and restart-on-crash for the node and the console | Killed process restarts within 10 s on each platform; restart audited; a 24 h run with three induced crashes loses no evidence |
| REL-02 | ✓ **closed 2026-09-05** (toggle, flag, sweep; 12 tests) — the watermark *alert* remains as OBS-01 | A console toggle records; retention runs on a cadence without a human; a fake full disk produces an alert and never deletes preserved evidence; tests |
| OPS-01 | Signed installers per platform | MSI/`.deb`/AppImage/notarised `.dmg` produced by CI from a tag; a fresh VM installs, runs `exetest`, uninstalls cleanly |
| TEST-01 | Run CI, remotely, on all three platforms including the offline job | Green on GitHub; branch protection requires it; the package job runs on tags |
| DOC-01a | ✓ **closed 2026-09-05**: kept, announced at every start, logged with the address on every connection it allows, documented in USAGE, SECURITY and `.env.example`; 2 tests | Code and docs agree; a test pins the decision; if kept, use is logged at WARNING and audited |

---

## 4. P1 — Critical Pre-Launch Work

Must be resolved before production unless explicitly risk-accepted in writing.

| ID | Task |
|---|---|
| SEC-04 | Process-wide egress guard; connect to the address that was checked (DNS TOCTOU); refuse unresolvable names explicitly |
| SEC-05 | Decide onnxruntime telemetry: source build with `--no_telemetry` for shipped bundles, or documented risk acceptance with a Linux runtime egress proof |
| SEC-06 | Lock and hash-pin Python dependencies; `pip-audit` and `cargo audit` in CI; SBOM per release; wheelhouse procedure tested air-gapped |
| SEC-14 | Document that the Monitor/Configure lock is not a security boundary; make it require a credential once SEC-01 exists |
| REL-03 | ✓ **closed 2026-09-05**: the console exports through the node; the package says how many clips it carries; 1 test |
| REL-04 | Decode independent of analysis: a dead analytic must not stop recording (STATUS says it still can) |
| REL-05 | Move persistence and correlation off the GUI thread, or bound them; measure UI stalls |
| REL-06 | Process exit with a wedged decoder tested and bounded |
| REL-07 | 72-hour soak on the packaged exe on the camera; bound `_sighted` / `_plate_refused`; hourly metrics |
| TEST-02 | CI launches the packaged console with `--for --screenshots` on a file; `exetest` pictures attached to every release |
| TEST-03 | Offline RTSP server in CI (mediamtx or equivalent) exercising connect, redaction, reconnect, backoff, starvation |
| TEST-04 | Real-footage evaluation set from the laptop camera; recall/precision/false-track rate at the 0.50 floor; a "real footage" column in STATUS |
| PERF-01 | Load the ONNX model once per process, not four times per Start; capacity table per detector |
| UX-01 | First-run flow (add → start → place → zone) with an inline checklist; tested on a naive user |
| UX-02 | HighDPI verified with photographs at 125/150/200 % on the operator's monitor |
| DATA-01 / DR-01 | Backup and restore tooling, tested by restoring |
| DATA-02 | DATABASE.md vs code: migrations are not checksummed; `synchronous` is not set; PostgreSQL/`SqlDriver` do not exist |
| AI-01 | Measure the false-object rate after the watch list and floor (TEST-04) |
| AI-02 | Model licence: YOLOv8 weights are AGPL-3.0; legal review before any commercial deployment; permissive alternative evaluated |
| NET-01 | One physical IP camera for one hour without a reconnect storm (ROADMAP 0.3b) |
| UI-01 | ✓ **closed 2026-09-05**: `sentinel.version`, `build.json` stamped at package time, shown in About, `where`, the log's first line, the evidence report and `HOW TO RUN.txt`; 7 tests |
| BE-01 | A clean stop channel for `sentinel node` on Windows (signals do not reach it) |
| OPS-02 | Release process: semver tags, changelog, CI release job with checksums |
| OPS-03 | CI hygiene: scheduled runs, dependency update bot, `cargo audit`, package job on tags |
| OPS-04 | Offline update flow; schema-version gate at open so an older build refuses a newer database |
| OBS-01 | Alerting: dark camera, recording stopped early, disk watermark, stuck thread — something leaves the process |
| DOC-01 | Documentation truth pass (counts, deleted-design sections, unmarked SECURITY sections, DATABASE claims) |
| XP-01 | Linux and macOS console actually run with a camera and photographed |
| UX-12 | Every dialog's OK path driven by a test that presses OK the way Qt does; `WA_DeleteOnClose` forbidden on any dialog read after `exec()` |

---

## 5. P2 — High-Priority Work

| ID | Task |
|---|---|
| SEC-07 | Caps and fuzzing for operator-supplied files (ONNX, PNG basemaps, JSON, video containers) |
| SEC-08 | Data-at-rest posture: permissions on creation, disk-encryption guidance, evidence-dir integrity check on demand |
| SEC-09 | Audit chain over every row, `detail` inside the hash, head exported with evidence, `sentinel audit verify` |
| SEC-10 | Camera enumeration without PowerShell (locked-down hosts) surfaces a reason instead of "no cameras" |
| SEC-13 | Atomic evidence export (temp dir + rename); partial packages removed |
| REL-08 | Tracker output cap (256 per camera) logged/raised instead of silently truncated |
| REL-09 | Restart semantics: incidents open before a restart, incomplete segments, re-correlation from the store |
| REL-10 | Startup `integrity_check`; operator-readable message and recovery path for a corrupt database |
| REL-11 / DATA-06 | Versioned zone geometry so past events point at the shape that existed |
| REL-12 | Console dark-camera alert (banner, audit, optional sound) |
| REL-15 | Single-instance guard per data directory |
| TEST-05 | Native-platform screenshot job with baseline diffs (offscreen renders no text) |
| TEST-06 | Stress test: N file cameras with the segmenter; drop rate bound; capacity documented |
| TEST-07 | Failure injection: disk full, DB locked, camera unplugged, model deleted mid-run, kill -9 then restart |
| TEST-08 | Migration fixtures v1…v9, forward and back |
| TEST-12 / TEST-13 | Split the 3,900-line console test file; measure coverage with a floor |
| PERF-02 | Stop rebuilding three widgets 30× a second; diff updates; measured at 16 cameras |
| PERF-03 | Incremental correlation instead of re-running over the whole window every 1.5 s |
| PERF-04 | Decide the accelerator story (DirectML/CUDA providers) and target hardware |
| PERF-08 | Alert when analysed fps falls below half the camera's rate |
| UX-03 | Undo for zone reshape/kind/name from the audit's before/after |
| UX-04 | Camera position entry in site metres / DMS / UTM (FEATURES PLAN) |
| UX-05 | Incident acknowledge / resolve / notes; the list must not grow forever |
| UX-06 | Faults and dark cameras announced on the wall, not only in a list |
| UX-10 | Packaged non-dev build shows a "something went wrong — log at …" dialog on a critical error |
| DATA-03 | Retention for events, incidents, audit, plate reads; evidence-linked rows protected |
| DATA-04 | Rollback strategy while the app is open; version gate |
| DATA-08 | Concurrent writer tests: retention sweep and CLI during a live console run |
| AI-03 | Per-class floors and a minimum box size |
| AI-04 | Appearance in the tracker (ABI 7); fragmentation ≤ 1.2 on the exetest scene |
| AI-05 | Inference watchdog: a hung provider must not freeze a camera silently |
| AI-08 | DPIA, scheduled register retention, subject-access export |
| NET-02 / NET-03 | ONVIF discovery; state plainly that there is no TLS and no control plane |
| UI-02 | Chase the `QFont::setPointSize` warning with `QT_FATAL_WARNINGS=1` |
| UI-04 | Wall layouts, pinning, fullscreen |
| BE-02 / BE-03 | Node API contract; split `app.py`, `node.py`, `map_view.py` before the control plane |
| OPS-05 | `sentinel health` with exit codes for external monitors |
| OPS-06 | One configuration story (today: registry, env vars, flags, database) |
| OPS-08 / DOC-03 | Runbooks: camera dark, disk full, DB locked, restore, chain break |
| OBS-02 | Structured logs with run/camera ids; per-camera counters exported |
| OBS-03 | ✓ **closed 2026-09-05** for the `faulthandler` half: `crash.log` beside the log, 2 tests. Minidumps remain |
| DOC-02 | Operator manual for the packaged product |
| DOC-06 | Consolidated known-limitations page |
| DEBT-01 | One preprocessing module for both ONNX detectors |
| DEBT-02 | Initialise `_zone_warnings` / `_zone_reports` in `__init__` instead of `getattr` defaults |
| DEBT-05 | Read class names without loading a full inference session |
| XP-02 / XP-03 | Windows backend matrix on more than one laptop; macOS permission and notarisation |
| DR-02 / DR-03 | Disk-full and power-loss drills |

---

## 6. P3 — Medium-Priority Work

| ID | Task |
|---|---|
| SEC-11 | Model manifest with expected digests; refuse or flag a mismatch |
| SEC-12 | Per-section status markers in SECURITY.md; product-grade vulnerability policy |
| SEC-15 | Log classification: DEBUG never logs full zone rings or positions |
| REL-13 | Schedule tests across DST transitions in the site zone |
| REL-14 | Degradation matrix, each failure injected and documented |
| TEST-09 | DPI, keyboard-only and accessibility test passes |
| TEST-10 | More concurrent-Store tests |
| TEST-11 | Install / uninstall / upgrade on fresh VMs |
| PERF-05 / PERF-06 / PERF-07 | Startup time target; memory per camera bound; H.264 instead of `mp4v` |
| UX-07 / UX-08 / UX-09 / UX-11 | Accessibility, localization readiness, multi-monitor, the two Add-zone buttons |
| DATA-05 / DATA-07 | Cascade verification; site config import/export |
| AI-06 / AI-07 | Model registry with version and licence; ONNX determinism across runtime versions |
| NET-04 | IPv6 camera test |
| UI-05 / UI-07 | Remember geometry and column widths; shortcuts for Start/Stop/Configure |
| BE-04 / BE-05 / DEBT-03 / DEBT-04 / DEBT-06 / DEBT-07 | Public CLI parsers; dead TypeScript-era directories; private import in `session.py`; `.env.example` trim; test-file split |
| OPS-07 | Container image scanning |
| OBS-04 / OBS-05 | Automate chain-head export; inference latency in the UI |
| DOC-04 / DOC-05 | ADRs, API reference, contribution and security policy |
| XP-04 | Non-ASCII data directories; long paths; cp1252 |
| DR-04 | Evidence-tampering response procedure |
| UX-13 | The status bar's transient message is clipped by its permanent labels |

---

## 7. P4 — Low-Priority / Post-Launch Work

| ID | Task |
|---|---|
| AI-09 | The grounded analyst, last, with the validator AI.md specifies and no tool calls |
| FEATURES "the ten" | Mission timeline, coverage map, track replay, why-alert, what-changed, heatmaps, incident graph, digital twin, copilot |
| ROADMAP Tier 5 | Occlusion, terrain and lens distortion, fuzzing beyond the hostile paths already listed |
| UX-14 / UX-15 | Zone label on the camera marker at default fit; dev-exe stdout is cp1252 |

---

## 8. Security Audit

Format for every item from here on: **ID · Priority · Task** — *Why* · *Do* ·
*Done when* · *Depends / Test / Risk*.

**SEC-01 · P0 · No authentication or authorization exists.**
*Why:* every audit row says `console`; the chain of custody cannot name a
person; anyone at the keyboard can delete a camera, export evidence or change a
zone. SECURITY.md's "Authorization" section describes roles that do not exist
and carries no status marker. *Do:* local accounts (argon2id), permission
checks inside `Node` (not only in the UI), a session with a lock timeout, an
application lock, audit rows carrying the user id; CLI commands authenticate
too. *Done when:* `test_node.py` proves every mutating method refuses without a
permission; the evidence report names the exporting user; the lock screen exists.
*Depends:* nothing. *Test:* allowed/denied/escalation, expired session, brute
force lockout. *Risk:* large; touches every path.

**SEC-02 · P0 · Camera credentials are never persisted and travel on the command line.**
*Why:* `Node.needs_credentials` makes a restored RTSP camera refuse to start
until a person retypes the password — an unattended restart at 03:00 leaves the
site unwatched. `sentinel run rtsp://user:pw@…` and the new console
`--camera rtsp://user:pw@…` put the password in process listings and shell
history (the console redacts what it *logs*, not what the OS shows). *Do:*
keychain bindings (Windows Credential Manager, macOS Keychain, Secret Service)
behind `credentials_ref`; prompt or read from an environment variable/file
for the CLI; warn when a credential is seen in argv. *Done when:* a restored
RTSP camera starts unattended; a test asserts argv-borne credentials are
refused or warned; the schema-walk test still finds no credential column.
*Risk:* keychain APIs differ per platform; a wrong abstraction leaks.

**SEC-03 / DOC-01a · P0 · `SENTINEL_ALLOW_PUBLIC_SOURCES` contradicts the documentation.**
*Why:* `decode.py:83` defines an override and `_require_private` tells the
operator to set it; USAGE §11/§12 say *"There is no override, and there will
not be one"*; SECURITY.md says refused, address named. A promise the code does
not keep is worse than no promise. *Do:* decide. If removed: delete the
variable and the sentence in the error. If kept: log at WARNING on every
start, audit its use, document it in SECURITY.md and USAGE. *Done when:* a test
pins whichever it is.

**SEC-04 · P1 · The egress guard is decode-only and has a resolve-then-connect gap.**
*Why:* `_require_private` resolves the host, checks the addresses, returns —
and `_require_reachable` and OpenCV then resolve *again*; a DNS answer that
changes between the two (rebinding) reaches a public address. An unresolvable
name is passed through ("left to the connect below"). Every future outbound path
(control plane, LLM endpoint) must route through one guard. *Do:* a single
`EgressGuard` returning the vetted address; connect by IP; refuse
unresolvable names with the reason. *Done when:* a test with a resolver stub
that changes its answer fails the connect; the guard is the only place a socket
is opened (asserted by the offline audit).

**SEC-05 · P1 · onnxruntime's telemetry uploader ships in the Linux/macOS wheels.**
*Why:* SECURITY.md documents it honestly: disarmed by environment variable and
API, but the uploader, its TLS stack and device-id database remain in the
binary; the offline CI job proves only the paths the tests reach. *Do:* either
build onnxruntime from source with `--no_telemetry` for shipped bundles, or
record a risk acceptance and add a Linux job that runs the packaged analyser
with `strace`/`ss` proving no socket is opened. *Done when:* the decision is in
SECURITY.md with the proof beside it.

**SEC-06 · P1 · No dependency lock, no hash pinning, no vulnerability scanning, no SBOM.**
*Why:* `numpy>=2.0`, `onnxruntime>=1.20`, `PySide6>=6.7` resolve to whatever
is current; two installs a month apart are different software (SECURITY.md says
so). Development is on Python 3.14 while CI targets 3.12. *Do:* `requirements.lock`
with `--require-hashes`, `pip-audit` and `cargo audit` gates, CycloneDX SBOM in
the release job, the air-gapped wheelhouse procedure exercised in CI. *Done
when:* CI fails on a known CVE; the bundle's SBOM matches its contents.

**SEC-07 · P2 · Operator-supplied files are parsed with no limits.**
*Why:* ONNX models (a large native parser), basemap PNGs through
`cv2.imdecode`, basemap/orthophoto JSON, evidence manifests, arbitrary video
containers through FFmpeg. "Hostile camera" is in the threat model and ROADMAP
5.4 says nothing has been fuzzed. *Do:* size and dimension caps before decode;
a fuzz corpus (malformed MP4, PNG bombs, deep JSON) run in CI on the parsers;
consider decoding in a subprocess. *Done when:* each parser rejects the corpus
within bounded memory and time.

**SEC-08 · P2 · Data at rest.** SQLite, logs, recordings and evidence live
under `%LOCALAPPDATA%` with default permissions and no encryption. *Do:* create
directories with restrictive ACLs/0700, document full-disk encryption as a
commissioning requirement, add an on-demand integrity check of the evidence
directory. *Done when:* a second local user cannot read the database; the
check exists as `sentinel verify`.

**SEC-09 · P2 · The audit chain is partial.** Only rows with before/after
carry a hash; `detail` is outside the hash by design; the head must be copied
out by hand. *Do:* chain every row, include `detail`, write the head into every
evidence package and an optional append-only file, add `sentinel audit verify`.
*Done when:* the Audit tab's verify reports full coverage; a modified prose row
breaks the chain in a test.

**SEC-10 · P2 · Camera enumeration shells out to PowerShell.** On a host with
AppLocker or Constrained Language Mode it fails to "no cameras". *Do:* fall
back to a WMI/COM query or DirectShow enumeration; surface the failure reason.
*Done when:* a test with PowerShell absent still lists or explains.

**SEC-11 · P3 · Model integrity is recorded, never verified.** *Do:* a
`models/manifest.json` with expected digests; refuse or flag a mismatch;
audit a model change. **SEC-12 · P3** per-section states in SECURITY.md; a
vulnerability policy for a deployed product rather than "a portfolio".
**SEC-13 · P2** atomic evidence export (write to a temporary directory, rename;
remove partial packages; test with an injected copy failure).
**SEC-14 · P1** the Configure lock is a guard against stray clicks, not access
control; the new command-line flags bypass it by design — document, and make it
credential-backed once SEC-01 lands. **SEC-15 · P3** DEBUG logs may carry
zone rings and positions (site layout is sensitive per the threat model);
classify and redact.

**Explicitly verified and not a finding:** SQL is parameterised everywhere
(`search.py` escapes `%`/`_`); evidence export resolves paths inside the root;
no webview; no network handler in logging; the redaction filter covers message,
args and traceback; `panic = "abort"` and pointer checks at the C ABI.

---

## 9. Reliability & Failure Recovery

**REL-01 · P0 · Nothing restarts.** No Windows service, no systemd unit, no
launchd plist, no watchdog; the console is a desktop process; `sentinel node`
exits on a crash. *Do:* service wrappers per platform (NSSM or a native
service host, systemd `Restart=always`, launchd `KeepAlive`), a heartbeat file
the wrapper watches, a restart audit row. *Done when:* `taskkill /F` on the
node during an exetest run is followed by a restart within 10 s with the same
cameras and no lost evidence; repeated three times in a 24 h run.

**REL-02 · P0 · Recording cannot be switched on from the console and retention is manual.**
*Why:* FEATURES marks the console toggle `PLAN`; `apply_retention` is reached
only by `sentinel retention`; its own shortfall message reads *"Recording will
continue until the disk is full and then stop."* *Do:* per-camera recording
checkbox → `Node.add_camera/start` (as the CLI does); a retention cadence
inside `Node.run_forever`/`poll`; a disk watermark that alerts (OBS-01). *Done
when:* a console run records segments, a fake-full disk triggers retention and
an alert, preserved evidence is untouched (existing test extended to the node
loop).

**REL-03 · P1 · The console's export bypasses the node.** `app.py`
`_export_incident` calls `sentinel.evidence.export_incident` directly; the
node's `export_incident` docstring says *"The console's own export had neither
half — no `footage=`, no `preserve_segments`."* It still has neither. *Done
when:* the console calls `self.node.export_incident`; a test exports from the
console and finds `footage.json` and a preserved segment.

**REL-04 · P1 · Decode and analysis share one loop** (STATUS: "survives a slow
analytic but not a dead decode"). *Do:* fan the decode thread out to a
recorder queue and an analytic queue. *Done when:* an analytic that raises or
blocks does not stop segments being written; tested.

**REL-05 · P1 · Persistence and correlation run on the GUI thread.** `_collect`
→ `node.poll()` → `store.save_events`, `save_incident`, `correlate` every 33 ms.
*Done when:* a 200 ms disk stall injected into the store does not freeze
painting; or the work is moved to a worker with a bounded queue.

**REL-06 · P1 · A runner that will not stop is "left running rather than killed".**
`closeEvent` calls `node.close()` and returns; a decoder wedged in a native
read keeps the process alive or crashes it at interpreter shutdown. *Done
when:* an injected blocking `read()` still lets the process exit within 15 s
(daemon thread + audited abandonment, or a hard exit path).

**REL-07 · P1 · No soak has ever run.** `Node._sighted`, `_plate_refused` grow
per (subject, camera, track, run) with no bound; `keep_images=True` holds a full
frame per camera per poll; face dicts are cleared per run. *Done when:* a
72-hour exetest-style run on the camera with hourly RSS, handle and thread
counts shows a flat line; the sets are bounded.

**REL-08 · P2 · 256-track output cap silently truncates** (`core.py`
`_MAX_TRACKS`, `ffi.rs` `min(len, capacity)`). *Done when:* the binding raises
or logs once and the cap is measured against a crowd scene.

**REL-09 · P2 · Restart semantics are undefined.** On restart `record.events`
is empty, so an incident open before the restart is never extended; an
incomplete segment is flagged `complete=0` but never checked for playability.
*Done when:* documented and tested: kill during an incident → restart → the
incident is re-correlated from the store or a new one opens with a link.

**REL-10 · P2 · Corrupt database at startup** produces a traceback. *Done
when:* `PRAGMA integrity_check` on open (or on a schedule), an operator message
naming the backup to restore (DATA-01).

**REL-11 · P2 · Zone geometry is overwritten in place** (HANDOFF's "honest
limit"): events raised before a reshape point at a shape that no longer exists.
*Done when:* versioned geometry (ROADMAP slice 5).

**REL-12 · P2 · A starved or dark camera in the console is a word in a list.**
The CLI exits 1 on starvation; the console does not alert. *Done when:* a
banner, an audit row and an optional sound after `DARK_AFTER_SECONDS`.

**REL-13 · P3** DST transitions for schedules in the site zone. **REL-14 · P3**
a degradation matrix (model file removed mid-run, database read-only, log
directory unwritable, camera unplugged). **REL-15 · P2** two consoles on one
data directory both open `device:0` — the OS shares the webcam and both starve
(HANDOFF phase C); add a per-data-directory instance lock.

---

## 10. Testing & QA

**TEST-01 · P0 · CI has never executed.** `.github/workflows/ci.yml` targets
Ubuntu/Windows/macOS and an `iptables` offline job; ROADMAP 0.4 says it has
never run. *Done when:* every job is green remotely; branch protection.

**TEST-02 · P1 · The packaged binary is only launched by `where` and `coverage`.**
*Do:* CI runs `SentinelVision --settings … --camera <reference.mp4> --place …
--start --for 5 --screenshots …` offscreen and asserts exit 0 and six PNGs;
before a release a person runs `python tasks.py exetest` and attaches the
pictures. *Done when:* both are in the release checklist and the CI log.

**TEST-03 · P1 · RTSP is untested.** *Do:* a container job with an RTSP
server (mediamtx) publishing the reference file on the LAN address space;
tests for connect, credential redaction end to end, reconnect with backoff,
resolution change mid-stream, starvation. *Done when:* `decode.py`'s RTSP path
rises from `SKEL` to `TESTED` with a real server behind it.

**TEST-04 · P1 · No real-footage evaluation.** *Do:* label frames from
`dist/exetest/*/` runs (person present / absent / seated / partial), compute
recall, precision and false tracks per hour at the default floor and watch
list, keep thresholds floored in a test that reads the labelled set. *Done
when:* STATUS carries a real-footage column.

**TEST-05 · P2 · Offscreen renders no text** (the screenshot tool's own
docstring: zero font families). *Do:* a Windows-runner job on the native
platform with baseline images and a perceptual diff. **TEST-06 · P2** stress:
N file cameras with the segmenter, drop rate bound, capacity documented per
detector. **TEST-07 · P2** failure injection scripts (disk full, DB locked,
camera unplugged, model deleted, kill -9 + restart). **TEST-08 · P2** schema
fixtures v1…v9 with forward and backward migration tests. **TEST-09 · P3**
DPI / keyboard / accessibility. **TEST-10 · P3** more concurrent-store tests.
**TEST-11 · P3** install/uninstall/upgrade on fresh VMs. **TEST-12 · P2** split
`test_console.py` (3,900+ lines; a helper-name collision cost time tonight).
**TEST-13 · P2** coverage measurement with a floor (`pytest-cov`, `cargo
llvm-cov`).

**Verified tonight:** the suites do prove what they claim for the paths they
cover; the freeing test catches reference cycles; the structural lock test
catches a new mutator. **Not proven by any test:** anything on a real IP
camera; anything over an hour; anything a human sees rendered.

---

## 11. Performance

| ID | P | Finding | Acceptance |
|---|---|---|---|
| PERF-01 | P1 | The segmenter runs 11–17 fps per camera on CPU; the 16-camera figure is motion-only. The console loads the ONNX session **four times** per Start for one camera (`segmenter ready` ×4 in the log: `_detector_labels`, `_vocabulary`, `_start`'s check, the factory) | One load per process for name reads; a capacity table per detector per machine class in STATUS; Start < 3 s with a model |
| PERF-02 | P2 | Track table, camera list and incident tree are rebuilt every 33 ms | Diff updates; 16 cameras × 20 tracks stays under 30 % of one core for the UI |
| PERF-03 | P2 | Correlation re-runs over up to 5,000 events per camera every 1.5 s | Incremental correlation; measured at 16 cameras × 5,000 events |
| PERF-04 | P2 | CPU only; providers `PLAN` | A decision on DirectML/CUDA and the reference hardware |
| PERF-05 | P3 | Packaged startup time unmeasured | < 5 s to first frame on the reference laptop |
| PERF-06 | P3 | Full frame per update per camera with `keep_images` | Bound and measured |
| PERF-07 | P3 | `mp4v` at ~17.5 GB/camera/day | H.264 evaluated; a decision recorded |
| PERF-08 | P2 | Frame drops are counted, not alarmed | Alert when analysed fps < ½ camera fps for 30 s |

---

## 12. UX & Accessibility

**UX-01 · P1 · First run.** A new operator sees an empty wall and, since
tonight, a status bar that says the site is locked and buttons that explain
themselves — but nothing tells them the order (add → start → place → zone) or
that a zone needs a placed camera until they hit the message box. *Done when:*
an inline checklist in the empty wall and a guided first run; a naive-user
session recorded.

**UX-02 · P1 · HighDPI.** No `highDpiScaleFactorRoundingPolicy` is set; the
2560 px screenshot showed clipped buttons before the two-row toolbar. *Done
when:* exetest pictures at the operator's scale (and 125/150/200 %) show no
clipping; a test resizes at each scale.

**UX-03 · P2** no undo beyond drag-revert. **UX-04 · P2** placement asks for
six-decimal coordinates operators do not have (site metres / DMS / UTM entry).
**UX-05 · P2** incidents have no acknowledge/resolve/notes and accumulate
forever. **UX-06 · P2** faults are a list entry, never a wall-level alarm.
**UX-07 · P3** accessibility: no accessible names, unaudited contrast on
`TEXT_FAINT` captions, no focus-order or screen-reader pass. **UX-08 · P3**
strings inline; Arabic/French/RTL `PLAN`. **UX-09 · P3** multi-monitor,
fullscreen, tray `PLAN`. **UX-10 · P2** the non-dev packaged build has no
terminal: a critical error now reaches the log and status bar; it should also
raise a dialog naming the log path. **UX-11 · P3** two "Add zone…" buttons
(toolbar and tab) are intentional; say so or unify.

**Fixed tonight and verified by test:** greyed controls answer a click with an
offer to unlock; Draw does the same; the status bar always states the mode;
the tooltip no longer promises that Escape relocks; **and the OK buttons of
the placement, zone and add-camera dialogs work** — until tonight the first
two raised on a deleted widget and did nothing, which is the complaint as the
operator actually experienced it (see §2).

**UX-13 · P3 · The status bar clips its own message.** In the third camera
run's picture the transient line reads `0 tracked no` before the four permanent
labels (lock state, placement, detector, ground point) take the rest of a
2,000-px window. *Done when:* the permanent labels elide from the right, or
the detector line moves to a tooltip on a short label, and a test holds the
transient message fully visible at 1280 px. **UX-14 · P4** the zone label
`Room · restricted` sits on the camera marker at the default fit; offset labels
away from markers. **UX-15 · P4** the dev executable's stdout is cp1252, so
`≥` prints as `?` in `stdout.txt`; emit UTF-8 once the terminal story is decided.

**UX-12 · P1 · Every dialog's OK path must be exercised by a test that presses
OK the way Qt does.** *Why:* four dialogs were read after `exec()` with
`WA_DeleteOnClose` set; every test called the slot beneath the dialog, so
the suite was green while the two most important buttons on the screen did
nothing. *Done when:* `_press_ok` (accept plus the deferred delete) drives
every dialog in the console from its real entry point; a structural test
forbids `WA_DeleteOnClose` on a dialog read after `exec()` (added tonight);
the same is applied to the register, audit and investigation panels' dialogs.

---

## 13. Data & Database

**DATA-01 · P1 · No backup, no restore.** Copying `sentinel.db` while WAL is
active is unsafe. *Do:* `sentinel backup DIR` using `VACUUM INTO` or the
backup API plus the evidence directory and its manifest; `sentinel restore`
that validates and refuses to overwrite a live database. *Done when:* a
restore from backup on a fresh machine reproduces incidents and passes
`verify_export` on every package.

**DATA-02 · P1 · DATABASE.md disagrees with `store.py`.** It claims migrations
are checksummed (the `schema_migrations` table has `version, name, applied_at`
— no checksum), `synchronous = NORMAL` (no pragma is set; SQLite's default under
WAL is FULL), PostgreSQL behind a `SqlDriver` (deleted with the prototype),
`incident_notes`/`ai_inferences`/`users` tables (absent). *Done when:* each
claim is built or removed; a test asserts the pragmas the document states.

**DATA-03 · P2** no retention for `events`, `incidents`, `audit_logs`,
`plate_reads` (only recordings); the database grows without bound. **DATA-04
· P2** `db-rollback` while the console is open; a newer database opened by an
older build must be refused with a message. **DATA-05 · P3** verify
`incident_events` cascade when `save_incident` deletes a superseded incident.
**DATA-06 · P2** = REL-11. **DATA-07 · P3** site configuration import/export.
**DATA-08 · P2** concurrent writers: a retention sweep and CLI reads during a
live console run, under load.

**Verified:** indexes exist on events (time, camera, zone), incidents, audit,
recordings, plate reads; foreign keys are enabled; migrations run statement by
statement inside a transaction; deterministic ids make re-correlation an
upsert; no column holds a credential (schema-walk test).

---

## 14. AI/ML/LLM Reliability & Safety

**AI-01 · P1 · The operator's "hallucinations" are unmeasured after the fix.**
The evidence: the model saw a couch at 0.39 and a jar at 0.43–0.51 with all 80
classes watched at a 0.35 floor; the person held 0.86. The watch list (person +
five vehicle classes) and the 0.50 floor remove those cases by construction,
but no number says what remains: a coat as `person` at 0.55, a reflection, a
poster. *Done when:* TEST-04's set reports false tracks per hour and recall at
the defaults, and the defaults are chosen from that table rather than from one
evening.

**AI-02 · P1 · Licence.** `devtools/export_model.py` exports YOLOv8n-seg from
Ultralytics, whose weights and code are AGPL-3.0. Shipping them inside a
proprietary appliance is a legal question, not a technical one. *Done when:* a
licence review is recorded per AI.md's registry rule and a permissively
licensed model has been evaluated as the default.

**AI-03 · P2** per-class floors, minimum box size, aspect sanity. **AI-04 · P2**
appearance in the tracker (1.3–2.7 tracks per moving object measured). **AI-05
· P2** an inference watchdog — a hung provider blocks the decode thread and the
camera looks dark with no reason. **AI-06 · P3** a model registry with version
and licence; About shows it. **AI-07 · P3** ONNX determinism across onnxruntime
versions. **AI-08 · P2** faces/plates: DPIA, scheduled register retention,
subject-access export; the register's lawful-basis field is free text. **AI-09
· P4** the analyst, last, exactly as AI.md specifies: validator, citations,
local only, no tool calls, prompt versioning; treat model output as untrusted
input.

**Not a finding:** motion emits `UNCLASSIFIED` and the console will not label
it; class names come from the model's own metadata; the model digest is on
every event; nothing is ever downloaded — all asserted by tests.

---

## 15. Networking & APIs

**NET-01 · P1** one physical IP camera, one hour, no reconnect storm (ROADMAP
0.3b); OpenCV's hard-coded 30 s connect timeout is mitigated by the socket
probe, the read timeout is `LIVE_FRAME_TIMEOUT_SECONDS = 10`, backoff is
1→30 s jittered — all unverified against hardware. **NET-02 · P2** ONVIF/mDNS
discovery `PLAN`; RTSP URLs are typed by hand. **NET-03 · P2** there is no
TLS, no control plane, no multi-machine mode; DEPLOYMENT.md's "LAN distributed"
section describes a design that does not exist and must say so in every
subsection. **NET-04 · P3** IPv6 camera test. **There is no API** today —
nothing listens, by design — so the REST/WebSocket items in PROTOCOL.md are
entirely `PLAN` and gate on SEC-04 (the hoisted egress guard) and SEC-01.

---

## 16. Frontend / Desktop / Mobile

**UI-01 · P1 · No version anywhere on screen.** `APPLICATION_VERSION = "0.1.0"`
lives in `evidence.py`; the About dialog has no version, commit or build date;
`sentinel where` prints none; `HOW TO RUN.txt` none. A bug report cannot say
which build. *Done when:* one `sentinel.__version__` derived at build time
(version + commit + date) appears in About, `where`, the log header, the
evidence report and the run notes.

**UI-02 · P2** `QFont::setPointSize: Point size <= 0 (-1)` at startup
(HANDOFF), unchased. **UI-03 · P2** = UX-02. **UI-04 · P2** wall layouts.
**UI-05 · P3** window geometry and column widths not remembered. **UI-06 · P1**
= REL-05. **UI-07 · P3** shortcuts for Start/Stop/Configure.

**Mobile:** none, and none planned; FEATURES lists operator notifications as
`PLAN`.

---

## 17. Backend & Services

**BE-01 · P1** `sentinel node` cannot be stopped cleanly on Windows (signals do
not reach it; documented) — add a stop channel (named pipe or a flag file the
loop watches) before any service wrapper. **BE-02 · P2** the node's methods are
the de facto API; define the contract before the control plane. **BE-03 · P2**
`app.py` (≈2,500 lines), `node.py` (≈2,600), `map_view.py` (≈2,500) are the god
objects ROADMAP §4 already worried about; split by responsibility. **BE-04 ·
P3** the console now imports `sentinel.cli._pose/_zone/_zone_classes/_apply_zone_classes`
— promote them to public names with their own tests. **BE-05 · P3** delete or
quarantine the TypeScript-era leftovers: `apps/desktop/`, `packages/shared-types`,
`packages/test-utils`, `services/inference`, `services/recorder`, `infrastructure/`.

---

## 18. Deployment / DevOps / CI/CD

**OPS-01 · P0** signed installers (see §3). **OPS-02 · P1** release process:
tags, changelog, checksums, SBOM, reproducibility statement (PyInstaller output
is not byte-reproducible; record what is). **OPS-03 · P1** CI never run,
package job only on push to `main`, no scheduled run, no dependency bot, no
`cargo audit`, no image scan. **OPS-04 · P1** offline update flow and a schema
version gate. **OPS-05 · P2** `sentinel health` with exit codes (nothing may
listen, by design, so monitoring must poll a command or a file). **OPS-06 ·
P2** configuration lives in four places — registry (`QSettings`), environment
(`SENTINEL_*`), flags, database — define precedence and one document.
**OPS-07 · P3** container scanning. **OPS-08 · P2** runbooks.

**Verified:** `onedir`, no UPX, models copied beside the executables, the
bundle's stubs removed, the launch check runs `where` and `coverage`; the
container runs as a non-root user with `network_mode: none`.

---

## 19. Observability

**OBS-01 · P1 · Nothing alerts.** A dark camera, a recording that stopped
early (`RECORDING STOPPED EARLY` at ERROR), a retention shortfall, a stuck
thread — all reach the log and, sometimes, a label. Nobody is watching the
cameras by assumption; nobody is watching the log either. *Done when:* an
alert sink exists (desktop notification, a local webhook to a LAN endpoint
through the egress guard, an audible alarm) and each of those four conditions
fires it in a test.

**OBS-02 · P2** structured logs (JSON option) with run and camera ids; per-camera
counters written to a metrics file. **OBS-03 · P2** `faulthandler.enable()` to
the log and Windows minidumps — the exit-time heap corruption of HANDOFF §2 left
no trace beyond an exit code. **OBS-04 · P3** automate the chain-head export.
**OBS-05 · P3** inference latency per camera in the UI.

**Verified:** rotating log capped at 30 MB; every record redacted; faults name
the exception type, never its text; since tonight slot exceptions are logged.

---

## 20. Documentation

**DOC-01 · P1 · Truth pass.** Specific discrepancies found: README says
648 tests (≈1,520 now); STATUS counts drift (corrected once tonight by a
concurrent editor); DEPLOYMENT/DATABASE/PROTOCOL/CAMERAS describe the deleted
design under a top banner while individual sections read as built;
SECURITY.md "Authorization", "Audit" and parts of "Privacy" have no state;
USAGE §11/§12 vs `SENTINEL_ALLOW_PUBLIC_SOURCES`; DATABASE.md checksums and
`synchronous`; TESTING.md counts (140). *Done when:* every claim maps to code or
is labelled `PLAN`, and a lint checks that every H2 in `docs/` carries a state.

**DOC-02 · P2** an operator manual for the packaged product. **DOC-03 · P2**
runbooks. **DOC-04 · P3** ADRs (C ABI over PyO3, `mp4v`, no PyInstaller
onefile) and a Node API reference. **DOC-05 · P3** contribution and security
policy for a product. **DOC-06 · P2** a consolidated known-limitations page.

---

## 21. Technical Debt (classified by production risk)

| ID | P | Debt | Risk if shipped |
|---|---|---|---|
| DEBT-01 | P2 | Letterbox/NMS duplicated between `OnnxDetector` and `Segmenter` (acknowledged in source) | Half-pixel drift shows up as geometry error |
| DEBT-02 | P2 | `getattr(self, "_zone_warnings", {})` / `_zone_reports` set lazily | An unexpected code path shows a zone without warnings |
| DEBT-03 | P3 | `session.py` imports private `_looks_live`; `is_live_source` is public | Breaks on refactor |
| DEBT-04 | P3 | `.env.example` advertises 12 unread variables (labelled as such) | Operators set them and believe them |
| DEBT-05 | P2 | The ONNX model is loaded to read class names, four times per Start | Slow starts; four sessions' memory briefly |
| DEBT-06 | P3 | One 3,900-line console test file | Collisions, slow triage |
| DEBT-07 | P3 | Dead TypeScript-era directories | Confuses audits and packaging |
| DEBT-08 | P2 | The console runs the node on the GUI thread (= REL-05) | UI stalls under disk pressure |
| DEBT-09 | P3 | Two `Add zone…` buttons; `_add_zone` still places squares while drawing exists | Confusion, not failure |
| DEBT-10 | P2 | Track/incident tables rebuilt at 30 Hz (= PERF-02) | CPU at scale |

Not debt: the C ABI instead of PyO3, `panic = "abort"`, Python over a
Makefile — all argued and measured in source.

---

## 22. Cross-Platform Verification

| Platform | Console | Camera capture | Package | Signed | CI ran | Status |
|---|---|---|---|---|---|---|
| Windows 11 (dev laptop) | run, photographed | MSMF → DirectShow fallback, one laptop | built tonight | no | never | the only platform anything has been seen on |
| Linux | never shown | V4L2 enumeration `IMPL`, never run | never | no | never | container runs the engine suite only |
| macOS | never built | AVFoundation `IMPL`, never run | never | no (Gatekeeper refuses unsigned) | never | unknown |

**XP-01 · P1** run the console with a camera on Linux and macOS and photograph
it; run CI. **XP-02 · P2** more than one Windows machine (a Hello IR sensor
and MSMF refusals were both surprises). **XP-03 · P2** macOS camera
permission, notarisation. **XP-04 · P3** non-ASCII data directories, Windows
long paths in evidence exports, cp1252 consoles (partly handled).

---

## 23. Disaster Recovery

**DR-01 · P1** backup and restore exist and have been exercised (= DATA-01).
**DR-02 · P2** disk-full drill: fill the volume during a recording run;
recording must stop with an alert, the database must keep working or fail
loudly, the console must keep painting. **DR-03 · P2** power-loss drill: kill
-9 mid-segment and mid-transaction, reopen, verify WAL recovery, an incomplete
segment flagged and a complete manifest. **DR-04 · P3** an evidence-tampering
response: how to verify the chain against an out-of-band head, who is told,
what is preserved.

---

## 24. 🚨 PRODUCTION RELEASE GATE

Sentinel Vision may be called production-ready only when every box is ticked
with a link to its evidence.

- [ ] All P0 items (§3) closed with tests: SEC-01, SEC-02, SEC-03/DOC-01a, REL-01, REL-02, OPS-01, TEST-01.
- [ ] All P1 items (§4) closed, or each risk-accepted in writing by the product owner with an expiry date.
- [ ] Core functionality covered by automated tests that have run **remotely** on all three platforms, including the offline job — and the run is linked.
- [ ] A security audit by someone other than the authors, against §8, with findings closed.
- [ ] Data integrity verified: migrations v1→current forward and back on fixtures; `integrity_check` clean on a soak database.
- [ ] Failure and recovery tested: kill -9, disk full, camera unplugged, wedged decoder, DB locked — each with a recorded outcome.
- [ ] Production builds reproducible to the extent documented; SBOM and checksums published with the release.
- [ ] Deployment tested on a fresh machine per platform from the signed installer; uninstall leaves only the data directory, and says so.
- [ ] Rollback tested: previous installer over the current one with a current database refuses or migrates down cleanly.
- [ ] Backups tested by actually restoring on another machine and verifying every evidence package.
- [ ] Monitoring and alerting demonstrated: a dark camera, a stopped recording and a full disk each produce an alert outside the process.
- [ ] Logging sufficient: a slot exception, a decode fault and a crash each leave a traceable record (excepthook + faulthandler).
- [ ] Documentation accurate: DOC-01 done; every document's H2 carries a state; USAGE matches the binary's `--help`.
- [ ] Performance measured on the reference hardware with the shipped model: cameras per node, fps per camera, startup time, RSS after 72 h.
- [ ] Cross-platform verified with photographs of the console running on a camera on each platform.
- [ ] No known critical vulnerability (SEC-06 scan clean); the onnxruntime telemetry decision recorded.
- [ ] No unresolved crash: the exit-time heap corruption stays fixed (freeing test), no reproduction open.
- [ ] No placeholder or mock in production paths: the register panel wired or hidden; `orthophoto` and `faces` callers exist or the modules are marked; the Investigation tab searches real data.
- [ ] No development secrets or debug configuration shipped: `.env` absent, `--verbose` off in the non-dev build, no test databases in the bundle.
- [ ] Every major feature has explicit acceptance criteria (FEATURES.md rows with `TESTED` and the test named).
- [ ] A fresh machine installs and runs the production build, and `python tasks.py exetest`'s equivalent passes on the shipped binary with a person in frame — pictures attached.
- [ ] The application survives realistic failure scenarios (§9, §23) in a 72-hour soak with induced faults.
- [ ] One physical IP camera has run for an hour (NET-01) and the result is in STATUS.

---

## 25. Remaining Unknowns / Things That Must Be Verified

| Unknown | How to find out |
|---|---|
| Behaviour on the operator's monitor at its DPI scale after the two-row toolbar | `exetest` pictures on that machine (tonight's run is on the dev laptop) |
| Whether onnxruntime opens a socket at runtime on Linux with the variable set | `strace -f -e trace=network` around the packaged analyser in the container |
| Memory and handle counts after 72 h | REL-07 soak |
| Exit behaviour with a wedged native `read()` | REL-06 injection |
| Whether `PRAGMA synchronous` default (FULL) is what a recorder can afford | measure write latency; DATA-02 |
| Whether `save_incident`'s delete of superseded incidents cascades `incident_events` | DATA-05 test |
| Schedules across DST in Asia/Beirut and Europe/London | REL-13 tests |
| Enumeration on a host without PowerShell | SEC-10 test |
| Two instances on one data directory | REL-15 |
| Real RTSP timestamps, reconnects, resolution changes | NET-01 on hardware; TEST-03 in CI |
| The false-object rate with the watch list and 0.50 floor on real footage | TEST-04 |
| Whether the AGPL model can ship at all | AI-02 legal review |
| Whether Linux and macOS enumeration/capture work at all | XP-01 |
| Whether `QFont::setPointSize` hides a real font bug | UI-02 |
| Whether the packaged non-dev build shows anything when the excepthook fires before the window exists | UX-10 |
| Whether any other slot reads a Qt object after the loop that owned it has ended (the `WA_DeleteOnClose` class of defect, now forbidden in `app.py` only) | grep every `exec()` and `deleteLater()` in `apps/console`; UX-12 |

---

## Appendix A — Evidence index

- Operator log: `%LOCALAPPDATA%\SentinelVision\logs\sentinel.log` (520 lines on 2026-09-04 23:12; 80-class loads at lines 465–516; duplicate-camera warnings at 418–419).
- Operator database audit trail: 74 rows; the 22:51–23:00 session described in §2.
- Test runs: `python tasks.py test` 2026-09-04 23:14 (Rust 60 / engine 1096 / console 330+2F); console suite 2026-09-05 (363 passed); `tasks.py ci --package` (§2.3).
- Code references: `decode.py:83,357–398` (egress override and guard); `app.py` `_export_incident` (bypasses node); `core.py` `_MAX_TRACKS`; `store.py` (no checksum column, no `synchronous`); `node.py` `_sighted`/`_plate_refused`; `evidence.py:55` (`APPLICATION_VERSION`).
- Documents contradicted: USAGE §11/§12; DATABASE.md "Migrations", "Performance notes"; SECURITY.md "Authorization"; README "Scale"; TESTING.md counts.
