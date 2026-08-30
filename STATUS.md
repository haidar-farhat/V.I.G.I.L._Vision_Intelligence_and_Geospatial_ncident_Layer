# Implementation status

This file exists because a UI existing is not the same as a feature working, and
a system that overstates its own readiness is dangerous in exactly the domain
this one operates in. Every capability carries one of five states:

| State | Meaning |
|---|---|
| `PLANNED` | Designed in ARCHITECTURE.md. No code. |
| `SKELETON` | Types, interfaces or scaffolding exist. Does not do the job yet. |
| `IMPLEMENTED` | Works. Not yet covered by tests that would catch a regression. |
| `TESTED` | Works, and tests fail if it stops working. |
| `PRODUCTION-READY` | Tested, hardened, documented, and exercised against real hardware. |

**Nothing in this repository is `PRODUCTION-READY`.** No part of this system has
been run against a physical IP camera, a GPU, or a multi-machine LAN. Everything
below marked `TESTED` is tested against the simulator and unit fixtures, which is
a real bar but not the same bar.

Last updated with the vertical-slice milestone. Current suite: **247 tests**,
clean typecheck under `strict` + `noUncheckedIndexedAccess`, clean architectural
lint.

---

## Core domain

| Capability | State | Notes |
|---|---|---|
| Domain model (`shared-types`) | `TESTED` | Branded ids, closed vocabularies, geo types. Compiler rejects a `CameraId` where an `IncidentId` is wanted. |
| Local tangent-plane geodesy | `TESTED` | Agrees with haversine to under a centimetre at site scale. |
| Polygon predicates | `TESTED` | Point-in-polygon, area, centroid, segment intersection, corridor distance. |
| Ground projection + uncertainty | `TESTED` | Uncertainty derived from the real error derivative; unprojectable rays return null rather than a guess. |
| Inverse projection (world to image) | `TESTED` | Round-trips to sub-centimetre. Lets the simulator drive production code. |
| Field-of-view footprint | `TESTED` | Annular sector, excluding the blind foreground under a tilted camera. |
| Zone engine | `TESTED` | Polygon, rectangle, circle, corridor, tripwire. Enter/exit/dwell/cross without boundary chatter. |
| Tracker | `TESTED` | Survives detection gaps; IoU plus a size-scaled proximity gate. Deterministic. |
| Cross-camera association | `TESTED` | Topology-aware scoring with reasons. Physically impossible hand-offs rejected outright. |
| Rule engine | `TESTED` | Structured rules only, never executable code. Rejections report why. |
| Risk scoring | `TESTED` | Additive, every point attributable to a stated reason. |
| Event correlation and incidents | `TESTED` | Three cameras seeing one person produce one incident. Transitive association chains. |
| AI analyst contract + guardrails | `TESTED` | Citations validated against the supplied bundle; prohibited claims refused. |
| Deterministic analyst | `TESTED` | Grounded by construction. Works with no model installed. |
| Secret handling | `TESTED` | `Secret<T>` redacts through every serialisation path; 24 tests assert no leak. |
| Egress guard (zero WAN) | `TESTED` | Address classification and refusal. Public DNS names refused, not resolved. |
| Authorization + rate limits + replay guard | `TESTED` | Permission-based checks; per-key limits; nonce + timestamp window. |
| Database driver + migrations | `TESTED` | `node:sqlite`, savepoint nesting, checksummed reversible migrations. |
| Schema | `TESTED` | Full ERD. Tests assert no credential column and no position without uncertainty. |
| Simulator (cameras, actors, scenarios) | `TESTED` | Seeded and reproducible. Drives the production pipeline, not a mock. |
| Worker camera pipeline | `TESTED` | Detections to tracks to zone observations. |
| Architectural lint | `TESTED` | Layering, zero-WAN, secrets, placeholders, erasable syntax. |

## Vertical slice

| Capability | State | Notes |
|---|---|---|
| Synthetic camera to incident, end to end | `TESTED` | 1 CRITICAL incident from 9 events across 3 cameras; mean position error 0.52 m; 100% of errors inside the stated uncertainty. |
| Evidence-bound incident report | `TESTED` | Generated and validated within the slice run. |
| Quiet-site behaviour | `TESTED` | An empty site raises nothing. |
| Determinism across runs | `TESTED` | Same seed reproduces identical event and incident ids. |

## Not yet built

Everything below is designed in [ARCHITECTURE.md](ARCHITECTURE.md) and has no
working implementation. Listed explicitly so the gap is visible rather than
inferred from silence.

| Capability | State | Notes |
|---|---|---|
| REST API + WebSocket hub (`services/api`) | `PLANNED` | Contracts specified in docs/PROTOCOL.md; no server yet. |
| Desktop shell (Tauri, Rust) | `SKELETON` | Scaffolded. Not built or run - no Rust toolchain was available in this environment. |
| Operator UI (React) | `SKELETON` | Command centre, map and incident views scaffolded; not wired to a live API. |
| RTSP ingestion | `PLANNED` | `VideoSource` abstraction defined; no decoder integration. |
| ONVIF discovery + Profile T | `PLANNED` | Camera and profile models exist; no protocol implementation. |
| Real detector (YOLO-family, ONNX/TensorRT) | `PLANNED` | `Detector` interface and model registry exist; only the simulated detector is implemented. |
| GPU device discovery | `SKELETON` | `selectDevice` chooses among reported devices; nothing enumerates real hardware yet. |
| VLM integration | `PLANNED` | Interface defined; no runtime. |
| Local LLM analyst | `PLANNED` | `AnalystEngine` interface is satisfied by the deterministic engine; no LLM client. |
| Recording + segmentation | `PLANNED` | Schema and retention policy exist; no recorder. |
| Evidence export + manifest | `PLANNED` | Format specified; not implemented. |
| Offline map packages (PMTiles) | `PLANNED` | Import and validation flow specified; not implemented. |
| MapLibre rendering | `SKELETON` | Component scaffolded; no tiles, no offline package. |
| LAN discovery (mDNS) | `PLANNED` | Protocol chosen; not implemented. |
| Node pairing + mTLS | `PLANNED` | Flow specified in ARCHITECTURE.md section 9.3; no implementation. |
| Worker buffering + reconciliation | `PLANNED` | Deterministic event ids make replay idempotent, which is the hard half; the buffer itself is not written. |
| OS keychain integration | `PLANNED` | `CredentialsRef` indirection exists throughout; no platform binding. |
| Authentication (login, sessions) | `PLANNED` | Password hash column and permission model exist; no auth flow. |
| Audit log writes | `SKELETON` | Table and record type exist; nothing writes to them yet. |
| PTZ control | `PLANNED` | Permission and confirmation model exist; no ONVIF PTZ. |
| Alerting (desktop, audible, webhook) | `PLANNED` | |
| Incident replay (synchronised playback) | `PLANNED` | Timeline data exists and is ordered; no player. |
| Search + natural-language retrieval | `PLANNED` | |
| Backup / restore | `PLANNED` | |
| Packaging + offline updates | `PLANNED` | |
| Internationalisation (EN/FR/AR, RTL) | `PLANNED` | No hard-coded UI strings introduced so far. |
| Chaos + network partition tests | `PLANNED` | Simulator hooks exist for camera dropout only. |

---

## Honest gaps worth naming

1. **No real video has ever passed through this system.** The pipeline consumes
   detections, and the only detector implemented is the simulated one. RTSP,
   decode, and hardware inference are the largest remaining unknowns, and the
   performance targets in the specification are design targets, not measurements.

2. **The desktop application has not been compiled.** The Tauri shell is
   scaffolded but no Rust toolchain was present, so it has never been built or
   launched. Treat the UI as a design artefact until that changes.

3. **The AI analyst is not a language model.** It is a deterministic, grounded
   report generator that satisfies the same contract an LLM would have to satisfy.
   That is deliberate - it is the reference implementation and the offline
   fallback - but nobody should read "AI analyst: TESTED" as "a local LLM works".

4. **Distributed mode is designed, not built.** Deterministic event identity and
   idempotent correlation - the parts that make reconciliation correct - are
   implemented and tested. The transport, pairing and buffering are not.

5. **Spatial accuracy is measured against simulated ground truth**, on a flat
   ground plane with a perfectly known camera pose. Real deployments have sloping
   ground, mis-surveyed masts and lens distortion. The uncertainty model is
   honest about its inputs; its inputs are currently ideal.
