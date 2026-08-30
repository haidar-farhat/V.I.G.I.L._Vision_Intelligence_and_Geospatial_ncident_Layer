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

Last updated at the end of Phase 5 (map, geospatial placement, FOV). Current
suite: **481 tests**,
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

## Camera discovery and ingestion (Phase 2)

| Capability | State | Notes |
|---|---|---|
| Bounded queues and backpressure | `TESTED` | Ring buffer with per-queue drop policy. Frames drop oldest; events refuse rather than lose evidence. |
| Bounded exponential backoff | `TESTED` | Full jitter, so cameras that dropped together do not retry in lockstep. Abortable mid-wait. |
| SDP parsing | `TESTED` | Bounded against hostile input; reports what it could not parse instead of failing a working camera. |
| RTSP Digest/Basic authentication | `TESTED` | RFC 7616. Unimplemented algorithms refused rather than downgraded. Basic is opt-in. |
| RTSP control client | `TESTED` | Full OPTIONS/DESCRIBE/SETUP/PLAY/TEARDOWN handshake with keep-alive, against a mock camera that misbehaves the way real ones do. |
| ONVIF WS-Discovery | `TESTED` | Link-local multicast, probes every private interface. Parser tested; the multicast send path has no physical device to answer it. |
| SOAP + WS-Security | `TESTED` | UsernameToken PasswordDigest. Bounded extractor rather than an XML parser; DTDs refused. |
| ONVIF device/media client | `TESTED` | Device info, capabilities, profiles, stream URI. Credentials stripped from returned URIs. |
| Camera connection test | `TESTED` | The onboarding wizard's verification step. Every failure carries a remedy, enforced by test. |
| Camera persistence | `TESTED` | Cameras, profiles, topology edges. No column can hold a credential. |
| Supervised video source | `TESTED` | Reconnect with backoff, flapping detection, honest DEGRADED vs OFFLINE, decoder boundary. |
| Video decode (H.264/H.265) | `PLANNED` | The `Decoder` interface is defined and the supervisor is tested against a stub. No FFmpeg implementation is written - see the gaps below. |
| Live view rendering | `PLANNED` | Requires decode. |

## Map and geospatial placement (Phase 5)

| Capability | State | Notes |
|---|---|---|
| PMTiles v3 reading | `TESTED` | Header and metadata, with every declared region checked against the real file length before any read. Tests build genuine archives, not stubs. |
| Map style validation | `TESTED` | Every URL a style can carry, including ones buried in a layer property, checked against the private ranges and the package's own file listing. |
| Map package import | `TESTED` | Content-addressed id, integrity hash, path-traversal refusal, warnings for coarse zoom and for style layers the archive lacks. |
| Map package persistence | `TESTED` | Install, list, set default, remove. Removing the default promotes the most detailed remaining package. |
| Camera coverage analysis | `TESTED` | Deterministic grid sampling against the real annular footprint. Reports covered, redundant, partial or blind, with the blind points themselves. |
| Camera placement UI | `IMPLEMENTED` | Live pose editing with footprints and blind spots recomputed in the browser by the production geometry. Verified in a headless browser. Not persisted - no API. |
| Map rendering | `IMPLEMENTED` | Cameras, footprints, zones, events and blind spots. Reports `OFFLINE MAP DATA NOT INSTALLED` and never fetches tiles. |
| Tile rendering from a package | `PLANNED` | The reader and validator exist; wiring an imported archive into MapLibre as a source does not. |
| Zone drawing on the map | `PLANNED` | Zones render and are analysed; drawing and editing them by hand is not built. |

## Not yet built

Everything below is designed in [ARCHITECTURE.md](ARCHITECTURE.md) and has no
working implementation. Listed explicitly so the gap is visible rather than
inferred from silence.

| Capability | State | Notes |
|---|---|---|
| REST API + WebSocket hub (`services/api`) | `PLANNED` | Contracts specified in docs/PROTOCOL.md; no server yet. |
| Desktop shell (Tauri, Rust) | `SKELETON` | Manifest, config and keychain command surface written. **Never compiled** - no Rust toolchain was available in this environment. |
| Operator UI (React) | `IMPLEMENTED` | Command centre, map, timeline and analysis panels render real pipeline output; verified in a headless browser with zero console errors and zero network requests. Reads a generated snapshot, not a live API. |
| ONVIF Profile T events | `PLANNED` | Discovery, device and media services are implemented; the event service is not. |
| Real detector (YOLO-family, ONNX/TensorRT) | `PLANNED` | `Detector` interface and model registry exist; only the simulated detector is implemented. |
| GPU device discovery | `SKELETON` | `selectDevice` chooses among reported devices; nothing enumerates real hardware yet. |
| VLM integration | `PLANNED` | Interface defined; no runtime. |
| Local LLM analyst | `PLANNED` | `AnalystEngine` interface is satisfied by the deterministic engine; no LLM client. |
| Recording + segmentation | `PLANNED` | Schema and retention policy exist; no recorder. |
| Evidence export + manifest | `PLANNED` | Format specified; not implemented. |
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
| Internationalisation (EN/FR/AR, RTL) | `PLANNED` | The React UI hard-codes English throughout and would need extracting before any of this starts. Domain code emits structured codes rather than prose, so the event and risk vocabularies are already translatable. |
| Chaos + network partition tests | `PLANNED` | Simulator hooks exist for camera dropout only. |

---

## Honest gaps worth naming

1. **No real video has ever passed through this system, and no physical camera
   has ever been contacted.** The RTSP and ONVIF protocol layers are implemented
   and tested, but against mock devices written from the specifications - which
   means they are tested against my reading of those specifications. Real cameras
   deviate from both in ways no mock anticipates, and that gap will only close on
   hardware.

   Decode is not implemented at all. The `Decoder` interface is defined and the
   supervisor is tested against a stub, but no FFmpeg integration exists, because
   no FFmpeg was available in the environment where this was written and shipping
   an unverifiable subprocess wrapper would be exactly the fake implementation
   this file exists to prevent. Hardware inference is likewise untouched, and the
   performance targets in the specification remain design targets rather than
   measurements.

2. **The Tauri shell has never been compiled.** No Rust toolchain was present, so
   the native layer - keychain access, tray, service supervision - has never been
   built or run. The React interface *has* been built and verified in a headless
   browser, but it reads a generated snapshot rather than a live API, because
   there is no API yet.

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

   Coverage analysis inherits the same assumption, and one more: it models what a
   camera can *geometrically* reach, not what it can usefully *see*. A person at
   85 m may be four pixels tall and undetectable, and no wall, fence, vehicle or
   tree occludes anything. Coverage is therefore an upper bound - real coverage is
   never better than this and is usually worse.

6. **No tiles have ever been rendered from an imported package.** The PMTiles
   reader and the package validator are tested against archives built byte by
   byte, but nothing has yet handed a real basemap to MapLibre. The map draws site
   geometry over an empty background, which is the correct behaviour with no
   package installed and also the only behaviour so far exercised.
