# Sentinel Vision — Architecture

> Local-first AI multi-camera security & situational awareness platform.
> Repository codename: **VIGIL** (Vision Intelligence and Geospatial Incident Layer).

**Status of this document:** authoritative. Every architectural change must land here in the
same commit as the code. Implementation state per subsystem is tracked in [STATUS.md](STATUS.md).

---

## 1. Purpose

Sentinel Vision turns ordinary IP cameras into a local **event intelligence system**.

The camera is a sensor. The AI is an analyst. The operator is the decision maker.

The system is architected around the assumption that **nobody is watching the cameras**.
Its job is to extract signal from continuous video, attach spatial and temporal context to
that signal, correlate it across cameras into a small number of reviewable incidents, and
present each one with the evidence that produced it.

The transformation the whole system exists to perform:

```
VIDEO -> DETECTION -> TRACKING -> SPATIAL CONTEXT -> TEMPORAL CONTEXT
      -> EVENT ANALYSIS -> MULTI-CAMERA CORRELATION -> RISK SCORING
      -> HUMAN REVIEW -> INCIDENT
```

---

## 2. Architectural invariants

These are non-negotiable. Any change that violates one is a defect, not a trade-off.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | **Zero WAN.** No feature may require Internet access. No silent online fallback. | CI offline job: drops all outbound traffic, proves the drop took effect, runs every suite; test asserting the map view references no URL or HTTP client |
| I2 | **Secrets never leave the vault.** Camera/node credentials never appear in logs, DB tables, API payloads, URLs, or errors. | A URL's password is held in one place and redacted for every other purpose; tests assert it is unreachable through `repr`, `str`, display URL, source id and every error message, including its length |
| I3 | **No AI claim without evidence.** Every AI statement carries evidence IDs, model ID, prompt version, confidence. | evidence-bound analyst contract; `OBSERVED / INFERRED / UNKNOWN` partition |
| I4 | **Graceful degradation.** Loss of GPU, AI, DB, control node, or map never stops recording. | failure matrix (section 16); chaos tests |
| I5 | **Bounded resources.** Every queue is bounded; every retry is backed off; no unbounded growth. | `BoundedQueue` with explicit drop policy; backoff utility |
| I6 | **Auditable.** Every privileged action produces an immutable audit record with before/after. | `audit_logs`, append-only |
| I7 | **Process separability.** Co-location is a deployment choice, never a code assumption. | services communicate only over the versioned protocol |
| I8 | **Deterministic core.** Geometry, rules, scoring and correlation are pure and seedable. | zero-dependency pure packages + golden tests |

---

## 3. System context

```
                     +----------------------------------------+
                     |        SECURITY LAN (RFC1918)          |
   IP cameras        |                                        |
   (Profile T,       |   +----------+                         |
    RTSP)  ----------+-->|  Worker  |--+                      |
                     |   +----------+  |                      |
   USB / file        |   +----------+  |  mTLS 1.3            |
   sources ----------+-->|  Worker  |--+---> +---------------+|
                     |   +----------+  |     | Control node  ||
   Local NTP / PTP   |   +----------+  |     | API + DB + UI ||
            ---------+-->|  Worker  |--+     +---------------+|
                     |   +----------+              |          |
                     +-----------------------------+----------+
                                                   |
                              Operator <-----------+
                              (desktop UI)         |
                                        +----------+----------+
                                        |                     |
                                Local LLM/VLM         Offline map
                                 (optional)            packages

   ##########################################################
   #  WAN / Internet -- NOT REACHABLE, NOT REQUIRED         #
   ##########################################################
```

Nothing crosses the WAN boundary. Ever. See section 13.

---

## 4. Component architecture

```
+-----------------------------------------------------------------------------+
|                        apps/console   (PySide6, native)                     |
|   camera view . plan view . track table . placement . no embedded browser    |
+-------------------------------+---------------------------------------------+
                                | in-process today; REST + WebSocket when the
                                | control plane is rebuilt
+-------------------------------v---------------------------------------------+
|                         engine  (Python)                                    |
|   decode . detect . pipeline . orchestration . policy . persistence         |
+-------------------------------+---------------------------------------------+
                                | C ABI (ctypes) - see 18.1
+-------------------------------v---------------------------------------------+
|                       core  (Rust, zero dependencies)                       |
|   geometry . ground projection . field of view . zones . tracking           |
+-----------------------------------------------------------------------------+
```

### The split, and why it falls where it does

**Rust holds what runs per detection, per frame, per camera** — projection,
field-of-view geometry, polygon tests, and the tracker. It is the only code whose
cost multiplies by camera count and frame rate, and it is also the code where a
subtle error is hardest to see: bad geometry produces plausible numbers.

**Python holds everything that runs per event, per second, or per operator
action** — decode orchestration, model execution, rules, correlation,
persistence, the interface. This is where the system changes most often, and
where the libraries that matter (OpenCV, onnxruntime, Qt) live.

The dividing line is *rate*, not importance. Anything called at frame rate goes
below the boundary; anything called at human rate stays above it.

### Layering rule

Dependencies point **downward and never back up**. `core` knows nothing about
Python. `engine` knows nothing about Qt. The console imports the engine; the
engine never imports the console. `core` has **no dependencies at all** — for a
security appliance the dependency list is part of the attack surface, and
everything in it is arithmetic.

---

## 5. Process topology

### 5.1 Standalone local mode

One machine. The console owns the process; analysis runs on its own thread, one
per camera, and the interface pulls results from it rather than being pushed to.

```
 sentinel-console  (Python + Qt)
   +-- analysis thread per camera
   |     decode -> detect -> track -> project        [engine]
   |                          |
   |                          +-- sentinel_core      [Rust, in-process]
   +-- UI thread: repaint timer pulls the newest result
```

Works with the network cable physically removed. Nothing in this path opens a
socket except a camera stream.

The threading direction matters and is an architectural decision rather than an
implementation detail: Qt's queued signal delivery is unbounded, so a pipeline
producing faster than the display consumes would build a backlog of already-stale
frames until the process died. The interface therefore takes the newest result on
a timer, and frames skipped for *drawing* are counted and shown. Nothing is
skipped from the analysis.

### 5.2 LAN distributed mode

Designed; not built in the current codebase. See STATUS.md.

```
                       CONTROL NODE
                 +-----------------------+
                 | console . api . db    |
                 | event-engine          |
                 +-----------+-----------+
                             | mTLS 1.3, pinned node identities
        +--------------------+--------------------+
        v                    v                    v
   WORKER PC 1          WORKER PC 2          WORKER PC 3
   cams 01-08           cams 09-16           cams 17-24
   decode.infer.track   decode.infer.track   decode.infer.track
   local buffer         local buffer         local buffer
```

Workers are autonomous. If the control node disappears they keep decoding,
detecting, tracking, recording and generating events into a local durable buffer,
and reconcile on reconnect (section 9.4). Scale is horizontal: add worker nodes,
not bigger constants. There is no hard-coded camera limit anywhere.

A worker is the `engine` package with no console attached — which is why the
engine has no Qt dependency and the pipeline is usable without one.

---

## 6. Data flow -- the spine

```
 CAMERA                                                       [worker node]
   |  RTSP / ONVIF / USB / file
   v
 [ VideoSource ]        reconnect w/ bounded exponential backoff
   |  encoded packets
   v
 [ Decoder ]            hardware decode when available, else software
   |  raw frames
   v
 [ BoundedQueue ]  <-- backpressure: DROP_OLDEST. A slow model can never
   |                   grow memory; it can only lose frames, visibly, with a
   |                   counter the health screen reports.
   v
 [ FrameSampler ]       adaptive: idle 2fps -> active 15fps
   |
   v
 [ Preprocess ]         letterbox . normalize . to device tensor
   |
   v
 [ Detector ]           YOLO-family or equivalent; model is swappable, its ID
   |  detections        is recorded on every observation it produces
   v
 [ Postprocess ]        NMS . class filter . confidence gate
   |
   v
 [ Tracker ]            IoU + motion association, survives detection gaps
   |  tracks
   v
 [ Projection ]         image point -> ground plane -> lat/lon (+ uncertainty)
   |
   v
 [ ZoneEngine ]         enter . exit . cross . dwell . direction
   |  observations
   v
 ======== node boundary ======== (buffered, at-least-once, idempotent)
   |
   v
 [ RuleEngine ]         structured rules -> EVENTS (not alerts)
   |
   v
 [ Correlator ]         same-object across cameras . dedup . merge
   |
   v
 [ RiskScorer ]         explainable additive score -> LOW/MEDIUM/HIGH/CRITICAL
   |
   v
 [ IncidentEngine ]     events -> one incident with a timeline
   |
   v
 [ AI Analyst ]         evidence-bound summary (optional, local LLM/VLM)
   |
   v
 [ Operator ]           acknowledge . classify . resolve . escalate . export
```

The critical design point: **detections are not alerts**. Three cameras seeing the same
person produce **one** incident, not three alerts. Alert fatigue is a system failure mode
and is treated as one.

---

## 7. Domain model (ERD)

```
 +----------+        +----------+        +-----------+
 |  users   |---+    |  nodes   |<--+    | locations |<-+ self-ref:
 +----+-----+   |    +----+-----+   |    +-----+-----+  | site -> building
      | 1:N     |         | 1:N     |          | 1:N    | -> floor -> room
 +----v-----+   |    +----v---------+--+  +----v--------+-+
 |  roles   |   |    |    cameras      |->|    zones      |
 +----------+   |    | geo . fov . cal |  +-------+-------+
                |    +----+------------+          |
 +----------+   |         | 1:N          camera_zone_links
 |audit_logs|<--+    +----v----------+           |
 +----------+        |camera_profiles|           |
                     +---------------+           |
 +----------------+                              |
 |camera_topology |  edges: from_camera -> to_camera, distance, travel time
 +----------------+                              |
                                                 |
 +----------+  1:N   +------------------+        |
 |  tracks  |------->|track_observations|        |
 +----+-----+        +------------------+        |
      | N:M via event_tracks                     |
 +----v-----+<------------------------------------+
 |  events  |  type . severity . confidence . evidence . status
 +----+-----+
      | N:M via incident_events
 +----v------+  1:N   +----------+
 | incidents |------->| evidence |--> recordings / frames / exports
 +----+------+        +----------+
      | 1:N
 +----v------------+  +----------+  +------------+  +---------------+
 | incident_notes  |  |  alerts  |  |   rules    |  | model_registry|
 | (append-only)   |  +----------+  +------------+  +---------------+
 +-----------------+
 +---------------+  +--------------+  +--------------+
 |system_settings|  | map_packages |  | ai_inferences|  prompt ver, evidence IDs
 +---------------+  +--------------+  +--------------+
```

Rules:

- **All timestamps stored UTC**, displayed operator-local. Camera-reported time and
  node-received time are stored **separately and never reconciled silently** -- clock skew is
  evidence, not noise.
- **No plaintext credentials in any table.** `cameras.credentials_ref` is an opaque handle
  into the OS keychain.
- **Append-only tables**: `audit_logs`, `incident_notes`, `ai_inferences`. No UPDATE, no DELETE.
- Full DDL and migration policy: [docs/DATABASE.md](docs/DATABASE.md).

---

## 8. Storage abstraction

```
        repositories (typed, domain-shaped)
                    |
            +-------v--------+
            |  SqlDriver     |  interface: exec . query . transaction . migrate
            +---+--------+---+
                |        |
        sqlite3|        |PostgreSQL
     (standalone|        |(multi-node / large deployments)
      stdlib)   |        |
```

Standalone uses SQLite in WAL mode — embedded, transactional, no daemon, no
install, and in Python's standard library, so a single-machine deployment adds
nothing to the dependency list. Serious multi-node deployments point the same
driver interface at PostgreSQL. Repository code never sees the difference;
migrations are authored once in portable SQL.

Neither is built yet. See STATUS.md.

---

## 9. Worker protocol

Full wire format: [docs/PROTOCOL.md](docs/PROTOCOL.md). Summary:

### 9.1 Envelope

Every message on every channel carries:

```
{ v: protocol version, request_id, timestamp (UTC, monotonic-checked),
  node_id, kind, payload }
```

Unversioned internal APIs are forbidden. Version mismatch is a hard, explicit failure with
a human-readable message -- never a best-effort parse.

### 9.2 Channels

| Channel | Transport | Direction | Purpose |
|---------|-----------|-----------|---------|
| `control` | REST/HTTPS | control -> worker | config, camera assignment, drain, restart |
| `telemetry` | WebSocket | worker -> control | health, FPS, load, camera state |
| `observations` | WebSocket | worker -> control | tracks, zone observations, events |
| `discovery` | mDNS/DNS-SD + UDP | broadcast | node presence on the local link |
| `ui` | WebSocket | api -> desktop | events, camera_status, tracks, incidents, nodes, system |

### 9.3 Pairing

```
 worker                                     control
   |  discovery announce (node_id, pubkey fingerprint, caps)
   |------------------------------------------>
   |                     operator sees pending node, reads the 6-word
   |                     fingerprint off-channel, approves
   |  <---- pairing challenge (nonce, control cert) ----
   |  ---- signed response + CSR ------------->
   |  <---- signed worker cert, trust anchor ----------
   |
   |  === mTLS 1.3 from here on; certs pinned by node_id ===
```

Pairing is **explicit and human-approved**. The LAN is treated as hostile: possession of a
LAN address grants nothing.

### 9.4 Disconnection and reconciliation

Worker behaviour on control-node loss:

1. Keeps decoding, detecting, tracking, recording. **No degradation of the sensor.**
2. Appends events/observations to a durable local buffer with a monotonic sequence number.
3. Retries with bounded exponential backoff + jitter -- never a tight reconnect loop.

On reconnect the worker replays from the last control-acknowledged sequence. Events carry a
deterministic ID derived from `(node_id, camera_id, rule_id, track_id, time bucket)`, so a
replayed event is an idempotent upsert, not a duplicate. Reconciliation converges regardless
of replay order.

---

## 10. Network architecture

- **Discovery:** mDNS/DNS-SD (`_sentinel._tcp.local`) with UDP fallback and manual IP entry.
  No external discovery service, no rendezvous server, no STUN/TURN.
- **Binding:** services bind explicit LAN interfaces; loopback by default.
- **Addressing:** private ranges only (10/8, 172.16/12, 192.168/16, 169.254/16, ::1, fc00::/7).
- **Node identity:** every node has a keypair; `node_id` is derived from the public key, so a
  node cannot rename itself into another node's trust slot.

Node record: `node_id . name . role(CONTROL|WORKER|STORAGE|HYBRID) . addresses . capabilities
. gpu . cpu . os . app version . protocol version . last heartbeat`. A node may hold several
roles simultaneously.

---

## 11. Security model

Threat model and controls: [docs/SECURITY.md](docs/SECURITY.md). Core positions:

**The LAN is hostile.** An attacker on the same subnet is assumed. Therefore: mutual TLS,
pinned identities, explicit pairing, per-request authorization, replay protection
(nonce + timestamp window + monotonic request IDs), and rate limits on every entry point.

**Trust boundaries:**

```
  camera ---+  untrusted input (malformed RTSP, hostile firmware)
  operator -+--> [ worker ] --> [ api ] --> [ db ]
  LAN peer -+         ^             ^
                      |             +- authz per action, audit every privileged one
                      +- decoder runs on untrusted bytes: bounded, restartable, isolated
```

**Credential handling.** Camera passwords live in the OS keychain -- Windows Credential
Manager, macOS Keychain, Linux Secret Service. The database stores only a reference. In
process they are carried in a `Secret<T>` wrapper whose `toString`, `toJSON` and inspection
hooks all render `[redacted]`, so a credential cannot reach a log line or an error payload
even by accident. A dedicated test asserts no camera password appears in any test output.

**Authorization.** Roles `ADMIN . OPERATOR . ANALYST . VIEWER` map to granular permissions.
High-risk actions (delete camera, delete evidence, export incident, change retention, PTZ
control, modify node, change rules) require explicit confirmation and are always audited.

---

## 12. Event, correlation and incident engine

### 12.1 Events

Observations become **events** only through structured, data-defined rules -- never through
arbitrary executable code stored in the database.

```
 rule := { when: { object_type, zone, time_window, duration, direction,
                   confidence_threshold, cooldown },
           then: { event_type, severity, notify } }
```

### 12.2 Multi-camera correlation

When a track leaves camera A and reappears on camera B, a candidate association is scored --
never asserted:

```
 association_score = w1 * class_match
                   + w2 * travel_time_plausibility(topology edge)
                   + w3 * direction_consistency
                   + w4 * appearance_similarity (optional embedding)
                   + w5 * topology_edge_prior
```

The camera **topology graph** (`C01->C02->C04->...` with distance and expected travel time
per edge, operator-editable) is what makes this tractable. The output is always a score with
its contributing reasons, shown to the operator as such.

### 12.3 Incidents

Correlated events collapse into one incident with a timeline, related cameras/tracks/events,
evidence, AI summary and human assessment. Status: `NEW . ACKNOWLEDGED . INVESTIGATING .
RESOLVED . FALSE_POSITIVE . ARCHIVED`.

### 12.4 Risk scoring

Additive and **explainable** -- every contribution is stored with its reason:

```
 base(event type)  +restricted zone  +after hours  +dwell duration
 +multi-camera confirmation  +repeat behaviour  -known normal condition
 --------------------------------------------------------------------
 -> LOW | MEDIUM | HIGH | CRITICAL
```

Developers see the number; operators see the reasons.

---

## 13. Offline architecture (zero WAN)

Enforcement is layered, not aspirational:

1. **No online SDKs.** No cloud map API, no CDN fonts, no analytics, no crash reporting, no
   auto-updater, no telemetry. Nothing to disable, because nothing is there.
2. **Egress guard.** A runtime address guard rejects connections to non-private addresses
   when `SENTINEL_ENFORCE_OFFLINE=true`, so a future dependency cannot quietly phone home.
3. **Assets vendored.** Fonts, styles, icons, map glyphs and sprites ship in the bundle.
4. **Explicit absence.** Missing map data renders `OFFLINE MAP DATA NOT INSTALLED`. A missing
   model renders an import prompt. The system never substitutes an online source.
5. **Diagnostic screen.** `NETWORK ISOLATION` states WAN blocked/not required, LAN active,
   Internet dependency NONE, cloud services DISABLED, telemetry DISABLED.
6. **Test proof.** The acceptance suite runs with WAN blocked and must pass.

Everything imported -- models, map packages, datasets, updates -- enters through **local file
import** with validation. Nothing is ever auto-downloaded.

---

## 14. AI architecture

```
  every frame          selected frames only        selected events only
      |                        |                           |
      v                        v                           v
 +---------+            +------------+              +--------------+
 |Detector |--tracks--->|    VLM     |--structured->|  LLM analyst |
 | (fast)  |            | (targeted) |  observation |  (summarize) |
 +---------+            +------------+              +--------------+
```

Model abstraction: `Detector . Tracker . Classifier . Segmenter . PoseEstimator .
EmbeddingModel . VLM`. Any is replaceable without touching the event engine -- the engine
consumes typed observations, not model output.

**The VLM never sees the whole stream.** It is invoked on representative frames of candidate
events only. This is the difference between a system that runs on one GPU and one that cannot.

**Every model is registered**: `model_id . name . version . format . classes . input size .
runtime . precision . hash . license . installed_at`. Every event records which model
produced it, so a detection can always be traced to the exact artifact that made it.

**Analyst guardrails** (hard requirements, tested):

- Reasons only over supplied structured evidence; insufficient evidence yields exactly
  *"Insufficient evidence."*
- Partitions output into **OBSERVED / INFERRED / UNKNOWN**.
- Cites evidence IDs on every factual statement.
- Never identifies people, infers protected traits, or asserts criminality.
- Never controls cameras, PTZ, or any physical actuator. Advisory only.

Prompts are versioned configuration; each generated report records model, prompt version,
input evidence IDs, output and timestamp.

**Privacy by default:** no facial recognition, no biometric identification, no identity
database. Tracking is appearance-based and identity-free. Optional face blurring for exports.

---

## 15. Map architecture

```
 React -> MapLibre GL -> local style.json -> PMTiles / local tiles (file://)
                              |
                              +- glyphs (bundled)     no network at any layer
                              +- sprites (bundled)
                              +- GeoJSON overlays: cameras . FOV wedges . zones
                                                   tracks . events . incidents
```

Offline map packages are imported as local files (`region.pmtiles`, `style.json`,
`metadata.json`) and validated for CRS, bounding box, zoom range, tile type, integrity, and
style dependency completeness. Managed in the UI: import, remove, validate, set default.

Style validation is the check that matters most. A style that imports cleanly and
then reaches for a font server at render time produces a map that works in the lab
and shows unlabelled roads on the isolated site - so every URL a style can carry,
including ones buried in a layer property, is checked against the private ranges
and against the package's own file listing.

**Coverage analysis.** Given a zone and the placed cameras, the geometry package
samples the zone on a deterministic grid and reports what fraction is visible, to
how many cameras, and exactly where the blind spots are. A site with four cameras
around a restricted zone looks protected on a map; if none of their footprints
reach one corner, nothing detects an intrusion there and nothing reports that it
could not. That gap is otherwise found by whoever walks through it.

**Camera geospatial model:** `lat . lon . altitude . heading . pitch . roll . hfov . vfov .
range`, rendered as an FOV wedge the operator can drag and rotate directly on the map.

**Ground projection:** image-space observations are projected onto the ground plane using the
camera pose to give approximate map positions. Uncertainty is rendered explicitly -- an
uncalibrated camera produces a wide uncertainty ellipse, not a confident dot. **False
precision is a bug.**

Indoor sites (site -> building -> floor -> room) share the same zone and event model with a
local coordinate frame instead of a geographic one.

---

## 16. Failure model

| Failure | Behaviour | Never |
|---------|-----------|-------|
| GPU unavailable | fall back to CPU, reduce model + AI FPS, warn | stop recording |
| Inference crashes | restart the worker's inference stage, keep ingest | lose the video |
| Camera unreachable | bounded exponential backoff + jitter, DEGRADED then OFFLINE | tight reconnect loop |
| Decoder wedged | restart decoder, count the restart, surface in health | silently stall |
| Database unavailable | worker buffers durably, API returns explicit 503 | drop events |
| Control node down | workers run autonomously, buffer, reconcile on return | halt the edge |
| Map package missing | `OFFLINE MAP DATA NOT INSTALLED` | fetch online tiles |
| Disk near full | enforce quota, apply retention, warn early | delete incident evidence |
| Queue saturated | drop oldest **frames**, count and report | drop events, or grow memory |

Retention is tiered: normal recordings, event recordings, incident evidence and audit logs
expire independently. **Incident evidence is never removed by routine cleanup.**

---

## 17. Evidence integrity

An exported incident package is independently verifiable without Sentinel Vision:

```
 incident-<id>/
   incident.json  timeline.json  events.json  manifest.json
   video/  frames/
```

`manifest.json` lists every file with its SHA-256, plus application version, model versions,
prompt version, exporting user, and export timestamp. Chain of custody -- who exported what,
when, with which software and models -- is preserved and audited.

---

## 18. Technology decisions

| Decision | Choice | Why |
|---|---|---|
| Hot path | Rust, no dependencies | runs per detection per frame per camera; also the code where an error is hardest to see, because bad geometry produces plausible numbers |
| Binding | **C ABI via ctypes, not PyO3** | see 18.1 |
| Orchestration | Python 3.12+ | where the libraries that matter live — OpenCV, onnxruntime — and where the system changes most often |
| Console | PySide6, native widgets | no embedded browser: a control-room console that ships a browser engine inherits its update cadence, memory profile and network assumptions, for a machine expected to run for months offline |
| Decode | OpenCV with bundled FFmpeg | one dependency covering files, RTSP and USB, with no subprocess to supervise |
| Inference | onnxruntime, operator-supplied models | one runtime across CPU and several accelerators; **nothing is ever downloaded** |
| Baseline detector | MOG2 background subtraction | the system must not be useless without a model file; it is explicit that it cannot classify |
| Embedded DB | SQLite (`sqlite3`, WAL) | real SQL, transactional, in the standard library — standalone mode needs no install and no daemon |
| Scale-out DB | PostgreSQL behind the same driver interface | multi-node deployments without rewriting repositories |
| Map | local raster/vector packages, no tile server | the only genuinely offline-capable option; the plan view works with no package at all |
| Realtime | WebSocket, versioned envelopes | the operator UI must reflect state changes immediately |
| Transport security | mTLS 1.3, pinned node identities | the LAN is hostile |
| Tests | pytest, `cargo test` | offline, no runner to configure |

### 18.1 Why a C ABI rather than PyO3

PyO3 is the obvious choice and it is the wrong one here, for two reasons.

**The build boundary.** The core is built with the GNU toolchain; CPython on
Windows is built with MSVC. PyO3 links against CPython's ABI, so it inherits that
mismatch. The C ABI is the C ABI, and it crosses cleanly.

**The runtime lock-in.** A PyO3 module is a Python module and nothing else. A
cdylib with a C ABI is loadable from anything — a future worker written in
another language, a diagnostic tool, a test harness. The engine is deliberately
not welded to one runtime.

The cost is real: struct layouts are maintained by hand on both sides of the
boundary. That cost is *guarded* rather than absorbed. The core exports the size
of every struct that crosses, and the Python binding compares each against its
own declaration and refuses to load on a mismatch. This matters more than it
sounds: a drifted layout does not crash. It reads the wrong bytes and produces
geometry that looks entirely reasonable.

Three further rules hold the boundary:

- **Every pointer is checked** before it is dereferenced. A caller's bug must
  produce a defined failure, not a segfault inside a security appliance.
- **Every entry point taking a pointer is `unsafe` and carries a `# Safety`
  contract.** Null-checking cannot establish that a non-null pointer is live, and
  a function that implies otherwise is lying to its callers.
- **Panics cannot cross.** The crate is built `panic = "abort"`; unwinding into C
  is undefined behaviour.

---

## 19. Repository layout

```
sentinel-vision/
+-- core/                  Rust engine core - the hot path, zero dependencies
|   +-- src/geometry.rs      geodesy . projection . field of view . polygons
|   +-- src/tracking.rs      track lifecycle . association . motion
|   +-- src/ffi.rs           the C ABI
+-- engine/                Python engine
|   +-- sentinel/core.py     ctypes bindings; struct-layout guard
|   +-- sentinel/decode.py   decode . credential redaction . live streams
|   +-- sentinel/detect.py   motion + ONNX detectors
|   +-- sentinel/pipeline.py decode -> detect -> track -> project
+-- apps/console/          PySide6 operator console - native, no webview
+-- models/                operator-imported model artifacts (never committed)
+-- map-data/              operator-imported offline map packages (never committed)
+-- infrastructure/        packaging + deployment
+-- docs/                  DEVELOPMENT SECURITY PROTOCOL DATABASE AI MAPS DEPLOYMENT TESTING
+-- tasks.py               build . lint . test . run, identical on every platform
```

Components described in sections 9 through 15 but absent from this tree — the
API, worker protocol, event engine, recorder, map import — are designed and
documented, not built. STATUS.md is the authority on which is which.

---

## 20. Implementation phases

| Phase | Scope | State |
|---|---|---|
| 1 | engine core, bindings, build, CI | done |
| 2 | decode, detection, tracking, projection on real video | done |
| 3 | native console: camera view, plan view, evidence table | done |
| 4 | real detection model, measured on real footage | next |
| 5 | zones, rules, events, persistence | |
| 6 | multi-camera correlation, topology graph | |
| 7 | incident engine, replay, evidence export | |
| 8 | camera discovery, RTSP against physical hardware | |
| 9 | LAN workers, node pairing, distributed inference | |
| 10 | AI incident analyst | |
| 11 | hardening, offline packaging, deployment | |

The first executable milestone was the **vertical slice** — one thin path through
every layer that existed:

```
 real video file -> decode -> detect -> track -> ground projection -> map
```

That slice runs today, on a real encoded file through a real decoder, with no
mocks in the path. Its purpose was to prove the seams between subsystems rather
than to be feature-complete, and what it proved most usefully was where the seams
leaked: the vertical morphology kernel, the elliptical association gate, the
three-state motion model, and OpenCV's unbypassable RTSP timeout were all found
by building it rather than by designing it.

The full spine — correlation, risk, incident, evidence — remains the target:

```
 VIDEO -> DETECTION -> TRACKING -> SPATIAL CONTEXT -> TEMPORAL CONTEXT
       -> EVENT ANALYSIS -> MULTI-CAMERA CORRELATION -> RISK SCORING
       -> HUMAN REVIEW -> INCIDENT
```

Per-subsystem implementation state is tracked honestly in [STATUS.md](STATUS.md);
a UI existing never counts as a feature being complete.
