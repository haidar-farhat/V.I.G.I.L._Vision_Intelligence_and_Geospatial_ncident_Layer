# Sentinel Vision

**Local-first AI multi-camera security & situational awareness platform.**
Repository codename: **VIGIL** — Vision Intelligence and Geospatial Incident Layer.

Sentinel Vision turns ordinary IP cameras into a local event intelligence system.
It runs entirely on your own hardware, on your own network, with the Internet
disconnected.

> The camera is a sensor. The AI is an analyst. The operator is the decision maker.

---

## What it is for

The system is built on the assumption that **nobody is watching the cameras**.
Its job is to extract signal from continuous video, attach spatial and temporal
context to it, correlate it across cameras into a small number of reviewable
incidents, and present each one with the evidence that produced it.

```
VIDEO -> DETECTION -> TRACKING -> SPATIAL CONTEXT -> TEMPORAL CONTEXT
      -> EVENT ANALYSIS -> MULTI-CAMERA CORRELATION -> RISK SCORING
      -> HUMAN REVIEW -> INCIDENT
```

The measure of the system is how *few* incidents it raises, not how many
detections it makes. Three cameras seeing the same person produce **one**
incident, not three alerts.

It produces statements like:

> Three people entered Restricted Zone A after scheduled hours.

It never produces statements like "this is a criminal". Every conclusion carries
its timestamp, camera, evidence, confidence, triggering conditions, related
events and model version.

---

## Non-negotiables

- **Zero WAN.** No feature requires the Internet. No cloud services, no telemetry,
  no external map tiles, no CDN assets, no auto-updater. Block all outbound WAN
  traffic and the system keeps working. Enforced by a runtime egress guard and a
  build-time lint, not by intention.
- **Credentials never leak.** Camera passwords live in the OS keychain and travel
  in a wrapper that renders `[redacted]` through every serialisation path.
- **No AI claim without evidence.** Reports that cite evidence they were not given
  are rejected before an operator ever sees them.
- **Privacy by default.** No facial recognition, no biometric identification, no
  identity database. Objects are tracked; people are not identified.
- **Graceful degradation.** Losing the GPU, the AI, the database, the map or the
  control node never stops recording.

---

## Current state

See **[STATUS.md](STATUS.md)** for a per-capability breakdown. In short: the
domain core, geometry, tracking, correlation, incident engine, security controls,
database, an end-to-end vertical slice, the ONVIF and RTSP protocol layers, and
offline map packages with camera coverage analysis, and the control plane (REST,
WebSocket, authentication) are implemented and tested. Video decode, real
inference, the Tauri shell and distributed mode are designed but not built.

The API runs but nothing is connected to it yet: the desktop still reads a
generated snapshot, and no worker has spoken to it.

Nothing here has been run against a physical camera. The camera protocols are
tested against mock devices written from the specifications, which is a real bar
and not the same one.

**619 tests.** Clean typecheck under `strict` + `noUncheckedIndexedAccess`. Clean
architectural lint.

---

## Quick start

Requires **Node 22.6+** (Node 24 recommended). Nothing else — the core has zero
third-party runtime dependencies, and the embedded database ships inside Node.

```bash
npm install          # dev dependencies only: TypeScript and Node types
npm test             # 619 tests, no network
npm run lint         # architectural invariants
npm run typecheck    # strict TypeScript across the workspace
```

### See it work

```bash
npm run slice        # run the full pipeline end to end and print the incident
npm run slice quiet  # the same pipeline on an empty site: raises nothing
```

`npm run slice` runs a scripted scenario — three people walk a perimeter road
after hours, cross a restricted zone, pass through a gap in camera coverage, and
are reacquired near a substation — through the **production** pipeline. Only the
camera and the detector are simulated. It prints the detections, the tracks, the
measured spatial error against ground truth, the scored cross-camera hand-offs,
the single incident produced, its risk breakdown, its timeline, and the
evidence-bound analyst report.

Typical output:

```
tracks per camera        3 / 3 / 3          (three people, three cameras)
mean position error      0.52 m
within stated 2-sigma    100.0 %
associations             6 hand-offs, 93-98%
events                   9
incidents                1                  <- the whole point
  INC-53CB4075B8  [CRITICAL]  3 people in Restricted Zone A (2 cameras)
```

### Run the stack

```bash
npm run up           # API + realtime + embedded database, on loopback
```

Prints where it is listening and a development administrator password. Needs no
network, no services, and no configuration.

### Database

```bash
npm run db status
npm run db migrate
npm run db rollback
```

---

## Development commands

Identical on Windows, Linux and macOS — everything routes through Node, so there
is one set of instructions and no shell-script pair to drift apart.

| Command | Does |
|---|---|
| `npm test` | Full test suite |
| `npm run lint` | Layering, zero-WAN, secrets, placeholders, erasable syntax |
| `npm run typecheck` | Strict TypeScript across the workspace |
| `npm run slice` | End-to-end vertical slice with printed results |
| `npm run simulator` | Camera simulator on its own |
| `npm run up` | Start the local stack (API, realtime, database) |
| `npm run db <cmd>` | `migrate` / `rollback` / `status` |
| `npm run build` | Typecheck, then build the desktop bundle |

---

## Repository layout

```
apps/desktop/       Tauri shell + React operator UI
services/
  api/              REST + WebSocket control plane
  worker/           headless edge: ingest -> infer -> track -> observe
  inference/        model runtimes and device discovery
  recorder/         segmented recording and retention
  event-engine/     rules, correlation, risk, incidents
packages/
  shared-types/     the domain model
  protocol/         versioned wire schemas
  geometry/         geodesy, polygons, projection, FOV, zones
  tracking/         track lifecycle and cross-camera association
  ai/               model abstraction, registry, analyst guardrails
  maps/             PMTiles reading, style validation, package integrity
  database/         driver, migrations, schema
  security/         Secret<T>, redaction, egress guard, authz
  test-utils/       deterministic clock, seeded RNG, fixtures
simulator/          synthetic cameras, actors, scenarios
models/             operator-imported models (never committed)
map-data/           operator-imported map packages (never committed)
```

`packages/*` have **zero third-party runtime dependencies**. For a security
product the supply chain is part of the threat model, and it keeps the domain
auditable by reading and testable in milliseconds.

---

## Documentation

| Document | Covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, processes, ERD, protocol, security model, data flow, AI and map architecture |
| [STATUS.md](STATUS.md) | What is actually built, per capability |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Working on the codebase |
| [docs/SECURITY.md](docs/SECURITY.md) | Threat model and controls |
| [docs/CAMERAS.md](docs/CAMERAS.md) | Discovery, ONVIF, RTSP, and diagnosing a camera that will not add |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Wire format between nodes |
| [docs/DATABASE.md](docs/DATABASE.md) | Schema and migration policy |
| [docs/AI.md](docs/AI.md) | Model pipeline and analyst guardrails |
| [docs/MAPS.md](docs/MAPS.md) | Offline map architecture |
| [docs/TESTING.md](docs/TESTING.md) | Testing strategy |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Packaging and deployment |

---

## Licence

See `LICENSE`. Third-party model and map-data licences are tracked in the model
registry and map package metadata; nothing is redistributed without checking them.
