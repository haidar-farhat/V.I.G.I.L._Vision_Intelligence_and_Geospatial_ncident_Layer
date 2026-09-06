# Sentinel Vision v2 — architecture

*Describes what is built. Anything not built is a decision in
[DECISIONS.md](DECISIONS.md), and the state of every capability is generated
into [CAPABILITIES.md](CAPABILITIES.md) by `python tasks.py capabilities`.*

## 1. The one sentence

A single Python package, `vigil`, in four layers with a strict import
direction, one service boundary through which every change to a site passes
with a named principal, and one executable that is both the product and the
thing the tests run.

```
interfaces  (cli, console)      →  service  (site, runtime, auth, alerts, evidence)
                                        ↓                          ↓
                                   storage (sqlite)          adapters (decode, detect, record, keychain)
                                        ↓                          ↓
                                   domain  (geo, tracking, zones, events, incidents)  — pure, no I/O
```

An arrow is an allowed import. `domain` imports nothing above it and does no
I/O; `adapters` and `storage` never import `service`; `interfaces` never
import `storage` or `adapters` directly. `tests/test_layering.py` fails the
build on any other edge.

## 2. Layers

### domain — pure, deterministic, golden-tested
- `geo.py` — WGS84 local frame, haversine, bearings, `CameraPose`, rectilinear
  ray model, ground projection with 1-σ uncertainty, `PositionEstimate` that
  says *how* it was obtained, field-of-view wedge, point-in-polygon.
- `detection.py` — normalised `BoundingBox`, `Detection` with its ground
  contact point, `DetectorInfo` (what drew the conclusion).
- `tracking.py` — IoU + size-gated greedy association, cumulative
  confirmation, coasting, windowed ground speed. Behind `TrackerProtocol`.
- `zones.py` — `Zone` (ring, kind, watch list, entry/exit holds),
  `Presence` with hysteresis, `PresenceChange`.
- `events.py` — `Event` with `Evidence`, `Rule` protocol, `ZoneEntryRule`,
  `LoiteringRule`, `AfterHoursRule`.
- `incidents.py` — time-and-place association, identity union-find,
  `Correlator`, `Incident`, risk scoring.

### adapters — the outside world, each behind a small interface
- `decode.py` — `VideoSource` for files, `device:N`, RTSP; the egress guard
  (private addresses only); `LiveReader` thread with a one-slot latest
  frame.
- `detectors.py` — `MotionDetector` (MOG2), `OnnxDetector` (box or
  segmentation heads), letterbox, NMS, per-process model cache, telemetry
  silenced before the runtime loads.
- `recorder.py` — segment writer, file-safe names, sha256 per clip.
- `keychain.py` — OS keychain via `keyring` with an in-memory backend for
  tests.

### storage — one SQLite file, one owning thread
- `schema.py` — migrations with a way back, **users, audit and alerts in
  migration 1**.
- `store.py` — `Store`: entity persistence, append-only audit, integrity
  check on open, WAL, backup/verify/restore. Asserts at runtime that it is
  used from the thread that opened it.

### service — the application; the only place rules about *who* live
- `auth.py` — `Principal`, roles as permission sets, scrypt hashes, lockout.
- `site.py` — `SiteService`: every mutation takes a `Principal`; permission
  is checked here, the audit row is written here.
- `runtime.py` — `CameraWorker` (thread: decode → detect → track → presence
  → rules → record; bounded outbox) and `Runtime` (start/stop/poll:
  persist, correlate, health, alerts).
- `alerts.py` — raised once per condition until cleared, audited, fanned out
  to file / command / local webhook.
- `evidence.py` — incident export: JSON report, clips, manifest with hashes.

### interfaces
- `cli.py` — `vigil` command. Every service method has a command.
- console — **not built** (DECISIONS.md D-07).

## 3. Threads and ownership

| Thread | Owns | Talks to others by |
|---|---|---|
| main (CLI / UI) | `Runtime`, `Store`, `SiteService` | calls `Runtime.poll()` |
| one per camera | `VideoSource`, detector, tracker, presence, rules, recorder | a bounded `Outbox` (latest frame slot + event queue); never the store |
| alert delivery | nothing | daemon thread per delivery |

The store raises `ThreadOwnership` if touched from another thread. A worker
that cannot deliver drops the *oldest* result and counts the drop.

## 4. Identity and permission

Roles: `VIEWER` (site.view), `OPERATOR` (+ site.configure, analysis.control,
incident.export), `ANALYST` (site.view, incident.export, audit.read), `ADMIN`
(everything + users.manage). Code asks `principal.may(permission)`. The CLI
authenticates by `--as NAME` with the password on stdin, or runs as the
operating-system account when no user exists yet (`Principal.system`). A
store with no users is *open*: every principal may; the CLI says so on every
command until an account exists.

## 5. Alerts

Kinds: `camera.dark`, `recording.stopped`, `retention.shortfall`,
`analysis.thread_stuck`, `disk.low`. Raised by the runtime from `poll`;
cleared when the condition clears. Sinks from the environment:
`VIGIL_ALERT_FILE`, `VIGIL_ALERT_COMMAND`, `VIGIL_ALERT_WEBHOOK` (private
network only).

## 6. Invariants and how each is enforced

| Invariant | Enforced by |
|---|---|
| Zero WAN | `tools/offline_audit.py` over shipped source (no hostnames, no URLs); decode and webhook refuse public addresses |
| Secrets never persist | schema test: no credential-shaped column except `users.password_hash`; keychain refs only; argv passwords warned |
| Append-only audit | `Store` has no update/delete on `audit`; test asserts by AST |
| Every mutation names a principal | `SiteService` methods all take `principal`; test by signature scan |
| Nothing tested but unreachable | capability manifest test |
| Module size budget | `test_layering.py`: ≤ 800 lines per module |
| Store thread ownership | runtime check |

## 7. Running

```
python tasks.py test          # every suite
python tasks.py check         # offline audit + layering + tests (what CI runs)
python tasks.py capabilities  # regenerate CAPABILITIES.md
python -m vigil --help        # the product
```
