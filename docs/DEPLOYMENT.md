# Deployment

> **Status:** this describes a design that has not been rebuilt since the move to
> Python and Rust. The reasoning is intact and is what the implementation will
> follow; the code it refers to no longer exists. See [STATUS.md](../STATUS.md).

> **Status:** partly built. `python tasks.py package` produces standalone
> executables today, and there is a container image — see
> [docs/USAGE.md](USAGE.md) for how to use both. Still `PLANNED`: signed
> installers (MSI/NSIS, `.deb`/`.rpm`/AppImage, notarised `.dmg`), the offline
> update flow, and everything below that describes more than one machine. See
> [STATUS.md](../STATUS.md).

## Deployment shapes

### Standalone

One machine runs everything: desktop UI, API, database, event engine, recorder and
a local worker. Suitable up to roughly a dozen cameras on a machine with a
mid-range GPU.

```
 sentinel-console (Python + Qt)
   +-- supervises --> sentinel-api        127.0.0.1:8787
                        +-- database        sqlite3 (WAL)
                        +-- event-engine    in-process
                        +-- recorder        in-process
                        +-- worker (local)  child process
```

Nothing binds beyond loopback unless the operator asks. Works with the network
cable removed.

### LAN distributed

A control node plus worker nodes. Scale is horizontal: add workers, not bigger
constants. There is no hard-coded camera limit anywhere in the codebase.

```
                       CONTROL NODE
                 +-----------------------+
                 | desktop . api . db    |
                 +-----------+-----------+
                             | mTLS 1.3, pinned identities
        +--------------------+--------------------+
        v                    v                    v
   WORKER PC 1          WORKER PC 2          WORKER PC 3
```

Workers run **headless**. They keep decoding, detecting, tracking and recording
when the control node is unreachable, buffer locally, and reconcile on reconnect.

## Sizing

Design targets, not measurements — nothing has been benchmarked against real
video.

| Configuration | Target |
|---|---|
| Standalone, mid-range GPU | 8 x 1080p, 5-15 fps inference each |
| Standalone, high-end GPU | 16-32 cameras depending on model, codec and workload |
| Add a worker node | Add its capacity |

Adaptive inference matters more than raw throughput: idle cameras run at 2 fps and
step up only when something happens, so an average site sits far below its peak.

## Storage

Plan for recording, not for the database. The database is small; video is not.

```
data/
  sentinel.db        database (small: metadata only)
  recordings/        segmented video, retention-managed
  evidence/          content-addressed, protected from routine cleanup
  thumbnails/
  exports/
  logs/
models/              operator-imported model artifacts
map-data/            operator-imported map packages
```

Every path is configurable. Nothing is hard-coded to a machine-specific location.
The UI shows free space per configured directory and warns before exhaustion.

**Incident evidence is never removed by routine cleanup.** Size the disk for the
retention policy plus expected incident volume, and set a minimum-free-space
threshold that leaves recording room when cleanup is behind.

## Network

- Bind services to explicit LAN interfaces. Loopback by default.
- Cameras, nodes and any local LLM endpoint must sit in private address space; the
  egress guard refuses anything else.
- mDNS (`_sentinel._tcp.local`) for discovery, with manual IP entry as a fallback
  for networks that block multicast.
- **No WAN access is required for any function.** A firewall rule blocking all
  outbound traffic is a supported configuration and part of the acceptance criteria.

### Time

Timestamp accuracy matters for correlation. Use a LAN NTP source or an
administrator-configured local time server. **Internet NTP is not required.** Node
clock offset is reported in every heartbeat and surfaced in the UI; skew is shown,
never silently corrected.

## Installation

### What exists today

`python tasks.py package` produces a folder in `dist/SentinelVision` holding
three executables and the libraries they share. Copy the folder, run it, delete
the folder — nothing is installed, nothing is written to the registry, nothing
is downloaded. There is also a container image for headless analysis. Both are
documented in [docs/USAGE.md](USAGE.md).

It is `onedir` rather than a single self-extracting binary, and it is not UPX
packed. Both are deliberate: a onefile build unpacks 300 MB to a temporary
directory on every launch and can be blocked outright on a locked-down machine,
and a packed binary looks exactly like malware to every endpoint product an
operator runs. A security appliance that trips the antivirus is a security
appliance that gets uninstalled.

Neither is signed. On Windows that means a SmartScreen warning; on macOS it
means Gatekeeper refuses it outright.

### What is planned

| Platform | Format |
|---|---|
| Windows | MSI / NSIS installer, signed |
| Linux | `.deb`, `.rpm`, AppImage |
| macOS | `.dmg` (signed and notarised) |

The base installer contains the desktop application, local services and
configuration bootstrap. **AI models are not bundled** unless deliberately
configured: they are large, licence-encumbered, and hardware-specific. They are
imported after installation from local files.

## First run

```
Welcome -> create administrator -> choose operating mode -> choose storage
-> import offline map -> detect GPU -> configure AI -> discover cameras
-> add first camera -> place it on the map -> create first zone
-> run a test event -> complete
```

Every step is skippable and revisitable. A deployment with no map package, no
model and no cameras is a valid state that reports itself clearly rather than
failing.

## Updates

Offline by default:

```
Import update package -> verify signature -> show version -> confirm -> install -> restart
```

There is **no mandatory online updater** and no background update check. An
air-gapped site cannot use one, and a security appliance that fetches and executes
code from the Internet has a different threat model than this one claims.

Database migrations run on start-up and are reversible; see
[DATABASE.md](DATABASE.md).

## Backup and restore

Back up the database, configuration, rules, camera metadata, map metadata and the
model registry. Do **not** blindly duplicate recordings — they are large, already
retention-managed, and usually reproducible from the source only if you still have
it.

Restore validates the backup and never silently overwrites a live database.

## Operational checks

The diagnostics screen runs, and should pass, before a site is considered
commissioned:

```
GPU . AI runtime . camera connections . RTSP . ONVIF . storage . database
LAN . node health . map package . TLS certificates . clock synchronisation
```

Plus the network isolation panel, which should read:

```
WAN                  BLOCKED / NOT REQUIRED
LAN                  ACTIVE
Internet dependency  NONE
Cloud services       DISABLED
Telemetry            DISABLED
```

## Acceptance

A deployment is not commissioned until, **with the Internet physically
disconnected**, it can demonstrate: a camera added via discovery or RTSP, placed
on an offline map; local detection and tracking; a drawn zone whose crossing
creates an event; correlated events forming one incident; the incident opening
synchronised evidence; a grounded AI summary; a worker processing a camera over the
LAN; recovery after that worker is killed; an unauthorised user unable to alter
configuration; and a verifiable exported incident package.
