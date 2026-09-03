# Using Sentinel Vision

How to install it, run it, and read what it tells you.

> **What this build is.** A local-first multi-camera analyser. It decodes video,
> finds moving objects, keeps their identity across frames, works out where they
> are *on the ground*, decides whether that matters, groups everything one
> intrusion caused into a single incident, and hands you an evidence package you
> can verify later. It does all of that with the network cable unplugged.
>
> **What this build is not.** It does not record video yet, so an evidence
> package contains the record and not the footage. It has no server, no
> multi-machine mode and no user accounts. A camera attached to this machine
> works and has been run end to end; **no *network* camera has ever been
> contacted** — RTSP remains a code path. Nothing here has been run against real
> footage or trained detection weights. See [STATUS.md](../STATUS.md) for the
> honest state of every capability and [ROADMAP.md](../ROADMAP.md) for what is
> coming and in what order.

---

## Contents

1. [Install](#1-install)
2. [Your first five minutes](#2-your-first-five-minutes)
3. [The console, screen by screen](#3-the-console-screen-by-screen)
4. [The command line](#4-the-command-line)
5. [Docker](#5-docker)
6. [Cameras](#6-cameras)
7. [Detection models](#7-detection-models)
8. [Recording](#8-recording)
9. [Where your files are](#9-where-your-files-are)
10. [Logs, and the developer build](#10-logs-and-the-developer-build)
11. [When something is wrong](#11-when-something-is-wrong)
12. [What it will refuse to do](#12-what-it-will-refuse-to-do)

---

## 1. Install

Three ways, for three different people.

```mermaid
flowchart TD
    Q{"who are you?"}
    Q -->|"an operator"| A["<b>the packaged build</b><br/>unzip a folder, double-click<br/>no Python, no Rust, no install"]
    Q -->|"a developer"| B["<b>from source</b><br/>Python 3.12+ and a Rust toolchain<br/>everything runnable and testable"]
    Q -->|"running it on a server"| C["<b>Docker</b><br/>headless analysis<br/>network_mode: none"]

    style A fill:#1e3f2f,stroke:#4ade80,color:#e2e8f0
    style B fill:#1e3a5f,stroke:#4a9eff,color:#e2e8f0
    style C fill:#3f2f1e,stroke:#fbbf24,color:#e2e8f0
```

### The packaged build

```bash
python tasks.py package
```

That builds the Rust core, bundles everything, and leaves a folder in `dist/`:

```
dist/SentinelVision/
    SentinelVision.exe        the console
    SentinelVision-dev.exe    the console with a terminal and verbose logging
    sentinel.exe              the headless analyser
    ... about 300 MB of libraries the three share
```

**Run it from `dist/SentinelVision/`, and nowhere else.** PyInstaller also
leaves a `build/` directory — that is its scratch space, and the executables in
it are bootloaders with no libraries beside them. `python tasks.py package`
deletes them once the real bundle exists, because running one produces

```
Failed to load Python DLL '...\build\sentinel\_internal\python314.dll'.
LoadLibrary: The specified module could not be found.
```

which reads like a broken build rather than the wrong file. If you ever see that
message, an executable has been separated from the `_internal` folder next to it.

**Ship the whole folder.** The executables need `_internal` beside them, which is
what every Qt application ships. It is deliberately not a single self-extracting
`.exe`: that would unpack 300 MB to a temporary directory on every launch, leave
debris when it is killed, and on a locked-down machine can be blocked outright.
The folder carries a `HOW TO RUN.txt` saying all of this, for whoever receives it
without this document.

Nothing is installed, nothing is written to the registry, and nothing is
downloaded — at build time or at run time. Copy the folder, run it, delete the
folder.

PyInstaller is needed to *build* the package and is not a runtime dependency:

```bash
python -m pip install pyinstaller
```

### From source

Requires **Python 3.12+** and a **Rust toolchain** (stable).

```bash
python -m pip install -e "engine[dev]" PySide6

python tasks.py build      # build the Rust engine core
python tasks.py check      # audit, lint, build, test — what CI runs
python tasks.py console    # run the console
python tasks.py cli --help # run the headless analyser
```

`cargo` reaches the network once to resolve crates, and `pip` once for wheels.
Nothing after that does, ever — `python tasks.py audit` fails the build if any
shipped source file so much as names a destination off the site.

### Docker

See [§5](#5-docker). Short version:

```bash
docker compose build
docker compose run --rm analyse run /media/gate.mp4 --place 33.8938,35.5018,6,180,-22
```

---

## 2. Your first five minutes

The fastest path to seeing it work, with a video file rather than a camera.

### In the console

1. **Add a camera.** *Add camera* offers three things: a camera attached to
   this machine (found through the operating system's own device interface), a
   camera on the network, or a video file. The pane appears in the wall on the
   left.
2. **Place it.** *Place camera* → type where it is: latitude, longitude, mast
   height, which way it points, how far down it tilts. As you type, the dialog
   tells you **the band of ground that pose actually covers** — see
   [§3](#placement-is-not-optional-and-there-is-no-default).
3. **Add a zone.** *Add zone* puts a square of the radius you choose just past
   the near edge of what that camera can genuinely see.
4. **Press Start.**

The frame appears with its overlay, the ground beside it, one table row per
tracked object, and — if anything crossed a rule — incidents in the panel below.

### On the command line

```bash
# 1. Where can this camera actually see?
sentinel coverage --place 33.8938,35.5018,6,180,-22,62,36,90

#    ground covered   7.2 m to 85.8 m ahead
#    A 12 m square just past that near edge:
#      --zone "Restricted Area A:33.893736,35.501800;..."

# 2. Analyse, with the zone it just handed you.
sentinel run gate.mp4 --id gate \
    --place 33.8938,35.5018,6,180,-22,62,36,90 \
    --zone "Restricted Area A:33.893736,35.501800;33.893628,35.501930;33.893520,35.501800;33.893628,35.501670" \
    --export ./evidence

# 3. Look at what it concluded.
sentinel incidents
```

From a checkout, every `sentinel` above is `python tasks.py cli` or
`python -m sentinel`.

**Run `coverage` first.** The single most common way to get zero events is to
put a zone somewhere the camera cannot see. A camera's *stated range* is not its
coverage: a 6 m mast tilted 22° with a 36° vertical field covers **7 m to 86 m**
however large the number on the datasheet is, and it sees no ground at all at
the mast.

---

## 3. The console, screen by screen

```mermaid
flowchart LR
    subgraph window["the console window"]
        direction TB
        WALL["<b>camera wall</b><br/>one pane per camera<br/>detections, confirmed tracks<br/>and coasting tracks drawn<br/><i>differently</i>"]
        MAP["<b>plan view</b><br/>metric grid, camera footprints,<br/>uncertainty discs, trails, zones<br/><i>no tiles, no network</i>"]
        TABLE["<b>track table</b><br/>one row per object: class,<br/>confidence, duration, speed,<br/>heading, position, uncertainty,<br/>provenance"]
        INC["<b>incident panel</b><br/>one row per incident, expandable<br/>into risk factors, cross-camera<br/>links and a timeline.<br/><i>sorted by severity, not arrival</i>"]
    end

    style WALL fill:#1e3a5f,stroke:#4a9eff,color:#e2e8f0
    style MAP fill:#1e3f2f,stroke:#4ade80,color:#e2e8f0
    style TABLE fill:#3f2f1e,stroke:#fbbf24,color:#e2e8f0
    style INC fill:#4c1d24,stroke:#f87171,color:#e2e8f0
```

### Placement is not optional, and there is no default

Until a camera is placed, objects are tracked and reported as **not placed**.
There is deliberately no default position: a nominal origin produces coordinates
that look exactly like measured ones, and somebody gets sent to them.

The placement dialog reports the ground band as you type because the number on a
camera's datasheet is not what it can see:

| you type | what it can actually see |
|---|---|
| 6 m mast, −22° pitch, 36° vertical field, range 90 m | **7.2 m to 85.8 m** ahead |
| the same camera at −45° | roughly 2.5 m to 12 m — a much smaller patch, much closer |
| the same camera level or tilted up | **nothing.** No ground is in frame, so nothing can be located |

The footprint drawn on the plan view is an **annular sector**, not a pie slice —
a downward-tilted camera cannot see the ground at its own mast, and drawing the
slice would claim coverage it does not have.

### Reading the plan view

| what you see | what it means |
|---|---|
| A dot with a circle around it | an object, and how well its position is known. The circle is 1σ horizontal uncertainty and it grows **super-linearly** with distance |
| A large circle centred on the camera | the projection failed and the system fell back to "something is happening at this camera", which is true. It is not a position |
| No dot at all | the camera is not placed. The object is still tracked and still in the table |
| A dashed box in the camera pane | a *coasting* track — held open through frames with no detection supporting it. A weaker claim than a solid one, drawn differently so you can tell |

### Reading an incident

Every incident expands into the reasoning that produced it:

```
inc_ABA008BFC4EF5A862535
HIGH     risk 75/100    6 events    1 camera
4 objects in Restricted Area A
  · +45.0  severity: most serious event is HIGH (ZONE_ENTRY)
  · +20.0  group: 4 distinct objects involved
  · +10.0  location: occurred in a restricted zone
```

The score never appears without its factors. A bare number invites you to
calibrate against it without understanding it, and then to ignore it the first
time it is wrong.

**Three cameras seeing one person is one incident, not three alerts.** That
collapse is the product. On the reference scene, 13 events become 1 incident —
92% less for a person to read.

### Exporting evidence

*Export incident* writes a folder containing the full record, a report a person
can read without any tooling, and a SHA-256 for every file plus one for the
manifest itself.

**Record the manifest digest separately.** It is what makes the package
checkable later by somebody who has only the folder.

---

## 4. The command line

The console is a *viewer*. `sentinel` is the same pipeline with no window, which
is what a container, a scheduled job, or a developer chasing one file wants.

It is **not a daemon**. It processes the sources it is given, in order, to
completion, and exits. Every frame of a file is processed, so a replay
reproduces the original result exactly.

```
sentinel [-v|-q] [--database PATH] <command>

  run <source>...      analyse and record what happened
  devices              cameras attached to this machine
  incidents            list what has been recorded
  export <id> --to DIR write one incident out as evidence
  coverage --place ... what a placed camera can actually see
  where                every path this build uses
```

### `run`

| option | meaning |
|---|---|
| `--id NAME` | camera id, once per source. Default `cam-01`, `cam-02`, … |
| `<source>` | a video file, an `rtsp://` URL, or `device:N` for a camera attached to this machine |
| `--place lat,lon,height,heading,pitch[,hfov,vfov,range]` | once, or once per source. Without it, objects are tracked but not located |
| `--zone "Name:lat,lon;lat,lon;lat,lon[;…]"` | a restricted polygon, three vertices up. Repeatable |
| `--detect-scale 0.75` | detection resolution. 0.75 is the default: **1.7× faster and slightly better recall**, because the downscale is a mild denoise |
| `--for SECONDS` | stop a live source after this long |
| `--frames N` | stop after N frames |
| `--export DIR` | write an evidence package per incident |
| `--node NAME` | node id recorded on every event |

**A live source has no end, so bound it.** A camera runs until the stream stops
or you press Ctrl-C at an interactive terminal — and a scheduled job or a
container has no terminal. On Windows an interrupt sent from outside does not
reach a Python process at all, measured, so there is no way to stop an unbounded
run short of killing it. `--for` and `--frames` are how a headless run ends
cleanly with everything it found recorded. A file ignores both: it stops on its
own.

Two cameras of one world, correlated into one incident:

```bash
sentinel run north.mp4 south.mp4 \
    --id north --id south \
    --place 33.8938,35.5018,6,180,-22 \
    --place 33.8942,35.5018,6,0,-22 \
    --zone "Yard:33.8940,35.5016;33.8940,35.5020;33.8936,35.5020;33.8936,35.5016"
```

Correlation runs across **every** source, deliberately. A camera correlating only
its own events raises one incident per camera for one intrusion, which is the
duplication the whole stage exists to remove.

### Placement is validated, not clamped

Every one of these is refused with a message, not silently corrected — because a
camera the system quietly "fixes" reports positions indistinguishable from
measured ones:

- a latitude or longitude out of range
- a mast of zero or negative height
- a pitch that is level or tilted up: it sees no ground to project onto
- five fields, then six — it takes five or eight
- more `--id` or `--place` than there are sources

### Exit codes

| code | meaning |
|---:|---|
| `0` | done |
| `1` | something failed. Re-run with `--verbose` for the traceback |
| `2` | the command line was wrong. Nothing was opened |
| `130` | interrupted. Everything already written stays written — each event and incident is its own transaction |

---

## 5. Docker

Every service runs with **`network_mode: none`**. That is not hardening, it is
the product's central claim under test: if any of it needs the network, this
system does not do what it says.

```bash
docker compose build

# Put footage in ./media, then:
docker compose run --rm analyse run /media/gate.mp4 \
    --place 33.8938,35.5018,6,180,-22 \
    --zone "Yard:33.8940,35.5016;33.8940,35.5020;33.8936,35.5020" \
    --export /evidence

docker compose run --rm incidents

# The offline acceptance proof: the whole Python suite, no network at all.
docker compose run --rm verify
```

| path | what goes there |
|---|---|
| `./media` | your footage, mounted read-only |
| `./evidence` | where exported packages land |
| the `data` volume | the database. **Named**, so it survives a container being replaced — a security system whose records vanish on `docker compose down` is not one |

**There is no server in the image.** No control plane, no REST API, no daemon.
The container runs an analysis and exits; nothing listens on a port, and if you
find something that does, that is a bug.

The one service that takes a network is `cameras`, commented out by default,
because a camera is on a LAN. It uses `network_mode: host` so RTSP over UDP works
without port games — and the runtime egress guard still refuses any address
outside RFC 1918 / 4193, so "has a network" does not become "can reach the
Internet".

The build stage uses the network to resolve crates and wheels, exactly as a
developer's machine does. Nothing after it may.

---

## 6. Cameras

There are three kinds of source, and they are genuinely different things rather
than three spellings of one.

```mermaid
flowchart TD
    Q{"what is the camera?"}
    Q -->|"attached to this machine"| L["<b>device:N</b><br/>found through the OS's own<br/>device interface, opened through<br/>its native capture API"]
    Q -->|"on the network"| N["<b>rtsp://…</b><br/>a URL, usually with a password.<br/>Must resolve to a private address"]
    Q -->|"footage"| F["<b>a file</b><br/>not a camera. The only source<br/>that is <i>evidence</i>: every frame,<br/>in order, reproducibly"]

    style L fill:#1e3f2f,stroke:#4ade80,color:#e2e8f0
    style N fill:#3f2f1e,stroke:#fbbf24,color:#e2e8f0
    style F fill:#1e3a5f,stroke:#4a9eff,color:#e2e8f0
```

### A camera attached to this machine

A USB or built-in camera is not a file and not a network stream: it is a device
the operating system owns, and the only honest way to find one is to ask the
operating system.

| Platform | Enumerated through | Opened through |
|---|---|---|
| **Windows** | `Win32_PnPEntity` — the PnP device registry | Media Foundation, falling back to DirectShow |
| **Linux** | `/sys/class/video4linux` — the V4L2 device tree | Video4Linux2 |
| **macOS** | `system_profiler SPCameraDataType` | AVFoundation |

No third-party dependency is added for any of it. Each is a query the platform
already answers, and all three are local.

```bash
sentinel devices            # what the OS reports. Opens nothing.
sentinel devices --probe    # opens each one to confirm which index is which

sentinel run device:0 --id front-door --place 33.8938,35.5018,3,90,-15 --for 60
```

In the console, **Add camera → This machine** shows the same list.

**Listing does not switch a camera on.** Enumeration reads metadata and captures
nothing, so opening the dialog does not light the webcam light and, on macOS,
does not raise a permission prompt for a camera nobody asked to use. *Detect* is
a separate button because opening a camera is a deliberate act — and on macOS it
is what triggers the permission prompt, which is the right place for that to
happen.

#### Why an index can say "assumed"

The operating system knows a camera's **name**. OpenCV opens one by **index**.
There is no supported way to map between them, so the pairing is an assumption
and the listing says so:

```
0: Integrated Camera  (index assumed)
```

`--probe` (or *Detect*) opens each index and turns the assumption into a fact:

```
0: Integrated Camera — 640x480
    use    device:0
    id     USB\VID_5986&PID_2174&MI_00\7&3B5246B5&1&0000
    opens  DirectShow
```

This matters. Two identical webcams are indistinguishable by name, and a USB bus
can enumerate differently after a reboot. A system that guessed, and was wrong,
would attribute an intrusion to the wrong side of a building. **The only way to
tell two identical cameras apart is to look at the picture** — so add the
camera, press Start, and check the frame is the one you meant.

The stable hardware id is recorded alongside the index precisely because the
index is not stable and the id is.

#### Windows needs two interfaces, and says which one worked

Media Foundation is the modern interface and the default. DirectShow still opens
devices that Media Foundation refuses outright — measured on the development
machine, where the integrated camera opens on DirectShow and not on Media
Foundation. The fallback is a fallback, and which one succeeded is recorded in
the source's provenance rather than forgotten, because an operator whose camera
works on one machine and not another needs to know which one each took.

#### Not every listed camera is a camera you can use

A Windows Hello infrared sensor lists as a camera and opens on nothing. A Linux
UVC camera exposes a metadata node next to its capture node, and the metadata
node opens happily and produces no image — those are filtered out by the
device's own reported capabilities rather than by guessing from the name. A
camera already in use by another application will not open. All of these are
ordinary, and `--probe` is what distinguishes "listed" from "works".

### A camera on the network

> **No physical camera has ever been contacted by this code.** RTSP is a code
> path, not a verified capability. Real cameras deviate from the specifications
> in ways local testing cannot anticipate. Treat this section as the design and
> expect surprises.

```bash
sentinel run "rtsp://admin:PASSWORD@192.168.1.64:554/Streaming/Channels/101" \
    --id gate --place 33.8938,35.5018,6,180,-22 --for 300
```

### Your password does not go anywhere

The URL is held in one private field, read in exactly one place — the call that
opens the capture — and everything downstream is given a redacted copy:

```
rtsp://admin:hunter2@10.0.0.5/s   →   rtsp://admin:***@10.0.0.5/s
```

That redacted form is what appears in the camera id, the display URL, **every
error message**, the log, the database, the interface and every export. The
replacement is a fixed `***` and never a run of stars matching the password's
length, because a password's length is part of the password.

Nine awkward URL shapes that each caused a real leak are now tests: a password
containing `@`, a password with no username, an IPv6 literal, a non-numeric
port, a credential in the query string, no scheme at all.

**Nothing persists a camera password today** — the keychain integration is
designed and not built, so the safest thing is that there is nowhere to put one.

### The egress guard

Every address a camera host resolves to must be loopback or inside RFC 1918 /
RFC 4193. One public answer refuses the whole connection, with the address
named — because an operator who genuinely means to reach a routable host needs
to know exactly what stopped them and that it was deliberate.

A name that does not resolve is refused rather than resolved onward. The system
will not reach the Internet to find out whether it is allowed to reach the
Internet.

### Unreachable cameras fail fast

A socket probe runs before the decoder is involved, because OpenCV's own RTSP
connect timeout is a hard-coded 30 seconds that its documented FFmpeg options do
not change — measured, not assumed. Thirty seconds per camera means a node with
twenty cameras behind a switch that has just lost power takes ten minutes to
work out that none of them are there.

Faults are reported **in place**, never in a modal dialog. Twenty cameras drop
together when a switch loses power, and twenty dialogs is not a user interface.

---

## 7. Detection models

Out of the box the system detects **motion**. That is free, needs nothing, and
has two limits worth stating plainly, because they decide what the rest of the
product can conclude:

- It does not classify. Every track is `unclassified`, and the system says so
  rather than guessing.
- It cannot see anything that has stopped moving. A person standing still
  disappears.

Point it at a model and both limits go away:

```bash
sentinel run device:0 --model models/yolov8n-seg.onnx
```

The console takes the same flag, and with no flag it uses the first `*-seg.onnx`
it finds in the models directory:

```bash
sentinel-console --model models/yolov8n-seg.onnx
sentinel-console --no-model          # motion only, whatever is installed
```

Which detector you get is decided by **reading the file**, not by a flag:

| The model has | You get | What it can conclude |
|---|---|---|
| *no model* | motion | something changed here |
| one output | detection | a *person* is in this box |
| two outputs | segmentation | a *person* is this shape, and touches the ground *here* |

That last column is the whole reason to bother. Every position this system
reports comes from one point per object — where it meets the ground — and
without a mask that point is the bottom-centre of a rectangle, which is only
correct for someone upright, unoccluded, and tightly boxed. With a mask it is
the object's own lowest pixel. The console draws that point as a small dot in
the track's colour, so you can see for yourself where the map position came
from: on the feet with a model, at the bottom of the box without one.

The line beside the toolbar always names what is actually running, with the
model's SHA-256 abbreviated. That digest is recorded on every event, so a
detection can be traced to the exact file months later.

### Getting a model

**Nothing is ever downloaded by the product.** Not on first run, not as a
fallback, not ever. You obtain a model once, on a machine with a network:

```bash
pip install ultralytics onnxslim
python devtools/export_model.py --task segment --size n
```

That writes `models/yolov8n-seg.onnx` (about 14 MB) and prints its SHA-256.
Copy the file to the offline machine and put it in the models directory —
`SENTINEL_MODELS_DIR` if you have set it, otherwise `models/` beside the install.
`ultralytics` is a developer tool and is deliberately **not** a dependency of
the product; see `devtools/README.md` for why that separation matters.

### What it costs

On an ordinary laptop CPU, YOLOv8n-seg runs at roughly **11–14 fps** against a
640×480 webcam, against a few hundred for motion. That is the trade: motion is
nearly free and tells you almost nothing; segmentation costs a core and tells
you what the object is and where it stands.

---

## 8. Recording

Off by default, because writing video is the single most expensive thing this
system can do to a disk. Turn it on per run:

```bash
sentinel run rtsp://admin:pw@192.168.1.64/stream --id gate \
    --place 33.8938,35.5018,6,180,-22 \
    --record --for 3600
```

`--record` with no value writes under the data directory
(`sentinel where` shows exactly where); `--record D:/footage` writes to the disk
you point it at — which is the normal case, because of the table below.

### The price, measured

| | 640×480 · 15 fps |
|---|---|
| One camera | ~12.7 MiB/minute · **~17.5 GB/day** |
| Sixteen cameras | **~280 GB/day** |
| Encoding cost | ~800 fps — the writer is never the bottleneck |

Recording uses `mp4v` (MPEG-4 Part 2) in ordinary `.mp4` files that play in any
player. **It is deliberately not H.264**, which would be about 4× smaller:
OpenCV's H.264 encoder is a DLL it offers to *download*, and this system never
downloads anything — that rule is enforced by the build, not by preference. The
4× is the honest price of the promise.

### Segments, not one growing file

Each camera writes bounded clips — 60 seconds each by default,
`--segment-seconds` to change it. That number is also **the most a power cut
can cost you**: a clip killed mid-write may not play at all, so shorter segments
bound the loss and longer ones make fewer files.

Filenames carry the camera and the wall-clock time
(`gate_20260902-141116_00000000.mp4`), every finished clip is SHA-256 hashed
the moment it closes, and all of it is indexed in the database so evidence and
retention can find it.

For a **file** being analysed, every frame is recorded — a replay is evidence,
and evidence with three frames in four missing is not. For a **live camera**,
a writer that falls behind drops frames rather than building the backlog that
kills the process — and counts every drop, because a recorder silently
discarding input is the worst failure a security system can have.

### Retention — the disk is finite

```bash
sentinel retention                      # report: what WOULD be deleted
sentinel retention --keep-days 7 --apply
```

Reporting is the default; nothing is deleted without `--apply`. The policy has
three independent bounds — age (`--keep-days`, default 14), total size
(`--max-gib`), and free space (`--min-free-gib`, default 5, because a disk at
100% stops the database too, not just the recording).

Two rules that do not bend:

- **A segment an incident depends on is never deleted** — however old, however
  full the disk. If the only way to meet the policy would be to delete
  evidence, the policy goes unmet and the command says so and exits non-zero.
- **Every deletion is audited.** "Where is the footage from the 3rd?" has an
  answer: deleted by retention on the 17th, with the camera, frame count and
  size on the audit row.

### Footage in evidence packages

An exported incident includes its clips — with a lead-in before the incident
opened, because an intrusion event fires *after* somebody is already inside the
zone, and the footage that explains it starts earlier. Adjust the window with
`--lead` and `--trail`:

```bash
sentinel export inc_ABA008BFC4EF5A862535 --to ./evidence --lead 60 --trail 30
```

Exporting also **preserves** the clips it used: they are marked as evidence in
the index and retention will not delete them afterwards, however old they get.
That preservation is itself audited, so "why can this segment not be deleted?"
has an answer on the record.

The package's `footage.json` states, per camera, exactly what the clips cover
— and **what they do not**:

```json
{
  "camera_id": "gate",
  "covered_fraction": 0.42,
  "complete": false,
  "gaps": [ { "from": "…14:11:08", "to": "…14:11:16", "seconds": 8.6 } ]
}
```

A package holding forty seconds of a ninety-second incident plays, verifies
clean, and misleads — unless it says so. This one says so. A camera the
incident names that has *no* footage at all is also listed, with 0% coverage,
because "this camera recorded nothing" is a finding, not an absence.

### What recording does not do yet

Honesty, per [STATUS.md](../STATUS.md): recording is **CLI-only** — the console
cannot switch it on yet. There is no playback or timeline scrubbing inside the
application (the clips are ordinary `.mp4`; any player opens them). There is no
motion- or event-triggered mode — continuous recording came first because,
with it, pre-event footage is simply already on disk. And nothing yet runs
retention on a schedule; run `sentinel retention --apply` from your scheduler
of choice.

---

## 9. Where your files are

```bash
sentinel where
```

```
data directory   C:\Users\you\AppData\Local\SentinelVision
database         C:\Users\you\AppData\Local\SentinelVision\sentinel.db
logs             C:\Users\you\AppData\Local\SentinelVision\logs
evidence         C:\Users\you\AppData\Local\SentinelVision\evidence
packaged build   True
```

| platform | default |
|---|---|
| Windows | `%LOCALAPPDATA%\SentinelVision` |
| macOS | `~/Library/Application Support/SentinelVision` |
| Linux | `$XDG_DATA_HOME/SentinelVision`, else `~/.local/share/SentinelVision` |

**`SENTINEL_DATA_DIR` overrides all of it**, which is the normal case for a
security appliance with a dedicated disk — continuous video does not belong on
the system volume.

Nothing is stored beside the code. A packaged install lives in Program Files or
`/usr/lib`, which the account running it cannot write to, and a security system
that silently fails to record because its install directory is read-only is
worse than one that refuses to start.

### Database maintenance

```bash
python tasks.py db           # what schema version is applied, and what is pending
python tasks.py db-migrate   # apply pending migrations
python tasks.py db-rollback  # undo the most recent one
```

`db` is read-only and deliberately so — applying a migration is something you do
knowingly. Every migration carries a way back, because an upgrade that cannot be
undone on a machine with no Internet and no spare hardware is a gamble.

---

## 10. Logs, and the developer build

| variable | effect |
|---|---|
| `SENTINEL_LOG_LEVEL` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. Turns a packaged build up in the field without a rebuild |
| `SENTINEL_LOG_FILE` | a specific path, or `""` to write nothing to disk — which is what a container wants, since its log is stdout |

The file rotates at 5 MB and keeps five, so it cannot become the thing that
fills the disk.

**`--quiet` silences the terminal, not the record.** The file is never quieter
than INFO: an operator who silences the console and then has an outage still
needs the log to say what happened.

### `SentinelVision-dev.exe`

Not a debug build — the same code with its output visible, plus `--verbose`
forced on.

It exists because a packaged Qt application on Windows has nowhere to print. An
exception raised before the window appears leaves no trace at all, and *"it just
closes"* is the least actionable bug report there is. The developer executable
turns that into a stack trace on a terminal you can read and copy.

```
17:24:51.676  INFO     sentinel.pipeline   MainThread   pipeline.py:251
              gate: analysis started (detector MOG2 background subtraction at
              0.75 scale, placed, 1 zone(s), 4 rule(s))
```

The operator format is timestamp / level / module / message. The developer
format adds milliseconds, the thread, and the file and line that logged it.

### A log line can never carry a password

Every record passes a redacting filter before it is formatted — the message, the
interpolation arguments, and the exception text. That is a backstop rather than
the defence: messages are built from the redacted URL in the first place. But it
is exactly the right backstop for the case that actually happens, which is a
library raising an exception containing a URL nobody sanitised.

There is no syslog handler, no HTTP handler, and no `logging.config` file
loading. A log configuration has no business being able to import arbitrary
modules or open a socket.

---

## 11. When something is wrong

| what you see | why | what to do |
|---|---|---|
| **No events at all** | The zone is somewhere the camera cannot see. This is the most common cause by a distance | `sentinel coverage --place …` and use the zone it hands you |
| **Objects tracked but "not placed"** | The camera has no pose. There is no default, deliberately | Place it — console *Place camera*, or `--place` |
| **A huge uncertainty circle at the camera** | Projection failed, so it fell back to "something is happening at this camera" | Check the pitch. A camera near level sees the horizon, where a pixel is hundreds of metres |
| **Objects reported near the horizon with metre-scale error** | Uncertainty grows **super-linearly** with distance; this is honest, not broken | Put zones near the camera. That is where the geometry is worth anything |
| **A person who stops moving disappears** | Background subtraction cannot see a stationary object. This is what background subtraction *is*, not a bug | Known and measured: 0.49 recall on the loiterer against 0.74 and 0.65. The real answer is a trained detector — [ROADMAP](../ROADMAP.md) 1.4 |
| **3 people reported as 4 objects** | Appearance-free tracking. When two people cross, box geometry alone cannot tell which is which | Known and bounded by tests. [ROADMAP](../ROADMAP.md) 1.3 |
| **`The Rust engine core was not found`** | The core is not built, or you moved the library | `python tasks.py build`, or set `SENTINEL_CORE_LIB` |
| **`reports ABI version N; this build expects M`** | A stale core beside a newer engine | `cargo build --release` in `core/`. The refusal is deliberate: calling a function whose signature moved produces plausible, wrong geometry |
| **`does not export sentinel_abi_version`** | That library is not the engine core, or is far older | Same fix |
| **`Refused to contact "…"`** | The egress guard: that address is not on a private network | Intended. Use a LAN address. There is no override, and there will not be one |
| **`Failed to load Python DLL '…\_internal\python3xx.dll'`** | An executable was run away from the `_internal` folder beside it — most often one out of `build/`, which is PyInstaller's scratch directory and not the product | Run from `dist/SentinelVision/`. `python tasks.py package` now deletes those stubs |
| **The console exits immediately, packaged** | An exception before the window appeared | Run `SentinelVision-dev.exe` — that is what it is for |
| **A camera is listed but will not open** | In use by another application, blocked by a privacy setting, or not a capture device at all — a Windows Hello IR sensor lists as a camera and opens on nothing | `sentinel devices --probe` shows which ones actually open |
| **`sentinel devices` finds nothing** | Some cameras do not appear in the device registry but do open | `sentinel devices --probe` scans indices directly |
| **The wrong camera opened** | The index was assumed from enumeration order, not confirmed | `--probe`, or press Start and look at the picture. Two identical cameras cannot be told apart any other way |
| **macOS refuses to open a camera** | The operating system's privacy control. The prompt appears the first time something opens one | Allow it in System Settings → Privacy & Security → Camera |
| **A headless run against a camera never ends** | A live source has no end, and Ctrl-C is not available to a scheduled job | `--for SECONDS` or `--frames N` |
| **Docker: `no such file /media/…`** | The file is not in the mounted directory | Put footage in `./media`; the container sees it as `/media` |
| **Docker: `device:0` will not open** | A container sees no hardware it was not given | Pass the device through: `--device /dev/video0`. Linux hosts only — Docker Desktop on Windows and macOS cannot forward a USB camera into a container |

Still stuck: run the developer executable or `sentinel --verbose`, then send the
folder that `sentinel where` prints. It contains the log and nothing else — and
the log cannot contain a camera password.

---

## 12. What it will refuse to do

Not limitations. Design rules, each enforced by something other than intention.

| It will not | Why |
|---|---|
| Reach the Internet | Zero WAN, checked three ways: a static source audit at commit, an offline CI job that drops all outbound traffic, and the runtime egress guard |
| Download a model | Models are supplied by you and placed in `models/`. A missing model is an error that says so, not a cue to go and find one |
| Send telemetry | Nothing here, and onnxruntime's own telemetry is switched off explicitly — the promise has to hold for every dependency |
| Identify a person | No facial recognition, no biometrics, no identity database. Objects are tracked; people are not identified. Enforced by absence |
| Control a camera or take a security action | The AI is an analyst. The operator is the decision maker |
| Invent a position | A projection that cannot be made returns nothing. It never guesses and never clamps |
| Report speed it cannot measure | Speed is withheld below 1.2 s of observation, because dividing a distance by one frame interval amplifies position error fivefold |
| Claim a class it did not find | The motion detector emits `UNCLASSIFIED` and the console will not label it |
| Edit an audit entry | There is no method to, and a test fails if somebody adds one |
| Write outside its own directories | Map, model and video imports are sandboxed. A path escaping the destination is **refused**, not sanitised — quietly rewriting one hides both the bug and the attack |

---

## See also

| Document | What it covers |
|---|---|
| [STATUS.md](../STATUS.md) | The honest state of every capability, with the measurements |
| [ROADMAP.md](../ROADMAP.md) | What is left, in the order it should be built |
| [docs/OVERVIEW.md](OVERVIEW.md) | A visual walk-through of the whole system |
| [docs/SECURITY.md](SECURITY.md) | Threat model, credential handling, the three zero-WAN enforcers |
| [docs/DEVELOPMENT.md](DEVELOPMENT.md) | Working on the code |
| [docs/TESTING.md](TESTING.md) | What is tested, and what only each layer can catch |
