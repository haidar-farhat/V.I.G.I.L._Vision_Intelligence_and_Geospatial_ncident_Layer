# Sentinel Vision v2

Local-first multi-camera incident intelligence: ordinary cameras in, a small
number of reviewable incidents out, with the evidence that produced each one,
and no route to the Internet at any point.

Read [REVIEW_OF_V1.md](REVIEW_OF_V1.md) for why this exists and what v1
taught, [ROADMAP.md](ROADMAP.md) for what comes next and what each step
refuses to claim, [ARCHITECTURE.md](ARCHITECTURE.md) for the shape,
[DECISIONS.md](DECISIONS.md) for what is and is not decided, and
[CAPABILITIES.md](CAPABILITIES.md) — generated from the manifest, never
hand-edited — for what exists and how well it is tested.

## Running it

```bash
pip install -e .[dev]
python tasks.py check            # offline audit + every suite (what CI runs)

python -m vigil console          # the operator window
python -m vigil where            # paths, principal, alert sinks
```

Or build the window on its own — one executable, no command line, no terminal
behind it — into the top of the repository:

```bash
python v2/tasks.py app           # -> VIGIL.exe in the repository root
```

`VIGIL.exe` and the `_internal` folder beside it are build output and are
gitignored. The command line is not removed by this build; it is simply not
what the folder hands to whoever opens it. Some things still need it —
`vigil identity enable` takes a written reason and a retention limit, and a
dialog with two boxes would invite treating that as a preference.

From the command line, without the window:

```bash
python -m vigil cameras add gate device:0 --place 33.8938,35.5018,2,180,-15
python -m vigil cameras calibrate gate --points "0.2,0.8,33.8937,35.5018; …"  # measure the pose
python -m vigil zones add yard "33.8937,35.5018;33.8937,35.5019;33.8936,35.5019" --watch person
python -m vigil run --for 30 --record
python -m vigil incidents
python -m vigil review inc-… ack             # somebody has seen it and it is real
python -m vigil review inc-… dismiss --note "a delivery"
python -m vigil export inc-…     # report, clips, and a manifest that verifies
python -m vigil audit            # who changed what, and when
```

The site's own geometry, from the cameras that watch it:

```bash
python tasks.py core             # build the Rust engine core (needs cargo)
python -m vigil coverage         # what these cameras reach, and what they miss
python -m vigil map build --seconds 60 --png yard.png
python -m vigil map show         # and how much of it is worth believing
python tasks.py bench            # what the pipeline costs and what it recovers
python tools/camera_check.py     # the whole pipeline against a real camera
```

Before trusting any of it on a new site — both read **unlabelled** footage, so
they can be run the day the cameras go up:

```bash
python tools/calibrate.py CLIP.mp4        # what real video says about the constants
python tools/detector_options.py --clip CLIP.mp4   # how to make detection affordable
```

And the corpus running the product has already written:

```bash
python -m vigil identity show                  # faces and plates: OFF, and why
python -m vigil eval ./corpus                  # score the detector — refuses a leaking split
python -m vigil dataset export --to ./corpus   # frames, pre-labels, split BY DAY
python -m vigil site detection --detect-every 3  # 3x less detection, measured
```

**Accounts.** `python -m vigil users add root --role ADMIN`. After the first
account exists every command runs as `--as NAME` (password prompted, or on
standard input with `--password-stdin`) and the console asks for a sign-in. A
store with no accounts is open and says so on every command and in the status
bar. Roles are sets of permissions: viewer watches; operator changes the site,
runs the analysis and exports; analyst exports and reads the audit trail;
admin does all of it and manages accounts.

**Changing what exists.** A camera that moved to a new address keeps its
placement and its zones (`vigil cameras source gate rtsp://…`, or *Edit…* in
the console, which also renames it), and a zone's
name, kind, watch list or hours change without touching the ring somebody
drew (`vigil zones edit yard --watch person --closed 22-6`, or *Edit zone…*
in the console). Both are audited with what the value was before.

**Finding what happened.** `vigil incidents --camera north-gate --since 2d
--severity HIGH` and `vigil events --contains person --until 2026-09-01`
answer a question about last Tuesday without reading the whole list. A time is
`2h`, `3d`, a date, or a full ISO moment, and anything else is refused rather
than quietly widened to everything. A severity means that one *and worse*. The
console has the same filters above its incident list.

**What things are doing together.** Beyond "a person entered the yard", the
system measures relations between tracks and says how sure it can be: somebody
*probably in* a vehicle, *appearing to carry* something, *with* somebody else,
*moving towards* a zone. Every one is inferred, never observed — one camera
cannot tell being inside a car from walking in front of it — so each carries
the overlap, the distance and the frames it held, and a zone entry that quotes
one repeats those conditions in its evidence. Distances are always given with
their error: `11.2 ± 2.4 m`, never a bare eleven.

**How many are in it.** A vehicle entering a zone is one event whether it
holds a driver or five people, and the difference is the whole reason somebody
is watching, so the entry says *apparently with 3 people inside* and quotes
the overlap each count was drawn from. The track table says the same on the
vehicle's row. Hedged, like every relation: from one camera, somebody standing
in front of a van overlaps it exactly as somebody sitting in it does.

**What a site watches for.** The watch list and the confidence threshold are
kept with the site (`vigil site detection --watch person,car --confidence
0.6`, or *Watch for…* in the console), not typed at each run: a service started at boot has nobody to type at
it, and until now it analysed with the built-in list while the operator
believed the choice they made once still applied. `--watch` and `--confidence`
still override it for one run. A label the installed model cannot produce is
refused where it is typed — the run prints the model's own vocabulary — and
`vigil doctor` fails on a stored one, because a site watching for nothing
looks exactly like a quiet night. A change made in the window lands at the
next start and says so: a detector is made once per camera thread and lives as
long as it does, so swapping one mid-run would mean two cameras drawing
conclusions from different settings inside one incident.

**Dangerous things, and what this refuses to claim.** A site names the labels
it treats as dangerous: `vigil site threats --set knife --suggest`. Nothing is
a threat by default, because the shipped model names `knife` and `scissors`
and a kitchen raising a critical alert every evening teaches an operator to
ignore the word. A threat claim needs a higher confidence and several frames
than an ordinary detection, it records the weights that made it, and when a
relation says somebody is carrying it the sentence says so and the severity
rises a step. `vigil doctor` **fails** when a configured label is one the
installed model can never produce, so a site is never told it is protected
when it is not.

**A warning before the breach.** When somebody is closing on a restricted or
perimeter zone and is already within ten metres, the system says so — once,
at `MEDIUM`, with the distance and its error — rather than waiting for the
entry. That is a rule acting on a *relation*, which is the only way a rule can
hear about something that has not arrived yet.

**Seen on screen and in the evidence.** The track table says how far each
object is from its camera, always with the error — `12.0 ± 1.5 m` — and what
it is doing, with the reasons one hover away. The *Why* panel says the same
for every event in an incident, beside the rule, the conditions it checked and
the risk weights. The exported report carries that distance too, because
"eleven metres from the gate" is the kind of thing somebody asks months later.
One measurement, computed in one place (`Evidence.distance_from`), so the
screen and the evidence cannot disagree. A track the geometry could not place
says so instead of printing a number.

**The window fits the window.** The toolbar wraps rather than shrinking its
buttons: Qt's answer to controls that do not fit is to elide their labels, and
a shipped build read "dd camera." and "ort evidenc" on a 1280-wide screen with
nothing failing and nothing logged. Every control keeps the width of its own
label and the toolbar takes another line instead. A test resizes the window to
1280 and checks exactly that.

**Working the queue.** `vigil incidents` shows what is still waiting on a
person; an incident is acknowledged or dismissed, and a dismissal needs a
reason, because "dismissed" with no reason cannot be told from nobody having
looked. The judgement names the person, is audited with what it was before,
and survives re-correlation: the system may learn more about an incident, but
it may not overrule somebody. In the console the same two buttons sit beside
Export, and dismissed incidents leave the list until you ask for them.

**Unattended.** `python -m vigil supervise -- run` restarts the analysis when
it dies, with a growing pause. `python -m vigil service install` registers
that with this operating system (a scheduled task, a launch agent or a systemd
user unit — `service print` shows exactly what it would write and run).
`python -m vigil run --stop` asks a running analysis to stop, on any platform.

**Alerts leave the process.** A dark camera, a recording that stopped early, a
retention sweep that cannot reach its target, a stuck analysis thread and a
disk below the watermark each raise one alert until it clears: a banner and a
sound in the console, an `alert.raised` row in the audit trail, a line in
`alerts.log`, and whatever `VIGIL_ALERT_COMMAND` and `VIGIL_ALERT_WEBHOOK`
(local network only) are pointed at. `vigil alerts --test` sends one through
every sink.

**Before leaving site.** `vigil doctor` checks the things a deployment fails
on quietly — the data directory, the database's integrity and schema, the site
clock, the model, the keychain, free disk, accounts, cameras, zones and where
alerts go — and says what to do about each. `--probe` opens every camera too.
It exits non-zero on any failure, so it can be the last line of an install
script.

**Watching it.** A run logs a metrics line every minute: cameras, live, dark,
faulted, recording, frames, fps, dropped, events, incidents, open alerts. Set
`VIGIL_LOG_JSON=1` and every line becomes one JSON object for a monitoring
agent, redacted the same way the prose is.

**Packaging.** `python tasks.py package` builds `dist/vigil/vigil.exe`.
`python tasks.py exetest --seconds 20 --record` runs it on `device:0` and
judges the run; `--console` drives the window instead and photographs it. It
refuses to run against an executable older than the source, because an old
binary runs perfectly and a pass against one is evidence for a change it does
not contain — which has already happened here once, quietly.

## The state of it

| | |
|---|---|
| Product code | 21,442 lines of Python, 5,351 lines of Rust behind a C ABI |
| Tests | 436 Python + 77 Rust, all green through `python tasks.py check` |
| Capabilities | 42 tested, 2 implemented, **0 planned** ([CAPABILITIES.md](CAPABILITIES.md)) |
| Packaged | `vigil.exe` for the command line and `vigil-console.exe` for the window, sharing one `_internal` |

**Two executables, one tree.** `vigil.exe` is console-subsystem so every
command prints, pipes and redirects; `vigil-console.exe` is GUI-subsystem so
double-clicking it opens the window with no black terminal behind it for the
whole shift. Both are built in the same run from the same sources, so they
cannot drift. Running `vigil.exe` with **no arguments** opens the window too —
it used to answer `error: the following arguments are required: command` and
exit 2, which is a usage message flashed into a console that closes before
anybody can read it.

The engine core is **optional but not decorative**. Without it the product
still analyses, tracks and raises events — the NumPy paths are the same
algorithms and `tests/test_native.py` holds them to the same answers — but
`vigil map` is unavailable and association runs an order of magnitude slower.
`vigil doctor` says which you have.

Not built, and not pretended: **any training or labelling pipeline** — the model is an
ONNX file the operator supplies and the shipped one is stock COCO weights,
which have never seen this site; installers and code signing, which need a
certificate; and **a model that can name a weapon**, which is the operator's
to supply — the mechanism is here and tested, the weights are not.

Faces, plates and the subject register **are** built, off by default behind one
switch (`vigil identity show`), and every threshold in them is an assumption
that says so: no face or plate model ships and none has been run. DECISIONS.md
D-08 records that the gate for building them — an hour on a physical IP camera
— was waived rather than met, by whom, and what is therefore unproven.
[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) is the honest account of
what is measured and what is not; [ROADMAP.md](ROADMAP.md) says what each of
those needs.

## Camera runs on this machine

| When (UTC) | What ran | Result |
|---|---|---|
| 2026-09-06 21:53 | `VIGIL.exe` from the repository root, 20 s on `device:0`, after the console was reworked | **PASS**, exit 0. The window only — no command line, no terminal behind it. Every verb now sits under the thing it acts on, and every panel heading reads in full: *1 of 1 placed*, *yolov8n-seg — watching 80 classes with masks · f828ccfa4b69*, *no map yet — 61 frame(s) folded in*. Two eliding bugs were found by photographing it and fixed: a right-aligned heading label given a cap wider than itself is clipped by Qt **from the left**, so the wall read *"olov8n-seg — watching 80 classes with …"* — correctly ellipsised at the end and missing its first letter; and a stretch spacer beside a stretching label split the heading between them, eliding a sentence that had room to be read. |
| 2026-09-06 21:20 | packaged `vigil.exe`, `exetest --seconds 20 --record --console`, after the identity, suppression-kernel and evaluation work | **PASS in 24 s**, with migrations 6 to 9 applied inside the packaged build. Faces and plates ship **off**: `vigil identity show` reads "faces and plates are OFF: no face is embedded, no plate is read, and no biometric row is written". |
| 2026-09-06 21:05 | the four Phase 8 candidates, measured before any was rewritten | The measurement decided what to write **and what not to**. NumPy soft-NMS on 300 proposals: **4.92 ms**, against about 12.5 ms for the detection itself — and tiling runs it once per tile. In Rust: **0.044 ms, 138x**, returning identical indices including ties. Two candidates on the same list were left alone after measuring: the assignment cost matrix at **5 microseconds** and mask decode at 0.9 ms. The appearance descriptor got neither — `cv2.calcHist` with a mask is 3.5x faster than indexing the pixels out and counting them in NumPy, with identical results over 300 randomised crops, so 12 detections went **3.61 ms to 1.49 ms** with no new implementation of a colour space to keep in step. |
| 2026-09-06 20:19 | packaged `vigil.exe`, `exetest --seconds 20 --record --console`, after the pose calibration, triangulation, cross-camera, live-map and detection work | **PASS in 23 s.** The window carries the new *Measure pose…* control, and the status bar reads **watching 80 classes** rather than six. It detected and tracked a **cell phone** — a class the old frozen watch list made invisible to the detector, the tracker, the plan and the map — drew it, inferred `carried` between it and the person, and raised **no event** for it, which is the whole point of splitting what is detected from what is alerted on. Migrations 6, 7 and 8 applied inside the packaged build. |
| 2026-09-06 20:18 | packaged `vigil.exe`, `exetest --seconds 25 --record` on `device:0` | **PASS in 29 s** at a sustained **20 fps** with the lowered 0.25 floor, soft-NMS and the full COCO vocabulary, on DirectML. |
| 2026-09-06 20:28 | `tools/camera_check.py --seconds 75 --map` on `device:0` — the rebuilt detection path end to end | 1,491 frames in 75.3 s = **19.8 fps**, against 14.1 fps on the CPU path earlier the same day. **2.3 detections per frame** where the shipped 0.50 threshold and six classes produced 0.00 on the same view; 3 distinct track ids for 3 objects, so no fragmentation came with them. Detection 12.45 ms median. Map: **123 m² usable of 163 m² seen (75%)**. Tiling correctly did nothing — a 640x480 webcam is already below the model's 640x640 input, and cropping an image the model sees whole buys nothing. |
| 2026-09-06 20:12 | the shipped `yolov8n-seg` on DirectML over twelve **1080p** frames, before and after the detection changes | The measurement Phase 6 rests on. At the old 0.50 floor with hard NMS and the six classes: **0.00 detections per frame**. At 0.25: **1.08**. With four tiles over the far ground: **1.33**, at **55.3 ms against 11.0 ms** — exactly the 1+4 inferences it costs. The far-half recall proxy came out **zero on both sides**, because a laptop webcam pointed at a room has no far ground; that number stays unmeasured until this runs on a camera that can see something distant, and it is the number tiling was built for. |
| 2026-09-06 18:04 | packaged `vigil.exe`, `exetest --seconds 20 --record --console`, after the two-executable split | **PASS in 22 s.** All six panels photographed; the camera wall drew a real frame at 15 fps with no warning banner, which is the correct answer for a good frame. Separately verified: `vigil.exe` **with no arguments** now opens the window instead of printing `error: the following arguments are required: command` and exiting 2, and the new GUI-subsystem `vigil-console.exe` runs with no console behind it. Migration 5 applied inside the packaged build. |
| 2026-09-06 17:47 | `onnxruntime-directml` in an **isolated virtualenv**, so this machine's environment was untouched | The integrated GPU runs the shipped `yolov8n-seg` session in **4.5 ms against the CPU's 38.5 ms — 8.6x**, 220 raw inferences a second. Session only: letterboxing, NMS and mask decoding are still Python, so end-to-end `detect()` would be nearer 10-12 ms. Not adopted — `onnxruntime-directml` *replaces* `onnxruntime`, which is a deployment decision. |
| 2026-09-06 16:56 | `tools/camera_check.py --seconds 25 --map` on `device:0` — **the rebuilt pipeline, end to end** | 356 frames in 25.3 s = **14.1 fps** with `yolov8n-seg` on the CPU. Detection 64.5 ms/frame; the three new perception stages cost **3.5 ms together** (quality 0.6, camera motion 2.9, appearance ~0). 0 unusable frames, 0 stale, the camera correctly reported still on 99% of frames. A ground map built from the same run: 95 m² usable of 160 m² seen, 0.05 m per source pixel — and, pointed at a room rather than a yard, it correctly marked 41% unusable, because a wall projected onto the ground plane is a smear and the confidence layer says so. |
| 2026-09-05 22:55 | packaged `vigil.exe`, `exetest --seconds 20 --record` | **PASS** in 21 s: 191 frames at 10 fps through the segmentation model, one clip written and indexed, exit 0, no traceback. The room was dark, so 0 detections. |
| 2026-09-06 05:44 | the console on `device:0`, 22 s, recording | **The whole chain, on a real camera.** A person detected and classified at 0.84, tracked, projected to ±0.1 m, entering a restricted zone; two zone-entry events; one HIGH incident at risk 0.46; a 384-frame clip; the exported package verifies against its manifest and carries the clip. |
| 2026-09-06 05:55 | packaged `vigil.exe`, `exetest --seconds 20 --record --console` | **PASS** in 22 s: the window drove the camera at 19 fps through the segmentation model, tracked a person, recorded, and photographed all six panels. |
| 2026-09-06 06:13 | the console with **two** cameras, the laptop and a file, both placed | Two workers, two wedges on the plan, per-camera health (one LIVE, one STOPPED when its file ended), and an incident from the live one. |
| 2026-09-06 06:16 | **ten-minute unattended soak**, `run --for 600 --record` | **Stable.** 602 s for a 600 s request, exit 0, no traceback. 8,565 frames — 14.3 fps average, 16 fps at the end — with the segmentation model on one CPU camera. Ten one-minute clips. Memory started at 45 MiB, peaked at 221 MiB while the model loaded, settled at 64 MiB and stayed flat for the last four minutes. |

| 2026-09-06 15:16 | packaged `vigil.exe`, `exetest --console` — **FAIL**, and worth keeping | The bundle was built while the source was being edited, so it shipped a window importing `describe_group` from a packaged domain module that did not have it. An earlier run had already passed against a thirty-minute-old binary. `package` now stamps a digest per source file and `exetest` refuses a tree that differs. |
| 2026-09-06 15:27 | packaged `vigil.exe`, `exetest --seconds 20 --record --console`, against a bundle proven to match the tree | **PASS** in 23 s. `site detection --watch person,car --confidence 0.55` survived a restart, `doctor` read it back as OK, and `run` printed `watching car, person; confidence at least 0.55` before any thread started. The window drew the new *Edit…* button greyed under the lock, and the track table read **3.1 ± 0.1 m** for a person at 0.88. |
| 2026-09-06 11:20 | the whole chain on a generated walk, through the real pipeline | A block walking towards the camera produced **"An object is approaching the Doorstep, 0.5 ± 0.2 m away"** at `MEDIUM`, then **"An object entered the Doorstep"** at `HIGH`, correlated into one incident. Warning first, breach second — which is the point. (Camera runs remain the product check; this one needed a known path.) |
| 2026-09-06 10:44 | the console on `device:0` with the new columns | The track table read **3.1 ± 0.1 m** for a bottle in front of the camera — the distance and its error, measured live. "Doing" was empty because a relation needs two tracks and only one was in view. |
| 2026-09-06 09:25 | packaged `vigil.exe`, `exetest --console` after the CLI split | **PASS** in 23 s. The shipped console carries the incident filter bar, and the split into parser, site commands and work commands changed nothing an operator can see. |
| 2026-09-06 09:02 | packaged `vigil.exe`: `site name Depot --timezone Asia/Beirut`, then `doctor`, then `exetest --console` | The bundle carries the IANA time zone database, so the site clock is read as `Asia/Beirut` rather than silently falling back to UTC. `doctor` passed with nothing failing; the console test passed in 23 s. |
| 2026-09-06 07:37 | the queue, end to end on the camera | A live incident raised on `device:0`, a dismissal **refused** for having no reason, then acknowledged with a note. The judgement shows in the console's State column, in `vigil incidents`, and in the audit trail under the person who made it. |

Two things that only a real process can prove, proven with real processes:

- **The supervisor restarts a child that keeps failing** — three starts with
  1 s then 2 s between them, then it gives up at the limit and returns the
  child's code, rather than spinning.
- **`vigil run --stop` reaches a supervised run from another process.** The
  run noticed on its next poll, exited cleanly within about a second, the
  supervisor saw the same file and did not restart, and the file was removed.
  This is the only way to stop a run on Windows, where an external Ctrl-C
  never reaches the process and a scheduled task has no terminal.

The soak also measured something an operator has to plan for: **one recorded
camera writes about 15 MiB a minute — roughly 21 GiB a day.** Retention is on
by default whenever recording is, and `vigil retention` sweeps on demand.

Four defects were found by looking at that console, not by a test:

1. The first packaged build failed `exetest` in 0 s — a frozen `__main__`
   cannot use relative imports (`packaging/entry.py` is the fix).
2. The screenshots were never written: a timer was connected to a bound
   method of an object nobody owned, so it was collected and the slot never
   ran. Silently. It is owned by the window now, and a test says so.
3. Every class read "—" because the model was looked for in one directory
   only, so the console had silently fallen back to motion detection. It now
   searches every place a model could be, and the status bar always says what
   is drawing the conclusions — including "does not classify, and cannot see
   a stationary object" when that is the truth.
4. `--record` set a destination and then recorded nothing, because each
   camera's own stored flag still decided. It now means what it says.

A sixth came from running the queue demonstration: `vigil run device:0`
created a *second* camera for a device the site already had, and ran that
unplaced duplicate — so the placement and the zones were silently not
applied. Naming a source the site already knows now uses that camera, and a
source it does not know is added under a name derived from the source, with
both the addition and its unplaced state said out loud.

A fifth came from reading the exported report: one person the detector
blinked on was counted as "2 persons". Same-camera fragments are now
rejoined within two seconds and half the spatial allowance, and each link
says in words that it rests on time and place alone.
