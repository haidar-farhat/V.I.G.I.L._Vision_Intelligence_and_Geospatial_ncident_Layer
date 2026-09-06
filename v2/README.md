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

From the command line, without the window:

```bash
python -m vigil cameras add gate device:0 --place 33.8938,35.5018,2,180,-15
python -m vigil zones add yard "33.8937,35.5018;33.8937,35.5019;33.8936,35.5019" --watch person
python -m vigil run --for 30 --record
python -m vigil incidents
python -m vigil review inc-… ack             # somebody has seen it and it is real
python -m vigil review inc-… dismiss --note "a delivery"
python -m vigil export inc-…     # report, clips, and a manifest that verifies
python -m vigil audit            # who changed what, and when
```

**Accounts.** `python -m vigil users add root --role ADMIN`. After the first
account exists every command runs as `--as NAME` (password prompted, or on
standard input with `--password-stdin`) and the console asks for a sign-in. A
store with no accounts is open and says so on every command and in the status
bar. Roles are sets of permissions: viewer watches; operator changes the site,
runs the analysis and exports; analyst exports and reads the audit trail;
admin does all of it and manages accounts.

**Changing what exists.** A camera that moved to a new address keeps its
placement and its zones (`vigil cameras source gate rtsp://…`), and a zone's
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
it is doing, with the reasons one hover away. The exported report carries the
same distance for every event, because "eleven metres from the gate" is the
kind of thing somebody asks months later. A track the geometry could not place
says so instead of printing a number.

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
judges the run; `--console` drives the window instead and photographs it.

## The state of it

| | |
|---|---|
| Product code | 9,592 lines of Python, no compiled core |
| Tests | 190, all green through `python tasks.py check` |
| Capabilities | 26 tested, 2 implemented, 1 planned ([CAPABILITIES.md](CAPABILITIES.md)) |
| Packaged | 751 MB bundle: one `vigil.exe` that is both the command line and the console |

Not built, and not pretended: faces, plates and a subject register
(DECISIONS.md D-08); appearance re-identification, so cross-camera identity
rests on time and place alone and says so; installers and code signing, which
need a certificate; and **a model that can name a weapon**, which is the
operator's to supply — the mechanism is here and tested, the weights are not.
[ROADMAP.md](ROADMAP.md) says what each of those needs.

## Camera runs on this machine

| When (UTC) | What ran | Result |
|---|---|---|
| 2026-09-05 22:55 | packaged `vigil.exe`, `exetest --seconds 20 --record` | **PASS** in 21 s: 191 frames at 10 fps through the segmentation model, one clip written and indexed, exit 0, no traceback. The room was dark, so 0 detections. |
| 2026-09-06 05:44 | the console on `device:0`, 22 s, recording | **The whole chain, on a real camera.** A person detected and classified at 0.84, tracked, projected to ±0.1 m, entering a restricted zone; two zone-entry events; one HIGH incident at risk 0.46; a 384-frame clip; the exported package verifies against its manifest and carries the clip. |
| 2026-09-06 05:55 | packaged `vigil.exe`, `exetest --seconds 20 --record --console` | **PASS** in 22 s: the window drove the camera at 19 fps through the segmentation model, tracked a person, recorded, and photographed all six panels. |
| 2026-09-06 06:13 | the console with **two** cameras, the laptop and a file, both placed | Two workers, two wedges on the plan, per-camera health (one LIVE, one STOPPED when its file ended), and an incident from the live one. |
| 2026-09-06 06:16 | **ten-minute unattended soak**, `run --for 600 --record` | **Stable.** 602 s for a 600 s request, exit 0, no traceback. 8,565 frames — 14.3 fps average, 16 fps at the end — with the segmentation model on one CPU camera. Ten one-minute clips. Memory started at 45 MiB, peaked at 221 MiB while the model loaded, settled at 64 MiB and stayed flat for the last four minutes. |

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
