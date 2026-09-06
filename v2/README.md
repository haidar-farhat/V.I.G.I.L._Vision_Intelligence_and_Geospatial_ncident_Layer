# Sentinel Vision v2

Local-first multi-camera incident intelligence: ordinary cameras in, a small
number of reviewable incidents out, with the evidence that produced each one,
and no route to the Internet at any point.

Read [REVIEW_OF_V1.md](REVIEW_OF_V1.md) for why this exists and what v1
taught, [ARCHITECTURE.md](ARCHITECTURE.md) for the shape,
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

**Packaging.** `python tasks.py package` builds `dist/vigil/vigil.exe`.
`python tasks.py exetest --seconds 20 --record` runs it on `device:0` and
judges the run; `--console` drives the window instead and photographs it.

## The state of it

| | |
|---|---|
| Product code | 7,393 lines of Python, no compiled core |
| Tests | 110, all green through `python tasks.py check` |
| Capabilities | 19 tested, 2 implemented, 1 planned ([CAPABILITIES.md](CAPABILITIES.md)) |
| Packaged | 749 MB bundle: one `vigil.exe` that is both the command line and the console |

Not built, and not pretended: faces, plates and a subject register
(DECISIONS.md D-08); appearance re-identification, so cross-camera identity
rests on time and place alone and says so; installers and code signing, which
need a certificate.

## Camera runs on this machine

| When (UTC) | What ran | Result |
|---|---|---|
| 2026-09-05 22:55 | packaged `vigil.exe`, `exetest --seconds 20 --record` | **PASS** in 21 s: 191 frames at 10 fps through the segmentation model, one clip written and indexed, exit 0, no traceback. The room was dark, so 0 detections. |
| 2026-09-06 05:44 | the console on `device:0`, 22 s, recording | **The whole chain, on a real camera.** A person detected and classified at 0.84, tracked, projected to ±0.1 m, entering a restricted zone; two zone-entry events; one HIGH incident at risk 0.46; a 384-frame clip; the exported package verifies against its manifest and carries the clip. |
| 2026-09-06 05:55 | packaged `vigil.exe`, `exetest --seconds 20 --record --console` | **PASS** in 22 s: the window drove the camera at 19 fps through the segmentation model, tracked a person, recorded, and photographed all six panels. |

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

A fifth came from reading the exported report: one person the detector
blinked on was counted as "2 persons". Same-camera fragments are now
rejoined within two seconds and half the spatial allowance, and each link
says in words that it rests on time and place alone.
