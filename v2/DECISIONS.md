# Decisions

One entry per decision that shapes v2. A decision not yet taken is listed as
such, so nothing here is a promise about code that does not exist.

## D-01 · One language for the core (taken; **reversed** 2026-09-06)
The Rust core is not ported. Measured in v1: the tracker and geometry cost
microseconds per frame; the model costs 60–90 ms. The C ABI cost six ABI
bumps, a struct-layout guard and a toolchain hazard, and was the site of two
"computed and unused" defects. `domain.tracking.TrackerProtocol` is the seam
for a compiled tracker if a measurement ever asks for one.

**A measurement asked for one.** `core/` exists again: a Rust crate behind a
C ABI, loaded by `vigil/kernel/native.py` with an ABI-version guard, with a
NumPy mirror for everything but the ground rasteriser. The numbers are in
PRODUCTION_READINESS.md §2: the ground raster at 79 ms per frame per camera
in NumPy against 0.62 ms in Rust, optimal assignment at 0.010 ms, suppression
at 0.044 ms against 4.92 ms. The rule this entry set is the one that governed
the reversal — a kernel is written *after* a measurement — and two candidates
on the same list, the assignment cost matrix and mask decode, were measured
and left in Python. The hazard this entry named is held by
`tests/test_native.py`, which fails when the two implementations disagree,
and by the CI step that refuses a run in which the core did not load.

*This paragraph was added on 2026-09-07. The entry said "not ported" for a
day after it was, and a decision reversed without a trace is
indistinguishable from one that was never made.*

## D-02 · Principal at the service boundary (taken)
Every `SiteService` and `Runtime` mutation takes a `Principal`. There is no
"actor string". A store with no users is open and says so.

## D-03 · Users, audit and alerts in migration 1 (taken)
See REVIEW_OF_V1.md §4.4.

## D-04 · Reachability is a test (taken)
`vigil/capabilities.py` is the manifest; `tests/test_capabilities.py` fails on
a symbol no test references and on a public service method no interface
calls. `CAPABILITIES.md` is generated.

## D-05 · Module size budget of 800 lines (taken)
`tests/test_layering.py`. A module over budget is split, not excused.

## D-06 · SQLite, one file, WAL, one owning thread (taken)
As v1, with the ownership checked at runtime.

## D-07 · The console (taken)
Qt again — native widgets, no embedded browser — as a thin view over
`SiteService` and `Runtime`. It holds no domain state: every change goes
through one `Commands` object that carries the principal and returns a
sentence rather than raising, so a refusal reaches the status bar instead of
being swallowed by a Qt slot. The three v1 Qt rules are structural tests: no
lambda closing over `self` in a connection, no `WA_DeleteOnClose` on a dialog
that is read after `exec()`, and a greyed control still answers a click with
the reason it is greyed. `vigil console`; `python tasks.py exetest --console`
drives it on the camera and photographs it.

## D-08 · Faces, plates, subject registry (taken — the gate was **waived**)
V1 built all three; each was a privacy switch, a model dependency and a
thousand lines. V2 said it would bring them back behind the same identity
switch **only after the spine had run on a physical IP camera for an hour**.

**That precondition was not met, and the work was done anyway**, on 2026-09-06,
at the explicit instruction of the repository owner (swteam@romisys.com), who
was told what the gate was for and chose to override it. This entry records the
waiver rather than quietly deleting the condition, because a decision that is
reversed without a trace is indistinguishable from one that was never made.

### What the gate was protecting against, and what is therefore unproven
The hour on a physical IP camera was meant to establish that the *spine* — one
camera, decoding, detecting, tracking, projecting, recording, for a sustained
period — was reliable before anything as consequential as biometrics was built
on top of it. What has actually been run is: a ten-minute unattended soak on a
**laptop webcam**, and repeated shorter runs on the same. No IP camera, no RTSP
transport, no hour.

So the following remain unproven and must not be presented otherwise:

- Behaviour under RTSP reconnection, packet loss and H.264 corruption, none of
  which a USB webcam produces.
- Any face or plate figure whatsoever. **No face or plate model ships**, none
  has been run, and every threshold in `perception/faces.py` and
  `perception/plates.py` is a stated default with no measurement behind it —
  each says so in its own docstring.
- That the identity features are worth their cost on this hardware.

### What was built, and the conditions it was built under
Mechanisms only, all **off by default** behind one switch, all audited:
`perception/faces.py`, `perception/plates.py`, `domain/identity.py`, the
`subjects` / `face_observations` / `plate_observations` tables, redaction on
export, and a `vigil doctor` check that **fails** when the switch is on and no
biometric retention limit is set. Weights are operator-supplied, as the
detector's already are; nothing is downloaded.

The register reports a **similarity with a threshold and a margin**, never an
identity, and the wording is enforced in the code rather than left to whoever
writes the report.

## D-09 · Packaging (taken; amended 2026-09-07)
One CLI executable via PyInstaller (`python tasks.py package`), exercised by
`python tasks.py exetest` on `device:0`.

**Installers** are built by `python tasks.py installer` from
`packaging/installers.py`: an archive on every platform, an MSI and an NSIS
setup on Windows, a `.deb` on Linux, a `.pkg` and a `.dmg` on macOS, every
file listed in a per-platform `SHA256SUMS`. The formats that can be opened
without their platform's tools are read back by `tests/test_installers.py`;
each installer is installed, and the installed product asked to run `doctor`
and find its engine core, on the CI runner that built it. Nothing is claimed
for a platform that has not done that.

**Signing waits on a certificate.** Until one exists every installer says
UNSIGNED in its own metadata rather than carrying a self-signed signature,
which would teach an operator to click through the warning that is supposed
to protect them.

## D-10 · Configuration (taken)
Environment variables with the `VIGIL_` prefix, read once into `Settings`.
No config file yet.

## D-11 · The public page (taken, 2026-09-07)
`site/` holds two templates and a stylesheet; `python tasks.py site` renders
them into `dist/site` from the capability manifest, the version, the commit,
the line counts and pytest's own count of the suite, and
`.github/workflows/pages.yml` publishes exactly that. Generated for the
reason CAPABILITIES.md is: a hand-kept page is a second description of the
product, and the copy that drifts is the one a stranger reads. The page
carries no analytics and no script beyond a theme switch. Its fonts are the
one thing it fetches from outside; it is a page about the product, not the
product, and the product's rule is unchanged.
