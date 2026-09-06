# Decisions

One entry per decision that shapes v2. A decision not yet taken is listed as
such, so nothing here is a promise about code that does not exist.

## D-01 · One language for the core (taken)
The Rust core is not ported. Measured in v1: the tracker and geometry cost
microseconds per frame; the model costs 60–90 ms. The C ABI cost six ABI
bumps, a struct-layout guard and a toolchain hazard, and was the site of two
"computed and unused" defects. `domain.tracking.TrackerProtocol` is the seam
for a compiled tracker if a measurement ever asks for one.

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

## D-08 · Faces, plates, subject registry (not yet taken)
V1 built all three; each was a privacy switch, a model dependency and a
thousand lines. V2 will bring them back behind the same identity switch only
after the spine has run on a physical IP camera for an hour.

## D-09 · Packaging (taken)
One CLI executable via PyInstaller (`python tasks.py package`), exercised by
`python tasks.py exetest` on `device:0`. Installers and signing wait on a
certificate.

## D-10 · Configuration (taken)
Environment variables with the `VIGIL_` prefix, read once into `Settings`.
No config file yet.
