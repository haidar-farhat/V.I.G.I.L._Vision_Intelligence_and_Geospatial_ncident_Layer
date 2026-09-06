# V1, reviewed without mercy

*Written 2026-09-06 against branch `Phase2` at `2bf57e6` plus the six
uncommitted slices of that day (backup/restore, supervision, keychain,
accounts, model cache, alerts). Every number below was measured on the tree,
not remembered.*

This is the document a v2 has to be built from. It is not a list of bugs —
the 137-row audit in `PRODUCTION_READINESS.md` is that — it is the answer to
one question: **what about v1's shape made those bugs inevitable, and what
shape would not?** Where v1 got something right, that is said too, because a
rewrite that throws away the hard-won parts is not a v2, it is a v0.

---

## 1. The verdict in one paragraph

V1 proved the *idea*: video → detection → tracks → ground positions → zone
presence → events → cross-camera incidents → an evidence package, all with no
route out of the site, and a console an operator can run on a laptop camera.
It did not become a *product*, and the reason is structural, not a shortage
of features. Its three biggest modules (`node.py` 3,141 lines / 102 methods,
`app.py` 3,108 / 111, `store.py` 2,294 / 71) each do five jobs; permission,
identity and alerting were bolted on in the last day instead of being the
frame everything hangs from; the Rust core paid a permanent tax (ABI versions,
struct-size guards, a GNU-vs-MSVC hazard) for a tracker of a few hundred
lines; and the recurring defect of the whole project — **correct, tested code
that nothing calls** — was found seven times because nothing in the design
made reachability a property that could be checked. V2 keeps the algorithms,
the invariants and the testing discipline, and replaces the skeleton.

## 2. What v1 measured, and what the numbers say

| Fact | Value | What it means |
|---|---|---|
| Lines of product Python | 26,235 engine + 11,656 console | Large for what it does; the spine is perhaps a third of it |
| Lines of Rust | 3,844 | Geometry and a tracker; every struct change bumped an ABI version |
| Tests | 60 Rust, 1,192 engine, 393 console | Real, and green; the discipline is the best thing about v1 |
| FEATURES.md rows | 211 TESTED · 24 IMPL · 30 SKEL · **228 PLAN** | Under half of the product definition exists |
| Audit rows | 137 (P0 7, P1 31, P2 58, P3 38, P4 3) | After six slices: P0 2 closed, 1 partial, 4 open |
| Packaged bundle | 834 MB, three executables | Ships onnxruntime, OpenCV, Qt, a 14 MB model; no installer, no signing |
| Empty scaffolding | `services/`, `packages/`, `infrastructure/` — 0 files | An architecture document describing microservices that never existed |
| Times "tested code nothing calls" was found | 7 | The recorder, `RecorderStats.fault`, `LiveStream`, the segmenter, `ground_contact`, the console export path, the correlator's plate hook |

## 3. What v1 got right — keep, port, do not "improve"

1. **The non-negotiables, and enforcing them mechanically.** Zero WAN with
   an offline audit over shipped source *and* a binary audit over wheels;
   passwords never in logs/DB/UI/argv; an append-only audit trail; honest
   capability states. These are the product. V2 keeps every one, and adds
   the enforcement v1 lacked for the last two.
2. **The projection maths and the tracker.** Rectilinear ray model, ground
   intersection with an honest uncertainty, `CameraFallback` when the ray
   misses the ground, an annular field-of-view wedge rather than a pie
   slice, cumulative-hit confirmation, coasting that never feeds the speed
   estimate. All measured on real footage. Ported line for line into Python.
3. **Conservative correlation that shows its work.** Time-and-place
   association with a stated 65/35 weighting, uncertainty-widened gates
   capped so one bad camera cannot swallow the site, summaries that say
   "object" unless every detector classified. Ported.
4. **Testing through the shipped executable on a real camera.** `exetest`
   was the single most productive verification in the project: it found the
   deleted-dialog bug, the NTFS stream bug, the cp1252 crash and the
   four-times model load. V2 makes it a first-class task from day one.
5. **Hysteresis everywhere a human reads a state.** Zone entry/exit holds,
   dark-camera grace, dedup'd alerts. Ported as a rule.

## 4. What went wrong, and *why* it went wrong

### 4.1 God objects: three modules, fifteen jobs

`Node` owns the store, the cameras, the correlator, the plate reader, the face
engine, the registry, retention, recording, restore-from-database, the audit
actor, and now alerts. `ConsoleWindow` owns the node *and* every panel *and*
the command-line entry point *and* the timed-run harness *and* the screenshot
tool. `Store` is schema, migrations, twelve entity tables, backup, integrity
and the audit trail in one class. Consequences that were observed, not
imagined:

- A change to recording semantics (`record_every_camera`) broke seven
  unrelated tests, because seven behaviours lived in one constructor.
- Permission checks (SEC-01) could only be put in the *console*, because
  `Node` has no notion of who is asking. The audit row says the right name
  now; the node still trusts any caller. That is a hole v2 closes by
  construction: every mutation takes a principal at the service boundary.
- The store is touched only from the node's thread — a rule enforced by
  comment. It was violated at least once (the console's export path) and
  found by a test that happened to run under a different thread.

### 4.2 Reachability was never a checkable property

Seven times a module was written, tested in isolation, marked `TESTED`, and
turned out to be called by nothing. FEATURES.md states were hand-edited
prose. V2's capability manifest is code: each capability names the symbols
that implement it and the tests that exercise them, and a test walks the
manifest and fails on a symbol no test imports *and* on a public service
method no interface calls. "Tested code nothing calls" becomes a red test.

### 4.3 The Rust core cost more than it returned

The core is 3,844 lines of geometry and tracking behind a C ABI with
`ctypes` structs, an ABI version bumped six times, a struct-layout guard,
and a documented hazard (a GNU-toolchain cdylib against an MSVC CPython).
Its performance argument never materialised: the measured bottleneck is the
ONNX model (11–17 fps per camera on CPU), and the tracker runs in
microseconds per frame in either language. Meanwhile the boundary was where
"computed in Python and used by nothing" happened twice (`ground_contact`,
the contact point in `CTrack`). V2 is one language for the product core,
with the tracker behind an interface so a compiled one can return if a
measurement ever asks for it.

### 4.4 Identity, permission and alerting were the last things built

Accounts arrived on the last day (migration 12), alerting the same day, and
the audit trail said `console` for a year. Everything downstream — the
export's chain of custody, the "who changed this zone" question, the
service's restart audit — was built without a principal and had to be
threaded through afterwards. V2's schema has `users`, `audit` and `alerts`
in migration 1, and no service method exists that does not take a principal.

### 4.5 Qt lifetime discipline was learned by crashing

Reference cycles holding `QWidget`s, lambdas closing over `self` in
connections, `WA_DeleteOnClose` on a dialog read after `exec()`, exceptions
swallowed in slots. Each was found by a crash or by a button that "did
nothing" in the operator's screenshot. V2 keeps Qt (native, offline, no
embedded browser) but the console is a thin *view* over the service: it holds
no domain state, every mutation goes through one `Commands` object, and the
lifetime rules are enforced by a structural test rather than by memory.

### 4.6 The documentation outran the code

`ARCHITECTURE.md` describes workers, a control node, mTLS 1.3 and a versioned
protocol; none exist. `FEATURES.md` carried 228 `PLAN` rows. The docs lint
checks diagrams parse, not that claims are true. V2's `CAPABILITIES.md` is
*generated* from the manifest by a task, and the architecture document
describes only what is built; anything else is in `DECISIONS.md` as a
decision not yet taken.

### 4.7 Operations were an afterthought

No installer, no signing, no service wrapper until day 366, no backup until
the same day, CI that had never executed remotely, a 834 MB bundle. V2 treats
the shipped executable as the product from the first commit: one CLI
executable, `package` and `exetest` tasks in the runner, a release checklist,
and a bundle that carries what it uses.

## 5. Things v1 never resolved (still open questions for v2)

- **Appearance re-identification.** Without it, cross-camera identity is
  time-and-place only; v1 said so honestly. V2 keeps the same honesty and the
  same hook.
- **Real IP cameras for hours.** Everything was laptop camera and files.
  V2's decode layer is built for RTSP first and the exetest takes an RTSP
  URL, but the hour on a physical camera is still a test to run, not a
  design to make.
- **Remote CI.** Never executed in v1. V2's `check` task is what CI runs;
  the workflow file is written but, until the user runs it, it is `PLAN`.
- **Installers and signing.** Need a certificate the user holds.

## 6. The principles v2 is built on (each traceable to a section above)

| # | Principle | From |
|---|---|---|
| P1 | Every mutation goes through a service method that takes a `Principal` and checks a permission; the audit row carries the principal. | 4.1, 4.4 |
| P2 | Modules have one job and a size budget (≤ 800 lines); a test fails the build above it. | 4.1 |
| P3 | Reachability is tested: the capability manifest names symbols and tests; unreferenced symbols fail. | 4.2 |
| P4 | One language for the core; interfaces where a compiled replacement could go. | 4.3 |
| P5 | Users, audit and alerts exist in migration 1; nothing is bolted on. | 4.4 |
| P6 | The UI is a view: no domain state, one command object, structural lifetime test. | 4.5 |
| P7 | Documentation is generated where it states facts; prose states intent only. | 4.6 |
| P8 | The shipped executable is the test medium from the first week: `package` and `exetest` are day-one tasks. | 4.7, 3.4 |
| P9 | The v1 invariants stand: zero WAN (audited), secrets in the keychain, append-only audit, honest states, hysteresis on every human-facing state. | 3 |
| P10 | Every thread has an owner and a bounded queue; the store belongs to one thread and that is checked at runtime, not by comment. | 4.1 |
