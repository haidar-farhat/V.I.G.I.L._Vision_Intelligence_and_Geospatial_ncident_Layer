# Handoff — 2026-09-03 (evening)

State of the work at the end of the ground-contact slice: the console exit
crash is resolved and explained, the mask-derived contact point now crosses the
Rust boundary, and CI is green. Written to be the first thing a new session
reads.

---

## 1. Standing rules (from the user, still in force)

1. **Run CI locally** — `python tasks.py ci --package`. Everything this machine
   can run, runs here before anything is called done.
2. **Build the three executables** each time, as before.
3. **Verify by screenshot.** Drive the real UI and look at the picture. Text
   assertions have twice passed on a screen that was visibly wrong.
4. **Start everywhere with masking and segmentation**, not boxes.
5. **Use a mature package rather than writing one**; use a **pretrained model
   rather than training one**.
6. **Use the laptop camera for real testing** — `device:0`, a real person in
   frame, not a synthetic fixture.
7. Conventional commits (§126). Honest capability states (§130). No fake
   implementations (§135).
8. **Test through the camera and the shipped binary — always.** Every check
   of the product runs `dist/SentinelVision/SentinelVision-dev.exe` against
   `device:0` with a real person in frame: never a prerecorded file, never
   `python tasks.py console`. `python tasks.py exetest` is that check — it
   seeds a camera, a placement and a person-only zone, runs for N seconds,
   photographs every panel, prints the summary and copies the log beside it
   into `dist/exetest/<stamp>/`. The console's command line exists for this
   (`--camera --place --zone --zone-classes --watch --confidence --settings
   --start --for --screenshots`; see USAGE §3), and anything the console
   grows must be reachable from it. The rendered reference scene stays for
   the unit suites and nowhere else.

Non-negotiable product constraints (§132) are unchanged: no cloud dependency, no
mandatory facial recognition, no autonomous enforcement, no hidden telemetry, **no
automatic Internet downloads**, no AI claim without evidence. Camera passwords
must never reach logs, UI payloads, URLs, error reports or tests. Network is
available at install time only; never at runtime.

---

## 2. The exit crash — resolved, and what it actually was

**Symptom.** The console suite passed every test and the process then died with
`0xC0000374` (heap corruption) during interpreter shutdown. `pytest` never
printed a summary and `tasks.py ci` failed at `console · tests`.

**Cause.** One line in `ConsoleWindow.__init__`:

```python
detector_factory=lambda: detector_for(self._model)
```

The closure over `self` put every window in a reference cycle with its own
node (window → node → factory → cell → window). A window was therefore no
longer freed when its last reference went — at fixture teardown, with the
`QApplication` alive — but whenever the cyclic collector next ran. For the last
few windows of a session that is interpreter shutdown, **after PySide has
already destroyed the `QApplication`** in its own atexit cleanup. Destroying a
`QMainWindow` at that point corrupts the heap.

**Evidence, in order.**

- A pytest plugin reporting live objects at `pytest_sessionfinish`:
  **9 `ConsoleWindow`s, 6 runners, 6 pipelines alive, every window held by a
  `cell`**, and only `MainThread` running. (No thread leak — the handoff's
  earlier check of `decode:` threads had missed nothing.)
- Forcing `gc.collect()` after **every** test: exit 0, zero windows alive at
  the end, `QApplication.instance()` already `None` by Python's atexit.
- Binding the factory to the value instead of `self`: **4/4 runs exit 0**, zero
  windows alive at session end. Then 3 further clean runs plus the CI run.
- The new regression test fails against the old lambda with
  `held by ['cell']` and passes against the fix. Verified both ways.

**Why the earlier bisection pointed at four files.** Whether the last windows
were collected before or after the app died depended on when an automatic
gen-2 collection happened to run, which depends on allocation counts — so
reverting *any* large enough diff shifted the timing and "fixed" it. The
`pipeline.py`+`node.py` result was that, not a cross-thread bug. The
lock/leak changes in `decode.py` are still correct and kept.

**One false lead of my own, for the record.** `pytest.ini` sets `-q` and the
runner adds another, and at `-qq` pytest prints **no summary line at all**. A
missing "N passed" is not evidence of a crash. Only the exit code is.

**Guard.** `test_a_closed_console_is_freed_the_moment_its_last_reference_goes`
in the console suite: a weakref to a closed window with cameras that have run
must be dead after `del`. Failure names the holder types so the fix is a lookup
rather than a bisection. `assert_freed()` beside it is the helper.

---

## 3. What else was completed this session

### The contact point crosses the Rust boundary (ABI 5 → 6)

`ground_contact()` — the mask's lowest lit row, the reason segmentation exists —
was **computed in Python and used by nothing**. The Rust projection still
received a box. That is the **fifth** instance of the repository's recurring
defect (correct, tested code that nothing calls). Now:

- `core/src/ffi.rs`: `CDetection` gains `has_contact: u32, contact_x, contact_y`
  (the `_pad` slot became the flag; 48 → 64 bytes). `CTrack` gains
  `contact_x, contact_y` (136 → 152). `ABI_VERSION = 6` on both sides; the
  struct-size guard covers both.
- `core/src/tracking.rs`: `Detection.contact: Vec2` (`Detection::from_box` gives
  the bottom-centre), `Track.contact`, and `project_point()` replaces the box
  projection everywhere. A coasting track moves its contact **with** the box
  rather than re-deriving it from the rectangle.
- `engine/sentinel/core.py`: `ground_contact()` moved here from `segment.py`
  (it must run whether or not a model is installed), `ContactPoint`,
  `Track.contact`, and `Tracker.update()` fills the flag for every detection.
  A box-only detector sends its bottom-centre and gets **exactly** the answer it
  always did.
- Flag set beside a NaN is a caller's bug; the core falls back to the box rather
  than projecting a NaN latitude. Tested.
- `video_view.py` draws the contact as a dot in the track colour; the console
  test reads it back **as pixels** at the contact and asserts nothing is drawn
  at the box's bottom-centre.

Tests: 3 new Rust (60 total), 2 new engine (561), 3 new console (54). All
green; `cargo fmt`, `clippy -D warnings`, docs lint, offline audit clean.

### Two things the live photograph found

- **The lower panels could be squeezed to a header row.** On a short window
  the video view's own minimum size took every pixel and the incident and
  track panels got what was left: the status bar said "2 tracked now" above
  an empty table. `LOWER_PANEL_MINIMUM_HEIGHT` and a non-collapsible
  splitter fix it; the test fails at 85 px without the guard.
- `python tasks.py console` refused arguments, so `--model` could not be
  passed through the task runner. It now passes everything after `console`.

(A first live run also showed three zones and overlapping zone labels. That
was a person at the keyboard clicking the real window the tool puts on the
desktop, not a defect — a trace of `_add_zone` calls in an unattended run
shows exactly one. Worth knowing: the screenshot tool's window is live.)

### Three things the operator's own log found

The user ran the packaged `SentinelVision-dev.exe`, pressed Start and Stop, and
pasted the log. It showed:

- **"no detection model"** — with `models/yolov8n-seg.onnx` in the checkout. A
  frozen build looked in `_internal/models` (`sys._MEIPASS`), where no operator
  would put anything. `paths.models_directory()` now uses `models/` **beside the
  executables**; `tasks.py package` copies any `models/*.onnx` there; `sentinel
  where` prints it; `HOW TO RUN.txt` says so.
- **"restored 3 camera(s)"**, all `device:0`, all started: MSMF refused two with
  `-1072873821`, every pane reconnected in a loop, and the one that worked
  managed one frame. `Node.add_camera` refuses a second camera on the same
  **live** source by name (the redacted form, never the credential); a duplicate
  already in the database is restored, faulted `same source as X; not started`,
  and `start()` leaves it alone. Files are exempt: a replay may back any number
  of cameras, which is how multi-camera correlation is tested. The console shows
  the refusal in a message box instead of raising out of the slot.
- `QFont::setPointSize: Point size <= 0 (-1)` once at start-up. Not chased:
  the stylesheet sets fonts in px, and something copies such a font and asks
  for its point size. Cosmetic; find it with `QT_FATAL_WARNINGS=1`.

Also verified: the packaged console, closed with `taskkill /PID` (WM_CLOSE, no
`/F`) after 8 s, **exits 0**. The shutdown path is clean in the real binary,
not only in the test process.

### The console can now take things back, and zones have kinds

The user's verdict on the packaged console was blunt and right: no way to remove
a camera, no control over zones beyond one red square, no camera positions on
the map. Now:

- `Node.remove_camera` (stops and drains first; evidence kept), `remove_zone`
  (drops the zone rules when the last goes), `replace_zone` (audited with the
  before/after kind). `Store.delete_camera` / `delete_zone`.
- Toolbar: **Remove camera**, **Move on map**, **Add zone…**. Lower-right panel
  is now tabs: *Tracked objects* | *Zones* (list with kind, size; Change…,
  Remove).
- `ZoneDialog`: name, the five `ZoneKind`s with a one-line meaning each, size,
  and placement in front of the camera or by clicking the map.
- `MapView.begin_pick()` / `picked` signal / `_from_local()` — the inverse
  projection, tested to 5 cm round trip. Picking needs a placed camera: with
  no origin a click is a click on nothing, and the dialog does not offer it.
- Zones coloured by kind (`theme.zone_colour`); labels carry the kind.
- Zone ids are the first unused, not count-plus-one, which after a removal
  overwrote a live zone through the upsert.
- Photographed on the reference scene: `03-running.png` (toolbar, two kinds on
  the map), `08-zones.png` (the tab).

### Zones can be drawn, reshaped, scheduled — the first slice of the map programme

- `MapView.begin_draw()` / `drawn`: corner-by-corner outlines; `begin_edit()` /
  `edited`: drag corners, click an edge to add one, right-click to remove (never
  below three), drag inside to move; Enter applies, Esc reverts. `zone_at()`,
  `select_zone()`, `zone_clicked`.
- `ZonePropertiesPanel` beside the zone list: name, kind, schedule, dwell,
  release, accept-uncertain. Apply / Revert. `ZoneDialog(ring_given=True)` asks
  only name and kind after a drawing.
- `zones.ring_problem()` (Shapely `is_valid` + convex-hull area) runs inside
  `Zone.__post_init__`, so no path can create a figure of eight. Found the hard
  way: a bow tie's signed area cancels to zero, so "no area" fired before
  "self-intersection"; the hull test fixed the order.
- `Node.replace_zone` audits *what* changed — every field, with corner counts —
  via `_describe_zone_change`.
- 8 console tests drive the map with `QtTest` mouse and key events (corners land
  within 0.5 m of the click); 3 engine tests for ring validity; 1 for the audit.
- Photographed: `08-zones.png` (list + properties), `04-plan-view.png` (a
  pentagon drawn beside two squares, selected with a solid outline).

**Still not done, honestly:** moving a zone by drag is there, but circles,
rectangles-by-drag, tripwires and corridors are not; a camera's first placement
still needs the dialog; there is no basemap under any of it.

### Docs

FEATURES.md +8 rows (155 `TESTED` of 379). STATUS.md counts corrected (they
were stale: 648 → 732 tests, 36 → 41 diagrams) and a row for the contact
crossing. USAGE §7 says the dot is where the position came from. README file
map.

---

## 4. Repository state

Branch `Phase2`. The previous session's work was already committed as three
commits (`90b11c1`, `1590ec1`, `86e6783`), so the five-commit split proposed in
the last handoff no longer applies. This session's changes are committed on top
of them — see `git log`.

The three executables were rebuilt by `tasks.py ci --package`; see §5 for the
launch check and screenshots.

---

## 5. Verification record (2026-09-03, 17:39)

- `python tasks.py ci --package`: **green**, twice on this tree's final shape
  (once before the layout guard, once after). Stages: source audit, binary
  audit, docs lint (41 diagrams), rustfmt, clippy, `cargo test` (60), release
  build, engine (561), console (54), engine with the network poisoned, package,
  and both `sentinel.exe` launch checks.
- Console suite exit code, offscreen, on the fixed tree: **0 in 8 of 8 runs**
  (4 with the diagnostic plugin, 3 plain, 1 inside CI). On the unfixed tree:
  `-1073740940` in 8 of 8.
- Executables in `dist/SentinelVision/`, each 45.1 MB, built 17:39:
  `SentinelVision.exe`, `SentinelVision-dev.exe`, `sentinel.exe`.
- `tools/screenshot_console.py --live --segment`, native platform, 290 font
  families, `device:0`, a real person in frame, unattended: 1 camera, 1 zone,
  2 tracks (`person` 0.86, `bottle` 0.58), 16 fps analysed. Looked at:
  `live-03-running.png` — model named with digest in the toolbar, both tracks in
  the table with class and confidence, both on the plan view, incident and track
  panels at full height. `live-07-camera-view.png` — the mask tint follows the
  silhouette (head, shoulder, raised arm), not the box; the contact dot sits at
  the bottle's base and at the frame edge where the body leaves the picture;
  no label overlaps another or the readout.
- Track ids reached #3 and #5 within 224 frames for two stationary objects:
  the fragmentation number to beat with appearance re-ID.

- Packaged `SentinelVision-dev.exe` launched, closed gracefully after 8 s:
  **exit 0**, log ends `console exited with 0`.
- After the console revamp, `--live --segment` again (`live-03-running.png`,
  `live-08-zones.png`): toolbar with Remove camera / Move on map / Add zone…,
  a restricted and an exclusion zone drawn in different colours and labelled
  with their kind, the Zones tab listing both, one person at 0.86 held 146
  frames / 9.9 s beside a couch at 0.39, 17 fps. Lower panels readable on a
  shorter window.

---

## 6. What is next

**The map/zones/control programme now has a written build order** — ROADMAP.md,
"Map, zones and control — the build order", sixteen slices from a four-lens
design pass (operator, geospatial correctness, evidence integrity, platform) and
a synthesis against FEATURES.md, which gained 47 rows and 22 amended notes.
Slice 1 ("Zone adjudicability: what the cameras can rule on, shown before a zone
is armed") is specified in the workflow output down to function names and test
names; see `synthesis.json` in the session scratchpad if it still exists, else
the ROADMAP text. Its separable part — the site clock — is already done:

- **Defect found by the design pass and fixed:** zone schedules were evaluated
  in UTC while the docstring promised local time, so 18:00–06:00 typed in
  Beirut armed at 21:00. `ZoneEvaluator(site_tz=…)`, `EventEngine(site_tz=…)`,
  `Pipeline(site_tz=…)`, `Node(site_tz=…)` defaulting to the machine's zone;
  the after-hours condition reads `19:30 UTC+0300 falls within …`;
  `Event.occurred_at` stays UTC; the properties panel says which clock. Tests
  use a fixed offset, never an IANA name: Windows has no tz database without
  the optional `tzdata` package.
- **Honest limit the pass surfaced:** `Store.save_zone` overwrites the ring in
  place, so an event raised before a reshape now points at a shape that no
  longer exists. Slice 5 (versioned geometry) is what lifts that; until then
  the audit row is the only record of the previous outline.

### Slice 1 of the map programme is built: zone adjudicability

The engine half was written by hand after both workflow implementers died on a
session limit — worth knowing before trusting a workflow with a long slice.

- `coverage.sigma_bands(pose)` — the 0.5 / 1 / 2 / 5 m 1σ contours, found by
  bisecting image rows (error falls monotonically down the frame, which is what
  makes bisection valid). `lru_cache`d on the frozen pose: ~2,000 FFI calls,
  fine once per placement, ruinous per repaint. A cache-hit test guards it.
- `coverage.zone_report(ring, cameras)` → area, covered / outside / confident
  fractions, contributing cameras, best and worst σ. `_footprint_polygons` is
  now the single union path shared with `analyse`, and a test asserts the two
  agree to 1e-6.
- `zones.zone_warnings(zone, report, others)` — six warnings in a fixed order,
  duck-typing the report so tests need no camera pose.
- Console: bands painted nested under the zones, a compact measured legend, the
  Covered column with a ⚠ glyph and warning tooltip, a "What the cameras can
  rule on" group at the *top* of the properties panel, the live coverage line
  in the drawing band, and a hatch over the part of the selected zone nothing
  can see.

**Three defects this slice found in its own code, all fixed and tested:**

1. A 4 m square built from geodesic points measures 3.999990 m, so
   `threshold <= half_width` dropped the 2 m band from a zone sized for exactly
   that band — 0% confident on a zone that is half inside it. Five microns.
   `_WIDTH_TOLERANCE` and `test_a_zone_the_width_of_a_threshold_is_not_dropped_by_rounding`.
2. `worst_sigma_m` was measured against the whole zone, so a zone with one
   corner in the blind foreground reported "beyond 5 m" when the real answer was
   1 m and the hole was already reported by `covered_fraction`. Now measured
   against the reachable part.
3. `_refresh_placement` called `set_cameras` (which refits) before `set_zones`,
   so the fit was always one zone behind and a zone drawn behind the camera was
   framed out of the only view that could explain it.

**And three the photograph found, which no test would have:** the properties
panel crushed its own rows into each other on a short window (a `QFormLayout`
given too little height compresses rather than clips — it is a `QScrollArea`
now); the zone list's five columns were cut off; and the legend was first
clipped to "…not de", then measured and 489 px wide on a 600 px view.

### Slice 2 is built: one selection, honest modes, a lock

Console only — no engine change. `selection.py` holds a `Selection` (a track
keyed by *(camera_id, track_id)*, because ids repeat across cameras) on a
`SelectionBus`; `MapView.hit_test` tests smallest-first; the map, the wall, the
track table and the incident list all show the same one thing, and Escape
clears it. Select ／ Draw ／ Measure are mutually exclusive checked buttons whose
state is *derived* from the gesture, so a drawing that closes on its own
unchecks Draw. The console opens in Monitor and everything that changes the
site needs Configure, which re-arms after ten idle minutes and audits both
edges. Hovering reports what is actually known; a `CAMERA_FALLBACK` position is
drawn as a dashed ring and says "not located". The status bar carries the
ground point under the pointer, named from the camera it is measured from, and
Ctrl+C copies it.

**What the freeing test caught, twice, while this was being written** — both
the same defect in different disguises, and both would have shipped silently:

1. `button.clicked.connect(lambda _checked, m=mode: self._choose_mode(m))` — a
   closure cell holding the window. Replaced with a bound slot that resolves
   the mode from `self.sender()`.
2. `_set_configuring(False)` called before `self.status` existed, so
   `_set_status` raised inside a Qt slot; Qt swallowed the exception and *its
   traceback* then held the window alive for ever. Construction order fixed.

Exceptions raised in Qt slots are silently retained. That is worth remembering:
it turns any construction-order slip into a leak with no visible symptom.

**Three more found by its own tests:** an empty `[]` meant "not measuring", so
measure mode ended the instant it began (an explicit flag now); `_mode_changed`
called `_refresh_status`, overwriting the very message that said why a mode was
refused; and a `_touch()` inserted by a sloppy patch landed in
`_selection_changed`, which would have held the Configure lock open for ever
just by clicking around — there is now a test for exactly that.

### A five-dimension adversarial review of slice 2, and what it found

Forty agents: five reviewers (Qt lifetime, logic, honesty, tests, integration)
and a skeptic per finding told to refute it. **35 findings, 21 confirmed, 14
refuted.** Every confirmed one is fixed. Worth running again on the next slice
— the two most expensive defects were invisible to a green suite.

**The blocker all five dimensions found independently.** `MapView._hover` held
two types: the rubber-band pointer (a `QPointF`, pre-existing) and what the
pointer is over (a `Selection`, new). One mouse move after pressing Draw raised
`AttributeError` inside `paintEvent`. Qt swallows that, so the plan view simply
stopped drawing — no corners, no rubber band, no scale bar, no banner — and the
retained traceback pinned the widget and a still-active `QPainter` past their
last reference. The rubber band is `_draw_cursor` now, and two tests cover
draw-plus-move-plus-repaint (verified failing against the old name).

**Two more ways past the Configure lock**, on top of the Ctrl+O / Ctrl+P
shortcuts found earlier: the Zones tab has its *own* Add zone button, and
`_teardown()` re-enabled Add camera flatly whenever a run stopped. A lock with
three doors and a bolt on one is not a lock.

**Honesty defects the geometry did not support:** "Copy position" put a bare
coordinate on the clipboard with neither its ±1σ nor the fact that it might be
a `CAMERA_FALLBACK` — pasted into a radio call, that sends somebody to a place
the system never claimed. Hovering a fallback led with "0.0 m at 0° from the
camera", a measurement of nothing. A zone no camera can see reported "beyond
5 m everywhere" as though the error were the problem. And coverage rounded
*up*, so a zone with real blind ground read "Covered 100%" — it reads 99% now,
and only a genuinely complete fraction may say 100%.

**Tests that could not fail:** the ground readout was only ever tested by
calling its slot, so the feature could be unwired and stay green (it *was*
unwired — the tooltip promised Ctrl+C and nothing was bound); the video-pane
test never set a selection; `VideoView` click-to-select had no test at all; and
the lock tests iterated the very list they were meant to police.

**Note for next time:** the reviewers wrote scratch `test_zz_*.py` files into
`apps/console/tests/` and one edited a source file to prove a finding. Both
were cleaned up, but a review workflow should be told to work outside the repo.

### Three parallel workflows, 41 agents, and the wiring that made them count

Run in this order, each with disjoint file ownership per agent, a skeptic per
piece and a repair pass — the only arrangement that has not lost work to a
collision:

1. **Slices 3 and 4** (4 builders): `Node.camera_health()`, the `sites`
   migration and `SiteFrame`, `CameraListPanel`, and camera dragging / dark
   hatching / far-edge styling in `MapView`.
2. **Six new engine modules** (6 builders): `plates`, `faces`, `registry`,
   `orthophoto`, `search`, `auditing` — 231 tests, none needing a model file,
   because every heavy model sits behind an injectable seam. Verified by hand:
   a lone 99%-confident frame yields `???????`, three disagreeing frames yield
   `B7?4921`, and `faces` is inert while disabled with the band at 0.363/0.5.
3. **Wiring** (4 builders): the register behind migration 5 and
   `Store.register`; structured audit records behind migration 6 with the prose
   unchanged; plates read per vehicle track in `Pipeline`; the camera list,
   dragging and dark cameras in `app.py`; the `InvestigationPanel`. Skeptics
   were told to *remove each wire and see whether a test failed*; three of four
   proved it, and the fourth's repair added the failing test.

Then by hand, the two things no agent could reach from its own files: the
Investigation tab placed in the window and fed the node's store, and a Plate
column in the track table showing `display` (never `text`) with the agreement
count for a thin read. Both have tests that fail if the wire is pulled.

**Lessons that cost something today.** Reviewers must be told to keep scratch
files out of the repo (probe files broke CI once). No literal URL may appear in
shipped source even as a docstring example (an illustrative RTSP credential in
a docstring failed the offline audit). Exceptions raised inside Qt slots are
silently retained and pin the widget — a third route to the exit-time heap
corruption, and the freeing test caught it. And the recurring defect, "tested
code nothing calls", is now the *default* outcome of a build workflow: budget a
wiring pass for every build pass, and leave operator-facing rows `PLAN` until
it has run.

**Still unwired, honestly:** nothing enrols into the register and the retention
job does not sweep it; no event is raised from a plate and no reading persists;
`orthophoto` and `faces` have no caller at all; the audit chain covers only rows
that carry a hash; the site record has no console screen.

### What the real camera said, answered: zone classes, and fragmentation measured

Driving the packaged console on the laptop camera produced "1 couch in Room
(HIGH, risk 55)" and, earlier, "A bottle entered Room". Three agents, one
workflow, every skeptic's mutation check passing:

- **Zone classes.** `Zone.classes` (empty = any), `Zone.watches(label)`,
  consulted by every presence rule; migration 7; a picker in the properties
  panel offering only the detector's own vocabulary; `zone_warnings(labels=…)`
  naming a filter the detector can never satisfy. The rule that matters most:
  a filtered zone **never fires from a motion detector**, which cannot say
  what it saw. On the reference scene a person-only zone raises exactly the
  person subset of the unfiltered zone's events, by deterministic event id.
- **Fragmentation, finally measured.** `tools/measure_fragmentation.py`:
  reference scene 4 tracks for 3 walkers (1.33 — identity swaps at the
  crossing, not temporal splits); laptop camera 7 tracks for one person in
  15 s with the segmenter. Every earlier figure (3, 10, 4, 11) was incidental.
- **Post-hoc re-ID** in `sentinel.reid` — masked HSV histogram, EMA per track,
  `link_fragments` joining only when gap, distance and appearance all agree.
  It reconciles the count; it cannot un-split a live track. That is the
  argument for ABI 7 (appearance into the Rust tracker), and the measurement
  tool is what will show whether it worked. Called by nothing yet.

**Verified in the packaged binary on the laptop camera (exetest4):** unfiltered,
the Room zone raised events for *a bottle, a couch and a person*; with
`classes = ["person"]` stored and the console restoring it (`--start`), the
incident is "1 person in Room" and nothing else fires. Two CLI gaps found on
the way: `sentinel run` builds its zones only from `--zone` and does not
restore stored ones (the console and `sentinel node` do), and there is no
`--zone-classes` option, so a filter can only be set from the console today.

### The recorded gaps, closed (4 builders, every mutation check passing)

- **CLI:** `--zone-classes`; `run` restores stored zones when given none (it
  used to watch nothing and say "No events"); every run prints what it
  watches; `retention` sweeps the register with `--face-days` / `--plate-days`.
- **Re-ID is used:** the correlator joins same-camera fragments into one object
  with reasons on the Association; one person in three fragments is one
  object. Same-camera links are labelled as such in the console, the report
  and the JSON (a new `same_camera_associations` key; the old key keeps its
  meaning).
- **Audit tab:** rows, filters, the field-by-field diff, and *Verify chain*.
  Two skeptic findings worth remembering: the `detail` column is outside the
  hash, and `Node._audit` was writing microsecond timestamps that broke the
  first record's hash every time — both fixed.
- **Plates persist** (migration 8) and a confident, resolved reading of an
  enrolled plate becomes a register sighting; unconfident readings never do.
- **The console's incident timeline** had the same epoch-as-offset defect the
  evidence report had this morning (`t+1788513275.8s`); fixed, with a test.

**Found by the photograph, not by any test or skeptic:** the Audit tab's first
picture read "the chain breaks at chained record 1 of 1" on a fresh database.
`Store.audit_record` hashed the record with its microsecond timestamp and
stored milliseconds, so no row could ever re-hash to its own chain hash. The
skeptic had reported this fixed; it was not. Fixed by truncating *before*
hashing — what is hashed is what is written — with a store test using a
microsecond stamp. The lesson generalises: a "verify" feature must be
photographed verifying real rows, not only its own fixtures.

**Still unwired, honestly:** nothing enrols into the register from the console
(no People/Vehicles panel yet); `orthophoto` and `faces` have no caller; the
site record has no console screen; the audit chain still covers only rows that
carry a hash.

### The harsh camera battery, and the three things it found

Four phases through the packaged binaries on `device:0`, nothing simulated,
every number below measured (`scratchpad/harsh_camera_battery.py`; a
person-only RESTRICTED zone `Room` beginning 2 m in front of the camera):

- **A.** 60 s, one seated person: 970 of 1787 frames analysed (16 fps), 12
  track ids, **0 events**. A 20 s re-run: 345 frames, 5 ids, `A person
  entered Room` at 1 s and `remained for 8 s` at 8 s, one incident, nothing
  else — no flapping for a seated person; one id held 19.2 s.
- **B.** Five open/close cycles of `SentinelVision-dev.exe --start`: exit
  codes `[0, 0, 0, 0, 0]`, nothing killed, no process left.
- **C.** Two `sentinel run device:0` at once: the second reconnected once,
  analysed **one frame in ten seconds, and exited 0**.
- **D.** Fragmentation, 3 × 30 s, one person plus furniture (`--objects 3`):
  7, 13 and 3 tracks; 1.33, 2.67 and 1.0 per object after linking.

Then a ten-second probe of *where* the person's contact landed, which is the
finding that matters: **every person sample sat on the frame's bottom edge
(rows 0.989–1.0) and every one projected to 2.16 m ± 0.13 m.** The person was
at the desk, half a metre from the lens, feet below the picture. The zone
began at 2 m. `A person entered Room` was somebody who never left their chair,
placed confidently two metres away by the nearest ground the camera could see.

What changed, each with a test that fails without it:

- **`FRAME_EDGE`** (`core.py`): a box whose lower side sits within 4 % of the
  frame's bottom (measured 0.975–1.0 for a seated person; 2 % missed it) is one the frame truncated; its projection becomes a bound —
  the middle of the camera-to-edge stretch, radius reaching both ends — and a
  zone beginning inside the stretch sees UNCERTAIN, not INSIDE. No ABI change:
  the Python tracker keeps the pose and judges the core's answer. The console
  words it ("feet below the frame — between the camera and 2.2 m") and draws
  it hollow; the linker keeps it (it is about the object, unlike a fallback).
- **A stay survives a track split** (`zones.py`): a young same-class track
  appearing within a plausible walk of where a stay's own track went quiet
  (no detection this frame, ≤ 3 s since its last) inherits the stay. No second
  ENTERED, the dwell continues, the superseded id cannot reopen it, and a stay
  whose id died waits the exit delay for the object to come back. Loitering is
  keyed on `Presence.identity` (zone, origin track, start), so it fires once
  per stay whichever id carries it. Phase D is why: every split inside a zone
  was a fresh ENTERED and a loiter timer back at zero.
- **The summary names its tracks** (`pipeline.py`): `track 1   person …`, and
  `distinct tracks … (ids issued; one object can hold several)` instead of
  "distinct objects". Phase A's silence was unreadable without it.
- **A starved live run fails** (`cli.py`): no frame, or under 1 fps after 5 s,
  prints `STARVED … is another program using the camera?` and exits 1. Phase C.

Re-tested through the rebuilt binaries on the camera, same seated person:
phase A raised **nothing** in two 20 s runs (person track held 16.9 s and
18.7 s, every track named in the summary); phase C's second process once
printed `STARVED 1 frame(s) in 10s` and exited 1, and once was given the
camera by the OS — 14 frames, 3 reconnects — and ran. The analysed frame rate
in that round (27–78 of ~520 in 20 s) is **not** a measurement of the build:
another application held the machine at 93 % CPU throughout; check
`Get-Process` before quoting throughput. The hand-off has unit tests only —
no moving person was in front of the camera to split a track on demand.

Not fixed, recorded: the core's speed comes from the raw projection, so a
subject with feet below the frame reads as standing still whatever they do;
the OS shares a webcam between processes and the one-camera guard is per node
only; the linker leaves 1.3–2.7 tracks per object on a moving scene — ABI 7.

### What the operator's screenshot said: unreadable buttons, unasked-for bottles

Two complaints, both right. The toolbar at a laptop's display scale read
`d camer`, `ve on m`, `onfigur` — twelve buttons, a picker, a spin box, a
checkbox and two captions in one `QHBoxLayout`, wider than the screen, every
button squeezed below its text. And a jar on the shelf and a phone on the desk
were tracked as `bottle` and `cell phone` at 0.43–0.51 in the same green as
the person. Fixed, with tests:

- **Toolbar in two rows** (cameras + the Configure lock; map + zones), captions
  in the status bar, a test that no button is narrower than its size hint at
  1280 px, and a **reason in every disabled control's tooltip** (`Locked. Press
  Configure…`, `Stop the analysis before adding a camera.`, `No incident to
  export yet.`). Qt shows a tooltip on a disabled widget; that was the one
  channel a greyed button had.
- **Watched classes** at the detector (`classes=` on both ONNX detectors,
  `detect.WATCHED_LABELS` = person and five vehicle classes as the console
  default, Detection → Watched classes… with a per-machine `QSettings`
  INI, applied at the next Start). The detector's reported vocabulary shrinks
  with the list so the zone picker cannot offer a class the detector drops;
  an unknown name is refused. Seen live from source: `segmenter ready … 6
  class name(s)`.

**Deferred, on purpose:** `sentinel run --classes` — `cli.py` was owned by
the basemap builder while this was done; add the flag (default
`WATCHED_LABELS` when a model is given) and a test. The watch list belongs on
the **site record** beside the identity switch, not in a per-machine INI;
move it when the site screen exists.

### 2026-09-05 — "hallucinations", "the buttons do nothing", and the exe as the test medium

The operator's two complaints, read from their own machine rather than guessed:

- **Every run on 2026-09-04 loaded the model with 80 classes** (`segmenter
  ready … 80 class name(s)`, log lines 465–516). The watch-list commit
  (`7a1b5df`, 23:10) post-dates the packaged build (22:00), so the binary they
  ran still tracked couches and jars. Nothing in the tree was wrong; the exe
  was stale — which is why rule 8 below exists.
- **The audit trail for 22:51–23:00** shows unlock → start → stop → remove
  camera → re-add → relock → start → stop, and never `camera.placed` or
  `zone.created`. Place… and Add zone… were grey whenever the console was
  locked, and a greyed button with a tooltip nobody hovers is "a button that
  does nothing". The current database: one `device:0`, unplaced, no zones.

What changed, each with tests that fail without it:

- **A confidence floor for classifiers**: `DEFAULT_CONFIDENCE = 0.50`
  (engine default stays 0.35; on the laptop camera the person held 0.86, the
  couch 0.39, the jar 0.43–0.51), per machine in the Watched-classes dialog
  (now "Watched classes and confidence…"), range 0.10–0.95, shown in the
  status line as `≥ 0.50`, applied at the next Start. `detector_for` drops
  `confidence_threshold` for motion the way it drops `classes`.
- **A locked control answers a click.** The window installs itself as an event
  filter on every control the lock disables (a disabled widget still runs its
  filters — checked with a probe, not assumed), and `_offer_unlock` names the
  control, says the site is locked, and offers to unlock and carry on; yes
  is the audited Configure entry and then `control.click()`. Draw goes through
  the same offer from `_mode_button_clicked`. `lock_label` in the status bar
  reads MONITOR / CONFIGURE at all times.
- **Escape never relocked.** The tooltip, USAGE and FEATURES said it did.
  Corrected and pinned by a behavioural test.
- **Exceptions inside Qt slots reach the log.** `_report_uncaught` is installed
  as `sys.excepthook` before the window shows (PySide's `PyErr_Print` calls it
  — probed), logs CRITICAL through the redacting filter, names the exception
  in the status bar, and clears `sys.last_*` so the traceback cannot pin the
  widget (§7's third route to the exit crash).
- **The console has a command line** (`build_parser()`): `--camera --place
  --zone --zone-classes` (the CLI's own parsers), `--watch --confidence`
  (this run only, never persisted), `--settings FILE`, `--for SECONDS`
  (prints the node summary plus every track with its class, then closes),
  `--screenshots DIR` (window and every panel). `seed_site` goes through the
  node and is audited; it is listed in the structural lock test as reachable
  from `run()` only. A real process run on the reference file: six PNGs,
  summary, exit 0.
- **`python tasks.py exetest`** (`tools/exe_camera_test.py`) drives
  `dist/SentinelVision/SentinelVision-dev.exe` on `device:0` in an isolated
  data directory and settings file, seeds a placement and a person-only
  `Room` zone 2 m ahead, runs `--start --for N --screenshots`, keeps stdout,
  stderr and the log beside the pictures under `dist/exetest/<stamp>/`, and
  reads the summary back into a PASS/FAIL table.
- **`PRODUCTION_READINESS.md`** — the hostile audit: 106 TODOs (7 P0, 28 P1,
  48 P2, 20 P3, 3 P4), each with evidence and a definition of done, plus the
  release gate. Published as an artifact as well.

**What the first packaged camera run found — the real "buttons do nothing".**
`exetest` at 09:03 on the rebuilt binary: the pipeline ran (1,335 frames, a
person at 0.86 for 28 s, no couch, no bottle, the floor and watch list in the
status bar), and the operator, sitting at the machine, clicked Place… and Add
zone… while it ran. Both raised `libshiboken: Internal C++ object … already
deleted` — `PlacementDialog`, `ZoneDialog` and `AddCameraDialog` carried
`WA_DeleteOnClose`, so `QDialog.done()` deleted them the instant OK was
pressed, *before* `exec()` returned, and the slot read a spin box that no
longer existed. Qt swallowed it; the excepthook added an hour earlier is what
made it visible. No test had ever pressed OK on those dialogs — every test
called the slot beneath them. Fixed (read, then `deleteLater()`), with five
tests that press OK the way Qt does (`accept()` plus
`sendPostedEvents(DeferredDelete)`) and a structural test forbidding the
attribute on a dialog read after `exec()`. Two more from the same run: the
report died printing `≥` to a cp1252 terminal and the timed run never closed
(stdout/stderr now replace; `_finish_timed_run` closes in a `finally` and
dismisses open dialogs), and `camera-device:0.png` became an NTFS alternate
data stream (`_file_safe`). The tool now fails a run that logged an exception
or outlived its `--for`.

**Verification, honestly.** Console suite 374 passed (was 332). Engine
`test_detect` green. Static guards green. **`tasks.py ci --package` is red at
HEAD `d38c64e`** on three `test_identity.py` tests — `migration_nine_is_the_newest`,
`migration_nine_arrives_and_leaves`, `plate_reader_is_built_…` — which
belong to the `site_declared` migration 10 and `_SwitchedPlateReader` that
arrived in the same commit from the other stream of work; reproduced in a
clean worktree at HEAD, and none of those symbols appear in this session's
patches. The package was therefore built directly (`tasks.py package`) and
the camera run made against it; see the verification record at the end of
this section. **Then fixed, on `continue`:** the three tests were stale
against deliberate work — migration 10 `site_declared` (the node's own
placeholder site row must not freeze an origin nobody chose) and
`_SwitchedPlateReader` (plates *off* reaches a running camera at its next
read). They now assert what the code promises, and four tests were added for
the promises themselves, which nothing had tested: the switched reader
returns nothing once the site says no; `set_identity` clears the event every
running reader watches; a switch written before any camera is placed leaves
the origin following the cameras (`declared=False`); a declared site is
returned as it is whatever the cameras say. `test_identity.py`: 40 passed.
`python tasks.py ci --package` was then run on the whole tree: **green**,
all thirteen stages (the source audit first caught an `rtsp://…` in a new
docstring — §7's rule, again — reworded), executables rebuilt at 10:05, and a
fourth camera run on that exact binary passed (34.3 s, 427 frames, six
pictures, no exception; nobody in frame). Everything is uncommitted; the
change set is `app.py`, `test_console.py`, `test_identity.py`,
`exe_camera_test.py`, HANDOFF, STATUS and the new `PRODUCTION_READINESS.md`.

Camera runs on the packaged binary (`dist/exetest/<stamp>/` holds the pictures,
stdout, stderr and the log of each):

| run | binary | what it showed |
|---|---|---|
| 09:03 | 09:02 build | 1,335 frames; one `person` at 0.86 for 28.2 s; no couch, no bottle; `≥ 0.50` and the watch list in the status bar. The operator clicked Place… and Add zone…: both raised on a deleted dialog (fixed since); the report died on `≥` in cp1252; `camera-device:0.png` became an alternate data stream |
| 09:17 | 09:16 build | dialog fixes in; `6 class name(s)`; a person tracked; closed by hand at 12 s, before the timer — no picture, no summary (fixed since: a closed window still reports) |
| 09:25 | 09:24 build | PASS with a caveat: 570 frames, six pictures, summary, no exception, closed itself at 30 s; nobody in frame |
| 10:06 | 10:05 build (CI) | PASS with a caveat: 427 frames, six pictures, summary, no exception, closed itself; nobody in frame |

**Hard-won facts from this session:**

- Two editors on one tree at once: files changed under the run twice
  (`test_console.py` at 23:18, then a whole commit at 00:17 that folded this
  session's files in with `basemap`, `cli`, `node`, `site`, `store`,
  `register_view`). Patch by exact string with a count assertion; never
  `Write` a file somebody else may hold open.
- A background `cmd; echo EXIT $? >> log` reports the *echo's* exit code to
  the harness. Read the log, not the task status.
- `local_ci.py` adds `-q` to a `pytest.ini` that already has it: `-qq`, no
  summary line. Count with `--collect-only -q | tail -1`.
- Appending tests to `test_console.py` shadowed an existing `_say_yes(monkeypatch)`
  helper and broke four unrelated tests; grep for helper names first (now
  `_answer_the_key_yes` / `_answer_the_key_no`).
- A full-page headless screenshot of a 26,000 px page tiles the masthead
  twice; the DOM had one. Ask the DOM before believing a picture of it.

### On "continue in features in to dos" — five audit items closed, FEATURES carries the rest

The audit's blockers and critical items are now rows in FEATURES.md (a
"Production readiness" section keyed by audit id), and the five that were
fully specified and needed no product decision were built, each with tests:

- **REL-03** — the console's export goes through `Node.export_incident`
  (footage + preservation); the "Evidence exported" box says how many clips
  the package carries.
- **UI-01** — `sentinel.version` is the one source of the version;
  `tasks.py package` writes `build.json` beside the executables (commit,
  `+dirty`, build time); About, `sentinel where`, the log's first line, the
  evidence report and `HOW TO RUN.txt` all name the build.
- **SEC-03** — `SENTINEL_ALLOW_PUBLIC_SOURCES` is kept, announced at WARNING
  on every start, logged with the address on every connection it allows, and
  USAGE/SECURITY/.env.example stop saying there is no override.
- **OBS-03** — `faulthandler` writes every thread's stack to `crash.log`
  beside the log; `logs.reset()` hands faulthandler back to whatever had it.
- **REL-02** — recording from the console. Migration 11 `cameras.record`;
  `Store.save_camera(record=None)` keeps the stored flag so a placement does
  not switch recording off; `Node.set_recording` (audited, chained);
  `Node(record_to, record_every_camera=True)` — **`record_to` alone still
  means every camera** (the CLI's `--record`, five existing tests); the
  console passes `record_every_camera=False` and records the cameras whose
  box is ticked; `Node.poll` sweeps retention every ten minutes (first poll
  sweeps at once) and `Node.retention_shortfall` carries the sweep's
  complaint; `CameraHealth` gained `asked_to_record`, `recording`,
  `clips_written`, `bytes_recorded`, `recording_fault`; the camera list has a
  Record column behind the Configure lock and prints `● rec N clip(s)` /
  `recording stopped: …` / `will record when restarted`; `--record` on the
  console's command line ticks the cameras `--camera` names.

Decisions worth knowing: the sweep runs on the node's thread (the GUI thread
in the console) — REL-05 is the item that moves it; a flag set on a running
camera applies at its next start and the status line says so; there is still
no alert that leaves the process (OBS-01).

**What the first camera run with `--record` found.** The camera's pipeline
died before a frame: `Recorder.start()` does `mkdir` on
`recordings/<camera_id>`, the id was `device:0`, and Windows refused the
colon — `NotADirectoryError`, "device:0 stopped unexpectedly", no frames, no
clips, and the exe test's new `--record` check called it FAIL. Two fixes:
`recording.file_safe()` (one rule for the folder, the clip names and the
console's pictures — `device:0` → `device-0`, the id itself unchanged in the
index and the evidence), and the pipeline guards `Recorder(...).start()` so a
recorder that cannot begin becomes `Pipeline.recording_fault`, reported in
`CameraHealth.recording_fault`, the status cell and `summary()` as
UNAVAILABLE while the analysis continues. The recorder is fed before analysis
precisely so the two cannot take each other down; the guard is what makes
that true at start-up too. Tests: a `device:0` camera records under
`device-0/`; a recordings path that is a *file* leaves the camera analysing
with the fault named.

**Verification of the to-dos.** `python tasks.py ci --package`: green, all
thirteen stages (engine 137.5 s, console 113.1 s, package 202 s, both launch
checks), executables 11:02, `build.json` stamped `0.1.0 (f98d754+dirty, …)`.
`python tasks.py exetest --seconds 30 --record` on that binary: PASS with the
caveat that nobody was in frame — 505 frames, **one clip of 3.4 MiB written as
`device-0_20260905-080245_00000000.mp4` at 17.5 fps measured**, six pictures,
no exception, closed itself. Suites now: 60 Rust, 1,138 engine, 382 console.

**23:44 — the executables updated from clean HEAD `2bf57e6`** (the user
committed the to-dos as `f98d754`, `45b591a`, `2bf57e6`): `ci --package`
green on all thirteen stages, `build.json` stamped `0.1.0 (2bf57e6, built
2026-09-05 20:44 UTC)`, and `exetest --record` on that binary passed with the
usual caveat — 691 frames, one clip of 12.5 MiB, six pictures, no exception,
nobody in frame.

**Immediate:**

1. **1.3 appearance re-ID.** The tracker fragments (17 tracks over 15 s on one
   webcam) because association is geometric only. Masks now make appearance
   embeddings possible — the mask selects the object's own pixels, not the
   background in its box. Start with a mature pretrained embedding (OSNet /
   a small re-ID ONNX export, obtained the same way as the segmentation model,
   never downloaded by the product), fall back to a masked colour histogram if
   no model is installed. Association score = geometric gate × appearance
   similarity. Carry the embedding on `Detection`; the Rust tracker only needs
   a similarity matrix, so the ABI change is an optional `f32` pointer per
   update, not per-detection vectors.
2. **2.1 control plane** — `aiohttp` (see previous handoff for why not FastAPI).
3. Console: a recording toggle, a detector picker in the UI, polygon zone
   drawing, and a **camera delete for stale database rows without opening
   the console** (`sentinel cameras remove`).

**Stage 3 of segmentation** (not started): use the mask for zone membership too
(fraction of the silhouette inside the polygon rather than one point), and for
occlusion-aware coasting.

---

### 2026-09-06 — "start doing all the unmade features": four slices, P0 first

The directive was to build what the audit left unbuilt, in its priority
order, one slice at a time with tests, and to say so after each slice rather
than after all of them. Four landed; each ran the full local CI with
packaging and the camera exetest on the rebuilt `SentinelVision-dev.exe`.

1. **Backup and restore (REL-02's data half, DOC-01's DATABASE.md half).**
   `Store.backup_to` uses SQLite's online backup and writes a sidecar
   checksum; `verify_backup` and `restore_backup` check it; `_require_intact`
   runs `quick_check` on open and a database from a newer schema is refused
   rather than mangled; `sentinel backup` / `sentinel restore`. DATABASE.md
   was rewritten to say what exists.
2. **Supervision (REL-01).** `sentinel.supervise`: a stop file, a backoff
   restart loop, and `service install|uninstall|print` that writes a Task
   Scheduler, launchd or systemd definition. `sentinel node` with no sources
   runs the stored cameras, so the service line has nothing to remember.
3. **Camera passwords in the keychain (SEC-02).** `sentinel.secrets` over
   `keyring`; `cameras.credentials_ref` is now populated; a restored RTSP
   camera gets its password back from the keychain; `sentinel password
   CAMERA`; `run`/`node` warn about a password in argv. Both test suites use
   an in-memory keychain so they never touch the machine's.
4. **Accounts and permissions (SEC-01, SEC-14).** `sentinel.accounts`:
   `users` table (migration 12), salted scrypt hashes, roles as sets of
   permissions, lockout after five failures; `sentinel users
   add|list|passwd|disable|enable`; the console offers to create the first
   administrator once (never on a timed `--for` run, which is unattended),
   then asks for a sign-in, or takes `--user NAME` with the password on
   standard input; Configure and Export need a permission and a refusal is
   audited; every console audit row carries `console:<name>`, every CLI row
   `cli:<os account>`. Not built, and said so in the audit: permission
   checks inside `Node`, sessions and an application lock.

5. **The model read once (PERF-01).** `detect.model_info` caches a model's
   description per process, keyed by path, size and mtime; `_output_count`
   is cached the same way. The console's zone picker, watch-list dialog,
   Start check and `--watch` validation all read the one description. One
   session per camera per Start, not four for one camera.
6. **Alerts (OBS-01, and REL-02's last half).** `sentinel.alerts`: raised
   once per (kind, subject) until cleared, audited, fanned out on a daemon
   thread to `alerts.log`, an operator's command and a local-network webhook
   (a public address is refused at construction). The node raises
   `camera.dark`, `recording.stopped`, `retention.shortfall`,
   `analysis.thread_stuck` and `disk.low` (2 GiB watermark) from `poll` and
   `stop`; the console shows a red banner and beeps once per alert. Found on
   the way: a store whose every clip is preserved measured *infinite* free
   space in the sweep and never reported a shortfall (`_free_bytes` now falls
   back to any segment).

Suites after the six slices: 60 Rust, 1,192 engine, 393 console; the full
local CI with packaging was green and the rebuilt `SentinelVision-dev.exe`
ran on the laptop camera through `python tasks.py exetest --record` (see the
verification record below this section).

Two things worth knowing about slice 4. The schema test that fails on any
credential-shaped column exempts exactly one, `users.password_hash`, by table
and name — keep it that narrow. And the first-administrator offer is
remembered when declined (`accounts/first_admin_declined` in the settings),
so a deployment that has not decided on accounts is told once, and the
status bar keeps saying "the audit trail names nobody" until one exists.

## 7. Hard-won facts worth not rediscovering

- **The recurring defect in this repository is correct, tested code that nothing
  calls.** Now found five times: the recorder, `RecorderStats.fault`,
  `LiveStream`, the segmenter before it was wired, and `ground_contact`. Before
  marking anything `TESTED`, grep for importers outside its own module and
  tests.
- **A `QWidget` in a Python reference cycle is a shutdown crash waiting to
  happen.** Its lifetime must be the plain refcount. Never close over `self`
  in a callable you hand to something `self` owns; bind the value.
- **`-qq` prints no pytest summary.** The exit code is the only evidence. Treat
  a single run of anything nondeterministic as noise; use four or more, and
  capture `$LASTEXITCODE` in PowerShell — Git Bash's `$?` after a pipe is the
  exit of `tail`.
- **Git Bash does not split a colon-separated `PYTHONPATH` for Windows
  Python.** Use PowerShell with `;`, or rely on `pytest.ini`'s `pythonpath`.
- **Heredocs are mangled here.** Write patch scripts with the Write tool and run
  them; a `<<'EOF'` with `\n` inside failed to parse.
- `core.py` already had an `ImagePoint` (field-of-view). Grep before naming.
- **Verify the premise, not just the result.** `strings` does not exist on this
  machine; OpenCV 5 substitutes codecs; offscreen Qt has 0 font families and
  renders all text as tofu.
- **onnxruntime ships a Microsoft telemetry uploader**, disarmed in
  `engine/sentinel/telemetry.py`; `tools/binary_audit.py` exists because the
  source audit cannot see a hostname inside a `.so`.
- **Only the thread that owns the node touches the store.** SQLite connections
  belong to their creating thread.
- The Rust core is **C ABI + ctypes, not PyO3** — a GNU-toolchain cdylib
  against an MSVC CPython is a real hazard.
