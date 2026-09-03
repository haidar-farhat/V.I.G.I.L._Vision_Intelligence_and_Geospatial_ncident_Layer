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

### Docs

FEATURES.md +5 rows (151 `TESTED` of 376). STATUS.md counts corrected (they
were stale: 648 → 679 tests, 36 → 41 diagrams) and a row for the contact
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

---

## 6. What is next

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
3. Console: a recording toggle, and a detector picker in the UI.

**Stage 3 of segmentation** (not started): use the mask for zone membership too
(fraction of the silhouette inside the polygon rather than one point), and for
occlusion-aware coasting.

---

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
