# Handoff — 2026-09-03

State of the work at the end of the segmentation slice, what is unfinished, and
the one open defect. Written to be the first thing a new session reads.

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

## 2. What was completed this session

### Instance segmentation, end to end

- `engine/sentinel/segment.py` — `Segmenter` (YOLOv8n-seg via ONNX Runtime) and
  `ground_contact()`, which takes an object's ground point from **the lowest row
  of its mask**, not the bottom edge of its box. That point is what the entire
  position layer rests on.
- `engine/sentinel/detect.py` — `detector_for()` picks motion / detection /
  segmentation by **reading the model file** (output count), never a flag,
  because a flag can disagree with the file and the operator cannot tell which
  won.
- `--model FILE` on `sentinel run`, `sentinel node`, and the console.
  `--no-model` forces motion. `paths.default_model_path()` finds a `*-seg.onnx`
  in the models directory; only segmentation models are auto-selected.
- `devtools/export_model.py` obtains the weights on a connected machine. The
  product downloads nothing, ever.
- The console draws **masks, not boxes**, and track labels no longer stack on
  top of each other.
- The toolbar names the detector actually running, with its SHA-256 —
  replacing a hardcoded `"MOG2 background subtraction"` string that became a
  false capability claim the moment a model was loaded.

**Verified on the live laptop camera, through the console:** one person track at
0.86 held **160 frames / 11.5 s** with speed and heading, plus two correctly
classified stationary bottles, at ~11–14 fps on CPU. The same camera under MOG2
produced **20 tracks and 30 events for one seated person** — face fragments,
curtains and a wall.

Model artifact: `models/yolov8n-seg.onnx`, 13.9 MB, sha256
`f828ccfa4b69332ad8b65c4ffebdf1f36ae9dd861c48faadd6c881e63bd68699` (gitignored,
operator-supplied).

### A live camera no longer stops for good on one dropped frame

This was the serious find. `VideoSource.read()` returns `None` for **both** the
end of a file and a single failed read, and `Pipeline.run()` iterated a live
camera exactly the way it iterated a file. One transient read failure ended the
run, and the log said `analysis finished` — the same line a file prints when it
is done. A security camera silently stopped watching and nothing reported a
fault.

`LiveStream` — reconnect with bounded backoff, drop counting, latest-wins queue
— **already existed, was marked `TESTED` in FEATURES.md, and was imported by
nothing.** The pipeline now uses it for live sources; files still iterate
directly, which is what keeps replay deterministic.

Also fixed alongside it:
- `LiveStream._state.error` was set on failure and never cleared on a successful
  reconnect, so a recovered stream would keep raising the outage it had already
  survived.
- `LiveStream.stop()` released the capture even when the join timed out — freeing
  a `cv2.VideoCapture` under a decode thread still reading through it. It now
  leaks the capture instead and says so.
- `VideoSource.open/close/read` had a check-then-act race now that a source has
  two owners.

### Tests and docs

- `engine/tests/test_segment.py` — 13 tests against the **real** model, plus an
  opt-in live-camera silhouette test (`SENTINEL_TEST_CAMERA=1`), which passes.
- `engine/tests/test_pipeline.py` — two regression tests pinning the live-camera
  behaviour above.
- Two pre-existing red recording tests fixed: they killed a writer with a bogus
  fourcc, and **OpenCV 5 silently substitutes a codec** (`tag 'ZZZZ' is not
  found ... fallback to use tag 'mp4v'`), so they stopped killing anything. The
  failure is now injected at the seam.
- `FEATURES.md` rescored to **146 TESTED / 23 IMPL / 35 SKEL / 167 PLAN** of 371.
- `docs/USAGE.md` gained **§7 Detection models**; later sections renumbered.
- `README.md` and `STATUS.md` updated, including an honest note that the
  reconnect rows were `TESTED` while the capability was absent from the product.

---

## 3. OPEN DEFECT — read this before doing anything else

**The console test process crashes at exit with `0xC0000374`
(STATUS_HEAP_CORRUPTION) under `QT_QPA_PLATFORM=offscreen`. Every one of the 51
tests passes; the process then dies during interpreter shutdown, so `pytest`
never prints its summary and the stage fails.** This makes
`python tasks.py ci --package` fail at `console · tests`.

Reproduce:

```bash
cd apps/console
QT_QPA_PLATFORM=offscreen PYTHONPATH="../../engine:." python -m pytest -q
echo $?     # 127 in bash; -1073740940 (0xC0000374) via PowerShell $LASTEXITCODE
```

What is established. **Treat any single run as noise** — the crash is
nondeterministic (~2 in 3 early on, deterministic later), and several of my own
bisections were invalidated by exactly that. Use 4+ runs per hypothesis.

Bisection results, each with everything else left modified:

| Reverted | Result |
|---|---|
| nothing (pristine `HEAD`) | **clean 4/4** — so this session caused it |
| `pipeline.py` + `node.py` together | **clean 3/3** ← strongest lead |
| `app.py` + `video_view.py` together | **clean 3/3** ← equally strong |
| `decode.py` alone | crash 3/3 |
| `node.py` alone | crash 3/3 |
| `app.py` alone | crash 4/4 |
| `pipeline.py` alone | crash 3/3 (also mass test failures — `node` calls the method it removes, so this run is not a clean isolation) |
| `video_view.py` alone | 1 clean, 1 crash |

Read together: **neither side alone is sufficient and both are necessary** —
reverting either pair clears it, reverting any single file does not. That points
at an interaction, not a single bad line.

Ruled out:
- **No decode threads leak** — verified with a per-test teardown hook that
  enumerates threads named `decode:`.
- Console tests use **file** sources only, so `LiveStream` is never constructed
  by them and `Pipeline._run_live` never executes. The live-camera path is very
  unlikely to be the trigger, which makes the `pipeline.py` + `node.py` result
  the interesting one: for a file, the only live changes are
  `Pipeline._stopping` / `ask_to_stop` and `CameraRunner.ask_to_stop` calling
  into the pipeline **from the Qt thread** while the camera thread is inside
  `pipeline.run()`.
- `_mask_image` over a live numpy buffer. Changed to `buffer.tobytes()` and
  re-run **6×: still crashed 6/6.** The change is correct and worth keeping, but
  it is not the cause. (`_mask_image` is in fact never called by these tests —
  MotionDetector produces no masks.)

Also attempted, correct, and worth keeping, but not the cause:
- a lock around `VideoSource._capture` for `open` / `close` / `read`;
- leaking rather than releasing a capture when the decode thread will not join.

Suggested next moves, cheapest first:
1. Revert **only** `CameraRunner.ask_to_stop`'s new call into the pipeline
   (leave `Pipeline.ask_to_stop` in place) and run 4×. That isolates the
   cross-thread call, which the table above implicates most directly.
2. Bisect *within* `video_view.py` — the `numpy` import, `_mask_image`, and the
   `placed` label-collision list are independent and removable separately.
3. Run the suite under `python -X dev -X faulthandler` or with
   `PYTHONMALLOC=debug` to move the abort nearer the bad free; also try
   `-p no:randomly` to see whether test order matters.
4. Bisect by test: split with `-k` into halves and find the minimal set that
   still crashes at exit. Slow (~75 s a run) but conclusive.

**Do not ship or commit as "green" until this is resolved.** The engine suite is
fully green (one skip, the opt-in camera test); only the console stage fails,
and it fails *after* passing every test.

---

## 4. Current repository state

Branch `Phase2`. **Nothing from this session is committed yet.** Modified:

```
apps/console/sentinel_console/app.py          model wiring, honest detector label
apps/console/sentinel_console/video_view.py   mask drawing, label collision
engine/sentinel/decode.py                     LiveStream hardening, capture races
engine/sentinel/detect.py                     detector_for(), _output_count()
engine/sentinel/cli.py                        --model on run and node
engine/sentinel/paths.py                      models_directory, default_model_path
engine/sentinel/pipeline.py                   LiveStream for live sources
engine/sentinel/node.py                       ask_to_stop reaches the pipeline
engine/sentinel/segment.py                    NEW — Segmenter, ground_contact
engine/tests/test_segment.py                  NEW
engine/tests/test_pipeline.py                 live-camera regressions
engine/tests/test_recording.py                injected writer failure
engine/tests/test_cli.py                      patch detect.MotionDetector, not cli
tools/screenshot_console.py                   --segment
FEATURES.md STATUS.md README.md docs/USAGE.md
```

Suggested commits once the console stage is green — small and separable:

1. `fix: a camera that drops one frame no longer stops for good`
   (decode, pipeline, node, test_pipeline)
2. `feat: the system can see shapes, not just boxes`
   (segment, detect, cli, paths, test_segment)
3. `feat: the console runs a model, and says which one`
   (app, video_view, screenshot_console)
4. `fix: two recording tests stopped testing anything`
   (test_recording, test_cli)
5. `docs: segmentation, and a capability that was never reachable`

---

## 5. What is next after that

**Immediate, in order:**

1. Resolve the console crash (§3), get `tasks.py ci --package` green, commit.
2. Rebuild the three executables and confirm each launches
   (`tools/local_ci.py` already has launch checks).
3. Re-capture screenshots with `--live --segment` and look at them.

### The executables as built on 2026-09-03 16:17

`python tasks.py package` succeeded and all three are in `dist/SentinelVision/`:

| Executable | Size | What it is |
|---|---:|---|
| `SentinelVision.exe` | 45.1 MB | the console, windowed |
| `SentinelVision-dev.exe` | 45.1 MB | the console with developer logging |
| `sentinel.exe` | 45.1 MB | the headless CLI |

`sentinel.exe run --help` confirms `--model FILE` is wired in the packaged
build. These were built **while the console test stage was failing at exit**
(§3) — the tests themselves all pass, so the binaries are usable for human
testing, but they are not a green build and should not be released as one.

To exercise segmentation in the packaged console, put a `*-seg.onnx` where it
can find it — `SENTINEL_MODELS_DIR`, or `models/` beside the install — or pass
`--model`. With no model it runs motion detection and says so in the toolbar.

**Stage 2 of segmentation (designed, not started):** carry the mask-derived
ground contact **through the Rust FFI**. `CDetection` needs `contact_x` /
`contact_y`, which bumps `ABI_VERSION` 5 → 6 on both sides plus the struct-size
guard. Today the mask improves the contact point in Python and the Rust
projection still receives a box.

**Then, from ROADMAP:**

- **1.3 appearance re-ID.** The tracker fragments — 17 tracks over 15 s on one
  webcam, because association is geometric only. Masks now make appearance
  embeddings possible, and this is the single biggest quality win available.
- **2.1 control plane.** A survey chose `aiohttp` over FastAPI: uvicorn and
  Hypercorn do not implement the ASGI TLS extension, so an ASGI stack cannot
  learn which node is on an mTLS connection — and FastAPI's `/docs` fetches
  Swagger UI from a CDN, which this product must never do.
- Console: a recording toggle (recording is still CLI-only), and a detector
  picker in the UI rather than only a flag.

---

## 6. Hard-won facts worth not rediscovering

- **The recurring defect in this repository is correct, tested code that nothing
  calls.** Found four times: the recorder, `RecorderStats.fault`, `LiveStream`,
  and the segmenter before it was wired. Unit tests instantiate a class
  directly, so they pass whether or not the product ever constructs it. Before
  marking anything `TESTED`, grep for importers outside its own module and
  tests.
- **Verify the premise, not just the result.** `strings` does not exist on this
  machine — a binary scan silently returned 0 matches for everything and "the
  binaries are clean" was worthless; redone in Python it found live Microsoft
  telemetry endpoints in `libonnxruntime.so`. OpenCV 5 substitutes codecs.
  A test hooked to `cli.MotionDetector` counted nothing after the call moved.
  Offscreen Qt has **0 font families** vs 290 native and renders all text as
  tofu — nearly reported as a product defect.
- **onnxruntime ships a Microsoft 1DS/OneCollector telemetry uploader** in the
  manylinux and macOS wheels, on by default. It is disarmed in
  `engine/sentinel/telemetry.py`, called at every entry point, and
  `tools/binary_audit.py` exists because the source audit can never see a
  hostname inside a 28 MB `.so`.
- **Only the thread that owns the node touches the store.** SQLite connections
  belong to their creating thread; this was violated three times. Everything
  else queues.
- The Rust core is **C ABI + ctypes, not PyO3** — a GNU-toolchain cdylib against
  an MSVC CPython is a real hazard.
- Heredocs mangle `\n`; write patch scripts with the Write tool instead.
