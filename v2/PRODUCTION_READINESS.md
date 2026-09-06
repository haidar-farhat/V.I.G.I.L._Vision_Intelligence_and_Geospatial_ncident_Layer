# Production readiness — v2, after the v1 audit

What was wrong, what was done about it, what it measures, and what still
stops this being deployable. Written to be argued with: every claim here is
either a number from `python tasks.py bench`, a number from a run against a
real camera, or a test that fails when the claim stops being true.

**The short version.** The perception, geometry and mapping half of this
system is now sound and measured. The half that decides whether it is
*useful* — the detection model and the data behind it — is untouched, because
there is no dataset and no labelling pipeline in this repository. Section 6 is
the reason this document does not say "production ready".

Section 8 is what to do about it, and it corrects one thing section 6 gets
wrong: "nothing can be measured without labels" is false. Four of this
system's constants have ground truth that comes from the structure of the
problem rather than from a person, `tools/calibrate.py` reads them off
unlabelled video, and the first run of it **contradicted two shipped
constants**. Both findings are in section 8; neither has been acted on yet,
because one twenty-second clip is not grounds for moving a core constant.

---

## 1. What the audit found

### Defects in the mathematics, both versions

| | Was | Cost | Now |
|---|---|---|---|
| **Camera model** | `yaw = atan(dx·tan(hfov/2))`, `elev = pitch + atan(dy·tan(vfov/2))` — documented as "rectilinear" | It is a *cylindrical* sensor. At the reference pose (4 m mast, 25° down, 62×36°) the bottom frame corner was **4.8° wrong in elevation = 21% wrong in distance**, about 1.5 m | One orthonormal basis, one ray, one plane intersection. Forward and inverse are exact inverses by construction |
| **`roll`** | A field on `CameraPose` since v1 ABI 1, read by nothing | A camera clamped 10° off level put a frame corner **0.98 m** from where it was | In the rotation. Tested, and `vigil coverage` moves with it |
| **Two Earths** | Track distances used a spherical haversine; zones and grids used a WGS84 tangent plane | The two disagree by **0.248%** — 25 cm per 100 m. A zone edge and a track measured from the same camera were on different planets | One tangent plane. `distance_meters`, `bearing_degrees` and `destination_point` are exact inverses of each other |
| **Uncertainty** | One radius from the contact point's angular error alone | A pose an operator clicked onto a map was treated as exact. The reported error was the smallest term in it | A Jacobian over heading, pitch, roll, mount height, contact point and terrain slope, reported as an **ellipse** — a shallow ray is vague along its own direction and sharp across it |
| **Process noise** | (introduced during this work, then fixed) `q = (3h)²`, conflating a peak acceleration with a spectral density | The filter was 4× more willing to believe in acceleration than acceleration exists, and chased detector jitter | `q = 0.18·h²`, derived as `(a/H)²·τ`, with the wrong derivation kept in the docstring |

### Defects in the tracker, both versions

- **Greedy association.** Sort pairs by score, take each in turn. Not optimal;
  the failure mode is two people crossing. Measured below.
- **EMA velocity.** No uncertainty to gate on, no way to tell detector jitter
  from motion, and a time constant measured in *frames* — so the smoothing
  changed whenever the camera stuttered, which is exactly when tracking is
  hard.
- **No appearance.** v1 measured the consequence and wrote it down: twenty
  seconds of one person counted **3, 10, 4 and 11 "distinct objects"** across
  four runs. v1 wrote `reid.py` to reconcile the count *after the fact* and
  said in its own docstring that it could not un-split a track mid-life. **v2
  deleted `reid.py` and kept the tracker**, so v2 had the fragmentation and
  none of the mitigation.
- **Low-confidence detections discarded.** A detector that is 70% sure drops
  to 30% behind a post, and the object is still exactly where the track
  predicts.
- **No camera-motion estimate.** A gust on a mast translates every box at
  once and the tracker read it as everything accelerating together.

### Defects in the detector

- **Class-agnostic NMS.** One suppression pass over every box in the frame, so
  a person standing in front of a car with 70% overlap deleted whichever
  scored lower. On a security camera that is not an edge case, it is a car
  park. A suppressed detection leaves no trace anywhere.
- **Execution provider pinned to the CPU.** `providers=["CPUExecutionProvider"]`
  hard-coded, so a machine with `onnxruntime-gpu` ran on the CPU and nothing
  said so.
- **Mask discarded.** The segmentation mask was computed, used to find the
  ground contact, and thrown away.

### Functionality v2 deleted from v1

| v1 module | Lines | In v2 before | Now |
|---|---|---|---|
| `orthophoto.py` + `basemap.py` | 1,974 | gone | `service/mapping.py` + `core/src/ortho.rs`, with a confidence layer v1 did not have |
| `coverage.py` | 577 | gone | `service/coverage.py`, on the corrected footprint |
| `reid.py` | 628 | gone | `domain/appearance.py` + `perception/appearance.py`, used **during** association rather than after it |
| the Rust core | ~3,800 | gone (reimplemented in Python) | `core/`, rebuilt around the corrected model |

### Gaps in neither version

- Nothing measured whether a **frame was worth looking at**. v1 and v2 both
  had `camera.dark` — no frames at all — and nothing between that and
  "working". An unfocused lens, a blown-out frame and a decoder repeating its
  last frame all failed silently while the frame counter climbed.

---

## 2. What was built

```
vigil/
  kernel/       native.py, filtering.py        the arithmetic, below the domain
  domain/       geo, tracking, appearance,     pure logic, no I/O, no OpenCV
                detection, zones, events, …
  perception/   motion, quality, appearance    pixels → measurements
  adapters/     decode, detectors, recorder    the outside world
  service/      runtime, mapping, coverage, …  policy
  interfaces/   cli, console, map_commands     what a person touches
core/src/       camera, geodesy, assign,       C ABI, no dependencies
                track, ortho, ffi
```

`kernel` is **below** the domain, not beside it, so the domain can say
"predict this track" without knowing a shared library exists. `perception` is
beside the adapters: it reads pixels, so it may use OpenCV, and it produces
domain types, so it may import the domain — but nothing in the domain may
import it back. `tests/test_layering.py` enforces the whole direction.

### The detection pipeline now

```
decode → frame quality → camera motion → detect → describe → track → project
           ↓ refuse         ↓ warp                   ↓ mask     ↓ 4 passes
        alert if sustained  incl. covariance                    Kalman + optimal assignment
```

Four association passes, in order:

1. **Confirmed tracks × strong detections** — cost blends appearance and
   overlap, gated by the filter's own Mahalanobis distance.
2. **Confirmed tracks × weak detections** — IoU only, position-gated. This is
   what recovers an object through an occlusion.
3. **Tentative tracks × what is left** — geometry alone and strictly.
4. **Lost tracks × what is still left** — appearance-dominant with a spatial
   gate that widens with the time gone. This is what prevents a fragment,
   rather than reconciling one afterwards.

### Why Rust, and where

By measurement, not taste. Three things:

| | Why | Measured |
|---|---|---|
| `ortho.rs` | 1,089 lattice projections and up to 230,000 cells **per frame per camera**. v1 did it in NumPy at **79 ms** and sampled 4 fps because of it | **0.62 ms** for 230,400 cells; 1.0 ms for the 21,840-cell grid a 40 m camera needs |
| `assign.rs` | Optimal association is O(n³), SciPy is not a dependency of an offline appliance, and a Python triple loop at 15 fps × 16 cameras is not one either | **0.010 ms** for 32×32 |
| `track.rs` | An 8-state predict and update per track per frame; NumPy pays more in call overhead on an 8×8 than the arithmetic costs | 0.057 ms for a whole track update including association |

Everything else stayed in Python, including per-detection geometry, because a
projection costs 36 µs and one readable implementation beats saving them.

**Two implementations are safe exactly as long as something proves they
agree.** `tests/test_native.py` drives the Rust and NumPy paths over a grid of
poses and holds them to 1e-9 for projection and inversion, to identical
matchings for assignment, and — for the ground rasteriser, which has no NumPy
mirror — against the *independent* exact inverse in `domain/geo.py`, which is
a stronger check than a mirror would be.

---

## 3. What it measures

`python tasks.py bench`. **The machine was at 33–40% background load**, so
every cost below is pessimistic and none of it is a product specification.

### Cost, median ms per call

| Stage | ms | Note |
|---|---|---|
| frame quality (1280×720) | 1.47 | |
| camera motion (1280×720) | 2.87–6.96 | the most expensive new stage |
| appearance, per detection | 0.16 | |
| track update, 1 track | 0.06 | |
| track update, 30 tracks | 0.61 | |
| assignment, 32×32 | 0.010 | Rust |
| ground projection, one point | 0.036 | Python |
| ground sample, 230,400 cells | 0.62 | Rust; v1's NumPy was 79 ms at 118k cells |
| **onnx detect+segment, 640 px** | **83.6** | CPU-only runtime — **this is the bottleneck** |

Perception and tracking together cost **~8 ms per frame**, about 10% of what
detection costs on this machine. On a real camera end to end: **14.1 fps**,
with the three new stages costing 3.5 ms of the 68 ms per frame.

### Quality, against known ground truth

| Scenario | v1/v2 design | Shipped | |
|---|---|---|---|
| One person, blinking detector, 20 s. Truth: **1 object** | **5 ids** | **1 id** | appearance during association |
| Four people milling in a small space, 8 runs — identity switches | **118** | **89** | **−25%**; optimal assignment gets it from 118→95, appearance and occlusion-awareness 95→89 |
| Two people crossing head-on, 12 runs | 0 swaps | 0 swaps | **a null result, reported as one** |

That last row matters. A head-on crossing does **not** discriminate between
greedy and optimal association, because the Kalman filter's velocity estimate
keeps the two predictions separated through it. That is a measurement of the
*filter*, not the association, and the benchmark says so rather than claiming
a win it did not find. The assignment's benefit is proven separately by a unit
test on the cost matrix (`core/src/assign.rs`) and by the milling scenario.

### Three findings the benchmark produced, and what they changed

1. **Appearance initially made association *worse*** — 105 identity switches
   against 95 without. Two causes, both real bugs, both now fixed and both
   held by a regression test:
   - The cost matrix mixed scales. A pair scored with appearance got
     `0.5·app + 0.5·(1−IoU)`; a pair with no usable appearance got the bare
     `1−IoU`. A good appearance match therefore scored *half* what the same
     geometry scored for a pair whose appearance was unknown, so the solver
     systematically preferred whichever candidate was not occluded — which is
     the wrong one. There is now a `NEUTRAL_APPEARANCE` value so every cell is
     on one scale.
   - A single appearance gate at 0.45 was doing two incompatible jobs. Inside
     a frame it **vetoed correct pairs** whose crop was momentarily
     contaminated; across a gap it is the only evidence there is. Split into
     `MAX_APPEARANCE_DISTANCE = 0.70` (association: appearance shades,
     geometry decides) and `MAX_REIDENTIFY_DISTANCE = 0.35` (re-identification:
     appearance decides). Sweeping the parameter is what found this.
2. **A crop of a partly occluded person is a crop of two people**, and storing
   it in a gallery goes on matching the wrong one for seconds afterwards.
   Which of two overlapping boxes is in front cannot be read off the boxes —
   but it can be read off the ground: for objects on a plane, the lower box
   bottom is the nearer one. Discarding the occluded one's descriptor was
   worth another 4 switches.
3. **The synthetic yard fixture failed the frame-quality gate twice**, and both
   times the gate was right: a frame uniform to within 8 levels has nothing in
   it, and one upscaled 10× from a coarse grid measures a Laplacian variance
   of 18 against a floor of 40 — it *is* out of focus.

---

## 4. Tests

| | Count | |
|---|---|---|
| Python | 298 | `python tasks.py test` |
| Rust | 53 | `cargo test --lib --release` |
| **Total** | **351** | `python tasks.py check` runs both plus the offline audit |

New suites: `test_geo.py` (rewritten around known-answer geometry),
`test_tracking.py` (rewritten; every test names a way the old tracker was
wrong), `test_native.py`, `test_perception.py`, `test_mapping.py`,
`test_coverage.py`.

Geometry is tested against closed forms wherever one exists — the centre
column of a frame, the height column of the Jacobian, a level camera — and
against invariants where one does not: orthonormality of the basis for every
orientation, exact invertibility of the projection under roll, monotonicity of
error with range.

---

## 5. Reliability

| Failure | Before | Now |
|---|---|---|
| Out-of-focus lens | silent | measured, alerted as `camera.degraded` after 150 frames |
| Blown out / crushed | silent | measured, named |
| Frozen decoder | silent; frame counter kept climbing | measured; separated from image faults, because a repeated frame is still worth detecting on and is *not* worth sampling into a map |
| Camera knocked | silent | measured; the warp is applied to every filter, covariance included |
| A moving lorry mistaken for a moving camera | n/a | refused: a fit explaining a minority of the frame is not the camera |
| A cut or PTZ slew | n/a | refused above 25% of a frame — warping tracks by half a frame is worse than admitting the scene is lost |
| Ray at the horizon / out of range / bad pose | all collapsed to `None` | four named reasons; `TOO_SHALLOW` is a pose problem and `OUT_OF_RANGE` is a siting problem |
| Degenerate Kalman update | n/a | refused; the track's identity survives and its numbers are restarted |
| Map file edited or truncated | n/a | refused — it no longer hashes to its own fingerprint |
| Engine core missing or wrong ABI | n/a | refused on load; `vigil doctor` says so; `vigil map` refuses rather than running 80× slower in silence |

---

## 6. Remaining limitations

**These are why this document does not say "production ready".**

### Blocking

1. **There is no data pipeline, no dataset, and no evaluation.** The model is
   an ONNX file the operator supplies; the one in this repository is stock
   `yolov8n-seg` COCO weights that have never seen this site, in this light,
   at this mounting height. Nothing here measures precision, recall, or the
   false-positive rate on the site it will run at, because there is nothing
   to measure them against. **This is the single largest gap and no amount of
   pipeline engineering closes it.** What would: a few thousand labelled
   frames from the actual cameras, at the actual hours, split by *day* rather
   than at random — a random split over consecutive frames leaks almost
   perfectly and would produce a meaningless number.
2. **Every appearance threshold is calibrated on synthetic scenes.** Solid
   colour blocks separate at a cosine distance of 0.56. Real clothing under
   real light will not, and the honest expectation is that
   `MAX_REIDENTIFY_DISTANCE` needs raising and `MAX_APPEARANCE_DISTANCE` needs
   watching. Until somebody runs the fragmentation measurement on real video,
   both numbers are provisional.
3. **83 ms per frame of detection on a CPU** is ~12 fps for **one** camera.
   The multi-camera claim in this product is not supported by this machine.
   `onnxruntime-gpu` or `onnxruntime-directml` is the answer and `vigil doctor`
   now says which one you have — but nobody has run this on a GPU.

### Known and bounded

4. **Flat ground plane.** Stated, not hidden: terrain enters as a 2%
   uncertainty term rather than being corrected for. A site on a slope will
   have positions biased along the line of sight, and the reported error will
   cover it only if the slope is under about 2%.
5. **No lens distortion model.** Intrinsics come from a datasheet FOV. On a
   90° security lens, radial distortion moves a corner by several percent, and
   nothing here corrects or bounds it. `Intrinsics::from_matrix` exists for a
   calibrated camera; nothing produces a calibration.
6. **Pose uncertainty defaults are assumptions, not measurements.** ±2° of
   heading and ±0.15 m of height are what an operator with a compass and a
   tape is worth. They are stated in `PoseUncertainty` and they dominate the
   error at range.
7. **The ground map is only right on the ground.** A wall, a van or a person
   smears radially. The median removes what moves and the confidence layer
   marks what it cannot vouch for — 41% of the map on the real camera run —
   but a static wall is static, and it will be drawn as ground with high
   confidence. That is the failure mode to watch for, and there is currently
   no test for it because it needs a real scene with a known wall.
8. **The error ellipse is rotated into (along, across) the line of sight**,
   which is within about a percent of the true eigenvectors for a mast camera
   but is not an eigendecomposition.
9. **`uncertainty_over` interpolates by range only**, ignoring the ~15%
   variation across a wide frame.
10. **Cross-camera identity still rests on time and place alone.** The
    appearance descriptor is per-camera; nothing yet carries it into the
    correlator, so two cameras seeing the same person still link on geometry.
11. **CI has never run.** The workflow is written; the numbers in this
    document come from one developer machine under load.
12. **No installer, no code signing**, and the packaged bundle's engine core
    has not been tested from a frozen build — only from a checkout.

### Not attempted

13. **Depth estimation, structure from motion, loop closure.** The camera
    motion estimator produces a 2-D affine, not a pose. Multi-frame
    reconstruction is a median over a *known* pose, not a solved one. A camera
    that drifts is detected, not corrected, and there is no bundle adjustment
    anywhere in this system. That is a deliberate limit: with one fixed camera
    per view and no baseline, there is no triangulation to do, and claiming
    otherwise would be the kind of thing this rewrite removed.
14. **Faces, plates, a subject register** — DECISIONS.md D-08.

---

## 7. Risks

- **The appearance thresholds are the most likely thing to be wrong in the
  field.** They are the only constants in this work calibrated purely on
  synthetic data, and getting them wrong in the permissive direction produces
  a *merge* — two people reported as one — which is worse than a fragment
  because a fragment is visible on screen and a merge is not.
- **The occlusion depth-ordering heuristic assumes objects stand on the
  ground.** A person on a balcony, an object on a shelf, or a camera looking
  down a stairwell breaks it. It fails safe — it discards a descriptor it
  should have kept — but it will discard more than it should indoors.
- **`camera.degraded` will false-positive on a genuinely featureless scene**:
  a camera watching a plain wall at night has low contrast because there is
  nothing there, not because it is broken. The 150-frame hold reduces this;
  it does not remove it.
- **`vigil map` is memory-hungry**: 10 MB per camera at the default cell and
  a 120 m footprint, and it is allocated up front. Sixteen cameras at once is
  160 MB before any frames arrive.

---

## 8. What to do about the blocking items

Recommendations, in the order they pay off. Every number here was measured on
this machine on 2026-09-06 with `tools/detector_options.py` and
`tools/calibrate.py`; both ship with the product, so the measurements can be
repeated on the site's own footage — which is the only place they mean
anything.

### A. Data, labelling and evaluation

**A0. Measure what needs no labels — this week, at no cost.**

Section 6 was too pessimistic, and the correction matters: "nothing measures
precision or recall because there is nothing to measure against" is true of
*precision and recall* and false of nearly everything else. Four constants
this system runs on have ground truth that comes from the **structure of the
problem** rather than from a person, and `tools/calibrate.py` reads all four
off unlabelled video:

- Two detections **in the same frame** are certainly different objects — one
  object cannot be in two places.
- Two appearances of the **same continuously-detected track** are almost
  certainly one object.
- A track whose **position is not moving** is measuring the detector's own box
  jitter and nothing else.
- A confirmed track whose **class flips** is a model guessing.

Run on one 573-frame clip it already found two problems (C and D below). Run
it on a night's footage from each camera before doing anything else.

**A1. Harvest the corpus the product already writes.**

Do not start a labelling project. Start by reading what running the product
has already produced:

- Timestamped MP4 clips with SHA-256 digests, indexed in the store.
- Events carrying camera, zone, track, class, confidence and time.
- **Every dismissal carries a mandatory human-written reason.** `review
  dismiss` refuses without one. That is a free, human-authored label on the
  system's own false positives, it has been accumulating since the queue was
  built, and **nothing reads it**.

An incident-level precision figure is available from data already on disk.
What is missing is a `vigil dataset export` that joins clips to events to
judgements and writes frames plus pre-labels in COCO or YOLO form. A day or
two of work, and it belongs before any labelling.

**A2. Then label the minimum that matters, and label it correctly.**

- **Stratify; do not sample uniformly.** By hour, by weather, by camera, and
  deliberately oversample dawn, dusk, rain and headlights. A thousand frames
  of a sunny afternoon measures a sunny afternoon.
- **Split by day, never at random.** Consecutive frames are near-duplicates; a
  random split leaks almost perfectly and produces a number that means
  nothing. Hold out whole days, and hold out days that look *different*.
- **Correct pre-labels rather than drawing from scratch** — three to five
  times faster.
- **Label only the watch list.** COCO has eighty classes; this site cares
  about six.
- 500–1000 frames per camera is enough for *domain adaptation*. It is not
  enough for a new class and not enough to certify anything.
- Tooling: CVAT or Label Studio. Both install and run offline, which this
  product requires.

**A3. Build the evaluation harness before fine-tuning anything.**

Precision and recall per class, per camera, per hour band, on the held-out
days. Fine-tuning without this is not engineering, it is hoping — and the
harness is worth more than the training run, because it is what tells you
whether the training run helped.

**A4. Fine-tune only if the evaluation says to — and the first evidence says
maybe not.**

On the clip measured here the model was confident (89% of detections above
0.80, strongly bimodal) and no track ever changed class. That is the shape of
a model *inside* its domain, and it is evidence against urgently retraining
for `person` at close range. The real risk is elsewhere, and A3 is what finds
it: small and distant objects, night and IR, weather, and the classes nobody
has exercised at all.

### B. Throughput: 5.4x is available on this machine today, measured

Baseline on a real 640x480 clip: **67.6 ms** per frame (14.8 fps) at the
shipped two-thread setting.

| Option | Measured | Cost | Status |
|---|---|---|---|
| **Threads 2 → 8** | 88.1 → **49.1 ms**, **1.8x** | none for one camera; worse for many, which is why the default is 2 | **available now**: `VIGIL_ORT_THREADS=8` |
| **Detect every 3rd frame, track between** | 67.6 → **22.5 ms**, **3.0x** | position lag **0.003 box heights median, 0.008 p95** — under a centimetre on a person, against a projection error over a metre at range | needs a config knob; **not built** |
| INT8 dynamic quantisation | 67.6 → **70.2 ms**, **0.96x — slower** | model 13.9 → 3.8 MB, 96% agreement | **do not bother** |
| GPU / NPU via DirectML | not measured | one `pip install onnxruntime-directml` | **untested; likely the largest single win** |

Three things worth saying about that table.

**The quantisation result is a useful negative.** Dynamic quantisation helps
matrix-multiply-heavy networks; this one is convolutions, and on this CPU it
came out slightly *slower* while shrinking the file to a quarter. The version
that might work is static QDQ quantisation with a calibration set — which
needs site footage, so it belongs after A1 rather than before it.

**The detection-interval result is the big one, and it is only safe because of
the tracker rewrite.** Skipping detections on the old EMA tracker would have
been reckless. With a Kalman filter that predicts properly, a second
association pass that recovers an object from a weak detection, and
re-identification across a gap, the measured cost of detecting a third as
often is negligible. **Caveat, and it matters:** the clip had one track and
little motion. Re-measure on footage with people walking and vehicles moving
before choosing an interval — the lag scales with speed.

**Combining the two available options** gives roughly 16 ms of detection per
frame: about four cameras at 15 fps on this CPU rather than one at twelve,
without a GPU and without touching the model.

### C. The appearance thresholds — and a result that contradicts them

`tools/calibrate.py` exists to answer this, and the first real measurement
**disagrees with the shipped values**:

```
same object (continuous tracks)     n=8911  p05=0.001  median=0.017  p95=0.068
different objects (same frame)      n=144   p05=0.011  median=0.105  p95=0.262
separation p95(same) → p05(different)       -0.057   ← they OVERLAP
```

The synthetic scenes separated two coats at a cosine distance of 0.56, which
is where `MAX_REIDENTIFY_DISTANCE = 0.35` came from. On real footage the
*different-object* median is **0.105** — so a gate at 0.35 admits essentially
everything, and would merge two objects rather than tell them apart. There is
no value that works here: the distributions overlap, so any threshold both
splits one object and merges two.

**This is the risk named in section 7, now confirmed with evidence.** The
honest reading is that a masked colour histogram may not carry enough
information to re-identify on real footage. What that leaves, in order:

1. **Re-measure on real site footage before concluding.** The clip used is a
   static indoor webcam scene with few objects of similar colour — close to
   the worst case for a colour descriptor, and nothing like a security site.
   Run `tools/calibrate.py` on a night from each camera. First thing to do,
   and it costs nothing.
2. **If it holds, a learned re-identification embedding is the answer** — a
   small OSNet or similar, exported to ONNX. It is a *download*, which this
   product forbids, so it takes the route the detector already takes: the
   operator supplies the file and its digest travels with the evidence. The
   interface in `domain/appearance.py` is already the right shape; only
   `perception/appearance.py` would change.
3. **Until then, prefer the split to the merge.** A fragment is visible — an
   operator sees two boxes on one person. A merge is invisible and reports two
   people as one. If a threshold has to be guessed, guess tight.

### D. And one more thing the calibration found

```
detector box jitter, measured 1-sigma    0.0042 of a box height
MEASURE_STD_FRACTION, assumed            0.05
```

The Kalman filter is being told the detector is **twelve times noisier than it
measurably is**, so it distrusts every measurement and lags real motion. That
constant is documented as "5%, measured on the shipped model", and nothing
held it there.

It has **not been changed**, deliberately: one twenty-second indoor clip of a
nearly-static scene is not grounds for moving a core constant, and jitter on a
distant moving object will be far worse than on a near stationary one. Run
`tools/calibrate.py` across several cameras and several hours first. If it
holds, lowering it is a free improvement in tracking responsiveness.

---

## 9. How to check any of this

```bash
python tasks.py core           # build and test the Rust core (53 tests)
python tasks.py check          # offline audit + 298 Python tests
python tasks.py bench          # the cost and quality numbers above
python tools/camera_check.py   # the whole pipeline against a real camera
python -m vigil doctor         # whether this installation will work
python -m vigil coverage       # what these cameras reach, and what they miss
```
