# Development

## Requirements

- **Python 3.12+** (3.14 is what this is developed on)
- **Rust**, stable toolchain, with `rustfmt` and `clippy`

```bash
rustup component add rustfmt clippy
python -m pip install -e "engine[dev]" PySide6
```

Nothing else, and nothing is fetched at runtime ever. On a machine without the
MSVC C++ build tools, install the GNU toolchain — `rustup default
stable-x86_64-pc-windows-gnu` — which bundles its own linker. The C ABI boundary
(see below) is what makes that mismatch harmless.

## Commands

Identical on Windows, Linux and macOS. Everything routes through Python, so
there is one set of instructions and no shell-script pair to drift apart.

| Command | Does |
|---|---|
| `python tasks.py build` | Build the Rust engine core |
| `python tasks.py test` | Rust, engine and console suites |
| `python tasks.py lint` | `cargo fmt --check` and `clippy -D warnings` |
| `python tasks.py check` | Lint, build, test — what CI runs |
| `python tasks.py console` | Run the operator console |

`tasks.py test` builds the core before running the Python suites. Testing Python
against a stale shared library is how a green run hides a broken change.

## Layout and layering

```
core/          Rust: the hot path. No dependencies.
engine/        Python: decode, detect, pipeline, orchestration.
apps/console/  PySide6: the operator console.
```

Dependencies point **downward and never back up**. `core` knows nothing about
Python. `engine` knows nothing about Qt. The console imports the engine; the
engine never imports the console — which is what makes the engine usable as a
headless worker.

### What goes in Rust

**Rate, not importance.** Anything called per detection, per frame, per camera
goes below the boundary: projection, field-of-view geometry, polygon tests,
tracking. Anything called per event, per second, or per operator action stays
above it, in Python, where the libraries that matter live and where the code
changes most often.

`core` has **no dependencies at all**. For a security appliance the dependency
list is part of the attack surface, and everything in it is arithmetic.

## The C ABI boundary

The struct layouts in `engine/sentinel/core.py` mirror `core/src/ffi.rs` field for
field. That duplication is the price of a C boundary, and it is guarded rather
than trusted.

**If you change any `#[repr(C)]` struct, or any exported signature:**

1. Update the mirror in `engine/sentinel/core.py`.
2. Bump `ABI_VERSION` in **both** files.
3. `python tasks.py build`, then run the tests.

The core exports the size of every struct that crosses, and the binding compares
each against its own declaration at load and refuses a mismatch. This matters
more than it sounds: a drifted layout does not crash. It reads the wrong bytes
and produces geometry that looks entirely reasonable.

Three further rules hold the boundary, and a new entry point must follow all
three:

- **Check every pointer** before dereferencing it. A caller's bug must produce a
  defined failure, not a segfault inside a security appliance.
- **Mark it `unsafe` and write a `# Safety` section.** Null-checking cannot
  establish that a non-null pointer is live, and a function implying otherwise is
  lying to its Rust callers. `clippy` enforces both.
- **Never allocate for the caller** except through a paired create/destroy, so
  ownership is never ambiguous.

Panics cannot cross: the crate is built `panic = "abort"`, because unwinding into
C is undefined behaviour.

## Working on the pipeline

The reference scene (`engine/tests/scene.py`) generates a real encoded video with
known ground truth. Use it to measure a change rather than to confirm one:

```python
import sys; sys.path[:0] = ["engine", "engine/tests"]
from pathlib import Path
import scene
from sentinel.core import CameraPose, LatLon
from sentinel.decode import VideoSource
from sentinel.detect import MotionDetector
from sentinel.pipeline import Pipeline

scene.write_scene(Path("scratch/scene.mp4"))
pose = CameraPose(LatLon(33.8938, 35.5018), 6.0, 180.0, -22.0,
                  horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0)

with Pipeline(VideoSource(Path("scratch/scene.mp4")), MotionDetector(), pose=pose) as p:
    for result in p.run():
        pass
    print(p.stats.summary())
```

Compare against `scene.ground_truth(index)`, which is the specification of what is
actually there. Then put the number you measured in STATUS.md and a floor under it
in a test.

**Do not tune against the reference scene alone.** It is synthetic, and a
parameter fitted to it is fitted to generated geometry. Two of the changes that
mattered most were found this way and are worth knowing about because both are
counter-intuitive:

- the closing kernel is **tall and narrow**, because objects of interest are
  upright and a background model splits them along their length;
- the association gate is an **ellipse, not a circle**, because a camera looking
  at the ground maps vertical image motion to *depth*.

## Adding a feature

1. Decide which side of the boundary it belongs on — by rate, not by importance.
2. Put the logic in the innermost layer that can hold it. Pure geometry belongs in
   `core`, not in a detector.
3. Write the test with the feature, not after it. Name it after the failure.
4. Measure anything you claim. A number that was not measured on this machine is
   marketing.
5. Update `ARCHITECTURE.md` in the same commit as any architectural change.
6. Update `STATUS.md`. A capability is not `TESTED` because a UI for it exists,
   and it is not `IMPLEMENTED` because the code path is written — the ONNX
   detector is fully written and has never had weights loaded into it, so it is
   `IMPLEMENTED`, not `TESTED`, and the distinction is the point.

## Things this codebase will not do

- Reach the Internet, for anything, ever. No tiles, no models, no telemetry, no
  update check.
- Report a position it cannot determine. An unprojectable ray returns nothing; an
  unplaced camera reports "not placed" rather than a nominal origin.
- Report a position without its uncertainty. They travel together or not at all.
- Label a detection with a class the detector cannot produce.
- Present inference as observation. A track held open with no detection behind it
  is drawn differently from one being watched.

## Commits

Conventional commits, scoped to the component:

```
feat(core): ...
feat(engine): ...
fix(console): ...
docs: ...
build: ...
security: ...
```

Explain *why* in the body, especially for a non-obvious trade-off, and give the
measurement behind any threshold. The commit log is the only place a future
reader will find the reasoning behind a number, a weighting or a rejected
alternative.
