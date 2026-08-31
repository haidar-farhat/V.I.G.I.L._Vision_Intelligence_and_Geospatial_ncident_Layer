# Testing

Three suites, one command:

```bash
python tasks.py test     # Rust, engine, console — 140 tests
python tasks.py check    # lint first, then all of the above (what CI runs)
```

Or individually:

```bash
cd core         && cargo test
cd engine       && python -m pytest
cd apps/console && python -m pytest        # QT_QPA_PLATFORM=offscreen
```

No network is used at any point. The CI `offline` job proves this rather than
assuming it: it drops all outbound traffic, **verifies the drop actually took
effect** — a firewall rule that silently failed would let the job pass while
testing nothing — and then runs every suite.

---

## What is tested where

| Suite | Count | Covers |
|---|---|---|
| `core` (Rust) | 32 | geodesy, projection, field of view, polygons, tracking, the C ABI |
| `engine` (Python) | 89 | the boundary, decode, detection, and the full pipeline on real video |
| `apps/console` | 19 | placement, overlay honesty, threading, redaction, fault handling |

The mathematics is tested in Rust, where it lives. The Python tests over the same
area deliberately do **not** re-test the mathematics; they test what only a caller
can break — struct layouts, ownership, buffer limits, and whether a value that is
correct in Rust is still correct after it crosses. Read them as "does the boundary
lie?" rather than "is the maths right?".

---

## Principles

### A test names the failure, not the behaviour

`test_a_ray_above_the_horizon_yields_nothing_rather_than_a_guess` says what goes
wrong if it breaks. `test_project_to_ground` does not. When one of these fails at
three in the morning, the name is the first thing anybody reads.

### Assert the imperfect truth, not the intention

Where the system currently does something badly, the test asserts the bad thing
and names it. The pipeline reports 5 distinct objects where 3 people walked past;
the test bounds that at 6 and explains why the number is what it is. A test that
asserted 3 would be marked skip within a week, and a skipped test protects
nothing.

### Numbers in tests are measured, then floored

Every threshold — recall, overlap, throughput, identity switches — was measured
first and the assertion set below it with room for machine-to-machine variation.
The measured value is written in a comment beside the assertion, so a later
reader can tell a genuine regression from noise. A threshold chosen by intuition
either fails constantly or never fails.

### Test the property, not an arbitrary sample of it

`test_uncertainty_grows_with_distance_from_the_camera` asserts a correlation and
a super-linear ratio across all samples, not that uncertainty at 15 m exceeds
uncertainty at 25 m. The bucketed version passed until the scene changed and
every sample landed in one bucket, at which point it silently proved nothing.

### Never assert on something your own fixture put there

A check for an error string matches the error you injected as readily as the one
you were looking for, and the test passes while proving nothing. Assert only on
values the system under test produced.

### A hanging test is a finding

`test_a_failed_source_reports_in_place_rather_than_in_a_modal` would hang forever
if the console opened a modal dialog — which is exactly what an operator would
experience. It found that defect by hanging.

---

## Determinism

Replaying evidence must reproduce it, or an incident review shows something other
than what the operator saw. Three tests hold this:

- decoding the same file twice yields byte-identical frames;
- the tracker produces identical output from identical input;
- the whole pipeline over the same video produces identical track ids and boxes.

Anything that would break these — wall-clock time in the analysis path, iteration
over an unordered set, an unseeded random number — is a defect regardless of
whether it changes any current result.

---

## The reference scene

`engine/tests/scene.py` generates a video and encodes it to a real file, once per
test session. It is not committed: it is fully determined by that module, and a
binary in version control that can be regenerated exactly is a binary that will
eventually disagree with the code that generates it.

**Be clear about what it establishes.** The *file* is real — a genuine container
written by a real encoder and read by a real decoder, so the decode path under
test is the one a camera exercises. The *scene* is generated geometry. It proves
the pipeline carries frames, detections, tracks and positions end to end without
lying about them. It proves nothing about real footage.

It is built to be honest work for the stages below it rather than easy on them:

- perspective, so objects further away are smaller and move less per frame;
- sensor noise and an illumination drift, so a background model has something to
  cope with;
- an object that walks behind an occluder, which decides whether a tracker keeps
  an identity or invents one;
- two objects that cross, which is where naive association swaps them;
- an object that stops moving, which is the loitering case and the one background
  subtraction cannot see.

The walkers are drawn with internal structure — head, torso, swinging legs — and
that is not decoration. A uniformly shaded rectangle is an unrealistically *hard*
case: a background model absorbs the unchanging interior of a slow solid block
and leaves only its leading edge. Testing against a shape real objects do not
have would mean tuning the detector for a problem nobody has.

`scene.ground_truth(index)` returns what is actually where in each frame, which
makes it the specification the measurements are taken against.

---

## Credentials in tests

No camera password may appear in test output. The decode tests use a fixed
sentinel string and assert it is unreachable through `repr`, `str`, the display
URL, the source id, and every error message — including its *length*, since a
redaction that prints one asterisk per character leaks that.

Unreachable network addresses in tests use `203.0.113.0/24` (TEST-NET-3), which is
reserved for documentation and routed nowhere. A test that reaches a real host is
a test that behaves differently on someone else's network.
