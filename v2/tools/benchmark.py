"""What this pipeline actually costs, and what it actually recovers.

`python tasks.py bench`.

Two kinds of number, and the distinction matters more than either:

**Cost** — milliseconds per frame for each stage, on this machine, against the
real model where one is present. These are honest only about *this* machine
and only about the frames given, and the harness prints both so nobody quotes
them as a product specification.

**Quality** — how many distinct objects the tracker reports for one person on
a sequence where the truth is known. This is the number v1 measured and wrote
down (3, 10, 4 and 11 across four runs of one person) and it is the reason
most of the tracker was rewritten. The harness runs the same sequence with
appearance on and off, so the improvement is measured against this code rather
than against a remembered figure from another version.

The scenes are synthetic. That bounds what any of this proves: a synthetic
walker has no motion blur, no shadow and no compression, and the fragmentation
figure on real video will be worse. `tasks.py exetest` is the run against a
real camera; this is the run that can go in CI.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.domain.detection import BoundingBox, Detection  # noqa: E402
from vigil.domain.geo import CameraPose, LatLon, project_to_ground  # noqa: E402
from vigil.domain.tracking import Tracker, TrackerConfig  # noqa: E402
from vigil.kernel import native  # noqa: E402
from vigil.perception.appearance import describe  # noqa: E402
from vigil.perception.motion import CameraMotionEstimator  # noqa: E402
from vigil.perception.quality import FrameQualityMonitor  # noqa: E402

SITE = LatLon(33.8938, 35.5018)
POSE = CameraPose(SITE, 4.0, 0.0, -25.0, 0.0, 62.0, 36.0, 60.0)
FPS = 15
STEP = 1000 // FPS


def _timed(label: str, work, runs: int = 30, warm: int = 3) -> tuple[str, float]:
    for _ in range(warm):
        work()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1000)
    # The median, not the mean: one scheduling hiccup on a laptop moves a mean
    # of thirty samples by more than the thing being measured.
    return label, statistics.median(samples)


def _scene(width=1280, height=720, people=1, seed=0):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(30, 200, (height // 20, width // 20, 3), dtype=np.uint8)
    frame = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    boxes = []
    for i in range(people):
        x = 0.15 + 0.25 * i
        box = BoundingBox(x, 0.55, 0.07, 0.25)
        colour = [(210, 60, 60), (60, 60, 210), (60, 200, 60)][i % 3]
        frame[int(box.y * height):int(box.bottom * height),
              int(box.x * width):int(box.right * width)] = colour
        boxes.append(box)
    return frame, boxes


def cost_report() -> list[tuple[str, float]]:
    """Milliseconds per frame, per stage, on this machine."""
    out: list[tuple[str, float]] = []
    frame, boxes = _scene()
    detections = [Detection(b, 0.9, 1) for b in boxes]

    quality = FrameQualityMonitor()
    out.append(_timed("frame quality (1280x720)", lambda: quality.measure(frame)))

    motion = CameraMotionEstimator()
    motion.estimate(frame)
    shifted = cv2.warpAffine(frame, np.array([[1, 0, 6.0], [0, 1, 0]], np.float32),
                             (frame.shape[1], frame.shape[0]))
    out.append(_timed("camera motion (1280x720)",
                      lambda: (motion.reset(), motion.estimate(frame), motion.estimate(shifted))))

    out.append(_timed("appearance, per detection",
                      lambda: describe(frame, (boxes[0].x, boxes[0].y, boxes[0].width, boxes[0].height))))

    looks = [describe(frame, (b.x, b.y, b.width, b.height)) for b in boxes]
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    counter = {"t": 0}

    def track_step():
        counter["t"] += STEP
        tracker.update(detections, counter["t"], appearances=looks)

    out.append(_timed("track update, 1 track", track_step))

    many = [Detection(BoundingBox(0.02 + 0.03 * i, 0.5, 0.02, 0.06), 0.9, 1) for i in range(30)]
    crowd = Tracker(TrackerConfig(min_hits_to_confirm=1))
    counter2 = {"t": 0}

    def crowd_step():
        counter2["t"] += STEP
        crowd.update(many, counter2["t"])

    out.append(_timed("track update, 30 tracks", crowd_step))

    cost = np.random.default_rng(0).random((32, 32))
    out.append(_timed("assignment, 32x32", lambda: native.assign(cost), runs=200))

    out.append(_timed("ground projection, one point",
                      lambda: project_to_ground(POSE, 0.5, 0.8), runs=500))

    if native.available():
        from vigil.service.mapping import site_lattice

        grid = site_lattice(SITE, 0.25, POSE)
        cells = grid.cells
        colour = np.zeros(cells * 3, dtype=np.uint8)
        valid = np.zeros(cells, dtype=np.uint8)
        resolution = np.full(cells, np.inf, dtype=np.float32)
        values = grid.values()
        out.append(_timed(
            f"ground sample ({cells:,} cells)",
            lambda: native.ortho_sample(POSE, values, frame, grid.rows, grid.cols, 33,
                                        colour, valid, resolution),
        ))
    return out


def detector_report() -> list[tuple[str, float]]:
    """The real model, when the operator has put one where it can be found."""
    from vigil.config import Settings
    from vigil.adapters.detectors import OnnxDetector, available_providers

    model = Settings.from_environment().default_model()
    if model is None:
        return [("onnx detection", float("nan"))]
    detector = OnnxDetector(model)
    frame, _ = _scene()
    print(f"  model: {model.name} on {detector.info.provider} "
          f"(available: {', '.join(available_providers())})")
    return [_timed(f"onnx detect+segment ({detector.info.input_size[0]}px)",
                   lambda: detector.detect(frame), runs=10, warm=2)]


def fragmentation(with_appearance: bool, blind_from_ms: int = 2500,
                  period_ms: int = 4000, seconds: int = 20) -> int:
    """Distinct ids reported for one person over `seconds` of blinking detector.

    The truth is one. v1 measured 3, 10, 4 and 11 on a laptop camera and wrote
    `reid.py` to reconcile the count afterwards; this measures whether
    consulting appearance *during* association prevents the split instead.
    """
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=2, coast_millis=400))
    ids: set[int] = set()
    width, height = 640, 360
    for i in range(seconds * FPS):
        millis = i * STEP
        rng = np.random.default_rng(i)
        coarse = rng.integers(30, 120, (height // 5, width // 5, 3), dtype=np.uint8)
        frame = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        box = BoundingBox(0.05 + 0.85 * (i / (seconds * FPS)), 0.55, 0.08, 0.24)
        frame[int(box.y * height):int(box.bottom * height),
              int(box.x * width):int(box.right * width)] = (200, 50, 50)
        blind = (millis % period_ms) > blind_from_ms
        detections = [] if blind else [Detection(box, 0.9, 1)]
        looks = None
        if with_appearance and detections:
            looks = [describe(frame, (box.x, box.y, box.width, box.height))]
        elif with_appearance:
            looks = []
        tracker.update(detections, millis, appearances=looks)
        ids.update(t.id for t in tracker.tracks())
    return len(ids)


def _greedy(cost: np.ndarray) -> np.ndarray:
    """The association v1 and v2 shared, as a drop-in for the optimal solver.

    Sort every pair by cost, walk the list, take a pair whenever neither side
    is already claimed. Here so that the claim "greedy swaps identities" is a
    measurement of this code rather than an assertion about another version.
    """
    rows, cols = cost.shape
    out = np.full(rows, -1, dtype=np.int64)
    taken_rows, taken_cols = set(), set()
    for r, c in sorted(((r, c) for r in range(rows) for c in range(cols)),
                       key=lambda rc: cost[rc]):
        if r in taken_rows or c in taken_cols or cost[r, c] >= 1.0e8:
            continue
        taken_rows.add(r)
        taken_cols.add(c)
        out[r] = c
    return out


def crossing_swaps(with_appearance: bool, runs: int = 12, greedy: bool = False) -> int:
    """How many of `runs` head-on crossings end with the identities swapped."""
    return _run_scenario(_crossing, with_appearance, runs, greedy)


def crowd_switches(with_appearance: bool, runs: int = 8, greedy: bool = False) -> int:
    """Identity switches when several people mill about in a small space.

    The scenario that actually discriminates. A head-on crossing does not:
    the filter's velocity estimate keeps the two predictions separated
    through it, so greedy and optimal both hold identity — a real result, and
    a measurement of the *filter* rather than of the association.

    Milling is different. People stop, turn and pass at walking pace inside a
    few of their own widths, so predictions genuinely overlap and the cost
    matrix is genuinely ambiguous. That is where a greedy pass takes the best
    single pair and is then forced into whatever remains.
    """
    return _run_scenario(_milling, with_appearance, runs, greedy)


def _run_scenario(scenario, with_appearance: bool, runs: int, greedy: bool) -> int:
    import vigil.domain.tracking as tracking_module

    original = tracking_module.native.assign
    if greedy:
        tracking_module.native.assign = _greedy
    try:
        return sum(scenario(with_appearance, run) for run in range(runs))
    finally:
        tracking_module.native.assign = original


def _render(boxes_and_colours, seed, width=640, height=360):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(20, 90, (height // 5, width // 5, 3), dtype=np.uint8)
    frame = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    for box, colour in boxes_and_colours:
        frame[max(0, int(box.y * height)):int(box.bottom * height),
              max(0, int(box.x * width)):int(box.right * width)] = colour
    return frame


def _switches(tracker, truth_boxes_over_time, colours, with_appearance) -> int:
    """Run a scripted scene and count how often a track changes person.

    A track is assigned the ground-truth person whose box it overlaps most.
    Every time that assignment changes for a track that already had one, the
    tracker has swapped somebody's identity.
    """
    owner: dict[int, int] = {}
    switches = 0
    for step, boxes in enumerate(truth_boxes_over_time):
        frame = _render(list(zip(boxes, colours)), step, )
        detections = [Detection(b, 0.9, 1) for b in boxes]
        looks = ([describe(frame, (b.x, b.y, b.width, b.height)) for b in boxes]
                 if with_appearance else None)
        tracker.update(detections, step * STEP, appearances=looks)
        for track in tracker.tracks():
            overlaps = [track.bbox.iou(b) for b in boxes]
            best = int(np.argmax(overlaps))
            if overlaps[best] <= 0.0:
                continue
            if track.id in owner and owner[track.id] != best:
                switches += 1
            owner[track.id] = best
    return switches


def _crossing(with_appearance: bool, run: int) -> int:
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    script = []
    for i in range(40):
        t = i / 39
        script.append([BoundingBox(0.15 + 0.5 * t, 0.55, 0.09, 0.22),
                       BoundingBox(0.65 - 0.5 * t, 0.57, 0.09, 0.22)])
    return min(1, _switches(tracker, script, [(220, 40, 40), (40, 40, 220)], with_appearance))


def _milling(with_appearance: bool, run: int) -> int:
    """Four people random-walking within a couple of body widths of each other."""
    rng = np.random.default_rng(1000 + run)
    tracker = Tracker(TrackerConfig(min_hits_to_confirm=1))
    positions = np.array([[0.35, 0.55], [0.45, 0.57], [0.40, 0.60], [0.50, 0.54]])
    script = []
    for _ in range(60):
        positions = np.clip(positions + rng.normal(0, 0.012, positions.shape), 0.05, 0.85)
        script.append([BoundingBox(x, y, 0.07, 0.20) for x, y in positions])
    colours = [(220, 40, 40), (40, 40, 220), (40, 200, 40), (200, 200, 40)]
    return _switches(tracker, script, colours, with_appearance)


def main() -> int:
    print(f"machine   {platform.processor() or platform.machine()}, "
          f"{platform.system()} {platform.release()}, Python {platform.python_version()}")
    print(f"core      {'loaded' if native.available() else 'NOT LOADED — ' + str(native.fault())}")
    print()
    print("COST  (median ms per call on this machine; a laptop under load reads differently)")
    rows = cost_report()
    try:
        rows += detector_report()
    except Exception as error:  # noqa: BLE001 - a missing model is not a failed benchmark
        print(f"  (detector not measured: {error})")
    for label, millis in rows:
        if millis != millis:  # NaN
            print(f"  {label:<34} not measured (no model)")
        else:
            print(f"  {label:<34} {millis:8.3f} ms")

    print()
    print("QUALITY  (synthetic; real video is harder)")
    without = fragmentation(with_appearance=False)
    with_looks = fragmentation(with_appearance=True)
    print(f"  one person, blinking detector, 20 s")
    print(f"    ids without appearance         {without:>4}   (truth: 1)")
    print(f"    ids with appearance            {with_looks:>4}   (truth: 1)")
    variants = (
        ("greedy, no appearance (v1/v2)", {"greedy": True, "with_appearance": False}),
        ("greedy + appearance", {"greedy": True, "with_appearance": True}),
        ("optimal, no appearance", {"greedy": False, "with_appearance": False}),
        ("optimal + appearance (shipped)", {"greedy": False, "with_appearance": True}),
    )
    print("  two people crossing head-on, 12 runs (swapped runs)")
    for label, kwargs in variants:
        print(f"    {label:<32} {crossing_swaps(**kwargs):>4} / 12")
    print("  four people milling in a small space, 8 runs (identity switches)")
    for label, kwargs in variants:
        print(f"    {label:<32} {crowd_switches(**kwargs):>4}")

    print()
    budget = sum(m for _, m in rows[:5] if m == m)
    print(f"per-frame perception+tracking budget on this machine: {budget:.1f} ms "
          f"({1000 / budget:.0f} fps for one camera, before detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
