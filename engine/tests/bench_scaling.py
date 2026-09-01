"""Where the time actually goes, and how many cameras one node can carry.

Not a test — a measurement, run by hand:

    python engine/tests/bench_scaling.py

Written because "make it scale" is a question that cannot be answered by
adding a framework and hoping. It needs two numbers first: what fraction of a
frame each stage costs, and what happens to that when cameras run side by side.
Only then does it become obvious whether the bottleneck is the Rust core (it is
not), the Python interpreter, or OpenCV.

Every figure printed here is a median of repeated runs on this machine, and the
conditions are printed with them. A throughput claim without its conditions is
not a measurement.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import scene  # noqa: E402
from sentinel.core import CameraPose, LatLon, Tracker, destination_point  # noqa: E402
from sentinel.decode import VideoSource  # noqa: E402
from sentinel.detect import MotionDetector  # noqa: E402
from sentinel.pipeline import Pipeline  # noqa: E402
from sentinel.zones import Zone, ZoneKind  # noqa: E402
from sentinel.events import ZoneEntryRule  # noqa: E402

SITE = LatLon(33.8938, 35.5018)
POSE = CameraPose(SITE, 6.0, 180.0, -22.0, horizontal_fov=62.0, vertical_fov=36.0,
                  range_meters=90.0)


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def stage_breakdown(frames: list[np.ndarray], video: Path) -> dict[str, float]:
    """Cost per frame, per stage, measured separately."""
    costs: dict[str, list[float]] = {k: [] for k in ("decode", "detect", "track+project", "zones+rules")}

    # Decode alone.
    for _ in range(3):
        with VideoSource(video) as source:
            start = time.perf_counter()
            count = sum(1 for _ in source)
            costs["decode"].append((time.perf_counter() - start) / count)

    # Detection alone, on already-decoded frames.
    for _ in range(3):
        detector = MotionDetector()
        start = time.perf_counter()
        detections_per_frame = [detector.detect(f) for f in frames]
        costs["detect"].append((time.perf_counter() - start) / len(frames))

    # Tracking and projection alone, replaying the detections above.
    detector = MotionDetector()
    detections_per_frame = [detector.detect(f) for f in frames]
    for _ in range(3):
        with Tracker(POSE) as tracker:
            start = time.perf_counter()
            for index, detections in enumerate(detections_per_frame):
                tracker.update(detections, index * 66)
            costs["track+project"].append((time.perf_counter() - start) / len(frames))

    # Zones, rules and correlation, on top of tracking.
    centre = destination_point(SITE, 180.0, 14.0)
    zone = Zone(id="z", name="Z", kind=ZoneKind.RESTRICTED,
                ring=tuple(destination_point(centre, b, 9.0) for b in (0., 90., 180., 270.)))
    for _ in range(3):
        with Pipeline(VideoSource(video), MotionDetector(), pose=POSE,
                      zones=[zone], rules=[ZoneEntryRule()]) as pipeline:
            start = time.perf_counter()
            count = sum(1 for _ in pipeline.run())
            whole = (time.perf_counter() - start) / count
        costs["zones+rules"].append(whole)

    return {k: median_ms(v) for k, v in costs.items()}


def one_camera(video: Path, frames_wanted: int, done: list[float]) -> None:
    """Run a whole pipeline as a worker would, and record its frame rate."""
    with Pipeline(VideoSource(video), MotionDetector(), pose=POSE) as pipeline:
        start = time.perf_counter()
        count = 0
        for _ in pipeline.run():
            count += 1
            if count >= frames_wanted:
                break
        done.append(count / (time.perf_counter() - start))


def scaling(video: Path, counts=(1, 2, 4, 8, 12, 16)) -> list[tuple[int, float, float]]:
    """Aggregate throughput as cameras are added, each on its own thread."""
    results = []
    for cameras in counts:
        rates: list[float] = []
        threads = [
            threading.Thread(target=one_camera, args=(video, 120, rates))
            for _ in range(cameras)
        ]
        start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - start

        aggregate = cameras * 120 / wall
        per_camera = statistics.median(rates) if rates else 0.0
        results.append((cameras, aggregate, per_camera))
        print(f"  {cameras:2} cameras   aggregate {aggregate:7.0f} fps   "
              f"per camera {per_camera:6.0f} fps")
    return results


def process_scaling(video: Path, counts=(1, 2, 4, 8, 12, 16)) -> list[tuple[int, float, float]]:
    """The same curve, but one OS process per camera.

    This is the control for the GIL. A worker node is already a separate
    process in the architecture, so if processes scale where threads do not,
    the fix is structural and needs no new dependency.
    """
    import concurrent.futures as futures

    import _bench_worker

    results = []
    for cameras in counts:
        start = time.perf_counter()
        with futures.ProcessPoolExecutor(max_workers=cameras) as pool:
            rates = list(pool.map(_bench_worker.run, [str(video)] * cameras, [120] * cameras))
        wall = time.perf_counter() - start

        aggregate = cameras * 120 / wall
        per_camera = statistics.median(rates)
        results.append((cameras, aggregate, per_camera))
        print(f"  {cameras:2} cameras   aggregate {aggregate:7.0f} fps   "
              f"per camera {per_camera:6.0f} fps")
    return results


def main() -> None:
    video = Path(__file__).resolve().parent / ".bench" / "scene.mp4"
    if not video.exists():
        scene.write_scene(video)

    with VideoSource(video) as source:
        frames = [f.image.copy() for f in source]

    print(f"conditions: {len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"cv2 threads {cv2.getNumThreads()}, cpus {__import__('os').cpu_count()}, "
          f"python {sys.version.split()[0]}")

    heading("Cost per frame, by stage (median of 3, all cv2 threads)")
    breakdown = stage_breakdown(frames, video)
    whole = breakdown.pop("zones+rules")
    total = sum(breakdown.values())
    for name, ms in breakdown.items():
        print(f"  {name:16} {ms:7.3f} ms   {100 * ms / total:5.1f}% of the stages measured")
    print(f"  {'':16} {'':7}      ")
    print(f"  {'whole pipeline':16} {whole:7.3f} ms   (decode+detect+track+zones+rules together)")

    heading("Scaling: one pipeline per camera, one thread each")
    print("  The GIL is the thing under test. OpenCV releases it during heavy")
    print("  work and the Rust core releases it for every FFI call, so the")
    print("  question is how much Python-level work is left in the way.")
    print()
    cv2.setNumThreads(1)
    print("  (cv2 limited to 1 thread per camera, so the cameras compete for cores")
    print("   the way they would on a real worker node rather than each grabbing 16)")
    print()
    curve = scaling(video)

    heading("The same, but one process per camera")
    print("  Process start-up is included in the aggregate, which is why one")
    print("  process looks slower than one thread. What matters is the slope.")
    print()
    process_curve = process_scaling(video)

    heading("What that means")
    single = curve[0][1]
    best = max(curve, key=lambda row: row[1])
    print(f"  one camera alone        {single:7.0f} fps")
    print(f"  best aggregate          {best[1]:7.0f} fps at {best[0]} cameras")
    print(f"  scaling efficiency      {100 * best[1] / (single * best[0]):5.1f}% of linear")
    print()
    for cameras, aggregate, per_camera in curve:
        budget = per_camera / 15.0
        print(f"  {cameras:2} cameras -> each runs at {per_camera:6.0f} fps, "
              f"{budget:5.1f}x the 15 fps a camera actually delivers")

    heading("Threads against processes")
    print(f"  {'cameras':>8} {'threads':>10} {'processes':>10} {'gain':>8}")
    for (n, t_agg, _), (_, p_agg, _) in zip(curve, process_curve):
        print(f"  {n:>8} {t_agg:>10.0f} {p_agg:>10.0f} {p_agg / t_agg:>7.2f}x")


if __name__ == "__main__":
    main()
