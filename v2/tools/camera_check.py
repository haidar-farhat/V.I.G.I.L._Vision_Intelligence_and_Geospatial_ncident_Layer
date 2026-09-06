"""The whole pipeline against a real camera, and what it actually did.

    python tools/camera_check.py [--source device:0] [--seconds 30] [--map]

Everything else in `tools/` measures synthetic frames. This measures the one
thing synthetic frames cannot: whether the pipeline holds together on real
pixels, with real noise, real light, real motion blur and a real detector.

It is a *check*, not a test — it needs a camera, so it cannot run in CI, and
it prints what happened rather than asserting. What it prints is chosen to
make a bad run obvious: frames that were unusable and why, whether the camera
was measured to move, how many objects were seen against how many track ids
were issued, and the wall-clock cost of every stage.

The ratio to watch is **ids issued against objects present**. If one person
walks through and it reports eleven ids, that is v1's defect and it is back.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.adapters.decode import LiveReader, VideoSource  # noqa: E402
from vigil.adapters.detectors import available_providers, detector_for  # noqa: E402
from vigil.config import Settings  # noqa: E402
from vigil.domain.geo import CameraPose, LatLon  # noqa: E402
from vigil.domain.tracking import Tracker, TrackerConfig  # noqa: E402
from vigil.kernel import native  # noqa: E402
from vigil.perception.appearance import describe  # noqa: E402
from vigil.perception.motion import CameraMotionEstimator  # noqa: E402
from vigil.perception.quality import FrameQualityMonitor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="device:0")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--map", action="store_true", help="also build a ground map from it")
    parser.add_argument("--png", help="write the map picture here")
    args = parser.parse_args(argv)

    settings = Settings.from_environment()
    model = settings.default_model()
    detector = detector_for(model)
    print(f"source    {args.source}")
    print(f"detector  {detector.info.kind} {detector.info.name}"
          f"{' on ' + detector.info.provider if detector.info.provider else ''}")
    print(f"providers {', '.join(available_providers()) or 'none'}")
    print(f"core      {'loaded' if native.available() else 'NOT LOADED'}")
    print()

    pose = CameraPose(LatLon(33.8938, 35.5018), 2.5, 0.0, -20.0, 0.0, 62.0, 36.0, 25.0)
    tracker = Tracker(TrackerConfig(), pose)
    quality = FrameQualityMonitor()
    motion = CameraMotionEstimator()

    source = VideoSource(args.source, source_id="check")
    info = source.open()
    print(f"opened    {info.width}x{info.height} @ {info.nominal_fps:.1f} fps nominal")
    reader = LiveReader(source) if source.live else None
    if reader is not None:
        reader.start()

    builder = None
    if args.map:
        if not native.available():
            print("(no map: the engine core is not loaded)")
        else:
            from vigil.service.mapping import MapBuilder

            builder = MapBuilder(pose.position, cell_size_m=0.1, depth=11)

    stage: dict[str, list[float]] = {k: [] for k in ("read", "quality", "motion", "detect", "describe", "track")}
    faults: Counter[str] = Counter()
    ids: set[int] = set()
    frames = unusable = stale = moved = detections_total = 0
    peak_tracks = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            t0 = time.perf_counter()
            frame = reader.read(timeout=1.0) if reader is not None else source.read()
            t1 = time.perf_counter()
            if frame is None:
                if reader is None:
                    break
                continue
            frames += 1
            stage["read"].append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            measured = quality.measure(frame.image)
            stage["quality"].append((time.perf_counter() - t0) * 1000)
            if measured.faults:
                unusable += 1
                faults[measured.faults[0].split("(")[0].strip()] += 1
            if measured.stale:
                stale += 1

            warp = None
            if measured.usable:
                t0 = time.perf_counter()
                camera = motion.estimate(frame.image)
                stage["motion"].append((time.perf_counter() - t0) * 1000)
                if camera.measured and not camera.still:
                    moved += 1
                    warp = camera.warp

            t0 = time.perf_counter()
            found = detector.detect(frame.image)
            stage["detect"].append((time.perf_counter() - t0) * 1000)
            detections_total += len(found)

            t0 = time.perf_counter()
            looks = [describe(frame.image, (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height), d.mask)
                     for d in found]
            stage["describe"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            tracker.update(found, frame.timestamp_millis, appearances=looks, warp=warp)
            stage["track"].append((time.perf_counter() - t0) * 1000)
            live = tracker.tracks()
            ids.update(t.id for t in live)
            peak_tracks = max(peak_tracks, len(live))

            if builder is not None and measured.worth_sampling:
                builder.observe("check", pose, frame.image, at_seconds=time.monotonic())
    finally:
        if reader is not None:
            reader.stop()
        source.close()

    elapsed = time.monotonic() - started
    print()
    print(f"frames    {frames} in {elapsed:.1f} s = {frames / max(elapsed, 1e-9):.1f} fps end to end")
    print(f"unusable  {unusable} ({unusable / max(1, frames):.0%})"
          + (f" — {', '.join(f'{k}: {v}' for k, v in faults.most_common())}" if faults else ""))
    print(f"stale     {stale} frame(s) identical to the one before")
    print(f"moved     the camera measurably moved on {moved} frame(s) "
          f"({moved / max(1, frames):.0%})")
    print(f"detected  {detections_total} detection(s), {detections_total / max(1, frames):.1f} per frame")
    print(f"tracks    {len(ids)} distinct id(s) issued, {peak_tracks} live at once")
    if peak_tracks and len(ids) > peak_tracks * 3:
        print("          ^ far more ids than were ever live at once. That is fragmentation: "
              "one object being counted as several.")
    print()
    print("cost, median ms per frame")
    for name, samples in stage.items():
        if samples:
            print(f"  {name:<10} {statistics.median(samples):7.2f}  (p90 "
                  f"{statistics.quantiles(samples, n=10)[-1] if len(samples) > 9 else float('nan'):.2f})")

    if builder is not None:
        ground = builder.build(minimum_samples=5)
        print()
        print(f"map       {ground.describe()}")
        if args.png:
            import cv2

            from vigil.service.mapping import visualise

            cv2.imwrite(args.png, visualise(ground))
            print(f"          picture written to {args.png}")
        builder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
