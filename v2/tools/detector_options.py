"""How to make detection affordable, measured rather than argued.

    python tools/detector_options.py [--clip CLIP.mp4] [--source device:0]

`PRODUCTION_READINESS.md` says 83 ms per frame of CPU detection is about 12 fps
for *one* camera, and that the multi-camera claim is not supported by this
hardware. That is the largest performance problem in the product, and it has
four possible answers. This measures three of them; the fourth needs hardware
this machine does not have.

**1. Detect on fewer frames and track in between.** The largest win available,
and it only became *safe* with the rewritten tracker: a Kalman filter that
predicts properly, a second association pass that recovers an object from a
weak detection, and re-identification that survives a gap. Detecting every
third frame cuts the detection bill by two thirds. What it costs is position
lag — measured here against the full-rate run, in fractions of a box height,
which is the unit a zone judgement cares about.

**2. INT8 dynamic quantisation.** onnxruntime ships it; no download, no
retraining, no new dependency. Typically 2-3x on a CPU for a convolutional
network. What it costs is accuracy, and this measures the *disagreement* with
the float model rather than assuming it is small.

**3. Threads.** The session is configured for sixteen cameras sharing a
machine, which is the wrong setting for one camera having it to itself.

**4. A GPU or NPU.** Not measured, because this machine's onnxruntime is
CPU-only. `vigil doctor` now reports which providers exist, and installing
`onnxruntime-directml` on this AMD APU is the single largest untested win.

Everything is measured on the same frames, in one process, back to back.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.adapters.detectors import OnnxDetector  # noqa: E402
from vigil.config import Settings  # noqa: E402
from vigil.domain.tracking import Tracker, TrackerConfig  # noqa: E402
from vigil.perception.appearance import describe  # noqa: E402

#: Frames used for every measurement. Enough that a median is stable and few
#: enough that the whole tool runs in a couple of minutes on a CPU.
FRAMES = 24


def _frames(clip: str | None, source: str | None, count: int) -> list[np.ndarray]:
    """Real frames if there are any, synthetic ones if not — and it says which."""
    import cv2

    if clip:
        capture = cv2.VideoCapture(clip)
        out = []
        while len(out) < count:
            ok, image = capture.read()
            if not ok:
                break
            out.append(image)
        capture.release()
        if out:
            print(f"frames    {len(out)} from {clip}")
            return out
        print(f"could not read {clip}")
    if source:
        from vigil.adapters.decode import LiveReader, VideoSource

        video = VideoSource(source, source_id="options")
        video.open()
        reader = LiveReader(video)
        reader.start()
        out = []
        deadline = time.monotonic() + count / 5.0 + 10.0
        while len(out) < count and time.monotonic() < deadline:
            frame = reader.read(timeout=1.0)
            if frame is not None:
                out.append(frame.image)
        reader.stop()
        video.close()
        if out:
            print(f"frames    {len(out)} from {source}")
            return out
    rng = np.random.default_rng(0)
    coarse = rng.integers(20, 220, (36, 64, 3), dtype=np.uint8)
    base = cv2.resize(coarse, (1280, 720), interpolation=cv2.INTER_CUBIC)
    print(f"frames    {count} synthetic — pass --clip or --source for a real measurement")
    return [np.roll(base, i * 3, axis=1) for i in range(count)]


def _time(detector, frames: list[np.ndarray]) -> tuple[float, list[list]]:
    for frame in frames[:2]:
        detector.detect(frame)
    samples: list[float] = []
    results: list[list] = []
    for frame in frames:
        started = time.perf_counter()
        results.append(detector.detect(frame))
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), results


def measure_threads(model: Path, frames: list[np.ndarray]) -> None:
    import os

    print()
    print("THREADS  (the session default is tuned for many cameras sharing a machine)")
    original = os.environ.get("VIGIL_ORT_THREADS")
    best = None
    for threads in (1, 2, 4, 8):
        os.environ["VIGIL_ORT_THREADS"] = str(threads)
        detector = OnnxDetector(model)
        millis, _ = _time(detector, frames)
        marker = ""
        if best is None or millis < best[1]:
            best = (threads, millis)
            marker = "  <- fastest so far"
        print(f"  {threads:>2} thread(s)   {millis:7.1f} ms   {1000 / millis:5.1f} fps{marker}")
        del detector
    if original is None:
        os.environ.pop("VIGIL_ORT_THREADS", None)
    else:
        os.environ["VIGIL_ORT_THREADS"] = original
    if best:
        print(f"  For ONE camera on this machine: VIGIL_ORT_THREADS={best[0]}.")
        print("  For N cameras, more threads per session is worse, not better: the sessions")
        print("  deschedule each other. The shipped default of 2 is for that case.")


def measure_quantised(model: Path, frames: list[np.ndarray], baseline_ms: float,
                      baseline: list[list]) -> None:
    print()
    print("INT8 DYNAMIC QUANTISATION  (no download, no retraining, no new dependency)")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("  onnxruntime.quantization is not available in this install")
        return
    work = Path(tempfile.mkdtemp(prefix="vigil-quant-"))
    try:
        quantised = work / "int8.onnx"
        try:
            quantize_dynamic(str(model), str(quantised), weight_type=QuantType.QUInt8)
        except Exception as error:  # noqa: BLE001 - a failed quantisation is a result
            print(f"  quantisation failed: {type(error).__name__}: {error}")
            return
        print(f"  model     {model.stat().st_size / 1e6:.1f} MB -> "
              f"{quantised.stat().st_size / 1e6:.1f} MB")
        detector = OnnxDetector(quantised)
        millis, results = _time(detector, frames)
        print(f"  speed     {baseline_ms:.1f} ms -> {millis:.1f} ms  "
              f"({baseline_ms / millis:.2f}x)")
        # Agreement, not "accuracy": without labels the float model is the only
        # reference there is, and the honest question is how far the int8 one
        # has moved from it.
        matched = missed = extra = 0
        drifts: list[float] = []
        for a, b in zip(baseline, results):
            used = set()
            for detection in a:
                best, best_iou = None, 0.0
                for i, other in enumerate(b):
                    if i in used or other.class_id != detection.class_id:
                        continue
                    iou = detection.bbox.iou(other.bbox)
                    if iou > best_iou:
                        best, best_iou = i, iou
                if best is not None and best_iou > 0.5:
                    used.add(best)
                    matched += 1
                    drifts.append(abs(detection.confidence - b[best].confidence))
                else:
                    missed += 1
            extra += len(b) - len(used)
        total = matched + missed
        if total:
            print(f"  agreement {matched}/{total} of the float model's detections kept "
                  f"({matched / total:.0%}), {extra} it did not have")
            if drifts:
                print(f"            confidence moved by {statistics.median(drifts):.3f} (median)")
            if matched / total < 0.9:
                print("  ^ it loses more than a tenth of the detections. Measure that against")
                print("    labelled footage before shipping it; a tenth at the far edge of the")
                print("    frame is not the same as a tenth in the middle.")
        else:
            print("  nothing was detected in these frames, so agreement cannot be judged.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def measure_interval(frames: list[np.ndarray], results: list[list], detect_ms: float) -> None:
    """Detect every Nth frame and track through the rest; what it costs."""
    print()
    print("DETECTION INTERVAL  (the largest win, and only safe since the tracker was rebuilt)")
    if not any(results):
        print("  nothing was detected in these frames, so the cost cannot be judged.")
        return
    step = 1000 // 15

    def run(interval: int) -> dict[int, list[tuple[int, float, float, float]]]:
        tracker = Tracker(TrackerConfig())
        seen: dict[int, list[tuple[int, float, float, float]]] = {}
        for index, (frame, found) in enumerate(zip(frames, results)):
            detections = found if index % interval == 0 else []
            looks = ([describe(frame, (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height), d.mask)
                      for d in detections] if detections else [])
            tracker.update(detections, index * step, appearances=looks)
            for track in tracker.tracks():
                box = track.bbox
                seen.setdefault(track.id, []).append(
                    (index, box.center.x, box.center.y, box.height))
        return seen

    reference = run(1)
    reference_ids = len(reference)
    print(f"  {'every':<8} {'detect ms':>10} {'ids':>5} {'median lag':>12} {'p95 lag':>10}")
    for interval in (1, 2, 3, 5):
        seen = run(interval)
        drifts: list[float] = []
        for tid, samples in seen.items():
            truth = {i: (x, y, h) for i, x, y, h in reference.get(tid, [])}
            for index, x, y, h in samples:
                if index in truth:
                    tx, ty, _ = truth[index]
                    drifts.append(np.hypot(x - tx, y - ty) / max(h, 1e-6))
        median = statistics.median(drifts) if drifts else 0.0
        p95 = np.percentile(drifts, 95) if drifts else 0.0
        print(f"  {interval:<8} {detect_ms / interval:>10.1f} {len(seen):>5} "
              f"{median:>11.3f}h {p95:>9.3f}h")
    print("  'lag' is how far a track's box sits from where full-rate detection put it,")
    print("  in box heights. A person is about 0.4 m wide and 1.7 m tall, so 0.1h is")
    print(f"  roughly 17 cm on the ground — compare that against the projection error the")
    print("  same camera reports, which is typically over a metre at range.")
    if reference_ids:
        print(f"  Watch the id count: if it climbs with the interval, tracking is not carrying")
        print(f"  the gap and the interval is too long for this scene ({reference_ids} at full rate).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="a recorded clip; the honest input")
    parser.add_argument("--source", help="a live camera instead")
    parser.add_argument("--frames", type=int, default=FRAMES)
    args = parser.parse_args(argv)

    model = Settings.from_environment().default_model()
    if model is None:
        print("no model found. This measures a detector, so there has to be one.")
        return 2
    frames = _frames(args.clip, args.source, args.frames)
    print(f"model     {model.name}")

    from vigil.adapters.detectors import available_providers

    print(f"providers {', '.join(available_providers())}")
    detector = OnnxDetector(model)
    baseline_ms, baseline = _time(detector, frames)
    found = sum(len(r) for r in baseline)
    print(f"baseline  {baseline_ms:.1f} ms per frame, {1000 / baseline_ms:.1f} fps, "
          f"{found} detection(s) over {len(frames)} frames")

    measure_interval(frames, baseline, baseline_ms)
    measure_quantised(model, frames, baseline_ms, baseline)
    measure_threads(model, frames)
    print()
    print("Not measured here: a GPU or NPU. `vigil doctor` reports the providers this")
    print("install offers; onnxruntime-directml on this machine's integrated GPU is the")
    print("largest untested win and needs one `pip install` to try.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
