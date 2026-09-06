"""What real video says about this system's constants — with no labels at all.

    python tools/calibrate.py CLIP.mp4 [CLIP2.mp4 ...]
    python tools/calibrate.py --source device:0 --seconds 60

`PRODUCTION_READINESS.md` says nothing measures precision or recall because
there is nothing to measure them against. That is true and it is not the whole
truth: **several of the constants this system runs on can be measured from
unlabelled video**, because the truth for them comes from the structure of the
problem rather than from a person.

Four of them, and what makes each one honest:

- **Appearance separation.** Two detections in the *same frame* are certainly
  different objects — one object cannot be in two places. Two appearances of
  the *same continuously-detected track* are almost certainly one object. So
  the within-track and between-track distance distributions are ground truth
  that nobody had to label, and where they cross is where
  `MAX_REIDENTIFY_DISTANCE` belongs. Today both gates are set from synthetic
  colour blocks, which is the weakest thing in the tracker.

- **Detector box jitter.** A track whose ground position is not moving is a
  stationary object, and everything its box does frame to frame is the
  detector's noise. That measures `MEASURE_STD_FRACTION` directly — a number
  currently asserted as "5%, measured on the shipped model" with nothing
  holding it there.

- **Class stability.** A confirmed track whose class flips is a model that is
  guessing. No label is needed to know that a person did not become a car and
  back; it is a lower bound on the error rate.

- **Confidence shape.** A detector that knows what it is looking at produces a
  bimodal confidence distribution. A flat one on this site's own footage means
  the stock weights are out of their domain, which is the argument for
  fine-tuning made from evidence rather than from suspicion.

**What this cannot do.** It cannot measure recall — an object nobody detected
leaves no trace here — and it cannot measure precision, because a confident
detection of nothing looks exactly like a confident detection of something.
Those need labels. This measures everything that does not.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.adapters.decode import LiveReader, VideoSource  # noqa: E402
from vigil.adapters.detectors import detector_for  # noqa: E402
from vigil.config import Settings  # noqa: E402
from vigil.domain.appearance import MAX_APPEARANCE_DISTANCE, MAX_REIDENTIFY_DISTANCE  # noqa: E402
from vigil.domain.tracking import Tracker, TrackerConfig  # noqa: E402
from vigil.kernel.filtering import MEASURE_STD_FRACTION  # noqa: E402
from vigil.perception.appearance import describe  # noqa: E402
from vigil.perception.quality import FrameQualityMonitor  # noqa: E402

#: A track has to be seen at least this many times before its appearances are
#: treated as one object. A two-frame track is as likely to be a detector
#: artefact as an object.
MIN_TRACK_HITS = 8

#: Frame-to-frame box-centre motion, as a fraction of box height, below which
#: a track is treated as stationary for the jitter measurement.
STATIONARY_FRACTION = 0.02


class Observations:
    """Everything one run saw, keyed so the truth is structural."""

    def __init__(self) -> None:
        self.looks: dict[int, list] = defaultdict(list)
        self.continuous: dict[int, bool] = defaultdict(lambda: True)
        self.classes: dict[int, list[int]] = defaultdict(list)
        self.confidences: list[float] = []
        self.same_frame: list[tuple[int, int]] = []
        self.boxes: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        self.hits: dict[int, int] = defaultdict(int)
        self.peak_live = 0
        self.frames = 0
        self.unusable = 0


def observe(source_url: str, seconds: float | None, out: Observations) -> None:
    settings = Settings.from_environment()
    detector = detector_for(settings.default_model())
    tracker = Tracker(TrackerConfig())
    quality = FrameQualityMonitor()
    source = VideoSource(source_url, source_id="calibrate")
    source.open()
    reader = LiveReader(source) if source.live else None
    if reader is not None:
        reader.start()
    import time

    started = time.monotonic()
    try:
        while seconds is None or time.monotonic() - started < seconds:
            frame = reader.read(timeout=1.0) if reader is not None else source.read()
            if frame is None:
                if reader is None:
                    break
                continue
            out.frames += 1
            if not quality.measure(frame.image).usable:
                out.unusable += 1
                continue
            detections = detector.detect(frame.image)
            looks = [describe(frame.image, (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height), d.mask)
                     for d in detections]
            out.confidences.extend(d.confidence for d in detections)
            tracker.update(detections, frame.timestamp_millis, appearances=looks)
            live = tracker.tracks()
            out.peak_live = max(out.peak_live, len(live))
            # Two detections in one frame are certainly different objects.
            present = [t for t in live if not t.coasting]
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    out.same_frame.append((a.id, b.id))
            for track in live:
                if track.coasting:
                    # A coasted frame breaks the "certainly one object" claim,
                    # because nothing was observed. Mark the track and keep it
                    # out of the within-track distances.
                    out.continuous[track.id] = False
                    continue
                out.hits[track.id] += 1
                out.classes[track.id].append(track.class_id)
                box = track.bbox
                out.boxes[track.id].append((box.center.x, box.center.y, box.height))
                index = min(range(len(detections)),
                            key=lambda k: -track.bbox.iou(detections[k].bbox), default=None) \
                    if detections else None
                if index is not None and looks[index].usable:
                    out.looks[track.id].append(looks[index])
    finally:
        if reader is not None:
            reader.stop()
        source.close()


def _percentiles(values: list[float]) -> str:
    if not values:
        return "no samples"
    a = np.asarray(values)
    return (f"n={len(a):<6} p05={np.percentile(a, 5):.3f} median={np.median(a):.3f} "
            f"p95={np.percentile(a, 95):.3f}")


def report(out: Observations) -> int:
    print(f"frames    {out.frames} ({out.unusable} unusable)")
    print(f"tracks    {len(out.hits)} distinct, {out.peak_live} live at once")
    if out.peak_live and len(out.hits) > out.peak_live * 3:
        print("          ^ far more ids than were ever live at once: fragmentation")
    print()

    # ------------------------------------------------ appearance separation
    print("APPEARANCE  (the truth here is structural: one object cannot be in two places)")
    usable = {tid: looks for tid, looks in out.looks.items()
              if out.hits[tid] >= MIN_TRACK_HITS and out.continuous[tid] and len(looks) >= 2}
    within: list[float] = []
    for looks in usable.values():
        # Every pair, not just consecutive ones: the gate has to hold across a
        # gap of seconds, which is when a person has turned round.
        for i, a in enumerate(looks):
            for b in looks[i + 1:]:
                d = a.distance(b)
                if np.isfinite(d):
                    within.append(d)
    between: list[float] = []
    for a_id, b_id in set(out.same_frame):
        for a in out.looks.get(a_id, [])[:6]:
            for b in out.looks.get(b_id, [])[:6]:
                d = a.distance(b)
                if np.isfinite(d):
                    between.append(d)
    print(f"  same object (continuous tracks, >={MIN_TRACK_HITS} hits)   {_percentiles(within)}")
    print(f"  different objects (same frame)                {_percentiles(between)}")
    if within and between:
        separation = np.percentile(between, 5) - np.percentile(within, 95)
        print(f"  separation p95(same) -> p05(different)        {separation:+.3f}")
        if separation <= 0:
            print("  ^ THE TWO DISTRIBUTIONS OVERLAP. No single threshold separates them on this")
            print("    footage, and any value chosen will both split one object and merge two.")
            print("    That is the argument for a learned descriptor, made from evidence.")
        else:
            suggested = (np.percentile(within, 95) + np.percentile(between, 5)) / 2
            print(f"  suggested MAX_REIDENTIFY_DISTANCE             {suggested:.2f}   "
                  f"(shipped: {MAX_REIDENTIFY_DISTANCE})")
            print(f"  suggested MAX_APPEARANCE_DISTANCE             "
                  f"{max(suggested, np.percentile(between, 25)):.2f}   "
                  f"(shipped: {MAX_APPEARANCE_DISTANCE}; looser on purpose — inside a frame "
                  f"geometry decides)")
    print()

    # ------------------------------------------------------- detector jitter
    print("DETECTOR BOX JITTER  (a track that is not moving is measuring the detector)")
    jitter: list[float] = []
    for tid, boxes in out.boxes.items():
        if out.hits[tid] < MIN_TRACK_HITS:
            continue
        arr = np.asarray(boxes)
        steps = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])) / np.maximum(arr[:-1, 2], 1e-6)
        still = steps[steps < STATIONARY_FRACTION]
        jitter.extend(still.tolist())
    if jitter:
        measured = float(np.std(jitter))
        print(f"  {_percentiles(jitter)}")
        print(f"  measured 1-sigma, as a fraction of box height  {measured:.4f}   "
              f"(assumed MEASURE_STD_FRACTION: {MEASURE_STD_FRACTION})")
        if measured > MEASURE_STD_FRACTION * 2:
            print("  ^ the filter is being told the detector is steadier than it is, so it will")
            print("    follow jitter. Raise MEASURE_STD_FRACTION.")
        elif measured < MEASURE_STD_FRACTION / 3:
            print("  ^ the filter is being told the detector is noisier than it is, so it will")
            print("    lag real motion. Lower MEASURE_STD_FRACTION.")
    else:
        print("  no stationary track long enough to measure. Point a camera at a parked car.")
    print()

    # ------------------------------------------------------ class stability
    print("CLASS STABILITY  (a confirmed track whose class flips is a model guessing)")
    flipped = sum(1 for c in out.classes.values() if len(set(c)) > 1)
    long_tracks = [tid for tid in out.classes if out.hits[tid] >= MIN_TRACK_HITS]
    if long_tracks:
        print(f"  {flipped} of {len(out.classes)} tracks changed class "
              f"({flipped / max(1, len(out.classes)):.0%})")
        print("  This is a *lower bound* on the model's error rate on this footage: a class that")
        print("  never flips can still be wrong every frame, but one that flips is wrong somewhere.")
    else:
        print("  no track long enough to judge")
    print()

    # ---------------------------------------------------- confidence shape
    print("CONFIDENCE  (a model in its own domain is bimodal; a flat one is guessing)")
    if out.confidences:
        a = np.asarray(out.confidences)
        print(f"  {_percentiles(out.confidences)}")
        counts, edges = np.histogram(a, bins=8, range=(0.0, 1.0))
        for count, low in zip(counts, edges):
            bar = "#" * int(40 * count / max(1, counts.max()))
            print(f"    {low:.2f}-{low + 0.125:.2f}  {count:>6}  {bar}")
        top = float((a > 0.8).mean())
        print(f"  share above 0.80: {top:.0%}")
        if top < 0.2:
            print("  ^ the model is rarely sure on this site's footage. That is the evidence-based")
            print("    argument for fine-tuning, and it is worth more than the suspicion.")
    else:
        print("  nothing was detected at all. Either the scene is empty or the model is wrong for it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", help="recorded clips, e.g. the ones `vigil run --record` writes")
    parser.add_argument("--source", help="a live source instead: device:0, or an RTSP address")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long, for a live source")
    args = parser.parse_args(argv)
    if not args.clips and not args.source:
        parser.error("give some clips, or --source")

    out = Observations()
    for clip in args.clips:
        print(f"reading {clip} …")
        observe(clip, None, out)
    if args.source:
        print(f"watching {args.source} for {args.seconds:.0f} s …")
        observe(args.source, args.seconds, out)
    print()
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
