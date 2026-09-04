#!/usr/bin/env python3
"""Measure how badly the tracker fragments, before anyone tries to fix it.

Driving the console on the laptop camera counted 3, 10, 4 and 11 "distinct
objects" over twenty seconds of one person and some furniture. Four numbers,
four runs, no control — so nobody can say whether that is the tracker, the
detector, the camera's auto-exposure, or the furniture. This tool produces the
number under controlled conditions: the synthetic reference scene has a known
object count, so **fragmentation = distinct track ids / true objects** is a
measurement rather than an impression, and the segments each object was
reported as are listed so the *kind* of fragmentation is visible too.

Two kinds matter, and they need different fixes:

- A **temporal fragment** is a track that dies and is reborn with a new id
  when the detector loses the object for longer than the tracker's gap
  budget. `sentinel.reid` can reconcile these after the fact, and this tool
  reports how many it did.
- An **identity swap** is a track that follows one object and then another —
  two people crossing. No post-hoc linking can fix that, only a tracker that
  knows what its objects look like (ABI 7). The per-object listing shows a
  track under two objects when it swapped.

The reference scene is the control; ``--device N --for S`` runs the same
measurement on a live camera, where the true count is whatever the operator
states with ``--objects``, or unknown. Nothing here downloads anything: a
model, if wanted, is supplied by path.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "engine" / "tests")]

#: The pose `engine/tests/conftest.py` gives the reference camera, restated
#: here so the tool measures the same run the suite does.
REFERENCE_POSE = dict(
    mount_height=6.0, heading=180.0, pitch=-22.0,
    horizontal_fov=62.0, vertical_fov=36.0, range_meters=90.0,
)
REFERENCE_SITE = (33.8938, 35.5018)

#: Frames where two true objects are closer than this, in pixels, are left out
#: of the attribution: nearest-centre cannot say which one a box belongs to,
#: and counting a guess would measure the metric's confusion, not the tracker's.
AMBIGUOUS_SEPARATION_PIXELS = 50


@dataclass(frozen=True, slots=True)
class Segment:
    """One track id's share of one true object."""

    track_id: int
    frames: int


@dataclass
class FragmentationReport:
    source: str
    frames: int
    detections: int
    track_ids: tuple[int, ...]
    #: ``None`` on a live camera the operator did not state a count for.
    true_objects: int | None
    #: Per true object, the track ids attributed to it and for how many frames.
    #: Empty for a live run: there is no ground truth to attribute against.
    segments: dict[str, tuple[Segment, ...]] = field(default_factory=dict)
    #: Track ids attributed to more than one object: identity swaps.
    swapped: tuple[int, ...] = ()
    #: After `sentinel.reid` linked temporal fragments.
    groups: tuple[str, ...] = ()
    linked: int = 0

    @property
    def distinct_tracks(self) -> int:
        return len(self.track_ids)

    @property
    def ratio(self) -> float | None:
        if not self.true_objects:
            return None
        return self.distinct_tracks / self.true_objects

    @property
    def ratio_after_linking(self) -> float | None:
        if not self.true_objects:
            return None
        return len(self.groups) / self.true_objects


def _attribute(results, scene_module) -> tuple[dict[str, tuple[Segment, ...]], tuple[int, ...]]:
    """Which true object each track followed, frame by frame.

    Nearest-centre, on unambiguous frames only — the same rule
    `test_pipeline.py` uses, so this tool and the suite cannot disagree about
    what a swap is.
    """
    frames_on: dict[int, Counter] = {}
    for result in results:
        truth = scene_module.ground_truth(result.index)
        centres = {
            name: (box[0] + box[2] / 2, box[1] + box[3] / 2) for name, box in truth.items()
        }
        names = list(centres)
        closest = min(
            (
                ((centres[a][0] - centres[b][0]) ** 2 + (centres[a][1] - centres[b][1]) ** 2) ** 0.5
                for i, a in enumerate(names)
                for b in names[i + 1:]
            ),
            default=float("inf"),
        )
        if closest < AMBIGUOUS_SEPARATION_PIXELS:
            continue
        for track in result.tracks:
            cx = (track.bbox.x + track.bbox.w / 2) * scene_module.WIDTH
            cy = (track.bbox.y + track.bbox.h / 2) * scene_module.HEIGHT
            best, distance = None, float("inf")
            for name, (gx, gy) in centres.items():
                d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                if d < distance:
                    best, distance = name, d
            if best is not None and distance < AMBIGUOUS_SEPARATION_PIXELS:
                frames_on.setdefault(track.id, Counter())[best] += 1

    segments: dict[str, list[Segment]] = {walker.name: [] for walker in scene_module.WALKERS}
    swapped = []
    for track_id in sorted(frames_on):
        counts = frames_on[track_id]
        if len(counts) > 1:
            swapped.append(track_id)
        for name, frames in counts.items():
            segments.setdefault(name, []).append(Segment(track_id, frames))
    return (
        {name: tuple(sorted(items, key=lambda s: s.track_id)) for name, items in segments.items()},
        tuple(swapped),
    )


def _track_ids(results) -> tuple[int, ...]:
    """Every id that ever appeared in a result.

    From the results rather than ``PipelineStats.track_ids``, which is a
    bounded reporting window: on a long live run it forgets the oldest ids,
    and a count that forgets is not a fragmentation measurement.
    """
    return tuple(sorted({track.id for result in results for track in result.tracks}))


def _link(results) -> tuple[tuple[str, ...], int]:
    from sentinel.reid import AppearanceLedger

    ledger = AppearanceLedger()
    for result in results:
        ledger.observe_result(result)
    groups = ledger.link()
    return (
        tuple(group.describe() for group in groups),
        sum(len(group.links) for group in groups),
    )


def measure_reference(video: Path | None = None) -> FragmentationReport:
    """The controlled measurement: the reference scene, reference pose, motion detector."""
    import scene
    from sentinel.core import CameraPose, LatLon
    from sentinel.decode import VideoSource
    from sentinel.detect import MotionDetector
    from sentinel.pipeline import Pipeline

    if video is None:
        video = Path(tempfile.mkdtemp(prefix="sentinel-fragmentation-")) / "reference.mp4"
        scene.write_scene(video)

    pose = CameraPose(position=LatLon(*REFERENCE_SITE), **REFERENCE_POSE)
    # Images kept so the appearance ledger has pixels to describe; the
    # measurement of the tracker itself does not need them.
    with Pipeline(
        VideoSource(video, source_id="reference"), MotionDetector(), pose=pose, keep_images=True
    ) as pipeline:
        results = list(pipeline.run())
        stats = pipeline.stats

    segments, swapped = _attribute(results, scene)
    groups, linked = _link(results)
    return FragmentationReport(
        source=f"reference scene ({video.name}, {len(scene.WALKERS)} walkers)",
        frames=stats.frames,
        detections=stats.detections,
        track_ids=_track_ids(results),
        true_objects=len(scene.WALKERS),
        segments=segments,
        swapped=swapped,
        groups=groups,
        linked=linked,
    )


def measure_live(
    index: int, seconds: float, *, objects: int | None, model: Path | None
) -> FragmentationReport | None:
    """The same measurement on a camera; ``None`` when no camera opened.

    Probing first, because `Pipeline` on a device that never opens retries
    with backoff and a tool that hangs on a missing webcam is a tool nobody
    runs twice.
    """
    from sentinel import devices
    from sentinel.decode import DecodeError, VideoSource
    from sentinel.detect import detector_for
    from sentinel.pipeline import Pipeline

    if devices.probe(index) is None:
        return None

    detector = detector_for(model) if model is not None else detector_for()
    deadline = time.monotonic() + seconds
    results = []
    try:
        with Pipeline(
            VideoSource(f"device:{index}", source_id=f"device-{index}"),
            detector, keep_images=True,
        ) as pipeline:
            for result in pipeline.run():
                results.append(result)
                if time.monotonic() >= deadline:
                    pipeline.ask_to_stop()
            stats = pipeline.stats
    except DecodeError as error:
        print(f"camera {index}: {error}", file=sys.stderr)
        return None

    groups, linked = _link(results)
    return FragmentationReport(
        source=f"camera {index} for {seconds:.0f} s ({detector.info.name})",
        frames=stats.frames,
        detections=stats.detections,
        track_ids=_track_ids(results),
        true_objects=objects,
        groups=groups,
        linked=linked,
    )


def render(report: FragmentationReport) -> str:
    lines = [
        f"fragmentation: {report.source}",
        f"  frames            {report.frames}",
        f"  detections        {report.detections}",
        f"  distinct tracks   {report.distinct_tracks}  "
        f"({', '.join(f'#{t}' for t in report.track_ids) or 'none'})",
    ]
    if report.true_objects is None:
        lines.append("  true objects      unknown (state them with --objects N)")
    else:
        lines.append(f"  true objects      {report.true_objects}")
        lines.append(f"  tracks per object {report.ratio:.2f}")

    if report.segments:
        lines.append("  segments per object:")
        for name, segments in report.segments.items():
            listed = ", ".join(f"#{s.track_id} ({s.frames} frames)" for s in segments) or "never tracked"
            lines.append(f"    {name:<12} {listed}")
        if report.swapped:
            lines.append(
                "  identity swaps    "
                + ", ".join(f"#{t}" for t in report.swapped)
                + "  (one track, two objects: no post-hoc link can fix this)"
            )

    lines.append(f"  after linking     {len(report.groups)} object(s), {report.linked} link(s)")
    for group in report.groups:
        lines.append(f"    {group}")
    if report.ratio_after_linking is not None:
        lines.append(f"  tracks per object {report.ratio_after_linking:.2f} after linking")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", type=int, default=None, help="camera index for a live run")
    parser.add_argument("--for", dest="seconds", type=float, default=20.0,
                        help="seconds to watch the camera (default 20)")
    parser.add_argument("--objects", type=int, default=None,
                        help="how many objects were actually in front of the camera")
    parser.add_argument("--model", type=Path, default=None,
                        help="an operator-supplied .onnx to detect with instead of motion")
    args = parser.parse_args(argv)

    from sentinel import logs

    logs.configure(level="WARNING", file="")

    print(render(measure_reference()))

    if args.device is not None:
        print()
        report = measure_live(
            args.device, args.seconds, objects=args.objects, model=args.model
        )
        if report is None:
            print(f"camera {args.device}: nothing opened, live measurement skipped")
        else:
            print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
