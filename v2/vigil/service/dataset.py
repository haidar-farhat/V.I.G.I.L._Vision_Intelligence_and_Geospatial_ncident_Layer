"""The corpus this product has been writing all along, made readable.

# Why this exists

`PRODUCTION_READINESS.md` says there is no dataset. There is no *labelled*
dataset, which is a different and much smaller problem than it sounds,
because running this system already produces almost all of one:

- **Clips.** Timestamped MP4 segments with a SHA-256 digest, indexed in the
  store with the camera, the frame count and the exact millisecond they
  start and end.
- **Events.** Every conclusion, with the camera, the zone, the track, the
  rule, the confidence and the moment it happened.
- **Judgements.** `review dismiss` **refuses without a reason**. So every
  dismissal is a sentence a person wrote about one of the system's own false
  positives — a hand-authored label on the error that matters most — and
  until this module nothing read them.

What was missing was the join. This is the join: it walks the incidents, finds
the clip covering each one, cuts the frame out of it, runs the detector to
produce a *pre-label*, and writes the lot in a form a labelling tool opens.

# What it does not do

It does not label anything. The boxes it writes are the detector's own
opinion, which is exactly what has to be checked rather than trusted —
correcting pre-labels is three to five times faster than drawing from
scratch, and that is the whole reason to emit them. Every image is written
with the model's digest beside it, so a corrected set can never be confused
about which model's mistakes it was correcting.

# The split, and why it is by day

Consecutive frames of video are near-duplicates. A random train/validation
split over them leaks almost perfectly: the model sees frame 100 in training
and is tested on frame 101, scores 0.98, and means nothing by it. So the split
here is **by day**, whole days held out, and it refuses to produce a
validation set at all when everything came from one day — because there is no
honest split of one day's footage, and silently making one is how a
meaningless number gets believed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Frames taken from around each incident. One frame at the moment of an event
#: is one sample of the hardest instant; a few either side catch the approach
#: and the aftermath, which is where a detector actually fails.
DEFAULT_FRAMES_PER_INCIDENT = 3

#: Seconds between those frames.
DEFAULT_SPACING_S = 1.0


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Sample:
    """One exported frame and everything known about it."""

    image_path: Path
    camera_id: str
    at_millis: int
    day: str
    incident_id: str | None
    #: NEW, ACKNOWLEDGED or DISMISSED, when the frame came from an incident.
    review_state: str | None
    #: What a person wrote when they dismissed it. The most valuable field
    #: here: a hand-written account of a false positive.
    review_note: str | None
    #: `(class_id, label, x, y, w, h)` in normalised coordinates, from the
    #: detector. An opinion to be corrected, never a label.
    predictions: tuple = ()
    clip_sha256: str | None = None
    #: The decoded frame, carried so `write` does not seek the clip a second
    #: time. Excluded from equality: an array does not compare to a bool, and
    #: two samples are the same sample when they name the same moment.
    image: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Export:
    samples: tuple[Sample, ...]
    train_days: tuple[str, ...]
    validation_days: tuple[str, ...]
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def days(self) -> tuple[str, ...]:
        return tuple(sorted({s.day for s in self.samples}))

    def describe(self) -> str:
        if not self.samples:
            return "nothing to export: no incident had a clip covering it"
        dismissed = sum(1 for s in self.samples if s.review_state == "DISMISSED")
        boxes = sum(len(s.predictions) for s in self.samples)
        lines = [
            f"{len(self.samples)} frame(s) from {len(self.days)} day(s) across "
            f"{len({s.camera_id for s in self.samples})} camera(s)",
            f"{boxes} pre-label(s) from the detector — opinions to correct, not labels",
            f"{dismissed} frame(s) come from incidents a person dismissed, with their reason",
        ]
        if self.validation_days:
            lines.append(f"split by day: train {', '.join(self.train_days)} | "
                         f"validate {', '.join(self.validation_days)}")
        else:
            lines.append("NO VALIDATION SPLIT: everything came from one day. A random split over "
                         "consecutive frames leaks almost perfectly, so none was invented. "
                         "Record on more days.")
        for reason, count in sorted(self.skipped.items()):
            lines.append(f"skipped {count}: {reason}")
        return "\n".join(lines)


def _day_of(at_millis: int) -> str:
    return datetime.fromtimestamp(at_millis / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _clip_covering(segments: Sequence, camera_id: str, at_millis: int):
    for segment in segments:
        if segment.camera_id != camera_id:
            continue
        if segment.started_millis <= at_millis <= segment.ended_millis:
            return segment
    return None


def _frame_at(clip, at_millis: int):
    """The frame of `clip` nearest `at_millis`, or `None`.

    By seeking on the millisecond rather than by counting frames: a recorder
    that dropped frames makes the two disagree, and the timestamp is what the
    event was recorded against.
    """
    import cv2

    path = Path(clip.path)
    if not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        offset = max(0, at_millis - int(clip.started_millis))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(offset))
        ok, image = capture.read()
        return image if ok else None
    finally:
        capture.release()


def collect(store, detector=None, *, since_millis: int | None = None, limit: int = 200,
            frames_per_incident: int = DEFAULT_FRAMES_PER_INCIDENT,
            spacing_s: float = DEFAULT_SPACING_S) -> Export:
    """Walk the incidents and cut out the frames they were drawn from."""
    segments = store.segments()
    if not segments:
        raise DatasetError(
            "no clip has been recorded, so there are no frames to export. Run with `--record`: "
            "the corpus is a by-product of the product doing its job."
        )
    incidents = store.incidents(limit=limit, since=since_millis)
    samples: list[Sample] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for incident in incidents:
        cameras = list(getattr(incident, "cameras", ()) or ())
        if not cameras:
            skip("the incident names no camera")
            continue
        camera_id = cameras[0]
        opened = int(incident.opened_at_millis)
        review = getattr(incident, "review", None)
        for step in range(max(1, frames_per_incident)):
            at = opened + int(step * spacing_s * 1000)
            clip = _clip_covering(segments, camera_id, at)
            if clip is None:
                skip("no clip covers the moment")
                continue
            image = _frame_at(clip, at)
            if image is None:
                skip("the clip would not yield that frame")
                continue
            predictions = ()
            if detector is not None:
                info = detector.info
                predictions = tuple(
                    (d.class_id, info.label_for(d.class_id) or f"class_{d.class_id}",
                     round(d.bbox.x, 6), round(d.bbox.y, 6),
                     round(d.bbox.width, 6), round(d.bbox.height, 6), round(d.confidence, 4))
                    for d in detector.detect(image)
                )
            samples.append(Sample(
                image_path=Path(f"{camera_id}-{at}.png"), camera_id=camera_id, at_millis=at,
                day=_day_of(at), incident_id=getattr(incident, "id", None),
                review_state=str(review.state) if review is not None else None,
                review_note=review.note if review is not None else None,
                predictions=predictions, clip_sha256=clip.sha256, image=image,
            ))

    days = sorted({s.day for s in samples})
    # One day in five held out, at least one, and none at all from a single
    # day — see the module docstring for why inventing one would be worse
    # than having none.
    if len(days) < 2:
        train, validate = tuple(days), ()
    else:
        held = max(1, len(days) // 5)
        validate = tuple(days[-held:])
        train = tuple(days[:-held])
    return Export(tuple(samples), train, validate, skipped)


def write(export: Export, destination: Path, *, model_sha256: str | None = None,
          class_names: dict[int, str] | None = None) -> Path:
    """Write images, YOLO pre-labels and a manifest.

    YOLO format because every offline labelling tool reads it and it is one
    line per box; the manifest carries everything YOLO has no room for — the
    incident, the human judgement, the clip's digest and the day.
    """
    import cv2

    destination.mkdir(parents=True, exist_ok=True)
    images = destination / "images"
    labels = destination / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)

    written: list[dict] = []
    order: dict[int, int] = {}
    for sample in export.samples:
        image = sample.image
        if image is None:
            continue
        target = images / sample.image_path.name
        if not cv2.imwrite(str(target), image):
            continue
        lines = []
        for class_id, label, x, y, w, h, confidence in sample.predictions:
            index = order.setdefault(int(class_id), len(order))
            lines.append(f"{index} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        (labels / f"{sample.image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written.append({
            "image": f"images/{sample.image_path.name}",
            "camera": sample.camera_id,
            "at_millis": sample.at_millis,
            "day": sample.day,
            "incident": sample.incident_id,
            "review_state": sample.review_state,
            "review_note": sample.review_note,
            "clip_sha256": sample.clip_sha256,
            "predictions": [
                {"class_id": c, "label": l, "x": x, "y": y, "w": w, "h": h, "confidence": conf}
                for c, l, x, y, w, h, conf in sample.predictions
            ],
        })
    names = {index: (class_names or {}).get(class_id, f"class_{class_id}")
             for class_id, index in sorted(order.items(), key=lambda kv: kv[1])}
    manifest = {
        "generated_by": "vigil dataset export",
        "model_sha256": model_sha256,
        "warning": ("Every box here is the detector's own output. It is a pre-label to be "
                    "corrected, never a label. Correcting is 3-5x faster than drawing from "
                    "scratch, which is the only reason to emit them."),
        "split": {"train_days": list(export.train_days),
                  "validation_days": list(export.validation_days),
                  "note": ("Split by day. Consecutive frames of video are near-duplicates and a "
                           "random split over them leaks almost perfectly.")},
        "classes": names,
        "samples": written,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_yolo_config(destination, export, names)
    _log.info("dataset written to %s: %d frame(s)", destination, len(written))
    return destination / "manifest.json"


def _write_yolo_config(destination: Path, export: Export, names: dict[int, str]) -> None:
    """A `data.yaml` a trainer reads, with the day split turned into file lists."""
    train = destination / "train.txt"
    validate = destination / "val.txt"
    train.write_text("\n".join(
        f"images/{s.image_path.name}" for s in export.samples if s.day in export.train_days
    ), encoding="utf-8")
    validate.write_text("\n".join(
        f"images/{s.image_path.name}" for s in export.samples if s.day in export.validation_days
    ), encoding="utf-8")
    lines = [
        "# Generated by `vigil dataset export`. The split is by DAY:",
        "# a random split over consecutive video frames leaks almost perfectly.",
        f"path: {destination.as_posix()}",
        "train: train.txt",
        "val: val.txt" if export.validation_days else "# val: none — everything came from one day",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in sorted(names.items()))
    (destination / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
