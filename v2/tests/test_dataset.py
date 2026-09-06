"""Exporting the corpus the product has already written.

The claim under test is that a labelled dataset is nearer than it looks: the
clips, the events and — most valuably — the *reasons a person typed when
dismissing a false positive* are all already on disk, and what was missing was
the join.
"""

import json

import cv2
import numpy as np
import pytest

from vigil.service.dataset import DatasetError, Sample, collect, write
from vigil.storage.store import Store


def _clip(path, *, frames=60, fps=15.0, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for i in range(frames):
        coarse = rng.integers(40, 200, (size[1] // 5, size[0] // 5, 3), dtype=np.uint8)
        writer.write(cv2.resize(coarse, size, interpolation=cv2.INTER_CUBIC))
    writer.release()
    return path


def _seeded(tmp_path, *, days=2):
    """A store with clips and incidents across `days` separate days."""
    from datetime import datetime, timezone

    from vigil.adapters.recorder import Segment
    from vigil.domain.events import Severity
    from vigil.domain.incidents import Incident, Risk

    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path / "d.db")
    day_ms = 24 * 3600 * 1000
    base = 1_760_000_000_000
    incidents = []
    for day in range(days):
        started = base + day * day_ms
        path = _clip(tmp_path / f"clip-{day}.mp4")
        store.save_segment(Segment("gate", path, started, started + 4000, 60, 320, 240, 15.0,
                                   path.stat().st_size, "0" * 64))
        incidents.append(Incident(
            id=f"inc-{day}", severity=Severity.HIGH, summary="something",
            opened_at_millis=started + 500, closed_at_millis=started + 3000,
            opened_at=datetime.fromtimestamp((started + 500) / 1000, tz=timezone.utc),
            distinct_objects=1, cameras=("gate",), zones=("yard",), events=(),
            associations=(), risk=Risk(0.5, ()),
        ))
    store.save_incidents(incidents)
    return store


def test_frames_are_cut_from_the_clips_that_cover_the_incidents(tmp_path):
    store = _seeded(tmp_path)
    try:
        export = collect(store, None, frames_per_incident=2, spacing_s=1.0)
    finally:
        store.close()
    assert len(export.samples) == 4, export.describe()
    assert all(isinstance(s, Sample) for s in export.samples)
    assert {s.camera_id for s in export.samples} == {"gate"}
    assert all(s.image is not None for s in export.samples)
    assert len(export.days) == 2


def test_a_store_with_no_clips_says_so_rather_than_writing_an_empty_dataset(tmp_path):
    store = Store(tmp_path / "empty.db")
    try:
        with pytest.raises(DatasetError) as caught:
            collect(store, None)
        assert "record" in str(caught.value)
    finally:
        store.close()


def test_the_split_is_by_day_and_a_single_day_gets_no_validation_set(tmp_path):
    """A random split over consecutive frames leaks almost perfectly. There is
    no honest split of one day's footage, and inventing one is how a
    meaningless number comes to be believed."""
    store = _seeded(tmp_path, days=1)
    try:
        export = collect(store, None, frames_per_incident=2)
    finally:
        store.close()
    assert export.validation_days == ()
    assert "NO VALIDATION SPLIT" in export.describe()

    store = _seeded(tmp_path / "many", days=6)
    (tmp_path / "many").mkdir(exist_ok=True)
    try:
        export = collect(store, None, frames_per_incident=1)
    finally:
        store.close()
    assert export.validation_days, export.describe()
    assert not set(export.train_days) & set(export.validation_days), "a day is in one side only"
    assert set(export.train_days) | set(export.validation_days) == set(export.days)


def test_what_is_written_is_openable_and_says_the_boxes_are_not_labels(tmp_path):
    store = _seeded(tmp_path)
    try:
        export = collect(store, None, frames_per_incident=2)
    finally:
        store.close()
    destination = tmp_path / "out"
    manifest_path = write(export, destination, model_sha256="abc123",
                          class_names={0: "person", 2: "car"})
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["model_sha256"] == "abc123", (
        "a corrected set must never be confused about whose mistakes it was correcting"
    )
    assert "pre-label" in manifest["warning"] and "never a label" in manifest["warning"]
    assert "leaks" in manifest["split"]["note"]
    assert len(manifest["samples"]) == len(export.samples)

    images = sorted((destination / "images").glob("*.png"))
    labels = sorted((destination / "labels").glob("*.txt"))
    assert len(images) == len(export.samples)
    assert len(labels) == len(export.samples)
    for image in images:
        assert cv2.imread(str(image)) is not None, f"{image.name} is not a readable image"

    config = (destination / "data.yaml").read_text(encoding="utf-8")
    assert "by DAY" in config
    assert (destination / "train.txt").is_file() and (destination / "val.txt").is_file()


def test_a_dismissal_reason_reaches_the_manifest(tmp_path):
    """The most valuable field in the export: a sentence a person wrote about
    one of the system's own false positives. `review dismiss` has required one
    since the queue was built and nothing read it until now."""
    store = _seeded(tmp_path, days=1)
    try:
        store.set_incident_review("inc-0", "DISMISSED", by="alice", at=1_760_000_100_000,
                                  note="a delivery van, every Tuesday")
        export = collect(store, None, frames_per_incident=1)
    finally:
        store.close()
    assert export.samples
    assert export.samples[0].review_state == "DISMISSED"
    assert export.samples[0].review_note == "a delivery van, every Tuesday"
    assert "1 frame(s) come from incidents a person dismissed" in export.describe()

    manifest = json.loads(write(export, tmp_path / "o").read_text(encoding="utf-8"))
    assert manifest["samples"][0]["review_note"] == "a delivery van, every Tuesday"


def test_pre_labels_come_out_in_yolo_form_when_a_detector_is_given(tmp_path):
    from vigil.domain.detection import BoundingBox, Detection, DetectorInfo

    class _Stub:
        info = DetectorInfo("stub", "stub", classifies=True, class_names={0: "person"})

        def detect(self, image):
            return [Detection(BoundingBox(0.25, 0.5, 0.10, 0.20), 0.9, 0)]

    store = _seeded(tmp_path, days=1)
    try:
        export = collect(store, _Stub(), frames_per_incident=1)
    finally:
        store.close()
    destination = tmp_path / "labelled"
    write(export, destination, class_names={0: "person"})
    label = next((destination / "labels").glob("*.txt")).read_text(encoding="utf-8").strip()
    index, cx, cy, w, h = label.split()
    # YOLO is centre-based; the box above is x=0.25 w=0.10, so cx is 0.30.
    assert index == "0"
    assert abs(float(cx) - 0.30) < 1e-6 and abs(float(cy) - 0.60) < 1e-6
    assert abs(float(w) - 0.10) < 1e-6 and abs(float(h) - 0.20) < 1e-6
    assert "0: person" in (destination / "data.yaml").read_text(encoding="utf-8")
