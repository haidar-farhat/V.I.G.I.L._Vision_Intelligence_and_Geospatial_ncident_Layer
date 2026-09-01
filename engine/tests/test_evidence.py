"""Tests for evidence export.

An export is the point where the system's conclusions leave the machine and
become something somebody else has to trust. These tests are about the three
ways that trust is broken:

**By claiming more than was known.** A position that was never determined must
not appear as a coordinate; a motion that could not be measured must not appear
as zero. Both are written down as unknowns, in the machine-readable file and in
the human-readable one.

**By being alterable without trace.** Every file is hashed and every hash is
listed, and a package must be checkable by somebody who has only the folder.

**By writing where it was not asked to.** A path that escapes the export
directory is refused, not sanitised.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentinel.evidence import (
    APPLICATION_VERSION,
    EXPORT_FORMAT_VERSION,
    Export,
    ExportError,
    export_incident,
    sha256_of,
    verify_export,
)
from sentinel.incidents import Correlator
from test_store import make_event

EXPORTED_BY = "operator:alice"
MOMENT = datetime(2026, 8, 30, 9, 30, tzinfo=timezone.utc)


def incident_with(*, placed: bool = True, classifies: bool = False, tracks=(1, 2)):
    events = [
        make_event(track=t, at_millis=t * 900, placed=placed, classifies=classifies)
        for t in tracks
    ]
    return Correlator().correlate(events)[0]


@pytest.fixture
def exported(tmp_path: Path) -> Export:
    return export_incident(
        incident_with(), tmp_path / "exports", exported_by=EXPORTED_BY, at=MOMENT
    )


# --------------------------------------------------------------- what is written


def test_an_export_produces_a_folder_named_for_the_incident(exported: Export):
    assert exported.directory.is_dir()
    assert exported.directory.name == exported.incident_id
    assert (exported.directory / "manifest.json").is_file()
    assert (exported.directory / "incident.json").is_file()
    assert (exported.directory / "report.txt").is_file()


def test_the_package_carries_the_incident_in_full(exported: Export):
    data = json.loads((exported.directory / "incident.json").read_text(encoding="utf-8"))

    assert data["id"] == exported.incident_id
    assert data["events"], "an incident exported without its events cannot be reviewed"
    assert data["timeline"]
    assert data["risk"]["factors"], "a risk score with no reasoning is a number to ignore"
    assert data["distinct_object_count"] >= 1


def test_the_object_count_and_the_segment_list_are_both_present(exported: Export):
    # They answer different questions. Conflating them titles an incident
    # "6 people" when three walked past.
    data = json.loads((exported.directory / "incident.json").read_text(encoding="utf-8"))

    assert "distinct_object_count" in data
    assert "track_segments" in data
    assert len(data["track_segments"]) >= data["distinct_object_count"]


def test_every_event_carries_the_grounds_for_itself(exported: Export):
    data = json.loads((exported.directory / "incident.json").read_text(encoding="utf-8"))

    for event in data["events"]:
        evidence = event["evidence"]
        assert evidence["camera_id"]
        assert evidence["detector"]
        assert evidence["observations"] > 0
        assert event["triggering_conditions"]
        assert "confidence" in event


def test_there_is_a_file_a_person_can_read_without_tooling(exported: Export):
    # Somebody opening this in five years may have no parser at all, and a
    # folder whose only readable file needs one will not be read.
    report = (exported.directory / "report.txt").read_text(encoding="utf-8")

    assert exported.incident_id in report
    assert "TIMELINE" in report
    assert "EVIDENCE" in report
    assert "RISK" in report
    assert "LIMITS OF THIS EVIDENCE" in report


# --------------------------------------------------------- claiming no more


def test_an_undetermined_position_is_stated_rather_than_omitted(tmp_path: Path):
    export = export_incident(
        incident_with(placed=False), tmp_path, exported_by=EXPORTED_BY, at=MOMENT
    )

    data = json.loads((export.directory / "incident.json").read_text(encoding="utf-8"))
    assert data["events"][0]["evidence"]["position"] is None

    report = (export.directory / "report.txt").read_text(encoding="utf-8")
    assert "NOT DETERMINED" in report, (
        "an absent line reads as an oversight; the export must say the position "
        "was never knowable"
    )


def test_a_position_never_loses_its_uncertainty(exported: Export):
    data = json.loads((exported.directory / "incident.json").read_text(encoding="utf-8"))
    position = data["events"][0]["evidence"]["position"]

    assert position is not None
    assert position["uncertainty_meters"] is not None
    assert position["source"] == "GROUND_PROJECTION"


def test_an_unclassified_detection_is_never_named(tmp_path: Path):
    # A blob is not a person, and an export is exactly where that distinction
    # gets lost if nobody guards it.
    export = export_incident(
        incident_with(classifies=False), tmp_path, exported_by=EXPORTED_BY, at=MOMENT
    )
    report = (export.directory / "report.txt").read_text(encoding="utf-8")

    assert "does not classify" in report
    assert "unclassified" in json.loads(
        (export.directory / "incident.json").read_text(encoding="utf-8")
    )["events"][0]["evidence"]["class_label"]


def test_a_model_that_produced_a_conclusion_is_named_by_digest(tmp_path: Path):
    # "Which model said this" gets asked long after the model was replaced.
    export = export_incident(
        incident_with(classifies=True), tmp_path, exported_by=EXPORTED_BY, at=MOMENT
    )
    manifest = json.loads((export.directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["models"], "no model digest was recorded"
    assert len(manifest["models"][0]) == 64


def test_unknown_motion_is_not_reported_as_zero(tmp_path: Path):
    from dataclasses import replace

    base = make_event()
    event = replace(
        base, evidence=replace(base.evidence, speed_mps=None, heading_degrees=None)
    )
    incident = Correlator().correlate([event])[0]
    export = export_incident(incident, tmp_path, exported_by=EXPORTED_BY, at=MOMENT)

    data = json.loads((export.directory / "incident.json").read_text(encoding="utf-8"))
    assert data["events"][0]["evidence"]["motion"]["speed_mps"] is None

    report = (export.directory / "report.txt").read_text(encoding="utf-8")
    assert "UNKNOWN" in report


# -------------------------------------------------------------- chain of custody


def test_the_package_records_who_exported_it_and_when(exported: Export):
    manifest = json.loads((exported.directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["exported_by"] == EXPORTED_BY
    assert manifest["exported_at_utc"] == MOMENT.isoformat()
    assert manifest["application_version"] == APPLICATION_VERSION
    assert manifest["format_version"] == EXPORT_FORMAT_VERSION
    assert manifest["platform"]


def test_every_file_is_hashed_and_listed(exported: Export):
    manifest = json.loads((exported.directory / "manifest.json").read_text(encoding="utf-8"))
    listed = {entry["name"] for entry in manifest["files"]}

    on_disk = {p.name for p in exported.directory.iterdir()} - {"manifest.json"}
    assert listed == on_disk

    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0


def test_the_manifest_itself_is_hashed(exported: Export):
    # Recorded somewhere other than the package, this makes the whole package
    # checkable rather than just its contents against a manifest that could
    # itself have been rewritten.
    assert len(exported.manifest_sha256) == 64
    assert exported.manifest_sha256 == sha256_of(exported.directory / "manifest.json")


# -------------------------------------------------------------- tamper evidence


def test_an_untouched_package_verifies(exported: Export):
    assert verify_export(exported.directory) == []
    assert exported.verify() == []


def test_an_altered_file_is_detected(exported: Export):
    report = exported.directory / "report.txt"
    report.write_text(
        report.read_text(encoding="utf-8") + "\nand then nothing happened",
        encoding="utf-8",
    )

    problems = verify_export(exported.directory)
    assert any("report.txt" in problem for problem in problems)
    assert any("altered" in problem for problem in problems)


def test_a_removed_file_is_detected(exported: Export):
    (exported.directory / "incident.json").unlink()

    assert any("missing" in problem for problem in verify_export(exported.directory))


def test_a_file_added_afterwards_is_detected(exported: Export):
    # As much a problem as one that changed: it may have been planted, and the
    # manifest would not know.
    (exported.directory / "extra.txt").write_text("added later", encoding="utf-8")

    problems = verify_export(exported.directory)
    assert any("not listed" in problem for problem in problems)


def test_a_package_with_no_manifest_does_not_silently_pass(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    assert verify_export(empty) == ["manifest.json: missing"]


def test_a_corrupt_manifest_does_not_silently_pass(exported: Export):
    (exported.directory / "manifest.json").write_text("{not json", encoding="utf-8")

    problems = verify_export(exported.directory)
    assert problems and "unreadable" in problems[0]


def test_verification_needs_only_the_folder(exported: Export, tmp_path: Path):
    # A package has to be checkable by somebody who has no access to the system
    # that made it.
    import shutil

    moved = tmp_path / "elsewhere" / exported.directory.name
    moved.parent.mkdir(parents=True)
    shutil.copytree(exported.directory, moved)

    assert verify_export(moved) == []


# ------------------------------------------------------------------ sandboxing


def test_an_attachment_that_escapes_the_package_is_refused(tmp_path: Path, monkeypatch):
    # An export that writes outside where it was asked to is a way to overwrite
    # something that mattered.
    from sentinel.evidence import _resolve_within

    with pytest.raises(ExportError, match="Refusing to write outside"):
        _resolve_within(tmp_path, "../escaped.txt")

    with pytest.raises(ExportError, match="Refusing to write outside"):
        _resolve_within(tmp_path, "a/../../escaped.txt")


def test_a_missing_attachment_fails_loudly(tmp_path: Path):
    with pytest.raises(ExportError, match="Attachment not found"):
        export_incident(
            incident_with(),
            tmp_path,
            exported_by=EXPORTED_BY,
            attachments=[tmp_path / "absent.mp4"],
        )


def test_an_attachment_is_copied_in_and_hashed(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video, but a real file")

    export = export_incident(
        incident_with(), tmp_path / "exports", exported_by=EXPORTED_BY,
        attachments=[clip], at=MOMENT,
    )

    assert (export.directory / "clip.mp4").is_file()
    names = {entry.name for entry in export.files}
    assert "clip.mp4" in names
    assert verify_export(export.directory) == []


# ------------------------------------------------------------------ determinism


def test_exporting_the_same_incident_twice_produces_the_same_content(tmp_path: Path):
    # The timestamp is supplied, so everything else must be reproducible — an
    # export whose bytes change for no reason cannot be compared against an
    # earlier copy.
    incident = incident_with()

    first = export_incident(incident, tmp_path / "a", exported_by=EXPORTED_BY, at=MOMENT)
    second = export_incident(incident, tmp_path / "b", exported_by=EXPORTED_BY, at=MOMENT)

    assert [f.sha256 for f in first.files] == [f.sha256 for f in second.files]
    assert first.manifest_sha256 == second.manifest_sha256


# ------------------------------------------------------- attachments collide


def test_two_attachments_with_the_same_name_both_survive(tmp_path: Path):
    """Two clips from two cameras are routinely both called clip.mp4.

    Copying the second over the first loses evidence *and still verifies clean*,
    because the manifest is written afterwards from whatever survived — the
    worst possible failure for a package whose purpose is to be trustworthy.
    """
    first_dir = tmp_path / "cam-07"
    second_dir = tmp_path / "cam-08"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "clip.mp4").write_bytes(b"footage from camera seven")
    (second_dir / "clip.mp4").write_bytes(b"footage from camera eight")

    export = export_incident(
        incident_with(),
        tmp_path / "exports",
        exported_by=EXPORTED_BY,
        attachments=[first_dir / "clip.mp4", second_dir / "clip.mp4"],
        at=MOMENT,
    )

    names = {entry.name for entry in export.files}
    assert "clip.mp4" in names
    assert len([n for n in names if n.startswith("clip")]) == 2, (
        "one attachment silently overwrote the other"
    )

    contents = {(export.directory / n).read_bytes() for n in names if n.startswith("clip")}
    assert len(contents) == 2, "both attachments are present but hold the same bytes"
    assert verify_export(export.directory) == []


def test_an_attachment_cannot_overwrite_the_record_itself(tmp_path: Path):
    # An attachment named incident.json would have destroyed the very thing the
    # package exists to carry, and the manifest would have hashed the imposter.
    source = tmp_path / "incident.json"
    source.write_text('{"not": "the real record"}', encoding="utf-8")

    export = export_incident(
        incident_with(), tmp_path / "exports", exported_by=EXPORTED_BY,
        attachments=[source], at=MOMENT,
    )

    real = json.loads((export.directory / "incident.json").read_text(encoding="utf-8"))
    assert real["id"] == export.incident_id, "the attachment overwrote the record"
    assert verify_export(export.directory) == []
