"""Evidence export.

An incident that cannot leave the machine is not evidence, it is a log entry.
This module produces a package somebody else can open — an investigator, a
manager, a court — and satisfy themselves that it says what it said when it was
made.

Four things the package has to carry, and this module exists to make sure it
carries all four:

**What the system concluded**, in full, including the reasoning: which rule, on
what grounds, with what confidence, and which risk factors produced the score.

**What it concluded that on.** Every event's evidence bundle: which camera, which
track, how many observations, which detector, which model by digest, where and
how well that was known.

**What it could not establish.** An export that quietly omits the uncertainty, or
the fact that a position was never determined, misrepresents the system as more
certain than it was. Unknowns are written down as unknowns.

**Whether it has been altered since.** Every file is hashed, the manifest lists
the hashes, and the manifest itself is hashed. That does not make tampering
impossible — nothing short of a signature does — but it makes silent tampering
impossible, which is the property that matters when the alternative is a folder
of files nobody can vouch for.

The export directory is sandboxed. A destination outside it is refused rather
than clamped, because an export that writes where it was not asked to is a way to
overwrite something that mattered.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .events import Event
from .incidents import Incident

#: Bumped when the package layout changes in a way a reader must know about.
EXPORT_FORMAT_VERSION = 1

#: Bumped when the *system* changes in a way that affects what it concludes.
#: Recorded in every export, because "which build said this" is a question that
#: gets asked long after the build has been replaced.
APPLICATION_VERSION = "0.1.0"


class ExportError(RuntimeError):
    """The export could not be written, or was refused."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(root: Path, name: str) -> Path:
    """A path inside ``root``, or a refusal.

    Refused rather than sanitised. A name that escapes the export directory is
    either a bug or an attack, and quietly rewriting it into something safe
    hides both — the caller ends up with a file somewhere it did not intend and
    no indication that anything was wrong.
    """
    root = root.resolve()
    candidate = (root / name).resolve()

    if root not in candidate.parents and candidate != root:
        raise ExportError(
            f"Refusing to write outside the export directory: {name!r} resolves "
            f"to {candidate}, which is not under {root}."
        )
    return candidate


@dataclass(frozen=True, slots=True)
class ExportedFile:
    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Export:
    """A written evidence package."""

    directory: Path
    incident_id: str
    files: tuple[ExportedFile, ...]
    #: SHA-256 of the manifest itself. Record this separately — somewhere other
    #: than the package — and the whole package becomes checkable against it.
    manifest_sha256: str
    exported_at: datetime
    exported_by: str

    def verify(self) -> list[str]:
        """Re-hash everything. Returns the names of files that no longer match.

        The reason to write hashes down at all: a package nobody can check is
        not better than one with no hashes, it is only more convincing.
        """
        problems: list[str] = []
        for entry in self.files:
            path = self.directory / entry.name
            if not path.exists():
                problems.append(f"{entry.name}: missing")
            elif sha256_of(path) != entry.sha256:
                problems.append(f"{entry.name}: altered since export")
        return problems


def _event_dict(event: Event) -> dict:
    """An event as plain data, losing nothing and inventing nothing.

    ``None`` stays ``None``. A position that was never determined must not
    become a zero, which reads as a coordinate off the west coast of Africa and
    is indistinguishable from a measurement.
    """
    evidence = event.evidence
    return {
        "id": event.id,
        "type": event.type.value,
        "severity": event.severity.value,
        "summary": event.summary,
        "rule": event.rule_id,
        "zone": {"id": event.zone_id, "name": event.zone_name},
        "occurred_at_media_millis": event.occurred_at_millis,
        "occurred_at_utc": event.occurred_at.isoformat(),
        "confidence": event.confidence,
        "triggering_conditions": list(event.triggering_conditions),
        "evidence": {
            "camera_id": evidence.camera_id,
            "track_id": evidence.track_id,
            "first_seen_millis": evidence.first_seen_millis,
            "last_seen_millis": evidence.last_seen_millis,
            "observations": evidence.observations,
            "detector": evidence.detector,
            "detector_classifies": evidence.detector_classifies,
            "model_digest": evidence.model_digest,
            "class_label": evidence.class_label,
            "position": (
                {
                    "latitude": evidence.latitude,
                    "longitude": evidence.longitude,
                    "uncertainty_meters": evidence.position_uncertainty_meters,
                    "source": evidence.position_source,
                }
                if evidence.latitude is not None
                else None
            ),
            "motion": {
                # Three states preserved: null means unknown, 0.0 means still.
                "speed_mps": evidence.speed_mps,
                "heading_degrees": evidence.heading_degrees,
            },
            "frame_indices": list(evidence.frame_indices),
        },
    }


def _incident_dict(incident: Incident) -> dict:
    return {
        "id": incident.id,
        "severity": incident.severity.value,
        "summary": incident.summary,
        "opened_at_media_millis": incident.opened_at_millis,
        "closed_at_media_millis": incident.closed_at_millis,
        "opened_at_utc": incident.opened_at.isoformat(),
        "duration_millis": incident.duration_millis,
        # Distinct objects, not track segments. The two answer different
        # questions and conflating them titles an incident "6 people" when three
        # walked past.
        "distinct_object_count": incident.distinct_objects,
        "track_segments": sorted(
            {
                f"{e.evidence.camera_id}#{e.evidence.track_id}"
                for e in incident.events
            }
        ),
        "cameras": list(incident.cameras),
        "zones": list(incident.zones),
        "risk": {
            "score": incident.risk.score,
            "band": incident.risk.band.value,
            "factors": [
                {"name": f.name, "points": f.points, "because": f.because}
                for f in incident.risk.factors
            ],
        },
        "cross_camera_associations": [
            {
                "a": f"{link.a[0]}#{link.a[1]}",
                "b": f"{link.b[0]}#{link.b[1]}",
                "score": link.score,
                "separation_meters": link.separation_meters,
                "allowance_meters": link.allowance_meters,
                "time_gap_millis": link.time_gap_millis,
                "reasons": list(link.reasons),
            }
            for link in incident.associations
        ],
        "timeline": [
            {
                "at_media_millis": entry.at_millis,
                "at_utc": entry.at.isoformat(),
                "camera_id": entry.camera_id,
                "severity": entry.severity.value,
                "summary": entry.summary,
                "event_id": entry.event_id,
            }
            for entry in incident.timeline()
        ],
        "events": [_event_dict(event) for event in incident.events],
    }


def _readable_report(incident: Incident, exported_by: str, at: datetime) -> str:
    """The same content as a person would read it.

    Present alongside the JSON, not instead of it. Somebody opening this package
    in five years may have no tooling at all, and a folder whose only readable
    file requires a parser is a folder that will not be read.
    """
    lines = [
        "SENTINEL VISION — INCIDENT EVIDENCE PACKAGE",
        "=" * 70,
        "",
        f"Incident        {incident.id}",
        f"Severity        {incident.severity.value}",
        f"Summary         {incident.summary}",
        f"Opened          {incident.opened_at:%Y-%m-%d %H:%M:%S} UTC",
        f"Duration        {incident.duration_millis / 1000:.1f} s",
        f"Distinct objects{incident.distinct_objects:>4}",
        f"Cameras         {', '.join(incident.cameras)}",
        f"Zones           {', '.join(incident.zones) if incident.zones else '—'}",
        "",
        "RISK",
        "-" * 70,
    ]
    lines.append(f"  {incident.risk.score:.0f}/100 ({incident.risk.band.value})")
    for factor in incident.risk.factors:
        lines.append(f"  {factor.points:+6.0f}  {factor.name}: {factor.because}")

    if incident.associations:
        lines += ["", "CROSS-CAMERA ASSOCIATIONS", "-" * 70]
        for link in incident.associations:
            lines.append(
                f"  {link.a[0]}#{link.a[1]} = {link.b[0]}#{link.b[1]}  "
                f"(score {link.score:.2f})"
            )
            for reason in link.reasons:
                lines.append(f"      {reason}")

    lines += ["", "TIMELINE", "-" * 70]
    for entry in incident.timeline():
        lines.append(
            f"  t+{entry.at_millis / 1000:7.1f}s  [{entry.severity.value:8}]  "
            f"{entry.camera_id}  {entry.summary}"
        )

    lines += ["", "EVIDENCE", "-" * 70]
    for event in incident.events:
        evidence = event.evidence
        lines.append(f"  {event.id}")
        lines.append(f"    rule          {event.rule_id}")
        lines.append(f"    because       {'; '.join(event.triggering_conditions)}")
        lines.append(f"    confidence    {event.confidence:.2f}")
        lines.append(f"    camera        {evidence.camera_id}, track #{evidence.track_id}")
        lines.append(f"    observations  {evidence.observations}")
        lines.append(
            f"    detector      {evidence.detector}"
            + ("" if evidence.detector_classifies else " (does not classify)")
        )
        if evidence.model_digest:
            lines.append(f"    model         sha256:{evidence.model_digest}")
        if evidence.latitude is not None:
            lines.append(
                f"    position      {evidence.latitude:.6f}, {evidence.longitude:.6f} "
                f"± {evidence.position_uncertainty_meters:.1f} m "
                f"({evidence.position_source})"
            )
        else:
            # Written down rather than omitted. An absent line reads as an
            # oversight; this reads as a fact about what was knowable.
            lines.append("    position      NOT DETERMINED — the camera was not placed")

        if evidence.speed_mps is None:
            lines.append("    motion        UNKNOWN — too little observation to measure")
        elif evidence.speed_mps < 0.3:
            lines.append("    motion        stationary")
        else:
            heading = (
                f", heading {evidence.heading_degrees:.0f}°"
                if evidence.heading_degrees is not None
                else ""
            )
            lines.append(f"    motion        {evidence.speed_mps:.2f} m/s{heading}")
        lines.append("")

    lines += [
        "PROVENANCE",
        "-" * 70,
        f"  Exported by   {exported_by}",
        f"  Exported at   {at:%Y-%m-%d %H:%M:%S} UTC",
        f"  Application   Sentinel Vision {APPLICATION_VERSION}",
        f"  Format        {EXPORT_FORMAT_VERSION}",
        f"  Platform      {platform.platform()}",
        f"  Python        {sys.version.split()[0]}",
        "",
        "  Every file in this package is listed in manifest.json with its",
        "  SHA-256. Verifying them detects any alteration since export.",
        "",
        "LIMITS OF THIS EVIDENCE",
        "-" * 70,
        "  This system reports what it observed and how well it observed it. It",
        "  does not identify people, and a detector that cannot classify says so",
        "  above rather than naming what it found. Positions carry the",
        "  uncertainty they were computed with; where a position could not be",
        "  determined, that is stated rather than omitted.",
        "",
    ]
    return "\n".join(lines)


def export_incident(
    incident: Incident,
    destination: Path,
    *,
    exported_by: str,
    attachments: Sequence[Path] = (),
    at: datetime | None = None,
) -> Export:
    """Write an evidence package for one incident.

    ``destination`` is the *containing* directory; a folder named for the
    incident is created inside it. Attachments are copied in and hashed with
    everything else, and any that would land outside the package are refused.
    """
    moment = at or datetime.now(timezone.utc)

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    package = _resolve_within(root, incident.id)
    package.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    incident_json = package / "incident.json"
    incident_json.write_text(
        json.dumps(_incident_dict(incident), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written.append(incident_json)

    report = package / "report.txt"
    report.write_text(_readable_report(incident, exported_by, moment), encoding="utf-8")
    written.append(report)

    for source in attachments:
        source = Path(source)
        if not source.is_file():
            raise ExportError(f"Attachment not found: {source}")
        target = _resolve_within(package, source.name)
        shutil.copy2(source, target)
        written.append(target)

    files = tuple(
        ExportedFile(
            name=path.name, bytes=path.stat().st_size, sha256=sha256_of(path)
        )
        for path in sorted(written, key=lambda p: p.name)
    )

    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "application": "Sentinel Vision",
        "application_version": APPLICATION_VERSION,
        "incident_id": incident.id,
        "exported_by": exported_by,
        "exported_at_utc": moment.isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        # Every model that contributed anything to this incident, by digest, so
        # a conclusion can be re-examined against the exact weights that made
        # it — three model versions later.
        "models": sorted(
            {
                event.evidence.model_digest
                for event in incident.events
                if event.evidence.model_digest
            }
        ),
        "detectors": sorted({event.evidence.detector for event in incident.events}),
        "files": [
            {"name": f.name, "bytes": f.bytes, "sha256": f.sha256} for f in files
        ],
    }

    manifest_path = package / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return Export(
        directory=package,
        incident_id=incident.id,
        files=files,
        manifest_sha256=sha256_of(manifest_path),
        exported_at=moment,
        exported_by=exported_by,
    )


def verify_export(package: Path) -> list[str]:
    """Check a package on disk against its own manifest.

    Returns the problems found, empty if none. Written so that verification does
    not require the object that produced the export — a package has to be
    checkable by somebody who has only the folder.
    """
    package = Path(package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json: missing"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"manifest.json: unreadable ({error})"]

    problems: list[str] = []
    listed = set()

    for entry in manifest.get("files", []):
        name = entry["name"]
        listed.add(name)
        path = package / name

        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        if sha256_of(path) != entry["sha256"]:
            problems.append(f"{name}: altered since export")
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"{name}: size changed since export")

    # A file nobody listed is as much a problem as one that changed: it may have
    # been added afterwards, and the manifest would not know.
    for path in package.iterdir():
        if path.name != "manifest.json" and path.name not in listed:
            problems.append(f"{path.name}: present but not listed in the manifest")

    return problems
