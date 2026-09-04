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

from .events import Event, utc_from_millis
from .incidents import Incident
from .recording import Segment

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
        # Relative to when the incident opened, which is what "t+" claims.
        # `at_millis` is a wall clock: for a file it starts near zero and this
        # read correctly, but for a live camera it is a Unix epoch, and the
        # first real evidence package from a webcam timed its own first event
        # at "t+1788513275.8s" — fifty-six years after the incident it belongs
        # to. The absolute UTC time is printed beside it, because a package
        # read a year later needs both.
        offset = (entry.at_millis - incident.opened_at_millis) / 1000
        lines.append(
            f"  t+{offset:7.1f}s  {entry.at:%H:%M:%S} UTC  "
            f"[{entry.severity.value:8}]  {entry.camera_id}  {entry.summary}"
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




# --------------------------------------------------------------------- footage

#: How much to include before an incident opened. An event fires *after*
#: somebody is already inside a zone, so the footage that explains it starts
#: earlier — usually a good deal earlier than anyone expects when they first
#: choose a number.
DEFAULT_LEAD_SECONDS = 30.0

#: And after it closed, because what somebody does on the way out is evidence
#: too.
DEFAULT_TRAIL_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Coverage:
    """What footage exists for an incident, and what does not.

    The second half is the point. A package that quietly contains forty seconds
    of a ninety-second incident looks complete — the clips play, the manifest
    verifies — and the missing part is discovered by whoever is relying on it,
    at the worst moment. Every gap is measured, named and written into the
    package.
    """

    camera_id: str
    #: The window asked for, including lead and trail.
    requested_start_millis: int
    requested_end_millis: int
    segments: tuple["Segment", ...]
    #: Windows inside the request that no segment covers.
    gaps: tuple[tuple[int, int], ...]

    @property
    def covered_millis(self) -> int:
        requested = self.requested_end_millis - self.requested_start_millis
        return requested - sum(end - start for start, end in self.gaps)

    @property
    def covered_fraction(self) -> float:
        requested = self.requested_end_millis - self.requested_start_millis
        return self.covered_millis / requested if requested > 0 else 0.0

    @property
    def is_complete(self) -> bool:
        return not self.gaps


def coverage_for(
    store,
    incident: Incident,
    *,
    lead_seconds: float = DEFAULT_LEAD_SECONDS,
    trail_seconds: float = DEFAULT_TRAIL_SECONDS,
) -> list[Coverage]:
    """Which recorded segments cover an incident, per camera, and what is missing.

    One entry per camera the incident names, including cameras with **no**
    footage at all — an empty result would read as "nothing to attach" when the
    truth is "this camera recorded nothing, and that is a finding".
    """
    # Wall clock, from `opened_at`, and **not** from `opened_at_millis`.
    # Those two are different clocks: an incident's millis are media time,
    # counted from the start of the footage, while recordings are indexed by
    # when they actually happened because that is what retention and an
    # operator both work in. Using the wrong one asked for footage from 1970
    # and reported, correctly and uselessly, that none existed.
    opened = int(incident.opened_at.timestamp() * 1000)
    start = opened - int(lead_seconds * 1000)
    end = opened + incident.duration_millis + int(trail_seconds * 1000)

    coverages: list[Coverage] = []
    for camera_id in incident.cameras:
        segments = tuple(
            store.segments(camera_id=camera_id, start_millis=start, end_millis=end)
        )
        coverages.append(
            Coverage(
                camera_id=camera_id,
                requested_start_millis=start,
                requested_end_millis=end,
                segments=segments,
                gaps=_gaps(start, end, segments),
            )
        )
    return coverages


def _gaps(start: int, end: int, segments) -> tuple[tuple[int, int], ...]:
    """Windows in [start, end] that no segment covers.

    Segments are merged before subtracting, because two that overlap — which
    happens across a resolution change, where one is closed and another opened
    on the same instant — would otherwise each punch a hole in the other.
    """
    spans = sorted(
        (max(start, s.started_millis), min(end, s.ended_millis))
        for s in segments
        if s.ended_millis >= start and s.started_millis <= end
    )
    if not spans:
        return ((start, end),) if end > start else ()

    merged: list[list[int]] = [list(spans[0])]
    for span_start, span_end in spans[1:]:
        if span_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span_end)
        else:
            merged.append([span_start, span_end])

    gaps: list[tuple[int, int]] = []
    cursor = start
    for span_start, span_end in merged:
        if span_start > cursor:
            gaps.append((cursor, span_start))
        cursor = max(cursor, span_end)
    if cursor < end:
        gaps.append((cursor, end))

    # A sub-second gap is a rounding artefact of frame boundaries, not a hole in
    # the evidence, and reporting one would train an operator to ignore the
    # field that reports real ones.
    return tuple((s, e) for s, e in gaps if e - s > 1000)


def _footage_document(coverages: list[Coverage], names: dict[str, str]) -> dict:
    return {
        "lead_and_trail": "the window extends before and after the incident on purpose",
        "cameras": [
            {
                "camera_id": coverage.camera_id,
                "window_start": utc_from_millis(coverage.requested_start_millis).isoformat(),
                "window_end": utc_from_millis(coverage.requested_end_millis).isoformat(),
                "covered_fraction": round(coverage.covered_fraction, 4),
                "complete": coverage.is_complete,
                "clips": [
                    {
                        "file": names[str(segment.path)],
                        "start": utc_from_millis(segment.started_millis).isoformat(),
                        "end": utc_from_millis(segment.ended_millis).isoformat(),
                        "frames": segment.frames,
                        "resolution": f"{segment.width}x{segment.height}",
                        "codec": segment.codec,
                        # The measured rate, not the container header's claim.
                        # For a live camera those differ, and this is the one
                        # that describes what actually happened.
                        "measured_fps": round(segment.measured_fps, 3),
                        "nominal_fps": round(segment.nominal_fps, 3),
                        "sha256_when_recorded": segment.sha256,
                        "complete": segment.complete,
                    }
                    for segment in coverage.segments
                    if str(segment.path) in names
                ],
                "gaps": [
                    {
                        "from": utc_from_millis(start).isoformat(),
                        "to": utc_from_millis(end).isoformat(),
                        "seconds": round((end - start) / 1000, 1),
                    }
                    for start, end in coverage.gaps
                ],
            }
            for coverage in coverages
        ],
    }



def _unique_name(name: str, used: set[str]) -> str:
    """A name that is not already in the package.

    Two clips from two cameras are routinely both called `clip.mp4`, and copying
    the second over the first loses evidence *and still verifies clean*, because
    the manifest is written afterwards from what survived. A file named
    `incident.json` would have destroyed the record itself. Names are made
    unique here rather than trusted.
    """
    if name not in used:
        used.add(name)
        return name

    stem, _, suffix = name.rpartition(".")
    stem, suffix = (stem, "." + suffix) if stem else (name, "")
    index = 2
    while f"{stem}-{index}{suffix}" in used:
        index += 1
    unique = f"{stem}-{index}{suffix}"
    used.add(unique)
    return unique


def export_incident(
    incident: Incident,
    destination: Path,
    *,
    exported_by: str,
    attachments: Sequence[Path] = (),
    footage: Sequence[Coverage] = (),
    at: datetime | None = None,
) -> Export:
    """Write an evidence package for one incident.

    ``destination`` is the *containing* directory; a folder named for the
    incident is created inside it. Attachments are copied in and hashed with
    everything else, and any that would land outside the package are refused.

    ``footage`` is what :func:`coverage_for` returned. Its clips are copied in
    like any other attachment, and a ``footage.json`` records what each one is
    and — the part that matters — **what is missing**. A package containing
    forty seconds of a ninety-second incident plays, verifies, and misleads.
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

    reserved = {"manifest.json", "incident.json", "report.txt", "footage.json"}
    used: set[str] = set(reserved)

    # Footage first, so a clip keeps its own name and a same-named attachment is
    # the one that gets a suffix. The recording is the evidence; an attachment
    # is somebody's addition to it.
    clip_names: dict[str, str] = {}
    for coverage in footage:
        for segment in coverage.segments:
            if not segment.path.is_file():
                # Indexed but gone. Named in footage.json as a gap rather than
                # failing the whole export: the rest of the package is still
                # evidence, and its absence is itself a finding.
                continue
            name = _unique_name(segment.path.name, used)
            target = _resolve_within(package, name)
            shutil.copy2(segment.path, target)
            written.append(target)
            clip_names[str(segment.path)] = name

    if footage:
        footage_json = package / "footage.json"
        footage_json.write_text(
            json.dumps(_footage_document(list(footage), clip_names), indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(footage_json)

    for source in attachments:
        source = Path(source)
        if not source.is_file():
            raise ExportError(f"Attachment not found: {source}")

        # Two clips from two cameras are routinely both called `clip.mp4`, and
        # copying the second over the first loses evidence *and still verifies
        # clean*, because the manifest is written afterwards from what survived.
        # An attachment named `incident.json` would have destroyed the record
        # itself. Names are made unique here rather than trusted.
        name = _unique_name(source.name, used)

        target = _resolve_within(package, name)
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
