"""An incident, as a package somebody else can check: report, clips, hashes."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.recorder import file_safe
from ..domain.incidents import Incident
from ..storage.store import Store
from ..version import describe
from .auth import INCIDENT_EXPORT, Principal

LEAD_MILLIS = 10_000
TRAIL_MILLIS = 10_000


def export_incident(store: Store, incident: Incident, destination: Path, *, by: Principal) -> Path:
    """Write `<destination>/<incident id>/` with report.json, report.txt, clips and manifest.json."""
    by.require(INCIDENT_EXPORT)
    folder = Path(destination) / file_safe(incident.id)
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    report = {
        "application": describe(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exported_by": by.actor,
        "incident": {
            "id": incident.id, "severity": incident.severity.value, "summary": incident.summary,
            "opened_at": incident.opened_at.isoformat(), "closed_at": datetime.fromtimestamp(incident.closed_at_millis / 1000, tz=timezone.utc).isoformat(),
            "distinct_objects": incident.distinct_objects, "cameras": list(incident.cameras), "zones": list(incident.zones),
            "risk": {"score": incident.risk.score, "factors": [asdict(f) for f in incident.risk.factors]},
            "associations": [asdict(a) for a in incident.associations],
        },
        "events": [_event_dict(e) for e in incident.events],
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    files["report.json"] = _sha256(folder / "report.json")

    clips = []
    window = (incident.opened_at_millis - LEAD_MILLIS, incident.closed_at_millis + TRAIL_MILLIS)
    for camera_id in incident.cameras:
        for segment in store.segments(camera_id=camera_id, between=window):
            if not segment.path.is_file():
                continue
            target = folder / "clips" / segment.path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(segment.path, target)
            digest = _sha256(target)
            if digest != segment.sha256:
                (folder / "WARNING.txt").write_text(f"{segment.path.name}: digest differs from the one recorded when the clip closed\n", encoding="utf-8")
            clips.append({"file": f"clips/{target.name}", "camera": camera_id, "started_at": segment.started_millis,
                          "ended_at": segment.ended_millis, "sha256": digest, "recorded_sha256": segment.sha256})
            files[f"clips/{target.name}"] = digest
    store.preserve_segments([s.path for cid in incident.cameras for s in store.segments(camera_id=cid, between=window)])

    lines = [f"{report['application']}", f"Incident {incident.id} — {incident.severity.value}", incident.summary, "",
             f"Opened {report['incident']['opened_at']}, closed {report['incident']['closed_at']}",
             f"{incident.distinct_objects} distinct object(s); cameras {', '.join(incident.cameras)}", "", "Events:"]
    for e in incident.events:
        lines.append(f"  {e.occurred_at.isoformat()} [{e.severity.value}] {e.summary} (rule {e.rule_id}, confidence {e.confidence:.2f})")
        for c in e.evidence.conditions:
            lines.append(f"      - {c}")
    lines += ["", f"Clips: {len(clips)}"] + [f"  {c['file']} ({c['camera']})" for c in clips]
    (folder / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files["report.txt"] = _sha256(folder / "report.txt")

    manifest = {"incident": incident.id, "files": files, "clips": clips, "chain": _chain(files)}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    store.audit(by.actor, "incident.exported", incident.id, f"{len(clips)} clip(s) to {folder}")
    return folder


def verify_package(folder: Path) -> list[str]:
    """Problems found; empty means every file matches the manifest."""
    manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for name, digest in manifest["files"].items():
        path = Path(folder) / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif _sha256(path) != digest:
            problems.append(f"{name}: digest differs")
    if _chain(manifest["files"]) != manifest["chain"]:
        problems.append("manifest chain differs")
    return problems


def _chain(files: dict[str, str]) -> str:
    return hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_dict(e) -> dict:
    return {"id": e.id, "type": e.type.value, "severity": e.severity.value, "summary": e.summary,
            "occurred_at": e.occurred_at.isoformat(), "rule": e.rule_id, "confidence": e.confidence,
            "zone": e.zone_name, "evidence": {
                "camera": e.evidence.camera_id, "track": e.evidence.track_id, "frame": e.evidence.frame_index,
                "detector": asdict(e.evidence.detector), "label": e.evidence.class_label,
                "latitude": e.evidence.latitude, "longitude": e.evidence.longitude,
                "uncertainty_m": e.evidence.position_uncertainty_meters, "observations": e.evidence.observations,
                "conditions": list(e.evidence.conditions)}}
