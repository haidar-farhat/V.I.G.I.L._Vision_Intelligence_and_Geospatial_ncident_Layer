"""Database rows to domain objects, and back.

Split from `store` at the line budget. The seam is a real one: everything here
is a pure function of a `sqlite3.Row`, with no connection, no thread rule and
no transaction — which is why these are the only parts of the storage layer
that can be read without holding the store's invariants in your head.

The direction matters. A row is turned into a domain object here and nowhere
else, so a column that changes meaning has exactly one place to be reconciled.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

from ..domain.detection import DetectorInfo
from ..domain.events import Event, EventType, Evidence, Severity
from ..domain.geo import CameraPose, Distortion, LatLon, PoseUncertainty
from ..domain.incidents import Association, Incident, Review, ReviewState, Risk, RiskFactor
from ..domain.zones import Membership, Schedule, Zone, ZoneKind


def _camera_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    pose = None
    if d["lat"] is not None:
        # A NULL sigma means nobody measured that parameter, so it keeps the
        # stated assumption. Per-parameter rather than all-or-nothing: a fit
        # that pinned the heading down is worth keeping even if it is only the
        # heading, and mixing a measured heading with an assumed roll is
        # honest as long as each says which it is.
        assumed = PoseUncertainty()
        uncertainty = PoseUncertainty(
            heading_deg=d["sigma_heading"] if d["sigma_heading"] is not None else assumed.heading_deg,
            pitch_deg=d["sigma_pitch"] if d["sigma_pitch"] is not None else assumed.pitch_deg,
            roll_deg=d["sigma_roll"] if d["sigma_roll"] is not None else assumed.roll_deg,
            mount_height_m=d["sigma_height"] if d["sigma_height"] is not None else assumed.mount_height_m,
            terrain_slope=assumed.terrain_slope,
        )
        pose = CameraPose(LatLon(d["lat"], d["lon"]), d["mount_height"], d["heading"], d["pitch"], d["roll"] or 0.0,
                          d["horizontal_fov"], d["vertical_fov"], d["range_meters"], uncertainty,
                          Distortion(d["k1"] or 0.0, d["k2"] or 0.0, d["p1"] or 0.0,
                                     d["p2"] or 0.0, d["k3"] or 0.0),
                          d["ground_tilt_east"] or 0.0, d["ground_tilt_north"] or 0.0)
    return {"id": d["id"], "name": d["name"], "source": d["source"], "credentials_ref": d["credentials_ref"],
            "pose": pose, "record": bool(d["record"]), "updated_at": d["updated_at"],
            "calibrated_at": d["calibrated_at"], "calibration_rms": d["calibration_rms"],
            "calibration_points": d["calibration_points"],
            "ground_solved_at": d["ground_solved_at"],
            "ground_observations": d["ground_observations"]}


def _zone_of(row: sqlite3.Row) -> Zone:
    schedule = Schedule(row["closed_from"], row["closed_until"]) if row["closed_from"] is not None else None
    return Zone(row["id"], row["name"], ZoneKind(row["kind"]), tuple(LatLon(p[0], p[1]) for p in json.loads(row["ring"])),
                frozenset(json.loads(row["watch"])), row["enter_after_millis"], row["exit_after_millis"],
                Membership(row["min_membership"]), schedule)


def _evidence_dict(e: Evidence) -> dict:
    return {"camera_id": e.camera_id, "track_id": e.track_id, "frame_index": e.frame_index,
            "detector": asdict(e.detector), "class_label": e.class_label, "latitude": e.latitude, "longitude": e.longitude,
            "position_uncertainty_meters": e.position_uncertainty_meters, "observations": e.observations,
            "conditions": list(e.conditions), "position_source": e.position_source,
            "appearance": [round(v, 5) for v in e.appearance]}


def _event_of(row: sqlite3.Row) -> Event:
    raw = json.loads(row["evidence"])
    det = raw["detector"]
    det["class_names"] = {int(k): v for k, v in det.get("class_names", {}).items()}
    det["input_size"] = tuple(det["input_size"]) if det.get("input_size") else None
    evidence = Evidence(raw["camera_id"], raw["track_id"], raw["frame_index"], DetectorInfo(**det), raw["class_label"],
                        raw["latitude"], raw["longitude"], raw["position_uncertainty_meters"], raw["observations"],
                        tuple(raw["conditions"]),
                        # Absent in rows written before triangulation and
                        # cross-camera appearance existed. Read with defaults
                        # rather than migrated: the evidence column is JSON
                        # precisely so a new field costs an old row nothing.
                        raw.get("position_source", "GROUND_PROJECTION"),
                        tuple(raw.get("appearance") or ()))
    return Event(row["id"], EventType(row["type"]), Severity(row["severity"]), row["summary"], row["occurred_at"],
                 datetime.fromtimestamp(row["occurred_at"] / 1000, tz=timezone.utc), row["node_id"], row["rule_id"],
                 row["confidence"], evidence, row["zone_id"], row["zone_name"])


def _review_of(row: sqlite3.Row) -> Review:
    keys = row.keys()
    if "state" not in keys:
        return Review()
    return Review(ReviewState(row["state"]), row["reviewed_by"], row["reviewed_at"], row["note"])


def _incident_of(row: sqlite3.Row, events: list[Event]) -> Incident:
    risk_raw = json.loads(row["risk"])
    risk = Risk(risk_raw["score"], tuple(RiskFactor(**f) for f in risk_raw["factors"]))
    associations = tuple(Association(tuple(a["a"]), tuple(a["b"]), a["score"], a["separation_meters"], a["allowance_meters"],
                                     a["time_gap_millis"], tuple(a["reasons"])) for a in json.loads(row["associations"]))
    return Incident(row["id"], Severity(row["severity"]), row["summary"], row["opened_at"], row["closed_at"],
                    datetime.fromtimestamp(row["opened_at"] / 1000, tz=timezone.utc), row["distinct_objects"],
                    tuple(json.loads(row["cameras"])), tuple(json.loads(row["zones"])), tuple(events), associations, risk,
                    _review_of(row))
