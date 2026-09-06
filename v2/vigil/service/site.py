"""Every change to the site passes through here, with a principal.

There is no other way to change a camera, a zone or a recording flag, and
every method's first thing is a permission check and its last an audit row
carrying the principal. The store never learns who is asking; this class
is the one that knows.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Sequence

from ..adapters.decode import is_live_source, redacted, split_password
from ..adapters.keychain import Keychain
from ..domain.geo import CameraPose, LatLon
from ..domain.zones import Membership, Schedule, Zone, ZoneKind
from ..logs import get as _get_logger
from ..storage.store import Store
from .auth import SITE_CONFIGURE, SITE_VIEW, Principal

_log = _get_logger(__name__)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
#: "Not given" for an edit, so that clearing a value and leaving it alone
#: are different requests. ``None`` means clear; absent means keep.
KEEP = object()


class SiteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Camera:
    id: str
    name: str
    source: str  # never carries a password
    credentials_ref: str | None
    pose: CameraPose | None
    record: bool
    #: When the pose was last measured rather than typed, and what that
    #: measurement was worth. `None` throughout means the pose's uncertainty
    #: is the stated assumption, not a measurement.
    calibrated_at: int | None = None
    calibration_rms: float | None = None
    calibration_points: int | None = None

    @property
    def live(self) -> bool:
        return is_live_source(self.source)

    @property
    def placed(self) -> bool:
        return self.pose is not None

    @property
    def calibrated(self) -> bool:
        return self.calibrated_at is not None


class SiteService:
    def __init__(self, store: Store, keychain: Keychain | None = None):
        self._store = store
        self._keychain = keychain if keychain is not None else Keychain(None)

    @property
    def store(self) -> Store:
        return self._store

    # ---------------------------------------------------------------- read

    def cameras(self, by: Principal) -> list[Camera]:
        by.require(SITE_VIEW)
        return [_camera(c) for c in self._store.cameras()]

    def camera(self, camera_id: str, by: Principal) -> Camera:
        by.require(SITE_VIEW)
        row = self._store.camera(camera_id)
        if row is None:
            raise SiteError(f"no camera called {camera_id!r}")
        return _camera(row)

    def zones(self, by: Principal) -> list[Zone]:
        by.require(SITE_VIEW)
        return self._store.zones()

    def source_with_credentials(self, camera: Camera) -> str:
        """The URL a decoder opens. Only the runtime asks; never logged or shown."""
        from ..adapters.decode import with_password

        secret = self._keychain.load(camera.credentials_ref)
        return with_password(camera.source, secret) if secret else camera.source

    # --------------------------------------------------------------- write

    def add_camera(self, camera_id: str, source: str, *, name: str | None = None, pose: CameraPose | None = None,
                   record: bool = False, by: Principal) -> Camera:
        by.require(SITE_CONFIGURE)
        camera_id = _valid_id(camera_id)
        if self._store.camera(camera_id) is not None:
            raise SiteError(f"there is already a camera called {camera_id!r}")
        if pose is not None:
            pose.validate()
        clean, password = split_password(str(source))
        ref = None
        if password:
            ref = self._keychain.store(password)
            if ref is None:
                _log.warning("%s: no keychain on this machine; the password is not kept and the camera will need it again after a restart", camera_id)
        self._store.save_camera(camera_id, name or camera_id, clean, credentials_ref=ref, pose=pose, record=record)
        self._store.audit(by.actor, "camera.added", camera_id, redacted(clean),
                          after={"source": redacted(clean), "placed": pose is not None, "record": record, "credential": ref is not None})
        return self.camera(camera_id, by)

    def place_camera(self, camera_id: str, pose: CameraPose, *, by: Principal) -> Camera:
        by.require(SITE_CONFIGURE)
        pose.validate()
        current = self.camera(camera_id, by)
        self._store.save_camera(camera_id, current.name, current.source, credentials_ref=current.credentials_ref,
                                pose=pose, record=current.record)
        self._store.audit(by.actor, "camera.placed", camera_id, f"{pose.position.lat:.6f},{pose.position.lon:.6f} heading {pose.heading:.0f}",
                          before=_pose_dict(current.pose), after=_pose_dict(pose))
        return self.camera(camera_id, by)

    def calibrate_camera(self, camera_id: str, points: Sequence, *, solve_position: bool = False,
                         lens=None, by: Principal):
        """Measure a placed camera's pose from marked ground points.

        Returns `(camera, calibration)`. Refuses, without saving, a fit that
        does not beat the uncertainty it would replace: a calibration that
        makes the pose *less* certain is not a calibration, and storing it
        would degrade every position this camera produces while looking like
        an improvement because somebody did the work.

        `lens` replaces the distortion coefficients if given, and the pose is
        fitted with it in place — the two are not independent, and refining
        the pose against a lens the camera does not have puts the lens's error
        into the pose.
        """
        from .calibration import CalibrationError, calibrate_pose

        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        if current.pose is None:
            raise SiteError(f"{camera_id} has no placement to refine; `cameras place` it first, "
                            f"roughly, and this will measure it properly")
        start = current.pose if lens is None else replace(current.pose, lens=lens)
        start.validate()
        try:
            result = calibrate_pose(start, points, solve_position=solve_position)
        except CalibrationError as error:
            raise SiteError(str(error)) from error
        result.pose.validate()
        if not result.better_than(current.pose.uncertainty):
            raise SiteError(
                f"this calibration is worse than what {camera_id} already assumes and has not been "
                f"saved:\n{result.describe()}\nSpread the points wider across the frame and the "
                f"range, or check for a mis-clicked one -- the worst was point "
                f"{result.worst_index + 1}"
            )
        self._store.save_camera(camera_id, current.name, current.source,
                                credentials_ref=current.credentials_ref, pose=result.pose,
                                record=current.record,
                                calibration=(int(time.time()), result.rms, len(points)))
        self._store.audit(by.actor, "camera.calibrated", camera_id,
                          f"{len(points)} points, {result.rms * 100:.2f}% RMS, "
                          f"heading +/- {result.uncertainty.heading_deg:.3f} deg",
                          before=_pose_dict(current.pose), after=_pose_dict(result.pose))
        _log.info("%s calibrated: %s", camera_id, result.describe().replace("\n", " | "))
        return self.camera(camera_id, by), result

    def set_recording(self, camera_id: str, on: bool, *, by: Principal) -> Camera:
        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        if current.record == on:
            return current
        self._store.save_camera(camera_id, current.name, current.source, credentials_ref=current.credentials_ref,
                                pose=current.pose, record=on)
        self._store.audit(by.actor, "camera.recording", camera_id, "on" if on else "off",
                          before={"record": current.record}, after={"record": on})
        return self.camera(camera_id, by)

    def set_password(self, camera_id: str, password: str, *, by: Principal) -> Camera:
        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        if not self._keychain.available:
            raise SiteError("no keychain on this machine; a password cannot be kept")
        self._keychain.forget(current.credentials_ref)
        ref = self._keychain.store(password)
        self._store.save_camera(camera_id, current.name, current.source, credentials_ref=ref, pose=current.pose, record=current.record)
        self._store.audit(by.actor, "camera.credential", camera_id, "password stored in the keychain")
        return self.camera(camera_id, by)

    def set_source(self, camera_id: str, source: str, *, by: Principal) -> Camera:
        """The camera moved to a new address. Everything else about it stays.

        Without this, a camera whose address changed had to be removed and
        added again — which threw away its placement, and with it every zone
        that acted on what it saw.
        """
        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        clean, password = split_password(str(source))
        ref = current.credentials_ref
        if password:
            self._keychain.forget(ref)
            ref = self._keychain.store(password)
        self._store.save_camera(camera_id, current.name, clean, credentials_ref=ref, pose=current.pose,
                                record=current.record)
        self._store.audit(by.actor, "camera.source_changed", camera_id, redacted(clean),
                          before={"source": redacted(current.source)}, after={"source": redacted(clean)})
        return self.camera(camera_id, by)

    def rename_camera(self, camera_id: str, name: str, *, by: Principal) -> Camera:
        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        if not (name or "").strip():
            raise SiteError("a camera needs a name")
        self._store.save_camera(camera_id, name.strip(), current.source, credentials_ref=current.credentials_ref,
                                pose=current.pose, record=current.record)
        self._store.audit(by.actor, "camera.renamed", camera_id, name.strip(),
                          before={"name": current.name}, after={"name": name.strip()})
        return self.camera(camera_id, by)

    def remove_camera(self, camera_id: str, *, by: Principal) -> None:
        by.require(SITE_CONFIGURE)
        current = self.camera(camera_id, by)
        self._keychain.forget(current.credentials_ref)
        self._store.delete_camera(camera_id)
        self._store.audit(by.actor, "camera.removed", camera_id, redacted(current.source),
                          before={"source": redacted(current.source), "placed": current.placed})

    def add_zone(self, zone_id: str, name: str, kind: ZoneKind | str, ring: Sequence[LatLon], *,
                 watch: Sequence[str] = (), enter_after_millis: int = 600, exit_after_millis: int = 2000,
                 min_membership: Membership | str = Membership.INSIDE, schedule: Schedule | None = None,
                 by: Principal) -> Zone:
        by.require(SITE_CONFIGURE)
        zone_id = _valid_id(zone_id)
        if self._store.zone(zone_id) is not None:
            raise SiteError(f"there is already a zone called {zone_id!r}")
        zone = Zone(zone_id, name, ZoneKind(str(kind).upper()), tuple(ring), frozenset(w.lower() for w in watch),
                    enter_after_millis, exit_after_millis, Membership(str(min_membership).upper()), schedule)
        zone.validate()
        self._store.save_zone(zone)
        self._store.audit(by.actor, "zone.added", zone_id, f"{name} ({zone.kind}, {len(zone.ring)} points)",
                          after={"name": name, "kind": zone.kind.value, "watch": sorted(zone.watch)})
        return zone

    def edit_zone(self, zone_id: str, *, name=KEEP, kind=KEEP, watch=KEEP, enter_after_millis=KEEP,
                  exit_after_millis=KEEP, min_membership=KEEP, schedule=KEEP, ring=KEEP, by: Principal) -> Zone:
        """Change what a zone means, keeping the ring somebody drew.

        Deleting and re-adding was the only way to fix a mistyped name or a
        wrong watch list, and it threw away the geometry — which is the part
        that took care to get right.
        """
        by.require(SITE_CONFIGURE)
        current = self._store.zone(zone_id)
        if current is None:
            raise SiteError(f"no zone called {zone_id!r}")
        updated = Zone(
            current.id,
            current.name if name is KEEP else (str(name).strip() or current.name),
            current.kind if kind is KEEP else ZoneKind(str(kind).upper()),
            current.ring if ring is KEEP else tuple(ring),
            current.watch if watch is KEEP else frozenset(w.lower() for w in watch),
            current.enter_after_millis if enter_after_millis is KEEP else int(enter_after_millis),
            current.exit_after_millis if exit_after_millis is KEEP else int(exit_after_millis),
            current.min_membership if min_membership is KEEP else Membership(str(min_membership).upper()),
            current.schedule if schedule is KEEP else schedule,
        )
        updated.validate()
        self._store.save_zone(updated)
        self._store.audit(by.actor, "zone.changed", zone_id, updated.name,
                          before=_zone_dict(current), after=_zone_dict(updated))
        return updated

    def remove_zone(self, zone_id: str, *, by: Principal) -> None:
        by.require(SITE_CONFIGURE)
        zone = self._store.zone(zone_id)
        if zone is None:
            raise SiteError(f"no zone called {zone_id!r}")
        self._store.delete_zone(zone_id)
        self._store.audit(by.actor, "zone.removed", zone_id, zone.name, before={"name": zone.name, "kind": zone.kind.value})

    @staticmethod
    def known_timezone(name: str):
        """The zone, or a `SiteError` naming what went wrong.

        Checked when it is typed rather than when a schedule is read: a site
        that learns at 03:00 that its clock was never valid has already been
        told the wrong thing all night.
        """
        from datetime import timezone
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        name = (name or "UTC").strip()
        if name.upper() == "UTC":
            return timezone.utc
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError as error:
            raise SiteError(f"this machine does not know the time zone {name!r}. Use an IANA name such as "
                            f"Asia/Beirut, or UTC. ({error})") from error
        except (ValueError, KeyError) as error:
            raise SiteError(f"{name!r} is not a time zone name: {error}") from error

    def threats(self):
        """What this site treats as dangerous. Empty until somebody says otherwise."""
        from ..domain.threats import ThreatVocabulary

        return ThreatVocabulary.from_labels(self._store.site().get("threat_labels") or [])

    def set_threats(self, labels: Sequence[str], *, by: Principal):
        """Name the labels this site calls dangerous, or clear them with an empty list."""
        from ..domain.threats import ThreatVocabulary

        by.require(SITE_CONFIGURE)
        before = self.threats()
        self._store.set_threat_labels(list(labels))
        after = ThreatVocabulary.from_labels(self._store.site().get("threat_labels") or [])
        self._store.audit(by.actor, "site.threats_changed", None, after.describe(),
                          before={"labels": list(before.labels)}, after={"labels": list(after.labels)})
        return after

    def detection(self):
        """What this site watches for and how sure it must be."""
        from .detection import DetectionSettings

        return DetectionSettings.from_site(self._store.site())

    def set_detection(self, labels, confidence: float | None, *, by: Principal,
                      detect_every: int | None = None, tile=KEEP):
        """Store the watch list, the threshold, how often to look, and whether
        to look closely at the far ground.

        `tile` defaults to `KEEP` rather than to `None`, because `None` is a
        meaningful value for it -- "decide from the provider" -- and an edit to
        the watch list must not silently turn that into a fixed choice.
        """
        from .detection import DetectionSettings

        by.require(SITE_CONFIGURE)
        before = self.detection()
        after = DetectionSettings.checked(
            labels, confidence, before.detect_every if detect_every is None else detect_every,
            before.tile if tile is KEEP else tile)
        self._store.set_detection(sorted(after.labels or ()), after.confidence, after.detect_every,
                                  after.tile)
        self._store.audit(by.actor, "site.detection_changed", None, after.describe(),
                          before={"labels": sorted(before.labels or ()), "confidence": before.confidence,
                                  "detect_every": before.detect_every, "tile": before.tile},
                          after={"labels": sorted(after.labels or ()), "confidence": after.confidence,
                                 "detect_every": after.detect_every, "tile": after.tile})
        return after

    def name_site(self, name: str, timezone_name: str, *, by: Principal) -> None:
        by.require(SITE_CONFIGURE)
        self.known_timezone(timezone_name)
        before = self._store.site()
        self._store.save_site(name, timezone_name)
        self._store.audit(by.actor, "site.named", name, timezone_name,
                          before={"name": before["name"], "timezone": before["timezone"]}, after={"name": name, "timezone": timezone_name})


def _valid_id(value: str) -> str:
    if not _ID.match(value or ""):
        raise SiteError(f"{value!r} is not a valid id (letters, digits, . _ : -, up to 64)")
    return value


def _camera(row: dict) -> Camera:
    return Camera(row["id"], row["name"], row["source"], row["credentials_ref"], row["pose"], row["record"],
                  row.get("calibrated_at"), row.get("calibration_rms"), row.get("calibration_points"))


def _zone_dict(zone: Zone) -> dict:
    return {"name": zone.name, "kind": zone.kind.value, "watch": sorted(zone.watch),
            "enter_after_millis": zone.enter_after_millis, "exit_after_millis": zone.exit_after_millis,
            "min_membership": zone.min_membership.value, "points": len(zone.ring),
            "closed": None if zone.schedule is None else zone.schedule.describe()}


def _pose_dict(pose: CameraPose | None) -> dict | None:
    """The pose as the audit records it.

    Roll, the sigmas and the lens are here because a calibration changes
    little else: an audit row whose before and after are identical is a record
    that something happened and no record of what.
    """
    if pose is None:
        return None
    u = pose.uncertainty
    row = {"lat": pose.position.lat, "lon": pose.position.lon, "height": pose.mount_height,
           "heading": pose.heading, "pitch": pose.pitch, "roll": pose.roll,
           "hfov": pose.horizontal_fov, "vfov": pose.vertical_fov, "range": pose.range_meters,
           "sigma_heading": u.heading_deg, "sigma_pitch": u.pitch_deg, "sigma_roll": u.roll_deg,
           "sigma_height": u.mount_height_m}
    if not pose.lens.is_identity:
        row["lens"] = [pose.lens.k1, pose.lens.k2, pose.lens.p1, pose.lens.p2, pose.lens.k3]
    return row
