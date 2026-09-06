"""Every change to the site passes through here, with a principal.

There is no other way to change a camera, a zone or a recording flag, and
every method's first thing is a permission check and its last an audit row
carrying the principal. The store never learns who is asking; this class
is the one that knows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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

    @property
    def live(self) -> bool:
        return is_live_source(self.source)

    @property
    def placed(self) -> bool:
        return self.pose is not None


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

    def remove_zone(self, zone_id: str, *, by: Principal) -> None:
        by.require(SITE_CONFIGURE)
        zone = self._store.zone(zone_id)
        if zone is None:
            raise SiteError(f"no zone called {zone_id!r}")
        self._store.delete_zone(zone_id)
        self._store.audit(by.actor, "zone.removed", zone_id, zone.name, before={"name": zone.name, "kind": zone.kind.value})

    def name_site(self, name: str, timezone_name: str, *, by: Principal) -> None:
        by.require(SITE_CONFIGURE)
        before = self._store.site()
        self._store.save_site(name, timezone_name)
        self._store.audit(by.actor, "site.named", name, timezone_name,
                          before={"name": before["name"], "timezone": before["timezone"]}, after={"name": name, "timezone": timezone_name})


def _valid_id(value: str) -> str:
    if not _ID.match(value or ""):
        raise SiteError(f"{value!r} is not a valid id (letters, digits, . _ : -, up to 64)")
    return value


def _camera(row: dict) -> Camera:
    return Camera(row["id"], row["name"], row["source"], row["credentials_ref"], row["pose"], row["record"])


def _pose_dict(pose: CameraPose | None) -> dict | None:
    if pose is None:
        return None
    return {"lat": pose.position.lat, "lon": pose.position.lon, "height": pose.mount_height, "heading": pose.heading,
            "pitch": pose.pitch, "hfov": pose.horizontal_fov, "vfov": pose.vertical_fov, "range": pose.range_meters}
