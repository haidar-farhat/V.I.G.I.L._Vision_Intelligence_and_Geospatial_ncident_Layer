import inspect

import pytest

from vigil.domain.geo import LatLon
from vigil.domain.zones import ZoneKind
from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.site import SiteError, SiteService
from vigil.storage.store import Store

OPERATOR = Principal("alice", Role.OPERATOR, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


def test_every_mutating_method_takes_a_principal_and_writes_an_audit_row(keychain, pose):
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        for name, method in inspect.getmembers(SiteService, inspect.isfunction):
            if name.startswith("_") or name in ("cameras", "camera", "zones", "source_with_credentials", "store"):
                continue
            assert "by" in inspect.signature(method).parameters, f"{name} takes no principal"
        camera = site.add_camera("gate", "rtsp" + "://admin:s3cret@10.0.0.9/s", pose=pose, by=OPERATOR)
        assert "s3cret" not in camera.source and camera.credentials_ref
        assert site.source_with_credentials(camera).count("s3cret") == 1
        site.place_camera("gate", pose, by=OPERATOR)
        site.set_recording("gate", True, by=OPERATOR)
        site.set_password("gate", "newer", by=OPERATOR)
        assert "newer" in site.source_with_credentials(site.camera("gate", OPERATOR))
        zone = site.add_zone("yard", "Yard", "restricted", [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0)], watch=["Person"], by=OPERATOR)
        assert zone.kind is ZoneKind.RESTRICTED and zone.watch == frozenset({"person"})
        site.name_site("Depot", "Asia/Beirut", by=OPERATOR)
        site.remove_zone("yard", by=OPERATOR)
        site.remove_camera("gate", by=OPERATOR)
        rows = store.audit_trail()
        assert {r["principal"] for r in rows} == {"user:alice"}
        assert {"camera.added", "camera.placed", "camera.recording", "camera.credential", "zone.added", "site.named", "zone.removed", "camera.removed"} <= {r["action"] for r in rows}
        assert "s3cret" not in " ".join(str(dict(r)) for r in rows) and "newer" not in " ".join(str(dict(r)) for r in rows)
        assert keychain.load(camera.credentials_ref) is None, "a removed camera's password is forgotten"


def test_a_viewer_may_read_and_may_not_change(keychain, pose):
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", "clip.mp4", by=OPERATOR)
        assert [c.id for c in site.cameras(VIEWER)] == ["gate"]
        with pytest.raises(Forbidden):
            site.place_camera("gate", pose, by=VIEWER)
        with pytest.raises(Forbidden):
            site.add_zone("z", "Z", "interest", [LatLon(0, 0)] * 3, by=VIEWER)


def test_ids_and_poses_are_validated_and_duplicates_refused(keychain, pose):
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        with pytest.raises(SiteError):
            site.add_camera("bad id!", "clip.mp4", by=OPERATOR)
        site.add_camera("gate", "clip.mp4", by=OPERATOR)
        with pytest.raises(SiteError, match="already"):
            site.add_camera("gate", "clip.mp4", by=OPERATOR)
        with pytest.raises(ValueError):
            site.add_zone("z", "Z", "restricted", [LatLon(0, 0), LatLon(0, 1)], by=OPERATOR)
        with pytest.raises(SiteError, match="no camera"):
            site.camera("nothing", OPERATOR)


def test_without_a_keychain_the_password_is_not_kept_and_that_is_said(pose, caplog):
    from vigil.adapters.keychain import Keychain

    with Store(":memory:") as store:
        site = SiteService(store, Keychain(None))
        camera = site.add_camera("gate", "rtsp" + "://u:p@10.0.0.9/s", by=Principal.open_site())
        assert camera.credentials_ref is None and "not kept" in caplog.text
        with pytest.raises(SiteError, match="no keychain"):
            site.set_password("gate", "x", by=Principal.open_site())
