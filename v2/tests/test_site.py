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
            # `known_timezone` is a pure validator that changes nothing.
            if name.startswith("_") or name in ("cameras", "camera", "zones", "threats", "detection", "source_with_credentials",
                                                "store", "known_timezone"):
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


def test_an_unknown_time_zone_is_refused_where_it_is_typed(keychain):
    """A site that learns at 03:00 that its clock was invalid has already been told the wrong thing all night."""
    from datetime import timezone
    from zoneinfo import ZoneInfo

    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        assert site.known_timezone("UTC") is timezone.utc
        assert site.known_timezone("  utc  ") is timezone.utc
        assert site.known_timezone("Asia/Beirut") == ZoneInfo("Asia/Beirut")
        with pytest.raises(SiteError, match="does not know the time zone"):
            site.known_timezone("Mars/Olympus")
        with pytest.raises(SiteError):
            site.name_site("Depot", "Mars/Olympus", by=OPERATOR)
        assert store.site()["name"] != "Depot", "a refused clock must not half-save the site"
        site.name_site("Depot", "Asia/Beirut", by=OPERATOR)
        assert store.site()["timezone"] == "Asia/Beirut"


def test_a_camera_that_moved_keeps_its_placement_and_its_password_follows(keychain, pose):
    """Removing and re-adding was the only way, and it threw away the placement."""
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        site.add_camera("gate", "rtsp" + "://admin:oldpass@10.0.0.9/s", pose=pose, record=True, by=OPERATOR)
        moved = site.set_source("gate", "rtsp" + "://admin:newpass@10.0.0.44/s", by=OPERATOR)
        assert moved.source.endswith("10.0.0.44/s") and moved.pose == pose and moved.record
        opened = site.source_with_credentials(moved)
        assert "newpass" in opened and "oldpass" not in opened
        renamed = site.rename_camera("gate", "North gate", by=OPERATOR)
        assert renamed.name == "North gate" and renamed.id == "gate"
        with pytest.raises(SiteError):
            site.rename_camera("gate", "   ", by=OPERATOR)
        actions = {r["action"] for r in store.audit_trail()}
        assert {"camera.source_changed", "camera.renamed"} <= actions
        trail = " ".join(str(dict(r)) for r in store.audit_trail())
        assert "oldpass" not in trail and "newpass" not in trail


def test_a_zone_can_be_changed_without_losing_the_ring_somebody_drew(keychain):
    from vigil.domain.zones import Schedule, ZoneKind

    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        ring = [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0.001), LatLon(0.001, 0)]
        site.add_zone("yard", "Yrad", "interest", ring, watch=["person"], by=OPERATOR)
        fixed = site.edit_zone("yard", name="Yard", kind=ZoneKind.RESTRICTED, watch=["Person", "car"],
                               schedule=Schedule(22, 6), by=OPERATOR)
        assert fixed.name == "Yard" and fixed.kind is ZoneKind.RESTRICTED
        assert fixed.watch == frozenset({"person", "car"}) and fixed.ring == tuple(ring)
        assert fixed.schedule == Schedule(22, 6)
        # Absent means keep; None means clear. They are different requests.
        kept = site.edit_zone("yard", enter_after_millis=1200, by=OPERATOR)
        assert kept.schedule == Schedule(22, 6) and kept.enter_after_millis == 1200 and kept.name == "Yard"
        cleared = site.edit_zone("yard", schedule=None, by=OPERATOR)
        assert cleared.schedule is None
        row = [r for r in store.audit_trail() if r["action"] == "zone.changed"][-1]
        assert "Yrad" in row["before"] and "Yard" in row["after"]
        with pytest.raises(SiteError, match="no zone"):
            site.edit_zone("nothing", name="x", by=OPERATOR)
        with pytest.raises(Forbidden):
            site.edit_zone("yard", name="x", by=VIEWER)


def test_a_site_names_what_it_treats_as_dangerous_and_the_change_is_audited(keychain):
    from vigil.domain.events import Severity

    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        assert not site.threats(), "nothing is a threat until a site says so"
        vocabulary = site.set_threats(["Knife", " gun ", ""], by=OPERATOR)
        assert vocabulary.labels == ("gun", "knife")
        assert vocabulary.of("knife").severity is Severity.HIGH
        assert site.threats().labels == ("gun", "knife"), "it must survive a re-read"
        row = [r for r in store.audit_trail() if r["action"] == "site.threats_changed"][0]
        assert '"labels": []' in row["before"] and "knife" in row["after"]

        # Naming or re-clocking the site must not quietly drop them.
        site.name_site("Depot", "UTC", by=OPERATOR)
        assert site.threats().labels == ("gun", "knife")
        assert site.set_threats([], by=OPERATOR).labels == ()
        with pytest.raises(Forbidden):
            site.set_threats(["knife"], by=VIEWER)
