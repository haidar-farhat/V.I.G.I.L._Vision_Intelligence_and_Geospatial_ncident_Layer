import ast
import inspect
import threading
from pathlib import Path

import pytest

from vigil.domain.geo import CameraPose, LatLon
from vigil.domain.zones import Membership, Schedule, Zone, ZoneKind
from vigil.storage import store as store_module
from vigil.storage.schema import MIGRATIONS, SCHEMA_VERSION
from vigil.storage.store import Store, StoreError, ThreadOwnership, restore_backup, verify_backup


def test_migrations_apply_and_every_one_has_a_way_back(tmp_path):
    with Store(tmp_path / "s.db") as store:
        assert store.applied_versions() == [m.version for m in MIGRATIONS]
        assert {"users", "audit", "alerts", "cameras", "zones", "events", "incidents", "recordings"} <= set(store.table_names())
        for migration in MIGRATIONS:
            assert migration.down.strip(), f"migration {migration.version} has no way back"
        while store.rollback() is not None:
            pass
        assert "users" not in store.table_names()
        store.migrate()
        assert store.applied_versions()[-1] == SCHEMA_VERSION


def test_no_column_in_the_schema_is_credential_shaped():
    forbidden = ("password", "passwd", "secret", "credential_value", "token", "api_key", "private_key")
    with Store(":memory:") as store:
        for table in store.table_names():
            for column in store.column_names(table):
                lowered = column.lower()
                if lowered == "credentials_ref" or (table == "users" and lowered == "password_hash"):
                    continue
                assert not any(word in lowered for word in forbidden), f"{table}.{column} looks like a credential"


def test_the_audit_table_is_append_only_by_construction():
    source = inspect.getsource(store_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            sql = node.value.strip().upper()
            assert not (sql.startswith(("UPDATE AUDIT", "DELETE FROM AUDIT"))), node.value


def test_the_store_belongs_to_the_thread_that_opened_it():
    with Store(":memory:") as store:
        errors = []

        def other():
            try:
                store.audit("x", "y")
            except ThreadOwnership as error:
                errors.append(error)

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert errors, "another thread touched the store and nothing said so"


def test_cameras_zones_and_site_round_trip(pose):
    with Store(":memory:") as store:
        store.save_camera("gate", "Gate", "rtsp" + "://user@10.0.0.9/s", credentials_ref="cam-abc", pose=pose, record=True)
        camera = store.camera("gate")
        assert camera["pose"] == pose and camera["record"] and camera["credentials_ref"] == "cam-abc"
        zone = Zone("z", "Yard", ZoneKind.RESTRICTED, (LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0)),
                    frozenset({"person"}), 700, 2500, Membership.UNCERTAIN, Schedule(22, 6))
        store.save_zone(zone)
        assert store.zones() == [zone]
        store.save_site("Depot", "Asia/Beirut")
        assert store.site()["timezone"] == "Asia/Beirut"
        store.delete_camera("gate")
        store.delete_zone("z")
        assert store.cameras() == [] and store.zones() == []


def test_a_measured_lens_and_pose_uncertainty_survive_a_round_trip(pose):
    """Migration 6. The uncertainty is the whole product of a calibration, so
    a round trip that quietly restored the default would throw away the
    measurement and leave a pose that merely looks confident."""
    from dataclasses import replace

    from vigil.domain.geo import Distortion, PoseUncertainty

    measured = replace(
        pose,
        uncertainty=PoseUncertainty(heading_deg=0.064, pitch_deg=0.243, roll_deg=0.105,
                                    mount_height_m=0.038),
        lens=Distortion(k1=-0.081, k2=0.012, p1=0.0004, p2=-0.0002, k3=0.0),
    )
    with Store(":memory:") as store:
        store.save_camera("gate", "Gate", "file:///x", pose=measured,
                          calibration=(1_757_000_000, 0.0021, 15))
        got = store.camera("gate")
        assert got["pose"] == measured
        assert got["calibration_points"] == 15 and got["calibration_rms"] == 0.0021


def test_turning_recording_on_does_not_erase_a_calibration(pose):
    """`save_camera` is how every edit to a camera is written, so provenance
    that vanished when it was not restated would be destroyed by an unrelated
    setting -- and the sigmas would survive with nothing to explain them."""
    from dataclasses import replace

    from vigil.domain.geo import PoseUncertainty

    measured = replace(pose, uncertainty=PoseUncertainty(0.064, 0.243, 0.105, 0.038))
    with Store(":memory:") as store:
        store.save_camera("gate", "Gate", "file:///x", pose=measured, calibration=(1_757_000_000, 0.0021, 15))
        store.save_camera("gate", "Gate", "file:///x", pose=measured, record=True)
        after = store.camera("gate")
        assert after["record"] and after["calibration_points"] == 15
        assert after["calibrated_at"] == 1_757_000_000
        assert after["pose"].uncertainty == measured.uncertainty


def test_an_uncalibrated_camera_comes_back_assumed_rather_than_measured(pose):
    """NULL sigmas must stay distinguishable from measured ones that happen to
    equal the assumption; otherwise nothing can tell a measurement from a
    default and the provenance is worthless."""
    with Store(":memory:") as store:
        store.save_camera("gate", "Gate", "file:///x", pose=pose)
        assert store.camera("gate")["calibrated_at"] is None
        row = store._connection.execute("SELECT sigma_heading, k1 FROM cameras").fetchone()
        assert row["sigma_heading"] is None, "an unmeasured sigma must be stored as NULL"
        assert row["k1"] == 0.0, "an uncalibrated lens is rectilinear, which is what it was"


def test_backup_verify_and_restore(tmp_path):
    database = tmp_path / "s.db"
    with Store(database) as store:
        store.audit("t", "something")
        backup = store.backup_to(tmp_path / "backups" / "b.db")
    assert verify_backup(backup)
    (tmp_path / "backups" / "b.db.sha256").write_text("0" * 64 + "  b.db\n", encoding="utf-8")
    with pytest.raises(StoreError, match="checksum"):
        verify_backup(backup)
    with Store(database) as store:
        backup = store.backup_to(tmp_path / "backups" / "c.db")
        store.audit("t", "after the backup")
    restore_backup(backup, database)
    with Store(database) as store:
        assert [r["action"] for r in store.audit_trail()][0] == "database.backup"
    assert list(tmp_path.glob("s.db.before-restore-*"))


def test_a_newer_schema_is_refused_and_a_corrupt_file_is_named(tmp_path):
    database = tmp_path / "s.db"
    with Store(database) as store:
        store._connection.execute("INSERT INTO schema_migrations VALUES (99, 'future', 0)")
    with pytest.raises(StoreError, match="newer build"):
        Store(database)
    garbage = tmp_path / "g.db"
    garbage.write_bytes(b"not a database at all" * 100)
    with pytest.raises(StoreError):
        Store(garbage)


def test_a_model_is_looked_for_in_every_place_it_could_be(tmp_path, monkeypatch):
    """Looking in one place is how a run silently falls back to motion detection."""
    from vigil.config import Settings

    settings = Settings(tmp_path / "data", None, None, None, False)
    places = settings.model_directories()
    assert (tmp_path / "data" / "models") in places
    assert len(places) == len(set(places)), "a place must not be searched twice"
    assert settings.default_model() is None or settings.default_model().suffix == ".onnx"

    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "plain.onnx").write_bytes(b"x")
    assert settings.default_model().name == "plain.onnx"
    (tmp_path / "data" / "models" / "world-seg.onnx").write_bytes(b"x")
    assert settings.default_model().name == "world-seg.onnx", "a segmentation model is preferred"
