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
