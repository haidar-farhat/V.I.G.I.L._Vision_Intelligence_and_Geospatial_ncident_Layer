"""Accounts, permissions and the audit trail that finally names a person."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.accounts import (
    INCIDENT_EXPORT, SITE_CONFIGURE, SITE_VIEW, USERS_MANAGE, LOCKOUT_AFTER,
    AccountError, Accounts, Role, User, hash_password, verify_password,
)
from sentinel.store import Store


@pytest.fixture
def store():
    with Store(":memory:") as opened:
        yield opened


def test_a_password_hashes_with_a_fresh_salt_and_verifies_in_constant_time():
    one, two = hash_password("hunter2-not-real"), hash_password("hunter2-not-real")
    assert one != two, "the same password must not hash the same twice"
    assert one.startswith("scrypt1$")
    assert verify_password("hunter2-not-real", one) and verify_password("hunter2-not-real", two)
    assert not verify_password("wrong", one)
    assert not verify_password("hunter2-not-real", "garbage")
    assert not verify_password("hunter2-not-real", "md5$00$00")
    with pytest.raises(AccountError):
        hash_password("")


def test_roles_are_sets_of_permissions_and_an_inactive_account_has_none():
    assert User("v", Role.VIEWER).may(SITE_VIEW)
    assert not User("v", Role.VIEWER).may(SITE_CONFIGURE)
    assert User("o", Role.OPERATOR).may(SITE_CONFIGURE)
    assert not User("o", Role.OPERATOR).may(USERS_MANAGE)
    assert User("a", Role.ANALYST).may(INCIDENT_EXPORT) and not User("a", Role.ANALYST).may(SITE_CONFIGURE)
    assert User("x", Role.ADMIN).may(USERS_MANAGE)
    assert not User("x", Role.ADMIN, active=False).may(SITE_VIEW)
    assert User("alice", Role.OPERATOR).actor == "console:alice"


def test_accounts_are_stored_hashed_and_authenticated_by_name(store):
    accounts = Accounts(store)
    assert not accounts.any()
    alice = accounts.add("Alice", "correct horse", Role.OPERATOR, actor="setup")
    assert alice.name == "alice" and accounts.any()
    row = store.user("alice")
    assert row["password_hash"].startswith("scrypt1$") and "correct horse" not in row["password_hash"]

    user = accounts.authenticate("ALICE", "correct horse")
    assert user == User("alice", Role.OPERATOR, True)
    with pytest.raises(AccountError, match="wrong"):
        accounts.authenticate("alice", "battery staple")
    with pytest.raises(AccountError, match="wrong"):
        accounts.authenticate("nobody", "correct horse")
    with pytest.raises(AccountError, match="already"):
        accounts.add("alice", "x", Role.VIEWER)
    for bad in ("", "two words", "x" * 65):
        with pytest.raises(AccountError):
            accounts.add(bad, "x", Role.VIEWER)

    actions = [row["action"] for row in store.audit_trail(limit=20)]
    assert "user.added" in actions and "login.succeeded" in actions and "login.failed" in actions
    trail = " ".join(str(dict(row)) for row in store.audit_trail(limit=20))
    assert "correct horse" not in trail and "battery" not in trail


def test_a_disabled_account_cannot_log_in_and_a_password_can_be_changed(store):
    accounts = Accounts(store)
    accounts.add("bob", "first", Role.VIEWER)
    accounts.set_password("bob", "second")
    with pytest.raises(AccountError):
        accounts.authenticate("bob", "first")
    accounts.authenticate("bob", "second")
    accounts.set_active("bob", False)
    with pytest.raises(AccountError):
        accounts.authenticate("bob", "second")
    assert [u.active for u in accounts.users()] == [False]
    with pytest.raises(AccountError, match="no account"):
        accounts.set_password("carol", "x")


def test_repeated_failures_make_the_next_attempt_wait(store):
    now = [0.0]
    accounts = Accounts(store, clock=lambda: now[0])
    accounts.add("eve", "right", Role.VIEWER)
    for _ in range(LOCKOUT_AFTER):
        with pytest.raises(AccountError, match="wrong"):
            accounts.authenticate("eve", "guess")
    assert accounts.seconds_until_allowed("eve") > 0
    with pytest.raises(AccountError, match="try again"):
        accounts.authenticate("eve", "right"), "the right password must wait too"
    now[0] += 3600
    assert accounts.authenticate("eve", "right").name == "eve"
    assert accounts.seconds_until_allowed("eve") == 0.0
    assert "login.locked" in [row["action"] for row in store.audit_trail(limit=30)]


def test_the_users_table_migration_carries_a_way_back_and_survives_a_round_trip(tmp_path: Path):
    with Store(tmp_path / "u.db") as store:
        Accounts(store).add("alice", "pw", Role.ADMIN)
        undone = store.rollback()
        assert undone is not None and undone.name == "users"
        assert "users" not in store.table_names()
        store.migrate()
        assert store.users() == [], "a table from before accounts existed holds nobody"
