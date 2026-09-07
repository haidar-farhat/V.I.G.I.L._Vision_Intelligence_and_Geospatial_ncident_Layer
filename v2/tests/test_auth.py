import pytest

from vigil.service.auth import (
    ANALYSIS_CONTROL, LOCKOUT_AFTER, SITE_CONFIGURE, SITE_VIEW, USERS_MANAGE, Accounts, AuthError, Forbidden, Principal,
    Role, hash_password, verify_password,
)
from vigil.storage.store import Store


def test_passwords_hash_with_fresh_salts_and_verify():
    one, two = hash_password("correct horse"), hash_password("correct horse")
    assert one != two and verify_password("correct horse", one) and not verify_password("wrong", two)
    assert not verify_password("x", "garbage")
    with pytest.raises(AuthError):
        hash_password("")


def test_principals_check_permissions_not_role_names():
    assert Principal("v", Role.VIEWER, "user").may(SITE_VIEW) and not Principal("v", Role.VIEWER, "user").may(SITE_CONFIGURE)
    assert Principal("o", Role.OPERATOR, "user").may(ANALYSIS_CONTROL) and not Principal("o", Role.OPERATOR, "user").may(USERS_MANAGE)
    assert Principal("a", Role.ADMIN, "user").may(USERS_MANAGE)
    assert not Principal("a", Role.ADMIN, "user", active=False).may(SITE_VIEW)
    assert Principal.open_site("me").may(USERS_MANAGE) and Principal.system().may(SITE_CONFIGURE)
    with pytest.raises(Forbidden):
        Principal("v", Role.VIEWER, "user").require(SITE_CONFIGURE)
    assert Principal("alice", Role.OPERATOR, "user").actor == "user:alice"


def test_the_first_account_opens_the_gate_and_later_ones_need_the_permission():
    with Store(":memory:") as store:
        accounts = Accounts(store)
        assert not accounts.any()
        open_site = accounts.principal_for(None, None)
        assert open_site.origin == "open"
        admin = accounts.add("Root", "a strong one", Role.ADMIN, by=open_site)
        assert admin.name == "root"
        with pytest.raises(AuthError, match="sign in"):
            accounts.principal_for(None, None)
        viewer = accounts.authenticate("root", "a strong one")
        assert viewer.role is Role.ADMIN
        bob = accounts.add("bob", "another one", Role.VIEWER, by=viewer)
        with pytest.raises(Forbidden):
            accounts.add("carol", "x", Role.VIEWER, by=bob)
        accounts.set_password("bob", "swapped-pw-77", by=bob)  # one's own password
        with pytest.raises(Forbidden):
            accounts.set_password("root", "swapped-pw-77", by=bob)
        accounts.set_role("bob", Role.OPERATOR, by=viewer)
        accounts.set_active("bob", False, by=viewer)
        with pytest.raises(AuthError, match="wrong"):
            accounts.authenticate("bob", "swapped-pw-77")
        actions = [r["action"] for r in store.audit_trail()]
        assert {"user.added", "login.succeeded", "user.password_changed", "user.role_changed", "user.disabled", "login.failed"} <= set(actions)
        trail = " ".join(str(dict(r)) for r in store.audit_trail())
        for secret in ("a strong one", "another one", "swapped-pw-77"):
            assert secret not in trail, "a password reached the audit trail"


def test_repeated_failures_lock_the_name_for_a_while():
    now = [0.0]
    with Store(":memory:") as store:
        accounts = Accounts(store, clock=lambda: now[0])
        accounts.add("eve", "right", Role.VIEWER, by=Principal.open_site())
        for _ in range(LOCKOUT_AFTER):
            with pytest.raises(AuthError, match="wrong"):
                accounts.authenticate("eve", "guess")
        with pytest.raises(AuthError, match="try again"):
            accounts.authenticate("eve", "right")
        now[0] += 3600
        assert accounts.authenticate("eve", "right").name == "eve"
        assert "login.locked" in [r["action"] for r in store.audit_trail()]
