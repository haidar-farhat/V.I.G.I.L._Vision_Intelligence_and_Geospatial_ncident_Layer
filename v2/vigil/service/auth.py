"""Who is asking, and what they may do. Permission by name, never by role name."""

from __future__ import annotations

import getpass
import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from enum import StrEnum

from ..logs import get as _get_logger
from ..storage.store import Store

_log = _get_logger(__name__)


class Role(StrEnum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"


SITE_VIEW = "site.view"
SITE_CONFIGURE = "site.configure"
ANALYSIS_CONTROL = "analysis.control"
INCIDENT_EXPORT = "incident.export"
#: Judging an incident — acknowledged, or dismissed with a reason. Separate
#: from export because reading out evidence and passing judgement on it are
#: different acts, and a site may want different people doing them.
INCIDENT_REVIEW = "incident.review"
AUDIT_READ = "audit.read"
USERS_MANAGE = "users.manage"

PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({SITE_VIEW}),
    Role.OPERATOR: frozenset({SITE_VIEW, SITE_CONFIGURE, ANALYSIS_CONTROL, INCIDENT_EXPORT, INCIDENT_REVIEW}),
    Role.ANALYST: frozenset({SITE_VIEW, INCIDENT_EXPORT, INCIDENT_REVIEW, AUDIT_READ}),
    Role.ADMIN: frozenset({SITE_VIEW, SITE_CONFIGURE, ANALYSIS_CONTROL, INCIDENT_EXPORT, INCIDENT_REVIEW,
                           AUDIT_READ, USERS_MANAGE}),
}
ALL_PERMISSIONS = frozenset().union(*PERMISSIONS.values())

LOCKOUT_AFTER = 5
LOCKOUT_SECONDS = (5.0, 15.0, 30.0, 60.0, 300.0)
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)
_HASH_VERSION = "scrypt1"


class AuthError(PermissionError):
    pass


class Forbidden(AuthError):
    pass


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is acting. `open` means the store has no users yet and gates nothing."""

    name: str
    role: Role | None
    origin: str  # "user" | "system" | "open"
    active: bool = True

    @property
    def actor(self) -> str:
        return f"{self.origin}:{self.name}"

    def may(self, permission: str) -> bool:
        if not self.active:
            return False
        if self.origin in ("system", "open"):
            return True
        return self.role is not None and permission in PERMISSIONS[self.role]

    def require(self, permission: str) -> None:
        if not self.may(permission):
            raise Forbidden(f"{self.name} may not {permission}")

    @classmethod
    def system(cls) -> "Principal":
        try:
            name = getpass.getuser()
        except Exception:  # noqa: BLE001
            name = "process"
        return cls(name, None, "system")

    @classmethod
    def open_site(cls, name: str | None = None) -> "Principal":
        return cls(name or cls.system().name, None, "open")


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not password:
        raise AuthError("a password cannot be empty")
    salt = os.urandom(16) if salt is None else salt
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return f"{_HASH_VERSION}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        version, salt_hex, digest_hex = stored.split("$")
        if version != _HASH_VERSION:
            return False
        salt = bytes.fromhex(salt_hex)
    except (AttributeError, ValueError):
        return False
    candidate = hashlib.scrypt((password or "").encode("utf-8"), salt=salt, **_SCRYPT)
    return hmac.compare_digest(candidate.hex(), digest_hex)


def _clean(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) > 64 or any(ch.isspace() for ch in cleaned):
        raise AuthError("an account name is one word, up to 64 characters")
    return cleaned.lower()


class Accounts:
    def __init__(self, store: Store, *, clock=time.monotonic):
        self._store = store
        self._clock = clock
        self._failures: dict[str, tuple[int, float]] = {}

    def any(self) -> bool:
        return bool(self._store.users())

    def users(self) -> list[Principal]:
        return [Principal(r["name"], Role(r["role"]), "user", bool(r["active"])) for r in self._store.users()]

    def add(self, name: str, password: str, role: Role | str, *, by: Principal) -> Principal:
        if self.any():
            by.require(USERS_MANAGE)
        name = _clean(name)
        role = Role(str(role).upper())
        if self._store.user(name) is not None:
            raise AuthError(f"there is already an account called {name!r}")
        self._store.save_user(name, hash_password(password), role.value)
        self._store.audit(by.actor, "user.added", name, f"role {role.value}")
        return Principal(name, role, "user")

    def set_password(self, name: str, password: str, *, by: Principal) -> None:
        name = _clean(name)
        if by.origin == "user" and by.name != name:
            by.require(USERS_MANAGE)
        if self._store.user(name) is None:
            raise AuthError(f"no account called {name!r}")
        self._store.update_user(name, password_hash=hash_password(password))
        self._store.audit(by.actor, "user.password_changed", name)

    def set_active(self, name: str, active: bool, *, by: Principal) -> None:
        by.require(USERS_MANAGE)
        name = _clean(name)
        if self._store.user(name) is None:
            raise AuthError(f"no account called {name!r}")
        self._store.update_user(name, active=active)
        self._store.audit(by.actor, "user.enabled" if active else "user.disabled", name)

    def set_role(self, name: str, role: Role | str, *, by: Principal) -> None:
        by.require(USERS_MANAGE)
        name = _clean(name)
        role = Role(str(role).upper())
        row = self._store.user(name)
        if row is None:
            raise AuthError(f"no account called {name!r}")
        self._store.update_user(name, role=role.value)
        self._store.audit(by.actor, "user.role_changed", name, None, before={"role": row["role"]}, after={"role": role.value})

    def seconds_until_allowed(self, name: str) -> float:
        _, not_before = self._failures.get(_clean(name), (0, 0.0))
        return max(0.0, not_before - self._clock())

    def authenticate(self, name: str, password: str) -> Principal:
        name = _clean(name)
        wait = self.seconds_until_allowed(name)
        if wait > 0:
            self._store.audit("anonymous", "login.locked", name, f"{wait:.0f}s to wait")
            raise AuthError(f"too many failed attempts; try again in {wait:.0f} seconds")
        row = self._store.user(name)
        stored = row["password_hash"] if row is not None else hash_password("x", salt=b"\x00" * 16)
        ok = verify_password(password, stored) and row is not None and bool(row["active"])
        if not ok:
            failures, _ = self._failures.get(name, (0, 0.0))
            failures += 1
            delay = LOCKOUT_SECONDS[min(failures - LOCKOUT_AFTER, len(LOCKOUT_SECONDS) - 1)] if failures >= LOCKOUT_AFTER else 0.0
            self._failures[name] = (failures, self._clock() + delay)
            self._store.audit("anonymous", "login.failed", name)
            _log.warning("login failed for %r (%d failure(s))", name, failures)
            raise AuthError("the name or the password is wrong")
        self._failures.pop(name, None)
        principal = Principal(name, Role(row["role"]), "user")
        self._store.audit(principal.actor, "login.succeeded", name)
        return principal

    def principal_for(self, name: str | None, password: str | None) -> Principal:
        """The principal a command runs as: a user, or the open site while none exists."""
        if not self.any():
            return Principal.open_site()
        if name is None:
            raise AuthError("accounts exist; sign in with --as NAME and the password on standard input")
        return self.authenticate(name, password or "")
