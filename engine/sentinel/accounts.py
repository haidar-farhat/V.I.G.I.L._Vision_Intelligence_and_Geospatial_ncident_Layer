"""Who is at the console, and what they may do.

Until this module existed every action in the audit trail was attributed to
the literal string ``console``. A chain of custody that cannot name a person
is not one, and the audit's first blocker (SEC-01) said so. What is built here
is the design SECURITY.md has carried from the start: local accounts, a
one-way password hash, and **permission-based** checks — a role is a set of
permissions, and code asks for the permission, never for the role name, so
widening a role cannot open a door somewhere else.

Deliberately small and entirely local:

- Passwords are hashed with `hashlib.scrypt` (in the standard library; no
  dependency) with a random per-user salt. The hash is the one column in the
  schema that is allowed to look like a secret, and it is not one.
- A failed login is remembered per name, in this process only, and each
  failure after the fifth waits longer before the next attempt is even
  checked. That is brute-force protection sized for a console on a desk, not
  for an Internet-facing service, which this is not.
- Logins and failures are audited by *name*. No password, no hash, no
  attempt count reaches the trail.

There is no session token: the console holds the `User` it authenticated for
the life of the window, and the CLI runs as the operating-system account that
launched it, named as such.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from enum import StrEnum

from .logs import get as _get_logger

_log = _get_logger(__name__)


class Role(StrEnum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"


#: Permissions, named for what they allow, checked by name and never by role.
SITE_CONFIGURE = "site.configure"      # cameras, placement, zones, recording, watch list
INCIDENT_EXPORT = "incident.export"
AUDIT_READ = "audit.read"
USERS_MANAGE = "users.manage"
SITE_VIEW = "site.view"

PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({SITE_VIEW}),
    Role.OPERATOR: frozenset({SITE_VIEW, SITE_CONFIGURE, INCIDENT_EXPORT}),
    Role.ANALYST: frozenset({SITE_VIEW, INCIDENT_EXPORT, AUDIT_READ}),
    Role.ADMIN: frozenset({SITE_VIEW, SITE_CONFIGURE, INCIDENT_EXPORT, AUDIT_READ, USERS_MANAGE}),
}

#: Failures before the next attempt starts waiting, and how the wait grows.
LOCKOUT_AFTER = 5
LOCKOUT_SECONDS = (5.0, 15.0, 30.0, 60.0, 300.0)

#: scrypt parameters: 16 MiB of memory, a few tens of milliseconds on a
#: laptop. Enough to make an offline attack on a stolen database expensive;
#: small enough that a login does not feel like one.
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)
_HASH_VERSION = "scrypt1"


class AccountError(ValueError):
    """A request about accounts that cannot be honoured, said plainly."""


@dataclass(frozen=True, slots=True)
class User:
    name: str
    role: Role
    active: bool = True

    @property
    def actor(self) -> str:
        """How this user appears in the audit trail."""
        return f"console:{self.name}"

    def may(self, permission: str) -> bool:
        """Permission, never role: an inactive account may do nothing."""
        return self.active and permission in PERMISSIONS[self.role]


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """``scrypt1$<salt hex>$<hash hex>``. A fresh random salt unless given."""
    if not password:
        raise AccountError("a password cannot be empty")
    salt = salt if salt is not None else os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return f"{_HASH_VERSION}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison against a stored hash; False for anything malformed."""
    try:
        version, salt_hex, digest_hex = stored.split("$")
        if version != _HASH_VERSION:
            return False
        salt = bytes.fromhex(salt_hex)
    except (AttributeError, ValueError):
        return False
    candidate = hashlib.scrypt((password or "").encode("utf-8"), salt=salt, **_SCRYPT)
    return hmac.compare_digest(candidate.hex(), digest_hex)


class Accounts:
    """The users a store holds, and the way to check one.

    Bound to a `Store` rather than owning a connection, for the reasons
    `Store.register` gives: one connection, inside the store's transactions,
    and the audit rows written beside each change land in the same trail.
    """

    def __init__(self, store, *, clock=time.monotonic):
        self._store = store
        self._clock = clock
        #: Per name: (failures, not-before). In memory only — a lockout that
        #: survived a restart would be a way to lock everybody out for good.
        self._failures: dict[str, tuple[int, float]] = {}

    # ---------------------------------------------------------------- users

    def add(self, name: str, password: str, role: Role | str, *, actor: str = "system") -> User:
        name = _clean_name(name)
        role = Role(str(role).upper())
        if self._store.user(name) is not None:
            raise AccountError(f"there is already an account called {name!r}")
        self._store.save_user(name, hash_password(password), role.value, active=True)
        self._store.audit(actor, "user.added", name, f"role {role.value}")
        _log.info("account added: %s (%s)", name, role.value)
        return User(name, role, True)

    def set_password(self, name: str, password: str, *, actor: str = "system") -> None:
        name = _clean_name(name)
        if self._store.user(name) is None:
            raise AccountError(f"no account called {name!r}")
        self._store.set_user_hash(name, hash_password(password))
        self._store.audit(actor, "user.password_changed", name, None)

    def set_active(self, name: str, active: bool, *, actor: str = "system") -> None:
        name = _clean_name(name)
        if self._store.user(name) is None:
            raise AccountError(f"no account called {name!r}")
        self._store.set_user_active(name, active)
        self._store.audit(actor, "user.enabled" if active else "user.disabled", name, None)

    def users(self) -> list[User]:
        return [
            User(row["name"], Role(row["role"]), bool(row["active"]))
            for row in self._store.users()
        ]

    def any(self) -> bool:
        """Whether accounts exist at all. Until one does, nothing is gated."""
        return bool(self._store.users())

    # ------------------------------------------------------------- checking

    def seconds_until_allowed(self, name: str) -> float:
        failures, not_before = self._failures.get(_clean_name(name), (0, 0.0))
        return max(0.0, not_before - self._clock())

    def authenticate(self, name: str, password: str, *, actor: str | None = None) -> User:
        """The user, or an `AccountError` that says only that it failed.

        Never which half was wrong: naming the missing account, or the wrong
        password, is a way to enumerate accounts one login at a time.
        """
        name = _clean_name(name)
        wait = self.seconds_until_allowed(name)
        if wait > 0:
            self._store.audit(actor or "console", "login.locked", name, f"{wait:.0f}s to wait")
            raise AccountError(f"too many failed attempts; try again in {wait:.0f} seconds")
        row = self._store.user(name)
        # The hash is checked even for an unknown name, so the two cases take
        # the same time and cannot be told apart by a stopwatch.
        stored = row["password_hash"] if row is not None else hash_password("x", salt=b"\x00" * 16)
        ok = verify_password(password, stored) and row is not None and bool(row["active"])
        if not ok:
            failures, _ = self._failures.get(name, (0, 0.0))
            failures += 1
            delay = 0.0
            if failures >= LOCKOUT_AFTER:
                delay = LOCKOUT_SECONDS[min(failures - LOCKOUT_AFTER, len(LOCKOUT_SECONDS) - 1)]
            self._failures[name] = (failures, self._clock() + delay)
            self._store.audit(actor or "console", "login.failed", name, None)
            _log.warning("login failed for %r (%d failure(s))", name, failures)
            raise AccountError("the name or the password is wrong")
        self._failures.pop(name, None)
        user = User(name, Role(row["role"]), True)
        self._store.audit(user.actor, "login.succeeded", name, None)
        return user


def _clean_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) > 64 or any(ch.isspace() for ch in cleaned):
        raise AccountError("an account name is one word, up to 64 characters")
    return cleaned.lower()
