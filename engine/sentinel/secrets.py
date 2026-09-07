"""Camera passwords, kept by the operating system rather than by this program.

Until this module existed no secret was persisted at all: a network camera's
password lived in memory for one run and had to be typed again after every
restart, which made an unattended restart of a site with RTSP cameras
impossible — the node came back with cameras it could not open. The database
was given a `credentials_ref` column on the first day and nothing wrote to it.

The design in SECURITY.md is the one built here. The database holds an opaque
handle; the secret itself goes to the operating system's own store through
`keyring` — Windows Credential Manager, the macOS Keychain, the Secret Service
on Linux — which is what a password on a machine is supposed to live in: it is
encrypted with the account, it does not travel with a copied database, and a
`backup` never contains it.

`keyring` is a mature, offline package: it reaches no network, and its
backends are the platform APIs. Where none is usable — a container with no
D-Bus, a headless Linux box with no Secret Service — this says so, stores
nothing, and the camera has to be given its password each start, exactly as
before. Storing a secret somewhere worse than nowhere is not a fallback.

Nothing in this module ever logs a secret, and the handle it returns is random
so a handle in a log or a backup names nothing.
"""

from __future__ import annotations

import secrets as _random
from typing import Protocol

from .logs import get as _get_logger

_log = _get_logger(__name__)

#: What the operating system files these under.
SERVICE = "SentinelVision"


class Backend(Protocol):
    """The three things a keychain must do. `keyring` does them; a test's
    dictionary does them too."""

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class InMemoryBackend:
    """For tests, and for nothing else: forgets everything at process end."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.entries.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.entries[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.entries.pop((service, username), None)


_backend: Backend | None = None
_resolved = False


def use(backend: Backend | None) -> None:
    """Replace the keychain. ``None`` goes back to discovering the platform's."""
    global _backend, _resolved
    _backend = backend
    _resolved = backend is not None


def _platform_backend() -> Backend | None:
    """The operating system's keychain through `keyring`, or ``None``.

    `keyring` picks a backend by priority; the two it falls back to when the
    platform offers nothing — `fail` and `null` — are not stores, and using
    them would look like storing and store nothing.
    """
    try:
        import keyring
        from keyring.backends import fail, null
    except Exception as error:  # noqa: BLE001 - not installed, or broken
        _log.warning("no keychain: keyring is unavailable (%s)", type(error).__name__)
        return None
    try:
        chosen = keyring.get_keyring()
    except Exception as error:  # noqa: BLE001
        _log.warning("no keychain: keyring could not choose a backend (%s)", type(error).__name__)
        return None
    if isinstance(chosen, (fail.Keyring, null.Keyring)):
        _log.warning(
            "no keychain on this machine (%s); camera passwords will not be kept",
            type(chosen).__name__,
        )
        return None
    _log.info("keychain: %s", type(chosen).__name__)
    return chosen


def backend() -> Backend | None:
    global _backend, _resolved
    if not _resolved:
        _backend = _platform_backend()
        _resolved = True
    return _backend


def available() -> bool:
    """Whether a secret given now would still be here after a restart."""
    return backend() is not None


def new_ref() -> str:
    """A handle that names nothing by itself."""
    return "cam-" + _random.token_urlsafe(18)


def store(ref: str, secret: str) -> bool:
    """Keep ``secret`` under ``ref``. Returns whether it was actually kept."""
    keychain = backend()
    if keychain is None:
        return False
    try:
        keychain.set_password(SERVICE, ref, secret)
    except Exception as error:  # noqa: BLE001 - the platform's store refused
        _log.error("the keychain refused to store a credential (%s)", type(error).__name__)
        return False
    return True


def load(ref: str | None) -> str | None:
    if not ref:
        return None
    keychain = backend()
    if keychain is None:
        return None
    try:
        return keychain.get_password(SERVICE, ref)
    except Exception as error:  # noqa: BLE001
        _log.error("the keychain could not be read (%s)", type(error).__name__)
        return None


def forget(ref: str | None) -> None:
    if not ref:
        return
    keychain = backend()
    if keychain is None:
        return
    try:
        keychain.delete_password(SERVICE, ref)
    except Exception:  # noqa: BLE001 - already gone, or refused; nothing to do
        pass
