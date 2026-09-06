"""Camera passwords live in the operating system's keychain under a random handle."""

from __future__ import annotations

import secrets
from typing import Protocol

from ..logs import get as _get_logger

_log = _get_logger(__name__)
SERVICE = "sentinel-vision"


class Backend(Protocol):
    def get_password(self, service: str, ref: str) -> str | None: ...
    def set_password(self, service: str, ref: str, secret: str) -> None: ...
    def delete_password(self, service: str, ref: str) -> None: ...


class InMemoryBackend:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, ref: str) -> str | None:
        return self._data.get((service, ref))

    def set_password(self, service: str, ref: str, secret: str) -> None:
        self._data[(service, ref)] = secret

    def delete_password(self, service: str, ref: str) -> None:
        self._data.pop((service, ref), None)


class Keychain:
    def __init__(self, backend: Backend | None = None):
        self._backend = backend

    @classmethod
    def system(cls) -> "Keychain":
        try:
            import keyring
            from keyring.backends.fail import Keyring as Fail

            if isinstance(keyring.get_keyring(), Fail):
                _log.warning("no usable keychain on this machine; passwords will not be kept")
                return cls(None)
            return cls(keyring)
        except Exception:  # noqa: BLE001
            _log.warning("keyring unavailable; passwords will not be kept")
            return cls(None)

    @property
    def available(self) -> bool:
        return self._backend is not None

    @staticmethod
    def new_ref() -> str:
        return "cam-" + secrets.token_hex(12)

    def store(self, secret: str) -> str | None:
        if self._backend is None:
            return None
        ref = self.new_ref()
        self._backend.set_password(SERVICE, ref, secret)
        return ref

    def load(self, ref: str | None) -> str | None:
        if self._backend is None or not ref:
            return None
        return self._backend.get_password(SERVICE, ref)

    def forget(self, ref: str | None) -> None:
        if self._backend is None or not ref:
            return
        try:
            self._backend.delete_password(SERVICE, ref)
        except Exception:  # noqa: BLE001 - a missing entry is not an error
            pass
