"""Opening, backing up and restoring the store, for an interface that must not touch storage directly."""

from __future__ import annotations

from pathlib import Path

from ..storage.store import Store, StoreError, restore_backup, verify_backup

__all__ = ["Store", "StoreError", "open_store", "backup", "verify_backup", "restore_backup"]


def open_store(path: str | Path) -> Store:
    return Store(path)


def backup(store: Store, destination: str | Path) -> Path:
    return store.backup_to(destination)
