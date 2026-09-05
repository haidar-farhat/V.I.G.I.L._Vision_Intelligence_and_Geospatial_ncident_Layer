"""Camera passwords survive a restart, in the operating system's keychain.

The keychain is stood in for by a dictionary here, so the tests prove the
wiring — what is stored where, what is rebuilt, what is forgotten — without
touching the developer's Credential Manager. One test uses the real backend
and is skipped where the machine has none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import secrets
from sentinel.decode import contains_credential
from sentinel.node import Node
from sentinel.redact import REDACTED, redact_url, split_password, with_password
from sentinel.store import Store

SECRET = "hunter2-not-a-real-password"


@pytest.fixture
def keychain():
    fake = secrets.InMemoryBackend()
    secrets.use(fake)
    yield fake
    secrets.use(None)


@pytest.mark.parametrize("url", [
    f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/101",
    f"rtsp://admin:p@ss:w0rd@10.0.0.5/s",
    f"rtsp://:{SECRET}@10.0.0.5/s",
    f"rtsp://user:{SECRET}@[2001:db8::1]:554/s",
    f"rtsp://user:{SECRET}@host:notaport/s",
    f"http://cam/stream?user=admin&password={SECRET}",
])
def test_a_password_splits_out_and_goes_back_in_exactly(url):
    stripped, password = split_password(url)
    assert password
    assert not contains_credential(stripped, url)
    assert stripped == redact_url(url), "the stripped form is the stored form"
    assert with_password(stripped, password) == url


def test_a_url_with_no_password_splits_to_itself():
    assert split_password("rtsp://admin@10.0.0.5/s") == ("rtsp://admin@10.0.0.5/s", None)
    assert split_password("device:0") == ("device:0", None)
    assert with_password("device:0", "x") == "device:0"


def test_the_node_keeps_the_password_in_the_keychain_never_in_the_database(tmp_path: Path, keychain):
    url = f"rtsp://admin:{SECRET}@10.0.0.5:554/s"
    with Node(tmp_path / "n.db") as node:
        record = node.add_camera(url, camera_id="gate")
        assert record.credentials_ref
        assert node.needs_credentials("gate") is False

    with Store(tmp_path / "n.db") as store:
        row = store.cameras()[0]
        assert REDACTED in row["source"] and not contains_credential(row["source"], url)
        assert row["credentials_ref"] == record.credentials_ref
    assert list(keychain.entries.values()) == [SECRET]
    trail = "\n".join(str(dict(r)) for r in Store(tmp_path / "n.db").audit_trail(limit=20))
    assert not contains_credential(trail, url)


def test_a_restarted_node_gets_its_camera_back_with_the_password(tmp_path: Path, keychain):
    url = f"rtsp://admin:{SECRET}@10.0.0.5:554/s"
    with Node(tmp_path / "n.db") as node:
        node.add_camera(url, camera_id="gate")

    with Node(tmp_path / "n.db") as node:
        assert node.needs_credentials("gate") is False, "the password did not come back"
        assert node.camera("gate").source == url
        assert node.camera("gate").display_source == redact_url(url)


def test_without_a_keychain_the_old_behaviour_stands(tmp_path: Path, monkeypatch):
    secrets.use(None)
    monkeypatch.setattr(secrets, "_platform_backend", lambda: None)
    url = f"rtsp://admin:{SECRET}@10.0.0.5:554/s"
    try:
        with Node(tmp_path / "n.db") as node:
            record = node.add_camera(url, camera_id="gate")
            assert record.credentials_ref is None
        with Node(tmp_path / "n.db") as node:
            assert node.needs_credentials("gate") is True
    finally:
        secrets.use(None)


def test_removing_a_camera_forgets_its_password(tmp_path: Path, keychain):
    with Node(tmp_path / "n.db") as node:
        node.add_camera(f"rtsp://admin:{SECRET}@10.0.0.5/s", camera_id="gate")
        assert keychain.entries
        node.remove_camera("gate")
        assert not keychain.entries


def test_a_password_can_be_given_later_for_a_stored_camera(tmp_path: Path, keychain):
    with Node(tmp_path / "n.db") as node:
        node.add_camera("rtsp://admin@10.0.0.5/s", camera_id="gate")
        assert node.needs_credentials("gate") is False
        node.set_password("gate", SECRET)
        assert node.camera("gate").source == f"rtsp://admin:{SECRET}@10.0.0.5/s"
    with Node(tmp_path / "n.db") as node:
        assert node.camera("gate").source == f"rtsp://admin:{SECRET}@10.0.0.5/s"
    with pytest.raises(Exception):
        with Node(tmp_path / "n.db") as node:
            node.set_password("gate", "")


def test_the_real_keychain_round_trips_when_the_machine_has_one():
    secrets.use(None)
    if not secrets.available():
        pytest.skip("no keychain on this machine")
    ref = secrets.new_ref()
    try:
        assert secrets.store(ref, SECRET)
        assert secrets.load(ref) == SECRET
    finally:
        secrets.forget(ref)
    assert secrets.load(ref) is None
