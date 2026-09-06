"""The install check: every way a deployment fails quietly, asked out loud."""

from __future__ import annotations

import pytest

from vigil.adapters.keychain import InMemoryBackend, Keychain
from vigil.config import Settings
from vigil.domain.geo import LatLon
from vigil.service.auth import Accounts, Principal, Role
from vigil.service.diagnostics import State, run_checks, worst
from vigil.service.site import SiteService
from vigil.storage.store import Store


def _checks(settings, store, keychain, **kw):
    return {c.name: c for c in run_checks(settings, store, keychain, **kw)}


def test_a_bare_installation_says_what_is_missing_and_how_to_fix_each(tmp_path, keychain):
    settings = Settings(tmp_path / "data", "", None, None, False)
    with Store(tmp_path / "d.db") as store:
        checks = _checks(settings, store, keychain)
        assert checks["data directory"].state is State.OK
        assert checks["database"].state is State.OK and "integrity ok" in checks["database"].detail
        assert checks["site clock"].state is State.OK
        for name in ("accounts", "cameras", "zones", "alerts"):
            assert checks[name].state is State.WARN, name
            assert checks[name].remedy, f"{name} says what is wrong but not what to do"
        assert worst(list(checks.values())) is State.WARN, "nothing here should fail a bare install"
        assert "not probed" in checks["camera sources"].detail


def test_a_site_that_is_ready_passes_and_an_unplaced_camera_is_flagged(tmp_path, reference_video):
    keychain = Keychain(InMemoryBackend())
    settings = Settings(tmp_path / "data", str(tmp_path / "a.log"), None, None, False)
    with Store(tmp_path / "d.db") as store:
        site = SiteService(store, keychain)
        by = Principal.open_site()
        Accounts(store).add("root", "a long password", Role.ADMIN, by=by)
        site.add_camera("gate", str(reference_video), by=by)
        site.add_zone("yard", "Yard", "RESTRICTED", [LatLon(0, 0), LatLon(0, 0.001), LatLon(0.001, 0)], by=by)
        checks = _checks(settings, store, keychain)
        assert checks["accounts"].state is State.OK and checks["zones"].state is State.OK
        assert checks["cameras"].state is State.WARN and "unplaced" in checks["cameras"].detail
        assert checks["alerts"].state is State.OK
        assert checks["keychain"].state is State.OK


def test_a_broken_clock_and_a_disabled_account_are_failures(tmp_path, keychain):
    settings = Settings(tmp_path / "data", "", None, None, False)
    with Store(tmp_path / "d.db") as store:
        store.save_site("Depot", "Mars/Olympus")
        by = Principal.open_site()
        accounts = Accounts(store)
        accounts.add("root", "a long password", Role.ADMIN, by=by)
        accounts.set_active("root", False, by=Principal.system())
        checks = _checks(settings, store, keychain)
        assert checks["site clock"].state is State.FAIL and "time zone" in checks["site clock"].detail
        assert checks["accounts"].state is State.FAIL and "nobody can sign in" in checks["accounts"].detail
        assert worst(list(checks.values())) is State.FAIL


def test_probing_opens_every_camera_and_names_the_one_that_will_not(tmp_path, keychain, reference_video):
    settings = Settings(tmp_path / "data", "", None, None, False)
    with Store(tmp_path / "d.db") as store:
        site = SiteService(store, keychain)
        by = Principal.open_site()
        site.add_camera("gate", str(reference_video), by=by)
        assert _checks(settings, store, keychain, probe=True)["camera sources"].state is State.OK
        site.add_camera("ghost", str(tmp_path / "missing.mp4"), by=by)
        broken = _checks(settings, store, keychain, probe=True)["camera sources"]
        assert broken.state is State.FAIL and "ghost" in broken.detail and broken.remedy


def test_a_check_that_throws_is_itself_a_finding(tmp_path, keychain, monkeypatch):
    from vigil.service import diagnostics

    monkeypatch.setattr(diagnostics, "_disk", lambda settings: 1 / 0)
    settings = Settings(tmp_path / "data", "", None, None, False)
    with Store(tmp_path / "d.db") as store:
        found = [c for c in run_checks(settings, store, keychain) if c.state is State.FAIL]
        assert any("could not run" in c.detail for c in found), "a check that throws must not vanish"


def test_a_remedy_never_uses_a_character_a_windows_console_cannot_print():
    """The prompt is cp1252; an arrow prints as `?`, which reads as a defect."""
    from vigil.service.diagnostics import Check

    line = Check("x", State.WARN, "something", "do this").describe()
    assert "->" in line and line.isascii()
