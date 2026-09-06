"""What a site watches for is a setting, not a flag somebody has to remember."""

import pytest

from vigil.service.auth import Forbidden, Principal, Role
from vigil.service.detection import (CONFIDENCE_CEILING, CONFIDENCE_FLOOR, DetectionError, DetectionSettings,
                                     detector_factory)
from vigil.service.site import SiteService
from vigil.storage.store import Store

OPERATOR = Principal("alice", Role.OPERATOR, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


def test_a_setting_nobody_could_satisfy_is_refused_where_it_is_typed():
    """Hours later it looks like a broken camera, which is the wrong thing to fix."""
    with pytest.raises(DetectionError):
        DetectionSettings.checked(["person"], CONFIDENCE_CEILING + 0.01)
    with pytest.raises(DetectionError):
        DetectionSettings.checked(["person"], CONFIDENCE_FLOOR - 0.01)
    with pytest.raises(DetectionError, match="not a label"):
        DetectionSettings.checked(["person; DROP TABLE"], None)

    chosen = DetectionSettings.checked([" Person ", "CAR", ""], 0.6)
    assert chosen.labels == frozenset({"person", "car"}) and chosen.confidence == 0.6
    assert "person" in chosen.describe() and "0.60" in chosen.describe()


def test_nothing_chosen_means_the_built_in_list_and_says_so():
    default = DetectionSettings()
    assert default.labels is None and default.confidence is None
    assert "built-in" in default.describe() and "the detector's own threshold" in default.describe()
    assert "person" in default.classes(), "the built-in list is what a detector is actually asked for"


def test_a_flag_overrides_the_stored_setting_for_one_run_and_absent_flags_keep_it():
    stored = DetectionSettings.checked(["person"], 0.6)
    assert stored.override(None, None) is stored
    assert stored.override(["car"], None) == DetectionSettings(frozenset({"car"}), 0.6)
    assert stored.override(None, 0.8) == DetectionSettings(frozenset({"person"}), 0.8)


def test_the_setting_survives_a_restart_naming_the_site_and_is_audited(keychain):
    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        assert site.detection() == DetectionSettings(), "a fresh site watches the built-in list"

        chosen = site.set_detection(["Person", "car"], 0.65, by=OPERATOR)
        assert chosen.labels == frozenset({"person", "car"})
        assert site.detection() == chosen, "it must survive a re-read"

        # Naming or re-clocking the site writes the same row, and must not
        # take the watch list with it. `INSERT ... VALUES (…)` did exactly
        # that to the threat labels once.
        site.name_site("Depot", "UTC", by=OPERATOR)
        assert site.detection() == chosen
        assert site.threats().labels == ()

        row = [r for r in store.audit_trail() if r["action"] == "site.detection_changed"][0]
        assert '"labels": []' in row["before"] and "person" in row["after"]

        assert site.set_detection([], None, by=OPERATOR) == DetectionSettings()
        with pytest.raises(Forbidden):
            site.set_detection(["person"], None, by=VIEWER)


def test_without_a_model_the_factory_says_no_watch_list_can_apply():
    """Motion detection answers "what changed", so a class list is a lie there."""
    factory = detector_factory(None, DetectionSettings.checked(["person"], 0.6))
    assert "motion only" in factory.describe()
    detector = factory()
    assert not detector.info.classifies


def test_the_doctor_fails_when_the_model_cannot_produce_a_watched_label(tmp_path, keychain):
    """A watch list the model cannot satisfy is a site that sees nothing, silently."""
    from vigil.service.diagnostics import State, _detection

    class _Settings:
        def __init__(self, model):
            self._model = model

        def default_model(self):
            return self._model

    with Store(":memory:") as store:
        site = SiteService(store, keychain)
        assert _detection(_Settings(None), store).state is State.OK

        site.set_detection(["person"], None, by=OPERATOR)
        without = _detection(_Settings(None), store)
        assert without.state is State.WARN and "cannot watch a class" in without.detail

        model = tmp_path / "not-a-model.onnx"
        model.write_bytes(b"not an onnx file")
        assert _detection(_Settings(model), store).state is State.WARN, "an unreadable model is the model check's job"
