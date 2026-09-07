"""Faces, plates and the register — and the refusals that make them shippable.

DECISIONS.md D-08's gate was waived, so nothing here has run against a real
face or plate model and no threshold in it has been measured. What *can* be
tested without a model is the part that actually matters: that the feature is
off, that it refuses to assert a name it has not earned, that a half-read
plate is not a plate, and that erasing somebody erases them.
"""

import json

import numpy as np
import pytest

from vigil.domain.identity import (
    FRAMES_FOR_A_MATCH, IdentityError, Register, Subject, Verdict, could_be_the_same_plate,
    normalise, plate_agrees,
)
from vigil.perception import plates
from vigil.service.auth import Principal, Role
from vigil.service.identity import IdentityDisabled, IdentityService
from vigil.storage.store import Store

ADMIN = Principal("root", Role.ADMIN, "user")
VIEWER = Principal("vic", Role.VIEWER, "user")


@pytest.fixture
def service():
    with Store(":memory:") as store:
        yield IdentityService(store), store


def _vector(seed: int, jitter: float = 0.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=128)
    if jitter:
        base = base + np.random.default_rng(seed + 900).normal(scale=jitter, size=128)
    return normalise(base)


# ------------------------------------------------------------------ the switch


def test_the_feature_is_off_and_nothing_biometric_can_happen(service):
    identity, _store = service
    state = identity.state()
    assert not state.enabled and not state.unbounded
    assert "OFF" in state.describe()
    with pytest.raises(IdentityDisabled):
        identity.require_enabled()
    with pytest.raises(IdentityDisabled):
        identity.enrol("Ana", _vector(1), "sha", basis="x", by=ADMIN)
    with pytest.raises(IdentityDisabled):
        identity.record_face("gate", 1, [_vector(1)], "sha", detector_score=0.9)


def test_turning_it_on_requires_a_reason_and_a_retention_limit(service):
    identity, _store = service
    with pytest.raises(IdentityError, match="needs a reason"):
        identity.enable(30, reason="   ", by=ADMIN)
    with pytest.raises(IdentityError, match="not a retention policy"):
        identity.enable(9999, reason="policy", by=ADMIN)
    with pytest.raises(IdentityError, match="useless"):
        identity.enable(0, reason="policy", by=ADMIN)
    state = identity.enable(30, reason="site policy SP-4", by=ADMIN)
    assert state.enabled and state.retention_days == 30 and not state.unbounded


def test_a_viewer_cannot_turn_it_on(service):
    from vigil.service.auth import AuthError

    identity, _store = service
    with pytest.raises(AuthError):
        identity.enable(30, reason="policy", by=VIEWER)


def test_turning_it_off_does_not_destroy_the_register_unless_asked(service):
    """Turning a feature off and destroying a register are different decisions,
    and one of them cannot be undone."""
    identity, _store = service
    identity.enable(30, reason="policy", by=ADMIN)
    identity.enrol("Ana", _vector(1), "sha", basis="pass 41", by=ADMIN)
    identity.disable(reason="out of hours", by=ADMIN)
    assert identity.counts()["subjects"] == 1, "an experiment with a setting lost the enrolments"

    identity.enable(30, reason="policy", by=ADMIN)
    identity.disable(reason="decommissioned", by=ADMIN, erase=True)
    assert identity.counts()["subjects"] == 0


def test_the_doctor_fails_on_a_site_that_keeps_biometrics_for_ever(service):
    """Loud, because the failure is silent: nothing looks wrong about a face
    table with four million rows in it."""
    from vigil.service.diagnostics import State, _identity

    identity, store = service
    assert _identity(store).state is State.OK
    identity.enable(30, reason="policy", by=ADMIN)
    assert _identity(store).state is State.OK
    # The state a hand-edited database, or a future bug, could reach.
    store.set_identity(True, None)
    check = _identity(store)
    assert check.state is State.FAIL
    assert "for ever" in check.detail and "retention-days" in (check.remedy or "")


# ----------------------------------------------------------------- the claim


def test_one_frame_can_never_assert_a_name():
    """v1 stated this rule in its headline and left a public method that broke
    it. Here it is structural: `compare` has no path to MATCH at all."""
    register = Register()
    register.add(Subject("1", "Ana", _vector(1), "sha"))
    register.add(Subject("2", "Bo", _vector(2), "sha"))

    once = register.compare(_vector(1, jitter=0.05), "sha")
    assert once.verdict is Verdict.POSSIBLE
    with pytest.raises(IdentityError, match="no name to give"):
        once.label

    enough = register.identify([_vector(1, jitter=0.05)] * FRAMES_FOR_A_MATCH, "sha")
    assert enough.verdict is Verdict.MATCH and enough.label == "Ana"

    too_few = register.identify([_vector(1, jitter=0.05)] * (FRAMES_FOR_A_MATCH - 1), "sha")
    assert too_few.verdict is Verdict.POSSIBLE
    assert "not enough to assert a name" in register.explain(
        [_vector(1, jitter=0.05)] * (FRAMES_FOR_A_MATCH - 1), "sha")


def test_two_models_embeddings_are_never_compared():
    """The check v1 asserted at length and never wrote. A distance between two
    encoders' vectors is a meaningless number that looks exactly like a score."""
    register = Register()
    register.add(Subject("1", "Ana", _vector(1), "model-a"))
    assert register.identify([_vector(1)] * 5, "model-b") is None
    assert "different model" in register.explain([_vector(1)] * 5, "model-b")


def test_two_people_who_look_alike_are_refused_rather_than_chosen_between():
    """A register in which the nearest two are both inside the threshold has
    no right answer, and picking is guessing."""
    base = _vector(7)
    register = Register()
    register.add(Subject("1", "Ana", base, "sha"))
    register.add(Subject("2", "Ada", normalise(base + np.full(128, 0.001)), "sha"))
    result = register.identify([base] * 5, "sha")
    assert result.verdict is not Verdict.MATCH
    assert "refused to choose" in register.explain([base] * 5, "sha")


def test_the_wording_is_a_resemblance_and_cannot_be_strengthened():
    register = Register()
    register.add(Subject("1", "Ana", _vector(1), "sha"))
    said = register.identify([_vector(1, jitter=0.03)] * 5, "sha").describe()
    assert "resembles" in said and "not an identification" in said
    assert " is Ana" not in said


def test_an_unnormalised_embedding_is_refused_at_the_door():
    """Cosine distance is only a distance for unit vectors."""
    register = Register()
    with pytest.raises(IdentityError, match="normalise it first"):
        register.add(Subject("1", "Ana", np.full(128, 5.0), "sha"))
    with pytest.raises(IdentityError, match="no length"):
        normalise(np.zeros(128))


# ----------------------------------------------------------------- the plate


def test_a_half_read_plate_is_not_a_plate():
    """The best idea in v1, kept: the completed string never exists as a value,
    so nothing downstream can treat it as one."""
    accumulator = plates.Accumulator()
    for i in range(4):
        accumulator.add(plates.Read("ABC123", (0.9,) * 6, i))
    for i in range(4):
        accumulator.add(plates.Read("ABC1B3", (0.9,) * 6, i + 4))
    reading = accumulator.resolve()
    assert reading.text is None, "a 4-4 split resolved to a plate"
    assert reading.display == "ABC1?3"
    assert not reading.resolved and not reading.confident
    assert "Not a plate" in reading.describe()


def test_a_character_needs_the_count_and_the_lead():
    """The count alone resolves a position four frames called 8 and four B."""
    clean = plates.Accumulator()
    for i in range(5):
        clean.add(plates.Read("XY7", (0.9,) * 3, i))
    assert clean.resolve().text == "XY7"

    thin = plates.Accumulator()
    for i in range(2):
        thin.add(plates.Read("XY7", (0.9,) * 3, i))
    assert thin.resolve().text is None, "two reads is one observation counted twice"


def test_reads_of_another_length_are_set_aside_and_counted():
    accumulator = plates.Accumulator()
    for i in range(5):
        accumulator.add(plates.Read("ABC123", (0.9,) * 6, i))
    for i in range(3):
        accumulator.add(plates.Read("ABC12", (0.9,) * 5, i + 5))
    reading = accumulator.resolve()
    assert reading.text == "ABC123" and reading.set_aside == 3
    assert "3 read(s) of a different length set aside" in reading.describe()


def test_a_model_that_cannot_report_confidence_does_not_get_a_free_vote():
    """v1's docstring said an empty tuple was better than a fabricated 1.0 and
    then let it skip the threshold, which is what a 1.0 would have done."""
    accumulator = plates.Accumulator()
    for i in range(6):
        accumulator.add(plates.Read("ABC123", (), i))
    assert accumulator.resolve().text is None


def test_the_reading_reports_its_weakest_character_not_its_mean():
    accumulator = plates.Accumulator()
    for i in range(5):
        accumulator.add(plates.Read("AB1", (0.95, 0.93, 0.55), i))
    reading = accumulator.resolve()
    assert reading.text == "AB1"
    assert abs(reading.weakest_confidence - 0.55) < 1e-9, "the mean would have hidden it"


def test_probabilities_are_not_softmaxed_twice():
    """v1 did, which flattened every confidence under the threshold so no
    position ever resolved — silently, and totally."""
    already = np.full((4, 5), 0.0025)
    already[:, 1] = 0.99
    assert plates.probabilities(already)[0, 1] == pytest.approx(0.99)

    logits = np.tile(np.array([0.0, 10.0, 0.0, 0.0, 0.0]), (4, 1))
    assert plates.probabilities(logits)[0, 1] > 0.99


def test_the_ctc_blank_can_sit_anywhere_in_the_vocabulary():
    """v1's mapping was off by one for every class above a blank in the middle
    — fluent, confident, systematically wrong plates."""
    vocabulary = ("A", "B", "C")
    for blank in (0, 1, 3):
        probs = np.full((3, 4), 0.01)
        wanted = [i for i in range(4) if i != blank]
        for step, index in enumerate(wanted[:3]):
            probs[step, index] = 0.97
        text, confidences = plates.ctc_decode(probs, vocabulary, blank)
        assert text == "ABC", f"blank at {blank} decoded {text!r}"
        assert len(confidences) == 3

    with pytest.raises(plates.PlateError, match="wrong length"):
        plates.ctc_decode(np.full((3, 9), 0.1), vocabulary, 0)


def test_two_readings_that_differ_only_where_an_ocr_confuses():
    # Genuinely ambiguous to a reader, so the same vehicle read twice.
    assert could_be_the_same_plate("ABC1O3", "ABC103")
    assert could_be_the_same_plate("ABCI23", "ABC123")
    # A different character is a different vehicle.
    assert not could_be_the_same_plate("ABC123", "ABC124")
    assert not could_be_the_same_plate("ABC123", "ABC12")


def test_glyphs_that_are_plainly_different_are_not_treated_as_confusable():
    """8 and B are distinct on a plate typeface. An OCR that disagreed about
    them made a mistake, and calling it an ambiguity turns a wrong reading
    into a confident one — hiding the misread instead of repairing a
    confusable, which is the opposite of the point."""
    from vigil.domain.identity import CONFUSABLE, REFUSED_FOLDS

    folded = {frozenset(p) for p in CONFUSABLE}
    for pair in REFUSED_FOLDS:
        assert frozenset(pair) not in folded, f"{pair} must not be folded"
    assert not could_be_the_same_plate("ABC8 23".replace(" ", ""), "ABCB23")
    assert not could_be_the_same_plate("AB5123", "ABS123")
    assert plate_agrees("ABC123", "ABC123") == 1.0
    assert 0.0 < plate_agrees("ABC123", "ABC124") < 1.0


# ------------------------------------------------------------------- erasure


def test_forgetting_somebody_takes_their_observations_with_them(service):
    identity, store = service
    identity.enable(30, reason="policy", by=ADMIN)
    subject = identity.enrol("Ana", _vector(1), "sha", basis="pass 41", by=ADMIN)
    for _ in range(4):
        identity.record_face("gate", 1, [_vector(1, jitter=0.02)] * 4, "sha", detector_score=0.9)
    assert identity.counts()["faces"] == 4

    erased = identity.forget(subject, by=ADMIN)
    assert erased["subject"] == 1 and erased["faces"] == 4
    assert identity.counts() == {"subjects": 0, "faces": 0, "plates": 0}

    # Idempotent: a retried erasure must not fail in a way that leaves an
    # operator unsure whether the data survived.
    assert identity.forget(subject, by=ADMIN)["subject"] == 0


def test_an_unmatched_face_is_still_recorded_so_it_can_be_deleted(service):
    """A face that matched nobody is still a face that was processed. Not
    recording it would leave the retention sweep nothing to delete."""
    identity, _store = service
    identity.enable(30, reason="policy", by=ADMIN)
    identity.enrol("Ana", _vector(1), "sha", basis="pass 41", by=ADMIN)
    result = identity.record_face("gate", 2, [_vector(50)] * 4, "sha", detector_score=0.8)
    assert result.verdict is Verdict.NONE
    assert identity.counts()["faces"] == 1


def test_observations_expire_and_enrolments_do_not(service):
    """Deleting a subject on a timer would be a different feature, and a
    surprising one: somebody put them there deliberately."""
    identity, _store = service
    identity.enable(1, reason="policy", by=ADMIN)
    identity.enrol("Ana", _vector(1), "sha", basis="pass 41", by=ADMIN)
    identity.record_face("gate", 1, [_vector(1)] * 4, "sha", detector_score=0.9, at_millis=1000)
    assert identity.counts()["faces"] == 1

    removed = identity.sweep(now_millis=1000 + 3 * 86_400_000)
    assert removed["faces"] == 1
    assert identity.counts() == {"subjects": 1, "faces": 0, "plates": 0}


def test_nothing_expires_without_a_limit_and_the_sweep_says_so(service):
    identity, store = service
    identity.enable(30, reason="policy", by=ADMIN)
    store.set_identity(True, None)
    assert identity.sweep() == {}


# --------------------------------------------------------------- the audit


def test_every_consequential_act_is_audited_and_no_name_reaches_the_trail(service):
    """The audit trail is append-only, so a name written into it outlives the
    erasure it was recording."""
    identity, store = service
    identity.enable(30, reason="site policy SP-4", by=ADMIN)
    subject = identity.enrol("Ana Halabi", _vector(1), "sha", basis="contractor pass 41", by=ADMIN)
    identity.forget(subject, by=ADMIN)
    identity.disable(reason="finished", by=ADMIN)

    rows = store.audit_trail()
    actions = [r["action"] for r in rows]
    for expected in ("identity.enabled", "identity.enrolled", "identity.forgotten",
                     "identity.disabled"):
        assert expected in actions, f"{expected} left no trace"

    trail = json.dumps([dict(r) for r in rows])
    assert "Ana Halabi" not in trail, "a name reached the append-only audit trail"
    assert subject in trail, "the identifier should be there so the act can be traced"
    assert "contractor pass 41" in trail, "the basis is the point of recording it"


def test_the_register_is_assembled_fresh_every_time(service):
    """A cached register would be a second copy of the site's biometrics,
    outliving the erasure that was meant to remove them."""
    identity, _store = service
    identity.enable(30, reason="policy", by=ADMIN)
    subject = identity.enrol("Ana", _vector(1), "sha", basis="pass", by=ADMIN)
    assert len(identity.register()) == 1
    identity.forget(subject, by=ADMIN)
    assert len(identity.register()) == 0


# ------------------------------------------------------------------- faces


def test_a_face_reader_cannot_be_built_without_its_models_and_names_each():
    """"A model is missing" sends an installer looking through two paths."""
    from pathlib import Path

    from vigil.perception.faces import FaceError, FaceModels, FaceReader

    models = FaceModels(Path("no-such-detector.onnx"), Path("no-such-embedder.onnx"))
    assert len(models.missing()) == 2
    with pytest.raises(FaceError) as raised:
        FaceReader(models)
    said = str(raised.value)
    assert "no-such-detector.onnx" in said and "no-such-embedder.onnx" in said
    assert "nothing is downloaded" in said


def test_a_runaway_person_box_is_refused_rather_than_clamped_to_the_frame():
    """The v1 hole, named. A box of width 200.0 in a normalised frame passed
    an x/y-only check and handed the face detector the entire picture — which
    is the exact thing requiring a person box exists to prevent. Clamping is
    not rejecting."""
    from vigil.domain.detection import BoundingBox
    from vigil.perception.faces import _crop
    from vigil.perception.plates import _crop as plate_crop

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    assert _crop(frame, BoundingBox(0.0, 0.0, 200.0, 150.0), 64)[0] is None
    assert _crop(frame, BoundingBox(-5.0, 0.0, 6.0, 0.9), 64)[0] is None
    assert plate_crop(frame, BoundingBox(0.0, 0.0, 200.0, 150.0)) is None

    # A real box at the frame edge, overshooting by a float hair, still works.
    crop, origin = _crop(frame, BoundingBox(0.0, 0.0, 1.0000001, 1.0), 64)
    assert crop is not None and origin == (0, 0)

    # And a genuinely small crop is refused for being small, not for being odd.
    assert _crop(frame, BoundingBox(0.4, 0.4, 0.02, 0.02), 64)[0] is None


def test_a_face_is_blurred_beyond_recognition_on_export():
    """A blur rather than a black rectangle: an exported clip should still
    show that somebody was there and what they did, which is usually the point
    of the clip."""
    from vigil.domain.detection import BoundingBox
    from vigil.perception.faces import blur

    rng = np.random.default_rng(4)
    frame = np.full((200, 300, 3), 128, dtype=np.uint8)
    frame[40:140, 90:210] = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    box = BoundingBox(90 / 300, 40 / 200, 120 / 300, 100 / 200)

    before = float(frame[40:140, 90:210].var())
    after = float(blur(frame, [box])[40:140, 90:210].var())
    assert after < before / 20, f"detail survived: {before:.0f} -> {after:.0f}"
    # Outside the box is untouched, so the rest of the evidence is intact.
    assert (blur(frame, [box])[:30, :30] == frame[:30, :30]).all()
    assert blur(frame, []) is frame
