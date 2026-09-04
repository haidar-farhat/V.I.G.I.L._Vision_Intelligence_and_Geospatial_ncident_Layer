"""Tests for the register behind People and Vehicles.

Almost every test here is about a *refusal* or an *erasure*, because that is
where a register of faces and plates is defensible or is not. Enrolling
something and reading it back is the easy half; the half worth testing is that
nothing gets in without an actor and a basis, that a delete really deletes, and
that a retention clock removes what nobody renewed.

**No model is loaded anywhere in this file, and none exists on the machine that
runs it.** A face template is bytes some encoder produced, so the encoder is
stood in for by `PretendEncoder` below: deterministic, offline, and the same
shape as the operator-supplied SFace it replaces. Every rule the register
enforces is reachable with it, which is the point — logic that could only be
tested with a 90 MB ONNX present would in practice never be tested at all.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from sentinel.registry import (
    Confidence,
    FaceTemplate,
    IdentifierKind,
    Plate,
    PlateFormat,
    Register,
    RegistryError,
    RetentionPolicy,
    SubjectKind,
    create_schema,
)

#: An arbitrary but fixed instant, so ages in these tests are exact.
NOW = 1_700_000_000_000
DAY = 86_400_000


class PretendEncoder:
    """A stand-in for the operator-supplied face encoder.

    The real one is `cv2.FaceRecognizerSF` reading an ONNX the operator
    installs; there is none on this machine and nothing here may download one.
    What the register actually needs from an encoder is a name and some bytes of
    a fixed width, and this produces both — deterministically, so a template
    enrolled twice in a test is byte-identical, which is what the idempotence
    test depends on.
    """

    def __init__(self, name: str = "pretend-sface-v1", dimensions: int = 128):
        self.name = name
        self.dimensions = dimensions

    def template(self, who: str) -> FaceTemplate:
        rng = np.random.default_rng(abs(hash((self.name, who))) % (2**32))
        vector = rng.standard_normal(self.dimensions).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return FaceTemplate(vector=vector.tobytes(), model=self.name, quality=0.87)


@pytest.fixture
def connection() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def register(connection: sqlite3.Connection) -> Register:
    return Register(connection)


@pytest.fixture
def encoder() -> PretendEncoder:
    return PretendEncoder()


def enrol_person(register: Register, encoder: PretendEncoder, **overrides):
    arguments = dict(
        subject_id="person-ali",
        display_name="Ali Hassan",
        identifier=encoder.template("ali"),
        actor="operator:nadia",
        basis="employment contract, staff register",
        now_millis=NOW,
    )
    arguments.update(overrides)
    return register.enrol(**arguments)  # type: ignore[arg-type]


def enrol_vehicle(register: Register, **overrides):
    arguments = dict(
        subject_id="vehicle-van",
        display_name="Contractor van",
        identifier=Plate("B 7421", frames_agreeing=9),
        actor="operator:nadia",
        basis="site access list",
        now_millis=NOW,
    )
    arguments.update(overrides)
    return register.enrol(**arguments)  # type: ignore[arg-type]


# ------------------------------------------------------------------- the schema


def test_the_schema_is_created_on_a_bare_connection_and_again_without_complaint(
    connection: sqlite3.Connection,
):
    Register(connection)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    print("tables after one construction:", sorted(tables))
    assert {
        "register_subjects",
        "register_identifiers",
        "register_sightings",
    } <= tables

    # Twice on the same connection, and once more through the bare function: a
    # console that opens the register on two screens must not fail on the
    # second, and a store migration may have created these already.
    Register(connection)
    create_schema(connection)

    again = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert again == tables


def test_the_schema_survives_a_connection_that_is_already_in_a_transaction():
    # The store hands out a connection that may be mid-transaction, and every
    # write here uses a savepoint for exactly that reason.
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("BEGIN")
    register = Register(connection)
    enrol_vehicle(register)
    connection.execute("COMMIT")

    assert register.find_plate("B-7421") is not None


# ----------------------------------------------------------------- enrolment


def test_a_person_enrolled_with_a_template_is_found_again_with_their_provenance(
    register: Register, encoder: PretendEncoder
):
    enrolment = enrol_person(register, encoder)

    assert enrolment.created_subject is True
    assert enrolment.replaced is False
    assert enrolment.subject.kind is SubjectKind.PERSON
    assert enrolment.subject.pinned is False

    identifier = enrolment.identifier
    assert identifier.kind is IdentifierKind.FACE_TEMPLATE
    assert identifier.enrolled_by == "operator:nadia"
    assert identifier.basis == "employment contract, staff register"
    assert identifier.model == encoder.name

    stored = register.identifiers("person-ali")
    print(f"{len(stored)} identifier(s), template {len(stored[0].template or b'')} bytes")
    assert len(stored) == 1
    # 128 float32s. Measured rather than asserted blind: the point is that the
    # bytes made the round trip, not that a particular encoder was used.
    assert len(stored[0].template or b"") == encoder.dimensions * 4
    assert stored[0].template == identifier.template


def test_a_vehicle_is_found_by_a_plate_written_differently_from_the_one_enrolled(
    register: Register
):
    enrol_vehicle(register, identifier=Plate("B 7421", frames_agreeing=9))

    for written in ("B-7421", "b7421", " b 7421 ", "B.7421"):
        found = register.find_plate(written)
        assert found is not None, f"{written!r} did not find the enrolled van"
        assert found.id == "vehicle-van"

    assert register.find_plate("B7422") is None

    # The raw read is kept beside the normalised form, so an operator asking
    # why the register matched can be shown the characters actually seen.
    identifier = register.identifiers("vehicle-van")[0]
    assert identifier.plate == "B7421"
    assert identifier.raw_text == "B 7421"
    assert identifier.frames_agreeing == 9


def test_arabic_indic_digits_normalise_to_the_same_plate_as_ascii_ones():
    # Unicode's own compatibility decomposition leaves these alone, so without
    # the explicit fold a Beirut plate photographed with Arabic numerals
    # normalises to nothing and is silently unmatchable.
    assert PlateFormat().normalise("ب ٧٤٢١".replace("ب", "B")) == "B7421"


def test_a_plate_with_an_unresolved_character_is_refused_rather_than_completed(
    register: Register
):
    with pytest.raises(RegistryError, match="unresolved"):
        register.find_plate("B?7 4?21")

    with pytest.raises(RegistryError, match="unresolved"):
        enrol_vehicle(register, identifier=Plate("B?7 4?21"))

    assert register.subjects() == ()


def test_enrolment_without_an_actor_is_refused(register: Register, encoder: PretendEncoder):
    for actor in ("", "   "):
        with pytest.raises(RegistryError, match="actor"):
            enrol_person(register, encoder, actor=actor)

    assert register.subjects() == (), "a refused enrolment left a subject behind"


def test_enrolment_without_a_lawful_basis_is_refused(
    register: Register, encoder: PretendEncoder
):
    with pytest.raises(RegistryError, match="lawful basis"):
        enrol_person(register, encoder, basis="  ")

    with pytest.raises(RegistryError, match="lawful basis"):
        enrol_vehicle(register, basis="")

    assert register.subjects() == ()
    rows = register.templates(model=PretendEncoder().name)
    assert rows == (), "a refused enrolment stored a template anyway"


def test_a_face_template_with_no_model_named_is_refused(register: Register):
    # A cosine distance between templates from two different encoders is a
    # well-formed float that means nothing, so an untagged template is refused
    # rather than stored and compared against everything.
    with pytest.raises(RegistryError, match="model"):
        enrol_person(
            register, PretendEncoder(), identifier=FaceTemplate(vector=b"\x00" * 8, model="")
        )


def test_enrolling_the_same_template_twice_refreshes_one_row_rather_than_adding_a_second(
    register: Register, encoder: PretendEncoder
):
    first = enrol_person(register, encoder)
    second = enrol_person(register, encoder, actor="operator:sami", now_millis=NOW + 5000)

    assert second.created_subject is False
    assert second.replaced is True
    assert second.identifier.id == first.identifier.id

    stored = register.identifiers("person-ali")
    print(f"{len(stored)} identifier(s) after two identical enrolments")
    assert len(stored) == 1
    # The provenance is the *latest* deliberate act, because that is the one
    # somebody would be asked about.
    assert stored[0].enrolled_by == "operator:sami"
    assert stored[0].enrolled_at_millis == NOW + 5000


def test_enrolling_a_second_template_adds_to_the_same_person(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    enrol_person(register, encoder, identifier=encoder.template("ali-in-a-hat"))

    stored = register.identifiers("person-ali")
    print(f"{len(stored)} templates for one person")
    assert len(stored) == 2
    assert len(register.subjects(kind=SubjectKind.PERSON)) == 1


def test_enrolling_onto_an_existing_subject_does_not_rename_them(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    again = enrol_person(
        register, encoder, display_name="Someone Else", identifier=encoder.template("second")
    )

    assert again.subject.display_name == "Ali Hassan", (
        "enrolling a second template renamed the person as a side effect"
    )


def test_a_plate_already_enrolled_to_another_subject_is_refused_not_merged(
    register: Register
):
    enrol_vehicle(register)

    with pytest.raises(RegistryError, match="already enrolled"):
        enrol_vehicle(register, subject_id="vehicle-other", display_name="Someone's car")

    # And the refused call left nothing half-written: the savepoint took the
    # subject row back out with the identifier.
    assert register.subject("vehicle-other") is None
    assert len(register.subjects(kind=SubjectKind.VEHICLE)) == 1


def test_a_subject_id_cannot_be_a_person_and_a_vehicle_at_once(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)

    with pytest.raises(RegistryError, match="already a PERSON"):
        register.enrol(
            subject_id="person-ali",
            display_name="Ali Hassan",
            identifier=Plate("C 1234"),
            actor="operator:nadia",
            basis="site access list",
        )


def test_templates_are_only_offered_to_the_encoder_that_produced_them(
    register: Register
):
    sface = PretendEncoder("pretend-sface-v1")
    other = PretendEncoder("pretend-other-v2")
    enrol_person(register, sface)
    enrol_person(
        register, other, subject_id="person-rana", display_name="Rana",
        identifier=other.template("rana"),
    )

    ours = register.templates(model=sface.name)
    theirs = register.templates(model=other.name)
    print(f"{len(ours)} candidate(s) for {sface.name}, {len(theirs)} for {other.name}")
    assert [row.subject_id for row in ours] == ["person-ali"]
    assert [row.subject_id for row in theirs] == ["person-rana"]
    assert register.templates(model="a-model-nobody-installed") == ()


# ------------------------------------------------------------------- sightings


def test_a_sighting_never_enrols_anybody(register: Register):
    with pytest.raises(RegistryError, match="never enrols"):
        register.record_sighting(
            subject_id="person-nobody",
            camera_id="cam-01",
            track_id=7,
            first_seen_millis=NOW,
            last_seen_millis=NOW + 1000,
            confidence=Confidence.MATCH,
            score=0.91,
        )

    assert register.subjects() == ()


def test_a_match_without_its_score_is_refused_and_a_declaration_with_one_is_too(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)

    def sighting(**overrides):
        arguments = dict(
            subject_id="person-ali",
            camera_id="cam-01",
            track_id=7,
            first_seen_millis=NOW,
            last_seen_millis=NOW + 1000,
            confidence=Confidence.MATCH,
            score=0.91,
        )
        arguments.update(overrides)
        return register.record_sighting(**arguments)  # type: ignore[arg-type]

    with pytest.raises(RegistryError, match="score"):
        sighting(score=None)
    with pytest.raises(RegistryError, match="score"):
        sighting(confidence=Confidence.POSSIBLE, score=None)
    with pytest.raises(RegistryError, match="no score"):
        sighting(confidence=Confidence.DECLARED, score=0.4)

    # And the legitimate forms are all accepted, keeping the distinction the
    # console renders differently.
    assert sighting().confidence is Confidence.MATCH
    assert sighting(track_id=8, confidence=Confidence.POSSIBLE, score=0.55).score == 0.55
    assert sighting(track_id=9, confidence=Confidence.DECLARED, score=None).score is None


def test_a_sighting_that_ends_before_it_began_is_refused(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    with pytest.raises(RegistryError, match="before it began"):
        register.record_sighting(
            subject_id="person-ali",
            camera_id="cam-01",
            track_id=7,
            first_seen_millis=NOW + 5000,
            last_seen_millis=NOW,
            confidence=Confidence.DECLARED,
        )


def test_history_comes_back_in_order_and_holds_only_that_subjects_sightings(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    enrol_vehicle(register)

    for camera, track, offset in (
        ("cam-03", 31, 9 * 60_000),
        ("cam-01", 11, 1 * 60_000),
        ("cam-02", 21, 5 * 60_000),
    ):
        register.record_sighting(
            subject_id="person-ali",
            camera_id=camera,
            track_id=track,
            first_seen_millis=NOW + offset,
            last_seen_millis=NOW + offset + 20_000,
            confidence=Confidence.MATCH,
            score=0.9,
        )
    register.record_sighting(
        subject_id="vehicle-van",
        camera_id="cam-01",
        track_id=99,
        first_seen_millis=NOW,
        last_seen_millis=NOW + 30_000,
        confidence=Confidence.DECLARED,
    )

    history = register.history("person-ali")
    cameras = [sighting.camera_id for sighting in history]
    print("history:", cameras)
    assert cameras == ["cam-01", "cam-02", "cam-03"], "the trail was out of order"
    assert [sighting.first_seen_millis for sighting in history] == sorted(
        sighting.first_seen_millis for sighting in history
    )
    assert all(sighting.subject_id == "person-ali" for sighting in history)

    van = register.history("vehicle-van")
    assert [sighting.track_id for sighting in van] == [99]
    assert register.history("person-nobody") == ()


def test_re_recording_a_track_widens_its_window_instead_of_duplicating_it(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)

    def see(first: int, last: int, score: float):
        return register.record_sighting(
            subject_id="person-ali",
            camera_id="cam-01",
            track_id=7,
            first_seen_millis=first,
            last_seen_millis=last,
            confidence=Confidence.MATCH,
            score=score,
        )

    see(NOW + 2_000, NOW + 4_000, 0.71)
    latest = see(NOW + 1_000, NOW + 9_000, 0.93)

    history = register.history("person-ali")
    seconds = latest.duration_millis / 1000
    print(f"{len(history)} row(s) for one track, spanning {seconds:.1f} s")
    assert len(history) == 1, "one track produced two rows in the movement history"
    assert seconds >= 8.0
    assert latest.score == 0.93


# ------------------------------------------------------------------ forgetting


def test_forgetting_a_subject_deletes_everything_and_reports_what_it_deleted(
    register: Register, encoder: PretendEncoder, connection: sqlite3.Connection
):
    enrol_person(register, encoder)
    enrol_person(register, encoder, identifier=encoder.template("ali-in-a-hat"))
    for track in (11, 12, 13):
        register.record_sighting(
            subject_id="person-ali",
            camera_id="cam-01",
            track_id=track,
            first_seen_millis=NOW + track * 1000,
            last_seen_millis=NOW + track * 1000 + 500,
            confidence=Confidence.MATCH,
            score=0.9,
        )

    forgotten = register.forget("person-ali", now_millis=NOW + 60_000)

    print(
        f"forgot {forgotten.identifiers_deleted} identifier(s) and "
        f"{forgotten.sightings_unlinked} sighting(s)"
    )
    assert forgotten.found is True
    assert forgotten.identifiers_deleted == 2
    assert forgotten.sightings_unlinked == 3
    assert forgotten.display_name == "Ali Hassan"
    assert forgotten.kind is SubjectKind.PERSON

    # Asked of the tables directly, not of the API that just claimed to delete.
    remaining = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("register_subjects", "register_identifiers", "register_sightings")
    }
    print("rows left:", remaining)
    assert remaining == {
        "register_subjects": 0,
        "register_identifiers": 0,
        "register_sightings": 0,
    }
    assert register.subject("person-ali") is None
    assert register.history("person-ali") == ()
    assert register.templates(model=encoder.name) == ()


def test_forgetting_somebody_who_is_not_there_is_not_an_error(register: Register):
    # A retried delete must not fail: an operator who sees an error has no way
    # to tell whether the data survived.
    forgotten = register.forget("person-nobody", now_millis=NOW)

    assert forgotten.found is False
    assert forgotten.identifiers_deleted == 0
    assert forgotten.display_name is None


def test_forgetting_one_subject_leaves_the_others_alone(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    enrol_vehicle(register)
    register.record_sighting(
        subject_id="vehicle-van",
        camera_id="cam-01",
        track_id=99,
        first_seen_millis=NOW,
        last_seen_millis=NOW + 1000,
        confidence=Confidence.DECLARED,
    )

    register.forget("person-ali")

    assert register.find_plate("B-7421") is not None
    assert len(register.history("vehicle-van")) == 1


def test_the_audit_detail_carries_ids_and_counts_but_never_the_name_or_the_plate(
    register: Register, encoder: PretendEncoder
):
    # The audit trail is append-only. A name written into it would outlive the
    # erasure it is recording, leaving the log holding the last copy of the
    # thing that was deleted.
    enrolment = enrol_vehicle(register)
    detail = enrolment.detail()
    print("enrol detail:", detail)
    assert "Contractor van" not in detail
    assert "B7421" not in detail and "B 7421" not in detail
    assert "vehicle-van" in detail
    assert enrolment.identifier.id in detail
    assert enrolment.action == "register.enrol"

    forgotten = register.forget("vehicle-van")
    detail = forgotten.detail()
    print("forget detail:", detail)
    assert "Contractor van" not in detail
    assert "B7421" not in detail
    assert '"sightings_unlinked":0' in detail
    assert forgotten.display_name == "Contractor van", (
        "the operator's own confirmation lost the name it is confirming"
    )


# ------------------------------------------------------------------- retention


def test_an_expired_template_is_swept_and_a_pinned_subjects_is_not(
    register: Register, encoder: PretendEncoder
):
    policy = RetentionPolicy(face_template_days=30.0, plate_days=365.0)
    enrol_person(register, encoder)
    enrol_person(
        register,
        encoder,
        subject_id="person-rana",
        display_name="Rana",
        identifier=encoder.template("rana"),
    )
    register.set_pinned("person-rana", True, actor="operator:nadia")

    later = NOW + 40 * DAY
    age_days = (later - NOW) / DAY
    print(f"both templates are {age_days:.1f} days old, policy is {policy.describe()}")
    assert age_days > (policy.face_template_days or 0)

    sweep = register.sweep_expired(later, policy)

    print(
        f"examined {sweep.examined}, deleted {len(sweep.deleted)}, "
        f"kept by a pin {len(sweep.kept_pinned)}"
    )
    assert sweep.examined == 2
    assert [row.subject_id for row in sweep.deleted] == ["person-ali"]
    assert [row.subject_id for row in sweep.kept_pinned] == ["person-rana"]

    assert register.identifiers("person-ali") == ()
    assert len(register.identifiers("person-rana")) == 1
    # The subject survives a sweep that emptied it: deleting the name because a
    # template expired would discard the notes and the history with it, and
    # that is a person's decision rather than a timer's.
    assert register.subject("person-ali") is not None


def test_nothing_inside_its_retention_is_swept(
    register: Register, encoder: PretendEncoder
):
    policy = RetentionPolicy(face_template_days=30.0)
    enrol_person(register, encoder)

    later = NOW + 29 * DAY
    print(f"template is {(later - NOW) / DAY:.1f} days old against a 30 day policy")
    sweep = register.sweep_expired(later, policy)

    assert sweep.deleted == ()
    assert len(register.identifiers("person-ali")) == 1


def test_shortening_the_policy_takes_effect_on_what_is_already_stored(
    register: Register, encoder: PretendEncoder
):
    # Retention is evaluated against enrolled_at at sweep time rather than
    # frozen into the row, so a site tightening its policy actually reaches the
    # oldest templates — the ones the change was made for.
    enrol_person(register, encoder)
    later = NOW + 10 * DAY

    assert register.sweep_expired(later, RetentionPolicy(face_template_days=30.0)).deleted == ()

    sweep = register.sweep_expired(later, RetentionPolicy(face_template_days=7.0))
    print(f"a 7 day policy deleted {len(sweep.deleted)} of {sweep.examined}")
    assert len(sweep.deleted) == 1


def test_faces_and_plates_expire_on_their_own_clocks(register: Register, encoder: PretendEncoder):
    policy = RetentionPolicy(face_template_days=30.0, plate_days=365.0)
    enrol_person(register, encoder)
    enrol_vehicle(register)

    sweep = register.sweep_expired(NOW + 100 * DAY, policy)

    kinds = [row.kind for row in sweep.deleted]
    print("deleted at 100 days:", [kind.value for kind in kinds])
    assert kinds == [IdentifierKind.FACE_TEMPLATE], (
        "the weaker claim expired at the same time as the biometric"
    )
    assert register.find_plate("B 7421") is not None


def test_a_retention_of_none_keeps_an_identifier_and_says_so(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    policy = RetentionPolicy(face_template_days=None)

    sweep = register.sweep_expired(NOW + 10_000 * DAY, policy)

    assert sweep.deleted == ()
    assert "indefinitely" in policy.describe()


def test_a_negative_retention_is_refused_rather_than_read_as_delete_everything():
    with pytest.raises(RegistryError, match="not a duration"):
        RetentionPolicy(face_template_days=-1.0).retention_millis(
            IdentifierKind.FACE_TEMPLATE
        )


def test_unpinning_puts_a_subject_back_under_the_sweep(
    register: Register, encoder: PretendEncoder
):
    enrol_person(register, encoder)
    pinned = register.set_pinned("person-ali", True, actor="operator:nadia")
    assert pinned.changed is True and pinned.subject.pinned is True

    policy = RetentionPolicy(face_template_days=1.0)
    later = NOW + 5 * DAY
    assert register.sweep_expired(later, policy).deleted == ()

    unpinned = register.set_pinned("person-ali", False, actor="operator:nadia")
    assert unpinned.changed is True
    print("pin detail:", unpinned.detail())

    sweep = register.sweep_expired(later, policy)
    assert len(sweep.deleted) == 1


def test_pinning_a_subject_who_is_not_there_is_refused(register: Register):
    # Silently pinning nothing would leave an operator believing a record is
    # protected while the next sweep deletes it.
    with pytest.raises(RegistryError, match="no subject"):
        register.set_pinned("person-nobody", True, actor="operator:nadia")
