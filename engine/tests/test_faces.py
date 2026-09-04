"""Tests for faces: the switch, the three-way verdict, and the arithmetic.

There is no face model on this machine and none may be downloaded, so every
test here runs against a stand-in backend that returns vectors the test wrote
itself. That is not a compromise. The things that can be wrong in a way that
matters — a feature that runs while it is switched off, a *possible* match read
as a match, a template that scores differently because somebody rescaled it, a
name printed without its score — are all above the model, and this is where they
are held.

Most of these tests are about *not* claiming something: not detecting, not
naming, not being sure. A biometric feature's failures are all in that
direction.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from sentinel.core import BoundingBox
from sentinel.faces import (
    DEFAULT_RETENTION_DAYS,
    FRAMES_FOR_A_MATCH,
    MATCH_SIMILARITY,
    MINIMUM_FACE_PIXELS,
    MINIMUM_FACE_SCORE,
    POSSIBLE_SIMILARITY,
    TEMPLATE_DIMENSIONS,
    EnrolledPerson,
    FaceBox,
    FaceEngine,
    FaceError,
    FaceTemplate,
    TrackIdentity,
    Verdict,
    expired_templates,
    identify_track,
    match,
    similarity,
    verdict_for,
)

DAY_MILLIS = 86_400_000


# ------------------------------------------------------------ the stand-ins


def unit(*components: float) -> tuple[float, ...]:
    """A 128-float vector with these leading components and zeros after.

    Two vectors built from disjoint components are orthogonal, and one built
    from ``(cos t, sin t)`` sits exactly ``t`` away from ``(1, 0)`` — which is
    what lets a test place a face at a chosen similarity instead of hoping a
    random vector lands in the band it wanted.
    """
    vector = [0.0] * TEMPLATE_DIMENSIONS
    for index, value in enumerate(components):
        vector[index] = value
    return tuple(vector)


def template(
    *components: float,
    quality: float = 0.95,
    model: str = "stand-in embedder",
    source: str = "camera-1/track-7",
    created: int = 0,
) -> FaceTemplate:
    return FaceTemplate(
        vector=unit(*components),
        quality=quality,
        model=model,
        source=source,
        created_unix_millis=created,
    )


def at_similarity(target: float) -> FaceTemplate:
    """A template whose similarity with ``template(1.0)`` is ``target``."""
    angle = math.acos(target)
    return template(math.cos(angle), math.sin(angle))


class StandInModels:
    """What YuNet and SFace would return, decided by the test rather than a file.

    Records every image it is handed, which is how the "only inside the person
    box" promise is checked: the assertion is not that the crop looked right,
    it is that the detector was never shown anything else.
    """

    def __init__(self, *faces: tuple[FaceBox, tuple[float, ...]]):
        self._faces = list(faces)
        self.images: list[np.ndarray] = []
        self.embedded: list[FaceBox] = []

    @property
    def name(self) -> str:
        return "stand-in embedder"

    def detect(self, image):
        self.images.append(image)
        return tuple(face for face, _ in self._faces)

    def embed(self, image, face):
        self.embedded.append(face)
        for candidate, vector in self._faces:
            if candidate is face:
                return vector
        raise AssertionError("the engine embedded a face the detector never found")


def face_box(
    *, score: float = 0.99, size: float = 64.0, x: float = 4.0, y: float = 4.0
) -> FaceBox:
    return FaceBox(
        x=x, y=y, w=size, h=size, score=score, raw=tuple([x, y, size, size] + [0.0] * 11)
    )


def frame_with_bright_box(box: BoundingBox, *, width: int = 400, height: int = 300):
    """A dark frame with one bright rectangle exactly where the person is.

    Lets a test prove the crop came from the box rather than from anywhere else,
    by looking at the pixels the detector was given.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[
        int(box.y * height) : int((box.y + box.h) * height),
        int(box.x * width) : int((box.x + box.w) * width),
    ] = 255
    return frame


PERSON = BoundingBox(0.25, 0.50, 0.25, 0.25)


# ------------------------------------------------------ off until turned on


def test_the_feature_is_off_until_somebody_turns_it_on():
    models = StandInModels((face_box(), unit(1.0)))
    engine = FaceEngine(backend=models)
    enrolled = [EnrolledPerson("p1", "Ali", (template(1.0),))]

    assert engine.enabled is False
    assert engine.templates_in(frame_with_bright_box(PERSON), PERSON) == ()
    assert engine.match(template(1.0), enrolled).verdict is Verdict.NONE
    assert engine.identify_track([template(1.0)], enrolled).verdict is Verdict.NONE
    assert engine.identify(
        [(frame_with_bright_box(PERSON), PERSON)], enrolled
    ).verdict is Verdict.NONE

    # Off does not mean "the answer was hidden". Nothing ran at all.
    assert models.images == []
    assert models.embedded == []
    assert engine.info.backend is None
    assert "off" in engine.info.describe()


def test_every_public_entry_point_on_the_engine_is_gated_by_the_switch():
    # Walked rather than listed, because the failure this guards against is a
    # method added next year by somebody who did not read the module docstring.
    gated = {
        name
        for name, member in vars(FaceEngine).items()
        if inspect.isfunction(member) and hasattr(member, "_disabled_result")
    }
    ungated = {
        name
        for name, member in vars(FaceEngine).items()
        if inspect.isfunction(member)
        and not name.startswith("_")
        and name not in gated
    }
    print("gated methods:", sorted(gated))

    assert ungated == set(), f"these run while the feature is off: {sorted(ungated)}"
    assert len(gated) >= 4, "the walk found almost nothing, so it proved nothing"

    # The only public members that are not gated, and both are documented as
    # statements about configuration rather than about anybody's face.
    properties = {
        name
        for name, member in vars(FaceEngine).items()
        if isinstance(member, property) and not name.startswith("_")
    }
    assert properties == {"enabled", "info"}


def test_what_a_gated_method_returns_while_off_claims_nothing():
    for name, member in vars(FaceEngine).items():
        if not inspect.isfunction(member) or not hasattr(member, "_disabled_result"):
            continue
        result = member._disabled_result()
        if isinstance(result, tuple):
            assert result == (), f"{name} returns something while off"
        else:
            assert result.verdict is Verdict.NONE, f"{name} names somebody while off"
            assert result.person_id is None
            assert result.score is None


def test_the_switch_cannot_be_flipped_on_an_engine_that_is_already_running():
    engine = FaceEngine(backend=StandInModels())

    with pytest.raises(AttributeError):
        engine.enabled = True  # type: ignore[misc]


def test_an_engine_that_is_off_never_opens_a_model_file(tmp_path):
    # A site with the feature off must not need face models to exist, which is
    # the state every site starts in.
    engine = FaceEngine(
        detector_model=tmp_path / "not-here.onnx",
        embedder_model=tmp_path / "also-not-here.onnx",
    )

    assert engine.enabled is False


# ------------------------------------------------------------ missing models


def test_a_missing_model_file_is_named_in_the_error(tmp_path):
    missing = tmp_path / "face_detection_yunet.onnx"

    with pytest.raises(FaceError) as raised:
        FaceEngine(
            enabled=True,
            detector_model=missing,
            embedder_model=tmp_path / "face_recognition_sface.onnx",
        )

    message = str(raised.value)
    print("error:", message)
    assert str(missing.resolve()) in message
    assert "supplied by the operator" in message
    assert "nothing is ever downloaded" in message


def test_turning_the_feature_on_with_no_models_says_which_two_are_needed():
    with pytest.raises(FaceError) as raised:
        FaceEngine(enabled=True)

    message = str(raised.value)
    print("error:", message)
    assert "YuNet" in message and "SFace" in message


# --------------------------------------------------------- the three verdicts


def test_identical_faces_score_one_and_read_as_a_match():
    face = template(1.0)
    result = match(face, [EnrolledPerson("p1", "Ali", (template(1.0),))])

    print("score:", result.score)
    assert result.score is not None and result.score >= 0.999
    assert result.score <= 1.0, "a cosine above one is arithmetic, not certainty"
    assert result.verdict is Verdict.MATCH
    assert result.is_match is True
    assert result.person_id == "p1"


def test_orthogonal_faces_are_not_a_match_at_all():
    result = match(template(1.0), [EnrolledPerson("p1", "Ali", (template(0.0, 1.0),))])

    print("score:", result.score)
    assert result.score is not None and abs(result.score) <= 1e-12
    assert result.score < POSSIBLE_SIMILARITY
    assert result.verdict is Verdict.NONE
    # Nothing is claimed, so nobody is named — but the score survives, as the
    # evidence that the silence was reasoned rather than an empty loop.
    assert result.person_id is None and result.name is None
    assert result.compared == 1


def test_a_face_between_the_thresholds_is_possible_and_is_not_a_match():
    midpoint = (MATCH_SIMILARITY + POSSIBLE_SIMILARITY) / 2
    result = match(
        template(1.0), [EnrolledPerson("p1", "Ali", (at_similarity(midpoint),))]
    )

    print("score:", result.score, "thresholds:", POSSIBLE_SIMILARITY, MATCH_SIMILARITY)
    assert result.score is not None
    assert POSSIBLE_SIMILARITY <= result.score < MATCH_SIMILARITY
    assert result.verdict is Verdict.POSSIBLE
    assert result.verdict != Verdict.MATCH
    assert result.is_match is False
    assert result.is_possible is True
    # The name is still carried, because an operator is meant to see it — the
    # hedge is in the verdict and in what may be done with it, not in hiding it.
    assert result.name == "Ali"


def test_a_possible_match_cannot_be_mistaken_for_a_match_by_being_truthy():
    midpoint = (MATCH_SIMILARITY + POSSIBLE_SIMILARITY) / 2
    result = match(
        template(1.0), [EnrolledPerson("p1", "Ali", (at_similarity(midpoint),))]
    )

    # `if match:` is the line this design exists to prevent.
    with pytest.raises(TypeError):
        bool(result)
    with pytest.raises(TypeError):
        bool(result.verdict)


def test_the_boundary_scores_land_on_the_side_their_constants_promise():
    print("at match threshold:", verdict_for(MATCH_SIMILARITY))
    print("just below it:", verdict_for(MATCH_SIMILARITY - 1e-9))
    print("at possible threshold:", verdict_for(POSSIBLE_SIMILARITY))
    print("just below it:", verdict_for(POSSIBLE_SIMILARITY - 1e-9))

    assert verdict_for(MATCH_SIMILARITY) is Verdict.MATCH
    assert verdict_for(MATCH_SIMILARITY - 1e-9) is Verdict.POSSIBLE
    assert verdict_for(POSSIBLE_SIMILARITY) is Verdict.POSSIBLE
    assert verdict_for(POSSIBLE_SIMILARITY - 1e-9) is Verdict.NONE
    assert verdict_for(-1.0) is Verdict.NONE
    assert POSSIBLE_SIMILARITY < MATCH_SIMILARITY, "the band would be empty"


def test_the_best_of_several_candidates_wins():
    register = [
        EnrolledPerson("p1", "Ali", (at_similarity(0.20),)),
        EnrolledPerson("p2", "Rana", (at_similarity(0.90),)),
        EnrolledPerson("p3", "Sami", (at_similarity(0.55),)),
    ]

    result = match(template(1.0), register)

    print("score:", result.score, "compared:", result.compared)
    assert result.score is not None and result.score >= 0.89
    assert result.person_id == "p2"
    assert result.name == "Rana"
    assert result.compared == 3, "somebody was not compared"


def test_the_best_of_one_persons_several_templates_is_the_one_used():
    person = EnrolledPerson(
        "p1", "Ali", (at_similarity(0.10), at_similarity(0.95), at_similarity(0.30))
    )

    result = match(template(1.0), [person])

    print("score:", result.score, "compared:", result.compared)
    assert result.score is not None and result.score >= 0.94
    assert result.compared == 3
    assert result.verdict is Verdict.MATCH


def test_an_empty_register_is_no_match_and_carries_no_score():
    result = match(template(1.0), [])

    assert result.verdict is Verdict.NONE
    assert result.person_id is None
    assert result.compared == 0
    # None, not zero: nobody was rejected, because nobody was looked at.
    assert result.score is None
    assert "nobody is enrolled" in result.describe()


def test_a_person_enrolled_with_no_templates_is_compared_against_nothing():
    result = match(template(1.0), [EnrolledPerson("p1", "Ali", ())])

    assert result.verdict is Verdict.NONE
    assert result.score is None
    assert result.compared == 0


# ------------------------------------------------------------ the template


def test_a_rescaled_vector_is_the_same_face():
    # Two model builds at different output scales must not disagree about a
    # face. Normalisation on construction is what makes that true.
    plain = template(3.0, 4.0)
    scaled = template(300.0, 400.0)

    score = similarity(plain, scaled)
    print("score:", score)
    assert score >= 0.999999
    assert plain.vector == scaled.vector

    length = float(np.linalg.norm(plain.as_array()))
    print("length:", length)
    assert abs(length - 1.0) <= 1e-12


def test_a_template_hands_out_a_copy_and_not_its_own_memory():
    face = template(1.0)
    borrowed = face.as_array()
    borrowed[0] = 99.0

    assert face.vector[0] != 99.0


def test_a_vector_of_the_wrong_length_is_refused():
    with pytest.raises(FaceError) as raised:
        FaceTemplate(
            vector=(1.0, 0.0, 0.0),
            quality=0.9,
            model="stand-in",
            source="camera-1",
            created_unix_millis=0,
        )

    print("error:", raised.value)
    assert str(TEMPLATE_DIMENSIONS) in str(raised.value)


def test_a_vector_with_no_magnitude_is_not_a_weak_face_but_no_face():
    with pytest.raises(FaceError):
        template(0.0)


def test_a_vector_with_a_non_finite_value_is_refused():
    with pytest.raises(FaceError):
        template(float("nan"), 1.0)


def test_a_quality_that_is_not_a_confidence_is_refused():
    with pytest.raises(FaceError):
        template(1.0, quality=1.4)


# -------------------------------------------------- only inside the person box


def test_the_detector_is_shown_the_person_and_never_the_frame():
    models = StandInModels((face_box(), unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)
    frame = frame_with_bright_box(PERSON)

    engine.templates_in(frame, PERSON)

    assert len(models.images) == 1
    shown = models.images[0]
    print("frame:", frame.shape, "shown to the detector:", shown.shape)
    assert shown.shape == (75, 100, 3)
    assert shown.shape != frame.shape
    # Every pixel it saw is inside the bright rectangle, so the crop is the
    # person and not a rectangle that happens to be the right size.
    assert int(shown.min()) == 255


def test_the_api_cannot_be_asked_to_search_a_whole_frame():
    parameter = inspect.signature(FaceEngine.templates_in).parameters["person_box"]

    assert parameter.default is inspect.Parameter.empty, "a default box means the frame"

    engine = FaceEngine(enabled=True, backend=StandInModels())
    with pytest.raises(TypeError):
        engine.templates_in(frame_with_bright_box(PERSON))  # type: ignore[call-arg]


def test_a_pixel_box_is_refused_rather_than_clamped_to_the_frame():
    models = StandInModels((face_box(), unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)

    with pytest.raises(FaceError) as raised:
        engine.templates_in(
            frame_with_bright_box(PERSON), BoundingBox(100.0, 150.0, 100.0, 75.0)
        )

    print("error:", raised.value)
    assert "normalised" in str(raised.value)
    assert models.images == [], "it looked at something before refusing"


def test_a_box_with_no_area_is_refused():
    engine = FaceEngine(enabled=True, backend=StandInModels())

    with pytest.raises(FaceError):
        engine.templates_in(frame_with_bright_box(PERSON), BoundingBox(0.2, 0.2, 0.0, 0.3))


def test_a_person_too_small_to_hold_a_face_is_never_looked_at():
    models = StandInModels((face_box(), unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)
    tiny = BoundingBox(0.5, 0.5, 0.01, 0.01)
    print("box in pixels:", tiny.w * 400, "x", tiny.h * 300)

    assert engine.templates_in(frame_with_bright_box(PERSON), tiny) == ()
    assert models.images == []


# ------------------------------------------------- what becomes a template


def test_a_face_the_detector_is_unsure_of_produces_no_template():
    unsure = face_box(score=MINIMUM_FACE_SCORE - 0.05)
    models = StandInModels((unsure, unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)

    print("detector score:", unsure.score, "floor:", MINIMUM_FACE_SCORE)
    assert engine.templates_in(frame_with_bright_box(PERSON), PERSON) == ()
    assert models.embedded == [], "an unsure face was embedded anyway"


def test_a_face_too_few_pixels_across_produces_no_template():
    small = face_box(size=MINIMUM_FACE_PIXELS - 1)
    models = StandInModels((small, unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)

    print("face size:", small.w, "floor:", MINIMUM_FACE_PIXELS)
    assert engine.templates_in(frame_with_bright_box(PERSON), PERSON) == ()
    assert models.embedded == []


def test_a_template_records_the_model_and_the_track_that_produced_it():
    models = StandInModels((face_box(score=0.97), unit(1.0, 1.0)))
    engine = FaceEngine(enabled=True, backend=models)

    templates = engine.templates_in(
        frame_with_bright_box(PERSON),
        PERSON,
        source="camera-1/track-7",
        timestamp_unix_millis=1_700_000_000_000,
    )

    assert len(templates) == 1
    made = templates[0]
    print("quality:", made.quality, "model:", made.model)
    assert made.model == "stand-in embedder"
    assert made.source == "camera-1/track-7"
    assert made.created_unix_millis == 1_700_000_000_000
    assert abs(made.quality - 0.97) <= 1e-9
    assert len(made.vector) == TEMPLATE_DIMENSIONS


def test_a_model_that_returns_an_unusable_vector_does_not_take_the_camera_down():
    models = StandInModels((face_box(), (0.0,) * TEMPLATE_DIMENSIONS))
    engine = FaceEngine(enabled=True, backend=models)

    assert engine.templates_in(frame_with_bright_box(PERSON), PERSON) == ()


# ------------------------------------------------------- a track, not a frame


def test_a_track_is_identified_from_its_frames_and_not_from_one_lucky_frame():
    register = [EnrolledPerson("p1", "Ali", (template(1.0),))]
    midpoint = (MATCH_SIMILARITY + POSSIBLE_SIMILARITY) / 2
    # Four frames that only ever say "possible", plus one that would say match.
    frames = [at_similarity(midpoint) for _ in range(4)] + [template(1.0)]

    identity = identify_track(frames, register)

    print("median:", identity.score, "best frame:", identity.best_frame_score)
    assert identity.best_frame_score is not None
    assert identity.best_frame_score >= 0.999
    assert identity.score is not None and identity.score < MATCH_SIMILARITY
    assert identity.verdict is Verdict.POSSIBLE
    assert identity.is_match is False


def test_a_track_that_mostly_agrees_is_a_match():
    register = [EnrolledPerson("p1", "Ali", (template(1.0),))]
    frames = [template(1.0), template(1.0), template(1.0), at_similarity(0.1)]

    identity = identify_track(frames, register)

    print("median:", identity.score, "frames:", identity.frames)
    assert identity.score is not None and identity.score >= 0.999
    assert identity.verdict is Verdict.MATCH
    assert identity.person_id == "p1"
    assert identity.frames == 4


def test_too_few_frames_can_never_be_a_match_however_well_they_score():
    register = [EnrolledPerson("p1", "Ali", (template(1.0),))]
    frames = [template(1.0)] * (FRAMES_FOR_A_MATCH - 1)

    identity = identify_track(frames, register)

    print("median:", identity.score, "frames:", identity.frames, "of", FRAMES_FOR_A_MATCH)
    assert identity.score is not None and identity.score >= 0.999
    assert identity.verdict is Verdict.POSSIBLE, "one frame's worth became a name"
    assert identity.name == "Ali"


def test_a_track_with_no_faces_claims_nothing():
    identity = identify_track([], [EnrolledPerson("p1", "Ali", (template(1.0),))])

    assert identity.verdict is Verdict.NONE
    assert identity.score is None
    assert identity.frames == 0
    with pytest.raises(TypeError):
        bool(identity)


def test_a_track_of_a_stranger_names_nobody():
    register = [EnrolledPerson("p1", "Ali", (template(0.0, 1.0),))]
    frames = [template(1.0)] * 5

    identity = identify_track(frames, register)

    print("median:", identity.score)
    assert identity.verdict is Verdict.NONE
    assert identity.person_id is None and identity.name is None
    assert identity.score is not None, "the rejection kept no evidence"


def test_the_engine_aggregates_the_frames_of_a_track_it_was_given():
    models = StandInModels((face_box(), unit(1.0)))
    engine = FaceEngine(enabled=True, backend=models)
    register = [EnrolledPerson("p1", "Ali", (template(1.0),))]
    frames = [(frame_with_bright_box(PERSON), PERSON)] * FRAMES_FOR_A_MATCH

    identity = engine.identify(frames, register, source="camera-1/track-7")

    print("median:", identity.score, "frames:", identity.frames)
    assert identity.frames == FRAMES_FOR_A_MATCH
    assert identity.verdict is Verdict.MATCH
    assert len(models.images) == FRAMES_FOR_A_MATCH


# ----------------------------------------------------- the name and the score


def test_a_name_is_never_shown_without_its_score():
    midpoint = (MATCH_SIMILARITY + POSSIBLE_SIMILARITY) / 2
    register = [EnrolledPerson("p1", "Ali", (at_similarity(midpoint),))]

    possible = match(template(1.0), register).describe()
    certain = match(template(1.0), [EnrolledPerson("p1", "Ali", (template(1.0),))]).describe()
    nothing = match(template(1.0), [EnrolledPerson("p1", "Ali", (template(0.0, 1.0),))]).describe()
    print(possible, "|", certain, "|", nothing)

    for line in (possible, certain):
        assert "Ali" in line
        assert any(character.isdigit() for character in line), "a name with no score"
    assert "possible match" in possible
    assert "possible" not in certain
    # Nothing was claimed, so no name appears at all.
    assert "Ali" not in nothing


def test_a_track_identity_prints_how_many_frames_agreed():
    register = [EnrolledPerson("p1", "Ali", (template(1.0),))]
    line = identify_track([template(1.0)] * 4, register).describe()
    print(line)

    assert "Ali" in line and "4 frame" in line


# --------------------------------------------------------------- retention


def test_templates_older_than_the_policy_are_the_ones_swept():
    now = 100 * DAY_MILLIS
    fresh = template(1.0, created=now - 5 * DAY_MILLIS)
    stale = template(0.0, 1.0, created=now - (DEFAULT_RETENTION_DAYS + 5) * DAY_MILLIS)

    doomed = expired_templates([fresh, stale], now_unix_millis=now)

    print("retention:", DEFAULT_RETENTION_DAYS, "days; swept:", len(doomed))
    assert doomed == (stale,)


def test_a_pinned_person_keeps_every_template():
    now = 100 * DAY_MILLIS
    ancient = template(1.0, created=0)

    assert expired_templates([ancient], now_unix_millis=now) == (ancient,)
    assert expired_templates([ancient], now_unix_millis=now, pinned=True) == ()


def test_a_shorter_site_policy_sweeps_more():
    now = 100 * DAY_MILLIS
    templates = [template(1.0, created=now - days * DAY_MILLIS) for days in (1, 10, 40)]

    default = expired_templates(templates, now_unix_millis=now)
    strict = expired_templates(templates, now_unix_millis=now, retention_days=7)
    print("swept at", DEFAULT_RETENTION_DAYS, "days:", len(default), "at 7 days:", len(strict))

    assert len(default) == 1
    assert len(strict) == 2


# ------------------------------------------------------------- housekeeping


def test_nothing_here_can_build_a_person_out_of_a_sighting():
    # Nobody is enrolled by being seen. The check is that no function in this
    # module returns an EnrolledPerson — enrolment lives where the audit row is.
    import sentinel.faces as faces

    returns_a_person = [
        name
        for name, member in vars(faces).items()
        if callable(member)
        and getattr(inspect.signature(member).return_annotation, "__name__", "")
        == "EnrolledPerson"
    ]
    print("functions returning an EnrolledPerson:", returns_a_person)

    assert returns_a_person == []


def test_the_identity_types_report_the_same_three_verdicts():
    assert {member.name for member in Verdict} == {"MATCH", "POSSIBLE", "NONE"}
    assert isinstance(identify_track([], []), TrackIdentity)
