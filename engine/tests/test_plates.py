"""Tests for number-plate reading.

There is no plate detector, no recogniser and no character set on this machine,
and none may be downloaded. That is not a gap in the coverage — it is the
condition the module was designed for. Everything that decides anything here
(normalisation, voting, thresholds, the crop geometry, the CTC decode) is
reachable with no model present, and these tests reach all of it through the two
protocols the reader takes its collaborators as.

Most of these tests are about refusals: the fold that is not made, the character
that stays ``?``, the reading that is resolved but not yet confident. A plate
reader's failure mode is not silence — it is a confident seven-character string
belonging to somebody else's car.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sentinel.core import BoundingBox
from sentinel.plates import (
    CONFIDENT_AGREEMENT,
    GENERIC,
    MIN_AGREEMENT,
    PlateAccumulator,
    PlateError,
    PlateModels,
    PlateRead,
    PlateReader,
    ctc_greedy_decode,
    load_charset,
    normalise,
    softmax,
    supported_countries,
)

ANY_BOX = BoundingBox(0.3, 0.6, 0.1, 0.05)


def make_read(
    text: str,
    *,
    confidences: tuple[float, ...] | None = None,
    frame: int = 0,
    country: str = GENERIC,
) -> PlateRead:
    """A read as the recogniser would have produced it, with no model involved."""
    return PlateRead.from_raw(
        text,
        confidences if confidences is not None else (),
        box=ANY_BOX,
        frame_index=frame,
        country=country,
    )


# ------------------------------------------------------------ normalisation


def test_separators_and_case_fold_so_one_vehicle_is_one_key():
    # The row the vehicles register was specified against: `B 7421` and
    # `B-7421` are one German plate, not two.
    keys = {normalise(text, "DE") for text in ("B 7421", "B-7421", "b.7421", " B7421 ")}

    assert keys == {"B7421"}


def test_a_letter_position_folds_a_read_zero_into_the_letter_it_must_be():
    # Two letters, two digits, three letters. A `0` in the last block cannot be
    # a zero, because that position never holds one.
    assert normalise("AB12 CD0", "UK") == "AB12CDO"


def test_a_digit_position_folds_a_read_letter_into_the_digit_it_must_be():
    assert normalise("ABI2 CDE", "UK") == "AB12CDE"


def test_it_refuses_to_fold_a_confusable_when_no_format_is_known():
    # Without a format there is no position class, so there is no way to tell an
    # O that should be a zero from an O. Case and punctuation still fold.
    assert normalise("ab12-cd0", GENERIC) == "AB12CD0"
    assert normalise("O0I1", GENERIC) == "O0I1"


def test_it_refuses_to_fold_a_country_it_does_not_know():
    # An unknown country is read and kept, never folded to somebody else's
    # format. Inventing a format merges plates the country keeps apart.
    assert "XX" not in supported_countries()
    assert normalise("AB12 CD0", "XX") == "AB12CD0"


def test_it_refuses_to_fold_when_no_shape_of_the_country_fits_the_read():
    # Six characters is not a UK plate. Folding it to the nearest shape would be
    # inventing the format as well as the character.
    assert normalise("AB12C0", "UK") == "AB12C0"


def test_it_refuses_to_fold_a_position_two_fitting_shapes_disagree_about():
    # `BO7421` fits German shapes with two district-and-series letters; a shape
    # putting a letter at position 2 would need `7` to be a letter, so only one
    # shape fits and the fold is sound.
    assert normalise("B07421", "DE") == "BO7421"
    # `B0O421` is fitted by two shapes that disagree about position 2 — three
    # letters then three digits, or two letters then four digits. Position 2 is
    # therefore left exactly as read; the positions both shapes agree on fold.
    assert normalise("B0O421", "DE") == "BOO421"


def test_it_refuses_to_fold_the_confusables_that_are_merely_misread():
    # S/5 is not a font artefact — the glyphs differ plainly on a plate,
    # including in the fonts drawn to be machine-readable. Folding it would hide
    # a wrong read rather than repair a confusable one.
    assert normalise("ABS2CDE", "UK") == "ABS2CDE"
    assert normalise("AB12CDS", "UK") == "AB12CDS"


def test_it_never_folds_an_unresolved_character():
    # The central promise, at the normalisation stage: a character nobody could
    # read is not evidence for the character the format would prefer there.
    folded = normalise("AB?2CD0", "UK")

    assert folded == "AB?2CDO"
    assert folded.count("?") == 1


# ------------------------------------------------------------------ one read


def test_a_read_keeps_the_raw_text_beside_the_normalised_key():
    read = make_read("b-7421", country="DE")

    assert read.raw_text == "b-7421"
    assert read.text == "B7421"


def test_confidences_follow_their_characters_across_a_dropped_separator():
    # The misalignment this is here to prevent is silent: one dropped hyphen and
    # every threshold in the module tests the wrong character.
    read = make_read("B-7421", confidences=(0.9, 0.1, 0.8, 0.7, 0.6, 0.5), country="DE")

    print("normalised", read.text, "confidences", read.char_confidences)
    assert read.text == "B7421"
    assert read.char_confidences == (0.9, 0.8, 0.7, 0.6, 0.5)
    assert read.weakest_confidence == pytest.approx(0.5)


def test_a_read_with_no_reported_confidence_claims_none():
    # Not 1.0. A recogniser that cannot report per character has not reported
    # certainty, and a fabricated one would outvote a measured one.
    read = make_read("AB12CDE")

    assert read.char_confidences == ()
    assert read.weakest_confidence is None


def test_a_confidence_list_that_does_not_match_the_text_is_an_error():
    with pytest.raises(ValueError, match="one to one"):
        make_read("AB12CDE", confidences=(0.9, 0.9))


# --------------------------------------------------------------- the voting


def test_a_character_is_resolved_only_once_enough_reads_agree():
    accumulator = PlateAccumulator(min_agreement=3)

    seen = []
    for frame in range(3):
        accumulator.add(make_read("AB12CDE", frame=frame))
        reading = accumulator.resolve()
        seen.append((len(accumulator), reading.display, reading.weakest_agreement))

    for entry in seen:
        print(entry)

    assert seen[0][1] == "???????"
    assert seen[1][1] == "???????"
    assert seen[2][1] == "AB12CDE"
    assert seen[2][2] >= 3


def test_a_disagreeing_character_stays_unresolved_while_its_neighbours_resolve():
    accumulator = PlateAccumulator(min_agreement=3)
    for frame, text in enumerate(("AB12CDE", "AB12CDE", "AB12XDE")):
        accumulator.add(make_read(text, frame=frame))

    reading = accumulator.resolve()
    print(reading.describe(), reading.agreement)

    assert reading.display == "AB12?DE"
    assert reading.agreement[4] == 0
    assert min(reading.agreement[:4]) >= 3


def test_a_reading_never_completes_an_unresolved_character():
    # `B?7 4?21` must never be presented, matched or exported as `BX7 4921`, so
    # the completed string does not exist as a value while a `?` remains.
    accumulator = PlateAccumulator(min_agreement=3)
    for frame, text in enumerate(("B17421", "B17421", "B27421", "B37421")):
        accumulator.add(make_read(text, frame=frame, country="DE"))

    reading = accumulator.resolve()
    print(reading.describe())

    assert reading.text is None
    assert reading.is_resolved is False
    assert reading.display == "B?7421"
    assert reading.unresolved_count == 1


def test_a_tie_at_the_threshold_does_not_resolve():
    # Four frames called it 8 and four called it B. The count alone would
    # resolve this position; the strict lead is what stops it.
    accumulator = PlateAccumulator(min_agreement=3)
    for frame in range(4):
        accumulator.add(make_read("8B12CDE", frame=frame))
    for frame in range(4, 8):
        accumulator.add(make_read("BB12CDE", frame=frame))

    reading = accumulator.resolve()
    print(reading.describe(), reading.agreement)

    assert reading.display == "?B12CDE"
    assert reading.agreement[0] == 0


def test_the_agreement_count_is_exposed_for_every_character():
    accumulator = PlateAccumulator(min_agreement=2)
    for frame, text in enumerate(("AB12CDE", "AB12CDE", "AB12CDX")):
        accumulator.add(make_read(text, frame=frame))

    reading = accumulator.resolve()
    print(list(zip(reading.characters, reading.agreement)))

    assert len(reading.agreement) == len(reading.characters) == 7
    assert reading.agreement[:6] == (3, 3, 3, 3, 3, 3)
    assert reading.agreement[6] == 2
    assert reading.weakest_agreement == 2


def test_a_character_read_with_low_confidence_does_not_vote():
    # A CTC decoder emits a character for every timestep it does not call blank,
    # including the ones it is barely committed to.
    accumulator = PlateAccumulator(min_agreement=3, min_character_confidence=0.5)
    strong = (0.9,) * 7
    weak_first = (0.2,) + (0.9,) * 6
    for frame, confidences in enumerate((strong, strong, strong, weak_first)):
        accumulator.add(make_read("AB12CDE", confidences=confidences, frame=frame))

    reading = accumulator.resolve()
    print(reading.describe(), reading.agreement)

    assert reading.display == "AB12CDE"
    assert reading.agreement[0] == 3, "the 0.2 character voted"
    assert reading.agreement[1] == 4


def test_reads_of_a_different_length_are_set_aside_rather_than_aligned():
    # Aligning a six-character read against seven-character ones needs a guess
    # about which character was dropped. The count of what was set aside is
    # reported so a track that never settled on a length looks like one.
    accumulator = PlateAccumulator(min_agreement=3)
    for frame, text in enumerate(("AB12CDE", "AB12CDE", "AB12CDE", "AB12CD", "B12CDE")):
        accumulator.add(make_read(text, frame=frame))

    reading = accumulator.resolve()
    print(reading.describe(), "set aside", reading.set_aside_reads)

    assert reading.display == "AB12CDE"
    assert reading.contributing_reads == 3
    assert reading.set_aside_reads == 2
    assert reading.total_reads == 5


# ----------------------------------------------------------- the confidence


def test_a_reading_agreed_by_only_the_minimum_is_resolved_but_not_confident():
    accumulator = PlateAccumulator(min_agreement=MIN_AGREEMENT)
    for frame in range(MIN_AGREEMENT):
        accumulator.add(make_read("AB12CDE", frame=frame))

    reading = accumulator.resolve()
    print("weakest agreement", reading.weakest_agreement, "of", CONFIDENT_AGREEMENT)

    assert reading.is_resolved is True
    assert reading.text == "AB12CDE"
    assert reading.weakest_agreement >= MIN_AGREEMENT
    assert reading.weakest_agreement < CONFIDENT_AGREEMENT
    assert reading.is_confident is False


def test_a_reading_becomes_confident_once_its_weakest_character_has_the_margin():
    accumulator = PlateAccumulator(min_agreement=MIN_AGREEMENT)
    for frame in range(CONFIDENT_AGREEMENT):
        accumulator.add(make_read("AB12CDE", frame=frame))

    reading = accumulator.resolve()
    print(reading.describe())

    assert reading.weakest_agreement >= CONFIDENT_AGREEMENT
    assert reading.is_confident is True


def test_an_unresolved_character_keeps_a_much_agreed_reading_unconfident():
    # Six characters agreed by ten frames is still not a plate if the seventh
    # was never read.
    accumulator = PlateAccumulator(min_agreement=3)
    for frame in range(10):
        text = "AB12CDE" if frame % 2 else "AB12CDX"
        accumulator.add(make_read(text, frame=frame))

    reading = accumulator.resolve()
    print(reading.describe())

    assert reading.display == "AB12CD?"
    assert reading.is_confident is False
    assert reading.text is None


def test_an_empty_accumulator_claims_nothing():
    reading = PlateAccumulator().resolve()

    assert reading.display == ""
    assert reading.text is None
    assert reading.is_confident is False
    assert reading.total_reads == 0
    assert reading.best_read is None


def test_the_clearest_read_is_kept_so_its_crop_can_be_shown():
    # Unresolved characters are drawn with the crop beside them, which needs a
    # frame number and a box to cut it from.
    accumulator = PlateAccumulator(min_agreement=2)
    accumulator.add(make_read("AB12CDE", confidences=(0.6,) * 7, frame=11))
    accumulator.add(make_read("AB12CDE", confidences=(0.95,) * 7, frame=12))

    best = accumulator.resolve().best_read
    print("best read from frame", best.frame_index, "weakest", best.weakest_confidence)

    assert best.frame_index == 12
    assert best.box == ANY_BOX


# ------------------------------------------------------------- the decoder


def test_the_decoder_merges_a_repeated_character_and_keeps_its_best_timestep():
    vocabulary = ("A", "B", "1")
    scores = np.zeros((5, len(vocabulary) + 1), dtype=np.float32)
    scores[0, 0] = 9.0            # blank
    scores[1, 1] = 1.0            # A, weakly
    scores[2, 1] = 6.0            # A again, clearly — merged with the timestep before
    scores[3, 0] = 9.0            # blank
    scores[4, 2] = 3.0            # B

    text, confidences = ctc_greedy_decode(softmax(scores), vocabulary)
    print(text, confidences)

    assert text == "AB"
    assert len(confidences) == 2
    assert confidences[0] > 0.9, "the clearer of the two merged timesteps"
    assert 0.5 < confidences[1] < 1.0


def test_the_decoder_refuses_a_class_the_character_set_does_not_describe():
    # A charset and a set of weights that do not belong together decode into
    # fluent, confident, wrong plates. This is the only place that is catchable.
    scores = np.zeros((2, 5), dtype=np.float32)
    scores[0, 4] = 9.0

    with pytest.raises(PlateError, match="do not belong together"):
        ctc_greedy_decode(softmax(scores), ("A", "B"))


def test_softmax_survives_a_logit_large_enough_to_overflow():
    rows = softmax(np.array([[1000.0, 999.0], [0.0, 0.0]], dtype=np.float64))
    print(rows)

    assert np.all(np.isfinite(rows))
    assert rows.sum(axis=1) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------- the files


@pytest.fixture()
def models(tmp_path: Path) -> PlateModels:
    """Three files standing where the operator's models would be.

    Their contents are never read by anything but the digest and the charset
    loader, because both model calls are injected in these tests. The files
    exist so the presence check is exercised as it would be in an installation.
    """
    detector = tmp_path / "plate-detector.onnx"
    recogniser = tmp_path / "plate-crnn.onnx"
    charset = tmp_path / "charset-uk.txt"
    detector.write_bytes(b"not a model, and never fetched")
    recogniser.write_bytes(b"not a model either")
    charset.write_text("\n".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), encoding="utf-8")
    return PlateModels(detector, recogniser, charset)


def test_a_missing_model_file_raises_and_names_the_file(tmp_path: Path):
    missing = tmp_path / "plate-detector.onnx"
    recogniser = tmp_path / "plate-crnn.onnx"
    charset = tmp_path / "charset.txt"
    recogniser.write_bytes(b"present")
    charset.write_text("A\nB\n", encoding="utf-8")

    with pytest.raises(PlateError) as raised:
        PlateReader(PlateModels(missing, recogniser, charset))

    message = str(raised.value)
    print(message)
    assert "plate-detector.onnx" in message
    assert "nothing is ever downloaded" in message


def test_every_missing_file_is_named_not_just_the_first(tmp_path: Path):
    with pytest.raises(PlateError) as raised:
        PlateModels(
            tmp_path / "a.onnx", tmp_path / "b.onnx", tmp_path / "c.txt"
        ).require_present()

    message = str(raised.value)
    print(message)
    assert "a.onnx" in message and "b.onnx" in message and "c.txt" in message


def test_an_injected_reader_cannot_stand_in_for_a_model_an_installation_lacks(
    tmp_path: Path,
):
    # The presence check runs even when both collaborators are supplied, so a
    # fake in a test — or a stub in a pipeline — can never hide a missing model.
    with pytest.raises(PlateError, match="plate detector"):
        PlateReader(
            PlateModels(tmp_path / "gone.onnx", tmp_path / "gone2.onnx", tmp_path / "g.txt"),
            box_finder=FakeBoxFinder([]),
            text_reader=FakeTextReader("AB12CDE"),
        )


def test_an_empty_character_set_is_an_error_not_an_empty_alphabet(tmp_path: Path):
    charset = tmp_path / "charset.txt"
    charset.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(PlateError, match="empty"):
        load_charset(charset)


def test_the_character_set_keeps_the_order_the_model_was_trained_in(tmp_path: Path):
    charset = tmp_path / "charset.txt"
    charset.write_text("Z\n\nA\n0\n", encoding="utf-8")

    assert load_charset(charset) == ("Z", "A", "0")


# --------------------------------------------------------------- the reader


class FakeBoxFinder:
    """Returns the boxes a test dictated, and records every image it was shown.

    The recording is the point: it is what lets a test prove the detector was
    given a vehicle crop and never the frame.
    """

    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.seen: list[np.ndarray] = []

    def find(self, image):
        self.seen.append(np.array(image, copy=True))
        return list(self.boxes)


class FakeTextReader:
    """Returns dictated characters, and records the shape of each crop it read."""

    def __init__(self, text: str, confidences: tuple[float, ...] = ()):
        self.text = text
        self.confidences = confidences
        self.shapes: list[tuple[int, ...]] = []

    def read_text(self, image):
        self.shapes.append(tuple(image.shape))
        return self.text, self.confidences


def a_frame() -> np.ndarray:
    """A 400x200 frame that is 0 everywhere except inside one vehicle box."""
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[100:150, 100:200] = 200
    return frame


VEHICLE = BoundingBox(0.25, 0.5, 0.25, 0.25)


def test_the_detector_is_shown_the_vehicle_crop_and_never_the_frame(models):
    finder = FakeBoxFinder([])
    reader = PlateReader(models, box_finder=finder, text_reader=FakeTextReader(""))

    reader.read(a_frame(), VEHICLE, frame_index=7)

    shown = finder.seen[0]
    print("frame (200, 400, 3), detector saw", shown.shape)
    assert shown.shape == (50, 100, 3)
    # Everything outside the vehicle box is 0 in this frame, so a crop taken
    # anywhere else would carry a zero.
    assert int(shown.min()) == 200


def test_a_plate_box_comes_back_in_whole_frame_coordinates(models):
    finder = FakeBoxFinder([(BoundingBox(0.2, 0.4, 0.4, 0.4), 0.9)])
    reader = PlateReader(
        models,
        country="UK",
        box_finder=finder,
        text_reader=FakeTextReader("AB12CDE", (0.8,) * 7),
    )

    read = reader.read(a_frame(), VEHICLE, frame_index=7)[0]
    print("plate box", read.box)

    # The vehicle occupies x 100..200, y 100..150 of a 400x200 frame; the plate
    # is 20..60 by 20..40 within that crop.
    assert read.box.x == pytest.approx(0.30)
    assert read.box.y == pytest.approx(0.60)
    assert read.box.w == pytest.approx(0.10)
    assert read.box.h == pytest.approx(0.10)
    assert read.frame_index == 7
    assert read.text == "AB12CDE"


def test_a_vehicle_too_small_to_hold_characters_is_not_read_at_all(models):
    finder = FakeBoxFinder([(BoundingBox(0.1, 0.1, 0.8, 0.8), 0.99)])
    reader = PlateReader(models, box_finder=finder, text_reader=FakeTextReader("AB12CDE"))

    # Four pixels across, at the far end of a car park.
    reads = reader.read(a_frame(), BoundingBox(0.5, 0.5, 0.01, 0.01), frame_index=1)

    assert reads == []
    assert finder.seen == [], "the detector was run on an unreadable crop"


def test_a_plate_box_too_small_to_hold_characters_is_not_recognised(models):
    finder = FakeBoxFinder([(BoundingBox(0.1, 0.1, 0.02, 0.02), 0.99)])
    text_reader = FakeTextReader("AB12CDE")
    reader = PlateReader(models, box_finder=finder, text_reader=text_reader)

    reads = reader.read(a_frame(), VEHICLE, frame_index=1)

    assert reads == []
    assert text_reader.shapes == [], "a two-pixel crop was handed to the recogniser"


def test_a_box_below_the_detector_threshold_is_not_read(models):
    finder = FakeBoxFinder([(BoundingBox(0.2, 0.4, 0.4, 0.4), 0.10)])
    reader = PlateReader(
        models, box_finder=finder, text_reader=FakeTextReader("AB12CDE"),
        min_box_confidence=0.35,
    )

    assert reader.read(a_frame(), VEHICLE, frame_index=1) == []


def test_a_vehicle_box_running_off_the_frame_edge_is_clamped_not_wrapped(models):
    # A negative slice index in numpy means "from the end", so an unclamped crop
    # reads the wrong side of the image instead of failing.
    finder = FakeBoxFinder([])
    reader = PlateReader(models, box_finder=finder, text_reader=FakeTextReader(""))

    reader.read(a_frame(), BoundingBox(-0.2, 0.8, 0.6, 0.6), frame_index=1)

    shown = finder.seen[0]
    print("clamped crop", shown.shape)
    assert shown.shape[0] > 0 and shown.shape[1] > 0
    assert shown.shape[0] <= 200 and shown.shape[1] <= 400


def test_the_reader_records_which_weights_produced_a_reading(models):
    reader = PlateReader(
        models, country="UK", box_finder=FakeBoxFinder([]), text_reader=FakeTextReader("")
    )

    info = reader.info
    print(info.detector_sha256[:16], "charset of", info.charset_size)
    assert len(info.detector_sha256) == 64
    assert info.detector_sha256 != info.recogniser_sha256
    assert info.charset_size == 36
    assert info.country == "UK"


def test_several_frames_of_one_track_collapse_to_one_confident_reading(models):
    # The whole path, with the two models faked: crop, detect, recognise,
    # normalise, vote. `AB12CD0` is read every frame and the UK format resolves
    # its last character to a letter.
    finder = FakeBoxFinder([(BoundingBox(0.2, 0.4, 0.4, 0.4), 0.9)])
    reader = PlateReader(
        models,
        country="UK",
        box_finder=finder,
        text_reader=FakeTextReader("AB12 CD0", (0.9,) * 8),
    )
    accumulator = PlateAccumulator(country="UK")

    frame = a_frame()
    for index in range(CONFIDENT_AGREEMENT):
        accumulator.add_all(reader.read(frame, VEHICLE, frame_index=index))

    reading = accumulator.resolve()
    print(reading.describe())

    assert reading.text == "AB12CDO"
    assert reading.is_confident is True
    assert reading.set_aside_reads == 0
    assert reading.best_read.raw_text == "AB12 CD0"
