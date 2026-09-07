"""Tests for post-hoc re-identification and the fragmentation measurement.

The laptop camera counted 3, 10, 4 and 11 "distinct objects" over twenty
seconds of one person. This file is where the two answers to that are held to
account: the measurement that says how bad fragmentation actually is under
controlled conditions, and the linker that reconciles the count afterwards.

The linker's tests are built by hand, the way `test_incidents.py` builds its
events, because each of its three conditions has to be shown to refuse on its
own. A red coat leaving and a red coat arriving are two people, and the test
that proves it is the one that matters most here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import scene
from sentinel.core import (
    BoundingBox,
    Detection,
    LatLon,
    PositionEstimate,
    Track,
    destination_point,
)
from sentinel.reid import (
    DEFAULT_MAX_GAP_MILLIS,
    DEFAULT_MIN_SIMILARITY,
    EMA_ALPHA,
    Appearance,
    AppearanceLedger,
    TrackAppearance,
    link_fragments,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import measure_fragmentation  # noqa: E402

SITE = LatLon(33.8938, 35.5018)

RED = (30, 30, 200)
BLUE = (200, 40, 30)
GREY = (90, 90, 90)
WHITE = (230, 230, 230)


# ------------------------------------------------------------------ helpers


def patch(colour: tuple[int, int, int], seed: int, size=(120, 160)) -> np.ndarray:
    """A flat colour with sensor-like noise, so no two patches are identical."""
    rng = np.random.default_rng(seed)
    image = np.full((*size, 3), colour, dtype=np.uint8)
    return np.clip(image.astype(np.float32) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)


def frame(*patches: tuple[tuple[int, int, int], int, int], background=(70, 70, 70)) -> np.ndarray:
    """A 640x480 frame with 160x120 patches at the given top-left pixels."""
    image = np.full((480, 640, 3), background, dtype=np.uint8)
    for index, (colour, x, y) in enumerate(patches, start=1):
        image[y : y + 120, x : x + 160] = patch(colour, index)
    return image


def box(x: int, y: int, w: int = 160, h: int = 120) -> BoundingBox:
    return BoundingBox(x / 640, y / 480, w / 640, h / 480)


def placed(bearing: float, metres: float, radius: float = 1.0) -> PositionEstimate:
    return PositionEstimate(
        point=destination_point(SITE, bearing, metres), radius_meters=radius,
        source="GROUND_PROJECTION",
    )


def make_track(
    track_id: int, bbox: BoundingBox, first: int, last: int,
    position: PositionEstimate | None = None,
) -> Track:
    return Track(
        id=track_id, class_id=9999, bbox=bbox, confidence=0.8, hits=5,
        first_seen_millis=first, last_seen_millis=last, position=position,
        speed_mps=None, heading_degrees=None,
    )


def fragment(
    track_id: int,
    *,
    first: int,
    last: int,
    histogram: np.ndarray | None,
    camera: str = "cam-01",
    first_position: PositionEstimate | None = None,
    last_position: PositionEstimate | None = None,
    first_box: BoundingBox = box(200, 200),
    last_box: BoundingBox = box(200, 200),
    samples: int = 10,
) -> TrackAppearance:
    """A track's record as the ledger would have built it, written by hand."""
    return TrackAppearance(
        camera_id=camera, track_id=track_id,
        first_seen_millis=first, last_seen_millis=last, last_confirmed_millis=last,
        first_box=first_box, last_box=last_box,
        first_position=first_position, last_position=last_position,
        histogram=None if histogram is None else histogram.copy(),
        samples=samples if histogram is not None else 0,
    )


@pytest.fixture(scope="module")
def looks() -> dict[str, np.ndarray]:
    """Descriptors for a red and a blue object, taken from real pixels."""
    image = frame((RED, 40, 40), (BLUE, 400, 40))
    red = Appearance.of(image, box(40, 40))
    blue = Appearance.of(image, box(400, 40))
    assert red is not None and blue is not None
    return {"red": red.histogram, "blue": blue.histogram}


# --------------------------------------------------------------- descriptor


def test_the_histogram_does_not_care_where_the_box_is():
    # The same red object at the top-left and the bottom-right of the frame.
    # A descriptor that leaked position would fail to recognise an object that
    # walked across the frame, which is the whole case for having one.
    image = frame((RED, 40, 40), (RED, 400, 300))
    here = Appearance.of(image, box(40, 40))
    there = Appearance.of(image, box(400, 300))

    assert here is not None and there is not None
    # Measured at 0.99999; the noise on the two patches differs, nothing else.
    assert here.similarity(there) > 0.99


def test_the_histogram_tells_red_from_blue():
    image = frame((RED, 40, 40), (BLUE, 400, 40))
    red = Appearance.of(image, box(40, 40))
    blue = Appearance.of(image, box(400, 40))

    assert red is not None and blue is not None
    # Measured at 0.62. Not lower because the value channel — half the mass —
    # sees two equally bright objects; hue and saturation do the separating.
    assert red.similarity(blue) < 0.7
    assert red.similarity(blue) < DEFAULT_MIN_SIMILARITY


def test_the_value_channel_tells_grey_from_white():
    # Neither has a hue. Without the value bins these two would be the same
    # object, and most clothing on most cameras has no hue either.
    image = frame((GREY, 40, 40), (WHITE, 400, 40))
    grey = Appearance.of(image, box(40, 40))
    white = Appearance.of(image, box(400, 40))

    assert grey is not None and white is not None
    # Measured at 0.06.
    assert grey.similarity(white) < 0.2


def test_a_mask_keeps_the_background_out_of_the_descriptor():
    """The same red object on a grey wall and on a white one.

    Over the box, the wall is a third of the sample and the two descriptors
    disagree. Over the mask, only the object is sampled and they agree — which
    is what makes a silhouette worth carrying into the descriptor at all.
    """
    on_grey = frame((RED, 40, 40), background=(70, 70, 70))
    on_white = frame((RED, 40, 40), background=(230, 230, 230))
    # A box a third wider than the object, so the wall is in it.
    wide = box(40, 40, 240, 120)
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[:, :160] = 1

    boxed_grey = Appearance.of(on_grey, wide)
    boxed_white = Appearance.of(on_white, wide)
    masked_grey = Appearance.of(on_grey, wide, mask)
    masked_white = Appearance.of(on_white, wide, mask)
    assert None not in (boxed_grey, boxed_white, masked_grey, masked_white)

    assert boxed_grey.source == "box" and masked_grey.source == "mask"
    assert masked_grey.pixels == 160 * 120
    assert masked_grey.similarity(masked_white) > 0.99
    assert boxed_grey.similarity(boxed_white) < masked_grey.similarity(masked_white)


def test_a_mask_with_almost_nothing_in_it_falls_back_to_the_box():
    # A speck of activation is not a silhouette. Sampling four pixels would
    # produce a descriptor that links to anything of roughly that colour.
    image = frame((RED, 40, 40))
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[0, :4] = 1

    appearance = Appearance.of(image, box(40, 40), mask)

    assert appearance is not None
    assert appearance.source == "box"


def test_a_box_outside_the_frame_is_no_appearance():
    image = frame((RED, 40, 40))

    assert Appearance.of(image, BoundingBox(1.2, 0.1, 0.1, 0.1)) is None
    assert Appearance.of(image, BoundingBox(-0.3, 0.1, 0.2, 0.1)) is None
    # Two pixels is not an appearance either.
    assert Appearance.of(image, BoundingBox(0.1, 0.1, 0.002, 0.002)) is None


def test_the_descriptor_is_normalised_so_size_does_not_matter():
    # A near object and a far one are the same colours in different numbers of
    # pixels. Unit mass is what lets them compare as colours.
    image = frame((RED, 40, 40))
    whole = Appearance.of(image, box(40, 40))
    quarter = Appearance.of(image, box(40, 40, 80, 60))

    assert whole is not None and quarter is not None
    assert whole.histogram.sum() == pytest.approx(1.0, abs=1e-5)
    assert quarter.histogram.sum() == pytest.approx(1.0, abs=1e-5)
    assert whole.similarity(quarter) > 0.99


# ------------------------------------------------------------- running mean


def test_the_running_descriptor_counts_its_samples_and_converges(looks):
    image = frame((BLUE, 40, 40))
    blue = Appearance.of(image, box(40, 40))
    assert blue is not None

    record = TrackAppearance.begin("cam-01", make_track(3, box(40, 40), 0, 0), 0)
    assert not record.has_appearance

    # Seed it red by hand, then feed it blue frames.
    record.histogram = looks["red"].copy()
    record.samples = 1

    record.observe(make_track(3, box(40, 40), 0, 66), 66, blue, confirmed=True)
    assert record.samples == 2
    # One frame moves it a fifth of the way: still nearer red than blue, so a
    # single bad frame cannot flip an identity.
    still_red = TrackAppearance.begin("cam-01", make_track(0, box(40, 40), 0, 0), 0)
    still_red.histogram, still_red.samples = looks["red"], 1
    assert record.similarity(still_red) > record.similarity(
        fragment(1, first=0, last=0, histogram=looks["blue"])
    )

    for index in range(2, 32):
        record.observe(make_track(3, box(40, 40), 0, index * 66), index * 66, blue, confirmed=True)
    assert record.samples == 32
    # (1 - alpha) ** 30 of the red seed is left: nothing.
    assert (1 - EMA_ALPHA) ** 30 < 0.002
    assert record.similarity(fragment(1, first=0, last=0, histogram=looks["blue"])) > 0.99


def test_a_coasted_frame_extends_the_life_but_not_the_last_known_place():
    # The tracker drags a predicted box towards the frame edge while it waits.
    # Measuring the gap or the distance from that prediction would measure the
    # tracker's guess, not the object's last observed position.
    record = TrackAppearance.begin("cam-01", make_track(3, box(200, 200), 0, 0, placed(0, 10)), 0)

    drifted = make_track(3, box(600, 400), 0, 2000, placed(0, 30))
    record.observe(drifted, 2000, None, confirmed=False)

    assert record.last_seen_millis == 2000
    assert record.last_confirmed_millis == 0
    assert record.last_box == box(200, 200)
    assert record.last_position == placed(0, 10)
    assert record.samples == 0


# ------------------------------------------------------------------ linking


def test_two_fragments_that_meet_all_three_conditions_link(looks):
    earlier = fragment(
        3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20),
    )
    later = fragment(
        7, first=11_000, last=15_000, histogram=looks["red"], first_position=placed(0, 21),
    )

    groups = link_fragments([earlier, later])

    assert len(groups) == 1
    group = groups[0]
    assert group.members == (3, 7)
    assert group.canonical == 3
    assert group.describe() == "#7 = #3"
    assert len(group.links) == 1
    link = group.links[0]
    assert link.earlier == ("cam-01", 3) and link.later == ("cam-01", 7)
    assert link.gap_millis == 1000
    assert link.separation_unit == "m"
    assert link.separation <= link.allowance
    assert link.similarity >= DEFAULT_MIN_SIMILARITY
    # Three reasons, one per condition, so an operator can see each.
    assert len(link.reasons) == 3


def test_a_gap_too_long_does_not_link_however_alike(looks):
    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    later = fragment(
        7, first=10_000 + DEFAULT_MAX_GAP_MILLIS + 1, last=30_000,
        histogram=looks["red"], first_position=placed(0, 20),
    )

    assert len(link_fragments([earlier, later])) == 2


def test_too_far_does_not_link_however_alike(looks):
    # Thirty metres in one second is not walking, and the same colour is not
    # evidence that it happened.
    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    later = fragment(
        7, first=11_000, last=15_000, histogram=looks["red"], first_position=placed(0, 50),
    )

    assert len(link_fragments([earlier, later])) == 2


def test_a_different_look_does_not_link_however_close_in_time_and_place(looks):
    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    later = fragment(
        7, first=11_000, last=15_000, histogram=looks["blue"], first_position=placed(0, 21),
    )

    assert len(link_fragments([earlier, later])) == 2


def test_a_red_coat_leaving_and_a_red_coat_arriving_are_two_people(looks):
    """Similarity alone never links. This is the test the module exists to pass.

    Identical descriptors, and every other condition failed one at a time:
    the linker must refuse each time.
    """
    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))

    across_the_site = fragment(
        7, first=10_500, last=15_000, histogram=looks["red"], first_position=placed(90, 80),
    )
    a_minute_later = fragment(
        7, first=70_000, last=75_000, histogram=looks["red"], first_position=placed(0, 20),
    )

    assert len(link_fragments([earlier, across_the_site])) == 2
    assert len(link_fragments([earlier, a_minute_later])) == 2


def test_tracks_that_were_confirmed_at_the_same_time_are_two_objects(looks):
    # A track still being confirmed by detections when the other begins is not
    # the other's predecessor, whatever it looks like. This is the crossing
    # case: two people, and the linker must not fold them into one.
    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    overlapping = fragment(
        7, first=9_500, last=15_000, histogram=looks["red"], first_position=placed(0, 20),
    )

    assert len(link_fragments([earlier, overlapping])) == 2


def test_a_track_nothing_was_sampled_from_never_links(looks):
    # Without appearance, time and place are all there is, and the correlator's
    # own docstring says what that merges. Refusing is the honest answer.
    earlier = fragment(3, first=0, last=10_000, histogram=None, last_position=placed(0, 20))
    later = fragment(
        7, first=11_000, last=15_000, histogram=looks["red"], first_position=placed(0, 20),
    )

    assert len(link_fragments([earlier, later])) == 2


def test_linking_never_crosses_cameras(looks):
    # The same object handed from one camera to the next is the correlator's
    # claim to make, with position uncertainty in the gate. Here it would be a
    # same-camera claim made about two cameras, on the wrong evidence.
    earlier = fragment(
        3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20), camera="cam-07",
    )
    later = fragment(
        7, first=11_000, last=15_000, histogram=looks["red"], first_position=placed(0, 20), camera="cam-08",
    )

    groups = link_fragments([earlier, later])

    assert len(groups) == 2
    assert {group.camera_id for group in groups} == {"cam-07", "cam-08"}


def test_a_chain_of_three_fragments_is_one_object(looks):
    # 3 -> 7 -> 9. Pairwise, 3 and 9 never meet: 9 begins twelve seconds after
    # 3 ended. Transitively they are one person, and the count must say so.
    first = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    second = fragment(
        7, first=11_000, last=20_000, histogram=looks["red"],
        first_position=placed(0, 21), last_position=placed(0, 25),
    )
    third = fragment(
        9, first=22_000, last=30_000, histogram=looks["red"], first_position=placed(0, 27),
    )

    groups = link_fragments([third, first, second])

    assert len(groups) == 1
    assert groups[0].members == (3, 7, 9)
    assert groups[0].describe() == "#9 = #7 = #3"
    assert len(groups[0].links) == 2


def test_one_departure_continues_into_at_most_one_arrival(looks):
    """Two tracks appear after one vanishes. At most one of them is it.

    Union-find would happily merge all three into one object, and the count
    would say one person where there were at least two. The better-supported
    continuation wins; the other is somebody else.
    """
    image = frame((RED, 40, 40))
    reddish = Appearance.of(image, box(40, 40, 200, 120))  # red with a strip of wall
    assert reddish is not None

    earlier = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    exact = fragment(
        7, first=11_000, last=15_000, histogram=looks["red"], first_position=placed(0, 21),
    )
    close = fragment(
        8, first=11_000, last=15_000, histogram=reddish.histogram, first_position=placed(0, 21),
    )

    groups = link_fragments([earlier, exact, close])

    assert len(groups) == 2
    assert {group.members for group in groups} == {(3, 7), (8,)}


def test_an_unplaced_camera_is_gated_in_the_frame(looks):
    # No pose, so no metres. The gate falls back to normalised image distance
    # and says so in the unit, because a frame unit is a metre near the camera
    # and ten at the horizon and an operator should know which gate spoke.
    earlier = fragment(
        3, first=0, last=10_000, histogram=looks["red"], last_box=box(200, 200),
    )
    beside = fragment(
        7, first=10_500, last=15_000, histogram=looks["red"], first_box=box(210, 205),
    )
    far = fragment(
        7, first=10_500, last=15_000, histogram=looks["red"], first_box=box(500, 40),
    )

    linked = link_fragments([earlier, beside])
    assert len(linked) == 1
    assert linked[0].links[0].separation_unit == "frame"

    assert len(link_fragments([earlier, far])) == 2


def test_every_track_is_in_exactly_one_group_and_singletons_count(looks):
    alone = fragment(3, first=0, last=10_000, histogram=looks["red"], last_position=placed(0, 20))
    also_alone = fragment(
        4, first=0, last=10_000, histogram=looks["blue"], last_position=placed(90, 20),
    )

    groups = link_fragments([alone, also_alone])

    assert sorted(group.members for group in groups) == [(3,), (4,)]
    assert all(group.links == () for group in groups)


# ------------------------------------------------------------------ ledger


def test_the_ledger_builds_a_descriptor_per_track_from_frame_results():
    # A red object walking across three frames with a mask each time, and a
    # blue one that stands still. One record per id, samples counted, masks
    # counted, ends recorded.
    ledger = AppearanceLedger()
    mask = np.ones((120, 160), dtype=np.uint8)
    for index, x in enumerate((40, 60, 80)):
        at = index * 66
        image = frame((RED, x, 40), (BLUE, 400, 300))
        tracks = [
            make_track(1, box(x, 40), 0, at, placed(0, 20 + index)),
            make_track(2, box(400, 300), 0, at, placed(90, 30)),
        ]
        detections = [
            Detection(bbox=box(x, 40), confidence=0.9, class_id=0, mask=mask),
            Detection(bbox=box(400, 300), confidence=0.9, class_id=0),
        ]
        ledger.observe("cam-01", at, tracks, detections, image)

    records = {record.track_id: record for record in ledger.tracks}
    assert set(records) == {1, 2}
    red, blue = records[1], records[2]
    assert red.samples == 3 and red.mask_samples == 3
    assert blue.samples == 3 and blue.mask_samples == 0
    assert red.last_confirmed_millis == 132
    assert red.last_box == box(80, 40)
    assert red.last_position == placed(0, 22)
    assert red.similarity(blue) < DEFAULT_MIN_SIMILARITY
    # Nothing to link: both lived the whole time.
    assert len(ledger.link()) == 2


def test_the_ledger_takes_no_appearance_from_a_coasted_frame():
    # The tracker reports a predicted box on a frame the detector missed. The
    # pixels under that box are whatever happens to be there, not the object.
    ledger = AppearanceLedger()
    image = frame((RED, 40, 40))
    detection = Detection(bbox=box(40, 40), confidence=0.9, class_id=0)

    ledger.observe("cam-01", 0, [make_track(1, box(40, 40), 0, 0)], [detection], image)
    # Coasted: a predicted box drifted onto the wall, and no detection at all.
    ledger.observe("cam-01", 66, [make_track(1, box(300, 300), 0, 66)],
                   [Detection(bbox=box(500, 10, 30, 30), confidence=0.5, class_id=0)], image)

    record = ledger.tracks[0]
    assert record.samples == 1
    assert record.last_confirmed_millis == 0
    assert record.last_seen_millis == 66
    assert record.last_box == box(40, 40)


def test_the_ledger_links_a_fragment_that_reappears():
    # The laptop-camera case in miniature: one red object, tracked as #1,
    # lost for two seconds, tracked again as #2 a step further on. Two ids,
    # one object, and the ledger says so.
    ledger = AppearanceLedger()
    for index, x in enumerate((40, 50, 60)):
        at = index * 66
        image = frame((RED, x, 40))
        ledger.observe("cam-01", at, [make_track(1, box(x, 40), 0, at, placed(0, 20))],
                       [Detection(bbox=box(x, 40), confidence=0.9, class_id=0)], image)
    for index, x in enumerate((90, 100, 110)):
        at = 2_200 + index * 66
        image = frame((RED, x, 40))
        ledger.observe("cam-01", at, [make_track(2, box(x, 40), 2_200, at, placed(0, 23))],
                       [Detection(bbox=box(x, 40), confidence=0.9, class_id=0)], image)

    groups = ledger.link()

    assert len(groups) == 1
    assert groups[0].describe() == "#2 = #1"


def test_without_images_the_ledger_learns_ends_but_links_nothing():
    ledger = AppearanceLedger()
    ledger.observe("cam-01", 0, [make_track(1, box(40, 40), 0, 0, placed(0, 20))],
                   [Detection(bbox=box(40, 40), confidence=0.9, class_id=0)], image=None)
    ledger.observe("cam-01", 1_000, [make_track(2, box(40, 40), 1_000, 1_000, placed(0, 20))],
                   [Detection(bbox=box(40, 40), confidence=0.9, class_id=0)], image=None)

    assert all(not record.has_appearance for record in ledger.tracks)
    assert len(ledger.link()) == 2


# ------------------------------------------------------------- measurement


@pytest.fixture(scope="module")
def reference_report(reference_video: Path) -> measure_fragmentation.FragmentationReport:
    return measure_fragmentation.measure_reference(reference_video)


def test_the_reference_scene_fragmentation_is_measured_not_guessed(reference_report):
    """The number this whole effort is organised around.

    Measured at 1.33: four tracks for three walkers, and — this is the finding
    — zero links, because the reference scene's over-count is identity swaps
    at the crossing, which post-hoc linking cannot touch. The laptop camera's
    3, 10, 4, 11 is a different kind of fragmentation and needs a live run.
    """
    report = reference_report

    assert report.true_objects == len(scene.WALKERS)
    assert report.ratio is not None
    # A tool that reports fewer tracks than objects measured nothing.
    assert report.ratio >= 1.0
    # The pipeline suite's own bound on the same run, restated as a ratio, so
    # the two cannot disagree about what "regressed" means.
    assert report.ratio <= (len(scene.WALKERS) + 2) / len(scene.WALKERS)
    assert report.ratio == report.distinct_tracks / report.true_objects
    assert math.floor(report.ratio * 100) / 100 >= 1.0


def test_the_report_prints_the_number_it_measured(reference_report):
    text = measure_fragmentation.render(reference_report)

    assert f"tracks per object {reference_report.ratio:.2f}" in text
    assert f"distinct tracks   {reference_report.distinct_tracks}" in text
    for walker in scene.WALKERS:
        assert walker.name in text
    # Every id the tracker issued appears in the listing; a segment table that
    # lost a track would understate the fragmentation it exists to show.
    for track_id in reference_report.track_ids:
        assert f"#{track_id}" in text


def test_the_segments_account_for_the_tracks(reference_report):
    report = reference_report
    attributed = {
        segment.track_id for segments in report.segments.values() for segment in segments
    }

    assert attributed <= set(report.track_ids)
    assert attributed, "no track was attributed to any walker"
    # Linking can only lower the count, and by exactly the links it made.
    assert len(report.groups) == report.distinct_tracks - report.linked
    assert report.ratio_after_linking is not None
    assert 1.0 <= report.ratio_after_linking <= report.ratio


def test_a_live_run_with_no_camera_is_skipped_not_failed(monkeypatch):
    # Index 99 opens nothing on any machine this runs on; the tool must say so
    # and return, not hang in the stream's reconnect loop.
    from sentinel import devices

    monkeypatch.setattr(devices, "probe", lambda index: None)

    assert measure_fragmentation.measure_live(99, 1.0, objects=None, model=None) is None
