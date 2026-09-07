"""Tests for structured before/after audit records.

The prose audit line — ``"kind RESTRICTED -> EXCLUSION"`` — is readable and
nothing else. It cannot be filtered, replayed or checked, and it omits whatever
fields nobody wrote a branch for. Four properties carry the weight here, and
each of them is what makes an audit trail worth having rather than worth
believing:

- **The same object always produces the same bytes.** Insertion order, hash
  seed, ``1`` against ``1.0``, ``-0.0`` against ``0.0``: every one of those
  would otherwise report a difference where there is none, and a hash that
  changes when nothing did is a tamper alarm that gets switched off.
- **A change is named where it happened.** ``ring[2].lat``, not "the outline
  changed". Which corner and how far is the question an auditor asks.
- **The prose does not change voice, and where it does, the suite says so.**
  Flat fields, a schedule appearing, an added or removed corner and "no change"
  are compared character for character against `node._describe_zone_change`,
  which is what operators have been reading. Three cases diverge, and each has a
  test that spells out both strings rather than quietly skipping the comparison:
  a same-length moved ring (which says more), and a schedule edited in place or
  by its days (which say *less*, losing `Schedule.describe`'s wording). Recorded
  here so that adopting `describe` is a decision made with the diff in hand.
- **The chain detects alteration, over every field it claims to cover.** Editing
  one earlier record must break every hash after it — and the test says *which*
  record, because "the log was altered" is not actionable. Every one of the
  eight hashed fields is varied in turn, because a test suite that still passes
  when `at` or `before` is dropped from the digest is a suite that would let the
  chain silently stop protecting them.

No model, no database and no network is needed for any of it: everything below
is built from `Zone`, `Schedule` and `LatLon`, which are values.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from sentinel.auditing import (
    MISSING,
    AuditError,
    AuditRecord,
    FieldChange,
    canonical,
    canonical_bytes,
    chain_hashes,
    describe,
    diff,
    verify_chain,
)
from sentinel.core import LatLon
from sentinel.zones import Schedule, Zone, ZoneKind

ENGINE = Path(__file__).resolve().parents[1]

#: A four-corner yard. Plain offsets rather than geodesy: nothing here measures
#: distance, and a ring built from constants is a ring two tests can agree on.
RING = (
    LatLon(33.8938, 35.5018),
    LatLon(33.8938, 35.5028),
    LatLon(33.8948, 35.5028),
    LatLon(33.8948, 35.5018),
)

WHEN = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def zone(**overrides) -> Zone:
    settings = dict(id="z1", name="Yard", kind=ZoneKind.RESTRICTED, ring=RING)
    settings.update(overrides)
    return Zone(**settings)


def moved_corner(index: int, *, north: float = 0.0, east: float = 0.0) -> tuple:
    """The ring with one corner dragged, which is the edit that matters most."""
    corner = RING[index]
    replacement = LatLon(corner.lat + north, corner.lon + east)
    return RING[:index] + (replacement,) + RING[index + 1:]


def record(**overrides) -> AuditRecord:
    settings = dict(
        actor="operator",
        action="zone.changed",
        subject="z1",
        node_id="local",
        at=WHEN,
    )
    settings.update(overrides)
    return AuditRecord.of(**settings)


# ------------------------------------------------------------- canonical bytes


def test_the_same_object_built_twice_produces_identical_text():
    first, second = zone(), zone()

    assert first == second
    assert canonical(first) == canonical(second)


def test_two_equal_objects_hash_equally():
    one = hashlib.sha256(canonical_bytes(zone())).hexdigest()
    two = hashlib.sha256(canonical_bytes(zone())).hexdigest()
    print("digest", one)

    assert one == two
    assert len(one) == 64


def test_a_dictionary_canonicalises_the_same_whatever_order_it_was_built_in():
    forward = {"zone": zone(), "actor": "operator", "at": WHEN}
    backward = {"at": WHEN, "actor": "operator", "zone": zone()}

    assert list(forward) != list(backward), "the two dicts must differ in order"
    assert canonical(forward) == canonical(backward)


def test_keys_come_out_sorted_whatever_went_in():
    text = canonical({"zulu": 1, "alpha": 2, "mike": 3})
    print(text)

    assert text == '{"alpha":2,"mike":3,"zulu":1}'


def test_a_set_canonicalises_the_same_however_it_was_assembled():
    # A frozenset has no order to preserve, so the output is what must be
    # ordered. `Schedule.days` is one of these.
    assert canonical(frozenset({3, 1, 2})) == canonical(frozenset({2, 3, 1}))


def test_canonical_text_is_stable_across_separate_runs():
    """A frozenset of strings iterates in an order that depends on the hash seed.

    That is the one thing in this codebase whose in-memory order really does
    change between processes, so it is the only honest way to test "stable
    across runs" — two interpreters, two seeds, one answer.
    """
    program = (
        "import sys; sys.path.insert(0, %r);"
        "from sentinel.auditing import canonical;"
        "print(canonical({'days': frozenset({'mon', 'tue', 'wed', 'thu', 'fri'})}))"
        % str(ENGINE)
    )
    outputs = []
    for seed in ("0", "1", "12345"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, env=environment, check=True,
        )
        outputs.append(result.stdout.strip())
    print(outputs)

    assert len(set(outputs)) == 1, "the text changed with the hash seed"


def test_an_integer_and_the_same_number_as_a_float_are_one_spelling():
    # They compare equal in Python, so two dataclasses holding them are equal —
    # and equal objects must not hash differently.
    assert 600 == 600.0
    assert canonical({"dwell": 600}) == canonical({"dwell": 600.0})


def test_negative_zero_is_written_as_zero():
    # -0.0 == 0.0. A heading normalised through a subtraction lands on one or
    # the other, and a zone that hashed differently after a no-op edit would
    # report a tamper that did not happen.
    assert canonical(LatLon(-0.0, 0.0)) == canonical(LatLon(0.0, 0.0))


def test_a_fractional_coordinate_survives_the_round_trip():
    text = canonical(LatLon(33.893812345, 35.5018))
    print(text)

    assert "33.893812345" in text


def test_a_moment_written_two_ways_is_one_instant():
    beirut = timezone(timedelta(hours=3))
    same_instant = WHEN.astimezone(beirut)

    assert same_instant.utcoffset() != WHEN.utcoffset()
    assert canonical(same_instant) == canonical(WHEN)


def test_an_aware_moment_and_a_naive_one_are_not_conflated():
    naive = WHEN.replace(tzinfo=None)

    assert canonical(naive) != canonical(WHEN)


def test_a_schedule_canonicalises_its_times_and_its_days():
    text = canonical(Schedule(start=time(18, 0), end=time(6, 0), days=frozenset({1, 2})))
    print(text)

    assert text == '{"days":[1,2],"end":"06:00:00","start":"18:00:00"}'


def test_a_date_is_recorded_without_pretending_to_a_time():
    assert canonical(date(2026, 9, 4)) == '"2026-09-04"'


def test_an_enum_is_recorded_by_its_value():
    assert canonical(ZoneKind.EXCLUSION) == '"EXCLUSION"'


def test_a_nan_coordinate_is_refused_rather_than_written():
    with pytest.raises(AuditError) as raised:
        canonical(LatLon(float("nan"), 35.5))

    assert "nan" in str(raised.value).lower()


def test_an_unrecordable_value_names_its_own_type():
    with pytest.raises(AuditError) as raised:
        canonical({"who": object()})

    assert "object" in str(raised.value)


def test_a_non_string_key_is_refused_at_the_call_site():
    with pytest.raises(AuditError) as raised:
        canonical({1: "one"})

    assert "int" in str(raised.value)


def test_absence_is_not_encodable_as_a_value():
    # Encoding MISSING as null would make an appended ring corner
    # indistinguishable from one set to null.
    with pytest.raises(AuditError):
        canonical(MISSING)


# -------------------------------------------------------------------- the diff


def test_a_single_changed_field_yields_exactly_one_change():
    changes = diff(zone(), zone(name="North yard"))
    print([change.path for change in changes])

    assert len(changes) == 1
    assert changes[0] == FieldChange("name", "Yard", "North yard")


def test_two_identical_zones_differ_in_nothing():
    assert diff(zone(), zone()) == []


def test_a_moved_ring_corner_names_that_corner():
    changes = diff(zone(), zone(ring=moved_corner(2, north=0.0001)))
    print([(c.path, c.before, c.after) for c in changes])

    assert len(changes) == 1, "the whole ring was reported instead of the corner"
    assert changes[0].path == "ring[2].lat"
    assert changes[0].before == RING[2].lat
    assert changes[0].after > changes[0].before


def test_a_corner_dragged_diagonally_names_both_of_its_coordinates():
    changes = diff(zone(), zone(ring=moved_corner(1, north=0.0002, east=-0.0003)))
    print([change.path for change in changes])

    assert [change.path for change in changes] == ["ring[1].lat", "ring[1].lon"]


def test_an_added_corner_is_absent_before_rather_than_null():
    grown = RING + (LatLon(33.8944, 35.5013),)
    changes = diff(zone(), zone(ring=grown))
    print([(c.path, c.before, c.after) for c in changes])

    assert len(changes) == 1
    assert changes[0].path == "ring[4]"
    assert changes[0].added
    assert changes[0].before is MISSING
    assert changes[0].before is not None


def test_a_removed_corner_is_absent_after():
    changes = diff(zone(ring=RING + (LatLon(33.8944, 35.5013),)), zone())

    assert [change.path for change in changes] == ["ring[4]"]
    assert changes[0].removed


def test_a_schedule_change_and_a_kind_change_in_one_edit_both_appear():
    # The edit the prose line was losing: an operator who changes a zone's kind
    # almost always changes its schedule in the same breath.
    after = zone(kind=ZoneKind.EXCLUSION, schedule=Schedule(time(18, 0), time(6, 0)))
    changes = diff(zone(), after)
    print([change.path for change in changes])

    assert [change.path for change in changes] == ["kind", "schedule"]
    assert changes[0].before is ZoneKind.RESTRICTED
    assert changes[1].before is None


def test_a_schedule_edited_in_place_reports_the_field_inside_it():
    before = zone(schedule=Schedule(time(18, 0), time(6, 0)))
    after = zone(schedule=Schedule(time(19, 0), time(6, 0)))
    changes = diff(before, after)
    print([change.path for change in changes])

    assert [change.path for change in changes] == ["schedule.start"]


def test_a_change_to_the_days_of_a_schedule_is_reported_whole():
    # A frozenset has no index, so there is no honest path to one of its
    # members. Reporting the set is the truthful thing to do.
    before = zone(schedule=Schedule(time(18, 0), time(6, 0), frozenset({1, 2})))
    after = zone(schedule=Schedule(time(18, 0), time(6, 0), frozenset({1, 2, 3})))
    changes = diff(before, after)
    print([(c.path, sorted(c.before), sorted(c.after)) for c in changes])

    assert [change.path for change in changes] == ["schedule.days"]


def test_an_everything_edit_reports_every_field_once():
    after = zone(
        name="North yard",
        kind=ZoneKind.EXCLUSION,
        ring=moved_corner(0, east=0.0004),
        schedule=Schedule(time(18, 0), time(6, 0)),
        enter_after_millis=900,
        exit_after_millis=3000,
        accept_uncertain=True,
    )
    paths = [change.path for change in diff(zone(), after)]
    print(paths)

    assert paths == [
        "name", "kind", "ring[0].lon", "schedule",
        "enter_after_millis", "exit_after_millis", "accept_uncertain",
    ]


def test_a_mapping_of_zones_names_the_zone_that_changed():
    before = {"z1": zone(), "z2": zone(id="z2", name="Gate")}
    after = {"z1": zone(name="North yard"), "z2": zone(id="z2", name="Gate")}
    changes = diff(before, after)
    print([change.path for change in changes])

    assert [change.path for change in changes] == ["z1.name"]


def test_a_key_that_is_not_an_identifier_is_bracketed_not_dotted():
    before = {"cam-01": zone()}
    after = {"cam-01": zone(name="North yard")}
    paths = [change.path for change in diff(before, after)]
    print(paths)

    assert paths == ['["cam-01"].name']


# ------------------------------------------------------------------- the prose


def existing_line(before: Zone, after: Zone) -> str:
    """What the log writes today, for a character-for-character comparison."""
    from sentinel.node import _describe_zone_change

    return _describe_zone_change(before, after)


def test_describe_reads_exactly_like_the_existing_audit_line():
    before = zone()
    after = zone(
        name="North yard",
        kind=ZoneKind.EXCLUSION,
        enter_after_millis=900,
        exit_after_millis=3000,
        accept_uncertain=True,
    )
    line = describe(diff(before, after))
    print(line)

    assert line == existing_line(before, after)


def test_a_schedule_appearing_still_reads_as_always_before():
    before = zone()
    after = zone(schedule=Schedule(time(18, 0), time(6, 0), frozenset({1, 2})))
    line = describe(diff(before, after))
    print(line)

    assert line == existing_line(before, after)
    assert line.startswith("schedule always -> 18:00")


def test_a_corner_added_to_the_outline_keeps_the_old_wording():
    before = zone()
    after = zone(ring=RING + (LatLon(33.8944, 35.5013),))
    line = describe(diff(before, after))
    print(line)

    assert line == existing_line(before, after) == "outline 4 -> 5 points, moved"


def test_a_corner_that_only_moved_is_named_rather_than_counted():
    # The third divergence, and the only one that says *more* than the old line:
    # that threw away which corner moved and printed "4 -> 4 points, moved".
    before = zone()
    after = zone(ring=moved_corner(2, north=0.0001))
    old, new = existing_line(before, after), describe(diff(before, after))
    print("old:", old)
    print("new:", new)

    assert old == "outline 4 -> 4 points, moved"
    assert new == "outline corner 2 moved"


def test_several_moved_corners_are_all_named():
    ring = moved_corner(1, north=0.0001)
    ring = ring[:3] + (LatLon(ring[3].lat, ring[3].lon + 0.0002),)
    line = describe(diff(zone(), zone(ring=ring)))
    print(line)

    assert line == "outline corners 1, 3 moved"


def test_describe_of_nothing_is_no_change():
    assert describe(diff(zone(), zone())) == existing_line(zone(), zone()) == "no change"


def test_clauses_are_joined_the_way_the_existing_line_joins_them():
    line = describe(diff(zone(), zone(name="North yard", kind=ZoneKind.INTEREST)))

    assert "; " in line
    assert line.count(";") == 1


def scheduled(start: time, days=frozenset({1, 2})) -> Zone:
    return zone(schedule=Schedule(start, time(6, 0), days))


def test_a_schedule_edited_in_place_names_the_field_and_loses_the_old_wording():
    """The first of the two divergences that read *worse* than the old line.

    `diff` recurses into the schedule, so the change carries one `time` and not
    the `Schedule` around it, and `describe` cannot rebuild what it never got.
    Both strings are spelled out here rather than the comparison being skipped:
    a divergence recorded only in prose is a divergence that widens unnoticed.
    """
    before, after = scheduled(time(18, 0)), scheduled(time(19, 0))
    old, new = existing_line(before, after), describe(diff(before, after))
    print("old:", old)
    print("new:", new)

    assert old == "schedule 18:00–06:00 on Mon, Tue -> 19:00–06:00 on Mon, Tue"
    assert new == "schedule start 18:00:00 -> 19:00:00"
    assert new != old, "the divergence has been closed; update the docstrings"


def test_a_schedule_whose_days_changed_loses_the_old_wording_too():
    """The second. `Schedule.describe` writes "on Mon, Tue"; this writes [1,2]."""
    before = scheduled(time(18, 0))
    after = scheduled(time(18, 0), frozenset({1, 2, 3}))
    old, new = existing_line(before, after), describe(diff(before, after))
    print("old:", old)
    print("new:", new)

    assert old == (
        "schedule 18:00–06:00 on Mon, Tue -> 18:00–06:00 on Mon, Tue, Wed"
    )
    assert new == "schedule days [1,2] -> [1,2,3]"
    assert new != old, "the divergence has been closed; update the docstrings"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "describe() cannot rebuild a whole Schedule from a change at "
        "schedule.start. Closing this needs either the two subjects passed in "
        "alongside the changes or a diff that stops at the schedule; both are "
        "named in describe()'s docstring. Strict, so whoever closes it is told "
        "to update the three docstrings that record the divergence."
    ),
)
def test_a_schedule_edited_in_place_would_ideally_read_like_the_existing_line():
    before, after = scheduled(time(18, 0)), scheduled(time(19, 0))

    assert describe(diff(before, after)) == existing_line(before, after)


# ------------------------------------------------------------------ the record


def test_a_creation_has_an_after_and_no_before():
    created = record(action="zone.added", after=zone())

    assert created.before_json is None
    assert created.after_json == canonical(zone())
    # Nothing *changed*: the zone came into being. Reporting every field as
    # changed from nothing would make a creation look like a total rewrite.
    assert created.changes == ()
    assert created.describe() == "no change"


def test_a_removal_has_a_before_and_no_after():
    removed = record(action="zone.removed", before=zone())

    assert removed.after_json is None
    assert removed.before_json == canonical(zone())


def test_a_record_carries_both_the_structure_and_the_prose():
    edited = record(before=zone(), after=zone(name="North yard"))
    print(edited.describe())

    assert [change.path for change in edited.changes] == ["name"]
    assert edited.describe() == "name 'Yard' -> 'North yard'"


def test_the_same_record_hashes_the_same_way_twice():
    one = record(before=zone(), after=zone(name="North yard"))
    two = record(before=zone(), after=zone(name="North yard"))
    print(one.chain(None))

    assert one.chain(None) == two.chain(None)
    assert len(one.chain(None)) == 64


def test_a_records_hash_depends_on_the_one_before_it():
    edited = record(before=zone(), after=zone(name="North yard"))

    assert edited.chain(None) != edited.chain("0" * 64)


def test_two_records_differing_only_in_actor_hash_differently():
    mine = record(before=zone(), after=zone(name="North yard"))
    theirs = record(actor="someone else", before=zone(), after=zone(name="North yard"))

    assert mine.chain(None) != theirs.chain(None)


def test_the_hashed_dict_names_exactly_the_fields_the_chain_protects():
    """What goes into the digest is a decision, so it is written down as one.

    `canonical_bytes` builds its dict by hand rather than from the dataclass's
    fields, precisely so a field added for the console cannot change every hash
    in an existing log. The reverse needs pinning too: without this, a later edit
    could *narrow* the digest — drop `at`, drop `before` — and the log would go
    on verifying while no longer covering when the edit happened or what it
    undid. Change this list only on purpose; every stored hash moves with it.
    """
    hashed = json.loads(record(before=zone(), after=zone(name="North yard")).canonical_bytes())
    print(sorted(hashed))

    assert sorted(hashed) == [
        "action", "actor", "after", "at", "before", "changes", "node_id", "subject",
    ]


@pytest.mark.parametrize(
    "field, value",
    [
        ("actor", "someone else"),
        ("action", "zone.removed"),
        ("subject", "z2"),
        ("subject", None),
        ("node_id", "gate-01"),
        ("at", WHEN + timedelta(seconds=1)),
        ("before_json", canonical(zone(name="Something else"))),
        ("before_json", None),
        ("after_json", canonical(zone(name="Something else"))),
        ("after_json", None),
        ("changes", (FieldChange("name", "Yard", "Somewhere else"),)),
        ("changes", ()),
    ],
)
def test_altering_any_recorded_field_moves_the_hash(field, value):
    """Every field the digest claims to cover, varied one at a time.

    The chain's whole promise is that altering a record breaks it. Testing that
    through `actor` alone proves only that `actor` is in there — and the two
    things an audit log exists to pin down are the moment it happened and the
    state before it, neither of which was covered. Each case below differs from
    the base record in exactly one field, so a digest that stopped hashing that
    field fails here rather than passing quietly.
    """
    base = record(before=zone(), after=zone(name="North yard"))
    varied = replace(base, **{field: value})
    print(field, "->", varied.chain(None))
    print(field, "was", base.chain(None))

    assert getattr(base, field) != value, "the case does not vary anything"
    assert varied.chain(None) != base.chain(None)


def test_an_added_corner_and_a_nulled_one_do_not_hash_alike():
    # MISSING and None must not collapse into the same hashed bytes, or an
    # appended corner and a corner set to null would be one edit.
    added = FieldChange("ring[4]", MISSING, LatLon(33.8944, 35.5013))
    nulled = FieldChange("ring[4]", None, LatLon(33.8944, 35.5013))
    one = AuditRecord(
        actor="operator", action="zone.changed", subject="z1", node_id="local",
        at=WHEN, changes=(added,),
    )
    two = AuditRecord(
        actor="operator", action="zone.changed", subject="z1", node_id="local",
        at=WHEN, changes=(nulled,),
    )

    assert one.chain(None) != two.chain(None)


def log() -> list[AuditRecord]:
    """Four edits to one zone, in order."""
    steps = [
        zone(),
        zone(name="North yard"),
        zone(name="North yard", kind=ZoneKind.EXCLUSION),
        zone(name="North yard", kind=ZoneKind.EXCLUSION, ring=moved_corner(2, north=1e-4)),
        zone(name="North yard", kind=ZoneKind.EXCLUSION, ring=moved_corner(2, north=1e-4),
             schedule=Schedule(time(18, 0), time(6, 0))),
    ]
    return [
        record(at=WHEN + timedelta(minutes=index), before=before, after=after)
        for index, (before, after) in enumerate(zip(steps, steps[1:]))
    ]


def test_a_clean_log_verifies():
    records = log()
    hashes = chain_hashes(records)
    print(len(records), hashes[-1])

    assert len(hashes) == len(records) == 4
    assert verify_chain(records, hashes) is None


def test_the_chain_changes_if_any_earlier_record_is_altered():
    records = log()
    hashes = chain_hashes(records)

    # Somebody rewrites the second edit so it never mentions EXCLUSION.
    tampered = list(records)
    tampered[1] = record(
        at=records[1].at,
        before=zone(name="North yard"),
        after=zone(name="North yard", kind=ZoneKind.INTEREST),
    )
    recomputed = chain_hashes(tampered)
    print(hashes[1][:16], "->", recomputed[1][:16])

    assert recomputed[0] == hashes[0], "the record before the edit is untouched"
    assert recomputed[1] != hashes[1]
    # And everything after it, which is the property the chain exists for: one
    # edited row cannot be hidden by leaving the rest alone.
    assert all(new != old for new, old in zip(recomputed[2:], hashes[2:]))
    assert verify_chain(tampered, hashes) == 1


def test_altering_the_last_record_is_still_caught():
    records = log()
    hashes = chain_hashes(records)
    tampered = list(records)
    tampered[-1] = record(at=records[-1].at, before=zone(), after=zone(name="Gate"))

    assert verify_chain(tampered, hashes) == len(records) - 1


def test_a_deleted_row_is_caught_rather_than_shortening_the_log():
    records = log()
    hashes = chain_hashes(records)
    without_the_second = records[:1] + records[2:]
    print(verify_chain(without_the_second, hashes))

    assert verify_chain(without_the_second, hashes) == 1


def test_an_appended_row_that_nobody_hashed_is_caught():
    records = log()
    hashes = chain_hashes(records)[:-1]

    assert verify_chain(records, hashes) == len(hashes)


def test_a_removed_trailing_row_is_caught():
    records = log()
    hashes = chain_hashes(records)

    assert verify_chain(records[:-1], hashes) == len(records) - 1


def test_a_log_can_be_continued_from_an_earlier_head():
    records = log()
    whole = chain_hashes(records)
    first_two = chain_hashes(records[:2])
    rest = chain_hashes(records[2:], previous_hash=first_two[-1])
    print(whole[-1][:16], rest[-1][:16])

    assert first_two + rest == whole
    assert verify_chain(records[2:], rest, previous_hash=first_two[-1]) is None
