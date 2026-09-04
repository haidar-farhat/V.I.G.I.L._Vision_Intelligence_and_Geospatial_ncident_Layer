"""What changed, in a form a machine can query and a person can read.

Today an edit reaches the audit log as one prose string — ``"kind RESTRICTED ->
EXCLUSION"`` — written by `node._describe_zone_change`. That line is good for a
human scrolling a list and useless for everything else. It cannot be filtered
("show me every zone whose schedule was shortened"), it cannot be replayed, it
cannot be checked, and it is lossy in the direction that matters: the operator
who changed a zone's kind almost always changed its ring and its schedule in the
same edit, and the string keeps whichever fields somebody remembered to write a
branch for.

So this module produces the structured half and keeps the prose half, from one
source, so the two can never disagree:

- :func:`canonical` — the same object always produces the same bytes. Sorted
  keys, one spelling per number, no incidental whitespace. This is the
  precondition for everything else: a diff between two serialisations that
  differ by key order is noise, and a hash over bytes that depend on a dict's
  insertion order is a hash that changes when nothing did.
- :func:`diff` — a list of :class:`FieldChange`, recursing into nested
  dataclasses and walking tuples element by element. A dragged polygon corner
  shows up as ``ring[2].lat``, not as "the outline changed", which is the
  difference between an audit trail that can answer "which corner, and by how
  much" and one that can only say that somebody touched it.
- :func:`describe` — the prose line, rebuilt from those changes in the voice
  `_describe_zone_change` already uses. It reproduces that function's output
  exactly for flat fields, for a schedule appearing or vanishing, and for a ring
  that gained or lost corners; it differs in three cases, named and measured in
  :func:`describe`'s own docstring, two of which read *worse* than the line they
  would replace. Adopting it is therefore a visible change to an existing log
  line and a decision for whoever wires it in, not a drop-in.
- :class:`AuditRecord` — actor, action, subject, node, both states as canonical
  JSON, the changes, and :meth:`AuditRecord.chain`, a SHA-256 over this record's
  canonical bytes and the previous record's hash.

**What the chain is, stated plainly.** It makes alteration *detectable*: change
one field of one record and every hash after it stops matching, so a log that
was edited no longer verifies. It does not make alteration *impossible* —
anybody who can rewrite a row can rewrite the hashes too — and it is emphatically
not a signature: there is no key, so it proves nothing about who wrote the log or
that this is the log that was written. It is an integrity check over a sequence,
and it becomes evidence of tampering only when the head hash has been recorded
somewhere the process that keeps the log cannot reach. Claiming more than that
for a bare hash chain is how audit trails come to be trusted for something they
never supported.

Nothing here stores anything. Persistence belongs to `store.py`, which owns the
schema and the append-only guarantee; this module owns only the shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "AuditError",
    "AuditRecord",
    "FieldChange",
    "MISSING",
    "canonical",
    "canonical_bytes",
    "chain_hashes",
    "describe",
    "diff",
    "verify_chain",
]


class AuditError(ValueError):
    """A value could not be recorded honestly.

    Raised rather than coerced. A serialiser that silently turns something it
    does not understand into ``"<object at 0x...>"`` writes an audit row that
    looks complete, hashes stably, and records nothing — and the address in it
    changes every run, so the hash is not even stable. An audit trail is allowed
    to fail loudly; it is not allowed to lie quietly.
    """


#: Floats at or above this are no longer integral in the reals, so writing one
#: without a fractional part would be a different number. 2**53 is where a
#: double stops being able to name every integer.
_INTEGRAL_LIMIT = 2 ** 53


class _Missing:
    """The absence of a value, which is not the same as ``None``.

    A zone whose ``schedule`` is ``None`` has a schedule field and it is
    "always". A ring that grew from four corners to five has no fourth index
    *before* the edit. Collapsing those two into ``None`` would report the added
    corner as "changed from nothing", which is what a null already means
    elsewhere in the same record.
    """

    __slots__ = ()
    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


#: The singleton absence. Compare with ``is``.
MISSING = _Missing()


# --------------------------------------------------------------- canonical form


def canonical(value: Any) -> str:
    """Deterministic JSON for any value this codebase records.

    Two equal objects must produce identical text, and that is the entire point:
    a diff and a hash are both meaningless if the same zone serialises two ways
    depending on the order somebody happened to build a dict. So:

    - **Keys are sorted**, by UTF-16 code unit, which is what RFC 8785 specifies
      and is identical to code-point order for every field name in this
      codebase. Sorting matters for dicts read back from SQLite, whose column
      order follows the query.
    - **One spelling per number.** ``1`` and ``1.0`` compare equal in Python, so
      a dataclass holding one is equal to a dataclass holding the other; both
      are written ``1``. Otherwise two equal zones would hash differently, which
      would report a tamper that did not happen. Non-integral floats use
      Python's shortest round-tripping repr.
    - **No incidental whitespace**, and UTF-8 throughout — the text is intended
      to be hashed, not read.

    Understood: dataclasses (recursively, in field order, then sorted),
    :class:`~enum.Enum` by value, ``str``, ``int``, ``float``, ``bool``,
    ``None``, ``datetime``/``date``/``time``, mappings with string keys,
    sequences, sets, and anything with ``tolist`` (numpy scalars and arrays).
    Anything else raises :class:`AuditError` naming its type, because the
    alternative is an audit row that records a placeholder.

    Two honest limits. A tuple and a list of the same elements produce the same
    text, and so do two different dataclasses with the same field names and
    values — the encoding carries no type tag, so that the JSON is the readable
    object an Audit tab can render rather than a tagged wire format. Equal
    objects therefore always agree; agreeing text does not by itself prove the
    objects were the same type. And ``NaN`` and infinity are refused rather than
    written, because JSON has no spelling for them.
    """
    return _encode(value)


def canonical_bytes(value: Any) -> bytes:
    """:func:`canonical` as the UTF-8 bytes that actually get hashed.

    Separate because a hash is over bytes, and leaving the encoding to the call
    site is how a chain computed on one machine comes to differ from the same
    chain computed on another.
    """
    return canonical(value).encode("utf-8")


def _encode(value: Any) -> str:
    # Order matters here and is not stylistic. `bool` is a subclass of `int`,
    # `ZoneKind` is a subclass of `str`, and `datetime` is a subclass of `date`;
    # each of those, checked in the wrong order, silently produces the wrong
    # spelling for a value that looks fine in a debugger.
    if value is None:
        return "null"
    if value is MISSING:
        raise AuditError(
            "MISSING is the absence of a value, not a value: encode it by "
            "leaving the key out, so that an absent field cannot be read back "
            "as a null one"
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return _encode(value.value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, datetime):
        return json.dumps(_encode_datetime(value))
    if isinstance(value, date):
        return json.dumps(value.isoformat())
    if isinstance(value, time):
        return json.dumps(value.isoformat())
    if is_dataclass(value) and not isinstance(value, type):
        return _encode_mapping(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )
    if isinstance(value, Mapping):
        return _encode_mapping(value)
    if isinstance(value, (set, frozenset)):
        # No order to preserve, so the *output* is sorted rather than the input.
        # Sorting the members themselves would need them to be mutually
        # comparable, which a mixed set is not; sorting their encodings always
        # works and gives the same answer for equal sets, which is the property
        # being bought.
        return "[" + ",".join(sorted((_encode(item) for item in value), key=_sort_key)) + "]"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if hasattr(value, "tolist"):
        # numpy scalars and arrays. Converted rather than refused because a
        # pose or a ring that has been through numpy holds `np.float64`, which
        # is not a `float` and would otherwise fail at the audit row rather than
        # where it was introduced.
        return _encode(value.tolist())
    raise AuditError(
        f"cannot record a {type(value).__name__} in an audit record: no "
        "deterministic spelling for it is defined, and inventing one would "
        "produce a row that hashes stably and says nothing"
    )


def _encode_float(value: float) -> str:
    if not math.isfinite(value):
        raise AuditError(
            f"{value!r} has no JSON spelling; a coordinate or angle that is "
            "NaN or infinite is a bug upstream, and writing it as null would "
            "hide the bug inside the record meant to expose it"
        )
    if value == int(value) and abs(value) < _INTEGRAL_LIMIT:
        # Also normalises -0.0, which equals 0.0 and must not hash differently.
        return str(int(value))
    return repr(value)


def _encode_datetime(value: datetime) -> str:
    """ISO 8601, normalised to UTC when the moment knows its offset.

    An aware datetime is a fixed instant however it is written, so two spellings
    of one instant must not produce two hashes. A naive one is written as it
    stands: it carries no offset, so there is nothing to normalise, and guessing
    the local zone would make the same text mean different instants on two
    machines.
    """
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None).isoformat() + "Z"
    return value.isoformat()


def _encode_mapping(mapping: Mapping[Any, Any]) -> str:
    parts = []
    for key in sorted(mapping, key=_mapping_key):
        parts.append(f"{json.dumps(key, ensure_ascii=False)}:{_encode(mapping[key])}")
    return "{" + ",".join(parts) + "}"


def _mapping_key(key: Any) -> bytes:
    if not isinstance(key, str):
        raise AuditError(
            f"a {type(key).__name__} cannot be a JSON object key; convert it at "
            "the call site, where it is known whether two different keys could "
            "convert to the same string"
        )
    return _sort_key(key)


def _sort_key(text: str) -> bytes:
    """UTF-16 code-unit order, as RFC 8785 requires.

    Identical to code-point order for everything in this codebase; spelled out
    anyway so a field name with an emoji or a CJK ideograph in it — a zone named
    by an operator, reaching this through a mapping — sorts the way another
    implementation of the same canonicalisation would sort it.
    """
    return text.encode("utf-16-be")


# ------------------------------------------------------------------ differences


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One field that differs, named by where it lives.

    ``path`` is a dotted, indexed path from the subject: ``"kind"``,
    ``"schedule.start"``, ``"ring[2].lat"``. The empty string means the subject
    itself was replaced by something of another type — a schedule that went from
    ``None`` to a :class:`~sentinel.zones.Schedule` reports at ``"schedule"``,
    but ``diff`` called on two unrelated objects reports at ``""``.

    ``before`` or ``after`` is :data:`MISSING` when the field did not exist on
    that side, which is how an appended polygon corner is distinguished from one
    that was set to null.
    """

    path: str
    before: Any
    after: Any

    @property
    def added(self) -> bool:
        return self.before is MISSING

    @property
    def removed(self) -> bool:
        return self.after is MISSING


def diff(before: Any, after: Any, *, path: str = "") -> list[FieldChange]:
    """Every field that differs, deepest first within each field.

    Recurses into dataclasses by field, mappings by key and sequences by index,
    so a moved polygon corner is reported as ``ring[2].lat`` rather than as the
    whole ring. That is the difference the FEATURES row asks for: an outline
    change reported as "moved" tells an auditor that somebody dragged something,
    and a path plus two coordinates tells them which corner and how far.

    Changes come out in field order — the declaration order of the dataclass —
    which is what lets :func:`describe` rebuild the existing prose line
    unchanged.

    **Sets are compared whole.** A ``frozenset`` has no index, so there is no
    honest path for one of its members; a change to ``Schedule.days`` is
    reported at ``schedule.days`` with both sets. Anything else would invent an
    ordering and then report a change when the ordering, not the data, differed.
    """
    found: list[FieldChange] = []
    _diff_into(before, after, path, found)
    return found


def _diff_into(before: Any, after: Any, path: str, found: list[FieldChange]) -> None:
    if before is MISSING or after is MISSING:
        if not (before is MISSING and after is MISSING):
            found.append(FieldChange(path, before, after))
        return

    if type(before) is not type(after):
        # Not a field-by-field comparison: a `None` schedule and a `Schedule`
        # share no fields, and pretending they do would report every field of
        # the new one as having changed from nothing.
        found.append(FieldChange(path, before, after))
        return

    if _equal(before, after):
        return

    if is_dataclass(before) and not isinstance(before, type):
        for field in fields(before):
            _diff_into(
                getattr(before, field.name),
                getattr(after, field.name),
                _join(path, field.name),
                found,
            )
        return

    if isinstance(before, Mapping):
        for key in sorted(set(before) | set(after), key=_mapping_key):
            _diff_into(
                before.get(key, MISSING),
                after.get(key, MISSING),
                _join(path, key),
                found,
            )
        return

    if isinstance(before, (set, frozenset)):
        found.append(FieldChange(path, before, after))
        return

    if isinstance(before, (tuple, list)):
        for index in range(max(len(before), len(after))):
            _diff_into(
                before[index] if index < len(before) else MISSING,
                after[index] if index < len(after) else MISSING,
                f"{path}[{index}]",
                found,
            )
        return

    found.append(FieldChange(path, before, after))


def _join(path: str, name: str) -> str:
    """Extend a path by one field or key.

    A key that is not a Python identifier — a camera id with a dash in it, a
    zone named by an operator — is bracketed and quoted rather than dotted, so
    the path can be read back unambiguously instead of splitting in the middle
    of a name.
    """
    if not name.isidentifier():
        return f"{path}[{json.dumps(name, ensure_ascii=False)}]"
    return f"{path}.{name}" if path else name


def _equal(before: Any, after: Any) -> bool:
    """Whether two values are the same, without trusting ``==`` to say so.

    A numpy array's ``==`` returns an array, whose truthiness raises; a mask
    reaching an audit record through a detection is exactly that. Falling back
    to comparing canonical text means such a value is compared the same way it
    is recorded, so a difference the record would show is never reported as
    equality.
    """
    try:
        result = before == after
        if isinstance(result, bool):
            return result
    except (TypeError, ValueError):
        pass
    try:
        return canonical(before) == canonical(after)
    except AuditError:
        return False


# ----------------------------------------------------------------------- prose


@dataclass(frozen=True, slots=True)
class _Style:
    """How one field is spoken about in the prose line."""

    label: str
    #: Trails the whole clause: ``dwell 600 -> 900 ms``.
    suffix: str = ""
    #: What ``None`` is called for this field. "A zone with no schedule applies
    #: at all times" — writing that as ``none`` would read as a missing value.
    absent: str = "none"


#: The fields whose prose name is not their field name. Everything else falls
#: back to the field name with underscores opened out, which is already what
#: `node._describe_zone_change` writes for ``accept uncertain``.
_STYLES: dict[str, _Style] = {
    "ring": _Style("outline"),
    "schedule": _Style("schedule", absent="always"),
    "enter_after_millis": _Style("dwell", suffix=" ms"),
    "exit_after_millis": _Style("exit", suffix=" ms"),
}


def describe(changes: Sequence[FieldChange]) -> str:
    """The line a person reads, in the voice the log already uses.

    `node._describe_zone_change` writes ``"name 'Yard' -> 'North yard'; kind
    RESTRICTED -> EXCLUSION"``, and operators have been reading that. Rebuilding
    it from :class:`FieldChange` rather than replacing it means the structured
    record and the prose cannot drift apart — there is one comparison, and the
    string is a rendering of it — while the existing log keeps its voice.

    It is character-identical to `_describe_zone_change` for every flat field,
    for a schedule appearing or vanishing, for a ring that gained or lost
    corners, and for no change at all. **Three cases differ, and only the first
    is an improvement.** They are stated here rather than discovered later,
    because two of them would quietly make the log read worse:

    1. A ring whose corners moved without changing in number is named corner by
       corner — ``outline corner 2 moved`` where the old line said ``outline 4
       -> 4 points, moved``. More information, not different information: the
       changes carry which corner and "moved" threw it away.
    2. A schedule edited in place reports the field inside it: ``schedule start
       18:00:00 -> 19:00:00`` where the old line said ``schedule 18:00–06:00 on
       Mon, Tue -> 19:00–06:00 on Mon, Tue``.
    3. A change to a schedule's days reports the set: ``schedule days [1,2] ->
       [1,2,3]`` where the old line said ``schedule 18:00–06:00 on Mon, Tue ->
       18:00–06:00 on Mon, Tue, Wed``.

    Cases 2 and 3 **lose** `Schedule.describe`'s wording — the operator-readable
    form is replaced by a raw ``time.isoformat()`` and by a canonical JSON array
    inside a prose line — and that is a degradation, not a trade. The cause is
    structural rather than an oversight: :func:`diff` recurses into the schedule,
    so a change at ``schedule.start`` carries one ``time`` and not the enclosing
    :class:`~sentinel.zones.Schedule`, and the whole schedule cannot be rendered
    from what the change holds. Fixing it means either passing the two subjects
    in alongside the changes, or stopping the diff at the schedule and losing the
    ``schedule.start`` / ``schedule.days`` paths that make it queryable. Both are
    real designs and both change something already reviewed, so the choice is
    named here and left to whoever adopts this rather than made silently. The
    divergence is pinned by tests so it cannot widen unnoticed.

    Empty changes give ``"no change"``, which is what the existing function
    returns and what a caller comparing two identical zones should see.
    """
    if not changes:
        return "no change"

    clauses: list[str] = []
    for head, group in _group_by_field(changes):
        # An empty head is the subject itself, replaced wholesale by something
        # of another type. It has no field name, so it is named for what it is.
        style = _STYLES.get(head) or _Style(head.replace("_", " ") or "subject")
        clauses.append(_clause(style, head, group))
    return "; ".join(clauses)


def _group_by_field(
    changes: Sequence[FieldChange],
) -> list[tuple[str, list[FieldChange]]]:
    """Consecutive changes under one top-level field, in the order given.

    Grouped rather than sorted so the clause order follows the dataclass's field
    order, which is the order the existing prose line uses.
    """
    grouped: list[tuple[str, list[FieldChange]]] = []
    for change in changes:
        head = _head(change.path)
        if grouped and grouped[-1][0] == head:
            grouped[-1][1].append(change)
        else:
            grouped.append((head, [change]))
    return grouped


def _head(path: str) -> str:
    for index, character in enumerate(path):
        if character in ".[":
            return path[:index]
    return path


def _clause(style: _Style, head: str, group: list[FieldChange]) -> str:
    indices = [_index_under(head, change.path) for change in group]
    if all(index is not None for index in indices):
        return _sequence_clause(style, group, [index for index in indices if index is not None])

    if len(group) == 1 and group[0].path == head:
        change = group[0]
        return (
            f"{style.label} {_render(change.before, style)} -> "
            f"{_render(change.after, style)}{style.suffix}"
        )

    parts = []
    for change in group:
        tail = change.path[len(head):].lstrip(".")
        name = f"{style.label} {tail}" if tail else style.label
        parts.append(
            f"{name} {_render(change.before, style)} -> "
            f"{_render(change.after, style)}{style.suffix}"
        )
    return "; ".join(parts)


def _index_under(head: str, path: str) -> int | None:
    """The sequence index this path sits under, if it sits under one."""
    if not path.startswith(head + "["):
        return None
    rest = path[len(head) + 1:]
    close = rest.find("]")
    if close < 0:
        return None
    try:
        return int(rest[:close])
    except ValueError:
        return None


def _sequence_clause(
    style: _Style, group: Sequence[FieldChange], indices: Sequence[int]
) -> str:
    added = sorted({i for i, c in zip(indices, group) if c.added})
    removed = sorted({i for i, c in zip(indices, group) if c.removed})

    # A sequence only ever gains or loses elements at its tail as far as an
    # index-wise diff can tell, so the two lengths are recoverable from which
    # indices appeared or vanished — and that is what reproduces the existing
    # "4 -> 5 points, moved" wording exactly.
    if added:
        return f"{style.label} {min(added)} -> {max(added) + 1} points, moved"
    if removed:
        return f"{style.label} {max(removed) + 1} -> {min(removed)} points, moved"

    moved = sorted({index for index in indices})
    if len(moved) == 1:
        return f"{style.label} corner {moved[0]} moved"
    return f"{style.label} corners {', '.join(str(index) for index in moved)} moved"


def _render(value: Any, style: _Style) -> str:
    """One value, as prose rather than as JSON."""
    if value is MISSING:
        return "absent"
    if value is None:
        return style.absent
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        # Quoted, so a rename to a name with a trailing space is visible.
        return repr(value)
    if isinstance(value, (int, float)):
        return str(value)
    describer = getattr(value, "describe", None)
    if callable(describer):
        # `Schedule.describe` already writes "18:00-06:00 on Mon, Tue", which is
        # what the existing audit line prints. Reused rather than reimplemented:
        # two spellings of one schedule is how a log comes to disagree with the
        # screen it was written from.
        try:
            described = describer()
        except TypeError:
            described = None
        if isinstance(described, str):
            return described
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    try:
        return canonical(value)
    except AuditError:
        return repr(value)


# ---------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One thing somebody did, in both forms.

    ``before_json`` and ``after_json`` are :func:`canonical` text, or ``None``
    where there was no such state — a creation has no before, a deletion no
    after. ``changes`` is the structured diff and is empty for both of those,
    which is honest rather than lazy: nothing *changed* when a zone came into
    being, and reporting every field as "changed from nothing" would make a
    creation indistinguishable from an edit that rewrote everything.

    ``at`` is required rather than defaulted to now. A record whose timestamp
    depends on when it happened to be constructed cannot be rebuilt from the
    database and re-hashed, and a chain that cannot be recomputed cannot be
    checked.
    """

    actor: str
    action: str
    subject: str | None
    node_id: str
    at: datetime
    before_json: str | None = None
    after_json: str | None = None
    changes: tuple[FieldChange, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        actor: str,
        action: str,
        subject: str | None,
        node_id: str,
        at: datetime,
        before: Any = MISSING,
        after: Any = MISSING,
    ) -> "AuditRecord":
        """Build a record from the two states, serialising and diffing once.

        Pass :data:`MISSING` — the default — for the side that does not exist.
        ``None`` is a state: a zone whose schedule became ``None`` still has a
        before and an after.
        """
        return cls(
            actor=actor,
            action=action,
            subject=subject,
            node_id=node_id,
            at=at,
            before_json=None if before is MISSING else canonical(before),
            after_json=None if after is MISSING else canonical(after),
            changes=(
                tuple(diff(before, after))
                if before is not MISSING and after is not MISSING
                else ()
            ),
        )

    def describe(self) -> str:
        """The prose line, for the ``detail`` column and for a person."""
        return describe(self.changes)

    def canonical_bytes(self) -> bytes:
        """Exactly the bytes this record is hashed over.

        Spelled out as a dict rather than handed to ``canonical(self)`` so that
        what is hashed is a decision, not a consequence of the dataclass's field
        list. Adding a field to :class:`AuditRecord` for the console's
        convenience must not silently change every hash in an existing log.

        ``before_json`` and ``after_json`` go in as the strings they are, escaped
        rather than embedded: a record whose bytes could be read two ways is a
        record whose hash could be met two ways.

        Because it is a decision, it is pinned: one test asserts the exact key
        set and another varies each of the eight in turn and requires the hash to
        move. Without those, a later edit could *narrow* what the chain protects
        — drop ``at``, drop ``before`` — and nothing would fail, which is a chain
        that still verifies while no longer covering the two things an audit log
        exists to fix: when it happened and what it was before.
        """
        return canonical_bytes(
            {
                "actor": self.actor,
                "action": self.action,
                "subject": self.subject,
                "node_id": self.node_id,
                "at": self.at,
                "before": self.before_json,
                "after": self.after_json,
                "changes": [_change_dict(change) for change in self.changes],
            }
        )

    def chain(self, previous_hash: str | None) -> str:
        """This record's SHA-256, over its own bytes and the one before it.

        Pass ``None`` for the first record in a log. The result is the hex
        digest to store on this row and to pass as ``previous_hash`` for the
        next one.

        **This detects alteration; it does not prevent it.** Editing any earlier
        record changes its hash, so every hash after it stops matching and the
        log fails :func:`verify_chain`. It is not a signature: there is no key
        here, nothing is signed, and anyone able to rewrite a row is equally
        able to recompute every hash after it. What it buys is that alteration
        must be *complete* to go unnoticed — a row edited in a database browser
        will not be — and that an excerpt can be checked against a full log. It
        becomes evidence of tampering only once the head hash has been written
        somewhere this process cannot reach.
        """
        digest = hashlib.sha256()
        # A separator, so that a previous hash and the bytes after it cannot be
        # re-cut at a different place to make two different logs agree.
        digest.update((previous_hash or "").encode("ascii"))
        digest.update(b"\n")
        digest.update(self.canonical_bytes())
        return digest.hexdigest()


def _change_dict(change: FieldChange) -> dict[str, Any]:
    """A change as it is hashed: an absent side is an absent key.

    Not ``null``, because ``null`` is a value a field can hold, and a ring
    corner that was added must not hash the same as one that was set to null.
    """
    entry: dict[str, Any] = {"path": change.path}
    if change.before is not MISSING:
        entry["before"] = canonical(change.before)
    if change.after is not MISSING:
        entry["after"] = canonical(change.after)
    return entry


def chain_hashes(
    records: Iterable[AuditRecord], *, previous_hash: str | None = None
) -> list[str]:
    """The hash of every record in order, folding each into the next.

    ``previous_hash`` continues an existing log; ``None`` starts one.
    """
    hashes: list[str] = []
    current = previous_hash
    for record in records:
        current = record.chain(current)
        hashes.append(current)
    return hashes


def verify_chain(
    records: Sequence[AuditRecord],
    hashes: Sequence[str],
    *,
    previous_hash: str | None = None,
) -> int | None:
    """The index of the first record whose stored hash is wrong, or ``None``.

    The index, not a bare boolean, because "the log was altered" is not
    actionable and "everything from record 412 onward no longer verifies" is:
    the first mismatch is where the edit was, and everything after it fails
    only because it follows.

    A length mismatch is reported at the first missing position, since a log
    with rows removed is exactly what this is meant to catch.
    """
    current = previous_hash
    for index, record in enumerate(records):
        current = record.chain(current)
        if index >= len(hashes) or hashes[index] != current:
            return index
    if len(hashes) > len(records):
        return len(records)
    return None
