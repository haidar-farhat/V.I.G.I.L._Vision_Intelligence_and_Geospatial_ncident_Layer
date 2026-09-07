"""The investigation surface: asking the record what happened.

... -> INCIDENT -> RETENTION -> **INVESTIGATION**

Everything upstream of this module writes. This is the half that reads, and until
it existed the stored evidence could only be reached the way it was written: the
most recent hundred incidents, or one camera's events since a timestamp. That is
enough to keep a live wall populated and nothing else. The question an
investigation actually starts from — *what happened at the north gate between two
and four on Tuesday, and was any of it after hours* — could not be expressed at
all, and an operator's only recourse was to page through a list until they gave
up. Evidence nobody can find is evidence nobody has.

Five things this module is careful about, each of them a way a query layer
quietly lies to the person using it:

**A truncated answer says so.** Every search returns the page *and* the number of
rows that matched it, so a panel can say "showing 50 of 1,284" rather than
showing fifty and letting a reader conclude there were fifty. A silently
truncated result is worse than an error: it is a wrong answer wearing the clothes
of a complete one, and the reader has no way to tell.

**A term is data, never SQL.** The search box is the one place in this system
where a person's typing reaches the database, which makes it the injection
surface. Every value travels as a bound parameter; the only text this module ever
interpolates into a statement is its own ``?`` placeholders. ``%`` and ``_`` are
escaped too, so a term containing them is matched literally rather than as a
wildcard nobody typed — a subtler failure than injection and a far more likely
one.

**Filters combine by AND, and an unset filter is not a filter.** An empty
``Query`` matches everything, so a panel that has not been touched shows the
record rather than an empty table with no explanation. Adding a filter can only
ever narrow.

**Time is one clock.** Events are filtered and ordered on ``occurred_at``, the
wall-clock column, never on media time — the store learned that lesson already:
filtering on one and ordering by the other interleaves two cameras by two
incompatible clocks, and the result looks entirely plausible. Incidents are
matched by *overlap*, because an investigator scrubbing to 03:00 must see the
incident that opened at 02:58 and was still running.

**Order says which way the reader is going.** Searches come back newest first,
because a list is read from the top and the recent thing is the thing being
looked for. `timeline` comes back oldest first, because it is laid out along an
axis and an axis runs forwards.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generic, Iterable, Sequence, TypeVar

from .events import Event, EventType, Severity, severity_rank, utc_from_millis
from .incidents import Incident
from .store import Store

# The store's own row reader, not a second copy of it. Rebuilding an event here
# would be a second answer to "what does a stored event contain", and the two
# would come to differ the first time a column is added — silently, because a
# read that drops a field looks complete. It is private today only because
# nothing outside that module has needed it.
from .store import _event_from_row as event_from_row

#: How many rows a page holds when the caller does not say. Enough to fill a
#: panel, small enough that the count beside it is what tells the reader there is
#: more.
DEFAULT_LIMIT = 100

#: The character SQLite is told to treat as the escape inside a LIKE pattern.
#: Any character would do; a backslash is the one a reader expects.
_ESCAPE = "\\"

#: What a free-text term is matched against on an event. Listed rather than
#: implied: a search that silently declines to look at the field the operator was
#: thinking of teaches them the record does not contain what it contains.
EVENT_TEXT_COLUMNS: tuple[str, ...] = (
    "summary",
    "zone_name",
    "class_label",
    "camera_id",
    "rule_id",
)


class SearchError(ValueError):
    """A query could not be expressed as it was asked.

    Raised rather than answered with an empty page. A filter naming an event
    type that does not exist matches nothing, and "nothing matched" and "you
    asked for something that cannot match" render as the same screen — which is
    how a typo in a filter gets read as an absence of evidence.
    """


# ------------------------------------------------------------------- the window


@dataclass(frozen=True, slots=True)
class Window:
    """A span of wall-clock time, in milliseconds since the epoch, UTC.

    Both ends inclusive. Two adjacent windows therefore share the instant
    between them, which for an investigation is the harmless direction to err
    in: an event on the boundary appears in both views rather than in neither.

    UTC, because that is what the store holds. Local time exists in the
    presentation layer and nowhere else, and a window built from a local clock
    without converting it is how a search for "last night" quietly returns the
    wrong three hours for half the year.
    """

    start_millis: int
    end_millis: int

    def __post_init__(self) -> None:
        if self.end_millis < self.start_millis:
            raise SearchError(
                f"a window from {self.start_millis} to {self.end_millis} ends "
                "before it starts, so nothing can be inside it. Reversed ends are "
                "a swapped argument, not an empty result."
            )

    @classmethod
    def around(cls, at_millis: int, radius_millis: int) -> "Window":
        """The span either side of an instant — a scrub bar centred on a thing.

        ``radius_millis`` may not be negative: such a window would end before it
        starts, and would be refused above for a reason naming numbers the
        caller never typed.
        """
        if radius_millis < 0:
            raise SearchError(
                f"a window cannot have a negative radius ({radius_millis} ms)"
            )
        return cls(at_millis - radius_millis, at_millis + radius_millis)

    @property
    def duration_millis(self) -> int:
        return self.end_millis - self.start_millis

    def contains(self, at_millis: int) -> bool:
        return self.start_millis <= at_millis <= self.end_millis


# -------------------------------------------------------------------- the query


def _one_or_many(values: object, what: str) -> tuple[str, ...]:
    """Ids, whether the caller passed one or several.

    A bare ``"cam-07"`` is one camera. Iterating it instead — which is what a
    plain ``tuple(values)`` does — searches for cameras named ``c``, ``a``,
    ``m``, and returns an empty page with no hint that the question was mangled
    on the way in. That failure is silent, so it is prevented here rather than
    documented and left to bite.
    """
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, Iterable):
        raise SearchError(f"{what} takes an id or a sequence of ids, not {values!r}")
    return tuple(str(value) for value in values)


def _members(values: object, enum_type: type, what: str) -> tuple:
    """Enum members, whether the caller passed strings, members, or one of them.

    ``Severity`` and ``EventType`` are string enums, so a single member is also a
    string and would fall apart into its characters exactly as above. An
    unrecognised value raises: a filter that cannot match anything is a mistake
    to report, not a result to return.
    """
    if isinstance(values, (enum_type, str)):
        values = (values,)
    elif not isinstance(values, Iterable):
        raise SearchError(f"{what} takes {enum_type.__name__} values, not {values!r}")

    members = []
    for value in values:
        try:
            members.append(enum_type(value))
        except ValueError:
            known = ", ".join(member.value for member in enum_type)
            raise SearchError(
                f"{enum_type.__name__} has no value {value!r}. Known values: {known}"
            ) from None
    return tuple(members)


@dataclass(frozen=True, slots=True)
class Query:
    """What to look for. Every field optional, and all of them combinable.

    The default instance — no cameras, no window, no types, no term — matches
    everything, which is what a panel shows before anybody has touched it.
    Filters narrow and never widen, so an operator adding one can only ever see
    less, which is the only behaviour a filter can have that a person is able to
    reason about.

    ``zones`` are zone *ids*, matched against the zone each event was raised in —
    not against the zone names an incident carries in its own row. A zone can be
    renamed, and a search matching the stored name would stop finding the
    incidents it found yesterday. The id does not move.

    ``window`` and the event-level filters are asked of *different things* when
    the subject is an incident, and that is deliberate. `search_incidents` asks
    the window of the incident's whole span, because an investigator scrubbing
    into the middle of something wants the whole of it; the camera, zone and type
    filters are then asked of that incident's events with no time bound at all.
    So a filter bar reading "03:00 to 03:05; after-hours" can return an incident
    that was still running at 03:00 whose after-hours event happened at 02:58:
    for an incident, a type or camera names evidence *anywhere in* it, not
    evidence inside the window. Pushing the window into that clause as well would
    undo the overlap it was written for — the incident found because it began
    before the window opened would then be dropped for having nothing inside it,
    hiding exactly the thing being scrubbed to. `search_events` has no such
    subtlety: there the filters are all asked of the one row, and the same query
    returns only what happened inside the window.
    """

    #: Camera ids. Empty means every camera.
    cameras: tuple[str, ...] = ()
    #: Wall-clock span. ``None`` means all of recorded time.
    window: Window | None = None
    types: tuple[EventType, ...] = ()
    severities: tuple[Severity, ...] = ()
    zones: tuple[str, ...] = ()
    #: Free text, matched case-insensitively as a substring against the columns
    #: in `EVENT_TEXT_COLUMNS`. Whitespace-only is treated as absent, because an
    #: empty search box must not be a filter.
    term: str | None = None
    #: How many rows one page holds. The total returned beside them is what keeps
    #: a page from reading as the whole answer.
    limit: int = DEFAULT_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", _one_or_many(self.cameras, "cameras"))
        object.__setattr__(self, "zones", _one_or_many(self.zones, "zones"))
        object.__setattr__(self, "types", _members(self.types, EventType, "types"))
        object.__setattr__(
            self, "severities", _members(self.severities, Severity, "severities")
        )

        term = self.term.strip() if isinstance(self.term, str) else self.term
        object.__setattr__(self, "term", term or None)

        if self.limit < 1:
            raise SearchError(
                f"a page of {self.limit} rows is not a page. Ask for at least one; "
                "the total that comes back beside it says how many there were."
            )

    @property
    def is_empty(self) -> bool:
        """Whether this narrows anything at all. The limit is not a filter."""
        return not (
            self.cameras
            or self.zones
            or self.types
            or self.severities
            or self.term
            or self.window
        )

    def describe(self) -> str:
        """The query in the words a panel can put above its results."""
        parts: list[str] = []
        if self.cameras:
            parts.append("cameras " + ", ".join(self.cameras))
        if self.window is not None:
            parts.append(
                f"{utc_from_millis(self.window.start_millis):%Y-%m-%d %H:%M} to "
                f"{utc_from_millis(self.window.end_millis):%Y-%m-%d %H:%M} UTC"
            )
        if self.types:
            parts.append("types " + ", ".join(t.value for t in self.types))
        if self.severities:
            parts.append("severity " + ", ".join(s.value for s in self.severities))
        if self.zones:
            parts.append("zones " + ", ".join(self.zones))
        if self.term:
            parts.append(f'matching "{self.term}"')
        return "; ".join(parts) if parts else "everything"


def at_least(severity: Severity) -> tuple[Severity, ...]:
    """Every severity at or above one, for the commonest filter there is.

    `Query.severities` is a set, deliberately: "exactly these" is the only
    membership test that composes. But "HIGH and above" is what an operator means
    nine times in ten, and a caller building that list by hand gets it wrong the
    day a severity is added between two others. Derived from `severity_rank`, so
    this list and the ordering it comes from cannot disagree.
    """
    floor = severity_rank(severity)
    return tuple(s for s in Severity if severity_rank(s) >= floor)


# ------------------------------------------------------------------- the answer

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Results(Generic[T]):
    """One page, and how much there was.

    ``total`` is the number of rows that matched, not the number returned. The
    two travelling together is the whole point of this type: a page handed back
    on its own is indistinguishable from a complete answer, and a reader who
    believes they are looking at everything stops looking.
    """

    items: tuple[T, ...]
    total: int
    limit: int

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def truncated(self) -> bool:
        return self.total > len(self.items)

    def describe(self) -> str:
        if not self.truncated:
            return f"{self.total:,} result(s)"
        return f"showing {len(self.items):,} of {self.total:,}"


class MomentKind(str, Enum):
    """What a point on the timeline is.

    Carried on the moment rather than inferred from which list it came out of,
    because the two are merged and a reader has to be able to tell an incident
    from one of the events inside it.
    """

    EVENT = "EVENT"
    INCIDENT = "INCIDENT"


@dataclass(frozen=True, slots=True)
class Moment:
    """One thing that happened, at one instant, for a scrub bar to sit on.

    Deliberately thin. A timeline over an eight-hour shift is thousands of these,
    and rebuilding every event's full evidence to draw a tick mark would make the
    bar cost more than the video it sits under. The id is here so that a click
    can go and ask for the real object.
    """

    at_millis: int
    at: datetime
    kind: MomentKind
    id: str
    severity: Severity
    summary: str
    #: Cameras involved. One for an event, however many for an incident.
    cameras: tuple[str, ...]
    #: Zone *names*, for display. The ids live on the objects themselves.
    zones: tuple[str, ...]


# ----------------------------------------------------------- statement building


def _like(term: str) -> str:
    """A substring pattern that matches the term and nothing cleverer.

    ``%`` and ``_`` are wildcards inside LIKE, so a term containing either would
    silently match far more than it says: a search for the plate fragment
    ``AB_12`` would match ``AB712``, and a bare ``%`` would return every row in
    the table while looking like a narrowing filter. Escaped here, and every
    statement that uses this names `_ESCAPE` so SQLite honours it.
    """
    escaped = (
        term.replace(_ESCAPE, _ESCAPE * 2)
        .replace("%", _ESCAPE + "%")
        .replace("_", _ESCAPE + "_")
    )
    return f"%{escaped}%"


def _in_clause(
    column: str, values: Sequence[str], clauses: list[str], params: list[object]
) -> None:
    """``column IN (?, ?, ?)`` — nearly the only text this module builds.

    The placeholders are generated from the *count* of values; not one value ever
    reaches a statement as text. That distinction is the whole of this module's
    injection story, so it lives in one function every filter goes through rather
    than being retyped at six call sites, five of which would stay right.
    """
    if not values:
        return
    placeholders = ",".join("?" * len(values))
    clauses.append(f"{column} IN ({placeholders})")
    params.extend(values)


def _event_filters(
    query: Query,
    *,
    prefix: str = "",
    with_time: bool = True,
    with_severity: bool = True,
    with_term: bool = True,
) -> tuple[list[str], list[object]]:
    """The event-level predicates, over a table named by ``prefix``.

    One function, because the same predicates are asked twice: directly of the
    events table, and inside the EXISTS that decides whether an incident has an
    event like this. Two copies would drift, and the drift shows up as a search
    that finds an event but not the incident containing it — which reads as
    missing evidence rather than as a bug.
    """
    clauses: list[str] = []
    params: list[object] = []

    _in_clause(f"{prefix}camera_id", query.cameras, clauses, params)
    _in_clause(f"{prefix}zone_id", query.zones, clauses, params)
    _in_clause(f"{prefix}type", [t.value for t in query.types], clauses, params)
    if with_severity:
        _in_clause(
            f"{prefix}severity", [s.value for s in query.severities], clauses, params
        )

    if with_time and query.window is not None:
        # Filtered on `occurred_at`, which is also what the results are ordered
        # by. `occurred_at_millis` is media time and means nothing across two
        # cameras; cutting on one and ordering by the other produces a page whose
        # ends are decided by a different clock than its middle.
        clauses.append(f"{prefix}occurred_at BETWEEN ? AND ?")
        params.extend([query.window.start_millis, query.window.end_millis])

    if with_term and query.term:
        pattern = _like(query.term)
        matches = " OR ".join(
            f"{prefix}{column} LIKE ? ESCAPE '{_ESCAPE}'"
            for column in EVENT_TEXT_COLUMNS
        )
        clauses.append(f"({matches})")
        params.extend([pattern] * len(EVENT_TEXT_COLUMNS))

    return clauses, params


def _incident_filters(query: Query) -> tuple[list[str], list[object]]:
    """The predicates for an incident, over the ``incidents`` table.

    Severity is the incident's own, which is not always the worst of its events —
    risk scoring can raise it — so it is read from the incident row. Taking it
    from the events would mean an incident promoted to CRITICAL could not be
    found by a search for CRITICAL, which is the search somebody runs first.

    Camera, zone and type are asked of the events instead, through one EXISTS, so
    that "cam-07 and LOITERING" means *one event that is both* rather than an
    incident that happens to contain a cam-07 event and, separately, somebody
    loitering in front of a different camera.

    That EXISTS carries no time bound — ``with_time=False`` — even when the query
    has a window, and the omission is the point rather than an oversight. The
    window has already been asked, just above, of the incident's own span, and
    asking it a second time of the events would quietly undo it: an incident
    matched *because* it was already running when the window opened would then be
    dropped for containing no event inside that window, which is to say the
    filter would hide precisely what the overlap test exists to find. The price
    is worth saying out loud, because it surprises people — a matched incident's
    cam-07 or LOITERING evidence may lie outside the stated span, since the
    filters name evidence anywhere in the incident. It is stated on `Query` too,
    where a caller reads it, and
    `test_an_incident_is_matched_by_evidence_the_window_does_not_contain` is what
    holds it in place.
    """
    clauses: list[str] = []
    params: list[object] = []

    _in_clause("severity", [s.value for s in query.severities], clauses, params)

    if query.window is not None:
        # Overlap, not containment, and not "opened inside". An incident that
        # opened at 02:58 and ran for four minutes is the answer to "what
        # happened at 03:00", and a search that missed it because it began a
        # moment early would hide the very thing being looked for.
        #
        # No wall-clock close is stored, so the end is derived: the observed open
        # plus the duration, which is the difference of the two media timestamps
        # and is a real elapsed time for a live source and for a replayed one
        # alike. MAX(..., 0) so a row whose close precedes its open collapses to
        # an instant rather than disappearing from every window there is.
        clauses.append("opened_at <= ?")
        params.append(query.window.end_millis)
        clauses.append("opened_at + MAX(closed_at_millis - opened_at_millis, 0) >= ?")
        params.append(query.window.start_millis)

    event_clauses, event_params = _event_filters(
        query, prefix="e.", with_time=False, with_severity=False, with_term=False
    )
    if event_clauses:
        clauses.append(
            "EXISTS (SELECT 1 FROM incident_events ie "
            "JOIN events e ON e.id = ie.event_id "
            "WHERE ie.incident_id = incidents.id AND "
            + " AND ".join(event_clauses)
            + ")"
        )
        params.extend(event_params)

    if query.term:
        pattern = _like(query.term)
        # The incident's own text first, then its events'. An operator searching
        # a camera name expects the incident that camera contributed to, and the
        # summary alone does not always name it.
        clauses.append(
            f"(summary LIKE ? ESCAPE '{_ESCAPE}' "
            f"OR cameras LIKE ? ESCAPE '{_ESCAPE}' "
            f"OR zones LIKE ? ESCAPE '{_ESCAPE}' "
            "OR EXISTS (SELECT 1 FROM incident_events ie "
            "JOIN events e ON e.id = ie.event_id "
            "WHERE ie.incident_id = incidents.id AND "
            f"(e.summary LIKE ? ESCAPE '{_ESCAPE}' "
            f"OR e.zone_name LIKE ? ESCAPE '{_ESCAPE}')))"
        )
        params.extend([pattern] * 5)

    return clauses, params


def _where(clauses: Sequence[str]) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


# -------------------------------------------------------------------- searching


def search_events(store: Store, query: Query | None = None) -> Results[Event]:
    """Events matching the query, newest first, with the number that matched.

    The count and the page are read inside one transaction so that they agree. A
    pipeline inserting events between two separate reads would produce "showing
    50 of 1,284" from a database that never held exactly those two numbers — a
    small lie, and the kind that teaches a reader to stop trusting the large
    ones.

    `Store.transaction` is the connection this module is allowed to reach: the
    store owns its own handle, and a query layer that reached past the public
    surface into it would silently break the first time that surface changed.
    """
    query = query or Query()
    clauses, params = _event_filters(query)
    where = _where(clauses)

    with store.transaction() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) AS n FROM events {where}", params
        ).fetchone()["n"]
        rows = connection.execute(
            # `id` breaks the tie so a page is deterministic. Two events in the
            # same millisecond ordered arbitrarily would swap places between two
            # reads of the same page, which reads as the record changing under
            # the person reviewing it.
            f"SELECT * FROM events {where} ORDER BY occurred_at DESC, id DESC LIMIT ?",
            (*params, query.limit),
        ).fetchall()

    return Results(
        items=tuple(event_from_row(row) for row in rows),
        total=int(total),
        limit=query.limit,
    )


def search_incidents(store: Store, query: Query | None = None) -> Results[Incident]:
    """Incidents matching the query, newest first, with the number that matched.

    Rebuilt through the store's own `Store.incident`, so what comes back is the
    incident as it was concluded — its events, its associations and the risk
    reasoning that produced its score. Nothing is recomputed here: re-deriving
    risk from a newer rule set would silently rewrite what was decided at the
    time, which is the one thing an evidence trail must never do.

    A row that vanishes between the page query and its rebuild — supersession
    deletes a merged incident — is skipped rather than returned as a ``None``
    that every caller would have to remember to filter. It still counted toward
    ``total``, which is honest: it matched at the moment it was counted.
    """
    query = query or Query()
    clauses, params = _incident_filters(query)
    where = _where(clauses)

    with store.transaction() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) AS n FROM incidents {where}", params
        ).fetchone()["n"]
        rows = connection.execute(
            f"SELECT id FROM incidents {where} ORDER BY opened_at DESC, id DESC LIMIT ?",
            (*params, query.limit),
        ).fetchall()
        found = [store.incident(row["id"]) for row in rows]

    return Results(
        items=tuple(incident for incident in found if incident is not None),
        total=int(total),
        limit=query.limit,
    )


def timeline(
    store: Store, window: Window, *, limit: int = DEFAULT_LIMIT
) -> Results[Moment]:
    """Everything in a window, across every camera, in the order it happened.

    Oldest first, unlike the searches above, because this is laid out along an
    axis rather than read down a list — a scrub bar that ran backwards would be a
    scrub bar nobody could use.

    Events and incidents are merged rather than handed back as two lists. The
    incident is the thing that happened and its events are the grounds for saying
    so; an operator dragging along the bar wants to see the second inside the
    first, not in another panel with its own scroll position to keep in sync.

    When more falls in the window than fits, the *earliest* are returned and
    ``total`` says how many there were, so the caller narrows the window.
    Returning some scattered subset would draw a bar with invisible holes in it,
    which is the one thing worse than a bar that is honestly too short.
    """
    if limit < 1:
        raise SearchError(f"a timeline of {limit} moments is not a timeline")

    span = (window.start_millis, window.end_millis)

    with store.transaction() as connection:
        event_total = connection.execute(
            "SELECT COUNT(*) AS n FROM events WHERE occurred_at BETWEEN ? AND ?", span
        ).fetchone()["n"]
        event_rows = connection.execute(
            "SELECT id, severity, summary, camera_id, zone_name, occurred_at "
            "FROM events WHERE occurred_at BETWEEN ? AND ? "
            "ORDER BY occurred_at, id LIMIT ?",
            (*span, limit),
        ).fetchall()

        # Overlap, for the reason spelled out in `_incident_filters`: the
        # incident that was already running when the window opened is the one
        # being looked for.
        overlap = (window.end_millis, window.start_millis)
        incident_total = connection.execute(
            "SELECT COUNT(*) AS n FROM incidents WHERE opened_at <= ? "
            "AND opened_at + MAX(closed_at_millis - opened_at_millis, 0) >= ?",
            overlap,
        ).fetchone()["n"]
        incident_rows = connection.execute(
            "SELECT id, severity, summary, cameras, zones, opened_at FROM incidents "
            "WHERE opened_at <= ? "
            "AND opened_at + MAX(closed_at_millis - opened_at_millis, 0) >= ? "
            "ORDER BY opened_at, id LIMIT ?",
            (*overlap, limit),
        ).fetchall()

    moments = [_moment_from_event_row(row) for row in event_rows]
    moments += [_moment_from_incident_row(row) for row in incident_rows]
    # The id breaks the tie, for the same reason the searches order by it: an
    # incident and its opening event share an instant, and marks that swapped
    # places between two reads would look like the record had changed.
    moments.sort(key=lambda moment: (moment.at_millis, moment.id))

    return Results(
        items=tuple(moments[:limit]),
        total=int(event_total) + int(incident_total),
        limit=limit,
    )


def _moment_from_event_row(row: sqlite3.Row) -> Moment:
    at_millis = row["occurred_at"]
    return Moment(
        at_millis=at_millis,
        at=utc_from_millis(at_millis),
        kind=MomentKind.EVENT,
        id=row["id"],
        severity=Severity(row["severity"]),
        summary=row["summary"],
        cameras=(row["camera_id"],),
        # An event that was in no zone has no zone name, which is a different
        # thing from an empty one — and is why this is a tuple and not a string.
        zones=() if row["zone_name"] is None else (row["zone_name"],),
    )


def _moment_from_incident_row(row: sqlite3.Row) -> Moment:
    at_millis = row["opened_at"]
    return Moment(
        at_millis=at_millis,
        at=utc_from_millis(at_millis),
        kind=MomentKind.INCIDENT,
        id=row["id"],
        severity=Severity(row["severity"]),
        summary=row["summary"],
        cameras=tuple(json.loads(row["cameras"])),
        zones=tuple(json.loads(row["zones"])),
    )
