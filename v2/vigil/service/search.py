"""Finding what happened: "the north gate, last Tuesday, anything serious".

A site that has been running for a week holds more incidents than anybody
will scroll. Without this the only way to answer a question about last
Tuesday is to read the whole list, which nobody does, which means the
history is written for nobody — the same failure the audit trail had in v1
before anything could read it.

Every filter is applied in SQL, so a search does not pull a month of events
into memory to throw most of them away. Times may be written the way a
person says them: `2h`, `3d`, `2026-09-01`, or a full ISO moment.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from ..domain.events import Event, Severity
from ..domain.incidents import Incident, ReviewState
from ..storage.store import Store
from .auth import SITE_VIEW, Principal

_RELATIVE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days|w|week|weeks)$", re.I)
_UNIT_SECONDS = {"s": 1, "sec": 1, "secs": 1, "m": 60, "min": 60, "mins": 60, "h": 3600, "hr": 3600, "hrs": 3600,
                 "d": 86400, "day": 86400, "days": 86400, "w": 604800, "week": 604800, "weeks": 604800}


class SearchError(ValueError):
    pass


def moment(text: str | None, *, now_millis: int | None = None) -> int | None:
    """A time a person typed, as milliseconds. ``None`` passes through.

    Accepts `90m`, `2h`, `3d`, `1w` (that long ago), a date `2026-09-01`, or
    a full ISO moment. Anything else is refused by name rather than quietly
    treated as the epoch, which would silently widen the search to
    everything.
    """
    if text is None or not str(text).strip():
        return None
    text = str(text).strip()
    now = now_millis if now_millis is not None else int(time.time() * 1000)
    match = _RELATIVE.match(text)
    if match:
        return now - int(float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()] * 1000)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SearchError(f"{text!r} is not a time. Use 2h, 3d, 2026-09-01, or a full ISO moment.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class Query:
    """What somebody asked for. Every field is optional and means "no filter"."""

    since: str | None = None
    until: str | None = None
    camera: str | None = None
    zone: str | None = None
    severity: str | None = None
    state: str | None = None
    #: Matched against an incident's own words — its summary and any note
    #: left on it — and against an event's summary and evidence. Not against
    #: an incident's events, because the summary already names the objects
    #: and the zones they were in.
    contains: str | None = None
    limit: int = 200

    def describe(self) -> str:
        said = [f"{name} {value}" for name, value in (
            ("since", self.since), ("until", self.until), ("camera", self.camera), ("zone", self.zone),
            ("severity", self.severity), ("state", self.state), ("containing", self.contains)) if value]
        return ", ".join(said) if said else "everything"


class Search:
    """Reads only. Every method takes the principal, because reading is a permission too."""

    def __init__(self, store: Store):
        self._store = store

    def incidents(self, query: Query, *, by: Principal, now_millis: int | None = None) -> list[Incident]:
        by.require(SITE_VIEW)
        return self._store.incidents(
            limit=_limit(query.limit), states=_states(query.state),
            since=moment(query.since, now_millis=now_millis), until=moment(query.until, now_millis=now_millis),
            camera_id=query.camera or None, zone=query.zone or None,
            severities=_severities(query.severity), contains=query.contains or None,
        )

    def events(self, query: Query, *, by: Principal, now_millis: int | None = None) -> list[Event]:
        by.require(SITE_VIEW)
        return self._store.events(
            since=moment(query.since, now_millis=now_millis), until=moment(query.until, now_millis=now_millis),
            camera_id=query.camera or None, zone_id=query.zone or None,
            severities=_severities(query.severity), contains=query.contains or None, limit=_limit(query.limit),
        )


def _limit(limit: int) -> int:
    if limit <= 0:
        raise SearchError("a limit must be positive")
    return min(int(limit), 10_000)


def _severities(name: str | None) -> list[str] | None:
    """One severity, or "that and worse", which is what a person means."""
    if not name:
        return None
    order = list(Severity)
    try:
        chosen = Severity(str(name).upper())
    except ValueError as error:
        raise SearchError(f"{name!r} is not a severity; use one of {', '.join(s.value for s in order)}") from error
    return [s.value for s in order[order.index(chosen):]]


def _states(name: str | None) -> list[str] | None:
    if not name:
        return None
    if str(name).lower() in ("queue", "open"):
        return [ReviewState.NEW.value, ReviewState.ACKNOWLEDGED.value]
    if str(name).lower() == "all":
        return None
    try:
        return [ReviewState(str(name).upper()).value]
    except ValueError as error:
        raise SearchError(f"{name!r} is not a review state; use new, acknowledged, dismissed, queue or all") from error
