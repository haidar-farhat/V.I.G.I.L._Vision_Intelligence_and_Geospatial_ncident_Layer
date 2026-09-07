"""Working the queue: acknowledging an incident, or dismissing it with a reason.

Without this every incident stays new for ever, the list only grows, and the
one that matters is buried under the ones somebody already looked at.

A judgement is never anonymous and never silent: it carries the principal,
it is written to the audit trail with what it was before, and a dismissal
without a reason is refused — "dismissed" with no reason is indistinguishable
from "nobody looked", and the difference is the whole point.
"""

from __future__ import annotations

import time

from ..domain.incidents import Incident, Review, ReviewState
from ..logs import get as _get_logger
from ..storage.store import Store
from .auth import INCIDENT_REVIEW, Principal

_log = _get_logger(__name__)

#: The most a note may be. Long enough for a sentence about what was seen,
#: short enough that the audit trail stays readable.
MAX_NOTE = 500


class ReviewError(ValueError):
    pass


class IncidentReview:
    def __init__(self, store: Store):
        self._store = store

    def queue(self, *, limit: int = 200, include_dismissed: bool = False) -> list[Incident]:
        """What is still waiting on a person, newest first."""
        states = [ReviewState.NEW.value, ReviewState.ACKNOWLEDGED.value]
        if include_dismissed:
            states.append(ReviewState.DISMISSED.value)
        return self._store.incidents(limit=limit, states=states)

    def acknowledge(self, incident_id: str, *, by: Principal, note: str | None = None) -> Incident:
        """Somebody has seen it and it is real."""
        return self._set(incident_id, ReviewState.ACKNOWLEDGED, by=by, note=note)

    def dismiss(self, incident_id: str, *, by: Principal, note: str) -> Incident:
        """Somebody has seen it and it is not worth acting on. The reason is required."""
        if not (note or "").strip():
            raise ReviewError("a dismissal needs a reason; without one it cannot be told from nobody looking")
        return self._set(incident_id, ReviewState.DISMISSED, by=by, note=note)

    def reopen(self, incident_id: str, *, by: Principal, note: str | None = None) -> Incident:
        """A judgement was wrong. The trail keeps both."""
        return self._set(incident_id, ReviewState.NEW, by=by, note=note)

    def _set(self, incident_id: str, state: ReviewState, *, by: Principal, note: str | None) -> Incident:
        by.require(INCIDENT_REVIEW)
        incident = self._store.incident(incident_id)
        if incident is None:
            raise ReviewError(f"no incident {incident_id!r}")
        if note is not None and len(note) > MAX_NOTE:
            raise ReviewError(f"a note is at most {MAX_NOTE} characters")
        before = incident.review
        at = int(time.time() * 1000)
        self._store.set_incident_review(incident_id, state.value, by=by.actor, at=at, note=note)
        self._store.audit(by.actor, f"incident.{str(state).lower()}", incident_id, note,
                          before={"state": str(before.state), "by": before.by, "note": before.note},
                          after={"state": str(state), "by": by.actor, "note": note})
        _log.info("incident %s %s by %s", incident_id, str(state).lower(), by.actor)
        return self._store.incident(incident_id)
