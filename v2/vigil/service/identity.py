"""The identity switch, and everything that may only happen while it is on.

# One switch, and it is off

Faces and plates are a single setting, off by default. Not two switches:
"faces on, plates off" is a distinction nobody has asked for and it doubles
the number of states an operator has to reason about when answering "is this
site processing biometrics". One question, one answer.

The switch is enforced **here**, at the one door, rather than by each caller
remembering to check. `FaceReader` and `PlateReader` are constructed only when
it is on, so a site with the switch off has no object to call, no models
loaded and no code path to reach — rather than a disabled object that
remembers to return nothing.

# Retention is not optional

Turning the switch on without a retention limit is refused by
`enable`, and `vigil doctor` fails if a database is ever found in that state.
An unbounded biometric store is the worst thing this feature can become, and
the check is loud because the failure is silent: nothing looks wrong about a
`face_observations` table with four million rows in it.

`sweep` deletes observations, never subjects. Somebody put a subject there
deliberately; removing them on a timer would be a different feature and a
surprising one. `forget` is how a subject goes, and it takes their
observations with them.

# What "delete" means here

v1's register documented "a delete that really deletes" and then relied on
`DELETE` alone. SQLite leaves the freed pages readable in the freelist and in
the write-ahead log until something overwrites them, so a face template
survived its own erasure. `forget` here sets `PRAGMA secure_delete` and
vacuums afterwards, which is what makes the sentence true.

# Every consequential act is audited, and the audit holds no names

An enrolment, a match, an enable, a disable and an erasure all leave a row
under the principal who caused it. The rows carry **identifiers, not names**:
the audit trail is append-only, so a name written into it outlives the
erasure it was recording. That is the sharpest idea v1 had and it is kept
exactly.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import numpy as np

from ..domain.identity import (
    MAX_DISTANCE, MIN_MARGIN, IdentityError, Match, Register, Subject, Verdict, normalise,
)
from ..logs import get as _get_logger
from ..storage.store import Store
from .auth import SITE_CONFIGURE, Principal

_log = _get_logger(__name__)

#: Retention limits an operator may set, in days.
#:
#: The ceiling is a year and the floor is a day. Neither is a legal figure --
#: this product does not know which jurisdiction it is in -- they are the
#: bounds outside which the setting is almost certainly a mistake: an hour is
#: too short for a register to be useful, and a decade is not a retention
#: policy.
MIN_RETENTION_DAYS, MAX_RETENTION_DAYS = 1, 365

DAY_MILLIS = 86_400_000


class IdentityDisabled(IdentityError):
    """Something biometric was asked for while the switch is off."""


@dataclass(frozen=True, slots=True)
class IdentityState:
    """Whether this site processes biometrics, and for how long it keeps them."""

    enabled: bool
    retention_days: int | None

    @property
    def unbounded(self) -> bool:
        """On, with nothing saying when the data goes. The state `doctor` fails on."""
        return self.enabled and self.retention_days is None

    def describe(self) -> str:
        if not self.enabled:
            return ("faces and plates are OFF: no face is embedded, no plate is read, and no "
                    "biometric row is written")
        if self.retention_days is None:
            return ("faces and plates are ON with NO RETENTION LIMIT — biometric data will be "
                    "kept for ever. Set one now")
        return (f"faces and plates are ON; observations are deleted after "
                f"{self.retention_days} day(s). Enrolled subjects stay until somebody removes them")


class IdentityService:
    """The one door to anything biometric."""

    def __init__(self, store: Store):
        self._store = store

    # ------------------------------------------------------------ the switch

    def state(self) -> IdentityState:
        site = self._store.site()
        return IdentityState(bool(site.get("identity_enabled")),
                             site.get("biometric_retention_days"))

    @property
    def enabled(self) -> bool:
        return self.state().enabled

    def enable(self, retention_days: int, *, reason: str, by: Principal) -> IdentityState:
        """Turn face and plate processing on. A retention limit is required.

        `reason` is required and may not be blank. There is no default -- not
        "operator request", not "enabled": a default here is how the answer to
        "why is this site processing biometrics" becomes a lie six months
        later, and this is the one setting where that question gets asked.
        """
        by.require(SITE_CONFIGURE)
        if not (reason or "").strip():
            raise IdentityError(
                "turning face and plate processing on needs a reason, recorded against your name. "
                "There is no default because this is the setting somebody will be asked about"
            )
        days = int(retention_days)
        if not MIN_RETENTION_DAYS <= days <= MAX_RETENTION_DAYS:
            raise IdentityError(
                f"a retention of {days} day(s) is outside {MIN_RETENTION_DAYS}-"
                f"{MAX_RETENTION_DAYS}. Shorter than a day makes the register useless; longer "
                f"than a year is not a retention policy"
            )
        before = self.state()
        self._store.set_identity(True, days)
        self._store.audit(by.actor, "identity.enabled", None, reason.strip(),
                          before={"enabled": before.enabled, "retention_days": before.retention_days},
                          after={"enabled": True, "retention_days": days})
        _log.warning("identity processing ENABLED by %s, retention %d day(s): %s",
                     by.actor, days, reason.strip())
        return self.state()

    def disable(self, *, reason: str, by: Principal, erase: bool = False) -> IdentityState:
        """Turn it off, and optionally erase what was collected.

        `erase` is not the default. Turning the feature off and destroying an
        entire register are different decisions, and one of them is not
        undoable; conflating them would mean an operator experimenting with a
        setting loses a day's enrolments.
        """
        by.require(SITE_CONFIGURE)
        before = self.state()
        self._store.set_identity(False, before.retention_days)
        removed = self.erase_everything(by=by, reason=reason) if erase else {}
        self._store.audit(by.actor, "identity.disabled", None, (reason or "").strip() or None,
                          before={"enabled": before.enabled},
                          after={"enabled": False, "erased": removed or None})
        _log.warning("identity processing DISABLED by %s%s", by.actor,
                     f", erasing {removed}" if removed else "")
        return self.state()

    def require_enabled(self) -> None:
        if not self.enabled:
            raise IdentityDisabled(
                "face and plate processing is off for this site. `vigil identity enable "
                "--retention-days N --reason ...` turns it on, under your name"
            )

    # --------------------------------------------------------------- subjects

    def enrol(self, label: str, embedding, model_sha256: str, *, basis: str,
              by: Principal, note: str | None = None) -> str:
        """Enrol somebody. Returns the subject id.

        `basis` is why this person is in the register -- a case number, an
        instruction, a contract -- and is required with no default, for the
        same reason `reason` is on the switch.

        The id is random, not derived from the label or the embedding. A
        content-derived id is not a pseudonym: it is the thing itself,
        recoverable by anybody willing to spend an afternoon on it, and it
        would then be the identifier written into the audit trail that is
        supposed to hold no personal data.
        """
        by.require(SITE_CONFIGURE)
        self.require_enabled()
        if not (label or "").strip():
            raise IdentityError("a subject needs a label to be enrolled under")
        if not (basis or "").strip():
            raise IdentityError(
                "enrolling somebody needs a stated basis, recorded against your name. There is no "
                "default because 'system' or 'unknown' six months later is not an answer"
            )
        vector = normalise(embedding)
        subject_id = secrets.token_hex(12)
        self._store.save_subject(subject_id, label.strip(), vector.tolist(), model_sha256,
                                 enrolled_by=by.actor, note=note)
        # Identifier only. A name here would outlive the erasure that removes
        # it from the register, in a table that cannot be edited.
        self._store.audit(by.actor, "identity.enrolled", subject_id, basis.strip(),
                          after={"subject": subject_id, "model": model_sha256[:12],
                                 "dimensions": int(vector.size)})
        _log.info("subject %s enrolled by %s", subject_id, by.actor)
        return subject_id

    def register(self) -> Register:
        """The enrolled subjects, assembled fresh.

        Never cached. A module-level cache would be a second copy of the
        site's biometrics, outliving the `forget` that was meant to delete
        them -- and it would be a copy nobody thinks to look for.
        """
        out = Register(max_distance=MAX_DISTANCE, min_margin=MIN_MARGIN)
        for row in self._store.subjects():
            import json

            try:
                vector = normalise(json.loads(row["embedding"]))
            except (ValueError, IdentityError) as error:
                _log.error("subject %s has an unusable embedding and was not loaded: %s",
                           row["id"], error)
                continue
            out.add(Subject(row["id"], row["label"], vector, row["model_sha256"],
                            row["note"] or ""))
        return out

    def forget(self, subject_id: str, *, by: Principal) -> dict:
        """Erase a subject and every observation of them, irreversibly.

        Idempotent: forgetting somebody who is not there is not an error. A
        retried erasure must not fail in a way that leaves an operator unsure
        whether the data survived.
        """
        by.require(SITE_CONFIGURE)
        counted = self._store.biometric_counts()
        removed = self._store.delete_subject(subject_id)
        after = self._store.biometric_counts()
        erased = {"subject": removed, "faces": counted["faces"] - after["faces"]}
        self._secure_erase()
        self._store.audit(by.actor, "identity.forgotten", subject_id, None, after=erased)
        _log.warning("subject %s erased by %s: %s", subject_id, by.actor, erased)
        return erased

    def erase_everything(self, *, by: Principal, reason: str = "") -> dict:
        by.require(SITE_CONFIGURE)
        counted = self._store.biometric_counts()
        for row in self._store.subjects():
            self._store.delete_subject(row["id"])
        self._store.sweep_biometrics(int(time.time() * 1000) + DAY_MILLIS)
        self._secure_erase()
        self._store.audit(by.actor, "identity.erased_all", None, (reason or "").strip() or None,
                          after=counted)
        return counted

    # -------------------------------------------------------------- recording

    def record_face(self, camera_id: str, track_id: int | None, embeddings,
                    model_sha256: str, *, detector_score: float,
                    at_millis: int | None = None) -> Match | None:
        """Compare a track's faces with the register and record what happened.

        The observation is written **whether or not anything matched**. A face
        that matched nobody is still a face that was processed, and if it were
        not recorded the retention sweep would have nothing to delete and the
        site could not say how much biometric processing it had done.
        """
        self.require_enabled()
        at = int(time.time() * 1000) if at_millis is None else at_millis
        many = list(embeddings)
        result = self.register().identify(many, model_sha256) if many else None
        subject = result.subject.id if result and result.verdict is Verdict.MATCH else None
        self._store.save_face_observation(
            camera_id, track_id, at, subject_id=subject,
            distance=result.distance if result else None,
            margin=None if result is None or result.margin == float("inf") else result.margin,
            quality=float(detector_score), model_sha256=model_sha256)
        return result

    def record_plate(self, camera_id: str, track_id: int | None, reading,
                     model_sha256: str, *, at_millis: int | None = None) -> bool:
        """Record a resolved plate. Returns whether anything was written.

        An unresolved reading is **not** recorded. `Reading.text` is `None`
        while any character is unresolved, and writing a half-read plate into
        a table is exactly how it becomes a plate somebody searches for.
        """
        self.require_enabled()
        text = reading.text
        if text is None:
            return False
        at = int(time.time() * 1000) if at_millis is None else at_millis
        self._store.save_plate_observation(
            camera_id, track_id, at, text=text,
            characters=[float(c) for c in getattr(reading, "agreement", ())],
            confidence=float(reading.weakest_confidence), model_sha256=model_sha256)
        return True

    # -------------------------------------------------------------- retention

    def sweep(self, *, now_millis: int | None = None) -> dict:
        """Delete observations past the retention limit.

        Nothing happens without a limit set, and that is not a quiet
        no-op: `doctor` fails on exactly that state, so the site is told
        rather than left accumulating.
        """
        state = self.state()
        if state.retention_days is None:
            return {}
        now = int(time.time() * 1000) if now_millis is None else now_millis
        removed = self._store.sweep_biometrics(now - state.retention_days * DAY_MILLIS)
        if any(removed.values()):
            self._secure_erase()
            _log.info("biometric retention swept: %s", removed)
        return removed

    def _secure_erase(self) -> None:
        """Make the deletion true rather than merely recorded.

        `DELETE` returns pages to SQLite's freelist with the row bytes still
        in them, readable by anybody with the file until something happens to
        overwrite them. For ordinary rows that is an acceptable trade; for a
        face template it is the difference between erasing somebody's
        biometrics and saying that you did.
        """
        try:
            self._store._connection.execute("PRAGMA secure_delete = ON")
            self._store._connection.execute("VACUUM")
        except Exception as error:  # noqa: BLE001 - a vacuum may fail on a busy database
            _log.error("the database could not be vacuumed after an erasure, so deleted biometric "
                       "rows may remain readable in freed pages until it is: %s", error)

    def counts(self) -> dict:
        return self._store.biometric_counts()

    def describe(self) -> str:
        state = self.state()
        if not state.enabled:
            return state.describe()
        counted = self.counts()
        return (f"{state.describe()}. {counted['subjects']} subject(s), "
                f"{counted['faces']} face observation(s), {counted['plates']} plate reading(s)")
