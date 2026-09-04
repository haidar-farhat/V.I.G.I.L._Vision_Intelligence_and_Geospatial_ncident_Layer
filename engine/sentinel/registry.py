"""The register behind People and Vehicles: who is enrolled, and how to undo it.

... -> A TRACK -> AN IDENTIFIER -> A NAME, ON PURPOSE -> ...

A camera produces tracks. A track is anonymous, and that is the default this
system has always kept. This module is where an operator may deliberately
attach a name to one — a face template for a member of staff, a plate for a
contractor's van — and, far more importantly, where the machinery for taking
that name away again lives.

**One module for both features, deliberately.** A person's face and a vehicle's
plate carry very different weights of claim, but the safeguards around them are
the same safeguards: an actor, a lawful basis, a retention clock and a delete
that really deletes. Written twice they would drift, and the copy that drifted
would be the one holding biometrics — the half nobody may get wrong. So the two
features share these tables, and the differences between them are a column, not
a codebase.

Four properties hold the whole module up, each of them naming a way a register
like this becomes indefensible:

**Nothing is ever enrolled by being observed.** :meth:`Register.enrol` is the
only path by which an identifier reaches the database, and it demands an actor
and a lawful basis before it will run. There is no code path from "a face was
detected" to "a row exists", because that path is how an identity database gets
built by accident out of passers-by who were never asked.

**A register with no delete is not lawful in most jurisdictions, and not
defensible in any of them.** :meth:`Register.forget` removes every identifier
and every link to a movement history, and returns exactly what it removed so
the caller can audit the erasure. :meth:`Register.sweep_expired` does the same
on a clock, so that forgetting is the default and keeping is the exception a
pin has to be set for.

**A stored claim carries its evidence.** A sighting recorded from a match
carries the score that produced it and whether that score was over the high
threshold or between the two. A sighting an operator declared carries no score
and says so. Nothing in here can express "this is Ali" with nothing behind it,
because the column that would have to be empty is refused.

**Personal data travels with the return value; the audit log gets identifiers
only.** Every mutating call returns a value object rich enough for the console
to show the operator what just happened — including the name or the plate — and
a ``detail()`` string that carries ids, kinds and counts and nothing else. The
audit trail is append-only by design, so writing a forgotten person's name into
it would leave the log holding the last copy of the thing that was erased. This
module writes no audit rows itself; it hands the caller the material for one.

**This module loads no model and reads no image.** A face template arrives here
as bytes some operator-supplied encoder produced, tagged with the name of the
model that produced it, and a plate arrives as text some operator-supplied
recogniser already resolved. Nothing is downloaded, and a missing model is a
problem for the module that wanted to run one — never for this one.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import string
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Iterator, Sequence

from .logs import get as _get_logger

_log = _get_logger(__name__)

#: The character a plate recogniser writes where it could not resolve a
#: character. A plate containing one is a partial read, and this module refuses
#: to enrol or match on it: ``B?7 4?21`` presented as ``BX7 4921`` is an
#: invented registration attached to a real person.
UNRESOLVED = "?"

#: Milliseconds in a day, spelled once. Retention is configured in days because
#: that is the unit an operator's policy is written in.
_DAY_MILLIS = 86_400_000

_savepoints = count()


class RegistryError(RuntimeError):
    """The register refused a write, and the message says what was missing."""


class SubjectKind(str, Enum):
    """What kind of thing is enrolled.

    Kept as a column rather than as two tables, so that the delete, the
    retention sweep and the provenance columns cannot come to differ between
    the biometric half and the other one.
    """

    PERSON = "PERSON"
    VEHICLE = "VEHICLE"


class IdentifierKind(str, Enum):
    """What is stored *about* a subject, and therefore how heavy it is.

    A face template is a biometric: derived from somebody's body, unchangeable
    if it leaks, and lawful to hold only under a basis the operator can name. A
    plate is a legally displayed identifier photographed in public — still
    personal data almost everywhere, but a materially weaker claim on the person
    behind it. The retention policy prices that difference; nothing else here
    does.
    """

    FACE_TEMPLATE = "FACE_TEMPLATE"
    PLATE = "PLATE"

    @property
    def subject_kind(self) -> SubjectKind:
        """Which kind of subject this identifier can belong to.

        Derived rather than passed, so that "a vehicle with a face template" is
        not a state the database can reach and then have to be interpreted.
        """
        return (
            SubjectKind.PERSON
            if self is IdentifierKind.FACE_TEMPLATE
            else SubjectKind.VEHICLE
        )


class Confidence(str, Enum):
    """How strong the claim behind a sighting is.

    The three values are not a scale of feeling. ``DECLARED`` is an operator
    saying so and carries no score, because there is no score behind a human
    assertion and inventing one would be worse than having none. ``MATCH`` and
    ``POSSIBLE`` come from a matcher and must both carry the score that produced
    them — the distinction between them is the console's *match* / *possible
    match* rendering, and a stored row that lost it would let one panel show the
    certain form of a name the next panel hedges.
    """

    DECLARED = "DECLARED"
    MATCH = "MATCH"
    POSSIBLE = "POSSIBLE"


# --------------------------------------------------------------- normalisation


@dataclass(frozen=True, slots=True)
class PlateFormat:
    """How this site's country writes a registration down.

    ``B 7421`` and ``B-7421`` are one vehicle, and a register that stored them
    as two would answer "no such vehicle" to a watchlist query about a van that
    is on the watchlist. So a plate is stored in a normalised form, with the raw
    read kept beside it: the normalised form is what is matched, the raw read is
    what an operator is shown when they ask why.

    What normalisation deliberately does **not** do is fold confusable
    characters. ``B0X`` and ``BOX`` look alike on a wet plate at thirty metres,
    and mapping ``O`` to ``0`` would quietly merge two vehicles into one entry
    that no later correction can separate, because the evidence for which was
    which has been thrown away. Resolving that ambiguity is the recogniser's
    job, and where it cannot, its output contains ``?`` and this module refuses
    it.
    """

    #: Everything a plate may consist of once punctuation and spacing are gone.
    alphabet: str = string.ascii_uppercase + string.digits

    def normalise(self, text: str) -> str:
        """The matchable form of a read, or a refusal explaining why not.

        Compatibility-normalises first, so full-width and presentation forms
        collapse onto the plain characters, then folds Arabic-Indic digits onto
        ASCII ones — a faithful transliteration of the same digit rather than a
        guess about which character was meant — then upper-cases and drops
        everything outside the alphabet.
        """
        if UNRESOLVED in text:
            raise RegistryError(
                f"the read {text!r} has unresolved characters, so it is not a "
                "plate: a partial read must never be enrolled or matched, "
                "because completing it invents a registration"
            )
        folded = unicodedata.normalize("NFKC", text).translate(_ARABIC_INDIC_DIGITS)
        kept = "".join(c for c in folded.upper() if c in self.alphabet)
        if not kept:
            raise RegistryError(
                f"the read {text!r} contains no plate characters at all"
            )
        return kept


#: Arabic-Indic and Eastern Arabic-Indic digits onto ASCII. Unicode's own
#: compatibility decomposition leaves these alone — they are distinct digits,
#: not presentation forms — so a Beirut plate photographed with Arabic numerals
#: would otherwise normalise to nothing and be silently unmatchable.
_ARABIC_INDIC_DIGITS = {
    **{0x0660 + n: ord(str(n)) for n in range(10)},
    **{0x06F0 + n: ord(str(n)) for n in range(10)},
}

#: The format assumed when the site has not configured one.
DEFAULT_PLATE_FORMAT = PlateFormat()


# ---------------------------------------------------------------- what is held


@dataclass(frozen=True, slots=True)
class FaceTemplate:
    """A face reduced to a vector, ready to be enrolled.

    ``vector`` is whatever the operator-supplied encoder produced — SFace's 128
    floats, packed. It is not an image and cannot be turned back into one, which
    is the whole reason storing it is defensible where storing the crop would
    need its own justification.

    ``model`` names the file that produced it, and it is required. Cosine
    distance between templates from two different encoders is a number, it looks
    exactly like a score, and it means nothing at all — the most dangerous kind
    of number a security product can print. Recording the model is what lets
    :meth:`Register.templates` refuse to hand a matcher a candidate it cannot
    legitimately compare.
    """

    #: Not in the repr: a few hundred bytes of binary in a traceback is noise,
    #: and it is noise derived from somebody's face.
    vector: bytes = field(repr=False)
    model: str
    #: The encoder's own quality estimate, where it offers one. Kept beside the
    #: template because a template taken from a blurred, half-turned face is a
    #: worse thing to match against and an operator reviewing the register
    #: deserves to see which ones those are.
    quality: float | None = None

    kind = IdentifierKind.FACE_TEMPLATE


@dataclass(frozen=True, slots=True)
class Plate:
    """A registration as the recogniser resolved it, ready to be enrolled.

    ``text`` is the raw read. The register normalises it and stores both, so
    that matching is done on the normalised form and any argument about it can
    be had against the characters actually seen.
    """

    text: str
    #: Which reading agreed. A plate is read from a track, not from a frame:
    #: one frame is a guess, and the same characters agreeing across several
    #: frames is a reading. Kept because an operator asked to trust an entry is
    #: entitled to know whether it rests on two frames or twenty.
    frames_agreeing: int | None = None

    kind = IdentifierKind.PLATE


#: What :meth:`Register.enrol` accepts. The union is the whole vocabulary: there
#: is no free-form "identifier" this module will store on trust.
NewIdentifier = FaceTemplate | Plate


@dataclass(frozen=True, slots=True)
class Subject:
    """A person or a vehicle somebody decided to name."""

    id: str
    kind: SubjectKind
    display_name: str
    notes: str | None
    #: Exempt from the retention sweep. A pin is the operator saying "this one
    #: is still current", and it is the only thing that survives a sweep, so it
    #: is deliberately an explicit act with its own audit row rather than a
    #: default.
    pinned: bool
    created_at_millis: int
    updated_at_millis: int


@dataclass(frozen=True, slots=True)
class Identifier:
    """One stored identifier, with the provenance that makes holding it lawful.

    ``enrolled_by``, ``enrolled_at_millis`` and ``basis`` are not decoration.
    "Who put this here, when, and under what basis" is the first question asked
    of any register of this kind, and a row that cannot answer it is a row that
    has to be deleted rather than defended.
    """

    id: str
    subject_id: str
    kind: IdentifierKind
    enrolled_by: str
    enrolled_at_millis: int
    basis: str
    #: For a face: the packed vector and the encoder that produced it.
    template: bytes | None = field(default=None, repr=False)
    model: str | None = None
    quality: float | None = None
    #: For a plate: the matchable form, and the characters actually read.
    plate: str | None = None
    raw_text: str | None = None
    frames_agreeing: int | None = None
    #: Where the enrolment was taken from, when it came off a track. Null when
    #: an operator typed a plate in from a contractor's paperwork.
    source_camera: str | None = None
    source_track: int | None = None

    def age_millis(self, now_millis: int) -> int:
        return max(0, now_millis - self.enrolled_at_millis)


@dataclass(frozen=True, slots=True)
class Sighting:
    """One track this subject was recognised on: the movement history.

    This is the feature the rest exists to serve — *where has this person been*
    — and it is also the most sensitive thing here, because it is a record of
    somebody's movements rather than of an incident. It holds no coordinates:
    the position of a track and its uncertainty live with the track, and a
    second copy here would be a second answer nobody could reconcile.
    """

    subject_id: str
    camera_id: str
    track_id: int
    first_seen_millis: int
    last_seen_millis: int
    confidence: Confidence
    #: The score behind a ``MATCH`` or ``POSSIBLE``; ``None`` for ``DECLARED``,
    #: where there is no score and pretending otherwise would be a fabrication.
    score: float | None
    #: The identifier that matched, where one did, so an operator asking "why
    #: does it think this is her?" can be shown the enrolment it came from.
    identifier_id: str | None

    @property
    def duration_millis(self) -> int:
        return max(0, self.last_seen_millis - self.first_seen_millis)


# ------------------------------------------------------------- what is returned


@dataclass(frozen=True, slots=True)
class Enrolment:
    """What :meth:`Register.enrol` did, in enough detail to audit it."""

    subject: Subject
    identifier: Identifier
    #: True when this call created the subject rather than adding to one.
    created_subject: bool
    #: True when the identical identifier was already enrolled and this call
    #: refreshed its provenance instead of adding a duplicate row.
    replaced: bool

    action = "register.enrol"

    def detail(self) -> str:
        return _detail(
            subject_id=self.subject.id,
            subject_kind=self.subject.kind.value,
            identifier_id=self.identifier.id,
            identifier_kind=self.identifier.kind.value,
            basis=self.identifier.basis,
            model=self.identifier.model,
            created_subject=self.created_subject,
            replaced=self.replaced,
        )


@dataclass(frozen=True, slots=True)
class Forgotten:
    """What :meth:`Register.forget` erased.

    ``display_name`` is here for the operator's confirmation — "Forgot 'Ali
    Hassan': 3 templates, 41 sightings" is what makes an irreversible action
    reviewable at the moment it is taken. It is deliberately absent from
    :meth:`detail`, because the audit trail is append-only and a name written
    into it would outlive the erasure it is recording.
    """

    subject_id: str
    #: False when there was nothing to forget. Not an error: forgetting is
    #: idempotent on purpose, so a retried delete cannot fail and leave an
    #: operator believing data survived.
    found: bool
    kind: SubjectKind | None
    display_name: str | None
    identifier_ids: tuple[str, ...]
    sightings_unlinked: int
    at_millis: int

    action = "register.forget"

    @property
    def identifiers_deleted(self) -> int:
        return len(self.identifier_ids)

    def detail(self) -> str:
        return _detail(
            subject_id=self.subject_id,
            found=self.found,
            subject_kind=self.kind.value if self.kind else None,
            identifiers_deleted=self.identifiers_deleted,
            identifier_ids=list(self.identifier_ids),
            sightings_unlinked=self.sightings_unlinked,
        )


@dataclass(frozen=True, slots=True)
class Sweep:
    """What :meth:`Register.sweep_expired` deleted, and what a pin saved.

    ``kept_pinned`` is reported rather than passed over in silence. An
    identifier that is past its retention and still in the database is exactly
    the thing an audit asks about, and the honest answer — "a named operator
    pinned that subject" — only exists if the sweep says which ones they were.
    """

    at_millis: int
    deleted: tuple[Identifier, ...]
    kept_pinned: tuple[Identifier, ...]
    examined: int

    action = "register.sweep"

    def detail(self) -> str:
        return _detail(
            at=self.at_millis,
            examined=self.examined,
            deleted=len(self.deleted),
            deleted_ids=[identifier.id for identifier in self.deleted],
            kept_pinned=len(self.kept_pinned),
            kept_pinned_ids=[identifier.id for identifier in self.kept_pinned],
        )


@dataclass(frozen=True, slots=True)
class Pinning:
    """What :meth:`Register.set_pinned` changed."""

    subject: Subject
    pinned: bool
    changed: bool

    action = "register.pin"

    def detail(self) -> str:
        return _detail(
            subject_id=self.subject.id,
            subject_kind=self.subject.kind.value,
            pinned=self.pinned,
            changed=self.changed,
        )


def _detail(**fields: object) -> str:
    """The audit payload: ids, kinds and counts, never names or plates.

    Sorted keys so two runs of the same action produce byte-identical detail,
    which is what makes an audit trail diffable.
    """
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------- retention


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long an identifier may be kept before the sweep deletes it.

    The face default is short on purpose. A biometric held indefinitely because
    nobody chose a number is held without a decision behind it, and "we never
    got round to setting it" is not a defence anybody has ever won with. A plate
    is kept longer because the claim is weaker and the operational case — a
    contractor's van returning next quarter — is real.

    ``None`` means *no expiry*, which is a deliberate operator choice with an
    audit row behind it and never a default.

    Retention is evaluated against ``enrolled_at`` at sweep time rather than
    written into the row as an expiry date, so shortening the policy takes
    effect on everything already stored. A policy change that only applied to
    future enrolments would leave the oldest and most exposed templates — the
    ones the change was made for — untouched.
    """

    face_template_days: float | None = 30.0
    plate_days: float | None = 365.0

    def retention_millis(self, kind: IdentifierKind) -> int | None:
        days = (
            self.face_template_days
            if kind is IdentifierKind.FACE_TEMPLATE
            else self.plate_days
        )
        if days is None:
            return None
        if days < 0:
            raise RegistryError(
                f"a retention of {days} days is not a duration; use None to "
                "mean 'keep indefinitely', so that the choice is explicit"
            )
        return int(days * _DAY_MILLIS)

    def describe(self) -> str:
        def phrase(days: float | None) -> str:
            return "kept indefinitely" if days is None else f"{days:g} days"

        return (
            f"face templates {phrase(self.face_template_days)}, "
            f"plates {phrase(self.plate_days)}"
        )


# ---------------------------------------------------------------------- schema


#: The tables, as separate statements so a migration can execute them one at a
#: time inside its own transaction. ``IF NOT EXISTS`` throughout, because this
#: schema is created both by the store's migration and by a `Register` handed a
#: bare connection, and the two must not fight.
SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS register_subjects (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL CHECK (kind IN ('PERSON', 'VEHICLE')),
        display_name  TEXT NOT NULL,
        notes         TEXT,
        -- Exempt from the retention sweep, and the only thing that is.
        pinned        INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
        created_at    INTEGER NOT NULL,
        updated_at    INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS register_identifiers (
        id            TEXT PRIMARY KEY,
        subject_id    TEXT NOT NULL
                      REFERENCES register_subjects(id) ON DELETE CASCADE,
        kind          TEXT NOT NULL
                      CHECK (kind IN ('FACE_TEMPLATE', 'PLATE')),
        -- Provenance. Not nullable, because a row that cannot say who put it
        -- here and under what basis is a row that has to be deleted rather
        -- than defended.
        enrolled_by   TEXT NOT NULL,
        enrolled_at   INTEGER NOT NULL,
        basis         TEXT NOT NULL,
        -- A face: the packed vector and the encoder that produced it. The
        -- model is stored because a distance between templates from two
        -- encoders is a meaningless number that looks exactly like a score.
        template      BLOB,
        model         TEXT,
        quality       REAL,
        -- A plate: the matchable form and the characters actually read.
        plate         TEXT,
        raw_text      TEXT,
        frames_agreeing INTEGER,
        source_camera TEXT,
        source_track  INTEGER,
        -- One shape or the other, never both and never neither. Without this a
        -- FACE_TEMPLATE row with a null template is representable, and it
        -- would be read as an enrolment that matches nothing rather than as
        -- the corruption it is.
        CHECK (
            (kind = 'FACE_TEMPLATE' AND template IS NOT NULL
                 AND model IS NOT NULL AND plate IS NULL)
            OR
            (kind = 'PLATE' AND plate IS NOT NULL AND template IS NULL)
        )
    )
    """,
    # One vehicle per plate. Two subjects sharing a registration means a
    # watchlist hit points at the wrong owner, and there is no evidence left to
    # say which.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS register_plate_unique
        ON register_identifiers (plate) WHERE plate IS NOT NULL
    """,
    # One row per (subject, template). Re-enrolling the same face refreshes
    # its provenance; without this a nervous double-click would double the
    # amount of biometric data held, and every later count would be wrong.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS register_template_unique
        ON register_identifiers (subject_id, template) WHERE template IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS register_identifiers_by_subject
        ON register_identifiers (subject_id)
    """,
    # The sweep's query: oldest first, within a kind.
    """
    CREATE INDEX IF NOT EXISTS register_identifiers_by_age
        ON register_identifiers (kind, enrolled_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS register_sightings (
        subject_id    TEXT NOT NULL
                      REFERENCES register_subjects(id) ON DELETE CASCADE,
        camera_id     TEXT NOT NULL,
        track_id      INTEGER NOT NULL,
        first_seen    INTEGER NOT NULL,
        last_seen     INTEGER NOT NULL,
        -- DECLARED carries no score; MATCH and POSSIBLE must carry one. The
        -- constraint is here as well as in Python because a claim without its
        -- evidence must not be reachable by any writer of this database.
        confidence    TEXT NOT NULL
                      CHECK (confidence IN ('DECLARED', 'MATCH', 'POSSIBLE')),
        score         REAL,
        identifier_id TEXT
                      REFERENCES register_identifiers(id) ON DELETE SET NULL,
        CHECK (
            (confidence = 'DECLARED' AND score IS NULL)
            OR (confidence <> 'DECLARED' AND score IS NOT NULL)
        ),
        -- One row per track. A track is the unit a match is decided over, so a
        -- second row for the same track would be the same encounter counted
        -- twice in a movement history.
        PRIMARY KEY (subject_id, camera_id, track_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS register_sightings_by_subject_time
        ON register_sightings (subject_id, first_seen)
    """,
)

_SUBJECT_COLUMNS = (
    "id, kind, display_name, notes, pinned, created_at, updated_at"
)
_IDENTIFIER_COLUMNS = (
    "id, subject_id, kind, enrolled_by, enrolled_at, basis, template, model, "
    "quality, plate, raw_text, frames_agreeing, source_camera, source_track"
)
_SIGHTING_COLUMNS = (
    "subject_id, camera_id, track_id, first_seen, last_seen, confidence, "
    "score, identifier_id"
)


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the register's tables on this connection if they are absent.

    Idempotent, and safe to call on a database the store has already migrated:
    a `Register` may be handed a connection from anywhere, and refusing to open
    one whose tables happen to exist already would make the console's startup
    order load-bearing.
    """
    for statement in SCHEMA:
        connection.execute(statement)


def _now() -> int:
    return int(time.time() * 1000)


def new_identifier_id() -> str:
    """An opaque id for an identifier: random, not derived from the content.

    Everywhere else in this system an id is a hash of what the thing *is*, so
    that replay is idempotent. Not here, and the exception is the point. A
    plate is drawn from a space of a few tens of billions of strings, so a
    ``sha256`` of one is not a pseudonym — it is the plate, recoverable by
    anybody willing to spend an afternoon on it. Since the id is the part that
    goes into the append-only audit log, deriving it from the content would
    quietly undo the separation this module is built around.

    Idempotence is kept by looking the identifier up on its natural key — this
    subject, this template or this plate — before inserting, so the obvious
    operator gesture of clicking "Enrol" twice refreshes one row rather than
    doubling the amount of biometric data held.
    """
    return "id_" + secrets.token_hex(12)


def _required(value: str | None, what: str, because: str) -> str:
    if value is None or not value.strip():
        raise RegistryError(f"{what} is required: {because}")
    return value.strip()


# -------------------------------------------------------------------- the register


class Register:
    """Subjects, their identifiers and their movement history, over a connection.

    Owns its own tables and creates them on construction, so that a caller with
    any `sqlite3.Connection` — the store's, or one opened for a migration tool —
    gets a working register without a separate setup step.

    Every multi-statement write is wrapped in a savepoint rather than in
    ``BEGIN``. The connection belongs to the caller and may already be inside a
    transaction of theirs; a savepoint composes with that, while a ``BEGIN``
    would either fail or, worse, commit half of somebody else's work. For the
    same reason nothing here changes the connection's pragmas: foreign keys may
    or may not be on, so :meth:`forget` deletes each table explicitly instead of
    trusting a cascade that may not fire.
    """

    __slots__ = ("_connection", "_plate_format")

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        plate_format: PlateFormat = DEFAULT_PLATE_FORMAT,
    ):
        self._connection = connection
        self._plate_format = plate_format
        create_schema(connection)

    @property
    def plate_format(self) -> PlateFormat:
        return self._plate_format

    @contextmanager
    def _unit(self) -> Iterator[sqlite3.Connection]:
        name = f"registry_{next(_savepoints)}"
        self._connection.execute(f"SAVEPOINT {name}")
        try:
            yield self._connection
        except Exception:
            self._connection.execute(f"ROLLBACK TO {name}")
            raise
        finally:
            self._connection.execute(f"RELEASE {name}")

    # ---------------------------------------------------------------- enrolment

    def enrol(
        self,
        *,
        subject_id: str,
        display_name: str,
        identifier: NewIdentifier,
        actor: str,
        basis: str,
        notes: str | None = None,
        source_camera: str | None = None,
        source_track: int | None = None,
        now_millis: int | None = None,
    ) -> Enrolment:
        """Attach an identifier to a named subject. The only way one is added.

        There is deliberately no other path into `register_identifiers`. A face
        that was detected, embedded and matched produces a sighting at most;
        **nothing is ever enrolled by being observed**, because a system that
        enrols what it sees builds a gallery of strangers who were never asked
        and could never have objected. Enrolment is an operator naming a track,
        once, on purpose.

        ``actor`` and ``basis`` are refused when blank, and that refusal is the
        point of the signature. "Who decided this, and under what lawful basis"
        is the first question asked of a register like this one, and a default
        value for either — ``"system"``, ``"unknown"`` — is how the answer
        becomes a lie six months later. The basis is free text because it is the
        site's own list; this module will not pretend to know a jurisdiction's
        vocabulary.

        Enrolling onto an existing subject adds to it and does **not** rename
        it. Renaming somebody as a side effect of enrolling a second template is
        an edit nobody asked for and nobody would see; a rename needs its own
        call and its own audit row.

        Raises `RegistryError` on a missing actor or basis, an empty or partial
        plate, an empty template, a template with no model named, or a subject
        id already held by the other kind of subject.
        """
        at = _now() if now_millis is None else now_millis
        subject_id = _required(subject_id, "a subject id", "it is the handle every later delete and audit row uses")
        display_name = _required(
            display_name,
            "a display name",
            "an entry nobody can recognise cannot be reviewed, renamed or forgotten on purpose",
        )
        actor = _required(
            actor,
            "an actor",
            "an enrolment nobody is recorded as having made cannot be defended, "
            "and nothing is ever enrolled by being observed",
        )
        basis = _required(
            basis,
            "a lawful basis",
            "holding an identifier without one is not lawful in most "
            "jurisdictions, and the operator must choose it, not this code",
        )

        kind = identifier.kind
        subject_kind = kind.subject_kind
        row = self._subject_row(subject_id)
        if row is not None and row[1] != subject_kind.value:
            raise RegistryError(
                f"{subject_id!r} is already a {row[1]} and cannot also hold a "
                f"{kind.value}: reusing an id across kinds would merge a person "
                "and a vehicle into one entry"
            )

        if isinstance(identifier, FaceTemplate):
            if not identifier.vector:
                raise RegistryError(
                    "an empty face template matches everything or nothing "
                    "depending on the metric, and either way it is not a face"
                )
            model = _required(
                identifier.model,
                "the model that produced a template",
                "a distance between templates from two different encoders is a "
                "meaningless number that looks exactly like a score",
            )
            values = dict(
                template=identifier.vector,
                model=model,
                quality=identifier.quality,
                plate=None,
                raw_text=None,
                frames_agreeing=None,
            )
            # The natural key of a face enrolment: this subject, this vector.
            already = (
                "SELECT id FROM register_identifiers WHERE subject_id = ? "
                "AND kind = ? AND template = ?",
                (subject_id, kind.value, identifier.vector),
            )
        else:
            normalised = self._plate_format.normalise(identifier.text)
            values = dict(
                template=None,
                model=None,
                quality=None,
                plate=normalised,
                raw_text=identifier.text,
                frames_agreeing=identifier.frames_agreeing,
            )
            # A plate is unique across the whole register, so the natural key
            # does not include the subject: enrolling a registration that
            # belongs to somebody else must be refused, not silently added.
            already = (
                "SELECT id FROM register_identifiers WHERE plate = ?",
                (normalised,),
            )

        with self._unit() as connection:
            created = row is None
            if created:
                connection.execute(
                    "INSERT INTO register_subjects "
                    "(id, kind, display_name, notes, pinned, created_at, updated_at) "
                    "VALUES (?,?,?,?,0,?,?)",
                    (subject_id, subject_kind.value, display_name, notes, at, at),
                )
            else:
                connection.execute(
                    "UPDATE register_subjects SET updated_at = ? WHERE id = ?",
                    (at, subject_id),
                )

            existing = connection.execute(*already).fetchone()
            stored_id = existing[0] if existing is not None else new_identifier_id()
            try:
                connection.execute(
                    "INSERT INTO register_identifiers "
                    "(id, subject_id, kind, enrolled_by, enrolled_at, basis, "
                    " template, model, quality, plate, raw_text, "
                    " frames_agreeing, source_camera, source_track) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    " enrolled_by = excluded.enrolled_by, "
                    " enrolled_at = excluded.enrolled_at, "
                    " basis = excluded.basis, "
                    " quality = excluded.quality, "
                    " raw_text = excluded.raw_text, "
                    " frames_agreeing = excluded.frames_agreeing, "
                    " source_camera = excluded.source_camera, "
                    " source_track = excluded.source_track",
                    (
                        stored_id,
                        subject_id,
                        kind.value,
                        actor,
                        at,
                        basis,
                        values["template"],
                        values["model"],
                        values["quality"],
                        values["plate"],
                        values["raw_text"],
                        values["frames_agreeing"],
                        source_camera,
                        source_track,
                    ),
                )
            except sqlite3.IntegrityError as error:
                # The plate index fired: this registration is already enrolled
                # to a different subject. Reported rather than merged, because
                # merging two vehicles is not reversible and a watchlist hit
                # against the merged entry would name the wrong owner.
                raise RegistryError(
                    f"that plate is already enrolled to another subject: {error}"
                ) from error

            stored = self._identifier(stored_id, connection)
            subject = self.subject(subject_id)

        assert stored is not None and subject is not None
        # Deliberately no name, plate or template in the log: this file's
        # reason for existing is that those live in exactly one place.
        _log.info(
            "register: %s enrolled for subject %s by %s (basis recorded)",
            kind.value, subject_id, actor,
        )
        return Enrolment(
            subject=subject,
            identifier=stored,
            created_subject=created,
            replaced=existing is not None,
        )

    def set_pinned(
        self, subject_id: str, pinned: bool, *, actor: str, now_millis: int | None = None
    ) -> Pinning:
        """Exempt a subject from the retention sweep, or stop exempting it.

        A pin is the one thing that keeps an identifier past its retention, so
        it is an explicit act by a named actor and returns enough to audit. An
        unknown subject is a `RegistryError`: silently pinning nothing would
        leave an operator believing a record is protected when the sweep is
        about to delete it.
        """
        at = _now() if now_millis is None else now_millis
        actor = _required(
            actor, "an actor", "a pin is what keeps data past its retention"
        )
        subject = self.subject(subject_id)
        if subject is None:
            raise RegistryError(f"no subject {subject_id!r} to pin")
        changed = subject.pinned != pinned
        if changed:
            with self._unit() as connection:
                connection.execute(
                    "UPDATE register_subjects SET pinned = ?, updated_at = ? "
                    "WHERE id = ?",
                    (1 if pinned else 0, at, subject_id),
                )
            subject = self.subject(subject_id)
            assert subject is not None
        return Pinning(subject=subject, pinned=pinned, changed=changed)

    # ------------------------------------------------------------------ reading

    def subject(self, subject_id: str) -> Subject | None:
        row = self._subject_row(subject_id)
        return None if row is None else _subject_from(row)

    def subjects(self, *, kind: SubjectKind | None = None) -> tuple[Subject, ...]:
        """Everybody enrolled, by name, so the People and Vehicles menus can list them."""
        sql = f"SELECT {_SUBJECT_COLUMNS} FROM register_subjects"
        params: tuple[object, ...] = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            params = (kind.value,)
        sql += " ORDER BY display_name, id"
        rows = self._connection.execute(sql, params).fetchall()
        return tuple(_subject_from(row) for row in rows)

    def identifiers(self, subject_id: str) -> tuple[Identifier, ...]:
        """Everything held about one subject, oldest enrolment first."""
        rows = self._connection.execute(
            f"SELECT {_IDENTIFIER_COLUMNS} FROM register_identifiers "
            "WHERE subject_id = ? ORDER BY enrolled_at, id",
            (subject_id,),
        ).fetchall()
        return tuple(_identifier_from(row) for row in rows)

    def find_plate(self, text: str) -> Subject | None:
        """The subject whose plate this read matches, or ``None``.

        Exact equality on the normalised form, and the only place in this module
        where a name is returned without a score beside it. That is allowed
        precisely because the evidence *is* exact: two identical strings, not
        two similar ones. Whether the characters were read correctly is the
        recogniser's question, and a read it could not resolve carries ``?`` and
        is refused here rather than completed.
        """
        normalised = self._plate_format.normalise(text)
        row = self._connection.execute(
            f"SELECT s.{', s.'.join(_SUBJECT_COLUMNS.split(', '))} "
            "FROM register_subjects s "
            "JOIN register_identifiers i ON i.subject_id = s.id "
            "WHERE i.plate = ?",
            (normalised,),
        ).fetchone()
        return None if row is None else _subject_from(row)

    def templates(self, *, model: str) -> tuple[Identifier, ...]:
        """Every face template a matcher may legitimately compare against.

        Filtered by model, and that filter is the whole point of the method. A
        cosine distance between an SFace template and one from another encoder
        is a well-formed float with no meaning; handing a matcher the union of
        both would produce confident matches against nobody in particular.
        """
        rows = self._connection.execute(
            f"SELECT {_IDENTIFIER_COLUMNS} FROM register_identifiers "
            "WHERE kind = ? AND model = ? ORDER BY subject_id, enrolled_at",
            (IdentifierKind.FACE_TEMPLATE.value, model),
        ).fetchall()
        return tuple(_identifier_from(row) for row in rows)

    # ------------------------------------------------------------- the history

    def record_sighting(
        self,
        *,
        subject_id: str,
        camera_id: str,
        track_id: int,
        first_seen_millis: int,
        last_seen_millis: int,
        confidence: Confidence,
        score: float | None = None,
        identifier_id: str | None = None,
    ) -> Sighting:
        """Note that a subject was recognised on a track.

        A sighting never creates a subject. An unknown ``subject_id`` is a
        `RegistryError`, because the alternative — inserting the subject —
        would be exactly the automatic enrolment this whole module exists to
        make impossible, arriving through the back door.

        The claim must carry its evidence: ``MATCH`` and ``POSSIBLE`` are
        refused without a score, and ``DECLARED`` is refused with one, since an
        operator naming somebody produces no score and any number attached to it
        would be invented.

        One row per track, and re-recording the same track widens the window and
        replaces the claim. Aggregation across the frames of a track belongs to
        the matcher: a decision made on one frame is a guess, and this method
        stores the conclusion the matcher reached, not the frames behind it.
        """
        if last_seen_millis < first_seen_millis:
            raise RegistryError(
                f"a sighting cannot end ({last_seen_millis}) before it began "
                f"({first_seen_millis})"
            )
        if confidence is Confidence.DECLARED and score is not None:
            raise RegistryError(
                "a declared sighting carries no score: an operator naming "
                "somebody produces a decision, not a measurement"
            )
        if confidence is not Confidence.DECLARED and score is None:
            raise RegistryError(
                f"a {confidence.value} sighting must carry the score that "
                "produced it; a name shown without its evidence is the failure "
                "this register is built to prevent"
            )
        if self._subject_row(subject_id) is None:
            raise RegistryError(
                f"no subject {subject_id!r}: a sighting never enrols anybody, "
                "so the subject must have been enrolled deliberately first"
            )

        with self._unit() as connection:
            connection.execute(
                "INSERT INTO register_sightings "
                f"({_SIGHTING_COLUMNS}) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(subject_id, camera_id, track_id) DO UPDATE SET "
                " first_seen = MIN(first_seen, excluded.first_seen), "
                " last_seen = MAX(last_seen, excluded.last_seen), "
                " confidence = excluded.confidence, "
                " score = excluded.score, "
                " identifier_id = excluded.identifier_id",
                (
                    subject_id,
                    camera_id,
                    track_id,
                    first_seen_millis,
                    last_seen_millis,
                    confidence.value,
                    score,
                    identifier_id,
                ),
            )
            row = connection.execute(
                f"SELECT {_SIGHTING_COLUMNS} FROM register_sightings "
                "WHERE subject_id = ? AND camera_id = ? AND track_id = ?",
                (subject_id, camera_id, track_id),
            ).fetchone()
        assert row is not None
        return _sighting_from(row)

    def history(self, subject_id: str) -> tuple[Sighting, ...]:
        """Where this subject has been, oldest first.

        Chronological because that is what a movement history *is*: the panel
        draws a trail and the trail has a direction. Ordered by camera and track
        within a millisecond so that two sightings that begin in the same
        millisecond — which happens on a multi-camera site — come back in a
        stable order rather than in whatever order the pages happen to be read.
        """
        rows = self._connection.execute(
            f"SELECT {_SIGHTING_COLUMNS} FROM register_sightings "
            "WHERE subject_id = ? ORDER BY first_seen, camera_id, track_id",
            (subject_id,),
        ).fetchall()
        return tuple(_sighting_from(row) for row in rows)

    # -------------------------------------------------------------- forgetting

    def forget(self, subject_id: str, *, now_millis: int | None = None) -> Forgotten:
        """Delete every identifier for a subject and unlink its history.

        **A register with no delete is not lawful in most jurisdictions and not
        defensible in any of them.** This is that delete. Every template and
        every plate goes, every row linking the subject to a track goes, and the
        subject itself goes with them.

        What is deliberately *not* touched is the footage and the tracks. Those
        are evidence about an incident, they are governed by the recording
        retention policy, and they were never claims about who anybody is — the
        register only ever held the *link* between a track and a name, and
        removing the link is the whole of forgetting. A person who has been
        forgotten leaves anonymous tracks behind, which is the state everybody
        who was never enrolled has always been in.

        Returns what it removed, so the caller can write one audit row proving
        the erasure happened. Forgetting an unknown subject is not an error: it
        returns ``found=False`` with nothing deleted, because a retried delete
        must not fail in a way that leaves an operator unsure whether data
        survived.
        """
        at = _now() if now_millis is None else now_millis
        row = self._subject_row(subject_id)
        if row is None:
            return Forgotten(
                subject_id=subject_id,
                found=False,
                kind=None,
                display_name=None,
                identifier_ids=(),
                sightings_unlinked=0,
                at_millis=at,
            )

        subject = _subject_from(row)
        identifiers = self.identifiers(subject_id)

        with self._unit() as connection:
            # Explicit deletes in dependency order rather than a cascade: the
            # connection is the caller's and PRAGMA foreign_keys may be off, in
            # which case a cascade silently does nothing and this method would
            # report an erasure that did not happen.
            sightings = connection.execute(
                "DELETE FROM register_sightings WHERE subject_id = ?", (subject_id,)
            ).rowcount
            connection.execute(
                "DELETE FROM register_identifiers WHERE subject_id = ?", (subject_id,)
            )
            connection.execute(
                "DELETE FROM register_subjects WHERE id = ?", (subject_id,)
            )

        _log.info(
            "register: forgot subject %s — %d identifier(s), %d sighting(s)",
            subject_id, len(identifiers), max(0, sightings),
        )
        return Forgotten(
            subject_id=subject_id,
            found=True,
            kind=subject.kind,
            display_name=subject.display_name,
            identifier_ids=tuple(identifier.id for identifier in identifiers),
            sightings_unlinked=max(0, sightings),
            at_millis=at,
        )

    def sweep_expired(
        self, now_millis: int, policy: RetentionPolicy | None = None
    ) -> Sweep:
        """Delete identifiers past their retention, unless the subject is pinned.

        Forgetting on a clock, so that keeping is the exception rather than the
        thing that happens when nobody does anything. Run by the same retention
        job that deletes recorded video — the two policies are separate numbers
        but they are one decision, and a deployment where the video expires and
        the biometrics do not is the wrong way round.

        A pin is the only exemption, it belongs to the subject rather than to
        the identifier, and the pinned rows are reported rather than passed over
        in silence: an audit asking why a two-year-old template is still here
        deserves the answer "a named operator pinned it" rather than nothing.

        The subject row survives a sweep that empties it. A named entry with no
        identifiers left is a person the operator can re-enrol, and deleting the
        name because the template expired would silently discard the notes and
        the history along with it — that is :meth:`forget`'s job, and it is
        somebody's decision, not a timer's.
        """
        policy = RetentionPolicy() if policy is None else policy
        rows = self._connection.execute(
            f"SELECT i.{', i.'.join(_IDENTIFIER_COLUMNS.split(', '))}, s.pinned "
            "FROM register_identifiers i "
            "JOIN register_subjects s ON s.id = i.subject_id "
            "ORDER BY i.enrolled_at, i.id"
        ).fetchall()

        expired: list[Identifier] = []
        pinned: list[Identifier] = []
        for row in rows:
            identifier = _identifier_from(row)
            retention = policy.retention_millis(identifier.kind)
            if retention is None:
                continue
            if identifier.age_millis(now_millis) < retention:
                continue
            (pinned if row[len(_IDENTIFIER_COLUMNS.split(", "))] else expired).append(
                identifier
            )

        if expired:
            with self._unit() as connection:
                connection.executemany(
                    "DELETE FROM register_identifiers WHERE id = ?",
                    [(identifier.id,) for identifier in expired],
                )

        _log.info(
            "register sweep: %d of %d identifier(s) deleted, %d kept by a pin (%s)",
            len(expired), len(rows), len(pinned), policy.describe(),
        )
        return Sweep(
            at_millis=now_millis,
            deleted=tuple(expired),
            kept_pinned=tuple(pinned),
            examined=len(rows),
        )

    # ------------------------------------------------------------------ private

    def _subject_row(self, subject_id: str):
        return self._connection.execute(
            f"SELECT {_SUBJECT_COLUMNS} FROM register_subjects WHERE id = ?",
            (subject_id,),
        ).fetchone()

    def _identifier(self, identifier: str, connection: sqlite3.Connection) -> Identifier | None:
        row = connection.execute(
            f"SELECT {_IDENTIFIER_COLUMNS} FROM register_identifiers WHERE id = ?",
            (identifier,),
        ).fetchone()
        return None if row is None else _identifier_from(row)


# ------------------------------------------------------------- reconstruction

# Rows are read positionally rather than by name. The connection belongs to the
# caller, its ``row_factory`` may be anything or nothing, and reaching into it
# to set one would be this module editing somebody else's object. Every SELECT
# above names its columns explicitly, and both a tuple and an `sqlite3.Row`
# index the same way.


def _subject_from(row: Sequence) -> Subject:
    return Subject(
        id=row[0],
        kind=SubjectKind(row[1]),
        display_name=row[2],
        notes=row[3],
        pinned=bool(row[4]),
        created_at_millis=row[5],
        updated_at_millis=row[6],
    )


def _identifier_from(row: Sequence) -> Identifier:
    return Identifier(
        id=row[0],
        subject_id=row[1],
        kind=IdentifierKind(row[2]),
        enrolled_by=row[3],
        enrolled_at_millis=row[4],
        basis=row[5],
        template=row[6],
        model=row[7],
        quality=row[8],
        plate=row[9],
        raw_text=row[10],
        frames_agreeing=row[11],
        source_camera=row[12],
        source_track=row[13],
    )


def _sighting_from(row: Sequence) -> Sighting:
    return Sighting(
        subject_id=row[0],
        camera_id=row[1],
        track_id=row[2],
        first_seen_millis=row[3],
        last_seen_millis=row[4],
        confidence=Confidence(row[5]),
        score=row[6],
        identifier_id=row[7],
    )
