"""Faces: the opt-in half of "who is that?", and the guards that make it opt-in.

Everywhere else this system refuses to name anybody. This module is the
exception an operator may deliberately switch on — a warehouse with twelve
staff, a depot with a contractor list — and it is written so that switching it
on stays a decision somebody made rather than a default nobody noticed.

Three promises from FEATURES.md hold this file up, and each one is a mechanism
here rather than a note in a document:

**It is off until somebody turns it on.** Off does not mean the name column is
hidden; it means no face is detected, no template computed and none returned.
That is enforced by :class:`FaceEngine` itself — every public method is wrapped
by :func:`_off_returns`, so a method added later that forgets the switch fails
the test that walks the class. A caller cannot get this wrong because the caller
is not asked to get it right.

**Nobody is enrolled by being seen.** There is no function here that turns a
sighting into an :class:`EnrolledPerson`. Templates come out; a person register
is built by an operator naming a track, in code that owns the database, and
this module has no way to write one. A stranger walking past produces a
template that is compared and discarded — never an "unknown persons" gallery,
which is an identity database built by accident.

**A name is never asserted without the evidence for it.** :class:`Match` has
three outcomes, not two. Above :data:`MATCH_SIMILARITY` it is a match; between
the thresholds it is *possible*, drawn differently and firing no rule; below
:data:`POSSIBLE_SIMILARITY` nothing is claimed. The score travels with the name
everywhere — :meth:`Match.describe` cannot print one without the other — and
``bool(match)`` raises rather than quietly answering True for a maybe. That
last one is not pedantry: ``if match:`` is exactly the line that turns "we think
this might be Ali" into a door that opens for Ali.

**Why storing a template is defensible where storing a crop is not.** A
:class:`FaceTemplate` is 128 floats from an embedding network. It is not an
image, it cannot be rendered, and it cannot be inverted back into the face it
came from — the mapping is many-to-one and the network is not run backwards
here or anywhere. A face crop is a photograph of somebody, and needs its own
justification, its own retention and its own audit row. That asymmetry is the
whole reason the default is a template and the crop is discarded.

**The models are operator-supplied.** ``cv2.FaceDetectorYN`` (YuNet) and
``cv2.FaceRecognizerSF`` (SFace) ship with OpenCV 5; their weights do not. A
missing file is an error naming the file, never a reason to fetch one — this
appliance has no network at runtime and no code here that could acquire
anything.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .core import BoundingBox
from .detect import _sha256
from .logs import get as _get_logger

_log = _get_logger(__name__)

#: Length of an SFace template. Fixed by the model, asserted here because a
#: vector of the wrong length compared against a register of the right length
#: is not a weak match, it is a meaningless number — and numpy will happily
#: broadcast some of the wrong lengths into one.
TEMPLATE_DIMENSIONS = 128

#: Above this cosine similarity the name is asserted.
#:
#: OpenCV publishes 0.363 as SFace's same-identity operating point, which is
#: where a *research* benchmark balances false accepts against false rejects.
#: A security product asserting a person's name to an operator who will act on
#: it needs more margin than that, so the assertion threshold is raised well
#: clear of it. The gap between the two is not wasted: it is the band this
#: module reports as POSSIBLE instead of throwing away.
MATCH_SIMILARITY = 0.50

#: Below this, nothing is claimed at all — no name, no maybe, no row.
#:
#: This is OpenCV's own published SFace threshold. Using the library's number
#: rather than one invented here matters: it is the point the model was
#: measured at, and a lower floor of our own devising would manufacture
#: "possible" matches out of pairs the model itself calls strangers.
POSSIBLE_SIMILARITY = 0.363

#: Faces the detector is less sure of than this produce no template.
#:
#: A low-confidence YuNet box is as often a hand, a poster or a wheel arch as a
#: face. A template built from one is not a bad match, it is a permanent wrong
#: answer sitting in the register, and it will match the next poster.
MINIMUM_FACE_SCORE = 0.6

#: A face smaller than this on either side produces no template. SFace is fed a
#: 112x112 aligned crop; below roughly a third of that there is no detail left
#: to embed and the template describes the upscaling filter.
MINIMUM_FACE_PIXELS = 32

#: Face templates a track must contribute before a MATCH may be claimed for it.
#:
#: One frame is a guess. A single frame of motion blur, a profile view, or half
#: a second of somebody's hand across their face all produce a template that
#: scores wherever it likes, and the whole point of a track is that it offers
#: several. Below this count :func:`identify_track` refuses to go past POSSIBLE
#: however high the score is.
FRAMES_FOR_A_MATCH = 3

#: Default life of a stored template, in days. Short on purpose: a biometric
#: register that grows without bound is one nobody can defend, and a site that
#: needs longer sets it deliberately.
DEFAULT_RETENTION_DAYS = 30

#: A vector shorter than this has no direction, so it has no cosine with
#: anything. Rejected rather than normalised, because dividing by it produces
#: infinities that compare greater than every real match.
_MINIMUM_NORM = 1e-9

#: Slack when checking a box lies inside the frame. A box assembled from
#: floating-point arithmetic can exceed 1.0 by an ulp or two, and refusing that
#: would reject a legitimate person box over a rounding error.
_BOX_TOLERANCE = 1e-6


class FaceError(RuntimeError):
    """The face pipeline could not be constructed, or was handed nonsense."""


class Verdict(Enum):
    """What may be said about an identity, in three values rather than two.

    ``__bool__`` raises on purpose. A two-valued answer is where a hedge gets
    lost: ``if verdict:`` is True for every member of an ordinary enum,
    including :data:`NONE`, and the same line reads as "we matched somebody".
    Forcing the comparison to be written out — ``verdict is Verdict.MATCH`` —
    is the difference between an operator seeing "possible match, 0.42" and a
    rule firing on it.

    Deliberately not a ``str`` enum, unlike :class:`~sentinel.zones.ZoneKind`.
    A string member survives being formatted, concatenated and compared against
    a literal, and every one of those is a way for POSSIBLE to reach a screen
    looking like every other string. Persist ``.value``; compare the member.
    """

    #: Above :data:`MATCH_SIMILARITY`. The name may be shown, with its score.
    MATCH = "MATCH"
    #: Between the thresholds. Shown differently, and fires no rule, ever.
    POSSIBLE = "POSSIBLE"
    #: Below :data:`POSSIBLE_SIMILARITY`, or nobody to compare against.
    NONE = "NONE"

    def __bool__(self) -> bool:
        raise TypeError(
            "a Verdict has three values and no truth value; write "
            "`verdict is Verdict.MATCH` — a POSSIBLE verdict is not a match"
        )


def verdict_for(score: float) -> Verdict:
    """Which of the three things this similarity is allowed to mean."""
    if score >= MATCH_SIMILARITY:
        return Verdict.MATCH
    if score >= POSSIBLE_SIMILARITY:
        return Verdict.POSSIBLE
    return Verdict.NONE


@dataclass(frozen=True, slots=True)
class FaceTemplate:
    """128 floats describing a face, and where they came from.

    **Not an image.** This is the output of an embedding network, not a picture:
    it cannot be rendered, and it cannot be inverted into the face that produced
    it — many faces map to nearby points and the network is never run backwards.
    That is the entire argument for storing one where storing the crop would
    need a separate justification, its own retention and its own audit row, and
    it is why the crop is discarded by default.

    The vector is L2-normalised on construction, so a cosine similarity is a
    plain dot product and a template always scores exactly 1.0 against itself.
    Normalising here rather than at each comparison closes a real hole: two
    templates written at different output scales would otherwise produce a
    similarity that depended on the scale rather than on the faces.

    ``quality`` is the *detector's* confidence that this was a face at all. It
    says nothing about whose face it is; that is what a :class:`Match` score is
    for, and conflating the two is how a crisp photograph of a stranger becomes
    a confident identification.
    """

    #: L2-normalised. A tuple, not an array: this is a value — hashed, compared
    #: and stored — and a mutable buffer shared with the frame it came from is a
    #: template that changes after it was recorded.
    vector: tuple[float, ...]
    #: The detector's confidence in the face, 0..1. Provenance, not identity.
    quality: float
    #: Which embedder produced it — name and digest. A template is only
    #: comparable with others from the same model, and a register that silently
    #: mixes two models matches nobody correctly for reasons nobody can see.
    model: str
    #: Where it came from: camera and track, for the audit trail behind a name.
    source: str
    #: When it was computed. Retention is measured from here, so a template
    #: with no time cannot be swept and would outlive the policy.
    created_unix_millis: int

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.shape != (TEMPLATE_DIMENSIONS,):
            raise FaceError(
                f"a face template is {TEMPLATE_DIMENSIONS} floats; "
                f"{vector.size} were given, which cannot be compared against "
                "the register"
            )
        if not np.all(np.isfinite(vector)):
            raise FaceError(
                "a face template contains a non-finite value; a NaN compares "
                "false against everything and would read as a stranger"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= _MINIMUM_NORM:
            raise FaceError(
                "a face template with no magnitude has no direction and no "
                "cosine with anything; it is not a weak face, it is not a face"
            )
        if not 0.0 <= self.quality <= 1.0:
            raise FaceError(f"quality is a 0..1 confidence; {self.quality} is not")
        object.__setattr__(self, "vector", tuple(float(v) for v in vector / norm))

    def as_array(self) -> np.ndarray:
        """A fresh array, never a view: a caller must not be able to edit this."""
        return np.asarray(self.vector, dtype=np.float64)


def similarity(left: FaceTemplate, right: FaceTemplate) -> float:
    """Cosine similarity of two templates, in [-1, 1].

    A dot product, because both vectors are unit length by construction. The
    clamp is not cosmetic: floating-point error puts a template's similarity
    with itself a few ulps above 1.0, and a score printed as 1.0000000000000002
    beside a name reads as a bug in the thing an operator is being asked to
    trust.

    This is the two-template helper, for a caller holding exactly two. Nothing
    that compares a track against a register is built out of it: see
    :class:`_Register` for why a per-pair loop over this function is the wrong
    shape by two orders of magnitude.
    """
    value = float(np.dot(left.as_array(), right.as_array()))
    return float(np.clip(value, -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class EnrolledPerson:
    """Somebody an operator named, and the templates they were named with.

    Nothing in this module constructs one of these from a sighting, and that
    absence is the design. Enrolment is an explicit act on a track — "name this
    person" — performed by code that owns the register and writes the audit row
    saying who did it, when, and on what lawful basis. A passer-by cannot walk
    into a database that has no door.
    """

    person_id: str
    name: str
    templates: tuple[FaceTemplate, ...]


@dataclass(frozen=True, slots=True)
class Match:
    """One comparison's outcome: a verdict, its score, and who it was against.

    ``score`` is ``None`` only when nothing was compared — an empty register.
    That is a different fact from a score of zero, which means somebody *was*
    compared and did not resemble this face; reporting the second when the first
    happened would let an empty register look like a considered rejection.

    ``bool(match)`` raises. See :class:`Verdict`: the one line this type exists
    to prevent is ``if match:`` treating POSSIBLE as MATCH.
    """

    verdict: Verdict
    score: float | None
    person_id: str | None
    name: str | None
    #: How many enrolled templates were actually compared. Evidence that the
    #: verdict was reached by looking rather than by an empty loop.
    compared: int

    def __bool__(self) -> bool:
        raise TypeError(
            "a Match has three outcomes and no truth value; ask for "
            "`match.is_match` — a POSSIBLE match must never open a door"
        )

    @property
    def is_match(self) -> bool:
        """True only for :data:`Verdict.MATCH`. POSSIBLE is not a match."""
        return self.verdict is Verdict.MATCH

    @property
    def is_possible(self) -> bool:
        """True only in the band between the two thresholds."""
        return self.verdict is Verdict.POSSIBLE

    def describe(self) -> str:
        """The name and the score, or neither. There is no third spelling.

        Every surface that shows a name — the track overlay, the table, the map
        — renders this, so an operator cannot meet the certain form in one panel
        and the hedged form in another.
        """
        if self.score is None:
            return "no comparison: nobody is enrolled"
        if self.verdict is Verdict.NONE:
            return f"no match (best {self.score:.2f} of {self.compared} compared)"
        word = "match" if self.verdict is Verdict.MATCH else "possible match"
        return f"{self.name} — {word}, {self.score:.2f}"


#: What a comparison against an empty register says about itself.
NO_MATCH = Match(
    verdict=Verdict.NONE, score=None, person_id=None, name=None, compared=0
)


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    """What a whole track's faces say, which is not what any one frame says.

    ``score`` is the *median* of the per-frame best similarities, not the mean
    and emphatically not the maximum. A track of twenty frames will contain one
    frame that resembles somebody it should not — a blur, a profile, a hand —
    and a maximum promotes exactly that frame to the answer. A median has to be
    convinced by most of the track.

    ``frames`` is on the face of the type because a verdict from two frames and
    a verdict from forty are different claims. Below :data:`FRAMES_FOR_A_MATCH`
    this refuses to say MATCH at all.
    """

    verdict: Verdict
    score: float | None
    person_id: str | None
    name: str | None
    #: Templates that went into the aggregate.
    frames: int
    #: The best single frame, kept beside the median so a reader can see the
    #: spread rather than a number with its own disagreement averaged away.
    best_frame_score: float | None

    def __bool__(self) -> bool:
        raise TypeError(
            "a TrackIdentity has three outcomes and no truth value; ask for "
            "`identity.is_match` — a POSSIBLE identity must fire no rule"
        )

    @property
    def is_match(self) -> bool:
        """True only for :data:`Verdict.MATCH`."""
        return self.verdict is Verdict.MATCH

    @property
    def is_possible(self) -> bool:
        """True only in the band between the two thresholds."""
        return self.verdict is Verdict.POSSIBLE

    def describe(self) -> str:
        """The name and the score and how many frames agreed, or none of them."""
        if self.score is None:
            return "no comparison: nobody is enrolled, or the track had no face"
        if self.verdict is Verdict.NONE:
            return f"no match (best {self.score:.2f} over {self.frames} frame(s))"
        word = "match" if self.verdict is Verdict.MATCH else "possible match"
        return f"{self.name} — {word}, {self.score:.2f} over {self.frames} frame(s)"


#: What a track with no faces, or a site with nobody enrolled, says.
NO_IDENTITY = TrackIdentity(
    verdict=Verdict.NONE,
    score=None,
    person_id=None,
    name=None,
    frames=0,
    best_frame_score=None,
)


@dataclass(frozen=True, slots=True, eq=False)
class _Register:
    """The whole register as one matrix, because a comparison is not a loop.

    Every comparison this module makes is the same arithmetic — a unit vector
    against every enrolled unit vector — and that is one matrix multiply, not a
    Python loop calling :func:`similarity` per pair. The difference is not
    academic: a thirty-frame track against fifty people of five templates each
    is 7,500 dot products, which measured 92 ms as a loop and a fraction of a
    millisecond as one ``A @ B.T`` on this machine. A camera worker identifying
    tracks on a cadence pays that per track, and it grows with the register the
    operator is being encouraged to build — so the loop punished a site for
    enrolling people.

    ``starts`` is where each person's block of templates begins, which is what
    lets ``np.maximum.reduceat`` collapse the full score matrix to a best score
    per person per frame in one pass. People with no templates are dropped when
    this is built rather than skipped in a branch further down: an empty person
    cannot contribute a score, and leaving them out is what keeps ``starts`` an
    honest index into ``vectors``.

    Deliberately built per call and never cached. A module-level cache of
    registers would be a second copy of the site's biometrics, outliving the
    ``forget_person`` that was supposed to delete them and held somewhere no
    audit row describes. Restacking costs microseconds; a hidden register costs
    the argument for the whole feature.
    """

    people: tuple[EnrolledPerson, ...]
    #: ``(M, TEMPLATE_DIMENSIONS)``, every enrolled template, people in order.
    vectors: np.ndarray
    #: ``(P,)``, the row where each person's templates start.
    starts: np.ndarray

    @property
    def compared(self) -> int:
        """How many templates a comparison against this actually looks at."""
        return int(self.vectors.shape[0])

    def scores_for(self, templates: Sequence[FaceTemplate]) -> np.ndarray:
        """``(frames, people)`` — each frame's best score against each person.

        The clamp is :func:`similarity`'s, for :func:`similarity`'s reason: a
        template against itself lands a few ulps above 1.0, and a score above
        one printed beside a name reads as a broken instrument.
        """
        faces = np.asarray(
            [template.vector for template in templates], dtype=np.float64
        )
        every = np.clip(faces @ self.vectors.T, -1.0, 1.0)
        return np.maximum.reduceat(every, self.starts, axis=1)


def _register_of(enrolled: Sequence[EnrolledPerson]) -> "_Register | None":
    """Stack a register once, or ``None`` when there is nobody to compare against.

    ``None`` rather than an empty matrix, because "nobody is enrolled" is a
    different answer from "everybody scored badly" and the callers must not be
    able to blur the two — that distinction is what :data:`NO_MATCH` carries.
    """
    people = tuple(person for person in enrolled if person.templates)
    if not people:
        return None
    counts = [len(person.templates) for person in people]
    return _Register(
        people=people,
        vectors=np.asarray(
            [template.vector for person in people for template in person.templates],
            dtype=np.float64,
        ),
        starts=np.cumsum([0] + counts[:-1], dtype=np.intp),
    )


def match(template: FaceTemplate, enrolled: Sequence[EnrolledPerson]) -> Match:
    """Compare one face against the register and say which of three things it is.

    The best-scoring person wins, and only the best: a face that scores 0.7
    against one person and 0.68 against another is reported as the first, and
    the second is not a second answer to be shown beside it. Ties resolve to
    whoever came first in ``enrolled``, which is arbitrary, and is why nothing
    downstream may read anything into the order.

    An empty register returns :data:`NO_MATCH`, whose score is ``None``. It has
    not decided this is a stranger; it has not looked at anybody.
    """
    register = _register_of(enrolled)
    if register is None:
        return NO_MATCH

    # One row — this face against every enrolled template, collapsed to the best
    # per person. ``argmax`` takes the first of any tie, which is the same
    # arbitrary order the loop it replaced resolved ties in.
    per_person = register.scores_for((template,))[0]
    winner = int(np.argmax(per_person))
    best_score = float(per_person[winner])
    best_person = register.people[winner]
    compared = register.compared

    verdict = verdict_for(best_score)
    if verdict is Verdict.NONE:
        # Nothing is claimed, so nobody is named — but the score stays, because
        # "we looked at four people and the closest was 0.11" is the evidence
        # that the silence was reasoned rather than a loop that never ran.
        return Match(
            verdict=verdict,
            score=best_score,
            person_id=None,
            name=None,
            compared=compared,
        )
    return Match(
        verdict=verdict,
        score=best_score,
        person_id=best_person.person_id,
        name=best_person.name,
        compared=compared,
    )


def identify_track(
    templates: Sequence[FaceTemplate], enrolled: Sequence[EnrolledPerson]
) -> TrackIdentity:
    """Decide a track's identity from all of its faces at once.

    Each person is scored by the median of their best similarity across the
    track's frames, and the highest median wins. This is deliberately harder to
    satisfy than :func:`match`: a name drawn on a track is a claim about a
    person moving through a site, and one lucky frame must not be able to make
    it.

    A track with fewer than :data:`FRAMES_FOR_A_MATCH` faces is capped at
    POSSIBLE however well it scores. Two frames of a moving person are two
    samples of one instant's lighting and pose, not corroboration.
    """
    if not templates or not enrolled:
        return NO_IDENTITY
    register = _register_of(enrolled)
    if register is None:
        return NO_IDENTITY

    # ``(frames, people)``: one matmul, then a median down the frame axis. The
    # median is taken over a column rather than over a Python list built per
    # person, so a longer track and a larger register cost arithmetic and not
    # interpreter time.
    per_person = register.scores_for(templates)
    medians = np.median(per_person, axis=0)
    winner = int(np.argmax(medians))
    best_median = float(medians[winner])
    best_frame = float(per_person[:, winner].max())
    best_person = register.people[winner]

    verdict = verdict_for(best_median)
    if verdict is Verdict.MATCH and len(templates) < FRAMES_FOR_A_MATCH:
        # Well-scoring, but on too little evidence. Reported as the hedge rather
        # than discarded: the operator should see it, and no rule should fire.
        verdict = Verdict.POSSIBLE
    if verdict is Verdict.NONE:
        return TrackIdentity(
            verdict=verdict,
            score=best_median,
            person_id=None,
            name=None,
            frames=len(templates),
            best_frame_score=best_frame,
        )
    return TrackIdentity(
        verdict=verdict,
        score=best_median,
        person_id=best_person.person_id,
        name=best_person.name,
        frames=len(templates),
        best_frame_score=best_frame,
    )


def expired_templates(
    templates: Sequence[FaceTemplate],
    *,
    now_unix_millis: int,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    pinned: bool = False,
) -> tuple[FaceTemplate, ...]:
    """Which stored templates the retention sweep should delete.

    Pure, and separate from whatever deletes them, so the policy can be tested
    without a database and the deletion can be audited where it happens. A
    biometric register with no delete is not lawful in most jurisdictions and
    not defensible in any of them, and the failure mode is silent: nothing
    breaks when a sweep quietly keeps everything.

    ``pinned`` is the operator's explicit decision that this person outlives the
    default — a permanent member of staff rather than a visiting contractor —
    and is the only thing that exempts a template.
    """
    if pinned:
        return ()
    cutoff = now_unix_millis - retention_days * 86_400_000
    return tuple(
        template
        for template in templates
        if template.created_unix_millis < cutoff
    )


@dataclass(frozen=True, slots=True)
class FaceBox:
    """One face the detector found, in pixels of the crop it was given.

    ``raw`` carries the detector's own output row — box, five landmarks and
    score — because the embedder aligns the face on those landmarks before it
    embeds. Passing the row along rather than re-deriving landmarks keeps one
    description of where the eyes are; two would drift, and an embedding from a
    differently-aligned crop is a different template for the same face.
    """

    x: float
    y: float
    w: float
    h: float
    score: float
    raw: tuple[float, ...]


class FaceBackend(Protocol):
    """The seam the models sit behind, so the logic above it can be tested.

    Everything worth getting right in this module — thresholds, aggregation,
    the three-way verdict, normalisation, the switch — is reachable through a
    stand-in that returns fixed vectors. That is deliberate: there is no model
    file on a developer machine and none may be downloaded, so logic reachable
    only with real weights would be logic that is never exercised.
    """

    @property
    def name(self) -> str:
        """What produced these templates, for :attr:`FaceTemplate.model`."""

    def detect(self, image: np.ndarray) -> Sequence[FaceBox]:
        """Faces inside ``image``, which is already a crop of one person."""

    def embed(self, image: np.ndarray, face: FaceBox) -> Sequence[float]:
        """The raw, un-normalised vector for one detected face."""


class _OpenCVBackend:
    """YuNet and SFace, loaded from files the operator put on this machine.

    Never constructed by the tests, and it must not need to be: it does nothing
    but load two files and forward two calls. Everything that could be wrong in
    a way worth testing lives above this line.
    """

    __slots__ = ("_detector", "_embedder", "_name", "_size")

    def __init__(
        self,
        detector_model: str | Path,
        embedder_model: str | Path,
        *,
        score_threshold: float = MINIMUM_FACE_SCORE,
        nms_threshold: float = 0.3,
        top_k: int = 50,
    ):
        detector_path = _model_file(detector_model, "face detector (YuNet)")
        embedder_path = _model_file(embedder_model, "face embedder (SFace)")

        import cv2

        self._size = (320, 320)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", self._size, score_threshold, nms_threshold, top_k
        )
        self._embedder = cv2.FaceRecognizerSF.create(str(embedder_path), "")
        # The digests are in the name because a register is only comparable
        # within one model build, and "SFace" alone does not distinguish two.
        self._name = (
            f"YuNet {detector_path.name}@{_sha256(detector_path)[:12]} + "
            f"SFace {embedder_path.name}@{_sha256(embedder_path)[:12]}"
        )

    @property
    def name(self) -> str:
        return self._name

    def detect(self, image: np.ndarray) -> Sequence[FaceBox]:
        height, width = image.shape[:2]
        if (width, height) != self._size:
            # YuNet must be told the exact input size before every differently
            # shaped image, and silently finds nothing when it is not.
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        _, rows = self._detector.detect(image)
        if rows is None:
            return ()
        return tuple(
            FaceBox(
                x=float(row[0]),
                y=float(row[1]),
                w=float(row[2]),
                h=float(row[3]),
                score=float(row[-1]),
                raw=tuple(float(value) for value in row),
            )
            for row in rows
        )

    def embed(self, image: np.ndarray, face: FaceBox) -> Sequence[float]:
        aligned = self._embedder.alignCrop(
            image, np.asarray(face.raw, dtype=np.float32)
        )
        return np.asarray(self._embedder.feature(aligned), dtype=np.float64).ravel()


def _model_file(path: str | Path, what: str) -> Path:
    """Resolve an operator-supplied model file, or say exactly what is missing.

    A missing model is an honest error naming the file, never a cue to fetch
    one. There is no network at runtime and no code here that could acquire
    anything; an error hinting otherwise would be describing a different system.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FaceError(
            f"No {what} model at {resolved}. Face models are supplied by the "
            "operator and placed in the models directory; nothing is ever "
            "downloaded. The People feature stays off until one is present."
        )
    return resolved


def _off_returns(disabled_result: Callable[[], object]):
    """Wrap a method so it does nothing at all while the feature is off.

    The switch is enforced here rather than at each call site because there are
    many call sites and one of them will forget. ``disabled_result`` is a
    factory rather than a value, so the disabled answer is written next to the
    method it belongs to and no two calls share one object.

    The marker attribute is not decoration: ``test_faces.py`` walks
    :class:`FaceEngine` and fails if any public method lacks it, so a method
    added next year cannot quietly become a way to run a face model on a site
    that never switched the feature on.
    """

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self.enabled:
                return disabled_result()
            return method(self, *args, **kwargs)

        wrapper._disabled_result = disabled_result
        return wrapper

    return decorate


@dataclass(frozen=True, slots=True)
class FaceEngineInfo:
    """What the face path is configured to be, for the record and the operator.

    Readable while the feature is off, and says so — an operator needs to see
    which models *would* be used, and that is a statement about configuration
    which claims nothing about anybody's face. It is the one public member of
    :class:`FaceEngine` that is not gated, for that reason.
    """

    enabled: bool
    backend: str | None
    match_similarity: float
    possible_similarity: float

    def describe(self) -> str:
        if not self.enabled:
            return "faces: off — no face is detected, embedded or stored"
        return (
            f"faces: on via {self.backend}; match at "
            f"{self.match_similarity:.2f}, possible from "
            f"{self.possible_similarity:.2f}"
        )


class FaceEngine:
    """Detection, embedding and matching — and the switch that gates all three.

    ``enabled`` is False unless the caller passes ``enabled=True``, and it
    cannot be changed afterwards. There is deliberately no ``enable()``: a
    mutable switch means a window in which one reference has the feature on and
    another has it off, and the one that is wrong is the reference still holding
    a frame. Turning the feature on for a site builds a new engine, which is
    also the moment the audit row is written.

    The models are loaded only when the feature is on. An engine constructed
    while off never opens a file, so a site with the feature off needs no face
    models to exist at all — which is the state every site starts in.

    ``backend`` exists for the tests and for any future embedder: pass one and
    the model paths are not consulted. Nothing else about the engine changes, so
    the thresholds, the gating and the aggregation are the same code in a test
    as in the field.
    """

    __slots__ = ("_enabled", "_backend", "_minimum_score", "_minimum_pixels")

    def __init__(
        self,
        *,
        enabled: bool = False,
        detector_model: str | Path | None = None,
        embedder_model: str | Path | None = None,
        backend: FaceBackend | None = None,
        minimum_face_score: float = MINIMUM_FACE_SCORE,
        minimum_face_pixels: int = MINIMUM_FACE_PIXELS,
    ):
        self._enabled = bool(enabled)
        self._minimum_score = minimum_face_score
        self._minimum_pixels = minimum_face_pixels
        self._backend: FaceBackend | None = None

        if not self._enabled:
            # Off means off: no file is opened and no model is loaded, so a site
            # that never switched this on needs no face models on disk.
            if backend is not None or detector_model or embedder_model:
                _log.info("faces: models are configured but the feature is off")
            return

        if backend is not None:
            self._backend = backend
        else:
            if detector_model is None or embedder_model is None:
                raise FaceError(
                    "the People feature is on but no face models were given; a "
                    "detector (YuNet) and an embedder (SFace) are both "
                    "required, and both are supplied by the operator"
                )
            self._backend = _OpenCVBackend(
                detector_model, embedder_model, score_threshold=minimum_face_score
            )
        _log.info("faces: on, using %s", self._backend.name)

    @property
    def enabled(self) -> bool:
        """The switch. False unless somebody turned it on, and then immutable."""
        return self._enabled

    @property
    def info(self) -> FaceEngineInfo:
        """Configuration, safe to read while off. See :class:`FaceEngineInfo`."""
        return FaceEngineInfo(
            enabled=self._enabled,
            backend=self._backend.name if self._backend is not None else None,
            match_similarity=MATCH_SIMILARITY,
            possible_similarity=POSSIBLE_SIMILARITY,
        )

    @_off_returns(tuple)
    def templates_in(
        self,
        frame: np.ndarray,
        person_box: BoundingBox,
        *,
        source: str = "",
        timestamp_unix_millis: int = 0,
    ) -> tuple[FaceTemplate, ...]:
        """Templates for the faces inside one person's box. Never a whole frame.

        ``person_box`` is required and positional, because the alternative — a
        default meaning "all of it" — is a face detector running over every
        frame of every camera on a site that asked for it to run on people. A
        scene with nobody in it must cost nothing, and it does, because there is
        no box to pass.

        Returns an empty tuple, not ``None``, when the feature is off, when the
        box is too small to hold a face, and when the detector finds none. Those
        are the same thing to a caller: there is nobody here to name.
        """
        crop = self._crop(frame, person_box)
        if crop is None:
            return ()

        backend = self._backend
        if backend is None:  # pragma: no cover - the switch guarantees otherwise
            return ()

        templates: list[FaceTemplate] = []
        for face in backend.detect(crop):
            if face.score < self._minimum_score:
                continue
            if min(face.w, face.h) < self._minimum_pixels:
                continue
            vector = backend.embed(crop, face)
            try:
                templates.append(
                    FaceTemplate(
                        vector=tuple(float(value) for value in vector),
                        quality=float(np.clip(face.score, 0.0, 1.0)),
                        model=backend.name,
                        source=source,
                        created_unix_millis=timestamp_unix_millis,
                    )
                )
            except FaceError as error:
                # A model that returned an unusable vector is a broken model,
                # not a stranger. Logged and dropped rather than raised: one bad
                # frame must not take a camera down.
                _log.warning("faces: unusable template discarded: %s", error)
        return tuple(templates)

    @_off_returns(lambda: NO_MATCH)
    def match(
        self, template: FaceTemplate, enrolled: Sequence[EnrolledPerson]
    ) -> Match:
        """Compare one template against the register, only while the feature is on.

        Gated as well as :meth:`templates_in`, although a template can only
        exist because an enabled engine made one: an engine switched off for a
        site must not answer questions about faces using templates that outlived
        the switch being turned off.
        """
        return match(template, enrolled)

    @_off_returns(lambda: NO_IDENTITY)
    def identify_track(
        self,
        templates: Sequence[FaceTemplate],
        enrolled: Sequence[EnrolledPerson],
    ) -> TrackIdentity:
        """A whole track's identity, aggregated. See :func:`identify_track`."""
        return identify_track(templates, enrolled)

    @_off_returns(lambda: NO_IDENTITY)
    def identify(
        self,
        frames: Sequence[tuple[np.ndarray, BoundingBox]],
        enrolled: Sequence[EnrolledPerson],
        *,
        source: str = "",
    ) -> TrackIdentity:
        """Faces from several frames of one track, aggregated into one verdict.

        Takes frames *with their boxes*, because a track's box moves. Pairing
        them at the call site is what keeps a crop taken from a stale box — a
        crop of whatever the person walked away from — out of the register.
        """
        templates: list[FaceTemplate] = []
        for frame, person_box in frames:
            templates.extend(self.templates_in(frame, person_box, source=source))
        return identify_track(templates, enrolled)

    def _crop(self, frame: np.ndarray, person_box: BoundingBox) -> np.ndarray | None:
        """The pixels of one person, or ``None`` when there are too few to matter.

        Rejects a box that is not in normalised coordinates rather than clamping
        it. A pixel box passed to a normalised API clamps to the whole frame,
        which turns "look inside this person" into "run a face detector over
        everything" — the exact behaviour this module promises never to do,
        arrived at silently.

        **The size is what decides that, not the corner.** Checking only ``x``
        and ``y`` left the whole failure open behind a legal-looking origin:
        ``BoundingBox(0.0, 0.0, 200.0, 150.0)`` starts at the top-left of any
        image, and clamping its width handed the detector every pixel of every
        frame. So ``w`` and ``h`` carry the same bound as ``x`` and ``y``, and
        they are the half that closes the hole — a box whose *extent* is a
        fraction of the frame can only ever crop a fraction of it.

        **A box running off an edge is a different thing, and is allowed.** The
        tracker's own ``clamp_box`` pins a coasting box to ``x ∈ [-w, 1]`` and
        deliberately leaves ``w`` alone, so a person walking out of the left or
        right of frame is *supposed* to arrive here with ``x < 0`` or
        ``x + w > 1``. Trimming that to the visible part is a crop of that
        person and of nobody else; refusing it would raise on the most ordinary
        event a camera sees. Rejecting the size while clamping the position is
        the distinction: one describes a person, the other stopped describing
        one at all.
        """
        if person_box.w <= 0 or person_box.h <= 0:
            raise FaceError(
                f"person box has no area ({person_box.w} x {person_box.h}); "
                "there is nothing inside it to look at"
            )
        for value, label in ((person_box.w, "w"), (person_box.h, "h")):
            if value > 1.0 + _BOX_TOLERANCE:
                raise FaceError(
                    f"person box {label}={value} is not in normalised 0..1 "
                    "coordinates; a pixel box here would silently become the "
                    "whole frame"
                )
        for value, extent, label in (
            (person_box.x, person_box.w, "x"),
            (person_box.y, person_box.h, "y"),
        ):
            # A box may begin off the top or left edge by at most its own size,
            # which is exactly the range the tracker clamps a coasting box into.
            if not -extent - _BOX_TOLERANCE <= value <= 1.0 + _BOX_TOLERANCE:
                raise FaceError(
                    f"person box {label}={value} is not in normalised 0..1 "
                    "coordinates; a pixel box here would silently become the "
                    "whole frame"
                )

        height, width = frame.shape[:2]
        left = max(0, int(round(person_box.x * width)))
        top = max(0, int(round(person_box.y * height)))
        right = min(width, int(round((person_box.x + person_box.w) * width)))
        bottom = min(height, int(round((person_box.y + person_box.h) * height)))

        if right - left < self._minimum_pixels or bottom - top < self._minimum_pixels:
            # A person a few pixels across at the far edge of the frame has no
            # face in any useful sense, and upscaling one produces a template
            # that describes the interpolation filter.
            return None
        return frame[top:bottom, left:right]
