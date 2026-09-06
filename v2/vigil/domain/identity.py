"""A register of enrolled subjects, and what a match is and is not.

# The claim this module refuses to make

A face embedding comparison produces a **distance**. It does not produce an
identity, and the difference is the entire reason this file is written the way
it is. "Camera 3 saw Mr Halabi" is a sentence a court can act on; what actually
happened is "a 128-number vector from a frame was 0.31 from a vector somebody
enrolled six months ago under the label `Mr Halabi`, and the next nearest
enrolled vector was 0.44 away". The second is true and the first may not be.

So `Match` carries the distance, the threshold it beat, and the margin over
the runner-up, and `describe()` says "resembles" rather than "is". The wording
is in the code rather than in a style guide, because a report is written by
whoever is on shift and the constraint has to survive them.

# Threshold *and* margin

A threshold alone answers "is this close enough". It does not answer "close
enough compared with what", and in a register of forty people the nearest two
are often both within any threshold generous enough to be useful. Then the
solver picks one, and picking is guessing.

The margin is the second gate: the best candidate must beat the runner-up by
enough that the ordering is not noise. A register of one has no runner-up and
so no margin to clear -- stated here because it is the case where this check
silently does nothing, and it is also the most common case in a small site.

# Every constant in here is a stated default with nothing behind it

No face model ships with this product and none has been run. `MAX_DISTANCE`
and `MIN_MARGIN` are the values the literature uses for L2-normalised
ArcFace-family embeddings, and they are **assumptions about a model this
repository has never seen**. `Register.describe()` says so. When a model is
supplied, `tools/calibrate.py`'s method applies here unchanged: enrol a
person twice from different footage, and measure what the same face scores
against what different faces score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

#: Furthest an embedding may be from an enrolled one and still be reported.
#:
#: **Assumed, not measured.** For L2-normalised embeddings of the ArcFace
#: family a cosine distance near 0.35 is the usual operating point. Nothing in
#: this repository has verified it, because no face model ships here.
MAX_DISTANCE = 0.35

#: How much better the nearest enrolled subject must be than the next.
#:
#: Also assumed. Without it, a register in which two people look alike returns
#: whichever of them happened to be marginally nearer, with a confident number
#: attached and no indication that the second was almost as close.
MIN_MARGIN = 0.08

#: Fewest enrolled subjects before a "no match" means anything.
#:
#: One. A register with nobody in it cannot fail to match; it simply has
#: nothing to compare against, and reporting "not a known subject" from an
#: empty register would be a claim made out of nothing.
MIN_ENROLLED = 1


class IdentityError(ValueError):
    """An enrolment or a comparison that cannot be made."""


@dataclass(frozen=True, slots=True)
class Subject:
    """Somebody an operator enrolled, and the vector they enrolled.

    `label` is what the operator typed. It is not a legal identity, it has not
    been verified against anything, and this class never treats it as more
    than a name on a row.
    """

    id: str
    label: str
    embedding: np.ndarray
    #: Digest of the model that produced the embedding. Embeddings from two
    #: different models are not comparable at all -- the distance between them
    #: is a number with no meaning -- so a match across a digest boundary is
    #: refused rather than scored.
    model_sha256: str
    note: str = ""

    def distance(self, embedding: np.ndarray) -> float:
        """Cosine distance in `[0, 2]`, 0 for identical."""
        return float(1.0 - float(self.embedding @ embedding))


@dataclass(frozen=True, slots=True)
class Match:
    """What the register found, with every number needed to disagree with it."""

    subject: Subject
    distance: float
    #: How much further away the next nearest enrolled subject was. `inf` when
    #: there was no second subject to compare with -- a register of one.
    margin: float
    threshold: float
    required_margin: float
    #: The runner-up's label, when there was one. Named so a reviewer can see
    #: who else it nearly was rather than being told only who it was.
    runner_up: str | None = None

    @property
    def confident(self) -> bool:
        return self.distance <= self.threshold and self.margin >= self.required_margin

    def describe(self) -> str:
        """Deliberately worded as a resemblance.

        "Resembles", never "is". A face embedding comparison produces a
        distance; the identity is an inference somebody else has to make, and
        wording it as a fact here would put that inference in the evidence
        package under this system's name.
        """
        near = (f", next nearest {self.runner_up} at {self.distance + self.margin:.3f}"
                if self.runner_up is not None else ", the only subject enrolled")
        return (f"resembles the subject enrolled as {self.subject.label!r}: "
                f"{self.distance:.3f} away, inside {self.threshold:.2f}{near}. "
                f"A similarity, not an identification")


@dataclass
class Register:
    """The enrolled subjects, and the only place a comparison happens."""

    subjects: list[Subject] = field(default_factory=list)
    max_distance: float = MAX_DISTANCE
    min_margin: float = MIN_MARGIN
    #: Whether the thresholds above have been measured on the model in use.
    #: False everywhere until somebody runs the measurement, and `describe`
    #: says so rather than presenting an assumption as a calibration.
    calibrated: bool = False

    def __len__(self) -> int:
        return len(self.subjects)

    def add(self, subject: Subject) -> None:
        if not np.isfinite(subject.embedding).all():
            raise IdentityError(f"{subject.label}'s embedding holds a NaN and cannot be compared")
        norm = float(np.linalg.norm(subject.embedding))
        if not 0.99 <= norm <= 1.01:
            raise IdentityError(
                f"{subject.label}'s embedding has length {norm:.3f}. These are compared by cosine "
                f"distance, which is only a distance for unit vectors; normalise it first"
            )
        self.subjects.append(subject)

    def match(self, embedding: np.ndarray, model_sha256: str) -> Match | None:
        """The nearest enrolled subject, or `None` when nobody is near enough.

        `None` covers three different situations and the caller must not read
        it as "a stranger": nobody is enrolled, nobody is within the
        threshold, or two people were equally close and the margin refused to
        choose between them. Only the second is "not a known subject", and
        `explain` says which happened.
        """
        best, _ = self._ranked(embedding, model_sha256)
        return best if best is not None and best.confident else None

    def explain(self, embedding: np.ndarray, model_sha256: str) -> str:
        """Why `match` answered as it did, in a sentence for the audit trail."""
        best, reason = self._ranked(embedding, model_sha256)
        if best is None:
            return reason
        if best.confident:
            return best.describe()
        if best.distance > best.threshold:
            return (f"no enrolled subject is near enough: the closest, {best.subject.label!r}, is "
                    f"{best.distance:.3f} away and the threshold is {best.threshold:.2f}")
        return (f"refused to choose: {best.subject.label!r} at {best.distance:.3f} and "
                f"{best.runner_up!r} at {best.distance + best.margin:.3f} are within "
                f"{best.required_margin:.2f} of each other, so which one it is would be a guess")

    def _ranked(self, embedding: np.ndarray, model_sha256: str) -> tuple[Match | None, str]:
        usable = [s for s in self.subjects if s.model_sha256 == model_sha256]
        if not self.subjects:
            return None, "nobody is enrolled, so there was nothing to compare against"
        if not usable:
            return None, (
                f"every enrolled subject was embedded by a different model than the one running "
                f"({model_sha256[:12]}). Distances between two models' embeddings are not "
                f"distances; re-enrol against this model"
            )
        if not np.isfinite(embedding).all() or not np.any(embedding):
            return None, "the embedding from this frame is empty or holds a NaN"
        distances = sorted(((s.distance(embedding), s) for s in usable), key=lambda p: p[0])
        nearest, subject = distances[0]
        if len(distances) > 1:
            margin = distances[1][0] - nearest
            runner_up = distances[1][1].label
        else:
            margin, runner_up = math.inf, None
        return Match(subject, nearest, margin, self.max_distance, self.min_margin, runner_up), ""

    def describe(self) -> str:
        how = ("measured on the model in use" if self.calibrated
               else "an ASSUMED default — no face model ships with this product and these numbers "
                    "have never been checked against one")
        return (f"{len(self.subjects)} subject(s) enrolled; a match must be within "
                f"{self.max_distance:.2f} and beat the runner-up by {self.min_margin:.2f} "
                f"({how})")


def normalise(embedding: Iterable[float]) -> np.ndarray:
    """A vector as this module compares them: float64, unit length.

    Refuses a zero vector rather than returning one. A zero embedding has a
    cosine distance of exactly 1.0 from everything, which sits inside no
    sensible threshold but is not obviously wrong when read off a screen.
    """
    vector = np.asarray(list(embedding), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise IdentityError("an embedding of no length is not a description of a face")
    return vector / norm


def plate_agrees(a: str, b: str) -> float:
    """How much two plate readings agree, in `[0, 1]`.

    Character by character over the common length, penalised for a difference
    in length. Not an edit distance: plate OCR substitutes far more often than
    it inserts or deletes -- 8 for B, 0 for O, 1 for I -- and an edit distance
    scores a substitution the same as a dropped character, which for this
    problem is the wrong shape.
    """
    a, b = (a or "").upper().strip(), (b or "").upper().strip()
    if not a or not b:
        return 0.0
    common = min(len(a), len(b))
    same = sum(1 for i in range(common) if a[i] == b[i])
    return same / max(len(a), len(b))


#: Character pairs an OCR confuses, so a near-miss can be reported as one
#: rather than as a different plate. Stated, not learnt: these are the shapes,
#: and a site that finds others should add them rather than have a model guess.
CONFUSABLE = (("0", "O"), ("1", "I"), ("1", "L"), ("8", "B"), ("5", "S"), ("2", "Z"), ("6", "G"))


def could_be_the_same_plate(a: str, b: str) -> bool:
    """Whether two readings differ only where an OCR is known to confuse.

    Used to say "this may be the same vehicle" without asserting it. Two
    readings that differ by a genuine character are two vehicles; two that
    differ only by 8/B are one vehicle read twice, probably.
    """
    a, b = (a or "").upper().strip(), (b or "").upper().strip()
    if not a or len(a) != len(b):
        return False
    pairs = {frozenset(p) for p in CONFUSABLE}
    for x, y in zip(a, b):
        if x == y:
            continue
        if frozenset((x, y)) not in pairs:
            return False
    return True
