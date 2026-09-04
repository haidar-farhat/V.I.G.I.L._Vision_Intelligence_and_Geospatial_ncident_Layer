"""Number plates: reading one off a vehicle, and never inventing the rest.

... -> VEHICLE TRACK -> PLATE READ -> READING -> ...

A plate is the one identifier a site can lift without touching anybody's
biometrics, which makes it the cheap half of the identity story and the half
most likely to be believed without qualification. That is the failure this
module exists to prevent. A single frame of a moving vehicle, at night, through
a windscreen-height glare, produces a string of seven characters that looks
exactly like a plate and is frequently not one. Presented as a plate, it becomes
a watchlist hit against a vehicle that was never there.

Three rules, each of which is a way a plate reader manufactures a wrong answer:

**A plate is read from a track, not from a frame.** One frame is a guess. The
same character agreeing across several frames of one vehicle's track is a
reading, and :class:`PlateAccumulator` keeps the count that says which it is.
The count travels with the reading everywhere it is shown.

**An unresolved character stays unresolved.** :class:`Reading` renders it ``?``
and refuses to hand out a plate string at all until every position is resolved —
:attr:`Reading.text` is ``None`` while any is not. A read of ``B?7 4?21`` must
never be presented, matched or exported as ``BX7 4921``, and the only way to
guarantee that is for the completed form never to exist as a value.

**Normalisation folds only what the format makes unambiguous.**
:func:`normalise` exists so ``B 7421`` and ``B-7421`` are one vehicle. Every
fold beyond punctuation merges two real plates into one identity, so the fold
set is deliberately tiny — ``O``/``0`` and ``I``/``1``, and only at a position
whose class the country's format fixes. See :func:`normalise` for what it
refuses to fold and why.

**Nothing is downloaded.** The detector, the recogniser and the character set
are files the operator supplies; a missing one is an error that names the file,
never a cue to fetch it. The two model calls sit behind :class:`PlateBoxFinder`
and :class:`PlateTextReader` so the logic above — normalisation, voting,
thresholds, geometry — is exercised without any model present, which is also the
only condition under which most of it can be exercised at all.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from .core import BoundingBox
from .logs import get as _get_logger

_log = _get_logger(__name__)


#: Reads that must agree before a character position is called resolved. Two is
#: not enough: a recogniser makes the *same* mistake on consecutive frames of
#: the same vehicle far more often than it makes two independent ones, so a pair
#: is close to one observation counted twice.
MIN_AGREEMENT = 3

#: Agreement the weakest character needs before a whole reading is confident
#: enough to match against a register or a watchlist. Deliberately above
#: :data:`MIN_AGREEMENT`: resolving a character and betting a police call on the
#: plate are different bars, and a reading agreed by exactly the minimum is a
#: reading with no margin left for one bad frame.
CONFIDENT_AGREEMENT = 4

#: Per-character recogniser confidence below which a character does not vote at
#: all. A CTC decoder emits a character for every timestep it does not call
#: blank, including the ones it is barely committed to; letting those vote lets
#: a smear of glare outvote three clean frames.
MIN_CHARACTER_CONFIDENCE = 0.5

#: Smallest plate crop worth handing to a recogniser, in pixels. Below this
#: there is no character shape left to read and the model returns a confident
#: string anyway, which is the worst of both.
MIN_PLATE_PIXELS = 8

#: The character that stands for "not resolved". Chosen because it is not in any
#: plate character set, so it can never be mistaken for a read character, and
#: because it survives every display path unchanged.
UNRESOLVED = "?"


class PlateError(RuntimeError):
    """A plate reader could not be constructed or run."""


# ------------------------------------------------------------- normalisation


class CharClass(str, Enum):
    """What a country's format says may stand at one position.

    This is the whole basis on which a fold is allowed. Without a fixed class
    there is no way to tell an ``O`` that should be a zero from an ``O`` that is
    an ``O``, which is why :data:`GENERIC` folds nothing.
    """

    LETTER = "LETTER"
    DIGIT = "DIGIT"


#: No format. Punctuation and case are folded; no character ever is.
GENERIC = "GENERIC"

#: Characters a recogniser confuses because the *glyphs* are near-identical in
#: the fonts plates are set in — not because it misread. Folding these recovers
#: a plate; folding anything else merges two plates.
_TO_LETTER: Mapping[str, str] = {"0": "O", "1": "I"}
_TO_DIGIT: Mapping[str, str] = {"O": "0", "I": "1"}

#: Pairs deliberately **not** folded, with the reason, because the omission is
#: the point and an undocumented omission reads as an oversight:
#:
#: * ``S``/``5``, ``B``/``8``, ``Z``/``2``, ``G``/``6``, ``D``/``0``, ``Q``/``0``
#:   — these glyphs are plainly different on a plate, including in the fonts
#:   drawn to be machine-readable. A recogniser returning ``S`` where a ``5`` is
#:   displayed has misread, not mis-rendered, and folding hides a wrong read
#:   rather than repairing a confusable one.
#: * Anything at all under :data:`GENERIC`, or when no single format fits the
#:   read — with no position class, a fold is a guess.
#: * :data:`UNRESOLVED`. A character nobody could read is not evidence for the
#:   character the format would prefer there.
_REFUSED_FOLDS = ("S/5", "B/8", "Z/2", "G/6", "D/0", "Q/0")


@dataclass(frozen=True, slots=True)
class PlateFormat:
    """One fixed-length shape a country's plates may take.

    A country is a *tuple* of these rather than one pattern because real formats
    vary in length — a German plate is one to three district letters, one or two
    series letters and one to four digits — and a fold is only sound when
    exactly one shape can account for what was read.
    """

    country: str
    name: str
    positions: tuple[CharClass, ...]

    def fits(self, text: str) -> bool:
        """Whether this shape can account for ``text``, folds included.

        A character fits its position when it is already of that class, when it
        is one of the two foldable glyphs for that class, or when it is
        :data:`UNRESOLVED` — an unread character rules nothing out.
        """
        if len(text) != len(self.positions):
            return False
        return all(
            _fits_class(character, klass)
            for character, klass in zip(text, self.positions)
        )


def _is_letter(character: str) -> bool:
    return "A" <= character <= "Z"


def _is_digit(character: str) -> bool:
    return "0" <= character <= "9"


def _fits_class(character: str, klass: CharClass) -> bool:
    if character == UNRESOLVED:
        return True
    if klass is CharClass.LETTER:
        return _is_letter(character) or character in _TO_LETTER
    return _is_digit(character) or character in _TO_DIGIT


def _fold(character: str, klass: CharClass) -> str:
    if character == UNRESOLVED:
        return character
    if klass is CharClass.LETTER:
        return _TO_LETTER.get(character, character)
    return _TO_DIGIT.get(character, character)


def _shapes(country: str, name: str, blocks: Sequence[tuple[CharClass, int, int]]):
    """Every fixed-length shape a block description allows.

    ``blocks`` is a sequence of ``(class, minimum, maximum)``. Expanding the
    ranges up front means the fitting test is a length comparison and a
    per-position class check, rather than a regular expression whose capture
    groups would have to be mapped back to positions to know what may be folded.
    """
    shapes: list[tuple[CharClass, ...]] = [()]
    for klass, low, high in blocks:
        grown: list[tuple[CharClass, ...]] = []
        for prefix in shapes:
            for count in range(low, high + 1):
                grown.append(prefix + (klass,) * count)
        shapes = grown
    return tuple(
        PlateFormat(country=country, name=f"{name} ({len(shape)})", positions=shape)
        for shape in shapes
    )


#: The formats this module knows. Two real ones and the empty one; a country
#: that is not here is read and stored, never folded, because inventing a format
#: for it would fold plates together that the country keeps apart.
FORMATS: Mapping[str, tuple[PlateFormat, ...]] = {
    GENERIC: (),
    # Current UK format: two memory letters, a two-digit age identifier, three
    # random letters. Fixed length, so every position's class is known.
    "UK": (
        PlateFormat(
            country="UK",
            name="AA00 AAA",
            positions=(
                CharClass.LETTER, CharClass.LETTER,
                CharClass.DIGIT, CharClass.DIGIT,
                CharClass.LETTER, CharClass.LETTER, CharClass.LETTER,
            ),
        ),
    ),
    # German format: one to three district letters, one or two series letters,
    # one to four digits. `B-7421` and `B 7421` are the same Berlin plate, which
    # is the case the vehicles register was specified against.
    "DE": _shapes(
        "DE",
        "district-series-number",
        ((CharClass.LETTER, 1, 3), (CharClass.LETTER, 1, 2), (CharClass.DIGIT, 1, 4)),
    ),
}


def supported_countries() -> tuple[str, ...]:
    """Country codes whose format is known well enough to fold a character."""
    return tuple(sorted(code for code, formats in FORMATS.items() if formats))


def _canonical(text: str) -> tuple[str, tuple[int, ...]]:
    """Upper-cased alphanumerics and ``?``, with the source index of each.

    Separators are dropped rather than replaced so a register key is the same
    whether the plate was painted with a space, a hyphen, a dot or a national
    badge between its blocks. The indices come back because a per-character
    confidence supplied against the raw read has to follow its character
    through — a confidence list silently misaligned by one dropped hyphen
    quietly votes every character against its neighbour.
    """
    kept: list[str] = []
    sources: list[int] = []
    for index, character in enumerate(text):
        upper = character.upper()
        if upper == UNRESOLVED or upper.isalnum():
            kept.append(upper)
            sources.append(index)
    return "".join(kept), tuple(sources)


def normalise(text: str, country: str = GENERIC) -> str:
    """The register key for a read plate, folding only what is unambiguous.

    Two stages, and the second one usually does nothing:

    1. **Punctuation and case.** ``b-7421``, ``B 7421`` and ``B.7421`` all
       become ``B7421``. This is the fold the vehicles register needs and it is
       safe everywhere, because no jurisdiction distinguishes two plates by the
       separator between their blocks.
    2. **Confusable glyphs, per position.** Only ``O``/``0`` and ``I``/``1``,
       and only when exactly one of ``country``'s shapes can account for the
       read. Where two shapes fit and disagree about a position, that position
       is left exactly as it was read: two candidate formats are not a format.

    What it refuses to fold, and why, is listed on :data:`_REFUSED_FOLDS`. The
    short version: a fold merges two real plates into one identity, so it is
    only worth making for glyphs that are hard to *tell apart*, never for
    characters a model got wrong.

    An unknown country is treated as :data:`GENERIC` — read, kept, never folded.
    """
    canonical, _ = _canonical(text)
    return _fold_to_format(canonical, country)


def _fold_to_format(canonical: str, country: str) -> str:
    formats = FORMATS.get(country.upper(), ())
    if not formats or not canonical:
        return canonical

    fitting = [shape for shape in formats if shape.fits(canonical)]
    if not fitting:
        # The read does not look like a plate of this country at all — a partial
        # read, a wrong crop, a foreign vehicle. Folding it to the nearest shape
        # would be inventing the format as well as the character.
        _log.debug("no %s format fits %r; folding nothing", country, canonical)
        return canonical

    folded: list[str] = []
    for index, character in enumerate(canonical):
        classes = {shape.positions[index] for shape in fitting}
        if len(classes) != 1:
            folded.append(character)
            continue
        folded.append(_fold(character, classes.pop()))
    return "".join(folded)


# --------------------------------------------------------------- one read


@dataclass(frozen=True, slots=True)
class PlateRead:
    """What one frame said the plate was. A guess, and labelled as one.

    Both forms are kept. ``raw_text`` is what the recogniser emitted and is what
    an operator is shown when they ask why a vehicle was matched; ``text`` is
    the normalised key the register and the watchlist compare against. Keeping
    only the second would make a wrong fold unfalsifiable.

    ``char_confidences`` is aligned to ``text``, one value per normalised
    character, so a caller never has to re-derive the mapping across dropped
    separators. It is empty when the recogniser cannot report per character —
    which is honest, and better than a fabricated ``1.0`` that would let a
    silent model outvote a measured one.
    """

    raw_text: str
    text: str
    char_confidences: tuple[float, ...]
    #: Where the plate was, in whole-frame normalised coordinates — not
    #: coordinates within the vehicle crop it was found in, which nothing
    #: downstream could interpret.
    box: BoundingBox
    #: The frame this came from, so the crop behind an unresolved character can
    #: be shown beside it.
    frame_index: int
    country: str = GENERIC
    #: The plate detector's own confidence that this box is a plate at all,
    #: which is a different claim from the characters being right.
    box_confidence: float | None = None

    @property
    def weakest_confidence(self) -> float | None:
        """The least confident character, or ``None`` when none were reported.

        The weakest character, not the mean: a plate is wrong if any one
        character is wrong, so averaging is exactly the operation that hides the
        one that matters.
        """
        if not self.char_confidences:
            return None
        return min(self.char_confidences)

    @classmethod
    def from_raw(
        cls,
        raw_text: str,
        confidences: Sequence[float],
        *,
        box: BoundingBox,
        frame_index: int,
        country: str = GENERIC,
        box_confidence: float | None = None,
    ) -> "PlateRead":
        """Normalise a recogniser's output, carrying its confidences across.

        Raises when the confidence list does not match the raw text, rather than
        padding or truncating it. A misaligned confidence list makes every
        threshold in this module test the wrong character, and it does so
        silently, which is the only reason this is a hard error.
        """
        if confidences and len(confidences) != len(raw_text):
            raise ValueError(
                f"{len(confidences)} confidences for {len(raw_text)} characters "
                f"in {raw_text!r}: the two must correspond one to one."
            )
        canonical, sources = _canonical(raw_text)
        text = _fold_to_format(canonical, country)
        kept = tuple(float(confidences[index]) for index in sources) if confidences else ()
        return cls(
            raw_text=raw_text,
            text=text,
            char_confidences=kept,
            box=box,
            frame_index=frame_index,
            country=country.upper(),
            box_confidence=box_confidence,
        )


# ------------------------------------------------------------ the reading


@dataclass(frozen=True, slots=True)
class Reading:
    """What a whole track said its plate was, with the evidence for each character.

    ``characters`` holds one entry per position: the resolved character, or
    :data:`UNRESOLVED`. ``agreement`` holds, position by position, how many
    reads agreed on it — zero where nothing resolved. The two are the same
    length and are meant to be read together; an operator shown a plate without
    its agreement count has been shown a conclusion without its evidence.

    There is deliberately no way to obtain a completed string from a reading
    that has an unresolved character. :attr:`text` is ``None`` until every
    position resolves, and :attr:`display` shows the ``?`` rather than filling
    it in.
    """

    country: str
    characters: tuple[str, ...]
    agreement: tuple[int, ...]
    #: Reads that voted, and reads set aside because their length disagreed with
    #: the length the majority of reads found. Both are reported: a reading
    #: built from four reads out of twenty is a different thing from one built
    #: from four out of four.
    contributing_reads: int
    set_aside_reads: int
    min_agreement: int
    #: The clearest read behind this reading, kept so its crop can be shown next
    #: to an unresolved character. ``None`` when nothing was ever added.
    best_read: PlateRead | None = None

    @property
    def total_reads(self) -> int:
        return self.contributing_reads + self.set_aside_reads

    @property
    def display(self) -> str:
        """What an operator is shown: the characters, with ``?`` for the rest."""
        return "".join(self.characters)

    @property
    def text(self) -> str | None:
        """The plate, or ``None`` while any character is unresolved.

        The central promise of this module, expressed as a type rather than a
        convention: there is no completed string to accidentally match, export
        or log while a character is missing.
        """
        if not self.characters or UNRESOLVED in self.characters:
            return None
        return self.display

    @property
    def unresolved_count(self) -> int:
        return sum(1 for character in self.characters if character == UNRESOLVED)

    @property
    def is_resolved(self) -> bool:
        return bool(self.characters) and UNRESOLVED not in self.characters

    @property
    def weakest_agreement(self) -> int:
        """Agreement behind the least-agreed character; ``0`` if any is unresolved."""
        if not self.agreement:
            return 0
        return min(self.agreement)

    @property
    def is_confident(self) -> bool:
        """Whether this reading may be matched against a register or watchlist.

        Two conditions, both necessary: every character is resolved, and the
        *weakest* of them was agreed by at least :data:`CONFIDENT_AGREEMENT`
        reads. The weakest rather than the average, because one wrong character
        is a different vehicle — often a real one, belonging to somebody who has
        no idea why they were stopped.
        """
        return self.is_resolved and self.weakest_agreement >= CONFIDENT_AGREEMENT

    def describe(self) -> str:
        """One line for a log or a panel, always carrying the qualification."""
        if not self.characters:
            return f"no reading from {self.total_reads} read(s)"
        state = "confident" if self.is_confident else "unconfirmed"
        return (
            f"{self.display} [{state}] agreement {self.weakest_agreement}"
            f"/{self.contributing_reads} read(s), {self.unresolved_count} unresolved"
        )


class PlateAccumulator:
    """Many reads of one track's plate, collapsed into one reading.

    Character positions vote independently. A position resolves when the leading
    candidate has at least ``min_agreement`` votes *and* strictly more than the
    runner-up; anything else stays :data:`UNRESOLVED`. Both halves matter: the
    count alone would resolve a position that four frames called ``8`` and four
    called ``B``, which is precisely the position that must not resolve.

    **Reads whose length disagrees are set aside, not aligned.** Aligning a
    six-character read against seven-character ones needs an edit-distance
    alignment, and every alignment it chooses is a guess about which character
    was dropped — the same guess this class exists to refuse. The count of
    reads set aside is reported so a track whose reads never settled on a length
    is visibly that, rather than quietly a short reading.
    """

    __slots__ = ("_country", "_min_agreement", "_min_confidence", "_reads")

    def __init__(
        self,
        *,
        country: str = GENERIC,
        min_agreement: int = MIN_AGREEMENT,
        min_character_confidence: float = MIN_CHARACTER_CONFIDENCE,
    ):
        if min_agreement < 1:
            raise ValueError("min_agreement must be at least 1")
        self._country = country.upper()
        self._min_agreement = int(min_agreement)
        self._min_confidence = float(min_character_confidence)
        self._reads: list[PlateRead] = []

    @property
    def country(self) -> str:
        return self._country

    @property
    def min_agreement(self) -> int:
        return self._min_agreement

    def __len__(self) -> int:
        return len(self._reads)

    def add(self, read: PlateRead) -> None:
        """Take one frame's read. Empty reads are dropped, not counted as votes."""
        if not read.text:
            return
        self._reads.append(read)

    def add_all(self, reads: Sequence[PlateRead]) -> None:
        for read in reads:
            self.add(read)

    def _votes(self, position: int, reads: Sequence[PlateRead]) -> Counter:
        tally: Counter = Counter()
        for read in reads:
            character = read.text[position]
            if character == UNRESOLVED:
                # A read that could not read this position is not evidence for
                # what stands there.
                continue
            if read.char_confidences:
                if read.char_confidences[position] < self._min_confidence:
                    continue
            tally[character] += 1
        return tally

    def resolve(self) -> Reading:
        """The reading so far. Cheap enough to call every frame."""
        if not self._reads:
            return Reading(
                country=self._country,
                characters=(),
                agreement=(),
                contributing_reads=0,
                set_aside_reads=0,
                min_agreement=self._min_agreement,
            )

        by_length: dict[int, list[PlateRead]] = {}
        for read in self._reads:
            by_length.setdefault(len(read.text), []).append(read)

        # Most reads wins; ties broken by the confidence behind them and then by
        # the longer read. A tie broken the wrong way costs agreement, never
        # correctness — the characters in the losing bucket simply never vote,
        # and a position with no votes stays unresolved.
        def weight(item: tuple[int, list[PlateRead]]) -> tuple[int, float, int]:
            length, reads = item
            confidence = sum(
                read.weakest_confidence if read.weakest_confidence is not None else 0.0
                for read in reads
            )
            return (len(reads), confidence, length)

        length, chosen = max(by_length.items(), key=weight)
        set_aside = len(self._reads) - len(chosen)

        characters: list[str] = []
        agreement: list[int] = []
        for position in range(length):
            tally = self._votes(position, chosen)
            ranked = tally.most_common(2)
            if not ranked:
                characters.append(UNRESOLVED)
                agreement.append(0)
                continue
            character, count = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0
            if count >= self._min_agreement and count > runner_up:
                characters.append(character)
                agreement.append(count)
            else:
                characters.append(UNRESOLVED)
                agreement.append(0)

        return Reading(
            country=self._country,
            characters=tuple(characters),
            agreement=tuple(agreement),
            contributing_reads=len(chosen),
            set_aside_reads=set_aside,
            min_agreement=self._min_agreement,
            best_read=_clearest(chosen),
        )


def _clearest(reads: Sequence[PlateRead]) -> PlateRead | None:
    """The read whose weakest character was strongest — the crop worth showing."""
    if not reads:
        return None
    return max(
        reads,
        key=lambda read: (
            read.weakest_confidence if read.weakest_confidence is not None else -1.0
        ),
    )


# ------------------------------------------------------------- the models


class PlateBoxFinder(Protocol):
    """Finds plate boxes inside an image that is already a vehicle crop.

    Boxes come back normalised to the image given, not to the frame, because
    that is the only coordinate system this collaborator can honestly know.
    """

    def find(self, image: np.ndarray) -> Sequence[tuple[BoundingBox, float]]: ...


class PlateTextReader(Protocol):
    """Turns a plate crop into characters and, where it can, per-character confidence.

    An implementation that cannot report per character returns an empty
    confidence sequence rather than filling it with ones.
    """

    def read_text(self, image: np.ndarray) -> tuple[str, Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class PlateModels:
    """The three files the operator supplies, and nothing else.

    Held together because they are useless apart — a recogniser without its
    country's character set decodes into whatever alphabet happened to be
    compiled in, which produces plausible characters from the wrong alphabet.
    """

    detector_path: Path
    recogniser_path: Path
    charset_path: Path

    def __post_init__(self) -> None:
        for field_name in ("detector_path", "recogniser_path", "charset_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

    def missing(self) -> tuple[tuple[str, Path], ...]:
        """Every named file that is not on disk, in the order they are needed."""
        named = (
            ("plate detector", self.detector_path),
            ("plate text recogniser", self.recogniser_path),
            ("plate character set", self.charset_path),
        )
        return tuple((name, path) for name, path in named if not path.is_file())

    def require_present(self) -> None:
        """Raise unless all three files exist, naming the ones that do not.

        Named individually because "a model is missing" sends an installer
        looking through three paths, and because the honest response to a
        missing model is to say which one — never to reach for a copy of it.
        """
        absent = self.missing()
        if not absent:
            return
        listed = "; ".join(f"no {name} at {path}" for name, path in absent)
        raise PlateError(
            f"{listed}. Plate models are supplied by the operator and placed in "
            "the models directory; nothing is ever downloaded."
        )


@dataclass(frozen=True, slots=True)
class PlateReaderInfo:
    """What read a plate, for the record.

    The digests are here for the same reason the detector's are: months later,
    "why did it say that" has to be answerable against the exact weights that
    said it.
    """

    country: str
    detector_path: str
    detector_sha256: str
    recogniser_path: str
    recogniser_sha256: str
    charset_path: str
    charset_size: int


def _digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large model is not held in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_charset(path: Path | str) -> tuple[str, ...]:
    """The recogniser's alphabet, one character per line, in model order.

    Order is the whole content of this file: the index a CTC decoder produces
    means nothing except through this list, so a charset that is right but
    reordered produces fluent, confident, wrong plates. Blank lines are dropped;
    an empty file is an error rather than an empty alphabet, because an empty
    alphabet decodes every plate to the empty string and looks like a camera
    problem.
    """
    file = Path(path)
    if not file.is_file():
        raise PlateError(
            f"No plate character set at {file}. It is supplied by the operator "
            "alongside the recogniser weights; nothing is ever downloaded."
        )
    entries = tuple(
        line.strip() for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not entries:
        raise PlateError(
            f"The plate character set at {file} is empty; it must list the "
            "recogniser's characters, one per line, in the model's own order."
        )
    return entries


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted so a large logit cannot overflow to ``inf``."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def ctc_greedy_decode(
    probabilities: np.ndarray,
    vocabulary: Sequence[str],
    *,
    blank_index: int = 0,
) -> tuple[str, tuple[float, ...]]:
    """Greedy CTC decode that keeps a confidence per emitted character.

    OpenCV's ``TextRecognitionModel.recognize`` does this decode internally and
    returns the string alone, which is why it is done here instead: without a
    per-character probability there is nothing to threshold, and every character
    of every read would have to be treated as equally believed. That is the
    difference between a reading that knows which character is weak and one that
    does not.

    ``probabilities`` is ``(timesteps, classes)``, already normalised. Repeated
    timesteps that a CTC decoder merges into one character take the *highest*
    of their probabilities: the merged run is one observation, and the timestep
    where the character was clearest is the one that observed it.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            f"expected (timesteps, classes); got shape {probabilities.shape}"
        )
    best = probabilities.argmax(axis=1)
    strength = probabilities.max(axis=1)

    characters: list[str] = []
    confidences: list[float] = []
    previous = -1
    for step, index in enumerate(int(value) for value in best):
        if index == blank_index:
            previous = index
            continue
        if index == previous and characters:
            confidences[-1] = max(confidences[-1], float(strength[step]))
            continue
        position = index - 1 if blank_index == 0 else index
        if not 0 <= position < len(vocabulary):
            raise PlateError(
                f"the recogniser produced class {index}, which the character set "
                f"of {len(vocabulary)} entries does not describe; the charset and "
                "the weights do not belong together."
            )
        characters.append(vocabulary[position])
        confidences.append(float(strength[step]))
        previous = index
    return "".join(characters), tuple(confidences)


class _OnnxPlateBoxFinder:
    """Plate boxes from the operator's ONNX detector, run on a vehicle crop only.

    Wraps the existing :class:`~sentinel.detect.OnnxDetector` rather than
    decoding model output again here. That decode — layout inference, letterbox
    unpadding, non-maximum suppression — is the part with the invisible bugs,
    and one copy of it that the whole system exercises is worth more than a
    second copy exercised only when a plate is in shot.
    """

    __slots__ = ("_detector", "_min_confidence")

    def __init__(self, model_path: Path, min_confidence: float):
        from .detect import DetectionError, OnnxDetector

        try:
            self._detector = OnnxDetector(model_path, confidence_threshold=min_confidence)
        except DetectionError as error:
            raise PlateError(f"Could not load the plate detector: {error}") from error
        self._min_confidence = min_confidence

    def find(self, image: np.ndarray) -> Sequence[tuple[BoundingBox, float]]:
        return [
            (detection.bbox, float(detection.confidence))
            for detection in self._detector.detect(image)
            if detection.confidence >= self._min_confidence
        ]


class _OpenCvTextReader:
    """`cv2.dnn.TextRecognitionModel` (CRNN), decoded here for the confidences.

    The model object is used for its preprocessing and its forward pass; the
    decode is :func:`ctc_greedy_decode` because OpenCV's own returns a string
    with nothing behind it. Nothing in this class can run without the operator's
    weights, which is exactly why it is this thin: everything that decides
    anything lives above it, behind :class:`PlateTextReader`, where a test can
    reach it.
    """

    __slots__ = ("_model", "_vocabulary", "_input_size")

    def __init__(
        self,
        model_path: Path,
        charset_path: Path,
        *,
        input_size: tuple[int, int] = (100, 32),
        scale: float = 1.0 / 127.5,
        mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
    ):
        self._vocabulary = load_charset(charset_path)
        try:
            model = cv2.dnn.TextRecognitionModel(str(model_path))
            model.setDecodeType("CTC-greedy")
            model.setVocabulary(list(self._vocabulary))
            model.setInputParams(scale, input_size, mean)
        except cv2.error as error:
            raise PlateError(
                f"Could not load the plate text recogniser at {model_path}: {error}"
            ) from error
        self._model = model
        self._input_size = input_size

    def read_text(self, image: np.ndarray) -> tuple[str, Sequence[float]]:
        outputs = self._model.predict(image)
        scores = np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
        scores = np.squeeze(scores)
        if scores.ndim != 2:
            raise PlateError(
                f"the recogniser produced an output of shape {scores.shape}; a CRNN "
                "head is expected to produce one score vector per timestep."
            )
        # A CRNN head may come back either way round. The class axis is the one
        # matching the character set, and guessing wrong would decode timesteps
        # as characters — fluent nonsense rather than an error.
        classes = len(self._vocabulary) + 1
        if scores.shape[1] != classes and scores.shape[0] == classes:
            scores = scores.T
        return ctc_greedy_decode(softmax(scores.astype(np.float32)), self._vocabulary)


# ------------------------------------------------------------- the reader


def _pixel_box(image: np.ndarray, box: BoundingBox) -> tuple[int, int, int, int]:
    """A normalised box as pixel bounds clamped to the image.

    Clamped rather than trusted: a track's box can extend past the frame edge
    when its estimate leads the object, and a negative slice index in numpy
    means "from the end", so an unclamped crop silently reads the wrong side of
    the image instead of failing.
    """
    height, width = image.shape[:2]
    left = int(round(max(0.0, box.x) * width))
    top = int(round(max(0.0, box.y) * height))
    right = int(round(min(1.0, box.x + box.w) * width))
    bottom = int(round(min(1.0, box.y + box.h) * height))
    return left, top, max(left, right), max(top, bottom)


class PlateReader:
    """Reads plates inside vehicle boxes, from models the operator supplied.

    Constructed against :class:`PlateModels`, whose files must all exist — the
    check runs even when both collaborators are injected, so a fake can never
    stand in for a model an installation is missing. Both model calls sit behind
    the two protocols above, which is what lets the geometry, the coordinate
    mapping and the refusal to read a too-small crop be tested on this machine,
    where no plate model exists.

    It never sees a whole frame. :meth:`read` crops the vehicle box first and
    the detector is given that crop, because a plate reader pointed at the
    frame is a plate reader pointed at the street outside the site boundary.
    """

    __slots__ = ("_models", "_country", "_finder", "_reader", "_min_box_confidence",
                 "_min_pixels", "_info")

    def __init__(
        self,
        models: PlateModels,
        *,
        country: str = GENERIC,
        box_finder: PlateBoxFinder | None = None,
        text_reader: PlateTextReader | None = None,
        min_box_confidence: float = 0.35,
        min_plate_pixels: int = MIN_PLATE_PIXELS,
    ):
        models.require_present()
        self._models = models
        self._country = country.upper()
        self._min_box_confidence = float(min_box_confidence)
        self._min_pixels = int(min_plate_pixels)
        self._finder = box_finder or _OnnxPlateBoxFinder(
            models.detector_path, self._min_box_confidence
        )
        self._reader = text_reader or _OpenCvTextReader(
            models.recogniser_path, models.charset_path
        )
        self._info = PlateReaderInfo(
            country=self._country,
            detector_path=str(models.detector_path),
            detector_sha256=_digest(models.detector_path),
            recogniser_path=str(models.recogniser_path),
            recogniser_sha256=_digest(models.recogniser_path),
            charset_path=str(models.charset_path),
            charset_size=len(load_charset(models.charset_path)),
        )

    @property
    def info(self) -> PlateReaderInfo:
        return self._info

    @property
    def country(self) -> str:
        return self._country

    def read(
        self,
        frame: np.ndarray,
        vehicle_box: BoundingBox,
        *,
        frame_index: int,
    ) -> list[PlateRead]:
        """Every plate read inside one vehicle's box on one frame.

        Returns an empty list, not an error, when the vehicle box is off the
        frame or too small to hold readable characters: a vehicle at the far end
        of a car park is a normal thing to see, and a reader that raised on it
        would take the pipeline down over ordinary distance.
        """
        left, top, right, bottom = _pixel_box(frame, vehicle_box)
        if right - left < self._min_pixels or bottom - top < self._min_pixels:
            return []

        crop = frame[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        frame_height, frame_width = frame.shape[:2]

        reads: list[PlateRead] = []
        for box, box_confidence in self._finder.find(crop):
            if box_confidence < self._min_box_confidence:
                continue
            plate_left, plate_top, plate_right, plate_bottom = _pixel_box(crop, box)
            if (plate_right - plate_left < self._min_pixels
                    or plate_bottom - plate_top < self._min_pixels):
                # Too few pixels to carry a character shape. A recogniser run on
                # one still returns a string, and that string is noise wearing a
                # plate's clothes.
                continue
            plate = crop[plate_top:plate_bottom, plate_left:plate_right]
            raw_text, confidences = self._reader.read_text(plate)
            if not raw_text:
                continue
            reads.append(
                PlateRead.from_raw(
                    raw_text,
                    confidences,
                    box=BoundingBox(
                        x=(left + plate_left) / frame_width,
                        y=(top + plate_top) / frame_height,
                        w=(plate_right - plate_left) / frame_width,
                        h=(plate_bottom - plate_top) / frame_height,
                    ),
                    frame_index=frame_index,
                    country=self._country,
                    box_confidence=float(box_confidence),
                )
            )
        _log.debug(
            "frame %d: %d plate read(s) in a %dx%d vehicle crop",
            frame_index, len(reads), crop_width, crop_height,
        )
        return reads
