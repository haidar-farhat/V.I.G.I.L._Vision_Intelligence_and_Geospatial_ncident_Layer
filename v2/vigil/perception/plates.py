"""Reading a number plate, and refusing to produce one that is half-guessed.

# The idea worth keeping from v1

**A reading has no text while any character is unresolved.** Not a string with
a `?` in it — `None`. The completed string never exists as a value, so it
cannot be logged, exported, matched against a watch list, or pasted into a
report by somebody who did not read the confidence beside it. Everything else
in this module exists to serve that.

# How a plate is resolved

Frame by frame the OCR proposes a string and a per-character confidence. Those
are accumulated per track and voted **per character position**:

- Reads of a different length are **set aside**, not aligned. Aligning them is
  guessing where the missing character was, and the count of set-aside reads
  is reported, because four agreeing reads out of twenty is a different claim
  from four out of four.
- A position resolves when one character has at least `MIN_AGREEMENT` votes
  **and strictly more than the runner-up**. The count alone would resolve a
  position that four frames called `8` and four called `B`.
- The reading's confidence is its **weakest** character, never the mean. A
  plate is wrong if any one character is wrong, so averaging is exactly the
  operation that hides the character that matters.

# What v1 got wrong here, and what this does instead

- **It softmaxed unconditionally.** A model whose head already emits
  probabilities was softmaxed again, flattening every confidence to about
  0.3, which fell under the character threshold, so every position stayed
  unresolved for ever with no error anywhere. Here the output is inspected:
  rows that already sum to one are left alone.
- **Its CTC blank index was wrong unless the blank was first.** `index - 1 if
  blank_index == 0 else index` is off by one for every class above a blank
  that sits anywhere in the middle — fluent, confident, systematically wrong
  plates. Here the mapping is built from the vocabulary and the blank's
  position, whatever it is.
- **A read with no per-character confidence voted unconditionally.** v1's own
  docstring said the empty tuple was honest and better than a fabricated 1.0,
  and then let it skip the threshold entirely, which is exactly what a
  fabricated 1.0 would have done. Here an unmeasured read is counted at
  `UNMEASURED_CONFIDENCE` and says so.
- **Preprocessing was nailed to one convention** — 100x32, mean 127.5, scale
  1/127.5 — with no way to change it. Wrong values do not error; they produce
  plausible characters. Here they are arguments with the common default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..kernel import onnx as _onnx
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Frames that must agree on a character before it is resolved.
#:
#: Three. A recogniser makes the *same* mistake on consecutive frames far more
#: often than two independent ones do, so a pair is close to one observation
#: counted twice. Carried from v1 with its reasoning intact.
MIN_AGREEMENT = 3

#: Agreement a reading needs before it may be acted on, as opposed to shown.
#: Four: a reading agreed by exactly the minimum has no margin left for one
#: bad frame. Resolving and acting are different decisions.
CONFIDENT_AGREEMENT = 4

#: A character below this confidence does not vote.
MIN_CHARACTER_CONFIDENCE = 0.5

#: What a read with no per-character confidences is counted at.
#:
#: Just under the threshold, so it is recorded and does **not** vote. A model
#: that cannot report per-character confidence has not told us its characters
#: are good; treating silence as certainty is the failure this replaces.
UNMEASURED_CONFIDENCE = MIN_CHARACTER_CONFIDENCE - 0.01

#: Shown for a position nothing has resolved.
UNRESOLVED = "?"

#: Characters a plate may contain, after normalisation.
ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


class PlateError(RuntimeError):
    """A plate model or charset that cannot be used."""


@dataclass(frozen=True, slots=True)
class Read:
    """One frame's attempt at the plate.

    `characters` is empty when the model cannot report per-character
    confidence. Empty, never a fabricated 1.0 — and `Accumulator` treats an
    empty one as `UNMEASURED_CONFIDENCE` rather than letting it past the gate.
    """

    text: str
    characters: tuple[float, ...] = ()
    frame_index: int = 0
    #: What the OCR actually emitted, before normalisation. Kept so a wrong
    #: normalisation is falsifiable rather than invisible.
    raw: str = ""

    @property
    def weakest(self) -> float:
        """The least confident character, or `UNMEASURED_CONFIDENCE`.

        The weakest, never the mean: a plate is wrong if any one character is
        wrong, so the mean is precisely the statistic that hides it.
        """
        return min(self.characters) if self.characters else UNMEASURED_CONFIDENCE

    def confidence_at(self, position: int) -> float:
        if not self.characters or position >= len(self.characters):
            return UNMEASURED_CONFIDENCE
        return self.characters[position]


@dataclass(frozen=True, slots=True)
class Reading:
    """What several frames agree the plate is, if they agree at all."""

    characters: tuple[str, ...]
    agreement: tuple[int, ...]
    reads: int
    set_aside: int
    min_agreement: int
    weakest_confidence: float

    @property
    def text(self) -> str | None:
        """The plate, or `None` while any character is unresolved.

        `None` rather than a string with a `?` in it. A half-read plate must
        not be a value that can be logged, exported or matched, because the
        moment it is, somebody downstream will treat it as a plate.
        """
        if not self.characters or UNRESOLVED in self.characters:
            return None
        return "".join(self.characters)

    @property
    def display(self) -> str:
        """For a screen, where showing the partial read is useful."""
        return "".join(self.characters) if self.characters else ""

    @property
    def resolved(self) -> bool:
        return self.text is not None

    @property
    def confident(self) -> bool:
        """Resolved, and with a margin left over for one bad frame."""
        return (self.resolved and bool(self.agreement)
                and min(self.agreement) >= CONFIDENT_AGREEMENT)

    @property
    def weakest_agreement(self) -> int:
        return min(self.agreement) if self.agreement else 0

    def describe(self) -> str:
        if not self.characters:
            return f"no plate read from {self.reads} attempt(s)"
        aside = (f", {self.set_aside} read(s) of a different length set aside"
                 if self.set_aside else "")
        if not self.resolved:
            unresolved = sum(1 for c in self.characters if c == UNRESOLVED)
            return (f"{self.display} — {unresolved} character(s) unresolved after "
                    f"{self.reads} read(s){aside}. Not a plate")
        return (f"{self.text} — every character agreed by at least "
                f"{self.weakest_agreement} of {self.reads} read(s), weakest character "
                f"{self.weakest_confidence:.2f}{aside}"
                f"{'' if self.confident else '. Below the bar for acting on'}")


@dataclass
class Accumulator:
    """Reads of one vehicle's plate, voted into a reading."""

    min_agreement: int = MIN_AGREEMENT
    min_confidence: float = MIN_CHARACTER_CONFIDENCE
    reads: list[Read] = field(default_factory=list)

    def add(self, read: Read) -> None:
        if read.text:
            self.reads.append(read)

    def resolve(self) -> Reading:
        if not self.reads:
            return Reading((), (), 0, 0, self.min_agreement, 0.0)
        # Bucket by length and take the majority. Reads of another length are
        # set aside rather than aligned: aligning them means guessing where
        # the missing character was, and a guess in the middle of a plate is
        # worse than an unresolved position.
        lengths = Counter(len(r.text) for r in self.reads)
        length = lengths.most_common(1)[0][0]
        agreeing = [r for r in self.reads if len(r.text) == length]
        set_aside = len(self.reads) - len(agreeing)

        characters: list[str] = []
        agreement: list[int] = []
        weakest = 1.0
        for position in range(length):
            votes: Counter[str] = Counter()
            best_confidence: dict[str, float] = {}
            for read in agreeing:
                confidence = read.confidence_at(position)
                if confidence < self.min_confidence:
                    continue
                character = read.text[position]
                votes[character] += 1
                best_confidence[character] = max(best_confidence.get(character, 0.0), confidence)
            if not votes:
                characters.append(UNRESOLVED)
                agreement.append(0)
                continue
            ordered = votes.most_common(2)
            (winner, count) = ordered[0]
            runner_up = ordered[1][1] if len(ordered) > 1 else 0
            # Both gates. The count alone resolves a position that four frames
            # called 8 and four called B.
            if count >= self.min_agreement and count > runner_up:
                characters.append(winner)
                agreement.append(count)
                weakest = min(weakest, best_confidence.get(winner, UNMEASURED_CONFIDENCE))
            else:
                characters.append(UNRESOLVED)
                agreement.append(count)
        return Reading(tuple(characters), tuple(agreement), len(self.reads), set_aside,
                       self.min_agreement, 0.0 if not characters else weakest)


def normalise(text: str) -> str:
    """The OCR's string as this module votes on it.

    Upper-cased, with anything outside `ALPHABET` dropped. Punctuation and
    spaces vary between frames on the same plate and would split the length
    buckets, which is the one thing that silently halves the evidence.
    """
    return "".join(c for c in (text or "").upper() if c in ALPHABET)


def probabilities(scores: np.ndarray) -> np.ndarray:
    """Per-timestep probabilities, softmaxing only if they are not already.

    The check is the point. Softmaxing a probability vector flattens it — a
    0.99 becomes about 0.3 — every character then falls under the confidence
    threshold, no position ever resolves, and nothing anywhere raises. v1 did
    this unconditionally and would have failed silently and totally on a model
    whose head already ends in a softmax.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise PlateError(f"expected (timesteps, classes) logits; got shape {matrix.shape}")
    sums = matrix.sum(axis=1)
    if np.all(matrix >= 0.0) and np.allclose(sums, 1.0, atol=1e-3):
        return matrix
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def ctc_decode(probs: np.ndarray, vocabulary: Sequence[str],
               blank_index: int = 0) -> tuple[str, tuple[float, ...]]:
    """Greedy CTC, keeping each character's own probability.

    OpenCV's `recognize()` returns a string and nothing else, which is why
    this is written out: the per-character confidence is the number the vote
    depends on, and a reading whose weakest character cannot be seen is a
    reading nobody can judge.

    The class-to-character mapping is built from where the blank actually is.
    v1 used `index - 1 if blank_index == 0 else index`, which is right by luck
    when the blank is first or last and off by one for every class above a
    blank anywhere else.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    classes = matrix.shape[1]
    if not 0 <= blank_index < classes:
        raise PlateError(f"blank index {blank_index} is outside the {classes} classes")
    if classes != len(vocabulary) + 1:
        raise PlateError(
            f"the model has {classes} classes and the charset has {len(vocabulary)} characters, "
            f"which with one blank should be {len(vocabulary) + 1}. A charset that is the wrong "
            f"length decodes into fluent, confident, wrong plates"
        )
    # Class index to vocabulary index, skipping the blank wherever it sits.
    mapping = {}
    seen = 0
    for index in range(classes):
        if index == blank_index:
            continue
        mapping[index] = seen
        seen += 1

    characters: list[str] = []
    confidences: list[float] = []
    previous = -1
    best = matrix.argmax(axis=1)
    for step, index in enumerate(best):
        index = int(index)
        if index == blank_index:
            previous = -1
            continue
        confidence = float(matrix[step, index])
        if index == previous:
            # A repeated timestep is the same character; keep the better view.
            if confidences:
                confidences[-1] = max(confidences[-1], confidence)
            continue
        previous = index
        characters.append(vocabulary[mapping[index]])
        confidences.append(confidence)
    return "".join(characters), tuple(confidences)


def load_charset(path: str | Path) -> tuple[str, ...]:
    """The character set, in the model's own order.

    Order is everything: a charset that holds the right characters in the
    wrong order produces fluent, confident, wrong plates, and nothing about
    the output looks unusual. An empty file is an error rather than an empty
    alphabet that decodes everything to nothing.
    """
    file = Path(path)
    if not file.is_file():
        raise PlateError(f"no charset at {file}; it is supplied with the model, never downloaded")
    characters = tuple(line.strip() for line in file.read_text(encoding="utf-8").splitlines()
                       if line.strip())
    if not characters:
        raise PlateError(f"the charset at {file} is empty, so nothing could be decoded")
    return characters


@dataclass(frozen=True, slots=True)
class PlateModels:
    """The three files a plate reader needs, which are useless apart.

    Grouped because a recogniser without its charset decodes into whatever
    alphabet happened to be compiled in, and the failure looks like bad OCR
    rather than like a missing file.
    """

    detector: Path
    recogniser: Path
    charset: Path

    def missing(self) -> list[Path]:
        return [p for p in (self.detector, self.recogniser, self.charset) if not Path(p).is_file()]

    def require_present(self) -> None:
        absent = self.missing()
        if absent:
            # Each one named. "A model is missing" sends an installer looking
            # through three paths.
            raise PlateError(
                "these plate files are missing and are supplied by the operator — nothing is "
                "downloaded:\n  " + "\n  ".join(str(p) for p in absent))


class PlateReader:
    """Plate boxes inside a vehicle box, and the characters inside those.

    **The vehicle box is required.** A plate reader pointed at the whole frame
    is a plate reader pointed at the road outside the site boundary, and there
    is deliberately no default that permits it.
    """

    def __init__(self, models: PlateModels, *, vocabulary: Sequence[str] | None = None,
                 input_size: tuple[int, int] = (100, 32),
                 mean: float = 127.5, scale: float = 1.0 / 127.5,
                 blank_index: int = 0, min_box_confidence: float = 0.35):
        models.require_present()
        self.models = models
        #: Preprocessing is configurable because it is model-specific and
        #: getting it wrong does not raise — it produces plausible characters.
        #: The defaults are the common CRNN convention, not a law.
        self.input_size = input_size
        self.mean = mean
        self.scale = scale
        self.blank_index = blank_index
        self.min_box_confidence = min_box_confidence
        self.vocabulary = tuple(vocabulary) if vocabulary else load_charset(models.charset)
        self._session, self.provider = _onnx.open_session(models.recogniser)
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]
        self.model_sha256 = _onnx.digest(models.recogniser)
        _log.info("plate reader ready: %s on %s, %d character(s)",
                  Path(models.recogniser).name, self.provider, len(self.vocabulary))

    def read(self, image: np.ndarray, vehicle_box, *, frame_index: int = 0) -> Read | None:
        """One vehicle's plate from one frame, or `None`.

        `vehicle_box` is a `BoundingBox` in frame fractions and is not
        optional.
        """
        import cv2

        crop = _crop(image, vehicle_box)
        if crop is None:
            return None
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        blob = ((grey.astype(np.float32) - self.mean) * self.scale)[None, None, :, :]
        try:
            outputs = self._session.run(self._outputs, {self._input: blob})
        except Exception as error:  # noqa: BLE001 - onnxruntime raises many types
            _log.debug("plate recognition failed on frame %d: %s", frame_index, error)
            return None
        scores = np.asarray(outputs[0])
        while scores.ndim > 2:
            scores = scores[0]
        text, confidences = ctc_decode(probabilities(scores), self.vocabulary, self.blank_index)
        cleaned = normalise(text)
        if not cleaned:
            return None
        # Confidences are dropped when normalisation changed the length: a
        # confidence list that no longer lines up with its characters is worse
        # than none, because it thresholds the wrong character.
        aligned = confidences if len(confidences) == len(cleaned) else ()
        return Read(cleaned, aligned, frame_index, raw=text)


#: How far outside `[0, 1]` a box may stray and still be treated as a box.
#:
#: A thousandth. Real boxes overshoot by an ulp or two after a clamp or a
#: rescale, and refusing those would reject legitimate detections at the frame
#: edge. Anything further out is not a normalised box at all, and the
#: important part is that it is **refused rather than clamped**: clamping
#: `BoundingBox(0, 0, 200.0, 150.0)` produces the whole frame, which is
#: exactly the thing requiring a person box was meant to prevent. v1 clamped,
#: and shipped that hole for a year.
BOX_TOLERANCE = 1e-3


def _sane(box) -> bool:
    return (box.width > 0 and box.height > 0
            and box.x >= -BOX_TOLERANCE and box.y >= -BOX_TOLERANCE
            and box.right <= 1.0 + BOX_TOLERANCE and box.bottom <= 1.0 + BOX_TOLERANCE)


def _crop(image: np.ndarray, box) -> np.ndarray | None:
    """The box's pixels, or `None` when there are too few to read.

    Both the offset **and the size** are bounds-checked. v1 checked only x and
    y, and a box of width 200.0 in a normalised frame passed the check and
    handed the reader the whole picture.
    """
    if image is None or image.ndim < 2 or not _sane(box):
        return None
    height, width = image.shape[:2]
    x1 = int(max(0.0, min(1.0, box.x)) * width)
    y1 = int(max(0.0, min(1.0, box.y)) * height)
    x2 = int(max(0.0, min(1.0, box.right)) * width)
    y2 = int(max(0.0, min(1.0, box.bottom)) * height)
    if x2 - x1 < MIN_PLATE_PIXELS or y2 - y1 < MIN_PLATE_PIXELS:
        return None
    return image[y1:y2, x1:x2]


#: Fewest pixels on a side for a crop to be worth reading.
#:
#: Twenty, not v1's eight. Eight pixels of plate is four pixels of character
#: and there is no character shape left at that size; the v1 docstring claimed
#: exactly that and then set the number below it.
MIN_PLATE_PIXELS = 20
