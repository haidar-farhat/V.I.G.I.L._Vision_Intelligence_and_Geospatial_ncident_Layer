"""What a site watches for, and how sure the detector has to be.

Both were command-line flags, which meant they existed only for as long as
somebody was standing at a keyboard: a service started at boot, or a console
opened by double-click, analysed with the built-in list while the operator
believed the choice they had typed once still applied. They are settings
now, kept with the site, and a flag overrides them for that one run.

The factory lives here rather than in each interface because there were two
copies of it, one in the console and one in the command line, and two copies
of a decision are two decisions waiting to differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..adapters.detectors import WATCHED_LABELS, Detector, detector_for, model_info
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: A label is a word from a model's class list, so it is deliberately narrow.
_LABEL = re.compile(r"^[a-z0-9][a-z0-9 _\-]{0,39}$")

#: Below this a detector reports mostly noise, above it mostly nothing. The
#: bounds refuse the two settings that make a site look broken.
CONFIDENCE_FLOOR, CONFIDENCE_CEILING = 0.05, 0.95


class DetectionError(ValueError):
    """A watch list or a threshold that would make the site behave absurdly."""


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """What to look for. `None` on either field means "whatever the build does"."""

    labels: frozenset[str] | None = None
    confidence: float | None = None

    @classmethod
    def from_site(cls, site: dict) -> "DetectionSettings":
        labels = frozenset(site.get("watch_labels") or ()) or None
        return cls(labels, site.get("min_confidence"))

    @classmethod
    def checked(cls, labels, confidence: float | None) -> "DetectionSettings":
        """The settings, or a `DetectionError` saying which value is wrong.

        Checked when it is typed. A watch list nobody can satisfy is only
        visible as an empty screen, hours later, and reads as a broken camera.
        """
        cleaned = {str(l).strip().lower() for l in (labels or ()) if str(l).strip()}
        for label in sorted(cleaned):
            if not _LABEL.match(label):
                raise DetectionError(f"{label!r} is not a label (lower-case words, digits, spaces, - and _)")
        if confidence is not None:
            if not CONFIDENCE_FLOOR <= float(confidence) <= CONFIDENCE_CEILING:
                raise DetectionError(f"a confidence of {confidence} is outside {CONFIDENCE_FLOOR}–"
                                     f"{CONFIDENCE_CEILING}; below that a site reports noise, above it nothing")
            confidence = float(confidence)
        return cls(frozenset(cleaned) or None, confidence)

    def classes(self) -> frozenset[str]:
        """The labels a model is asked for, including the built-in default."""
        return WATCHED_LABELS if self.labels is None else self.labels

    def describe(self) -> str:
        watch = ("the built-in list (" + ", ".join(sorted(WATCHED_LABELS)) + ")" if self.labels is None
                 else ", ".join(sorted(self.labels)))
        sure = "the detector's own threshold" if self.confidence is None else f"{self.confidence:.2f}"
        return f"watching {watch}; confidence at least {sure}"

    def override(self, labels, confidence: float | None) -> "DetectionSettings":
        """This run's flags on top of the stored setting; absent flags keep it."""
        if labels is None and confidence is None:
            return self
        return DetectionSettings.checked(self.labels if labels is None else labels,
                                         self.confidence if confidence is None else confidence)


class DetectorFactory:
    """Makes one detector per analysis thread, all with the same settings.

    A plain object rather than a closure so that what a thread was told to
    look for can be read back off it — the console prints it in the status
    bar, and a test asserts on it instead of on a lambda nobody can inspect.
    """

    def __init__(self, model, settings: DetectionSettings):
        self.model = model
        self.settings = settings

    def __call__(self) -> Detector:
        classes = self.settings.classes() if self.model is not None else None
        return detector_for(self.model, classes=classes, confidence=self.settings.confidence)

    def describe(self) -> str:
        if self.model is None:
            return "motion only: it does not classify, so no watch list applies"
        return f"{self.model.name}, {self.settings.describe()}"


def detector_factory(model, settings: DetectionSettings) -> DetectorFactory:
    """A factory whose watch list this model can actually satisfy.

    The model is opened once, here, before any thread exists: a label the
    model cannot produce is an error at start-up rather than a site that
    quietly sees nothing all night.
    """
    if model is not None:
        model_info(model, classes=settings.classes())
    _log.info("detection: %s", DetectorFactory(model, settings).describe())
    return DetectorFactory(model, settings)
