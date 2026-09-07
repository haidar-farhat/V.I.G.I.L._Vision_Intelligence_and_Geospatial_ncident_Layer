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

#: The most frames a site may skip between detections.
#:
#: Measured on this machine: every third frame costs 3.0x less detection and
#: moves a track 0.008 box heights from where full-rate detection put it. The
#: measurement was taken on a scene with one slow track, and the lag scales
#: with speed — so the ceiling is well short of where the arithmetic stops
#: working, because past it a running person is being *extrapolated* for a
#: third of a second and the boxes an operator sees are a guess.
MAX_DETECT_EVERY = 5


class DetectionError(ValueError):
    """A watch list or a threshold that would make the site behave absurdly."""


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """What to look for. `None` on either field means "whatever the build does"."""

    labels: frozenset[str] | None = None
    confidence: float | None = None
    #: Run the detector on one frame in this many, and track through the rest.
    #: 1 is every frame.
    detect_every: int = 1
    #: Also run it over crops of the far ground. `None` decides from the
    #: provider: worth it where inference is cheap, ruinous on CPU.
    tile: bool | None = None

    @classmethod
    def from_site(cls, site: dict) -> "DetectionSettings":
        labels = frozenset(site.get("watch_labels") or ()) or None
        tile = site.get("tile_far")
        return cls(labels, site.get("min_confidence"), int(site.get("detect_every") or 1),
                   None if tile is None else bool(tile))

    @classmethod
    def checked(cls, labels, confidence: float | None,
                detect_every: int | None = None,
                tile: bool | None = None) -> "DetectionSettings":
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
        every = 1 if detect_every is None else int(detect_every)
        if not 1 <= every <= MAX_DETECT_EVERY:
            raise DetectionError(
                f"detecting every {every} frames is outside 1-{MAX_DETECT_EVERY}. Past that a "
                f"running person is extrapolated for long enough that the boxes on screen are a "
                f"guess rather than a measurement"
            )
        return cls(frozenset(cleaned) or None, confidence, every, tile)

    def classes(self) -> frozenset[str]:
        """The labels a model is asked for, including the built-in default."""
        return WATCHED_LABELS if self.labels is None else self.labels

    def describe(self) -> str:
        watch = ("the built-in list (" + ", ".join(sorted(WATCHED_LABELS)) + ")" if self.labels is None
                 else ", ".join(sorted(self.labels)))
        sure = "the detector's own threshold" if self.confidence is None else f"{self.confidence:.2f}"
        often = ("" if self.detect_every <= 1
                 else f"; detecting every {self.detect_every} frames and tracking between")
        tiled = {True: "; tiling the far ground", False: "; whole frame only",
                 None: ""}[self.tile]
        return f"watching {watch}; confidence at least {sure}{often}{tiled}"

    def override(self, labels, confidence: float | None,
                 detect_every: int | None = None, tile: bool | None = None) -> "DetectionSettings":
        """This run's flags on top of the stored setting; absent flags keep it."""
        if labels is None and confidence is None and detect_every is None and tile is None:
            return self
        return DetectionSettings.checked(
            self.labels if labels is None else labels,
            self.confidence if confidence is None else confidence,
            self.detect_every if detect_every is None else detect_every,
            self.tile if tile is None else tile,
        )


class DetectorFactory:
    """Makes one detector per analysis thread, all with the same settings.

    A plain object rather than a closure so that what a thread was told to
    look for can be read back off it — the console prints it in the status
    bar, and a test asserts on it instead of on a lambda nobody can inspect.
    """

    def __init__(self, model, settings: DetectionSettings):
        self.model = model
        self.settings = settings

    def tiling_wanted(self, detector) -> bool:
        """Whether to spend `1 + tiles` inferences a frame on this machine.

        The setting when there is one, and otherwise the provider. On a GPU
        the extra passes are affordable — measured at 4.5 ms against 38.5 ms
        for the session alone, so four tiles still cost less than one whole
        frame did on CPU. On CPU a whole frame is already 38 ms, and five
        would put one camera under 6 fps: the recall would be bought with the
        frame rate, which on a security camera is the wrong trade and is not
        made silently.
        """
        if self.settings.tile is not None:
            return self.settings.tile
        provider = (getattr(detector.info, "provider", "") or "").lower()
        return bool(provider) and "cpu" not in provider

    def __call__(self) -> Detector:
        """A detector that reports **everything its model knows**.

        The watch list is not passed down any more. It used to restrict the
        detector's own vocabulary, which meant a trailer, a dog or a ladder
        against a fence was invisible to the tracker, the plan and the map,
        because the one thing that could have seen it had been told not to
        look. The list now governs what raises an event, in `CameraWorker`,
        and everything found is still tracked and drawn.
        """
        detector = detector_for(self.model, classes=None, confidence=self.settings.confidence)
        if self.model is not None and self.tiling_wanted(detector):
            from ..adapters.tiling import TiledDetector

            return TiledDetector(detector)
        return detector

    def watch(self) -> frozenset[str] | None:
        """Labels this site raises events for. `None` means the built-in list."""
        return self.settings.classes()

    def describe(self) -> str:
        if self.model is None:
            return "motion only: it does not classify, so no watch list applies"
        return (f"{self.model.name}, detecting every class it knows; "
                f"{self.settings.describe()}")


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
