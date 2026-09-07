"""Whether a frame is worth believing.

# Why a detector's confidence is not enough

A detector reports how sure it is *given the pixels it was shown*. It has no
way to say "these pixels are out of focus", "this frame is a wall of infrared
glare", or "the stream is handing me the same frame it handed me a second
ago". So a camera that has gone bad produces confident detections of nothing,
or no detections at all with no explanation, and both look identical to a
camera that is working and watching an empty yard.

v1 and v2 both had `CAMERA_DARK` — an alert for a camera producing no frames
at all — and nothing at all between that and "working". Every failure in
between was silent:

- **Out of focus.** A lens that has drifted, or a spider's web across it. The
  detector recalls almost nothing and reports it as an empty scene.
- **Blown out or crushed.** A low-sun glare or a failed IR cut filter. Half
  the frame carries no information at all.
- **Frozen.** Some IP cameras repeat the last frame indefinitely rather than
  dropping the connection, so the decoder is happy, the frame counter climbs,
  and the picture is an hour old.

Each of these is measurable in under a millisecond and none of them was being
measured.

# What the numbers mean

Every measure is scaled to `0..1` where 1 is good, and each carries the raw
figure it came from so an operator can argue with the threshold rather than
just seeing a verdict. `usable` is the conjunction, and `faults` says which
ones failed — a frame is not "70% good", it is sharp and badly exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Longest edge the measures run at. Sharpness and exposure are properties of
#: the image, not of its resolution, and measuring them on a full 4K frame
#: costs twenty times as much for the same answer.
WORKING_EDGE = 320

#: Variance of the Laplacian, on a 0..255 grey image at the working size,
#: below which a frame is out of focus.
#:
#: The absolute value is scene-dependent — a blank wall in perfect focus
#: scores low — which is why this is deliberately a *floor* rather than a
#: target, and why `sharpness_trend` on the run matters more than any single
#: frame. 40 is roughly where a 1080p security frame stops resolving a face at
#: 10 m.
BLUR_FLOOR = 40.0

#: Share of pixels at the very top or bottom of the range past which the frame
#: has lost information rather than merely being contrasty. 15% of a frame
#: being pure white is a glare, not a bright day.
CLIPPING_LIMIT = 0.15

#: Standard deviation of luminance below which there is nothing in the frame
#: to detect: a lens cap, a whiteout, a black night with the IR off.
MIN_CONTRAST = 8.0

#: Mean absolute difference from the previous frame below which the stream is
#: repeating itself. Not zero: a static scene from a real sensor still has
#: read noise, and a camera whose output is *bit-identical* between frames is
#: not a still scene, it is a frozen decoder.
FROZEN_DIFFERENCE = 0.35


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """What one frame is worth, with the raw numbers behind each verdict."""

    #: Variance of the Laplacian. Higher is sharper.
    sharpness: float
    #: Share of pixels clipped to 0 or 255.
    clipped: float
    #: Standard deviation of luminance, 0..255.
    contrast: float
    #: Mean luminance, 0..255.
    brightness: float
    #: Mean absolute difference from the previous frame, or `None` for the
    #: first frame of a run.
    change: float | None
    #: Why the *image* is unusable: blur, exposure, no contrast.
    faults: tuple[str, ...] = ()
    #: This frame is a repeat of the last one.
    #:
    #: Separate from `faults` because it is a fault of the *stream*, not of
    #: the image, and the two callers want different things. Detection should
    #: still run on a repeated frame — the picture may be an hour old, but it
    #: is a picture. A map must not sample it: a repeat adds no information to
    #: a median and only fills the ring with one moment.
    stale: bool = False

    @property
    def usable(self) -> bool:
        """The image is worth running a detector over."""
        return not self.faults

    @property
    def worth_sampling(self) -> bool:
        """The frame adds something a median did not already have."""
        return not self.faults and not self.stale

    @property
    def degraded(self) -> str | None:
        """One sentence for a health check, or `None` when the camera is fine."""
        if self.faults:
            return "; ".join(self.faults)
        if self.stale:
            return (f"the stream is repeating itself (frame-to-frame change {self.change:.2f}); "
                    "some cameras hold the last frame rather than dropping the connection")
        return None

    @property
    def score(self) -> float:
        """A single 0..1 number, for ranking frames rather than judging them.

        Deliberately not what `usable` is built on: a frame is not 70% good,
        it is sharp and badly exposed, and collapsing that into one number is
        how a specific fixable fault becomes a vague complaint. This exists so
        a map builder can prefer the best of the frames it has.
        """
        if self.faults or self.stale:
            return 0.0
        focus = min(1.0, self.sharpness / (BLUR_FLOOR * 4))
        exposure = 1.0 - min(1.0, self.clipped / CLIPPING_LIMIT)
        detail = min(1.0, self.contrast / (MIN_CONTRAST * 6))
        return float(focus * 0.5 + exposure * 0.25 + detail * 0.25)

    def describe(self) -> str:
        degraded = self.degraded
        if degraded is not None:
            return degraded
        return f"sharpness {self.sharpness:.0f}, contrast {self.contrast:.0f}, {self.clipped:.1%} clipped"


class FrameQualityMonitor:
    """Rolling frame quality for one camera. Not thread-safe; one owner.

    Stateful only because "frozen" needs the previous frame. Everything else
    is a property of the frame in hand.
    """

    def __init__(self, *, working_edge: int = WORKING_EDGE):
        self._working_edge = working_edge
        self._previous: np.ndarray | None = None
        self._recent: list[float] = []

    def reset(self) -> None:
        self._previous = None
        self._recent.clear()

    def measure(self, frame: np.ndarray) -> FrameQuality:
        grey = self._prepare(frame)
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        histogram = cv2.calcHist([grey], [0], None, [256], [0, 256]).reshape(-1)
        total = float(grey.size) or 1.0
        clipped = float(histogram[0] + histogram[255]) / total
        contrast = float(grey.std())
        brightness = float(grey.mean())

        change: float | None = None
        if self._previous is not None and self._previous.shape == grey.shape:
            change = float(np.mean(cv2.absdiff(grey, self._previous)))
        self._previous = grey
        stale = change is not None and change < FROZEN_DIFFERENCE

        faults: list[str] = []
        if sharpness < BLUR_FLOOR:
            faults.append(f"out of focus or badly blurred (sharpness {sharpness:.0f}, floor {BLUR_FLOOR:.0f})")
        if clipped > CLIPPING_LIMIT:
            where = "blown out" if histogram[255] > histogram[0] else "crushed to black"
            faults.append(f"{where}: {clipped:.0%} of pixels carry no information")
        if contrast < MIN_CONTRAST:
            faults.append(f"nothing in the frame to detect (contrast {contrast:.1f})")
        quality = FrameQuality(sharpness, clipped, contrast, brightness, change,
                               tuple(faults), stale)
        self._recent.append(quality.score)
        if len(self._recent) > 120:
            del self._recent[0]
        return quality

    @property
    def recent_score(self) -> float | None:
        """Mean score over the last few seconds, or `None` before there is one.

        What a health check should read rather than the latest frame: one bad
        frame is a bad frame, and a hundred is a broken camera.
        """
        return float(np.mean(self._recent)) if self._recent else None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        longest = max(grey.shape[:2])
        if longest > self._working_edge:
            scale = self._working_edge / longest
            grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(grey)
