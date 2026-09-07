"""How the camera itself moved between two frames.

# What this is for

Nothing in v1 or v2 estimated camera motion, and three things went wrong
because of it.

**The tracker read camera motion as object motion.** A gust on a mast, a
lorry passing a pole, an operator nudging a PTZ: every box in the frame
translates at once, and a tracker with no warp sees every object accelerate
together. Association survives a small shift by luck — the gate is generous —
and stops surviving it exactly when the shift is large enough to matter.

**A map cannot be built from a camera that has moved.** The ground sample
assumes the pose it was given. A camera knocked 3 degrees off its stored
heading puts every pixel it contributes 2 m out at 40 m, silently, until
somebody notices the map has two kerbs.

**Nobody could tell a moving camera from a busy scene.** Both look like "a lot
of pixels changed", which is why v2's motion detector reports a curtain and a
person with equal conviction.

# How

Sparse optical flow: corners from `goodFeaturesToTrack`, tracked forward with
Lucas-Kanade pyramids, and a partial affine (translation, rotation, uniform
scale) fitted through RANSAC. Partial rather than full affine deliberately —
a camera on a mount cannot shear, so allowing shear only lets the estimate
absorb the scene's own motion and call it the camera's.

The forward-backward check is what separates this from a flow estimate that
merely runs: every point is tracked forward to the next frame and back again,
and a point that does not return to within a pixel of where it started was not
actually tracked. Without it, a frame of moving foliage produces a confident
warp built from points that matched nothing.

# What it refuses to answer

A frame with too few corners, too few surviving matches, or too few RANSAC
inliers produces `measured=False` and an identity warp. That is not a
fallback dressed as a measurement: callers check `measured` before using the
warp for anything that matters, and the identity is there so that a caller
which only wants to compose transforms does not have to special-case `None`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Longest edge the estimate runs at. Camera motion is a global, low-frequency
#: signal; a quarter-size frame carries all of it and costs a sixteenth as
#: much. Measured at 1.1 ms for a 1080p frame reduced to this.
WORKING_EDGE = 480

#: Corners asked for. More than this stops improving the fit and starts
#: picking up the scene.
MAX_CORNERS = 200
#: Minimum spacing between corners, in working-resolution pixels. Spread out,
#: so the fit is constrained across the frame rather than by one textured
#: corner of it.
MIN_CORNER_DISTANCE = 12

#: Forward-backward reprojection error, in working pixels, past which a point
#: was not tracked. One pixel: at this resolution that is a quarter of a pixel
#: of real motion, and anything looser admits points that drifted onto a
#: different feature.
MAX_ROUND_TRIP_PX = 1.0

#: Fewest inliers for an answer. Three points determine a partial affine; ten
#: is enough that the fit is over-determined and a couple of bad ones cannot
#: carry it.
MIN_INLIERS = 10

#: Inlier share below which the fit describes a minority of the frame — a
#: lorry crossing it, not the camera moving.
MIN_INLIER_RATIO = 0.5

#: Translation, as a fraction of the frame, past which this is not camera
#: shake but a cut, a stream reconnect, or a PTZ slew. Reported rather than
#: applied: warping tracks by half a frame is worse than admitting the
#: tracker has lost its scene.
MAX_PLAUSIBLE_SHIFT = 0.25

_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True, slots=True)
class CameraMotion:
    """A 2x3 affine in **normalised** image coordinates, and its confidence.

    Normalised rather than pixels because everything downstream — boxes,
    tracks, the filter's covariance — is normalised, and a warp in pixels
    silently stops being right when the stream changes resolution on a
    reconnect.
    """

    warp: np.ndarray
    measured: bool
    inliers: int
    tracked: int
    #: Reason the estimate was refused, when it was.
    fault: str | None = None
    #: Frame width over height.
    #:
    #: Needed to *report* the warp, not to apply it. Normalising by width and
    #: height is a non-uniform scaling, so a pixel-space rotation is no longer
    #: a rotation matrix once normalised — its off-diagonals come out scaled
    #: by the aspect ratio and no longer negatives of each other. Applying the
    #: warp to normalised boxes is still exactly right; reading an angle off
    #: it without undoing the aspect is not, and did so by 77% on a 16:9 frame
    #: before this field existed.
    aspect: float = 1.0

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.tracked if self.tracked else 0.0

    @property
    def shift(self) -> tuple[float, float]:
        """Translation as a fraction of the frame."""
        return float(self.warp[0, 2]), float(self.warp[1, 2])

    @property
    def magnitude(self) -> float:
        return float(np.hypot(*self.shift))

    @property
    def rotation_degrees(self) -> float:
        """Rotation in the frame's own pixels, positive anticlockwise.

        Taken from the de-normalised linear part; see `aspect`.
        """
        return float(np.degrees(np.arctan2(self.warp[1, 0] / max(self.aspect, 1e-9),
                                           self.warp[0, 0])))

    @property
    def scale(self) -> float:
        """Uniform scale factor.

        The determinant is invariant under the aspect normalisation — it is a
        similarity transform — so this one needs no correction.
        """
        return float(np.sqrt(abs(np.linalg.det(self.warp[:, :2]))))

    @property
    def still(self) -> bool:
        """True when the camera measurably did not move.

        A quarter of a percent of the frame and a tenth of a degree: below
        that the estimate is reporting its own noise, and calling that motion
        makes a fixed camera look like it is drifting.
        """
        return (self.measured and self.magnitude < 0.0025
                and abs(self.rotation_degrees) < 0.1 and abs(self.scale - 1.0) < 0.002)

    @classmethod
    def unmeasured(cls, fault: str, tracked: int = 0, inliers: int = 0) -> "CameraMotion":
        return cls(_IDENTITY.copy(), False, inliers, tracked, fault)


class CameraMotionEstimator:
    """Frame-to-frame camera motion. One per camera; not thread-safe.

    Keeps the previous working-resolution grey frame, so a caller feeds frames
    in and gets motion out without managing history.
    """

    def __init__(self, *, working_edge: int = WORKING_EDGE, max_corners: int = MAX_CORNERS):
        self._working_edge = working_edge
        self._max_corners = max_corners
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous frame. Called on a reconnect: the frame before
        a stream dropped is not the frame before the one after it."""
        self._previous = None

    def estimate(self, frame: np.ndarray) -> CameraMotion:
        """Motion from the previous frame to this one."""
        grey = self._prepare(frame)
        previous, self._previous = self._previous, grey
        if previous is None:
            return CameraMotion.unmeasured("no previous frame")
        if previous.shape != grey.shape:
            # A resolution change mid-stream. Comparing across it would produce
            # a scale factor that is an artefact of the decoder.
            return CameraMotion.unmeasured("the frame size changed")
        return self._between(previous, grey)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        longest = max(grey.shape[:2])
        if longest > self._working_edge:
            scale = self._working_edge / longest
            grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(grey)

    def _between(self, previous: np.ndarray, current: np.ndarray) -> CameraMotion:
        height, width = previous.shape[:2]
        corners = cv2.goodFeaturesToTrack(
            previous, maxCorners=self._max_corners, qualityLevel=0.01,
            minDistance=MIN_CORNER_DISTANCE, blockSize=7,
        )
        if corners is None or len(corners) < MIN_INLIERS:
            # A blank wall, a black frame, fog. Not a failure of the estimator:
            # a frame with no features genuinely does not say where it moved.
            return CameraMotion.unmeasured("too few corners to track", 0 if corners is None else len(corners))

        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
        if forward is None:
            return CameraMotion.unmeasured("optical flow failed", len(corners))
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None)
        if backward is None:
            return CameraMotion.unmeasured("optical flow failed on the return pass", len(corners))

        # The forward-backward check: a point that does not come back to where
        # it started was not tracked, it was guessed.
        round_trip = np.linalg.norm(corners - backward, axis=2).reshape(-1)
        good = (
            (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
            & (round_trip < MAX_ROUND_TRIP_PX)
        )
        tracked = int(good.sum())
        if tracked < MIN_INLIERS:
            return CameraMotion.unmeasured("too few points survived the round trip", tracked)

        source = corners.reshape(-1, 2)[good]
        target = forward.reshape(-1, 2)[good]
        # Partial affine: translation, rotation, uniform scale. A camera on a
        # mount cannot shear, and allowing shear lets the fit absorb the
        # scene's motion and report it as the camera's.
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=2000, confidence=0.99,
        )
        if matrix is None or inlier_mask is None:
            return CameraMotion.unmeasured("no consistent transform", tracked)
        inliers = int(inlier_mask.sum())
        if inliers < MIN_INLIERS:
            return CameraMotion.unmeasured("too few inliers", tracked, inliers)
        if inliers / tracked < MIN_INLIER_RATIO:
            # A minority of the frame moved together. That is a lorry crossing
            # it, not the camera turning, and applying it would drag every
            # track sideways after the lorry.
            return CameraMotion.unmeasured("the transform explains a minority of the frame", tracked, inliers)

        warp = _to_normalised(matrix, width, height)
        motion = CameraMotion(warp, True, inliers, tracked, None, width / max(1, height))
        if motion.magnitude > MAX_PLAUSIBLE_SHIFT:
            return CameraMotion.unmeasured(
                f"the scene moved {motion.magnitude:.0%} of a frame, which is a cut or a slew, not shake",
                tracked, inliers,
            )
        if not np.isfinite(warp).all():
            return CameraMotion.unmeasured("the transform was not finite", tracked, inliers)
        return motion


def _to_normalised(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel affine to normalised affine.

    `S A S^-1` with `S = diag(1/w, 1/h)`: the linear part's off-diagonals pick
    up the aspect ratio and the translation is divided through. Getting this
    wrong is invisible on a square frame and wrong by the aspect ratio on
    every real one.
    """
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    return np.array([
        [a, b * height / width, tx / width],
        [c * width / height, d, ty / height],
    ], dtype=np.float64)
