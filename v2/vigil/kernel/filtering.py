"""The box filter in NumPy: the readable statement of the model.

`core/src/track.rs` is the same filter and is what actually runs. This exists
for two reasons and neither is "in case Rust breaks":

1. **It is the specification.** The model — what the state is, what the noise
   means, where the constants come from — is easier to read here than in Rust,
   and `tests/test_native.py` holds the two to 1e-9 over long runs. If they
   ever disagree, one of them is wrong and the test says which case found it.
2. **A checkout without a Rust toolchain still tracks.** Correctly, with the
   same numbers, more slowly. That is a legitimate fallback; a *different*
   algorithm that merely runs would not be.

# The state

`[cx, cy, a, h, vcx, vcy, va, vh]`: box centre, aspect ratio `w/h`, height,
and their rates, in normalised image coordinates. `h` is the object's apparent
size and doubles as the scale for every noise term, which is what lets one set
of constants cover a person at 5 m and the same person at 40 m.

# The noise, and where the numbers come from

Continuous white-noise acceleration, discretised exactly, so a dropped frame
widens the gate by the right amount instead of by whatever the frame counter
did:

    Q_block = q * [[dt^3/3, dt^2/2],
                   [dt^2/2, dt    ]]

`q` for the centre is `ACCEL_PSD_CENTRE * h^2`, and that constant carries the
derivation - including the units error the first version of it made.

# What this replaced

v1's Rust tracker and v2's Python port both carried velocity as a 60/40
exponential moving average of frame-to-frame displacement. That has no
uncertainty to gate on, no way to tell detector jitter from motion — so a
stationary object had a wandering velocity — and a time constant measured in
*frames*, so the smoothing changed silently whenever the camera stuttered,
which is exactly when tracking is hard.
"""

from __future__ import annotations

import numpy as np

#: Acceleration **power spectral density** for the box centre, in squared
#: object-heights per second cubed. Note the units: this is a PSD, not an
#: acceleration, and conflating the two is the error this constant was rewritten
#: to fix.
#:
#: The first version of this filter set q = (3h)^2, reasoning that a 1.7 m person
#: can accelerate at about 5 m/s^2 and that h is inversely proportional to range,
#: so image-space acceleration is 5h/1.7 ~ 3h. The arithmetic is right and the
#: substitution is not: in a continuous white-noise-acceleration model the
#: velocity variance grows as q*dt, so q = (3h)^2 lets the estimated velocity
#: wander by 3h*sqrt(dt) in one frame - 0.155 frames per second at 15 fps - while
#: the largest change a person can physically produce in that frame is 3h*dt,
#: which is 0.039. The filter was four times more willing to believe in
#: acceleration than acceleration exists, so it chased detector jitter and gave
#: parked cars a velocity.
#:
#: A PSD needs an acceleration *and a time over which it is held*:
#: q = (a/H)^2 * tau. A pedestrian sustains about 1 m/s^2 for about half a second
#: when starting, stopping or turning, which over a 1.7 m frame gives 0.17. A
#: 5 m/s^2 sprint start is real, lasts a fraction of a second, and arrives as an
#: innovation the chi-squared gate still accepts - which is where a brief
#: manoeuvre belongs.
ACCEL_PSD_CENTRE = 0.18
#: The same for apparent height, which changes only with range - second order
#: in the speed the centre term already covers.
ACCEL_PSD_SCALE = 0.02
#: The same for aspect ratio, absolute rather than scaled by h: a person
#: turning changes it by about 0.3 over about a second.
ACCEL_PSD_ASPECT = 0.09
#: Detector box jitter, 1-sigma, as a fraction of box height.
MEASURE_STD_FRACTION = 0.05
#: Jitter of the aspect ratio, absolute.
MEASURE_STD_ASPECT = 0.1

#: Chi-squared 0.95 quantile at 4 degrees of freedom.
CHI2_GATE_4DOF = 9.4877
#: The same at 2, for gating on centre position alone.
CHI2_GATE_2DOF = 5.9915

#: Length of the flat state array: 8 mean values then a flattened 8x8.
STATE_VALUES = 72

_N = 8
_M = 4


def _unpack(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return state[:_N], state[_N:].reshape(_N, _N)


def initiate(state: np.ndarray, cx: float, cy: float, aspect: float, height: float) -> np.ndarray:
    """Start a track from one measurement, in place.

    Velocity starts at zero with a *wide* covariance rather than at a guess:
    one box says nothing about speed, and a filter told otherwise spends its
    first second chasing an invented velocity.
    """
    state[:] = 0.0
    mean, covariance = _unpack(state)
    mean[:4] = (cx, cy, aspect, height)
    position = 2.0 * MEASURE_STD_FRACTION * height
    velocity = 10.0 * MEASURE_STD_FRACTION * height
    std = np.array([
        position, position, 2.0 * MEASURE_STD_ASPECT, position,
        velocity, velocity, 10.0 * MEASURE_STD_ASPECT, velocity,
    ])
    covariance[np.diag_indices(_N)] = std ** 2
    return state


def _transition(dt: float) -> np.ndarray:
    f = np.eye(_N)
    f[np.arange(_M), np.arange(_M) + _M] = dt
    return f


def _process_noise(height: float, dt: float) -> np.ndarray:
    h = max(abs(height), 1e-4)
    q = np.array([
        ACCEL_PSD_CENTRE * h * h,
        ACCEL_PSD_CENTRE * h * h,
        ACCEL_PSD_ASPECT,
        ACCEL_PSD_SCALE * h * h,
    ])
    noise = np.zeros((_N, _N))
    index = np.arange(_M)
    noise[index, index] = q * dt ** 3 / 3.0
    noise[index, index + _M] = q * dt ** 2 / 2.0
    noise[index + _M, index] = q * dt ** 2 / 2.0
    noise[index + _M, index + _M] = q * dt
    return noise


def predict(state: np.ndarray, dt_seconds: float) -> None:
    """Advance by `dt` seconds. A non-positive or non-finite `dt` is a no-op
    rather than an error: two detections carrying the same timestamp is a
    decoder quirk, not a reason to corrupt a filter."""
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        return
    mean, covariance = _unpack(state)
    height = mean[3]
    f = _transition(dt_seconds)
    mean[:] = f @ mean
    covariance[:] = f @ covariance @ f.T + _process_noise(height, dt_seconds)
    _symmetrise(covariance)


def _measurement_noise(height: float) -> np.ndarray:
    h = max(abs(height), 1e-4)
    return np.diag([
        (MEASURE_STD_FRACTION * h) ** 2,
        (MEASURE_STD_FRACTION * h) ** 2,
        MEASURE_STD_ASPECT ** 2,
        (MEASURE_STD_FRACTION * h) ** 2,
    ])


def project(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The measurement-space mean and its innovation covariance `S`."""
    mean, covariance = _unpack(state)
    return mean[:_M].copy(), covariance[:_M, :_M] + _measurement_noise(mean[3])


def update(state: np.ndarray, cx: float, cy: float, aspect: float, height: float) -> bool:
    """Fold in a measured box. False when `S` is not positive definite, which
    means the filter has been fed something degenerate and the caller should
    restart the track rather than carry on with a poisoned state."""
    mean, covariance = _unpack(state)
    projected, s = project(state)
    try:
        # Cholesky purely as the definiteness test; the gain goes through the
        # general solve, which is what the Rust side does with its factor.
        np.linalg.cholesky(s)
    except np.linalg.LinAlgError:
        return False
    gain = np.linalg.solve(s, covariance[:, :_M].T).T
    innovation = np.array([cx, cy, aspect, height]) - projected
    mean += gain @ innovation
    covariance -= gain @ s @ gain.T
    _symmetrise(covariance)
    # Height and aspect are positive by construction; a run of bad boxes can
    # drive them negative, and a negative height turns every downstream IoU
    # into nonsense.
    mean[2] = max(mean[2], 1e-4)
    mean[3] = max(mean[3], 1e-4)
    return True


def gate(state: np.ndarray, boxes: np.ndarray, position_only: bool = False) -> np.ndarray:
    """Squared Mahalanobis distance from this track to each measured box.

    `position_only` gates on the centre alone, which is the right test when a
    detector's box size is unreliable: a half-occluded person has a correct
    centre and a badly wrong height.
    """
    measurements = np.reshape(np.asarray(boxes, dtype=np.float64), (-1, 4))
    projected, s = project(state)
    if position_only:
        sub = s[:2, :2]
        delta = measurements[:, :2] - projected[:2]
        determinant = sub[0, 0] * sub[1, 1] - sub[0, 1] * sub[1, 0]
        if not abs(determinant) > 1e-18:
            return np.full(len(measurements), np.inf)
        inverse = np.array([[sub[1, 1], -sub[0, 1]], [-sub[1, 0], sub[0, 0]]]) / determinant
        return np.einsum("ij,jk,ik->i", delta, inverse, delta)
    try:
        chol = np.linalg.cholesky(s)
    except np.linalg.LinAlgError:
        return np.full(len(measurements), np.inf)
    delta = measurements - projected
    z = np.linalg.solve_triangular(chol, delta.T, lower=True) if hasattr(np.linalg, "solve_triangular") \
        else _forward_substitute(chol, delta.T)
    return np.sum(z ** 2, axis=0)


def _forward_substitute(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve `L z = rhs` for a lower-triangular `L`.

    NumPy has no triangular solve of its own — that is SciPy — and a general
    `solve` on a 4x4 is both slower and less accurate than four subtractions.
    """
    z = np.zeros_like(rhs)
    for i in range(lower.shape[0]):
        z[i] = (rhs[i] - lower[i, :i] @ z[:i]) / lower[i, i]
    return z


def warp(state: np.ndarray, affine: np.ndarray) -> bool:
    """Apply a 2x3 affine `[a, b, tx, c, d, ty]` for camera motion.

    The mean moves through the affine and the covariance is rotated by its
    linear part, so an estimate that was uncertain along one axis stays
    uncertain along that axis after the camera turns. A tracker that moves the
    boxes and leaves the covariance alone claims a precision the motion
    estimate does not have.
    """
    values = np.asarray(affine, dtype=np.float64).reshape(-1)[:6]
    if not np.isfinite(values).all():
        return False
    mean, covariance = _unpack(state)
    linear = values[[0, 1, 3, 4]].reshape(2, 2)
    translation = values[[2, 5]]
    mean[:2] = linear @ mean[:2] + translation
    mean[4:6] = linear @ mean[4:6]
    # Scale: the geometric mean of the affine's singular values. Anisotropic
    # scaling is not representable in this state, and a camera warp anisotropic
    # enough to matter is not a camera warp, it is a bad estimate.
    scale = np.sqrt(abs(np.linalg.det(linear)))
    if np.isfinite(scale) and scale > 1e-6:
        mean[3] *= scale
        mean[7] *= scale
    for block in (0, 4):
        rows = slice(block, block + 2)
        covariance[rows, :] = linear @ covariance[rows, :]
        covariance[:, rows] = covariance[:, rows] @ linear.T
    _symmetrise(covariance)
    return True


def _symmetrise(covariance: np.ndarray) -> None:
    """Force exact symmetry. Repeated rank updates drift out of symmetry by a
    few ulp per step, and a Cholesky of a matrix that is not quite symmetric
    fails in a way that looks like a modelling bug."""
    covariance += covariance.T
    covariance *= 0.5


def box_xywh(state: np.ndarray) -> tuple[float, float, float, float]:
    """The filtered box as `(x, y, width, height)`, top-left origin."""
    cx, cy, aspect, height = state[:4]
    width = aspect * height
    return float(cx - width / 2), float(cy - height / 2), float(width), float(height)


def to_measurement(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    """A top-left box as the filter's `(cx, cy, aspect, height)`."""
    h = max(height, 1e-4)
    return x + width / 2, y + h / 2, width / h, h
