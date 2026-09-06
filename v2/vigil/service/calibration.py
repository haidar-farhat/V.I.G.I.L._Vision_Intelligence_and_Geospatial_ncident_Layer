"""Measuring a camera's pose instead of assuming it.

# Why this exists

`PoseUncertainty`'s defaults — plus or minus two degrees of heading, 0.15 m of
mount height — are what an operator with a compass and a tape is worth. They
are honest, they are stated, and **they dominate every position this system
produces at range**: at 40 m, two degrees of heading is 1.4 m of sideways
error before the detector has contributed anything at all.

They are also assumptions. This module replaces them with a measurement, from
the only evidence available on a real site: somebody stands at a few places
that can be found on a map, and points at them in the picture.

# What is solved, and against what

Levenberg-Marquardt over the pose, minimising **reprojection error** — where
each known ground point *ought* to appear in the image against where it was
marked. Reprojection rather than ground-plane error for a specific reason: the
forward projection refuses rays at the horizon and past the range, so an
optimiser walking through a bad intermediate pose would hit a wall of `None`
and lose its gradient. `image_coordinates` has no such cliff; a point behind
the camera is the only failure and it is a real one.

The parameters are heading, pitch, roll and mount height. **Position is not
solved by default**: an operator's click on a map is worth about a metre, the
orientation is worth two degrees, and at 40 m the second is worth more than
the first — so solving both from a handful of points spends the geometry on
the term that matters less. `solve_position=True` is there for a site with
surveyed control points, where it is the right thing to do.

# The number this is really for

Not the refined pose — the **covariance**. `sigma^2 (J^T J)^-1` is what the
residuals and the geometry together say about how well each parameter is
known, and it is what replaces the assumption. A calibration from four points
in a tight cluster produces a *large* covariance, and that is the correct
answer: it says the measurement did not pin the pose down, which is far more
useful than a confident wrong one.

# When it refuses

- Fewer than four points: four parameters need more than four constraints, and
  eight residuals from four points is already thin.
- A **numerically** degenerate arrangement, on the condition number of `J^T J`.
  Measured, because the intuition here is wrong: points down a single image
  column are *not* degenerate (condition 1.6e4 on the test scene), because
  heading and roll move a near point and a far point by different amounts even
  when the two lie on one ray. What is actually degenerate is a **tight
  cluster** — points spanning 2% of the frame condition at 1e6, 0.2% at 9.6e7,
  0.02% at 9.5e9.
- A result *worse* than the assumption it would replace. A calibration that
  makes the pose less certain is not a calibration.

Those two refusals do different jobs and the division matters. The condition
number only catches a fit whose covariance is *arithmetically* meaningless —
an inverse dominated by rounding error. A fit that is merely **poor** passes it
and is caught by `better_than()`, which compares the measured uncertainty with
the assumption it would replace. Tightening `MAX_CONDITION` to do the second
job would refuse fits that `better_than` handles better, with a worse message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..domain.geo import (
    CameraPose, LatLon, LocalFrame, PoseUncertainty, Vec2, image_coordinates,
)
from ..domain.lens import Distortion
from ..logs import get as _get_logger

_log = _get_logger(__name__)

#: Fewest correspondences. Four parameters against eight residuals is the
#: thinnest thing worth calling a fit; six or more is where the covariance
#: starts to mean something.
MIN_POINTS = 4

#: Condition number of `J^T J` past which inverting it produces a covariance
#: made of rounding error rather than of information.
#:
#: Doubles as the floor: float64 carries ~16 digits, so a condition of 1e8
#: leaves about 8, which is where a variance stops being worth quoting. On the
#: test scene this admits any layout spanning more than roughly 0.2% of the
#: frame and refuses anything tighter. Fits that are poor but arithmetically
#: sound pass here and are stopped by `Calibration.better_than` instead.
MAX_CONDITION = 1e8

#: Levenberg-Marquardt iteration cap. Convergence is usually in under ten;
#: this is the guard against a pathological input, not a working limit.
MAX_ITERATIONS = 60

#: Relative improvement below which the fit has stopped moving.
CONVERGENCE = 1e-10

#: Steps for the numerical Jacobian, per parameter, in that parameter's units.
#: Central differences; the same reasoning as `geo`'s uncertainty Jacobian,
#: and the same reason for doing it numerically — six hand-derived partials
#: through a rotation composition stay wrong for a year without anybody
#: noticing.
_STEPS = {"heading": 1e-4, "pitch": 1e-4, "roll": 1e-4, "mount_height": 1e-4,
          "east": 1e-4, "north": 1e-4}

#: Position is solved as **metres east and north** of where the camera was
#: said to be, never as latitude and longitude.
#:
#: Not presentation — correctness. A degree of latitude is 111 km, so a lat/lon
#: column of the Jacobian is ~1e5 times the size of a heading column, `J^T J`
#: is ~1e10 out, and its condition number then measures the *units* rather than
#: whether the data separates the parameters. Measured on the test scene: the
#: same well-spread points condition at 2.2e3 in metres and 3.3e13 in degrees,
#: so the degeneracy check refused a perfectly good fit and `solve_position`
#: could never once have succeeded.
_POSITION = ("east", "north")


class CalibrationError(ValueError):
    """The data cannot pin down a pose, and says which way it fell short."""


@dataclass(frozen=True, slots=True)
class Correspondence:
    """A point in the picture, and where it is on the ground.

    `image` is normalised, `(0, 0)` top-left, the same coordinates a detection
    uses. `ground` is somewhere an operator can find on a map or a survey.
    """

    image: Vec2
    ground: LatLon
    #: What the operator was pointing at, for the report. Never used in the fit.
    label: str = ""


@dataclass(frozen=True, slots=True)
class Calibration:
    """A measured pose, and what the measurement is worth."""

    pose: CameraPose
    #: Per-point reprojection error, in fractions of the frame.
    residuals: tuple[float, ...]
    #: RMS of those, the single number for "how well does this pose explain
    #: the points it was fitted to".
    rms: float
    #: The worst single point. A good RMS with one terrible point is a
    #: mis-clicked correspondence, not a bad pose, and the two need telling
    #: apart.
    worst: float
    #: Which point that was.
    worst_index: int
    iterations: int
    #: How well the *data* separated the parameters. Large means the geometry
    #: was poor even if the residuals look small.
    condition: float
    #: 1-sigma of the camera's own position, in metres, when it was solved for.
    #: `None` when the position was taken as given, which is the default. It
    #: has no home on `PoseUncertainty`, which describes orientation.
    position_error_m: float | None = None

    @property
    def uncertainty(self) -> PoseUncertainty:
        return self.pose.uncertainty

    def describe(self) -> str:
        u = self.uncertainty
        lines = [
            f"reprojection {self.rms * 100:.2f}% of the frame RMS, worst {self.worst * 100:.2f}% "
            f"at point {self.worst_index + 1}, {self.iterations} iterations",
            f"  heading {self.pose.heading:.2f} +/- {u.heading_deg:.2f} deg",
            f"  pitch   {self.pose.pitch:.2f} +/- {u.pitch_deg:.2f} deg",
            f"  roll    {self.pose.roll:.2f} +/- {u.roll_deg:.2f} deg",
            f"  height  {self.pose.mount_height:.3f} +/- {u.mount_height_m:.3f} m",
        ]
        if self.position_error_m is not None:
            lines.append(f"  position {self.pose.position.lat:.6f}, {self.pose.position.lon:.6f} "
                         f"+/- {self.position_error_m:.3f} m")
        return "\n".join(lines)

    def better_than(self, other: PoseUncertainty) -> bool:
        """Whether this measurement beats the assumption it would replace.

        On every angular term at once. A fit that pins the heading down and
        loses the pitch has not improved the pose; it has traded one error for
        another, and the position on the map is no better for it.
        """
        u = self.uncertainty
        return (u.heading_deg <= other.heading_deg
                and u.pitch_deg <= other.pitch_deg
                and u.roll_deg <= other.roll_deg)


def _parameters(pose: CameraPose, names: Sequence[str]) -> np.ndarray:
    """The starting vector. `east`/`north` start at zero because they are an
    offset from where the camera was said to be, not a coordinate."""
    values = {"heading": pose.heading, "pitch": pose.pitch, "roll": pose.roll,
              "mount_height": pose.mount_height, "east": 0.0, "north": 0.0}
    return np.array([values[n] for n in names], dtype=np.float64)


def _with(pose: CameraPose, names: Sequence[str], values: np.ndarray,
          anchor: LatLon | None = None) -> CameraPose:
    """The pose with these parameters applied.

    `anchor` is where the camera started; `east`/`north` are metres from it. It
    has to be passed in rather than read from `pose`, because `pose` moves as
    the fit proceeds and an offset measured from a moving origin is applied
    twice.
    """
    fields = dict(zip(names, values.tolist()))
    position = pose.position
    if "east" in fields or "north" in fields:
        base = anchor if anchor is not None else pose.position
        position = LocalFrame(base).to_lat_lon(
            Vec2(fields.get("east", 0.0), fields.get("north", 0.0)))
    return replace(
        pose, position=position,
        heading=fields.get("heading", pose.heading),
        pitch=fields.get("pitch", pose.pitch),
        roll=fields.get("roll", pose.roll),
        mount_height=max(0.05, fields.get("mount_height", pose.mount_height)),
    )


def _residuals(pose: CameraPose, points: Sequence[Correspondence]) -> np.ndarray:
    """Reprojection error, two per point, in fractions of the frame.

    A point that falls behind the image plane gets a large finite residual
    rather than an infinite one: the optimiser has to be able to walk *out* of
    a bad pose, and a NaN in the Jacobian ends the fit instead of steering it.
    """
    out = np.empty(len(points) * 2, dtype=np.float64)
    for i, point in enumerate(points):
        seen = image_coordinates(pose, point.ground)
        if seen is None:
            out[i * 2] = out[i * 2 + 1] = 10.0
            continue
        out[i * 2] = seen.x - point.image.x
        out[i * 2 + 1] = seen.y - point.image.y
    return out


def _jacobian(pose: CameraPose, names: Sequence[str], points: Sequence[Correspondence],
              values: np.ndarray, anchor: LatLon | None = None) -> np.ndarray:
    rows = len(points) * 2
    out = np.zeros((rows, len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        step = _STEPS[name]
        forward = values.copy()
        backward = values.copy()
        forward[column] += step
        backward[column] -= step
        plus = _residuals(_with(pose, names, forward, anchor), points)
        minus = _residuals(_with(pose, names, backward, anchor), points)
        out[:, column] = (plus - minus) / (2 * step)
    return out


def calibrate_pose(pose: CameraPose, points: Sequence[Correspondence], *,
                   solve_position: bool = False) -> Calibration:
    """Fit the pose to marked ground points and measure what the fit is worth."""
    if len(points) < MIN_POINTS:
        raise CalibrationError(
            f"{len(points)} point(s) is not enough to measure a pose: four parameters need more "
            f"than four constraints, so at least {MIN_POINTS} points are required and six is where "
            f"the uncertainty starts to mean something"
        )
    names = ["heading", "pitch", "roll", "mount_height"]
    if solve_position:
        names += list(_POSITION)

    anchor = pose.position
    values = _parameters(pose, names)
    current = _residuals(pose, points)
    cost = float(current @ current)
    damping = 1e-3
    iterations = 0
    jacobian = _jacobian(pose, names, points, values, anchor)

    for iterations in range(1, MAX_ITERATIONS + 1):
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        step = None
        for _ in range(12):
            try:
                damped = normal + damping * np.diag(np.maximum(np.diag(normal), 1e-12))
                step = np.linalg.solve(damped, -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            trial = _with(pose, names, values + step, anchor)
            trial_residuals = _residuals(trial, points)
            trial_cost = float(trial_residuals @ trial_residuals)
            if trial_cost < cost:
                # Downhill: accept, and trust the linear model a little more.
                improvement = (cost - trial_cost) / max(cost, 1e-18)
                values = values + step
                pose = trial
                current = trial_residuals
                cost = trial_cost
                damping = max(damping / 10, 1e-12)
                jacobian = _jacobian(pose, names, points, values, anchor)
                if improvement < CONVERGENCE:
                    step = None
                break
            # Uphill: trust it less and try a shorter step.
            damping *= 10
        else:
            step = None
        if step is None:
            break

    normal = jacobian.T @ jacobian
    condition = float(np.linalg.cond(normal)) if normal.size else math.inf
    if not math.isfinite(condition) or condition > MAX_CONDITION:
        raise CalibrationError(
            "these points cannot separate the pose's parameters (condition number "
            f"{condition:.1e}). They are almost certainly clustered into one small part of the "
            "picture. Spread them across the frame and across the range: near and far, left and "
            "right"
        )

    residuals = np.hypot(current[0::2], current[1::2])
    degrees_of_freedom = max(1, len(current) - len(names))
    variance = cost / degrees_of_freedom
    try:
        covariance = variance * np.linalg.inv(normal)
    except np.linalg.LinAlgError as error:  # pragma: no cover - `condition` catches this first
        raise CalibrationError(f"the fit did not produce an invertible covariance: {error}") from error
    sigma = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    stated = dict(zip(names, sigma.tolist()))

    measured = PoseUncertainty(
        heading_deg=stated.get("heading", pose.uncertainty.heading_deg),
        pitch_deg=stated.get("pitch", pose.uncertainty.pitch_deg),
        roll_deg=stated.get("roll", pose.uncertainty.roll_deg),
        mount_height_m=stated.get("mount_height", pose.uncertainty.mount_height_m),
        # Terrain is not observable from points that are all *on* the ground:
        # they fit whatever plane the pose implies. It keeps its assumption
        # until `triangulation` solves the plane from points off it.
        terrain_slope=pose.uncertainty.terrain_slope,
    )
    position_error = None
    if solve_position:
        # East and north are independent draws; combine in quadrature, the
        # same rule `geo.separation` uses and for the same reason.
        position_error = float(math.hypot(stated.get("east", 0.0), stated.get("north", 0.0)))
    worst = int(np.argmax(residuals))
    calibration = Calibration(
        pose=replace(pose, uncertainty=measured),
        residuals=tuple(residuals.tolist()),
        rms=float(np.sqrt(np.mean(residuals ** 2))),
        worst=float(residuals[worst]),
        worst_index=worst,
        iterations=iterations,
        condition=condition,
        position_error_m=position_error,
    )
    _log.info("calibrated: %s", calibration.describe().replace("\n", " | "))
    return calibration


def calibrate_lens(images: Sequence, pattern: tuple[int, int] = (9, 6),
                   square_size_m: float = 0.025) -> tuple[Distortion, float, int]:
    """Distortion coefficients from photographs of a chessboard.

    Returns `(distortion, rms_pixels, boards_found)`. The count matters: OpenCV
    will happily fit a calibration to three views and produce coefficients that
    describe those three views, so a caller that does not check how many boards
    were actually found can ship a lens model built from nothing.

    Ten or more views, held at different angles and filling different parts of
    the frame, is the usual advice — and the reason is the same as for the pose
    above: a calibration is only as good as the geometry it saw.
    """
    import cv2

    if not images:
        raise CalibrationError("no images to calibrate from")
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size_m

    world: list[np.ndarray] = []
    found: list[np.ndarray] = []
    size = None
    for image in images:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        size = grey.shape[::-1]
        ok, corners = cv2.findChessboardCorners(grey, pattern, None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            grey, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        world.append(board)
        found.append(corners)

    if len(found) < 3 or size is None:
        raise CalibrationError(
            f"found the chessboard in {len(found)} image(s). A lens calibration needs it in at "
            f"least three and really wants ten, held at different angles and filling different "
            f"parts of the frame"
        )
    rms, _matrix, coefficients, _r, _t = cv2.calibrateCamera(world, found, size, None, None)
    flat = np.asarray(coefficients).reshape(-1)
    lens = Distortion(
        k1=float(flat[0]), k2=float(flat[1]),
        p1=float(flat[2]), p2=float(flat[3]),
        k3=float(flat[4]) if flat.size > 4 else 0.0,
    )
    _log.info("lens calibrated from %d board(s): %s, %.3f px RMS", len(found), lens.describe(), rms)
    return lens, float(rms), len(found)
