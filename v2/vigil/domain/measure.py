"""Measurements and the errors they carry. Never one without the other.

Split out of `geo` when that module outgrew its line budget, and the seam is
a real one: this file is about *what a number is worth*, and knows nothing
about cameras. The only geometry it needs is the tangent plane underneath it.

# The rule these types exist to enforce

Two positions each known to +/-1.4 m are eleven metres apart *give or take
about two*, and a plain "11 m" invites somebody to act on a precision nobody
measured. So every quantity here is a pair — the value and its 1-sigma error —
and the comparisons are deliberately not each other's negation: `within` and
`beyond`, `faster_than` and `slower_than`, all answer False when the
measurement cannot settle the question. A question the data cannot answer is
answered neither way.

Errors combine **in quadrature**, because independent measurements do. Adding
them would claim the errors always conspire; ignoring one would claim it does
not exist. `vigil.domain.incidents` used to add them, and that disagreement
between two modules about the same two numbers is what this file being one
place is meant to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .geodesy import LatLon, bearing_degrees, distance_meters, normalize_degrees


@dataclass(frozen=True, slots=True)
class ProjectionUncertainty:
    """The error ellipse of a projected position, on the ground, in metres.

    An ellipse and not a radius because the two are genuinely different: a
    shallow ray is sharp across its own direction and vague along it, and a
    camera 40 m away reporting "plus or minus 6 m" as a circle claims a
    sideways error it does not have.

    Two decompositions of the same covariance, because they answer different
    questions and both get asked:

    - `along`/`across` is the **line of sight** frame. "How well do I know how
      far away it is, against how well do I know which way it is" — the
      question a range or a bearing is judged by.
    - `semi_major`/`semi_minor`/`orientation` are the **true principal axes**,
      from an eigendecomposition. The question a *drawn* ellipse is judged by,
      and the one that bounds the error honestly whatever direction it points.

    For a mast camera the two are within about a percent of each other, which
    is why the line-of-sight pair was the only one here for a while. They come
    apart when the covariance is dominated by something other than range — a
    camera with a large heading error and a short reach, where the ellipse is
    a fat crescent across the view rather than a spike along it.
    """

    #: 1-sigma along the line of sight.
    along_meters: float
    #: 1-sigma across it.
    across_meters: float
    #: Bearing of the line of sight, degrees.
    orientation_deg: float
    #: 1-sigma along the true major axis, from the eigendecomposition.
    semi_major_meters: float = 0.0
    #: 1-sigma along the minor axis.
    semi_minor_meters: float = 0.0
    #: Bearing of the major axis, degrees clockwise from north.
    major_bearing_deg: float = 0.0

    @property
    def radius_meters(self) -> float:
        """The conservative single number: the true semi-major axis.

        A circle fitted inside the ellipse would understate the error in the
        direction it actually points, which is the direction somebody is sent
        to look.
        """
        return max(self.semi_major_meters, self.along_meters, self.across_meters)

    @property
    def rms_meters(self) -> float:
        """For combining errors rather than for drawing them."""
        return math.sqrt((self.semi_major_meters ** 2 + self.semi_minor_meters ** 2) / 2) \
            or math.sqrt((self.along_meters ** 2 + self.across_meters ** 2) / 2)

    @property
    def eccentricity(self) -> float:
        """0 for a circle, approaching 1 for a spike.

        Worth having because it says whether the single number above is
        telling most of the story: a position with an eccentricity of 0.1 is
        genuinely a circle and one at 0.99 is a line, and drawing both as
        circles of the same radius misleads about the second.
        """
        major, minor = self.semi_major_meters, self.semi_minor_meters
        if major <= 0:
            return 0.0
        ratio = min(1.0, (minor / major) ** 2)
        return math.sqrt(max(0.0, 1.0 - ratio))


def principal_axes(cov: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """`(semi_major, semi_minor, bearing)` of a 2x2 covariance in (east, north).

    The closed form for a symmetric 2x2, which is exact and needs no
    iteration: the eigenvalues are `(a+c)/2 +/- sqrt(((a-c)/2)^2 + b^2)` and
    the major axis sits at `atan2(2b, a-c)/2` from east.

    Guarded against the two degenerate cases that produce NaN rather than an
    answer: a covariance that is already circular, where the angle is
    arbitrary and any is right, and a negative eigenvalue from accumulated
    round-off, where the honest value is zero rather than the square root of
    a negative number.
    """
    a, b, c = float(cov[0][0]), float(cov[0][1]), float(cov[1][1])
    middle = (a + c) / 2
    spread = math.hypot((a - c) / 2, b)
    major_var, minor_var = middle + spread, middle - spread
    if spread <= 1e-18:
        # Circular: every direction is a principal axis. North, arbitrarily,
        # and the caller cannot tell because the two axes are equal.
        angle_from_east = 0.0
    else:
        angle_from_east = 0.5 * math.atan2(2 * b, a - c)
    bearing = normalize_degrees(math.degrees(math.atan2(math.cos(angle_from_east),
                                                        math.sin(angle_from_east))))
    return math.sqrt(max(0.0, major_var)), math.sqrt(max(0.0, minor_var)), bearing


@dataclass(frozen=True, slots=True)
class Distance:
    """A distance and how well it is known. Never one without the other.

    Two positions each known to ±1.4 m are eleven metres apart *give or take
    about two*, and a plain "11 m" invites somebody to act on a precision
    nobody measured.
    """

    meters: float
    error_meters: float

    def describe(self) -> str:
        return f"{self.meters:.1f} ± {self.error_meters:.1f} m"

    @property
    def at_most(self) -> float:
        return self.meters + self.error_meters

    @property
    def at_least(self) -> float:
        return max(0.0, self.meters - self.error_meters)

    def within(self, limit: float) -> bool:
        """True only when it is within `limit` even at its worst."""
        return self.at_most <= limit

    def beyond(self, limit: float) -> bool:
        """True only when it is past `limit` even at its best."""
        return self.at_least > limit


@dataclass(frozen=True, slots=True)
class Speed:
    """A speed and how well it is known. Never one without the other.

    Speed is a distance over a time, so it inherits the distance's error and
    nothing else: the clock is good to a millisecond and the positions are
    good to metres. `sigma_v = sigma_d / dt`.

    It exists because the track table printed `1.4 m/s` beside `12.0 ± 1.5 m`
    on the same row — the distance carrying its error and the speed, computed
    from two of those same distances, carrying none. An operator reading
    "walking pace" off a number whose error is larger than the number is being
    misled by the half of the row that looks most precise.
    """

    mps: float
    error_mps: float

    def describe(self) -> str:
        return f"{self.mps:.1f} ± {self.error_mps:.1f} m/s"

    @property
    def at_most(self) -> float:
        return self.mps + self.error_mps

    @property
    def at_least(self) -> float:
        return max(0.0, self.mps - self.error_mps)

    def faster_than(self, limit: float) -> bool:
        """True only when it is faster than `limit` even at its slowest."""
        return self.at_least > limit

    def slower_than(self, limit: float) -> bool:
        """True only when it is slower than `limit` even at its fastest."""
        return self.at_most <= limit

    @property
    def meaningful(self) -> bool:
        """False when the error is as large as the measurement.

        A speed of 0.4 +/- 0.9 m/s is not a slow walk, it is no measurement at
        all, and the honest thing to do with it is not to print it.
        """
        return self.mps > self.error_mps


@dataclass(frozen=True, slots=True)
class Bearing:
    """A direction of travel and how well it is known.

    The error is angular and comes from the same place the speed's does: an
    object that has moved `d` metres with `sigma_d` of position error has a
    heading known to about `atan(sigma_d / d)`. Something that has barely
    moved has a heading error approaching 90 degrees, which is the honest way
    of saying nobody knows which way it is going.
    """

    degrees: float
    error_degrees: float

    def describe(self) -> str:
        return f"{self.degrees:.0f} +/- {self.error_degrees:.0f} deg"

    @property
    def meaningful(self) -> bool:
        """False past 45 degrees: a heading that could be any of a quadrant
        does not tell an operator which way to look."""
        return self.error_degrees < 45.0


def motion_of(first: LatLon, last: LatLon, seconds: float, sigma_meters: float) -> tuple[Speed, Bearing]:
    """Speed and heading between two observed positions, with their errors.

    `sigma_meters` is the combined position error of the two endpoints, in
    quadrature — the same rule `separation` uses, because it is the same
    question.
    """
    if not seconds > 0:
        return Speed(0.0, math.inf), Bearing(0.0, 180.0)
    travelled = distance_meters(first, last)
    speed = Speed(travelled / seconds, sigma_meters / seconds)
    if travelled <= 1e-9:
        return speed, Bearing(0.0, 180.0)
    error = math.degrees(math.atan2(sigma_meters, travelled))
    return speed, Bearing(bearing_degrees(first, last), min(180.0, error))
