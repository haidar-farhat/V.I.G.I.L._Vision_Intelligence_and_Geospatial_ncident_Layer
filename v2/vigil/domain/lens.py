"""What a real lens does to a straight line, and how to undo it.

# Why this exists

`geo` documented a distortion-free lens as a stated assumption: intrinsics
came from a datasheet field of view and nothing corrected for the fact that a
real lens bends straight lines. On a 90-degree security lens radial distortion
moves a corner by several percent — and a corner of the frame is the far
ground, where the projection is already least certain. The two errors compound
in the same place.

# The model

Brown-Conrady, five coefficients, the same ones `cv2.calibrateCamera` returns
so a calibration can be dropped straight in:

- `k1, k2, k3` — **radial**. A barrel or pincushion bow; the dominant term on
  every wide lens.
- `p1, p2` — **tangential**. A sensor not quite parallel to the lens. Usually
  tiny, occasionally not, and free to carry once the radial terms are here.

```text
r2 = x^2 + y^2
radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x^2)
y_d = y*radial + p1*(r2 + 2*y^2) + 2*p2*x*y
```

# Which direction is which, because it is easy to get backwards

`distort` maps an **ideal** point to where it **actually lands** on the
sensor. That is the direction the projection needs: compute where a ground
point ideally falls, then bend it to find the pixel it is really at.

`undistort` is the inverse, and it is the direction a *ray* needs: a detection
gives a pixel that has already been bent, and the ray through it must be
computed from the ideal coordinate. There is no closed form, so it is solved
by fixed-point iteration — which converges in a handful of steps for any lens
that is not a fisheye, and says so rather than silently returning its last
guess when it does not.

# Zero is exactly today

All-zero coefficients make both functions the identity, bit for bit, so a
camera nobody has calibrated behaves exactly as it did before this module
existed. `tests/test_geo.py` holds that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Iterations for the undistortion fixed point.
#:
#: Measured at the frame corner with a typical wide security lens
#: (k1=-0.28, k2=0.09), as round-trip error in normalised camera coordinates:
#:
#: | field of view | 10 steps | 30 steps | 100 steps |
#: |---|---|---|---|
#: | 62 x 36  | 1.4e-07 | 9.8e-09 | 9.8e-09 |
#: | 90 x 50  | 1.3e-04 | 2.0e-08 | 2.0e-08 |
#: | 110 x 70 | 1.5e+05 | 1.5e+05 | 1.5e+05 |
#:
#: Thirty, because ninety degrees is an ordinary security lens and ten steps
#: is not converged there. A hundred buys nothing.
#:
#: The last row is the point: **a wide enough lens does not converge at any
#: iteration count.** It is outside this model's domain, not short of
#: arithmetic, and `converges` is how a caller finds out before storing a
#: calibration that would place a corner detection in the next county. A
#: fisheye needs a different model.
UNDISTORT_STEPS = 30

#: Convergence, in normalised image coordinates. A ten-thousandth is far below
#: one pixel on any sensor this product will meet.
UNDISTORT_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class Distortion:
    """Brown-Conrady coefficients, in OpenCV's order and sign convention.

    All zero means a perfect lens, which is what an uncalibrated camera is
    assumed to have — stated, not hidden, and the reason
    `vigil cameras calibrate` exists.
    """

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @property
    def is_identity(self) -> bool:
        """True when there is nothing to correct.

        Checked rather than assumed so the common case costs one comparison
        instead of a polynomial and five iterations per ray — and the map
        builder asks for a thousand rays a frame.
        """
        return self.k1 == 0.0 and self.k2 == 0.0 and self.p1 == 0.0 \
            and self.p2 == 0.0 and self.k3 == 0.0

    def describe(self) -> str:
        if self.is_identity:
            return "no lens correction (uncalibrated; assumed rectilinear)"
        return (f"k1={self.k1:+.4f} k2={self.k2:+.4f} k3={self.k3:+.4f} "
                f"p1={self.p1:+.5f} p2={self.p2:+.5f}")

    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Ideal coordinates to where they actually land on the sensor."""
        if self.is_identity:
            return x, y
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        return (
            x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
            y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        )

    def undistort(self, x: float, y: float) -> tuple[float, float]:
        """Where a pixel actually is, back to the ideal coordinate.

        Fixed-point iteration, because the forward model is a polynomial with
        no closed-form inverse. Starts from the distorted point, which is the
        right guess: the correction is small for any lens worth using.

        A point that will not converge — deep in the corner of a very strong
        barrel, or a nonsense calibration — is returned as its last iterate
        rather than looping, and `converges` is how a caller finds out.
        """
        if self.is_identity:
            return x, y
        u, v = x, y
        for _ in range(UNDISTORT_STEPS):
            r2 = u * u + v * v
            radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            if not radial > 1e-6:
                return u, v
            tangential_x = 2.0 * self.p1 * u * v + self.p2 * (r2 + 2.0 * u * u)
            tangential_y = self.p1 * (r2 + 2.0 * v * v) + 2.0 * self.p2 * u * v
            next_u = (x - tangential_x) / radial
            next_v = (y - tangential_y) / radial
            if abs(next_u - u) < UNDISTORT_TOLERANCE and abs(next_v - v) < UNDISTORT_TOLERANCE:
                return next_u, next_v
            u, v = next_u, next_v
        return u, v

    def converges(self, x: float, y: float, tolerance: float = 1e-4) -> bool:
        """Whether `undistort` actually inverted this point.

        Round-trip rather than a residual on the iteration: what a caller cares
        about is whether the two directions agree, and that is the thing to
        measure. `vigil cameras calibrate` checks it at the frame corners
        before accepting a calibration, because a set of coefficients that
        cannot be inverted at the edge of the frame is worse than none.
        """
        back = self.distort(*self.undistort(x, y))
        return math.hypot(back[0] - x, back[1] - y) < tolerance
