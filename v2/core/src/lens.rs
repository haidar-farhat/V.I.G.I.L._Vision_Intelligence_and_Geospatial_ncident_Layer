//! What a real lens does to a straight line, and how to undo it.
//!
//! The mirror of `vigil/domain/lens.py`, and it exists for the same reason
//! `camera.rs` mirrors `geo.py`: the ground rasteriser projects a thousand
//! points per frame and cannot cross the language boundary to do it. If the
//! two ever disagree, a calibrated camera would draw a map that does not
//! match the tracks placed on it. `tests/test_native.py` holds them together.
//!
//! Brown–Conrady, five coefficients, the same order and sign convention
//! `cv2.calibrateCamera` returns, so a calibration drops straight in.
//!
//! # Which direction is which
//!
//! [`Distortion::distort`] maps an **ideal** point to where it **actually
//! lands** on the sensor — the direction the inverse projection needs.
//! [`Distortion::undistort`] is the direction a *ray* needs: a detection gives
//! a pixel that has already been bent. There is no closed form, so it is a
//! fixed-point iteration.
//!
//! # Where it operates
//!
//! **Normalised camera coordinates** (`x/z`, `y/z`), which is the space
//! Brown–Conrady is defined in. Applying it to the `[-1, 1]` frame coordinate
//! instead is a different function of a differently scaled argument, and the
//! two directions then stop inverting each other — which is exactly the bug
//! the Python side was written with and had to be measured out of.

/// Iterations for the undistortion fixed point.
///
/// Measured at the frame corner with a typical wide security lens
/// (k1=-0.28, k2=0.09), as round-trip error in normalised camera coordinates:
///
/// | field of view | 10 steps | 30 steps | 100 steps |
/// |---|---|---|---|
/// | 62 x 36  | 1.4e-07 | 9.8e-09 | 9.8e-09 |
/// | 90 x 50  | 1.3e-04 | 2.0e-08 | 2.0e-08 |
/// | 110 x 70 | 1.5e+05 | 1.5e+05 | 1.5e+05 |
///
/// Thirty, because ninety degrees is an ordinary security lens and ten is not
/// converged there. The last row is the point: a wide enough lens does not
/// converge at any iteration count. It is outside this model's domain, and
/// [`Distortion::converges`] is how a caller finds out.
pub const UNDISTORT_STEPS: usize = 30;

/// Convergence, in normalised camera coordinates.
pub const UNDISTORT_TOLERANCE: f64 = 1e-7;

/// Brown–Conrady coefficients. All zero is a perfect lens, which is what an
/// uncalibrated camera is assumed to have.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl Distortion {
    /// True when there is nothing to correct.
    ///
    /// Checked rather than assumed so the common case costs one comparison
    /// instead of a polynomial and thirty iterations per ray — and the map
    /// builder asks for a thousand rays a frame.
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0 && self.k3 == 0.0
    }

    /// Ideal coordinates to where they actually land on the sensor.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        (
            x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x),
            y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y,
        )
    }

    /// Where a pixel actually is, back to the ideal coordinate.
    ///
    /// Starts from the distorted point, which is the right guess: the
    /// correction is small for any lens worth using. A point that will not
    /// converge is returned as its last iterate rather than looping forever,
    /// and [`Distortion::converges`] is how a caller finds out.
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (mut u, mut v) = (x, y);
        for _ in 0..UNDISTORT_STEPS {
            let r2 = u * u + v * v;
            let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
            if !(radial > 1e-6) {
                return (u, v);
            }
            let tangential_x = 2.0 * self.p1 * u * v + self.p2 * (r2 + 2.0 * u * u);
            let tangential_y = self.p1 * (r2 + 2.0 * v * v) + 2.0 * self.p2 * u * v;
            let next_u = (x - tangential_x) / radial;
            let next_v = (y - tangential_y) / radial;
            if (next_u - u).abs() < UNDISTORT_TOLERANCE
                && (next_v - v).abs() < UNDISTORT_TOLERANCE
            {
                return (next_u, next_v);
            }
            u = next_u;
            v = next_v;
        }
        (u, v)
    }

    /// Whether `undistort` actually inverted this point.
    ///
    /// A round trip, because what a caller cares about is whether the two
    /// directions agree. Checked at the **raw image corner**, where it fails
    /// first: checking a distorted-then-undistorted point instead tests a
    /// smaller radius than the one that fails, and passes a lens that does
    /// not work.
    pub fn converges(&self, x: f64, y: f64, tolerance: f64) -> bool {
        let (ux, uy) = self.undistort(x, y);
        let (bx, by) = self.distort(ux, uy);
        (bx - x).hypot(by - y) < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide() -> Distortion {
        Distortion { k1: -0.28, k2: 0.09, p1: 0.0006, p2: -0.0004, k3: -0.012 }
    }

    #[test]
    fn no_calibration_is_the_identity_bit_for_bit() {
        let none = Distortion::default();
        assert!(none.is_identity());
        for &(x, y) in &[(0.0, 0.0), (0.3, -0.2), (1.0, 0.47)] {
            assert_eq!(none.distort(x, y), (x, y));
            assert_eq!(none.undistort(x, y), (x, y));
        }
    }

    #[test]
    fn the_two_directions_invert_each_other_over_a_real_frame() {
        let lens = wide();
        let mut worst: f64 = 0.0;
        for i in 0..9 {
            for j in 0..9 {
                let x = -0.60 + i as f64 * 0.15;
                let y = -0.33 + j as f64 * 0.0825;
                let (dx, dy) = lens.distort(x, y);
                let (ux, uy) = lens.undistort(dx, dy);
                worst = worst.max((ux - x).hypot(uy - y));
            }
        }
        assert!(worst < 1e-6, "round trip off by {worst}");
    }

    #[test]
    fn a_lens_too_wide_for_this_model_says_so_rather_than_guessing() {
        let lens = wide();
        // The raw image corner, in normalised camera coordinates, for each
        // field of view. 110 degrees does not converge at any iteration count.
        let corner = |fov: f64| (fov / 2.0f64).to_radians().tan();
        assert!(lens.converges(corner(62.0), corner(36.0), 1e-4));
        assert!(!lens.converges(corner(110.0), corner(70.0), 1e-4));
    }

    #[test]
    fn distortion_actually_moves_a_point() {
        let lens = wide();
        let (dx, dy) = lens.distort(1.0, 0.466);
        assert!((dx - 1.0).abs() > 0.1, "a real lens bends the corner: {dx}");
        assert!(dy < 0.466);
    }
}
