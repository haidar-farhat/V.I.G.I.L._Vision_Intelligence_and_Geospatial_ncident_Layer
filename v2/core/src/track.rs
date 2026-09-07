//! The per-track filter: a constant-velocity Kalman filter over a box.
//!
//! # What was there before
//!
//! v1's Rust tracker and v2's Python port both carried velocity as an
//! exponential moving average of frame-to-frame box displacement, blended
//! 60/40, and "predicted" by adding `velocity * dt` to the last box. Three
//! things follow from that and all three were visible in the product:
//!
//!  - **No uncertainty.** The tracker could not say how sure it was where a
//!    track had got to, so association had nothing to gate on but a distance
//!    threshold scaled by box size — a hand-tuned constant standing in for a
//!    covariance.
//!  - **No noise model.** A detector's box jitters by a few percent of its
//!    own height every frame. An EMA cannot distinguish that jitter from
//!    motion, so a stationary object had a wandering velocity and a coasting
//!    prediction that walked away from it.
//!  - **Wrong under variable dt.** A dropped frame doubles `dt`. The EMA's
//!    time constant is in *frames*, so the smoothing silently changed
//!    whenever the camera stuttered, which is exactly when tracking is hard.
//!
//! # The state
//!
//! `[cx, cy, a, h, vcx, vcy, va, vh]`: box centre, aspect ratio `w/h`, height,
//! and their rates. Normalised image coordinates throughout, so `h` is the
//! object's apparent size and doubles as the scale for every noise term —
//! which is what makes one set of constants work for a person at 5 m and the
//! same person at 40 m.
//!
//! # The noise model, and why these numbers
//!
//! Continuous white-noise acceleration, discretised exactly:
//!
//! ```text
//! Q_block = q * [[dt^3/3, dt^2/2],
//!                [dt^2/2, dt    ]]
//! ```
//!
//! so a dropped frame widens the gate by the right amount instead of by
//! whatever the frame counter happened to do.
//!
//! `q` for the centre is `ACCEL_PSD_CENTRE * h^2`, and that constant carries
//! the derivation — including the units error the first version of it made.

/// Acceleration **power spectral density** for the box centre, in squared
/// object-heights per second cubed. Note the units: this is a PSD, not an
/// acceleration, and conflating the two is the error this constant was rewritten
/// to fix.
///
/// The first version of this filter set q = (3h)^2, reasoning that a 1.7 m person
/// can accelerate at about 5 m/s^2 and that h is inversely proportional to range,
/// so image-space acceleration is 5h/1.7 ~ 3h. The arithmetic is right and the
/// substitution is not: in a continuous white-noise-acceleration model the
/// velocity variance grows as q*dt, so q = (3h)^2 lets the estimated velocity
/// wander by 3h*sqrt(dt) in one frame - 0.155 frames per second at 15 fps - while
/// the largest change a person can physically produce in that frame is 3h*dt,
/// which is 0.039. The filter was four times more willing to believe in
/// acceleration than acceleration exists, so it chased detector jitter and gave
/// parked cars a velocity.
///
/// A PSD needs an acceleration *and a time over which it is held*:
/// q = (a/H)^2 * tau. A pedestrian sustains about 1 m/s^2 for about half a second
/// when starting, stopping or turning, which over a 1.7 m frame gives 0.17. A
/// 5 m/s^2 sprint start is real, lasts a fraction of a second, and arrives as an
/// innovation the chi-squared gate still accepts - which is where a brief
/// manoeuvre belongs.
pub const ACCEL_PSD_CENTRE: f64 = 0.18;

/// The same for apparent height, which changes only with range.
pub const ACCEL_PSD_SCALE: f64 = 0.02;

/// The same for aspect ratio, absolute rather than scaled by `h`.
pub const ACCEL_PSD_ASPECT: f64 = 0.09;

/// Detector box jitter, 1-sigma, as a fraction of box height. A YOLO box's
/// edges move by a few percent of the box between consecutive frames of a
/// stationary object; 5% is that, measured on the shipped model.
pub const MEASURE_STD_FRACTION: f64 = 0.05;

/// Jitter of the aspect ratio, absolute. Larger than the positional term
/// because the width of a box around a walking person genuinely does change.
pub const MEASURE_STD_ASPECT: f64 = 0.1;

/// Chi-squared 0.95 quantile at 4 degrees of freedom. A measurement further
/// than this from a track's prediction, in units of the track's own
/// uncertainty, is not that track — 5% of the time it is and the track
/// survives on the next frame anyway.
pub const CHI2_GATE_4DOF: f64 = 9.4877;

/// The same at 2 degrees of freedom, for gating on centre position alone.
pub const CHI2_GATE_2DOF: f64 = 5.9915;

const N: usize = 8;
const M: usize = 4;

/// A track's estimated box and how well it is known.
#[derive(Clone, Copy, Debug)]
pub struct KalmanBox {
    /// `[cx, cy, a, h, vcx, vcy, va, vh]`.
    pub mean: [f64; N],
    /// Row-major 8x8 covariance.
    pub covariance: [[f64; N]; N],
}

impl KalmanBox {
    /// Start a track from one measurement.
    ///
    /// Velocity is initialised to zero with a *wide* covariance rather than
    /// to a guess: one box says nothing at all about speed, and a filter told
    /// otherwise spends its first second chasing an invented velocity.
    pub fn initiate(cx: f64, cy: f64, a: f64, h: f64) -> Self {
        let mut covariance = [[0.0; N]; N];
        let position = 2.0 * MEASURE_STD_FRACTION * h;
        let velocity = 10.0 * MEASURE_STD_FRACTION * h;
        let std = [
            position,
            position,
            2.0 * MEASURE_STD_ASPECT,
            position,
            velocity,
            velocity,
            10.0 * MEASURE_STD_ASPECT,
            velocity,
        ];
        for i in 0..N {
            covariance[i][i] = std[i] * std[i];
        }
        Self {
            mean: [cx, cy, a, h, 0.0, 0.0, 0.0, 0.0],
            covariance,
        }
    }

    pub fn box_xywh(&self) -> (f64, f64, f64, f64) {
        let (cx, cy, a, h) = (self.mean[0], self.mean[1], self.mean[2], self.mean[3]);
        let w = a * h;
        (cx - w / 2.0, cy - h / 2.0, w, h)
    }

    /// Advance the state by `dt` seconds.
    ///
    /// `dt` of zero or less is a no-op rather than an error: two detections
    /// carrying the same timestamp is a decoder quirk, not a reason to
    /// corrupt a filter.
    pub fn predict(&mut self, dt: f64) {
        if !(dt > 0.0) || !dt.is_finite() {
            return;
        }
        // x = F x, with F the constant-velocity transition.
        for i in 0..M {
            self.mean[i] += dt * self.mean[i + M];
        }
        // P = F P F' + Q. F is I + dt*S where S shifts velocity into
        // position, so F P F' is three rank-updates rather than two 8x8
        // multiplies.
        let p = self.covariance;
        let mut out = p;
        for i in 0..M {
            for j in 0..N {
                out[i][j] = p[i][j] + dt * p[i + M][j];
            }
        }
        let mid = out;
        for j in 0..M {
            for i in 0..N {
                out[i][j] = mid[i][j] + dt * mid[i][j + M];
            }
        }
        let h = self.mean[3].abs().max(1e-4);
        let q = [
            ACCEL_PSD_CENTRE * h * h,
            ACCEL_PSD_CENTRE * h * h,
            ACCEL_PSD_ASPECT,
            ACCEL_PSD_SCALE * h * h,
        ];
        let (t3, t2, t1) = (dt * dt * dt / 3.0, dt * dt / 2.0, dt);
        for i in 0..M {
            out[i][i] += q[i] * t3;
            out[i][i + M] += q[i] * t2;
            out[i + M][i] += q[i] * t2;
            out[i + M][i + M] += q[i] * t1;
        }
        self.covariance = out;
        self.symmetrise();
    }

    /// The measurement-space mean and its innovation covariance `S`.
    pub fn project(&self) -> ([f64; M], [[f64; M]; M]) {
        let h = self.mean[3].abs().max(1e-4);
        let r = [
            (MEASURE_STD_FRACTION * h).powi(2),
            (MEASURE_STD_FRACTION * h).powi(2),
            MEASURE_STD_ASPECT.powi(2),
            (MEASURE_STD_FRACTION * h).powi(2),
        ];
        let mut s = [[0.0; M]; M];
        for i in 0..M {
            for j in 0..M {
                s[i][j] = self.covariance[i][j];
            }
            s[i][i] += r[i];
        }
        ([self.mean[0], self.mean[1], self.mean[2], self.mean[3]], s)
    }

    /// Fold in a measured box. Returns false when the innovation covariance
    /// is not invertible, which means the filter has been fed something
    /// degenerate and the caller should restart the track rather than carry
    /// on with a poisoned state.
    pub fn update(&mut self, cx: f64, cy: f64, a: f64, h: f64) -> bool {
        let (projected, s) = self.project();
        let Some(chol) = cholesky4(&s) else {
            return false;
        };
        // K = P H' S^-1, solved as K S = P H' by the Cholesky factor rather
        // than by forming S^-1.
        let mut gain = [[0.0; M]; N];
        for i in 0..N {
            let row: [f64; M] = [
                self.covariance[i][0],
                self.covariance[i][1],
                self.covariance[i][2],
                self.covariance[i][3],
            ];
            let solved = chol_solve4(&chol, &row);
            gain[i] = solved;
        }
        let innovation = [cx - projected[0], cy - projected[1], a - projected[2], h - projected[3]];
        for i in 0..N {
            let mut delta = 0.0;
            for k in 0..M {
                delta += gain[i][k] * innovation[k];
            }
            self.mean[i] += delta;
        }
        // Joseph-free form: P -= K S K'. Equivalent to (I - K H) P for a
        // consistent gain and cheaper than the Joseph form, with the
        // symmetrisation below standing in for its stability.
        let mut ks = [[0.0; M]; N];
        for i in 0..N {
            for j in 0..M {
                let mut acc = 0.0;
                for k in 0..M {
                    acc += gain[i][k] * s[k][j];
                }
                ks[i][j] = acc;
            }
        }
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..M {
                    acc += ks[i][k] * gain[j][k];
                }
                self.covariance[i][j] -= acc;
            }
        }
        self.symmetrise();
        // Height and aspect are positive by construction; a filter that has
        // been fed a run of bad boxes can drive them negative, and a negative
        // height turns every downstream IoU into nonsense.
        self.mean[2] = self.mean[2].max(1e-4);
        self.mean[3] = self.mean[3].max(1e-4);
        true
    }

    /// Squared Mahalanobis distance from this track's prediction to a
    /// measured box. `position_only` gates on the centre alone, which is the
    /// right test when a detector's box size is unreliable — a partly
    /// occluded person has a correct centre and a badly wrong height.
    pub fn gating_distance(&self, cx: f64, cy: f64, a: f64, h: f64, position_only: bool) -> f64 {
        let (projected, s) = self.project();
        if position_only {
            let sub = [[s[0][0], s[0][1]], [s[1][0], s[1][1]]];
            let d = [cx - projected[0], cy - projected[1]];
            let det = sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0];
            if !(det.abs() > 1e-18) {
                return f64::INFINITY;
            }
            let inv = [
                [sub[1][1] / det, -sub[0][1] / det],
                [-sub[1][0] / det, sub[0][0] / det],
            ];
            return d[0] * (inv[0][0] * d[0] + inv[0][1] * d[1])
                + d[1] * (inv[1][0] * d[0] + inv[1][1] * d[1]);
        }
        let Some(chol) = cholesky4(&s) else {
            return f64::INFINITY;
        };
        let d = [cx - projected[0], cy - projected[1], a - projected[2], h - projected[3]];
        // z = L^-1 d; the squared Mahalanobis distance is z'z.
        let mut z = [0.0; M];
        for i in 0..M {
            let mut acc = d[i];
            for k in 0..i {
                acc -= chol[i][k] * z[k];
            }
            z[i] = acc / chol[i][i];
        }
        z.iter().map(|v| v * v).sum()
    }

    /// Apply a 2x3 affine to the filter, for camera motion between frames.
    ///
    /// The mean moves through the affine and the covariance is rotated by its
    /// linear part, so an estimate that was uncertain along one axis stays
    /// uncertain along that axis after the camera turns. A tracker that moves
    /// the boxes and leaves the covariance alone claims a precision the
    /// motion estimate does not have.
    ///
    /// `warp` is `[[a, b, tx], [c, d, ty]]` in normalised image coordinates.
    pub fn apply_warp(&mut self, warp: &[[f64; 3]; 2]) {
        let (a, b, tx) = (warp[0][0], warp[0][1], warp[0][2]);
        let (c, d, ty) = (warp[1][0], warp[1][1], warp[1][2]);
        let (cx, cy) = (self.mean[0], self.mean[1]);
        self.mean[0] = a * cx + b * cy + tx;
        self.mean[1] = c * cx + d * cy + ty;
        let (vx, vy) = (self.mean[4], self.mean[5]);
        self.mean[4] = a * vx + b * vy;
        self.mean[5] = c * vx + d * vy;
        // Scale: the geometric mean of the affine's singular values, which is
        // sqrt|det|. Anisotropic scaling is not representable in this state,
        // and a camera warp that is anisotropic enough to matter is not a
        // camera warp, it is a bad estimate.
        let scale = (a * d - b * c).abs().sqrt();
        if scale.is_finite() && scale > 1e-6 {
            self.mean[3] *= scale;
            self.mean[7] *= scale;
        }
        // Rotate the covariance blocks that the affine touched.
        let l = [[a, b], [c, d]];
        for block in [0usize, 4usize] {
            for other in 0..N {
                let (p0, p1) = (
                    self.covariance[block][other],
                    self.covariance[block + 1][other],
                );
                self.covariance[block][other] = l[0][0] * p0 + l[0][1] * p1;
                self.covariance[block + 1][other] = l[1][0] * p0 + l[1][1] * p1;
            }
            for other in 0..N {
                let (p0, p1) = (
                    self.covariance[other][block],
                    self.covariance[other][block + 1],
                );
                self.covariance[other][block] = l[0][0] * p0 + l[0][1] * p1;
                self.covariance[other][block + 1] = l[1][0] * p0 + l[1][1] * p1;
            }
        }
        self.symmetrise();
    }

    /// Force exact symmetry. Repeated rank updates in floating point drift
    /// out of symmetry by a few ulp per step, and a Cholesky of a matrix that
    /// is not quite symmetric fails in a way that looks like a modelling bug.
    fn symmetrise(&mut self) {
        for i in 0..N {
            for j in (i + 1)..N {
                let m = 0.5 * (self.covariance[i][j] + self.covariance[j][i]);
                self.covariance[i][j] = m;
                self.covariance[j][i] = m;
            }
        }
    }
}

/// Lower-triangular Cholesky factor of a 4x4 SPD matrix, or `None` when the
/// matrix is not positive definite.
fn cholesky4(a: &[[f64; M]; M]) -> Option<[[f64; M]; M]> {
    let mut l = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(sum > 0.0) || !sum.is_finite() {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `A x = b` given `A`'s Cholesky factor.
fn chol_solve4(l: &[[f64; M]; M], b: &[f64; M]) -> [f64; M] {
    let mut y = [0.0; M];
    for i in 0..M {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i][k] * y[k];
        }
        y[i] = acc / l[i][i];
    }
    let mut x = [0.0; M];
    for i in (0..M).rev() {
        let mut acc = y[i];
        for k in (i + 1)..M {
            acc -= l[k][i] * x[k];
        }
        x[i] = acc / l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxes_close(k: &KalmanBox, cx: f64, cy: f64, tol: f64) -> bool {
        (k.mean[0] - cx).abs() < tol && (k.mean[1] - cy).abs() < tol
    }

    #[test]
    fn a_new_track_knows_where_it_is_and_nothing_about_its_speed() {
        let k = KalmanBox::initiate(0.5, 0.6, 0.5, 0.2);
        assert_eq!(k.mean[0], 0.5);
        assert_eq!(&k.mean[4..], &[0.0, 0.0, 0.0, 0.0]);
        // The velocity block must start much less certain than the position
        // block, or the first update is dominated by a velocity of zero.
        assert!(k.covariance[4][4] > 10.0 * k.covariance[0][0]);
    }

    #[test]
    fn it_converges_on_a_constant_velocity_and_then_predicts_it() {
        let dt = 1.0 / 15.0;
        let (speed_x, h) = (0.3, 0.2);
        let mut k = KalmanBox::initiate(0.1, 0.5, 0.5, h);
        let mut x = 0.1;
        for _ in 0..60 {
            x += speed_x * dt;
            k.predict(dt);
            assert!(k.update(x, 0.5, 0.5, h));
        }
        assert!(
            (k.mean[4] - speed_x).abs() < 0.02,
            "velocity should converge on {speed_x}, got {}",
            k.mean[4]
        );
        // And a coast of half a second lands where the object actually is.
        k.predict(0.5);
        assert!(boxes_close(&k, x + speed_x * 0.5, 0.5, 0.02));
    }

    #[test]
    fn it_rejects_jitter_that_an_average_would_have_chased() {
        // A stationary object whose box wobbles by 5% of its height. The old
        // EMA turned that into a velocity; the filter must not.
        let dt = 1.0 / 15.0;
        let h = 0.2;
        let mut k = KalmanBox::initiate(0.5, 0.5, 0.5, h);
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for _ in 0..90 {
            k.predict(dt);
            assert!(k.update(0.5 + next() * 0.01, 0.5 + next() * 0.01, 0.5, h));
        }
        let speed = k.mean[4].hypot(k.mean[5]);
        assert!(speed < 0.05, "a stationary object drifted at {speed}/s");
        assert!(boxes_close(&k, 0.5, 0.5, 0.01));
    }

    #[test]
    fn a_dropped_frame_widens_the_gate_by_the_right_amount() {
        // The whole point of a time-parameterised Q: uncertainty after one
        // 200 ms gap must match uncertainty after three 66 ms steps, not
        // whatever a per-frame constant would have given.
        let mut a = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        let mut b = a;
        a.predict(0.2);
        for _ in 0..3 {
            b.predict(0.2 / 3.0);
        }
        let ratio = a.covariance[0][0] / b.covariance[0][0];
        assert!(
            (0.8..1.25).contains(&ratio),
            "one long step vs three short: covariance ratio {ratio}"
        );
        // And it must actually grow: a track that has not been seen for
        // 200 ms is less certain than one seen 66 ms ago.
        let mut short = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        short.predict(1.0 / 15.0);
        assert!(a.covariance[0][0] > short.covariance[0][0]);
    }

    #[test]
    fn the_gate_accepts_the_truth_and_rejects_a_stranger() {
        let dt = 1.0 / 15.0;
        let mut k = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        for _ in 0..20 {
            k.predict(dt);
            k.update(0.5, 0.5, 0.5, 0.2);
        }
        k.predict(dt);
        assert!(k.gating_distance(0.5, 0.5, 0.5, 0.2, false) < CHI2_GATE_4DOF);
        assert!(k.gating_distance(0.9, 0.2, 0.5, 0.2, false) > CHI2_GATE_4DOF);
        // Position-only gating survives a box whose height the detector got
        // badly wrong, which is what a half-occluded person looks like.
        assert!(k.gating_distance(0.5, 0.5, 0.5, 0.6, true) < CHI2_GATE_2DOF);
        assert!(k.gating_distance(0.5, 0.5, 0.5, 0.6, false) > CHI2_GATE_4DOF);
    }

    #[test]
    fn the_covariance_stays_symmetric_and_positive_definite_under_load() {
        let dt = 1.0 / 15.0;
        let mut k = KalmanBox::initiate(0.3, 0.4, 0.6, 0.15);
        for i in 0..500 {
            k.predict(dt);
            let t = i as f64 * dt;
            assert!(k.update(0.3 + 0.1 * t.sin(), 0.4 + 0.05 * t.cos(), 0.6, 0.15));
            for r in 0..N {
                for c in 0..N {
                    assert!(
                        (k.covariance[r][c] - k.covariance[c][r]).abs() < 1e-18,
                        "asymmetric at {i}"
                    );
                }
                assert!(k.covariance[r][r] > 0.0, "variance went non-positive at {i}");
            }
            let (_, s) = k.project();
            assert!(cholesky4(&s).is_some(), "S lost definiteness at {i}");
        }
    }

    #[test]
    fn a_camera_pan_moves_the_track_and_its_uncertainty_together() {
        let mut k = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        k.predict(1.0 / 15.0);
        let before = k.covariance[0][0];
        // Pure translation: 0.1 of a frame to the right.
        k.apply_warp(&[[1.0, 0.0, 0.1], [0.0, 1.0, 0.0]]);
        assert!((k.mean[0] - 0.6).abs() < 1e-12);
        assert!((k.covariance[0][0] - before).abs() < 1e-12, "translation is not a rotation");

        // A 90-degree rotation must swap the two positional variances.
        let mut r = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        r.covariance[0][0] = 0.04;
        r.covariance[1][1] = 0.01;
        r.apply_warp(&[[0.0, -1.0, 1.0], [1.0, 0.0, 0.0]]);
        assert!((r.covariance[0][0] - 0.01).abs() < 1e-12);
        assert!((r.covariance[1][1] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn a_degenerate_update_is_refused_rather_than_poisoning_the_filter() {
        let mut k = KalmanBox::initiate(0.5, 0.5, 0.5, 0.2);
        k.predict(f64::NAN);
        assert!(k.mean[0].is_finite(), "a NaN dt must be a no-op");
        k.predict(-1.0);
        assert!((k.mean[0] - 0.5).abs() < 1e-12);
        // Height driven to nothing must not become negative.
        for _ in 0..40 {
            k.predict(1.0 / 15.0);
            k.update(0.5, 0.5, 0.5, 1e-6);
        }
        assert!(k.mean[3] > 0.0);
    }

    #[test]
    fn cholesky_refuses_a_matrix_that_is_not_positive_definite() {
        let bad = [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(cholesky4(&bad).is_none());
        let good = [
            [2.0, 0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.1],
            [0.0, 0.0, 0.1, 3.0],
        ];
        let l = cholesky4(&good).expect("SPD");
        // L L' must reproduce the matrix.
        for i in 0..M {
            for j in 0..M {
                let mut acc = 0.0;
                for k in 0..M {
                    acc += l[i][k] * l[j][k];
                }
                assert!((acc - good[i][j]).abs() < 1e-12);
            }
        }
        let x = chol_solve4(&l, &[1.0, 2.0, 3.0, 4.0]);
        for i in 0..M {
            let mut acc = 0.0;
            for j in 0..M {
                acc += good[i][j] * x[j];
            }
            assert!((acc - [1.0, 2.0, 3.0, 4.0][i]).abs() < 1e-12);
        }
    }
}
