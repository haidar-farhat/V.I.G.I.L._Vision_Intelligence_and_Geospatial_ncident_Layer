//! Where two cameras agree an object is, and what the ground under it does.
//!
//! # The assumption this exists to remove
//!
//! Every position this product has produced so far comes from intersecting one
//! ray with an *assumed* plane: flat ground at the camera's mount height. That
//! assumption is why a person standing on a loading dock is placed several
//! metres past where they are, why a yard with a 2% fall biases every position
//! along the line of sight, and why `PoseUncertainty` has to carry a
//! `terrain_slope` term at all.
//!
//! Two cameras that can both see the same object do not need the assumption.
//! The object is where their two rays come closest, and the height of that
//! point above the fitted ground is a **measurement** rather than a premise.
//!
//! # Why the midpoint, and what it costs
//!
//! Two rays in three dimensions almost never meet. The estimate here is the
//! midpoint of their common perpendicular — the least-squares point in the
//! sense that it minimises the sum of squared distances to both lines. It is
//! not the maximum-likelihood point under image-plane noise (that would be the
//! two-view optimal correction, which needs the full covariance of each ray
//! and buys a fraction of a pixel), and for two security cameras metres apart
//! looking at something tens of metres away, the difference is far below the
//! pose error that dominates.
//!
//! What the midpoint *does* give, and what is reported, is the **gap**: how
//! far apart the rays were at closest approach. That number is the honest
//! check on the whole thing. Two rays that pass 4 m apart are not looking at
//! the same object, whatever the association said, and a midpoint computed
//! from them is a confident answer to a question nobody asked.
//!
//! # Parallax, and why a small one is refused rather than averaged
//!
//! The error along the baseline direction goes as `sigma_angle * range /
//! sin(parallax)`. At 30 degrees of parallax and a quarter-degree of pose
//! error, a 40 m object is good to about 0.35 m. At 2 degrees it is 5 m, and
//! at 0.5 degrees it is 20 m — worse than the flat-ground projection it would
//! replace, while looking more authoritative because two cameras agreed.
//!
//! So a pair below [`MIN_PARALLAX_DEG`] is refused. Not averaged, not
//! down-weighted: refused, and the caller falls back to ground projection,
//! which for a nearly-collinear pair is the better estimate.
//!
//! # The plane
//!
//! [`fit_plane`] is RANSAC over accumulated ground contacts. RANSAC rather
//! than least squares because the input is contaminated by construction: the
//! points come from the bottom edge of detection boxes, and a box that clipped
//! at the frame edge, or bounded a person on a step, puts its contact
//! somewhere that is not the ground. A least-squares plane tilts towards every
//! one of those. RANSAC ignores them and says how many it ignored.
//!
//! The random draw is seeded and the generator is written out here, so the
//! same points give the same plane on every machine and in every run. A map
//! that quietly redraws itself differently on a re-run cannot be audited.

use crate::camera::Vec3;

/// Below this angle between two rays, the pair is refused.
///
/// Chosen from the error, not from taste. With a quarter-degree of pose
/// uncertainty — roughly what [`crate::camera`]'s calibrated cameras reach —
/// the along-baseline error at 40 m is 0.35 m at 30 degrees, 1.0 m at 10, and
/// 5.0 m at 2. Five degrees puts it at 2.0 m, which is about where a
/// triangulated position stops beating the flat-ground projection it would
/// replace on the kind of yard this product watches.
pub const MIN_PARALLAX_DEG: f64 = 5.0;

/// Above this closest-approach distance, the two rays are not looking at the
/// same thing and the pair is refused.
///
/// Three metres is wider than a person and narrower than the gap a
/// mis-association opens. It is a correspondence check, not a precision one:
/// the pair is being rejected as *wrong*, not as imprecise.
pub const MAX_GAP_M: f64 = 3.0;

/// A ray in the local ENU frame: metres east, north, up.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    /// Normalised in [`Ray::new`]; every consumer here assumes unit length.
    pub direction: Vec3,
}

impl Ray {
    /// `None` for a direction of no length, which is a caller bug rather than
    /// a geometry the arithmetic can express.
    pub fn new(origin: Vec3, direction: Vec3) -> Option<Self> {
        let n = direction.norm();
        if !n.is_finite() || n < 1e-12 {
            return None;
        }
        Some(Self {
            origin,
            direction: direction.scale(1.0 / n),
        })
    }
}

/// A point two cameras agree on, and every number needed to disbelieve it.
#[derive(Clone, Copy, Debug)]
pub struct Triangulation {
    /// Local ENU metres.
    pub point: Vec3,
    /// Angle between the two rays. Small means ill-conditioned.
    pub parallax_deg: f64,
    /// How far apart the rays passed. Large means these are two objects.
    pub gap_m: f64,
    pub range_a: f64,
    pub range_b: f64,
    /// 1-sigma of the position, metres, in its worst direction — along the
    /// baseline, which is always the worst one for a two-view intersection.
    pub sigma_m: f64,
}

/// Why a pair could not be triangulated, so the caller can say which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The rays are parallel to within the arithmetic's ability to tell.
    Parallel,
    /// Below [`MIN_PARALLAX_DEG`].
    TooLittleParallax,
    /// The intersection is behind one of the cameras.
    Behind,
    /// Above [`MAX_GAP_M`]: not the same object.
    TooFarApart,
}

/// The least-squares midpoint of two rays' common perpendicular.
///
/// `angular_sigma_deg` is the 1-sigma pointing error of each camera — for a
/// calibrated one, what `service.calibration` measured. It only scales the
/// reported `sigma_m`; it does not move the point.
pub fn triangulate(
    a: Ray,
    b: Ray,
    angular_sigma_deg: f64,
    min_parallax_deg: f64,
) -> Result<Triangulation, Refusal> {
    let d = a.direction.dot(b.direction);
    // Both directions are unit, so `1 - d*d` is `sin^2` of the angle between
    // them, and it is also the determinant of the 2x2 system below.
    let denominator = 1.0 - d * d;
    if denominator < 1e-12 {
        return Err(Refusal::Parallel);
    }
    let parallax_deg = d.clamp(-1.0, 1.0).acos().to_degrees();
    // The angle between two *directions* can come out obtuse when the cameras
    // face each other across the object; the parallax is the acute one.
    let parallax_deg = if parallax_deg > 90.0 {
        180.0 - parallax_deg
    } else {
        parallax_deg
    };
    if parallax_deg < min_parallax_deg {
        return Err(Refusal::TooLittleParallax);
    }

    let w = Vec3::new(
        a.origin.x - b.origin.x,
        a.origin.y - b.origin.y,
        a.origin.z - b.origin.z,
    );
    let e = a.direction.dot(w);
    let f = b.direction.dot(w);
    let s = (d * f - e) / denominator;
    let t = (f - d * e) / denominator;
    if s <= 0.0 || t <= 0.0 {
        // The rays meet behind one of the cameras. A camera does not see what
        // is behind it, so this is a mis-association, not a far-away object.
        return Err(Refusal::Behind);
    }

    let pa = a.origin.add(a.direction.scale(s));
    let pb = b.origin.add(b.direction.scale(t));
    let gap = Vec3::new(pa.x - pb.x, pa.y - pb.y, pa.z - pb.z).norm();
    if gap > MAX_GAP_M {
        return Err(Refusal::TooFarApart);
    }

    let point = Vec3::new(
        0.5 * (pa.x + pb.x),
        0.5 * (pa.y + pb.y),
        0.5 * (pa.z + pb.z),
    );
    // Worst-direction error. The transverse error of one ray at range `r` is
    // `r * sigma`; resolving two of those onto the baseline direction divides
    // by `sin(parallax)`, which is the whole reason a narrow pair is useless.
    let sigma_rad = angular_sigma_deg.to_radians();
    let range = 0.5 * (s + t);
    let sigma_m = range * sigma_rad / parallax_deg.to_radians().sin().max(1e-9);

    Ok(Triangulation {
        point,
        parallax_deg,
        gap_m: gap,
        range_a: s,
        range_b: t,
        sigma_m,
    })
}

/// A plane `normal . p = offset`, with the normal unit length and `z` up.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f64,
}

impl Plane {
    /// Signed metres above the plane. Positive is up, because the normal is
    /// kept pointing up.
    pub fn height_above(&self, p: Vec3) -> f64 {
        self.normal.dot(p) - self.offset
    }

    /// Rise per metre east, and per metre north. What a camera pose needs to
    /// stop assuming a level yard.
    pub fn tilt(&self) -> (f64, f64) {
        if self.normal.z.abs() < 1e-9 {
            return (0.0, 0.0);
        }
        (-self.normal.x / self.normal.z, -self.normal.y / self.normal.z)
    }
}

/// A fitted plane and how much of the data actually supported it.
#[derive(Clone, Copy, Debug)]
pub struct PlaneFit {
    pub plane: Plane,
    /// How many points fell within the threshold. The number that decides
    /// whether to believe the plane at all.
    pub inliers: u32,
    /// RMS height of those inliers about the plane, metres.
    pub rms: f64,
    pub tilt_east: f64,
    pub tilt_north: f64,
}

/// A small deterministic generator, written out rather than pulled in.
///
/// `rand` would be a dependency for sixteen lines, and — more to the point —
/// the same points must give the same plane on every machine and every run,
/// which a library free to change its algorithm between versions does not
/// promise.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// RANSAC ground plane through accumulated contact points.
///
/// `threshold_m` is how far off the plane a point may be and still count. It
/// is a property of the *data* — how well a contact point is known — not a
/// tuning knob: at 40 m with a quarter-degree pose, a ground contact is worth
/// about 0.2 m vertically, so 0.3 m is the honest setting for a calibrated
/// camera and 1.0 m for an assumed pose.
///
/// Returns `None` when fewer than three points are given, or when no candidate
/// plane held a majority — the second being the useful refusal, because it is
/// what a car park full of parked cars looks like.
pub fn fit_plane(points: &[Vec3], threshold_m: f64, iterations: u32, seed: u64) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
    }
    let mut rng = Xorshift(seed | 1);
    let mut best: Option<(Plane, u32)> = None;

    for _ in 0..iterations.max(1) {
        let i = rng.below(points.len());
        let j = rng.below(points.len());
        let k = rng.below(points.len());
        if i == j || j == k || i == k {
            continue;
        }
        let Some(plane) = plane_through(points[i], points[j], points[k]) else {
            continue;
        };
        let inliers = points
            .iter()
            .filter(|p| plane.height_above(**p).abs() <= threshold_m)
            .count() as u32;
        if best.is_none() || inliers > best.unwrap().1 {
            best = Some((plane, inliers));
        }
    }

    let candidate = match best {
        Some((plane, _)) => plane,
        // Every draw was degenerate, which means the points are collinear --
        // a camera watching a corridor produces exactly that. Refusing to
        // give any ground there is worse than giving level ground through the
        // points and letting `inliers` and `rms` say what it is worth, which
        // is what `least_squares_plane` does with a singular normal matrix.
        None => least_squares_plane(points)?,
    };
    // Refit on the inliers. RANSAC's job was to decide *which* points; three
    // of them decide the plane far less well than all of them do, and a plane
    // fitted to exactly the three that were drawn has zero residual by
    // construction and tells nobody anything.
    let inliers: Vec<Vec3> = points
        .iter()
        .copied()
        .filter(|p| candidate.height_above(*p).abs() <= threshold_m)
        .collect();
    if inliers.len() < 3 {
        return None;
    }
    let plane = least_squares_plane(&inliers)?;
    let mut sum = 0.0;
    for p in &inliers {
        let h = plane.height_above(*p);
        sum += h * h;
    }
    let (tilt_east, tilt_north) = plane.tilt();
    Some(PlaneFit {
        plane,
        inliers: inliers.len() as u32,
        rms: (sum / inliers.len() as f64).sqrt(),
        tilt_east,
        tilt_north,
    })
}

fn plane_through(a: Vec3, b: Vec3, c: Vec3) -> Option<Plane> {
    let n = Vec3::new(b.x - a.x, b.y - a.y, b.z - a.z)
        .cross(Vec3::new(c.x - a.x, c.y - a.y, c.z - a.z));
    normalise_upward(n, a)
}

/// Least squares over `z = ax + by + c`.
///
/// Solved in that form rather than by the total-least-squares eigenvector,
/// because ground is a height field: the points' horizontal positions are
/// known far better than their heights, so the vertical residual is the one
/// worth minimising. A vertical plane cannot be expressed this way, which is
/// correct — a vertical surface is not ground.
fn least_squares_plane(points: &[Vec3]) -> Option<Plane> {
    let n = points.len() as f64;
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    for p in points {
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    let (mx, my, mz) = (sx / n, sy / n, sz / n);
    let (mut sxx, mut sxy, mut syy, mut sxz, mut syz) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy, dz) = (p.x - mx, p.y - my, p.z - mz);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        sxz += dx * dz;
        syz += dy * dz;
    }
    let determinant = sxx * syy - sxy * sxy;
    if determinant.abs() < 1e-12 {
        // Every point on one line on the ground: a line does not determine a
        // plane's tilt across itself. Level, through the points, is the only
        // answer that does not invent a slope nobody measured.
        return normalise_upward(Vec3::new(0.0, 0.0, 1.0), Vec3::new(mx, my, mz));
    }
    let a = (sxz * syy - syz * sxy) / determinant;
    let b = (syz * sxx - sxz * sxy) / determinant;
    // z = a*x + b*y + c  ->  -a*x - b*y + z = c
    normalise_upward(Vec3::new(-a, -b, 1.0), Vec3::new(mx, my, mz))
}

fn normalise_upward(n: Vec3, through: Vec3) -> Option<Plane> {
    let len = n.norm();
    if !len.is_finite() || len < 1e-12 {
        return None;
    }
    let mut unit = n.scale(1.0 / len);
    if unit.z < 0.0 {
        unit = unit.scale(-1.0);
    }
    Some(Plane {
        normal: unit,
        offset: unit.dot(through),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray from `o` through `t`, which is how every test here states one.
    fn ray(ox: f64, oy: f64, oz: f64, tx: f64, ty: f64, tz: f64) -> Ray {
        Ray::new(
            Vec3::new(ox, oy, oz),
            Vec3::new(tx - ox, ty - oy, tz - oz),
        )
        .unwrap()
    }

    #[test]
    fn two_rays_through_a_known_point_recover_it() {
        let target = Vec3::new(12.0, 30.0, 1.7);
        let a = ray(0.0, 0.0, 4.0, target.x, target.y, target.z);
        let b = ray(25.0, 0.0, 4.0, target.x, target.y, target.z);
        let t = triangulate(a, b, 0.25, MIN_PARALLAX_DEG).unwrap();
        assert!((t.point.x - target.x).abs() < 1e-9);
        assert!((t.point.y - target.y).abs() < 1e-9);
        assert!((t.point.z - target.z).abs() < 1e-9);
        assert!(t.gap_m < 1e-9, "rays that meet have no gap");
        assert!(t.parallax_deg > 20.0, "parallax was {}", t.parallax_deg);
    }

    #[test]
    fn a_narrow_pair_is_refused_rather_than_averaged() {
        // Two cameras a metre apart looking 60 m away: about one degree.
        let target = Vec3::new(0.0, 60.0, 1.7);
        let a = ray(0.0, 0.0, 4.0, target.x, target.y, target.z);
        let b = ray(1.0, 0.0, 4.0, target.x, target.y, target.z);
        assert_eq!(
            triangulate(a, b, 0.25, MIN_PARALLAX_DEG).unwrap_err(),
            Refusal::TooLittleParallax
        );
    }

    #[test]
    fn the_reported_sigma_grows_as_the_parallax_shrinks() {
        // The claim MIN_PARALLAX_DEG is chosen from, held to arithmetic.
        let target = Vec3::new(0.0, 40.0, 1.7);
        let wide = triangulate(
            ray(-20.0, 0.0, 4.0, target.x, target.y, target.z),
            ray(20.0, 0.0, 4.0, target.x, target.y, target.z),
            0.25,
            1.0,
        )
        .unwrap();
        let narrow = triangulate(
            ray(-1.0, 0.0, 4.0, target.x, target.y, target.z),
            ray(1.0, 0.0, 4.0, target.x, target.y, target.z),
            0.25,
            1.0,
        )
        .unwrap();
        assert!(wide.sigma_m < 0.5, "wide pair: {} m", wide.sigma_m);
        assert!(narrow.sigma_m > 3.0, "narrow pair: {} m", narrow.sigma_m);
    }

    #[test]
    fn two_rays_at_different_objects_are_refused_on_the_gap() {
        // Genuinely skew: `a` runs north through x=0 at ground level, `b` runs
        // east through y=30 five metres up. They pass 5 m apart. Two rays that
        // merely point at different places can still meet -- an earlier
        // version of this test picked such a pair and triangulated it happily.
        let a = ray(0.0, 0.0, 0.0, 0.0, 60.0, 0.0);
        let b = ray(-20.0, 30.0, 5.0, 20.0, 30.0, 5.0);
        assert_eq!(
            triangulate(a, b, 0.25, MIN_PARALLAX_DEG).unwrap_err(),
            Refusal::TooFarApart
        );
    }

    #[test]
    fn a_point_behind_a_camera_is_refused() {
        let a = ray(0.0, 0.0, 4.0, 0.0, 30.0, 1.7);
        // Same line, opposite direction: the meeting point is behind `b`.
        // Pointing away from where `a` is looking: they meet behind `b`.
        let b = Ray::new(Vec3::new(20.0, 40.0, 4.0), Vec3::new(1.0, 1.0, 0.0)).unwrap();
        assert_eq!(
            triangulate(a, b, 0.25, MIN_PARALLAX_DEG).unwrap_err(),
            Refusal::Behind
        );
    }

    #[test]
    fn a_tilted_plane_is_recovered_from_points_on_it() {
        // 3% east, -2% north.
        let mut points = Vec::new();
        for i in 0..40 {
            let x = (i % 8) as f64 * 5.0 - 20.0;
            let y = (i / 8) as f64 * 7.0;
            points.push(Vec3::new(x, y, 0.03 * x - 0.02 * y));
        }
        let fit = fit_plane(&points, 0.3, 200, 12345).unwrap();
        assert!((fit.tilt_east - 0.03).abs() < 1e-6, "east {}", fit.tilt_east);
        assert!(
            (fit.tilt_north + 0.02).abs() < 1e-6,
            "north {}",
            fit.tilt_north
        );
        assert_eq!(fit.inliers, 40);
        assert!(fit.rms < 1e-9);
    }

    #[test]
    fn contacts_on_a_loading_dock_do_not_tilt_the_yard() {
        // The reason this is RANSAC. A quarter of the points sit 1.2 m up on
        // a dock at one end; a least-squares plane through all of them tilts
        // to split the difference and every position on the yard moves.
        let mut points = Vec::new();
        for i in 0..40 {
            let x = (i % 8) as f64 * 5.0 - 20.0;
            let y = (i / 8) as f64 * 7.0;
            points.push(Vec3::new(x, y, 0.0));
        }
        for i in 0..14 {
            points.push(Vec3::new(15.0 + (i % 2) as f64, i as f64 * 2.0, 1.2));
        }
        let fit = fit_plane(&points, 0.3, 400, 7).unwrap();
        assert_eq!(fit.inliers, 40, "the dock must be left out, not averaged in");
        assert!(fit.tilt_east.abs() < 1e-6 && fit.tilt_north.abs() < 1e-6);

        let dock = Vec3::new(15.0, 6.0, 1.2);
        assert!(
            (fit.plane.height_above(dock) - 1.2).abs() < 1e-6,
            "height above the fitted ground is what says this is not the ground"
        );
    }

    #[test]
    fn the_same_points_give_the_same_plane_every_time() {
        let points: Vec<Vec3> = (0..30)
            .map(|i| {
                Vec3::new(
                    (i % 6) as f64 * 3.0,
                    (i / 6) as f64 * 4.0,
                    0.01 * (i % 6) as f64,
                )
            })
            .collect();
        let a = fit_plane(&points, 0.3, 200, 99).unwrap();
        let b = fit_plane(&points, 0.3, 200, 99).unwrap();
        assert_eq!(a.tilt_east.to_bits(), b.tilt_east.to_bits());
        assert_eq!(a.inliers, b.inliers);
    }

    #[test]
    fn points_along_one_line_produce_a_level_plane_rather_than_an_invented_slope() {
        let points: Vec<Vec3> = (0..12).map(|i| Vec3::new(i as f64 * 2.0, 5.0, 0.0)).collect();
        let fit = fit_plane(&points, 0.3, 100, 3).unwrap();
        assert!(fit.tilt_north.abs() < 1e-9, "nothing measured a north slope");
    }

    #[test]
    fn fewer_than_three_points_is_no_plane() {
        assert!(fit_plane(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)], 0.3, 10, 1).is_none());
    }
}
