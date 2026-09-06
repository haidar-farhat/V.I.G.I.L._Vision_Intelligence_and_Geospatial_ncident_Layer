//! The camera model: a real pinhole, and what its answers are worth.
//!
//! # What was wrong before
//!
//! v1's Rust core and the v2 Python port of it both computed a ray like this:
//!
//! ```text
//! yaw       = atan(dx * tan(hfov/2))
//! elevation = pitch + atan(dy * tan(vfov/2))
//! ```
//!
//! and called it "rectilinear". It is not. Treating yaw and elevation as
//! independent describes a sensor bent into a cylinder around the vertical
//! axis, and a camera does not have one: a rectilinear lens puts the scene on
//! a *plane*, and on a plane the two axes are coupled through the tilt. The
//! error is zero along the centre row and grows into the corners with the
//! pitch, which is exactly where a mast camera does its work.
//!
//! Measured on this product's own default pose — 4 m mast, 25 degrees down,
//! 62 by 36 degrees — the bottom corner of the frame comes out 4.8 degrees
//! wrong in elevation, which is 21% of the distance to the thing standing
//! there. A person at the corner of the frame was being placed on the map
//! about a metre and a half from where they were.
//!
//! `roll` made it worse by being ignored. `CameraPose` has carried a roll
//! field since v1 ABI 1; nothing has ever read it. A camera clamped a few
//! degrees off level — which is most cameras on most masts — had that error
//! silently folded into every position it produced.
//!
//! # What this does instead
//!
//! One orthonormal camera basis in world ENU (east, north, up), built from
//! heading, pitch and roll; one ray `x*right + y*up + forward`; one
//! intersection with the ground plane. Forward and inverse are then exact
//! inverses of each other by construction rather than by two hand-derived
//! formulas that have to be kept in step.
//!
//! # Conventions, stated once
//!
//! - World frame is **ENU**: `+x` east, `+y` north, `+z` up. Metres.
//! - `heading` is degrees **clockwise from true north**, the compass sense.
//! - `pitch` is the **elevation of the optical axis**: negative looks down.
//! - `roll` is degrees **clockwise about the optical axis as seen from behind
//!   the camera**, which is the direction a horizon tips when you tilt your
//!   head to the right.
//! - Image coordinates are normalised `[0, 1]`, `(0, 0)` **top-left**.
//! - The ground is the plane `z = 0`; the camera is at `z = mount_height`.
//!
//! # What it is still assuming
//!
//! A flat ground plane and a distortion-free lens. Both are stated, neither
//! is hidden, and both bound the result rather than being corrected for:
//! [`ProjectionUncertainty`] carries a terrain term so a slope shows up as
//! error rather than as a confident wrong answer, and
//! [`Intrinsics::from_fov`] is documented as the fallback for a camera whose
//! calibration nobody has measured.

use crate::lens::Distortion;
use crate::geodesy::{
    angle_difference, destination_point, distance_meters, normalize_degrees, LatLon, LocalFrame,
};

/// Below this depression angle a ray is refused rather than projected. At 2
/// degrees a 6 m mast reaches 172 m, and one pixel of contact-point error is
/// worth 3 m of range there: the answer is not wrong so much as meaningless.
pub const MIN_DEPRESSION_DEG: f64 = 2.0;

/// 1-sigma angular error assumed for a detection's ground-contact point when
/// the caller states none. Three quarters of a degree is roughly 13 px on a
/// 1080-line frame at 36 degrees vertical: about what a bounding box's bottom
/// edge is worth against a real foot.
pub const DEFAULT_CONTACT_SIGMA_DEG: f64 = 0.75;

/// 1-sigma error assumed for an operator-typed heading or pitch. A camera
/// aimed by eye and a compass is not better than this, and pretending
/// otherwise is how a map gains precision nobody measured.
pub const DEFAULT_ANGLE_SIGMA_DEG: f64 = 2.0;

/// 1-sigma error assumed for a tape-measured mount height.
pub const DEFAULT_HEIGHT_SIGMA_M: f64 = 0.15;

/// 1-sigma ground-plane departure, as a fraction of ground distance. 2% is a
/// 1-in-50 slope: a yard that drains. It is the term that says "the ground is
/// not actually flat" out loud instead of assuming it away.
pub const DEFAULT_TERRAIN_SLOPE: f64 = 0.02;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn scale(self, k: f64) -> Vec3 {
        Vec3::new(self.x * k, self.y * k, self.z * k)
    }
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

/// What a lens does to angles, in normalised image coordinates.
///
/// Held as the tangent of each half-angle because that is what the projection
/// actually multiplies by; storing degrees would mean a `tan` per pixel.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    /// tan(hfov/2): half-width of the image plane at unit depth.
    pub tan_half_h: f64,
    /// tan(vfov/2): half-height of the image plane at unit depth.
    pub tan_half_v: f64,
}

impl Intrinsics {
    /// From the two fields of view an operator can read off a datasheet.
    ///
    /// This is the fallback, and it is worth naming as one. A datasheet FOV
    /// is nominal, it is quoted for the widest zoom, and it says nothing
    /// about the lens's radial distortion — which on a 90-degree security
    /// lens moves a corner by several percent. When somebody has actually
    /// calibrated a camera, `fx`/`fy` from that calibration belong here
    /// instead; the rest of the model does not change.
    pub fn from_fov(horizontal_fov_deg: f64, vertical_fov_deg: f64) -> Self {
        Self {
            tan_half_h: (horizontal_fov_deg.to_radians() / 2.0).tan(),
            tan_half_v: (vertical_fov_deg.to_radians() / 2.0).tan(),
        }
    }

    /// From a calibrated pinhole matrix and the frame it was measured on.
    pub fn from_matrix(fx: f64, fy: f64, width: f64, height: f64) -> Self {
        Self {
            tan_half_h: (width / 2.0) / fx,
            tan_half_v: (height / 2.0) / fy,
        }
    }

    pub fn horizontal_fov_deg(&self) -> f64 {
        2.0 * self.tan_half_h.atan().to_degrees()
    }

    pub fn vertical_fov_deg(&self) -> f64 {
        2.0 * self.tan_half_v.atan().to_degrees()
    }
}

/// Where a camera is, where it looks, and how far its answers are worth
/// having.
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub position: LatLon,
    /// Metres above the ground plane.
    pub mount_height: f64,
    /// Degrees clockwise from true north.
    pub heading: f64,
    /// Elevation of the optical axis; negative looks down.
    pub pitch: f64,
    /// Clockwise about the optical axis, seen from behind the camera.
    pub roll: f64,
    pub horizontal_fov: f64,
    pub vertical_fov: f64,
    /// Ground distance past which a projection is refused, not clamped.
    pub range_meters: f64,
    /// What the lens does to a straight line. Default is a perfect one.
    pub lens: Distortion,
}

/// How well each input to a projection is known, 1-sigma.
///
/// Separate from the pose because a pose is a claim about the world and this
/// is a claim about the claim. They have different lifetimes: a camera that
/// gets surveyed keeps its position and gains a smaller `heading_deg`.
#[derive(Clone, Copy, Debug)]
pub struct PoseUncertainty {
    pub heading_deg: f64,
    pub pitch_deg: f64,
    pub roll_deg: f64,
    pub mount_height_m: f64,
    /// Fractional ground-plane departure over the projected distance.
    pub terrain_slope: f64,
}

impl Default for PoseUncertainty {
    fn default() -> Self {
        Self {
            heading_deg: DEFAULT_ANGLE_SIGMA_DEG,
            pitch_deg: DEFAULT_ANGLE_SIGMA_DEG,
            roll_deg: DEFAULT_ANGLE_SIGMA_DEG,
            mount_height_m: DEFAULT_HEIGHT_SIGMA_M,
            terrain_slope: DEFAULT_TERRAIN_SLOPE,
        }
    }
}

impl PoseUncertainty {
    /// Everything known perfectly. For tests that want the contact point's
    /// own error and nothing else; never for a real camera.
    pub fn exact() -> Self {
        Self {
            heading_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            mount_height_m: 0.0,
            terrain_slope: 0.0,
        }
    }
}

/// The orthonormal camera basis in world ENU, and the pose it came from.
///
/// Built once per pose and reused across every ray: the six trig calls are
/// the whole cost of a projection, and the orthophoto lattice asks for a
/// thousand rays off one pose.
#[derive(Clone, Copy, Debug)]
pub struct CameraBasis {
    /// Optical axis.
    pub forward: Vec3,
    /// Image `+x`.
    pub right: Vec3,
    /// Image `+y`: world-up when level and unrolled.
    pub up: Vec3,
    pub intrinsics: Intrinsics,
    pub mount_height: f64,
    pub lens: Distortion,
}

impl CameraBasis {
    pub fn of(pose: &CameraPose) -> Self {
        Self::build_with_lens(
            pose.heading,
            pose.pitch,
            pose.roll,
            Intrinsics::from_fov(pose.horizontal_fov, pose.vertical_fov),
            pose.mount_height,
            pose.lens,
        )
    }

    /// The basis from angles rather than from a pose, so that the Jacobian
    /// can perturb one angle without rebuilding a whole pose around it.
    pub fn build(
        heading: f64,
        pitch: f64,
        roll: f64,
        intrinsics: Intrinsics,
        mount_height: f64,
    ) -> Self {
        Self::build_with_lens(heading, pitch, roll, intrinsics, mount_height, Distortion::default())
    }

    /// The basis, with a lens. `build` is the same thing for a perfect one.
    pub fn build_with_lens(
        heading: f64,
        pitch: f64,
        roll: f64,
        intrinsics: Intrinsics,
        mount_height: f64,
        lens: Distortion,
    ) -> Self {
        let (psi, theta, phi) = (heading.to_radians(), pitch.to_radians(), roll.to_radians());
        let (sin_psi, cos_psi) = (psi.sin(), psi.cos());
        let (sin_th, cos_th) = (theta.sin(), theta.cos());
        let (sin_ph, cos_ph) = (phi.sin(), phi.cos());

        // Heading clockwise from north, pitch as elevation. ENU components.
        let forward = Vec3::new(sin_psi * cos_th, cos_psi * cos_th, sin_th);
        // Right is horizontal by construction: roll tips the image, it does
        // not move where the lens points.
        let right0 = Vec3::new(cos_psi, -sin_psi, 0.0);
        let up0 = Vec3::new(-sin_th * sin_psi, -sin_th * cos_psi, cos_th);
        // Roll clockwise seen from behind: the right axis dips.
        let right = right0.scale(cos_ph).add(up0.scale(-sin_ph));
        let up = right0.scale(sin_ph).add(up0.scale(cos_ph));
        Self {
            forward,
            right,
            up,
            intrinsics,
            mount_height,
            lens,
        }
    }

    /// The unnormalised world-frame direction through a normalised image
    /// point. Length is arbitrary and never used as a distance.
    pub fn ray(&self, u: f64, v: f64) -> Vec3 {
        let x = (2.0 * u - 1.0) * self.intrinsics.tan_half_h;
        let y = (1.0 - 2.0 * v) * self.intrinsics.tan_half_v;
        // The pixel has already been bent by the lens, so undo that first: the
        // ray belongs to the ideal coordinate, not to where the sensor
        // recorded it. In normalised camera coordinates, which is the space
        // the model is defined in — see `lens`.
        let (x, y) = self.lens.undistort(x, y);
        self.right.scale(x).add(self.up.scale(y)).add(self.forward)
    }

    /// (bearing, elevation) of that ray, in degrees. Reported for humans and
    /// for the field-of-view outline; the projection itself never needs it.
    pub fn ray_angles(&self, u: f64, v: f64) -> (f64, f64) {
        let d = self.ray(u, v);
        let horizontal = d.x.hypot(d.y);
        (
            normalize_degrees(d.x.atan2(d.y).to_degrees()),
            d.z.atan2(horizontal).to_degrees(),
        )
    }

    /// Where a ray meets the ground plane, as (east, north) metres from the
    /// camera. `None` when it points at or above the horizon.
    ///
    /// This is the whole projection: `t = h / -d_up`, and the horizontal part
    /// of `t*d`. No `tan`, no case analysis, no decoupled axes.
    pub fn ground_offset(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let d = self.ray(u, v);
        if !(d.z < 0.0) || !d.z.is_finite() {
            return None;
        }
        let t = self.mount_height / -d.z;
        if !t.is_finite() || t <= 0.0 {
            return None;
        }
        let (east, north) = (t * d.x, t * d.y);
        if !east.is_finite() || !north.is_finite() {
            return None;
        }
        Some((east, north))
    }

    /// Camera-frame coordinates of a world offset: (image x, image y, depth).
    pub fn to_camera(&self, offset: Vec3) -> (f64, f64, f64) {
        (
            offset.dot(self.right),
            offset.dot(self.up),
            offset.dot(self.forward),
        )
    }
}

/// A ground intersection and everything needed to argue about it.
#[derive(Clone, Copy, Debug)]
pub struct GroundProjection {
    pub position: LatLon,
    pub ground_distance_meters: f64,
    pub bearing_deg: f64,
    pub uncertainty: ProjectionUncertainty,
}

/// The error ellipse of a projected position, in metres, on the ground plane.
///
/// An ellipse and not a radius because the two are genuinely different: a
/// shallow ray is precise across its own direction and vague along it, and a
/// camera 40 m away that reports "plus or minus 6 m" as a circle claims a
/// sideways error it does not have. `radius_meters` keeps the single
/// conservative number for callers that need one, and it is the semi-major
/// axis: a circle that fits inside the ellipse would understate the error in
/// the direction it actually points.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionUncertainty {
    /// 1-sigma along the line of sight, metres.
    pub along_meters: f64,
    /// 1-sigma across the line of sight, metres.
    pub across_meters: f64,
    /// Bearing of the along-axis, degrees. Equal to the projection's bearing.
    pub orientation_deg: f64,
}

impl ProjectionUncertainty {
    /// The conservative single number: the semi-major axis.
    pub fn radius_meters(&self) -> f64 {
        self.along_meters.max(self.across_meters)
    }

    /// The RMS radius, for combining errors rather than for drawing them.
    pub fn rms_meters(&self) -> f64 {
        ((self.along_meters.powi(2) + self.across_meters.powi(2)) / 2.0).sqrt()
    }
}

/// A projection refused, and why. The caller has to be told which: "too
/// shallow" is a pose problem and "out of range" is a siting problem, and
/// collapsing them into `None` was how v1 and v2 both told an operator
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionFailure {
    /// The ray is at or above the horizon.
    AboveHorizon,
    /// Below `MIN_DEPRESSION_DEG`: geometrically a hit, numerically noise.
    TooShallow,
    /// Past `range_meters`.
    OutOfRange,
    /// The pose itself is not usable.
    BadPose,
}

/// Steps for the finite-difference Jacobian, per parameter, in that
/// parameter's own units.
///
/// Central differences, so truncation error is O(step^2) and round-off is
/// O(eps/step); these sit near the minimum of the sum for a projection whose
/// scale is metres. The alternative — six hand-derived partial derivatives
/// through a rotation composition — is the kind of algebra that stays wrong
/// for a year without anybody noticing, and this is checked against the
/// closed form in the tests for the parameter that has one.
const STEP_ANGLE_DEG: f64 = 1e-4;
const STEP_HEIGHT_M: f64 = 1e-4;
const STEP_IMAGE: f64 = 1e-5;

/// Project a normalised image point onto the ground.
///
/// `contact_sigma_deg` is the angular error of the image point itself; the
/// rest of the error comes from `pose_sigma`. `enforce_range` off is for
/// asking how far a camera *could* see, not for placing anything.
pub fn project_to_ground(
    pose: &CameraPose,
    u: f64,
    v: f64,
    contact_sigma_deg: f64,
    pose_sigma: &PoseUncertainty,
    enforce_range: bool,
) -> Result<GroundProjection, ProjectionFailure> {
    if !pose.position.is_valid() || !(pose.mount_height > 0.0) || !pose.mount_height.is_finite() {
        return Err(ProjectionFailure::BadPose);
    }
    if !(pose.horizontal_fov > 0.0 && pose.horizontal_fov < 180.0)
        || !(pose.vertical_fov > 0.0 && pose.vertical_fov < 180.0)
        || !(pose.range_meters > 0.0)
    {
        return Err(ProjectionFailure::BadPose);
    }
    let basis = CameraBasis::of(pose);
    let (_, elevation) = basis.ray_angles(u, v);
    if elevation >= 0.0 {
        return Err(ProjectionFailure::AboveHorizon);
    }
    if -elevation < MIN_DEPRESSION_DEG {
        return Err(ProjectionFailure::TooShallow);
    }
    let (east, north) = basis
        .ground_offset(u, v)
        .ok_or(ProjectionFailure::AboveHorizon)?;
    let distance = east.hypot(north);
    if enforce_range && distance > pose.range_meters {
        return Err(ProjectionFailure::OutOfRange);
    }
    let bearing = normalize_degrees(east.atan2(north).to_degrees());
    let uncertainty = ground_covariance(pose, u, v, contact_sigma_deg, pose_sigma, east, north);
    Ok(GroundProjection {
        position: destination_point(pose.position, bearing, distance),
        ground_distance_meters: distance,
        bearing_deg: bearing,
        uncertainty,
    })
}

/// Propagate every stated input error through the projection.
///
/// `Sigma_g = J Sigma_p J^T` with `J` by central differences over
/// (heading, pitch, roll, mount height, u, v), then the terrain term added
/// along the line of sight — terrain tilts the plane the ray lands on, which
/// moves the hit away from or towards the camera and barely sideways.
///
/// The result is rotated into (along, across) the line of sight, where the
/// two axes are nearly the eigenvectors already: the residual off-diagonal
/// term for a mast camera is under a percent of the diagonal, and reporting
/// a rotated ellipse whose axes are 1% off is a great deal more honest than
/// reporting a circle that is 300% off.
fn ground_covariance(
    pose: &CameraPose,
    u: f64,
    v: f64,
    contact_sigma_deg: f64,
    sigma: &PoseUncertainty,
    east: f64,
    north: f64,
) -> ProjectionUncertainty {
    let intrinsics = Intrinsics::from_fov(pose.horizontal_fov, pose.vertical_fov);
    let at = |heading: f64, pitch: f64, roll: f64, height: f64, du: f64, dv: f64| {
        CameraBasis::build_with_lens(heading, pitch, roll, intrinsics, height, pose.lens)
            .ground_offset(u + du, v + dv)
    };

    // A contact point's angular error, expressed in the image coordinates it
    // is actually measured in. The vertical axis carries the range error and
    // the two half-angles differ, so one shared value would be wrong on one
    // axis or the other.
    let sigma_u = contact_sigma_deg.to_radians() / (2.0 * intrinsics.tan_half_h.atan());
    let sigma_v = contact_sigma_deg.to_radians() / (2.0 * intrinsics.tan_half_v.atan());

    let (h, p, r, m) = (pose.heading, pose.pitch, pose.roll, pose.mount_height);
    let columns: [(f64, f64, [Option<(f64, f64)>; 2]); 6] = [
        (
            sigma.heading_deg,
            STEP_ANGLE_DEG,
            [
                at(h + STEP_ANGLE_DEG, p, r, m, 0.0, 0.0),
                at(h - STEP_ANGLE_DEG, p, r, m, 0.0, 0.0),
            ],
        ),
        (
            sigma.pitch_deg,
            STEP_ANGLE_DEG,
            [
                at(h, p + STEP_ANGLE_DEG, r, m, 0.0, 0.0),
                at(h, p - STEP_ANGLE_DEG, r, m, 0.0, 0.0),
            ],
        ),
        (
            sigma.roll_deg,
            STEP_ANGLE_DEG,
            [
                at(h, p, r + STEP_ANGLE_DEG, m, 0.0, 0.0),
                at(h, p, r - STEP_ANGLE_DEG, m, 0.0, 0.0),
            ],
        ),
        (
            sigma.mount_height_m,
            STEP_HEIGHT_M,
            [
                at(h, p, r, m + STEP_HEIGHT_M, 0.0, 0.0),
                at(h, p, r, m - STEP_HEIGHT_M, 0.0, 0.0),
            ],
        ),
        (
            sigma_u,
            STEP_IMAGE,
            [
                at(h, p, r, m, STEP_IMAGE, 0.0),
                at(h, p, r, m, -STEP_IMAGE, 0.0),
            ],
        ),
        (
            sigma_v,
            STEP_IMAGE,
            [
                at(h, p, r, m, 0.0, STEP_IMAGE),
                at(h, p, r, m, 0.0, -STEP_IMAGE),
            ],
        ),
    ];

    let mut cov = [[0.0f64; 2]; 2];
    for (param_sigma, step, pair) in columns {
        if !(param_sigma > 0.0) {
            continue;
        }
        // A perturbation that pushes the ray over the horizon has no
        // derivative worth taking; the depression floor has already refused
        // the rays where that can happen at any distance from the horizon.
        let (Some((ep, np)), Some((em, nm))) = (pair[0], pair[1]) else {
            continue;
        };
        let de = (ep - em) / (2.0 * step) * param_sigma;
        let dn = (np - nm) / (2.0 * step) * param_sigma;
        cov[0][0] += de * de;
        cov[0][1] += de * dn;
        cov[1][0] += de * dn;
        cov[1][1] += dn * dn;
    }

    // Rotate into (across, along) the line of sight. `along` points away from
    // the camera; `across` is to its left.
    let distance = east.hypot(north);
    let (ux, uy) = if distance > 1e-9 {
        (east / distance, north / distance)
    } else {
        (0.0, 1.0)
    };
    let (px, py) = (-uy, ux);
    let along_var = ux * (cov[0][0] * ux + cov[0][1] * uy) + uy * (cov[1][0] * ux + cov[1][1] * uy);
    let across_var = px * (cov[0][0] * px + cov[0][1] * py) + py * (cov[1][0] * px + cov[1][1] * py);

    // Terrain: a plane tilted by `slope` moves the hit along the ray by about
    // `distance * slope / tan(depression)`. Capped at the distance itself,
    // because "it might be twice as far as it looks" is the most a slope can
    // honestly say before the answer should simply be refused.
    let terrain = if sigma.terrain_slope > 0.0 && distance > 1e-9 {
        let depression = (pose.mount_height / distance).atan();
        (distance * sigma.terrain_slope / depression.tan().max(1e-6)).min(distance)
    } else {
        0.0
    };

    ProjectionUncertainty {
        along_meters: (along_var.max(0.0) + terrain * terrain).sqrt(),
        across_meters: across_var.max(0.0).sqrt(),
        orientation_deg: normalize_degrees(east.atan2(north).to_degrees()),
    }
}

/// Where a ground position appears in the image. The exact inverse of
/// [`project_to_ground`], unclipped: a value outside `[0, 1]` is meaningful
/// and means "off the side of the frame by this much".
///
/// `None` only when the point is behind the image plane.
pub fn image_coordinates(pose: &CameraPose, point: LatLon) -> Option<(f64, f64)> {
    let basis = CameraBasis::of(pose);
    let frame = LocalFrame::new(pose.position);
    let (east, north) = frame.to_local(point);
    // The ground is `mount_height` below the camera, and the offset has to be
    // taken in the camera's own frame or the tilt gets applied twice.
    let offset = Vec3::new(east, north, -pose.mount_height);
    let (x, y, depth) = basis.to_camera(offset);
    if !(depth > 1e-9) {
        return None;
    }
    // Normalised camera coordinates, then bent to where the sensor records
    // them, so this stays the exact inverse of `ray`, which unbends.
    let (camera_x, camera_y) = basis.lens.distort(x / depth, y / depth);
    let u = (camera_x / basis.intrinsics.tan_half_h + 1.0) / 2.0;
    let v = (1.0 - camera_y / basis.intrinsics.tan_half_v) / 2.0;
    Some((u, v))
}

/// True when a ground point is inside the frame and inside the range.
pub fn camera_sees(pose: &CameraPose, point: LatLon) -> bool {
    match image_coordinates(pose, point) {
        Some((u, v)) => {
            (0.0..=1.0).contains(&u)
                && (0.0..=1.0).contains(&v)
                && distance_meters(pose.position, point) <= pose.range_meters
        }
        None => false,
    }
}

/// Ground distance to the nearest point of the frame's bottom edge.
pub fn near_ground_distance(pose: &CameraPose) -> Option<f64> {
    edge_distance(pose, 1.0, f64::min)
}

/// Ground distance to the farthest point of the frame's top edge.
pub fn far_ground_distance(pose: &CameraPose) -> Option<f64> {
    edge_distance(pose, 0.0, f64::max)
}

/// The extreme ground distance along one horizontal edge of the frame.
///
/// Sampled across the edge rather than taken at its centre, because with roll
/// or a wide lens the nearest ground in view is at a *corner*, not below the
/// middle of the frame. v1 read the centre column and drew a footprint whose
/// near edge cut through ground the camera could see.
fn edge_distance(pose: &CameraPose, v: f64, pick: fn(f64, f64) -> f64) -> Option<f64> {
    let sigma = PoseUncertainty::exact();
    let mut best: Option<f64> = None;
    for i in 0..=16 {
        let u = i as f64 / 16.0;
        if let Ok(p) = project_to_ground(pose, u, v, 0.0, &sigma, false) {
            best = Some(match best {
                Some(current) => pick(current, p.ground_distance_meters),
                None => p.ground_distance_meters,
            });
        }
    }
    best
}

/// The ground a camera covers, as a closed ring of lat/lon.
///
/// An annular sector: a tilted camera cannot see the ground at the foot of
/// its own mast, and drawing the pie slice tells an operator the camera
/// covers ground it is blind to. Empty when the camera sees no ground.
///
/// Traced by walking the frame's border in image space and projecting, so a
/// rolled camera produces the rotated footprint it actually has rather than a
/// symmetric wedge that is right only when roll is zero.
pub fn footprint(pose: &CameraPose, segments: usize) -> Vec<LatLon> {
    let segments = segments.max(4);
    let sigma = PoseUncertainty::exact();
    let mut ring: Vec<LatLon> = Vec::with_capacity(2 * segments + 4);
    let clamp = |p: GroundProjection| -> LatLon {
        if p.ground_distance_meters <= pose.range_meters {
            p.position
        } else {
            destination_point(pose.position, p.bearing_deg, pose.range_meters)
        }
    };
    for i in 0..=segments {
        let u = i as f64 / segments as f64;
        if let Ok(p) = project_to_ground(pose, u, 0.0, 0.0, &sigma, false) {
            ring.push(clamp(p));
        }
    }
    for i in (0..=segments).rev() {
        let u = i as f64 / segments as f64;
        if let Ok(p) = project_to_ground(pose, u, 1.0, 0.0, &sigma, false) {
            ring.push(clamp(p));
        }
    }
    if ring.len() < 3 {
        return Vec::new();
    }
    ring
}

/// True when a compass bearing falls inside the camera's horizontal spread.
pub fn bearing_in_view(pose: &CameraPose, bearing_deg: f64) -> bool {
    let basis = CameraBasis::of(pose);
    let (left, _) = basis.ray_angles(0.0, 0.5);
    let (right, _) = basis.ray_angles(1.0, 0.5);
    let (mid, _) = basis.ray_angles(0.5, 0.5);
    let half = angle_difference(right, left).abs() / 2.0;
    angle_difference(bearing_deg, mid).abs() <= half
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> CameraPose {
        CameraPose {
            position: LatLon::new(33.8938, 35.5018),
            mount_height: 4.0,
            heading: 0.0,
            pitch: -25.0,
            roll: 0.0,
            horizontal_fov: 62.0,
            vertical_fov: 36.0,
            range_meters: 60.0,
            lens: Distortion::default(),
        }
    }

    #[test]
    fn the_basis_is_orthonormal_for_every_orientation() {
        for heading in [0.0, 37.0, 180.0, 300.0] {
            for pitch in [-60.0, -25.0, 0.0, 15.0] {
                for roll in [-30.0, 0.0, 12.0] {
                    let b =
                        CameraBasis::build(heading, pitch, roll, Intrinsics::from_fov(62.0, 36.0), 4.0);
                    for v in [b.forward, b.right, b.up] {
                        assert!((v.norm() - 1.0).abs() < 1e-12, "not unit: {v:?}");
                    }
                    assert!(b.forward.dot(b.right).abs() < 1e-12);
                    assert!(b.forward.dot(b.up).abs() < 1e-12);
                    assert!(b.right.dot(b.up).abs() < 1e-12);
                    // (right, up, forward) is left-handed, because a
                    // compass heading turns clockwise while ENU turns
                    // anticlockwise. The identity that holds is
                    // `up x right = forward` — the same triad OpenCV writes
                    // as (right, down, forward) and calls right-handed.
                    let c = b.up.cross(b.right);
                    assert!((c.x - b.forward.x).abs() < 1e-12);
                    assert!((c.y - b.forward.y).abs() < 1e-12);
                    assert!((c.z - b.forward.z).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn the_centre_ray_is_the_pose_whatever_the_roll() {
        let intrinsics = Intrinsics::from_fov(62.0, 36.0);
        for heading in [0.0, 91.0, 271.0] {
            for pitch in [-45.0, -10.0] {
                for roll in [-20.0, 0.0, 20.0] {
                    let b = CameraBasis::build(heading, pitch, roll, intrinsics, 4.0);
                    let (bearing, elevation) = b.ray_angles(0.5, 0.5);
                    assert!(angle_difference(bearing, heading).abs() < 1e-9);
                    assert!((elevation - pitch).abs() < 1e-9, "roll must not tilt the axis");
                }
            }
        }
    }

    #[test]
    fn a_level_camera_reduces_to_the_simple_formula() {
        // With no pitch and no roll the pinhole model and the old decoupled
        // one agree exactly. That is the case the old model was checked
        // against, which is why the error went unnoticed for two versions.
        let mut pose = reference();
        pose.pitch = 0.0;
        let basis = CameraBasis::of(&pose);
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (bearing, _) = basis.ray_angles(u, 0.5);
            let dx = 2.0 * u - 1.0;
            let expected = (dx * basis.intrinsics.tan_half_h).atan().to_degrees();
            assert!(angle_difference(bearing, expected).abs() < 1e-9);
        }
    }

    #[test]
    fn the_pinhole_model_disagrees_with_the_decoupled_one_where_it_matters() {
        // The regression this module exists for. At the bottom corner of the
        // reference pose the old formula is out by degrees, and degrees of
        // depression are metres of range.
        let pose = reference();
        let basis = CameraBasis::of(&pose);
        let (_, elevation) = basis.ray_angles(1.0, 1.0);
        let old = pose.pitch + (-basis.intrinsics.tan_half_v).atan().to_degrees();
        assert!(
            (elevation - old).abs() > 4.0,
            "expected a large disagreement, got {elevation} vs {old}"
        );
        let correct = pose.mount_height / (-elevation).to_radians().tan();
        let wrong = pose.mount_height / (-old).to_radians().tan();
        assert!(
            (correct - wrong).abs() / correct > 0.15,
            "corner range error should exceed 15%: {correct} vs {wrong}"
        );
    }

    #[test]
    fn forward_and_inverse_are_exact_inverses() {
        for roll in [-15.0, 0.0, 8.0] {
            for heading in [0.0, 47.0, 300.0] {
                let mut pose = reference();
                pose.roll = roll;
                pose.heading = heading;
                for &(u, v) in &[(0.5, 0.7), (0.05, 0.95), (0.95, 0.6), (0.2, 0.55), (0.8, 0.99)] {
                    let p = project_to_ground(
                        &pose,
                        u,
                        v,
                        DEFAULT_CONTACT_SIGMA_DEG,
                        &PoseUncertainty::default(),
                        false,
                    )
                    .expect("should project");
                    let (bu, bv) = image_coordinates(&pose, p.position).expect("in front");
                    assert!(
                        (bu - u).abs() < 1e-6 && (bv - v).abs() < 1e-6,
                        "roll {roll} heading {heading} ({u},{v}) -> ({bu},{bv})"
                    );
                }
            }
        }
    }

    #[test]
    fn distance_matches_the_closed_form_down_the_centre_column() {
        // Down the middle of the frame the two models do agree, so there is a
        // closed form to check the whole pipeline against end to end.
        let pose = reference();
        let basis = CameraBasis::of(&pose);
        for v in [0.55, 0.7, 0.9, 1.0] {
            let p = project_to_ground(&pose, 0.5, v, 0.0, &PoseUncertainty::exact(), false).unwrap();
            let (_, elevation) = basis.ray_angles(0.5, v);
            let expected = pose.mount_height / (-elevation).to_radians().tan();
            assert!((p.ground_distance_meters - expected).abs() < 1e-9);
            assert!((distance_meters(pose.position, p.position) - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn roll_moves_a_position_and_the_amount_is_not_small() {
        let level = reference();
        let mut rolled = reference();
        rolled.roll = 10.0;
        let a = project_to_ground(&level, 0.15, 0.9, 0.0, &PoseUncertainty::exact(), false).unwrap();
        let b = project_to_ground(&rolled, 0.15, 0.9, 0.0, &PoseUncertainty::exact(), false).unwrap();
        let moved = distance_meters(a.position, b.position);
        // Measured: 0.98 m at this pose. The threshold is well under it so
        // the test states "roll is not negligible" rather than pinning a
        // number that a change of default FOV would move.
        assert!(moved > 0.5, "10 degrees of roll moved a corner by {moved} m");
    }

    #[test]
    fn uncertainty_grows_with_range_and_is_longer_than_it_is_wide() {
        let pose = reference();
        let sigma = PoseUncertainty::default();
        let near =
            project_to_ground(&pose, 0.5, 0.95, DEFAULT_CONTACT_SIGMA_DEG, &sigma, false).unwrap();
        let far =
            project_to_ground(&pose, 0.5, 0.55, DEFAULT_CONTACT_SIGMA_DEG, &sigma, false).unwrap();
        assert!(far.ground_distance_meters > near.ground_distance_meters);
        assert!(far.uncertainty.radius_meters() > near.uncertainty.radius_meters());
        // A shallow ray is vague about range and sharp about direction.
        assert!(far.uncertainty.along_meters > far.uncertainty.across_meters);
    }

    #[test]
    fn a_perfect_pose_still_has_the_contact_points_own_error() {
        let pose = reference();
        let p = project_to_ground(
            &pose,
            0.5,
            0.8,
            DEFAULT_CONTACT_SIGMA_DEG,
            &PoseUncertainty::exact(),
            false,
        )
        .unwrap();
        assert!(p.uncertainty.along_meters > 0.0 && p.uncertainty.across_meters > 0.0);
        let exact = project_to_ground(&pose, 0.5, 0.8, 0.0, &PoseUncertainty::exact(), false).unwrap();
        assert!(exact.uncertainty.radius_meters() < 1e-9);
    }

    #[test]
    fn the_finite_difference_jacobian_matches_the_closed_form_for_height() {
        // Ground offset is exactly proportional to mount height, so the height
        // column of the Jacobian has a closed form: dg/dh = g/h. If the
        // numerical derivative is right here it is right in general.
        let pose = reference();
        let sigma = PoseUncertainty {
            mount_height_m: 0.15,
            ..PoseUncertainty::exact()
        };
        let p = project_to_ground(&pose, 0.5, 0.8, 0.0, &sigma, false).unwrap();
        let expected = p.ground_distance_meters / pose.mount_height * 0.15;
        assert!(
            (p.uncertainty.along_meters - expected).abs() < 1e-6,
            "{} vs {expected}",
            p.uncertainty.along_meters
        );
        assert!(
            p.uncertainty.across_meters < 1e-9,
            "height is a range error only"
        );
    }

    #[test]
    fn rays_outside_the_useful_band_are_refused_by_name() {
        let mut level = reference();
        level.pitch = 0.0;
        assert_eq!(
            project_to_ground(&level, 0.5, 0.5, 0.0, &PoseUncertainty::exact(), true).err(),
            Some(ProjectionFailure::AboveHorizon)
        );
        let mut shallow = reference();
        shallow.pitch = -1.0;
        shallow.vertical_fov = 1.0;
        assert_eq!(
            project_to_ground(&shallow, 0.5, 0.5, 0.0, &PoseUncertainty::exact(), true).err(),
            Some(ProjectionFailure::TooShallow)
        );
        let mut short = reference();
        short.range_meters = 3.0;
        assert_eq!(
            project_to_ground(&short, 0.5, 0.6, 0.0, &PoseUncertainty::exact(), true).err(),
            Some(ProjectionFailure::OutOfRange)
        );
        let mut broken = reference();
        broken.mount_height = 0.0;
        assert_eq!(
            project_to_ground(&broken, 0.5, 0.6, 0.0, &PoseUncertainty::exact(), true).err(),
            Some(ProjectionFailure::BadPose)
        );
    }

    #[test]
    fn the_footprint_is_a_ring_that_excludes_the_mast() {
        let pose = reference();
        let ring = footprint(&pose, 8);
        assert!(ring.len() >= 18);
        let nearest = ring
            .iter()
            .map(|p| distance_meters(pose.position, *p))
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest > 1.0,
            "a tilted camera does not see its own feet: {nearest}"
        );
        let farthest = ring
            .iter()
            .map(|p| distance_meters(pose.position, *p))
            .fold(0.0, f64::max);
        assert!(farthest <= pose.range_meters + 1e-6);
        let mut sky = reference();
        sky.pitch = 30.0;
        assert!(footprint(&sky, 8).is_empty());
    }

    #[test]
    fn near_distance_is_taken_across_the_edge_not_at_its_centre() {
        // With roll, the nearest ground in view is at a corner. Reading the
        // centre column alone reports ground the camera cannot see as its
        // near limit.
        let mut pose = reference();
        pose.roll = 20.0;
        let across = near_ground_distance(&pose).unwrap();
        let centre = project_to_ground(&pose, 0.5, 1.0, 0.0, &PoseUncertainty::exact(), false)
            .unwrap()
            .ground_distance_meters;
        assert!(across <= centre + 1e-9);
        assert!(
            centre - across > 0.1,
            "roll should move the near point off centre"
        );
    }
}
