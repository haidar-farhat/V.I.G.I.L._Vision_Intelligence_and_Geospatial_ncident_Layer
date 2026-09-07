//! Deterministic spatial mathematics.
//!
//! Pure and allocation-light: this runs per detection, per frame, per camera, so
//! it sits on the hottest path in the system. It is also the layer that decides
//! where the map says something happened, which makes it the layer most able to
//! be confidently wrong.
//!
//! The governing principle throughout is that **uncertainty travels with the
//! position**. Ground range is `h / tan(theta)`, so error grows super-linearly as
//! a ray flattens toward the horizon: a detection at a camera's feet may be
//! metre-accurate while one near the horizon is uncertain by tens of metres.
//! Rendering both as identical dots is a lie, so every projected position carries
//! the error the geometry actually implies, and a ray that cannot meet the ground
//! returns nothing rather than a clamped guess.

pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Assumed angular error of a ray, combining detection jitter and pose error.
pub const DEFAULT_ANGULAR_UNCERTAINTY_DEG: f64 = 1.5;

/// Rays flatter than this are treated as unusable rather than projected.
pub const MIN_DEPRESSION_ANGLE_DEG: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLon {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

/// Detection box in normalised image space: 0..1, origin top-left.
///
/// Normalised rather than pixels so a box survives a resolution change, a
/// sub-stream switch, or a model input-size change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl BoundingBox {
    /// Where the object meets the ground: bottom edge, horizontal centre.
    ///
    /// For a standing person or a vehicle this is the contact patch, which is the
    /// only part of the box whose ground position means anything.
    pub fn ground_contact(&self) -> Vec2 {
        Vec2 {
            x: self.x + self.w / 2.0,
            y: self.y + self.h,
        }
    }

    pub fn center(&self) -> Vec2 {
        Vec2 {
            x: self.x + self.w / 2.0,
            y: self.y + self.h / 2.0,
        }
    }

    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);

        let w = x2 - x1;
        let h = y2 - y1;
        if w <= 0.0 || h <= 0.0 {
            return 0.0;
        }

        let intersection = w * h;
        let union = self.w * self.h + other.w * other.h - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// Camera pose and optics, enough to project image points onto the ground plane.
///
/// Angles in degrees. `heading` is a compass bearing (0 = north, 90 = east).
/// `pitch` is negative when looking down, which is the normal mounting.
#[derive(Debug, Clone, Copy)]
pub struct CameraPose {
    pub position: LatLon,
    /// Lens height above the local ground plane, metres.
    pub mount_height: f64,
    pub heading: f64,
    pub pitch: f64,
    pub roll: f64,
    pub horizontal_fov: f64,
    pub vertical_fov: f64,
    /// Useful observation distance. Beyond this, detections are not mapped.
    pub range_meters: f64,
}

/// How a position was obtained. Kept with the position, never separated from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSource {
    /// Projected through a calibrated pose onto the ground plane.
    GroundProjection,
    /// The camera's own position, used when projection is impossible.
    CameraFallback,
}

/// A map position together with an honest statement of how well it is known.
#[derive(Debug, Clone, Copy)]
pub struct PositionEstimate {
    pub point: LatLon,
    /// 1-sigma horizontal uncertainty, metres.
    pub radius_meters: f64,
    pub source: PositionSource,
}

// ---------------------------------------------------------------------- geodesy

/// Metres per degree of latitude. Series expansion of the WGS84 meridian arc;
/// sub-centimetre across a site.
pub fn meters_per_degree_latitude(latitude_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    111_132.92 - 559.82 * (2.0 * lat).cos() + 1.175 * (4.0 * lat).cos() - 0.0023 * (6.0 * lat).cos()
}

pub fn meters_per_degree_longitude(latitude_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos() + 0.118 * (5.0 * lat).cos()
}

/// A local east-north-up frame anchored at a site origin.
///
/// Built once per site and reused: the conversion factors depend on latitude, and
/// recomputing them per point would be both slower and subtly inconsistent
/// between a polygon and the point being tested against it.
#[derive(Debug, Clone, Copy)]
pub struct LocalFrame {
    pub origin: LatLon,
    pub meters_per_lat: f64,
    pub meters_per_lon: f64,
}

impl LocalFrame {
    pub fn new(origin: LatLon) -> Self {
        Self {
            origin,
            meters_per_lat: meters_per_degree_latitude(origin.lat),
            meters_per_lon: meters_per_degree_longitude(origin.lat),
        }
    }

    /// Geographic to local metres (x = east, y = north).
    pub fn to_local(&self, point: LatLon) -> Vec2 {
        Vec2 {
            x: (point.lon - self.origin.lon) * self.meters_per_lon,
            y: (point.lat - self.origin.lat) * self.meters_per_lat,
        }
    }

    /// Local metres to geographic. Exact inverse of `to_local`.
    pub fn to_lat_lon(&self, local: Vec2) -> LatLon {
        LatLon {
            lat: self.origin.lat + local.y / self.meters_per_lat,
            lon: self.origin.lon + local.x / self.meters_per_lon,
        }
    }
}

/// Great-circle distance in metres.
pub fn haversine_distance(a: LatLon, b: LatLon) -> f64 {
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let d_lat = lat2 - lat1;
    let d_lon = (b.lon - a.lon).to_radians();

    let sin_lat = (d_lat / 2.0).sin();
    let sin_lon = (d_lon / 2.0).sin();
    let h = sin_lat * sin_lat + lat1.cos() * lat2.cos() * sin_lon * sin_lon;

    2.0 * EARTH_RADIUS_M * h.sqrt().min(1.0).asin()
}

/// Initial compass bearing from `a` to `b`, degrees, 0 = north.
pub fn bearing_degrees(a: LatLon, b: LatLon) -> f64 {
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let d_lon = (b.lon - a.lon).to_radians();

    let y = d_lon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * d_lon.cos();

    normalize_degrees(y.atan2(x).to_degrees())
}

/// The point reached by travelling `distance_meters` from `origin` on a bearing.
pub fn destination_point(origin: LatLon, bearing_deg: f64, distance_meters: f64) -> LatLon {
    let angular = distance_meters / EARTH_RADIUS_M;
    let bearing = bearing_deg.to_radians();
    let lat1 = origin.lat.to_radians();
    let lon1 = origin.lon.to_radians();

    let sin_lat2 = lat1.sin() * angular.cos() + lat1.cos() * angular.sin() * bearing.cos();
    let lat2 = sin_lat2.clamp(-1.0, 1.0).asin();
    let lon2 = lon1
        + (bearing.sin() * angular.sin() * lat1.cos()).atan2(angular.cos() - lat1.sin() * sin_lat2);

    LatLon {
        lat: lat2.to_degrees(),
        lon: (lon2.to_degrees() + 540.0) % 360.0 - 180.0,
    }
}

/// Wrap any angle into [0, 360).
pub fn normalize_degrees(deg: f64) -> f64 {
    let wrapped = deg % 360.0;
    if wrapped < 0.0 {
        wrapped + 360.0
    } else {
        wrapped
    }
}

/// Smallest signed difference a - b, in (-180, 180].
pub fn angle_difference(a: f64, b: f64) -> f64 {
    let diff = normalize_degrees(a - b);
    if diff > 180.0 {
        diff - 360.0
    } else {
        diff
    }
}

// ------------------------------------------------------------------ projection

/// Ray direction for a normalised image point.
///
/// Uses a rectilinear (tangent) model rather than a linear angle sweep, matching
/// how a real lens maps angle to sensor position.
pub fn ray_angles(pose: &CameraPose, u: f64, v: f64) -> (f64, f64) {
    let half_h = (pose.horizontal_fov / 2.0).to_radians();
    let half_v = (pose.vertical_fov / 2.0).to_radians();

    let dx = u.clamp(0.0, 1.0) * 2.0 - 1.0;
    let dy = 1.0 - v.clamp(0.0, 1.0) * 2.0;

    let yaw = (dx * half_h.tan()).atan();
    let pitch_offset = (dy * half_v.tan()).atan();

    (
        normalize_degrees(pose.heading + yaw.to_degrees()),
        pose.pitch + pitch_offset.to_degrees(),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct GroundProjection {
    pub position: LatLon,
    pub ground_distance_meters: f64,
    pub bearing_deg: f64,
    pub uncertainty_meters: f64,
}

/// Project a normalised image point onto the ground plane.
///
/// `None` when the ray cannot meet the ground: it points at or above the horizon,
/// or the intersection lies beyond the camera's useful range. Returning nothing
/// rather than a clamped guess is deliberate — a position the system cannot
/// determine must not appear on a map at all.
pub fn project_to_ground(
    pose: &CameraPose,
    u: f64,
    v: f64,
    angular_uncertainty_deg: f64,
    enforce_range: bool,
) -> Option<GroundProjection> {
    let (bearing_deg, elevation_deg) = ray_angles(pose, u, v);

    let depression_deg = -elevation_deg;
    if depression_deg < MIN_DEPRESSION_ANGLE_DEG {
        return None;
    }

    let depression = depression_deg.to_radians();
    let ground_distance = pose.mount_height / depression.tan();
    if !ground_distance.is_finite() || ground_distance <= 0.0 {
        return None;
    }
    if enforce_range && ground_distance > pose.range_meters {
        return None;
    }

    let sigma = angular_uncertainty_deg.to_radians();
    // d = h / tan(theta)  =>  |dd/dtheta| = h / sin^2(theta)
    let sin_depression = depression.sin();
    let range_sigma = pose.mount_height * sigma / (sin_depression * sin_depression);
    let lateral_sigma = ground_distance * sigma;

    Some(GroundProjection {
        position: destination_point(pose.position, bearing_deg, ground_distance),
        ground_distance_meters: ground_distance,
        bearing_deg,
        uncertainty_meters: (range_sigma * range_sigma + lateral_sigma * lateral_sigma).sqrt(),
    })
}

/// Project a detection box to a map position, degrading honestly.
///
/// When the geometry does not permit a real projection the camera's own position
/// is returned, tagged `CameraFallback` with an uncertainty covering the whole
/// field of view. The operator still learns "something is happening at this
/// camera" — which is true — without the map implying precision that is not there.
pub fn project_detection(pose: &CameraPose, bbox: &BoundingBox) -> PositionEstimate {
    project_point(pose, bbox.ground_contact())
}

/// Project one image point — where an object meets the ground — to a map
/// position, degrading honestly. See [`project_detection`] for the fallback.
///
/// Separate from the box version because the point is now measured rather than
/// assumed: a segmentation mask gives the lowest row that has any of the object
/// in it, and for a person leaning, carrying something or half behind a car
/// that is not the bottom-centre of their rectangle.
pub fn project_point(pose: &CameraPose, contact: Vec2) -> PositionEstimate {
    match project_to_ground(
        pose,
        contact.x,
        contact.y,
        DEFAULT_ANGULAR_UNCERTAINTY_DEG,
        true,
    ) {
        Some(projection) => PositionEstimate {
            point: projection.position,
            radius_meters: projection.uncertainty_meters,
            source: PositionSource::GroundProjection,
        },
        None => PositionEstimate {
            point: pose.position,
            radius_meters: pose.range_meters,
            source: PositionSource::CameraFallback,
        },
    }
}

/// Nearest ground distance the camera sees: the bottom edge of the frame, which
/// points most steeply down and therefore lands closest to the mast.
pub fn near_ground_distance(pose: &CameraPose) -> Option<f64> {
    // `None` when even the bottom of the frame misses the ground, which means
    // the camera sees no ground at all. An earlier version answered 0.0 there,
    // and `field_of_view_wedge` read that as "coverage starts at the mast" and
    // drew the solid pie slice its own documentation forbids — claiming the
    // whole foreground for a camera pointed at the sky.
    project_to_ground(pose, 0.5, 1.0, DEFAULT_ANGULAR_UNCERTAINTY_DEG, false)
        .map(|p| p.ground_distance_meters)
}

/// Farthest ground distance: the top edge. `None` when it is above the horizon,
/// in which case the pose's stated range is what actually bounds the view.
pub fn far_ground_distance(pose: &CameraPose) -> Option<f64> {
    project_to_ground(pose, 0.5, 0.0, DEFAULT_ANGULAR_UNCERTAINTY_DEG, false)
        .map(|p| p.ground_distance_meters)
}

/// Ground footprint of a camera's field of view.
///
/// An annular sector, not a pie slice. A downward-tilted camera does not see the
/// ground at its own feet, and drawing the slice would tell an operator the
/// camera covers ground it is physically blind to — which is exactly the kind of
/// false coverage assumption that gets a site burgled.
pub fn field_of_view_wedge(pose: &CameraPose, arc_segments: usize) -> Vec<LatLon> {
    let segments = arc_segments.max(2);
    let half_fov = pose.horizontal_fov / 2.0;

    // No near edge means no ground in view at all, so there is no footprint to
    // draw. An empty ring is the honest answer and every caller already handles
    // one: the map skips it, and point-in-polygon rejects it.
    let Some(near) = near_ground_distance(pose) else {
        return Vec::new();
    };

    let far_range = far_ground_distance(pose)
        .map(|far| far.min(pose.range_meters))
        .unwrap_or(pose.range_meters);
    let near_range = near.min(far_range);

    let bearing_at = |t: f64| normalize_degrees(pose.heading - half_fov + t * pose.horizontal_fov);

    let mut points = Vec::with_capacity(segments * 2 + 2);

    for i in 0..=segments {
        let t = i as f64 / segments as f64;
        points.push(destination_point(pose.position, bearing_at(t), far_range));
    }

    if near_range > 0.5 {
        for i in (0..=segments).rev() {
            let t = i as f64 / segments as f64;
            points.push(destination_point(pose.position, bearing_at(t), near_range));
        }
    } else {
        points.push(pose.position);
    }

    points
}

/// Whether a bearing falls inside the camera's horizontal field of view.
pub fn bearing_in_fov(pose: &CameraPose, bearing_deg: f64) -> bool {
    angle_difference(bearing_deg, pose.heading).abs() <= pose.horizontal_fov / 2.0
}

/// The exact inverse of `project_to_ground`: where a ground point appears in the
/// image. Unclipped — a point outside 0..1 is meaningful and is returned as such.
pub fn image_coordinates(
    pose: &CameraPose,
    point: LatLon,
    height_meters: f64,
) -> Option<(f64, f64, f64, bool)> {
    let distance = haversine_distance(pose.position, point);
    if distance <= 0.0 {
        return None;
    }

    let bearing = bearing_degrees(pose.position, point);
    let yaw_deg = angle_difference(bearing, pose.heading);
    let half_h = pose.horizontal_fov / 2.0;
    if yaw_deg.abs() >= 90.0 {
        return None;
    }

    let elevation_deg = (height_meters - pose.mount_height)
        .atan2(distance)
        .to_degrees();
    let pitch_offset_deg = elevation_deg - pose.pitch;
    let half_v = pose.vertical_fov / 2.0;
    if pitch_offset_deg.abs() >= 90.0 {
        return None;
    }

    // A zero or negative field of view has no image plane to project onto.
    // Dividing by tan(0) returned Some((NaN, inf, ..)), which propagates
    // silently through every consumer instead of failing here.
    if half_h <= 0.0 || half_v <= 0.0 {
        return None;
    }

    let dx = yaw_deg.to_radians().tan() / half_h.to_radians().tan();
    let dy = pitch_offset_deg.to_radians().tan() / half_v.to_radians().tan();

    let in_frame = yaw_deg.abs() <= half_h
        && pitch_offset_deg.abs() <= half_v
        && distance <= pose.range_meters;

    Some(((dx + 1.0) / 2.0, (1.0 - dy) / 2.0, distance, in_frame))
}

// ----------------------------------------------------------------------- zones

/// Crossing-number point-in-polygon.
///
/// The half-open vertical test counts each vertex exactly once, so a point level
/// with a vertex is classified consistently rather than flickering — which
/// matters when a track walks along a zone edge and would otherwise emit a burst
/// of spurious enter/exit events.
pub fn point_in_polygon(point: Vec2, ring: &[Vec2]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let a = ring[i];
        let b = ring[j];

        if (a.y > point.y) != (b.y > point.y)
            && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Shortest distance from a point to a segment.
pub fn point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let len_sq = abx * abx + aby * aby;

    if len_sq == 0.0 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }

    let t = (((p.x - a.x) * abx + (p.y - a.y) * aby) / len_sq).clamp(0.0, 1.0);
    let cx = a.x + abx * t;
    let cy = a.y + aby * t;

    ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt()
}

/// Where a point sits relative to a polygon, given how well the point is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneMembership {
    /// The whole uncertainty disc lies outside the zone.
    Outside,
    /// The whole uncertainty disc lies inside the zone.
    Inside,
    /// The disc straddles the boundary. The object may be in the zone or may
    /// not, and the system does not know which.
    Uncertain,
}

/// The smallest area a ring must enclose to be treated as a zone at all.
///
/// A square 30 cm on a side. Below this a "zone" is a drawing accident — three
/// clicks in nearly the same place — and treating it as an area produces
/// intrusion events from a shape nobody meant to draw.
const MINIMUM_ZONE_AREA_M2: f64 = 0.09;

/// Twice the signed area of a polygon, halved and made positive: the shoelace
/// formula. Used only to reject degenerate rings.
fn polygon_area(ring: &[Vec2]) -> f64 {
    let mut sum = 0.0;
    for index in 0..ring.len() {
        let a = ring[index];
        let b = ring[(index + 1) % ring.len()];
        sum += a.x * b.y - b.x * a.y;
    }
    (sum / 2.0).abs()
}

/// Zone membership that accounts for how well the position is actually known.
///
/// A plain point-in-polygon test answers a question nobody asked. The position
/// it is given is an estimate with a 1σ radius that grows toward the horizon —
/// at 40 m from a mast that radius is metres across — so "is this point inside"
/// is not the same question as "is this object inside", and treating them as one
/// produces intrusion alerts from an object that was never in the zone.
///
/// Three answers instead of two. `Uncertain` is the honest one, and it exists so
/// a rule can decide what to do with it: an intrusion alarm should demand
/// `Inside` and stay silent, while a coverage report should count `Uncertain` as
/// a gap rather than as clear ground.
///
/// The boundary distance is computed in a local metric frame anchored at the
/// point, so the comparison against a radius in metres is meaningful. Over a
/// site-sized polygon that planar approximation agrees with the spherical
/// distance to well under a centimetre.
pub fn zone_membership(ring: &[LatLon], point: LatLon, uncertainty_meters: f64) -> ZoneMembership {
    if ring.len() < 3 {
        // Two points are a line, not an area. A half-drawn zone must not start
        // producing intrusion events.
        return ZoneMembership::Outside;
    }

    let frame = LocalFrame::new(point);
    let local_point = Vec2 { x: 0.0, y: 0.0 };
    let local_ring: Vec<Vec2> = ring.iter().map(|&p| frame.to_local(p)).collect();

    // Checked before containment. A ring whose vertices are all within a few
    // centimetres has three points but no area, and the zero-uncertainty path
    // below would otherwise report Inside for the point at its own centre.
    if polygon_area(&local_ring) < MINIMUM_ZONE_AREA_M2 {
        return ZoneMembership::Outside;
    }

    let inside = point_in_polygon(local_point, &local_ring);

    let radius = uncertainty_meters.max(0.0);
    if radius <= 0.0 {
        return if inside {
            ZoneMembership::Inside
        } else {
            ZoneMembership::Outside
        };
    }

    let mut nearest = f64::INFINITY;
    for index in 0..local_ring.len() {
        let a = local_ring[index];
        let b = local_ring[(index + 1) % local_ring.len()];
        nearest = nearest.min(point_to_segment(local_point, a, b));
    }

    if nearest <= radius {
        ZoneMembership::Uncertain
    } else if inside {
        ZoneMembership::Inside
    } else {
        ZoneMembership::Outside
    }
}

/// Proper segment intersection.
///
/// Deliberately excludes collinear-touching and endpoint-grazing: a track whose
/// interpolated path merely touches a tripwire should not fire it.
pub fn segments_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    let orientation = |a: Vec2, b: Vec2, c: Vec2| -> i32 {
        let v = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    };

    let o1 = orientation(p1, p2, q1);
    let o2 = orientation(p1, p2, q2);
    let o3 = orientation(q1, q2, p1);
    let o4 = orientation(q1, q2, p2);

    o1 != o2 && o3 != o4 && o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: LatLon = LatLon {
        lat: 33.8938,
        lon: 35.5018,
    };

    /// A 10 m mast looking due north, tilted 45 degrees down. At that tilt the
    /// image centre lands exactly one mount-height away, which makes every
    /// expected value below checkable by hand.
    fn pose() -> CameraPose {
        CameraPose {
            position: SITE,
            mount_height: 10.0,
            heading: 0.0,
            pitch: -45.0,
            roll: 0.0,
            horizontal_fov: 60.0,
            vertical_fov: 34.0,
            range_meters: 120.0,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: expected {expected} +/- {tolerance}, got {actual}"
        );
    }

    #[test]
    fn local_frame_round_trips() {
        let frame = LocalFrame::new(SITE);
        let point = LatLon {
            lat: SITE.lat + 0.0012,
            lon: SITE.lon - 0.0008,
        };
        let back = frame.to_lat_lon(frame.to_local(point));

        close(back.lat, point.lat, 1e-12, "latitude");
        close(back.lon, point.lon, 1e-12, "longitude");
    }

    #[test]
    fn planar_distance_agrees_with_haversine_at_site_scale() {
        // Under a centimetre at 250 m: far more accurate than the metre-scale
        // uncertainty the positions themselves carry.
        let frame = LocalFrame::new(SITE);
        let target = destination_point(SITE, 47.0, 250.0);
        let local = frame.to_local(target);

        let planar = (local.x * local.x + local.y * local.y).sqrt();
        close(
            planar,
            haversine_distance(SITE, target),
            0.01,
            "planar vs geodesic",
        );
    }

    #[test]
    fn destination_lands_at_the_requested_distance_and_bearing() {
        for bearing in [0.0, 45.0, 90.0, 180.0, 271.0, 359.0] {
            let target = destination_point(SITE, bearing, 500.0);
            close(haversine_distance(SITE, target), 500.0, 0.001, "distance");
            close(bearing_degrees(SITE, target), bearing, 0.001, "bearing");
        }
    }

    #[test]
    fn forty_five_degree_depression_lands_at_one_mount_height() {
        let p = project_to_ground(&pose(), 0.5, 0.5, DEFAULT_ANGULAR_UNCERTAINTY_DEG, true)
            .expect("centre ray must reach the ground");

        close(p.ground_distance_meters, 10.0, 1e-9, "ground distance");
        close(p.bearing_deg, 0.0, 1e-9, "bearing");
    }

    #[test]
    fn a_ray_at_or_above_the_horizon_has_no_ground_intersection() {
        let level = CameraPose {
            pitch: 0.0,
            ..pose()
        };
        assert!(project_to_ground(&level, 0.5, 0.5, 1.5, true).is_none());

        let upward = CameraPose {
            pitch: 10.0,
            ..pose()
        };
        assert!(project_to_ground(&upward, 0.5, 0.5, 1.5, true).is_none());
    }

    #[test]
    fn out_of_range_projections_are_rejected_not_clamped() {
        let shallow = CameraPose {
            pitch: -3.0,
            ..pose()
        };
        assert!(project_to_ground(&shallow, 0.5, 0.5, 1.5, true).is_none());

        let unbounded = project_to_ground(&shallow, 0.5, 0.5, 1.5, false)
            .expect("range enforcement is opt-out for FOV rendering");
        assert!(unbounded.ground_distance_meters > pose().range_meters);
    }

    #[test]
    fn uncertainty_grows_sharply_toward_the_horizon() {
        let near = project_to_ground(
            &CameraPose {
                pitch: -60.0,
                ..pose()
            },
            0.5,
            0.5,
            1.5,
            true,
        )
        .unwrap();
        let far = project_to_ground(
            &CameraPose {
                pitch: -10.0,
                ..pose()
            },
            0.5,
            0.5,
            1.5,
            true,
        )
        .unwrap();

        assert!(near.uncertainty_meters < far.uncertainty_meters);

        // Error is super-linear in distance, so a detection near the horizon must
        // never be shown with the same confidence as one at the camera's feet.
        let near_ratio = near.uncertainty_meters / near.ground_distance_meters;
        let far_ratio = far.uncertainty_meters / far.ground_distance_meters;
        assert!(
            far_ratio > near_ratio * 2.0,
            "relative error must worsen with distance"
        );
    }

    #[test]
    fn projection_round_trips_through_image_space() {
        // The property the whole simulator and every accuracy claim rests on.
        for bearing_offset in [-20.0, -5.0, 0.0, 5.0, 20.0] {
            for distance in [12.0, 20.0, 40.0, 80.0] {
                let truth =
                    destination_point(pose().position, pose().heading + bearing_offset, distance);

                let Some((u, v, _, in_frame)) = image_coordinates(&pose(), truth, 0.0) else {
                    continue;
                };
                if !in_frame {
                    continue;
                }

                let recovered = project_to_ground(&pose(), u, v, 1.5, false)
                    .expect("a point that is in frame must project back");

                let error = haversine_distance(truth, recovered.position);
                assert!(error < 0.01, "round-trip error {error} m at {distance} m");
            }
        }
    }

    #[test]
    fn a_ray_below_nadir_does_not_project_behind_the_camera() {
        // A camera tilted steeply enough that the bottom of the frame passes
        // *under* the mast. Those rays have a depression above 90 degrees, so
        // `h / tan(theta)` is negative — they meet the ground plane behind the
        // camera, which the forward-looking model cannot represent.
        //
        // The refusal is what is being pinned. A negative distance walked along
        // the bearing would place the object 180 degrees from where it is, and
        // a clamp to zero would place it at the operator's feet. Both are
        // confident lies; `None` is the truth.
        let steep = CameraPose {
            pitch: -80.0,
            vertical_fov: 40.0,
            ..pose()
        };

        // Bottom edge of the frame: depression 100 degrees, past straight down.
        assert!(
            ray_angles(&steep, 0.5, 1.0).1 < -90.0,
            "not actually past nadir"
        );
        assert!(project_to_ground(&steep, 0.5, 1.0, 1.5, false).is_none());

        // And the caller degrades to the camera's own position rather than
        // dropping the object, with the uncertainty saying how little is known.
        let estimate = project_detection(
            &steep,
            &BoundingBox {
                x: 0.45,
                y: 0.9,
                w: 0.1,
                h: 0.1,
            },
        );
        assert_eq!(estimate.source, PositionSource::CameraFallback);
        close(
            estimate.radius_meters,
            steep.range_meters,
            1e-9,
            "below-nadir fallback radius",
        );
    }

    #[test]
    fn detection_falls_back_rather_than_inventing_a_position() {
        let level = CameraPose {
            pitch: 5.0,
            ..pose()
        };
        let estimate = project_detection(
            &level,
            &BoundingBox {
                x: 0.45,
                y: 0.4,
                w: 0.1,
                h: 0.2,
            },
        );

        assert_eq!(estimate.source, PositionSource::CameraFallback);
        // The uncertainty must cover the whole field of view, not imply precision.
        close(
            estimate.radius_meters,
            level.range_meters,
            1e-9,
            "fallback radius",
        );
    }

    #[test]
    fn the_footprint_excludes_the_blind_foreground() {
        // A tilted camera cannot see the ground at its own mast.
        let near = near_ground_distance(&pose()).expect("this camera does see ground");
        assert!(near > 0.0);

        let wedge = field_of_view_wedge(&pose(), 8);
        let closest = wedge
            .iter()
            .map(|p| haversine_distance(pose().position, *p))
            .fold(f64::INFINITY, f64::min);

        close(closest, near, 0.5, "closest footprint point");
    }

    #[test]
    fn the_footprint_stays_within_range_and_field_of_view() {
        for point in field_of_view_wedge(&pose(), 8) {
            let distance = haversine_distance(pose().position, point);
            assert!(distance <= pose().range_meters + 0.5);

            if distance < 0.01 {
                continue;
            }
            let offset = angle_difference(bearing_degrees(pose().position, point), pose().heading);
            assert!(offset.abs() <= pose().horizontal_fov / 2.0 + 1e-6);
        }
    }

    #[test]
    fn point_in_polygon_does_not_chatter_on_an_edge() {
        let square = [
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: 40.0, y: 0.0 },
            Vec2 { x: 40.0, y: 40.0 },
            Vec2 { x: 0.0, y: 40.0 },
        ];

        assert!(point_in_polygon(Vec2 { x: 20.0, y: 20.0 }, &square));
        assert!(!point_in_polygon(Vec2 { x: -5.0, y: 20.0 }, &square));
        assert!(!point_in_polygon(Vec2 { x: 60.0, y: 20.0 }, &square));

        // A pass straight through must produce exactly one enter and one exit.
        let mut transitions = 0;
        let mut inside = false;
        for step in -10..60 {
            let now = point_in_polygon(
                Vec2 {
                    x: step as f64,
                    y: 20.0,
                },
                &square,
            );
            if now != inside {
                transitions += 1;
                inside = now;
            }
        }
        assert_eq!(transitions, 2, "a single pass must not chatter");
    }

    #[test]
    fn iou_behaves() {
        let a = BoundingBox {
            x: 0.2,
            y: 0.5,
            w: 0.1,
            h: 0.3,
        };
        close(a.iou(&a), 1.0, 1e-9, "identical");
        assert_eq!(
            a.iou(&BoundingBox {
                x: 0.9,
                y: 0.5,
                w: 0.1,
                h: 0.3
            }),
            0.0
        );

        let partial = a.iou(&BoundingBox {
            x: 0.25,
            y: 0.5,
            w: 0.1,
            h: 0.3,
        });
        assert!(partial > 0.0 && partial < 1.0);
    }

    #[test]
    fn tripwire_crossings_are_proper_intersections() {
        let a = Vec2 { x: 0.0, y: -20.0 };
        let b = Vec2 { x: 0.0, y: 20.0 };

        assert!(segments_intersect(
            Vec2 { x: -10.0, y: 0.0 },
            Vec2 { x: 10.0, y: 0.0 },
            a,
            b
        ));
        assert!(!segments_intersect(
            Vec2 { x: -10.0, y: 0.0 },
            Vec2 { x: -5.0, y: 0.0 },
            a,
            b
        ));
        assert!(!segments_intersect(
            Vec2 { x: -10.0, y: 50.0 },
            Vec2 { x: 10.0, y: 50.0 },
            a,
            b
        ));
    }
    #[test]
    fn a_confident_position_well_inside_a_zone_is_inside() {
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let ring = square_zone(site, 40.0);

        assert_eq!(zone_membership(&ring, site, 1.0), ZoneMembership::Inside);
    }

    #[test]
    fn a_confident_position_well_outside_a_zone_is_outside() {
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let ring = square_zone(site, 40.0);
        let far = destination_point(site, 0.0, 200.0);

        assert_eq!(zone_membership(&ring, far, 1.0), ZoneMembership::Outside);
    }

    #[test]
    fn a_position_whose_uncertainty_straddles_the_boundary_is_uncertain() {
        // The object is 3 m outside the fence line, known to plus or minus 8 m.
        // Reporting that as "outside" is a guess dressed as a measurement, and
        // reporting it as "inside" is an intrusion alarm nobody can justify.
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let ring = square_zone(site, 40.0);
        let near_edge = destination_point(site, 0.0, 43.0);

        assert_eq!(
            zone_membership(&ring, near_edge, 8.0),
            ZoneMembership::Uncertain
        );
        assert_eq!(
            zone_membership(&ring, near_edge, 1.0),
            ZoneMembership::Outside
        );
    }

    #[test]
    fn uncertainty_reaching_out_from_inside_is_also_uncertain() {
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let ring = square_zone(site, 40.0);
        let just_inside = destination_point(site, 0.0, 37.0);

        assert_eq!(
            zone_membership(&ring, just_inside, 8.0),
            ZoneMembership::Uncertain
        );
        assert_eq!(
            zone_membership(&ring, just_inside, 0.5),
            ZoneMembership::Inside
        );
    }

    #[test]
    fn a_degenerate_zone_contains_nothing_however_uncertain_the_point() {
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let line = vec![site, destination_point(site, 0.0, 10.0)];

        assert_eq!(zone_membership(&line, site, 50.0), ZoneMembership::Outside);
    }

    #[test]
    fn membership_is_stable_as_uncertainty_shrinks() {
        // A track walking into a zone must not oscillate between answers as its
        // position estimate improves. Each state may only advance in one
        // direction: outside -> uncertain -> inside.
        let site = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let ring = square_zone(site, 40.0);
        let inside_point = destination_point(site, 0.0, 20.0);

        let mut seen = Vec::new();
        for step in 0..20 {
            let radius = 40.0 - step as f64 * 2.0;
            seen.push(zone_membership(&ring, inside_point, radius.max(0.0)));
        }

        // Only ever Uncertain then Inside, and never back.
        let first_inside = seen.iter().position(|&m| m == ZoneMembership::Inside);
        assert!(
            first_inside.is_some(),
            "a shrinking disc must eventually be inside"
        );
        assert!(
            seen[first_inside.unwrap()..]
                .iter()
                .all(|&m| m == ZoneMembership::Inside),
            "membership went back to uncertain after being inside"
        );
    }

    fn square_zone(centre: LatLon, half_side: f64) -> Vec<LatLon> {
        [45.0, 135.0, 225.0, 315.0]
            .iter()
            .map(|&bearing| {
                destination_point(centre, bearing, half_side * std::f64::consts::SQRT_2)
            })
            .collect()
    }
    #[test]
    fn a_camera_that_sees_no_ground_has_no_footprint() {
        // Level or upward. near_ground_distance previously answered 0.0 here,
        // and field_of_view_wedge read that as "coverage starts at the mast" —
        // drawing the solid pie slice its own documentation forbids, claiming
        // the entire foreground for a camera pointed at the sky.
        let skyward = CameraPose {
            pitch: 20.0,
            vertical_fov: 34.0,
            ..pose()
        };

        assert_eq!(near_ground_distance(&skyward), None);
        assert!(
            field_of_view_wedge(&skyward, 12).is_empty(),
            "a camera with no ground in view was given a footprint"
        );
    }

    #[test]
    fn a_camera_pointed_at_the_sky_has_no_near_edge_to_measure_from() {
        // `camera_sees` lives behind the FFI, where the same absence is turned
        // into "no". Here the property it depends on is asserted directly.
        let skyward = CameraPose {
            pitch: 20.0,
            vertical_fov: 34.0,
            ..pose()
        };

        assert_eq!(near_ground_distance(&skyward), None);
        assert_eq!(far_ground_distance(&skyward), None);
    }

    #[test]
    fn a_zero_field_of_view_has_no_image_plane() {
        // Dividing by tan(0) returned Some((NaN, inf, ..)), which then
        // propagated silently through every consumer.
        let degenerate = CameraPose {
            horizontal_fov: 0.0,
            ..pose()
        };
        let ahead = destination_point(degenerate.position, 0.0, 20.0);

        assert_eq!(image_coordinates(&degenerate, ahead, 0.0), None);
    }

    #[test]
    fn a_ring_that_encloses_nothing_is_not_a_zone() {
        // Three clicks in nearly the same place. It has three vertices and no
        // area, and previously returned Uncertain — which an uncertain-accepting
        // zone would have turned into presences.
        let centre = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let collapsed = vec![
            centre,
            destination_point(centre, 0.0, 0.05),
            destination_point(centre, 90.0, 0.05),
        ];

        assert_eq!(
            zone_membership(&collapsed, centre, 0.0),
            ZoneMembership::Outside
        );
        assert_eq!(
            zone_membership(&collapsed, centre, 25.0),
            ZoneMembership::Outside
        );
    }

    #[test]
    fn a_real_zone_is_still_a_zone() {
        // The degeneracy guard must not reject anything an operator would draw.
        let centre = LatLon {
            lat: 33.8938,
            lon: 35.5018,
        };
        let small = square_zone(centre, 0.5);

        assert_eq!(zone_membership(&small, centre, 0.0), ZoneMembership::Inside);
    }
}
