//! Latitude, longitude, and metres between them — on **one** Earth.
//!
//! # The defect this module was rewritten to remove
//!
//! v1 and v2 both used two different Earths at once and never noticed.
//! Distances and bearings between tracks went through a haversine on a sphere
//! of radius 6 371 008.8 m; zone tests, polygon distances and the orthophoto
//! grid went through a tangent plane built from the WGS84 latitude series.
//! Those two disagree by **0.248%** at this product's reference latitude —
//! 25 cm per 100 m, 1.24 m across a 500 m site.
//!
//! It is not a rounding difference, it is two models. The sphere's 111 195
//! metres per degree of latitude is a global average; the real meridian at 34
//! degrees is 110 920. So a zone edge drawn 30 m from a camera and a track
//! measured 30 m from the same camera were 7 cm apart for no reason anybody
//! could see, and a `distance_from` that read 0.02 m outside a fence was
//! inside it on the other calculation.
//!
//! Everything here is now the **tangent plane**, because that is the model
//! the rest of the system already had to use: a polygon test needs metres in
//! a plane, and there is no spherical version of "is this point in this
//! ring". [`distance_meters`], [`bearing_degrees`] and [`destination_point`]
//! are all defined on it, which makes them exact inverses of each other
//! rather than approximately consistent.
//!
//! [`spherical_distance`] is kept, used by nothing, and tested — as the
//! reference that says how far the two models are apart and over what
//! distances the tangent plane stops being the right one.
//!
//! # What the tangent plane costs
//!
//! It is exact at its origin and degrades with the square of the distance
//! from it. Over 500 m — a large site — it is within a centimetre of the
//! ellipsoid. Over 50 km it is not, and nothing in this product works at 50
//! km. [`distance_meters`] builds its frame at its first argument, so
//! `distance(a, b)` and `distance(b, a)` differ. Measured: 5.5 mm over 500 m
//! and half a millimetre over 100 m, first order in the north-south
//! separation, and there is a test that holds it there.
//!
//! Two *different* frames are a separate hazard and a real one: the
//! convergence of meridians between frames `D` apart, for a point `d` from
//! the origin, is `D*d*tan(lat)/R`. That is 0.6 mm for frames 60 m apart over
//! a 100 m span here, and metres across a country. Nothing in this crate ever
//! builds a second frame for a job the caller already has one for.

/// Mean Earth radius, IUGG. Used only by [`spherical_distance`], which the
/// product does not call.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct LatLon {
    pub lat: f64,
    pub lon: f64,
}

impl LatLon {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// True for a point that is actually on the planet. Every entry point
    /// that takes a position from outside checks this: a transposed pair of
    /// coordinates produces plausible arithmetic and a camera in the sea.
    pub fn is_valid(&self) -> bool {
        self.lat.is_finite()
            && self.lon.is_finite()
            && self.lat >= -90.0
            && self.lat <= 90.0
            && self.lon >= -180.0
            && self.lon <= 180.0
    }
}

/// Metres per degree of latitude at a latitude. Standard series expansion of
/// the WGS84 meridian arc; error under a millimetre per degree.
pub fn meters_per_degree_latitude(latitude_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    111_132.92 - 559.82 * (2.0 * lat).cos() + 1.175 * (4.0 * lat).cos()
}

/// Metres per degree of longitude at a latitude.
pub fn meters_per_degree_longitude(latitude_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos() + 0.118 * (5.0 * lat).cos()
}

/// An east/north tangent plane at an origin.
///
/// Construct once and reuse: the two trig-series evaluations are the whole
/// cost, and building a second frame for the same job is how two answers
/// about one place come to disagree.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    pub origin: LatLon,
    m_lat: f64,
    m_lon: f64,
}

impl LocalFrame {
    pub fn new(origin: LatLon) -> Self {
        Self {
            origin,
            // A frame at a pole has no east. Clamped rather than left to
            // divide by zero: this product does not run at 90 degrees, and a
            // NaN that reaches a polygon test is far worse than a frame that
            // is merely useless.
            m_lat: meters_per_degree_latitude(origin.lat).max(1.0),
            m_lon: meters_per_degree_longitude(origin.lat).max(1.0),
        }
    }

    /// (east, north) metres from the origin.
    pub fn to_local(&self, point: LatLon) -> (f64, f64) {
        (
            wrapped_longitude_delta(point.lon, self.origin.lon) * self.m_lon,
            (point.lat - self.origin.lat) * self.m_lat,
        )
    }

    pub fn to_lat_lon(&self, east: f64, north: f64) -> LatLon {
        LatLon::new(
            self.origin.lat + north / self.m_lat,
            normalize_longitude(self.origin.lon + east / self.m_lon),
        )
    }
}

/// Shortest signed difference in longitude, so a site on the date line does
/// not measure itself as most of the way round the world.
fn wrapped_longitude_delta(a: f64, b: f64) -> f64 {
    let mut d = (a - b + 180.0) % 360.0;
    if d < 0.0 {
        d += 360.0;
    }
    d - 180.0
}

/// Metres between two points, on the tangent plane at `a`.
///
/// This is the product's distance. See the module docs for why it is not a
/// haversine.
pub fn distance_meters(a: LatLon, b: LatLon) -> f64 {
    let (east, north) = LocalFrame::new(a).to_local(b);
    east.hypot(north)
}

/// Bearing from `a` to `b`, degrees clockwise from true north, on the tangent
/// plane at `a`.
pub fn bearing_degrees(a: LatLon, b: LatLon) -> f64 {
    let (east, north) = LocalFrame::new(a).to_local(b);
    if east == 0.0 && north == 0.0 {
        return 0.0;
    }
    normalize_degrees(east.atan2(north).to_degrees())
}

/// The point a given distance along a given bearing, on the tangent plane at
/// `origin`. The exact inverse of [`distance_meters`] and [`bearing_degrees`]
/// taken from the same origin.
pub fn destination_point(origin: LatLon, bearing_deg: f64, distance_meters: f64) -> LatLon {
    let theta = bearing_deg.to_radians();
    LocalFrame::new(origin).to_lat_lon(distance_meters * theta.sin(), distance_meters * theta.cos())
}

/// Great-circle distance on a sphere.
///
/// Kept for reference and for the test that measures how far it is from
/// [`distance_meters`]. The product does not use it: mixing it with the
/// tangent plane is the defect this module was rewritten to remove.
pub fn spherical_distance(a: LatLon, b: LatLon) -> f64 {
    let (phi1, phi2) = (a.lat.to_radians(), b.lat.to_radians());
    let dphi = phi2 - phi1;
    let dlambda = wrapped_longitude_delta(b.lon, a.lon).to_radians();
    let h = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlambda / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * h.sqrt().min(1.0).asin()
}

pub fn normalize_degrees(deg: f64) -> f64 {
    let d = deg % 360.0;
    if d < 0.0 {
        d + 360.0
    } else {
        d
    }
}

pub fn normalize_longitude(deg: f64) -> f64 {
    wrapped_longitude_delta(deg, 0.0)
}

/// Signed smallest difference `a - b`, in `(-180, 180]`.
pub fn angle_difference(a: f64, b: f64) -> f64 {
    let d = wrapped_longitude_delta(a, b);
    if d == -180.0 {
        180.0
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: LatLon = LatLon {
        lat: 33.8938,
        lon: 35.5018,
    };

    #[test]
    fn destination_distance_and_bearing_are_exact_inverses() {
        for bearing in [0.0, 45.0, 137.0, 271.5, 359.0] {
            for distance in [0.5, 50.0, 500.0, 5_000.0] {
                let there = destination_point(SITE, bearing, distance);
                let back = distance_meters(SITE, there);
                // A micrometre. The residual is the f64 round-trip through
                // degrees of latitude, not the model.
                assert!(
                    (back - distance).abs() < 1e-6,
                    "{bearing} deg, {distance} m -> {back} m"
                );
                // Expressed as the sideways error it causes, because a
                // bearing tolerance alone is meaningless without a range: a
                // nanometre at half a metre is 1e-7 degrees.
                let off = angle_difference(bearing_degrees(SITE, there), bearing);
                assert!(
                    off.to_radians().abs() * distance < 1e-6,
                    "{bearing} deg at {distance} m: off by {off} deg"
                );
            }
        }
    }

    #[test]
    fn the_two_earths_disagree_by_a_quarter_of_a_percent() {
        // The defect, held as a measurement so it cannot come back quietly.
        // If somebody reintroduces a haversine into a distance path, this
        // number is what it costs.
        let there = destination_point(SITE, 0.0, 1000.0);
        let plane = distance_meters(SITE, there);
        let sphere = spherical_distance(SITE, there);
        let divergence = (sphere / plane - 1.0).abs();
        assert!(
            (0.002..0.003).contains(&divergence),
            "expected about 0.25%, got {}%",
            divergence * 100.0
        );
        assert!(sphere - plane > 2.0, "over a kilometre it is metres");
    }

    #[test]
    fn distance_is_symmetric_to_a_hundredth_of_a_percent() {
        // The frame is built at the first argument, so the two directions are
        // not identical: the metres-per-degree series is evaluated at two
        // latitudes. The gap is first order in the north-south separation —
        // measured at 5.5 mm over 500 m, half a millimetre over 100 m — and
        // it has to stay there, or "how far is A from B" becomes a question
        // with two answers an operator can see.
        for bearing in [0.0, 90.0, 210.0] {
            for distance in [10.0, 100.0, 500.0] {
                let b = destination_point(SITE, bearing, distance);
                let forward = distance_meters(SITE, b);
                let backward = distance_meters(b, SITE);
                assert!(
                    (forward - backward).abs() < 2e-5 * distance,
                    "{bearing}/{distance}: {forward} vs {backward}"
                );
            }
        }
    }

    #[test]
    fn the_local_frame_round_trips_and_stays_metric() {
        let frame = LocalFrame::new(SITE);
        for bearing in [0.0, 30.0, 90.0, 210.0, 315.0] {
            let point = destination_point(SITE, bearing, 500.0);
            let (e, n) = frame.to_local(point);
            assert!((e.hypot(n) - 500.0).abs() < 1e-6, "{bearing}: {} m", e.hypot(n));
            let back = frame.to_lat_lon(e, n);
            assert!(distance_meters(point, back) < 1e-6);
            // The same frame both ways, so this round trip is exact to f64.
        }
    }

    #[test]
    fn a_site_on_the_date_line_measures_itself_correctly() {
        // Longitudes either side of 180 differ by 359.99 degrees numerically
        // and by a few metres in fact.
        let west = LatLon::new(0.0, 179.9995);
        let east = LatLon::new(0.0, -179.9995);
        let d = distance_meters(west, east);
        assert!(d < 200.0, "the date line is not half the planet: {d} m");
        assert!(d > 100.0);
    }

    #[test]
    fn angles_normalise_without_drifting() {
        assert_eq!(normalize_degrees(-1.0), 359.0);
        assert_eq!(normalize_degrees(361.0), 1.0);
        assert_eq!(normalize_longitude(181.0), -179.0);
        assert_eq!(normalize_longitude(-181.0), 179.0);
        assert_eq!(angle_difference(10.0, 350.0), 20.0);
        assert_eq!(angle_difference(350.0, 10.0), -20.0);
        assert_eq!(angle_difference(180.0, 0.0), 180.0);
    }

    #[test]
    fn a_point_at_its_own_position_has_a_bearing_rather_than_a_nan() {
        assert_eq!(bearing_degrees(SITE, SITE), 0.0);
        assert_eq!(distance_meters(SITE, SITE), 0.0);
    }
}
