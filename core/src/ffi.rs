//! The C ABI.
//!
//! A plain C surface rather than PyO3 bindings, for two reasons. The practical
//! one: this crate is built with the GNU toolchain while CPython on Windows is
//! built with MSVC, and PyO3 across that boundary is an ABI hazard — whereas the
//! C ABI is the C ABI. The better one: a C surface is loadable from anything, so
//! the engine is not welded to Python.
//!
//! Every function here is `extern "C"` and takes only plain data. The rules that
//! follow from crossing an unsafe boundary:
//!
//!  - **Every pointer is checked** before it is dereferenced. A null from a
//!    caller's bug must produce a defined failure, not a segfault inside a
//!    security appliance.
//!  - **Nothing allocates for the caller** except through paired create/destroy
//!    functions, so ownership is never ambiguous.
//!  - **Panics cannot cross the boundary.** The crate is built with
//!    `panic = "abort"`; unwinding into C is undefined behaviour, so the code
//!    below simply does not panic — every fallible path returns a status.

use crate::geometry::{self, BoundingBox, CameraPose, LatLon, PositionSource, Vec2};
use crate::tracking::{Detection, Tracker, TrackerConfig};

/// Version of this ABI. Python checks it on load and refuses a mismatch rather
/// than calling functions whose signatures may have moved.
pub const ABI_VERSION: u32 = 6;

#[no_mangle]
pub extern "C" fn sentinel_abi_version() -> u32 {
    ABI_VERSION
}

/// Sizes of every struct that crosses the boundary, in declaration order:
/// `CPose, CDetection, CTrack, CProjection, CPoint`.
///
/// A binding in another language declares these layouts by hand, and a layout
/// that silently disagrees with this one does not fail — it reads the wrong
/// bytes and produces plausible, wrong geometry. So the sizes are exported and
/// checked, rather than assumed to have stayed in step.
///
/// Returns the number written, or -1 on a null or undersized buffer.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `out` must have room for at least `capacity` `u32` values.
#[no_mangle]
pub unsafe extern "C" fn sentinel_struct_sizes(out: *mut u32, capacity: u32) -> i32 {
    const COUNT: usize = 5;
    if out.is_null() || (capacity as usize) < COUNT {
        return -1;
    }

    let sizes = [
        core::mem::size_of::<CPose>() as u32,
        core::mem::size_of::<CDetection>() as u32,
        core::mem::size_of::<CTrack>() as u32,
        core::mem::size_of::<CProjection>() as u32,
        core::mem::size_of::<CPoint>() as u32,
    ];

    // SAFETY: checked non-null and large enough for COUNT elements above.
    unsafe {
        core::ptr::copy_nonoverlapping(sizes.as_ptr(), out, COUNT);
    }
    COUNT as i32
}

// --------------------------------------------------------------- plain structs

/// Camera pose, laid out for C. Field order is part of the ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPose {
    pub lat: f64,
    pub lon: f64,
    pub mount_height: f64,
    pub heading: f64,
    pub pitch: f64,
    pub roll: f64,
    pub horizontal_fov: f64,
    pub vertical_fov: f64,
    pub range_meters: f64,
}

impl CPose {
    fn to_pose(self) -> CameraPose {
        CameraPose {
            position: LatLon {
                lat: self.lat,
                lon: self.lon,
            },
            mount_height: self.mount_height,
            heading: self.heading,
            pitch: self.pitch,
            roll: self.roll,
            horizontal_fov: self.horizontal_fov,
            vertical_fov: self.vertical_fov,
            range_meters: self.range_meters,
        }
    }
}

/// A detection as it crosses the boundary.
///
/// `has_contact` is a flag, not an `Option`: an absent ground contact is the
/// normal state for any detector that produces boxes only. When it is 0 the
/// contact is the box's bottom-centre, which is what every position was
/// projected from before masks existed. When it is 1, `contact_x` /
/// `contact_y` are where the object actually meets the ground — measured from
/// its silhouette — and that point, not the box, is what gets projected.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CDetection {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub class_id: u32,
    /// 1 when `contact_x` / `contact_y` carry a measured point (ABI 6).
    pub has_contact: u32,
    pub contact_x: f64,
    pub contact_y: f64,
}

impl CDetection {
    fn to_detection(self) -> Detection {
        let bbox = BoundingBox {
            x: self.x,
            y: self.y,
            w: self.w,
            h: self.h,
        };
        // A flag set beside a non-finite value is a caller's bug, and the
        // defined response is the answer the box would have given — not a NaN
        // that projects to a NaN latitude and is drawn nowhere without a word.
        let measured =
            self.has_contact != 0 && self.contact_x.is_finite() && self.contact_y.is_finite();
        Detection {
            bbox,
            confidence: self.confidence,
            class_id: self.class_id,
            contact: if measured {
                Vec2 {
                    x: self.contact_x,
                    y: self.contact_y,
                }
            } else {
                bbox.ground_contact()
            },
        }
    }
}

/// A track as it crosses the boundary.
///
/// `has_position` rather than a nullable pointer: an absent map position is a
/// normal, frequent state (no pose, or a ray above the horizon), and encoding it
/// as a flag keeps the struct flat and copyable.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CTrack {
    pub id: u64,
    pub class_id: u32,
    pub confirmed: u32,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub first_seen_millis: i64,
    pub last_seen_millis: i64,
    pub hits: u32,
    pub has_position: u32,
    pub lat: f64,
    pub lon: f64,
    pub uncertainty_meters: f64,
    /// 0 = ground projection, 1 = camera fallback.
    pub position_source: u32,
    /// Motion is three states, not two. `has_speed = 0` means too few
    /// observations to say anything; `has_speed = 1, has_heading = 0` means
    /// standing still, which is a different and often more interesting fact
    /// than not knowing. Collapsing them into one flag would erase the
    /// loitering signal entirely.
    pub has_speed: u32,
    pub has_heading: u32,
    pub _pad2: u32,
    pub speed_mps: f64,
    pub heading_degrees: f64,
    /// Where this track last met the ground, in normalised image coordinates
    /// (ABI 6). Always set: measured when the detector could see the shape,
    /// the box's bottom-centre when it could not. It is the point the map
    /// position was projected from, so a viewer can draw exactly that.
    pub contact_x: f64,
    pub contact_y: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CProjection {
    /// 1 when the ray met the ground; 0 when it could not and nothing else is set.
    pub valid: u32,
    pub _pad: u32,
    pub lat: f64,
    pub lon: f64,
    pub ground_distance_meters: f64,
    pub bearing_deg: f64,
    pub uncertainty_meters: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPoint {
    pub lat: f64,
    pub lon: f64,
}

// ------------------------------------------------------------------- geometry

/// Project a normalised image point onto the ground plane.
///
/// `valid = 0` means the ray is at or above the horizon, or beyond range. The
/// caller must check it: the remaining fields are meaningless when it is zero.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
#[no_mangle]
pub unsafe extern "C" fn sentinel_project_to_ground(
    pose: *const CPose,
    u: f64,
    v: f64,
    angular_uncertainty_deg: f64,
    enforce_range: u32,
    out: *mut CProjection,
) -> i32 {
    if pose.is_null() || out.is_null() {
        return -1;
    }
    let pose = unsafe { *pose }.to_pose();

    let result =
        geometry::project_to_ground(&pose, u, v, angular_uncertainty_deg, enforce_range != 0);

    let projection = match result {
        Some(p) => CProjection {
            valid: 1,
            _pad: 0,
            lat: p.position.lat,
            lon: p.position.lon,
            ground_distance_meters: p.ground_distance_meters,
            bearing_deg: p.bearing_deg,
            uncertainty_meters: p.uncertainty_meters,
        },
        None => CProjection {
            valid: 0,
            _pad: 0,
            lat: 0.0,
            lon: 0.0,
            ground_distance_meters: 0.0,
            bearing_deg: 0.0,
            uncertainty_meters: 0.0,
        },
    };

    unsafe { *out = projection };
    0
}

/// Ground footprint of a camera's field of view.
///
/// Writes at most `capacity` points and reports how many were produced. A caller
/// that under-sized its buffer gets a truncated polygon and a count it can check,
/// rather than a heap overflow.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `out` must have room for at least `capacity` `CPoint` values.
#[no_mangle]
pub unsafe extern "C" fn sentinel_field_of_view(
    pose: *const CPose,
    arc_segments: u32,
    out: *mut CPoint,
    capacity: u32,
    written: *mut u32,
) -> i32 {
    if pose.is_null() || out.is_null() || written.is_null() {
        return -1;
    }
    let pose = unsafe { *pose }.to_pose();

    let wedge = geometry::field_of_view_wedge(&pose, arc_segments as usize);
    let count = wedge.len().min(capacity as usize);

    let slice = unsafe { std::slice::from_raw_parts_mut(out, count) };
    for (index, point) in wedge.iter().take(count).enumerate() {
        slice[index] = CPoint {
            lat: point.lat,
            lon: point.lon,
        };
    }

    unsafe { *written = count as u32 };
    if wedge.len() > capacity as usize {
        1
    } else {
        0
    }
}

/// Whether a camera can see a ground point, accounting for the blind foreground.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
#[no_mangle]
pub unsafe extern "C" fn sentinel_camera_sees(pose: *const CPose, lat: f64, lon: f64) -> i32 {
    if pose.is_null() {
        return -1;
    }
    let pose = unsafe { *pose }.to_pose();
    let point = LatLon { lat, lon };

    let distance = geometry::haversine_distance(pose.position, point);

    // No near edge means this camera sees no ground at all, so it sees nothing.
    // Treating the absence as 0.0 would have answered "yes" for every point in
    // front of a camera pointed at the sky.
    let Some(near) = geometry::near_ground_distance(&pose) else {
        return 0;
    };

    let far = geometry::far_ground_distance(&pose)
        .map(|f| f.min(pose.range_meters))
        .unwrap_or(pose.range_meters);

    if distance < near || distance > far || distance < 0.5 {
        return 0;
    }

    let bearing = geometry::bearing_degrees(pose.position, point);
    if geometry::bearing_in_fov(&pose, bearing) {
        1
    } else {
        0
    }
}

/// Point-in-polygon over a ring supplied in geographic coordinates.
///
/// The ring is converted into a local metric frame anchored at its first vertex,
/// which is what keeps the test consistent between the polygon and the point.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `ring` must point to at least `count` contiguous `CPoint` values.
#[no_mangle]
pub unsafe extern "C" fn sentinel_point_in_zone(
    ring: *const CPoint,
    count: u32,
    lat: f64,
    lon: f64,
) -> i32 {
    if ring.is_null() || count < 3 {
        return 0;
    }
    let ring = unsafe { std::slice::from_raw_parts(ring, count as usize) };

    let anchor = LatLon {
        lat: ring[0].lat,
        lon: ring[0].lon,
    };
    let frame = geometry::LocalFrame::new(anchor);

    let local: Vec<Vec2> = ring
        .iter()
        .map(|p| {
            frame.to_local(LatLon {
                lat: p.lat,
                lon: p.lon,
            })
        })
        .collect();

    if geometry::point_in_polygon(frame.to_local(LatLon { lat, lon }), &local) {
        1
    } else {
        0
    }
}

/// Where a point at a given height appears in a camera's image.
///
/// The exact inverse of `sentinel_project_to_ground`. Two uses, both real:
/// drawing a map object onto a camera view, and generating test footage of a
/// known world from a known pose — which is the only way to check that two
/// cameras looking at one scene agree about what they are seeing.
///
/// `u` and `v` are normalised image coordinates and are **not clipped**: a value
/// outside 0..1 means the point is off frame in that direction, which is a
/// meaningful answer rather than an error. `in_frame` is 1 only when the point
/// is inside the field of view on both axes and within range.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
#[no_mangle]
pub unsafe extern "C" fn sentinel_image_coordinates(
    pose: *const CPose,
    lat: f64,
    lon: f64,
    height_meters: f64,
    out_u: *mut f64,
    out_v: *mut f64,
    out_distance: *mut f64,
    out_in_frame: *mut u32,
) -> i32 {
    if pose.is_null()
        || out_u.is_null()
        || out_v.is_null()
        || out_distance.is_null()
        || out_in_frame.is_null()
    {
        return -1;
    }

    // SAFETY: every pointer checked non-null above.
    let camera = unsafe { (*pose).to_pose() };
    let point = LatLon { lat, lon };

    match geometry::image_coordinates(&camera, point, height_meters) {
        None => 0,
        Some((u, v, distance, in_frame)) => {
            // SAFETY: as above.
            unsafe {
                *out_u = u;
                *out_v = v;
                *out_distance = distance;
                *out_in_frame = u32::from(in_frame);
            }
            1
        }
    }
}

/// Zone membership accounting for the position's own uncertainty.
///
/// Returns `0` outside, `1` inside, `2` uncertain, or `-1` on a bad argument.
///
/// The three-way answer is the point. A rule that raises an intrusion alarm must
/// demand `1`; treating `2` as inside produces alerts from objects that were
/// never in the zone, and treating it as outside hides ones that were.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `ring` must point to at least `count` contiguous `CPoint` values.
#[no_mangle]
pub unsafe extern "C" fn sentinel_zone_membership(
    ring: *const CPoint,
    count: u32,
    lat: f64,
    lon: f64,
    uncertainty_meters: f64,
) -> i32 {
    if ring.is_null() || count < 3 {
        return -1;
    }

    // SAFETY: checked non-null and the caller promises `count` elements.
    let points = unsafe { std::slice::from_raw_parts(ring, count as usize) };
    let polygon: Vec<LatLon> = points
        .iter()
        .map(|p| LatLon {
            lat: p.lat,
            lon: p.lon,
        })
        .collect();

    match geometry::zone_membership(&polygon, LatLon { lat, lon }, uncertainty_meters) {
        geometry::ZoneMembership::Outside => 0,
        geometry::ZoneMembership::Inside => 1,
        geometry::ZoneMembership::Uncertain => 2,
    }
}

#[no_mangle]
pub extern "C" fn sentinel_haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    geometry::haversine_distance(
        LatLon {
            lat: lat1,
            lon: lon1,
        },
        LatLon {
            lat: lat2,
            lon: lon2,
        },
    )
}

#[no_mangle]
pub extern "C" fn sentinel_bearing_degrees(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    geometry::bearing_degrees(
        LatLon {
            lat: lat1,
            lon: lon1,
        },
        LatLon {
            lat: lat2,
            lon: lon2,
        },
    )
}

///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
#[no_mangle]
pub unsafe extern "C" fn sentinel_destination_point(
    lat: f64,
    lon: f64,
    bearing_deg: f64,
    distance_meters: f64,
    out: *mut CPoint,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let result = geometry::destination_point(LatLon { lat, lon }, bearing_deg, distance_meters);
    unsafe {
        *out = CPoint {
            lat: result.lat,
            lon: result.lon,
        }
    };
    0
}

// ------------------------------------------------------------------- tracking

/// An opaque tracker handle.
///
/// The caller holds a pointer and never inspects it. Paired with
/// `sentinel_tracker_destroy`; every other tracker function tolerates null.
pub struct TrackerHandle {
    tracker: Tracker,
    /// Scratch buffer reused across calls so a per-frame call does not allocate.
    scratch: Vec<CTrack>,
    ended: Vec<u64>,
}

///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// The returned handle must eventually be passed to
/// `sentinel_tracker_destroy`, exactly once.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_create(
    pose: *const CPose,
    iou_threshold: f64,
    gate_factor: f64,
    max_gap_millis: i64,
    min_hits_to_confirm: u32,
) -> *mut TrackerHandle {
    let config = TrackerConfig {
        iou_threshold,
        gate_factor,
        max_gap_millis,
        min_hits_to_confirm,
        motion_window_millis: 3000,
        min_motion_span_millis: 1200,
    };

    // A null pose is the normal state for a camera nobody has placed on the map.
    let camera = if pose.is_null() {
        None
    } else {
        Some(unsafe { *pose }.to_pose())
    };

    Box::into_raw(Box::new(TrackerHandle {
        tracker: Tracker::new(config, camera),
        scratch: Vec::new(),
        ended: Vec::new(),
    }))
}

///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `handle` must come from `sentinel_tracker_create` and must not have been
/// destroyed already. Passing the same handle twice is a double free.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_destroy(handle: *mut TrackerHandle) {
    if handle.is_null() {
        return;
    }
    // Reconstituting the Box drops it, which is the only place this frees.
    unsafe { drop(Box::from_raw(handle)) };
}

///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `handle` must come from `sentinel_tracker_create` and still be live.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_set_pose(
    handle: *mut TrackerHandle,
    pose: *const CPose,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *handle };
    handle.tracker.set_pose(if pose.is_null() {
        None
    } else {
        Some(unsafe { *pose }.to_pose())
    });
    0
}

/// Feed one frame of detections.
///
/// Returns the number of confirmed tracks, or a negative status. The tracks
/// themselves are then read with `sentinel_tracker_tracks`, which keeps this call
/// free of output-buffer sizing decisions the caller cannot make in advance.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `detections` must point to at least `count` contiguous `CDetection`
/// values, and `handle` must come from `sentinel_tracker_create`.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_update(
    handle: *mut TrackerHandle,
    detections: *const CDetection,
    count: u32,
    at_millis: i64,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if count > 0 && detections.is_null() {
        return -1;
    }

    let handle = unsafe { &mut *handle };

    let input: Vec<Detection> = if count == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(detections, count as usize) }
            .iter()
            .map(|d| d.to_detection())
            .collect()
    };

    let update = handle.tracker.update(&input, at_millis);
    handle.ended = update.ended;

    handle.scratch.clear();
    for track in handle.tracker.tracks() {
        handle.scratch.push(CTrack {
            id: track.id,
            class_id: track.class_id,
            confirmed: u32::from(track.confirmed),
            x: track.bbox.x,
            y: track.bbox.y,
            w: track.bbox.w,
            h: track.bbox.h,
            confidence: track.confidence,
            first_seen_millis: track.first_seen_millis,
            last_seen_millis: track.last_seen_millis,
            hits: track.hits,
            has_position: u32::from(track.position.is_some()),
            lat: track.position.map(|p| p.point.lat).unwrap_or(0.0),
            lon: track.position.map(|p| p.point.lon).unwrap_or(0.0),
            uncertainty_meters: track.position.map(|p| p.radius_meters).unwrap_or(0.0),
            position_source: track
                .position
                .map(|p| u32::from(p.source == PositionSource::CameraFallback))
                .unwrap_or(0),
            has_speed: u32::from(track.speed_mps.is_some()),
            has_heading: u32::from(track.heading_degrees.is_some()),
            _pad2: 0,
            speed_mps: track.speed_mps.unwrap_or(0.0),
            heading_degrees: track.heading_degrees.unwrap_or(0.0),
            contact_x: track.contact.x,
            contact_y: track.contact.y,
        });
    }

    handle.scratch.len() as i32
}

/// Copy the confirmed tracks from the last update into the caller's buffer.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `out` must have room for at least `capacity` `CTrack` values, and
/// `handle` must come from `sentinel_tracker_create`.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_tracks(
    handle: *const TrackerHandle,
    out: *mut CTrack,
    capacity: u32,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return -1;
    }
    let handle = unsafe { &*handle };

    let count = handle.scratch.len().min(capacity as usize);
    let slice = unsafe { std::slice::from_raw_parts_mut(out, count) };
    slice.copy_from_slice(&handle.scratch[..count]);

    count as i32
}

/// Track ids closed on the last update, so downstream state can be released.
///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `out` must have room for at least `capacity` `u64` values, and `handle`
/// must come from `sentinel_tracker_create`.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_ended(
    handle: *const TrackerHandle,
    out: *mut u64,
    capacity: u32,
) -> i32 {
    if handle.is_null() || out.is_null() {
        return -1;
    }
    let handle = unsafe { &*handle };

    let count = handle.ended.len().min(capacity as usize);
    let slice = unsafe { std::slice::from_raw_parts_mut(out, count) };
    slice.copy_from_slice(&handle.ended[..count]);

    count as i32
}

///
/// # Safety
///
/// Null is accepted and produces a defined failure. A **non-null** pointer,
/// however, is taken at its word: it must point to a live, aligned,
/// initialised value of the named type. Null-checking cannot establish that,
/// which is why this is `unsafe` despite checking.
/// `handle` must come from `sentinel_tracker_create` and still be live.
#[no_mangle]
pub unsafe extern "C" fn sentinel_tracker_reset(handle: *mut TrackerHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *handle };
    let ended = handle.tracker.reset();
    handle.scratch.clear();

    let count = ended.len() as i32;
    handle.ended = ended;
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zeroed track, for the read buffers a caller must supply.
    fn blank_track() -> CTrack {
        CTrack {
            id: 0,
            class_id: 0,
            confirmed: 0,
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
            confidence: 0.0,
            first_seen_millis: 0,
            last_seen_millis: 0,
            hits: 0,
            has_position: 0,
            lat: 0.0,
            lon: 0.0,
            uncertainty_meters: 0.0,
            position_source: 0,
            has_speed: 0,
            has_heading: 0,
            _pad2: 0,
            speed_mps: 0.0,
            heading_degrees: 0.0,
            contact_x: 0.0,
            contact_y: 0.0,
        }
    }

    fn pose() -> CPose {
        CPose {
            lat: 33.8938,
            lon: 35.5018,
            mount_height: 10.0,
            heading: 0.0,
            pitch: -45.0,
            roll: 0.0,
            horizontal_fov: 60.0,
            vertical_fov: 34.0,
            range_meters: 120.0,
        }
    }

    #[test]
    fn every_entry_point_tolerates_null() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            // A caller's bug must produce a defined failure, not a segfault inside a
            // security appliance.
            let mut projection = CProjection {
                valid: 0,
                _pad: 0,
                lat: 0.0,
                lon: 0.0,
                ground_distance_meters: 0.0,
                bearing_deg: 0.0,
                uncertainty_meters: 0.0,
            };

            assert_eq!(
                sentinel_project_to_ground(std::ptr::null(), 0.5, 0.5, 1.5, 1, &mut projection),
                -1
            );
            assert_eq!(
                sentinel_project_to_ground(&pose(), 0.5, 0.5, 1.5, 1, std::ptr::null_mut()),
                -1
            );
            assert_eq!(sentinel_camera_sees(std::ptr::null(), 0.0, 0.0), -1);
            assert_eq!(
                sentinel_tracker_update(std::ptr::null_mut(), std::ptr::null(), 0, 0),
                -1
            );
            assert_eq!(
                sentinel_tracker_tracks(std::ptr::null(), std::ptr::null_mut(), 0),
                -1
            );
            assert_eq!(sentinel_tracker_reset(std::ptr::null_mut()), -1);
            assert_eq!(sentinel_tracker_set_pose(std::ptr::null_mut(), &pose()), -1);
            assert_eq!(sentinel_struct_sizes(std::ptr::null_mut(), 5), -1);
            assert_eq!(
                sentinel_zone_membership(std::ptr::null(), 4, 0.0, 0.0, 1.0),
                -1
            );
            assert_eq!(
                sentinel_image_coordinates(
                    std::ptr::null(),
                    0.0,
                    0.0,
                    1.7,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                -1
            );

            // Destroying null is a no-op, so a double free is survivable.
            sentinel_tracker_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_measured_contact_crosses_the_boundary_and_comes_back() {
        unsafe {
            let handle = sentinel_tracker_create(&pose(), 0.2, 2.5, 2000, 1);
            assert!(!handle.is_null());

            let with_contact = CDetection {
                x: 0.4,
                y: 0.5,
                w: 0.2,
                h: 0.3,
                confidence: 0.9,
                class_id: 0,
                has_contact: 1,
                contact_x: 0.42,
                contact_y: 0.78,
            };
            assert_eq!(sentinel_tracker_update(handle, &with_contact, 1, 0), 1);

            let mut out = [blank_track()];
            assert_eq!(sentinel_tracker_tracks(handle, out.as_mut_ptr(), 1), 1);
            assert_eq!((out[0].contact_x, out[0].contact_y), (0.42, 0.78));

            // Flag clear: the values beside it are ignored and the box answers.
            let without = CDetection {
                has_contact: 0,
                contact_x: 123.0,
                contact_y: 456.0,
                ..with_contact
            };
            assert_eq!(sentinel_tracker_update(handle, &without, 1, 200), 1);
            assert_eq!(sentinel_tracker_tracks(handle, out.as_mut_ptr(), 1), 1);
            assert_eq!((out[0].contact_x, out[0].contact_y), (0.5, 0.8));

            // Flag set beside a NaN is a caller's bug; the defined answer is
            // the box's, not a NaN latitude.
            let broken = CDetection {
                has_contact: 1,
                contact_x: f64::NAN,
                ..with_contact
            };
            assert_eq!(sentinel_tracker_update(handle, &broken, 1, 400), 1);
            assert_eq!(sentinel_tracker_tracks(handle, out.as_mut_ptr(), 1), 1);
            assert_eq!((out[0].contact_x, out[0].contact_y), (0.5, 0.8));
            assert_eq!(out[0].has_position, 1);
            assert!(out[0].lat.is_finite());

            sentinel_tracker_destroy(handle);
        }
    }

    #[test]
    fn standing_still_is_distinguishable_from_not_knowing() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            // The three motion states must survive the boundary. A person standing
            // in one place for four minutes is the loitering signal the system
            // exists to notice; reporting it as "motion unknown" throws that away.
            let handle = sentinel_tracker_create(&pose(), 0.2, 2.5, 5_000, 2);
            let detection = CDetection {
                x: 0.45,
                y: 0.55,
                w: 0.06,
                h: 0.12,
                confidence: 0.9,
                class_id: 0,
                has_contact: 0,
                contact_x: 0.0,
                contact_y: 0.0,
            };

            let mut out = [blank_track(); 4];

            sentinel_tracker_update(handle, &detection, 1, 0);
            sentinel_tracker_update(handle, &detection, 1, 200);
            for step in 2..14 {
                sentinel_tracker_update(handle, &detection, 1, step * 200);
            }

            assert_eq!(sentinel_tracker_tracks(handle, out.as_mut_ptr(), 4), 1);
            assert_eq!(out[0].has_speed, 1, "we know the speed");
            assert_eq!(out[0].speed_mps, 0.0, "and the speed is zero");
            assert_eq!(
                out[0].has_heading, 0,
                "a heading from jitter is worse than none"
            );

            sentinel_tracker_destroy(handle);
        }
    }

    #[test]
    fn image_coordinates_invert_the_projection() {
        // Project a ray to the ground, then ask where that ground point appears.
        // It has to come back where it started, or the two halves of the camera
        // model disagree and every overlay drawn from the map is wrong.
        unsafe {
            let camera = pose();
            let mut projected = CProjection {
                valid: 0,
                _pad: 0,
                lat: 0.0,
                lon: 0.0,
                ground_distance_meters: 0.0,
                bearing_deg: 0.0,
                uncertainty_meters: 0.0,
            };
            assert_eq!(
                sentinel_project_to_ground(&camera, 0.62, 0.71, 1.5, 1, &mut projected),
                0
            );
            assert_eq!(projected.valid, 1);

            let (mut u, mut v, mut distance, mut in_frame) = (0.0, 0.0, 0.0, 0u32);
            let found = sentinel_image_coordinates(
                &camera,
                projected.lat,
                projected.lon,
                0.0,
                &mut u,
                &mut v,
                &mut distance,
                &mut in_frame,
            );

            assert_eq!(found, 1);
            assert!((u - 0.62).abs() < 1e-6, "u came back as {u}");
            assert!((v - 0.71).abs() < 1e-6, "v came back as {v}");
            assert_eq!(in_frame, 1);
            assert!((distance - projected.ground_distance_meters).abs() < 1e-6);
        }
    }

    #[test]
    fn a_point_behind_the_camera_has_no_image_coordinates() {
        unsafe {
            let camera = pose();
            let (mut u, mut v, mut d, mut f) = (0.0, 0.0, 0.0, 0u32);

            assert_eq!(
                sentinel_image_coordinates(
                    &camera,
                    camera.lat - 0.001,
                    camera.lon,
                    0.0,
                    &mut u,
                    &mut v,
                    &mut d,
                    &mut f,
                ),
                0,
                "a point behind the camera must not be given a position in its image"
            );
        }
    }

    #[test]
    fn zone_membership_crosses_the_boundary_as_three_states() {
        // Two states would be a lie: a position known to plus or minus 8 m,
        // 3 m from a fence, is neither in nor out and the caller must be able
        // to tell.
        let ring = [
            CPoint {
                lat: 33.8930,
                lon: 35.5010,
            },
            CPoint {
                lat: 33.8930,
                lon: 35.5026,
            },
            CPoint {
                lat: 33.8946,
                lon: 35.5026,
            },
            CPoint {
                lat: 33.8946,
                lon: 35.5010,
            },
        ];

        unsafe {
            let centre = (33.8938, 35.5018);
            assert_eq!(
                sentinel_zone_membership(ring.as_ptr(), 4, centre.0, centre.1, 1.0),
                1
            );
            assert_eq!(
                sentinel_zone_membership(ring.as_ptr(), 4, 33.9100, 35.5018, 1.0),
                0
            );
            // Sitting on the northern edge with a large radius.
            assert_eq!(
                sentinel_zone_membership(ring.as_ptr(), 4, 33.8946, 35.5018, 25.0),
                2
            );
            // Fewer than three points is not an area.
            assert_eq!(
                sentinel_zone_membership(ring.as_ptr(), 2, centre.0, centre.1, 1.0),
                -1
            );
        }
    }

    #[test]
    fn struct_sizes_are_reported_for_every_boundary_type() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            // These are what a hand-written binding in another language must match.
            let mut sizes = [0u32; 5];
            assert_eq!(sentinel_struct_sizes(sizes.as_mut_ptr(), 5), 5);
            assert!(sizes.iter().all(|&s| s > 0));

            // An undersized buffer is refused rather than partially filled: a caller
            // that got the count wrong has a stale layout, which is the exact
            // condition this function exists to catch.
            let mut small = [0u32; 4];
            assert_eq!(sentinel_struct_sizes(small.as_mut_ptr(), 4), -1);
            assert_eq!(small, [0u32; 4]);
        }
    }

    #[test]
    fn projection_crosses_the_boundary_intact() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            let mut out = CProjection {
                valid: 0,
                _pad: 0,
                lat: 0.0,
                lon: 0.0,
                ground_distance_meters: 0.0,
                bearing_deg: 0.0,
                uncertainty_meters: 0.0,
            };

            assert_eq!(
                sentinel_project_to_ground(&pose(), 0.5, 0.5, 1.5, 1, &mut out),
                0
            );
            assert_eq!(out.valid, 1);
            assert!((out.ground_distance_meters - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn an_impossible_projection_reports_invalid_rather_than_a_guess() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            let level = CPose {
                pitch: 5.0,
                ..pose()
            };
            let mut out = CProjection {
                valid: 1,
                _pad: 0,
                lat: 9.0,
                lon: 9.0,
                ground_distance_meters: 9.0,
                bearing_deg: 9.0,
                uncertainty_meters: 9.0,
            };

            assert_eq!(
                sentinel_project_to_ground(&level, 0.5, 0.5, 1.5, 1, &mut out),
                0
            );
            assert_eq!(out.valid, 0, "the caller must be told, not handed a number");
        }
    }

    #[test]
    fn an_undersized_buffer_truncates_and_says_so() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            let mut points = [CPoint { lat: 0.0, lon: 0.0 }; 4];
            let mut written = 0u32;

            let status = sentinel_field_of_view(&pose(), 24, points.as_mut_ptr(), 4, &mut written);

            assert_eq!(status, 1, "truncation is reported");
            assert_eq!(written, 4, "and never overruns the buffer");
        }
    }

    #[test]
    fn a_tracker_round_trips_through_the_boundary() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            let handle = sentinel_tracker_create(&pose(), 0.2, 2.5, 2000, 2);
            assert!(!handle.is_null());

            let detections = [CDetection {
                x: 0.45,
                y: 0.5,
                w: 0.08,
                h: 0.2,
                confidence: 0.9,
                class_id: 0,
                has_contact: 0,
                contact_x: 0.0,
                contact_y: 0.0,
            }];

            assert_eq!(
                sentinel_tracker_update(handle, detections.as_ptr(), 1, 0),
                0,
                "not confirmed yet"
            );
            let confirmed = sentinel_tracker_update(handle, detections.as_ptr(), 1, 200);
            assert_eq!(confirmed, 1);

            let mut out = [blank_track(); 8];

            assert_eq!(sentinel_tracker_tracks(handle, out.as_mut_ptr(), 8), 1);
            assert_eq!(out[0].confirmed, 1);
            assert_eq!(
                out[0].has_position, 1,
                "a posed camera yields a map position"
            );
            assert!(
                out[0].uncertainty_meters > 0.0,
                "uncertainty always travels with it"
            );

            sentinel_tracker_destroy(handle);
        }
    }

    #[test]
    fn zone_membership_crosses_intact() {
        // Every call below crosses the C ABI, which is what this test is for.
        unsafe {
            let anchor = LatLon {
                lat: 33.8938,
                lon: 35.5018,
            };
            let corner = |east: f64, north: f64| {
                let p = geometry::destination_point(
                    geometry::destination_point(anchor, 0.0, north),
                    90.0,
                    east,
                );
                CPoint {
                    lat: p.lat,
                    lon: p.lon,
                }
            };

            let ring = [
                corner(0.0, 0.0),
                corner(40.0, 0.0),
                corner(40.0, 40.0),
                corner(0.0, 40.0),
            ];
            let inside = geometry::destination_point(
                geometry::destination_point(anchor, 0.0, 20.0),
                90.0,
                20.0,
            );

            assert_eq!(
                sentinel_point_in_zone(ring.as_ptr(), 4, inside.lat, inside.lon),
                1
            );
            assert_eq!(
                sentinel_point_in_zone(ring.as_ptr(), 4, anchor.lat - 0.01, anchor.lon),
                0
            );
            assert_eq!(
                sentinel_point_in_zone(ring.as_ptr(), 2, inside.lat, inside.lon),
                0,
                "a ring needs 3 points"
            );
        }
    }
}
