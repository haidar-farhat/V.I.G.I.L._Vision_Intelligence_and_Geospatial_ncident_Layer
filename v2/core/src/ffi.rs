//! The C ABI.
//!
//! A plain C surface rather than PyO3, for the reason v1 gave and which still
//! holds: this crate builds with the GNU toolchain while CPython on Windows
//! is built with MSVC, and PyO3 across that boundary is an ABI hazard. The C
//! ABI is the C ABI. It also means the core is loadable from anything, so the
//! engine is not welded to one runtime.
//!
//! # The rules that follow from crossing an unsafe boundary
//!
//!  - **Every pointer is checked** before it is dereferenced. A null from a
//!    caller's bug must produce a defined failure, not a segfault inside a
//!    security appliance.
//!  - **Nothing allocates for the caller** except through the paired
//!    create/destroy pair for the accumulator, so ownership is never
//!    ambiguous. Everything else writes into buffers the caller owns and
//!    whose length the caller states.
//!  - **Panics cannot cross.** The crate is built with `panic = "abort"`;
//!    unwinding into C is undefined behaviour, so the code below does not
//!    panic — every fallible path returns a status.
//!
//! # Why the layouts are flat
//!
//! v1's ABI passed six `#[repr(C)]` structs and exported their sizes so a
//! binding could check it had declared them the same way. That works, and it
//! is a standing tax: every field added is an ABI break, and the Python side
//! carries a mirror of each layout that has to be kept in step by hand.
//!
//! Here everything crosses as `f64` arrays with a documented order and an
//! exported length. A mismatch is then a length check rather than a silent
//! misread, and NumPy hands over a contiguous array's pointer without any
//! declaration at all.

use core::ffi::c_void;

use crate::assign;
use crate::camera::{
    self, CameraPose, PoseUncertainty, ProjectionFailure,
};
use crate::geodesy::{self, LatLon};
use crate::ortho::{self, CellStats, GroundGrid, GroundSample, MedianAccumulator};
use crate::track::KalmanBox;

/// Version of this ABI. Python checks it on load and refuses a mismatch
/// rather than calling functions whose meaning may have moved.
pub const ABI_VERSION: u32 = 1;

/// `[lat, lon, mount_height, heading, pitch, roll, hfov, vfov, range]`.
pub const POSE_VALUES: u32 = 9;
/// `[heading, pitch, roll, mount_height, terrain_slope]`, all 1-sigma.
pub const SIGMA_VALUES: u32 = 5;
/// `[lat, lon, ground_distance, bearing, along_sigma, across_sigma]`.
pub const PROJECTION_VALUES: u32 = 6;
/// `[origin_lat, origin_lon, cell_size_m, rows, cols]`.
pub const GRID_VALUES: u32 = 5;
/// 8 mean + 64 covariance.
pub const KALMAN_VALUES: u32 = 72;

#[no_mangle]
pub extern "C" fn vigil_abi_version() -> u32 {
    ABI_VERSION
}

/// The array lengths this ABI expects, in the order
/// `[POSE, SIGMA, PROJECTION, GRID, KALMAN]`. A binding checks these rather
/// than assuming they have stayed put.
///
/// Returns the number written, or -1 on a null or undersized buffer.
///
/// # Safety
/// `out` must be null or point to `capacity` writable `u32`.
#[no_mangle]
pub unsafe extern "C" fn vigil_layout(out: *mut u32, capacity: u32) -> i32 {
    const COUNT: usize = 5;
    if out.is_null() || (capacity as usize) < COUNT {
        return -1;
    }
    let values = [
        POSE_VALUES,
        SIGMA_VALUES,
        PROJECTION_VALUES,
        GRID_VALUES,
        KALMAN_VALUES,
    ];
    for (i, v) in values.iter().enumerate() {
        *out.add(i) = *v;
    }
    COUNT as i32
}

// ------------------------------------------------------------------ geodesy

#[no_mangle]
pub extern "C" fn vigil_distance_meters(a_lat: f64, a_lon: f64, b_lat: f64, b_lon: f64) -> f64 {
    geodesy::distance_meters(LatLon::new(a_lat, a_lon), LatLon::new(b_lat, b_lon))
}

#[no_mangle]
pub extern "C" fn vigil_bearing_degrees(a_lat: f64, a_lon: f64, b_lat: f64, b_lon: f64) -> f64 {
    geodesy::bearing_degrees(LatLon::new(a_lat, a_lon), LatLon::new(b_lat, b_lon))
}

/// Writes `[lat, lon]`. Returns 0, or -1 on a null buffer.
///
/// # Safety
/// `out` must be null or point to two writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_destination_point(
    lat: f64,
    lon: f64,
    bearing_deg: f64,
    distance_meters: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let p = geodesy::destination_point(LatLon::new(lat, lon), bearing_deg, distance_meters);
    *out = p.lat;
    *out.add(1) = p.lon;
    0
}

// ------------------------------------------------------------------- camera

/// # Safety
/// `pose` must point to [`POSE_VALUES`] readable `f64`.
unsafe fn read_pose(pose: *const f64) -> Option<CameraPose> {
    if pose.is_null() {
        return None;
    }
    let v = core::slice::from_raw_parts(pose, POSE_VALUES as usize);
    if v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(CameraPose {
        position: LatLon::new(v[0], v[1]),
        mount_height: v[2],
        heading: v[3],
        pitch: v[4],
        roll: v[5],
        horizontal_fov: v[6],
        vertical_fov: v[7],
        range_meters: v[8],
    })
}

/// # Safety
/// `sigma` must be null or point to [`SIGMA_VALUES`] readable `f64`. Null
/// means the documented defaults.
unsafe fn read_sigma(sigma: *const f64) -> PoseUncertainty {
    if sigma.is_null() {
        return PoseUncertainty::default();
    }
    let v = core::slice::from_raw_parts(sigma, SIGMA_VALUES as usize);
    PoseUncertainty {
        heading_deg: v[0],
        pitch_deg: v[1],
        roll_deg: v[2],
        mount_height_m: v[3],
        terrain_slope: v[4],
    }
}

fn failure_code(failure: ProjectionFailure) -> i32 {
    match failure {
        ProjectionFailure::AboveHorizon => 1,
        ProjectionFailure::TooShallow => 2,
        ProjectionFailure::OutOfRange => 3,
        ProjectionFailure::BadPose => 4,
    }
}

/// Project `count` normalised image points onto the ground.
///
/// `uv` is `count` pairs. `out` receives `count * PROJECTION_VALUES`, and
/// `status` `count` codes: 0 for a projection, otherwise the refusal reason
/// (1 above horizon, 2 too shallow, 3 out of range, 4 bad pose). A refused
/// point's six output values are left untouched.
///
/// Batched because the map builder asks for a thousand of these per frame and
/// a per-call boundary crossing would cost more than the arithmetic.
///
/// Returns the number projected, or -1 on a null or invalid argument.
///
/// # Safety
/// Every pointer must be null or point to at least the stated length.
#[no_mangle]
pub unsafe extern "C" fn vigil_project_batch(
    pose: *const f64,
    uv: *const f64,
    count: u32,
    contact_sigma_deg: f64,
    sigma: *const f64,
    enforce_range: u32,
    out: *mut f64,
    status: *mut i32,
) -> i32 {
    let Some(pose) = read_pose(pose) else {
        return -1;
    };
    if uv.is_null() || out.is_null() || status.is_null() {
        return -1;
    }
    let n = count as usize;
    let points = core::slice::from_raw_parts(uv, n * 2);
    let results = core::slice::from_raw_parts_mut(out, n * PROJECTION_VALUES as usize);
    let codes = core::slice::from_raw_parts_mut(status, n);
    let sigma = read_sigma(sigma);
    let mut projected = 0i32;
    for i in 0..n {
        match camera::project_to_ground(
            &pose,
            points[i * 2],
            points[i * 2 + 1],
            contact_sigma_deg,
            &sigma,
            enforce_range != 0,
        ) {
            Ok(p) => {
                let at = i * PROJECTION_VALUES as usize;
                results[at] = p.position.lat;
                results[at + 1] = p.position.lon;
                results[at + 2] = p.ground_distance_meters;
                results[at + 3] = p.bearing_deg;
                results[at + 4] = p.uncertainty.along_meters;
                results[at + 5] = p.uncertainty.across_meters;
                codes[i] = 0;
                projected += 1;
            }
            Err(failure) => codes[i] = failure_code(failure),
        }
    }
    projected
}

/// Where `count` ground positions appear in the image.
///
/// `points` is `count` `[lat, lon]` pairs; `out` receives `count` `[u, v]`
/// pairs, unclipped. `status` is 0 for a point in front of the camera and 1
/// for one behind it, whose `out` pair is left untouched.
///
/// Returns the number in front, or -1 on a null or invalid argument.
///
/// # Safety
/// Every pointer must be null or point to at least the stated length.
#[no_mangle]
pub unsafe extern "C" fn vigil_image_coordinates_batch(
    pose: *const f64,
    points: *const f64,
    count: u32,
    out: *mut f64,
    status: *mut i32,
) -> i32 {
    let Some(pose) = read_pose(pose) else {
        return -1;
    };
    if points.is_null() || out.is_null() || status.is_null() {
        return -1;
    }
    let n = count as usize;
    let input = core::slice::from_raw_parts(points, n * 2);
    let output = core::slice::from_raw_parts_mut(out, n * 2);
    let codes = core::slice::from_raw_parts_mut(status, n);
    let mut ahead = 0i32;
    for i in 0..n {
        match camera::image_coordinates(&pose, LatLon::new(input[i * 2], input[i * 2 + 1])) {
            Some((u, v)) => {
                output[i * 2] = u;
                output[i * 2 + 1] = v;
                codes[i] = 0;
                ahead += 1;
            }
            None => codes[i] = 1,
        }
    }
    ahead
}

/// The camera's ground footprint as `[lat, lon]` pairs.
///
/// Returns the number of points written, 0 when the camera sees no ground, or
/// -1 on a null buffer, an invalid pose, or a ring that would not fit in
/// `capacity` pairs.
///
/// # Safety
/// `out` must be null or point to `capacity * 2` writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_footprint(
    pose: *const f64,
    segments: u32,
    out: *mut f64,
    capacity: u32,
) -> i32 {
    let Some(pose) = read_pose(pose) else {
        return -1;
    };
    if out.is_null() {
        return -1;
    }
    let ring = camera::footprint(&pose, segments as usize);
    if ring.len() > capacity as usize {
        return -1;
    }
    let buffer = core::slice::from_raw_parts_mut(out, ring.len() * 2);
    for (i, p) in ring.iter().enumerate() {
        buffer[i * 2] = p.lat;
        buffer[i * 2 + 1] = p.lon;
    }
    ring.len() as i32
}

/// Writes `[near, far]` ground distances. A value is negative where the
/// camera reaches no ground on that edge.
///
/// # Safety
/// `out` must be null or point to two writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_ground_span(pose: *const f64, out: *mut f64) -> i32 {
    let Some(pose) = read_pose(pose) else {
        return -1;
    };
    if out.is_null() {
        return -1;
    }
    *out = camera::near_ground_distance(&pose).unwrap_or(-1.0);
    *out.add(1) = camera::far_ground_distance(&pose).unwrap_or(-1.0);
    0
}

// --------------------------------------------------------------- assignment

/// Optimal rectangular assignment. `out` receives one entry per row: the
/// column it takes, or -1 for a row left unassigned.
///
/// Returns the number of rows assigned, or -1 on a null or oversized problem.
///
/// # Safety
/// `cost` must point to `rows * cols` readable `f64` and `out` to `rows`
/// writable `i64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_assign(
    cost: *const f64,
    rows: u32,
    cols: u32,
    out: *mut i64,
) -> i32 {
    if cost.is_null() || out.is_null() {
        return -1;
    }
    // A ceiling rather than a limit anybody will meet: 4096 tracks against
    // 4096 detections is 16 M cells and 68 billion operations. Refusing is
    // the right answer to a caller that has lost track of its own sizes.
    if rows > 4096 || cols > 4096 {
        return -1;
    }
    let (r, c) = (rows as usize, cols as usize);
    let matrix = core::slice::from_raw_parts(cost, r * c);
    let assignment = assign::solve(matrix, r, c);
    let output = core::slice::from_raw_parts_mut(out, r);
    let mut count = 0i32;
    for (i, column) in assignment.iter().enumerate() {
        if *column == usize::MAX {
            output[i] = -1;
        } else {
            output[i] = *column as i64;
            count += 1;
        }
    }
    count
}

// ------------------------------------------------------------------- kalman

/// # Safety
/// `state` must point to [`KALMAN_VALUES`] readable `f64`.
unsafe fn read_kalman(state: *const f64) -> KalmanBox {
    let v = core::slice::from_raw_parts(state, KALMAN_VALUES as usize);
    let mut filter = KalmanBox {
        mean: [0.0; 8],
        covariance: [[0.0; 8]; 8],
    };
    filter.mean.copy_from_slice(&v[..8]);
    for i in 0..8 {
        filter.covariance[i].copy_from_slice(&v[8 + i * 8..8 + i * 8 + 8]);
    }
    filter
}

/// # Safety
/// `state` must point to [`KALMAN_VALUES`] writable `f64`.
unsafe fn write_kalman(state: *mut f64, filter: &KalmanBox) {
    let v = core::slice::from_raw_parts_mut(state, KALMAN_VALUES as usize);
    v[..8].copy_from_slice(&filter.mean);
    for i in 0..8 {
        v[8 + i * 8..8 + i * 8 + 8].copy_from_slice(&filter.covariance[i]);
    }
}

/// # Safety
/// `state` must be null or point to [`KALMAN_VALUES`] writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_initiate(
    state: *mut f64,
    cx: f64,
    cy: f64,
    aspect: f64,
    height: f64,
) -> i32 {
    if state.is_null() {
        return -1;
    }
    write_kalman(state, &KalmanBox::initiate(cx, cy, aspect, height));
    0
}

/// # Safety
/// `state` must be null or point to [`KALMAN_VALUES`] writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_predict(state: *mut f64, dt_seconds: f64) -> i32 {
    if state.is_null() {
        return -1;
    }
    let mut filter = read_kalman(state);
    filter.predict(dt_seconds);
    write_kalman(state, &filter);
    0
}

/// Returns 0 on success, 1 when the innovation covariance was singular and
/// the state was left untouched, -1 on a null pointer.
///
/// # Safety
/// `state` must be null or point to [`KALMAN_VALUES`] writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_update(
    state: *mut f64,
    cx: f64,
    cy: f64,
    aspect: f64,
    height: f64,
) -> i32 {
    if state.is_null() {
        return -1;
    }
    let mut filter = read_kalman(state);
    if !filter.update(cx, cy, aspect, height) {
        return 1;
    }
    write_kalman(state, &filter);
    0
}

/// Squared Mahalanobis distance, or infinity when it cannot be computed.
///
/// # Safety
/// `state` must be null or point to [`KALMAN_VALUES`] readable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_gate(
    state: *const f64,
    cx: f64,
    cy: f64,
    aspect: f64,
    height: f64,
    position_only: u32,
) -> f64 {
    if state.is_null() {
        return f64::INFINITY;
    }
    read_kalman(state).gating_distance(cx, cy, aspect, height, position_only != 0)
}

/// Gate `count` measurements against one track in a single crossing.
///
/// # Safety
/// `state` must point to [`KALMAN_VALUES`] readable `f64`, `boxes` to
/// `count * 4`, and `out` to `count` writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_gate_batch(
    state: *const f64,
    boxes: *const f64,
    count: u32,
    position_only: u32,
    out: *mut f64,
) -> i32 {
    if state.is_null() || boxes.is_null() || out.is_null() {
        return -1;
    }
    let filter = read_kalman(state);
    let n = count as usize;
    let input = core::slice::from_raw_parts(boxes, n * 4);
    let output = core::slice::from_raw_parts_mut(out, n);
    for i in 0..n {
        output[i] = filter.gating_distance(
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
            position_only != 0,
        );
    }
    0
}

/// Apply a 2x3 affine `[a, b, tx, c, d, ty]` in normalised image coordinates.
///
/// # Safety
/// `state` must point to [`KALMAN_VALUES`] writable `f64` and `warp` to six
/// readable `f64`.
#[no_mangle]
pub unsafe extern "C" fn vigil_kalman_warp(state: *mut f64, warp: *const f64) -> i32 {
    if state.is_null() || warp.is_null() {
        return -1;
    }
    let w = core::slice::from_raw_parts(warp, 6);
    if w.iter().any(|x| !x.is_finite()) {
        return -1;
    }
    let mut filter = read_kalman(state);
    filter.apply_warp(&[[w[0], w[1], w[2]], [w[3], w[4], w[5]]]);
    write_kalman(state, &filter);
    0
}

// -------------------------------------------------------------------- ortho

/// # Safety
/// `grid` must point to [`GRID_VALUES`] readable `f64`.
unsafe fn read_grid(grid: *const f64) -> Option<GroundGrid> {
    if grid.is_null() {
        return None;
    }
    let v = core::slice::from_raw_parts(grid, GRID_VALUES as usize);
    if v.iter().any(|x| !x.is_finite()) || v[2] <= 0.0 || v[3] < 1.0 || v[4] < 1.0 {
        return None;
    }
    Some(GroundGrid {
        origin: LatLon::new(v[0], v[1]),
        cell_size_m: v[2],
        rows: v[3] as usize,
        cols: v[4] as usize,
    })
}

/// Resample one BGR frame onto a ground grid.
///
/// Returns the number of cells filled, or -1 on a null or invalid argument.
///
/// # Safety
/// `image` must point to `height * stride` readable bytes; `out_colour` to
/// `rows * cols * 3` writable bytes, `out_valid` to `rows * cols`, and
/// `out_resolution` to `rows * cols` writable `f32`.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn vigil_ortho_sample(
    pose: *const f64,
    grid: *const f64,
    image: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    lattice: u32,
    out_colour: *mut u8,
    out_valid: *mut u8,
    out_resolution: *mut f32,
) -> i64 {
    let Some(pose) = read_pose(pose) else {
        return -1;
    };
    let Some(grid) = read_grid(grid) else {
        return -1;
    };
    if image.is_null() || out_colour.is_null() || out_valid.is_null() || out_resolution.is_null() {
        return -1;
    }
    if width == 0 || height == 0 || (stride as usize) < width as usize * 3 {
        return -1;
    }
    let cells = grid.cells();
    let pixels = core::slice::from_raw_parts(image, height as usize * stride as usize);
    let mut sample = GroundSample {
        colour: core::slice::from_raw_parts_mut(out_colour, cells * 3),
        valid: core::slice::from_raw_parts_mut(out_valid, cells),
        resolution: core::slice::from_raw_parts_mut(out_resolution, cells),
    };
    ortho::sample_frame(
        &pose,
        &grid,
        pixels,
        width as usize,
        height as usize,
        stride as usize,
        lattice as usize,
        &mut sample,
    ) as i64
}

/// Create a per-cell median accumulator. The caller owns it until
/// [`vigil_median_destroy`]. Null on a size that would not fit in memory.
#[no_mangle]
pub extern "C" fn vigil_median_create(cells: u32, capacity: u32) -> *mut c_void {
    if cells == 0 || capacity == 0 {
        return core::ptr::null_mut();
    }
    // Refuse rather than attempt a several-gigabyte allocation: a caller that
    // asks for this has computed a grid size wrongly, and a failed allocation
    // inside a `Vec` aborts the process.
    if (cells as u64) * (capacity.min(255) as u64) * 3 > 1 << 31 {
        return core::ptr::null_mut();
    }
    let accumulator = Box::new(MedianAccumulator::new(cells as usize, capacity as usize));
    Box::into_raw(accumulator) as *mut c_void
}

/// # Safety
/// `handle` must be null or a pointer returned by [`vigil_median_create`] and
/// not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn vigil_median_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut MedianAccumulator));
    }
}

/// Fold one sample in.
///
/// # Safety
/// `handle` must be live; `colour` must point to `cells * 3` readable bytes
/// and `valid` to `cells`.
#[no_mangle]
pub unsafe extern "C" fn vigil_median_add(
    handle: *mut c_void,
    colour: *const u8,
    valid: *const u8,
    cells: u32,
) -> i32 {
    if handle.is_null() || colour.is_null() || valid.is_null() {
        return -1;
    }
    let accumulator = &mut *(handle as *mut MedianAccumulator);
    let n = cells as usize;
    accumulator.add(
        core::slice::from_raw_parts(colour, n * 3),
        core::slice::from_raw_parts(valid, n),
    );
    0
}

/// Read the per-cell median and its statistics.
///
/// Returns the number of cells that met `minimum_samples`, or -1 on a null
/// argument.
///
/// # Safety
/// Every output must point to at least `cells` entries (`cells * 3` for
/// `out_colour`).
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn vigil_median_result(
    handle: *mut c_void,
    minimum_samples: u32,
    cells: u32,
    out_colour: *mut u8,
    out_valid: *mut u8,
    out_samples: *mut u16,
    out_deviation: *mut u8,
    out_disturbed: *mut u8,
) -> i64 {
    if handle.is_null()
        || out_colour.is_null()
        || out_valid.is_null()
        || out_samples.is_null()
        || out_deviation.is_null()
        || out_disturbed.is_null()
    {
        return -1;
    }
    let accumulator = &*(handle as *mut MedianAccumulator);
    let n = cells as usize;
    let mut stats = vec![CellStats::default(); n];
    let filled = accumulator.median(
        minimum_samples as usize,
        core::slice::from_raw_parts_mut(out_colour, n * 3),
        core::slice::from_raw_parts_mut(out_valid, n),
        &mut stats,
    );
    let samples = core::slice::from_raw_parts_mut(out_samples, n);
    let deviation = core::slice::from_raw_parts_mut(out_deviation, n);
    let disturbed = core::slice::from_raw_parts_mut(out_disturbed, n);
    for i in 0..n {
        samples[i] = stats[i].samples;
        deviation[i] = stats[i].deviation;
        disturbed[i] = stats[i].disturbed_percent;
    }
    filled as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose_values() -> [f64; 9] {
        [33.8938, 35.5018, 4.0, 0.0, -25.0, 0.0, 62.0, 36.0, 60.0]
    }

    #[test]
    fn every_entry_point_refuses_null_rather_than_dereferencing_it() {
        unsafe {
            assert_eq!(vigil_layout(core::ptr::null_mut(), 5), -1);
            assert_eq!(vigil_layout([0u32; 5].as_mut_ptr(), 4), -1);
            assert_eq!(vigil_destination_point(0.0, 0.0, 0.0, 1.0, core::ptr::null_mut()), -1);
            let pose = pose_values();
            assert_eq!(
                vigil_project_batch(
                    core::ptr::null(),
                    core::ptr::null(),
                    0,
                    0.0,
                    core::ptr::null(),
                    1,
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                -1
            );
            assert_eq!(
                vigil_project_batch(
                    pose.as_ptr(),
                    core::ptr::null(),
                    1,
                    0.0,
                    core::ptr::null(),
                    1,
                    core::ptr::null_mut(),
                    core::ptr::null_mut()
                ),
                -1
            );
            assert_eq!(vigil_assign(core::ptr::null(), 1, 1, core::ptr::null_mut()), -1);
            assert_eq!(vigil_kalman_predict(core::ptr::null_mut(), 0.1), -1);
            assert_eq!(vigil_kalman_gate(core::ptr::null(), 0.0, 0.0, 1.0, 1.0, 0), f64::INFINITY);
            assert_eq!(vigil_footprint(pose.as_ptr(), 8, core::ptr::null_mut(), 100), -1);
            assert_eq!(vigil_ground_span(core::ptr::null(), core::ptr::null_mut()), -1);
            assert_eq!(vigil_median_add(core::ptr::null_mut(), core::ptr::null(), core::ptr::null(), 1), -1);
            vigil_median_destroy(core::ptr::null_mut());
        }
    }

    #[test]
    fn a_pose_full_of_nonsense_is_refused_at_the_boundary() {
        unsafe {
            let mut pose = pose_values();
            pose[0] = f64::NAN;
            let uv = [0.5, 0.8];
            let mut out = [0.0f64; 6];
            let mut status = [0i32; 1];
            assert_eq!(
                vigil_project_batch(
                    pose.as_ptr(),
                    uv.as_ptr(),
                    1,
                    0.0,
                    core::ptr::null(),
                    1,
                    out.as_mut_ptr(),
                    status.as_mut_ptr()
                ),
                -1
            );
        }
    }

    #[test]
    fn a_batch_projection_reports_each_point_separately() {
        unsafe {
            let pose = pose_values();
            // Ground, horizon, and past the far edge but inside the range.
            let uv = [0.5, 0.9, 0.5, 0.0, 0.5, 0.5];
            let mut out = [0.0f64; 18];
            let mut status = [-9i32; 3];
            let count = vigil_project_batch(
                pose.as_ptr(),
                uv.as_ptr(),
                3,
                0.75,
                core::ptr::null(),
                1,
                out.as_mut_ptr(),
                status.as_mut_ptr(),
            );
            assert!(count >= 1);
            assert_eq!(status[0], 0, "the bottom of the frame is ground");
            assert!(out[2] > 0.0 && out[2] < 60.0);
            assert!(out[4] > 0.0 && out[5] > 0.0, "and it carries its error ellipse");
            assert!(status[2] == 0, "the centre row of a tilted camera is ground too");
        }
    }

    #[test]
    fn the_kalman_state_survives_a_round_trip_through_the_boundary() {
        unsafe {
            let mut state = [0.0f64; KALMAN_VALUES as usize];
            assert_eq!(vigil_kalman_initiate(state.as_mut_ptr(), 0.5, 0.6, 0.5, 0.2), 0);
            assert_eq!(state[0], 0.5);
            for _ in 0..30 {
                assert_eq!(vigil_kalman_predict(state.as_mut_ptr(), 1.0 / 15.0), 0);
                assert_eq!(vigil_kalman_update(state.as_mut_ptr(), 0.5, 0.6, 0.5, 0.2), 0);
            }
            assert!(vigil_kalman_gate(state.as_ptr(), 0.5, 0.6, 0.5, 0.2, 0) < 1.0);
            let warp = [1.0, 0.0, 0.05, 0.0, 1.0, 0.0];
            assert_eq!(vigil_kalman_warp(state.as_mut_ptr(), warp.as_ptr()), 0);
            assert!((state[0] - 0.55).abs() < 1e-9);
            let bad = [f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0];
            assert_eq!(vigil_kalman_warp(state.as_mut_ptr(), bad.as_ptr()), -1);
            assert!((state[0] - 0.55).abs() < 1e-9, "a refused warp changes nothing");
        }
    }

    #[test]
    fn assignment_crosses_the_boundary_intact() {
        unsafe {
            let cost = [0.10, 0.20, 0.25, 0.90];
            let mut out = [0i64; 2];
            assert_eq!(vigil_assign(cost.as_ptr(), 2, 2, out.as_mut_ptr()), 2);
            assert_eq!(out, [1, 0]);
            let three = [0.7, 0.2, 0.9];
            let mut out3 = [0i64; 3];
            assert_eq!(vigil_assign(three.as_ptr(), 3, 1, out3.as_mut_ptr()), 1);
            assert_eq!(out3, [-1, 0, -1]);
            assert_eq!(vigil_assign(cost.as_ptr(), 5000, 2, out.as_mut_ptr()), -1);
        }
    }

    #[test]
    fn the_accumulator_is_owned_and_released_exactly_once() {
        unsafe {
            let handle = vigil_median_create(4, 5);
            assert!(!handle.is_null());
            let colour = [10u8; 12];
            let valid = [1u8; 4];
            for _ in 0..5 {
                assert_eq!(vigil_median_add(handle, colour.as_ptr(), valid.as_ptr(), 4), 0);
            }
            let mut out_colour = [0u8; 12];
            let mut out_valid = [0u8; 4];
            let mut samples = [0u16; 4];
            let mut deviation = [0u8; 4];
            let mut disturbed = [0u8; 4];
            let filled = vigil_median_result(
                handle,
                1,
                4,
                out_colour.as_mut_ptr(),
                out_valid.as_mut_ptr(),
                samples.as_mut_ptr(),
                deviation.as_mut_ptr(),
                disturbed.as_mut_ptr(),
            );
            assert_eq!(filled, 4);
            assert_eq!(out_colour, [10u8; 12]);
            assert_eq!(samples, [5u16; 4]);
            vigil_median_destroy(handle);
            assert!(vigil_median_create(u32::MAX, 255).is_null(), "an absurd size is refused");
        }
    }

    #[test]
    fn the_layout_is_what_the_binding_will_check() {
        let mut out = [0u32; 5];
        unsafe {
            assert_eq!(vigil_layout(out.as_mut_ptr(), 5), 5);
        }
        assert_eq!(out, [9, 5, 6, 5, 72]);
        assert_eq!(vigil_abi_version(), ABI_VERSION);
    }
}
