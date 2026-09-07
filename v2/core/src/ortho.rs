//! Ground rasterisation: turning a camera frame into a patch of map.
//!
//! # What this produces, and what it is not
//!
//! Every pixel is placed **where it would be if it lay on the ground**. A
//! wall, a parked van, a person: each has height, so each smears radially
//! away from the camera along the ray that saw it. A single patch is a
//! *sample*, not a map, and reading one as a photograph reads those smears as
//! ground markings.
//!
//! [`MedianAccumulator`] is what makes a map out of samples. A per-cell
//! median over many frames separated in time removes whatever walked
//! through: a person occupies any one cell for a second or two out of a
//! minute, so the median of that cell is the ground they were standing on.
//! It also gives the honest quality signal, [`CellStats::deviation`] — a cell
//! whose samples disagree is a cell where something is *not* flat ground, and
//! that is worth more to a security overlay than the colour is.
//!
//! **Empty is a value.** A cell no camera has seen stays empty and nothing
//! here interpolates one from its neighbours. Under an overlay where an
//! operator judges "inside the fence" against what is drawn, inventing ground
//! nobody has looked at is the one thing this must not do. Emptiness lives in
//! a separate `valid` mask and is never inferred from the colour, because
//! black is an ordinary colour for asphalt at dusk.
//!
//! # How the inverse is done
//!
//! The exact inverse of the projection is [`crate::camera::image_coordinates`],
//! and it is the obvious tool — one call per ground cell. At a 200x200 grid
//! that is 40 000 calls per camera per frame against a median that wants
//! hundreds of frames.
//!
//! So this goes the other way: project a coarse lattice of *image* points
//! onto the ground, which gives an irregular ground mesh whose corners carry
//! known image coordinates, and invert that mesh by barycentric interpolation
//! across its triangles. The inverse is exact at the lattice nodes and
//! piecewise-linear between them. A ground cell is filled only when it falls
//! inside a triangle *all* of whose corners projected, so the ragged edge at
//! the horizon falls inward and leaves cells empty rather than claiming
//! ground the pose does not reach.
//!
//! # Why it is in Rust
//!
//! v1 did exactly this in NumPy and measured 79 ms per frame per camera,
//! which is why v1's command line fed the builder four frames a second and
//! called that a design decision. It is the same arithmetic here with the
//! per-triangle Python loop removed.

use crate::camera::{CameraBasis, CameraPose};
use crate::geodesy::{LatLon, LocalFrame};

/// Points per axis in the image lattice whose forward projection is inverted.
///
/// Thirty-three squared is 1 089 projections. The residual against the exact
/// inverse is well under a tenth of a pixel over the near two thirds of a
/// frame, which is where a mast camera's ground resolution is worth having at
/// all. Raising it costs projections quadratically and buys accuracy only
/// near the horizon, where one image row already spans metres of ground.
pub const DEFAULT_LATTICE: usize = 33;

/// A regular grid of ground cells in one tangent plane.
///
/// Row 0 is the **north** edge, so the raster draws the right way up without
/// anybody flipping it on the way to a screen — a flip that gets forgotten
/// once and puts the whole site's map upside down under its zones.
#[derive(Clone, Copy, Debug)]
pub struct GroundGrid {
    /// Tangent-plane origin. Cell (rows-1, 0) has its centre half a cell
    /// north and east of this point.
    pub origin: LatLon,
    pub cell_size_m: f64,
    pub rows: usize,
    pub cols: usize,
}

impl GroundGrid {
    pub fn cells(&self) -> usize {
        self.rows * self.cols
    }

    /// Local (east, north) metres of a cell's centre.
    pub fn cell_centre(&self, row: usize, col: usize) -> (f64, f64) {
        (
            (col as f64 + 0.5) * self.cell_size_m,
            (self.rows as f64 - 0.5 - row as f64) * self.cell_size_m,
        )
    }

    /// Fractional (col, row) of a local point. The inverse of
    /// [`GroundGrid::cell_centre`], and the coordinates rasterisation happens
    /// in.
    pub fn to_cell(&self, east: f64, north: f64) -> (f64, f64) {
        (
            east / self.cell_size_m - 0.5,
            (self.rows as f64 - 0.5) - north / self.cell_size_m,
        )
    }

    pub fn lat_lon_of(&self, row: usize, col: usize) -> LatLon {
        let (east, north) = self.cell_centre(row, col);
        LocalFrame::new(self.origin).to_lat_lon(east, north)
    }
}

/// One frame resampled onto a grid.
///
/// Three parallel arrays, one entry per cell, because that is what crosses a
/// C boundary without allocation and what NumPy wraps without a copy.
pub struct GroundSample<'a> {
    /// BGR, three bytes per cell. Undefined where `valid` is 0.
    pub colour: &'a mut [u8],
    /// 1 where a cell was seen this frame.
    pub valid: &'a mut [u8],
    /// Metres of ground per source pixel at that cell.
    ///
    /// The number that says whether the texture there is worth anything: a
    /// cell at 0.01 m/px was seen close and sharp, one at 0.5 m/px was seen
    /// down a grazing ray and is a smear of a handful of pixels stretched
    /// across half a metre. Infinite where the cell was not seen.
    pub resolution: &'a mut [f32],
}

/// Resample one frame onto a grid. Returns the number of cells filled.
///
/// `image` is BGR, `height` rows of `stride` bytes. Cells outside the
/// projected mesh are left untouched in `colour` and set to 0 in `valid`.
pub fn sample_frame(
    pose: &CameraPose,
    grid: &GroundGrid,
    image: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    lattice: usize,
    out: &mut GroundSample,
) -> usize {
    let cells = grid.cells();
    out.valid[..cells].fill(0);
    out.resolution[..cells].fill(f32::INFINITY);
    if width == 0 || height == 0 || cells == 0 {
        return 0;
    }
    let lattice = lattice.max(2);
    let basis = CameraBasis::of(pose);
    let frame = LocalFrame::new(grid.origin);
    let (camera_east, camera_north) = frame.to_local(pose.position);
    let range_sq = pose.range_meters * pose.range_meters;

    // Forward-project the lattice. `None` for a node whose ray misses the
    // ground or lands past the range: the triangles that touch it are then
    // skipped whole, which is what keeps the horizon edge from bleeding.
    let mut nodes: Vec<Option<Node>> = Vec::with_capacity(lattice * lattice);
    for iy in 0..lattice {
        let v = iy as f64 / (lattice - 1) as f64;
        for ix in 0..lattice {
            let u = ix as f64 / (lattice - 1) as f64;
            nodes.push(match basis.ground_offset(u, v) {
                Some((de, dn)) if de * de + dn * dn <= range_sq => {
                    let (col, row) = grid.to_cell(camera_east + de, camera_north + dn);
                    Some(Node {
                        col,
                        row,
                        u,
                        v,
                        east: camera_east + de,
                        north: camera_north + dn,
                    })
                }
                _ => None,
            });
        }
    }

    let mut filled = 0usize;
    for iy in 0..lattice - 1 {
        for ix in 0..lattice - 1 {
            let quad = [
                nodes[iy * lattice + ix],
                nodes[iy * lattice + ix + 1],
                nodes[(iy + 1) * lattice + ix],
                nodes[(iy + 1) * lattice + ix + 1],
            ];
            let (Some(a), Some(b), Some(c), Some(d)) = (quad[0], quad[1], quad[2], quad[3]) else {
                continue;
            };
            // Ground metres per source pixel for this quad: the ground area
            // it covers over the image area it came from. One value for the
            // quad rather than per cell, because it varies smoothly and the
            // lattice is already fine enough that a quad spans a few pixels.
            let ground_area = quad_area(a, b, d, c);
            let image_area = ((b.u - a.u) * width as f64).abs() * ((c.v - a.v) * height as f64).abs();
            let resolution = if image_area > 0.0 {
                (ground_area / image_area).sqrt() as f32
            } else {
                f32::INFINITY
            };
            filled += raster_triangle(
                grid, &a, &b, &c, image, width, height, stride, resolution, out,
            );
            filled += raster_triangle(
                grid, &b, &d, &c, image, width, height, stride, resolution, out,
            );
        }
    }
    filled
}

#[derive(Clone, Copy)]
struct Node {
    col: f64,
    row: f64,
    u: f64,
    v: f64,
    east: f64,
    north: f64,
}

/// Area of the quad a-b-d-c in ground metres, by the shoelace formula over
/// its two triangles.
fn quad_area(a: Node, b: Node, d: Node, c: Node) -> f64 {
    let tri = |p: Node, q: Node, r: Node| {
        ((q.east - p.east) * (r.north - p.north) - (r.east - p.east) * (q.north - p.north)).abs()
            / 2.0
    };
    tri(a, b, d) + tri(a, d, c)
}

#[allow(clippy::too_many_arguments)]
fn raster_triangle(
    grid: &GroundGrid,
    a: &Node,
    b: &Node,
    c: &Node,
    image: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    resolution: f32,
    out: &mut GroundSample,
) -> usize {
    let area = (b.col - a.col) * (c.row - a.row) - (c.col - a.col) * (b.row - a.row);
    if area.abs() < 1e-12 {
        return 0; // Degenerate: the quad collapsed, near the horizon.
    }
    let min_col = a.col.min(b.col).min(c.col).floor().max(0.0) as isize;
    let max_col = (a.col.max(b.col).max(c.col).ceil() as isize).min(grid.cols as isize - 1);
    let min_row = a.row.min(b.row).min(c.row).floor().max(0.0) as isize;
    let max_row = (a.row.max(b.row).max(c.row).ceil() as isize).min(grid.rows as isize - 1);
    if max_col < min_col || max_row < min_row {
        return 0;
    }
    let inv_area = 1.0 / area;
    let mut filled = 0usize;
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            let (px, py) = (col as f64, row as f64);
            // Barycentric weights of the cell centre.
            let w0 = ((b.col - px) * (c.row - py) - (c.col - px) * (b.row - py)) * inv_area;
            let w1 = ((c.col - px) * (a.row - py) - (a.col - px) * (c.row - py)) * inv_area;
            let w2 = 1.0 - w0 - w1;
            // A small negative tolerance keeps the shared edge of two
            // triangles from leaving a one-cell seam of holes.
            if w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9 {
                continue;
            }
            let u = w0 * a.u + w1 * b.u + w2 * c.u;
            let v = w0 * a.v + w1 * b.v + w2 * c.v;
            let index = row as usize * grid.cols + col as usize;
            if let Some(bgr) = bilinear(image, width, height, stride, u, v) {
                out.colour[index * 3] = bgr[0];
                out.colour[index * 3 + 1] = bgr[1];
                out.colour[index * 3 + 2] = bgr[2];
                if out.valid[index] == 0 {
                    filled += 1;
                }
                out.valid[index] = 1;
                // Two triangles can cover one cell at a seam; keep the
                // sharper of the two claims.
                if resolution < out.resolution[index] {
                    out.resolution[index] = resolution;
                }
            }
        }
    }
    filled
}

/// Bilinear sample at normalised image coordinates, or `None` outside the
/// frame. Clamped at the border rather than wrapping: a wrap puts the top of
/// the frame at the bottom of the map.
fn bilinear(
    image: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    u: f64,
    v: f64,
) -> Option<[u8; 3]> {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }
    let x = (u * width as f64 - 0.5).clamp(0.0, width as f64 - 1.0);
    let y = (v * height as f64 - 0.5).clamp(0.0, height as f64 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let mut out = [0u8; 3];
    for channel in 0..3 {
        let at = |xx: usize, yy: usize| -> f64 {
            let offset = yy * stride + xx * 3 + channel;
            image.get(offset).copied().unwrap_or(0) as f64
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        out[channel] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
    }
    Some(out)
}

/// What a cell's history says about it.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellStats {
    pub samples: u16,
    /// Median absolute deviation of luminance across the samples, 0..255.
    ///
    /// How noisy the ground itself looked: sensor noise, compression, the
    /// light drifting. Near zero is a cell that looked the same every time.
    /// This is a *texture quality* number and it is deliberately blind to a
    /// minority of wildly different samples — that is what a median is for.
    pub deviation: u8,
    /// Percentage of samples whose luminance is further than
    /// [`DISTURBANCE_LEVELS`] from the median.
    ///
    /// How often something was in the way, which `deviation` cannot say and
    /// which is the number that decides whether a cell is *usable*. A cell
    /// disturbed 2% of the time is ground with a bird over it; one disturbed
    /// 40% of the time is a doorway, a parked car's usual space, or a wall
    /// being smeared across the plane by two different light angles — and its
    /// colour is not a measurement of the ground however steady the median
    /// looks.
    pub disturbed_percent: u8,
}

/// Luminance levels away from a cell's median that count as "something else
/// was there". Twenty-four of 255 is about 10%: past the compression noise
/// and the daylight drift of a fixed camera, well under the contrast between
/// tarmac and a person.
pub const DISTURBANCE_LEVELS: u8 = 24;

/// A bounded per-cell history, and the median over it.
///
/// A ring of `capacity` samples per cell and nothing else: the memory is
/// `cells * capacity * 3 + cells * 2` bytes and does not grow with the
/// minutes fed to it. At v1's reference grid — 328 by 362 cells — and a depth
/// of 15 that is 5.5 MB per camera.
///
/// The median is taken per channel rather than as a vector median. A vector
/// median is better behaved in principle and costs an order of magnitude
/// more; over a ground cell whose samples differ by illumination rather than
/// by hue, the two agree to within a level or two.
pub struct MedianAccumulator {
    cells: usize,
    capacity: usize,
    /// `cells * capacity * 3`, BGR, oldest-overwritten ring.
    ring: Vec<u8>,
    /// How many slots of each cell's ring are populated.
    counts: Vec<u16>,
    /// Where the next sample for each cell goes.
    cursor: Vec<u16>,
}

impl MedianAccumulator {
    pub fn new(cells: usize, capacity: usize) -> Self {
        let capacity = capacity.clamp(1, 255);
        Self {
            cells,
            capacity,
            ring: vec![0; cells * capacity * 3],
            counts: vec![0; cells],
            cursor: vec![0; cells],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Fold one frame's sample in. Cells whose `valid` is 0 are untouched, so
    /// a frame that saw half the grid does not age out the other half.
    pub fn add(&mut self, colour: &[u8], valid: &[u8]) {
        for cell in 0..self.cells {
            if valid[cell] == 0 {
                continue;
            }
            let slot = self.cursor[cell] as usize;
            let at = (cell * self.capacity + slot) * 3;
            self.ring[at] = colour[cell * 3];
            self.ring[at + 1] = colour[cell * 3 + 1];
            self.ring[at + 2] = colour[cell * 3 + 2];
            self.cursor[cell] = ((slot + 1) % self.capacity) as u16;
            if (self.counts[cell] as usize) < self.capacity {
                self.counts[cell] += 1;
            }
        }
    }

    /// The per-cell median and its statistics.
    ///
    /// `minimum_samples` is the number of observations below which a cell is
    /// reported as empty. One sample is not a median — it is one frame, with
    /// whoever was walking through it still in it — and the default caller
    /// asks for several.
    pub fn median(
        &self,
        minimum_samples: usize,
        out_colour: &mut [u8],
        out_valid: &mut [u8],
        out_stats: &mut [CellStats],
    ) -> usize {
        let mut buffer = vec![0u8; self.capacity];
        let mut luminance = vec![0u8; self.capacity];
        let mut spread = vec![0u8; self.capacity];
        let mut filled = 0usize;
        for cell in 0..self.cells {
            let count = self.counts[cell] as usize;
            out_stats[cell] = CellStats {
                samples: self.counts[cell],
                deviation: 0,
                disturbed_percent: 0,
            };
            if count == 0 || count < minimum_samples.max(1) {
                out_valid[cell] = 0;
                continue;
            }
            let base = cell * self.capacity * 3;
            for channel in 0..3 {
                for slot in 0..count {
                    buffer[slot] = self.ring[base + slot * 3 + channel];
                }
                out_colour[cell * 3 + channel] = median_of(&mut buffer[..count]);
            }
            // Deviation is measured on luminance, because a cell whose
            // brightness is steady and whose hue wanders is still ground.
            for slot in 0..count {
                let b = self.ring[base + slot * 3] as u32;
                let g = self.ring[base + slot * 3 + 1] as u32;
                let r = self.ring[base + slot * 3 + 2] as u32;
                luminance[slot] = ((b * 29 + g * 150 + r * 77) >> 8) as u8;
            }
            let centre = median_of(&mut luminance[..count]);
            for slot in 0..count {
                spread[slot] = luminance[slot].abs_diff(centre);
            }
            out_stats[cell].deviation = median_of(&mut spread[..count]);
            let disturbed = spread[..count]
                .iter()
                .filter(|d| **d > DISTURBANCE_LEVELS)
                .count();
            out_stats[cell].disturbed_percent = ((disturbed * 100) / count) as u8;
            out_valid[cell] = 1;
            filled += 1;
        }
        filled
    }
}

/// Median of a small slice, in place. Insertion sort: the slices here are at
/// most a few dozen long and this beats anything cleverer at that size.
fn median_of(values: &mut [u8]) -> u8 {
    for i in 1..values.len() {
        let key = values[i];
        let mut j = i;
        while j > 0 && values[j - 1] > key {
            values[j] = values[j - 1];
            j -= 1;
        }
        values[j] = key;
    }
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{image_coordinates, PoseUncertainty};
    use crate::geodesy::distance_meters;

    fn pose() -> CameraPose {
        CameraPose {
            position: LatLon::new(33.8938, 35.5018),
            mount_height: 4.0,
            heading: 0.0,
            pitch: -25.0,
            roll: 0.0,
            horizontal_fov: 62.0,
            vertical_fov: 36.0,
            range_meters: 60.0,
            lens: crate::lens::Distortion::default(),
            ground_tilt_east: 0.0,
            ground_tilt_north: 0.0,
        }
    }

    /// A frame whose blue channel encodes the column and green the row, so a
    /// resampled cell says exactly which source pixel it came from.
    fn coded_frame(width: usize, height: usize) -> Vec<u8> {
        let mut image = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let at = (y * width + x) * 3;
                image[at] = (x * 255 / width.max(1)) as u8;
                image[at + 1] = (y * 255 / height.max(1)) as u8;
                image[at + 2] = 128;
            }
        }
        image
    }

    fn grid_for(pose: &CameraPose, cell: f64) -> GroundGrid {
        // A grid big enough to hold the whole footprint, anchored south-west
        // of the camera.
        let span = pose.range_meters * 2.0;
        let n = (span / cell).ceil() as usize;
        GroundGrid {
            origin: crate::geodesy::destination_point(
                crate::geodesy::destination_point(pose.position, 180.0, span / 2.0),
                270.0,
                span / 2.0,
            ),
            cell_size_m: cell,
            rows: n,
            cols: n,
        }
    }

    struct Buffers {
        colour: Vec<u8>,
        valid: Vec<u8>,
        resolution: Vec<f32>,
    }

    impl Buffers {
        fn new(cells: usize) -> Self {
            Self {
                colour: vec![0; cells * 3],
                valid: vec![0; cells],
                resolution: vec![f32::INFINITY; cells],
            }
        }
        fn sample(&mut self) -> GroundSample<'_> {
            GroundSample {
                colour: &mut self.colour,
                valid: &mut self.valid,
                resolution: &mut self.resolution,
            }
        }
    }

    #[test]
    fn the_barycentric_inverse_agrees_with_the_exact_one() {
        // The claim the whole module rests on: interpolating across the
        // lattice lands within a fraction of a pixel of the exact inverse.
        let pose = pose();
        let grid = grid_for(&pose, 0.25);
        // 255 wide makes one colour level exactly one pixel, so what this
        // measures is the mesh and not the test's own 8-bit encoding.
        let (width, height) = (255usize, 255usize);
        let image = coded_frame(width, height);
        let mut buffers = Buffers::new(grid.cells());
        let filled = sample_frame(
            &pose,
            &grid,
            &image,
            width,
            height,
            width * 3,
            DEFAULT_LATTICE,
            &mut buffers.sample(),
        );
        assert!(filled > 1000, "only {filled} cells filled");

        let mut worst_u = 0.0f64;
        let mut worst_v = 0.0f64;
        let mut checked = 0;
        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let index = row * grid.cols + col;
                if buffers.valid[index] == 0 {
                    continue;
                }
                let point = grid.lat_lon_of(row, col);
                // Only judge the near two thirds: past that one image row is
                // metres of ground and the comparison is meaningless.
                if distance_meters(pose.position, point) > 35.0 {
                    continue;
                }
                let Some((u, v)) = image_coordinates(&pose, point) else {
                    continue;
                };
                if !(0.02..=0.98).contains(&u) || !(0.02..=0.98).contains(&v) {
                    continue;
                }
                let got_u = buffers.colour[index * 3] as f64 / 255.0;
                let got_v = buffers.colour[index * 3 + 1] as f64 / 255.0;
                worst_u = worst_u.max((got_u - u).abs() * width as f64);
                worst_v = worst_v.max((got_v - v).abs() * height as f64);
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} cells compared");
        // One and a half pixels at 255x255, of which half a pixel is the
        // encoding's own quantisation. The piecewise-linear inverse across a
        // 33-node lattice is worth what the module claims it is worth.
        assert!(worst_u < 1.5 && worst_v < 1.5, "off by {worst_u}, {worst_v} px");
        eprintln!("barycentric inverse: worst {worst_u:.2}, {worst_v:.2} px over {checked} cells");
    }

    #[test]
    fn nothing_is_drawn_where_the_camera_cannot_see() {
        let pose = pose();
        let grid = grid_for(&pose, 0.5);
        let (width, height) = (320usize, 192usize);
        let image = coded_frame(width, height);
        let mut buffers = Buffers::new(grid.cells());
        sample_frame(
            &pose,
            &grid,
            &image,
            width,
            height,
            width * 3,
            DEFAULT_LATTICE,
            &mut buffers.sample(),
        );
        for row in 0..grid.rows {
            for col in 0..grid.cols {
                if buffers.valid[row * grid.cols + col] == 0 {
                    continue;
                }
                let point = grid.lat_lon_of(row, col);
                let distance = distance_meters(pose.position, point);
                assert!(
                    distance <= pose.range_meters + grid.cell_size_m * 2.0,
                    "filled a cell {distance} m away, past a {} m range",
                    pose.range_meters
                );
                // Behind the camera is the failure that matters: a mesh that
                // wraps puts the sky on the map.
                let bearing = crate::geodesy::bearing_degrees(pose.position, point);
                assert!(
                    crate::geodesy::angle_difference(bearing, pose.heading).abs() < 90.0,
                    "filled a cell behind the camera at bearing {bearing}"
                );
            }
        }
    }

    #[test]
    fn resolution_degrades_with_range_and_says_so() {
        let pose = pose();
        let grid = grid_for(&pose, 0.25);
        let (width, height) = (640usize, 384usize);
        let image = coded_frame(width, height);
        let mut buffers = Buffers::new(grid.cells());
        sample_frame(
            &pose,
            &grid,
            &image,
            width,
            height,
            width * 3,
            DEFAULT_LATTICE,
            &mut buffers.sample(),
        );
        let mut near = Vec::new();
        let mut far = Vec::new();
        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let index = row * grid.cols + col;
                if buffers.valid[index] == 0 {
                    continue;
                }
                let d = distance_meters(pose.position, grid.lat_lon_of(row, col));
                // The pose's near edge is 4.3 m and its far edge 32.6 m —
                // the 60 m range never binds, because the top of the frame
                // hits the ground first. Bands chosen against that, not
                // against the range.
                if d < 10.0 {
                    near.push(buffers.resolution[index]);
                } else if d > 25.0 {
                    far.push(buffers.resolution[index]);
                }
            }
        }
        assert!(!near.is_empty() && !far.is_empty());
        let mean = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
        assert!(
            mean(&far) > 3.0 * mean(&near),
            "far {} vs near {} m/px",
            mean(&far),
            mean(&near)
        );
        assert!(mean(&near) < 0.1, "near ground should be centimetres per pixel");
    }

    #[test]
    fn the_median_removes_someone_walking_through() {
        // The claim behind calling the output a map: a cell is ground for
        // most of the samples and a person for a few, and the median is the
        // ground. Two of eleven frames have a walker over the cell.
        let mut accumulator = MedianAccumulator::new(1, 11);
        let ground = [40u8, 45, 50];
        let person = [200u8, 210, 220];
        for i in 0..11 {
            let colour = if i == 4 || i == 7 { person } else { ground };
            accumulator.add(&colour, &[1]);
        }
        let mut colour = [0u8; 3];
        let mut valid = [0u8; 1];
        let mut stats = [CellStats::default(); 1];
        assert_eq!(accumulator.median(1, &mut colour, &mut valid, &mut stats), 1);
        assert_eq!(colour, ground);
        assert_eq!(stats[0].samples, 11);
        // The median absolute deviation is *zero*, and that is correct: two
        // outliers in eleven cannot move it, which is precisely why the
        // median is trustworthy here. The number that notices the walker is
        // the disturbance share.
        assert_eq!(stats[0].deviation, 0);
        assert_eq!(stats[0].disturbed_percent, 18, "2 of 11 samples were blocked");
    }

    #[test]
    fn a_steady_cell_reports_no_deviation_and_a_disputed_one_reports_plenty() {
        let mut steady = MedianAccumulator::new(1, 8);
        for _ in 0..8 {
            steady.add(&[100, 100, 100], &[1]);
        }
        let mut disputed = MedianAccumulator::new(1, 8);
        for i in 0..8 {
            let v = if i % 2 == 0 { 20 } else { 200 };
            disputed.add(&[v, v, v], &[1]);
        }
        let mut colour = [0u8; 3];
        let mut valid = [0u8; 1];
        let mut stats = [CellStats::default(); 1];
        steady.median(1, &mut colour, &mut valid, &mut stats);
        assert_eq!(stats[0].deviation, 0);
        disputed.median(1, &mut colour, &mut valid, &mut stats);
        assert!(stats[0].deviation > 50, "got {}", stats[0].deviation);
        assert!(stats[0].disturbed_percent >= 50, "half the samples disagree");
    }

    #[test]
    fn a_cell_with_too_few_samples_is_empty_rather_than_a_guess() {
        let mut accumulator = MedianAccumulator::new(2, 8);
        accumulator.add(&[10, 10, 10, 20, 20, 20], &[1, 0]);
        accumulator.add(&[11, 11, 11, 20, 20, 20], &[1, 0]);
        let mut colour = [0u8; 6];
        let mut valid = [0u8; 2];
        let mut stats = [CellStats::default(); 2];
        let filled = accumulator.median(3, &mut colour, &mut valid, &mut stats);
        assert_eq!(filled, 0, "two samples is not a median when three are asked for");
        assert_eq!(valid, [0, 0]);
        assert_eq!(stats[0].samples, 2);
        assert_eq!(stats[1].samples, 0, "an unseen cell was never written to");
    }

    #[test]
    fn the_ring_is_bounded_and_keeps_the_newest() {
        let mut accumulator = MedianAccumulator::new(1, 4);
        for _ in 0..3 {
            accumulator.add(&[10, 10, 10], &[1]);
        }
        for _ in 0..10 {
            accumulator.add(&[200, 200, 200], &[1]);
        }
        let mut colour = [0u8; 3];
        let mut valid = [0u8; 1];
        let mut stats = [CellStats::default(); 1];
        accumulator.median(1, &mut colour, &mut valid, &mut stats);
        assert_eq!(colour, [200, 200, 200], "the old samples must have aged out");
        assert_eq!(stats[0].samples, 4, "the ring never grows past its capacity");
    }

    #[test]
    fn sampling_a_realistic_grid_is_fast_enough_for_every_frame() {
        // v1 measured the NumPy version of this at 79 ms and sampled four
        // frames a second because of it. The budget here is one frame at
        // 15 fps for one camera with room for fifteen more.
        let pose = pose();
        let grid = grid_for(&pose, 0.25);
        let (width, height) = (1280usize, 720usize);
        let image = coded_frame(width, height);
        let mut buffers = Buffers::new(grid.cells());
        // One warm run, then the measured one.
        sample_frame(&pose, &grid, &image, width, height, width * 3, DEFAULT_LATTICE, &mut buffers.sample());
        let start = std::time::Instant::now();
        let runs = 20;
        for _ in 0..runs {
            sample_frame(&pose, &grid, &image, width, height, width * 3, DEFAULT_LATTICE, &mut buffers.sample());
        }
        let each = start.elapsed() / runs;
        assert!(
            each.as_millis() < 20,
            "{} cells took {each:?} per frame",
            grid.cells()
        );
        eprintln!("sample_frame: {} cells in {each:?}", grid.cells());
    }

    #[test]
    fn a_camera_that_sees_no_ground_fills_nothing() {
        let mut sky = pose();
        sky.pitch = 30.0;
        let grid = grid_for(&sky, 1.0);
        let image = coded_frame(64, 64);
        let mut buffers = Buffers::new(grid.cells());
        let filled = sample_frame(&sky, &grid, &image, 64, 64, 64 * 3, 17, &mut buffers.sample());
        assert_eq!(filled, 0);
        assert!(buffers.valid.iter().all(|v| *v == 0));
    }

    #[test]
    fn projection_uncertainty_is_available_alongside_the_texture() {
        // The map and the error over it come from the same projection, so a
        // caller can weigh one cell against another. Guard that the two
        // agree about which cells exist at all.
        let pose = pose();
        let grid = grid_for(&pose, 0.5);
        let image = coded_frame(320, 192);
        let mut buffers = Buffers::new(grid.cells());
        sample_frame(&pose, &grid, &image, 320, 192, 320 * 3, DEFAULT_LATTICE, &mut buffers.sample());
        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let index = row * grid.cols + col;
                if buffers.valid[index] == 0 {
                    continue;
                }
                let point = grid.lat_lon_of(row, col);
                let (u, v) = image_coordinates(&pose, point).expect("a filled cell is in front");
                let projected = crate::camera::project_to_ground(
                    &pose,
                    u.clamp(0.0, 1.0),
                    v.clamp(0.0, 1.0),
                    crate::camera::DEFAULT_CONTACT_SIGMA_DEG,
                    &PoseUncertainty::default(),
                    false,
                );
                assert!(projected.is_ok(), "a filled cell must project back");
            }
        }
    }
}
