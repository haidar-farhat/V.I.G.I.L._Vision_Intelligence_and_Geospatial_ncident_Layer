//! Rectangular linear assignment: the Jonker–Volgenant shortest augmenting
//! path.
//!
//! # Why this is here at all
//!
//! v1's Rust tracker and v2's Python port both associated detections to
//! tracks *greedily*: sort every pair by score, walk the list, take a pair
//! whenever neither side is already claimed. Greedy is not optimal and the
//! way it fails is the way that matters — two people crossing. The pair that
//! happens to score highest locks in, and the second person is then matched
//! against whatever is left, which is frequently the first person's old
//! track. Two tracks swap identities, and every count, every loitering clock
//! and every zone judgement downstream is about the wrong person.
//!
//! Optimal assignment costs O(n^3) and fixes exactly that: it minimises the
//! *total* cost, so a locally attractive pair that forces an expensive
//! remainder is rejected. At the sizes a camera produces — tens of tracks —
//! O(n^3) is microseconds, and the reason it was not done in Python is that
//! SciPy is not a dependency of an offline appliance and a Python triple loop
//! at 15 fps across sixteen cameras is.
//!
//! # Infeasible pairs
//!
//! There is no such thing as "not allowed" in an assignment problem, so a
//! forbidden pair is given [`FORBIDDEN`] as its cost and the caller drops any
//! resulting pair at or above [`FORBIDDEN_THRESHOLD`]. That is exact rather
//! than approximate: with every real cost bounded by 1 and the matrix no
//! larger than a camera can produce, no sum of real costs can reach the
//! threshold, so the solver can never prefer a forbidden pair to a feasible
//! completion that exists.

/// Cost given to a pair that must not be matched.
pub const FORBIDDEN: f64 = 1.0e9;

/// Any assignment at or above this is a forbidden pair the solver was forced
/// into, and the caller drops it.
pub const FORBIDDEN_THRESHOLD: f64 = 1.0e8;

/// Solve `min sum(cost[i][assignment[i]])` over injective assignments.
///
/// `cost` is row-major, `rows` by `cols`. Returns one entry per row: the
/// column it takes, or `usize::MAX` for a row left unassigned (which happens
/// only when `cols < rows`).
///
/// Costs must be finite. A caller with an infeasible pair uses [`FORBIDDEN`];
/// NaN is treated as forbidden rather than being allowed to poison the
/// potentials, because a NaN that reaches the dual variables makes every
/// subsequent comparison false and the solver silently returns nonsense.
pub fn solve(cost: &[f64], rows: usize, cols: usize) -> Vec<usize> {
    if rows == 0 || cols == 0 {
        return vec![usize::MAX; rows];
    }
    debug_assert_eq!(cost.len(), rows * cols);
    if rows <= cols {
        let sane: Vec<f64> = cost.iter().map(|c| sanitise(*c)).collect();
        jv(&sane, rows, cols)
    } else {
        // The algorithm needs at least as many columns as rows, so transpose
        // and turn the answer back the right way up.
        let mut transposed = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = sanitise(cost[r * cols + c]);
            }
        }
        let by_column = jv(&transposed, cols, rows);
        let mut out = vec![usize::MAX; rows];
        for (col, row) in by_column.iter().enumerate() {
            if *row != usize::MAX {
                out[*row] = col;
            }
        }
        out
    }
}

fn sanitise(c: f64) -> f64 {
    if c.is_finite() {
        c
    } else {
        FORBIDDEN
    }
}

/// The e-maxx formulation of Jonker–Volgenant, 1-indexed internally with a
/// sentinel row and column. Requires `rows <= cols`.
fn jv(cost: &[f64], rows: usize, cols: usize) -> Vec<usize> {
    let at = |i: usize, j: usize| cost[(i - 1) * cols + (j - 1)];
    let inf = f64::INFINITY;

    // Dual potentials, and `p[j]` = the row currently assigned to column j.
    let mut u = vec![0.0f64; rows + 1];
    let mut v = vec![0.0f64; cols + 1];
    let mut p = vec![0usize; cols + 1];
    let mut way = vec![0usize; cols + 1];

    for i in 1..=rows {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; cols + 1];
        let mut used = vec![false; cols + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=cols {
                if used[j] {
                    continue;
                }
                let cur = at(i0, j) - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            // A finite matrix always yields a finite delta; guard anyway so a
            // caller's degenerate input cannot spin here forever.
            if !delta.is_finite() {
                break;
            }
            for j in 0..=cols {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Walk the augmenting path back, flipping the assignment along it.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut out = vec![usize::MAX; rows];
    for j in 1..=cols {
        if p[j] != 0 {
            out[p[j] - 1] = j - 1;
        }
    }
    out
}

/// Total cost of an assignment, skipping unassigned rows. For tests and for
/// the caller that wants to know how good the matching it got actually was.
pub fn total_cost(cost: &[f64], cols: usize, assignment: &[usize]) -> f64 {
    assignment
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != usize::MAX)
        .map(|(r, c)| cost[r * cols + c])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force over every injective assignment, for checking optimality
    /// on matrices small enough to enumerate.
    fn brute(cost: &[f64], rows: usize, cols: usize) -> f64 {
        let mut best = f64::INFINITY;
        if rows <= cols {
            // Choose which column each row takes.
            let mut columns: Vec<usize> = (0..cols).collect();
            permute(&mut columns, 0, &mut |order: &[usize]| {
                let total: f64 = (0..rows).map(|r| cost[r * cols + order[r]]).sum();
                best = best.min(total);
            });
        } else {
            // Choose which rows get served; the rest go unassigned. Permuting
            // columns alone would force row 0 to always be one of them, which
            // is not the problem being solved.
            let mut order: Vec<usize> = (0..rows).collect();
            permute(&mut order, 0, &mut |rows_in_order: &[usize]| {
                let total: f64 = (0..cols)
                    .map(|c| cost[rows_in_order[c] * cols + c])
                    .sum();
                best = best.min(total);
            });
        }
        best
    }

    fn permute(items: &mut Vec<usize>, k: usize, visit: &mut impl FnMut(&[usize])) {
        if k == items.len() {
            visit(items);
            return;
        }
        for i in k..items.len() {
            items.swap(k, i);
            permute(items, k + 1, visit);
            items.swap(k, i);
        }
    }

    // A tiny deterministic generator; the crate has no dependencies and a
    // seeded LCG is enough to hit a few thousand random matrices.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn it_matches_brute_force_on_square_and_rectangular_matrices() {
        let mut rng = Lcg(0x5eed);
        for rows in 1..=5usize {
            for cols in 1..=5usize {
                for _ in 0..40 {
                    let cost: Vec<f64> = (0..rows * cols).map(|_| rng.next()).collect();
                    let assignment = solve(&cost, rows, cols);
                    let got = total_cost(&cost, cols, &assignment);
                    let want = brute(&cost, rows, cols);
                    assert!(
                        (got - want).abs() < 1e-9,
                        "{rows}x{cols}: got {got}, optimal {want}"
                    );
                    // Injective, and every row assigned when there is room.
                    let mut seen = vec![false; cols];
                    let mut count = 0;
                    for c in &assignment {
                        if *c != usize::MAX {
                            assert!(!seen[*c], "column used twice");
                            seen[*c] = true;
                            count += 1;
                        }
                    }
                    assert_eq!(count, rows.min(cols));
                }
            }
        }
    }

    #[test]
    fn greedy_swaps_two_crossing_tracks_and_this_does_not() {
        // Two tracks, two detections. Greedy takes the single best pair (0,0)
        // at 0.10 and is then forced into (1,1) at 0.90, total 1.00. The
        // optimal matching is the other diagonal, 0.20 + 0.25 = 0.45 — less
        // than half the cost. This is the crossing-pedestrian case in the
        // smallest form that shows it, and it is why the greedy pass that v1
        // and v2 shared swapped two people's identities when they met.
        let cost = vec![0.10, 0.20, 0.25, 0.90];
        let assignment = solve(&cost, 2, 2);
        assert_eq!(assignment, vec![1, 0]);
        assert!((total_cost(&cost, 2, &assignment) - 0.45).abs() < 1e-12);
    }

    #[test]
    fn forbidden_pairs_are_never_preferred_to_a_feasible_matching() {
        // Row 0 may only take column 1; row 1 may only take column 0.
        let cost = vec![FORBIDDEN, 0.9, 0.9, FORBIDDEN];
        let assignment = solve(&cost, 2, 2);
        assert_eq!(assignment, vec![1, 0]);
        // With no feasible completion the solver still returns a matching, and
        // the caller can see which pairs to throw away.
        let all_forbidden = vec![FORBIDDEN; 4];
        let forced = solve(&all_forbidden, 2, 2);
        for (r, c) in forced.iter().enumerate() {
            assert!(all_forbidden[r * 2 + *c] >= FORBIDDEN_THRESHOLD);
        }
    }

    #[test]
    fn more_rows_than_columns_leaves_the_dearest_rows_unassigned() {
        // Three tracks, one detection: exactly one match, and it is the
        // cheapest one.
        let cost = vec![0.7, 0.2, 0.9];
        let assignment = solve(&cost, 3, 1);
        assert_eq!(assignment, vec![usize::MAX, 0, usize::MAX]);
    }

    #[test]
    fn more_columns_than_rows_assigns_every_row() {
        let cost = vec![0.5, 0.1, 0.9, 0.4, 0.8, 0.2];
        let assignment = solve(&cost, 2, 3);
        assert_eq!(assignment, vec![1, 2]);
    }

    #[test]
    fn degenerate_shapes_and_non_finite_costs_do_not_break_it() {
        assert!(solve(&[], 0, 0).is_empty());
        assert_eq!(solve(&[], 2, 0), vec![usize::MAX, usize::MAX]);
        let cost = vec![f64::NAN, 0.5, 0.4, f64::INFINITY];
        let assignment = solve(&cost, 2, 2);
        assert_eq!(assignment, vec![1, 0], "NaN must behave as forbidden");
    }

    #[test]
    fn it_stays_fast_at_the_sizes_a_camera_produces() {
        // Not a benchmark, a ceiling: 64 tracks against 64 detections is far
        // past what one camera yields, and it must not be milliseconds.
        let mut rng = Lcg(7);
        let n = 64;
        let cost: Vec<f64> = (0..n * n).map(|_| rng.next()).collect();
        let start = std::time::Instant::now();
        let assignment = solve(&cost, n, n);
        let elapsed = start.elapsed();
        assert_eq!(assignment.iter().filter(|c| **c != usize::MAX).count(), n);
        assert!(elapsed.as_millis() < 50, "64x64 took {elapsed:?}");
    }
}
