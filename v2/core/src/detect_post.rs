//! Suppressing overlapping detections, which is a quarter of what detection costs.
//!
//! # Why this is here and the assignment solver's cost matrix is not
//!
//! Measured on this machine before anything was written, because a kernel
//! goes into Rust only with a number behind it:
//!
//! | | per frame |
//! |---|---|
//! | NMS, 300 proposals over 8 classes | 3.14 ms |
//! | **soft-NMS, same** | **4.92 ms** |
//! | mask decode, 12 detections | 0.91 ms |
//! | rectangular assignment, 30x30 | 0.005 ms |
//!
//! Detection is about 12.5 ms a frame on this machine's GPU, so suppression
//! alone was a quarter to a third of it — and tiling runs it once per tile,
//! so a five-pass frame spent nearly 25 ms in Python deciding which boxes to
//! throw away. The assignment cost matrix, which the plan also listed, turned
//! out to be five microseconds; it stays in Python, because rewriting it
//! would buy nothing and cost a second implementation to keep in step.
//!
//! # Ordering, and why it is defined rather than left to the sort
//!
//! Both implementations order by **descending score, then ascending index**.
//! NumPy's `argsort` is not stable by default, so two proposals with equal
//! scores could come out in either order, and the one kept would differ
//! between the two implementations for reasons neither could be blamed for.
//! Equal scores are common: a quantised model emits them constantly.
//!
//! # Soft-NMS
//!
//! Hard NMS assumes two boxes overlapping past a threshold are two proposals
//! for one object. In a queue of people or a row of parked cars that is
//! false, and the second real object is deleted leaving no trace anywhere —
//! which is the more expensive mistake, because nobody can see it. Gaussian
//! soft-NMS multiplies the score by `exp(-iou^2 / sigma)` instead: a genuine
//! duplicate at 0.9 overlap keeps 6% of its score and falls under the floor,
//! while a real neighbour at 0.55 keeps 45% and survives with an honestly
//! lower score that the tracker's confirmation can then take or leave.

/// `[x1, y1, x2, y2]` per box, plus a score and a class.
pub struct Boxes<'a> {
    pub xyxy: &'a [f64],
    pub scores: &'a [f64],
    pub classes: &'a [i64],
}

impl<'a> Boxes<'a> {
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    fn area(&self, i: usize) -> f64 {
        let b = &self.xyxy[i * 4..i * 4 + 4];
        (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
    }

    fn iou(&self, i: usize, j: usize, area_i: f64, area_j: f64) -> f64 {
        let a = &self.xyxy[i * 4..i * 4 + 4];
        let b = &self.xyxy[j * 4..j * 4 + 4];
        let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let inter = w * h;
        inter / (area_i + area_j - inter + 1e-9)
    }
}

/// Descending score, ascending index. See the module note on ties.
fn ordered(indices: &mut [usize], scores: &[f64]) {
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
}

/// Hard suppression within one class. Indices are into `boxes`.
pub fn nms(boxes: &Boxes, members: &[usize], iou_threshold: f64) -> Vec<usize> {
    let mut order: Vec<usize> = members.to_vec();
    ordered(&mut order, boxes.scores);
    let areas: Vec<f64> = members.iter().map(|&i| boxes.area(i)).collect();
    let area_of = |i: usize| areas[members.iter().position(|&m| m == i).unwrap_or(0)];

    let mut keep = Vec::new();
    let mut alive: Vec<bool> = vec![true; order.len()];
    for pos in 0..order.len() {
        if !alive[pos] {
            continue;
        }
        let i = order[pos];
        keep.push(i);
        let area_i = area_of(i);
        for (other, &j) in order.iter().enumerate().skip(pos + 1) {
            if !alive[other] {
                continue;
            }
            if boxes.iou(i, j, area_i, area_of(j)) > iou_threshold {
                alive[other] = false;
            }
        }
    }
    keep
}

/// Gaussian soft suppression within one class.
pub fn soft_nms(
    boxes: &Boxes,
    members: &[usize],
    iou_threshold: f64,
    sigma: f64,
    floor: f64,
) -> Vec<usize> {
    let mut order: Vec<usize> = members.to_vec();
    ordered(&mut order, boxes.scores);
    let mut working: Vec<f64> = order.iter().map(|&i| boxes.scores[i]).collect();
    let areas: Vec<f64> = order.iter().map(|&i| boxes.area(i)).collect();
    let mut taken = vec![false; order.len()];
    let mut keep = Vec::new();

    loop {
        // The highest remaining score. Not a re-sort: the decay only ever
        // lowers scores, so a linear scan is both correct and cheaper than
        // sorting again for every kept box.
        let mut best: Option<usize> = None;
        for pos in 0..order.len() {
            if taken[pos] || working[pos] < floor {
                continue;
            }
            if best.is_none() || working[pos] > working[best.unwrap()] {
                best = Some(pos);
            } else if working[pos] == working[best.unwrap()] && order[pos] < order[best.unwrap()] {
                best = Some(pos);
            }
        }
        let Some(pos) = best else { break };
        taken[pos] = true;
        keep.push(order[pos]);
        for other in 0..order.len() {
            if taken[other] || working[other] < floor {
                continue;
            }
            let overlap = boxes.iou(order[pos], order[other], areas[pos], areas[other]);
            if overlap > iou_threshold {
                working[other] *= (-(overlap * overlap) / sigma).exp();
            }
        }
    }
    keep
}

/// Suppress within each class, never across them, and return in score order.
///
/// Class-aware because one pass over every box in a frame lets a person
/// standing in front of a car delete the car: a 70% overlap between two
/// *different* things is not a duplicate detection, it is a car park.
pub fn suppress_per_class(
    boxes: &Boxes,
    iou_threshold: f64,
    soft: bool,
    sigma: f64,
    floor: f64,
) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let mut classes: Vec<i64> = boxes.classes.to_vec();
    classes.sort_unstable();
    classes.dedup();

    let mut keep = Vec::new();
    for class in classes {
        let members: Vec<usize> = (0..boxes.len())
            .filter(|&i| boxes.classes[i] == class)
            .collect();
        let kept = if soft {
            soft_nms(boxes, &members, iou_threshold, sigma, floor)
        } else {
            nms(boxes, &members, iou_threshold)
        };
        keep.extend(kept);
    }
    // Back into confidence order, which is what a caller reading the first
    // few detections expects.
    ordered(&mut keep, boxes.scores);
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxes<'a>(xyxy: &'a [f64], scores: &'a [f64], classes: &'a [i64]) -> Boxes<'a> {
        Boxes {
            xyxy,
            scores,
            classes,
        }
    }

    #[test]
    fn a_duplicate_is_removed_and_a_neighbour_is_not() {
        let xyxy = [
            0.10, 0.50, 0.30, 0.66, // a car
            0.105, 0.505, 0.305, 0.665, // the same car again
            0.55, 0.50, 0.75, 0.66, // a different car, no overlap
        ];
        let scores = [0.90, 0.72, 0.66];
        let classes = [1, 1, 1];
        let kept = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, false, 0.5, 0.2);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn a_person_in_front_of_a_car_does_not_delete_the_car() {
        let xyxy = [0.10, 0.40, 0.40, 0.90, 0.15, 0.42, 0.38, 0.88];
        let scores = [0.88, 0.70];
        let classes = [0, 1];
        let kept = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, false, 0.5, 0.2);
        assert_eq!(kept.len(), 2, "class-agnostic suppression deleted the car");
    }

    #[test]
    fn soft_suppression_keeps_the_second_car_in_a_row() {
        // Overlap about 0.55: two parked cars, not two proposals for one.
        let xyxy = [0.10, 0.50, 0.30, 0.66, 0.163, 0.50, 0.363, 0.66];
        let scores = [0.86, 0.71];
        let classes = [1, 1];
        let hard = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, false, 0.5, 0.2);
        let soft = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, true, 0.5, 0.2);
        assert_eq!(hard.len(), 1, "this test needs hard NMS to delete one");
        assert_eq!(soft.len(), 2);
    }

    #[test]
    fn soft_suppression_still_removes_a_true_duplicate() {
        let xyxy = [0.10, 0.50, 0.30, 0.66, 0.105, 0.505, 0.305, 0.665];
        let scores = [0.90, 0.34];
        let classes = [1, 1];
        let kept = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, true, 0.5, 0.2);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn equal_scores_break_towards_the_lower_index_every_time() {
        // Two identical boxes with identical scores. Which survives must not
        // depend on the sort's mood, or the Rust and the NumPy disagree for a
        // reason neither can be blamed for.
        let xyxy = [0.10, 0.50, 0.30, 0.66, 0.10, 0.50, 0.30, 0.66];
        let scores = [0.75, 0.75];
        let classes = [1, 1];
        for _ in 0..20 {
            let kept = suppress_per_class(&boxes(&xyxy, &scores, &classes), 0.45, false, 0.5, 0.2);
            assert_eq!(kept, vec![0]);
        }
    }

    #[test]
    fn nothing_in_gives_nothing_out() {
        assert!(suppress_per_class(&boxes(&[], &[], &[]), 0.45, false, 0.5, 0.2).is_empty());
    }

    #[test]
    fn it_is_fast_enough_that_tiling_can_afford_five_of_them() {
        // The measurement this module exists for: NumPy took 3.1 ms hard and
        // 4.9 ms soft on 300 proposals, and tiling runs it once per tile.
        let count = 300usize;
        let mut xyxy = Vec::with_capacity(count * 4);
        let mut scores = Vec::with_capacity(count);
        let mut classes = Vec::with_capacity(count);
        let mut seed = 12345u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for i in 0..count {
            let x = next() * 0.9;
            let y = next() * 0.9;
            xyxy.extend_from_slice(&[x, y, x + 0.02 + next() * 0.08, y + 0.02 + next() * 0.08]);
            scores.push(0.25 + next() * 0.74);
            classes.push((i % 8) as i64);
        }
        let set = boxes(&xyxy, &scores, &classes);
        let started = std::time::Instant::now();
        let mut kept = 0usize;
        for _ in 0..20 {
            kept += suppress_per_class(&set, 0.45, true, 0.5, 0.2).len();
        }
        let each = started.elapsed().as_secs_f64() * 1000.0 / 20.0;
        assert!(kept > 0);
        assert!(
            each < 1.0,
            "soft suppression of {count} proposals took {each:.3} ms; NumPy did it in 4.9 and \
             the whole point was to be well under that"
        );
    }
}
