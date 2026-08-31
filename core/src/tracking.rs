//! Single-camera multi-object tracking.
//!
//! Associates detections to persistent tracks by overlap with motion
//! compensation. Deliberately not a Kalman filter: at the 5–15 fps this platform
//! runs inference, a constant-velocity predictor matches nearly as well, has no
//! tuning matrices to get wrong, and stays fully deterministic — which is what
//! lets the same recorded video produce identical tracks in a test and in the
//! field.
//!
//! The property that matters operationally is **gap tolerance**. A person walking
//! behind a pillar must come out the other side as the same track. A tracker that
//! splits identity there turns one incident into three and destroys cross-camera
//! correlation before it starts.
//!
//! The second property, learned the hard way, is that **overlap alone is not
//! enough**. A person 25 m from a wide-angle camera occupies a box under two
//! percent of the frame width and crosses half a box width between inference
//! frames, so consecutive detections of the same person overlap by barely a
//! third — and detector jitter regularly pushes that under any sane threshold. A
//! pure-IoU tracker shatters one person into a dozen tracks. So a detection whose
//! centre lands within a size-scaled gate of the *predicted* position associates
//! too, ranked below any real overlap.

use crate::geometry::{BoundingBox, CameraPose, LatLon, PositionEstimate, PositionSource, Vec2};
use crate::geometry::{bearing_degrees, haversine_distance, project_detection};

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f64,
    /// Index into the model's class list. Class identity is opaque here.
    pub class_id: u32,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub class_id: u32,
    pub first_seen_millis: i64,
    pub last_seen_millis: i64,
    /// Last frame in which a detection actually matched, as opposed to coasting.
    pub last_detected_millis: i64,
    pub bbox: BoundingBox,
    /// Normalised image units per millisecond.
    pub velocity: Vec2,
    pub confidence: f64,
    pub hits: u32,
    pub confirmed: bool,
    pub position: Option<PositionEstimate>,
    /// Recent ground positions, for deriving speed and heading over the map.
    ground_history: Vec<(i64, LatLon)>,
    pub speed_mps: Option<f64>,
    pub heading_degrees: Option<f64>,
}

/// How much tighter the vertical half of the proximity gate is than the
/// horizontal. An upright object is about a third as wide as it is tall, and
/// vertical image motion is depth rather than lateral movement.
const VERTICAL_GATE_RATIO: f64 = 0.35;

/// The most the gate may grow for a track that has gone unobserved. Unbounded
/// growth would let a track that vanished thirty seconds ago claim anything that
/// appears anywhere.
const MAX_GATE_WIDENING: f64 = 3.0;

#[derive(Debug, Clone, Copy)]
pub struct TrackerConfig {
    /// Minimum overlap for a detection to be considered the same object.
    pub iou_threshold: f64,
    /// Fallback gate for small, fast boxes, as a multiple of object size.
    pub gate_factor: f64,
    /// How long a track coasts without detections before it is closed.
    pub max_gap_millis: i64,
    /// Consecutive detections required before a track is reported at all.
    pub min_hits_to_confirm: u32,
    /// Window over which ground speed and heading are averaged.
    pub motion_window_millis: i64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.2,
            gate_factor: 2.5,
            max_gap_millis: 2000,
            min_hits_to_confirm: 2,
            motion_window_millis: 3000,
        }
    }
}

pub struct Tracker {
    config: TrackerConfig,
    pose: Option<CameraPose>,
    tracks: Vec<Track>,
    next_id: u64,
    last_update_millis: Option<i64>,
}

/// What one frame produced.
pub struct TrackerUpdate {
    /// Track ids closed on this frame, so downstream state can be released.
    pub ended: Vec<u64>,
}

impl Tracker {
    pub fn new(config: TrackerConfig, pose: Option<CameraPose>) -> Self {
        Self { config, pose, tracks: Vec::new(), next_id: 1, last_update_millis: None }
    }

    pub fn set_pose(&mut self, pose: Option<CameraPose>) {
        self.pose = pose;
    }

    pub fn active_count(&self) -> usize {
        self.tracks.len()
    }

    /// Confirmed, currently-live tracks.
    pub fn tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.iter().filter(|t| t.confirmed)
    }

    /// Predict where a track's box will be, from its last box and velocity.
    ///
    /// This is what lets a track survive a detection gap: it keeps moving through
    /// the occlusion so the reacquired detection still lands near it.
    fn predict(track: &Track, at_millis: i64) -> BoundingBox {
        let dt = (at_millis - track.last_seen_millis) as f64;
        if dt <= 0.0 {
            return track.bbox;
        }
        BoundingBox {
            x: track.bbox.x + track.velocity.x * dt,
            y: track.bbox.y + track.velocity.y * dt,
            w: track.bbox.w,
            h: track.bbox.h,
        }
    }

    /// How well a detection matches a predicted box, or `None` if it cannot.
    ///
    /// Two tiers so the ranking is unambiguous: a real overlap scores above 1 and
    /// always wins; a proximity match scores below 1.
    ///
    /// The proximity gate is an **ellipse, not a circle**, and that asymmetry is
    /// the whole point. A camera looking at the ground maps horizontal image
    /// motion to lateral movement and vertical image motion to *depth*, and those
    /// are not interchangeable. An upright object is roughly three times taller
    /// than it is wide, so a circular gate scaled by the larger dimension permits
    /// a vertical jump of several body-heights — which in world terms is a leap
    /// of tens of metres directly toward or away from the camera, in one frame.
    /// That is how a track hands its identity to a different object that has just
    /// appeared nearby.
    ///
    /// So the gate is generous across the frame and tight up and down. Measured
    /// on the reference scene, this is the difference between a track that
    /// abandons the person it was following to grab someone who walked into shot
    /// 100 pixels above it, and one that does not.
    ///
    /// `elapsed_ratio` widens the gate for a track that has not been seen for
    /// several frames: something unobserved for two seconds really could be
    /// further away than something unobserved for one frame, and a gate that
    /// ignores time is simultaneously too loose frame-to-frame and too tight
    /// after an occlusion.
    fn association_score(
        &self,
        predicted: &BoundingBox,
        detected: &BoundingBox,
        elapsed_ratio: f64,
    ) -> Option<f64> {
        let overlap = predicted.iou(detected);
        if overlap >= self.config.iou_threshold {
            return Some(1.0 + overlap);
        }

        let widen = elapsed_ratio.clamp(1.0, MAX_GATE_WIDENING);
        let gate_x = self.config.gate_factor * predicted.w.max(detected.w) * widen;
        let gate_y = self.config.gate_factor
            * predicted.h.max(detected.h)
            * VERTICAL_GATE_RATIO
            * widen;

        if gate_x <= 0.0 || gate_y <= 0.0 {
            return None;
        }

        let a = predicted.center();
        let b = detected.center();
        let normalized = ((b.x - a.x) / gate_x).powi(2) + ((b.y - a.y) / gate_y).powi(2);

        if normalized > 1.0 { None } else { Some(1.0 - normalized.sqrt()) }
    }

    /// Feed one frame's detections.
    ///
    /// Association is greedy by descending score, which for the handful of objects
    /// one camera sees at a time is both effectively optimal and stable: the same
    /// input always yields the same assignment, with no dependence on ordering.
    pub fn update(&mut self, detections: &[Detection], at_millis: i64) -> TrackerUpdate {
        let predicted: Vec<BoundingBox> =
            self.tracks.iter().map(|t| Self::predict(t, at_millis)).collect();

        // The observed frame interval, used to judge how stale a track is. Taken
        // from the stream rather than configured, because a camera's real rate is
        // rarely the rate it advertises.
        let interval = match self.last_update_millis {
            Some(previous) if at_millis > previous => (at_millis - previous) as f64,
            _ => 0.0,
        };
        self.last_update_millis = Some(at_millis);

        let mut candidates: Vec<(usize, usize, f64)> = Vec::new();

        for (track_index, track) in self.tracks.iter().enumerate() {
            let box_predicted = predicted[track_index];

            for (detection_index, detection) in detections.iter().enumerate() {
                // A "person" track must never silently absorb a "vehicle".
                if detection.class_id != track.class_id {
                    continue;
                }
                let gap = (at_millis - track.last_detected_millis).max(0) as f64;
                let elapsed_ratio = if interval > 0.0 { gap / interval } else { 1.0 };

                if let Some(score) =
                    self.association_score(&box_predicted, &detection.bbox, elapsed_ratio)
                {
                    candidates.push((track_index, detection_index, score));
                }
            }
        }

        // Ties break on track id so the result never depends on iteration order.
        candidates.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.tracks[a.0].id.cmp(&self.tracks[b.0].id))
        });

        let mut claimed_tracks = vec![false; self.tracks.len()];
        let mut claimed_detections = vec![false; detections.len()];

        for (track_index, detection_index, _) in candidates {
            if claimed_tracks[track_index] || claimed_detections[detection_index] {
                continue;
            }
            claimed_tracks[track_index] = true;
            claimed_detections[detection_index] = true;

            let pose = self.pose;
            let config = self.config;
            let track = &mut self.tracks[track_index];
            apply_detection(track, &detections[detection_index], at_millis, pose, &config);
        }

        // Unmatched detections start new tracks.
        for (index, detection) in detections.iter().enumerate() {
            if claimed_detections[index] {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;

            self.tracks.push(Track {
                id,
                class_id: detection.class_id,
                first_seen_millis: at_millis,
                last_seen_millis: at_millis,
                last_detected_millis: at_millis,
                bbox: detection.bbox,
                velocity: Vec2 { x: 0.0, y: 0.0 },
                confidence: detection.confidence,
                hits: 1,
                confirmed: self.config.min_hits_to_confirm <= 1,
                position: self.pose.map(|p| project_detection(&p, &detection.bbox)),
                ground_history: Vec::new(),
                speed_mps: None,
                heading_degrees: None,
            });
        }

        // Unmatched tracks coast, then expire.
        let mut ended = Vec::new();
        let pose = self.pose;
        let config = self.config;

        for (index, track) in self.tracks.iter_mut().enumerate() {
            if claimed_tracks.get(index).copied().unwrap_or(false) {
                continue;
            }
            if at_millis - track.last_detected_millis > config.max_gap_millis {
                ended.push(track.id);
                continue;
            }
            if track.confirmed {
                // Coast along the velocity so the track keeps a plausible position
                // through the occlusion.
                let coasted = Self::predict(track, at_millis);
                track.bbox = clamp_box(coasted);
                track.last_seen_millis = at_millis;
                track.position = pose.map(|p| project_detection(&p, &track.bbox));
                record_ground(track, at_millis, &config);
            }
        }

        self.tracks.retain(|t| !ended.contains(&t.id));
        TrackerUpdate { ended }
    }

    /// Close every live track, e.g. when a camera goes offline.
    pub fn reset(&mut self) -> Vec<u64> {
        let ended = self.tracks.iter().map(|t| t.id).collect();
        self.tracks.clear();
        ended
    }
}

fn apply_detection(
    track: &mut Track,
    detection: &Detection,
    at_millis: i64,
    pose: Option<CameraPose>,
    config: &TrackerConfig,
) {
    let dt = (at_millis - track.last_seen_millis) as f64;
    let previous = track.bbox.center();
    let next = detection.bbox.center();

    if dt > 0.0 {
        // Exponential smoothing: responsive enough to follow a turn, damped
        // enough that one noisy box does not fling the prediction away.
        let instant = Vec2 { x: (next.x - previous.x) / dt, y: (next.y - previous.y) / dt };
        track.velocity = Vec2 {
            x: track.velocity.x * 0.6 + instant.x * 0.4,
            y: track.velocity.y * 0.6 + instant.y * 0.4,
        };
    }

    track.bbox = detection.bbox;
    track.last_seen_millis = at_millis;
    track.last_detected_millis = at_millis;
    track.confidence = track.confidence * 0.7 + detection.confidence * 0.3;
    track.hits += 1;
    if track.hits >= config.min_hits_to_confirm {
        track.confirmed = true;
    }

    track.position = pose.map(|p| project_detection(&p, &detection.bbox));
    record_ground(track, at_millis, config);
}

/// Append a ground position and recompute motion over the window.
///
/// Camera-fallback positions are excluded: they are the mast's location, not the
/// object's, so feeding them in would compute the speed of a stationary pole and
/// report it as the target's.
fn record_ground(track: &mut Track, at_millis: i64, config: &TrackerConfig) {
    let Some(position) = track.position else { return };
    if position.source != PositionSource::GroundProjection {
        return;
    }

    track.ground_history.push((at_millis, position.point));

    let cutoff = at_millis - config.motion_window_millis;
    while track.ground_history.len() > 2 && track.ground_history[0].0 < cutoff {
        track.ground_history.remove(0);
    }

    let (Some(first), Some(last)) = (track.ground_history.first(), track.ground_history.last())
    else {
        return;
    };

    let seconds = (last.0 - first.0) as f64 / 1000.0;
    if seconds <= 0.0 {
        track.speed_mps = None;
        track.heading_degrees = None;
        return;
    }

    let distance = haversine_distance(first.1, last.1);

    // A heading derived from projection noise is worse than no heading, because
    // correlation would weigh it as evidence.
    if distance < position.radius_meters.max(1.0) {
        track.speed_mps = Some(0.0);
        track.heading_degrees = None;
        return;
    }

    track.speed_mps = Some(distance / seconds);
    track.heading_degrees = Some(bearing_degrees(first.1, last.1));
}

/// Keep a coasting box inside the frame so it cannot drift into nonsense.
fn clamp_box(b: BoundingBox) -> BoundingBox {
    BoundingBox {
        x: b.x.max(-b.w).min(1.0),
        y: b.y.max(-b.h).min(1.0),
        w: b.w,
        h: b.h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walking(x: f64) -> BoundingBox {
        BoundingBox { x, y: 0.5, w: 0.1, h: 0.3 }
    }

    fn detection(bbox: BoundingBox, class_id: u32) -> Detection {
        Detection { bbox, confidence: 0.9, class_id }
    }

    #[test]
    fn a_track_is_not_reported_until_confirmed() {
        let mut tracker = Tracker::new(TrackerConfig::default(), None);

        tracker.update(&[detection(walking(0.2), 0)], 0);
        assert_eq!(tracker.tracks().count(), 0, "one detection is not yet a track");

        tracker.update(&[detection(walking(0.22), 0)], 200);
        assert_eq!(tracker.tracks().count(), 1, "confirmed on the second hit");
    }

    #[test]
    fn a_walk_keeps_one_stable_identity() {
        let mut tracker = Tracker::new(TrackerConfig::default(), None);
        let mut ids = std::collections::HashSet::new();

        for step in 0..20 {
            let at = step * 200;
            tracker.update(&[detection(walking(0.05 + step as f64 * 0.03), 0)], at);
            for track in tracker.tracks() {
                ids.insert(track.id);
            }
        }

        assert_eq!(ids.len(), 1, "a single walk must not fragment");
    }

    #[test]
    fn small_fast_boxes_still_hold_identity() {
        // The failure that shattered three people into 31 tracks: a box under 2%
        // of frame width crossing half its own width between frames.
        let mut tracker = Tracker::new(TrackerConfig::default(), None);
        let mut ids = std::collections::HashSet::new();

        for step in 0..30 {
            let at = step * 200;
            let bbox = BoundingBox {
                x: 0.1 + step as f64 * 0.009,
                y: 0.5,
                w: 0.018,
                h: 0.15,
            };
            tracker.update(&[detection(bbox, 0)], at);
            for track in tracker.tracks() {
                ids.insert(track.id);
            }
        }

        assert_eq!(ids.len(), 1, "the proximity gate must hold small fast boxes together");
    }

    #[test]
    fn a_track_survives_an_occlusion() {
        let mut tracker = Tracker::new(TrackerConfig::default(), None);

        let mut id = 0;
        for step in 0..4 {
            let at = step * 200;
            tracker.update(&[detection(walking(0.1 + step as f64 * 0.05), 0)], at);
            if let Some(track) = tracker.tracks().next() {
                id = track.id;
            }
        }
        assert!(id > 0);

        // Occluded for three frames.
        for step in 4..7 {
            let update = tracker.update(&[], step * 200);
            assert!(update.ended.is_empty(), "must not close during a short gap");
        }

        tracker.update(&[detection(walking(0.1 + 7.0 * 0.05), 0)], 1400);
        assert_eq!(tracker.tracks().next().map(|t| t.id), Some(id), "same object, same id");
    }

    #[test]
    fn a_track_closes_once_the_gap_exceeds_the_limit() {
        let config = TrackerConfig { min_hits_to_confirm: 1, max_gap_millis: 1000, ..Default::default() };
        let mut tracker = Tracker::new(config, None);

        tracker.update(&[detection(walking(0.2), 0)], 0);
        assert!(tracker.update(&[], 900).ended.is_empty());

        let expired = tracker.update(&[], 1500);
        assert_eq!(expired.ended.len(), 1);
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn different_classes_never_merge() {
        let config = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };
        let mut tracker = Tracker::new(config, None);

        tracker.update(&[detection(walking(0.3), 0)], 0);
        tracker.update(&[detection(walking(0.3), 1)], 200);

        assert_eq!(tracker.active_count(), 2, "a person track must not absorb a vehicle");
    }

    #[test]
    fn two_objects_get_two_tracks() {
        let config = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };
        let mut tracker = Tracker::new(config, None);

        for step in 0..5 {
            let at = step * 200;
            tracker.update(
                &[
                    detection(BoundingBox { x: 0.1 + step as f64 * 0.02, y: 0.5, w: 0.08, h: 0.25 }, 0),
                    detection(BoundingBox { x: 0.7 - step as f64 * 0.02, y: 0.5, w: 0.08, h: 0.25 }, 0),
                ],
                at,
            );
        }

        assert_eq!(tracker.active_count(), 2);
    }

    #[test]
    fn tracking_is_deterministic() {
        let run = || {
            let mut tracker = Tracker::new(TrackerConfig::default(), None);
            let mut output = Vec::new();

            for step in 0..12 {
                let at = step * 200;
                tracker.update(
                    &[
                        detection(BoundingBox { x: 0.1 + step as f64 * 0.03, y: 0.5, w: 0.08, h: 0.25 }, 0),
                        detection(BoundingBox { x: 0.6 - step as f64 * 0.02, y: 0.4, w: 0.08, h: 0.25 }, 0),
                    ],
                    at,
                );
                for track in tracker.tracks() {
                    output.push(format!("{}@{:.4}", track.id, track.bbox.x));
                }
            }
            output
        };

        assert_eq!(run(), run(), "the same input must reproduce the same tracks");
    }

    #[test]
    fn a_stationary_object_reports_zero_speed_and_no_heading() {
        let pose = CameraPose {
            position: LatLon { lat: 33.8938, lon: 35.5018 },
            mount_height: 10.0,
            heading: 0.0,
            pitch: -45.0,
            roll: 0.0,
            horizontal_fov: 60.0,
            vertical_fov: 34.0,
            range_meters: 200.0,
        };
        let config = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };
        let mut tracker = Tracker::new(config, Some(pose));

        for step in 0..10 {
            tracker.update(
                &[detection(BoundingBox { x: 0.45, y: 0.6, w: 0.08, h: 0.15 }, 0)],
                step * 500,
            );
        }

        let track = tracker.tracks().next().expect("track exists");
        assert_eq!(track.speed_mps, Some(0.0), "standing still is zero, not noise");
        assert_eq!(track.heading_degrees, None, "a heading from jitter is worse than none");
    }

    #[test]
    fn no_pose_means_no_map_position() {
        let config = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };
        let mut tracker = Tracker::new(config, None);
        tracker.update(&[detection(walking(0.2), 0)], 0);

        let track = tracker.tracks().next().unwrap();
        assert!(track.position.is_none());
        assert!(track.speed_mps.is_none());
    }
}
