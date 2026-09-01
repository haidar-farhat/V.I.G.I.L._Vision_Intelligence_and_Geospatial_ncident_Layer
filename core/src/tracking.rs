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

use crate::geometry::{bearing_degrees, haversine_distance, project_detection};
use crate::geometry::{BoundingBox, CameraPose, LatLon, PositionEstimate, PositionSource, Vec2};

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

/// How much the vertical half of the proximity gate is tightened, relative to
/// the object's own height.
///
/// A measurement rather than a principle. On the reference scene 0.35 gives 5
/// tracks for three people, 0.25 gives 6, 0.20 gives 7 and 0.15 gives 8.
/// Tightening does refuse more identity hand-offs, but it refuses legitimate
/// depth motion faster than it helps.
const VERTICAL_GATE_RATIO: f64 = 0.35;

/// The coarsest speed resolution at which reporting "stationary" is still
/// honest.
///
/// Below a slow walk. When the position is so uncertain that the smallest
/// detectable movement would exceed this, "not moving" and "walking" are
/// indistinguishable, and the tracker says it does not know rather than
/// choosing one.
const STATIONARY_RESOLUTION_MPS: f64 = 0.5;

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
    /// Detections required before a track is reported at all.
    ///
    /// Cumulative, not consecutive, and that is deliberate — an earlier comment
    /// here said "consecutive" and the code never enforced it. Making the code
    /// match the comment was tried and measured: on the reference scene it took
    /// the object count from 5 to 8 for three people, because requiring an
    /// unbroken run is brittle exactly when the detector is unreliable, which at
    /// 0.69 recall is the normal case. A track seen twice has been seen twice,
    /// whether or not there was a miss in between.
    pub min_hits_to_confirm: u32,
    /// Window over which ground speed and heading are averaged.
    pub motion_window_millis: i64,
    /// The shortest span of observation from which a speed may be reported.
    ///
    /// Below this, speed is `None` rather than a number. Dividing a distance by
    /// a very short interval amplifies position error by the reciprocal of that
    /// interval: with a position known to ±1.3 m, two frames 200 ms apart give a
    /// velocity uncertainty of about ±9 m/s, so a reported 53 m/s is a
    /// measurement of the projection rather than of the object. Over two seconds
    /// the same error contributes under 1 m/s.
    pub min_motion_span_millis: i64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.2,
            gate_factor: 2.5,
            max_gap_millis: 2000,
            min_hits_to_confirm: 2,
            motion_window_millis: 3000,
            min_motion_span_millis: 1200,
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
        Self {
            config,
            pose,
            tracks: Vec::new(),
            next_id: 1,
            last_update_millis: None,
        }
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

        // Each axis is scaled by the object's extent ON THAT AXIS, and the
        // vertical one is then tightened. **The per-axis scaling is what does
        // the work**, not the ratio. What this replaced used
        // `gate_factor * max(w, h)` as a single radius — for an upright box
        // that is its height — so a track could leap most of a body-height in
        // depth and pick up whoever had just walked into shot. Measured on the
        // case that prompted it (a 25x71 px box, an 82 px vertical jump): the
        // single-radius gate scored 0.17 against a 0.37 limit and accepted; this
        // one scores 1.82 against 1.0 and refuses.
        //
        // The resulting gate is close to circular in pixels, because an upright
        // object is about three times taller than it is wide and 3 x 0.35 is
        // roughly 1. That is an honest description of the arithmetic, and an
        // earlier comment here — "generous across the frame and tight up and
        // down" — was not.
        //
        // Scaling the vertical half by the width instead, to make that stated
        // asymmetry real, was tried and measured. It still refuses the leap, but
        // it takes the reference scene from 5 tracks to 8 for three people: a
        // person walking toward the camera genuinely moves down the frame, and
        // the gate has to allow it.
        let gate_x = self.config.gate_factor * predicted.w.max(detected.w) * widen;
        let gate_y =
            self.config.gate_factor * predicted.h.max(detected.h) * VERTICAL_GATE_RATIO * widen;

        if gate_x <= 0.0 || gate_y <= 0.0 {
            return None;
        }

        let a = predicted.center();
        let b = detected.center();
        let normalized = ((b.x - a.x) / gate_x).powi(2) + ((b.y - a.y) / gate_y).powi(2);

        if normalized > 1.0 {
            None
        } else {
            Some(1.0 - normalized.sqrt())
        }
    }

    /// Feed one frame's detections.
    ///
    /// Association is greedy by descending score, which for the handful of objects
    /// one camera sees at a time is both effectively optimal and stable: the same
    /// input always yields the same assignment, with no dependence on ordering.
    pub fn update(&mut self, detections: &[Detection], at_millis: i64) -> TrackerUpdate {
        let predicted: Vec<BoundingBox> = self
            .tracks
            .iter()
            .map(|t| Self::predict(t, at_millis))
            .collect();

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
            apply_detection(
                track,
                &detections[detection_index],
                at_millis,
                pose,
                &config,
            );
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
            // `unwrap_or(true)`, not false: tracks created from this frame's
            // unmatched detections were pushed above and have no entry in
            // `claimed_tracks`, but they were matched by definition — a
            // detection is what created them. Treating them as missed zeroes
            // the hit count on the very frame the track is born, so nothing
            // ever reaches min_hits_to_confirm.
            if claimed_tracks.get(index).copied().unwrap_or(true) {
                continue;
            }
            if at_millis - track.last_detected_millis > config.max_gap_millis {
                ended.push(track.id);
                continue;
            }
            if track.confirmed {
                // Coast along the velocity so the track keeps a plausible
                // position through the occlusion.
                let coasted = Self::predict(track, at_millis);
                track.bbox = clamp_box(coasted);
                track.last_seen_millis = at_millis;
                track.position = pose.map(|p| project_detection(&p, &track.bbox));

                // Deliberately NOT recorded into ground_history. A coasted box
                // is extrapolation, and clamp_box pins it to the frame edge once
                // the object has left the scene — so feeding it in would let an
                // object that walked out of shot be reported as standing at a
                // confident map position, indefinitely. The position is still
                // published, because the tracker does believe the object is
                // near there; what it must not do is treat its own guess as a
                // new measurement of speed.
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
        let instant = Vec2 {
            x: (next.x - previous.x) / dt,
            y: (next.y - previous.y) / dt,
        };
        track.velocity = Vec2 {
            x: track.velocity.x * 0.6 + instant.x * 0.4,
            y: track.velocity.y * 0.6 + instant.y * 0.4,
        };
    }

    track.bbox = detection.bbox;
    track.last_seen_millis = at_millis;
    track.last_detected_millis = at_millis;
    track.confidence = track.confidence * 0.7 + detection.confidence * 0.3;
    // Cumulative. See TrackerConfig::min_hits_to_confirm for why a miss does not
    // reset this.
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
/// Forget a motion estimate that no longer has evidence behind it.
///
/// `None` means "not known", which is the only honest answer once the
/// observations that produced a speed have stopped arriving.
fn forget_motion(track: &mut Track) {
    track.speed_mps = None;
    track.heading_degrees = None;
}

fn record_ground(track: &mut Track, at_millis: i64, config: &TrackerConfig) {
    // Both of these were bare `return`s, and that was a real defect: the moment
    // a track's position stopped being a ground projection — the object walked
    // beyond `range_meters`, its ground contact rose toward the horizon, or the
    // operator un-placed the camera — the last speed and heading computed
    // minutes earlier were frozen in place and re-emitted every frame for the
    // life of the track. An evidence package could then read "position NOT
    // DETERMINED" directly above "motion 4.38 m/s, heading 0 degrees".
    let Some(position) = track.position else {
        forget_motion(track);
        track.ground_history.clear();
        return;
    };
    if position.source != PositionSource::GroundProjection {
        forget_motion(track);
        // The history goes too, so samples from before the blackout cannot
        // later be paired with samples from after it and averaged into a speed
        // that nothing observed.
        track.ground_history.clear();
        return;
    }

    track.ground_history.push((at_millis, position.point));

    // Prune to the window, keeping at most one sample older than the cutoff so
    // there is still a pair to measure across. An earlier `len() > 2` floor
    // stopped pruning while two stale samples remained, so after a long gap the
    // speed could be averaged over a span far longer than motion_window_millis
    // — reporting a minutes-old average as if it were current.
    let cutoff = at_millis - config.motion_window_millis;
    while track.ground_history.len() > 2 && track.ground_history[1].0 < cutoff {
        track.ground_history.remove(0);
    }

    let (Some(first), Some(last)) = (track.ground_history.first(), track.ground_history.last())
    else {
        return;
    };

    let span_millis = last.0 - first.0;
    if span_millis < config.min_motion_span_millis {
        // Not yet measurable. `None` means "we do not know", which is a
        // different statement from "it is not moving", and the two must not be
        // confused: a rule that treats an unknown speed as zero misses a
        // sprinting intruder, and one that treats it as a number invents one.
        track.speed_mps = None;
        track.heading_degrees = None;
        return;
    }

    let seconds = span_millis as f64 / 1000.0;
    if seconds <= 0.0 {
        track.speed_mps = None;
        track.heading_degrees = None;
        return;
    }

    let distance = haversine_distance(first.1, last.1);

    // Below the noise floor. A heading derived from projection noise is worse
    // than no heading, because correlation would weigh it as evidence.
    //
    // Whether that also means "stationary" depends on how coarse the floor is.
    // Reporting Some(0.0) asserts the object is not moving; that is only
    // defensible when the measurement could have detected it moving. At 40 m
    // from a mast the floor can exceed a walking pace, and there Some(0.0)
    // would be a claim the geometry cannot support — so the answer is None,
    // "not known", which is a different statement and the true one.
    let floor = position.radius_meters.max(1.0);
    if distance < floor {
        let resolvable_mps = floor / seconds;
        track.speed_mps = if resolvable_mps <= STATIONARY_RESOLUTION_MPS {
            Some(0.0)
        } else {
            None
        };
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
        BoundingBox {
            x,
            y: 0.5,
            w: 0.1,
            h: 0.3,
        }
    }

    fn detection(bbox: BoundingBox, class_id: u32) -> Detection {
        Detection {
            bbox,
            confidence: 0.9,
            class_id,
        }
    }

    #[test]
    fn a_track_is_not_reported_until_confirmed() {
        let mut tracker = Tracker::new(TrackerConfig::default(), None);

        tracker.update(&[detection(walking(0.2), 0)], 0);
        assert_eq!(
            tracker.tracks().count(),
            0,
            "one detection is not yet a track"
        );

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

        assert_eq!(
            ids.len(),
            1,
            "the proximity gate must hold small fast boxes together"
        );
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
        assert_eq!(
            tracker.tracks().next().map(|t| t.id),
            Some(id),
            "same object, same id"
        );
    }

    #[test]
    fn a_track_closes_once_the_gap_exceeds_the_limit() {
        let config = TrackerConfig {
            min_hits_to_confirm: 1,
            max_gap_millis: 1000,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, None);

        tracker.update(&[detection(walking(0.2), 0)], 0);
        assert!(tracker.update(&[], 900).ended.is_empty());

        let expired = tracker.update(&[], 1500);
        assert_eq!(expired.ended.len(), 1);
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn different_classes_never_merge() {
        let config = TrackerConfig {
            min_hits_to_confirm: 1,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, None);

        tracker.update(&[detection(walking(0.3), 0)], 0);
        tracker.update(&[detection(walking(0.3), 1)], 200);

        assert_eq!(
            tracker.active_count(),
            2,
            "a person track must not absorb a vehicle"
        );
    }

    #[test]
    fn two_objects_get_two_tracks() {
        let config = TrackerConfig {
            min_hits_to_confirm: 1,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, None);

        for step in 0..5 {
            let at = step * 200;
            tracker.update(
                &[
                    detection(
                        BoundingBox {
                            x: 0.1 + step as f64 * 0.02,
                            y: 0.5,
                            w: 0.08,
                            h: 0.25,
                        },
                        0,
                    ),
                    detection(
                        BoundingBox {
                            x: 0.7 - step as f64 * 0.02,
                            y: 0.5,
                            w: 0.08,
                            h: 0.25,
                        },
                        0,
                    ),
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
                        detection(
                            BoundingBox {
                                x: 0.1 + step as f64 * 0.03,
                                y: 0.5,
                                w: 0.08,
                                h: 0.25,
                            },
                            0,
                        ),
                        detection(
                            BoundingBox {
                                x: 0.6 - step as f64 * 0.02,
                                y: 0.4,
                                w: 0.08,
                                h: 0.25,
                            },
                            0,
                        ),
                    ],
                    at,
                );
                for track in tracker.tracks() {
                    output.push(format!("{}@{:.4}", track.id, track.bbox.x));
                }
            }
            output
        };

        assert_eq!(
            run(),
            run(),
            "the same input must reproduce the same tracks"
        );
    }

    /// A camera on a 10 m mast looking north, tilted 45 degrees down.
    fn pose() -> CameraPose {
        CameraPose {
            position: LatLon {
                lat: 33.8938,
                lon: 35.5018,
            },
            mount_height: 10.0,
            heading: 0.0,
            pitch: -45.0,
            roll: 0.0,
            horizontal_fov: 60.0,
            vertical_fov: 34.0,
            range_meters: 200.0,
        }
    }

    #[test]
    fn a_stationary_object_reports_zero_speed_and_no_heading() {
        let config = TrackerConfig {
            min_hits_to_confirm: 1,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, Some(pose()));

        for step in 0..10 {
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.45,
                        y: 0.6,
                        w: 0.08,
                        h: 0.15,
                    },
                    0,
                )],
                step * 500,
            );
        }

        let track = tracker.tracks().next().expect("track exists");
        assert_eq!(
            track.speed_mps,
            Some(0.0),
            "standing still is zero, not noise"
        );
        assert_eq!(
            track.heading_degrees, None,
            "a heading from jitter is worse than none"
        );
    }

    #[test]
    fn no_pose_means_no_map_position() {
        let config = TrackerConfig {
            min_hits_to_confirm: 1,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, None);
        tracker.update(&[detection(walking(0.2), 0)], 0);

        let track = tracker.tracks().next().unwrap();
        assert!(track.position.is_none());
        assert!(track.speed_mps.is_none());
    }
    #[test]
    fn a_speed_is_withheld_until_it_can_be_measured() {
        // Two frames 200 ms apart amplify position error fivefold. A track that
        // has only just appeared must report "unknown", not a number derived
        // from its own settling.
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));

        for step in 0..3 {
            let detection = Detection {
                bbox: BoundingBox {
                    x: 0.4 + step as f64 * 0.02,
                    y: 0.6,
                    w: 0.05,
                    h: 0.1,
                },
                confidence: 0.9,
                class_id: 0,
            };
            tracker.update(&[detection], step * 200);
        }

        let track = tracker.tracks().next().expect("no track");
        assert!(
            track.speed_mps.is_none(),
            "reported {:?} m/s from 400 ms of observation",
            track.speed_mps
        );
    }

    #[test]
    fn a_speed_appears_once_the_span_is_long_enough() {
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));

        for step in 0..14 {
            let detection = Detection {
                bbox: BoundingBox {
                    x: 0.4 + step as f64 * 0.01,
                    y: 0.6,
                    w: 0.05,
                    h: 0.1,
                },
                confidence: 0.9,
                class_id: 0,
            };
            tracker.update(&[detection], step * 200);
        }

        let track = tracker.tracks().next().expect("no track");
        assert!(
            track.speed_mps.is_some(),
            "2.6 s of observation still gave no speed"
        );
    }

    #[test]
    fn a_single_projection_jump_cannot_produce_an_absurd_speed() {
        // The failure this guard exists for: a detector box that suddenly covers
        // a whole body instead of a fragment moves the ground-contact point
        // metres in one frame. Divided by 200 ms that is a sprinting cheetah.
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));

        let mut boxes = vec![0.60_f64; 6];
        boxes.push(0.40); // the jump

        for (step, y) in boxes.iter().enumerate() {
            let detection = Detection {
                bbox: BoundingBox {
                    x: 0.5,
                    y: *y,
                    w: 0.05,
                    h: 0.1,
                },
                confidence: 0.9,
                class_id: 0,
            };
            tracker.update(&[detection], step as i64 * 200);
        }

        let track = tracker.tracks().next().expect("no track");
        if let Some(speed) = track.speed_mps {
            assert!(speed < 25.0, "a single jump produced {speed} m/s");
        }
    }
    #[test]
    fn motion_is_forgotten_when_the_projection_is_lost() {
        // The defect: an object walks beyond the camera's range, its position
        // degrades to the mast, and the last speed measured while it was still
        // projectable stays frozen and is re-emitted for the life of the track.
        // An evidence package could read "position NOT DETERMINED" directly
        // above "motion 4.38 m/s".
        let camera = CameraPose {
            range_meters: 40.0,
            ..pose()
        };
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(camera));

        // Walk up the frame, well inside range, long enough to earn a speed.
        for step in 0..16 {
            let y = 0.80 - step as f64 * 0.01;
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.5,
                        y,
                        w: 0.05,
                        h: 0.10,
                    },
                    0,
                )],
                step * 200,
            );
        }
        let moving = tracker.tracks().next().unwrap();
        assert!(
            moving.speed_mps.is_some(),
            "the setup never produced a speed"
        );

        // Now step to a row whose ray lands beyond the range: the projection
        // falls back to the camera and there is no ground measurement any more.
        for step in 16..24 {
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.5,
                        y: 0.30,
                        w: 0.05,
                        h: 0.10,
                    },
                    0,
                )],
                step * 200,
            );
        }

        let track = tracker.tracks().next().unwrap();
        if track
            .position
            .map(|p| p.source != PositionSource::GroundProjection)
            .unwrap_or(true)
        {
            assert!(
                track.speed_mps.is_none(),
                "a track with no ground projection still reports {:?} m/s",
                track.speed_mps
            );
            assert!(track.heading_degrees.is_none());
        }
    }

    #[test]
    fn un_placing_the_camera_forgets_the_motion_it_measured() {
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));

        for step in 0..16 {
            let y = 0.60 + step as f64 * 0.005;
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.5,
                        y,
                        w: 0.05,
                        h: 0.10,
                    },
                    0,
                )],
                step * 200,
            );
        }
        assert!(tracker.tracks().next().unwrap().speed_mps.is_some());

        tracker.set_pose(None);
        tracker.update(
            &[detection(
                BoundingBox {
                    x: 0.5,
                    y: 0.68,
                    w: 0.05,
                    h: 0.10,
                },
                0,
            )],
            16 * 200,
        );

        let track = tracker.tracks().next().unwrap();
        assert!(track.position.is_none());
        assert!(
            track.speed_mps.is_none(),
            "a track with no position reported {:?} m/s",
            track.speed_mps
        );
    }

    #[test]
    fn the_gate_scales_each_axis_by_that_axis_extent() {
        // The property that matters, and the one the bug violated: the gate is
        // an ellipse sized per axis, not a circle sized by the larger dimension.
        //
        // A single radius of `gate_factor * max(w, h)` is, for an upright box,
        // `gate_factor * height` — which lets a track jump more than a whole
        // body-height in DEPTH and adopt whoever just walked into shot. That is
        // the failure this replaced, reproduced here at the shape that produced
        // it.
        //
        // Asserted on `association_score` directly, because for any vertical
        // displacement small enough to keep two upright boxes touching the
        // overlap tier answers first — a test routed through `update` would be
        // measuring IoU rather than the gate.
        let tracker = Tracker::new(TrackerConfig::default(), None);
        let from = BoundingBox {
            x: 0.50,
            y: 0.50,
            w: 0.04,
            h: 0.11,
        };

        // A leap in depth: clear of the box vertically, so the gate decides.
        let leap = BoundingBox {
            x: 0.52,
            y: 0.63,
            ..from
        };
        // The circular gate this replaced would have accepted it comfortably.
        let single_radius = TrackerConfig::default().gate_factor * from.w.max(from.h);
        let separation = ((0.02f64).powi(2) + (0.13f64).powi(2)).sqrt();
        assert!(
            separation < single_radius,
            "the fixture no longer reproduces the original bug"
        );

        assert!(
            tracker.association_score(&from, &leap, 1.0).is_none(),
            "a leap of more than a body-height in depth was accepted"
        );

        // The same object drifting sideways is still the same object.
        let sideways = BoundingBox { x: 0.56, ..from };
        assert!(
            tracker.association_score(&from, &sideways, 1.0).is_some(),
            "a lateral drift within the gate must still associate"
        );
    }

    #[test]
    fn a_speed_too_coarse_to_resolve_is_unknown_rather_than_zero() {
        // At range the position uncertainty can exceed a walking pace. Saying
        // "stationary" there asserts something the geometry cannot support:
        // still and slow are indistinguishable, so the answer is "not known".
        let far = CameraPose {
            pitch: -3.0,
            range_meters: 400.0,
            ..pose()
        };
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(far));

        for step in 0..16 {
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.5,
                        y: 0.55,
                        w: 0.02,
                        h: 0.04,
                    },
                    0,
                )],
                step * 200,
            );
        }

        let track = tracker.tracks().next().unwrap();
        let radius = track.position.map(|p| p.radius_meters).unwrap_or(0.0);
        if radius > 2.0 {
            assert!(
                track.speed_mps.is_none(),
                "reported {:?} m/s as 'stationary' from a position known only to +/-{radius:.1} m",
                track.speed_mps
            );
        }
    }

    #[test]
    fn a_confident_stationary_object_still_reports_zero() {
        // The other half of the rule: close to the camera the floor is small,
        // so "not moving" is a conclusion the measurement supports and must
        // still be stated as 0.0 rather than as unknown.
        let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));

        for step in 0..16 {
            tracker.update(
                &[detection(
                    BoundingBox {
                        x: 0.5,
                        y: 0.85,
                        w: 0.06,
                        h: 0.12,
                    },
                    0,
                )],
                step * 200,
            );
        }

        let track = tracker.tracks().next().unwrap();
        assert_eq!(
            track.speed_mps,
            Some(0.0),
            "standing still is zero, not unknown"
        );
        assert_eq!(track.heading_degrees, None);
    }

    #[test]
    fn a_coasted_track_does_not_accumulate_a_speed_from_its_own_guesses() {
        // clamp_box pins a coasted box to the frame edge once the object has
        // left the scene. Feeding those extrapolations back in as ground
        // measurements let an object that walked out of shot be reported as
        // standing at a confident map position, forever.
        let config = TrackerConfig {
            max_gap_millis: 30_000,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, Some(pose()));

        for step in 0..14 {
            let x = 0.50 + step as f64 * 0.02;
            tracker.update(
                &[detection(
                    BoundingBox {
                        x,
                        y: 0.70,
                        w: 0.05,
                        h: 0.10,
                    },
                    0,
                )],
                step * 200,
            );
        }
        let before = tracker.tracks().next().unwrap().speed_mps;

        // Now nothing at all, for a long time. The track coasts.
        for step in 14..80 {
            tracker.update(&[], step * 200);
        }

        let track = tracker
            .tracks()
            .next()
            .expect("the track should still be coasting");
        assert_eq!(
            track.speed_mps, before,
            "coasting changed the measured speed, so extrapolation was recorded as evidence"
        );
    }

    #[test]
    fn confirmation_survives_a_detector_that_misses() {
        // `min_hits_to_confirm` is cumulative, and this pins that down because
        // the field was once documented as "consecutive".
        //
        // Enforcing consecutiveness was tried and measured: it took the
        // reference scene from 5 tracks to 8 for three people. An unbroken run
        // is the wrong requirement when the detector's recall is 0.69 — the
        // misses are the normal case, not the exception, and demanding a clean
        // streak just means the same person keeps being rediscovered as
        // somebody new.
        let config = TrackerConfig {
            min_hits_to_confirm: 3,
            max_gap_millis: 30_000,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config, None);
        let box_ = BoundingBox {
            x: 0.5,
            y: 0.5,
            w: 0.05,
            h: 0.10,
        };

        // Seen, missed, seen, missed, seen. Three sightings of one object.
        tracker.update(&[detection(box_, 0)], 0);
        tracker.update(&[], 1000);
        tracker.update(&[detection(box_, 0)], 2000);
        tracker.update(&[], 3000);
        tracker.update(&[detection(box_, 0)], 4000);

        assert_eq!(
            tracker.tracks().count(),
            1,
            "three sightings of one object should confirm it, misses notwithstanding"
        );
    }
}
