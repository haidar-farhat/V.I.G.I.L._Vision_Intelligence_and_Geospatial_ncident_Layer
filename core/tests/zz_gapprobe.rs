use sentinel_core::geometry::{BoundingBox, CameraPose, LatLon, PositionSource};
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn pose() -> CameraPose {
    CameraPose {
        position: LatLon { lat: 33.8938, lon: 35.5018 },
        mount_height: 10.0,
        heading: 0.0,
        pitch: -10.0,
        roll: 0.0,
        horizontal_fov: 60.0,
        vertical_fov: 34.0,
        range_meters: 200.0,
    }
}

fn det(x: f64, y: f64) -> Detection {
    Detection {
        bbox: BoundingBox { x, y, w: 0.05, h: 0.1 },
        confidence: 0.9,
        class_id: 0,
    }
}

fn dump(tracker: &Tracker, at: i64, tag: &str) {
    for t in tracker.tracks() {
        let (src, rad) = match t.position {
            Some(p) => (format!("{:?}", p.source), p.radius_meters),
            None => ("none".into(), f64::NAN),
        };
        println!(
            "{tag} t={at:>6} id={} src={src:<16} r={rad:8.2} speed={:?} heading={:?}",
            t.id, t.speed_mps, t.heading_degrees
        );
    }
}

#[test]
fn probe_projection_blackout() {
    let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));
    let mut at = 0i64;

    // Phase A: object standing still, feet visible low in frame -> projects.
    for _ in 0..16 {
        tracker.update(&[det(0.5, 0.45)], at);
        dump(&tracker, at, "A");
        at += 200;
    }

    // Phase B: feet occluded, box bottom climbs the frame -> ground contact goes
    // above the horizon -> CameraFallback. Detections continue every frame, so
    // the track never expires.
    let mut y = 0.45;
    for _ in 0..8 {
        y -= 0.04;
        tracker.update(&[det(0.5, y)], at);
        dump(&tracker, at, "B");
        at += 200;
    }
    // Hold there for 60 seconds of continuous detections.
    let hold_start = at;
    while at < hold_start + 60_000 {
        tracker.update(&[det(0.5, y)], at);
        at += 200;
    }
    dump(&tracker, at - 200, "Bhold");

    // Phase C: feet visible again, object is back near where it left.
    let mut y2 = y;
    for i in 0..8 {
        y2 += 0.04;
        tracker.update(&[det(0.5, y2)], at);
        dump(&tracker, at, &format!("C{i}"));
        at += 200;
    }
    for i in 0..4 {
        tracker.update(&[det(0.5, 0.45)], at);
        dump(&tracker, at, &format!("D{i}"));
        at += 200;
    }
    assert!(tracker.tracks().count() > 0, "track must survive");
    let _ = PositionSource::GroundProjection;
}
