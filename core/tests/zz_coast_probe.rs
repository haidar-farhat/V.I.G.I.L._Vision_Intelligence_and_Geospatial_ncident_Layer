use sentinel_core::geometry::{BoundingBox, CameraPose, LatLon, PositionSource};
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn pose() -> CameraPose {
    CameraPose {
        position: LatLon { lat: 33.8938, lon: 35.5018 },
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
fn probe_coast_off_frame() {
    let cfg = TrackerConfig::default();
    println!("config: max_gap {} window {} min_span {}", cfg.max_gap_millis, cfg.motion_window_millis, cfg.min_motion_span_millis);
    let mut tracker = Tracker::new(cfg, Some(pose()));

    // Walk right across the frame, 200 ms frames, 0.03 normalized units/frame.
    let mut step = 0i64;
    let mut x = 0.30f64;
    while x <= 0.90 {
        let bbox = BoundingBox { x, y: 0.60, w: 0.10, h: 0.30 };
        tracker.update(&[Detection { bbox, confidence: 0.9, class_id: 0 }], step * 200);
        if let Some(t) = tracker.tracks().next() {
            println!(
                "det  t={:>5} x={:.4} src={:?} r={:.2} speed={:?}",
                step * 200, t.bbox.x,
                t.position.map(|p| p.source), t.position.map(|p| p.radius_meters).unwrap_or(0.0),
                t.speed_mps
            );
        }
        step += 1;
        x += 0.03;
    }

    // Detections stop entirely: the object has left the scene.
    for coast in 0..14 {
        let at = step * 200;
        let upd = tracker.update(&[], at);
        step += 1;
        let live = tracker.tracks().next().map(|t| (
            t.bbox.x,
            t.position.map(|p| p.source),
            t.position.map(|p| p.radius_meters).unwrap_or(0.0),
            t.position.map(|p| (p.point.lat, p.point.lon)),
            t.speed_mps,
            t.heading_degrees,
        ));
        println!("coast {:>2} t={:>5} ended={:?} -> {:?}", coast, at, upd.ended, live);
    }
    let _ = PositionSource::GroundProjection;
}
