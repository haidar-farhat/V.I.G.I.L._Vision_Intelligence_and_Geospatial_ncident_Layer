use sentinel_core::geometry::{BoundingBox, CameraPose, LatLon, haversine_distance};
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn pose() -> CameraPose {
    CameraPose { position: LatLon { lat: 33.8938, lon: 35.5018 }, mount_height: 12.0,
        heading: 0.0, pitch: -20.0, roll: 0.0, horizontal_fov: 60.0, vertical_fov: 34.0, range_meters: 200.0 }
}

#[test]
fn coast_far() {
    let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose()));
    let mut step = 0i64;
    let mut x = 0.55f64;
    let mut last_pt = None;
    // bottom edge at 0.40 => y = 0.40 - h
    let h = 0.08f64;
    let w = 0.03f64;
    while x <= 0.99 {
        let bbox = BoundingBox { x, y: 0.40 - h, w, h };
        tracker.update(&[Detection { bbox, confidence: 0.9, class_id: 0 }], step * 200);
        if let Some(t) = tracker.tracks().next() {
            let p = t.position.unwrap();
            let moved = last_pt.map(|q| haversine_distance(q, p.point)).unwrap_or(0.0);
            println!("det   t={:>5} x={:.4} r={:.2} step_m={:.3} speed={:?}", step*200, t.bbox.x, p.radius_meters, moved, t.speed_mps);
            last_pt = Some(p.point);
        }
        step += 1;
        x += 0.011;
    }
    for coast in 0..12 {
        let at = step * 200;
        let upd = tracker.update(&[], at);
        step += 1;
        match tracker.tracks().next() {
            Some(t) => {
                let p = t.position.unwrap();
                println!("coast {:>2} t={:>5} x={:.4} src={:?} r={:.2} pt=({:.7},{:.7}) speed={:?} head={:?}",
                    coast, at, t.bbox.x, p.source, p.radius_meters, p.point.lat, p.point.lon, t.speed_mps, t.heading_degrees);
            }
            None => println!("coast {:>2} t={:>5} ended={:?} (no live track)", coast, at, upd.ended),
        }
    }
}
