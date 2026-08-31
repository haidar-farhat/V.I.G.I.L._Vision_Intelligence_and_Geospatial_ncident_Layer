use sentinel_core::geometry::{BoundingBox, CameraPose, LatLon, PositionSource};
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn mast() -> CameraPose {
    CameraPose {
        position: LatLon { lat: 33.8938, lon: 35.5018 },
        mount_height: 10.0,
        heading: 0.0,
        pitch: -20.0,
        roll: 0.0,
        horizontal_fov: 60.0,
        vertical_fov: 34.0,
        range_meters: 40.0,
    }
}

fn det(x: f64, y: f64) -> Detection {
    Detection {
        bbox: BoundingBox { x, y, w: 0.06, h: 0.12 },
        confidence: 0.9,
        class_id: 0,
    }
}

fn dump(tag: &str, tracker: &Tracker, t: i64) {
    for tr in tracker.tracks() {
        let src = match tr.position.map(|p| p.source) {
            Some(PositionSource::GroundProjection) => "ground",
            Some(PositionSource::CameraFallback) => "FALLBACK",
            None => "none",
        };
        println!(
            "{tag:>12} t={t:>7} id={} src={src:>8} speed={:?} heading={:?}",
            tr.id, tr.speed_mps, tr.heading_degrees
        );
    }
}

#[test]
fn probe_speed_after_a_long_unprojectable_gap() {
    let mut tracker = Tracker::new(TrackerConfig::default(), Some(mast()));
    let mut t = 0i64;

    // Phase A: 4 s in range, slowly walking. Establishes a real measurement.
    let mut x = 0.20;
    for _ in 0..20 {
        tracker.update(&[det(x, 0.48)], t);
        t += 200;
        x += 0.002;
    }
    dump("in-range", &tracker, t);

    // Phase B1: climb out of projectable range over 8 frames.
    let mut y = 0.48;
    for _ in 0..8 {
        y -= 0.04;
        tracker.update(&[det(x, y)], t);
        t += 200;
    }
    dump("leaving", &tracker, t);

    // Phase B2: 60 s beyond range_meters, walking right across the frame the
    // whole time. Detected every frame, so the track never expires.
    let gap_start = t;
    for _ in 0..300 {
        tracker.update(&[det(x, y)], t);
        t += 200;
        x += 0.002;
    }
    println!("gap of {} ms with no projectable position", t - gap_start);
    dump("out-of-range", &tracker, t);

    // Phase C: walk back down into range.
    for step in 0..10 {
        y += 0.04;
        if y > 0.48 { y = 0.48; }
        tracker.update(&[det(x, y)], t);
        dump(&format!("re-entry{step}"), &tracker, t);
        t += 200;
        x += 0.002;
    }
}
