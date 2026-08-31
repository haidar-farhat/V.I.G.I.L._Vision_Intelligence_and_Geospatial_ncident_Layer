use sentinel_core::geometry::*;
use sentinel_core::tracking::*;

const SITE: LatLon = LatLon { lat: 33.8938, lon: 35.5018 };

fn mast() -> CameraPose {
    CameraPose {
        position: SITE,
        mount_height: 10.0,
        heading: 0.0,
        pitch: -20.0,
        roll: 0.0,
        horizontal_fov: 60.0,
        vertical_fov: 34.0,
        range_meters: 90.0,
    }
}

const W: f64 = 0.05;
const H: f64 = 0.10;

/// Image column whose ray, at the fixed depression used below, lands `lateral`
/// metres to the side of boresight at ~60 m range.
fn column_for(lateral_m: f64, range_m: f64, pose: &CameraPose) -> f64 {
    let yaw = (lateral_m / range_m).atan();
    let half_h = (pose.horizontal_fov / 2.0).to_radians();
    let dx = yaw.tan() / half_h.tan();
    (dx + 1.0) / 2.0
}

/// Image row whose ray meets the ground at `range_m`.
fn row_for(range_m: f64, pose: &CameraPose) -> f64 {
    let depression = (pose.mount_height / range_m).atan().to_degrees();
    let elevation = -depression;
    let pitch_offset = (elevation - pose.pitch).to_radians();
    let half_v = (pose.vertical_fov / 2.0).to_radians();
    let dy = pitch_offset.tan() / half_v.tan();
    (1.0 - dy) / 2.0
}

fn det_at(lateral_m: f64, range_m: f64, pose: &CameraPose) -> Detection {
    let u = column_for(lateral_m, range_m, pose);
    let v = row_for(range_m, pose);
    Detection {
        bbox: BoundingBox { x: u - W / 2.0, y: v - H, w: W, h: H },
        confidence: 0.9,
        class_id: 0,
    }
}

fn run(label: &str, speed_mps: f64) {
    let pose = mast();
    let mut tracker = Tracker::new(TrackerConfig::default(), Some(pose));
    let mut t = 0i64;
    let mut lateral = -3.0f64;
    println!("--- {label} (true speed {speed_mps:.2} m/s) ---");
    for step in 0..26 {
        tracker.update(&[det_at(lateral, 60.0, &pose)], t);
        if step % 5 == 0 || step == 25 {
            if let Some(tr) = tracker.tracks().next() {
                println!(
                    "  t={:>5} ms  lateral={:>6.2} m  proj={:?} r={:>5.2} m  speed={:?} heading={:?}",
                    t,
                    lateral,
                    tr.position.map(|p| p.source),
                    tr.position.map(|p| p.radius_meters).unwrap_or(f64::NAN),
                    tr.speed_mps,
                    tr.heading_degrees
                );
            }
        }
        lateral += speed_mps * 0.2;
        t += 200;
    }
    let tr = tracker.tracks().next().expect("track alive");
    println!(
        "  FINAL: speed {:?} heading {:?}  (walked {:.2} m in 5.0 s)",
        tr.speed_mps,
        tr.heading_degrees,
        speed_mps * 5.0
    );
}

#[test]
fn probe_slow_lateral_walk_at_range() {
    let pose = mast();
    let contact = det_at(0.0, 60.0, &pose).bbox.ground_contact();
    let p = project_to_ground(&pose, contact.x, contact.y, DEFAULT_ANGULAR_UNCERTAINTY_DEG, true)
        .expect("ray meets ground");
    println!(
        "geometry check: ground distance {:.2} m, uncertainty {:.2} m",
        p.ground_distance_meters, p.uncertainty_meters
    );

    run("standing perfectly still", 0.0);
    run("walking laterally", 1.4);
    run("jogging laterally", 2.4);
    run("running laterally", 4.0);
}
