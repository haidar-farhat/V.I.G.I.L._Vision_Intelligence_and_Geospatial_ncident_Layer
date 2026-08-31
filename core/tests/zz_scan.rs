use sentinel_core::geometry::*;

fn pose() -> CameraPose {
    CameraPose { position: LatLon { lat: 33.8938, lon: 35.5018 }, mount_height: 10.0,
        heading: 0.0, pitch: -45.0, roll: 0.0, horizontal_fov: 60.0, vertical_fov: 34.0, range_meters: 200.0 }
}

#[test]
fn scan() {
    let p = pose();
    for i in 0..=20 {
        let v = i as f64 / 20.0;
        let a = project_to_ground(&p, 0.50, v, DEFAULT_ANGULAR_UNCERTAINTY_DEG, true);
        let b = project_to_ground(&p, 0.51, v, DEFAULT_ANGULAR_UNCERTAINTY_DEG, true);
        match (a, b) {
            (Some(a), Some(b)) => println!(
                "v={:.2} dist={:7.2} r={:6.2} m per 0.01x = {:.3}",
                v, a.ground_distance_meters, a.uncertainty_meters,
                haversine_distance(a.position, b.position)),
            _ => println!("v={:.2} none", v),
        }
    }
}
