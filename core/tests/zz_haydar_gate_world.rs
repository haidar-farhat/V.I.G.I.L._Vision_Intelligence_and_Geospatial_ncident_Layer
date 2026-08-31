use sentinel_core::geometry::*;
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn det(x: f64, y: f64, w: f64, h: f64) -> Detection {
    Detection { bbox: BoundingBox { x, y, w, h }, confidence: 0.9, class_id: 0 }
}

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
fn probe() {
    let config = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };

    // --- 1. reproduce the reviewer's claim -------------------------------
    for jump in [0.245_f64, 0.251, 0.255, 0.263, 0.27] {
        let mut up = Tracker::new(config, None);
        up.update(&[det(0.45, 0.5, 0.1, 0.3)], 0);
        up.update(&[det(0.45, 0.5 - jump, 0.1, 0.3)], 200);

        let mut down = Tracker::new(config, None);
        down.update(&[det(0.45, 0.5, 0.1, 0.3)], 0);
        down.update(&[det(0.45, 0.5 + jump, 0.1, 0.3)], 200);

        let mut side = Tracker::new(config, None);
        side.update(&[det(0.45, 0.5, 0.1, 0.3)], 0);
        side.update(&[det(0.45 + jump, 0.5, 0.1, 0.3)], 200);

        println!(
            "jump {:.3}: up={} down={} side={}",
            jump, up.active_count(), down.active_count(), side.active_count()
        );
    }

    // --- 2. what the gate is worth in the world --------------------------
    let p = pose();
    let w = 0.1_f64;
    let h = 0.3_f64;
    let gate_x = 2.5 * w;              // 0.250 normalised frame widths
    let gate_y = 2.5 * h * 0.35;       // 0.2625 normalised frame heights
    println!("\ngate_x={gate_x:.4} gate_y={gate_y:.4}  gate_y/gate_x={:.3}", gate_y / gate_x);

    // pixels, 1920x1080 and 1280x720 (both 16:9) and 4:3
    for (pw, ph) in [(1920.0, 1080.0), (704.0, 576.0)] {
        println!(
            "  frame {}x{}: gate_x={:.0}px gate_y={:.0}px  vertical/horizontal={:.2}",
            pw, ph, gate_x * pw, gate_y * ph, (gate_y * ph) / (gate_x * pw)
        );
    }
    // angular, using the pose's own optics
    println!(
        "  angular: gate_x={:.2} deg  gate_y={:.2} deg  vertical/horizontal={:.2}",
        gate_x * 60.0, gate_y * 34.0, (gate_y * 34.0) / (gate_x * 60.0)
    );

    // world: a person standing mid-frame, then displaced to each gate edge
    let base = BoundingBox { x: 0.45, y: 0.45, w, h };
    let up_edge = BoundingBox { x: 0.45, y: 0.45 - gate_y, w, h };
    let down_edge = BoundingBox { x: 0.45, y: 0.45 + gate_y, w, h };
    let side_edge = BoundingBox { x: 0.45 + gate_x, y: 0.45, w, h };

    let pe = |b: &BoundingBox| project_detection(&p, b);
    let b0 = pe(&base);
    println!("\nbase contact v={:.3} -> {:?} r={:.2}m", base.ground_contact().y, b0.source, b0.radius_meters);
    for (name, bb) in [("up", up_edge), ("down", down_edge), ("side", side_edge)] {
        let e = pe(&bb);
        let d = haversine_distance(b0.point, e.point);
        println!(
            "  {name}: contact v={:.3} source={:?} world jump={:.2} m (radius {:.2} m)",
            bb.ground_contact().y, e.source, d, e.radius_meters
        );
    }

    // --- 3. distant small box, the case the module says it designed for ---
    let (w2, h2) = (0.018_f64, 0.15_f64);
    let gx2 = 2.5 * w2;
    let gy2 = 2.5 * h2 * 0.35;
    println!("\nsmall box w={w2} h={h2}: gate_x={gx2:.4} gate_y={gy2:.4} ratio={:.2}", gy2 / gx2);
    let b = BoundingBox { x: 0.5, y: 0.30, w: w2, h: h2 };
    let bu = BoundingBox { x: 0.5, y: 0.30 - gy2, w: w2, h: h2 };
    let bd = BoundingBox { x: 0.5, y: 0.30 + gy2, w: w2, h: h2 };
    let bs = BoundingBox { x: 0.5 + gx2, y: 0.30, w: w2, h: h2 };
    let e0 = pe(&b);
    println!("  base source={:?} radius={:.2}", e0.source, e0.radius_meters);
    for (name, bb) in [("up", bu), ("down", bd), ("side", bs)] {
        let e = pe(&bb);
        println!("  {name}: source={:?} world jump={:.2} m radius={:.2}", e.source, haversine_distance(e0.point, e.point), e.radius_meters);
    }
    assert!(false, "probe");
}
