use sentinel_core::geometry::*;
use sentinel_core::tracking::{Detection, Tracker, TrackerConfig};

fn pose() -> CameraPose {
    CameraPose {
        position: LatLon { lat: 33.8938, lon: 35.5018 },
        mount_height: 10.0, heading: 0.0, pitch: -45.0, roll: 0.0,
        horizontal_fov: 60.0, vertical_fov: 34.0, range_meters: 200.0,
    }
}

fn depth_of(p: &CameraPose, v: f64) -> Option<f64> {
    project_to_ground(p, 0.5, v, 0.5, false).map(|g| g.ground_distance_meters)
}

#[test]
fn probe2() {
    let p = pose();
    // (a) what the 0.35 ratio actually prevents: circle-of-max-dimension gate
    let contact = 0.75_f64;
    let d0 = depth_of(&p, contact).unwrap();
    for (name, gate) in [("with ratio (0.2625)", 0.2625_f64), ("no ratio, h-scaled (0.75)", 0.75)] {
        let up = (contact - gate).max(0.0);
        let d = depth_of(&p, up);
        println!("{name}: contact {contact:.3} -> {up:.3};  depth {:.1} m -> {:?} m",
            d0, d.map(|x| (x*10.0).round()/10.0));
    }

    // (b) the projection's own error ellipse: depth sigma vs lateral sigma
    for v in [0.9_f64, 0.75, 0.6, 0.5, 0.4, 0.35] {
        let g = project_to_ground(&p, 0.5, v, 0.5, false);
        if let Some(g) = g {
            let depression = (-(ray_angles(&p, 0.5, v).1)).to_radians();
            let sigma = 0.5_f64.to_radians();
            let range_sigma = p.mount_height * sigma / (depression.sin() * depression.sin());
            let lateral_sigma = g.ground_distance_meters * sigma;
            println!("v={v:.2} range={:6.1} m  depth_sigma={:5.2}  lateral_sigma={:5.2}  ratio={:.2}",
                g.ground_distance_meters, range_sigma, lateral_sigma, range_sigma / lateral_sigma);
        } else { println!("v={v:.2} above horizon"); }
    }

    // (c) realistic 16:9 person box: 0.5 m wide x 1.75 m tall at ~15 m
    // normalised aspect h/w about 6 -> gate_y/gate_x about 2.1
    let (w, h) = (0.03_f64, 0.18_f64);
    let gx = 2.5*w; let gy = 2.5*h*0.35;
    println!("\nrealistic box w={w} h={h}: gate_x={gx:.4} gate_y={gy:.4} norm ratio={:.2}, pixel ratio(16:9)={:.2}",
        gy/gx, (gy*1080.0)/(gx*1920.0));
    let cfg = TrackerConfig { min_hits_to_confirm: 1, ..Default::default() };
    for jump in [0.05_f64, 0.075, 0.10, 0.13, 0.16] {
        let mut up = Tracker::new(cfg, None);
        up.update(&[Detection{bbox:BoundingBox{x:0.45,y:0.5,w,h},confidence:0.9,class_id:0}], 0);
        up.update(&[Detection{bbox:BoundingBox{x:0.45,y:0.5-jump,w,h},confidence:0.9,class_id:0}], 200);
        let mut side = Tracker::new(cfg, None);
        side.update(&[Detection{bbox:BoundingBox{x:0.45,y:0.5,w,h},confidence:0.9,class_id:0}], 0);
        side.update(&[Detection{bbox:BoundingBox{x:0.45+jump,y:0.5,w,h},confidence:0.9,class_id:0}], 200);
        // world equivalents at the reference pose
        let b0 = BoundingBox{x:0.45,y:0.5,w,h};
        let bu = BoundingBox{x:0.45,y:0.5-jump,w,h};
        let bs = BoundingBox{x:0.45+jump,y:0.5,w,h};
        let e0 = project_detection(&p,&b0);
        let du = haversine_distance(e0.point, project_detection(&p,&bu).point);
        let ds = haversine_distance(e0.point, project_detection(&p,&bs).point);
        println!("  jump {jump:.3}: up={} ({:.2} m depth)  side={} ({:.2} m lateral)",
            up.active_count(), du, side.active_count(), ds);
    }
    assert!(false, "probe");
}
