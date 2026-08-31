use sentinel_core::geometry::*;

fn pose() -> CameraPose {
    CameraPose { position: LatLon{lat:33.8938,lon:35.5018}, mount_height:10.0, heading:0.0,
        pitch:-45.0, roll:0.0, horizontal_fov:60.0, vertical_fov:34.0, range_meters:200.0 }
}

fn sigmas(p:&CameraPose, v:f64) -> (f64,f64,f64) {
    let (_b, elev) = ray_angles(p, 0.5, v);
    let dep = (-elev).to_radians();
    let d = p.mount_height / dep.tan();
    let s = DEFAULT_ANGULAR_UNCERTAINTY_DEG.to_radians();
    (d, p.mount_height*s/(dep.sin()*dep.sin()), d*s)
}

#[test]
fn probe3() {
    let p = pose();
    for (label, w, h, contact) in [("module reference box", 0.1_f64, 0.3_f64, 0.75_f64),
                                   ("realistic 16:9 person", 0.03, 0.18, 0.68)] {
        let gx = 2.5*w; let gy = 2.5*h*0.35;
        let (d0, ds, ls) = sigmas(&p, contact);
        let b0 = BoundingBox{x:0.45,y:contact-h,w,h};
        let up = BoundingBox{x:0.45,y:contact-h-gy,w,h};
        let side = BoundingBox{x:0.45+gx,y:contact-h,w,h};
        let e0 = project_detection(&p,&b0);
        let du = haversine_distance(e0.point, project_detection(&p,&up).point);
        let dl = haversine_distance(e0.point, project_detection(&p,&side).point);
        println!("{label}: range {d0:.1} m  depth_sigma {ds:.2} m  lateral_sigma {ls:.2} m (sigma ratio {:.2})", ds/ls);
        println!("   gate_x={gx:.4} gate_y={gy:.4} (normalised ratio {:.2})", gy/gx);
        println!("   vertical gate  = {du:.2} m of depth   = {:.1} sigma", du/ds);
        println!("   horizontal gate= {dl:.2} m lateral    = {:.1} sigma", dl/ls);
        println!("   -> vertical gate is {:.0}% of the horizontal gate in sigma terms\n", 100.0*(du/ds)/(dl/ls));
    }
    assert!(false,"probe");
}
