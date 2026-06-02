//! Probe rapier's own foot–ground contact while the biped holds its default pose
//! (zero action → PD holds the standing pose). Reports, per foot, the lowest
//! collider Z, body Z, active ground-contact count, penetration, and contact
//! normal — to see exactly how the feet meet the ground in the physics engine.
//!
//! Run: `cargo run --release --example biped_footcheck --features cpu`

#[path = "biped_env.rs"]
mod biped_env;

use biped_env::{BipedEnv, default_mjcf_path};

fn main() {
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let mut env = BipedEnv::new(&xml, 1);
    let zero = [0.0f32; 12]; // hold default (straight-leg) standing pose
    println!("AT SPAWN (no physics step yet) — feet should already be on the ground:");
    println!("  torso_z={:.3}", env.torso_height());
    print!("{}", env.foot_report());
    println!("\nHolding default pose; rapier foot–ground contact (ground top at z=0):\n");
    for step in 0..160 {
        let _ = env.step(&zero);
        if step % 20 == 0 {
            let [tr, tl] = env.foot_tilts();
            println!(
                "step {step:>3}  torso_z={:.3}  sole tilt: R={:.1}deg L={:.1}deg",
                env.torso_height(),
                tr.to_degrees(),
                tl.to_degrees()
            );
            print!("{}", env.foot_report());
        }
    }
    // Verdict: if the feet stay near-flat here (zero policy input, pure physics),
    // the simulator supports flat stance — edge-standing was a reward/policy issue.
    let [tr, tl] = env.foot_tilts();
    println!(
        "\nPHYSICS-ONLY VERDICT (held flat default pose, no policy): R={:.1}deg L={:.1}deg  -> sim {} hold flat feet",
        tr.to_degrees(),
        tl.to_degrees(),
        if tr.to_degrees() < 10.0 && tl.to_degrees() < 10.0 {
            "CAN"
        } else {
            "struggles to"
        }
    );
}
