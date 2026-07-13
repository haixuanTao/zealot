//! Passive-stability probe for the TRAINER's sim: create the batched nexus env
//! (same MJCF / colliders / PD setup the trainer uses), apply ZERO action
//! (PD holds the nominal pose), do NOT reset, and watch torso height + fall
//! fraction over time. If the robot stands, mean torso stays ~constant and
//! fell_frac → 0; if the sim can't hold a stand, torso drops / fell_frac stays high.
//!
//! Run: `BIPED_SPAWN_DR=0 BIPED_CUDA=1 cargo run --release --example passive_stand \
//!       --features "gpu biped_gpu cuda_backend" -- [num_envs] [steps]`

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "cutile_gemm.rs"]
mod cutile_gemm;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    // BIPED_ACT: constant action on every joint. 0.0 = hold the all-zero pose;
    // 1.0 = hold the bent "home" crouch (action_scale == WBC-AGILE's nominal
    // angles, so q_target = default_pos + scale·1 = the bent stance).
    let act: f32 = std::env::var("BIPED_ACT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("mjcf");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let zero = vec![[act; NUM_JOINTS]; n];
        println!(
            "(action level = {act}: {})",
            if act == 0.0 {
                "all-zero pose"
            } else {
                "bent home crouch"
            }
        );
        let z0 = env.torso_heights().await;
        let mean0 = z0.iter().sum::<f32>() / n as f32;
        println!(
            "\npassive stand (zero action, no reset): {n} envs, spawn mean torso_z={mean0:.3}"
        );
        println!(
            "{:>5}  {:>10}  {:>9}  {:>9}  {:>9}",
            "step", "mean_torso", "min_torso", "max_torso", "fell_frac"
        );
        let dump_motors = std::env::var("BIPED_DEBUG_MOTORS").is_ok();
        for s in 0..steps {
            let outs = env.step(&zero).await;
            if s == 0 && dump_motors {
                env.debug_dump_motors(0).await;
            }
            if s % 20 == 0 || s == steps - 1 {
                let zs = env.torso_heights().await;
                let mean = zs.iter().sum::<f32>() / n as f32;
                let min = zs.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let fell = outs.iter().filter(|o| o.fell).count() as f32 / n as f32;
                println!("{s:>5}  {mean:>10.3}  {min:>9.3}  {max:>9.3}  {fell:>9.3}");
            }
        }
    });
}
