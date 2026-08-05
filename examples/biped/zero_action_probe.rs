//! Zero-action survival probe: how long does the robot live under a pure
//! PD hold at the default pose (action = 0), with NO policy at all?
//!
//! If this can't reach the episode timeout, the env is lethal independent of
//! learning — the terrain-training failure is a physics/config problem
//! (gains, contact, spawn, termination thresholds), not an RL problem.
//! Reports the survival curve (alive fraction per second) and mean episode
//! length over `--steps` control ticks.
//!
//!   cargo run --release --example zero_action_probe --features "gpu biped_gpu cuda_backend" -- [num_envs] [steps]
//!
//! Honors the full BIPED_* env-var stack (TERRAIN, ROBOT, MOTOR_DELAY, DR,
//! RESET_VEL, PUSH_*, …), so it probes exactly the training configuration.

mod biped_env;
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use zealot_env::robots::NUM_JOINTS;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1100);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let _ = env.initial_obs().await;
        let actions = vec![[0.0f32; NUM_JOINTS]; n];

        // Per-env: has this env's FIRST episode ended yet, and at which step?
        let mut first_len = vec![usize::MAX; n];
        let mut ep_lens: Vec<usize> = Vec::new();
        let mut cur_len = vec![0usize; n];
        let dt = 0.02f32; // VelocityFlatTask control_dt (1/200 x 4)

        for t in 0..steps {
            let outs = env.step(&actions).await;
            for e in 0..n {
                cur_len[e] += 1;
                if outs[e].done {
                    if first_len[e] == usize::MAX {
                        first_len[e] = t + 1;
                    }
                    ep_lens.push(cur_len[e]);
                    cur_len[e] = 0;
                }
            }
            if t < 40 || (t + 1) % 10 == 0 {
                let zs = env.torso_heights().await;
                let zmin = zs.iter().cloned().fold(f32::INFINITY, f32::min);
                let zmean = zs.iter().sum::<f32>() / zs.len() as f32;
                println!("t={t} torso z env0={:.3} mean={:.3} min={:.3}", zs[0], zmean, zmin);
            }
            if (t + 1) % 50 == 0 {
                let alive_first = first_len.iter().filter(|&&v| v == usize::MAX).count();
                println!(
                    "t={:.1}s  first-episode-still-alive {}/{} ({:.0}%)",
                    (t + 1) as f32 * dt,
                    alive_first,
                    n,
                    100.0 * alive_first as f32 / n as f32
                );
            }
        }
        let survived = first_len.iter().filter(|&&v| v == usize::MAX).count();
        let ended: Vec<usize> = first_len.iter().copied().filter(|&v| v != usize::MAX).collect();
        let mean_first = if ended.is_empty() {
            f32::NAN
        } else {
            ended.iter().sum::<usize>() as f32 / ended.len() as f32
        };
        let mean_ep = if ep_lens.is_empty() {
            f32::NAN
        } else {
            ep_lens.iter().sum::<usize>() as f32 / ep_lens.len() as f32
        };
        println!(
            "RESULT: first-episode survival to t={:.1}s: {}/{} ({:.0}%); mean first-episode len {:.1} steps ({:.2}s); mean episode len over all resets {:.1} steps ({:.2}s)",
            steps as f32 * dt,
            survived,
            n,
            100.0 * survived as f32 / n as f32,
            mean_first,
            mean_first * dt,
            mean_ep,
            mean_ep * dt,
        );
    });
}
