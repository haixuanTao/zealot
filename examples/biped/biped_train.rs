//! Stage 4 — train flat velocity tracking on the LeRobot bipedal (rapier CPU,
//! MJCF model) with the zealot-rl PPO stack. The env (`biped_env`), the MDP
//! (`zealot-env`), and the learner (`zealot-rl`) meet on the real robot model.
//!
//! Envs step in parallel across cores (rayon) and observations are normalized
//! (rsl_rl-style). Still CPU + a hand-rolled MLP, so a modest net/budget; the
//! deployed config is [512,256,128] × 4096 envs on a GPU. CLI-tunable.
//!
//! Run: `cargo run --release --example biped_train --features cpu -- [iters] [num_envs]`

#[path = "biped_env.rs"]
mod biped_env;

use biped_env::{BipedEnv, default_mjcf_path, ppo_iteration};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig};

const T: usize = 32; // steps per env per iteration

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let num_envs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    println!("building {num_envs} envs (stepping in parallel across cores)...");
    let mut envs: Vec<BipedEnv> = (0..num_envs)
        .map(|i| BipedEnv::new(&xml, 0xC0FFEE ^ (i as u64).wrapping_mul(2654435761)))
        .collect();

    let (obs_dim, critic_dim, act_dim) = (
        envs[0].obs_dim(),
        envs[0].critic_obs_dim(),
        envs[0].action_dim(),
    );
    println!("obs={obs_dim} critic_obs={critic_dim} action={act_dim}");

    let mut rng = Lcg::new(7);
    let mut ac = ActorCritic::new(
        &[obs_dim, 256, 128, act_dim],
        &[critic_dim, 256, 128, 1],
        0.5,
        5e-4,
        &mut rng,
    );
    // Fixed LR (the adaptive schedule ratchets down at these scales); obs
    // normalization is on by default inside ActorCritic.
    let cfg = PpoConfig {
        adaptive_lr: false,
        entropy_coef: 0.005,
        ..PpoConfig::default()
    };

    let mut cur: Vec<Vec<f32>> = Vec::with_capacity(num_envs);
    let mut cur_c: Vec<Vec<f32>> = Vec::with_capacity(num_envs);
    for e in envs.iter_mut() {
        let (o, c) = e.reset_full();
        cur.push(o);
        cur_c.push(c);
    }

    println!(
        "\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}  {:>8}  {:>8}",
        "iter", "curr", "step_rew", "falls", "torso_z", "cmd_spd", "fwd_spd"
    );
    // Command curriculum: stand-only at first, ramp to full velocity ranges over
    // the first 40% of training (learn to balance before walking).
    let warmup = (iters as f32 * 0.4).max(1.0);
    for it in 0..iters {
        let scale = (it as f32 / warmup).min(1.0);
        for e in envs.iter_mut() {
            e.set_command_scale(scale);
        }
        let s = ppo_iteration(&mut ac, &mut envs, &mut cur, &mut cur_c, &cfg, &mut rng, T);
        if it % 10 == 0 || it == iters - 1 {
            println!(
                "{:>4}  {:>5.2}  {:>9.4}  {:>7}  {:>8.3}  {:>8.3}  {:>8.3}",
                it, scale, s.mean_step_reward, s.falls, s.mean_torso_z, s.mean_cmd, s.mean_speed
            );
        }
    }
    println!("\n`fwd_spd` tracking `cmd_spd` (and not just standing) ⇒ it's walking.");
}
