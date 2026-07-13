//! Real rollout-forward benchmark: CPU per-env path vs GPU batched path, on the
//! actual biped nexus env, INCLUDING the per-step GPU→CPU readback the GPU path
//! pays. This is the honest "does the GPU policy help the rollout?" number —
//! unlike `policy_forward_bench` (isolated resident compute), this is what
//! `biped_render_nexus` actually does per control step.
//!
//! Run: `cargo run --release --example rollout_bench --features "gpu biped_gpu" -- [num_envs] [steps]`
//!
//! Both paths produce action + logp + value for every env each step; the only
//! difference is CPU serial `ac.sample`/`ac.value` vs `GpuPolicy::forward`
//! (batched GEMM + readback) followed by cheap CPU sampling. Physics is excluded
//! (identical for both; forward cost is obs-value-independent so we reuse obs).

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "cutile_gemm.rs"]
mod cutile_gemm;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use gpu_policy::GpuPolicy;
use std::time::Instant;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::ActorCritic;
use zealot_rl::rng::Lcg;

fn main() {
    let num_envs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let mut rng = Lcg::new(7);

    pollster::block_on(async {
        println!("building {num_envs} envs...");
        let mut env = BipedNexusBatchEnv::new(&xml, num_envs, 32, 0xC0FFEE).await;
        let (obs_dim, critic_dim, act_dim) =
            (env.obs_dim(), env.critic_obs_dim(), env.action_dim());
        let mut ac = ActorCritic::new(
            &[obs_dim, 256, 256, 128, act_dim],
            &[critic_dim, 512, 256, 128, 1],
            1.0,
            1e-3,
            &mut rng,
        );
        let mut gpu = GpuPolicy::new(env.backend(), &ac, num_envs).expect("gpu policy");
        let (cur, cur_c) = env.initial_obs().await;
        let n = cur.len();

        // ---- CPU path: serial per-env sample + value ----
        for _ in 0..2 {
            for e in 0..n {
                ac.record_obs(&cur[e], &cur_c[e]);
                let _ = ac.sample(&cur[e], &mut rng);
                let _ = ac.value(&cur_c[e]);
            }
        }
        let t0 = Instant::now();
        for _ in 0..steps {
            for e in 0..n {
                ac.record_obs(&cur[e], &cur_c[e]);
                let (_a, _lp, _m) = ac.sample(&cur[e], &mut rng);
                let _v = ac.value(&cur_c[e]);
            }
        }
        let cpu = t0.elapsed();

        // ---- GPU path: batched forward (+ readback) then cheap CPU sampling ----
        let _ = gpu.forward(env.backend(), &ac, &cur, &cur_c).await.unwrap(); // warmup
        let t1 = Instant::now();
        for _ in 0..steps {
            for e in 0..n {
                ac.record_obs(&cur[e], &cur_c[e]);
            }
            let (means, _values) = gpu.forward(env.backend(), &ac, &cur, &cur_c).await.unwrap();
            for e in 0..n {
                let mean = &means[e];
                let mut action = [0.0f32; NUM_JOINTS];
                for k in 0..NUM_JOINTS {
                    action[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
                }
                let _lp = ac.logp(&action, mean);
            }
        }
        let gpu_t = t1.elapsed();

        let cpu_ms = cpu.as_secs_f64() / steps as f64 * 1e3;
        let gpu_ms = gpu_t.as_secs_f64() / steps as f64 * 1e3;
        println!("\nrollout forward — {num_envs} envs, {steps} steps (physics excluded)");
        println!("  CPU serial sample+value : {cpu_ms:8.3} ms/step");
        println!("  GPU batched + readback  : {gpu_ms:8.3} ms/step");
        if gpu_ms < cpu_ms {
            println!(
                "  -> GPU {:.2}x faster end-to-end (readback included)",
                cpu_ms / gpu_ms
            );
        } else {
            println!(
                "  -> GPU {:.2}x SLOWER end-to-end (readback dominates)",
                gpu_ms / cpu_ms
            );
        }
    });
}
