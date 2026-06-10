//! End-to-end rollout throughput: FULL CPU vs FULL GPU.
//!   full CPU = rapier multibody physics (rayon) + CPU MLP policy (serial)
//!   full GPU = nexus batched GPU physics + vortx GPU policy
//! One control step = policy forward (produce an action per env) + physics step.
//! Reports env-control-steps/second, same unit as `biped_fps`. No PPO update.
//!
//! Run: `cargo run --release --example rollout_e2e_bench --features "gpu biped_gpu" -- [num_envs] [steps]`

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env::BipedEnv;
use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use gpu_policy::GpuPolicy;
use rayon::prelude::*;
use std::time::Instant;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::rng::Lcg;
use zealot_rl::ActorCritic;

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

/// One full-CPU control step: serial CPU policy forward + rayon rapier physics.
fn cpu_step(
    ac: &mut ActorCritic,
    envs: &mut [BipedEnv],
    cur: &mut [Vec<f32>],
    cur_c: &mut [Vec<f32>],
    rng: &mut Lcg,
) {
    let n = envs.len();
    let mut actions = Vec::with_capacity(n);
    for e in 0..n {
        ac.record_obs(&cur[e], &cur_c[e]);
        let (a, _, _) = ac.sample(&cur[e], rng);
        let _ = ac.value(&cur_c[e]);
        actions.push(to_action(&a));
    }
    let outs: Vec<_> = envs
        .par_iter_mut()
        .enumerate()
        .map(|(e, env)| env.step(&actions[e]))
        .collect();
    for e in 0..n {
        cur[e].clone_from(&outs[e].obs);
        cur_c[e].clone_from(&outs[e].critic_obs);
    }
}

/// One full-GPU control step: batched vortx policy forward + nexus GPU physics.
async fn gpu_step(
    ac: &mut ActorCritic,
    env: &mut BipedNexusBatchEnv,
    gpu: &mut GpuPolicy,
    cur: &mut [Vec<f32>],
    cur_c: &mut [Vec<f32>],
    rng: &mut Lcg,
) {
    let n = cur.len();
    let tf = std::time::Instant::now();
    for e in 0..n {
        ac.record_obs(&cur[e], &cur_c[e]);
    }
    let (means, _) = gpu.forward(env.backend(), ac, cur, cur_c).await.unwrap();
    T_FWD.fetch_add(tf.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let ts = std::time::Instant::now();
    let mut actions = Vec::with_capacity(n);
    for e in 0..n {
        let mean = &means[e];
        let mut a = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            a[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
        }
        actions.push(a);
    }
    T_SAMPLE.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let tp = std::time::Instant::now();
    let outs = env.step(&actions).await;
    for e in 0..n {
        cur[e].clone_from(&outs[e].obs);
        cur_c[e].clone_from(&outs[e].critic_obs);
    }
    T_STEP.fetch_add(tp.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

static T_FWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static T_SAMPLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static T_STEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn main() {
    let num_envs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let n = num_envs;
    let mut rng = Lcg::new(7);

    println!("building {n} CPU rapier envs...");
    let mut cpu_envs: Vec<BipedEnv> = (0..n).map(|e| BipedEnv::new(&xml, e as u64)).collect();
    let mut cur: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut cur_c: Vec<Vec<f32>> = Vec::with_capacity(n);
    for env in cpu_envs.iter_mut() {
        let (o, c) = env.reset_full();
        cur.push(o);
        cur_c.push(c);
    }
    let (obs_dim, critic_dim) = (cpu_envs[0].obs_dim(), cpu_envs[0].critic_obs_dim());
    let mut ac = ActorCritic::new(
        &[obs_dim, 256, 256, 128, NUM_JOINTS],
        &[critic_dim, 512, 256, 128, 1],
        1.0,
        1e-3,
        &mut rng,
    );

    // ---------- FULL CPU ----------
    for _ in 0..2 {
        cpu_step(&mut ac, &mut cpu_envs, &mut cur, &mut cur_c, &mut rng);
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        cpu_step(&mut ac, &mut cpu_envs, &mut cur, &mut cur_c, &mut rng);
    }
    let cpu = t0.elapsed();
    drop(cpu_envs);

    // ---------- FULL GPU ----------
    let gpu = pollster::block_on(async {
        println!("building {n} GPU nexus envs...");
        let mut env = BipedNexusBatchEnv::new(&xml, n, 4, 0xC0FFEE).await;
        let mut gpu = GpuPolicy::new(env.backend(), &ac, n).expect("gpu policy");
        let (mut gcur, mut gcur_c) = env.initial_obs().await;
        for _ in 0..2 {
            gpu_step(&mut ac, &mut env, &mut gpu, &mut gcur, &mut gcur_c, &mut rng).await;
        }
        use std::sync::atomic::Ordering::Relaxed;
        T_FWD.store(0, Relaxed);
        T_SAMPLE.store(0, Relaxed);
        T_STEP.store(0, Relaxed);
        let t1 = Instant::now();
        for _ in 0..steps {
            gpu_step(&mut ac, &mut env, &mut gpu, &mut gcur, &mut gcur_c, &mut rng).await;
        }
        let el = t1.elapsed();
        let ps = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed) as f64 / steps as f64 / 1e6;
        println!(
            "  [rollout split] forward {:.2} ms  sample {:.2} ms  env.step {:.2} ms",
            ps(&T_FWD),
            ps(&T_SAMPLE),
            ps(&T_STEP)
        );
        el
    });

    let cpu_ms = cpu.as_secs_f64() / steps as f64 * 1e3;
    let gpu_ms = gpu.as_secs_f64() / steps as f64 * 1e3;
    let cpu_eps = n as f64 / (cpu_ms / 1e3) / 1e3;
    let gpu_eps = n as f64 / (gpu_ms / 1e3) / 1e3;
    println!("\nend-to-end rollout (policy + physics) — {n} envs, {steps} steps");
    println!("  FULL CPU (rapier + CPU MLP) : {cpu_ms:8.2} ms/step  = {cpu_eps:6.1} k env/s");
    println!("  FULL GPU (nexus + vortx)    : {gpu_ms:8.2} ms/step  = {gpu_eps:6.1} k env/s");
    println!("  speedup                     : {:.2}x", cpu_ms / gpu_ms);
}
