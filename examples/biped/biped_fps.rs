//! FPS benchmark — rapier CPU multibody vs nexus GPU physics for the LeRobot
//! biped. Steps both implementations for the same number of control ticks with
//! the same N envs, neutral actions, and the same control decimation, then
//! reports wall-clock throughput.
//!
//! Numbers reported:
//! - **env-ctrl-steps/s** — N · control_steps / elapsed. This is the figure
//!   the trainer actually sees (one observation/action per env per tick).
//! - **sim-steps/s** — env-ctrl-steps/s · decimation. The underlying physics
//!   sub-tick throughput (4× the control rate by `VelocityFlatTask` default).
//!
//! Both envs do the same per-step work in spirit (PD targets → physics step
//! ×decimation → readback → obs/reward), so the ratio reflects the engine
//! difference, not policy / reward overhead.
//!
//! Run: `cargo run --release --example biped_fps --features "cpu biped_gpu" -- [num_envs] [control_steps]`

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env::{BipedEnv, default_mjcf_path};
use biped_env_nexus::{BipedNexusBatchEnv, StepTimings};
use rayon::prelude::*;
use std::time::Instant;
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;

// Match VelocityFlatTask::new() — sim_dt = 1/200, decimation = 4 → 50 Hz control.
const DECIMATION: f64 = 4.0;

fn fmt_rate(x: f64) -> String {
    if x >= 1.0e6 {
        format!("{:.2} M", x / 1.0e6)
    } else if x >= 1.0e3 {
        format!("{:.1} k", x / 1.0e3)
    } else {
        format!("{x:.0}")
    }
}

fn main() {
    let num_envs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let control_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let warmup: usize = 10;

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let zero_action = [0.0f32; NUM_JOINTS];

    println!(
        "biped FPS benchmark — {num_envs} envs, {control_steps} control steps \
         (decimation {DECIMATION}, warmup {warmup})\n"
    );

    // ------------------------------------------------------------------
    // CPU: independent rapier multibody worlds, stepped in parallel via rayon.
    // ------------------------------------------------------------------
    let build_t0 = Instant::now();
    let mut cpu_envs: Vec<BipedEnv> = (0..num_envs)
        .map(|i| BipedEnv::new(&xml, 0xC0FFEE ^ (i as u64).wrapping_mul(2654435761)))
        .collect();
    for e in cpu_envs.iter_mut() {
        e.reset_full();
    }
    let cpu_build = build_t0.elapsed().as_secs_f64();

    for _ in 0..warmup {
        let _outs: Vec<_> = cpu_envs
            .par_iter_mut()
            .map(|e| e.step(&zero_action))
            .collect();
    }
    let t0 = Instant::now();
    for _ in 0..control_steps {
        let _outs: Vec<_> = cpu_envs
            .par_iter_mut()
            .map(|e| e.step(&zero_action))
            .collect();
    }
    let cpu_elapsed = t0.elapsed().as_secs_f64();
    let cpu_env_steps = (num_envs * control_steps) as f64;
    let cpu_ctrl_fps = cpu_env_steps / cpu_elapsed;
    let cpu_sim_fps = cpu_ctrl_fps * DECIMATION;

    println!(
        "CPU  (rapier, rayon)  build {cpu_build:>5.2}s  step {cpu_elapsed:>6.2}s  \
         → {:>8} env-ctrl/s   ({:>8} sim/s)",
        fmt_rate(cpu_ctrl_fps),
        fmt_rate(cpu_sim_fps),
    );

    // ------------------------------------------------------------------
    // GPU: one batched RbdState holding all N envs, one dispatch/step.
    // ------------------------------------------------------------------
    pollster::block_on(async {
        let build_t0 = Instant::now();
        // 4 templates is the minimum that still gives some DR diversity; more
        // templates only affect reset cost, not steady-state step throughput.
        let mut gpu_env = BipedNexusBatchEnv::new(&xml, num_envs, 4, 0xC0FFEE).await;
        let gpu_build = build_t0.elapsed().as_secs_f64();

        let actions: Vec<[f32; NUM_JOINTS]> = vec![zero_action; num_envs];
        for _ in 0..warmup {
            let _ = gpu_env.step(&actions).await;
        }
        // Reset the per-phase timing counters AFTER warmup so we measure
        // steady-state only (the first few steps include pipeline warmup,
        // buffer-pool warmup, contact buffer growth, etc.).
        let _ = gpu_env.take_step_timings();

        let t0 = Instant::now();
        for _ in 0..control_steps {
            let _ = gpu_env.step(&actions).await;
        }
        let gpu_elapsed = t0.elapsed().as_secs_f64();
        let gpu_env_steps = (num_envs * control_steps) as f64;
        let gpu_ctrl_fps = gpu_env_steps / gpu_elapsed;
        let gpu_sim_fps = gpu_ctrl_fps * DECIMATION;
        let timings = gpu_env.take_step_timings();

        println!(
            "GPU  (nexus, batched) build {gpu_build:>5.2}s  step {gpu_elapsed:>6.2}s  \
             → {:>8} env-ctrl/s   ({:>8} sim/s)",
            fmt_rate(gpu_ctrl_fps),
            fmt_rate(gpu_sim_fps),
        );

        // Per-phase breakdown (averaged over the timed window). Helps
        // identify the actual per-step bottleneck instead of guessing.
        print_timing_breakdown(&timings, gpu_elapsed);

        // Stability probe: under zero action the biped PD-holds its default
        // standing pose, so after the run torso heights should cluster near the
        // spawn height. A solver run with too few substeps destabilises contacts
        // → collapsed/exploded torsos (height out of [0.3,1.0]) or NaN. Report
        // the spread so a fidelity/speed tradeoff isn't accepted blind.
        let h = gpu_env.torso_heights().await;
        let (mut nan, mut blow, mut lo, mut hi, mut sum) =
            (0usize, 0usize, f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &z in &h {
            if !z.is_finite() {
                nan += 1;
                continue;
            }
            if z.abs() > 5.0 {
                blow += 1;
            } // exploded out of the arena
            lo = lo.min(z);
            hi = hi.max(z);
            sum += z as f64;
        }
        let ok = h.len() - nan;
        println!(
            "  stability ({} envs): mean z {:.3}  range [{:.3}, {:.3}]  blowup(|z|>5) {}  NaN {}",
            h.len(),
            sum / ok.max(1) as f64,
            lo,
            hi,
            blow,
            nan,
        );

        let ratio = gpu_ctrl_fps / cpu_ctrl_fps;
        let tag = if ratio >= 1.0 {
            format!("nexus is {ratio:.2}× faster than rapier")
        } else {
            format!("rapier is {:.2}× faster than nexus", 1.0 / ratio)
        };
        println!("\nspeedup (steady-state, per-env): {tag}");
        println!(
            "(realtime factor: CPU {:.1}×, GPU {:.1}×; one env / one ctrl step = 0.02s sim)",
            cpu_ctrl_fps * 0.02 / num_envs as f64,
            gpu_ctrl_fps * 0.02 / num_envs as f64,
        );
    });
}

/// Pretty-print the per-phase nanosecond accumulators as mean-µs-per-step
/// + share-of-total. The sum of phases may not exactly equal the outer wall
/// time (rayon worker contention, sub-step kernel launch overlap with host,
/// etc.), but it's close enough to point at the dominant cost.
fn print_timing_breakdown(t: &StepTimings, wall_elapsed_s: f64) {
    if t.steps == 0 {
        return;
    }
    let n = t.steps as f64;
    let wall_us = wall_elapsed_s * 1e6;
    let mean_step_us = wall_us / n;
    let total_ns = t.total_ns();
    println!(
        "  per-step breakdown ({} steps, mean wall {:.2} ms/step):",
        t.steps,
        mean_step_us / 1000.0
    );
    let row = |label: &str, ns: u64| {
        let mean_ms = ns as f64 / n / 1_000_000.0;
        let pct = if total_ns > 0 {
            100.0 * ns as f64 / total_ns as f64
        } else {
            0.0
        };
        println!("    {:<24} {:>8.3} ms  ({:>5.1}%)", label, mean_ms, pct);
    };
    row("stage_motors", t.stage_motors_ns);
    row("flush_links_static", t.flush_static_ns);
    row("pipeline.step (encode)", t.pipeline_step_ns);
    row("gpu.synchronize (compute)", t.gpu_wait_ns);
    row("auto_resize_buffers", t.auto_resize_ns);
    row("slurp_poses (transfer)", t.readback_ns);
    row("serial pre (rng/cmd)", t.serial_pre_ns);
    row("par obs+reward (rayon)", t.par_compute_ns);
    row("serial commit", t.serial_commit_ns);
    let accounted_ms = total_ns as f64 / n / 1_000_000.0;
    let unaccounted_ms = mean_step_us / 1000.0 - accounted_ms;
    println!(
        "    {:<24} {:>8.3} ms  (wall − Σphases; rayon / async overhead)",
        "unaccounted",
        unaccounted_ms.max(0.0)
    );
}
