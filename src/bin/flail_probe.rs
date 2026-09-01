//! Flail probe: drive every env with an IDENTICAL seeded random action
//! sequence (~N(0,1), the distribution an untrained policy emits) and record
//! the joint-velocity statistics the power/vel fault detectors consume.
//! No policy, no resets, faults expected OFF via env vars. The same file is
//! dropped into both engine trees, so any distribution difference between the
//! binaries is pure solver behavior.
//!
//!   flail_probe [num_envs=512] [steps=150] [scale=1.0]
#[path = "../biped/biped_env.rs"]
mod biped_env;
#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "../biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../biped/gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use zealot_env::robots::NUM_JOINTS;
use zealot_rl::rng::Lcg;

/// PD torque estimate from the obs slots, same formula as the power fault:
/// tau = clamp(kp*(target - q) - kd*qd, +-effort). Ankle deploy override 40/2.
fn tau_table() -> (
    [f32; NUM_JOINTS],
    [f32; NUM_JOINTS],
    [f32; NUM_JOINTS],
    [f32; NUM_JOINTS],
    [f32; NUM_JOINTS],
) {
    let spec = zealot_env::robots::unitree_g1::unitree_g1_29dof_agile();
    let mut kp = [0.0; NUM_JOINTS];
    let mut kd = [0.0; NUM_JOINTS];
    let mut eff = [0.0; NUM_JOINTS];
    let mut scale = [0.0; NUM_JOINTS];
    let mut dpos = [0.0; NUM_JOINTS];
    for k in 0..NUM_JOINTS {
        let j = &spec.joints[k];
        let ankle = j.name.contains("ankle");
        kp[k] = if ankle { 40.0 } else { j.kp };
        kd[k] = if ankle { 2.0 } else { j.kd };
        eff[k] = j.effort_limit;
        scale[k] = j.action_scale;
        dpos[k] = j.default_pos;
    }
    (kp, kd, eff, scale, dpos)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let scale: f32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let mut rng = Lcg::new(1234);
        // Percentile accumulators over all (env, step) samples, steps >= 3.
        let mut qd_abs: Vec<f32> = Vec::new();
        let mut per_joint_max = [0.0f32; NUM_JOINTS];
        let (kp, kd, eff, ascale, _dpos) = tau_table();
        let mut pw: Vec<f32> = Vec::new(); // per-sample |tau*qd|
        // torque conditioned on velocity bands
        let mut band_sum = [0.0f64; 4];
        let mut band_n = [0u64; 4];
        let mut ep_lens: Vec<u32> = Vec::new();
        let mut steps_since = vec![0u32; n];
        let mut ssr_sum = [0.0f64; 6];
        let mut ssr_n = [0u64; 6];
        let mut ssr_max = [0.0f32; 6];
        let (mut g_up_sum, mut g_up_n, mut g_up_bad) = (0.0f64, 0u64, 0u64);
        // FLAIL_STEP_TEST=1: single sustained step-input on the left knee at
        // t=50 with zero action elsewhere — the response curve exposes the
        // delay implementation's true applied-target timing.
        let step_test = std::env::var("FLAIL_STEP_TEST").is_ok();
        for t in 0..steps {
            let mut acts: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut a = [0.0f32; NUM_JOINTS];
                if step_test {
                    if t >= 50 {
                        for k in 0..NUM_JOINTS {
                            a[k] = 1.5; // flail-scale sustained step, all joints
                        }
                    }
                } else {
                    for k in 0..NUM_JOINTS {
                        a[k] = scale * rng.gauss();
                    }
                }
                acts.push(a);
            }
            let outs = env.step(&acts).await;
            if step_test && (46..60).contains(&t) {
                let o0 = &outs[0].obs;
                if o0.len() >= 40 {
                    println!(
                        "STEPTEST t={t} kneeQ {:+.4} kneeQd {:+.2}",
                        o0[16 + 3],
                        o0[28 + 3]
                    );
                }
            }
            // FLAIL_RESET=1: reset tripped envs like the trainer does, and
            // histogram episode lengths — separates steady-state spikes from
            // reset-transient re-trip cascades.
            if std::env::var("FLAIL_RESET").is_ok() {
                for e in 0..n {
                    if outs[e].done {
                        ep_lens.push(steps_since[e]);
                        steps_since[e] = 0;
                        let _ = env.reset_env(e).await;
                    } else {
                        steps_since[e] += 1;
                    }
                }
            }
            if t < 3 {
                continue; // let finite-diff velocities become defined
            }
            for e in 0..n {
                let o = &outs[e].obs;
                if o.len() < 40 {
                    continue;
                }
                // obs slots 28..40 = joint_vel (identical layout both trees)
                for k in 0..NUM_JOINTS {
                    let v = o[28 + k];
                    let va = v.abs();
                    qd_abs.push(va);
                    if va > per_joint_max[k] {
                        per_joint_max[k] = va;
                    }
                    // obs: last_action slot k, joint_pos_rel slot 16+k (q - default)
                    let qrel = o[16 + k];
                    let err = ascale[k] * o[k] - qrel; // (default+scale*a) - q
                    let tau = (kp[k] * err - kd[k] * v).clamp(-eff[k], eff[k]);
                    pw.push((tau * v).abs());
                    let b = if va < 10.0 {
                        0
                    } else if va < 15.0 {
                        1
                    } else if va < 20.0 {
                        2
                    } else {
                        3
                    };
                    band_sum[b] += tau.abs() as f64;
                    band_n[b] += 1;
                    // upright check 1 step after reset: obs[42] = projected
                    // gravity up-component (-1 standing, ~0 lying down)
                    if k == 0 && steps_since[e] == 1 {
                        g_up_sum += o[42] as f64;
                        g_up_n += 1;
                        if o[42] > -0.5 {
                            g_up_bad += 1;
                        }
                    }
                    // post-reset transient: velocity by steps-since-reset
                    let ssr = steps_since[e].min(5) as usize;
                    ssr_sum[ssr] += va as f64;
                    ssr_n[ssr] += 1;
                    if va > ssr_max[ssr] {
                        ssr_max[ssr] = va;
                    }
                }
            }
        }
        qd_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| qd_abs[((qd_abs.len() as f64 - 1.0) * p) as usize];
        println!(
            "FLAIL n={n} steps={steps} scale={scale} samples={}",
            qd_abs.len()
        );
        println!(
            "|qd| rad/s: p50={:.2} p90={:.2} p99={:.2} p99.9={:.2} max={:.2}",
            pct(0.50),
            pct(0.90),
            pct(0.99),
            pct(0.999),
            qd_abs[qd_abs.len() - 1]
        );
        let over = |thr: f32| {
            qd_abs.iter().filter(|v| **v > thr).count() as f64 / qd_abs.len() as f64 * 100.0
        };
        println!(
            "tail fractions: >10={:.3}% >20={:.3}% >30={:.4}% >50={:.4}%",
            over(10.0),
            over(20.0),
            over(30.0),
            over(50.0)
        );
        println!(
            "per-joint max: {:?}",
            per_joint_max.map(|v| (v * 10.0).round() / 10.0)
        );
        pw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ppct = |p: f64| pw[((pw.len() as f64 - 1.0) * p) as usize];
        println!(
            "|tau*qd| W: p90={:.0} p99={:.0} p99.9={:.0} max={:.0}  frac>1500W={:.3}% >3000W={:.4}%",
            ppct(0.90),
            ppct(0.99),
            ppct(0.999),
            pw[pw.len() - 1],
            pw.iter().filter(|v| **v > 1500.0).count() as f64 / pw.len() as f64 * 100.0,
            pw.iter().filter(|v| **v > 3000.0).count() as f64 / pw.len() as f64 * 100.0
        );
        for i in 0..6 {
            if ssr_n[i] > 0 {
                println!(
                    "  steps-since-reset {}{}: mean|qd| {:.2}  max {:.1}  (n={})",
                    i,
                    if i == 5 { "+" } else { "" },
                    ssr_sum[i] / ssr_n[i] as f64,
                    ssr_max[i],
                    ssr_n[i]
                );
            }
        }
        if g_up_n > 0 {
            println!(
                "POST-RESET UPRIGHT: mean grav-up {:.3} ({} samples), NOT-upright(>-0.5): {:.1}%",
                g_up_sum / g_up_n as f64,
                g_up_n,
                g_up_bad as f64 / g_up_n as f64 * 100.0
            );
        }
        if !ep_lens.is_empty() {
            ep_lens.sort();
            let e = &ep_lens;
            let q = |p: f64| e[((e.len() as f64 - 1.0) * p) as usize];
            let frac_short = e.iter().filter(|v| **v <= 3).count() as f64 / e.len() as f64 * 100.0;
            println!(
                "EPISODES: n={} p10={} p50={} p90={}  trips<=3-steps-after-reset: {:.1}%",
                e.len(),
                q(0.10),
                q(0.50),
                q(0.90),
                frac_short
            );
        }
        for (i, name) in ["<10", "10-15", "15-20", ">20"].iter().enumerate() {
            if band_n[i] > 0 {
                println!(
                    "  mean|tau| @ |qd| {name} rad/s: {:.1} N.m (n={})",
                    band_sum[i] / band_n[i] as f64,
                    band_n[i]
                );
            }
        }
    });
}
