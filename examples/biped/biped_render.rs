//! Train briefly, then record a deterministic rollout of the LeRobot biped policy
//! to JSON (per-step link world positions + the kinematic tree) for rendering with
//! `examples/biped/render_biped.py`.
//!
//! The rollout uses the mean (noise-free) action and resets on falls, so the video
//! shows the controller's repeated balance attempts. Convergence is compute-bound
//! (see `biped_train`), so expect short attempts at modest train budgets.
//!
//! Run: `cargo run --release --example biped_render --features cpu -- [train_iters] [rollout_steps] [out.json]`

#[path = "biped_env.rs"]
mod biped_env;

use biped_env::{BipedEnv, Randomization, default_mjcf_path, ppo_iteration};
use std::fmt::Write as _;
use zealot_env::robots::lerobot_bipedal::JOINT_NAMES;
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig};

const NUM_JOINTS: usize = 12;
const T: usize = 32;

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

/// PPO training loop using the shared parallel iteration (see `biped_train`).
fn train(
    ac: &mut ActorCritic,
    envs: &mut [BipedEnv],
    cfg: &PpoConfig,
    rng: &mut Lcg,
    iters: usize,
) {
    let n = envs.len();
    let mut cur: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut cur_c: Vec<Vec<f32>> = Vec::with_capacity(n);
    for e in envs.iter_mut() {
        let (o, c) = e.reset_full();
        cur.push(o);
        cur_c.push(c);
    }
    let warmup = (iters as f32 * 0.4).max(1.0);
    for it in 0..iters {
        let scale = (it as f32 / warmup).min(1.0);
        for e in envs.iter_mut() {
            e.set_command_scale(scale);
        }
        let s = ppo_iteration(ac, envs, &mut cur, &mut cur_c, cfg, rng, T);
        if it % 25 == 0 || it == iters - 1 {
            println!(
                "  train iter {it:>4}  cmd={scale:.2}  step_rew={:.4}  falls={}  torso_z={:.3}  entropy={:.3}",
                s.mean_step_reward, s.falls, s.mean_torso_z, s.entropy
            );
        }
    }
}

fn main() {
    let train_iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(250);
    let rollout_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let out = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/biped_rollout.json".to_string());
    // 4th arg: number of starting poses to record (default 1 = backward compat).
    // Each pose pins the spawn joint offsets to a different perturbation; with
    // N>1 the JSON path gets a `_<i>` suffix per pose.
    let num_poses: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // 5th arg: policy binary. If `train_iters > 0`, train and save here. If
    // `train_iters == 0`, load from here and skip training (used for fast
    // multi-pose rollouts after a single training session).
    let policy_path = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "/tmp/biped_policy.safetensors".to_string());

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let mut rng = Lcg::new(7);

    let mut ac = if train_iters > 0 {
        // 96 envs on 12 cores = 8 envs/core. Doubling parallelism over 48 doubles
        // the samples per iter at the same wall-clock per iter.
        let num_envs = 96;
        println!("training {train_iters} iters ({num_envs} parallel envs)...");
        let mut envs: Vec<BipedEnv> = (0..num_envs)
            .map(|i| BipedEnv::new(&xml, 0xC0FFEE ^ (i as u64).wrapping_mul(2654435761)))
            .collect();
        let (obs_dim, critic_dim, act_dim) = (
            envs[0].obs_dim(),
            envs[0].critic_obs_dim(),
            envs[0].action_dim(),
        );
        // [256, 256, 128] — matches rsl_rl's locomotion preset shape.
        let mut ac = ActorCritic::new(
            &[obs_dim, 256, 256, 128, act_dim],
            &[critic_dim, 256, 256, 128, 1],
            0.5,
            5e-4,
            &mut rng,
        );
        let cfg = PpoConfig::default();
        train(&mut ac, &mut envs, &cfg, &mut rng, train_iters);
        ac.save(&policy_path).expect("save policy");
        println!("saved policy → {policy_path}");
        ac
    } else {
        println!("loading policy from {policy_path}...");
        ActorCritic::load(&policy_path).expect("load policy")
    };

    // Deterministic rollout on a fresh env, recording link positions per step.
    let mut env = BipedEnv::new(&xml, 12345);
    // Deterministic rollout for the demo: turn DR off (no random spawn yaw, no
    // action noise, no friction/PD/push perturbations) and pin the command.
    env.set_randomization(Randomization::off());
    env.pin_command(0.4, 0.0, 0.0);
    let (names, edges, feet) = env.skeleton();

    // Joint-perturbation poses to record from. Pose 0 is the canonical default
    // (zeros) for baseline; pose i>0 applies independent Gaussian noise per joint
    // with stddev = `0.05 * i` rad (so pose 1 ≈ 3°, pose 2 ≈ 6°, pose 3 ≈ 9°
    // perturbation per joint). Tests how far the policy can drift from default
    // before it fails to recover.
    for pose_idx in 0..num_poses.max(1) {
        let mut offsets = [0.0f32; NUM_JOINTS];
        if pose_idx > 0 {
            let mut prng = Lcg::new(31 + pose_idx as u64);
            let sigma = 0.05 * pose_idx as f32;
            for k in 0..NUM_JOINTS {
                offsets[k] = sigma * prng.gauss();
            }
        }
        env.pin_yaw(0.0);
        env.pin_joint_offsets(offsets);
        let mut obs = env.reset_full().0;
        let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(rollout_steps);
        let mut bases: Vec<[f32; 7]> = Vec::with_capacity(rollout_steps);
        let mut joints: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(rollout_steps);
        let mut resets: Vec<usize> = Vec::new();
        let pose_label = if pose_idx == 0 {
            "default (zeros)".to_string()
        } else {
            format!(
                "Gaussian σ={:.2} rad/joint  (max |offset| = {:.2} rad)",
                0.05 * pose_idx as f32,
                offsets.iter().fold(0.0_f32, |a, &b| a.max(b.abs()))
            )
        };
        println!("  pose {pose_idx}: {pose_label}");
        for step in 0..rollout_steps {
            frames.push(env.body_positions());
            let (p, q) = env.base_pose();
            bases.push([p[0], p[1], p[2], q[0], q[1], q[2], q[3]]);
            joints.push(env.joint_angles());
            let action = to_action(&ac.mean(&obs));
            let outp = env.step(&action);
            obs = outp.obs;
            if outp.done {
                resets.push(step);
                obs = env.reset_full().0;
            }
        }

        // Output path: `out` for single pose; `out_<i>.json` for multi-pose runs.
        let out_path = if num_poses <= 1 {
            out.clone()
        } else {
            let stem = out.trim_end_matches(".json");
            format!("{stem}_{pose_idx}.json")
        };

        // Manual JSON (no serde dependency).
        let mut s = String::new();
        s.push_str("{\n");
        let _ = write!(s, "  \"dt\": {:.4},\n", 0.02);
        let names_json: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
        let _ = write!(s, "  \"names\": [{}],\n", names_json.join(", "));
        let edges_json: Vec<String> = edges.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
        let _ = write!(s, "  \"edges\": [{}],\n", edges_json.join(", "));
        let feet_json: Vec<String> = feet.iter().map(|i| i.to_string()).collect();
        let _ = write!(s, "  \"feet\": [{}],\n", feet_json.join(", "));
        let resets_json: Vec<String> = resets.iter().map(|i| i.to_string()).collect();
        let _ = write!(s, "  \"resets\": [{}],\n", resets_json.join(", "));
        let jn: Vec<String> = JOINT_NAMES.iter().map(|n| format!("\"{n}\"")).collect();
        let _ = write!(s, "  \"joint_names\": [{}],\n", jn.join(", "));
        let base_json: Vec<String> = bases
            .iter()
            .map(|b| {
                format!(
                    "[{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}]",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6]
                )
            })
            .collect();
        let _ = write!(s, "  \"base\": [{}],\n", base_json.join(","));
        let joints_json: Vec<String> = joints
            .iter()
            .map(|j| {
                let v: Vec<String> = j.iter().map(|a| format!("{a:.5}")).collect();
                format!("[{}]", v.join(","))
            })
            .collect();
        let _ = write!(s, "  \"joints\": [{}],\n", joints_json.join(","));
        s.push_str("  \"frames\": [\n");
        for (fi, frame) in frames.iter().enumerate() {
            let pts: Vec<String> = frame
                .iter()
                .map(|p| format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]))
                .collect();
            let comma = if fi + 1 < frames.len() { "," } else { "" };
            let _ = write!(s, "    [{}]{}\n", pts.join(","), comma);
        }
        s.push_str("  ]\n}\n");
        std::fs::write(&out_path, &s).expect("write json");
        println!("    → {out_path}");
    }
}
