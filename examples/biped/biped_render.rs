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

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let mut rng = Lcg::new(7);

    // 96 envs on 12 cores = 8 envs/core. Doubling parallelism over 48 doubles the
    // samples per iter at the same wall-clock per iter (rayon-bound critical path).
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
    // [256, 256, 128] — one more hidden layer than before, matching rsl_rl's
    // locomotion preset shape (they use [512, 256, 128]; we go smaller for CPU).
    let mut ac = ActorCritic::new(
        &[obs_dim, 256, 256, 128, act_dim],
        &[critic_dim, 256, 256, 128, 1],
        0.5,
        5e-4,
        &mut rng,
    );
    // rsl_rl defaults: adaptive LR (KL-targeted), 0.01 entropy bonus. Re-enabled
    // after they were explicitly disabled earlier this session for an unrelated
    // reason; both are essential for stable PPO at our scale.
    let cfg = PpoConfig::default();
    train(&mut ac, &mut envs, &cfg, &mut rng, train_iters);

    // Deterministic rollout on a fresh env, recording link positions per step.
    println!("recording {rollout_steps}-step deterministic rollout...");
    let mut env = BipedEnv::new(&xml, 12345);
    // Deterministic rollout for the demo: turn DR off (no random spawn yaw, no
    // action noise, no friction/PD/push perturbations) and pin the command.
    env.set_randomization(Randomization::off());
    env.pin_command(0.4, 0.0, 0.0);
    let (names, edges, feet) = env.skeleton();
    let mut obs = env.reset_full().0;
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(rollout_steps);
    let mut bases: Vec<[f32; 7]> = Vec::with_capacity(rollout_steps); // pos(3) + quat xyzw(4)
    let mut joints: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(rollout_steps);
    let mut resets: Vec<usize> = Vec::new();
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
    // qpos playback fields (for the MuJoCo mesh renderer).
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
    std::fs::write(&out, &s).expect("write json");

    println!(
        "wrote {} frames + skeleton → {out}\nrender: python3 examples/biped/render_biped.py {out} /tmp/biped.mp4",
        frames.len()
    );
}
