//! Train briefly on **nexus GPU physics**, then record a deterministic rollout
//! of env 0 to JSON for rendering with `examples/biped/render_biped.py` (or the
//! MuJoCo mesh renderer). Same output format as `biped_render.rs` — same python
//! script reads both.
//!
//! Run:
//!   `cargo run --release --example biped_render_nexus --features biped_gpu -- \
//!         [train_iters] [rollout_steps] [out.json]`
//! then:
//!   `python3 examples/biped/render_biped.py /tmp/biped_rollout_nexus.json /tmp/biped_nexus.mp4`

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, StepOut, default_mjcf_path};
use std::fmt::Write as _;
use zealot_env::robots::lerobot_bipedal::{JOINT_NAMES, NUM_JOINTS};
use zealot_rl::ppo::{Sample, gae};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig};

const T: usize = 32;

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

/// Train on the batched nexus env for `iters` PPO iterations. Writes a
/// checkpoint to `checkpoint_path` every `checkpoint_every` iters (and once
/// more at the end) so a killed run leaves a resumable state. Set
/// `checkpoint_every = 0` to disable mid-training saves.
async fn train(
    ac: &mut ActorCritic,
    env: &mut BipedNexusBatchEnv,
    cfg: &PpoConfig,
    rng: &mut Lcg,
    iters: usize,
    checkpoint_path: &str,
    checkpoint_every: usize,
) {
    let n = env.obs_dim();
    let _ = n;
    let num_envs: usize = env.obs_dim(); // placeholder — we re-read below

    // num_envs from the env itself.
    let num_envs = env.action_dim() / NUM_JOINTS; // == 1; unused, replaced
    let _ = num_envs;

    // Initial obs.
    let (mut cur, mut cur_c) = env.initial_obs().await;
    let n = cur.len();

    let warmup = (iters as f32 * 0.4).max(1.0);
    for it in 0..iters {
        let scale = (it as f32 / warmup).min(1.0);
        env.set_command_scale(scale);

        let mut samples: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut rs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut vs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut ds: Vec<Vec<bool>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let (mut total_reward, mut falls) = (0.0f32, 0u32);

        for _ in 0..T {
            let mut actions: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(n);
            for e in 0..n {
                ac.record_obs(&cur[e], &cur_c[e]);
                let (action, logp, mean) = ac.sample(&cur[e], rng);
                let value = ac.value(&cur_c[e]);
                actions.push(to_action(&action));
                samples[e].push(Sample {
                    obs: cur[e].clone(),
                    critic_obs: cur_c[e].clone(),
                    action,
                    mean_old: mean,
                    logp_old: logp,
                    value_old: value,
                    adv: 0.0,
                    ret: 0.0,
                });
                vs[e].push(value);
            }
            let outs: Vec<StepOut> = env.step(&actions).await;
            for e in 0..n {
                let out = &outs[e];
                total_reward += out.reward;
                rs[e].push(out.reward);
                ds[e].push(out.done);
                if out.fell {
                    falls += 1;
                }
                if out.done {
                    let (o, c) = env.reset_env(e).await;
                    cur[e] = o;
                    cur_c[e] = c;
                } else {
                    cur[e].clone_from(&out.obs);
                    cur_c[e].clone_from(&out.critic_obs);
                }
            }
        }

        let mut batch: Vec<Sample> = Vec::with_capacity(n * T);
        for e in 0..n {
            let last_v = ac.value(&cur_c[e]);
            let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], last_v, cfg.gamma, cfg.lam);
            for t in 0..T {
                samples[e][t].adv = adv[t];
                samples[e][t].ret = ret[t];
                batch.push(std::mem::take(&mut samples[e][t]));
            }
        }
        let _stats = ac.update(&mut batch, cfg);

        if it % 10 == 0 || it == iters - 1 {
            let steps = (n * T) as f32;
            println!(
                "iter {it:>4}  scale {scale:>4.2}  step_rew {:>8.4}  falls {falls:>5}",
                total_reward / steps
            );
        }
        // Periodic checkpoint. Writes the FULL policy via safetensors —
        // weights, log_std, both Normalizer states. Atomic enough for our
        // purposes (single fs::write call). Skip iter 0 so we don't overwrite
        // a resumed checkpoint with the un-trained starting state.
        if checkpoint_every > 0 && it > 0 && (it % checkpoint_every == 0 || it == iters - 1) {
            if let Err(e) = ac.save(checkpoint_path) {
                eprintln!("warning: checkpoint save failed at iter {it}: {e}");
            } else {
                println!("  checkpoint → {checkpoint_path}");
            }
        }
    }
}

fn main() {
    let train_iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let rollout_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let out = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/biped_rollout_nexus.json".to_string());
    // 4th arg: policy safetensors path. If `train_iters > 0`, train and save
    // here. If `train_iters == 0`, load and skip training (fast re-rollouts).
    let policy_path = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "/tmp/biped_policy_nexus.safetensors".to_string());

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    let mut rng = Lcg::new(7);

    pollster::block_on(async {
        // After the bulk-motor + single-readback wins, 32 envs is the sweet
        // spot — N×12 motor updates collapse into one buffer write, so the
        // marginal cost per env at this scale is the per-link workspace
        // readback, ~constant. 32 templates give enough initial-pose variety
        // (yaw + roll/pitch + height noise) that PPO actually explores.
        let num_envs = 32;
        let num_templates = 32;
        println!("building {num_envs} envs on nexus...");
        let mut env = BipedNexusBatchEnv::new(&xml, num_envs, num_templates, 0xC0FFEE).await;

        let (obs_dim, critic_dim, act_dim) =
            (env.obs_dim(), env.critic_obs_dim(), env.action_dim());
        let mut ac = if train_iters > 0 {
            // Auto-resume: if a checkpoint already exists at `policy_path`,
            // pick up from there. Otherwise build a fresh net.
            let mut ac = if std::path::Path::new(&policy_path).exists() {
                println!("resuming from existing checkpoint {policy_path}...");
                ActorCritic::load(&policy_path).expect("load checkpoint")
            } else {
                // Wider net + extra hidden — closer to the deployed rsl_rl preset.
                ActorCritic::new(
                    &[obs_dim, 512, 256, 128, act_dim],
                    &[critic_dim, 512, 256, 128, 1],
                    0.5,
                    5e-4,
                    &mut rng,
                )
            };
            // rsl_rl-style: adaptive-KL LR, real entropy bonus.
            let cfg = PpoConfig::default();
            println!("training for {train_iters} iters on nexus GPU...");
            train(
                &mut ac,
                &mut env,
                &cfg,
                &mut rng,
                train_iters,
                &policy_path,
                50, // checkpoint every 50 iters
            )
            .await;
            // Final write — `train` writes one at iters-1 too, this is the
            // belt-and-braces.
            ac.save(&policy_path).expect("save policy");
            println!("saved final policy → {policy_path}");
            ac
        } else {
            println!("loading policy from {policy_path}...");
            ActorCritic::load(&policy_path).expect("load policy")
        };
        // Suppress the now-unused `act_dim` / `obs_dim` warning when train_iters=0.
        let _ = (obs_dim, critic_dim, act_dim);

        // Recording rollout: reset env 0 to the DR-OFF template + pin command
        // forward, then step deterministically (mean action) and record state.
        println!("recording {rollout_steps}-step deterministic rollout from env 0...");
        let _ = env.reset_env_to_default_template(0).await;
        env.pin_command_for(0, 0.4, 0.0, 0.0);
        let (names, edges, feet) = env.skeleton();

        let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(rollout_steps);
        let mut bases: Vec<[f32; 7]> = Vec::with_capacity(rollout_steps);
        let mut joints: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(rollout_steps);
        let mut resets: Vec<usize> = Vec::new();

        // Build initial obs from a fresh snapshot so the first action sees the
        // post-reset state (matching `BipedEnv::reset_full + step` pattern).
        let (mut cur, _) = env.initial_obs().await;
        // Pin only env 0's command — other envs idle along but we don't read them.

        for step in 0..rollout_steps {
            // Snapshot BEFORE stepping so we record the current pose, then act.
            // Both body positions and joint angles come from `body_poses` now
            // — `joint_angles_for` derives them via parent⇄child relative
            // rotation (the heavy `links_workspace` readback was removed when
            // the step path was switched to the same poses-only path).
            let poses = env.snapshot().await;
            frames.push(env.body_positions_for(0, &poses));
            let (p, q) = env.base_pose_for(0, &poses);
            bases.push([p[0], p[1], p[2], q[0], q[1], q[2], q[3]]);
            joints.push(env.joint_angles_for(0, &poses));

            // Mean (noise-free) action for env 0.
            let mean = ac.mean(&cur[0]);
            let mut actions: Vec<[f32; NUM_JOINTS]> = vec![[0.0; NUM_JOINTS]; num_envs];
            actions[0] = to_action(&mean);
            // Other envs: just hold zero (we don't render them).
            let outs = env.step(&actions).await;

            if outs[0].done {
                resets.push(step);
                let (o, _) = env.reset_env_to_default_template(0).await;
                cur[0] = o;
                env.pin_command_for(0, 0.4, 0.0, 0.0);
            } else {
                cur[0].clone_from(&outs[0].obs);
            }
        }

        // JSON (no serde dep — same hand-rolled format as biped_render.rs so
        // render_biped.py / render_biped_mujoco.py read both interchangeably).
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
        std::fs::write(&out, &s).expect("write json");

        println!(
            "wrote {} frames + skeleton → {out}\nrender: python3 examples/biped/render_biped.py {out} /tmp/biped_nexus.mp4",
            frames.len()
        );
    });
}
