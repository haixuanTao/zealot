//! Train briefly on **nexus GPU physics**, then record a deterministic rollout
//! of env 0 to JSON for rendering with `scripts/render_biped.py` (or the
//! MuJoCo mesh renderer). Same output format as `biped_render.rs` — same python
//! script reads both.
//!
//! Run:
//!   `cargo run --release --example biped_render_nexus --features biped_gpu -- \
//!         [train_iters] [rollout_steps] [out.json]`
//! then:
//!   `python3 scripts/render_biped.py /tmp/biped_rollout_nexus.json /tmp/biped_nexus.mp4`

#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "../biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../biped/gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{BipedNexusBatchEnv, StepOut, default_mjcf_path};
use gpu_policy::GpuPolicy;
use std::fmt::Write as _;
use zealot_env::robots::{RobotSpec, NUM_JOINTS};
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

    // GPU-resident actor/critic for the batched rollout forward, on the env's
    // own backend. Re-synced from `ac` after every PPO update below.
    let mut gpu = GpuPolicy::new(env.backend(), ac, n).expect("build gpu policy");

    // Curriculum/resume position: persist the iteration index alongside the
    // weight checkpoint so a killed-and-resumed run continues the command
    // curriculum instead of restarting the scale ramp from 0. The safetensors
    // checkpoint stores weights + normalizers but not this loop counter, so
    // without the sidecar a resume silently rewinds `scale` to 0.
    let progress_path = format!("{checkpoint_path}.iter");
    let start_iter = std::fs::read_to_string(&progress_path)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&i| i < iters)
        .unwrap_or(0);
    if start_iter > 0 {
        println!("resuming curriculum at iter {start_iter}/{iters}");
    }

    let warmup = (iters as f32 * 0.4).max(1.0);
    for it in start_iter..iters {
        let scale = (it as f32 / warmup).min(1.0);
        env.set_command_scale(scale);

        let mut samples: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut rs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut vs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let mut ds: Vec<Vec<bool>> = (0..n).map(|_| Vec::with_capacity(T)).collect();
        let (mut total_reward, mut falls) = (0.0f32, 0u32);

        for _ in 0..T {
            // Fold this step's obs into the running normalizers, then run ONE
            // batched GPU forward for all envs (the old per-env CPU actor/critic
            // forward loop was the rollout bottleneck — see gpu_policy.rs).
            for e in 0..n {
                ac.record_obs(&cur[e], &cur_c[e]);
            }
            let (means, values) = gpu
                .forward(env.backend(), ac, &cur, &cur_c)
                .await
                .expect("gpu policy forward");

            let mut actions: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(n);
            for e in 0..n {
                // Sample a = mean + std·ε and its log-prob — cheap CPU work
                // (std-only diagonal Gaussian), matching `ActorCritic::sample`.
                let mean = means[e].to_vec();
                let mut action = vec![0.0f32; NUM_JOINTS];
                for k in 0..NUM_JOINTS {
                    action[k] = mean[k] + ac.log_std[k].exp() * rng.gauss();
                }
                let logp = ac.logp(&action, &mean);
                let value = values[e];
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
        // Weights changed — push them to the GPU policy for next iter's rollout.
        gpu.sync_weights(env.backend(), ac);

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
                // Record the next iter to resume at, so the curriculum scale
                // continues across a kill/restart. Written after the weights so
                // the two stay consistent on resume.
                let _ = std::fs::write(&progress_path, (it + 1).to_string());
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
        // Train at the deployed-scale batch size — 4096 envs is where the
        // nexus GPU path actually beats CPU rapier on a 5090 (~40k env/s,
        // see README "Cross-engine reference" table). Templates stay at 32:
        // each template defines one DR scene (friction / restitution / PD
        // scale / contact softness / spawn pose) and N_envs/N_templates =
        // 128 envs share each template at construction time, then mix
        // freely as `reset_env` cycles them.
        // BIPED_RENDER_ENVS: shrink for rollout-only runs (train_iters=0) — a
        // 1-env eval doesn't need the deployed-scale batch, and a small build
        // coexists with a training run on the same GPU.
        let num_envs = std::env::var("BIPED_RENDER_ENVS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        let num_templates = 32.min(num_envs);
        println!("building {num_envs} envs on nexus ({num_templates} DR templates)...");
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
                // Fresh run — clear any stale curriculum-progress sidecar so the
                // scale ramp starts at iter 0.
                let _ = std::fs::remove_file(format!("{policy_path}.iter"));
                // Matches WBC-AGILE T1 velocity policy exactly: asymmetric net
                // (actor smaller, privileged critic wider), `init_noise_std=1.0`,
                // lr 1e-3 with adaptive-KL schedule.
                ActorCritic::new(
                    &[obs_dim, 256, 256, 128, act_dim],
                    &[critic_dim, 512, 256, 128, 1],
                    1.0,
                    1e-3,
                    &mut rng,
                )
            };
            // rsl_rl-style: adaptive-KL LR, entropy bonus at WBC-AGILE's 0.005.
            let cfg = PpoConfig {
                entropy_coef: 0.005,
                ..PpoConfig::default()
            };
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
        // Pinned command [vx,vy,yaw], default forward 0.4. Override with
        // BIPED_RENDER_CMD="0,0,0" to render a stand-trained policy fairly.
        let rcmd: Vec<f32> = std::env::var("BIPED_RENDER_CMD")
            .unwrap_or_else(|_| "0.4,0,0".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let (rvx, rvy, ryaw) = (
            rcmd.first().copied().unwrap_or(0.4),
            rcmd.get(1).copied().unwrap_or(0.0),
            rcmd.get(2).copied().unwrap_or(0.0),
        );
        let _ = env.reset_env_to_default_template(0).await;
        env.pin_command_for(0, rvx, rvy, ryaw);
        let (names, edges, feet) = env.skeleton();

        let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(rollout_steps);
        let mut frame_quats: Vec<Vec<[f32; 4]>> = Vec::with_capacity(rollout_steps);
        let mut bases: Vec<[f32; 7]> = Vec::with_capacity(rollout_steps);
        let mut joints: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(rollout_steps);
        let dump_vel = std::env::var("BIPED_DUMP_VEL").is_ok();
        let mut dof_vels: Vec<Vec<f32>> = Vec::new();
        let mut resets: Vec<usize> = Vec::new();
        // Ground-truth policy I/O per step, for the sim-to-sim cross-val parity
        // check: the exact 43-dim obs fed to the actor and the 12-dim mean action
        // it produced. A Python re-implementation of the obs+net must reproduce
        // these before we trust its MuJoCo closed-loop rollout.
        let mut obss: Vec<Vec<f32>> = Vec::with_capacity(rollout_steps);
        let mut acts: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(rollout_steps);

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
            frame_quats.push(env.body_rotations_for(0, &poses));
            let (p, q) = env.base_pose_for(0, &poses);
            bases.push([p[0], p[1], p[2], q[0], q[1], q[2], q[3]]);
            joints.push(env.joint_angles_for(0, &poses));
            // BIPED_DUMP_VEL=1: true generalized velocities from dof_state (a
            // readback per step, so opt-in). The divergence probe needs these
            // — FD velocities at 50 Hz are its noise floor.
            if dump_vel {
                dof_vels.push(env.true_dof_velocities(0).await);
            }

            // Mean (noise-free) action for env 0.
            let mean = ac.mean(&cur[0]);
            // Record the exact obs that produced this action + the action itself.
            obss.push(cur[0].clone());
            acts.push(to_action(&mean));
            let mut actions: Vec<[f32; NUM_JOINTS]> = vec![[0.0; NUM_JOINTS]; num_envs];
            actions[0] = to_action(&mean);
            // PASSIVE mode: ignore the policy and hold the default (zero) pose
            // every step (action = 0 → target = default_pos). No resets — record
            // the uninterrupted settle so we can compare nexus's passive dynamics
            // to MuJoCo's (isolates physics instability from the policy).
            let passive = std::env::var("BIPED_PASSIVE").is_ok();
            if passive {
                actions[0] = [0.0; NUM_JOINTS];
            }
            // Other envs: just hold zero (we don't render them).
            let outs = env.step(&actions).await;
            if std::env::var("BIPED_DBG_JACS").is_ok() && step == 5 {
                let (jacs, cols, (cpb, dpb, spb)) = env.dbg_mb_jac_columns().await;
                let (_, cons) = env.dbg_mb_contacts().await;
                let mut out = String::from("{\n");
                out.push_str(&format!("  \"cpb\": {cpb}, \"dpb\": {dpb}, \"spb\": {spb},\n"));
                out.push_str("  \"slots\": [\n");
                for (i, c) in cons.iter().take(192).enumerate() {
                    if c.kind == 0 { continue; }
                    let row: Vec<String> = (0..dpb as usize)
                        .map(|d| format!("{:.6}", jacs[i * dpb as usize + d]))
                        .collect();
                    let col: Vec<String> = (0..dpb as usize)
                        .map(|d| format!("{:.6}", cols[i * dpb as usize + d]))
                        .collect();
                    out.push_str(&format!(
                        "    {{\"s\": {i}, \"kind\": {}, \"link\": {}, \"inv_lhs\": {:.6}, \"lin_jac\": [{:.6},{:.6},{:.6}], \"ang_jac\": [{:.6},{:.6},{:.6}], \"jrow\": [{}], \"col\": [{}]}},\n",
                        c.kind, c.link_id, c.inv_lhs,
                        c.lin_jac.x, c.lin_jac.y, c.lin_jac.z,
                        c.ang_jac.x, c.ang_jac.y, c.ang_jac.z,
                        row.join(","), col.join(",")));
                }
                out.push_str("    {\"s\": -1, \"kind\": 0, \"link\": 0, \"inv_lhs\": 0, \"lin_jac\": [0,0,0], \"ang_jac\": [0,0,0], \"jrow\": [], \"col\": []}\n  ]\n}\n");
                let ls = env.dbg_links_static().await;
                // links_static is batch-interleaved: link i of batch b at i*NB+b.
                let nb = 64usize;
                for i in 0..13 {
                    let l = &ls[i * nb];
                    println!("[dbgl] link {i} mass {:.4} com {:?}", 1.0/l.local_mprops.inv_mass.x.max(1e-9), l.local_mprops.com);
                }
                let bj = env.dbg_body_jacobians().await;
                let bjs: Vec<String> = bj.iter().map(|v| format!("{v:.6}")).collect();
                std::fs::write("/tmp/bodyjacs.json", format!("[{}]", bjs.join(","))).unwrap();
                std::fs::write("/tmp/jacdump.json", out).unwrap();
                println!("[dbgj] wrote /tmp/jacdump.json");
            }
            let dbg_contacts_from: Option<usize> = std::env::var("BIPED_DBG_CONTACTS")
                .ok()
                .map(|v| v.parse().unwrap_or(0));
            if dbg_contacts_from.is_some_and(|s0| step >= s0 && step < s0 + 10) {
                let (_, cons) = env.dbg_mb_contacts().await;
                // Aligned with the CPU probe ([cpuc]): per-foot Fz + per-point
                // (x, normal imp, tangent imp). x = +ang_jac.y (ground-static:
                // ang_jac=(pt-com)x(-z) so ang_jac.y = pt_x - com_x, com_x=0).
                // Fz = last-substep accumulated impulse / substep dt.
                let decim: f32 = std::env::var("BIPED_DECIMATION")
                    .ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
                let sub_dt = 0.02 / decim / 8.0;
                let mut line = format!("[gpuc] step {step}:");
                for foot_link in [6u32, 12u32] {
                    let mut fz = 0.0f32;
                    let mut pts = String::new();
                    for c in cons.iter().take(192) {
                        if c.kind == 0 || c.link_id != foot_link { continue; }
                        if c.kind == 1 {
                            fz += c.impulse;
                            pts.push_str(&format!(" (x{:+.3} N{:.3} rhs{:+.2})", c.ang_jac.y, c.impulse, c.rhs));
                        } else if c.kind == 2 {
                            // tangent row: lin_jac = tangent dir; print its x-component,
                            // impulse and mu (clamp = mu * paired normal impulse).
                            pts.push_str(&format!(" [T{:+.3} tx{:+.2} slip{:+.4}]", c.impulse, c.lin_jac.x, c._unused_cfm));
                        }
                    }
                    line.push_str(&format!(" foot{}: Fz~{:7.1}N pts[{} ]",
                        if foot_link == 6 { 0 } else { 1 }, fz / sub_dt, pts));
                }
                println!("{line}");
            }

            if !passive && outs[0].done {
                resets.push(step);
                // Randomized resets BY DEFAULT (random DR template + spawn
                // perturbation), so each reset cycle starts differently and the
                // video shows the policy recovering from varied states. Set
                // BIPED_RENDER_DET=1 for the fixed DR-off default template (a
                // reproducible, frame-comparable eval).
                let o = if std::env::var("BIPED_RENDER_DET").is_ok() {
                    env.reset_env_to_default_template(0).await.0
                } else {
                    env.reset_env(0).await.0
                };
                cur[0] = o;
                env.pin_command_for(0, rvx, rvy, ryaw);
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
        let jn: Vec<String> = RobotSpec::from_env().joints.iter().map(|j| format!("\"{}\"", j.name)).collect();
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
        if dump_vel {
            let dofs: Vec<String> = env
                .policy_joint_dofs()
                .iter()
                .map(|d| d.to_string())
                .collect();
            let _ = write!(s, "  \"joint_dof_idx\": [{}],\n", dofs.join(","));
        }
        if !dof_vels.is_empty() {
            let dv_json: Vec<String> = dof_vels
                .iter()
                .map(|v| {
                    let e: Vec<String> = v.iter().map(|a| format!("{a:.6}")).collect();
                    format!("[{}]", e.join(","))
                })
                .collect();
            let _ = write!(s, "  \"dof_vel\": [{}],\n", dv_json.join(","));
        }
        let obs_json: Vec<String> = obss
            .iter()
            .map(|o| {
                let v: Vec<String> = o.iter().map(|a| format!("{a:.6}")).collect();
                format!("[{}]", v.join(","))
            })
            .collect();
        let _ = write!(s, "  \"obs\": [{}],\n", obs_json.join(","));
        let act_json: Vec<String> = acts
            .iter()
            .map(|a| {
                let v: Vec<String> = a.iter().map(|x| format!("{x:.6}")).collect();
                format!("[{}]", v.join(","))
            })
            .collect();
        let _ = write!(s, "  \"actions\": [{}],\n", act_json.join(","));
        // Terrain patch around the trajectory, so the offline renderer can
        // draw the ground (the step riser is otherwise invisible and the
        // robot appears to levitate as it climbs).
        {
            let n = bases.len().max(1);
            let (mcx, mcy) = (
                bases.iter().map(|b| b[0]).sum::<f32>() / n as f32,
                bases.iter().map(|b| b[1]).sum::<f32>() / n as f32,
            );
            let (half, hs, hf) = env.terrain_patch_for(0, mcx, mcy, 6.0, 0.15);
            let vals: Vec<String> = hf.iter().map(|h| format!("{:.3}", h)).collect();
            let _ = write!(
                s,
                "  \"terrain\": {{\"cx\": {:.3}, \"cy\": {:.3}, \"half\": {:.3}, \"hs\": {:.3}, \"heights\": [{}]}},\n",
                mcx, mcy, half, hs, vals.join(",")
            );
        }
        s.push_str("  \"frame_quats\": [\n");
        for (fi, frame) in frame_quats.iter().enumerate() {
            let pts: Vec<String> = frame
                .iter()
                .map(|q| format!("[{:.5},{:.5},{:.5},{:.5}]", q[0], q[1], q[2], q[3]))
                .collect();
            let comma = if fi + 1 < frame_quats.len() { "," } else { "" };
            let _ = write!(s, "    [{}]{}\n", pts.join(","), comma);
        }
        s.push_str("  ],\n");
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
            "wrote {} frames + skeleton → {out}\nrender: python3 scripts/render_biped.py {out} /tmp/biped_nexus.mp4",
            frames.len()
        );
    });
}
