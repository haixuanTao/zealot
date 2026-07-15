//! Train flat velocity tracking on the LeRobot bipedal — **nexus GPU physics**.
//!
//! LEGACY / REFERENCE TRAINER. The **default** end-to-end trainer is
//! `biped_train_gpu.rs`, which runs both the rollout policy forward AND the PPO
//! update on the GPU (~8–10× faster) and carries the full feature set
//! (stand/torque curricula, time-limit bootstrapping, adaptive-KL LR, log_std
//! re-flooring, mirror augmentation, per-component reward logging). This file
//! keeps the policy + PPO on the CPU and is retained only as a simple reference
//! / fallback; prefer `biped_train_gpu` for real runs.
//!
//! Mirror of `biped_train.rs` but uses `BipedNexusBatchEnv` (one batched
//! `GpuPhysicsState` holding N envs) instead of `Vec<BipedEnv>` over rapier CPU.
//! Same MDP (`zealot-env`), same PPO (`zealot-rl`), same obs/action layout — so
//! a policy trained here is swap-compatible with the CPU rollout/render path.
//!
//! Run: `cargo run --release --example biped_train_nexus --features biped_gpu -- [iters] [num_envs]`

#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;

use biped_env_nexus::{BipedNexusBatchEnv, StepOut, default_mjcf_path};
use std::time::Instant;
use zealot_env::robots::{RobotSpec, NUM_JOINTS};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig, Sample, gae};

const T: usize = 32; // steps per env per iteration

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

// --- Left/right mirror augmentation (forces a SYMMETRIC policy → straight,
// non-veering gait; the instantaneous bilateral_symmetry reward can't, since a
// walking gait is half-a-cycle out of phase). The mirror is an isometry of the
// action space (joint L/R swap + sign flips), so logp_old/adv are preserved when
// mean_old is mirrored too — the policy loss is EXACT. The joint mirror
// permutation/signs are per-robot (canonical joint orders differ): lateral
// families (roll/yaw) negate; sagittal (pitch/knee) keep.
static JMIRROR: std::sync::LazyLock<[usize; NUM_JOINTS]> =
    std::sync::LazyLock::new(|| RobotSpec::from_env().mirror);
static JSIGN: std::sync::LazyLock<[f32; NUM_JOINTS]> =
    std::sync::LazyLock::new(|| RobotSpec::from_env().mirror_sign);

fn jmirror(v: &[f32]) -> Vec<f32> {
    (0..NUM_JOINTS).map(|i| JSIGN[i] * v[JMIRROR[i]]).collect()
}

// obs frame layout (45): last_action[0:12], command[12:16]=(vx,vy,yaw,aux),
// joint_pos[16:28], joint_vel[28:40], proj_grav[40:43]=(fwd,lat,up),
// gait_phase[43:45]=(sin,cos).
fn mirror_frame(o: &[f32]) -> Vec<f32> {
    let mut m = o.to_vec();
    m[0..12].copy_from_slice(&jmirror(&o[0..12]));
    m[13] = -o[13]; // command vy
    m[14] = -o[14]; // command yaw_rate
    m[16..28].copy_from_slice(&jmirror(&o[16..28]));
    m[28..40].copy_from_slice(&jmirror(&o[28..40]));
    m[41] = -o[41]; // proj_grav lateral
    m[43] = -o[43]; // gait phase sin → contralateral (half-cycle)
    m[44] = -o[44]; // gait phase cos
    m
}

// With BIPED_OBS_HISTORY the actor obs is H stacked 45-frames — mirror each
// frame independently (block-diagonal). H=1 = the plain frame.
fn mirror_obs(o: &[f32]) -> Vec<f32> {
    o.chunks(45).flat_map(|f| mirror_frame(f)).collect()
}

// critic_obs (51) = obs frame(45) + base_lin_vel(3)[fwd,lat,up] +
// base_ang_vel(3)[roll,pitch,yaw] — single-frame (no history on the critic).
fn mirror_critic(c: &[f32]) -> Vec<f32> {
    let mut m = mirror_frame(&c[0..45]);
    m.extend_from_slice(&c[45..]);
    m[46] = -c[46]; // lin_vel lateral (polar vector)
    m[48] = -c[48]; // ang_vel roll  (axial: negate roll + yaw, keep pitch)
    m[50] = -c[50]; // ang_vel yaw
    m
}

fn mirror_sample(s: &Sample) -> Sample {
    Sample {
        obs: mirror_obs(&s.obs),
        critic_obs: mirror_critic(&s.critic_obs),
        action: jmirror(&s.action),
        mean_old: jmirror(&s.mean_old),
        logp_old: s.logp_old,
        value_old: s.value_old,
        adv: s.adv,
        ret: s.ret,
    }
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let num_envs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    // 3rd arg: where to save the trained policy (safetensors). Pass an empty
    // string to skip saving.
    let policy_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "/tmp/biped_policy_nexus.safetensors".to_string());

    let xml = std::fs::read_to_string(default_mjcf_path()).expect("read mjcf");
    println!("building {num_envs} envs (batched on one GpuPhysicsState)...");

    pollster::block_on(async {
        let mut env = BipedNexusBatchEnv::new(&xml, num_envs, 32, 0xC0FFEE).await;

        let (obs_dim, critic_dim, act_dim) =
            (env.obs_dim(), env.critic_obs_dim(), env.action_dim());
        println!("obs={obs_dim} critic_obs={critic_dim} action={act_dim}");

        let mut rng = Lcg::new(7);
        // Auto-resume: if a checkpoint already exists at `policy_path`, load
        // it and continue training from there. Otherwise build a fresh net.
        let mut ac = if !policy_path.is_empty() && std::path::Path::new(&policy_path).exists() {
            println!("resuming from existing checkpoint {policy_path}...");
            let ac = ActorCritic::load(&policy_path).expect("load checkpoint");
            assert_eq!(
                ac.actor.dims[0], obs_dim,
                "checkpoint actor input dim {} != env obs dim {obs_dim} — \
                 BIPED_OBS_HISTORY mismatch? delete {policy_path} or match the setting",
                ac.actor.dims[0]
            );
            assert_eq!(
                ac.critic.dims[0], critic_dim,
                "checkpoint critic input dim {} != env critic obs dim {critic_dim} — delete {policy_path}",
                ac.critic.dims[0]
            );
            ac
        } else {
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
        const CHECKPOINT_EVERY: usize = 50;

        let (mut cur, mut cur_c) = env.initial_obs().await;

        println!(
            "\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}",
            "iter", "curr", "step_rew", "falls", "torso_z"
        );

        let mirror_aug = std::env::var("BIPED_MIRROR_AUG").is_ok();
        if mirror_aug {
            println!("mirror augmentation ENABLED (symmetric policy)");
        }
        // Curriculum: command-velocity ramps 0 → 1 over the first 40% of iters.
        let warmup = (iters as f32 * 0.4).max(1.0);
        for it in 0..iters {
            let scale = (it as f32 / warmup).min(1.0);
            env.set_command_scale(scale);

            // Collect T steps across all envs.
            let mut samples: Vec<Vec<Sample>> =
                (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut rs: Vec<Vec<f32>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut vs: Vec<Vec<f32>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let mut ds: Vec<Vec<bool>> = (0..num_envs).map(|_| Vec::with_capacity(T)).collect();
            let (mut total_reward, mut falls, mut torso_sum) = (0.0f32, 0u32, 0.0f32);
            // [TIMING] ns accumulators: CPU policy fwd, GPU env.step, per-done
            // env resets, CPU PPO update.
            let (mut t_pol, mut t_step, mut t_commit, mut t_upd) = (0u128, 0u128, 0u128, 0u128);

            for _ in 0..T {
                // Sample actions + values for all envs (sequential — shared policy).
                let tp = Instant::now();
                let mut actions: Vec<[f32; NUM_JOINTS]> = Vec::with_capacity(num_envs);
                for e in 0..num_envs {
                    ac.record_obs(&cur[e], &cur_c[e]);
                    let (action, logp, mean) = ac.sample(&cur[e], &mut rng);
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
                t_pol += tp.elapsed().as_nanos();
                // ONE GPU dispatch advances every env.
                let ts = Instant::now();
                let outs: Vec<StepOut> = env.step(&actions).await;
                t_step += ts.elapsed().as_nanos();

                let tc = Instant::now();
                for e in 0..num_envs {
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
                t_commit += tc.elapsed().as_nanos();
                // Cheap torso-z aggregate (avoids another GPU readback — already
                // baked into the last step's StepOut via obs[ ... ] but we don't
                // expose that index here, so re-read once per T-block instead).
            }
            // One torso-Z readback at the end of the iteration is enough for telemetry.
            let zs = env.torso_heights().await;
            for z in &zs {
                torso_sum += *z;
            }

            // GAE per env, then flatten + PPO update.
            let mut batch: Vec<Sample> = Vec::with_capacity(num_envs * T);
            for e in 0..num_envs {
                let last_v = ac.value(&cur_c[e]);
                let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], last_v, cfg.gamma, cfg.lam);
                for t in 0..T {
                    samples[e][t].adv = adv[t];
                    samples[e][t].ret = ret[t];
                    batch.push(std::mem::take(&mut samples[e][t]));
                }
            }
            // Mirror augmentation: append the L/R-mirrored copy of every sample so
            // the policy is trained to be symmetric (fixes the lopsided, veering
            // gait). BIPED_MIRROR_AUG=1 to enable.
            if mirror_aug {
                let mirrored: Vec<Sample> = batch.iter().map(mirror_sample).collect();
                batch.extend(mirrored);
            }
            let tu = Instant::now();
            let _stats = ac.update(&mut batch, &cfg);
            t_upd += tu.elapsed().as_nanos();

            if it % 10 == 0 || it == iters - 1 {
                let ms = |x: u128| x as f64 / 1e6;
                println!(
                    "  [time] policy_cpu {:.0}ms | gpu_step {:.0}ms | resets {:.0}ms | ppo_update {:.0}ms",
                    ms(t_pol),
                    ms(t_step),
                    ms(t_commit),
                    ms(t_upd)
                );
            }
            if it % 10 == 0 || it == iters - 1 {
                let steps = (num_envs * T) as f32;
                println!(
                    "{:>4}  {:>5.2}  {:>9.4}  {:>7}  {:>8.3}",
                    it,
                    scale,
                    total_reward / steps,
                    falls,
                    torso_sum / num_envs as f32,
                );
            }
            // Periodic checkpoint so a killed run leaves a resumable state.
            // Skip iter 0 so we don't overwrite a resumed checkpoint with the
            // un-trained starting net.
            if !policy_path.is_empty() && it > 0 && (it % CHECKPOINT_EVERY == 0 || it == iters - 1)
            {
                if let Err(e) = ac.save(&policy_path) {
                    eprintln!("warning: checkpoint save failed at iter {it}: {e}");
                } else {
                    println!("  checkpoint → {policy_path}");
                }
            }
        }
        // Final save (belt-and-braces — the periodic write above already
        // handled iters-1, but a no-op extra write is cheap).
        if !policy_path.is_empty() {
            ac.save(&policy_path).expect("save policy");
            println!("saved final policy → {policy_path}");
        }
    });
}
