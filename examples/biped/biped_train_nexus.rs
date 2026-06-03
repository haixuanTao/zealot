//! Train flat velocity tracking on the LeRobot bipedal — **nexus GPU physics**.
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
use zealot_env::robots::lerobot_bipedal::NUM_JOINTS;
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig, Sample, gae};

const T: usize = 32; // steps per env per iteration

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
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
        const CHECKPOINT_EVERY: usize = 50;

        let (mut cur, mut cur_c) = env.initial_obs().await;

        println!(
            "\n{:>4}  {:>5}  {:>9}  {:>7}  {:>8}",
            "iter", "curr", "step_rew", "falls", "torso_z"
        );

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

            for _ in 0..T {
                // Sample actions + values for all envs (sequential — shared policy).
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
                // ONE GPU dispatch advances every env.
                let outs: Vec<StepOut> = env.step(&actions).await;

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
            let _stats = ac.update(&mut batch, &cfg);

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
