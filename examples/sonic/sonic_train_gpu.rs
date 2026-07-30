//! GPU policy + GPU PPO trainer for SONIC-style whole-body motion tracking.
//!
//! This is intentionally a separate entry point from `biped_train_gpu`: it
//! shares the policy helper, vortx primitives, Nexus physics, and checkpoint
//! format without adding motion-specific branches to the locomotion trainer.
//!
//! Run:
//!   BIPED_ROBOT=g1_29dof_agile cargo run --release \
//!     --example sonic_train_gpu --features "gpu biped_gpu" -- \
//!     /path/to/bones-seed/g1/csv [iterations] [num-envs] [checkpoint]

#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "../biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../biped/gpu_policy.rs"]
mod gpu_policy;
mod gpu_ppo;

use anyhow::{Context, Result, bail};
use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use gpu_policy::GpuPolicy;
use gpu_ppo::{GpuPpoConfig, GpuPpoUpdater};
use std::path::Path;
use std::time::Instant;
use zealot_env::tasks::motion_tracking::{
    ACTOR_OBS_DIM, CONTROL_HZ, CRITIC_OBS_DIM, G1_CONTROLLED_JOINTS, MotionLibrary,
    MotionReference, MotionState, MotionTrackingTask,
};
use zealot_rl::ppo::PendingNorm;
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, Sample, gae};

const ROLLOUT_STEPS: usize = 24;
const GAMMA: f32 = 0.99;
const LAMBDA: f32 = 0.95;
const GPU_COPY_WORKGROUP_SIZE: usize = 128;
const MAX_DISPATCH_GROUPS: usize = 65_535;

fn usage() -> &'static str {
    "Usage: sonic_train_gpu <motion-directory> [iterations] [num-envs] [checkpoint]"
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = args.first().context(usage())?.clone();
    let iterations = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(5_000);
    let num_envs = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4_096);
    let checkpoint = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "/tmp/sonic_gpu.safetensors".into());
    if num_envs == 0 || iterations == 0 {
        bail!("iterations and num-envs must be positive");
    }
    pollster::block_on(train(source, iterations, num_envs, checkpoint))
}

async fn make_env(num_envs: usize) -> Result<BipedNexusBatchEnv> {
    let robot = std::env::var("BIPED_ROBOT").unwrap_or_default();
    if !matches!(
        robot.as_str(),
        "g1_29dof" | "g1_29dof_agile" | "g1full" | "g1_29"
    ) {
        bail!("set BIPED_ROBOT=g1_29dof_agile for whole-body tracking");
    }
    let path = default_mjcf_path();
    let xml =
        std::fs::read_to_string(&path).with_context(|| format!("read whole-body MJCF {path}"))?;
    let env = BipedNexusBatchEnv::new(&xml, num_envs, 32, 0x50_4e_49_43).await;
    if !env.supports_fullbody_control() {
        bail!("selected MJCF does not expose all 25 controlled G1 joints");
    }
    Ok(env)
}

fn action_array(action: &[f32]) -> [f32; G1_CONTROLLED_JOINTS] {
    let mut output = [0.0; G1_CONTROLLED_JOINTS];
    output.copy_from_slice(&action[..G1_CONTROLLED_JOINTS]);
    output
}

fn validate_policy(actor_critic: &ActorCritic) -> Result<()> {
    if actor_critic.actor.dims[0] != ACTOR_OBS_DIM
        || actor_critic.critic.dims[0] != CRITIC_OBS_DIM
        || actor_critic.action_dim() != G1_CONTROLLED_JOINTS
    {
        bail!(
            "checkpoint shape mismatch: actor={} critic={} action={}, expected {} {} {}",
            actor_critic.actor.dims[0],
            actor_critic.critic.dims[0],
            actor_critic.action_dim(),
            ACTOR_OBS_DIM,
            CRITIC_OBS_DIM,
            G1_CONTROLLED_JOINTS
        );
    }
    Ok(())
}

fn gpu_minibatches(total_samples: usize, native_cuda: bool) -> usize {
    if native_cuda {
        return 4;
    }
    let widest_input = ACTOR_OBS_DIM.max(CRITIC_OBS_DIM);
    let max_elements_per_dispatch = GPU_COPY_WORKGROUP_SIZE * MAX_DISPATCH_GROUPS;
    let minimum = (total_samples * widest_input)
        .div_ceil(max_elements_per_dispatch)
        .max(4);
    (minimum..=total_samples)
        .find(|count| total_samples % count == 0)
        .expect("total sample count always divides itself")
}

async fn train(
    source: String,
    iterations: usize,
    num_envs: usize,
    checkpoint: String,
) -> Result<()> {
    let max_motions = std::env::var("SONIC_MAX_MOTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_096);
    let load_started = Instant::now();
    let library = MotionLibrary::load(&source, Some(max_motions))?;
    println!(
        "loaded {} motions / {} frames in {:.2}s",
        library.clips.len(),
        library.total_frames(),
        load_started.elapsed().as_secs_f32()
    );
    println!("building {num_envs} GPU Nexus environments");
    let mut env = make_env(num_envs).await?;
    let backend = env.backend().clone();
    let mut rng = Lcg::new(7);
    let resume = Path::new(&checkpoint).exists();
    let mut actor_critic = if resume {
        println!("resuming weights from {checkpoint}");
        ActorCritic::load(&checkpoint)?
    } else {
        ActorCritic::new(
            &[ACTOR_OBS_DIM, 256, 256, 128, G1_CONTROLLED_JOINTS],
            &[CRITIC_OBS_DIM, 512, 256, 128, 1],
            0.5,
            3e-4,
            &mut rng,
        )
    };
    validate_policy(&actor_critic)?;
    let mut gpu_policy = GpuPolicy::new(&backend, &actor_critic, num_envs)?;
    let total_samples = num_envs * ROLLOUT_STEPS;
    let ppo_config = GpuPpoConfig {
        minibatches: gpu_minibatches(total_samples, backend.is_cuda()),
        ..GpuPpoConfig::default()
    };
    let mut updater = GpuPpoUpdater::new(&backend, &actor_critic, total_samples, ppo_config)?;

    let mut tasks = vec![MotionTrackingTask::default(); num_envs];
    let mut clip_ids: Vec<usize> = (0..num_envs).map(|env| env % library.clips.len()).collect();
    let mut times = vec![0.0f32; num_envs];
    let initial_targets: Vec<_> = clip_ids
        .iter()
        .map(|&clip| library.clips[clip].frames[0].joint_pos)
        .collect();
    let initial = env.step_fullbody(&initial_targets).await;
    let mut states: Vec<MotionState> = initial.into_iter().map(|step| step.state).collect();
    for (task, state) in tasks.iter_mut().zip(&states) {
        task.reset_history(state);
    }
    let mut warming = vec![false; num_envs];
    let mut pending_actor_norm = PendingNorm::default();
    let mut pending_critic_norm = PendingNorm::default();

    println!(
        "actor={} critic={} action={} rollout={} samples/iter={} minibatches={}",
        ACTOR_OBS_DIM,
        CRITIC_OBS_DIM,
        G1_CONTROLLED_JOINTS,
        ROLLOUT_STEPS,
        total_samples,
        ppo_config.minibatches
    );
    println!(
        "{:>5} {:>8} {:>8} {:>7} {:>7} {:>9} {:>8}",
        "iter", "reward", "resets", "warmup", "roll_s", "update_s", "steps/s"
    );

    for iteration in 0..iterations {
        let iteration_started = Instant::now();
        let mut samples: Vec<Vec<Sample>> = (0..num_envs)
            .map(|_| Vec::with_capacity(ROLLOUT_STEPS))
            .collect();
        let mut rewards: Vec<Vec<f32>> = (0..num_envs)
            .map(|_| Vec::with_capacity(ROLLOUT_STEPS))
            .collect();
        let mut values: Vec<Vec<f32>> = (0..num_envs)
            .map(|_| Vec::with_capacity(ROLLOUT_STEPS))
            .collect();
        let mut dones: Vec<Vec<bool>> = (0..num_envs)
            .map(|_| Vec::with_capacity(ROLLOUT_STEPS))
            .collect();
        let mut reward_sum = 0.0f32;
        let mut resets = 0usize;
        let mut warmup_steps = 0usize;
        let rollout_started = Instant::now();

        while samples.iter().any(|samples| samples.len() < ROLLOUT_STEPS) {
            let mut actor_obs = Vec::with_capacity(num_envs);
            let mut critic_obs = Vec::with_capacity(num_envs);
            let mut references = Vec::with_capacity(num_envs);
            let mut active = Vec::with_capacity(num_envs);
            for e in 0..num_envs {
                let reference = MotionReference::at(&library.clips[clip_ids[e]], times[e]);
                let is_active = !warming[e] && samples[e].len() < ROLLOUT_STEPS;
                let obs = tasks[e].actor_obs(&states[e], &reference);
                let privileged = tasks[e].critic_obs(&states[e], &reference);
                if is_active {
                    pending_actor_norm.push(&obs);
                    pending_critic_norm.push(&privileged);
                }
                actor_obs.push(obs);
                critic_obs.push(privileged);
                references.push(reference);
                active.push(is_active);
            }

            let (means, predicted_values) = gpu_policy
                .forward_dynamic(&backend, &actor_critic, &actor_obs, &critic_obs)
                .await?;
            let mut commands = Vec::with_capacity(num_envs);
            let mut previous_actions = Vec::with_capacity(num_envs);
            for e in 0..num_envs {
                previous_actions.push(states[e].last_action);
                if warming[e] {
                    commands.push(library.clips[clip_ids[e]].frames[0].joint_pos);
                } else if active[e] {
                    let mut action = vec![0.0; G1_CONTROLLED_JOINTS];
                    for joint in 0..G1_CONTROLLED_JOINTS {
                        action[joint] =
                            means[e][joint] + actor_critic.log_std[joint].exp() * rng.gauss();
                    }
                    let logp = actor_critic.logp(&action, &means[e]);
                    commands.push(tasks[e].joint_targets(&action_array(&action)));
                    values[e].push(predicted_values[e]);
                    samples[e].push(Sample {
                        obs: actor_obs[e].clone(),
                        critic_obs: critic_obs[e].clone(),
                        action,
                        mean_old: means[e].clone(),
                        logp_old: logp,
                        value_old: predicted_values[e],
                        adv: 0.0,
                        ret: 0.0,
                    });
                } else {
                    commands.push(references[e].now.joint_pos);
                }
            }

            let outputs = env.step_fullbody(&commands).await;
            for e in 0..num_envs {
                states[e] = outputs[e].state.clone();
                if warming[e] {
                    tasks[e].reset_history(&states[e]);
                    warming[e] = false;
                    warmup_steps += 1;
                    continue;
                }
                if !active[e] {
                    tasks[e].push_state(&states[e]);
                    continue;
                }
                times[e] += 1.0 / CONTROL_HZ;
                tasks[e].push_state(&states[e]);
                let clip = &library.clips[clip_ids[e]];
                let next_reference = MotionReference::at(clip, times[e]);
                let reward = tasks[e].reward(&states[e], &next_reference, &previous_actions[e]);
                let done =
                    tasks[e].terminated(&states[e], &next_reference) || times[e] >= clip.duration();
                reward_sum += reward.total;
                rewards[e].push(reward.total);
                dones[e].push(done);
                if done {
                    resets += 1;
                    env.reset_fullbody_env(e).await;
                    clip_ids[e] = (clip_ids[e] + 1 + iteration) % library.clips.len();
                    times[e] = 0.0;
                    states[e] = MotionState::default();
                    tasks[e].reset_history(&states[e]);
                    warming[e] = true;
                }
            }
        }
        let rollout_seconds = rollout_started.elapsed().as_secs_f64();

        let mut bootstrap_actor = Vec::with_capacity(num_envs);
        let mut bootstrap_critic = Vec::with_capacity(num_envs);
        for e in 0..num_envs {
            let reference = MotionReference::at(&library.clips[clip_ids[e]], times[e]);
            bootstrap_actor.push(tasks[e].actor_obs(&states[e], &reference));
            bootstrap_critic.push(tasks[e].critic_obs(&states[e], &reference));
        }
        let (_, bootstrap_values) = gpu_policy
            .forward_dynamic(&backend, &actor_critic, &bootstrap_actor, &bootstrap_critic)
            .await?;
        let mut batch = Vec::with_capacity(num_envs * ROLLOUT_STEPS);
        for e in 0..num_envs {
            let last_value = if dones[e].last().copied().unwrap_or(true) {
                0.0
            } else {
                bootstrap_values[e]
            };
            let (advantages, returns) = gae(
                &rewards[e],
                &values[e],
                &dones[e],
                last_value,
                GAMMA,
                LAMBDA,
            );
            for (step, mut sample) in samples[e].drain(..).enumerate() {
                sample.adv = advantages[step];
                sample.ret = returns[step];
                batch.push(sample);
            }
        }

        let stats = updater
            .update(&backend, &mut actor_critic, &mut batch)
            .await?;
        actor_critic.obs_norm.commit(&mut pending_actor_norm);
        actor_critic.critic_norm.commit(&mut pending_critic_norm);
        gpu_policy.sync_weights(&backend, &actor_critic);
        let elapsed = iteration_started.elapsed().as_secs_f64();
        println!(
            "{:5} {:8.3} {:8} {:7} {:7.2} {:9.2} {:8.0} kl={:.4} lr={:.2e}",
            iteration + 1,
            reward_sum / (num_envs * ROLLOUT_STEPS) as f32,
            resets,
            warmup_steps,
            rollout_seconds,
            stats.seconds,
            (num_envs * ROLLOUT_STEPS) as f64 / elapsed,
            stats.kl,
            stats.lr
        );
        if (iteration + 1) % 10 == 0 || iteration + 1 == iterations {
            actor_critic.save(&checkpoint)?;
        }
    }
    println!("saved {checkpoint}");
    Ok(())
}
