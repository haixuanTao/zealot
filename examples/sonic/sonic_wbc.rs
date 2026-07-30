//! SONIC-style reference-motion tracking on zealot's nexus G1 environment.
//!
//! This reproduces the core idea—future motion conditioning and whole-body
//! joint-position targets—while reusing zealot's locomotion physics and PPO.

#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;

use anyhow::{Context, Result, bail};
use biped_env_nexus::{BipedNexusBatchEnv, default_mjcf_path};
use std::fmt::Write as _;
use std::path::Path;
use zealot_env::tasks::motion_tracking::{
    ACTOR_OBS_DIM, CONTROL_HZ, CRITIC_OBS_DIM, G1_CONTROLLED_JOINTS, MotionLibrary,
    MotionReference, MotionState, MotionTrackingTask,
};
use zealot_rl::rng::Lcg;
use zealot_rl::{ActorCritic, PpoConfig, Sample, gae};

const ROLLOUT_STEPS: usize = 24;

fn usage() -> &'static str {
    "Usage:
  sonic_wbc inspect-motion <csv-or-directory> [max-motions]
  sonic_wbc train <csv-or-directory> [iterations] [num-envs] [checkpoint]
  sonic_wbc eval <motion.csv> <checkpoint> [rollout.json]

The GPU commands require BIPED_ROBOT=g1_29dof_agile (or g1_29dof)."
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect-motion") => inspect(&args[1..]),
        Some("train") => pollster::block_on(train(&args[1..])),
        Some("eval") => pollster::block_on(evaluate(&args[1..])),
        _ => bail!(usage()),
    }
}

fn inspect(args: &[String]) -> Result<()> {
    let source = args.first().context(usage())?;
    let max = args.get(1).map(|s| s.parse()).transpose()?;
    let library = MotionLibrary::load(source, max)?;
    println!(
        "{} motions, {} resampled frames at {CONTROL_HZ:.0} Hz",
        library.clips.len(),
        library.total_frames()
    );
    for clip in &library.clips {
        println!(
            "{:7.2}s  {:6} frames  {}",
            clip.duration(),
            clip.frames.len(),
            clip.source.display()
        );
    }
    Ok(())
}

async fn make_env(num_envs: usize) -> Result<BipedNexusBatchEnv> {
    let robot = std::env::var("BIPED_ROBOT").unwrap_or_default();
    if !matches!(
        robot.as_str(),
        "g1_29dof" | "g1_29dof_agile" | "g1full" | "g1_29"
    ) {
        bail!("set BIPED_ROBOT=g1_29dof_agile for the 25-joint SONIC example");
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

fn checked_policy(path: &str) -> Result<ActorCritic> {
    let ac = ActorCritic::load(path).with_context(|| format!("load checkpoint {path}"))?;
    if ac.actor.dims[0] != ACTOR_OBS_DIM
        || ac.critic.dims[0] != CRITIC_OBS_DIM
        || ac.action_dim() != G1_CONTROLLED_JOINTS
    {
        bail!(
            "checkpoint shape mismatch: actor {} critic {} action {}, expected {} {} {}",
            ac.actor.dims[0],
            ac.critic.dims[0],
            ac.action_dim(),
            ACTOR_OBS_DIM,
            CRITIC_OBS_DIM,
            G1_CONTROLLED_JOINTS
        );
    }
    Ok(ac)
}

fn action_array(action: &[f32]) -> [f32; G1_CONTROLLED_JOINTS] {
    let mut out = [0.0; G1_CONTROLLED_JOINTS];
    out.copy_from_slice(&action[..G1_CONTROLLED_JOINTS]);
    out
}

async fn train(args: &[String]) -> Result<()> {
    let source = args.first().context(usage())?;
    let iterations = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(100);
    let num_envs = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let checkpoint = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("/tmp/sonic_wbc.safetensors");
    if num_envs == 0 {
        bail!("num-envs must be positive");
    }
    let max_motions = std::env::var("SONIC_MAX_MOTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let library = MotionLibrary::load(source, Some(max_motions))?;
    println!(
        "loaded {} motions / {} frames; building {num_envs} GPU environments",
        library.clips.len(),
        library.total_frames()
    );
    let mut env = make_env(num_envs).await?;
    let mut rng = Lcg::new(7);
    let mut ac = if Path::new(checkpoint).exists() {
        println!("resuming {checkpoint}");
        checked_policy(checkpoint)?
    } else {
        ActorCritic::new(
            &[ACTOR_OBS_DIM, 256, 256, 128, G1_CONTROLLED_JOINTS],
            &[CRITIC_OBS_DIM, 512, 256, 128, 1],
            0.5,
            3e-4,
            &mut rng,
        )
    };
    let cfg = PpoConfig {
        entropy_coef: 0.005,
        ..PpoConfig::default()
    };
    let mut tasks = vec![MotionTrackingTask::default(); num_envs];
    let mut clip_ids: Vec<usize> = (0..num_envs).map(|e| e % library.clips.len()).collect();
    let mut times = vec![0.0f32; num_envs];
    let initial_targets: Vec<_> = clip_ids
        .iter()
        .map(|&i| library.clips[i].frames[0].joint_pos)
        .collect();
    let initial = env.step_fullbody(&initial_targets).await;
    let mut states: Vec<MotionState> = initial.into_iter().map(|x| x.state).collect();
    for (task, state) in tasks.iter_mut().zip(&states) {
        task.reset_history(state);
    }

    println!(
        "actor={} critic={} action={}; rollout={} steps",
        ACTOR_OBS_DIM, CRITIC_OBS_DIM, G1_CONTROLLED_JOINTS, ROLLOUT_STEPS
    );
    for iteration in 0..iterations {
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
        let mut reward_sum = 0.0;
        let mut terminations = 0usize;

        for _ in 0..ROLLOUT_STEPS {
            let mut commands = Vec::with_capacity(num_envs);
            let mut previous_actions = Vec::with_capacity(num_envs);
            for e in 0..num_envs {
                let reference = MotionReference::at(&library.clips[clip_ids[e]], times[e]);
                tasks[e].push_state(&states[e]);
                let obs = tasks[e].actor_obs(&states[e], &reference);
                let critic_obs = tasks[e].critic_obs(&states[e], &reference);
                ac.record_obs(&obs, &critic_obs);
                let (action, logp, mean) = ac.sample(&obs, &mut rng);
                let value = ac.value(&critic_obs);
                let raw = action_array(&action);
                previous_actions.push(states[e].last_action);
                commands.push(tasks[e].joint_targets(&raw));
                samples[e].push(Sample {
                    obs,
                    critic_obs,
                    action,
                    mean_old: mean,
                    logp_old: logp,
                    value_old: value,
                    adv: 0.0,
                    ret: 0.0,
                });
                values[e].push(value);
            }

            let next = env.step_fullbody(&commands).await;
            for e in 0..num_envs {
                states[e] = next[e].state.clone();
                times[e] += 1.0 / CONTROL_HZ;
                let clip = &library.clips[clip_ids[e]];
                let reference = MotionReference::at(clip, times[e]);
                let reward = tasks[e].reward(&states[e], &reference, &previous_actions[e]);
                let done =
                    tasks[e].terminated(&states[e], &reference) || times[e] >= clip.duration();
                reward_sum += reward.total;
                rewards[e].push(reward.total);
                dones[e].push(done);
                if done {
                    terminations += 1;
                    env.reset_fullbody_env(e).await;
                    clip_ids[e] = (clip_ids[e] + 1 + iteration) % library.clips.len();
                    times[e] = 0.0;
                    states[e] = MotionState::default();
                    tasks[e].reset_history(&states[e]);
                }
            }
        }

        let mut batch = Vec::with_capacity(num_envs * ROLLOUT_STEPS);
        for e in 0..num_envs {
            let last_value = if dones[e].last().copied().unwrap_or(true) {
                0.0
            } else {
                let reference = MotionReference::at(&library.clips[clip_ids[e]], times[e]);
                let critic_obs = tasks[e].critic_obs(&states[e], &reference);
                ac.value(&critic_obs)
            };
            let (adv, ret) = gae(
                &rewards[e],
                &values[e],
                &dones[e],
                last_value,
                cfg.gamma,
                cfg.lam,
            );
            for (i, mut sample) in samples[e].drain(..).enumerate() {
                sample.adv = adv[i];
                sample.ret = ret[i];
                batch.push(sample);
            }
        }
        let stats = ac.update(&mut batch, &cfg);
        println!(
            "{:4}  reward {:7.3}  resets {:4}  kl {:.4}  value {:.4}  lr {:.2e}",
            iteration + 1,
            reward_sum / (num_envs * ROLLOUT_STEPS) as f32,
            terminations,
            stats.kl,
            stats.value_loss,
            stats.lr
        );
        if (iteration + 1) % 10 == 0 {
            ac.save(checkpoint)?;
        }
    }
    ac.save(checkpoint)?;
    println!("saved {checkpoint}");
    Ok(())
}

async fn evaluate(args: &[String]) -> Result<()> {
    let motion = args.first().context(usage())?;
    let checkpoint = args.get(1).context(usage())?;
    let output = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/tmp/sonic_wbc_rollout.json");
    let library = MotionLibrary::load(motion, Some(1))?;
    let clip = &library.clips[0];
    let ac = checked_policy(checkpoint)?;
    let mut env = make_env(1).await?;
    let mut task = MotionTrackingTask::default();
    let first = env.step_fullbody(&[clip.frames[0].joint_pos]).await;
    let mut state = first[0].state.clone();
    task.reset_history(&state);
    let (names, edges, feet) = env.skeleton();
    let mut frames = vec![first[0].body_positions.clone()];
    let mut time = 0.0f32;
    let mut squared_joint_error = 0.0f32;
    let mut squared_root_error = 0.0f32;
    let mut measured = 0usize;
    let mut terminated = false;

    while time < clip.duration() {
        let reference = MotionReference::at(clip, time);
        task.push_state(&state);
        let action = action_array(&ac.mean(&task.actor_obs(&state, &reference)));
        let previous_action = state.last_action;
        let out = env.step_fullbody(&[task.joint_targets(&action)]).await;
        state = out[0].state.clone();
        frames.push(out[0].body_positions.clone());
        time += 1.0 / CONTROL_HZ;
        let next_reference = MotionReference::at(clip, time);
        for j in 0..G1_CONTROLLED_JOINTS {
            squared_joint_error += (state.joint_pos[j] - next_reference.now.joint_pos[j]).powi(2);
        }
        for k in 0..3 {
            squared_root_error += (state.root_pos[k] - next_reference.now.root_pos[k]).powi(2);
        }
        measured += 1;
        let _ = task.reward(&state, &next_reference, &previous_action);
        if task.terminated(&state, &next_reference) {
            terminated = true;
            break;
        }
    }
    write_rollout(output, &names, &edges, &feet, &frames)?;
    let completion = if clip.duration() > 0.0 {
        (time / clip.duration()).min(1.0)
    } else {
        1.0
    };
    println!(
        "completion {:5.1}%{}  joint RMSE {:.4} rad  root RMSE {:.4} m",
        100.0 * completion,
        if terminated { " (terminated)" } else { "" },
        (squared_joint_error / (measured * G1_CONTROLLED_JOINTS).max(1) as f32).sqrt(),
        (squared_root_error / (measured * 3).max(1) as f32).sqrt()
    );
    println!("wrote {output}");
    Ok(())
}

fn write_rollout(
    path: &str,
    names: &[String],
    edges: &[(usize, usize)],
    feet: &[usize],
    frames: &[Vec<[f32; 3]>],
) -> Result<()> {
    fn strings(values: &[String]) -> String {
        values
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
            .join(",")
    }
    let mut json = String::new();
    write!(
        json,
        "{{\"dt\":{},\"names\":[{}],\"edges\":[",
        1.0 / CONTROL_HZ,
        strings(names)
    )?;
    for (i, (a, b)) in edges.iter().enumerate() {
        write!(json, "{}[{},{}]", if i == 0 { "" } else { "," }, a, b)?;
    }
    write!(json, "],\"feet\":{:?},\"resets\":[],\"frames\":[", feet)?;
    for (i, frame) in frames.iter().enumerate() {
        write!(json, "{}[", if i == 0 { "" } else { "," })?;
        for (j, p) in frame.iter().enumerate() {
            write!(
                json,
                "{}[{:.6},{:.6},{:.6}]",
                if j == 0 { "" } else { "," },
                p[0],
                p[1],
                p[2]
            )?;
        }
        json.push(']');
    }
    json.push_str("]}");
    std::fs::write(path, json).with_context(|| format!("write rollout {path}"))?;
    Ok(())
}
