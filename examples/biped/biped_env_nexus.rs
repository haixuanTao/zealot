//! Vectorized N-env biped environment on nexus GPU physics.
//!
//! `BipedNexusBatchEnv` owns one `GpuPhysicsState` holding N parallel envs and
//! the host-side bookkeeping each env needs (RNG, current command, step counter,
//! action history, air-time per foot). One `pipeline.step(...)` advances every
//! env on the GPU; one `slow_read_buffer(links_workspace)` brings the full
//! per-link state back to host where we compute obs/reward per env using the
//! same `VelocityFlatTask` the CPU env uses.
//!
//! What's mirrored from `biped_env.rs`:
//! - MJCF scene build (per env), foot box collider, PD motors, dynamic root.
//! - Per-env friction / restitution / contact-softness / PD-scale randomization
//!   (baked into the rapier scene + `GpuSimParams` before `from_rapier`).
//! - Episode-end reset via pre-built spawn templates + `state.reset_env_from`.
//!
//! What's NOT mirrored (nexus host API doesn't expose them):
//! - Push perturbations (no `apply_impulse` API on the GPU side).
//! - True foot-ground contact pairs (synthesized via foot Z < threshold).
//!
//! Joint angles / velocities, base linear / angular velocity all come from
//! `links_workspace[k].{coords, joint_rot, rb_vels}` (rb_vels is world-space).

use khal::backend::{Backend, Buffer, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose as NexusPose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use nexus3d::rbd::shaders::dynamics::MultibodyLinkWorkspace;
use rapier3d::prelude::*;
use roxmltree::Node;
use std::collections::HashMap;
use zealot_env::rng::Lcg;
use zealot_env::robots::LeRobotBipedal;
use zealot_env::robots::lerobot_bipedal::{JOINT_NAMES, NUM_JOINTS};
use zealot_env::tasks::velocity_flat::{
    BaseState, CRITIC_OBS_DIM, CommandSampler, FootObs, NUM_FEET, OBS_DIM, RobotState,
    VelocityCommand, VelocityFlatTask,
};

const SPAWN_Z: f32 = 0.72;
// Match the CPU env's `IntegrationParameters::num_solver_iterations = 8` — at 16
// the inner solver loop doubles the per-step kernel work for marginal stability
// gain at our timescales.
const SOLVER_ITERS: u32 = 8;

// --- MJCF parsing (duplicated from biped_env.rs — small, self-contained) ----

pub struct MjBody {
    #[allow(dead_code)]
    pub name: String,
    pub parent: Option<usize>,
    pub local_pos: Vec3,
    pub local_quat: Rotation,
    pub joint: Option<String>,
    pub com: Vec3,
    pub mass: f32,
    pub inertia_diag: Vec3,
    pub capsules: Vec<(Vec3, Vec3, f32)>,
}

fn floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}
fn vec3(node: &Node, attr: &str, default: Vec3) -> Vec3 {
    node.attribute(attr).map_or(default, |s| {
        let f = floats(s);
        Vec3::new(f[0], f[1], f[2])
    })
}
fn quat_wxyz(node: &Node) -> Rotation {
    node.attribute("quat").map_or(Rotation::IDENTITY, |s| {
        let f = floats(s);
        Rotation::from_xyzw(f[1], f[2], f[3], f[0]).normalize()
    })
}

fn parse_body(node: &Node, parent: Option<usize>, out: &mut Vec<MjBody>) {
    let mut joint = None;
    let mut is_free = false;
    let (mut com, mut mass, mut inertia_diag) = (Vec3::ZERO, 0.0, Vec3::splat(1e-4));
    let mut capsules = Vec::new();
    for c in node.children().filter(Node::is_element) {
        match c.tag_name().name() {
            "freejoint" => is_free = true,
            "joint" => joint = Some(c.attribute("name").unwrap_or("").to_string()),
            "inertial" => {
                com = vec3(&c, "pos", Vec3::ZERO);
                mass = c
                    .attribute("mass")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                if let Some(s) = c.attribute("fullinertia") {
                    let f = floats(s);
                    inertia_diag = Vec3::new(f[0], f[1], f[2]);
                }
            }
            "geom" if c.attribute("class") == Some("collision") => {
                if let Some(ft) = c.attribute("fromto") {
                    let f = floats(ft);
                    let r = floats(c.attribute("size").unwrap_or("0.01"))[0];
                    capsules.push((Vec3::new(f[0], f[1], f[2]), Vec3::new(f[3], f[4], f[5]), r));
                }
            }
            _ => {}
        }
    }
    let idx = out.len();
    let keep = parent.is_none() || joint.is_some() || is_free;
    if keep {
        out.push(MjBody {
            name: node.attribute("name").unwrap_or("").to_string(),
            parent,
            local_pos: vec3(node, "pos", Vec3::ZERO),
            local_quat: quat_wxyz(node),
            joint,
            com,
            mass,
            inertia_diag,
            capsules,
        });
    }
    let this = if keep { Some(idx) } else { parent };
    for c in node.children().filter(Node::is_element) {
        if c.tag_name().name() == "body" {
            parse_body(&c, this, out);
        }
    }
}

pub fn parse_mjcf(xml: &str) -> Vec<MjBody> {
    let doc = roxmltree::Document::parse(xml).expect("parse mjcf");
    let world = doc
        .descendants()
        .find(|n| n.tag_name().name() == "worldbody")
        .expect("worldbody");
    let mut out = Vec::new();
    for c in world.children().filter(Node::is_element) {
        if c.tag_name().name() == "body" {
            parse_body(&c, None, &mut out);
        }
    }
    out
}

pub fn default_mjcf_path() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml")
}

// --- Per-env scene parameters (the bits a single rapier scene needs) --------

/// Domain randomization knobs the GPU side CAN honour. Push-perturbation and
/// contact-pair readback are dropped vs the CPU `Randomization` struct.
///
/// Initial-pose fields (`joint_pos_noise`, `base_z_noise`, `base_tilt_noise`)
/// perturb each spawn template's starting configuration so the policy sees a
/// distribution of starts rather than the same neutral pose every episode.
/// Crucial for PPO to explore the relevant state space.
#[derive(Clone, Copy, Debug)]
pub struct DrParams {
    pub friction: f32,
    pub restitution: f32,
    pub pd_scale: f32,
    pub contact_natural_frequency: f32,
    pub contact_damping_ratio: f32,
    /// Sampled base orientation at spawn — separate axes so a single template
    /// can mix yaw / roll / pitch. Each in rad.
    pub spawn_yaw: f32,
    pub spawn_roll: f32,
    pub spawn_pitch: f32,
    /// Sampled additive jitter on `SPAWN_Z`, m. May be negative.
    pub spawn_z_offset: f32,
}

impl Default for DrParams {
    fn default() -> Self {
        Self {
            friction: 1.0,
            restitution: 0.0,
            pd_scale: 1.0,
            contact_natural_frequency: 30.0,
            contact_damping_ratio: 5.0,
            spawn_yaw: 0.0,
            spawn_roll: 0.0,
            spawn_pitch: 0.0,
            spawn_z_offset: 0.0,
        }
    }
}

/// Static per-env scene + index bookkeeping (kept once per env so we can
/// rebuild a fresh single-env GPU state for `reset_env_from`).
pub struct EnvScene {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse: ImpulseJointSet,
    pub multibody: MultibodyJointSet,
    pub sim_params: GpuSimParams,
}

/// Indices into the per-env link layout, common across every env (the topology
/// is identical, so these are computed once at the first scene build).
#[derive(Clone, Debug)]
pub struct LinkIndices {
    /// Number of multibody links per env (1 root + 12 leg children = 13).
    pub links_per_batch: u32,
    /// Number of generalized DOFs per env (6 root + 12 revolute = 18).
    pub dofs_per_batch: u32,
    /// Number of colliders per env (1 root + 12 inert/foot + 1 ground = 14).
    #[allow(dead_code)]
    pub colliders_per_batch: u32,
    /// Multibody link index of the torso (always 0 — the root).
    pub torso_link: u32,
    /// Multibody link indices of the two feet (assembly order).
    pub foot_links: [u32; NUM_FEET],
    /// (multibody_link_index, joint_name) for each actuated revolute. In
    /// `JOINT_NAMES` (canonical policy) order, so observation/action layout
    /// lines up with the CPU env.
    pub actuated: Vec<(u32, String)>,
    /// `(joint_idx_in_JOINT_NAMES, dof_offset_within_env)` for each leg joint.
    /// Root DOFs occupy 0..6; leg joints fill 6..18 in the order they were
    /// inserted into the multibody. Used to index into `dof_state` for joint
    /// angular velocities.
    pub joint_dof_offset: [u32; NUM_JOINTS],
    /// Foot sole-normal in foot-local frame at spawn (sole = +Z world there).
    pub foot_sole_local: [Vec3; NUM_FEET],
    /// Multibody link index for each MJCF body (in `mjcf: Vec<MjBody>` order).
    /// Used by `body_positions_for` to render the skeleton in MJCF order — the
    /// same order the CPU env's `body_positions()` returns and the python
    /// renderer (`render_biped.py`) expects.
    pub mjcf_to_link: Vec<u32>,
}

/// Build one env's rapier scene + sim params with the given DR sample.
/// Mirrors `biped_nexus.rs::build_scene` minus the Scene-id wrappers (we don't
/// need nexus_id lookups here — link indices are stable across envs).
fn build_env_scene(
    mjcf: &[MjBody],
    robot: &LeRobotBipedal,
    dr: &DrParams,
    task_dt: f32,
) -> (EnvScene, LinkIndices) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();

    // FK world poses with initial-pose jitter on the root: yaw + roll + pitch
    // + height. Joint angles stay at neutral (the multibody rest pose).
    // Composing intrinsic ZYX so yaw is the outermost rotation (the typical
    // RL convention — yaw randomises heading, roll/pitch perturb upright).
    let root_rot = Rotation::from_rotation_z(dr.spawn_yaw)
        * Rotation::from_rotation_y(dr.spawn_pitch)
        * Rotation::from_rotation_x(dr.spawn_roll);
    let root_pos = Vec3::new(0.0, 0.0, SPAWN_Z + dr.spawn_z_offset);
    let root_pose = Pose::from_parts(root_pos, root_rot);
    let mut world: Vec<Pose> = Vec::with_capacity(mjcf.len());
    for b in mjcf {
        let w = match b.parent {
            None => root_pose,
            Some(p) => world[p] * Pose::from_parts(b.local_pos, b.local_quat),
        };
        world.push(w);
    }

    let mut handles = Vec::with_capacity(mjcf.len());
    let mut torso_handle = RigidBodyHandle::invalid();
    let mut foot_handles: Vec<(usize, RigidBodyHandle)> = Vec::new();
    for (i, b) in mjcf.iter().enumerate() {
        let h = bodies.insert(
            RigidBodyBuilder::dynamic()
                .position(world[i])
                .additional_mass_properties(MassProperties::new(
                    b.com,
                    b.mass.max(1e-3),
                    b.inertia_diag,
                ))
                .build(),
        );
        handles.push(h);
        if b.parent.is_none() {
            torso_handle = h;
        }
        if b.capsules.is_empty() {
            // Inert placeholder (nexus requires exactly one collider per body).
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.01, 0.01, 0.01)
                    .density(0.0)
                    .collision_groups(InteractionGroups::none()),
                h,
                &mut bodies,
            );
        } else {
            // Foot box.
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for (a, c, r) in &b.capsules {
                lo = lo.min(a.min(*c) - Vec3::splat(*r));
                hi = hi.max(a.max(*c) + Vec3::splat(*r));
            }
            let he = ((hi - lo) * 0.5).max(Vec3::splat(1e-3));
            let center = (hi + lo) * 0.5;
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(he.x, he.y, he.z)
                    .position(Pose::from_parts(center, Rotation::IDENTITY))
                    .density(0.0)
                    .friction(dr.friction)
                    .restitution(dr.restitution),
                h,
                &mut bodies,
            );
            foot_handles.push((i, h));
        }
    }

    // Revolute multibody joints (free AngZ) — build in MJCF order, then reorder
    // to canonical JOINT_NAMES so action layout matches the CPU env.
    let locked = JointAxesMask::LIN_X
        | JointAxesMask::LIN_Y
        | JointAxesMask::LIN_Z
        | JointAxesMask::ANG_X
        | JointAxesMask::ANG_Y;
    // Track (mjcf_idx, joint_name) → link assembly index (monotone with insert
    // order, equals the rapier multibody link id).
    let mut mb_link_of_mjcf: HashMap<usize, u32> = HashMap::new();
    mb_link_of_mjcf.insert(0, 0); // torso is multibody root → link 0
    let mut next_mb_link: u32 = 1;
    let mut name_to_link: HashMap<String, u32> = HashMap::new();
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some(jname)) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let spec = robot.joints.iter().find(|j| &j.name == jname);
        let (kp, kd, effort) = spec
            .map(|s| (s.kp * dr.pd_scale, s.kd * dr.pd_scale, s.effort_limit))
            .unwrap_or((50.0, 1.0, 20.0));
        let mut joint = GenericJointBuilder::new(locked)
            .local_frame1(Pose::from_parts(b.local_pos, b.local_quat))
            .local_frame2(Pose::IDENTITY)
            .build();
        joint.set_motor_model(JointAxis::AngZ, MotorModel::ForceBased);
        joint.set_motor_position(JointAxis::AngZ, 0.0, kp, kd);
        joint.set_motor_max_force(JointAxis::AngZ, effort);
        multibody.insert(handles[parent], handles[i], joint, true);
        mb_link_of_mjcf.insert(i, next_mb_link);
        name_to_link.insert(jname.clone(), next_mb_link);
        next_mb_link += 1;
    }

    // Ground (Z-up).
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5)
            .friction(dr.friction)
            .restitution(dr.restitution),
        ground,
        &mut bodies,
    );

    // Rapier's `local_mprops` is populated by its step pipeline; we hand the
    // scene to nexus without stepping rapier first, so call recompute here. See
    // `biped_nexus.rs` module docs / dimforge/nexus-rustgpu#1 follow-up.
    let colliders_snapshot = colliders.clone();
    for (_, rb) in bodies.iter_mut() {
        rb.recompute_mass_properties_from_colliders(&colliders_snapshot);
    }

    // Sim params: per-env contact softness via DR.
    let mut sp = GpuSimParams::default();
    sp.dt = task_dt;
    sp.num_solver_iterations = SOLVER_ITERS;
    sp.contact_natural_frequency = dr.contact_natural_frequency;
    sp.contact_damping_ratio = dr.contact_damping_ratio;

    // Build the index table from the canonical joint ordering.
    let mut actuated: Vec<(u32, String)> = Vec::with_capacity(NUM_JOINTS);
    let mut joint_dof_offset = [0u32; NUM_JOINTS];
    for (k, &name) in JOINT_NAMES.iter().enumerate() {
        let link = *name_to_link
            .get(name)
            .unwrap_or_else(|| panic!("missing joint {name} in MJCF"));
        actuated.push((link, name.to_string()));
        // Each leg joint has 1 DOF and sits at offset (6 root DOFs + insertion order).
        // Insertion order = link - 1 (since torso is link 0).
        joint_dof_offset[k] = 6 + (link - 1);
    }

    // Sole-normal in foot-local frame at spawn (sole = world +Z, so the local
    // sole-normal is R_spawn⁻¹·Z).
    let mut foot_sole_local = [Vec3::Z; NUM_FEET];
    let mut foot_links = [0u32; NUM_FEET];
    for (i, (mjcf_idx, h)) in foot_handles.iter().enumerate() {
        let link = *mb_link_of_mjcf.get(mjcf_idx).unwrap_or(&0);
        foot_links[i] = link;
        foot_sole_local[i] = bodies[*h].rotation().conjugate() * Vec3::Z;
    }

    let mjcf_to_link: Vec<u32> = (0..mjcf.len())
        .map(|i| *mb_link_of_mjcf.get(&i).unwrap_or(&0))
        .collect();

    let idx = LinkIndices {
        links_per_batch: next_mb_link, // 1 (torso) + 12 (legs) = 13
        dofs_per_batch: 6 + NUM_JOINTS as u32,
        colliders_per_batch: (mjcf.len() + 1) as u32, // robot bodies + ground
        torso_link: 0,
        foot_links,
        actuated,
        joint_dof_offset,
        foot_sole_local,
        mjcf_to_link,
    };

    let _ = torso_handle;
    (
        EnvScene {
            bodies,
            colliders,
            impulse,
            multibody,
            sim_params: sp,
        },
        idx,
    )
}

// --- The batched env ---------------------------------------------------------

/// Outcome of one control step for one env (same shape as `BipedEnv::StepOut`).
pub struct StepOut {
    pub obs: Vec<f32>,
    pub critic_obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub fell: bool,
}

/// One vectorized env over nexus GPU physics.
///
/// All N envs share a single `GpuPhysicsState`. Per-env host state (RNG,
/// command, step counter, action history, air-time, sole-normals) lives in
/// parallel vectors keyed by env index. Reset uses pre-built single-env spawn
/// templates and `state.reset_env_from(env_i, template)`.
pub struct BipedNexusBatchEnv {
    // Topology + indexing
    mjcf: Vec<MjBody>,
    robot: LeRobotBipedal,
    task: VelocityFlatTask,
    idx: LinkIndices,

    // Per-env host state
    n: usize,
    rng: Vec<Lcg>,
    sampler: CommandSampler,
    cmd: Vec<VelocityCommand>,
    step_count: Vec<u32>,
    resample_at: Vec<u32>,
    last_action: Vec<[f32; NUM_JOINTS]>,
    prev_action: Vec<[f32; NUM_JOINTS]>,
    air_time: Vec<[f32; NUM_FEET]>,
    /// Previous control-step joint angles per env. Used to compute joint
    /// velocities by finite-diff `(q_now - q_prev) / control_dt` instead of
    /// reading nexus's `dof_state` buffer — saves one slow_read per step.
    /// Initialised lazily to the first-step coords so step 1's vel is 0.
    prev_joint_pos: Vec<[f32; NUM_JOINTS]>,
    has_prev_joint_pos: Vec<bool>,
    /// Per-env foot-local sole-normal (depends on the spawn template that
    /// seeded the env — we keep one copy per env, updated on reset).
    foot_sole_local: Vec<[Vec3; NUM_FEET]>,
    /// Default sampler (full ranges) — kept so `set_command_scale` can derive
    /// scaled ranges from a known baseline, mirroring the CPU env.
    sampler_default: CommandSampler,

    // GPU state
    gpu: KhalGpuBackend,
    pipeline: GpuPhysicsPipeline,
    state: GpuPhysicsState,

    // Pre-built spawn templates for reset_env_from (different DR samples).
    templates: Vec<GpuPhysicsState>,
    template_dr: Vec<DrParams>,
}

impl BipedNexusBatchEnv {
    /// Build N envs sharing one batched GpuPhysicsState. `num_templates` controls
    /// how many distinct DR samples are pre-built and cycled across the N envs
    /// at construction and reset time (higher = better coverage, more GPU mem).
    pub async fn new(mjcf_xml: &str, num_envs: usize, num_templates: usize, seed: u64) -> Self {
        let mjcf = parse_mjcf(mjcf_xml);
        let robot = LeRobotBipedal::new();
        let task = VelocityFlatTask::new();

        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);

        // Sample DR for the templates first (each defines one rapier scene).
        let mut tpl_rng = Lcg::new(seed);
        let mut template_dr: Vec<DrParams> = (0..num_templates)
            .map(|_| sample_dr(&mut tpl_rng))
            .collect();
        // Always include one DR-OFF template at index 0 — keeps deterministic
        // replay possible and provides a stable initialiser.
        template_dr[0] = DrParams::default();

        // Build the per-env scenes — cycle across the templates so envs get
        // mixed DR from the start. We keep the LinkIndices from the first one
        // (topology is invariant).
        let mut idx_out: Option<LinkIndices> = None;
        let mut env_scenes: Vec<EnvScene> = Vec::with_capacity(num_envs);
        for e in 0..num_envs {
            let dr = template_dr[e % num_templates];
            let (scene, ix) = build_env_scene(&mjcf, &robot, &dr, task.sim_dt);
            if idx_out.is_none() {
                idx_out = Some(ix);
            }
            env_scenes.push(scene);
        }
        let idx = idx_out.expect("at least one env");

        // Batched from_rapier.
        let envs_refs: Vec<_> = env_scenes
            .iter()
            .map(|s| {
                (
                    &s.bodies,
                    &s.colliders,
                    &s.impulse,
                    &s.multibody,
                    &s.sim_params,
                )
            })
            .collect();
        let mut state = GpuPhysicsState::from_rapier(&gpu, &envs_refs);
        state.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);

        // Spawn templates: one single-env GPU state per DR sample.
        let mut templates: Vec<GpuPhysicsState> = Vec::with_capacity(num_templates);
        for dr in &template_dr {
            let (scene, _) = build_env_scene(&mjcf, &robot, dr, task.sim_dt);
            let envs_refs = vec![(
                &scene.bodies,
                &scene.colliders,
                &scene.impulse,
                &scene.multibody,
                &scene.sim_params,
            )];
            let mut tpl = GpuPhysicsState::from_rapier(&gpu, &envs_refs);
            tpl.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);
            templates.push(tpl);
        }

        // Per-env initial sole-normal: every env starts from the corresponding
        // template, so its foot_sole_local matches that template's at-spawn
        // computation. Re-derive by re-building the rapier scene with the same
        // DR (cheap; bodies are tiny).
        let mut foot_sole_local: Vec<[Vec3; NUM_FEET]> = Vec::with_capacity(num_envs);
        for e in 0..num_envs {
            let dr = template_dr[e % num_templates];
            let (_, ix) = build_env_scene(&mjcf, &robot, &dr, task.sim_dt);
            foot_sole_local.push(ix.foot_sole_local);
        }

        let cmd = vec![VelocityCommand::default(); num_envs];
        let step_count = vec![0u32; num_envs];
        let resample_at = vec![0u32; num_envs];
        let last_action = vec![[0.0f32; NUM_JOINTS]; num_envs];
        let prev_action = vec![[0.0f32; NUM_JOINTS]; num_envs];
        let air_time = vec![[0.0f32; NUM_FEET]; num_envs];
        let prev_joint_pos = vec![[0.0f32; NUM_JOINTS]; num_envs];
        let has_prev_joint_pos = vec![false; num_envs];
        let rng: Vec<Lcg> = (0..num_envs)
            .map(|e| Lcg::new(seed ^ ((e as u64).wrapping_mul(2654435761))))
            .collect();
        let sampler = CommandSampler::default();
        let sampler_default = CommandSampler::default();

        let mut env = Self {
            mjcf,
            robot,
            task,
            idx,
            n: num_envs,
            rng,
            sampler,
            cmd,
            step_count,
            resample_at,
            last_action,
            prev_action,
            air_time,
            prev_joint_pos,
            has_prev_joint_pos,
            foot_sole_local,
            sampler_default,
            gpu,
            pipeline,
            state,
            templates,
            template_dr,
        };
        // Seed every env's command and resample schedule (mirrors `reset_full`
        // on the CPU side without an actual GPU reset — the GPU state is
        // already at the correct spawn pose from `from_rapier`).
        for e in 0..num_envs {
            env.cmd[e] = env.sampler.sample(&mut env.rng[e]);
            env.resample_at[e] = env
                .sampler
                .resample_steps(&mut env.rng[e], env.task.control_dt());
        }
        env
    }

    #[allow(dead_code)]
    pub fn num_envs(&self) -> usize {
        self.n
    }

    pub fn obs_dim(&self) -> usize {
        OBS_DIM
    }

    pub fn critic_obs_dim(&self) -> usize {
        CRITIC_OBS_DIM
    }

    pub fn action_dim(&self) -> usize {
        NUM_JOINTS
    }

    /// Curriculum hook — scales every env's command range by `s` (mirrors the
    /// CPU env: shrinks `lin_vel_x`/`lin_vel_y`/`ang_vel_z` proportionally).
    pub fn set_command_scale(&mut self, s: f32) {
        let s = s.clamp(0.0, 1.0);
        let d = &self.sampler_default;
        self.sampler.lin_vel_x = (d.lin_vel_x.0 * s, d.lin_vel_x.1 * s);
        self.sampler.lin_vel_y = (d.lin_vel_y.0 * s, d.lin_vel_y.1 * s);
        self.sampler.ang_vel_z = (d.ang_vel_z.0 * s, d.ang_vel_z.1 * s);
    }

    /// Read every link's workspace + every body's world pose for ALL envs.
    /// `ws.rb_vels` carries velocities (only valid after the first FK pass);
    /// `body_poses` carries world positions/orientations and is correct at all
    /// times (initialised by `from_rapier`, refreshed by FK each step). Joint
    /// velocities are reconstructed from successive `ws.coords[5]` via
    /// finite-diff in `read_state`, so we skip the `dof_state` readback (also
    /// untrustworthy per dimforge/nexus-rustgpu#1).
    async fn slurp_state(&mut self) -> (Vec<MultibodyLinkWorkspace>, Vec<NexusPose>) {
        let mut ws: Vec<MultibodyLinkWorkspace> = vec![
            unsafe { std::mem::zeroed() };
            self.state
                .multibodies_mut()
                .links_workspace()
                .buffer()
                .len()
        ];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().links_workspace().buffer(),
                &mut ws,
            )
            .await
            .expect("links_workspace readback");
        let mut poses: Vec<NexusPose> =
            vec![NexusPose::default(); self.state.body_poses().buffer().len()];
        self.gpu
            .slow_read_buffer(self.state.body_poses().buffer(), &mut poses)
            .await
            .expect("body_poses readback");
        (ws, poses)
    }

    /// Build the per-env `RobotState` from a workspace + body_poses slurp.
    /// Joint angles from `ws.coords[5]` (AngZ revolute); joint velocities by
    /// finite-diff against the cached previous joint angles (zero velocity on
    /// first step). Base pose from `body_poses` (always correct, including at
    /// step 0); base linear/angular velocity from `ws.rb_vels` (zero before any
    /// pipeline.step runs — fine, the biped starts at rest).
    fn read_state(
        &mut self,
        env: usize,
        ws: &[MultibodyLinkWorkspace],
        poses: &[NexusPose],
    ) -> RobotState {
        let lpb = self.idx.links_per_batch as usize;
        let cpb = self.idx.colliders_per_batch as usize;
        let env_base_ws = env * lpb;
        let env_base_pose = env * cpb;
        let control_dt = self.task.control_dt();

        let torso_pose = &poses[env_base_pose + self.idx.torso_link as usize];
        let t = torso_pose.translation;
        let r = torso_pose.rotation;
        let root_ws = &ws[env_base_ws + self.idx.torso_link as usize];
        let lv = root_ws.rb_vels.linear;
        let av = root_ws.rb_vels.angular;
        let base = BaseState {
            orientation: [r.x, r.y, r.z, r.w],
            lin_vel_world: [lv.x, lv.y, lv.z],
            ang_vel_world: [av.x, av.y, av.z],
            height: t.z,
        };

        let mut joint_pos = [0.0f32; NUM_JOINTS];
        let mut joint_vel = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            let link = self.idx.actuated[k].0 as usize;
            joint_pos[k] = ws[env_base_ws + link].coords[5];
        }
        if self.has_prev_joint_pos[env] {
            for k in 0..NUM_JOINTS {
                joint_vel[k] = (joint_pos[k] - self.prev_joint_pos[env][k]) / control_dt;
            }
        }
        self.prev_joint_pos[env] = joint_pos;
        self.has_prev_joint_pos[env] = true;

        RobotState {
            base,
            joint_pos,
            joint_vel,
            last_action: self.last_action[env],
            prev_action: self.prev_action[env],
            feet: [FootObs::default(); NUM_FEET],
        }
    }

    /// Compute per-foot observation for one env from the slurp + advance the
    /// per-foot air-time counter. Positions come from `body_poses` (always
    /// correct), velocities from `ws.rb_vels` (zero before first step — fine,
    /// biped starts at rest). Contact is synthesised: foot below a small Z
    /// threshold (nexus doesn't expose narrow-phase pairs).
    fn update_feet(
        &mut self,
        env: usize,
        ws: &[MultibodyLinkWorkspace],
        poses: &[NexusPose],
    ) -> [FootObs; NUM_FEET] {
        const CONTACT_Z: f32 = 0.025;
        let dt = self.task.control_dt();
        let lpb = self.idx.links_per_batch as usize;
        let cpb = self.idx.colliders_per_batch as usize;
        let env_base_ws = env * lpb;
        let env_base_pose = env * cpb;

        let base_rot = poses[env_base_pose + self.idx.torso_link as usize].rotation;
        let base_rot_inv = base_rot.conjugate();
        let sole_local = self.foot_sole_local[env];
        let mut out = [FootObs::default(); NUM_FEET];
        for i in 0..NUM_FEET {
            let link = self.idx.foot_links[i] as usize;
            let foot_pose = &poses[env_base_pose + link];
            let foot_ws = &ws[env_base_ws + link];
            let pos = foot_pose.translation;
            let lv = foot_ws.rb_vels.linear;
            let world_normal = foot_pose.rotation * sole_local[i];
            let tilt = world_normal.z.abs().clamp(0.0, 1.0).acos();
            let foot_x_in_base = (base_rot_inv * foot_pose.rotation) * Vec3::X;
            let yaw_rel_base = foot_x_in_base.y.atan2(foot_x_in_base.x);
            let contact = pos.z < CONTACT_Z;
            let prev_air = self.air_time[env][i];
            let first_contact = contact && prev_air > 0.0;
            self.air_time[env][i] = if contact { 0.0 } else { prev_air + dt };
            out[i] = FootObs {
                contact,
                first_contact,
                air_time: if contact {
                    prev_air
                } else {
                    self.air_time[env][i]
                },
                height: pos.z,
                planar_speed: (lv.x * lv.x + lv.y * lv.y).sqrt(),
                tilt,
                yaw_rel_base,
                pos_xy: [pos.x, pos.y],
            };
        }
        out
    }

    /// Step every env one control tick. Returns per-env `StepOut`s in
    /// env-index order. Async because both pipeline.step and the readback are
    /// async on the WebGPU backend.
    pub async fn step(&mut self, actions: &[[f32; NUM_JOINTS]]) -> Vec<StepOut> {
        assert_eq!(actions.len(), self.n);

        // Stage every env's motor targets host-side in the mirror, then push
        // the whole `links_static` buffer in ONE write_buffer call. Replaces
        // `num_envs * NUM_JOINTS` per-step write_buffer calls.
        for e in 0..self.n {
            let targets = self.task.joint_targets(&actions[e]);
            for k in 0..NUM_JOINTS {
                let link = self.idx.actuated[k].0;
                self.state.multibodies_mut().stage_motor_position(
                    e as u32,
                    link,
                    JointAxis::AngZ,
                    targets[k],
                );
            }
        }
        self.state
            .multibodies_mut()
            .flush_links_static(&self.gpu)
            .expect("flush motor targets");
        // Advance physics at the control decimation.
        for _ in 0..self.task.decimation {
            let _ = self.pipeline.step(&self.gpu, &mut self.state, None).await;
        }
        self.gpu.synchronize().expect("sync");
        self.pipeline
            .auto_resize_buffers(&self.gpu, &mut self.state)
            .await;

        let (ws, poses) = self.slurp_state().await;

        let mut outs = Vec::with_capacity(self.n);
        for e in 0..self.n {
            let feet = self.update_feet(e, &ws, &poses);
            let mut state = self.read_state(e, &ws, &poses);
            state.feet = feet;

            // Resample command on schedule.
            self.step_count[e] += 1;
            if self.step_count[e] >= self.resample_at[e] {
                self.cmd[e] = self.sampler.sample(&mut self.rng[e]);
                self.resample_at[e] = self.step_count[e]
                    + self
                        .sampler
                        .resample_steps(&mut self.rng[e], self.task.control_dt());
            }

            let fell = self.task.fell_over(&state.base) || !state.base.height.is_finite();
            let timeout = self.step_count[e] >= self.task.max_steps();
            let mut reward = self.task.reward(&state, &self.cmd[e]).total();
            if fell {
                reward += self.task.weights.termination;
            }

            let mut obs = vec![0.0; OBS_DIM];
            self.task.observe(&state, &self.cmd[e], &mut obs);
            let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
            self.task
                .observe_critic(&state, &self.cmd[e], &mut critic_obs);

            self.prev_action[e] = self.last_action[e];
            self.last_action[e] = actions[e];

            outs.push(StepOut {
                obs,
                critic_obs,
                reward,
                done: fell || timeout,
                fell,
            });
        }
        outs
    }

    /// Reset one env by copying a randomly-chosen spawn template into its slot.
    /// Returns the fresh obs / critic_obs for that env.
    pub async fn reset_env(&mut self, env: usize) -> (Vec<f32>, Vec<f32>) {
        // Pick a template via this env's RNG so reset choices are deterministic
        // for a given seed.
        let r = self.rng[env].range(0.0, 1.0);
        let t = ((r * self.templates.len() as f32) as usize).min(self.templates.len() - 1);
        self.state
            .reset_env_from(&self.gpu, env as u32, &self.templates[t])
            .await;
        // Mirror the template's sole-normal so update_feet's tilt makes sense.
        let dr = self.template_dr[t];
        let (_, ix) = build_env_scene(&self.mjcf, &self.robot, &dr, self.task.sim_dt);
        self.foot_sole_local[env] = ix.foot_sole_local;

        // Reset host state.
        self.cmd[env] = self.sampler.sample(&mut self.rng[env]);
        self.step_count[env] = 0;
        self.resample_at[env] = self
            .sampler
            .resample_steps(&mut self.rng[env], self.task.control_dt());
        self.last_action[env] = [0.0; NUM_JOINTS];
        self.prev_action[env] = [0.0; NUM_JOINTS];
        self.air_time[env] = [0.0; NUM_FEET];

        // Cached prev joint angles are stale across a reset; clear so the next
        // step seeds them again with zero velocity.
        self.has_prev_joint_pos[env] = false;
        // Build the initial obs from the freshly-copied state.
        let (ws, poses) = self.slurp_state().await;
        let feet = self.update_feet(env, &ws, &poses);
        let mut state = self.read_state(env, &ws, &poses);
        state.feet = feet;
        let mut obs = vec![0.0; OBS_DIM];
        self.task.observe(&state, &self.cmd[env], &mut obs);
        let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
        self.task
            .observe_critic(&state, &self.cmd[env], &mut critic_obs);
        (obs, critic_obs)
    }

    /// Bulk fresh-reset: rebuild every env's obs (no GPU reset — caller uses
    /// this once after construction to seed the policy loop).
    pub async fn initial_obs(&mut self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (ws, poses) = self.slurp_state().await;
        let mut obs = Vec::with_capacity(self.n);
        let mut critic_obs = Vec::with_capacity(self.n);
        for e in 0..self.n {
            let feet = self.update_feet(e, &ws, &poses);
            let mut state = self.read_state(e, &ws, &poses);
            state.feet = feet;
            let mut o = vec![0.0; OBS_DIM];
            self.task.observe(&state, &self.cmd[e], &mut o);
            let mut c = vec![0.0; CRITIC_OBS_DIM];
            self.task.observe_critic(&state, &self.cmd[e], &mut c);
            obs.push(o);
            critic_obs.push(c);
        }
        (obs, critic_obs)
    }

    // --- Render-recording helpers (mirror BipedEnv's body_positions /
    // base_pose / joint_angles / skeleton API on top of `links_workspace`). ---

    /// Reset env `e` to the deterministic (DR-OFF) template at index 0 — the
    /// one `BipedNexusBatchEnv::new` always installs there. Use this before
    /// a rendering rollout so the recorded trajectory doesn't drift on the
    /// per-env DR sample the env was originally seeded with.
    pub async fn reset_env_to_default_template(&mut self, e: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(!self.templates.is_empty());
        self.state
            .reset_env_from(&self.gpu, e as u32, &self.templates[0])
            .await;
        self.foot_sole_local[e] = self.idx.foot_sole_local;
        self.cmd[e] = VelocityCommand::default();
        self.step_count[e] = 0;
        // Pin the resample so the command stays where the caller pins it.
        self.resample_at[e] = u32::MAX;
        self.last_action[e] = [0.0; NUM_JOINTS];
        self.prev_action[e] = [0.0; NUM_JOINTS];
        self.air_time[e] = [0.0; NUM_FEET];
        self.has_prev_joint_pos[e] = false;
        let (ws, poses) = self.slurp_state().await;
        let feet = self.update_feet(e, &ws, &poses);
        let mut state = self.read_state(e, &ws, &poses);
        state.feet = feet;
        let mut obs = vec![0.0; OBS_DIM];
        self.task.observe(&state, &self.cmd[e], &mut obs);
        let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
        self.task
            .observe_critic(&state, &self.cmd[e], &mut critic_obs);
        (obs, critic_obs)
    }

    /// Pin env `e`'s commanded velocity to a fixed `(vx, vy, yaw)` — overrides
    /// the resample schedule so the command stays put. Mirrors
    /// `BipedEnv::pin_command`.
    pub fn pin_command_for(&mut self, e: usize, vx: f32, vy: f32, yaw: f32) {
        self.cmd[e] = VelocityCommand {
            vx,
            vy,
            yaw_rate: yaw,
        };
        self.resample_at[e] = u32::MAX;
    }

    /// World-space positions of every MJCF body for env `e`, returned in MJCF
    /// order (matches `BipedEnv::body_positions` so the python renderer reads
    /// both the same way). Reads from `body_poses` — correct at all times,
    /// including step 0 (before any FK has run).
    pub fn body_positions_for(&self, e: usize, poses: &[NexusPose]) -> Vec<[f32; 3]> {
        let cpb = self.idx.colliders_per_batch as usize;
        let base = e * cpb;
        // MJCF body i has collider index i (we insert one collider per body in
        // mjcf order), so its body_poses index is base + i.
        (0..self.idx.mjcf_to_link.len())
            .map(|i| {
                let t = poses[base + i].translation;
                [t.x, t.y, t.z]
            })
            .collect()
    }

    /// `(position, quaternion xyzw)` of the torso for env `e`. Mirrors
    /// `BipedEnv::base_pose`.
    pub fn base_pose_for(&self, e: usize, poses: &[NexusPose]) -> ([f32; 3], [f32; 4]) {
        let cpb = self.idx.colliders_per_batch as usize;
        let pose = &poses[e * cpb + self.idx.torso_link as usize];
        let t = pose.translation;
        let r = pose.rotation;
        ([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
    }

    /// Joint angles (rad) in `JOINT_NAMES` order for env `e`. Read straight
    /// from `ws.coords[5]` — the integrated coord for an AngZ-free revolute.
    pub fn joint_angles_for(&self, e: usize, ws: &[MultibodyLinkWorkspace]) -> [f32; NUM_JOINTS] {
        let lpb = self.idx.links_per_batch as usize;
        let base = e * lpb;
        let mut q = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            let link = self.idx.actuated[k].0 as usize;
            q[k] = ws[base + link].coords[5];
        }
        q
    }

    /// Kinematic tree for the skeleton renderer: `(link names, parent→child
    /// edges, foot link indices)`, all indexed in MJCF order (mirrors
    /// `BipedEnv::skeleton`).
    pub fn skeleton(&self) -> (Vec<String>, Vec<(usize, usize)>, Vec<usize>) {
        let names: Vec<String> = self.mjcf.iter().map(|b| b.name.clone()).collect();
        let edges: Vec<(usize, usize)> = self
            .mjcf
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.parent.map(|p| (p, i)))
            .collect();
        let feet: Vec<usize> = self
            .mjcf
            .iter()
            .enumerate()
            .filter_map(|(i, b)| (!b.capsules.is_empty()).then_some(i))
            .collect();
        (names, edges, feet)
    }

    /// One slurped snapshot for rendering — returns `(ws, body_poses)`. Use
    /// `body_positions_for` / `base_pose_for` with `body_poses` (correct at all
    /// times); use `joint_angles_for` with `ws` for the integrated joint coords.
    pub async fn snapshot(&mut self) -> (Vec<MultibodyLinkWorkspace>, Vec<NexusPose>) {
        self.slurp_state().await
    }

    /// Telemetry: torso heights across all envs.
    pub async fn torso_heights(&mut self) -> Vec<f32> {
        let (_, poses) = self.slurp_state().await;
        (0..self.n)
            .map(|e| {
                let i = e * self.idx.colliders_per_batch as usize + self.idx.torso_link as usize;
                poses[i].translation.z
            })
            .collect()
    }
}

// --- Helpers -----------------------------------------------------------------

async fn webgpu_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut w = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("webgpu");
    w.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(w)
}

/// Sample one DR point. Ranges mirror `Randomization::default()` from the CPU
/// env (minus push perturbations, which nexus can't apply at runtime).
/// Initial-pose jitter ranges are conservative — wider tilts make every
/// episode start mid-fall, which the policy can't recover from at small T.
fn sample_dr(rng: &mut Lcg) -> DrParams {
    DrParams {
        friction: rng.range(0.5, 1.5),
        restitution: rng.range(0.0, 0.15),
        pd_scale: rng.range(0.85, 1.15),
        contact_natural_frequency: rng.range(10.0, 50.0),
        contact_damping_ratio: rng.range(2.0, 8.0),
        spawn_yaw: rng.range(-std::f32::consts::PI, std::f32::consts::PI),
        spawn_roll: rng.range(-0.10, 0.10),     // ±~6°
        spawn_pitch: rng.range(-0.10, 0.10),    // ±~6°
        spawn_z_offset: rng.range(-0.03, 0.03), // ±3 cm
    }
}
