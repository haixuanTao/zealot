//! The vectorizable flat-velocity environment for the LeRobot bipedal on rapier
//! CPU, built from the curated MuJoCo model (see `biped_mjcf.rs` for the bring-up
//! rationale). Used by `biped_train.rs`.
//!
//! `BipedEnv::step` applies a 12-dim policy action as per-joint PD position
//! targets, advances rapier at the control decimation, then reads back joint and
//! base state to build the observation/reward/termination via the shared
//! `VelocityFlatTask` MDP. `reset` rebuilds the world at a fixed forward yaw by
//! default (low randomization for a clean first training; see `set_spawn_yaw_range`).

#![allow(dead_code)]

use rapier3d::prelude::*;
use rayon::prelude::*;
use roxmltree::Node;
use zealot_env::rng::Lcg;
use zealot_env::robots::LeRobotBipedal;
use zealot_env::robots::lerobot_bipedal::{JOINT_NAMES, NUM_JOINTS};
use zealot_env::tasks::velocity_flat::{
    BaseState, CRITIC_OBS_DIM, CommandSampler, FootObs, NUM_FEET, OBS_DIM, RobotState,
    VelocityCommand, VelocityFlatTask,
};
use zealot_rl::{ActorCritic, PpoConfig, Sample, gae};

// ---------------------------------------------------------------------------
// MJCF parsing (focused reader for this model — same as biped_mjcf.rs).
// ---------------------------------------------------------------------------

struct MjBody {
    name: String,
    parent: Option<usize>,
    local_pos: Vec3,
    local_quat: Rotation,
    joint: Option<String>,
    com: Vec3,
    mass: f32,
    inertia_diag: Vec3,
    capsules: Vec<(Vec3, Vec3, f32)>,
}

fn floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}
fn vec3(node: &Node, attr: &str, default: Vec3) -> Vec3 {
    match node.attribute(attr) {
        Some(s) => {
            let f = floats(s);
            Vec3::new(f[0], f[1], f[2])
        }
        None => default,
    }
}
fn quat_wxyz(node: &Node) -> Rotation {
    match node.attribute("quat") {
        Some(s) => {
            let f = floats(s);
            Rotation::from_xyzw(f[1], f[2], f[3], f[0]).normalize()
        }
        None => Rotation::IDENTITY,
    }
}

fn parse_body(node: &Node, parent: Option<usize>, out: &mut Vec<MjBody>) {
    let mut joint = None;
    let mut is_free = false;
    let (mut com, mut mass, mut inertia_diag) = (Vec3::ZERO, 0.0, Vec3::splat(1e-4));
    let mut capsules = Vec::new();
    for child in node.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "freejoint" => is_free = true,
            "joint" => joint = Some(child.attribute("name").unwrap_or("").to_string()),
            "inertial" => {
                com = vec3(&child, "pos", Vec3::ZERO);
                mass = child
                    .attribute("mass")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                if let Some(s) = child.attribute("fullinertia") {
                    let f = floats(s);
                    inertia_diag = Vec3::new(f[0], f[1], f[2]);
                }
            }
            "geom" if child.attribute("class") == Some("collision") => {
                if let Some(ft) = child.attribute("fromto") {
                    let f = floats(ft);
                    let r = floats(child.attribute("size").unwrap_or("0.01"))[0];
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
    for child in node.children().filter(Node::is_element) {
        if child.tag_name().name() == "body" {
            parse_body(&child, this, out);
        }
    }
}

fn parse_mjcf(xml: &str) -> Vec<MjBody> {
    let doc = roxmltree::Document::parse(xml).expect("parse mjcf");
    let world = doc
        .descendants()
        .find(|n| n.tag_name().name() == "worldbody")
        .expect("worldbody");
    let mut out = Vec::new();
    for child in world.children().filter(Node::is_element) {
        if child.tag_name().name() == "body" {
            parse_body(&child, None, &mut out);
        }
    }
    out
}

/// Default MJCF path for the LeRobot bipedal (the deployed policy's model).
pub fn default_mjcf_path() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml")
}

// ---------------------------------------------------------------------------
// World + environment.
// ---------------------------------------------------------------------------

/// Per-joint runtime references, in policy (`JOINT_NAMES`) order.
struct JointRef {
    handle: MultibodyJointHandle,
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    rest_quat: Rotation,
    kp: f32,
    kd: f32,
    effort: f32,
}

struct World {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    islands: IslandManager,
    bp: BroadPhaseBvh,
    np: NarrowPhase,
    ccd: CCDSolver,
    pipeline: PhysicsPipeline,
    joints: Vec<JointRef>,
    torso: RigidBodyHandle,
    feet: Vec<RigidBodyHandle>,
    ground_collider: ColliderHandle,
    /// All robot link bodies, in MJCF parse order (for rendering).
    all_handles: Vec<RigidBodyHandle>,
}

const SPAWN_Z: f32 = 0.72;

fn build_world(mjcf: &[MjBody], robot: &LeRobotBipedal, base_yaw: f32) -> World {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();

    // FK world poses (root lifted to SPAWN_Z with a yaw about +Z).
    let mut world: Vec<Pose> = Vec::with_capacity(mjcf.len());
    for b in mjcf {
        let w = match b.parent {
            None => Pose::from_parts(
                Vec3::new(0.0, 0.0, SPAWN_Z),
                Rotation::from_rotation_z(base_yaw),
            ),
            Some(p) => world[p] * Pose::from_parts(b.local_pos, b.local_quat),
        };
        world.push(w);
    }

    let mut handles = Vec::with_capacity(mjcf.len());
    let mut feet = Vec::new();
    let mut torso = RigidBodyHandle::invalid();
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
            torso = h;
        }
        if !b.capsules.is_empty() {
            feet.push(h);
            // ONE solid foot box covering the sole footprint, NOT the MJCF's 6 thin
            // capsules: rapier makes an unstable line-contact on thin capsules (the
            // feet pop and never rest, so the robot gets no ground support and
            // collapses). A flat box gives stable area contact so the feet plant.
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
                    .friction(1.0),
                h,
                &mut bodies,
            );
        }
    }

    // Revolute joints (free child-local AngZ) with PD motors. Collect refs by name.
    let locked = JointAxesMask::LIN_X
        | JointAxesMask::LIN_Y
        | JointAxesMask::LIN_Z
        | JointAxesMask::ANG_X
        | JointAxesMask::ANG_Y;
    let mut by_name: std::collections::HashMap<String, JointRef> = std::collections::HashMap::new();
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some(jname)) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let spec = robot.joints.iter().find(|j| &j.name == jname);
        let (kp, kd, effort) = spec
            .map(|s| (s.kp, s.kd, s.effort_limit))
            .unwrap_or((50.0, 1.0, 20.0));
        let mut joint = GenericJointBuilder::new(locked)
            .local_frame1(Pose::from_parts(b.local_pos, b.local_quat))
            .local_frame2(Pose::IDENTITY)
            .build();
        joint.set_motor_model(JointAxis::AngZ, MotorModel::ForceBased);
        joint.set_motor_position(JointAxis::AngZ, 0.0, kp, kd);
        joint.set_motor_max_force(JointAxis::AngZ, effort);
        let handle = multibody
            .insert(handles[parent], handles[i], joint, true)
            .expect("insert joint");
        by_name.insert(
            jname.clone(),
            JointRef {
                handle,
                parent: handles[parent],
                child: handles[i],
                rest_quat: b.local_quat,
                kp,
                kd,
                effort,
            },
        );
    }

    // Order the joint refs into policy (JOINT_NAMES) order.
    let joints: Vec<JointRef> = JOINT_NAMES
        .iter()
        .map(|n| {
            by_name
                .remove(*n)
                .unwrap_or_else(|| panic!("missing joint {n}"))
        })
        .collect();

    // FK so coords-0 link poses are exact.
    if let Some(link_id) = multibody.rigid_body_link(torso).copied() {
        if let Some(mb) = multibody.get_multibody_mut(link_id.multibody) {
            mb.forward_kinematics(&bodies, true);
            mb.update_rigid_bodies(&mut bodies, true);
        }
    }

    // Ground (Z-up), top at z = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    let ground_collider = colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );

    World {
        bodies,
        colliders,
        impulse,
        multibody,
        islands: IslandManager::new(),
        bp: BroadPhaseBvh::new(),
        np: NarrowPhase::new(),
        ccd: CCDSolver::new(),
        pipeline: PhysicsPipeline::new(),
        joints,
        torso,
        feet,
        ground_collider,
        all_handles: handles,
    }
}

/// Outcome of one control step.
pub struct StepOut {
    pub obs: Vec<f32>,
    pub critic_obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub fell: bool,
}

/// One flat-velocity environment instance.
pub struct BipedEnv {
    mjcf: Vec<MjBody>,
    robot: LeRobotBipedal,
    task: VelocityFlatTask,
    sampler: CommandSampler,
    rng: Lcg,
    world: World,
    cmd: VelocityCommand,
    step_count: u32,
    resample_at: u32,
    last_action: [f32; NUM_JOINTS],
    prev_action: [f32; NUM_JOINTS],
    /// Seconds each foot has been airborne (tracked across steps for air-time reward).
    air_time: [f32; NUM_FEET],
    /// Each foot's sole-normal in its own link frame, captured at the flat spawn
    /// pose (world sole-normal = +Z there). Lets `update_feet` measure sole tilt.
    foot_sole_local: [Vec3; NUM_FEET],
    /// Domain randomization config (applied at every reset + every step).
    randomization: Randomization,
    /// If `Some(yaw)`, every reset (including in-rollout restarts after a fall)
    /// spawns at this exact yaw, overriding `randomization.spawn_yaw_range`. Used
    /// for reproducible multi-pose rollouts.
    pinned_yaw: Option<f32>,
    /// If `Some`, these per-joint angle offsets are applied after every reset,
    /// so post-fall restarts return to the SAME perturbed pose. Free root joint
    /// is left untouched. Used for multi-pose rollouts.
    pinned_joint_offsets: Option<[f32; NUM_JOINTS]>,
    /// Step at which the next random push impulse fires (`u32::MAX` = none queued).
    push_at: u32,
    gravity: Vec3,
    ip: IntegrationParameters,
}

/// Domain randomization knobs. Defaults are training-grade (broad enough to break
/// memorisation of fixed friction/PD/spawn quirks, narrow enough to stay learnable
/// at our sample budget). Use `Randomization::off()` for clean deterministic replay.
#[derive(Clone, Copy, Debug)]
pub struct Randomization {
    /// Half-range of initial yaw at reset, rad. `0.0` = always fixed forward.
    pub spawn_yaw_range: f32,
    /// Gaussian noise stddev added to each action target every control step (rad).
    pub action_noise_std: f32,
    /// `(min, max)` friction coefficient sampled per-episode for foot & ground.
    pub friction_range: (f32, f32),
    /// `(min, max)` multiplicative scale on every joint's PD gains, per-episode.
    pub pd_scale_range: (f32, f32),
    /// `(min, max)` seconds between random impulse perturbations to the torso.
    pub push_interval_s: (f32, f32),
    /// Max horizontal linear-velocity kick from one push, m/s (sign + magnitude
    /// are sampled uniformly). 0 = pushes disabled.
    pub push_lin_vel_max: f32,
    /// `(min, max)` contact natural frequency (Hz), sampled per-episode. Sets
    /// how stiff/soft the foot–ground contact is. Default rapier value is 30 Hz;
    /// MuJoCo's effective value is ~8 Hz. Randomising both targets the dominant
    /// cross-engine difference revealed by the MuJoCo replay test.
    pub contact_freq_range: (f32, f32),
    /// `(min, max)` contact damping ratio, sampled per-episode. Default 5.0.
    pub contact_damping_range: (f32, f32),
    /// `(min, max)` restitution coefficient for foot & ground colliders.
    pub restitution_range: (f32, f32),
}

impl Default for Randomization {
    fn default() -> Self {
        // Cross-engine-transfer DR: randomise the parameters that actually differ
        // between rapier and MuJoCo/Newton — friction coefficient, contact
        // stiffness, contact damping, restitution, PD gains. Spawn yaw and pushes
        // stay off (those are regularisers, not transfer DR).
        Self {
            spawn_yaw_range: 0.0,
            action_noise_std: 0.0,
            friction_range: (0.5, 1.5),
            pd_scale_range: (0.85, 1.15), // ±15% on PD gains
            push_interval_s: (1e9, 1e9),
            push_lin_vel_max: 0.0,
            contact_freq_range: (10.0, 50.0), // straddles MuJoCo (~8) & rapier (30)
            contact_damping_range: (2.0, 8.0),
            restitution_range: (0.0, 0.15),
        }
    }
}

impl Randomization {
    /// All DR disabled — deterministic, reproducible. Use for rendering / probes.
    pub fn off() -> Self {
        Self {
            spawn_yaw_range: 0.0,
            action_noise_std: 0.0,
            friction_range: (1.0, 1.0),
            pd_scale_range: (1.0, 1.0),
            push_interval_s: (1e9, 1e9),
            push_lin_vel_max: 0.0,
            contact_freq_range: (30.0, 30.0), // rapier default
            contact_damping_range: (5.0, 5.0),
            restitution_range: (0.0, 0.0),
        }
    }
}

impl BipedEnv {
    pub fn new(mjcf_xml: &str, seed: u64) -> Self {
        let mjcf = parse_mjcf(mjcf_xml);
        let robot = LeRobotBipedal::new();
        let task = VelocityFlatTask::new();
        let mut ip = IntegrationParameters::default();
        ip.dt = task.sim_dt;
        ip.num_solver_iterations = 8;
        let world = build_world(&mjcf, &robot, 0.0);
        // At the freshly-built default pose the feet are flat (world sole-normal =
        // +Z), so each foot's local sole-normal is R_spawn⁻¹·Z. Tilt is then the
        // angle between the rotated local normal and +Z at any later step.
        let mut foot_sole_local = [Vec3::Z; NUM_FEET];
        for i in 0..NUM_FEET {
            if let Some(&foot) = world.feet.get(i) {
                foot_sole_local[i] = world.bodies[foot].rotation().conjugate() * Vec3::Z;
            }
        }
        let mut env = Self {
            mjcf,
            robot,
            task,
            sampler: CommandSampler::default(),
            rng: Lcg::new(seed),
            world,
            cmd: VelocityCommand::default(),
            step_count: 0,
            resample_at: 0,
            last_action: [0.0; NUM_JOINTS],
            prev_action: [0.0; NUM_JOINTS],
            air_time: [0.0; NUM_FEET],
            foot_sole_local,
            randomization: Randomization::default(),
            pinned_yaw: None,
            pinned_joint_offsets: None,
            push_at: u32::MAX,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            ip,
        };
        env.reset();
        env
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

    /// Set the spawn-yaw randomization half-range (rad). Convenience wrapper that
    /// only touches the yaw knob; for everything else use [`set_randomization`].
    pub fn set_spawn_yaw_range(&mut self, r: f32) {
        self.randomization.spawn_yaw_range = r;
    }

    /// Replace the full domain-randomization config. Pass `Randomization::off()`
    /// for deterministic replay (rendering, probes).
    pub fn set_randomization(&mut self, r: Randomization) {
        self.randomization = r;
    }

    /// Current DR config (immutable view).
    pub fn randomization(&self) -> &Randomization {
        &self.randomization
    }

    /// Pin the spawn yaw to a specific value (rad). Every subsequent `reset_full`
    /// uses this exact yaw, ignoring `randomization.spawn_yaw_range`. Pass `None`
    /// (via [`unpin_yaw`]) to return to the DR-driven yaw.
    pub fn pin_yaw(&mut self, yaw: f32) {
        self.pinned_yaw = Some(yaw);
    }

    /// Clear the pinned yaw (restore DR-driven yaw sampling at reset).
    pub fn unpin_yaw(&mut self) {
        self.pinned_yaw = None;
    }

    /// Pin per-joint initial angle offsets — applied after every reset (incl.
    /// post-fall restarts) so multi-pose rollouts stay at the chosen pose.
    pub fn pin_joint_offsets(&mut self, offsets: [f32; NUM_JOINTS]) {
        self.pinned_joint_offsets = Some(offsets);
    }

    /// Clear the pinned joint offsets (reset returns to MJCF default pose).
    pub fn unpin_joint_offsets(&mut self) {
        self.pinned_joint_offsets = None;
    }

    /// Add per-joint angle offsets (rad) to the multibody's current pose, then
    /// re-run forward kinematics so the link rigid-body world transforms match.
    /// Call after [`reset_full`] to start the rollout from a non-default joint
    /// configuration. The free root joint is left untouched.
    pub fn perturb_initial_joints(&mut self, offsets: &[f32; NUM_JOINTS]) {
        let world = &mut self.world;
        let torso = world.torso;
        let Some(link) = world.multibody.rigid_body_link(torso).copied() else {
            return;
        };
        if let Some(mb) = world.multibody.get_multibody_mut(link.multibody) {
            // Displacement vector: 6 free-joint DOFs (zero) + NUM_JOINTS hinge offsets,
            // in the same canonical joint order the env's `joints` field uses.
            let mut disp = vec![0.0_f32; mb.ndofs()];
            for k in 0..NUM_JOINTS {
                disp[6 + k] = offsets[k];
            }
            mb.apply_displacements(&disp);
            mb.forward_kinematics(&world.bodies, true);
            mb.update_rigid_bodies(&mut world.bodies, true);
        }
    }

    /// Pin the velocity command (e.g. for a clean straight-walk demo): every
    /// resample/reset returns exactly `(vx, vy, yaw)`, no standing, no turning.
    pub fn pin_command(&mut self, vx: f32, vy: f32, yaw: f32) {
        self.sampler.lin_vel_x = (vx, vx);
        self.sampler.lin_vel_y = (vy, vy);
        self.sampler.ang_vel_z = (yaw, yaw);
        self.sampler.standing_prob = 0.0;
        self.cmd = VelocityCommand {
            vx,
            vy,
            yaw_rate: yaw,
        };
    }

    /// Curriculum: scale the commanded-velocity ranges by `s` (0 = stand only,
    /// 1 = full deployed ranges). Lets the policy learn to balance before walking.
    pub fn set_command_scale(&mut self, s: f32) {
        let d = CommandSampler::default();
        self.sampler.lin_vel_x = (d.lin_vel_x.0 * s, d.lin_vel_x.1 * s);
        self.sampler.lin_vel_y = (d.lin_vel_y.0 * s, d.lin_vel_y.1 * s);
        self.sampler.ang_vel_z = (d.ang_vel_z.0 * s, d.ang_vel_z.1 * s);
    }

    /// Rebuild the world at a random yaw, resample the command, return the obs.
    pub fn reset(&mut self) -> Vec<f32> {
        self.reset_full().0
    }

    /// Like [`reset`](Self::reset) but returns both the policy and critic obs.
    pub fn reset_full(&mut self) -> (Vec<f32>, Vec<f32>) {
        let dr = self.randomization;
        // Spawn yaw — pinned value wins; otherwise uniformly sampled in [-range, +range].
        let yaw = if let Some(y) = self.pinned_yaw {
            y
        } else if dr.spawn_yaw_range > 0.0 {
            self.rng.range(-dr.spawn_yaw_range, dr.spawn_yaw_range)
        } else {
            0.0
        };
        self.world = build_world(&self.mjcf, &self.robot, yaw);

        // Per-episode contact material: friction + restitution applied to the
        // ground and every foot collider.
        let sample = |rng: &mut Lcg, r: (f32, f32)| {
            if r.0 != r.1 { rng.range(r.0, r.1) } else { r.0 }
        };
        let friction = sample(&mut self.rng, dr.friction_range);
        let restitution = sample(&mut self.rng, dr.restitution_range);
        self.world.colliders[self.world.ground_collider].set_friction(friction);
        self.world.colliders[self.world.ground_collider].set_restitution(restitution);
        for &foot in &self.world.feet {
            for &c in self.world.bodies[foot].colliders() {
                self.world.colliders[c].set_friction(friction);
                self.world.colliders[c].set_restitution(restitution);
            }
        }
        // Per-episode contact stiffness/damping — the cross-engine difference test
        // showed these were the dominant source of replay divergence.
        self.ip.contact_softness.natural_frequency = sample(&mut self.rng, dr.contact_freq_range);
        self.ip.contact_softness.damping_ratio = sample(&mut self.rng, dr.contact_damping_range);
        // Per-episode PD gain scale: sample once, multiply every joint's kp/kd.
        let pd_scale = if dr.pd_scale_range.0 != dr.pd_scale_range.1 {
            self.rng.range(dr.pd_scale_range.0, dr.pd_scale_range.1)
        } else {
            dr.pd_scale_range.0
        };
        for jr in self.world.joints.iter_mut() {
            jr.kp *= pd_scale;
            jr.kd *= pd_scale;
        }

        self.cmd = self.sampler.sample(&mut self.rng);
        self.step_count = 0;
        self.resample_at = self
            .sampler
            .resample_steps(&mut self.rng, self.task.control_dt());
        // Schedule first push.
        self.push_at = self.next_push_step();
        self.last_action = [0.0; NUM_JOINTS];
        self.prev_action = [0.0; NUM_JOINTS];
        self.air_time = [0.0; NUM_FEET];
        // Apply pinned joint offsets (multi-pose rollout) before observing.
        if let Some(offsets) = self.pinned_joint_offsets {
            self.perturb_initial_joints(&offsets);
        }
        let mut state = self.read_state();
        state.feet = self.update_feet();
        let mut obs = vec![0.0; OBS_DIM];
        self.task.observe(&state, &self.cmd, &mut obs);
        let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
        self.task.observe_critic(&state, &self.cmd, &mut critic_obs);
        (obs, critic_obs)
    }

    pub fn step(&mut self, action: &[f32; NUM_JOINTS]) -> StepOut {
        // Action noise: independent Gaussian per joint, added to the position
        // target. Prevents the policy from memorising sub-millimeter precision.
        let noise_std = self.randomization.action_noise_std;
        let mut noisy_action = *action;
        if noise_std > 0.0 {
            for v in noisy_action.iter_mut() {
                *v += noise_std * self.rng.gauss();
            }
        }
        // Apply PD position targets.
        let targets = self.task.joint_targets(&noisy_action);
        for k in 0..NUM_JOINTS {
            let jr = &self.world.joints[k];
            let (kp, kd) = (jr.kp, jr.kd);
            if let Some((mb, link_id)) = self.world.multibody.get_mut(jr.handle) {
                if let Some(link) = mb.link_mut(link_id) {
                    link.joint
                        .data
                        .set_motor_position(JointAxis::AngZ, targets[k], kp, kd);
                }
            }
        }
        // Push perturbation: if the schedule fired this step, apply a small
        // horizontal velocity impulse to the torso, then schedule the next push.
        if self.step_count >= self.push_at {
            self.apply_push();
            self.push_at = self.step_count + self.next_push_step();
        }
        // Advance physics at the control decimation.
        for _ in 0..self.task.decimation {
            self.world.pipeline.step(
                self.gravity,
                &self.ip,
                &mut self.world.islands,
                &mut self.world.bp,
                &mut self.world.np,
                &mut self.world.bodies,
                &mut self.world.colliders,
                &mut self.world.impulse,
                &mut self.world.multibody,
                &mut self.world.ccd,
                &(),
                &(),
            );
        }
        self.prev_action = self.last_action;
        self.last_action = *action;
        self.step_count += 1;

        let feet = self.update_feet();
        let mut state = self.read_state();
        state.feet = feet;

        // Resample command on schedule.
        if self.step_count >= self.resample_at {
            self.cmd = self.sampler.sample(&mut self.rng);
            self.resample_at = self.step_count
                + self
                    .sampler
                    .resample_steps(&mut self.rng, self.task.control_dt());
        }

        let fell = self.task.fell_over(&state.base) || !state.base.height.is_finite();
        let timeout = self.step_count >= self.task.max_steps();
        let mut reward = self.task.reward(&state, &self.cmd).total();
        if fell {
            // One-shot termination penalty (per `task.weights.termination`).
            reward += self.task.weights.termination;
        }
        let mut obs = vec![0.0; OBS_DIM];
        self.task.observe(&state, &self.cmd, &mut obs);
        let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
        self.task.observe_critic(&state, &self.cmd, &mut critic_obs);
        StepOut {
            obs,
            critic_obs,
            reward,
            done: fell || timeout,
            fell,
        }
    }

    /// Torso world height (Z), for logging.
    pub fn torso_height(&self) -> f32 {
        self.world.bodies[self.world.torso].translation().z
    }

    /// Direct rapier-side foot probe: for each foot, the lowest world-Z of its
    /// colliders, the number of active ground contacts, and the most-penetrating
    /// contact distance + that contact's world normal. This is rapier's OWN view
    /// (not MuJoCo's), to see exactly how the feet meet the ground.
    pub fn foot_report(&self) -> String {
        use std::fmt::Write as _;
        let w = &self.world;
        let mut s = String::new();
        for (fi, &foot) in w.feet.iter().enumerate() {
            let body = &w.bodies[foot];
            let mut lowest = f32::INFINITY;
            let mut ncol = 0;
            let mut pairs = 0; // broad-phase contact pairs found
            let mut npts = 0; // total manifold points
            let mut min_dist = f32::INFINITY;
            for &c in body.colliders() {
                ncol += 1;
                lowest = lowest.min(w.colliders[c].compute_aabb().mins.z);
                if let Some(pair) = w.np.contact_pair(c, w.ground_collider) {
                    pairs += 1;
                    for m in &pair.manifolds {
                        for p in &m.points {
                            npts += 1;
                            min_dist = min_dist.min(p.dist);
                        }
                    }
                }
            }
            let _ = writeln!(
                s,
                "  foot{fi}: lowest_z={lowest:+.3} body_z={:+.3} contact_pairs={pairs} points={npts} min_dist={:+.3}",
                body.translation().z,
                if min_dist.is_finite() { min_dist } else { 9.99 },
            );
        }
        s
    }

    /// Torso forward speed in the body frame (m/s) — the "is it actually walking"
    /// signal. Projects world linear velocity onto the body's forward (+X) axis.
    pub fn base_forward_speed(&self) -> f32 {
        let b = &self.world.bodies[self.world.torso];
        let fwd = b.rotation() * Vec3::X;
        b.linvel().dot(fwd)
    }

    /// Magnitude of the current commanded planar velocity (for logging).
    pub fn command_speed(&self) -> f32 {
        self.cmd.speed()
    }

    /// Per-foot sole tilt from horizontal (rad; 0 = sole flat). Same measure the
    /// flat-foot reward uses — exposed so a probe can read it under pure physics.
    pub fn foot_tilts(&self) -> [f32; NUM_FEET] {
        let mut out = [0.0; NUM_FEET];
        for i in 0..NUM_FEET {
            if let Some(&foot) = self.world.feet.get(i) {
                let n = self.world.bodies[foot].rotation() * self.foot_sole_local[i];
                out[i] = n.z.abs().clamp(0.0, 1.0).acos();
            }
        }
        out
    }

    /// World positions of all robot link bodies, in MJCF order (for rendering).
    pub fn body_positions(&self) -> Vec<[f32; 3]> {
        self.world
            .all_handles
            .iter()
            .map(|&h| {
                let t = self.world.bodies[h].translation();
                [t.x, t.y, t.z]
            })
            .collect()
    }

    /// Base (torso) world pose: `(position, quaternion xyzw)`. For MuJoCo qpos
    /// playback (note MuJoCo wants the quaternion in `wxyz` order).
    pub fn base_pose(&self) -> ([f32; 3], [f32; 4]) {
        let b = &self.world.bodies[self.world.torso];
        let t = b.translation();
        let r = b.rotation();
        ([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])
    }

    /// Joint angles (rad) in policy/`JOINT_NAMES` order — same computation as the
    /// observation, exposed for rendering / logging.
    pub fn joint_angles(&self) -> [f32; NUM_JOINTS] {
        let mut q = [0.0f32; NUM_JOINTS];
        for (k, jr) in self.world.joints.iter().enumerate() {
            let rp = *self.world.bodies[jr.parent].rotation();
            let rc = *self.world.bodies[jr.child].rotation();
            let qrel = jr.rest_quat.conjugate() * (rp.conjugate() * rc);
            q[k] = 2.0 * qrel.z.atan2(qrel.w);
        }
        q
    }

    /// The kinematic tree for rendering: `(link names, parent→child edges, foot
    /// link indices)`, all indexed in MJCF order.
    pub fn skeleton(&self) -> (Vec<String>, Vec<(usize, usize)>, Vec<usize>) {
        let names = self.mjcf.iter().map(|b| b.name.clone()).collect();
        let edges = self
            .mjcf
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.parent.map(|p| (p, i)))
            .collect();
        let feet = self
            .mjcf
            .iter()
            .enumerate()
            .filter_map(|(i, b)| (!b.capsules.is_empty()).then_some(i))
            .collect();
        (names, edges, feet)
    }

    /// Sample the number of control steps until the next push perturbation, given
    /// `randomization.push_interval_s`. Returns `u32::MAX` when pushes are disabled.
    fn next_push_step(&mut self) -> u32 {
        let dr = self.randomization;
        if dr.push_lin_vel_max <= 0.0 || dr.push_interval_s.0 >= 1e8 {
            return u32::MAX;
        }
        let s = self.rng.range(dr.push_interval_s.0, dr.push_interval_s.1);
        (s / self.task.control_dt()).max(1.0) as u32
    }

    /// Apply a single random horizontal velocity-impulse to the torso.
    fn apply_push(&mut self) {
        let max_v = self.randomization.push_lin_vel_max;
        if max_v <= 0.0 {
            return;
        }
        let angle = self.rng.range(-std::f32::consts::PI, std::f32::consts::PI);
        let mag = self.rng.range(0.0, max_v);
        let dv = Vec3::new(angle.cos() * mag, angle.sin() * mag, 0.0);
        let body = &mut self.world.bodies[self.world.torso];
        let impulse = dv * body.mass();
        body.apply_impulse(impulse, true);
    }

    /// Read foot–ground contact / height / speed and advance the per-foot air-time
    /// tracker. Returns the per-foot observation for the reward.
    fn update_feet(&mut self) -> [FootObs; NUM_FEET] {
        let dt = self.task.control_dt();
        // Base orientation (for foot-yaw-relative-to-base computation below).
        let base_rot = *self.world.bodies[self.world.torso].rotation();
        let base_rot_inv = base_rot.conjugate();
        let mut out = [FootObs::default(); NUM_FEET];
        for i in 0..NUM_FEET {
            let Some(&foot) = self.world.feet.get(i) else {
                continue;
            };
            let body = &self.world.bodies[foot];
            let pos = body.translation();
            let lv = body.linvel();
            // Sole tilt: angle between the (rotated) local sole-normal and world +Z.
            let world_normal = body.rotation() * self.foot_sole_local[i];
            let tilt = world_normal.z.abs().clamp(0.0, 1.0).acos();
            // Foot yaw relative to base: rotate the body-X axis by the relative
            // quaternion (base⁻¹·foot), then take atan2(y, x) of the result. At
            // zero relative orientation this gives 0; rotational misalignment in
            // the horizontal plane gives ±yaw — used by WBC's feet_yaw_mean term.
            let foot_x_in_base = (base_rot_inv * *body.rotation()) * Vec3::X;
            let yaw_rel_base = foot_x_in_base.y.atan2(foot_x_in_base.x);
            // In contact if any of the foot's collision capsules touches the ground.
            let contact = body.colliders().iter().any(|&c| {
                self.world
                    .np
                    .contact_pair(c, self.world.ground_collider)
                    .is_some_and(|p| p.has_any_active_contact())
            });
            let prev_air = self.air_time[i];
            let first_contact = contact && prev_air > 0.0;
            self.air_time[i] = if contact { 0.0 } else { prev_air + dt };
            out[i] = FootObs {
                contact,
                first_contact,
                // Air time *at touchdown* (so the reward sees the completed swing).
                air_time: if contact { prev_air } else { self.air_time[i] },
                height: pos.z,
                planar_speed: (lv.x * lv.x + lv.y * lv.y).sqrt(),
                tilt,
                yaw_rel_base,
                pos_xy: [pos.x, pos.y],
            };
        }
        out
    }

    fn read_state(&self) -> RobotState {
        let torso = &self.world.bodies[self.world.torso];
        let t = torso.translation();
        let r = torso.rotation();
        let lv = torso.linvel();
        let av = torso.angvel();
        let base = BaseState {
            orientation: [r.x, r.y, r.z, r.w],
            lin_vel_world: [lv.x, lv.y, lv.z],
            ang_vel_world: [av.x, av.y, av.z],
            height: t.z,
        };
        let mut joint_pos = [0.0f32; NUM_JOINTS];
        let mut joint_vel = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            let jr = &self.world.joints[k];
            let rp = *self.world.bodies[jr.parent].rotation();
            let rc = *self.world.bodies[jr.child].rotation();
            // Joint angle about child-local Z: Rz(q) = rest⁻¹ · Rp⁻¹ · Rc.
            let qrel = jr.rest_quat.conjugate() * (rp.conjugate() * rc);
            joint_pos[k] = 2.0 * qrel.z.atan2(qrel.w);
            // Joint rate: relative angular velocity projected on the world axis.
            let wp = self.world.bodies[jr.parent].angvel();
            let wc = self.world.bodies[jr.child].angvel();
            let axis = rc * Vec3::Z;
            joint_vel[k] = (wc - wp).dot(axis);
        }
        RobotState {
            base,
            joint_pos,
            joint_vel,
            last_action: self.last_action,
            prev_action: self.prev_action,
            feet: [FootObs::default(); NUM_FEET], // overwritten by the caller
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel PPO iteration (shared by the trainers).
// ---------------------------------------------------------------------------

fn to_action(v: &[f32]) -> [f32; NUM_JOINTS] {
    let mut a = [0.0; NUM_JOINTS];
    a.copy_from_slice(&v[..NUM_JOINTS]);
    a
}

/// Diagnostics from one PPO iteration.
pub struct IterStats {
    pub mean_step_reward: f32,
    pub falls: u32,
    pub mean_torso_z: f32,
    /// Mean torso forward speed (m/s) — the "is it walking" signal.
    pub mean_speed: f32,
    /// Mean commanded speed (m/s) — what it's being asked to track.
    pub mean_cmd: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub lr: f32,
}

/// Run one PPO iteration over `envs`: collect `t_steps` per env (stepping all envs
/// in parallel across cores via rayon), then update `ac`. `cur`/`cur_c` hold each
/// env's current policy / critic observation and are advanced in place. The
/// observation normalizer is fed every collected step.
#[allow(clippy::too_many_arguments)]
pub fn ppo_iteration(
    ac: &mut ActorCritic,
    envs: &mut [BipedEnv],
    cur: &mut [Vec<f32>],
    cur_c: &mut [Vec<f32>],
    cfg: &PpoConfig,
    rng: &mut zealot_rl::rng::Lcg,
    t_steps: usize,
) -> IterStats {
    let n = envs.len();
    let mut samples: Vec<Vec<Sample>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
    let mut rs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
    let mut vs: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
    let mut ds: Vec<Vec<bool>> = (0..n).map(|_| Vec::with_capacity(t_steps)).collect();
    let (mut total_reward, mut falls, mut torso_sum) = (0.0f32, 0u32, 0.0f32);
    let (mut speed_sum, mut cmd_sum) = (0.0f32, 0.0f32);

    for _ in 0..t_steps {
        // Sample actions + values for all envs (sequential: shared policy + rng).
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
        // Advance every env's physics in parallel (independent rapier worlds).
        let outs: Vec<StepOut> = envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, a)| env.step(a))
            .collect();
        // Record outcomes + reset fallen envs (sequential).
        for e in 0..n {
            let out = &outs[e];
            total_reward += out.reward;
            rs[e].push(out.reward);
            ds[e].push(out.done);
            if out.fell {
                falls += 1;
            }
            // Tracking signal: actual forward speed vs commanded, BEFORE any reset.
            cmd_sum += envs[e].command_speed();
            speed_sum += envs[e].base_forward_speed();
            if out.done {
                let (o, c) = envs[e].reset_full();
                cur[e] = o;
                cur_c[e] = c;
            } else {
                cur[e].clone_from(&out.obs);
                cur_c[e].clone_from(&out.critic_obs);
            }
            torso_sum += envs[e].torso_height();
        }
    }

    // GAE per env, then flatten into one batch.
    let mut batch: Vec<Sample> = Vec::with_capacity(n * t_steps);
    for e in 0..n {
        let last_v = ac.value(&cur_c[e]);
        let (adv, ret) = gae(&rs[e], &vs[e], &ds[e], last_v, cfg.gamma, cfg.lam);
        for t in 0..t_steps {
            samples[e][t].adv = adv[t];
            samples[e][t].ret = ret[t];
            batch.push(std::mem::take(&mut samples[e][t]));
        }
    }
    let stats = ac.update(&mut batch, cfg);
    let steps = (n * t_steps) as f32;
    IterStats {
        mean_step_reward: total_reward / steps,
        falls,
        mean_torso_z: torso_sum / steps,
        mean_speed: speed_sum / steps,
        mean_cmd: cmd_sum / steps,
        value_loss: stats.value_loss,
        entropy: stats.entropy,
        lr: stats.lr,
    }
}
