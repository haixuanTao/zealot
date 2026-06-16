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
use rayon::prelude::*;
use roxmltree::Node;
use std::collections::HashMap;
use std::time::Instant;
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

/// Per-phase wall-time accumulators populated by `BipedNexusBatchEnv::step`.
/// Use `take_step_timings` to read + reset. `Instant::now()` is cheap (~50 ns
/// per call, ~10 calls per step → ~0.5 µs/step overhead) so the
/// instrumentation is always on. Lets us answer "where does the per-step
/// time actually go?" without external profilers.
#[derive(Default, Clone, Copy, Debug)]
pub struct StepTimings {
    /// Number of `step()` calls accumulated into this struct.
    pub steps: u64,
    /// Host loop staging motor targets into `links_static_mirror`.
    pub stage_motors_ns: u64,
    /// `flush_links_static` — single `write_buffer` for the whole mirror.
    pub flush_static_ns: u64,
    /// `decimation × pipeline.step.await` — encoder build + queue submit
    /// (host-side; GPU work is fire-and-forget here, waited on later).
    pub pipeline_step_ns: u64,
    /// `auto_resize_buffers` (only fires every `AUTO_RESIZE_PERIOD` steps).
    pub auto_resize_ns: u64,
    /// Explicit `gpu.synchronize()` between the pipeline step and the
    /// readback — this is where the host actually blocks waiting for the
    /// physics dispatches we enqueued above to finish. So this is "true
    /// GPU compute time per ctrl step", separated from the byte transfer.
    pub gpu_wait_ns: u64,
    /// `slurp_poses` — `slow_read_buffer` of body_poses (the only readback
    /// remaining after Tier 1). After the explicit sync above, this should
    /// be just the staging copy + map_async + memcpy.
    pub readback_ns: u64,
    /// Serial pre-pass: `step_count++` + occasional command resample.
    pub serial_pre_ns: u64,
    /// Parallel rayon block (feet/state/obs/reward across N envs).
    pub par_compute_ns: u64,
    /// Serial commit pass: per-env state writes + StepOut assembly.
    pub serial_commit_ns: u64,
}

impl StepTimings {
    /// Total wall time accounted for across all phases (ns).
    pub fn total_ns(&self) -> u64 {
        self.stage_motors_ns
            + self.flush_static_ns
            + self.pipeline_step_ns
            + self.auto_resize_ns
            + self.gpu_wait_ns
            + self.readback_ns
            + self.serial_pre_ns
            + self.par_compute_ns
            + self.serial_commit_ns
    }
}
// `pipeline.auto_resize_buffers` only needs to fire when nexus's internal
// buffers (contacts mostly) grow. Once the scene settles after a few warmup
// steps, sizes stop changing — calling it every step adds dispatch latency
// for no work. 32 control steps ≈ 0.64 s of sim time, plenty fast to react.
const AUTO_RESIZE_PERIOD: u32 = 32;

// --- MJCF parsing (duplicated from biped_env.rs — small, self-contained) ----

pub struct MjBody {
    #[allow(dead_code)]
    pub name: String,
    pub parent: Option<usize>,
    pub local_pos: Vec3,
    pub local_quat: Rotation,
    pub joint: Option<String>,
    /// Real per-joint position limits `(lo, hi)` from the MJCF `range` (rad).
    /// `None` if unlimited. Used instead of the ±π JointSpec placeholder so the
    /// ankle/knee can't over-flex (e.g. the foot folding into its own shin).
    pub joint_range: Option<(f32, f32)>,
    /// Passive joint damping (N·m·s/rad) from the MJCF `damping`. `None` if the
    /// model omits it (then the JointSpec value is used).
    pub joint_damping: Option<f32>,
    pub com: Vec3,
    pub mass: f32,
    /// Diagonal inertia (Ixx, Iyy, Izz) from MJCF `fullinertia`.
    pub inertia_diag: Vec3,
    /// Off-diagonal inertia products (Ixy, Ixz, Iyz) from MJCF `fullinertia`.
    /// Several links have these comparable to the diagonal (~50–100%), so the
    /// inertia tensor is significantly rotated — must not be dropped.
    pub inertia_offdiag: Vec3,
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
    let mut joint_range = None;
    let mut joint_damping = None;
    let mut is_free = false;
    let (mut com, mut mass, mut inertia_diag) = (Vec3::ZERO, 0.0, Vec3::splat(1e-4));
    let mut inertia_offdiag = Vec3::ZERO;
    let mut capsules = Vec::new();
    for c in node.children().filter(Node::is_element) {
        match c.tag_name().name() {
            "freejoint" => is_free = true,
            "joint" => {
                joint = Some(c.attribute("name").unwrap_or("").to_string());
                joint_range = c.attribute("range").map(|s| {
                    let f = floats(s);
                    (f[0], f[1])
                });
                joint_damping = c.attribute("damping").and_then(|s| s.parse().ok());
            }
            "inertial" => {
                com = vec3(&c, "pos", Vec3::ZERO);
                mass = c
                    .attribute("mass")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                if let Some(s) = c.attribute("fullinertia") {
                    // MuJoCo order: Ixx Iyy Izz Ixy Ixz Iyz.
                    let f = floats(s);
                    inertia_diag = Vec3::new(f[0], f[1], f[2]);
                    if f.len() >= 6 {
                        inertia_offdiag = Vec3::new(f[3], f[4], f[5]);
                    }
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
            joint_range,
            joint_damping,
            com,
            mass,
            inertia_diag,
            inertia_offdiag,
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
    /// Links that must NEVER touch the ground (thigh / shin / hip) — only the
    /// feet have ground colliders in nexus, so the policy can otherwise clip
    /// these straight through the floor for free support. Used for a
    /// WBC-AGILE-style `illegal_contact` termination (terminate if any of these
    /// drops below `BIPED_ILLEGAL_Z`).
    pub illegal_ground_links: Vec<u32>,
    /// Left/right link pairs (foot, shin, thigh) for a WBC-AGILE-style
    /// `feet_distance`/`knee_distance` self-collision guard: nexus can't do
    /// physical leg-leg self-collision (the leg colliders are inert), so instead
    /// terminate if any pair gets closer than `BIPED_SELF_COLL_DIST` — i.e. the
    /// legs cross. Each entry is `(left_link, right_link)`.
    pub self_collision_pairs: Vec<(u32, u32)>,
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

    /// Parent multibody link index for each actuated joint (in `JOINT_NAMES`
    /// order). Used to compute joint angles from `body_poses` alone — the
    /// parent's world rotation, the joint's rest local quat, and the child's
    /// world rotation suffice (no `links_workspace` readback needed).
    pub actuated_parent_links: [u32; NUM_JOINTS],
    /// Rest orientation of each actuated joint in its parent's local frame
    /// (i.e. the body's `local_frame1.rotation` at zero joint angle). With
    /// this, `q_child = q_parent · rest_quat · R_z(θ)`, so the current angle
    /// is `θ = 2·atan2(rel.z, rel.w)` where
    /// `rel = rest_quat⁻¹ · q_parent⁻¹ · q_child`.
    pub actuated_rest_quat: [Rotation; NUM_JOINTS],
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
        // Armature (rotor inertia) is NO LONGER added to the link inertia tensor.
        // It's now seeded into the multibody's mass-matrix DIAGONAL via
        // set_dof_armature (see the seeding block after from_rapier). Baking it
        // into Izz inflated M=JᵀIJ inconsistently with the gravity bias force, so
        // a free-falling body spuriously buckled (joints to limits in ~0.1s) —
        // the core nexus instability. The diagonal is the correct, consistent
        // place (matches MuJoCo/rapier).
        // Full inertia tensor (Ixx,Iyy,Izz + Ixy,Ixz,Iyz), diagonalized by parry
        // (`with_inertia_matrix` → principal moments + frame, which nexus consumes).
        let (d, o) = (b.inertia_diag, b.inertia_offdiag);
        // DIAG-INERTIA A/B (diagnostic): zero the off-diagonals to isolate the
        // principal-frame inertia from other effects.
        let o = if std::env::var("BIPED_DIAG_INERTIA").is_ok() { Vec3::ZERO } else { o };
        let inertia_mat = Mat3::from_cols(
            Vec3::new(d.x, o.x, o.y), // col 0: Ixx, Ixy, Ixz
            Vec3::new(o.x, d.y, o.z), // col 1: Ixy, Iyy, Iyz
            Vec3::new(o.y, o.z, d.z), // col 2: Ixz, Iyz, Izz
        );
        let h = bodies.insert(
            RigidBodyBuilder::dynamic()
                .position(world[i])
                .additional_mass_properties(MassProperties::with_inertia_matrix(
                    b.com,
                    b.mass.max(1e-3),
                    inertia_mat,
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
    // Joint position limits cost ~1.7x iter time (extra per-step constraints);
    // only worth it with real (tight) ranges, so gate them off by default.
    let joint_limits_on = std::env::var_os("BIPED_JOINT_LIMITS").is_some();
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some(jname)) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let spec = robot.joints.iter().find(|j| &j.name == jname);
        let pi = std::f32::consts::PI;
        let (kp, kd, effort, pos_limit, spec_damping) = spec
            .map(|s| {
                (
                    s.kp * dr.pd_scale,
                    s.kd * dr.pd_scale,
                    s.effort_limit,
                    s.pos_limit,
                    s.damping,
                )
            })
            .unwrap_or((50.0, 1.0, 20.0, (-pi, pi), 0.0));
        // Passive joint damping (N·m·s/rad): the real joints are damped 0.5–2.3,
        // but nexus's passive-damping buffer is a hardcoded 0.1 default, so the
        // sim joints slew at ~50 rad/s. Fold the real damping into the motor's
        // velocity gain (kd) — the chosen no-shader-change fix. Prefer the MJCF
        // `damping` attr when the model provides it; else the JointSpec value.
        // It's NOT scaled by `pd_scale` (it's a physical property, not a gain).
        let damping = b.joint_damping.unwrap_or(spec_damping);
        let kd = kd + damping;
        let mut joint = GenericJointBuilder::new(locked)
            .local_frame1(Pose::from_parts(b.local_pos, b.local_quat))
            .local_frame2(Pose::IDENTITY)
            .build();
        // AccelerationBased motor: compliance is mass-normalized (cfm_coeff path),
        // so commanded kp realizes consistent stiffness regardless of the joint's
        // (tiny) link inertia — like MuJoCo's actuators. ForceBased (raw cfm_gain)
        // under-realized kp on the low-inertia leg joints, so the motors couldn't
        // hold and a passive robot buckled to its limits in ~0.1s. BIPED_FORCE_MOTOR
        // reverts to ForceBased for A/B.
        let motor_model = if std::env::var("BIPED_FORCE_MOTOR").is_ok() {
            MotorModel::ForceBased
        } else {
            MotorModel::AccelerationBased
        };
        joint.set_motor_model(JointAxis::AngZ, motor_model);
        // Diagnostic knobs to test the "constraint-motor too soft to hold" theory
        // without a cubin rebuild (gains are host-side). BIPED_KP_SCALE/KD_SCALE
        // scale the motor stiffness/damping.
        let kp_scale: f32 = std::env::var("BIPED_KP_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let kd_scale: f32 = std::env::var("BIPED_KD_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        joint.set_motor_position(JointAxis::AngZ, 0.0, kp * kp_scale, kd * kd_scale);
        joint.set_motor_max_force(JointAxis::AngZ, effort);
        // Enforce the free axis's position limits — OFF by default (set
        // BIPED_JOINT_LIMITS=1 to enable). Setting a limit makes the multibody
        // solver emit a limit constraint (kind=1) alongside each motor
        // constraint, ~doubling per-step joint constraints and costing ~1.7x
        // iter time, so it's gated. When enabled, use the REAL per-joint range
        // from the MJCF (`joint_range`) when present — the ankle is only
        // ~[-10°,+20°], so the ±π JointSpec placeholder let the foot fold into
        // its own shin — falling back to the placeholder if the model omits one.
        // (The PD target is separately clamped to the joint range in
        // VelocityFlatTask::joint_targets regardless of this physical limit.)
        if joint_limits_on {
            let (lo, hi) = b.joint_range.unwrap_or(pos_limit);
            joint.set_limits(JointAxis::AngZ, [lo, hi]);
        }
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

    // Sim params: per-env contact softness via DR. Env overrides let us A/B the
    // contact-solver knobs against the WBC-AGILE-matched config without a rebuild
    // each time (BIPED_SOLVER_ITERS / BIPED_CONTACT_NF / BIPED_CONTACT_DR).
    let env_f32 = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f32>().ok());
    let mut sp = GpuSimParams::default();
    sp.dt = task_dt;
    sp.num_solver_iterations = std::env::var("BIPED_SOLVER_ITERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(SOLVER_ITERS);
    sp.contact_natural_frequency = env_f32("BIPED_CONTACT_NF").unwrap_or(dr.contact_natural_frequency);
    sp.contact_damping_ratio = env_f32("BIPED_CONTACT_DR").unwrap_or(dr.contact_damping_ratio);

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

    // Thigh / shin / hip links — the parts that have NO ground collider (only
    // the feet do), so they must never legitimately touch the floor. (Names use
    // the model's spelling: `tigh_subassembly`, `shin_subassembly`, `hip*`.)
    // Ankle + foot are excluded (they sit legitimately low next to the sole).
    let illegal_ground_links: Vec<u32> = (0..mjcf.len())
        .filter(|&i| {
            let n = &mjcf[i].name;
            n.contains("shin") || n.contains("tigh") || n.contains("hip")
        })
        .filter_map(|i| mb_link_of_mjcf.get(&i).copied())
        .collect();

    // Left/right link pairs for the self-collision (leg-crossing) guard. The
    // model's right side is the bare name and the left side gets a `_sym`/`_2`
    // suffix, so pair each base body with its mirrored twin.
    let link_of_name = |name: &str| -> Option<u32> {
        mjcf.iter()
            .position(|b| b.name == name)
            .and_then(|i| mb_link_of_mjcf.get(&i).copied())
    };
    let self_collision_pairs: Vec<(u32, u32)> = [
        ("foot_subassembly", "foot_subassembly_2"),
        ("shin_subassembly", "shin_subassembly_sym"),
        ("tigh_subassembly", "tigh_subassembly_sym"),
    ]
    .iter()
    .filter_map(|(r, l)| Some((link_of_name(r)?, link_of_name(l)?)))
    .collect();

    // Per-joint parent link + rest quat, used by the ws-free joint-angle
    // extraction (`q_child = q_parent · rest_quat · R_z(θ)`).
    let mut actuated_parent_links = [0u32; NUM_JOINTS];
    let mut actuated_rest_quat = [Rotation::IDENTITY; NUM_JOINTS];
    for (k, &name) in JOINT_NAMES.iter().enumerate() {
        let mjcf_idx = mjcf
            .iter()
            .position(|b| b.joint.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing joint {name} in MJCF"));
        let parent_mjcf_idx = mjcf[mjcf_idx]
            .parent
            .expect("actuated joint's body must have a parent");
        actuated_parent_links[k] = *mb_link_of_mjcf
            .get(&parent_mjcf_idx)
            .expect("joint parent not in multibody");
        // The joint's local_frame1.rotation is the body's MJCF `local_quat`
        // (set above in the GenericJointBuilder call).
        actuated_rest_quat[k] = mjcf[mjcf_idx].local_quat;
    }

    let idx = LinkIndices {
        links_per_batch: next_mb_link, // 1 (torso) + 12 (legs) = 13
        dofs_per_batch: 6 + NUM_JOINTS as u32,
        colliders_per_batch: (mjcf.len() + 1) as u32, // robot bodies + ground
        torso_link: 0,
        foot_links,
        illegal_ground_links,
        self_collision_pairs,
        actuated,
        joint_dof_offset,
        foot_sole_local,
        mjcf_to_link,
        actuated_parent_links,
        actuated_rest_quat,
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
    /// Previous control-step `body_poses` slice per env (one `NexusPose` per
    /// collider in this env's slot). Used to finite-diff base linear /
    /// angular velocity and per-foot linear velocity at the control rate
    /// (20 ms) instead of reading `links_workspace.rb_vels` back from the
    /// GPU — kills the dominant per-step readback. Layout matches the body
    /// poses returned by `slurp_poses`: `colliders_per_batch` poses per env,
    /// concatenated in env-index order.
    prev_body_poses: Vec<NexusPose>,
    has_prev_pose: Vec<bool>,
    /// Per-env foot-local sole-normal (depends on the spawn template that
    /// seeded the env — we keep one copy per env, updated on reset).
    foot_sole_local: Vec<[Vec3; NUM_FEET]>,
    /// Default sampler (full ranges) — kept so `set_command_scale` can derive
    /// scaled ranges from a known baseline, mirroring the CPU env.
    sampler_default: CommandSampler,

    /// Curriculum scale on the torque (effort) penalty (0 = off, 1 = full WBC
    /// weight). Set per-iteration by the trainer via `set_torque_scale` so the
    /// penalty ramps in only AFTER the policy can stand — a torque penalty at
    /// full strength from scratch fights learning to stand at all. Initialised
    /// from `BIPED_TORQUE_W` so non-curriculum callers (e.g. render) still get a
    /// fixed value.
    torque_scale: f32,

    // GPU state
    gpu: KhalGpuBackend,
    pipeline: GpuPhysicsPipeline,
    state: GpuPhysicsState,

    // Pre-built spawn templates for reset_env_from (different DR samples).
    templates: Vec<GpuPhysicsState>,
    template_dr: Vec<DrParams>,

    /// Counter for the periodic `pipeline.auto_resize_buffers` call (see
    /// `AUTO_RESIZE_PERIOD`). Resets to 0 after each resize.
    tick_since_resize: u32,

    /// Phase-level timing accumulators — read + reset via `take_step_timings`.
    timings: StepTimings,

    /// Per-component reward + termination-cause accumulators for W&B logging.
    /// `rlog_comps[i]` sums component `i` (see `REWARD_COMP_NAMES`) over every
    /// (env, step) sample since the last `take_reward_log`; `rlog_steps` is the
    /// sample count (divide to get the per-step mean). The three termination
    /// counters are episode totals over the same window. Read + reset via
    /// `take_reward_log` so the trainer can emit one structured line per iter.
    rlog_comps: [f64; NUM_REWARD_COMPS],
    rlog_steps: u64,
    rlog_illegal: u64,
    rlog_fell: u64,
    rlog_timeout: u64,
}

/// Number of logged reward components (see [`REWARD_COMP_NAMES`]).
pub const NUM_REWARD_COMPS: usize = 24;

/// Names of the per-component reward terms, in `rlog_comps` / `RewardLog::comps`
/// order. The first 20 mirror `RewardBreakdown`'s live terms; the last four are
/// env-side penalties applied after `total()` (leg torque, ankle torque,
/// self-collision) plus the termination penalty.
pub const REWARD_COMP_NAMES: [&str; NUM_REWARD_COMPS] = [
    "track_lin_vel",
    "track_ang_vel",
    "upright",
    "base_height",
    "pose",
    "bilateral_symmetry",
    "action_rate",
    "action_rate_hipz_hipx",
    "body_ang_vel",
    "lin_vel_z",
    "dof_pos_limits",
    "dof_vel",
    "air_time",
    "flight",
    "single_support",
    "foot_slip",
    "foot_clearance",
    "foot_orientation",
    "feet_yaw_mean",
    "feet_distance",
    "torque_leg",
    "torque_ankle",
    "self_coll",
    "termination",
];

/// One window of accumulated reward/termination stats (see `take_reward_log`).
pub struct RewardLog {
    /// Per-step mean of each reward component, in `REWARD_COMP_NAMES` order.
    pub comps: [f32; NUM_REWARD_COMPS],
    /// Episodes ended by illegal ground contact over the window.
    pub illegal: u64,
    /// Episodes ended by a fall (tilt / low base height), excluding `illegal`.
    pub fell: u64,
    /// Episodes ended by hitting the max-step timeout (not a failure).
    pub timeout: u64,
    /// Number of (env, step) samples averaged into `comps`.
    pub samples: u64,
}

impl BipedNexusBatchEnv {
    /// Build N envs sharing one batched GpuPhysicsState. `num_templates` controls
    /// how many distinct DR samples are pre-built and cycled across the N envs
    /// at construction and reset time (higher = better coverage, more GPU mem).
    pub async fn new(mjcf_xml: &str, num_envs: usize, num_templates: usize, seed: u64) -> Self {
        let mjcf = parse_mjcf(mjcf_xml);
        let robot = LeRobotBipedal::new();
        let task = VelocityFlatTask::new();

        let gpu = make_backend().await;
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

        // Seed per-DOF Coulomb joint friction (MJCF `frictionloss`) into the
        // multibody. Env-major `[env][dof]` layout matching the velocity section:
        // 0 for the 6 root DOFs, each leg joint's frictionloss at its DOF offset.
        // Static across envs (same robot), set once — the per-env reset copies
        // dof_state/values, not this separate `dof_frictionloss` buffer.
        {
            let dpb = idx.dofs_per_batch as usize;
            let mut fl_per_dof = vec![0.0f32; dpb];
            for k in 0..NUM_JOINTS {
                let dof = idx.joint_dof_offset[k] as usize;
                if let Some(s) = robot.joints.iter().find(|j| j.name == idx.actuated[k].1) {
                    if dof < dpb {
                        fl_per_dof[dof] = s.frictionloss;
                    }
                }
            }
            let fl_flat: Vec<f32> = (0..num_envs).flat_map(|_| fl_per_dof.iter().copied()).collect();
            state.multibodies_mut().set_dof_frictionloss(&gpu, &fl_flat);
        }

        // Seed per-DOF armature (rotor inertia) into the multibody's mass-matrix
        // diagonal — the CORRECT place for armature. Previously armature was baked
        // into each link's inertia tensor (izz_extra), which inflated M=JᵀIJ
        // inconsistently with the gravity bias force and made a free-falling body
        // spuriously buckle (joints slammed to limits in ~0.1s — the nexus
        // instability that blocked all training). Same env-major `[env][dof]`
        // layout as frictionloss; 0 for the root DOFs. Scaled by BIPED_ARM (A/B).
        {
            let dpb = idx.dofs_per_batch as usize;
            let arm_scale: f32 = std::env::var("BIPED_ARM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let mut arm_per_dof = vec![0.0f32; dpb];
            for k in 0..NUM_JOINTS {
                let dof = idx.joint_dof_offset[k] as usize;
                if let Some(s) = robot.joints.iter().find(|j| j.name == idx.actuated[k].1) {
                    if dof < dpb {
                        arm_per_dof[dof] = s.armature * arm_scale;
                    }
                }
            }
            let arm_flat: Vec<f32> = (0..num_envs).flat_map(|_| arm_per_dof.iter().copied()).collect();
            state.multibodies_mut().set_dof_armature(&gpu, &arm_flat);
        }

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
        // One pose entry per collider per env (matches `body_poses` layout).
        let prev_body_poses =
            vec![NexusPose::default(); num_envs * idx.colliders_per_batch as usize];
        let has_prev_pose = vec![false; num_envs];
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
            prev_body_poses,
            has_prev_pose,
            foot_sole_local,
            sampler_default,
            torque_scale: std::env::var("BIPED_TORQUE_W")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.1),
            gpu,
            pipeline,
            state,
            templates,
            template_dr,
            tick_since_resize: 0,
            timings: StepTimings::default(),
            rlog_comps: [0.0; NUM_REWARD_COMPS],
            rlog_steps: 0,
            rlog_illegal: 0,
            rlog_fell: 0,
            rlog_timeout: 0,
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

    /// The shared GPU backend driving the physics. Exposed so a vortx GPU policy
    /// can run its batched forward on the *same* device (no second backend, and a
    /// future on-device obs path can skip the CPU round-trip).
    pub fn backend(&self) -> &KhalGpuBackend {
        &self.gpu
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

    /// Curriculum hook — scales the torque (effort) penalty by `s`. The trainer
    /// ramps this from 0 up to the target so the penalty engages only after the
    /// policy can stand (full strength from scratch fights learning to stand).
    pub fn set_torque_scale(&mut self, s: f32) {
        self.torque_scale = s.max(0.0);
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

    /// Hot-path readback: ONLY `body_poses` (no `links_workspace`). The fast
    /// step path uses parent⇄child relative rotation off `body_poses` to derive
    /// joint angles, and finite-diffs the previous step's poses for base /
    /// foot velocities — eliminating the ~13 MB-per-step `links_workspace`
    /// readback that dominated the host loop.
    async fn slurp_poses(&mut self) -> Vec<NexusPose> {
        let mut poses: Vec<NexusPose> =
            vec![NexusPose::default(); self.state.body_poses().buffer().len()];
        self.gpu
            .slow_read_buffer(self.state.body_poses().buffer(), &mut poses)
            .await
            .expect("body_poses readback");
        poses
    }

    /// Debug probe for the inert-motor bug: read `links_static` back FROM THE
    /// GPU and print env `e`'s actuated links' motor state (target_pos,
    /// motor_axes, gains, model). If the targets staged by the last `step()`
    /// show up here, the upload path (stage → flush → write_buffer) works and
    /// the bug is in the solver's consumption; if they don't, the upload is
    /// broken. Expected target for constant action a: `default_pos + scale·a`.
    pub async fn debug_dump_motors(&mut self, e: usize) {
        use nexus3d::rbd::shaders::dynamics::MultibodyLinkStatic;
        let lpb = self.state.multibodies_mut().links_per_batch() as usize;
        let n = self.state.multibodies_mut().links_static_mut().buffer().len();
        let mut st: Vec<MultibodyLinkStatic> = vec![unsafe { std::mem::zeroed() }; n];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().links_static_mut().buffer(),
                &mut st,
            )
            .await
            .expect("links_static readback");
        println!("links_static GPU readback: env {e}, links_per_batch={lpb}");
        for k in 0..NUM_JOINTS {
            let (link, name) = &self.idx.actuated[k];
            let s = &st[e * lpb + *link as usize];
            let m = &s.data.motors[5]; // AngZ
            println!(
                "  {name:<14} link={link:>2} ndofs={} locked={:#04x} motor_axes={:#04x} \
                 target_pos={:+.4} target_vel={:+.3} kp={} kd={} maxF={} model={}",
                s.ndofs,
                s.data.locked_axes,
                s.data.motor_axes,
                m.target_pos,
                m.target_vel,
                m.stiffness,
                m.damping,
                m.max_force,
                m.model
            );
        }

        // Raw f32 view of one actuated link's full MultibodyLinkStatic — used
        // to fit which byte offset the (misreading) CUDA kernel's motors[5]
        // access actually lands on.
        {
            let (link, name) = &self.idx.actuated[9]; // hipz_right, kp=30
            let s = &st[e * lpb + *link as usize];
            let words: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    (s as *const MultibodyLinkStatic) as *const f32,
                    std::mem::size_of::<MultibodyLinkStatic>() / 4,
                )
            };
            println!(
                "raw f32 dump of {name} (link {link}), {} words (idx: value, zeros elided):",
                words.len()
            );
            for (i, w) in words.iter().enumerate() {
                if *w != 0.0 {
                    println!("  [{i:>3}] byte {:>3}: {w:+.6e}", i * 4);
                }
            }
        }

        // The constraint slots the limit/motor solve kernel should have filled
        // last substep. kind=0 ⇒ init never wrote this slot; kind=2 with rhs
        // tracking `-(target_pos)·erp_inv_dt` ⇒ init consumed the target and
        // the bug is in the solve/apply.
        use nexus3d::rbd::shaders::dynamics::MultibodyJointConstraint;
        let cpb = self.state.multibodies_mut().joint_constraints_per_batch() as usize;
        let nc = self.state.multibodies_mut().joint_constraints().buffer().len();
        let mut cons: Vec<MultibodyJointConstraint> = vec![unsafe { std::mem::zeroed() }; nc];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().joint_constraints().buffer(),
                &mut cons,
            )
            .await
            .expect("joint_constraints readback");
        println!("joint_constraints GPU readback: env {e}, slots_per_batch={cpb}");
        for (s, c) in cons[e * cpb..(e + 1) * cpb].iter().enumerate().take(14) {
            println!(
                "  slot {s:>2}: dof_id={:>2} kind={} rhs={:+.4} rhs_wo_bias={:+.4} \
                 inv_lhs={:+.4e} impulse={:+.4e} lo={:+.3e} hi={:+.3e} cfm_c={:.3} cfm_g={:.3e}",
                c.dof_id,
                c.kind,
                c.rhs,
                c.rhs_wo_bias,
                c.inv_lhs,
                c.impulse,
                c.impulse_lo,
                c.impulse_hi,
                c.cfm_coeff,
                c.cfm_gain
            );
        }
    }

    /// Build the per-env `RobotState` from a `body_poses` slurp ONLY (no
    /// `links_workspace`). Pure with respect to `&self` — the parallel post-
    /// step loop calls this read-only and the caller commits the returned
    /// `new_joint_pos` into `self.prev_joint_pos[env]` afterwards.
    ///
    /// Joint angles come from `q_child = q_parent · rest_quat · R_z(θ)`,
    /// inverted to `θ = 2·atan2(rel.z, rel.w)` with
    /// `rel = rest_quat⁻¹ · q_parent⁻¹ · q_child` (see `LinkIndices`).
    /// Joint velocities, base linear/angular velocity, and base height are
    /// finite-diffed at the control rate (20 ms) against the cached previous
    /// poses — first step gets zero velocity (mirrors the existing
    /// `has_prev_joint_pos` semantics).
    fn read_state_from_poses(
        &self,
        env: usize,
        poses: &[NexusPose],
    ) -> (RobotState, [f32; NUM_JOINTS]) {
        let cpb = self.idx.colliders_per_batch as usize;
        let env_base = env * cpb;
        let control_dt = self.task.control_dt();

        let torso_pose = &poses[env_base + self.idx.torso_link as usize];
        let t = torso_pose.translation;
        let r = torso_pose.rotation;

        // Base linear / angular velocity by finite-diff vs last step's torso
        // pose. ω from the small-rotation approximation
        // `ω ≈ 2 · (Δq.xyz)/dt` with hemisphere correction so antipodal
        // quaternions don't blow it up. Zero on the first step (no prev).
        let (lv, av) = if self.has_prev_pose[env] {
            let prev = &self.prev_body_poses[env_base + self.idx.torso_link as usize];
            let pt = prev.translation;
            let lv = Vec3::new(
                (t.x - pt.x) / control_dt,
                (t.y - pt.y) / control_dt,
                (t.z - pt.z) / control_dt,
            );
            let dq_raw = r * prev.rotation.conjugate();
            let s = if dq_raw.w >= 0.0 { 1.0 } else { -1.0 };
            let av = Vec3::new(
                2.0 * s * dq_raw.x / control_dt,
                2.0 * s * dq_raw.y / control_dt,
                2.0 * s * dq_raw.z / control_dt,
            );
            (lv, av)
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };
        let base = BaseState {
            orientation: [r.x, r.y, r.z, r.w],
            lin_vel_world: [lv.x, lv.y, lv.z],
            ang_vel_world: [av.x, av.y, av.z],
            height: t.z,
        };

        // Joint angles from parent⇄child relative rotation (see doc comment).
        let mut joint_pos = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            let parent_link = self.idx.actuated_parent_links[k] as usize;
            let child_link = self.idx.actuated[k].0 as usize;
            let qp = poses[env_base + parent_link].rotation;
            let qc = poses[env_base + child_link].rotation;
            let rest = self.idx.actuated_rest_quat[k];
            let rel = rest.conjugate() * qp.conjugate() * qc;
            joint_pos[k] = 2.0 * rel.z.atan2(rel.w);
        }
        let mut joint_vel = [0.0f32; NUM_JOINTS];
        if self.has_prev_joint_pos[env] {
            for k in 0..NUM_JOINTS {
                joint_vel[k] = (joint_pos[k] - self.prev_joint_pos[env][k]) / control_dt;
            }
        }

        (
            RobotState {
                base,
                joint_pos,
                joint_vel,
                last_action: self.last_action[env],
                prev_action: self.prev_action[env],
                feet: [FootObs::default(); NUM_FEET],
            },
            joint_pos,
        )
    }

    /// Per-foot observation for one env from `body_poses` ONLY.
    /// Pure with respect to `&self` — returns the new air-time array alongside
    /// the `FootObs` row; the caller commits it into `self.air_time[env]`.
    /// Foot linear velocity is finite-diffed against the previous step's foot
    /// pose (so we don't need `ws.rb_vels`); contact is still synthesised by
    /// foot Z < threshold (nexus doesn't expose narrow-phase pairs).
    fn compute_feet_from_poses(
        &self,
        env: usize,
        poses: &[NexusPose],
    ) -> ([FootObs; NUM_FEET], [f32; NUM_FEET]) {
        const CONTACT_Z: f32 = 0.025;
        let dt = self.task.control_dt();
        let cpb = self.idx.colliders_per_batch as usize;
        let env_base = env * cpb;

        let base_rot = poses[env_base + self.idx.torso_link as usize].rotation;
        let base_rot_inv = base_rot.conjugate();
        let sole_local = self.foot_sole_local[env];
        let has_prev = self.has_prev_pose[env];
        let mut out = [FootObs::default(); NUM_FEET];
        let mut new_air = [0.0f32; NUM_FEET];
        for i in 0..NUM_FEET {
            let link = self.idx.foot_links[i] as usize;
            let foot_pose = &poses[env_base + link];
            let pos = foot_pose.translation;
            let planar_speed = if has_prev {
                let prev_pos = self.prev_body_poses[env_base + link].translation;
                let dx = (pos.x - prev_pos.x) / dt;
                let dy = (pos.y - prev_pos.y) / dt;
                (dx * dx + dy * dy).sqrt()
            } else {
                0.0
            };
            let world_normal = foot_pose.rotation * sole_local[i];
            let tilt = world_normal.z.abs().clamp(0.0, 1.0).acos();
            let foot_x_in_base = (base_rot_inv * foot_pose.rotation) * Vec3::X;
            let yaw_rel_base = foot_x_in_base.y.atan2(foot_x_in_base.x);
            let contact = pos.z < CONTACT_Z;
            let prev_air = self.air_time[env][i];
            let first_contact = contact && prev_air > 0.0;
            new_air[i] = if contact { 0.0 } else { prev_air + dt };
            out[i] = FootObs {
                contact,
                first_contact,
                air_time: if contact { prev_air } else { new_air[i] },
                height: pos.z,
                planar_speed,
                tilt,
                yaw_rel_base,
                pos_xy: [pos.x, pos.y],
            };
        }
        (out, new_air)
    }

    /// Step every env one control tick. Returns per-env `StepOut`s in
    /// env-index order. Async because both pipeline.step and the readback are
    /// async on the WebGPU backend.
    ///
    /// Hot-path layout (after the Tier-1 perf rework):
    /// 1. Stage motor targets + flush → `pipeline.step × decimation`.
    /// 2. ONE readback: `body_poses` only (was `body_poses + links_workspace`
    ///    every step; the latter dominated host time at large N).
    /// 3. Serial pre-pass: bump `step_count`, resample commands on schedule.
    /// 4. **Parallel** rayon block: derive joint angles from parent⇄child
    ///    relative rotation, finite-diff base + foot velocities, build obs /
    ///    critic_obs / reward. All read-only against `&self`, so envs run
    ///    independently across worker threads.
    /// 5. Serial post-pass: commit per-env mutable state (air_time, prev_*,
    ///    action history), assemble `StepOut`s.
    /// Physics-only throughput A/B for the GPU-resident rollout: time the
    /// decimation loop run with a host `synchronize()` per control step (the
    /// current rollout pattern — the per-step stall we diagnosed) vs captured
    /// ONCE into a CUDA graph and replayed with a single `cuGraphLaunch` per
    /// step (zero host encode/submit/sync between the ~decimation×N dispatches).
    /// Returns `(sync_ms, graph_ms)` for `t_steps` control steps; `graph_ms` is
    /// `None` on non-CUDA backends. A fixed zero-action target is staged once so
    /// the captured sequence has stable inputs (and `BIPED_FIXED_GRID=1` must be
    /// set so there are no indirect-dispatch host readbacks to break capture).
    #[cfg(feature = "cuda_backend")]
    pub async fn bench_physics_modes(&mut self, t_steps: usize) -> (f64, Option<f64>) {
        // Stage one fixed (zero-action) motor target + flush — stable physics
        // input, no per-step staging in the timed loops.
        let targets = self.task.joint_targets(&[0.0; NUM_JOINTS]);
        for e in 0..self.n {
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
            .expect("flush");
        let decim = self.task.decimation;

        // Warmup so the color count / buffers stabilise (capture must not realloc).
        for _ in 0..32 {
            for _ in 0..decim {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None).await;
            }
        }
        self.gpu.synchronize().expect("warmup sync");

        // ---- SYNC: host synchronize() per control step ----
        let t0 = Instant::now();
        for _ in 0..t_steps {
            for _ in 0..decim {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None).await;
            }
            self.gpu.synchronize().expect("sync");
        }
        let sync_ms = t0.elapsed().as_secs_f64() * 1e3;

        // ---- GRAPH: capture one decimation loop, replay it per step ----
        let graph_ms = if let Some(cuda) = self.gpu.as_cuda() {
            cuda.begin_capture().expect("begin_capture");
            for _ in 0..decim {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None).await;
            }
            let graph = cuda.end_capture().expect("end_capture");
            graph.upload().ok();
            graph.launch().expect("first graph launch"); // capture records, run once
            self.gpu.synchronize().expect("sync after first launch");
            let t0 = Instant::now();
            for _ in 0..t_steps {
                graph.launch().expect("graph replay");
            }
            self.gpu.synchronize().expect("graph sync");
            Some(t0.elapsed().as_secs_f64() * 1e3)
        } else {
            None
        };

        (sync_ms, graph_ms)
    }

    pub async fn step(&mut self, actions: &[[f32; NUM_JOINTS]]) -> Vec<StepOut> {
        assert_eq!(actions.len(), self.n);

        // (1) Stage every env's motor targets host-side in the mirror, then
        // push the whole `links_static` buffer in ONE write_buffer call.
        // Replaces `num_envs * NUM_JOINTS` per-step write_buffer calls.
        let t = Instant::now();
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
        self.timings.stage_motors_ns += t.elapsed().as_nanos() as u64;

        let t = Instant::now();
        self.state
            .multibodies_mut()
            .flush_links_static(&self.gpu)
            .expect("flush motor targets");
        self.timings.flush_static_ns += t.elapsed().as_nanos() as u64;

        // (2) Advance physics at the control decimation. Each `pipeline.step`
        // is async — the await may include queue submit + any implicit GPU
        // sync the backend needs between sub-steps.
        let t = Instant::now();
        for _ in 0..self.task.decimation {
            let _ = self.pipeline.step(&self.gpu, &mut self.state, None).await;
        }
        self.timings.pipeline_step_ns += t.elapsed().as_nanos() as u64;

        // Explicit `gpu.synchronize()` so the timing buckets cleanly split
        // "wait for GPU compute to finish" from "transfer bytes back". In
        // production this sync isn't needed — the next `slow_read_buffer`
        // syncs implicitly — but for profiling it lets us see how much of
        // the per-step budget is actual GPU work vs host-side transfer.
        let t = Instant::now();
        self.gpu.synchronize().expect("sync");
        self.timings.gpu_wait_ns += t.elapsed().as_nanos() as u64;

        // `auto_resize_buffers` runs only every `AUTO_RESIZE_PERIOD` steps;
        // for a static scene it stabilises after warmup and per-step calls
        // just add dispatch latency for no work.
        self.tick_since_resize += 1;
        if self.tick_since_resize >= AUTO_RESIZE_PERIOD {
            let t = Instant::now();
            self.pipeline
                .auto_resize_buffers(&self.gpu, &mut self.state)
                .await;
            self.timings.auto_resize_ns += t.elapsed().as_nanos() as u64;
            self.tick_since_resize = 0;
        }

        // (3) Single readback: body_poses (the only one left post-Tier-1).
        // After the explicit sync above, this should be just staging copy +
        // map_async + memcpy — the time *attributed* to the readback now is
        // close to its real cost, not the GPU compute that piggybacks on the
        // implicit drain.
        let t = Instant::now();
        let poses = self.slurp_poses().await;
        self.timings.readback_ns += t.elapsed().as_nanos() as u64;

        // (4) Serial pre-pass: step_count + command resample. Cheap; can't
        // easily live in the parallel block (needs `&mut self.rng[e]`).
        let t = Instant::now();
        for e in 0..self.n {
            self.step_count[e] += 1;
            if self.step_count[e] >= self.resample_at[e] {
                self.cmd[e] = self.sampler.sample(&mut self.rng[e]);
                self.resample_at[e] = self.step_count[e]
                    + self
                        .sampler
                        .resample_steps(&mut self.rng[e], self.task.control_dt());
            }
        }
        self.timings.serial_pre_ns += t.elapsed().as_nanos() as u64;

        // (4) Parallel heavy compute. Inputs: read-only `&self` slices indexed
        // by env. Output: per-env tuple of obs/critic/reward/fell + the new
        // air-time + new joint-pos snapshot (committed serially below).
        // `with_min_len(64)` chunks the work so rayon's per-task overhead
        // (~µs) amortises across many envs.
        struct PerEnv {
            obs: Vec<f32>,
            critic_obs: Vec<f32>,
            reward: f32,
            fell: bool,
            illegal: bool,
            // Per-term reward breakdown for logging (W&B). Indices:
            // Per-term reward contributions, in `REWARD_COMP_NAMES` order.
            comps: [f32; NUM_REWARD_COMPS],
            new_air: [f32; NUM_FEET],
            new_joint_pos: [f32; NUM_JOINTS],
        }
        let t = Instant::now();
        // WBC-AGILE-style illegal-ground-contact termination: only the feet have
        // ground colliders in nexus, so the policy can clip thigh/shin/hip links
        // through the floor for free support (we measured shins ~3 cm below the
        // floor in an early policy). Terminate if any monitored link drops below
        // `BIPED_ILLEGAL_Z`. Default 0.0 = actual floor penetration only: a
        // trained policy's shins sit ~+0.046 m, so 0.0 catches real clipping
        // (the −0.03 case) without over-terminating legitimate low stances —
        // 0.06 was too tight and killed the learning gradient. Set large-negative
        // to disable entirely.
        let illegal_z = std::env::var("BIPED_ILLEGAL_Z")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        // WBC-AGILE-style self-collision avoidance, as a SOFT reward penalty
        // (not a hard termination). nexus can't do physical leg-leg collision
        // (inert leg colliders), and a hard distance-termination is the ONLY
        // guard here so it fires every episode for a from-scratch policy and
        // buries the gradient (measured: falls 6.8k→46k). Instead, smoothly
        // penalize each left/right link pair (foot/shin/thigh) by how far it
        // intrudes inside `sc_margin`: `penalty = w · Σ max(0, margin − dist)`.
        // DEFAULT OFF (weight 0): the real per-joint angle limits already keep the
        // legs apart — measured min L/R separation is 0.105 m (shins) with limits
        // and no penalty, well above the ~0.07 crossing threshold. The joint
        // ranges (esp. hipx ±20° ad/abduction) are designed so the reachable
        // workspace doesn't self-collide, so an explicit distance penalty is
        // redundant AND competes with learning. Kept as opt-in (`BIPED_SELF_COLL_W`)
        // for cases the limits don't cover (e.g. foot↔torso). margin 0.12 m.
        let sc_margin = std::env::var("BIPED_SELF_COLL_DIST")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.12);
        let sc_weight = std::env::var("BIPED_SELF_COLL_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let sc_dt = self.task.control_dt();
        // Torque (effort) penalty: we're PD position-controlled and had NO cost
        // on joint torque, so the policy reward-hacks strained high-torque poses
        // (e.g. balancing on one leg at saturated effort). Reconstruct the
        // applied PD torque τ = clamp(kp·(q_target−q) − kd·q̇, ±effort) and
        // penalize Στ², mirroring WBC-AGILE's lerobot config: base -5e-4 on all
        // leg joints, an extra -1e-3 on the (weaker) ankles, and an extra -5e-3
        // on ankle-roll. Scaled by `self.torque_scale` (the trainer's curriculum
        // hook, init from `BIPED_TORQUE_W`): full WBC weight from scratch breaks
        // learning (falls 6k→12.6k) because a torque penalty fights "learn to
        // stand at all", so the trainer ramps it in only after standing is
        // learned (set_torque_scale 0→target). 0 disables.
        let torque_w = self.torque_scale;
        // Ankle torque is penalized at FULL strength AT ALL TIMES (not ramped by
        // the curriculum) — the real ankle motor is fragile (~11 N·m diamond vs
        // the sim's 44), so we discourage ankle torque from iter 0. Soft (a
        // penalty, not a hard effort cap) to keep learning feasible. Scale via
        // BIPED_ANKLE_TORQUE_W (default 1.0; raise if rollouts still show high
        // ankle torque, 0 disables).
        let ankle_torque_w = std::env::var("BIPED_ANKLE_TORQUE_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        let cpb_idx = self.idx.colliders_per_batch as usize;
        let computed: Vec<PerEnv> = (0..self.n)
            .into_par_iter()
            .with_min_len(64)
            .map(|e| {
                let (feet, new_air) = self.compute_feet_from_poses(e, &poses);
                let (mut state, new_joint_pos) = self.read_state_from_poses(e, &poses);
                state.feet = feet;
                let env_base = e * cpb_idx;
                let illegal = self
                    .idx
                    .illegal_ground_links
                    .iter()
                    .any(|&l| poses[env_base + l as usize].translation.z < illegal_z);
                let fell = illegal
                    || self.task.fell_over(&state.base)
                    || !state.base.height.is_finite();
                let rb = self.task.reward(&state, &self.cmd[e]);
                let mut reward = rb.total();
                let mut comps = [0.0f32; NUM_REWARD_COMPS];
                comps[0] = rb.track_lin_vel;
                comps[1] = rb.track_ang_vel;
                comps[2] = rb.upright;
                comps[3] = rb.base_height;
                comps[4] = rb.pose;
                comps[5] = rb.bilateral_symmetry;
                comps[6] = rb.action_rate;
                comps[7] = rb.action_rate_hipz_hipx;
                comps[8] = rb.body_ang_vel;
                comps[9] = rb.lin_vel_z;
                comps[10] = rb.dof_pos_limits;
                comps[11] = rb.dof_vel;
                comps[12] = rb.air_time;
                comps[13] = rb.flight;
                comps[14] = rb.single_support;
                comps[15] = rb.foot_slip;
                comps[16] = rb.foot_clearance;
                comps[17] = rb.foot_orientation;
                comps[18] = rb.feet_yaw_mean;
                comps[19] = rb.feet_distance;
                if fell {
                    comps[23] = self.task.weights.termination;
                    reward += self.task.weights.termination;
                }
                // Soft self-collision penalty: ramp up as any L/R pair intrudes
                // inside `sc_margin` (legs crossing). ~0 for a clean stance.
                if sc_weight > 0.0 {
                    let intrusion: f32 = self
                        .idx
                        .self_collision_pairs
                        .iter()
                        .map(|&(a, b)| {
                            let pa = poses[env_base + a as usize].translation;
                            let pb = poses[env_base + b as usize].translation;
                            (sc_margin - (pa - pb).length()).max(0.0)
                        })
                        .sum();
                    let sc_pen = sc_weight * intrusion * sc_dt;
                    comps[22] = -sc_pen;
                    reward -= sc_pen;
                }
                // Torque (effort) penalty — reconstruct the applied PD torque per
                // joint and penalize Στ². The ANKLE motors are fragile hardware
                // (real diamond limit ~11 N·m vs the sim's 44), so the ankle term
                // is FULL-STRENGTH AT ALL TIMES (`ankle_torque_w`, not ramped),
                // while the leg term ramps with the curriculum (`torque_w`). WBC
                // lerobot base weights: -5e-4 legs, -1.5e-3 ankle pitch, -6.5e-3
                // ankle roll (coupled, weakest).
                if torque_w > 0.0 || ankle_torque_w > 0.0 {
                    let q_target = self.task.joint_targets(&actions[e]);
                    let mut leg_pen = 0.0f32;
                    let mut ankle_pen = 0.0f32;
                    for i in 0..NUM_JOINTS {
                        let j = &self.task.robot.joints[i];
                        let tau = (j.kp * (q_target[i] - state.joint_pos[i])
                            - j.kd * state.joint_vel[i])
                            .clamp(-j.effort_limit, j.effort_limit);
                        let t2 = tau * tau;
                        if j.name.contains("ankle") {
                            let w = if j.name.contains("anklex") { 6.5e-3 } else { 1.5e-3 };
                            ankle_pen += w * t2;
                        } else {
                            leg_pen += 5e-4 * t2;
                        }
                    }
                    comps[20] = -(torque_w * leg_pen) * sc_dt;
                    comps[21] = -(ankle_torque_w * ankle_pen) * sc_dt;
                    reward -= (torque_w * leg_pen + ankle_torque_w * ankle_pen) * sc_dt;
                }
                let mut obs = vec![0.0; OBS_DIM];
                self.task.observe(&state, &self.cmd[e], &mut obs);
                let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
                self.task
                    .observe_critic(&state, &self.cmd[e], &mut critic_obs);
                PerEnv {
                    obs,
                    critic_obs,
                    reward,
                    fell,
                    illegal,
                    comps,
                    new_air,
                    new_joint_pos,
                }
            })
            .collect();
        self.timings.par_compute_ns += t.elapsed().as_nanos() as u64;

        // (5) Serial commit: per-env mutable state + StepOut assembly.
        let t = Instant::now();
        let cpb = self.idx.colliders_per_batch as usize;
        let mut outs = Vec::with_capacity(self.n);
        for (e, c) in computed.into_iter().enumerate() {
            self.air_time[e] = c.new_air;
            self.prev_joint_pos[e] = c.new_joint_pos;
            self.has_prev_joint_pos[e] = true;
            // Snapshot poses for this env into prev_body_poses for the next
            // step's finite-diff base / foot velocities.
            let env_base = e * cpb;
            self.prev_body_poses[env_base..env_base + cpb]
                .copy_from_slice(&poses[env_base..env_base + cpb]);
            self.has_prev_pose[e] = true;
            self.prev_action[e] = self.last_action[e];
            self.last_action[e] = actions[e];
            let timeout = self.step_count[e] >= self.task.max_steps();
            // Accumulate per-component reward + termination causes for W&B
            // (drained by `take_reward_log`). Every (env, step) contributes to
            // the component means; termination counters tally episode ends.
            for i in 0..NUM_REWARD_COMPS {
                self.rlog_comps[i] += c.comps[i] as f64;
            }
            self.rlog_steps += 1;
            if c.illegal {
                self.rlog_illegal += 1;
            } else if c.fell {
                self.rlog_fell += 1;
            } else if timeout {
                self.rlog_timeout += 1;
            }
            outs.push(StepOut {
                obs: c.obs,
                critic_obs: c.critic_obs,
                reward: c.reward,
                done: c.fell || timeout,
                fell: c.fell,
            });
        }
        self.timings.serial_commit_ns += t.elapsed().as_nanos() as u64;
        self.timings.steps += 1;
        outs
    }

    /// Drain the accumulated per-component reward + termination stats since the
    /// last call and reset the counters. Returns `None` if no steps were taken
    /// (nothing to log). The trainer calls this once per PPO iteration to emit a
    /// structured line the W&B sidecar logs.
    pub fn take_reward_log(&mut self) -> Option<RewardLog> {
        if self.rlog_steps == 0 {
            return None;
        }
        let n = self.rlog_steps as f64;
        let mut comps = [0.0f32; NUM_REWARD_COMPS];
        for i in 0..NUM_REWARD_COMPS {
            comps[i] = (self.rlog_comps[i] / n) as f32;
        }
        let out = RewardLog {
            comps,
            illegal: self.rlog_illegal,
            fell: self.rlog_fell,
            timeout: self.rlog_timeout,
            samples: self.rlog_steps,
        };
        self.rlog_comps = [0.0; NUM_REWARD_COMPS];
        self.rlog_steps = 0;
        self.rlog_illegal = 0;
        self.rlog_fell = 0;
        self.rlog_timeout = 0;
        Some(out)
    }

    /// Read the accumulated per-phase timings and reset the counters.
    /// Pair with the timed loop in `biped_fps.rs` to get a breakdown of
    /// where the per-step budget went.
    pub fn take_step_timings(&mut self) -> StepTimings {
        std::mem::take(&mut self.timings)
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

        // Cached prev joint angles + poses are stale across a reset; clear so
        // the next step seeds them again with zero velocity.
        self.has_prev_joint_pos[env] = false;
        self.has_prev_pose[env] = false;
        // Build the initial obs from the freshly-copied state.
        let poses = self.slurp_poses().await;
        let (feet, _) = self.compute_feet_from_poses(env, &poses);
        let (mut state, _) = self.read_state_from_poses(env, &poses);
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
        let poses = self.slurp_poses().await;
        let mut obs = Vec::with_capacity(self.n);
        let mut critic_obs = Vec::with_capacity(self.n);
        for e in 0..self.n {
            let (feet, _) = self.compute_feet_from_poses(e, &poses);
            let (mut state, _) = self.read_state_from_poses(e, &poses);
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
        self.has_prev_pose[e] = false;
        let poses = self.slurp_poses().await;
        let (feet, _) = self.compute_feet_from_poses(e, &poses);
        let (mut state, _) = self.read_state_from_poses(e, &poses);
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

    /// Joint angles (rad) in `JOINT_NAMES` order for env `e`. Derived from
    /// `body_poses` via the same parent⇄child relative-rotation formula the
    /// step path uses — no `links_workspace` readback needed.
    pub fn joint_angles_for(&self, e: usize, poses: &[NexusPose]) -> [f32; NUM_JOINTS] {
        let cpb = self.idx.colliders_per_batch as usize;
        let base = e * cpb;
        let mut q = [0.0f32; NUM_JOINTS];
        for k in 0..NUM_JOINTS {
            let parent_link = self.idx.actuated_parent_links[k] as usize;
            let child_link = self.idx.actuated[k].0 as usize;
            let qp = poses[base + parent_link].rotation;
            let qc = poses[base + child_link].rotation;
            let rest = self.idx.actuated_rest_quat[k];
            let rel = rest.conjugate() * qp.conjugate() * qc;
            q[k] = 2.0 * rel.z.atan2(rel.w);
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

    /// One slurped snapshot for rendering — returns only `body_poses` now.
    /// `body_positions_for` / `base_pose_for` / `joint_angles_for` all consume
    /// it directly; the `links_workspace` readback was only needed for
    /// joint-angle extraction, which now goes through parent⇄child relative
    /// rotation off `body_poses` (same as the step path).
    pub async fn snapshot(&mut self) -> Vec<NexusPose> {
        self.slurp_poses().await
    }

    /// Telemetry: torso heights across all envs.
    pub async fn torso_heights(&mut self) -> Vec<f32> {
        let poses = self.slurp_poses().await;
        (0..self.n)
            .map(|e| {
                let i = e * self.idx.colliders_per_batch as usize + self.idx.torso_link as usize;
                poses[i].translation.z
            })
            .collect()
    }
}

// --- Helpers -----------------------------------------------------------------

/// Pick the GPU backend for the batched physics. Defaults to WebGPU; when the
/// `cuda_backend` feature is compiled in AND `BIPED_CUDA=1`, runs the native
/// CUDA (cuda-oxide) backend instead — used by the all-native e2e benchmark.
/// The nexus + vortx cubins are embedded at build time via the per-crate
/// `CUDA_OXIDE_SHADERS_PTX_*` env vars (see khal-builder `build_ptx`).
async fn make_backend() -> KhalGpuBackend {
    #[cfg(feature = "cuda_backend")]
    {
        if std::env::var("BIPED_CUDA").as_deref() == Ok("1") {
            use khal::backend::Cuda;
            eprintln!("[biped] backend = native CUDA (cuda-oxide)");
            return KhalGpuBackend::Cuda(Cuda::new(0).expect("init CUDA backend"));
        }
    }
    webgpu_backend().await
}

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
    // BIPED_SPAWN_DR scales the initial-pose tilt/height randomization (default
    // 1.0). Set to 0.0 to start every episode upright at nominal height — used to
    // test whether aggressive spawn DR is what's preventing the policy from
    // getting a learning gradient (the rng draws are still consumed, so dynamics
    // DR and determinism are unchanged).
    let sdr: f32 = std::env::var("BIPED_SPAWN_DR").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // BIPED_FRICTION: force a fixed Coulomb μ on every env (overrides the random
    // draw) — used to A/B-test that friction actually reaches the GPU contact
    // solver. The rng draw is still consumed so other DR + determinism are
    // unchanged.
    let friction = match std::env::var("BIPED_FRICTION").ok().and_then(|s| s.parse::<f32>().ok()) {
        Some(f) => { let _ = rng.range(0.5, 1.5); f }
        None => rng.range(0.5, 1.5),
    };
    DrParams {
        friction,
        restitution: rng.range(0.0, 0.15),
        pd_scale: rng.range(0.85, 1.15),
        contact_natural_frequency: rng.range(10.0, 50.0),
        contact_damping_ratio: rng.range(2.0, 8.0),
        // Initial-pose DR — aggressive ranges so the policy sees a wide
        // distribution of starts and learns to recover from non-trivial
        // perturbations. Comparable to WBC-AGILE / Isaac Lab humanoid
        // defaults (±15–25° on tilts, a few cm on height). Wider than this
        // (e.g. ±30° tilts) makes most episodes start mid-fall and PPO
        // can't get a useful gradient with the curriculum's early
        // command-velocity scale.
        spawn_yaw: rng.range(-std::f32::consts::PI, std::f32::consts::PI),
        spawn_roll: rng.range(-0.35, 0.35) * sdr,     // ±~20° (× BIPED_SPAWN_DR)
        spawn_pitch: rng.range(-0.35, 0.35) * sdr,    // ±~20° (× BIPED_SPAWN_DR)
        spawn_z_offset: rng.range(-0.08, 0.08) * sdr, // ±8 cm (× BIPED_SPAWN_DR)
    }
}
