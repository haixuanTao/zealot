//! Vectorized N-env biped environment on nexus GPU physics.
//!
//! `BipedNexusBatchEnv` owns one `RbdState` holding N parallel envs and
//! the host-side bookkeeping each env needs (RNG, current command, step counter,
//! action history, air-time per foot). One `pipeline.step(...)` advances every
//! env on the GPU; one `slow_read_buffer(links_workspace)` brings the full
//! per-link state back to host where we compute obs/reward per env using the
//! same `VelocityFlatTask` the CPU env uses.
//!
//! What's mirrored from `biped_env.rs`:
//! - MJCF scene build (per env), foot box collider, PD motors, dynamic root.
//! - Per-env friction / restitution / contact-softness / PD-scale randomization
//!   (baked into the rapier scene + `RbdSimParams` before `from_rapier`).
//! - Episode-end reset via pre-built spawn templates + `state.reset_env_from`.
//!
//! What's NOT mirrored (nexus host API doesn't expose them):
//! - True foot-ground contact pairs (synthesized via foot Z < threshold).
//!
//! Push perturbations ARE supported (Isaac's `push_by_setting_velocity`
//! equivalent): a read-modify-write of the root free-joint velocity DOFs in
//! `dof_state` — see `apply_random_pushes` (BIPED_PUSH_VEL / BIPED_PUSH_ANGVEL
//! / BIPED_PUSH_INTERVAL).
//!
//! Joint angles / velocities, base linear / angular velocity all come from
//! `links_workspace[k].{coords, joint_rot, rb_vels}` (rb_vels is world-space).

use khal::backend::{Backend, Buffer, GpuBackend as KhalGpuBackend};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::RbdSimParams;
use nexus3d::rbd::math::Pose as NexusPose;
use nexus3d::rbd::pipeline::{RbdPipeline, RbdSnapshot, RbdState};
use nexus3d::rbd::queries::GpuIndexedContact as NexusIndexedContact;
use nexus3d::rbd::shaders::dynamics::MultibodyContactConstraint as NexusMbContact;
use nexus3d::rbd::shaders::dynamics::MultibodyLinkWorkspace;
use rapier3d::prelude::*;
use rayon::prelude::*;
use roxmltree::Node;
use std::collections::HashMap;
use std::time::Instant;
use zealot_env::obs_history::ObsHistory;
use zealot_env::rng::Lcg;
use zealot_env::terrain::{TerrainCurriculum, TerrainFamily, TerrainStrip};
use zealot_env::robots::{RobotSpec, NUM_JOINTS};
use zealot_env::tasks::velocity_flat::{
    BaseState, CRITIC_OBS_DIM, CommandSampler, FootObs, NUM_FEET, OBS_DIM, RobotState,
    VelocityCommand, VelocityFlatTask,
};

// Spawn height comes from the robot spec (`RobotSpec::spawn_z` — the
// straight-leg sole-on-ground height; the multibody rest pose is q = 0).
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
/// Steps to run eager before capturing the physics CUDA graph — long enough for
/// the dispatch structure (color count / buffer sizes) to stabilise through a
/// couple of `auto_resize_buffers` cycles, so the captured graph stays valid.
const GRAPH_CAPTURE_AT: u32 = 64;

/// `Send`+`Sync` wrapper for a captured physics graph. `CapturedGraph` holds raw
/// CUDA handles (not thread-safe), but the env is shared by-ref with rayon in the
/// par-compute closure — which NEVER touches the graph (it's launched only on the
/// main thread in `step`). The unsafe impls assert that main-thread-only access,
/// which holds for our usage.
#[cfg(feature = "cuda_backend")]
struct SyncGraph(khal::backend::cuda::CapturedGraph);
#[cfg(feature = "cuda_backend")]
unsafe impl Send for SyncGraph {}
#[cfg(feature = "cuda_backend")]
unsafe impl Sync for SyncGraph {}

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
    /// Visual mesh geoms on this link: `(mesh_name, local_pos, local_quat)`.
    /// Collected for the optional convex-hull collider path (`BIPED_FOOT_SHAPE=convex`).
    pub mesh_geoms: Vec<(String, Vec3, Rotation)>,
    /// Mesh vertices (link frame) used to build a convex-hull collider. Filled by
    /// `load_mesh_hulls` only when the convex foot shape is requested; empty otherwise.
    pub mesh_pts: Vec<Vec3>,
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
    let mut mesh_geoms = Vec::new();
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
            "geom" if c.attribute("type") == Some("mesh") => {
                if let Some(name) = c.attribute("mesh") {
                    mesh_geoms.push((name.to_string(), vec3(&c, "pos", Vec3::ZERO), quat_wxyz(&c)));
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
            mesh_geoms,
            mesh_pts: Vec::new(),
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

/// MJCF path for the robot selected by `BIPED_ROBOT` (see
/// [`RobotSpec::from_env`]); `BIPED_MJCF` overrides it with an explicit path.
pub fn default_mjcf_path() -> String {
    if let Ok(p) = std::env::var("BIPED_MJCF") {
        return p;
    }
    RobotSpec::from_env().mjcf_path().to_string_lossy().into_owned()
}

/// Minimal binary-STL vertex loader. Returns every triangle vertex (unindexed) —
/// `ColliderBuilder::convex_hull` only needs the point cloud. Handles binary STL
/// (the format the onshape CAD export uses). Returns empty on any error.
fn load_stl_vertices(path: &std::path::Path) -> Vec<Vec3> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    if bytes.len() < 84 {
        return Vec::new();
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    if bytes.len() < 84 + n * 50 {
        return Vec::new(); // not a well-formed binary STL
    }
    let rd = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut pts = Vec::with_capacity(n * 3);
    for t in 0..n {
        let base = 84 + t * 50 + 12; // skip the 12-byte triangle normal
        for v in 0..3 {
            let o = base + v * 12;
            pts.push(Vec3::new(rd(o), rd(o + 4), rd(o + 8)));
        }
    }
    pts
}

/// Fill `mesh_pts` (link-frame vertices for the convex-hull collider path) on each
/// link that carries visual mesh geoms. Resolves STL files via the MJCF
/// `<asset><mesh>` table + `<compiler meshdir>` relative to the robot.xml dir
/// (override with `BIPED_MESH_DIR`). Called only when the convex foot shape is
/// requested, so the one-time STL read is skipped otherwise.
pub fn load_mesh_hulls(mjcf: &mut [MjBody], xml: &str) {
    let Ok(doc) = roxmltree::Document::parse(xml) else {
        return;
    };
    let assets: HashMap<String, String> = doc
        .descendants()
        .filter(|n| n.tag_name().name() == "mesh")
        .filter_map(|n| {
            let file = n.attribute("file")?.to_string();
            // MuJoCo: an unnamed `<mesh file="foo.stl"/>` is referenced by the file
            // basename sans extension ("foo"). This model omits `name` entirely.
            let name = n.attribute("name").map(str::to_string).unwrap_or_else(|| {
                std::path::Path::new(&file)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&file)
                    .to_string()
            });
            Some((name, file))
        })
        .collect();
    let meshdir = doc
        .descendants()
        .find(|n| n.tag_name().name() == "compiler")
        .and_then(|n| n.attribute("meshdir"))
        .unwrap_or("assets");
    let base = std::path::Path::new(&default_mjcf_path())
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let asset_dir = std::env::var("BIPED_MESH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| base.join(meshdir));
    for b in mjcf.iter_mut() {
        // Only links that actually get a collider (the feet — the others are inert
        // placeholders, see the scene builder) need a hull. Skip the rest so we
        // don't hull the huge thigh/shin meshes (~700k verts) for nothing.
        if b.capsules.is_empty() {
            continue;
        }
        let mut pts = Vec::new();
        for (name, pos, quat) in &b.mesh_geoms {
            let Some(file) = assets.get(name) else {
                continue;
            };
            for v in load_stl_vertices(&asset_dir.join(file)) {
                pts.push(*quat * v + *pos);
            }
        }
        // Reduce the raw mesh cloud (~10^5 verts) to the convex-hull vertices ONCE
        // here, so each of the thousands of per-env colliders re-hulls only a few
        // dozen points instead of the whole mesh (env build was minutes otherwise).
        if !pts.is_empty() {
            if let Ok((hull, _)) = rapier3d::parry::transformation::try_convex_hull(&pts) {
                eprintln!(
                    "[convex] link '{}': {} mesh verts -> {}-vertex hull collider",
                    b.name,
                    pts.len(),
                    hull.len()
                );
                pts = hull;
            }
        }
        b.mesh_pts = pts;
    }
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
    /// Extra multiplier on kd ONLY (default 1.0 → kd follows `pd_scale`).
    /// AGILE-parity DR (`BIPED_AGILE_DR=1`) randomizes damping on a wider,
    /// independent range from stiffness (kd ×U(0.8,2.0) vs kp ×U(0.9,1.1)).
    pub kd_scale: f32,
    /// Per-env multiplier on every link's mass (and, to stay physically
    /// consistent, its inertia tensor). Models payload / build-tolerance /
    /// CAD-vs-reality mass error. ~±20% by default.
    pub mass_scale: f32,
    /// Additive payload on the ROOT link only, kg (AGILE randomize_base_mass:
    /// +U(−1,5) kg on the pelvis; mass only, inertia untouched — matches
    /// Isaac's `operation: add`). Default 0.
    pub base_payload_kg: f32,
    pub contact_natural_frequency: f32,
    pub contact_damping_ratio: f32,
    /// Sampled base orientation at spawn — separate axes so a single template
    /// can mix yaw / roll / pitch. Each in rad.
    pub spawn_yaw: f32,
    pub spawn_roll: f32,
    pub spawn_pitch: f32,
    /// Sampled additive jitter on the spawn height, m. May be negative.
    pub spawn_z_offset: f32,
    /// Per-actuated-joint gain/torque multiplier (independent draw per joint).
    /// Models actuator-strength asymmetry — e.g. one hip motor stronger than its
    /// mirror, or a worn/weaker joint — the asymmetry a perfectly symmetric
    /// policy must handle REACTIVELY on the real robot. Independent per joint so
    /// left/right differ; scales kp, kd, and the effort (torque) limit together.
    /// Default `[1.0; NUM_JOINTS]` (symmetric, nominal).
    pub pd_scale_per_joint: [f32; NUM_JOINTS],
}

impl Default for DrParams {
    fn default() -> Self {
        Self {
            friction: 1.0,
            restitution: 0.0,
            pd_scale: 1.0,
            kd_scale: 1.0,
            mass_scale: 1.0,
            base_payload_kg: 0.0,
            contact_natural_frequency: 30.0,
            contact_damping_ratio: 5.0,
            spawn_yaw: 0.0,
            spawn_roll: 0.0,
            spawn_pitch: 0.0,
            spawn_z_offset: 0.0,
            pd_scale_per_joint: [1.0; NUM_JOINTS],
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
    pub sim_params: RbdSimParams,
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
    robot: &RobotSpec,
    dr: &DrParams,
    task_dt: f32,
    // BIPED_TERRAIN: this env's terrain trimesh (the SAME `SharedShape` Arc is
    // cloned across envs of one family so nexus dedupes the mesh buffers).
    // Appended LAST so all existing collider/link indices are unchanged.
    // None = flag off = byte-identical scene.
    terrain_shape: Option<&SharedShape>,
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
    // BIPED_FREEFALL_Z lifts the spawn clear of the ground so the robot is in
    // TRUE contact-free free-fall — the clean g/M-consistency test (pre-contact
    // generalized accel `a` must equal pure free-fall: base linear = g, all joints
    // ≈ 0). Diagnostic only.
    let ff_z: f32 = std::env::var("BIPED_FREEFALL_Z")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let root_pos = Vec3::new(0.0, 0.0, robot.spawn_z + dr.spawn_z_offset + ff_z);
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
        let o = if std::env::var("BIPED_DIAG_INERTIA").is_ok() {
            Vec3::ZERO
        } else {
            o
        };
        let inertia_mat = Mat3::from_cols(
            Vec3::new(d.x, o.x, o.y), // col 0: Ixx, Ixy, Ixz
            Vec3::new(o.x, d.y, o.z), // col 1: Ixy, Iyy, Iyz
            Vec3::new(o.y, o.z, d.z), // col 2: Ixz, Iyz, Izz
        );
        // Mass DR: scale mass and inertia together so the body stays physically
        // consistent (fixed geometry/density → I ∝ m). Applied per-env from the
        // template's sampled `dr.mass_scale`.
        let ms = dr.mass_scale;
        // Root-only additive payload (AGILE randomize_base_mass, mass only).
        let payload = if b.parent.is_none() {
            dr.base_payload_kg
        } else {
            0.0
        };
        let h = bodies.insert(
            RigidBodyBuilder::dynamic()
                .position(world[i])
                .additional_mass_properties(MassProperties::with_inertia_matrix(
                    b.com,
                    (b.mass * ms + payload).max(1e-3),
                    inertia_mat * ms,
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
            // Foot box. The MJCF sole is a cage of thin capsules; nexus allows
            // only ONE collider per body, so we approximate it with a single
            // cuboid. CRUCIAL: the cuboid spans the capsule CENTERLINES, and the
            // capsule radius is added back ONLY on the sole-thickness axis (the
            // one where the centerlines are ~coplanar) — NOT on the two footprint
            // axes. A horizontal capsule resting on the floor contacts along the
            // line directly beneath its axis, so the real support polygon is the
            // centerline rectangle; the old `±r` bounding box inflated the sole
            // footprint by ~1.5 cm per side (~+47% area), giving an
            // unrealistically large/stable support base the policy braced on
            // (stands rock-solid in nexus, falls in ~1.3 s in MuJoCo). Keeping the
            // radius on the thickness axis preserves the foot's bottom-surface
            // height, so the spawn height / penetration are unchanged.
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            let mut rmax = 0.0f32;
            for (a, c, r) in &b.capsules {
                lo = lo.min(a.min(*c));
                hi = hi.max(a.max(*c));
                rmax = rmax.max(*r);
            }
            // Add the radius back only on axes whose centerline extent is below
            // the radius (i.e. the capsules are essentially coplanar there → the
            // sole-thickness direction). Footprint axes keep centerline bounds.
            let ext = hi - lo;
            let pad = Vec3::new(
                if ext.x < rmax { rmax } else { 0.0 },
                if ext.y < rmax { rmax } else { 0.0 },
                if ext.z < rmax { rmax } else { 0.0 },
            );
            lo -= pad;
            hi += pad;
            let he = ((hi - lo) * 0.5).max(Vec3::splat(1e-3));
            let mut center = (hi + lo) * 0.5;
            // Foot collider shape. Default CAPSULE (rounded sole): nexus's flat box
            // foot caught on its sharp edges at foot-strike, so a dynamic gait
            // diverged at the ankles in MuJoCo (whose sole is 6 ROUNDED capsules) —
            // the walking sim2sim gap. A capsule rounds the heel/toe so the foot
            // ROLLS through strike/push-off like MuJoCo's. Axis = longest footprint
            // axis; radius = the foot half-width; the center is shifted on the
            // thickness axis so the sole-bottom height is unchanged. BIPED_FOOT_SHAPE=box reverts.
            // BIPED_FOOT_SHAPE=convex adds a third option: a convex hull of the
            // link's actual mesh geometry, so nexus collides with the real foot
            // shape rather than the capsule/box approximation. The hull points are
            // already in the link frame, so the collider pose is identity (unlike
            // box/capsule, which sit at the computed `center`). NOTE the rounded-
            // capsule rationale above: a hull reintroduces sharp foot-strike edges,
            // so this is opt-in for fidelity experiments, not the tuned default.
            let foot_shape =
                std::env::var("BIPED_FOOT_SHAPE").unwrap_or_else(|_| "capsule".to_string());
            let convex_cb = if foot_shape == "convex" && !b.mesh_pts.is_empty() {
                ColliderBuilder::convex_hull(&b.mesh_pts)
            } else {
                None
            };
            let (cb, cpose) = if let Some(cb) = convex_cb {
                (cb, Pose::from_parts(Vec3::ZERO, Rotation::IDENTITY))
            } else if foot_shape == "box" {
                (
                    ColliderBuilder::cuboid(he.x, he.y, he.z),
                    Pose::from_parts(center, Rotation::IDENTITY),
                )
            } else {
                let he_arr = [he.x, he.y, he.z];
                // Thickness axis = where the radius pad was added (capsules ~coplanar).
                let tax = if pad.x > 0.0 {
                    0
                } else if pad.y > 0.0 {
                    1
                } else {
                    2
                };
                let foot_axes: Vec<usize> = (0..3).filter(|&a| a != tax).collect();
                // long = larger-extent footprint axis (capsule axis); wide = the other.
                let (long_ax, wide_ax) = if he_arr[foot_axes[0]] >= he_arr[foot_axes[1]] {
                    (foot_axes[0], foot_axes[1])
                } else {
                    (foot_axes[1], foot_axes[0])
                };
                let radius = he_arr[wide_ax].max(1e-3);
                let half_height = (he_arr[long_ax] - radius).max(1e-3);
                // Preserve the sole-bottom height: capsule bottom is center−radius
                // vs the box's center−he[tax]; shift the center up by the difference.
                let shift = radius - he_arr[tax];
                match tax {
                    0 => center.x += shift,
                    1 => center.y += shift,
                    _ => center.z += shift,
                }
                let cb = match long_ax {
                    0 => ColliderBuilder::capsule_x(half_height, radius),
                    1 => ColliderBuilder::capsule_y(half_height, radius),
                    _ => ColliderBuilder::capsule_z(half_height, radius),
                };
                (cb, Pose::from_parts(center, Rotation::IDENTITY))
            };
            colliders.insert_with_parent(
                cb.position(cpose)
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
    // Physical joint limits ON by default (opt out with BIPED_JOINT_LIMITS=0).
    // Without them the policy "stands" by jamming joints to the target-clamp
    // boundary — a degenerate brace that doesn't balance and doesn't transfer to
    // MuJoCo. Real limits (the per-joint MJCF range) force genuine balance,
    // matching WBC's soft_joint_pos_limit_factor=0.9. ~1.7x iter cost.
    let joint_limits_on = std::env::var("BIPED_JOINT_LIMITS")
        .map(|v| v != "0")
        .unwrap_or(true);
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some(jname)) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let spec = robot.joints.iter().find(|j| &j.name == jname);
        let pi = std::f32::consts::PI;
        // Per-joint actuator-strength DR (asymmetry): look up this joint's action
        // index and apply its independent gain/torque multiplier on top of the
        // global `pd_scale`. `1.0` for joints not in the canonical action set.
        let pj = robot
            .joints
            .iter()
            .position(|j| j.name == jname.as_str())
            .map(|k| dr.pd_scale_per_joint[k])
            .unwrap_or(1.0);
        // Non-action joints (e.g. the G1 29-DOF body's waist/arms) are PD-held
        // at the rest pose with the spec's `held_joints` gains (first matching
        // name fragment wins), falling back to generic holding gains.
        let held = robot
            .held_joints
            .iter()
            .find(|(frag, ..)| jname.contains(frag))
            .map(|&(_, kp, kd, effort)| (kp, kd, effort, (-pi, pi), 0.0))
            .unwrap_or((50.0 * pj, 1.0 * pj, 20.0 * pj, (-pi, pi), 0.0));
        let (kp, kd, effort, pos_limit, spec_damping) = spec
            .map(|s| {
                (
                    s.kp * dr.pd_scale * pj,
                    s.kd * dr.pd_scale * dr.kd_scale * pj,
                    s.effort_limit * pj,
                    s.pos_limit,
                    s.damping,
                )
            })
            .unwrap_or(held);
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
        // Motor model. ForceBased is now applied as an EXPLICIT generalized-force
        // PD torque inside the nexus solver (gpu_mb_gravity_and_lu: gen_forces +=
        // clamp(kp·(target−q) − kd·q̇, ±effort)), exactly matching the real robot
        // and MuJoCo's position actuator. AccelerationBased uses the mass-
        // normalized soft constraint (cfm_coeff): commanded kp realizes the same
        // stiffness regardless of the joint's (tiny) link inertia — crisp, but
        // UNREALISTIC (the real actuator is force-based, so the policy overfits to
        // nexus's inertia-decoupled tracking; sim-to-sim diverges at the ankles).
        // The OLD constraint-based ForceBased (raw cfm_gain) under-realized kp and
        // sagged — that path is bypassed now. BIPED_FORCE_MOTOR=1 selects the
        // explicit force-based PD (the sim-to-real-faithful default candidate).
        // Explicit force-based PD is now the DEFAULT: it matches the real robot /
        // MuJoCo actuator (τ = kp·err − kd·q̇), and with it the standing policy
        // survives the full 6 s in MuJoCo (vs 1.7 s on AccelerationBased — the
        // inertia-decoupled tracking the policy used to overfit to). Opt back into
        // the old AccelerationBased motor with BIPED_ACCEL_MOTOR=1 for A/B.
        let motor_model = if std::env::var("BIPED_ACCEL_MOTOR").is_ok() {
            MotorModel::AccelerationBased
        } else {
            MotorModel::ForceBased
        };
        joint.set_motor_model(JointAxis::AngZ, motor_model);
        // Motor gains come straight from the robot spec (RobotSpec::joints),
        // which already bakes in the physical torque-PD correction (STIFFNESS_SCALE
        // / DAMPING_SCALE) — kp is a real torque/rad gain, FIXED, identical to what
        // the MuJoCo transfer model and the real robot use. No runtime scaling.
        // BIPED_KP_SCALE / BIPED_KD_SCALE remain only as optional diagnostics
        // (default 1.0); leave them unset for the production gains.
        let kp_scale: f32 = std::env::var("BIPED_KP_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let kd_scale: f32 = std::env::var("BIPED_KD_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
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

    // Ground (Z-up). With terrain on, the cuboid stretches to backstop the
    // whole 160 m strip (x ∈ [8, 168]); its top stays at z = 0.
    let (g_pos, g_hx) = if terrain_shape.is_some() {
        (Vec3::new(75.0, 0.0, -0.5), 100.0)
    } else {
        (Vec3::new(0.0, 0.0, -0.5), 50.0)
    };
    // With terrain on, the ground cuboid and the strip trimesh fully overlap
    // (both fixed). nexus's broad-phase doesn't filter fixed-fixed pairs, and
    // cuboid-vs-strip would emit a PFM pair per overlapping TRIANGLE (~10^6) —
    // so statics get a group that excludes each other while still colliding
    // with the robot (which keeps default ALL/ALL groups). Flag-off ground
    // keeps rapier defaults (bit-identity).
    let static_groups = InteractionGroups::new(
        Group::GROUP_2,
        Group::ALL ^ Group::GROUP_2,
        InteractionTestMode::And,
    );
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(g_pos));
    let mut gb = ColliderBuilder::cuboid(g_hx, 50.0, 0.5)
        .friction(dr.friction)
        .restitution(dr.restitution);
    if terrain_shape.is_some() {
        gb = gb.collision_groups(static_groups);
    }
    colliders.insert_with_parent(gb, ground, &mut bodies);
    // BIPED_TERRAIN: the difficulty strip, one trimesh collider at identity.
    if let Some(shape) = terrain_shape {
        let tb = bodies.insert(RigidBodyBuilder::fixed());
        colliders.insert_with_parent(
            ColliderBuilder::new(shape.clone())
                .friction(dr.friction)
                .restitution(dr.restitution)
                .collision_groups(static_groups),
            tb,
            &mut bodies,
        );
    }

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
    let mut sp = RbdSimParams::default();
    sp.dt = task_dt;
    sp.num_solver_iterations = std::env::var("BIPED_SOLVER_ITERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(SOLVER_ITERS);
    sp.contact_natural_frequency =
        env_f32("BIPED_CONTACT_NF").unwrap_or(dr.contact_natural_frequency);
    sp.contact_damping_ratio = env_f32("BIPED_CONTACT_DR").unwrap_or(dr.contact_damping_ratio);

    // Build the index table from the canonical joint ordering.
    let mut actuated: Vec<(u32, String)> = Vec::with_capacity(NUM_JOINTS);
    let mut joint_dof_offset = [0u32; NUM_JOINTS];
    for (k, name) in robot.joints.iter().map(|j| j.name).enumerate() {
        let link = *name_to_link
            .get(name)
            .unwrap_or_else(|| panic!("missing joint {name} in MJCF"));
        actuated.push((link, name.to_string()));
        // Each leg joint has 1 DOF and sits at offset (6 root DOFs + insertion order).
        // Insertion order = link - 1 (since torso is link 0).
        joint_dof_offset[k] = 6 + (link - 1);
    }

    // Sole-normal in foot-local frame at spawn (sole = world +Z, so the local
    // sole-normal is R_spawn⁻¹·Z). Feet are matched BY NAME against the spec's
    // `foot_links` so index 0/1 order is the spec's, not MJCF document order
    // (lerobot lists right-then-left, the Unitree models left-then-right).
    let mut foot_sole_local = [Vec3::Z; NUM_FEET];
    let mut foot_links = [0u32; NUM_FEET];
    for (i, want) in robot.foot_links.iter().enumerate() {
        let (mjcf_idx, h) = foot_handles
            .iter()
            .find(|(m, _)| mjcf[*m].name == *want)
            .unwrap_or_else(|| panic!("foot link {want} carries no sole capsules in the MJCF"));
        foot_links[i] = *mb_link_of_mjcf.get(mjcf_idx).unwrap_or(&0);
        foot_sole_local[i] = bodies[*h].rotation().conjugate() * Vec3::Z;
    }

    let mjcf_to_link: Vec<u32> = (0..mjcf.len())
        .map(|i| *mb_link_of_mjcf.get(&i).unwrap_or(&0))
        .collect();

    // Thigh / shin / hip links — the parts that have NO ground collider (only
    // the feet do), so they must never legitimately touch the floor. The
    // name fragments are per-robot (`RobotSpec::illegal_ground_fragments`);
    // ankle + foot links never match (they sit legitimately low next to the sole).
    let illegal_ground_links: Vec<u32> = (0..mjcf.len())
        .filter(|&i| {
            let n = &mjcf[i].name;
            robot.illegal_ground_fragments.iter().any(|f| n.contains(f))
        })
        .filter_map(|i| mb_link_of_mjcf.get(&i).copied())
        .collect();

    // Left/right link pairs for the self-collision (leg-crossing) guard —
    // per-robot (`RobotSpec::self_collision_pairs`): feet, shins, thighs.
    let link_of_name = |name: &str| -> Option<u32> {
        mjcf.iter()
            .position(|b| b.name == name)
            .and_then(|i| mb_link_of_mjcf.get(&i).copied())
    };
    let self_collision_pairs: Vec<(u32, u32)> = robot
        .self_collision_pairs
        .iter()
        .filter_map(|(r, l)| Some((link_of_name(r)?, link_of_name(l)?)))
        .collect();

    // Per-joint parent link + rest quat, used by the ws-free joint-angle
    // extraction (`q_child = q_parent · rest_quat · R_z(θ)`).
    let mut actuated_parent_links = [0u32; NUM_JOINTS];
    let mut actuated_rest_quat = [Rotation::IDENTITY; NUM_JOINTS];
    for (k, name) in robot.joints.iter().map(|j| j.name).enumerate() {
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
        links_per_batch: next_mb_link, // 1 (root) + jointed links
        // 6 root DOFs + one per hinge — counts ALL model joints, not just the
        // actuated ones (the G1 29-DOF body carries 13 extra held joints).
        dofs_per_batch: 6 + mjcf.iter().filter(|b| b.joint.is_some()).count() as u32,
        // robot bodies + ground (+ terrain trimesh when BIPED_TERRAIN=1)
        colliders_per_batch: (mjcf.len() + 1 + terrain_shape.is_some() as usize) as u32,
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

/// Debug-only per-foot stance-phase accumulator (env 0). Records the foot
/// origin's world XY + heading at the moment it became loaded, so when it lifts
/// we can report how far the planted foot's origin actually drifted (slide) and
/// how much it rotated (roll) over the whole single-support phase.
#[derive(Clone, Copy, Default)]
struct DbgStance {
    loaded: bool,
    start_x: f32,
    start_y: f32,
    start_quat: [f32; 4],
    steps: u32,
    prev_x: f32,
    prev_y: f32,
    path_len: f32, // accumulated horizontal path of the origin (total, not net)
}

/// One vectorized env over nexus GPU physics.
///
/// All N envs share a single `RbdState`. Per-env host state (RNG,
/// command, step counter, action history, air-time, sole-normals) lives in
/// parallel vectors keyed by env index. Reset uses pre-built single-env spawn
/// templates and `state.reset_env_from(env_i, template)`.
/// BIPED_TERRAIN=1 state: the three family strips + per-env curriculum
/// (WBC-AGILE's terrain_levels_vel_curriculum — see zealot_env::terrain).
struct TerrainSetup {
    strips: [TerrainStrip; 3],
    /// Per-env curriculum state (level + success/failure counters).
    curriculum: Vec<TerrainCurriculum>,
    /// Dedicated RNG stream (levels, spawn jitter) — the env's command/DR
    /// streams stay untouched, keeping flag-off runs bit-identical.
    rng: Vec<Lcg>,
    /// Chord-sum traveled distance since episode start (AGILE's metric:
    /// straight-line segments between command-resample points).
    travel: Vec<f32>,
    last_xy: Vec<[f32; 2]>,
}

impl TerrainSetup {
    fn strip_for(&self, env: usize) -> &TerrainStrip {
        &self.strips[env % 3]
    }
}

pub struct BipedNexusBatchEnv {
    // Topology + indexing
    mjcf: Vec<MjBody>,
    robot: RobotSpec,
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
    /// Actuator delay (BIPED_MOTOR_DELAY=min,max): the PD position target is
    /// delayed by a per-env lag of k physics SUBSTEPS (WBC-AGILE's
    /// DelayedPDActuator semantics — k ~ uniform int [min,max] inclusive,
    /// resampled at every episode reset; max ≤ decimation). None = off, and
    /// the staging path below is byte-identical to the no-delay build.
    motor_delay: Option<(u32, u32)>,
    /// Per-env sampled lag, in physics substeps.
    delay_k: Vec<u32>,
    /// Dedicated RNG stream for lag sampling, so enabling the delay leaves the
    /// env's command/DR stream untouched (keeps `0,0` bitwise-comparable to
    /// the no-delay build — the staging-equivalence check).
    delay_rng: Vec<Lcg>,
    /// Joint targets applied last control step (post `joint_targets` clamp).
    delay_prev_targets: Vec<[f32; NUM_JOINTS]>,
    /// Set on reset: the first post-reset command applies from substep 0
    /// regardless of `delay_k` (AGILE replicates the first command into the
    /// delay buffer — a fresh env never sees another episode's targets).
    delay_fresh: Vec<bool>,
    /// Per-step scratch: this step's targets + the packed GPU delay-state
    /// upload ([tick, k, prev targets per link] per env).
    delay_now: Vec<[f32; NUM_JOINTS]>,
    delay_state_buf: Vec<f32>,
    /// Observation history (BIPED_OBS_HISTORY=H): the ACTOR obs becomes the
    /// last H noised 45-frames stacked oldest→newest (WBC-AGILE semantics —
    /// replicated on reset). Critic stays single-frame privileged. None = off.
    obs_hist: Option<ObsHistory>,
    /// Rough-terrain difficulty curriculum (BIPED_TERRAIN=1). None = off.
    terrain: Option<TerrainSetup>,
    air_time: Vec<[f32; NUM_FEET]>,
    /// Index of the foot that most recently touched down, per env (-1 = none yet,
    /// reset on episode reset). Drives `FootObs.alt_step`: a touchdown only counts
    /// as a step if it's the OTHER foot than this, enforcing L→R→L→R alternation.
    last_td_foot: Vec<i8>,
    /// Gait-clock phase ∈ [0,1) per env, advanced by `control_dt / gait_period`
    /// each step (wraps at 1), reset to 0 on episode reset. Fed to the policy as
    /// (sin,cos) and used by the periodic gait reward to prescribe swing/stance.
    gait_phase: Vec<f32>,
    /// Seconds per full gait cycle (both feet step once). BIPED_GAIT_PERIOD.
    gait_period: f32,
    /// Global control-step counter (for push-perturbation scheduling).
    global_step: u64,
    /// Debug-only per-foot stance-phase tracker (env 0). Tracks, while a foot is
    /// continuously loaded, the net horizontal travel + rotation of its origin so
    /// we can tell a planted-but-vaulting foot (origin ~fixed) from a SLIDING one
    /// (origin drifts across the floor). Lazily init'd in debug_contact_impulses.
    dbg_stance: Vec<DbgStance>,
    /// Random torso-push magnitude, m/s (BIPED_PUSH_VEL, 0 = off) and mean
    /// interval in control steps (BIPED_PUSH_INTERVAL, default 175 ≈ 3.5 s —
    /// the midpoint of WBC-AGILE's 2–5 s). On each push every env gets an
    /// independent random horizontal velocity kick to the torso, forcing the
    /// policy to learn genuine balance recovery (sim-to-real robustness)
    /// rather than a brittle nexus-specific reflex.
    push_vel: f32,
    /// AGILE reset_base/reset_robot_joints velocity randomization
    /// (BIPED_RESET_VEL=1): every reset writes base lin ±0.25 m/s (x,y), base
    /// ang ±0.5 rad/s (r/p/y) and joint vels ±1.0 rad/s — episodes START in
    /// motion, so a statically stable stand is never the t=0 state.
    reset_vel: bool,
    push_interval: u64,
    /// Angular kick magnitude, rad/s (BIPED_PUSH_ANGVEL, default 0 = linear-only
    /// pushes). WBC-AGILE uses ±0.25 on roll/pitch/yaw.
    push_angvel: f32,
    /// Next `global_step` at which to push. Rescheduled after each push with
    /// ±50% jitter around `push_interval` so the policy can't phase-lock a
    /// recovery reflex to a fixed cadence.
    next_push_at: u64,
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
    pipeline: RbdPipeline,
    state: RbdState,

    /// CUDA-graph capture of one control step's `decimation × pipeline.step`
    /// physics sequence. The per-step host re-encode of those dispatches is
    /// ~half the physics time (~24 ms/step measured); capturing once and
    /// replaying via `cuGraphLaunch` removes it. Opt-in via `BIPED_GRAPH=1`
    /// (eager dispatch is the default). Captured lazily after warmup; replayed
    /// thereafter with the freshly-staged motor buffer (the graph records kernel
    /// launches, not data, so per-step buffer writes + resets are honoured).
    #[cfg(feature = "cuda_backend")]
    physics_graph: Option<SyncGraph>,
    /// Steps taken since construction — used to delay graph capture until the
    /// dispatch structure (color count / buffers) has stabilised.
    graph_warmup_steps: u32,

    // Pre-built spawn templates for reset_env_from (different DR samples).
    templates: Vec<RbdState>,
    /// CPU snapshot of each template, read off the GPU once at setup so resets
    /// are write-only (no per-reset `slow_read_buffer` stalls — the dominant
    /// reset cost on WebGPU). Parallel to `templates`.
    template_snapshots: Vec<RbdSnapshot>,
    template_dr: Vec<DrParams>,
    /// Cached per-template `foot_sole_local` (constant per template) so reset_env
    /// doesn't rebuild the rapier scene every reset.
    template_foot_sole: Vec<[Vec3; NUM_FEET]>,
    /// Cached per-template spawn obs / critic-obs (populated by `initial_obs`).
    /// The post-reset obs is deterministic from the template spawn state; the
    /// velocity command enters ONLY obs[12:16], so reset_env serves these cached
    /// vectors with the fresh command patched in — eliminating the per-reset
    /// `slurp_poses` full readback (the dominant reset cost). Empty until
    /// `initial_obs` runs, in which case reset_env falls back to the readback.
    template_spawn_obs: Vec<Vec<f32>>,
    template_spawn_critic_obs: Vec<Vec<f32>>,

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
pub const NUM_REWARD_COMPS: usize = 28;

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
    "power",         // Σ|τ·q̇| mechanical-power (energy / cost-of-transport) penalty
    "gait_clock",    // dense periodic swing/stance-matching reward
    "com_centering", // CoM-over-support-foot (low-ankle-torque single-support)
    "stand_planted", // per-airborne-foot penalty at standing command (balance, don't step)
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
    /// Build N envs sharing one batched RbdState. `num_templates` controls
    /// how many distinct DR samples are pre-built and cycled across the N envs
    /// at construction and reset time (higher = better coverage, more GPU mem).
    pub async fn new(mjcf_xml: &str, num_envs: usize, num_templates: usize, seed: u64) -> Self {
        let mut mjcf = parse_mjcf(mjcf_xml);
        // Convex-hull foot collider path: load the link meshes once so the scene
        // builder can hull them (BIPED_FOOT_SHAPE=convex). Default capsule path
        // skips this entirely.
        if std::env::var("BIPED_FOOT_SHAPE").as_deref() == Ok("convex") {
            load_mesh_hulls(&mut mjcf, mjcf_xml);
        }
        let robot = RobotSpec::from_env();
        let mut task = VelocityFlatTask::for_robot(robot);
        // BIPED_DECIMATION: shift physics work between narrow-phase refreshes
        // (decimation) and solver substeps while KEEPING control_dt=0.02 fixed
        // (sim_dt = 0.02/decimation = the contact-staleness window). Used with
        // BIPED_SOLVER_ITERS to hold total substeps + substep dt' constant and
        // vary ONLY how often the contact manifold is refreshed — the
        // deconfounding test for the "stale multibody contact across substeps"
        // hypothesis. Diagnostic only.
        if let Some(d) = std::env::var("BIPED_DECIMATION")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            task.decimation = d;
            task.sim_dt = 0.02 / d as f32;
        }
        // Gait-cadence knobs. The left/right alternation itself comes from
        // the gait clock (foot 1 runs half a cycle behind foot 0); its
        // PERIOD is BIPED_GAIT_PERIOD (read below — larger = slower,
        // lower-frequency weight transfer). These two shape how hard the
        // policy locks to that clock and the swing/stance split:
        if let Some(w) = std::env::var("BIPED_GAIT_CLOCK_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.gait_clock = w;
        }
        if let Some(sr) = std::env::var("BIPED_GAIT_SWING_RATIO")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.gait_swing_ratio = sr;
        }
        // CoM-over-support-foot reward weight. Long single-support holds
        // (slow gait clocks) are only cheap for the fragile ankles if the
        // CoM rides over the stance foot — raise this together with
        // BIPED_GAIT_PERIOD.
        if let Some(w) = std::env::var("BIPED_COM_CENTERING_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.com_centering = w;
        }
        // Balance-don't-step at stand: per-airborne-foot penalty while the
        // command is standing (NEGATIVE, e.g. -1.0; 0 = off). Pair with a
        // raised BIPED_STAND_PROB so the policy actually trains the quiet
        // stance, and with pushes on so it learns the ankle/hip strategy.
        if let Some(w) = std::env::var("BIPED_STAND_PLANTED_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.stand_planted = w;
        }

        let gpu = make_backend().await;
        let mut pipeline = RbdPipeline::new(&gpu).unwrap();
        // BIPED_CONTACT_REDUCE=1: merge per-triangle terrain contacts to ≤4
        // points per collider pair before the solvers (training-grade
        // approximation; flat-ground contacts unaffected). Biggest terrain
        // perf lever — the mb contact-constraint kernels scale with points.
        if std::env::var("BIPED_CONTACT_REDUCE").as_deref() == Ok("1") {
            pipeline.contact_reduction = true;
            println!("contact reduction ENABLED (per-pair manifolds merged to ≤4 points)");
        }

        // Sample DR for the templates first (each defines one rapier scene).
        let mut tpl_rng = Lcg::new(seed);
        let mut template_dr: Vec<DrParams> = (0..num_templates)
            .map(|_| sample_dr(&mut tpl_rng))
            .collect();
        // Always include one DR-OFF template at index 0 — keeps deterministic
        // replay possible and provides a stable initialiser. BIPED_FRICTION still
        // pins its contact μ (the render uses this template, so the knob must reach
        // it — otherwise friction A/B on the rendered env is a no-op).
        template_dr[0] = DrParams::default();
        if let Some(f) = std::env::var("BIPED_FRICTION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            template_dr[0].friction = f;
        }

        // BIPED_TERRAIN=1: generate the three family strips once and wrap each
        // in ONE SharedShape (cloned across that family's envs so nexus dedupes
        // the mesh buffers to 3 uploads). ORIENTED pseudo-normals are required
        // by the nexus trimesh contact path; the strips are closed slabs.
        let terrain_on = std::env::var("BIPED_TERRAIN").as_deref() == Ok("1");
        let terrain_build = if terrain_on {
            let t0 = Instant::now();
            let strips = [
                TerrainStrip::generate(TerrainFamily::Boxes, seed),
                TerrainStrip::generate(TerrainFamily::Rough, seed),
                TerrainStrip::generate(TerrainFamily::Wave, seed),
            ];
            let mk_shape = |verts: Vec<[f32; 3]>, tris: Vec<[u32; 3]>| -> SharedShape {
                let pts: Vec<_> = verts
                    .into_iter()
                    .map(|v| Vec3::new(v[0], v[1], v[2]))
                    .collect();
                SharedShape::trimesh_with_flags(
                    pts,
                    tris,
                    TriMeshFlags::ORIENTED | TriMeshFlags::FIX_INTERNAL_EDGES,
                )
                .expect("terrain trimesh build")
            };
            let shapes: Vec<SharedShape> = strips
                .iter()
                .map(|s| {
                    let (v, t) = s.mesh();
                    mk_shape(v, t)
                })
                .collect();
            let (sv, st) = TerrainStrip::flat_stub_mesh();
            let stub = mk_shape(sv, st);
            println!(
                "terrain curriculum ENABLED: 3 family strips ({} rows x {} m patches), built in {:.1}s",
                zealot_env::terrain::ROWS,
                zealot_env::terrain::PATCH,
                t0.elapsed().as_secs_f64()
            );
            Some((strips, shapes, stub))
        } else {
            None
        };

        // Build the per-env scenes — cycle across the templates so envs get
        // mixed DR from the start. We keep the LinkIndices from the first one
        // (topology is invariant).
        let mut idx_out: Option<LinkIndices> = None;
        let mut env_scenes: Vec<EnvScene> = Vec::with_capacity(num_envs);
        for e in 0..num_envs {
            let dr = template_dr[e % num_templates];
            let tshape = terrain_build.as_ref().map(|(_, shapes, _)| &shapes[e % 3]);
            let (scene, ix) = build_env_scene(&mjcf, &robot, &dr, task.sim_dt, tshape);
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
        let mut state = RbdState::from_rapier(
            &gpu,
            &envs_refs,
            nexus3d::rbd::pipeline::RbdCapacities {
                batches: envs_refs.len() as u32,
                body_capacity: (envs_refs.len() as u32 * 32).max(1024),
                // Per-batch contact/constraint slots; the Grow policy lazy-
                // resizes from the previous frame's counts, so start small
                // (the default 4096/batch OOMs at 4096 envs).
                collisions_capacity: 64,
                ..Default::default()
            },
        );
        state.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);
        // BIPED_CONTACT_CAP: eagerly pre-size the contact/constraint buffers
        // (per batch). Required before BIPED_GRAPH capture on terrain — the
        // lazy in-step resize can't run once a CUDA graph is captured, and
        // overflowing pairs are silently dropped (feet sink into the mesh).
        if let Some(cap) = std::env::var("BIPED_CONTACT_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            state.reserve_contacts(&gpu, cap);
            println!("contact buffers pre-sized to {cap}/batch");
        }
        // BIPED_MAX_COLORS: bound for the contact-graph coloring (nexus default
        // 8 → the solver runs max_colors+1 passes per phase whether or not the
        // colors are used). The biped scene's rigid-rigid contact graph is tiny
        // (dynamics live in the multibody solver), so the default mostly buys
        // empty solver dispatches. Under-provisioning is self-healing but bad:
        // the coloring-failed ratchet adds +5. Keep constant per run (graph
        // capture records the pass count).
        if let Some(mc) = std::env::var("BIPED_MAX_COLORS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            state.set_max_colors(mc);
            println!("contact-coloring max_colors = {mc} (default 8)");
        }
        // Implicit-coriolis OFF by default (BIPED_IMPLICIT_CORIOLIS=1 restores
        // the old nexus default). Two reasons:
        // 1. FIDELITY: implicit coriolis augments the mass matrix with `dt·C` —
        //    at fewer substeps that over-damps, at more it under-damps, so
        //    passive feet creep ∝ num_solver_iterations (the sim-to-real
        //    foot-slip bug). MuJoCo (whose RECOMMENDED `implicitfast`
        //    integrator deliberately skips the Coriolis derivatives), Genesis,
        //    PhysX and Bullet all treat Coriolis explicitly with one dynamics
        //    linearization per step; rapier's per-substep rebuild is the outlier.
        // 2. SPEED: with it on, nexus rebuilds M/LU/accelerations EVERY TGS
        //    substep (8×/step — compute_dynamics_pre + gravity_and_lu were 51%
        //    of ALL GPU time); off = once per step, measured 1.9 s → 1.0 s per
        //    training iteration @2048 envs.
        // NOTE: this changes the physics slightly — train and eval with the
        // same setting.
        let implicit_coriolis = std::env::var("BIPED_IMPLICIT_CORIOLIS")
            .map(|v| v != "0")
            .unwrap_or(false);
        state
            .multibodies_mut()
            .set_implicit_coriolis(implicit_coriolis);

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
            let fl_flat: Vec<f32> = (0..num_envs)
                .flat_map(|_| fl_per_dof.iter().copied())
                .collect();
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
            // Every joint DOF gets a floor armature of 0.01 (the official G1
            // models set `armature="0.01"` on ALL joints): PD-held non-action
            // joints (e.g. the G1 29-DOF body's arms) with ZERO armature go
            // numerically unstable once implicit-coriolis no longer refreshes
            // the mass matrix每 substep (passive stand → NaN in <20 steps).
            let mut arm_per_dof = vec![0.0f32; dpb];
            for a in arm_per_dof.iter_mut().skip(6) {
                *a = 0.01 * arm_scale;
            }
            for k in 0..NUM_JOINTS {
                let dof = idx.joint_dof_offset[k] as usize;
                if let Some(s) = robot.joints.iter().find(|j| j.name == idx.actuated[k].1) {
                    if dof < dpb {
                        arm_per_dof[dof] = s.armature * arm_scale;
                    }
                }
            }
            let arm_flat: Vec<f32> = (0..num_envs)
                .flat_map(|_| arm_per_dof.iter().copied())
                .collect();
            state.multibodies_mut().set_dof_armature(&gpu, &arm_flat);
        }

        // Spawn templates: one single-env GPU state per DR sample. Also CACHE
        // each template's `foot_sole_local` — it depends only on the (fixed) DR
        // sample, so it's constant per template. reset_env looks it up instead of
        // rebuilding the whole rapier scene every reset (build_env_scene is heavy:
        // bodies + colliders + joints + inertia eigendecomps — and reset_env runs
        // thousands of times per training iteration, once per fallen env).
        let mut templates: Vec<RbdState> = Vec::with_capacity(num_templates);
        let mut template_foot_sole: Vec<[Vec3; NUM_FEET]> = Vec::with_capacity(num_templates);
        for dr in &template_dr {
            // Templates carry a tiny far-below flat stub in the terrain slot:
            // collider count/order parity with the main batch (strides match),
            // zero mesh memory per template, and resets never copy geometry.
            let tstub = terrain_build.as_ref().map(|(_, _, stub)| stub);
            let (scene, ix) = build_env_scene(&mjcf, &robot, dr, task.sim_dt, tstub);
            template_foot_sole.push(ix.foot_sole_local);
            let envs_refs = vec![(
                &scene.bodies,
                &scene.colliders,
                &scene.impulse,
                &scene.multibody,
                &scene.sim_params,
            )];
            let mut tpl = RbdState::from_rapier(
            &gpu,
            &envs_refs,
            nexus3d::rbd::pipeline::RbdCapacities {
                batches: envs_refs.len() as u32,
                body_capacity: (envs_refs.len() as u32 * 32).max(1024),
                // Per-batch contact/constraint slots; the Grow policy lazy-
                // resizes from the previous frame's counts, so start small
                // (the default 4096/batch OOMs at 4096 envs).
                collisions_capacity: 64,
                ..Default::default()
            },
        );
            tpl.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);
            templates.push(tpl);
        }

        // Snapshot each template off the GPU ONCE so per-env resets are
        // write-only. reset_env runs thousands of times per iteration (once per
        // fallen env); the old reset_env_from re-read the constant template from
        // the GPU 6× per reset, and each slow_read_buffer stalls the WebGPU queue
        // (tens of seconds/iter on Metal). Reading once here makes resets cheap.
        let mut template_snapshots: Vec<RbdSnapshot> = Vec::with_capacity(num_templates);
        for tpl in &templates {
            template_snapshots.push(tpl.snapshot(&gpu).await);
        }

        // Per-env initial sole-normal: every env starts from the corresponding
        // template, so its foot_sole_local matches that template's. Look up the
        // cached per-template value (no rebuild).
        let foot_sole_local: Vec<[Vec3; NUM_FEET]> = (0..num_envs)
            .map(|e| template_foot_sole[e % num_templates])
            .collect();

        let cmd = vec![VelocityCommand::default(); num_envs];
        let step_count = vec![0u32; num_envs];
        let resample_at = vec![0u32; num_envs];
        let last_action = vec![[0.0f32; NUM_JOINTS]; num_envs];
        let prev_action = vec![[0.0f32; NUM_JOINTS]; num_envs];
        // BIPED_MOTOR_DELAY=min,max (or just max → min=0), in physics
        // substeps. `0,0` is a valid ENABLED config (constant zero delay —
        // used by the staging-equivalence check); unset/unparseable = off.
        let motor_delay: Option<(u32, u32)> = std::env::var("BIPED_MOTOR_DELAY").ok().and_then(
            |s| {
                let p: Vec<u32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
                match p.as_slice() {
                    [max] => Some((0, *max)),
                    [min, max] => Some((*min, *max)),
                    _ => None,
                }
            },
        );
        if let Some((min, max)) = motor_delay {
            assert!(
                min <= max && max <= task.decimation,
                "BIPED_MOTOR_DELAY: need min <= max <= decimation ({})",
                task.decimation
            );
            // GPU-side delay (gravity_and_lu selects prev-vs-current target by
            // a per-step tick): no mid-decimation host writes, so this is
            // CUDA-graph-compatible (the per-step delay-state upload sits next
            // to the motor flush, outside the captured region).
            println!(
                "actuator delay ENABLED: {min}..={max} physics substeps, resampled per env at reset"
            );
        }
        let air_time = vec![[0.0f32; NUM_FEET]; num_envs];
        let last_td_foot = vec![-1i8; num_envs];
        let gait_phase = vec![0.0f32; num_envs];
        let gait_period = std::env::var("BIPED_GAIT_PERIOD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        let reset_vel = std::env::var("BIPED_RESET_VEL").is_ok_and(|v| v == "1");
        if reset_vel {
            println!("reset-velocity randomization ENABLED (AGILE reset_base/joints: lin ±0.25, ang ±0.5, joints ±1.0)");
        }
        let push_vel = std::env::var("BIPED_PUSH_VEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let push_interval = std::env::var("BIPED_PUSH_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(175);
        let push_angvel = std::env::var("BIPED_PUSH_ANGVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let prev_joint_pos = vec![[0.0f32; NUM_JOINTS]; num_envs];
        let has_prev_joint_pos = vec![false; num_envs];
        // One pose entry per collider per env (matches `body_poses` layout).
        let prev_body_poses =
            vec![NexusPose::default(); num_envs * idx.colliders_per_batch as usize];
        let has_prev_pose = vec![false; num_envs];
        let rng: Vec<Lcg> = (0..num_envs)
            .map(|e| Lcg::new(seed ^ ((e as u64).wrapping_mul(2654435761))))
            .collect();
        // Dedicated delay RNG stream (never touches the command/DR stream).
        let mut delay_rng: Vec<Lcg> = (0..num_envs)
            .map(|e| Lcg::new(seed ^ ((e as u64).wrapping_mul(2654435761)) ^ 0xD31A7))
            .collect();
        let delay_k: Vec<u32> = if let Some((min, max)) = motor_delay {
            (0..num_envs)
                .map(|e| {
                    let r = delay_rng[e].range(0.0, 1.0);
                    min + ((r * (max - min + 1) as f32) as u32).min(max - min)
                })
                .collect()
        } else {
            vec![0; num_envs]
        };
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
            motor_delay,
            delay_k,
            delay_rng,
            delay_prev_targets: vec![[0.0f32; NUM_JOINTS]; num_envs],
            delay_fresh: vec![true; num_envs],
            delay_now: vec![[0.0f32; NUM_JOINTS]; num_envs],
            delay_state_buf: Vec::new(),
            obs_hist: ObsHistory::from_env(num_envs, OBS_DIM),
            terrain: terrain_build.map(|(strips, _shapes, _stub)| {
                let mut rng: Vec<Lcg> = (0..num_envs)
                    .map(|e| {
                        Lcg::new(
                            (seed ^ 0x7E22_A100)
                                .wrapping_add((e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                        )
                    })
                    .collect();
                let curriculum = rng.iter_mut().map(TerrainCurriculum::init).collect();
                TerrainSetup {
                    strips,
                    curriculum,
                    rng,
                    travel: vec![0.0; num_envs],
                    last_xy: vec![[0.0, 0.0]; num_envs],
                }
            }),
            air_time,
            last_td_foot,
            gait_phase,
            gait_period,
            global_step: 0,
            dbg_stance: Vec::new(),
            push_vel,
            reset_vel,
            push_interval,
            push_angvel,
            next_push_at: push_interval,
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
            template_snapshots,
            template_dr,
            template_foot_sole,
            template_spawn_obs: Vec::new(),
            template_spawn_critic_obs: Vec::new(),
            tick_since_resize: 0,
            #[cfg(feature = "cuda_backend")]
            physics_graph: None,
            graph_warmup_steps: 0,
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
        // BIPED_TERRAIN: teleport every env onto its initial-level patch (the
        // as-built state stands on flat ground at the origin). Uses the same
        // template each env was built from, so its DR sample is preserved.
        // Training is on-terrain from step 0, like AGILE.
        if env.terrain.is_some() {
            for e in 0..num_envs {
                let t = e % env.templates.len().max(1);
                let off = env.terrain_spawn_offset(e);
                env.state.reset_env_from_snapshot_offset(
                    &env.gpu,
                    e as u32,
                    &env.template_snapshots[t],
                    off,
                );
                if env.motor_delay.is_some() {
                    env.delay_fresh[e] = true;
                }
            }
        }
        env
    }

    /// BIPED_TERRAIN: pick env `e`'s spawn offset — its current level's patch
    /// center plus AGILE's ±2.5 m jitter, lifted to clear the local terrain —
    /// and reset its travel bookkeeping to the new spawn.
    fn terrain_spawn_offset(&mut self, e: usize) -> Vec3 {
        let ter = self.terrain.as_mut().expect("terrain on");
        let level = ter.curriculum[e].level;
        let (cx, cy) = TerrainStrip::patch_center(level);
        let rng = &mut ter.rng[e];
        let (sx, sy) = (cx + rng.range(-2.5, 2.5), cy + rng.range(-2.5, 2.5));
        // Clearance over a foot-sized neighborhood + a small epsilon: spawn
        // height is relative to flat ground (the template pose), so the offset
        // z lifts the whole robot by the local surface height.
        let hz = ter.strip_for(e).height_max_in(sx, sy, 0.3) + 0.02;
        ter.travel[e] = 0.0;
        ter.last_xy[e] = [sx, sy];
        Vec3::new(sx, sy, hz)
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
        OBS_DIM * self.obs_hist.as_ref().map_or(1, |h| h.h())
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

    /// DEBUG: read env 0's post-solve contact constraints and print, per contact,
    /// the normal impulse N, the friction impulse F, and the clamp μ·N — to settle
    /// whether the sliding foot is loaded (large N) with friction below the clamp
    /// (F < μ·N → solver issue) or unloaded (N≈0 → no contact / hovering).
    pub async fn debug_contact_impulses(&mut self) {
        use nexus3d::rbd::shaders::dynamics::MultibodyContactConstraint;
        let total = self
            .state
            .multibodies_mut()
            .contact_constraints()
            .buffer()
            .len();
        let cnt_total = self
            .state
            .multibodies_mut()
            .contact_constraint_count()
            .buffer()
            .len();
        let mut cons: Vec<MultibodyContactConstraint> =
            vec![MultibodyContactConstraint::default(); total];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().contact_constraints().buffer(),
                &mut cons,
            )
            .await
            .expect("contact_constraints readback");
        let mut cnt = vec![0u32; cnt_total];
        self.gpu
            .slow_read_buffer(
                self.state
                    .multibodies_mut()
                    .contact_constraint_count()
                    .buffer(),
                &mut cnt,
            )
            .await
            .expect("contact count readback");
        let stride = total / cnt_total.max(1); // MAX constraints per mb
        let n0 = (cnt[0] as usize).min(stride);
        // Generalized velocities (env 0) + per-constraint jacobian rows, so we can
        // recompute J·v = Σ_i jac[i]·dof_state[i] on the host — the contact-point
        // tangential velocity the solver ACTUALLY perceives — and compare it to the
        // foot's real world slip. For env 0 (batch 0, mb 0): v_base=0, the jac
        // column for slot s starts at s*dpb, ndofs = dpb.
        let dpb = self.state.multibodies_mut().dofs_per_batch_count() as usize;
        let dtot = self.state.multibodies_mut().dof_state().buffer().len();
        let mut dof = vec![0.0f32; dtot];
        self.gpu
            .slow_read_buffer(self.state.multibodies_mut().dof_state().buffer(), &mut dof)
            .await
            .expect("dof_state readback");
        let jtot = self
            .state
            .multibodies_mut()
            .contact_constraint_jacs()
            .buffer()
            .len();
        let mut jacs = vec![0.0f32; jtot];
        self.gpu
            .slow_read_buffer(
                self.state
                    .multibodies_mut()
                    .contact_constraint_jacs()
                    .buffer(),
                &mut jacs,
            )
            .await
            .expect("contact_constraint_jacs readback");
        let host_jv = |s: usize| -> f32 {
            let base = s * dpb;
            let mut v = 0.0f32;
            for i in 0..dpb {
                if base + i < jacs.len() {
                    v += jacs[base + i] * dof[i]; // env0 dof_state = dof[0..dpb]
                }
            }
            v
        };
        // |J| of a constraint's jac row — if a LOADED foot's tangent |J|≈0 the
        // jacobian is degenerate (can't apply friction in that direction); if |J|
        // is healthy then Jv_host≈0 means the velocity is genuinely zeroed.
        let jnorm = |s: usize| -> f32 {
            let base = s * dpb;
            let mut v = 0.0f32;
            for i in 0..dpb {
                if base + i < jacs.len() {
                    v += jacs[base + i] * jacs[base + i];
                }
            }
            v.sqrt()
        };
        let mut out = String::from("[contact] env0:");
        for s in 0..n0 {
            let c = cons[s];
            if c.kind == 1 {
                // Normal: print impulse + the bias rhs. rhs_wo_bias = dist·inv_dt
                // (speculative, dist>0) — the raw-inv_dt term suspected of the
                // ∝inv_dt energy injection; rhs includes the (saturating) erp bias.
                // _pad0 = with-bias-solve J·v, _pad1 = no-bias-stabilization J·v
                // (both pre-impulse). Comparing across iters localizes the energy
                // injection (with-bias = integrate adds too much; no-bias = removal
                // leaves a growing residual).
                let wbias_jv = f32::from_bits(c._pad0);
                let nobias_jv = f32::from_bits(c._pad1);
                out.push_str(&format!(
                    " N[link{} s{}]={:.3}(wJv={:+.4} nJv={:+.4} il={:.2})",
                    c.link_id, s, c.impulse, wbias_jv, nobias_jv, c.inv_lhs
                ));
            } else if c.kind == 2 {
                let n = cons[c.normal_constraint_slot as usize];
                // Jv_host = jac·dof_state (the foot's tangential velocity the solver
                // sees). If Jv_host≈0 while the foot world-slips ~2 m/s, the contact
                // jacobian is blind to the real motion (the bug). rhs is the target
                // (0 = stick). F is the applied impulse vs clamp μN.
                out.push_str(&format!(
                    " F[s{}]={:.3} Jv_host={:+.3} |J|={:.2} (clampμN={:.2})",
                    s,
                    c.impulse,
                    host_jv(s),
                    jnorm(s),
                    c.friction_coeff * n.impulse
                ));
            }
        }
        // Per-foot STANCE-PHASE kinematics from the link's WORLD POSE — no velocity
        // convention, no assumed moment arm. While a foot is continuously loaded we
        // track how far its origin (a) NET-drifts from where it touched down and
        // (b) wanders in total path length, plus how much it rotates. A foot that
        // is "planted" while the body vaults over it has NET drift ≈ 0; a foot that
        // SLIDES has net drift growing across the phase. We also test the
        // rolling-without-slip relation directly: v_origin ≈ ω·R ⇒ rolling;
        // v_origin ≫ ω·R ⇒ translating (sliding). Requires per-step calls.
        let wtotal = self
            .state
            .multibodies_mut()
            .links_workspace()
            .buffer()
            .len();
        let mut ws: Vec<MultibodyLinkWorkspace> = vec![unsafe { std::mem::zeroed() }; wtotal];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().links_workspace().buffer(),
                &mut ws,
            )
            .await
            .expect("links_workspace readback");
        if self.dbg_stance.len() < self.idx.foot_links.len() {
            self.dbg_stance = vec![DbgStance::default(); self.idx.foot_links.len()];
        }
        let cdt = self.task.control_dt();
        for (fi, &fl) in self.idx.foot_links.iter().enumerate() {
            // Loaded? Sum normal impulses on this link.
            let mut n_imp = 0.0f32;
            for s in 0..n0 {
                let c = cons[s];
                if c.kind == 1 && c.link_id == fl {
                    n_imp += c.impulse;
                }
            }
            let w = ws[fl as usize]; // env 0
            let p = w.local_to_world.translation;
            let q = w.local_to_world.rotation; // glam::Quat
            let loaded = n_imp > 0.05; // well-loaded (≈ ½ body weight), not a graze
            let st = &mut self.dbg_stance[fi];
            if loaded && !st.loaded {
                // Touchdown: start a fresh stance phase.
                *st = DbgStance {
                    loaded: true,
                    start_x: p.x,
                    start_y: p.y,
                    start_quat: [q.x, q.y, q.z, q.w],
                    steps: 1,
                    prev_x: p.x,
                    prev_y: p.y,
                    path_len: 0.0,
                };
            } else if loaded && st.loaded {
                let dx = p.x - st.prev_x;
                let dy = p.y - st.prev_y;
                st.path_len += (dx * dx + dy * dy).sqrt();
                st.prev_x = p.x;
                st.prev_y = p.y;
                st.steps += 1;
            } else if !loaded && st.loaded {
                // Lift-off: report the whole stance phase.
                let net = ((p.x - st.start_x).powi(2) + (p.y - st.start_y).powi(2)).sqrt();
                let q0 = nexus3d::rbd::math::Quat::from_xyzw(
                    st.start_quat[0],
                    st.start_quat[1],
                    st.start_quat[2],
                    st.start_quat[3],
                );
                let rot_deg = q0.angle_between(q).to_degrees();
                let dur = st.steps as f32 * cdt;
                eprintln!(
                    "[stance] foot{fl}: dur={dur:.2}s  net_drift={:.1}cm  path={:.1}cm  rot={rot_deg:.0}deg  (drift_rate={:.2} m/s)",
                    net * 100.0,
                    st.path_len * 100.0,
                    if dur > 1e-3 { net / dur } else { 0.0 },
                );
                st.loaded = false;
            }
            let tag = if loaded { "STANCE" } else { "swing " };
            out.push_str(&format!("  foot{fl}[{tag} N={n_imp:.2}]"));
        }
        // Generalized velocities (env 0): is the foot moved by the BASE translating
        // (root linear DOFs large → planted foot dragged) or by the LEG joints
        // spinning (joint q̇ large → motors actively swinging the stance foot)?
        // Layout: [root lin x,y,z, root ang x,y,z, joint q̇ ×NUM_JOINTS].
        // (`dpb` / `dof` were read at the top of this function.)
        let b = 0; // env 0
        let root_lin = (dof[b * dpb].powi(2) + dof[b * dpb + 1].powi(2)).sqrt();
        let root_ang =
            (dof[b * dpb + 3].powi(2) + dof[b * dpb + 4].powi(2) + dof[b * dpb + 5].powi(2)).sqrt();
        // Max |q̇| over the joint DOFs (everything past the 6 root DOFs).
        let mut qd_max = 0.0f32;
        for d in 6..dpb {
            qd_max = qd_max.max(dof[b * dpb + d].abs());
        }
        out.push_str(&format!(
            "  | root_lin={root_lin:.2}m/s root_ang={root_ang:.2} max|q̇|={qd_max:.2}rad/s"
        ));
        // Generalized acceleration `a = M⁻¹τ` (gravity bias, pre-contact). For a
        // PASSIVE standing robot under vertical gravity the HORIZONTAL base accel
        // (DOF 0,1) and base angular accel should be ~0; a spurious value here is
        // the task #27 g/M inconsistency that drives the foot creep.
        let atot = self
            .state
            .multibodies_mut()
            .gen_accelerations()
            .buffer()
            .len();
        let mut acc = vec![0.0f32; atot];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().gen_accelerations().buffer(),
                &mut acc,
            )
            .await
            .expect("gen_accelerations readback");
        if dpb <= atot {
            out.push_str(&format!(
                "  | a_base=[x={:+.2} y={:+.2} z={:+.2} | ωx={:+.2} ωy={:+.2} ωz={:+.2}] a_joints=[",
                acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]
            ));
            for d in 6..dpb {
                out.push_str(&format!("{:+.1} ", acc[d]));
            }
            out.push(']');
        }
        eprintln!("{out}");
    }

    /// PHASE-A substep trace: read env0's foot-link world XY + per-foot normal
    /// impulse and emit one `[sub]` line. Called per `pipeline.step` inside the
    /// (non-graph) decimation loop when `BIPED_SUBSTEP_TRACE` is set. With
    /// `BIPED_SOLVER_ITERS=1` each pipeline.step is ONE substep, so this gives
    /// per-substep resolution of the foot contact-point trajectory — to isolate
    /// the exact substep a loaded foot flips from planted to sliding. Reuses the
    /// `debug_contact_impulses` readback pattern (links_workspace + contacts).
    pub async fn trace_foot_substep(&mut self, gstep: u64, sub: u32) {
        use nexus3d::rbd::shaders::dynamics::MultibodyContactConstraint;
        let total = self
            .state
            .multibodies_mut()
            .contact_constraints()
            .buffer()
            .len();
        let cnt_total = self
            .state
            .multibodies_mut()
            .contact_constraint_count()
            .buffer()
            .len();
        let mut cons: Vec<MultibodyContactConstraint> =
            vec![MultibodyContactConstraint::default(); total];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().contact_constraints().buffer(),
                &mut cons,
            )
            .await
            .expect("contact_constraints readback");
        let mut cnt = vec![0u32; cnt_total];
        self.gpu
            .slow_read_buffer(
                self.state
                    .multibodies_mut()
                    .contact_constraint_count()
                    .buffer(),
                &mut cnt,
            )
            .await
            .expect("contact count readback");
        let stride = total / cnt_total.max(1);
        let n0 = (cnt[0] as usize).min(stride);
        let wtotal = self
            .state
            .multibodies_mut()
            .links_workspace()
            .buffer()
            .len();
        let mut ws: Vec<MultibodyLinkWorkspace> = vec![unsafe { std::mem::zeroed() }; wtotal];
        self.gpu
            .slow_read_buffer(
                self.state.multibodies_mut().links_workspace().buffer(),
                &mut ws,
            )
            .await
            .expect("links_workspace readback");
        let mut out = format!("[sub] g={gstep} s={sub}");
        for &fl in &self.idx.foot_links {
            let mut n_imp = 0.0f32;
            // Perceived tangent SPEED magnitude over BOTH orthonormal tangents of
            // the foot's contacts: wbias = entering the with-bias solve (_pad0, post
            // integrate_velocities+motor), nobias = entering stabilization (_pad1).
            // Compared to the foot's actual world velocity, this tells us if the
            // contact jacobian sees the real slip or is blind to it.
            let mut wb2 = 0.0f32;
            let mut nb2 = 0.0f32;
            // Geometric contact-point world XY of the MOST LOADED normal point
            // (the active load-bearing point), + how many points share load — so
            // we can see the single load-bearing point DANCE between candidate
            // points across substeps (the ratchet-forward hypothesis).
            let mut cpx = 0.0f32;
            let mut cpy = 0.0f32;
            let mut max_imp = 0.0f32;
            let mut n_loaded = 0u32;
            for c in cons.iter().take(n0) {
                if c.link_id == fl {
                    if c.kind == 1 {
                        n_imp += c.impulse;
                        if c.impulse > 0.02 {
                            n_loaded += 1;
                        }
                        if c.impulse > max_imp {
                            max_imp = c.impulse;
                            cpx = f32::from_bits(c._pad4[0]);
                            cpy = f32::from_bits(c._pad4[1]);
                        }
                    } else if c.kind == 2 {
                        let w = f32::from_bits(c._pad0);
                        let n = f32::from_bits(c._pad1);
                        wb2 += w * w;
                        nb2 += n * n;
                    }
                }
            }
            let wref = &ws[fl as usize];
            let p = wref.local_to_world.translation;
            // Independent contact-point horizontal velocity from the foot's rigid-body
            // velocity: v_contact = v_lin + ω × r, r = (0,0,-SOLE_DZ). If this ≈ tJv_wb
            // → jacobian is correct and the foot is PIVOTING (contact ~stationary, not a
            // slip bug). If v_contact ≫ tJv_wb → the jacobian is BLIND to the real slip.
            const SOLE_DZ: f32 = 0.04;
            let v = wref.rb_vels.linear;
            let a = wref.rb_vels.angular;
            let cx = v.x - a.y * SOLE_DZ;
            let cy = v.y + a.x * SOLE_DZ;
            let v_contact = (cx * cx + cy * cy).sqrt();
            out.push_str(&format!(
                " foot{fl}: ox={:+.5} oy={:+.5} z={:.4} N={n_imp:.3} nL={n_loaded} cp=({cpx:+.5},{cpy:+.5}) tJv={:.4}",
                p.x, p.y, p.z, wb2.sqrt()
            ));
            let _ = (v_contact, a);
        }
        eprintln!("{out}");
    }

    /// Inject a random velocity kick to every env's torso — a push perturbation,
    /// the GPU equivalent of Isaac's `push_by_setting_velocity`: ±push_vel m/s on
    /// the root's linear x/y DOFs and (when BIPED_PUSH_ANGVEL > 0) ±push_angvel
    /// rad/s on its angular x/y/z DOFs. The policy must re-establish balance over
    /// its feet after each shove, which is what makes the learned equilibrium
    /// ROBUST and engine-agnostic (sim-to-real) rather than a brittle
    /// nexus-specific reflex. Read-modify-write the generalized-velocity section of
    /// `dof_state` (env-major, `dofs_per_batch` DOFs per env; root linear = 0..3,
    /// root angular = 3..6, world frame — rapier free-joint DOF order).
    async fn apply_random_pushes(&mut self) {
        let dpb = self.state.multibodies_mut().dofs_per_batch_count() as usize;
        let n = self.n;
        let total = self.state.multibodies_mut().dof_state().buffer().len();
        let mut buf = vec![0.0f32; total];
        self.gpu
            .slow_read_buffer(self.state.multibodies_mut().dof_state().buffer(), &mut buf)
            .await
            .expect("dof_state readback for push");
        let pv = self.push_vel;
        let pa = self.push_angvel;
        for e in 0..n {
            let dvx = self.rng[e].range(-pv, pv);
            let dvy = self.rng[e].range(-pv, pv);
            buf[e * dpb] += dvx; // root linear x velocity
            buf[e * dpb + 1] += dvy; // root linear y velocity
            if pa > 0.0 {
                for d in 3..6 {
                    // root angular x/y/z velocity (world frame)
                    buf[e * dpb + d] += self.rng[e].range(-pa, pa);
                }
            }
        }
        let vel_len = dpb * n; // velocity section only (don't touch the damping section)
        self.gpu
            .write_buffer(
                self.state.multibodies_mut().dof_state_mut().buffer_mut(),
                0,
                &buf[..vel_len],
            )
            .expect("dof_state push write");
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
        let n = self
            .state
            .multibodies_mut()
            .links_static_mut()
            .buffer()
            .len();
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
        let nc = self
            .state
            .multibodies_mut()
            .joint_constraints()
            .buffer()
            .len();
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
        // BIPED_TERRAIN: heights are relative to the LOCAL ground surface so
        // the base-height reward, fall detection and obs semantics carry over
        // to rough patches unchanged (h = 0 off the strip / flag off).
        let ground_h = self
            .terrain
            .as_ref()
            .map_or(0.0, |ter| ter.strip_for(env).height(t.x, t.y));
        let base = BaseState {
            orientation: [r.x, r.y, r.z, r.w],
            lin_vel_world: [lv.x, lv.y, lv.z],
            ang_vel_world: [av.x, av.y, av.z],
            height: t.z - ground_h,
            pos_xy: [t.x, t.y],
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
                phase: 0.0, // overwritten with self.gait_phase[env] by the caller
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
        // Foot-contact threshold on the foot LINK-ORIGIN height (not the sole).
        // The link origin rests at z~0.035-0.045 when the sole is planted (the
        // sole/collider sits below it), so the old 0.025 was BELOW the planted
        // height — contact never registered, breaking every contact-based gait
        // reward (air_time/single_support/flight/foot_slip/clearance all saw the
        // feet as permanently airborne). 0.05 sits just above the planted height
        // and well below a real swing (foot_clearance_target 0.08), so a planted
        // foot reads contact and a lifted foot reads swing. Overridable for tuning.
        let contact_z: f32 = std::env::var("BIPED_CONTACT_Z")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.robot.foot_contact_z);
        #[allow(non_snake_case)]
        let CONTACT_Z = contact_z;
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
            // Foot "forward" is a per-robot local axis (the G1's foot frame is
            // axis-normalized, putting its forward at +Z instead of +X).
            let fwd = Vec3::from(self.robot.foot_forward_local);
            let foot_fwd_in_base = (base_rot_inv * foot_pose.rotation) * fwd;
            let yaw_rel_base = foot_fwd_in_base.y.atan2(foot_fwd_in_base.x);
            // BIPED_TERRAIN: contact + clearance are relative to the LOCAL
            // ground under the foot (0 off-strip / flag off).
            let foot_ground_h = self
                .terrain
                .as_ref()
                .map_or(0.0, |ter| ter.strip_for(env).height(pos.x, pos.y));
            let contact = pos.z - foot_ground_h < CONTACT_Z;
            let prev_air = self.air_time[env][i];
            let first_contact = contact && prev_air > 0.0;
            // Alternating touchdown: a step that lands on the OTHER foot than the
            // last one to touch down (or the first step ever, last_td_foot == -1).
            let alt_step = first_contact && self.last_td_foot[env] != i as i8;
            new_air[i] = if contact { 0.0 } else { prev_air + dt };
            out[i] = FootObs {
                contact,
                first_contact,
                air_time: if contact { prev_air } else { new_air[i] },
                height: pos.z - foot_ground_h,
                planar_speed,
                tilt,
                yaw_rel_base,
                pos_xy: [pos.x, pos.y],
                alt_step,
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
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
            }
        }
        self.gpu.synchronize().expect("warmup sync");

        // ---- SYNC: host synchronize() per control step ----
        let t0 = Instant::now();
        for _ in 0..t_steps {
            for _ in 0..decim {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
            }
            self.gpu.synchronize().expect("sync");
        }
        let sync_ms = t0.elapsed().as_secs_f64() * 1e3;

        // ---- GRAPH: capture one decimation loop, replay it per step ----
        let graph_ms = if let Some(cuda) = self.gpu.as_cuda() {
            cuda.begin_capture().expect("begin_capture");
            for _ in 0..decim {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
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

    /// Stage one env's joint position targets into the host `links_static`
    /// mirror (uploaded by the next `flush_links_static`).
    fn stage_env_targets(&mut self, e: usize, targets: &[f32; NUM_JOINTS]) {
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

    pub async fn step(&mut self, actions: &[[f32; NUM_JOINTS]]) -> Vec<StepOut> {
        assert_eq!(actions.len(), self.n);

        // (1) Stage every env's motor targets host-side in the mirror, then
        // push the whole `links_static` buffer in ONE write_buffer call.
        // Replaces `num_envs * NUM_JOINTS` per-step write_buffer calls.
        //
        // With BIPED_MOTOR_DELAY, the delay itself runs GPU-side (see the
        // delay-state upload below); staging is identical to the no-delay path.
        let t = Instant::now();
        if self.motor_delay.is_none() {
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
        } else {
            // GPU-side delay: stage the CURRENT targets for every env (exactly
            // the no-delay staging), then upload the per-batch delay state
            // [tick=0, k_eff, prev targets] in ONE additional pre-step write.
            // The gravity_and_lu kernel swaps in the previous target while its
            // per-step tick < k — ZERO mid-decimation host writes (the old
            // per-substep restage stalled the stream on a pageable H2D copy,
            // ~70 ms/step at 4096 envs).
            for e in 0..self.n {
                self.delay_now[e] = self.task.joint_targets(&actions[e]);
                let tg = self.delay_now[e];
                self.stage_env_targets(e, &tg);
            }
            self.timings.stage_motors_ns += t.elapsed().as_nanos() as u64;

            let t = Instant::now();
            self.state
                .multibodies_mut()
                .flush_links_static(&self.gpu)
                .expect("flush motor targets");
            let stride = self.state.multibodies_mut().motor_delay_stride() as usize;
            if self.delay_state_buf.len() != stride * self.n {
                self.delay_state_buf = vec![0.0; stride * self.n];
            }
            for e in 0..self.n {
                let base = e * stride;
                self.delay_state_buf[base] = 0.0; // tick
                self.delay_state_buf[base + 1] = if self.delay_fresh[e] {
                    self.delay_fresh[e] = false;
                    0.0 // first post-reset command applies from substep 0
                } else {
                    self.delay_k[e] as f32
                };
                for j in 0..NUM_JOINTS {
                    let link = self.idx.actuated[j].0 as usize;
                    self.delay_state_buf[base + 2 + link] = self.delay_prev_targets[e][j];
                }
            }
            let buf = std::mem::take(&mut self.delay_state_buf);
            self.state
                .multibodies_mut()
                .write_motor_delay_state(&self.gpu, &buf)
                .expect("write motor delay state");
            self.delay_state_buf = buf;
            self.timings.flush_static_ns += t.elapsed().as_nanos() as u64;
        }

        // (1b) Push perturbation: roughly every `push_interval` control steps
        // (±50% jitter), kick each torso with a random velocity so the policy
        // learns robust, engine-agnostic balance recovery (sim-to-real).
        // Applied BEFORE the physics advance so the kick propagates this step.
        // Off when push_vel=0.
        self.global_step += 1;
        if self.push_vel > 0.0 && self.global_step >= self.next_push_at {
            self.apply_random_pushes().await;
            let base = self.push_interval as f32;
            self.next_push_at = self.global_step + self.rng[0].range(0.5 * base, 1.5 * base) as u64;
        }

        // (2) Advance physics at the control decimation. With BIPED_GRAPH=1 on a
        // CUDA backend, capture the `decimation × pipeline.step` dispatch sequence
        // ONCE (after warmup) into a CUDA graph and replay it per step — removing
        // the ~24 ms/step host re-encode (~half the physics cost). The graph
        // records kernel launches, not data, so the per-step motor-buffer write
        // (above) and resets are honoured on replay. Eager dispatch otherwise.
        let t = Instant::now();
        let mut ran_physics = false;
        #[cfg(feature = "cuda_backend")]
        if std::env::var("BIPED_GRAPH").is_ok() {
            if let Some(g) = self.physics_graph.as_ref() {
                g.0.launch().expect("physics graph replay");
                ran_physics = true;
            } else if self.graph_warmup_steps >= GRAPH_CAPTURE_AT {
                let cuda = self.gpu.as_cuda().expect("cuda backend for BIPED_GRAPH");
                cuda.begin_capture().expect("begin_capture");
                for _ in 0..self.task.decimation {
                    let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
                }
                let g = cuda.end_capture().expect("end_capture");
                g.upload().ok();
                g.launch().expect("first graph launch"); // capture only records; execute once
                self.physics_graph = Some(SyncGraph(g));
                ran_physics = true;
            }
            self.graph_warmup_steps += 1;
        }
        if !ran_physics {
            // PHASE-A substep trace: when BIPED_SUBSTEP_TRACE is set, read env0's
            // foot pose + contact load AFTER each pipeline.step. With
            // BIPED_SOLVER_ITERS=1 each pipeline.step is one substep → per-substep
            // foot trajectory. Forces the non-graph path (this branch) implicitly
            // since the trace readback syncs per step.
            let trace = std::env::var("BIPED_SUBSTEP_TRACE").is_ok();
            for i in 0..self.task.decimation {
                let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
                if trace {
                    self.trace_foot_substep(self.global_step, i).await;
                }
            }
        }
        if self.motor_delay.is_some() {
            for e in 0..self.n {
                self.delay_prev_targets[e] = self.delay_now[e];
            }
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
        // Skip auto-resize once a physics graph is captured — reallocating the
        // state buffers would invalidate the graph's recorded buffer addresses.
        // (Buffers are already stable by capture time, so this is a no-op anyway.)
        let graph_captured = {
            #[cfg(feature = "cuda_backend")]
            {
                self.physics_graph.is_some()
            }
            #[cfg(not(feature = "cuda_backend"))]
            {
                false
            }
        };
        if self.tick_since_resize >= AUTO_RESIZE_PERIOD && !graph_captured {
            let t = Instant::now();
            self.pipeline
                .auto_resize_buffers(&self.gpu, &mut self.state)
                .unwrap();
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
                // BIPED_TERRAIN travel metric (AGILE's): accumulate the
                // straight-line chord from the last resample point.
                if let Some(ter) = &mut self.terrain {
                    let p = poses[e * self.idx.colliders_per_batch as usize
                        + self.idx.torso_link as usize]
                        .translation;
                    let [lx, ly] = ter.last_xy[e];
                    ter.travel[e] += ((p.x - lx).powi(2) + (p.y - ly).powi(2)).sqrt();
                    ter.last_xy[e] = [p.x, p.y];
                }
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
            // Foot index that touched down this step (-1 = none); committed to
            // `self.last_td_foot` in the serial pass to track gait alternation.
            td_foot: i8,
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
        // BIPED_ANKLE_TORQUE_W (0 disables). Default 4.0: at 1.0 the penalty
        // (~-0.003/step) was too cheap against the tracking reward and the
        // learned gait balanced by torquing the ankles (flat-footed shuffle).
        let ankle_torque_w = std::env::var("BIPED_ANKLE_TORQUE_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(4.0);
        // Mechanical-power (energy) penalty weight. Penalizes Σ|τᵢ·q̇ᵢ| — the rate
        // of mechanical work, the principled cost-of-transport proxy. Unlike Στ²
        // (effort, penalized even when static), this only charges for work done in
        // motion, so energy-economical (natural) gaits are favored and degenerate
        // high-energy modes (marching in place, frantic shuffling) are punished.
        // BIPED_POWER_W tunes it (0 = off). Default 4e-3 (was 2e-3): the
        // higher energy price further biases against shuffle/ankle-balance
        // gaits in favor of discrete weight-transferring steps.
        let power_w: f32 = std::env::var("BIPED_POWER_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4e-3);
        let cpb_idx = self.idx.colliders_per_batch as usize;
        let computed: Vec<PerEnv> = (0..self.n)
            .into_par_iter()
            .with_min_len(64)
            .map(|e| {
                let (feet, new_air) = self.compute_feet_from_poses(e, &poses);
                let (mut state, new_joint_pos) = self.read_state_from_poses(e, &poses);
                state.feet = feet;
                state.phase = self.gait_phase[e];
                let env_base = e * cpb_idx;
                let illegal = self.idx.illegal_ground_links.iter().any(|&l| {
                    let p = poses[env_base + l as usize].translation;
                    // BIPED_TERRAIN: threshold vs the LOCAL ground height.
                    let gh = self
                        .terrain
                        .as_ref()
                        .map_or(0.0, |ter| ter.strip_for(e).height(p.x, p.y));
                    p.z - gh < illegal_z
                });
                let fell =
                    illegal || self.task.fell_over(&state.base) || !state.base.height.is_finite();
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
                comps[25] = rb.gait_clock;
                comps[26] = rb.com_centering;
                comps[27] = rb.stand_planted;
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
                if torque_w > 0.0 || ankle_torque_w > 0.0 || power_w > 0.0 {
                    let q_target = self.task.joint_targets(&actions[e]);
                    let mut leg_pen = 0.0f32;
                    let mut ankle_pen = 0.0f32;
                    let mut power = 0.0f32; // Σ|τ·q̇| mechanical power (energy rate)
                    for i in 0..NUM_JOINTS {
                        let j = &self.task.robot.joints[i];
                        let tau = (j.kp * (q_target[i] - state.joint_pos[i])
                            - j.kd * state.joint_vel[i])
                            .clamp(-j.effort_limit, j.effort_limit);
                        let t2 = tau * tau;
                        power += (tau * state.joint_vel[i]).abs();
                        if j.name.contains("ankle") {
                            let w = if j.name.contains("anklex") {
                                6.5e-3
                            } else {
                                1.5e-3
                            };
                            ankle_pen += w * t2;
                        } else {
                            leg_pen += 5e-4 * t2;
                        }
                    }
                    comps[20] = -(torque_w * leg_pen) * sc_dt;
                    comps[21] = -(ankle_torque_w * ankle_pen) * sc_dt;
                    comps[24] = -(power_w * power) * sc_dt;
                    reward -=
                        (torque_w * leg_pen + ankle_torque_w * ankle_pen + power_w * power) * sc_dt;
                }
                let mut obs = vec![0.0; OBS_DIM];
                self.task.observe(&state, &self.cmd[e], &mut obs);
                let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
                self.task
                    .observe_critic(&state, &self.cmd[e], &mut critic_obs);
                // Which foot touched down this step (last wins if both did) — used
                // to advance the gait-alternation tracker in the serial pass.
                let mut td_foot: i8 = -1;
                for (i, f) in state.feet.iter().enumerate() {
                    if f.first_contact {
                        td_foot = i as i8;
                    }
                }
                PerEnv {
                    obs,
                    critic_obs,
                    reward,
                    fell,
                    illegal,
                    comps,
                    new_air,
                    new_joint_pos,
                    td_foot,
                }
            })
            .collect();
        self.timings.par_compute_ns += t.elapsed().as_nanos() as u64;

        // (5) Serial commit: per-env mutable state + StepOut assembly.
        let t = Instant::now();
        let cpb = self.idx.colliders_per_batch as usize;
        // Observation noise (sensor DR): uniform additive noise on the ACTOR obs
        // only (the critic keeps a clean privileged obs — asymmetric PPO). Models
        // encoder quantization / IMU noise so the policy can't overfit to
        // pixel-perfect proprioception. Amplitudes mirror Isaac Lab's UniformNoise
        // for proprioceptive humanoid obs; BIPED_OBS_NOISE scales them (0 = off).
        let obs_noise: f32 = std::env::var("BIPED_OBS_NOISE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let mut outs = Vec::with_capacity(self.n);
        for (e, mut c) in computed.into_iter().enumerate() {
            if obs_noise > 0.0 {
                let rng = &mut self.rng[e];
                // joint_pos_rel [16..28]: ±0.01 rad
                for v in &mut c.obs[NUM_JOINTS + 4..2 * NUM_JOINTS + 4] {
                    *v += rng.range(-0.01, 0.01) * obs_noise;
                }
                // joint_vel [28..40]: ±1.5 rad/s
                for v in &mut c.obs[2 * NUM_JOINTS + 4..3 * NUM_JOINTS + 4] {
                    *v += rng.range(-1.5, 1.5) * obs_noise;
                }
                // projected_gravity [40..43]: ±0.05
                for v in &mut c.obs[3 * NUM_JOINTS + 4..3 * NUM_JOINTS + 7] {
                    *v += rng.range(-0.05, 0.05) * obs_noise;
                }
            }
            // Obs history: push the final (noised) frame, emit the stacked
            // window. Must run after the noise block — the history records
            // exactly what the policy saw (WBC-AGILE ordering).
            if let Some(hist) = &mut self.obs_hist {
                c.obs = hist.push_stacked(e, &c.obs);
            }
            self.air_time[e] = c.new_air;
            if c.td_foot >= 0 {
                self.last_td_foot[e] = c.td_foot;
            }
            // Advance the gait clock (wraps at 1).
            self.gait_phase[e] =
                (self.gait_phase[e] + self.task.control_dt() / self.gait_period).fract();
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
            // BIPED_TERRAIN: on EVERY episode end (fall/illegal/timeout — AGILE
            // updates for all resets), close the travel chord and run the
            // promote/demote state machine; the next reset spawns at the
            // (possibly new) level's patch.
            if c.fell || timeout {
                if let Some(ter) = &mut self.terrain {
                    let p = poses[env_base + self.idx.torso_link as usize].translation;
                    let [lx, ly] = ter.last_xy[e];
                    let traveled =
                        ter.travel[e] + ((p.x - lx).powi(2) + (p.y - ly).powi(2)).sqrt();
                    let rng = &mut ter.rng[e];
                    ter.curriculum[e].on_episode_end(traveled, rng);
                }
            }
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
        // Per-step so the stance-phase path/drift accumulation is real (each call
        // does 2 readbacks — fine for a short diagnostic run, not for training).
        if std::env::var("BIPED_DEBUG_CONTACT").is_ok() {
            self.debug_contact_impulses().await;
        }
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

    /// Mean terrain-difficulty level across envs (BIPED_TERRAIN; the
    /// curriculum's progress metric — AGILE logs the same).
    pub fn mean_terrain_level(&self) -> Option<f32> {
        self.terrain.as_ref().map(|t| {
            t.curriculum.iter().map(|c| c.level as f32).sum::<f32>()
                / t.curriculum.len().max(1) as f32
        })
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
        if self.terrain.is_some() {
            // Teleport to the env's current difficulty patch (level was
            // already updated by the curriculum when the episode ended).
            let off = self.terrain_spawn_offset(env);
            self.state.reset_env_from_snapshot_offset(
                &self.gpu,
                env as u32,
                &self.template_snapshots[t],
                off,
            );
        } else {
            self.state
                .reset_env_from_snapshot(&self.gpu, env as u32, &self.template_snapshots[t]);
        }
        // AGILE reset-velocity randomization: overwrite the fresh env's dof
        // velocities (snapshot resets them to 0) so the episode starts in
        // motion. Layout per env in `dof_state`: [0..3) root lin, [3..6) root
        // ang, [6..dpb) joint velocities; element-offset write touches only
        // this env's slice of the velocity section.
        if self.reset_vel {
            let dpb = self.state.multibodies_mut().dofs_per_batch_count() as usize;
            let mut v = vec![0.0f32; dpb];
            v[0] = self.rng[env].range(-0.25, 0.25);
            v[1] = self.rng[env].range(-0.25, 0.25);
            for d in 3..6 {
                v[d] = self.rng[env].range(-0.5, 0.5);
            }
            for d in 6..dpb {
                v[d] = self.rng[env].range(-1.0, 1.0);
            }
            self.gpu
                .write_buffer(
                    self.state.multibodies_mut().dof_state_mut().buffer_mut(),
                    (env * dpb) as u64,
                    &v,
                )
                .expect("dof_state reset-velocity write");
        }
        // Mirror the template's sole-normal so update_feet's tilt makes sense.
        // Cached per-template (constant) — NO per-reset rapier-scene rebuild.
        self.foot_sole_local[env] = self.template_foot_sole[t];

        // Reset host state.
        self.cmd[env] = self.sampler.sample(&mut self.rng[env]);
        self.step_count[env] = 0;
        self.resample_at[env] = self
            .sampler
            .resample_steps(&mut self.rng[env], self.task.control_dt());
        self.last_action[env] = [0.0; NUM_JOINTS];
        self.prev_action[env] = [0.0; NUM_JOINTS];
        self.air_time[env] = [0.0; NUM_FEET];
        self.last_td_foot[env] = -1;
        self.gait_phase[env] = 0.0;
        // Actuator delay: resample the lag for the new episode (from the
        // DEDICATED delay stream — the command/DR stream stays untouched) and
        // mark fresh so the first post-reset command applies from substep 0
        // (`delay_prev_targets` is stale across the reset).
        if let Some((min, max)) = self.motor_delay {
            let r = self.delay_rng[env].range(0.0, 1.0);
            self.delay_k[env] = min + ((r * (max - min + 1) as f32) as u32).min(max - min);
            self.delay_fresh[env] = true;
        }

        // Cached prev joint angles + poses are stale across a reset; clear so
        // the next step seeds them again with zero velocity.
        self.has_prev_joint_pos[env] = false;
        self.has_prev_pose[env] = false;

        // Fast path: serve the cached per-template spawn obs with the fresh
        // command patched into [12:16] — NO `slurp_poses` readback (the dominant
        // per-reset cost). The post-reset state is the template spawn state
        // (joints 0, vel 0, last_action 0); the command is the only thing that
        // varies and it enters obs ONLY at [12:16] (see VelocityFlatTask::observe).
        if !self.template_spawn_obs.is_empty() {
            let mut obs = self.template_spawn_obs[t].clone();
            let mut critic_obs = self.template_spawn_critic_obs[t].clone();
            let c = self.cmd[env].obs(); // [vx, vy, yaw, 0]
            obs[NUM_JOINTS..NUM_JOINTS + 4].copy_from_slice(&c);
            critic_obs[NUM_JOINTS..NUM_JOINTS + 4].copy_from_slice(&c);
            // Opt-in self-check: confirm the cached obs equals the live readback
            // path bit-for-bit (run once with BIPED_VERIFY_RESET=1 to validate).
            if std::env::var("BIPED_VERIFY_RESET").is_ok() {
                let poses = self.slurp_poses().await;
                let (feet, _) = self.compute_feet_from_poses(env, &poses);
                let (mut state, _) = self.read_state_from_poses(env, &poses);
                state.feet = feet;
                let mut ref_obs = vec![0.0; OBS_DIM];
                self.task.observe(&state, &self.cmd[env], &mut ref_obs);
                let mut ref_co = vec![0.0; CRITIC_OBS_DIM];
                self.task
                    .observe_critic(&state, &self.cmd[env], &mut ref_co);
                let do_max = obs
                    .iter()
                    .zip(&ref_obs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let dc_max = critic_obs
                    .iter()
                    .zip(&ref_co)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "[verify_reset] env {env} tpl {t}: obs maxdiff={do_max:.3e} critic maxdiff={dc_max:.3e}"
                );
            }
            // Obs history: replicate the fresh 45-frame into all H slots
            // (the verify check above compares the pre-stack frame).
            if let Some(hist) = &mut self.obs_hist {
                return (hist.reset_stacked(env, &obs), critic_obs);
            }
            return (obs, critic_obs);
        }

        // Fallback (cache not yet populated): build obs from a readback.
        let poses = self.slurp_poses().await;
        let (feet, _) = self.compute_feet_from_poses(env, &poses);
        let (mut state, _) = self.read_state_from_poses(env, &poses);
        state.feet = feet;
        let mut obs = vec![0.0; OBS_DIM];
        self.task.observe(&state, &self.cmd[env], &mut obs);
        let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
        self.task
            .observe_critic(&state, &self.cmd[env], &mut critic_obs);
        if let Some(hist) = &mut self.obs_hist {
            obs = hist.reset_stacked(env, &obs);
        }
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
        // Cache the per-template spawn obs: env `t` was seeded from template `t`
        // at construction and no reset has happened yet, so obs[t] IS template
        // t's spawn obs. reset_env serves these (with the command patched into
        // [12:16]) instead of a per-reset readback. Command-agnostic: the baked-in
        // command is overwritten on reset.
        let nt = self.templates.len().min(self.n);
        self.template_spawn_obs = obs[..nt].to_vec();
        self.template_spawn_critic_obs = critic_obs[..nt].to_vec();
        // Obs history: the template cache above holds single 45-frames (the
        // reset fast path patches the command into the frame before stacking);
        // the vectors handed to the policy get the replicated H-stack.
        if let Some(hist) = &mut self.obs_hist {
            for (e, o) in obs.iter_mut().enumerate() {
                *o = hist.reset_stacked(e, o);
            }
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
        assert!(!self.template_snapshots.is_empty());
        self.state
            .reset_env_from_snapshot(&self.gpu, e as u32, &self.template_snapshots[0]);
        self.foot_sole_local[e] = self.idx.foot_sole_local;
        self.cmd[e] = VelocityCommand::default();
        self.step_count[e] = 0;
        // Pin the resample so the command stays where the caller pins it.
        self.resample_at[e] = u32::MAX;
        self.last_action[e] = [0.0; NUM_JOINTS];
        self.prev_action[e] = [0.0; NUM_JOINTS];
        self.air_time[e] = [0.0; NUM_FEET];
        self.last_td_foot[e] = -1;
        self.gait_phase[e] = 0.0;
        // Deterministic render path: pin the delay to `min` (no RNG draw).
        if let Some((min, _)) = self.motor_delay {
            self.delay_k[e] = min;
            self.delay_fresh[e] = true;
        }
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
        if let Some(hist) = &mut self.obs_hist {
            obs = hist.reset_stacked(e, &obs);
        }
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

    /// DEBUG: read back the narrow-phase contact manifolds (the shared
    /// collision-detection output consumed by the multibody contact solver).
    /// Returns `(reported_len[..], manifolds[..capacity])`. Used to diagnose
    /// foot↔ground contact on WebGpu vs CUDA: contact COUNT (is narrow-phase
    /// generating foot-ground pairs at all?) and the contact NORMAL direction.
    pub async fn dbg_contacts(&mut self) -> (Vec<u32>, Vec<NexusIndexedContact>) {
        let lbuf = self.state.dbg_contacts_len().buffer();
        let mut len = vec![0u32; lbuf.len()];
        self.gpu
            .slow_read_buffer(lbuf, &mut len)
            .await
            .expect("contacts_len readback");
        let cbuf = self.state.dbg_contacts().buffer();
        let mut v = vec![NexusIndexedContact::default(); cbuf.len()];
        self.gpu
            .slow_read_buffer(cbuf, &mut v)
            .await
            .expect("contacts readback");
        (len, v)
    }

    /// DEBUG: broad-phase pair count (how many collider pairs the LBVH found),
    /// and the raw pair list. Splits "broad-phase finds nothing" from
    /// "narrow-phase generates no manifold" when contacts come back empty.
    pub async fn dbg_collision_pairs(&mut self) -> (Vec<u32>, Vec<[u32; 2]>) {
        let lbuf = self.state.dbg_collision_pairs_len().buffer();
        let mut len = vec![0u32; lbuf.len()];
        self.gpu
            .slow_read_buffer(lbuf, &mut len)
            .await
            .expect("pairs_len readback");
        let pbuf = self.state.dbg_collision_pairs().buffer();
        let mut raw: Vec<nexus3d::rbd::shaders::broad_phase::CollisionPair> =
            vec![
                nexus3d::rbd::shaders::broad_phase::CollisionPair {
                    colliders: glamx::UVec2::new(0, 0).into(),
                };
                pbuf.len() as usize
            ];
        self.gpu
            .slow_read_buffer(pbuf, &mut raw)
            .await
            .expect("pairs readback");
        let v: Vec<[u32; 2]> = raw
            .iter()
            .map(|p| [p.colliders.x, p.colliders.y])
            .collect();
        (len, v)
    }

    /// DEBUG: read back the per-multibody contact-constraint bank
    /// (`inv_lhs` = 1/(J·M⁻¹·Jᵀ), `rhs`, accumulated `impulse`, jacobians) and
    /// the per-batch active counts. Diagnoses the WebGpu contact-solve blow-up.
    pub async fn dbg_mb_contacts(&mut self) -> (Vec<u32>, Vec<NexusMbContact>) {
        let mut cnt = vec![
            0u32;
            self.state
                .multibodies_mut()
                .dbg_contact_constraint_count()
                .buffer()
                .len()
        ];
        self.gpu
            .slow_read_buffer(
                self.state
                    .multibodies_mut()
                    .dbg_contact_constraint_count()
                    .buffer(),
                &mut cnt,
            )
            .await
            .expect("cc count readback");
        let mut ccs = vec![
            NexusMbContact::default();
            self.state
                .multibodies_mut()
                .dbg_contact_constraints()
                .buffer()
                .len()
        ];
        self.gpu
            .slow_read_buffer(
                self.state
                    .multibodies_mut()
                    .dbg_contact_constraints()
                    .buffer(),
                &mut ccs,
            )
            .await
            .expect("cc readback");
        (cnt, ccs)
    }

    /// DEBUG: world pose of every body for all envs (spawn-divergence check:
    /// print these BEFORE the first step on each backend and diff).
    pub async fn dbg_body_poses(&mut self) -> Vec<NexusPose> {
        self.slurp_poses().await
    }

    /// DEBUG: read back the per-constraint `Jᵀ` rows and `M⁻¹·Jᵀ` columns plus
    /// the strides to slice them: `(jacs, columns, (columns_per_batch,
    /// dofs_per_batch, constraints_per_batch))`. Slot `s` of batch `b` is
    /// `[b*columns_per_batch + s*dofs_per_batch ..][..ndofs]` in both banks.
    /// The columns are the prime suspect for the WebGpu contact divergence.
    pub async fn dbg_mb_jac_columns(&mut self) -> (Vec<f32>, Vec<f32>, (u32, u32, u32)) {
        let strides = self
            .state
            .multibodies_mut()
            .dbg_contact_constraint_strides();
        let jbuf = self
            .state
            .multibodies_mut()
            .dbg_contact_constraint_jacs()
            .buffer();
        let mut jacs = vec![0f32; jbuf.len()];
        self.gpu
            .slow_read_buffer(jbuf, &mut jacs)
            .await
            .expect("jacs readback");
        let cbuf = self
            .state
            .multibodies_mut()
            .dbg_contact_constraint_columns()
            .buffer();
        let mut cols = vec![0f32; cbuf.len()];
        self.gpu
            .slow_read_buffer(cbuf, &mut cols)
            .await
            .expect("columns readback");
        (jacs, cols, strides)
    }

    /// DEBUG: read back the packed dof state (velocities first,
    /// `dofs_per_batch` per batch) and the LU-factored mass matrices.
    pub async fn dbg_mb_dof_state_and_lu(&mut self) -> (Vec<f32>, Vec<f32>) {
        let dbuf = self.state.multibodies_mut().dbg_dof_state().buffer();
        let mut dofs = vec![0f32; dbuf.len()];
        self.gpu
            .slow_read_buffer(dbuf, &mut dofs)
            .await
            .expect("dof_state readback");
        let mbuf = self.state.multibodies_mut().dbg_mass_matrices().buffer();
        let mut mm = vec![0f32; mbuf.len()];
        self.gpu
            .slow_read_buffer(mbuf, &mut mm)
            .await
            .expect("mass_matrices readback");
        (dofs, mm)
    }

    /// Global collider index of the ground cuboid in env `e` (last collider
    /// per env, or second-to-last when the terrain trimesh is appended).
    pub fn ground_collider(&self, e: usize) -> u32 {
        let after_ground = self.terrain.is_some() as u32;
        (e as u32 + 1) * self.idx.colliders_per_batch - 1 - after_ground
    }
}

// --- Helpers -----------------------------------------------------------------

/// Pick the GPU backend for the batched physics via [`KhalGpuBackend::auto`]:
/// native CUDA on Blackwell (`sm_120`+, when built with `cuda_backend`), else
/// WebGPU. Override with `KHAL_BACKEND=cuda|webgpu`. The nexus + vortx cubins are
/// embedded at build time via the per-crate `CUDA_OXIDE_SHADERS_PTX_*` env vars.
async fn make_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut bk = KhalGpuBackend::auto(wgpu::Features::default(), limits)
        .await
        .expect("init GPU backend");
    // The WebGPU biped path needs buffer copy-src for state readbacks.
    if let KhalGpuBackend::WebGpu(w) = &mut bk {
        w.force_buffer_copy_src = true;
    }
    bk
}

/// Sample one DR point. Ranges mirror `Randomization::default()` from the CPU
/// env (minus push perturbations, which nexus can't apply at runtime).
/// Initial-pose jitter ranges are conservative — wider tilts make every
/// episode start mid-fall, which the policy can't recover from at small T.
fn sample_dr(rng: &mut Lcg) -> DrParams {
    // BIPED_AGILE_DR=1: sample the WBC-AGILE LocomotionEventCfg ranges instead
    // of zealot's (which are 2–4× harsher exactly where stepping is risky —
    // kp ±30% vs their ±10%, link mass ±20% vs ±5%, spawn tilt ±20° vs ±10°).
    // AGILE-side mapping: friction single-μ U(0.2,1.25) (their static 0.2–1.5 /
    // dynamic 0.2–1.0; nexus has one μ), restitution U(0,0.1), per-joint kp
    // ±10% (via pd_scale_per_joint — also touches effort ±10%, deviation:
    // AGILE leaves effort alone), kd ×U(0.8,2.0) per env (theirs is per joint),
    // link mass ×U(0.95,1.05) + pelvis payload +U(−1,5) kg, tilt ±10°, no z
    // jitter. Not modeled: CoM offsets, armature ×U(0,2), continuous wrenches.
    if std::env::var("BIPED_AGILE_DR").is_ok_and(|v| v == "1") {
        let pd_scale = 1.0;
        let kd_scale = rng.range(0.8, 2.0);
        let friction = rng.range(0.2, 1.25);
        let restitution = rng.range(0.0, 0.1);
        let mass_scale = rng.range(0.95, 1.05);
        let base_payload_kg = rng.range(-1.0, 5.0);
        let mut pd_scale_per_joint = [1.0f32; NUM_JOINTS];
        for v in pd_scale_per_joint.iter_mut() {
            *v = rng.range(0.9, 1.1);
        }
        return DrParams {
            friction,
            restitution,
            pd_scale,
            kd_scale,
            mass_scale,
            base_payload_kg,
            contact_natural_frequency: rng.range(10.0, 50.0),
            contact_damping_ratio: rng.range(2.0, 8.0),
            spawn_yaw: rng.range(-std::f32::consts::PI, std::f32::consts::PI),
            spawn_roll: rng.range(-0.1745, 0.1745),
            spawn_pitch: rng.range(-0.1745, 0.1745),
            spawn_z_offset: 0.0,
            pd_scale_per_joint,
        };
    }
    // BIPED_SPAWN_DR scales the initial-pose tilt/height randomization (default
    // 1.0). Set to 0.0 to start every episode upright at nominal height — used to
    // test whether aggressive spawn DR is what's preventing the policy from
    // getting a learning gradient (the rng draws are still consumed, so dynamics
    // DR and determinism are unchanged).
    let sdr: f32 = std::env::var("BIPED_SPAWN_DR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    // BIPED_FRICTION: force a fixed Coulomb μ on every env (overrides the random
    // draw) — used to A/B-test that friction actually reaches the GPU contact
    // solver. The rng draw is still consumed so other DR + determinism are
    // unchanged.
    // Friction range widened DOWN into the slip regime: 0.3–1.3 (was 0.5–1.5).
    // The low tail (μ≈0.3) makes the foot actually slip, so the policy can't
    // rely on a consistent grip to brace — this is the dominant "slippery
    // contact" DR lever both MuJoCo (geom friction randomization) and Isaac
    // (randomize_rigid_body_material) use. Center stays ≈ MuJoCo's default μ=1.
    // (Per-foot and static-vs-dynamic friction would express stick-slip even
    // better, but nexus stores a single Coulomb μ per multibody — engine-blocked.)
    let friction = match std::env::var("BIPED_FRICTION")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        Some(f) => {
            let _ = rng.range(0.3, 1.3);
            f
        }
        None => rng.range(0.3, 1.3),
    };
    // BIPED_MASS_DR scales the half-width of the per-link mass randomization
    // (default 1.0 → ±20%). Set 0.0 to disable (mass fixed at nominal); the rng
    // draw is still consumed so other DR + determinism are unchanged.
    let mdr: f32 = std::env::var("BIPED_MASS_DR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mass_scale = 1.0 + rng.range(-0.2, 0.2) * mdr;
    DrParams {
        friction,
        restitution: rng.range(0.0, 0.15),
        // Widened from ±15% to ±30%: PD-gain error is a major sim-to-real gap
        // (the real actuators' effective kp/kd differ from the modelled values),
        // and a policy that's robust to ±30% gain error transfers far better.
        pd_scale: rng.range(0.7, 1.3),
        kd_scale: 1.0,
        mass_scale,
        base_payload_kg: 0.0,
        // Contact-stiffness DR — now LIVE on the multibody contact solver (the
        // kernel reads per-env contact_natural_frequency / contact_damping_ratio
        // from SimParams; it used to hardcode 30/5). This is the analog of
        // MuJoCo's solref randomization. BIPED_CONTACT_FREQ / BIPED_CONTACT_DAMP
        // pin every env to a fixed value (rng draw still consumed) — set both to
        // 30 / 5 to reproduce the old hardcoded path and verify the new binding
        // is bit-identical.
        contact_natural_frequency: match std::env::var("BIPED_CONTACT_FREQ")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(f) => {
                let _ = rng.range(10.0, 50.0);
                f
            }
            None => rng.range(10.0, 50.0),
        },
        contact_damping_ratio: match std::env::var("BIPED_CONTACT_DAMP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(d) => {
                let _ = rng.range(2.0, 8.0);
                d
            }
            None => rng.range(2.0, 8.0),
        },
        // Initial-pose DR — aggressive ranges so the policy sees a wide
        // distribution of starts and learns to recover from non-trivial
        // perturbations. Comparable to WBC-AGILE / Isaac Lab humanoid
        // defaults (±15–25° on tilts, a few cm on height). Wider than this
        // (e.g. ±30° tilts) makes most episodes start mid-fall and PPO
        // can't get a useful gradient with the curriculum's early
        // command-velocity scale.
        spawn_yaw: rng.range(-std::f32::consts::PI, std::f32::consts::PI),
        spawn_roll: rng.range(-0.35, 0.35) * sdr, // ±~20° (× BIPED_SPAWN_DR)
        spawn_pitch: rng.range(-0.35, 0.35) * sdr, // ±~20° (× BIPED_SPAWN_DR)
        spawn_z_offset: rng.range(-0.08, 0.08) * sdr, // ±8 cm (× BIPED_SPAWN_DR)
        // Per-joint actuator-strength asymmetry (BIPED_ASYM_DR = half-width,
        // default ±15%; 0 disables). Each joint draws independently → left/right
        // gains differ, modelling "one motor stronger than the other". Drawn LAST
        // so enabling it doesn't perturb the rng order of the other DR fields.
        // A symmetric policy handles this REACTIVELY: the weaker side tracks its
        // target worse → shows up in the joint-pos/vel obs → the (symmetric) map
        // responds; the distribution is L/R-balanced, so the mirror prior stays
        // valid in expectation.
        pd_scale_per_joint: {
            let hw: f32 = std::env::var("BIPED_ASYM_DR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.15);
            let mut a = [1.0f32; NUM_JOINTS];
            for v in a.iter_mut() {
                *v = 1.0 + rng.range(-hw, hw);
            }
            a
        },
    }
}
