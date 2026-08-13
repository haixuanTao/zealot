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
use nexus3d::rbd::math::Vector as NexusVector;
use nexus3d::rbd::pipeline::{RbdPipeline, RbdSnapshot, RbdState};
use nexus3d::rbd::queries::GpuIndexedContact as NexusIndexedContact;
use nexus3d::rbd::shaders::dynamics::MultibodyContactConstraint as NexusMbContact;
use nexus3d::rbd::shaders::dynamics::MAX_CONTACT_SENSORS;
use nexus3d::rbd::shaders::dynamics::MultibodyLinkWorkspace;
use rapier3d::prelude::*;
use rayon::prelude::*;
use roxmltree::Node;
use std::collections::HashMap;
// `std::time::Instant::now()` panics on wasm32-unknown-unknown; `web-time`
// forwards to `performance.now()` there and re-exports std everywhere else.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;
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

/// Programmatic overrides for the `BIPED_*` sim-param knobs (same wasm
/// rationale as [`DECIMATION_OVERRIDE`]). Consulted BEFORE the process
/// environment by the sim-param reads in `build_scene`.
pub static ENV_OVERRIDES: std::sync::OnceLock<
    std::collections::HashMap<&'static str, &'static str>,
> = std::sync::OnceLock::new();

/// `std::env::var` with the [`ENV_OVERRIDES`] table taking precedence. Every
/// `BIPED_*`/`NEXUS_*` knob in this file reads through this, so wasm demos
/// (no process environment) can configure the env in code.
fn env_var(k: &str) -> Result<String, std::env::VarError> {
    if let Some(v) = ENV_OVERRIDES.get().and_then(|m| m.get(k)) {
        return Ok((*v).to_string());
    }
    std::env::var(k)
}

/// [`env_var`], `Option`-shaped.
fn env_or_override(k: &str) -> Option<String> {
    env_var(k).ok()
}

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
    if let Ok(p) = env_var("BIPED_MJCF") {
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
    let asset_dir = env_var("BIPED_MESH_DIR")
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
    /// Foot collider shape for this template: 0 = the BIPED_FOOT_SHAPE env
    /// default, 1 = box, 2 = capsule. `BIPED_FOOT_SHAPE=dr` samples 1/2 at
    /// 50/50 per template — STRUCTURAL-error DR: the box's sharp edges and
    /// the capsule's roll-through resolve contact so differently that a
    /// policy trained on both can't overfit either engine's handling of
    /// either shape (μ-DR alone can't cover model-shape bias — the box
    /// friction corner over-grant was multiplicative at every μ).
    pub foot_shape_id: u8,
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
            foot_shape_id: 0,
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
    /// Multibody link index of the CHEST — the `torso_link` body above the
    /// waist chain on full-body models. Every "stability" reward term reads the
    /// pelvis (link 0); the chest rides above three PD-held waist joints and is
    /// otherwise invisible to the reward. Falls back to 0 (pelvis) on models
    /// without a `torso_link` body (legs-only), where the chest term would
    /// duplicate `body_ang_vel` anyway.
    pub chest_link: u32,
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

    /// PD-held non-action joints (G1 waist + arms), in MJCF insertion order —
    /// the upper-body staging targets for the AMASS arm-motion disturbance
    /// (BIPED_ARM_MOTION). Empty when the model has none or BIPED_LOCK_HELD
    /// welded them (no motor to stage).
    pub held: Vec<HeldJoint>,
}

/// One PD-held (non-action) joint: enough to restage its motor target.
#[derive(Clone, Debug)]
pub struct HeldJoint {
    pub link: u32,
    pub name: String,
    /// The build-time hold target (`held_home` or 0) the joint returns to
    /// when no clip is playing.
    pub home: f32,
    /// MJCF joint range — clip poses are clamped into it (retargeted mocap
    /// can exceed the robot's mechanical range).
    pub range: (f32, f32),
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
    // (joint handle, injected angle) pairs collected while inserting joints —
    // consumed by the BIPED_INIT_POSE coordinate seeding below.
    let mut init_joint_handles: Vec<(MultibodyJointHandle, f32)> = Vec::new();

    // FK world poses with initial-pose jitter on the root: yaw + roll + pitch
    // + height. Joint angles stay at neutral (the multibody rest pose).
    // Composing intrinsic ZYX so yaw is the outermost rotation (the typical
    // RL convention — yaw randomises heading, roll/pitch perturb upright).
    // BIPED_INIT_RP="roll,pitch" (rad) adds a fixed base tilt on top of the
    // template's sampled spawn orientation — snapshot replay of a real-robot
    // IMU attitude (see BIPED_INIT_POSE below).
    let (init_roll, init_pitch) = env_var("BIPED_INIT_RP")
        .ok()
        .and_then(|s| {
            let (r, p) = s.split_once(',')?;
            Some((r.trim().parse().ok()?, p.trim().parse().ok()?))
        })
        .unwrap_or((0.0f32, 0.0f32));
    let root_rot = Rotation::from_rotation_z(dr.spawn_yaw)
        * Rotation::from_rotation_y(dr.spawn_pitch + init_pitch)
        * Rotation::from_rotation_x(dr.spawn_roll + init_roll);
    // BIPED_FREEFALL_Z lifts the spawn clear of the ground so the robot is in
    // TRUE contact-free free-fall — the clean g/M-consistency test (pre-contact
    // generalized accel `a` must equal pure free-fall: base linear = g, all joints
    // ≈ 0). Diagnostic only.
    let ff_z: f32 = env_var("BIPED_FREEFALL_Z")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let root_pos = Vec3::new(0.0, 0.0, robot.spawn_z + dr.spawn_z_offset + ff_z);
    let root_pose = Pose::from_parts(root_pos, root_rot);
    // BIPED_INIT_POSE="left_knee_joint=0.59;left_elbow_joint=1.4;..." spawns
    // the scene with the named hinges rotated to the given angles (rad) —
    // snapshot replay of a real-robot pose. Every hinge in these models turns
    // about the CHILD's local +Z, so the rotation composes after `local_quat`;
    // the multibody picks the joint coordinate up from the relative body poses
    // at insertion, and `joint_angles_for` reads it back unchanged because
    // `actuated_rest_quat` stays the UNROTATED MJCF local quat. Unknown joint
    // names are ignored. When set, the whole robot is auto-shifted vertically
    // so the lowest sole point spawns 2 mm above the ground (the recorded z of
    // the real base is unknown — feet-on-floor is the reconstruction).
    let init_pose: HashMap<String, f32> = env_var("BIPED_INIT_POSE")
        .map(|s| {
            s.split(';')
                .filter_map(|kv| {
                    let (k, v) = kv.split_once('=')?;
                    Some((k.trim().to_string(), v.trim().parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut world: Vec<Pose> = Vec::with_capacity(mjcf.len());
    for b in mjcf {
        let w = match b.parent {
            None => root_pose,
            Some(p) => {
                let q0 = b
                    .joint
                    .as_deref()
                    .and_then(|j| init_pose.get(j).copied())
                    .unwrap_or(0.0);
                world[p]
                    * Pose::from_parts(
                        b.local_pos,
                        b.local_quat * Rotation::from_rotation_z(q0),
                    )
            }
        };
        world.push(w);
    }
    if !init_pose.is_empty() {
        let mut min_z = f32::INFINITY;
        for (i, b) in mjcf.iter().enumerate() {
            for (a, c, r) in &b.capsules {
                for p in [a, c] {
                    let z = (world[i] * *p).z - r;
                    min_z = min_z.min(z);
                }
            }
        }
        if min_z.is_finite() {
            let dz = 0.002 - min_z;
            for w in &mut world {
                *w = Pose::from_parts(w.translation + Vec3::new(0.0, 0.0, dz), w.rotation);
            }
        }
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
        let o = if env_var("BIPED_DIAG_INERTIA").is_ok() {
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
            if env_var("BIPED_DBG_FOOT").is_ok() {
                eprintln!("[dbgf] link {} box lo {:?} hi {:?} center {:?} he {:?}", b.name, lo, hi, center, he);
            }
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
            let foot_shape = match dr.foot_shape_id {
                1 => "box".to_string(),
                2 => "capsule".to_string(),
                _ => env_or_override("BIPED_FOOT_SHAPE").unwrap_or_else(|| "capsule".to_string()),
            };
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
                // Preserve the sole-bottom height ON THE SOLE SIDE of the
                // thickness axis. The sole plane lies on whichever side of the
                // link origin the rail centerlines sit (the ankle joint is
                // always above the sole), so `down` = sign of the centerline
                // plane along `tax`. The capsule's sole-side extreme is
                // center ± radius vs the box's center ± he[tax]; shift the
                // center so those coincide. Shifting toward the WRONG side
                // (the old unconditional `+= radius − he`) leaves the capsule
                // 2·(radius − he) too thick under the sole — the G1's
                // converter frame has the sole on the +axis side and the robot
                // rode ~4 cm above the ground on an invisible fat foot.
                let center_tax = match tax {
                    0 => center.x,
                    1 => center.y,
                    _ => center.z,
                };
                let mut down = if center_tax >= 0.0 { 1.0 } else { -1.0 };
                // BIPED_FOOT_FAT=1: replicate the PRE-FIX collider (shift
                // toward the wrong side → capsule 2·(radius−he) thicker under
                // the sole). A/B knob for comparing against results taken
                // before the sole-side fix — e.g. the champagne stand tuning.
                if env_or_override("BIPED_FOOT_FAT").as_deref() == Some("1") {
                    down = -down;
                }
                let shift = (radius - he_arr[tax]) * -down;
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
    let joint_limits_on = env_var("BIPED_JOINT_LIMITS")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Held (non-action) joints collected as they're built, for the AMASS
    // upper-body playback (BIPED_ARM_MOTION). Skipped under BIPED_LOCK_HELD
    // (welded joints carry no motor to restage).
    let mut held_joints: Vec<HeldJoint> = Vec::new();
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
        // Per-family ankle gain override (BIPED_ANKLE_KP / BIPED_ANKLE_KD),
        // applied before DR. AGILE's ankles are kp 20 / kd 0.2 (roll 0.1) —
        // with the foot planted the body's ~12 kg·m² reflected inertia makes
        // that a damping ratio of 0.003-0.006, i.e. effectively UNDAMPED, and
        // measured stand tremor peaks at 9-10 rad/s there while every other
        // joint sits near 0.08. unitree_rl_gym's deploy pair (40 / 2.0) gives
        // ~0.046 planted and stays ~critically damped in swing. Unset = the
        // spec value (AGILE parity preserved by default).
        let (ankle_kp_ovr, ankle_kd_ovr) = (
            std::env::var("BIPED_ANKLE_KP").ok().and_then(|s| s.parse::<f32>().ok())
                .or(robot.name.starts_with("unitree_g1").then_some(40.0)),
            std::env::var("BIPED_ANKLE_KD").ok().and_then(|s| s.parse::<f32>().ok())
                .or(robot.name.starts_with("unitree_g1").then_some(2.0)),
        );
        let (kp, kd, effort, pos_limit, spec_damping) = spec
            .map(|s| {
                let is_ankle = s.name.contains("ankle");
                let base_kp = match ankle_kp_ovr {
                    Some(v) if is_ankle => v,
                    _ => s.kp,
                };
                let base_kd = match ankle_kd_ovr {
                    Some(v) if is_ankle => v,
                    _ => s.kd,
                };
                (
                    base_kp * dr.pd_scale * pj,
                    base_kd * dr.pd_scale * dr.kd_scale * pj,
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
        // BIPED_LOCK_HELD=1: weld non-action joints (G1 waist/arms) rigid
        // instead of PD-holding them. The held-gains default is underdamped:
        // during a passive stand the arms swing ~7cm in 1.6s — a moving-COM
        // disturbance that alone drives a constant base-pitch drift.
        let lock_held = spec.is_none() && env_var("BIPED_LOCK_HELD").is_ok();
        let axes = if lock_held {
            locked | JointAxesMask::ANG_Z
        } else {
            locked
        };
        let mut joint = GenericJointBuilder::new(axes)
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
        let motor_model = if env_or_override("BIPED_ACCEL_MOTOR").is_some() {
            MotorModel::AccelerationBased
        } else {
            MotorModel::ForceBased
        };
        if !lock_held {
            joint.set_motor_model(JointAxis::AngZ, motor_model);
        }
        // Motor gains come straight from the robot spec (RobotSpec::joints),
        // which already bakes in the physical torque-PD correction (STIFFNESS_SCALE
        // / DAMPING_SCALE) — kp is a real torque/rad gain, FIXED, identical to what
        // the MuJoCo transfer model and the real robot use. No runtime scaling.
        // BIPED_KP_SCALE / BIPED_KD_SCALE remain only as optional diagnostics
        // (default 1.0); leave them unset for the production gains.
        let kp_scale: f32 = env_or_override("BIPED_KP_SCALE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let kd_scale: f32 = env_or_override("BIPED_KD_SCALE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        // Held joints hold at the spec's `held_home` target, falling back to 0
        // (the model's rest pose). For the G1 this puts the arms ALONG THE BODY
        // rather than out in front: the MJCF bakes a 73.2 deg elbow bend into
        // the body hierarchy, so q = 0 is a bent arm, and elbow = 1.28 cancels
        // it. Every sim2sim harness and the LeRobot controller already command
        // 1.28; training held 0, so it ran a different upper-body geometry than
        // every evaluator and the real robot.
        // BIPED_HELD_HOME=0 restores the pre-fix behaviour (every held joint at
        // the model rest pose = arms OUT IN FRONT). Required to evaluate v19-v27
        // checkpoints, which were TRAINED that way -- without it, rebuilding the
        // render binary silently re-poses those policies and their numbers stop
        // matching every measurement taken before this change.
        let held_home_on = std::env::var("BIPED_HELD_HOME")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        let hold_target = if held_home_on {
            robot
                .held_home
                .iter()
                .find(|(frag, _)| jname.contains(frag))
                .map(|&(_, t)| t)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        if !lock_held {
            joint.set_motor_position(
                JointAxis::AngZ,
                hold_target,
                kp * kp_scale,
                kd * kd_scale,
            );
            joint.set_motor_max_force(JointAxis::AngZ, effort);
        }
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
        if spec.is_none() && !lock_held {
            held_joints.push(HeldJoint {
                link: next_mb_link,
                name: jname.clone(),
                home: hold_target,
                range: b.joint_range.unwrap_or(pos_limit),
            });
        }
        let jh = multibody.insert(handles[parent], handles[i], joint, true);
        if let (Some(jh), Some(&q0)) = (jh, init_pose.get(jname.as_str())) {
            if q0 != 0.0 {
                init_joint_handles.push((jh, q0));
            }
        }
        mb_link_of_mjcf.insert(i, next_mb_link);
        name_to_link.insert(jname.clone(), next_mb_link);
        next_mb_link += 1;
    }

    // BIPED_INIT_POSE part 2: seed the multibody's GENERALIZED COORDINATES to
    // match the injected body poses. Joint insertion always starts every
    // coordinate at zero regardless of where the bodies sit, and the solver
    // runs FK from coordinates — without this the whole robot snaps back to
    // the neutral pose on step 1 (straight legs punch ~10 cm into the ground
    // and the depenetration launches it airborne).
    if let Some(&(h0, _)) = init_joint_handles.first() {
        let targets: Vec<(usize, f32)> = init_joint_handles
            .iter()
            .filter_map(|&(h, q0)| {
                let (mb, link_id) = multibody.get(h)?;
                Some((mb.link(link_id)?.assembly_id(), q0))
            })
            .collect();
        if let Some((mb, _)) = multibody.get_mut(h0) {
            let mut disp = vec![0.0f32; mb.ndofs()];
            for (aid, q0) in targets {
                disp[aid] = q0;
            }
            mb.apply_displacements(&disp);
            mb.forward_kinematics(&bodies, true);
        }
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
    let env_f32 = |k: &str| env_or_override(k).and_then(|s| s.parse::<f32>().ok());
    let mut sp = RbdSimParams::default();
    sp.dt = task_dt;
    sp.num_solver_iterations = env_or_override("BIPED_SOLVER_ITERS")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(SOLVER_ITERS);
    sp.contact_natural_frequency =
        env_f32("BIPED_CONTACT_NF").unwrap_or_else(|| zealot_env::knobs::CONTACT_NF.get());
    sp.contact_damping_ratio = env_f32("BIPED_CONTACT_DR").unwrap_or_else(|| zealot_env::knobs::CONTACT_DR.get());
    // Penetration-recovery knobs (A/B vs the Rapier game defaults 1mm/10m/s).
    // MuJoCo has NO velocity-level depenetration (critically-damped soft constraint
    // only); Isaac clamps max_depenetration_velocity to ~1 m/s in RL configs. The
    // 10 m/s default converts a 2mm spawn overlap into a ~0.6 m/s whole-robot
    // launch (~16 J from nothing) — see the 2026-07-27 pop investigation.
    sp.normalized_allowed_linear_error =
        env_f32("BIPED_ALLOWED_LIN_ERR").unwrap_or(sp.normalized_allowed_linear_error);
    // Contact prediction distance (speculative-contact margin, rapier default
    // 2mm). With the stiff normal holding equilibrium penetration ~0, a box
    // foot tilted a fraction of a degree lifts its far-edge corners past 2mm
    // and the manifold collapses to ONE EDGE (zero pitch moment capacity ->
    // the foot rocks). PhysX ships contactOffset ~2cm for exactly this.
    sp.normalized_prediction_distance =
        env_f32("BIPED_PREDICTION").unwrap_or(sp.normalized_prediction_distance);
    sp.normalized_max_corrective_velocity =
        env_f32("BIPED_MAX_CORR_VEL").unwrap_or(0.2);

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
        chest_link: link_of_name("torso_link").unwrap_or(0),
        foot_links,
        illegal_ground_links,
        self_collision_pairs,
        actuated,
        joint_dof_offset,
        foot_sole_local,
        mjcf_to_link,
        actuated_parent_links,
        actuated_rest_quat,
        held: held_joints,
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
/// BIPED_TERRAIN=1 state: the four family strips + per-env curriculum
/// (WBC-AGILE's terrain_levels_vel_curriculum — see zealot_env::terrain).
struct TerrainSetup {
    strips: [TerrainStrip; 4],
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
        // Route through of_env() so BIPED_TERRAIN_FAMILY affects the STRIP,
        // not just the family label. Indexing by env % 4 directly meant the
        // forced-family eval walked the wrong terrain: env 0 got the Boxes
        // strip while every log said "step" -- the negative cell heights in a
        // dumped patch (impossible for the 0-or-rise step field) were the
        // tell, and the "riser climb" in the first step videos was actually
        // the Boxes family's pyramid slope.
        &self.strips[Self::family_index(env)]
    }

    fn family_index(env: usize) -> usize {
        match TerrainFamily::of_env(env) {
            TerrainFamily::Boxes => 0,
            TerrainFamily::Rough => 1,
            TerrainFamily::Wave => 2,
            TerrainFamily::Step => 3,
        }
    }

    /// Oracle for the FOOT-CONTACT PROBE the real robot will run: walk a ray
    /// forward along the heading and report the first discrete height change.
    ///
    /// This deliberately models what a probe can actually measure -- distance
    /// to an edge and the height beyond it -- not what the simulator knows. It
    /// detects an edge in ANY family, not just Step, so the policy cannot use
    /// "cue present" as a proxy for "this is the step family".
    ///
    /// `EDGE_MIN` is the smallest height change worth reporting: below it the
    /// feature is terrain roughness the gait already absorbs, and a probe
    /// would not reliably resolve it either.
    fn probe(&self, env: usize, x: f32, y: f32, yaw: f32) -> zealot_env::tasks::velocity_flat::StepCue {
        use zealot_env::tasks::velocity_flat::StepCue;
        const EDGE_MIN: f32 = 0.04;
        const RANGE: f32 = 1.5;
        const DS: f32 = 0.05;
        let strip = self.strip_for(env);
        let (cx, sy) = (yaw.cos(), yaw.sin());
        let h0 = strip.height(x, y);
        let mut prev = h0;
        let mut d = DS;
        while d <= RANGE {
            let (px, py) = (x + cx * d, y + sy * d);
            let h = strip.height(px, py);
            if (h - prev).abs() > EDGE_MIN {
                // Edge ORIENTATION from the height gradient at the crossing.
                // The gradient points across the edge (up-slope), which is
                // exactly the edge normal; a camera's edge extraction recovers
                // the same direction from the depth discontinuity.
                let e = 0.10;
                let gx = strip.height(px + e, py) - strip.height(px - e, py);
                let gy = strip.height(px, py + e) - strip.height(px, py - e);
                let n = (gx * gx + gy * gy).sqrt();
                // Degenerate gradient (edge exactly on the sample grid) -> fall
                // back to "square-on", which is what a detector would also
                // report when it cannot resolve the orientation.
                let (nx, ny) = if n > 1e-6 { (gx / n, gy / n) } else { (cx, sy) };
                // Rotate the world-frame normal into the body frame.
                let ec = nx * cx + ny * sy;
                let es = -nx * sy + ny * cx;
                return StepCue {
                    distance: d,
                    height: h - h0,
                    edge_sin: es,
                    edge_cos: ec,
                    valid: 1.0,
                };
            }
            prev = h;
            d += DS;
        }
        StepCue::default()
    }
}

/// Gait cycle seconds at the slowest walking command (0.1 m/s).
const GAIT_PERIOD_SLOW: f32 = 0.8;
/// Gait cycle seconds at the full 0.5 m/s command.
const GAIT_PERIOD_FAST: f32 = 0.55;
/// Floor on the cycle time when the cap is raised past 0.5 m/s: the
/// slow->fast interpolation is linear and would extrapolate to 0.36 s at
/// 0.8 m/s, which is a sprint cadence this robot has never walked at.
/// 0.40 s at 0.8 m/s is a ~0.16 m step, which is in family with the
/// 0.14 m step the 0.5 m/s command already uses.
const GAIT_PERIOD_MIN: f32 = 0.40;

/// AMASS/SONIC upper-body playback config (see the `arm_motion` field).
struct ArmMotionCfg {
    clips: Vec<zealot_env::motion::MotionClip>,
    /// P(a command window plays a clip) — BIPED_ARM_MOTION_P, default 0.7:
    /// rolled on EVERY command resample (standing and walking alike), so most
    /// windows get moving arms but quiet-arm ones stay in the curriculum.
    p: f32,
    /// Amplitude blend home→clip — BIPED_ARM_MOTION_SCALE, default 1.0.
    /// <1 attenuates the retargeted motion toward `held_home` (curriculum
    /// knob if full-amplitude dance clips topple everything early on).
    scale: f32,
}


/// Per-step-constant reward / termination knobs, resolved ONCE.
///
/// These were re-read from the process environment on EVERY control step —
/// ~500 `getenv` calls per training iteration, each a lock plus a linear scan
/// of `environ`. They are also exactly the POD uniform a GPU reward kernel
/// needs, so gathering them is the first step of that port.
///
/// Process env cannot change mid-run, so resolving once is behaviour-identical.
#[derive(Clone, Copy, Debug)]
pub struct StepKnobs {
    pub illegal_z: f32,
    pub sc_margin: f32,
    pub sc_weight: f32,
    pub sc_term: f32,
    pub vel_term: f32,
    pub power_term: f32,
    pub env_term: f32,
    pub dwell_max: u16,
    pub slam_vel: f32,
    pub joint_power_term: f32,
    pub ankle_torque_w: f32,
    pub w_knee_torques: f32,
    pub power_w: f32,
    pub chest_w: f32,
    pub graph: bool,
    pub substep_trace: bool,
}

impl StepKnobs {
    fn from_env() -> Self {
        fn f(name: &str, dflt: f32) -> f32 {
            env_var(name).ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(dflt)
        }
        Self {
            illegal_z: f("BIPED_ILLEGAL_Z", 0.0),
            sc_margin: f("BIPED_SELF_COLL_DIST", 0.08),
            sc_weight: f("BIPED_SELF_COLL_W", 5.0),
            sc_term: f("BIPED_SELF_COLL_TERM", 0.05),
            vel_term: f("BIPED_DOF_VEL_TERM", 1.0),
            power_term: f("BIPED_POWER_TERM", 3000.0),
            env_term: f("BIPED_ENVELOPE_TERM", 0.0),
            dwell_max: env_var("BIPED_LIMIT_DWELL_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(40u16),
            slam_vel: f("BIPED_LIMIT_SLAM_VEL", 2.0),
            joint_power_term: f("BIPED_JOINT_POWER_TERM", 1500.0),
            ankle_torque_w: f("BIPED_ANKLE_TORQUE_W", 4.0),
            w_knee_torques: f("BIPED_W_KNEE_TORQUES", 7e-5),
            // MECH_POWER_W wins; POWER_W is the legacy alias.
            power_w: env_var("BIPED_MECH_POWER_W")
                .ok()
                .or_else(|| env_var("BIPED_POWER_W").ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(1e-4),
            chest_w: f("BIPED_CHEST_ANGVEL_W", 0.0),
            graph: env_var("BIPED_GRAPH").map(|v| v != "0").unwrap_or(true),
            substep_trace: env_var("BIPED_SUBSTEP_TRACE").is_ok(),
        }
    }
}

/// Reward-component indices the GPU kernel currently owns, paired with the
/// kernel output row that carries each. See [`zealot_gpu_obs::GpuRewardTerms`].
const GPU_REWARD_TERMS: [(usize, usize); 3] = [(6, 0), (7, 1), (29, 2)];

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
    /// Action before prev — third point for action_rate_rate's second difference.
    prev2_action: Vec<[f32; NUM_JOINTS]>,
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
    /// BIPED_CONTACT_SENSE=1: foot contact comes from the solver's summed
    /// normal-contact impulse per foot (nexus contact force sensor) instead of
    /// the geometric foot-height proxy. AGILE-parity contact semantics: a foot
    /// counts as planted only when it actually bears load.
    contact_sense: bool,
    /// Contact-force threshold, N (BIPED_CONTACT_FORCE_N, default 1.0 =
    /// AGILE's `feet_slip` contact_threshold). Compared against
    /// `sensed_force`.
    contact_force_n: f32,
    /// Impulse → force conversion: 1 / substep dt' (the sensor reads the last
    /// substep's converged normal impulse).
    sensor_inv_dt: f32,
    /// Per-env per-foot sensed normal force, N, from the last physics step of
    /// the control tick. Seeded to "planted" (half body weight) on reset so
    /// the spawn pose reads as standing before the first step's readback.
    sensed_force: Vec<[f32; NUM_FEET]>,
    /// Previous control step's `sensed_force` (same units), for the
    /// `force_rate` ground-reaction-smoothness reward. `has_prev_force` gates
    /// the first post-reset step so the seeded value never fabricates a ΔF.
    prev_sensed_force: Vec<[f32; NUM_FEET]>,
    has_prev_force: Vec<bool>,
    /// Index of the foot that most recently touched down, per env (-1 = none yet,
    /// reset on episode reset). Drives `FootObs.alt_step`: a touchdown only counts
    /// as a step if it's the OTHER foot than this, enforcing L→R→L→R alternation.
    last_td_foot: Vec<i8>,
    /// Gait-clock phase ∈ [0,1) per env, advanced by `control_dt / gait_period`
    /// each step (wraps at 1), reset to 0 on episode reset. Fed to the policy as
    /// (sin,cos) and used by the periodic gait reward to prescribe swing/stance.
    gait_phase: Vec<f32>,
    /// Speed at which the gait clock reaches its fastest cadence
    /// (`BIPED_GAIT_SPEED_CAP`, default 0.5 = historical). Raise alongside
    /// BIPED_VX or higher commands all share one cadence.
    gait_speed_cap: f32,
    /// Trainer-installed dropout override for the actor's step cue (annealed
    /// exploration schedule). None = the BIPED_STEP_CUE_DROPOUT env default.
    step_cue_dropout_override: Option<f32>,
    /// AMASS/SONIC upper-body playback (BIPED_ARM_MOTION=<clip dir>): under
    /// any command — standing or walking — the held waist+arm joints replay a
    /// random clip window as an unobserved moving-mass disturbance, and the
    /// legs must balance under it. None = off (bit-identical staging to before).
    arm_motion: Option<ArmMotionCfg>,
    /// Dedicated RNG stream (clip choice / start offset / activation roll) so
    /// enabling playback leaves the command/DR stream untouched.
    arm_rng: Vec<Lcg>,
    /// Per-env clip index / playback position (s) / active flag.
    arm_clip: Vec<u32>,
    arm_time: Vec<f32>,
    arm_active: Vec<bool>,
    /// Crossfade progress ∈ [0,1] from `arm_from` toward the current
    /// destination (clip pose or home), ramped over ~0.5 s. Reset to 0 on
    /// EVERY destination switch — activation, deactivation, AND a command
    /// resample that draws a fresh clip window — so the held
    /// PD targets never see a step input between unrelated poses.
    arm_blend: Vec<f32>,
    /// Externally-driven held-joint targets (VR teleop / web demo): when
    /// `Some`, they replace clip-or-home as the staging destination for
    /// EVERY env until cleared. Length = `idx.held.len()`, staging order.
    live_arm: Option<Vec<f32>>,
    /// Staging steps left after a live clear, so the fade back home
    /// completes even with no clips configured.
    live_fadeout: u32,
    /// Crossfade source: the held targets staged at the moment of the last
    /// destination switch (`env * held.len() + j`).
    arm_from: Vec<f32>,
    /// Scratch: one sampled pose (idx.held.len() radians).
    arm_scratch: Vec<f32>,
    /// Last staged held-joint targets (`env * held.len() + j`) — mirrored into
    /// the motor-delay state's held-link slots so the GPU delay kernel swaps
    /// prev==current (identity) for held links instead of a zero from the
    /// buffer default.
    arm_staged: Vec<f32>,
    /// Consecutive control steps each joint has spent inside its position-limit
    /// band, per env (`env * NUM_JOINTS + joint`). Atomic so the parallel
    /// per-env step closure can update its own entries. Drives the
    /// endstop-DWELL termination: touching a limit is normal in every gait
    /// (54-66% of frames), but LEANING on one is the standing pathology
    /// (median dwell 19 steps standing vs 9 walking).
    limit_dwell: Vec<std::sync::atomic::AtomicU16>,

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
    /// replaying via `cuGraphLaunch` removes it. DEFAULT ON; `BIPED_GRAPH=0`
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

    /// Reusable row-major `[NUM_JOINTS × n]` motor-target staging, uploaded to
    /// `motor_targets_gpu` and scattered by a kernel each control step.
    /// Reward / termination knobs, resolved once (see [`StepKnobs`]).
    knobs: StepKnobs,
    /// GPU reward: the ported terms + their staging. `None` until first use.
    /// Verified against the host terms with `BIPED_VERIFY_REWARD=1`.
    gpu_reward: Option<zealot_gpu_obs::GpuRewardTerms>,
    /// GPU joint state (q, qd), verified against `read_state_from_poses`.
    gpu_joints: Option<zealot_gpu_obs::GpuJointState>,
    /// Joint-only reward terms (pose / dof_pos_limits / dof_vel).
    gpu_joint_terms: Option<zealot_gpu_obs::GpuRewardJointTerms>,
    /// Torque / power reward terms.
    gpu_torque_terms: Option<zealot_gpu_obs::GpuRewardTorqueTerms>,
    /// Base pose / velocities / height.
    gpu_base: Option<zealot_gpu_obs::GpuBaseState>,
    /// Base-state reward terms.
    gpu_base_terms: Option<zealot_gpu_obs::GpuRewardBaseTerms>,
    /// Per-foot state.
    gpu_feet: Option<zealot_gpu_obs::GpuFeetState>,
    /// GPU assembly of the actor + critic observation frames. Verified with
    /// `BIPED_VERIFY_REWARD=1` against the host `observe`/`observe_critic`.
    gpu_observe: Option<zealot_gpu_obs::GpuObserve>,
    /// `BIPED_VERIFY_CUE`: feed a synthetic step cue so the obs harness
    /// exercises the cue gate + clamp, which a flat-curriculum run never hits.
    /// TEST FIXTURE ONLY — it overwrites the real cue.
    verify_cue: bool,
    /// `BIPED_SKIP_REWARD`: skip the host reward evaluation to price it. Gives
    /// WRONG training; measurement only.
    skip_reward: bool,
    /// `BIPED_GPU_REWARD=1`: consume the GPU reward terms as the source of
    /// truth — the host term math is skipped and `comps` are filled from the
    /// device stack (one fused encode + submit per step, term-matrix
    /// readbacks only). The host keeps state assembly, obs, the step cue,
    /// termination detection and feet bookkeeping. Ignored under
    /// `BIPED_VERIFY_REWARD`, which needs the host values to compare against.
    use_gpu_reward: bool,
    /// `BIPED_SKIP_OBS`: skip the host obs assembly to price it. Wrong
    /// training; measurement only.
    skip_obs: bool,
    /// Self-contained per-foot reward terms.
    gpu_feet_terms: Option<zealot_gpu_obs::GpuRewardFeetTerms>,
    /// Gated gait reward terms.
    gpu_gait_terms: Option<zealot_gpu_obs::GpuRewardGaitTerms>,
    /// self_coll / chest_ang_vel / termination.
    gpu_misc_terms: Option<zealot_gpu_obs::GpuRewardMiscTerms>,
    targets_row: Vec<f32>,
    /// Device copy of `targets_row`; persistent so the per-step upload is one
    /// small `write_buffer` instead of a fresh allocation.
    motor_targets_gpu: Option<vortx::tensor::Tensor<f32>>,
    /// Set when `stage_arm_motion` changed HELD-joint entries of the mirror.
    /// Those live in `links_static` alongside the actuated ones, and the motor
    /// scatter only rewrites the actuated entries — so a full mirror upload is
    /// still required on exactly those steps.
    held_dirty: bool,

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
pub const NUM_REWARD_COMPS: usize = 32;

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
    "stand_planted", // per-airborne-foot penalty at standing command (balance, don't step)
    "feet_yaw_diff", // WBC feet_yaw_diff_l2: L/R foot yaw splay penalty
    "force_rate",    // ground-reaction smoothness: |ΔF| above deadband, squared (slam + tremor)
    "action_rate_rate", // action 2nd difference — tremor at the source (engine-agnostic)
    "touchdown_vz",  // kinematic slam: descent speed near ground, above allowance
    "chest_ang_vel", // chest-link roll/pitch rate penalty (BIPED_CHEST_ANGVEL_W)
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
        if env_var("BIPED_FOOT_SHAPE").as_deref() == Ok("convex") {
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
        {
            let d = zealot_env::knobs::DECIMATION.get();
            task.decimation = d;
            task.sim_dt = 0.02 / d as f32;
        }
        // Gait-cadence knobs. The left/right alternation itself comes from
        // the gait clock (foot 1 runs half a cycle behind foot 0); its
        // PERIOD is BIPED_GAIT_PERIOD (read below — larger = slower,
        // lower-frequency weight transfer). These two shape how hard the
        // policy locks to that clock and the swing/stance split:
        task.weights.gait_clock = env_var("BIPED_GAIT_CLOCK_W")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(3.0);
        if let Some(sr) = env_var("BIPED_GAIT_SWING_RATIO")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.gait_swing_ratio = sr;
        }
        // CoM-over-support-foot reward weight. Long single-support holds
        // (slow gait clocks) are only cheap for the fragile ankles if the
        // CoM rides over the stance foot — raise this together with
        // BIPED_GAIT_PERIOD.
        task.weights.feet_distance = std::env::var("BIPED_W_FEET_DISTANCE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-1.0);
        // Foot-clearance weight. DEFAULT 0 -- this reward was dropped once for
        // being farmed into a one-foot statue (foot held up to collect it; zero
        // transfer, MuJoCo fell in 0.66 s). The guards that make it safe now
        // live in the reward (swing-only, air_time < 0.45 s so a held foot
        // stops earning, moving-gated, capped at the target), and the target
        // scales with a cued step riser -- but enable it deliberately, and
        // watch for the statue signature: one foot held, air_time saturating,
        // clearance high, travel low.
        task.weights.foot_clearance = std::env::var("BIPED_W_FOOT_CLEARANCE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);
        // Target for that reward (m). Overridden upward automatically while a
        // step is cued; this is the flat-ground value.
        if let Some(t) = std::env::var("BIPED_FOOT_CLEARANCE_TARGET")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.foot_clearance_target = t;
        }
        task.weights.foot_orientation = std::env::var("BIPED_W_FOOT_ORIENTATION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-0.5);
        task.weights.stand_planted = std::env::var("BIPED_W_STAND_PLANTED")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-1.0);
        // Ground-reaction-smoothness (force-rate) penalty: pass the weight
        // NEGATIVE (it's a penalty). Deadband in body weights per control
        // step. Requires BIPED_CONTACT_SENSE=1 — without the sensor the ΔF
        // input is identically 0 and the term is silently inert.
        if let Some(w) = std::env::var("BIPED_W_FORCE_RATE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.force_rate = w;
        }
        if let Some(d) = std::env::var("BIPED_FORCE_RATE_DB")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.force_rate_deadband = d;
        }
        // Action-side slam/tremor pair (the engine-agnostic replacement for
        // the retired sensor-side force_rate): both weights NEGATIVE.
        // -0.3 (was -0.1): v28's measured gait carried 68.7% of its knee
        // action power above 5 Hz (knee target slew p95 282 mrad/20ms) — at
        // -0.1 chatter cost ~14% of income, a tax the policy happily paid.
        // 3x makes tremor a first-order cost so the policy LEARNS smoothness
        // instead of needing a deploy-side low-pass (S2S_ACT_LPF measured the
        // headroom: filtering to 2% high-freq power cost only 4% travel).
        // Verify on the next run with S2S_TRACE=1; escalate to -0.5 or
        // train-with-filter if high-freq power stays >5%.
        task.weights.action_rate_rate = std::env::var("BIPED_W_ACTION_RATE_RATE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-0.3);
        task.weights.touchdown_vz = std::env::var("BIPED_W_TOUCHDOWN_VZ")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-1.0);
        if let Some(v) = std::env::var("BIPED_TOUCHDOWN_VZ_OK")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.touchdown_vz_ok = v;
        }
        if let Some(v) = std::env::var("BIPED_TOUCHDOWN_VZ_H")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            task.weights.touchdown_vz_h = v;
        }
        // Soft joint-limit penalty weight. At the spec default (-0.5) a fully
        // saturated ankle costs ~0.0009/step per joint — 3-4x below the
        // ~0.002/step where a term changes behaviour, which is why the policy
        // parks on the endstop and lets the constraint carry ~97% of the load.
        task.weights.dof_pos_limits = std::env::var("BIPED_W_DOF_LIMITS")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(-5.0);
        // Balance-don't-step at stand: per-airborne-foot penalty while the
        // command is standing (NEGATIVE, e.g. -1.0; 0 = off). Pair with a
        // raised BIPED_STAND_PROB so the policy actually trains the quiet
        // stance, and with pushes on so it learns the ankle/hip strategy.
        if let Some(w) = env_var("BIPED_STAND_PLANTED_W")
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
        if zealot_env::knobs::CONTACT_REDUCE.get() {
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
        if let Some(f) = env_var("BIPED_FRICTION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            template_dr[0].friction = f;
        }

        // BIPED_TERRAIN=1: generate the four family strips once and wrap each
        // in ONE SharedShape (cloned across that family's envs so nexus dedupes
        // the mesh buffers to 3 uploads). ORIENTED pseudo-normals are required
        // by the nexus trimesh contact path; the strips are closed slabs.
        let terrain_on = zealot_env::knobs::TERRAIN.get();
        let terrain_build = if terrain_on {
            let t0 = Instant::now();
            // BIPED_TERRAIN_AMP (amplitude multiplier, default 1) and
            // BIPED_TERRAIN_SLOPE_DEG (uphill grade along +X, default 0):
            // demo-facing shape knobs; defaults reproduce training terrain.
            let params = zealot_env::terrain::TerrainParams {
                amp: env_var("BIPED_TERRAIN_AMP")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1.0),
                slope: env_var("BIPED_TERRAIN_SLOPE_DEG")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .map_or(0.0, |deg| deg.clamp(0.0, 45.0).to_radians().tan()),
            };
            let strips = [
                TerrainStrip::generate_with(TerrainFamily::Boxes, seed, params),
                TerrainStrip::generate_with(TerrainFamily::Rough, seed, params),
                TerrainStrip::generate_with(TerrainFamily::Wave, seed, params),
                TerrainStrip::generate_with(TerrainFamily::Step, seed, params),
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
            let step_in_rotation = std::env::var("BIPED_TERRAIN_STEP")
                .map(|v| v != "0")
                .unwrap_or(true);
            println!(
                "terrain curriculum ENABLED: {} family strips ({} rows x {} m patches), built in {:.1}s{}",
                if step_in_rotation { 4 } else { 3 },
                zealot_env::terrain::ROWS,
                zealot_env::terrain::PATCH,
                t0.elapsed().as_secs_f64(),
                if step_in_rotation { "" } else { " [Step parked: BIPED_TERRAIN_STEP=0]" }
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
            let tshape = terrain_build.as_ref().map(|(_, shapes, _)| &shapes[TerrainSetup::family_index(e)]);
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
                collisions_capacity: 128,
                ..Default::default()
            },
        );
        state.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);
        // BIPED_CONTACT_CAP: eagerly pre-size the contact/constraint buffers
        // (per batch). Required before BIPED_GRAPH capture on terrain — the
        // lazy in-step resize can't run once a CUDA graph is captured, and
        // overflowing pairs are silently dropped (feet sink into the mesh).
        {
            let cap = zealot_env::knobs::CONTACT_CAP.get();
            state.reserve_contacts(&gpu, cap);
            println!("contact buffers pre-sized to {cap}/batch");
        }
        // BIPED_CONTACT_SENSE=1: force-sensed foot contact. The nexus MB
        // solver folds each foot link's solved NORMAL-constraint impulses into
        // a tiny per-env buffer at the end of every step (last substep, post-
        // stabilization); we read it back alongside body_poses and gate every
        // contact-derived gait signal on measured load instead of the
        // foot-height proxy (which can't tell a planted foot from one hovering
        // just under the threshold). Threshold BIPED_CONTACT_FORCE_N (default
        // 1.0 N = AGILE's feet_slip contact_threshold).
        let contact_sense = zealot_env::knobs::CONTACT_SENSE.get();
        let contact_force_n = env_var("BIPED_CONTACT_FORCE_N")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);
        let solver_iters = env_var("BIPED_SOLVER_ITERS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(SOLVER_ITERS);
        // Impulse→force divisor. On the upstream base the DEFAULT
        // explicit-coriolis mode builds contact constraints ONCE per step, so
        // the sensed impulse is accumulated over the whole physics step →
        // divide by sim_dt. Implicit-coriolis rebuilds per substep → the
        // readout is the last substep's impulse → divide by the substep dt.
        let implicit_coriolis = env_var("BIPED_IMPLICIT_CORIOLIS").as_deref() == Ok("1");
        let sensor_inv_dt = if implicit_coriolis {
            solver_iters as f32 / task.sim_dt
        } else {
            1.0 / task.sim_dt
        };
        if contact_sense {
            let mbs = state.multibodies_mut();
            let mut links = [0u32; NUM_FEET];
            for (i, l) in links.iter_mut().enumerate() {
                let bl = mbs.link_of_body(0, idx.foot_links[i]);
                assert!(
                    bl[1] != u32::MAX,
                    "foot body {} is not a multibody link",
                    idx.foot_links[i]
                );
                *l = bl[1];
            }
            mbs.set_contact_sensor_links(&gpu, &links);
            println!(
                "contact sensing ENABLED: force-based foot contact (links {links:?}, \
                 threshold {contact_force_n} N)"
            );
        }
        // BIPED_MAX_COLORS: bound for the contact-graph coloring (nexus default
        // 8 → the solver runs max_colors+1 passes per phase whether or not the
        // colors are used). The biped scene's rigid-rigid contact graph is tiny
        // (dynamics live in the multibody solver), so the default mostly buys
        // empty solver dispatches. Under-provisioning is self-healing but bad:
        // the coloring-failed ratchet adds +5. Keep constant per run (graph
        // capture records the pass count).
        if let Some(mc) = env_var("BIPED_MAX_COLORS")
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
        let implicit_coriolis = env_var("BIPED_IMPLICIT_CORIOLIS")
            .map(|v| v != "0")
            .unwrap_or(false);
        state
            .multibodies_mut()
            .set_implicit_coriolis(implicit_coriolis);
        // Decomposed refresh probe: implicit mode's per-substep dynamics +
        // constraint rebuild cadence with the explicit (coriolis-free) kernels.
        let refresh_mode = zealot_env::knobs::SUBSTEP_REFRESH.get();
        state
            .multibodies_mut()
            .set_substep_refresh(refresh_mode == 1);
        // "2": light split-cadence — constraints per substep, M/LU per sim-step.
        state
            .multibodies_mut()
            .set_substep_refresh_light(refresh_mode == 2);

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
            let arm_scale: f32 = env_var("BIPED_ARM")
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
                collisions_capacity: 128,
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
        // Publish the templates to the GPU ONCE. Resets then reference them by
        // index, so a reset uploads only (env, template id, offset, velocities)
        // instead of re-sending ~10 KB of template state per env.
        {
            let refs: Vec<&RbdSnapshot> = template_snapshots.iter().collect();
            state.publish_reset_templates(&gpu, &refs);
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
        let prev2_action = vec![[0.0f32; NUM_JOINTS]; num_envs];
        // BIPED_MOTOR_DELAY=min,max (or just max → min=0), in physics
        // substeps. `0,0` is a valid ENABLED config (constant zero delay —
        // used by the staging-equivalence check); unset/unparseable = off.
        let motor_delay: Option<(u32, u32)> = env_var("BIPED_MOTOR_DELAY")
            .ok()
            .and_then(|s| {
                let p: Vec<u32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
                match p.as_slice() {
                    [max] => Some((0, *max)),
                    [min, max] => Some((*min, *max)),
                    _ => None,
                }
            })
            .or(Some((0, 4)));
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
        let limit_dwell: Vec<std::sync::atomic::AtomicU16> = (0..num_envs * NUM_JOINTS)
            .map(|_| std::sync::atomic::AtomicU16::new(0))
            .collect();
        let reset_vel = std::env::var("BIPED_RESET_VEL").map_or(true, |v| v != "0");
        if reset_vel {
            println!("reset-velocity randomization ENABLED (AGILE reset_base/joints: lin ±0.25, ang ±0.5, joints ±1.0)");
        }
        let push_vel = env_var("BIPED_PUSH_VEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let push_interval = env_var("BIPED_PUSH_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(175);
        let push_angvel = env_var("BIPED_PUSH_ANGVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25);
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
        // AMASS/SONIC upper-body playback (BIPED_ARM_MOTION=<dir of SONIC
        // csv clips>). Loaded against the model's held joints BY NAME, so a
        // clip column set that doesn't cover them fails here, not silently.
        let arm_motion = std::env::var("BIPED_ARM_MOTION").ok().map(|dir| {
            assert!(
                !idx.held.is_empty(),
                "BIPED_ARM_MOTION needs PD-held joints (unset BIPED_LOCK_HELD, use a 29dof model)"
            );
            let fps: f32 = std::env::var("BIPED_ARM_MOTION_FPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30.0);
            let names: Vec<String> = idx.held.iter().map(|h| h.name.clone()).collect();
            let clips = zealot_env::motion::MotionClip::load_dir(
                std::path::Path::new(&dir),
                &names,
                fps,
            )
            .expect("BIPED_ARM_MOTION");
            let p: f32 = std::env::var("BIPED_ARM_MOTION_P")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7);
            let scale: f32 = std::env::var("BIPED_ARM_MOTION_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let total_s: f32 = clips.iter().map(|c| c.duration()).sum();
            println!(
                "arm-motion playback ENABLED: {} clips ({:.0} s total) driving {} held joints, p={p}, scale={scale}, fps={fps}",
                clips.len(),
                total_s,
                idx.held.len()
            );
            ArmMotionCfg { clips, p, scale }
        });
        if arm_motion.is_none() {
            println!(
                "arm-motion playback DISABLED (BIPED_ARM_MOTION unset) — held joints hold the home pose"
            );
        }
        let arm_rng: Vec<Lcg> = (0..num_envs)
            .map(|e| Lcg::new(seed ^ ((e as u64).wrapping_mul(2654435761)) ^ 0xA53A))
            .collect();
        let n_held = idx.held.len();
        let held_homes: Vec<f32> = idx.held.iter().map(|h| h.home).collect();
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
            arm_motion,
            arm_rng,
            arm_clip: vec![0; num_envs],
            arm_time: vec![0.0; num_envs],
            arm_active: vec![false; num_envs],
            // Blend starts settled-at-home (1.0, staged == home): the spawn
            // pose holds home, so there is nothing to fade from.
            arm_blend: vec![1.0; num_envs],
            live_arm: None,
            live_fadeout: 0,
            arm_from: held_homes
                .iter()
                .copied()
                .cycle()
                .take(n_held * num_envs)
                .collect(),
            arm_scratch: vec![0.0; n_held],
            arm_staged: held_homes
                .iter()
                .copied()
                .cycle()
                .take(n_held * num_envs)
                .collect(),
            cmd,
            step_count,
            resample_at,
            last_action,
            prev_action,
            prev2_action,
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
                // BIPED_TERRAIN_INIT_LEVEL: fixed starting difficulty (demo
                // knob; default = AGILE's U{0, 1}). Promotion/demotion still
                // runs from there.
                let init_level: Option<u32> = env_var("BIPED_TERRAIN_INIT_LEVEL")
                    .ok()
                    .and_then(|s| s.parse().ok());
                let curriculum = rng
                    .iter_mut()
                    .map(|r| {
                        let mut c = TerrainCurriculum::init(r);
                        if let Some(l) = init_level {
                            c.level = l.min(zealot_env::terrain::ROWS as u32 - 1);
                        }
                        c
                    })
                    .collect();
                TerrainSetup {
                    strips,
                    curriculum,
                    rng,
                    travel: vec![0.0; num_envs],
                    last_xy: vec![[0.0, 0.0]; num_envs],
                }
            }),
            air_time,
            contact_sense,
            contact_force_n,
            sensor_inv_dt,
            sensed_force: vec![[0.5 * robot.total_mass * 9.81; NUM_FEET]; num_envs],
            prev_sensed_force: vec![[0.5 * robot.total_mass * 9.81; NUM_FEET]; num_envs],
            has_prev_force: vec![false; num_envs],
            last_td_foot,
            gait_phase,
            gait_speed_cap: std::env::var("BIPED_GAIT_SPEED_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.8),
            step_cue_dropout_override: None,
            limit_dwell,
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
            torque_scale: env_var("BIPED_TORQUE_W")
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
            knobs: StepKnobs::from_env(),
            gpu_reward: None,
            gpu_joints: None,
            gpu_joint_terms: None,
            gpu_torque_terms: None,
            gpu_base: None,
            gpu_base_terms: None,
            gpu_feet: None,
            gpu_observe: None,
            verify_cue: std::env::var("BIPED_VERIFY_CUE").is_ok(),
            skip_reward: std::env::var("BIPED_SKIP_REWARD").is_ok(),
            use_gpu_reward: std::env::var("BIPED_GPU_REWARD").is_ok()
                && std::env::var("BIPED_VERIFY_REWARD").is_err(),
            skip_obs: std::env::var("BIPED_SKIP_OBS").is_ok(),
            gpu_feet_terms: None,
            gpu_gait_terms: None,
            gpu_misc_terms: None,
            targets_row: vec![0.0; NUM_JOINTS * num_envs],
            motor_targets_gpu: None,
            held_dirty: false,
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
            env.cmd[e] = eval_cmd_override().unwrap_or_else(|| env.sampler.sample(&mut env.rng[e]));
            env.resample_at[e] = env
                .sampler
                .resample_steps(&mut env.rng[e], env.task.control_dt());
            env.arm_resample(e);
        }
        // BIPED_TERRAIN: teleport every env onto its initial-level patch (the
        // as-built state stands on flat ground at the origin). Uses the same
        // template each env was built from, so its DR sample is preserved.
        // Training is on-terrain from step 0, like AGILE.
        if env.terrain.is_some() {
            for e in 0..num_envs {
                let t = e % env.templates.len().max(1);
                let off = env.terrain_spawn_offset(e, t);
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
    fn terrain_spawn_offset(&mut self, e: usize, template: usize) -> Vec3 {
        // Step-family APPROACH MODE: with probability BIPED_STEP_APPROACH_P
        // (default 0.5), spawn the robot so its heading POINTS AT the edge from
        // a short standoff, instead of AGILE's uniform +/-2.5 m jitter (which,
        // with the edge through the patch centre, spawns the robot ON the edge
        // line with a random heading -- clean approach->cross->continue
        // sequences are rare accidents, and the curriculum can promote through
        // episodes that walked ALONG the edge).
        //
        // The reset API has no yaw control, so instead of turning the robot we
        // pick the spawn point: place it standoff metres BEHIND the patch
        // centre along its own baked heading (template_dr[t].spawn_yaw), so
        // walking forward crosses the edge. The heading is uniform across
        // templates and the edge normal uniform per patch, so approach angles
        // stay fully diverse -- head-on, oblique, up-side, down-side all occur.
        let approach_p: f32 = std::env::var("BIPED_STEP_APPROACH_P")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let yaw_t = self
            .template_dr
            .get(template)
            .map(|d| d.spawn_yaw)
            .unwrap_or(0.0);
        let ter = self.terrain.as_mut().expect("terrain on");
        let level = ter.curriculum[e].level;
        let (cx, cy) = TerrainStrip::patch_center(level);
        let rng = &mut ter.rng[e];
        let is_step = matches!(
            TerrainFamily::of_env(e),
            TerrainFamily::Step
        );
        let (sx, sy) = if is_step && rng.range(0.0, 1.0) < approach_p {
            let standoff = rng.range(1.0, 2.5);
            // Lateral jitter perpendicular to the heading, so the crossing
            // point sweeps along the edge rather than always hitting centre.
            let lat = rng.range(-1.5, 1.5);
            (
                cx - yaw_t.cos() * standoff - yaw_t.sin() * lat,
                cy - yaw_t.sin() * standoff + yaw_t.cos() * lat,
            )
        } else {
            (cx + rng.range(-2.5, 2.5), cy + rng.range(-2.5, 2.5))
        };
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

    /// Sample the terrain height field around `(cx, cy)` on a regular grid,
    /// for the offline renderer to draw the ground the robot actually walked
    /// on. Returns (half_extent, spacing, row-major heights). Zeros if terrain
    /// is off.
    pub fn terrain_patch_for(&self, e: usize, cx: f32, cy: f32, half: f32, hs: f32) -> (f32, f32, Vec<f32>) {
        let n = (2.0 * half / hs).round() as usize + 1;
        let mut out = vec![0.0f32; n * n];
        if let Some(t) = self.terrain.as_ref() {
            let strip = t.strip_for(e);
            for j in 0..n {
                for i in 0..n {
                    let x = cx - half + i as f32 * hs;
                    let y = cy - half + j as f32 * hs;
                    out[j * n + i] = strip.height(x, y);
                }
            }
        }
        (half, hs, out)
    }

    /// Install the annealed actor-cue dropout for this iteration (see the
    /// rollout's cue block). The critic's clean copy is unaffected.
    pub fn set_step_cue_dropout(&mut self, p: f32) {
        self.step_cue_dropout_override = Some(p.clamp(0.0, 1.0));
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
        unimplemented!("slurp_state: links_workspace is SoA (Vec4) on the upstream base — probe not ported")
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
        unimplemented!("debug_contact_impulses: constraint-count tensor + AoS workspace absent on the upstream base — probe not ported")
    }

    /// PHASE-A substep trace: read env0's foot-link world XY + per-foot normal
    /// impulse and emit one `[sub]` line. Called per `pipeline.step` inside the
    /// (non-graph) decimation loop when `BIPED_SUBSTEP_TRACE` is set. With
    /// `BIPED_SOLVER_ITERS=1` each pipeline.step is ONE substep, so this gives
    /// per-substep resolution of the foot contact-point trajectory — to isolate
    /// the exact substep a loaded foot flips from planted to sliding. Reuses the
    /// `debug_contact_impulses` readback pattern (links_workspace + contacts).
    pub async fn trace_foot_substep(&mut self, _gstep: u64, _sub: u32) {
        unimplemented!("trace_foot_substep: constraint-count tensor absent on the upstream base — probe not ported")
    }

    /// Inject a random velocity kick to every env's torso — a push perturbation,
    /// the GPU equivalent of Isaac's `push_by_setting_velocity`: ±push_vel m/s on
    /// the root's linear x/y DOFs and (when BIPED_PUSH_ANGVEL > 0) ±push_angvel
    /// rad/s on its angular x/y/z DOFs. The policy must re-establish balance over
    /// its feet after each shove, which is what makes the learned equilibrium
    /// ROBUST and engine-agnostic (sim-to-real) rather than a brittle
    /// nexus-specific reflex. Read-modify-write the generalized-velocity section of
    /// DOF index of each policy joint within an env's `dof_state` slice, in
    /// canonical `JOINT_NAMES` order (root DOFs occupy 0..6).
    pub fn policy_joint_dofs(&self) -> [u32; NUM_JOINTS] {
        self.idx.joint_dof_offset
    }

    /// TRUE generalized velocities for one env, straight from `dof_state`
    /// (no finite differencing): `[root_lin(3), root_ang(3), joints…]` in the
    /// multibody's DOF order, world frame for the root. Used by the
    /// cross-engine divergence probe, where FD velocity error at the 50 Hz
    /// control rate is the dominant noise floor.
    pub async fn true_dof_velocities(&mut self, env: usize) -> Vec<f32> {
        let dpb = self.state.multibodies_mut().dofs_per_batch_count() as usize;
        let n = self.n;
        let total = self.state.multibodies_mut().dof_state().buffer().len();
        let mut buf = vec![0.0f32; total];
        self.gpu
            .slow_read_buffer(self.state.multibodies_mut().dof_state().buffer(), &mut buf)
            .await
            .expect("dof_state readback");
        // Batch-interleaved: dof `d` of env `e` at `d·n + e`.
        (0..dpb).map(|d| buf[d * n + env]).collect()
    }

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
        // Upstream-base layout is BATCH-INTERLEAVED: dof `d` of env `e` at
        // `d·n + e` (fork layout was `e·dpb + d`).
        for e in 0..n {
            let dvx = self.rng[e].range(-pv, pv);
            let dvy = self.rng[e].range(-pv, pv);
            buf[e] += dvx; // root linear x velocity
            buf[n + e] += dvy; // root linear y velocity
            if pa > 0.0 {
                for d in 3..6 {
                    // root angular x/y/z velocity (world frame)
                    buf[d * n + e] += self.rng[e].range(-pa, pa);
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
                prev2_action: self.prev2_action[env],
                feet: [FootObs::default(); NUM_FEET],
                phase: 0.0, // overwritten with self.gait_phase[env] by the caller
                // Filled by the caller (needs the terrain + this env's pose).
                step_cue: Default::default(),
            step_cue_clean: Default::default(),
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
        let contact_z: f32 = env_var("BIPED_CONTACT_Z")
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
            let (planar_speed, vz) = if has_prev {
                let prev_pos = self.prev_body_poses[env_base + link].translation;
                let dx = (pos.x - prev_pos.x) / dt;
                let dy = (pos.y - prev_pos.y) / dt;
                ((dx * dx + dy * dy).sqrt(), (pos.z - prev_pos.z) / dt)
            } else {
                (0.0, 0.0)
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
            // BIPED_CONTACT_SENSE=1: measured load-bearing contact from the
            // solver's normal impulses (AGILE force-sensor semantics — a foot
            // skimming just under CONTACT_Z no longer reads as planted).
            // Default: geometric foot-height proxy.
            let contact = if self.contact_sense {
                self.sensed_force[env][i] >= self.contact_force_n
            } else {
                pos.z - foot_ground_h < CONTACT_Z
            };
            let prev_air = self.air_time[env][i];
            let first_contact = contact && prev_air > 0.0;
            // Alternating touchdown: a step that lands on the OTHER foot than the
            // last one to touch down (or the first step ever, last_td_foot == -1).
            let alt_step = first_contact && self.last_td_foot[env] != i as i8;
            new_air[i] = if contact { 0.0 } else { prev_air + dt };
            // ΔF in body weights per control step (force_rate reward). Zero
            // without the sensor (forces frozen at seed) or pre-first-read.
            let force_rate = if self.contact_sense && self.has_prev_force[env] {
                (self.sensed_force[env][i] - self.prev_sensed_force[env][i]).abs()
                    / (self.robot.total_mass * 9.81)
            } else {
                0.0
            };
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
                vz,
                force_rate,
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

    /// Roll the upper-body playback state for env `e`. Called wherever the
    /// command (re)samples: every fresh command — standing OR walking — rolls
    /// `chance(p)` for a new clip window, so the legs learn to balance under
    /// arm motion in both regimes; a losing roll blends the arms back home.
    fn arm_resample(&mut self, e: usize) {
        let Some(am) = self.arm_motion.as_ref() else { return };
        let (n_clips, p) = (am.clips.len(), am.p);
        let n_held = self.idx.held.len();
        if self.arm_rng[e].chance(p) {
            let c = ((self.arm_rng[e].unit() * n_clips as f32) as usize).min(n_clips - 1);
            // Random start with runway: uniformly inside the clip minus the
            // longest plausible command dwell (a finished clip freezes on its
            // last frame — legal, just less motion than intended).
            let dur = self.arm_motion.as_ref().unwrap().clips[c].duration();
            let t0 = self.arm_rng[e].range(0.0, (dur - 8.0).max(0.0));
            self.arm_clip[e] = c as u32;
            self.arm_time[e] = t0;
            self.arm_active[e] = true;
            // New destination (fresh clip window — even stand→stand):
            // crossfade from wherever the arms are RIGHT NOW, never jump.
            self.arm_from[e * n_held..(e + 1) * n_held]
                .copy_from_slice(&self.arm_staged[e * n_held..(e + 1) * n_held]);
            self.arm_blend[e] = 0.0;
        } else if self.arm_active[e] {
            // Deactivating: crossfade back home from the current pose. (An
            // already-home env stays settled — no pointless re-blend.)
            self.arm_active[e] = false;
            self.arm_from[e * n_held..(e + 1) * n_held]
                .copy_from_slice(&self.arm_staged[e * n_held..(e + 1) * n_held]);
            self.arm_blend[e] = 0.0;
        }
    }

    /// Kill playback INSTANTLY (no blend-out): episode resets respawn the
    /// held joints at the home pose, so a lingering blend would drag the
    /// fresh episode's arms through a stale clip pose.
    fn arm_reset(&mut self, e: usize) {
        if self.arm_motion.is_none() && self.live_arm.is_none() {
            return;
        }
        let n_held = self.idx.held.len();
        self.arm_active[e] = false;
        // Settled at home (nothing to fade from) — unless live targets are
        // installed, in which case fade home → live instead of jumping.
        self.arm_blend[e] = if self.live_arm.is_some() { 0.0 } else { 1.0 };
        for (j, h) in self.idx.held.iter().enumerate() {
            self.arm_staged[e * n_held + j] = h.home;
            self.arm_from[e * n_held + j] = h.home;
        }
    }

    /// Advance every env's playback one control step and restage the held
    /// joints' PD targets in the `links_static` mirror (uploaded by the same
    /// `flush_links_static` that carries the leg targets). No-op when
    /// BIPED_ARM_MOTION is unset and no live targets are installed —
    /// staging stays bit-identical to before.
    fn stage_arm_motion(&mut self) {
        let dt = self.task.control_dt();
        if self.arm_motion.is_none() && self.live_arm.is_none() && self.live_fadeout == 0 {
            return;
        }
        // Held-joint entries of the mirror are about to change; the motor
        // scatter only rewrites ACTUATED entries, so this step needs the full
        // links_static upload as well.
        self.held_dirty = true;
        if self.live_arm.is_none() && self.live_fadeout > 0 {
            self.live_fadeout -= 1;
        }
        let clip_scale = self.arm_motion.as_ref().map(|am| am.scale).unwrap_or(1.0);
        let n_held = self.idx.held.len();
        let mbs = self.state.multibodies_mut();
        for e in 0..self.n {
            // Crossfade `arm_from` → destination over ~0.5 s (smoothstep, so
            // the fade starts and ends with zero velocity). The destination
            // is the LIVE external target, the clip pose (already moving
            // during the fade), or home; `arm_from` was snapshotted at the
            // last destination switch, so every transition — activation,
            // deactivation, resample clip swap, live install/clear —
            // leaves from wherever the arms currently are.
            let b = (self.arm_blend[e] + dt / 0.5).min(1.0);
            self.arm_blend[e] = b;
            let s = b * b * (3.0 - 2.0 * b);
            if self.arm_active[e] {
                self.arm_time[e] += dt;
                if let Some(am) = &self.arm_motion {
                    let clip = &am.clips[self.arm_clip[e] as usize];
                    clip.sample(self.arm_time[e], &mut self.arm_scratch);
                }
            }
            let live = self.live_arm.as_deref();
            for (j, h) in self.idx.held.iter().enumerate() {
                // Destination: live external target, or amplitude-scaled clip
                // pose (or home) — always clamped into the joint's mechanical
                // range (retargeted mocap/VR can exceed it).
                let dest = if let Some(lv) = live {
                    lv[j].clamp(h.range.0, h.range.1)
                } else if self.arm_active[e] {
                    (h.home + clip_scale * (self.arm_scratch[j] - h.home))
                        .clamp(h.range.0, h.range.1)
                } else {
                    h.home
                };
                let q = if s >= 1.0 {
                    dest
                } else {
                    let from = self.arm_from[e * n_held + j];
                    from + s * (dest - from)
                };
                mbs.stage_motor_position(e as u32, h.link, JointAxis::AngZ, q);
                self.arm_staged[e * n_held + j] = q;
            }
        }
    }

    /// Names of the PD-held (non-action) joints in staging order — the key
    /// for mapping external (VR) targets onto `set_live_arm_targets`.
    pub fn held_joint_names(&self) -> Vec<String> {
        self.idx.held.iter().map(|h| h.name.clone()).collect()
    }

    /// Hold-pose targets of the held joints, same order as
    /// `held_joint_names()` — the fallback for joints an external source
    /// doesn't provide.
    pub fn held_joint_homes(&self) -> Vec<f32> {
        self.idx.held.iter().map(|h| h.home).collect()
    }

    /// Install externally-driven held-joint targets (VR teleop): they
    /// override clip/home as the staging destination for EVERY env until
    /// cleared, crossfading from the current pose on install (same ~0.5 s
    /// smoothstep as clip transitions). Length must match
    /// `held_joint_names()`; subsequent calls update in place (no re-fade).
    pub fn set_live_arm_targets(&mut self, targets: &[f32]) {
        let n_held = self.idx.held.len();
        assert_eq!(targets.len(), n_held, "expected one target per held joint");
        if self.live_arm.is_none() {
            for e in 0..self.n {
                self.arm_from[e * n_held..(e + 1) * n_held]
                    .copy_from_slice(&self.arm_staged[e * n_held..(e + 1) * n_held]);
                self.arm_blend[e] = 0.0;
            }
        }
        self.live_arm = Some(targets.to_vec());
    }

    /// For GPU-resident callers (the web demo's single-submit control step
    /// bypasses `step()`, so `stage_arm_motion` never runs there): stage the
    /// live/clip held-joint targets and upload the links buffer now. The
    /// mirror's ACTUATED entries may be stale — harmless, the GPU scatter
    /// rewrites them from the policy output within the same control step,
    /// before any physics substep runs.
    pub fn stage_and_flush_arm_targets(&mut self) {
        if self.arm_motion.is_none() && self.live_arm.is_none() && self.live_fadeout == 0 {
            return;
        }
        self.stage_arm_motion();
        self.state
            .multibodies_mut()
            .flush_links_static(&self.gpu)
            .expect("flush arm targets");
    }

    /// Remove live targets, crossfading the held joints back to clip/home.
    pub fn clear_live_arm_targets(&mut self) {
        if self.live_arm.is_none() {
            return;
        }
        let n_held = self.idx.held.len();
        for e in 0..self.n {
            self.arm_from[e * n_held..(e + 1) * n_held]
                .copy_from_slice(&self.arm_staged[e * n_held..(e + 1) * n_held]);
            self.arm_blend[e] = 0.0;
        }
        self.live_arm = None;
        self.live_fadeout = 30; // ~0.6 s of staging to finish the fade home
    }

    pub async fn step(&mut self, actions: &[[f32; NUM_JOINTS]]) -> Vec<StepOut> {
        assert_eq!(actions.len(), self.n);

        // (0) Upper-body playback: restage the held-joint targets FIRST, so
        // both staging branches below flush them together with the leg
        // targets in the one links_static upload.
        self.stage_arm_motion();

        // (1) Stage every env's motor targets host-side in the mirror, then
        // push the whole `links_static` buffer in ONE write_buffer call.
        // Replaces `num_envs * NUM_JOINTS` per-step write_buffer calls.
        //
        // With BIPED_MOTOR_DELAY, the delay itself runs GPU-side (see the
        // delay-state upload below); staging is identical to the no-delay path.
        let t = Instant::now();
        if self.motor_delay.is_none() {
            let n = self.n;
            for e in 0..n {
                let targets = self.task.joint_targets(&actions[e]);
                for k in 0..NUM_JOINTS {
                    self.targets_row[k * n + e] = targets[k];
                }
            }
            self.timings.stage_motors_ns += t.elapsed().as_nanos() as u64;

            let t = Instant::now();
            // ORDER MATTERS: the held flush uploads the WHOLE mirror, whose
            // actuated entries are deliberately stale (we stopped staging them
            // per env — the scatter is the only writer). So flush first, then
            // scatter the fresh targets over the top, before any substep runs.
            if std::mem::take(&mut self.held_dirty) {
                self.state
                    .multibodies_mut()
                    .flush_links_static(&self.gpu)
                    .expect("flush held targets");
            }
            self.flush_motor_targets();
            self.timings.flush_static_ns += t.elapsed().as_nanos() as u64;
        } else {
            // GPU-side delay: stage the CURRENT targets for every env (exactly
            // the no-delay staging), then upload the per-batch delay state
            // [tick=0, k_eff, prev targets] in ONE additional pre-step write.
            // The gravity_and_lu kernel swaps in the previous target while its
            // per-step tick < k — ZERO mid-decimation host writes (the old
            // per-substep restage stalled the stream on a pageable H2D copy,
            // ~70 ms/step at 4096 envs).
            let n = self.n;
            for e in 0..n {
                self.delay_now[e] = self.task.joint_targets(&actions[e]);
                let tg = self.delay_now[e];
                for k in 0..NUM_JOINTS {
                    self.targets_row[k * n + e] = tg[k];
                }
            }
            self.timings.stage_motors_ns += t.elapsed().as_nanos() as u64;

            let t = Instant::now();
            // Flush BEFORE the scatter — see the no-delay branch.
            if std::mem::take(&mut self.held_dirty) {
                self.state
                    .multibodies_mut()
                    .flush_links_static(&self.gpu)
                    .expect("flush held targets");
            }
            self.flush_motor_targets();
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
                // Held links: mirror the just-staged playback target into the
                // prev slot (prev == current → the delay swap is an identity
                // for the upper body; the buffer default of 0.0 would dip the
                // elbows toward q=0 for the first k substeps of every step).
                if self.arm_motion.is_some() {
                    let n_held = self.idx.held.len();
                    for (j, h) in self.idx.held.iter().enumerate() {
                        self.delay_state_buf[base + 2 + h.link as usize] =
                            self.arm_staged[e * n_held + j];
                    }
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

        // (2) Advance physics at the control decimation. On a CUDA backend the
        // `decimation × pipeline.step` dispatch sequence is captured ONCE (after
        // warmup) into a CUDA graph and replayed per step — removing the
        // ~24 ms/step host re-encode. That re-encode cost GROWS over a run
        // (measured 2.3s → 12s/iter by iter 800 on the eager path); graph replay
        // holds it FLAT at ~2.3s/iter. The graph records kernel launches, not
        // data, so the per-step motor-buffer write (above) and resets are
        // honoured on replay. DEFAULT ON; `BIPED_GRAPH=0` forces eager dispatch
        // (fallback if a graph-replay driver issue ever surfaces).
        let t = Instant::now();
        let mut ran_physics = false;
        #[cfg(feature = "cuda_backend")]
        if self.knobs.graph {
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
            let trace = self.knobs.substep_trace;
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
        // Force-sensed contact (BIPED_CONTACT_SENSE=1): pull the per-foot
        // normal-impulse sums the MB solver folded out during the last physics
        // step of this control tick. Tiny buffer (4 f32 per env) — piggybacks
        // on the same sync point as the pose readback.
        if self.contact_sense {
            let mbs_per_batch = {
                let mbs = self.state.multibodies_mut();
                mbs.multibodies_per_batch() as usize
            };
            let buf_len = self
                .state
                .multibodies_mut()
                .contact_sensor_out()
                .buffer()
                .len();
            let mut imp = vec![0.0f32; buf_len];
            self.gpu
                .slow_read_buffer(
                    self.state.multibodies_mut().contact_sensor_out().buffer(),
                    &mut imp,
                )
                .await
                .unwrap();
            // Env e = multibody 0 of batch e; upstream-base buffers are
            // INTERLEAVED: slot index = (mb_idx·num_batches + batch)·MAX + s
            // = (0·n + e)·MAX + s with one robot per batch. Slot i = foot i
            // (set_contact_sensor_links order).
            let _ = mbs_per_batch; // 1 robot per batch on this stack
            for e in 0..self.n {
                let base = e * MAX_CONTACT_SENSORS as usize;
                // Roll the force history for the force_rate reward. On the
                // first read after a reset, prev := new (ΔF = 0) so the
                // half-body-weight seed never fabricates a force spike.
                if self.has_prev_force[e] {
                    self.prev_sensed_force[e] = self.sensed_force[e];
                }
                for i in 0..NUM_FEET {
                    self.sensed_force[e][i] = imp[base + i] * self.sensor_inv_dt;
                }
                if !self.has_prev_force[e] {
                    self.prev_sensed_force[e] = self.sensed_force[e];
                    self.has_prev_force[e] = true;
                }
            }
        }
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
                self.cmd[e] = eval_cmd_override().unwrap_or_else(|| self.sampler.sample(&mut self.rng[e]));
                self.resample_at[e] = self.step_count[e]
                    + self
                        .sampler
                        .resample_steps(&mut self.rng[e], self.task.control_dt());
                self.arm_resample(e);
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
            /// Host joint velocity, carried only so `BIPED_VERIFY_REWARD` can
            /// check the GPU joint-state kernel against it.
            joint_vel_dbg: [f32; NUM_JOINTS],
            /// Host base state, same purpose: quat xyzw, lin vel, ang vel, height.
            base_dbg: [f32; 11],
            /// The step cue AS THE HOST USED IT (noised/dropout-masked):
            /// valid, distance, edge_cos, edge_sin.
            cue_dbg: [f32; 4],
            /// Both cues in the OBS kernel's upload order — noisy 5 then clean
            /// 5, each (distance, height, edge_sin, edge_cos, valid). RAW: the
            /// gating and clamping happen in the kernel, matching `observe`.
            cue_obs: [f32; 10],
            /// Host per-foot state for the GPU feet check: per foot, 11 fields
            /// in the kernel's order.
            feet_dbg: [f32; 22],
            /// The host's `stepping` predicate and cue height, which gate the
            /// clearance target.
            stepping_dbg: f32,
            cue_h_dbg: f32,
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
        let illegal_z = self.knobs.illegal_z;
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
        let sc_margin = self.knobs.sc_margin;
        let sc_weight = self.knobs.sc_weight;
        // Leg-interpenetration termination distance (BIPED_SELF_COLL_TERM,
        // 0 disables). 0.05 m between link centers ≈ colliders overlapping.
        let sc_term = self.knobs.sc_term;
        // Joint-velocity fault (BIPED_DOF_VEL_TERM, multiplier on each
        // joint's hardware vel_limit; 0 disables). Real actuators fault past
        // rated speed — flail-speed swings end the episode like an e-stop.
        let vel_term: f32 = self.knobs.vel_term;
        // Mechanical-power fault (BIPED_POWER_TERM, watts of Σ|τ·q̇| in one
        // control step; 0 disables). Calibration: walking ≈ 350 W, the
        // stand-tremor pathology ≈ 2000 W — sustained dither burns past the
        // threshold while honest locomotion never approaches it.
        let power_term: f32 = self.knobs.power_term;
        // DC torque-speed envelope fault (BIPED_ENVELOPE_TERM, speed-axis
        // scale; 0 disables — and it DEFAULTS OFF: with 50 Hz finite-diff
        // joint velocities, contact aliasing throws single joints past rated
        // speed for one sample and even healthy gaits "violate" on 92% of
        // steps (measured). The correct form of this idea is kernel-level DC
        // torque-speed saturation in the actuator model, not a termination.
        let env_term: f32 = self.knobs.env_term;
        // Endstop-SLAM fault (BIPED_LIMIT_SLAM_VEL, rad/s; 0 disables). Resting
        // against a position limit is a static load the structure carries, but
        // ARRIVING at one carries ½Iω² into the gearbox — measured entries hit
        // 9-10 rad/s ≈ 0.9 J, like dropping the foot 15 cm through the ankle
        // drive. Terminating on contact would fire within 0.15 s of every
        // episode (the ankle is inside the band >50% of the time) and bury the
        // gradient; gating on approach SPEED targets only the damaging case.
        // 2.0 rad/s ≈ 43 mJ, the energy of a 7 mm drop.
        // Endstop-DWELL fault (BIPED_LIMIT_DWELL_STEPS, consecutive control
        // steps inside the band; 0 disables). The free-support exploit is
        // STATIC — the constraint carries ~97% of the ankle load while the
        // motor idles — so a velocity gate can't catch it: the policy can
        // drift onto the stop slowly and lean forever. Terminating on CONTACT
        // is not an option either (54-66% of frames touch, in every gait).
        // Dwell separates them: standing leans ~19 steps at a stretch,
        // walking brushes ~9.
        let dwell_max: u16 = self.knobs.dwell_max;
        let slam_vel: f32 = self.knobs.slam_vel;
        const SLAM_BAND: f32 = 0.05; // rad from the hard limit
        // Per-joint instantaneous power fault (BIPED_JOINT_POWER_TERM, watts;
        // 0 disables). Spikes are the failure mode averages cannot see, and
        // hardware limits are per actuator: healthy v17-style gait peaks at
        // ~700 W in the worst joint, the stand-tremor spikes to 20 kW.
        let joint_power_term: f32 = self.knobs.joint_power_term;
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
        // Multiplier on the ankle extras. Default 1.0 = AGILE parity (the old
        // 4.0 was tuned against the lerobot-magnitude constants below and
        // would put the G1 ankles 4× over WBC).
        let ankle_torque_w = self.knobs.ankle_torque_w;
        // AGILE-named torque regularizers, WBC-AGILE **G1** config verbatim
        // (`torques` -5e-5 on every controlled joint, `ankle_torques` an extra
        // -1e-4 on ankle joints, `ankle_roll_torques` an extra -1e-3 on
        // ankle-roll). The previous hardcoded constants were WBC's *lerobot*
        // magnitudes (5e-4/1.5e-3/6.5e-3 — 10× stronger) and the roll branch
        // matched the lerobot-only name "anklex", which never fires on the
        // G1's `ankle_roll` joints.
        let env_f32 = |k: &str| env_or_override(k).and_then(|s| s.parse::<f32>().ok());
        // 1e-4 = 2x WBC-AGILE's G1 value (5e-5). The lerobot-magnitude 5e-4 the
        // v28 run trained with prices the knee out of flexing: measured gait
        // never bends past the 0.3 rad home pose (ROM 0.02-0.30, mean 0.12)
        // and the two knees burn 20-50% of the positive reward at stance
        // peaks — the likely terrain-curriculum blocker (clearance needs
        // knee flexion). 2x (not 1x) keeps some extra effort pressure since
        // nexus's torque-to-stand runs higher than PhysX's.
        let w_torques = env_f32("BIPED_W_TORQUES").unwrap_or(1e-4);
        let w_ankle_torques = env_f32("BIPED_W_ANKLE_TORQUES").unwrap_or(1.5e-3);
        let w_ankle_roll_torques = env_f32("BIPED_W_ANKLE_ROLL_TORQUES").unwrap_or(0.0);
        // Knee-specific torque extra (BIPED_W_KNEE_TORQUES, per-step weight on
// tau^2; 0 = off). Unlike the generic ramped leg term this is
// FULL-STRENGTH from iter 0, like the ankle extras: the knee holds the
// crouch, and a sustained 105 N.m (75% of the 139 limit measured at
// cmd 0.4) is a thermal problem long before it is an electrical one.
// Extension is free for the knee - the load passes through the joint -
// so this term prices the crouch itself.
        // SIZING TRAP (v22): this was first set to 1.5e-3, copied from the ankle
// extra -- but the penalty is tau^2 and the knee runs at ~4.5x the ankle's
// torque, so the same weight costs ~20x more. At 1.5e-3 a 105 N.m walking
// peak charges 0.33/step, MORE than the entire positive reward (~0.26): the
// policy's only survivable answer was to stop using the knee at all
// (measured: -4 deg through the whole swing, a locked compass gait). Size
// this by the COST it should impose, not by another joint's weight. 7e-5
// puts a walking peak at ~0.016/step, comparable to the ankle extra.
let w_knee_torques: f32 = self.knobs.w_knee_torques;
        // Mechanical-power (energy) penalty weight. Penalizes Σ|τᵢ·q̇ᵢ| — the rate
        // of mechanical work, the principled cost-of-transport proxy. Unlike Στ²
        // (effort, penalized even when static), this only charges for work done in
        // motion, so energy-economical (natural) gaits are favored and degenerate
        // high-energy modes (marching in place, frantic shuffling) are punished.
        // BIPED_POWER_W tunes it (0 = off). Default 4e-3 (was 2e-3): the
        // higher energy price further biases against shuffle/ankle-balance
        // gaits in favor of discrete weight-transferring steps.
        // RENAMED: `BIPED_MECH_POWER_W` (mechanical-power Σ|τ·q̇| penalty, a
        // zealot-only term — WBC-AGILE has NO power cost; its gait economy is
        // the torque family above). Default now 0 = AGILE parity; the legacy
        // name `BIPED_POWER_W` is still honored if set.
        let power_w: f32 = self.knobs.power_w;
        // Chest angular-velocity penalty (BIPED_CHEST_ANGVEL_W, 0 = off).
        // Every other stability term reads the PELVIS (link 0); on the 29-DOF
        // body the chest above the waist joints is invisible to the reward, so
        // the policy never learns pelvis trajectories that don't shake what's
        // above them. Penalizes the chest link's roll/pitch rate ω²_xy
        // (finite-diff, same approximation as the base ang-vel obs) — yaw is
        // excluded so turning with the commanded heading isn't taxed. The
        // Isaac-Lab humanoid analogue is `ang_vel_xy_l2` on the torso link.
        let chest_w: f32 = self.knobs.chest_w;
        let chest_link = self.idx.chest_link as usize;
        let cpb_idx = self.idx.colliders_per_batch as usize;
        // Step-cue knobs, read once per step rather than per env.
        // BIPED_STEP_CUE=0 disables the cue entirely (obs slots stay zero), so
        // a run can be done with the wider observation but no step information
        // -- the control for "did the cue actually help".
        let step_cue_on = std::env::var("BIPED_STEP_CUE").as_deref() != Ok("0");
        // Detector-shaped error. A depth-based edge detector is decent when it
        // fires and occasionally does not fire at all; training on a clean
        // oracle would teach the policy to trust it, and the first dropout on
        // hardware becomes a fall rather than a refusal.
        // Base dropout from the env (deploy-level, default 0.10) unless the
        // trainer has installed an annealed override (exploration schedule:
        // hide the cue from the ACTOR early so it practices crossings blind,
        // then hand the cue back as skill accrues; the critic sees the clean
        // cue throughout, so the gradient credits crossings either way).
        let step_cue_dropout: f32 = self.step_cue_dropout_override.unwrap_or_else(|| {
            std::env::var("BIPED_STEP_CUE_DROPOUT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.10)
        });
        let step_cue_dn: f32 = std::env::var("BIPED_STEP_CUE_DIST_NOISE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.03);
        let step_cue_hn: f32 = std::env::var("BIPED_STEP_CUE_H_NOISE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
        #[allow(unused_mut)]
        let mut computed: Vec<PerEnv> = (0..self.n)
            .into_par_iter()
            .with_min_len(64)
            .map(|e| {
                let (feet, new_air) = self.compute_feet_from_poses(e, &poses);
                let (mut state, new_joint_pos) = self.read_state_from_poses(e, &poses);
                state.feet = feet;
                state.phase = self.gait_phase[e];
                // Step cue from the foot-probe oracle, with PROBE-shaped error.
                // A real probe is accurate when it succeeds (the foot touched
                // the edge) but fails outright now and then -- foot slips off,
                // sweep misses, operator probes the wrong spot. So: small
                // Gaussian-ish noise on both numbers, plus a dropout that
                // clears the cue entirely. Training on a perfect oracle would
                // teach the policy to trust it, and the first bad probe on
                // hardware becomes a fall instead of a refusal.
                if let Some(t) = self.terrain.as_ref() {
                    if step_cue_on {
                        let yaw = {
                            let q = state.base.orientation;
                            // q = [x, y, z, w]
                            (2.0 * (q[3] * q[2] + q[0] * q[1]))
                                .atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
                        };
                        let mut cue =
                            t.probe(e, state.base.pos_xy[0], state.base.pos_xy[1], yaw);
                        state.step_cue_clean = cue; // critic-only, un-noised
                        if cue.valid > 0.5 {
                            let r = &mut self.rng[e].clone();
                            if r.range(0.0, 1.0) < step_cue_dropout {
                                cue = Default::default();
                            } else {
                                cue.distance += r.range(-step_cue_dn, step_cue_dn);
                                cue.height += r.range(-step_cue_hn, step_cue_hn);
                            }
                        }
                        state.step_cue = cue;
                    }
                }
                // BIPED_VERIFY_CUE: deterministic synthetic cue for the obs
                // harness. The terrain curriculum starts flat, so a short run
                // never produces a live cue and the kernel's gate + clamp
                // branches would be compared as zeros against zeros. Values are
                // chosen to straddle BOTH clamp bounds and to differ between
                // the actor and clean copies, so a swapped source is caught.
                if self.verify_cue {
                    let f = |k: usize, m: usize, lo: f32, st: f32| lo + ((e + k) % m) as f32 * st;
                    let ang = (e % 13) as f32 * 0.5;
                    state.step_cue = zealot_env::tasks::velocity_flat::StepCue {
                        valid: if e % 3 == 0 { 0.0 } else { 1.0 },
                        distance: f(0, 17, -1.0, 0.25),
                        height: f(3, 11, -0.8, 0.16),
                        edge_sin: ang.sin(),
                        edge_cos: ang.cos(),
                    };
                    state.step_cue_clean = zealot_env::tasks::velocity_flat::StepCue {
                        valid: if e % 4 == 0 { 0.0 } else { 1.0 },
                        distance: f(5, 17, -1.0, 0.25),
                        height: f(7, 11, -0.8, 0.16),
                        edge_sin: (ang + 0.3).sin(),
                        edge_cos: (ang + 0.3).cos(),
                    };
                }
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
                // E-stop termination: legs truly interpenetrating (any L/R pair
                // inside `sc_term`). Neither engine has leg-leg collision, so on
                // hardware this state is an operator emergency stop — train the
                // same contract. The soft `sc_weight` ring (below) keeps this
                // rare after the first few hundred iters.
                let crossed = sc_term > 0.0
                    && self.idx.self_collision_pairs.iter().any(|&(a, b)| {
                        let pa = poses[env_base + a as usize].translation;
                        let pb = poses[env_base + b as usize].translation;
                        (pa - pb).length_squared() < sc_term * sc_term
                    });
                let vel_fault = vel_term > 0.0
                    && (0..NUM_JOINTS).any(|i| {
                        state.joint_vel[i].abs()
                            > self.task.robot.joints[i].vel_limit * vel_term
                    });
                // Leaning on an endstop: inside the band for `dwell_max`
                // consecutive control steps (own-env entries only, so the
                // relaxed atomics never race).
                let dwell_fault = if dwell_max > 0 {
                    use std::sync::atomic::Ordering::Relaxed;
                    let mut hit = false;
                    for i in 0..NUM_JOINTS {
                        let (lo, hi) = self.task.robot.joints[i].pos_limit;
                        let q = state.joint_pos[i];
                        let c = &self.limit_dwell[e * NUM_JOINTS + i];
                        if q < lo + SLAM_BAND || q > hi - SLAM_BAND {
                            if c.fetch_add(1, Relaxed) + 1 >= dwell_max {
                                hit = true;
                            }
                        } else {
                            c.store(0, Relaxed);
                        }
                    }
                    hit
                } else {
                    false
                };
                // Slamming a joint endstop: inside the band AND still moving
                // into it above the energy threshold.
                let slam_fault = slam_vel > 0.0
                    && (0..NUM_JOINTS).any(|i| {
                        let (lo, hi) = self.task.robot.joints[i].pos_limit;
                        let (q, v) = (state.joint_pos[i], state.joint_vel[i]);
                        (q < lo + SLAM_BAND && v < -slam_vel)
                            || (q > hi - SLAM_BAND && v > slam_vel)
                    });
                let (power_fault, envelope_fault) = if power_term > 0.0
                    || env_term > 0.0
                    || joint_power_term > 0.0
                {
                    let q_target = self.task.joint_targets(&actions[e]);
                    let mut p = 0.0f32;
                    let mut pj_max = 0.0f32;
                    let mut env_viol = false;
                    for i in 0..NUM_JOINTS {
                        let j = &self.task.robot.joints[i];
                        let tau = (j.kp * (q_target[i] - state.joint_pos[i])
                            - j.kd * state.joint_vel[i])
                            .clamp(-j.effort_limit, j.effort_limit);
                        let pj = (tau * state.joint_vel[i]).abs();
                        p += pj;
                        pj_max = pj_max.max(pj);
                        if env_term > 0.0 {
                            let avail = j.effort_limit
                                * (1.0 - state.joint_vel[i].abs() / (env_term * j.vel_limit))
                                    .max(0.0);
                            // only torque WITH the motion direction is motor
                            // work — braking torque comes for free in a DC
                            // motor and must not trip the envelope.
                            if tau * state.joint_vel[i] > 0.0 && tau.abs() > avail {
                                env_viol = true;
                            }
                        }
                    }
                    (
                        (power_term > 0.0 && p > power_term)
                            || (joint_power_term > 0.0 && pj_max > joint_power_term),
                        env_viol,
                    )
                } else {
                    (false, false)
                };
                let fell = illegal
                    || crossed
                    || vel_fault
                    || slam_fault
                    || dwell_fault
                    || power_fault
                    || envelope_fault
                    || self.task.fell_over(&state.base)
                    || !state.base.height.is_finite();
                // BIPED_SKIP_REWARD: A/B lever isolating the cost of the host
                // reward evaluation from the rest of the per-env block, so the
                // GPU port's ~0.55 ms/step can be priced against what it would
                // actually replace. Produces WRONG training — measurement only.
                let rb = if self.skip_reward || self.use_gpu_reward {
                    // BIPED_GPU_REWARD: the device stack computes every term;
                    // `comps` are filled after the parallel block.
                    Default::default()
                } else {
                    self.task.reward(&state, &self.cmd[e])
                };
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
                comps[26] = rb.stand_planted;
                comps[27] = rb.feet_yaw_diff;
                comps[28] = rb.force_rate;
                comps[29] = rb.action_rate_rate;
                comps[30] = rb.touchdown_vz;
                if fell {
                    use std::sync::atomic::Ordering::Relaxed;
                    for i in 0..NUM_JOINTS {
                        self.limit_dwell[e * NUM_JOINTS + i].store(0, Relaxed);
                    }
                    comps[23] = self.task.weights.termination;
                    reward += self.task.weights.termination;
                }
                // Soft self-collision penalty: ramp up as any L/R pair intrudes
                // inside `sc_margin` (legs crossing). ~0 for a clean stance.
                if sc_weight > 0.0 && !self.use_gpu_reward {
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
                if !self.use_gpu_reward
                    && (torque_w > 0.0 || ankle_torque_w > 0.0 || power_w > 0.0 || w_knee_torques > 0.0)
                {
                    let q_target = self.task.joint_targets(&actions[e]);
                    let mut leg_pen = 0.0f32;
                    let mut ankle_pen = 0.0f32;
                    let mut knee_pen = 0.0f32; // unramped, full-strength like the ankle extras
                    let mut power = 0.0f32; // Σ|τ·q̇| mechanical power (energy rate)
                    for i in 0..NUM_JOINTS {
                        let j = &self.task.robot.joints[i];
                        let tau = (j.kp * (q_target[i] - state.joint_pos[i])
                            - j.kd * state.joint_vel[i])
                            .clamp(-j.effort_limit, j.effort_limit);
                        let t2 = tau * tau;
                        power += (tau * state.joint_vel[i]).abs();
                        // AGILE structure: base `torques` on EVERY joint, plus
                        // additive ankle / ankle-roll extras (their RewTerms
                        // stack). Roll detection covers both naming schemes
                        // (G1 `ankle_roll`, lerobot `anklex`).
                        leg_pen += w_torques * t2;
                        if j.name.contains("ankle") {
                            let mut w = w_ankle_torques;
                            if j.name.contains("ankle_roll") || j.name.contains("anklex") {
                                w += w_ankle_roll_torques;
                            }
                            ankle_pen += w * t2;
                        }
                        if j.name.contains("knee") {
                            knee_pen += w_knee_torques * t2;
                        }
                    }
                    comps[20] = -(torque_w * leg_pen + knee_pen) * sc_dt;
                    comps[21] = -(ankle_torque_w * ankle_pen) * sc_dt;
                    comps[24] = -(power_w * power) * sc_dt;
                    reward -= (torque_w * leg_pen
                        + knee_pen
                        + ankle_torque_w * ankle_pen
                        + power_w * power)
                        * sc_dt;
                }
                // Chest roll/pitch rate penalty (see `chest_w` above). Same
                // finite-diff ω approximation as `read_state_from_poses`.
                if chest_w > 0.0 && self.has_prev_pose[e] && !self.use_gpu_reward {
                    let cur = &poses[env_base + chest_link];
                    let prev = &self.prev_body_poses[env_base + chest_link];
                    let dq = cur.rotation * prev.rotation.conjugate();
                    let s = if dq.w >= 0.0 { 1.0 } else { -1.0 };
                    let wx = 2.0 * s * dq.x / sc_dt;
                    let wy = 2.0 * s * dq.y / sc_dt;
                    let pen = chest_w * (wx * wx + wy * wy) * sc_dt;
                    comps[31] = -pen;
                    reward -= pen;
                }
                // BIPED_SKIP_OBS: A/B lever pricing the HOST obs assembly, the
                // thing the device-resident path removes. Wrong training;
                // measurement only. Vectors keep their length so every
                // downstream consumer (gc/gcc, fill_raw, the bootstrap) still
                // does its normal work — this isolates assembly, not transfer.
                let mut obs = vec![0.0; OBS_DIM];
                let mut critic_obs = vec![0.0; CRITIC_OBS_DIM];
                if !self.skip_obs {
                    self.task.observe(&state, &self.cmd[e], &mut obs);
                    self.task
                        .observe_critic(&state, &self.cmd[e], &mut critic_obs);
                }
                // Which foot touched down this step (last wins if both did) — used
                // to advance the gait-alternation tracker in the serial pass.
                let mut td_foot: i8 = -1;
                for (i, f) in state.feet.iter().enumerate() {
                    if f.first_contact {
                        td_foot = i as i8;
                    }
                }
                PerEnv {
                    joint_vel_dbg: state.joint_vel,
                    // The host's `stepping` predicate, recomputed here from the
                    // same state the reward uses, so the GPU gate is checked
                    // against the identical condition.
                    stepping_dbg: {
                        let vb = zealot_env::math::quat_rotate_inv(
                            state.base.orientation,
                            state.base.lin_vel_world,
                        );
                        let toward = vb[0] * state.step_cue.edge_cos
                            + vb[1] * state.step_cue.edge_sin;
                        let st = state.step_cue.valid > 0.5
                            && state.step_cue.distance.abs() < self.task.step_relax_dist
                            && toward > 0.1;
                        if st { 1.0 } else { 0.0 }
                    },
                    cue_h_dbg: state.step_cue.height,
                    feet_dbg: {
                        let mut f = [0.0f32; 22];
                        for (i, fo) in state.feet.iter().enumerate() {
                            let b = i * 11;
                            f[b] = fo.contact as u8 as f32;
                            f[b + 1] = fo.first_contact as u8 as f32;
                            f[b + 2] = fo.air_time;
                            f[b + 3] = fo.height;
                            f[b + 4] = fo.planar_speed;
                            f[b + 5] = fo.tilt;
                            f[b + 6] = fo.yaw_rel_base;
                            f[b + 7] = fo.pos_xy[0];
                            f[b + 8] = fo.pos_xy[1];
                            f[b + 9] = fo.vz;
                            f[b + 10] = fo.force_rate;
                        }
                        f
                    },
                    cue_dbg: [
                        state.step_cue.valid,
                        state.step_cue.distance,
                        state.step_cue.edge_cos,
                        state.step_cue.edge_sin,
                    ],
                    cue_obs: [
                        state.step_cue.distance,
                        state.step_cue.height,
                        state.step_cue.edge_sin,
                        state.step_cue.edge_cos,
                        state.step_cue.valid,
                        state.step_cue_clean.distance,
                        state.step_cue_clean.height,
                        state.step_cue_clean.edge_sin,
                        state.step_cue_clean.edge_cos,
                        state.step_cue_clean.valid,
                    ],
                    base_dbg: [
                        state.base.orientation[0],
                        state.base.orientation[1],
                        state.base.orientation[2],
                        state.base.orientation[3],
                        state.base.lin_vel_world[0],
                        state.base.lin_vel_world[1],
                        state.base.lin_vel_world[2],
                        state.base.ang_vel_world[0],
                        state.base.ang_vel_world[1],
                        state.base.ang_vel_world[2],
                        state.base.height,
                    ],
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

        // ---- GPU reward terms (partial port) ----
        // Verified against the host values with BIPED_VERIFY_REWARD=1. Until a
        // term is confirmed it stays host-owned, so totals are unaffected.
        if env_var("BIPED_VERIFY_REWARD").is_ok() {
            let n = self.n;
            let jn = NUM_JOINTS * n;
            if self.gpu_reward.is_none() {
                self.gpu_reward = Some(
                    zealot_gpu_obs::GpuRewardTerms::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &self.task.hip_yawroll_idx(),
                    )
                    .expect("gpu reward terms"),
                );
            }
            let mut la = vec![0.0f32; jn];
            let mut pa = vec![0.0f32; jn];
            let mut p2a = vec![0.0f32; jn];
            for e in 0..n {
                for k in 0..NUM_JOINTS {
                    la[k * n + e] = self.last_action[e][k];
                    pa[k * n + e] = self.prev_action[e][k];
                    p2a[k * n + e] = self.prev2_action[e][k];
                }
            }
            let w = &self.task.weights;
            let got = self
                .gpu_reward
                .as_mut()
                .unwrap()
                .compute(
                    &self.gpu,
                    &la,
                    &pa,
                    &p2a,
                    self.task.control_dt(),
                    w.action_rate,
                    w.action_rate_hipz_hipx,
                    w.action_rate_rate,
                )
                .await
                .expect("gpu reward compute");
            let mut worst = [0.0f32; GPU_REWARD_TERMS.len()];
            for (t_i, (comp, row)) in GPU_REWARD_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (got[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > worst[t_i] {
                        worst[t_i] = d;
                    }
                }
            }
            eprintln!(
                "[verify_reward] action_rate={:.3e} hipz_hipx={:.3e} action_rate_rate={:.3e}",
                worst[0], worst[1], worst[2]
            );

            // ---- joint state (q, qd) ----
            let bps = (self.state.body_poses().len() as usize / n) as u32;
            if self.gpu_joints.is_none() {
                let rest: Vec<glamx::Vec4> = (0..NUM_JOINTS)
                    .map(|k| {
                        let r = self.idx.actuated_rest_quat[k];
                        glamx::Vec4::new(r.x, r.y, r.z, r.w)
                    })
                    .collect();
                let children: Vec<u32> =
                    (0..NUM_JOINTS).map(|k| self.idx.actuated[k].0).collect();
                self.gpu_joints = Some(
                    zealot_gpu_obs::GpuJointState::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &self.idx.actuated_parent_links,
                        &children,
                        &rest,
                        bps,
                        self.task.control_dt(),
                    )
                    .expect("gpu joint state"),
                );
            }
            let hp: Vec<u32> = (0..n).map(|e| self.has_prev_joint_pos[e] as u32).collect();
            let poses_t = self.state.body_poses();
            let (gq, gqd) = self
                .gpu_joints
                .as_mut()
                .unwrap()
                .compute(&self.gpu, poses_t, &hp)
                .await
                .expect("gpu joint state compute");
            let (mut wq, mut wqd) = (0.0f32, 0.0f32);
            for e in 0..n {
                for k in 0..NUM_JOINTS {
                    let i = k * n + e;
                    wq = wq.max((gq[i] - computed[e].new_joint_pos[k]).abs());
                    wqd = wqd.max((gqd[i] - computed[e].joint_vel_dbg[k]).abs());
                }
            }
            eprintln!("[verify_joints] q={wq:.3e} qd={wqd:.3e}");

            // ---- joint-only reward terms ----
            if self.gpu_joint_terms.is_none() {
                let dpos: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].default_pos).collect();
                // The host applies the 0.9 soft band before comparing, so bake
                // it in here rather than in the kernel.
                let lo: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].pos_limit.0 * 0.9).collect();
                let hi: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].pos_limit.1 * 0.9).collect();
                self.gpu_joint_terms = Some(
                    zealot_gpu_obs::GpuRewardJointTerms::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &dpos,
                        &lo,
                        &hi,
                        &self.task.hip_yawroll_idx(),
                        &(0..NUM_JOINTS)
                            .map(|k| self.robot.mirror[k] as u32)
                            .collect::<Vec<_>>(),
                        &(0..NUM_JOINTS)
                            .map(|k| self.robot.mirror_sign[k])
                            .collect::<Vec<_>>(),
                    )
                    .expect("gpu joint terms"),
                );
            }
            let w = &self.task.weights;
            let (wp, wl, wv) = (w.pose, w.dof_pos_limits, w.dof_vel);
            let wb = w.bilateral_symmetry;
            let gate = self.task.sym_yaw_gate;
            let dtc = self.task.control_dt();
            let yaws: Vec<f32> = (0..n).map(|e| self.cmd[e].yaw_rate).collect();
            let joints_ref = self.gpu_joints.as_ref().unwrap();
            let jt = self
                .gpu_joint_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, joints_ref, dtc, wp, wl, wv, wb, gate, &yaws)
                .await
                .expect("gpu joint terms compute");
            const JOINT_TERMS: [(usize, usize); 4] = [(4, 0), (10, 1), (11, 2), (5, 3)];
            let mut wj = [0.0f32; 4];
            for (ti, (comp, row)) in JOINT_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (jt[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wj[ti] {
                        wj[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_joint_terms] pose={:.3e} dof_pos_limits={:.3e} dof_vel={:.3e} bilateral={:.3e}",
                wj[0], wj[1], wj[2], wj[3]
            );

            // ---- torque / power terms ----
            if self.gpu_torque_terms.is_none() {
                // Resolve the host's joint-NAME classification once, into
                // per-joint weights the kernel just multiplies by.
                let w_torques = env_f32("BIPED_W_TORQUES").unwrap_or(1e-4);
                let w_ankle_t = env_f32("BIPED_W_ANKLE_TORQUES").unwrap_or(1.5e-3);
                let w_ankle_roll = env_f32("BIPED_W_ANKLE_ROLL_TORQUES").unwrap_or(0.0);
                let w_knee = self.knobs.w_knee_torques;
                let js = &self.task.robot.joints;
                let kp: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].kp).collect();
                let kd: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].kd).collect();
                let eff: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].effort_limit).collect();
                let wl = vec![w_torques; NUM_JOINTS];
                let wa: Vec<f32> = (0..NUM_JOINTS)
                    .map(|k| {
                        let nm = &js[k].name;
                        if nm.contains("ankle") {
                            let roll = nm.contains("ankle_roll") || nm.contains("anklex");
                            w_ankle_t + if roll { w_ankle_roll } else { 0.0 }
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let wk: Vec<f32> = (0..NUM_JOINTS)
                    .map(|k| if js[k].name.contains("knee") { w_knee } else { 0.0 })
                    .collect();
                self.gpu_torque_terms = Some(
                    zealot_gpu_obs::GpuRewardTorqueTerms::new(
                        &self.gpu, n, NUM_JOINTS, &kp, &kd, &eff, &wl, &wa, &wk,
                    )
                    .expect("gpu torque terms"),
                );
            }
            let (tw, atw, pw) = (self.torque_scale, self.knobs.ankle_torque_w, self.knobs.power_w);
            let tr = self.targets_row.clone();
            let joints_ref2 = self.gpu_joints.as_ref().unwrap();
            let tt = self
                .gpu_torque_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, joints_ref2, &tr, dtc, tw, atw, pw)
                .await
                .expect("gpu torque terms compute");
            const TORQUE_TERMS: [(usize, usize); 3] = [(20, 0), (21, 1), (24, 2)];
            let mut wt = [0.0f32; 3];
            for (ti, (comp, row)) in TORQUE_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (tt[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wt[ti] {
                        wt[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_torque_terms] torque_leg={:.3e} torque_ankle={:.3e} power={:.3e}",
                wt[0], wt[1], wt[2]
            );

            // ---- base state ----
            if self.gpu_base.is_none() {
                self.gpu_base = Some(
                    zealot_gpu_obs::GpuBaseState::new(
                        &self.gpu,
                        n,
                        bps,
                        self.idx.torso_link,
                        dtc,
                    )
                    .expect("gpu base state"),
                );
            }
            // The terrain lookup stays host-side, so heights stay relative to
            // the LOCAL surface exactly as the host computes them.
            let gh: Vec<f32> = (0..n)
                .map(|e| {
                    let tp = &poses[e * bps as usize + self.idx.torso_link as usize];
                    self.terrain
                        .as_ref()
                        .map_or(0.0, |ter| ter.strip_for(e).height(tp.translation.x, tp.translation.y))
                })
                .collect();
            let hpp: Vec<u32> = (0..n).map(|e| self.has_prev_pose[e] as u32).collect();
            let poses_t2 = self.state.body_poses();
            let gb = self
                .gpu_base
                .as_mut()
                .unwrap()
                .compute(&self.gpu, poses_t2, &hpp, &gh)
                .await
                .expect("gpu base compute");
            let (mut wquat, mut wlv, mut wav, mut wh) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for e in 0..n {
                for k in 0..4 {
                    wquat = wquat.max((gb[k * n + e] - computed[e].base_dbg[k]).abs());
                }
                for k in 0..3 {
                    wlv = wlv.max((gb[(4 + k) * n + e] - computed[e].base_dbg[4 + k]).abs());
                    wav = wav.max((gb[(7 + k) * n + e] - computed[e].base_dbg[7 + k]).abs());
                }
                wh = wh.max((gb[10 * n + e] - computed[e].base_dbg[10]).abs());
            }
            eprintln!(
                "[verify_base] quat={wquat:.3e} lin_vel={wlv:.3e} ang_vel={wav:.3e} height={wh:.3e}"
            );

            // ---- base-dependent reward terms ----
            if self.gpu_base_terms.is_none() {
                self.gpu_base_terms =
                    Some(zealot_gpu_obs::GpuRewardBaseTerms::new(&self.gpu, n).expect("base terms"));
            }
            let tk = &self.task;
            let bp = zealot_obs_shaders::RewardBaseParams {
                n_envs: n as u32,
                dt: dtc,
                w_track_lin: tk.weights.track_lin_vel,
                w_forward_progress: tk.weights.forward_progress,
                w_track_ang: tk.weights.track_ang_vel,
                w_upright: tk.weights.upright,
                w_base_height: tk.weights.base_height,
                w_body_ang_vel: tk.weights.body_ang_vel,
                w_lin_vel_z: tk.weights.lin_vel_z,
                std_lin: tk.stds.lin_vel,
                std_ang: tk.stds.ang_vel,
                std_base_h: tk.stds.base_height,
                std_upright: tk.stds.upright,
                step_std_base_h: tk.step_std_base_h,
                step_std_upright: tk.step_std_upright,
                step_relax_dist: tk.step_relax_dist,
                h_target_stand: tk.weights.base_height_target_stand,
                h_target_walk: tk.weights.base_height_target,
                pad0: 0,
                pad1: 0,
            };
            let mut cmdb = vec![0.0f32; 4 * n];
            let mut cueb = vec![0.0f32; 4 * n];
            for e in 0..n {
                cmdb[e] = self.cmd[e].vx;
                cmdb[n + e] = self.cmd[e].vy;
                cmdb[2 * n + e] = self.cmd[e].yaw_rate;
                cmdb[3 * n + e] = self.cmd[e].speed();
                let c = computed[e].cue_dbg;
                cueb[e] = c[0];
                cueb[n + e] = c[1];
                cueb[2 * n + e] = c[2];
                cueb[3 * n + e] = c[3];
            }
            let bt = self
                .gpu_base_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, bp, &gb, &cmdb, &cueb)
                .await
                .expect("gpu base terms compute");
            const BASE_TERMS: [(usize, usize); 6] =
                [(0, 0), (1, 1), (2, 2), (3, 3), (8, 4), (9, 5)];
            let mut wbt = [0.0f32; 6];
            for (ti, (comp, row)) in BASE_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (bt[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wbt[ti] {
                        wbt[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_base_terms] track_lin={:.3e} track_ang={:.3e} upright={:.3e} base_h={:.3e} body_ang={:.3e} lin_vel_z={:.3e}",
                wbt[0], wbt[1], wbt[2], wbt[3], wbt[4], wbt[5]
            );

            // ---- per-foot state ----
            let cpb = self.idx.colliders_per_batch as usize;
            if self.gpu_feet.is_none() {
                let ff = self.robot.foot_forward_local;
                self.gpu_feet = Some(
                    zealot_gpu_obs::GpuFeetState::new(
                        &self.gpu,
                        n,
                        NUM_FEET,
                        &(0..NUM_FEET).map(|i| self.idx.foot_links[i]).collect::<Vec<_>>(),
                        &[ff[0], ff[1], ff[2]],
                        zealot_obs_shaders::FeetStateParams {
                            n_envs: n as u32,
                            colliders_per_env: cpb as u32,
                            torso_link: self.idx.torso_link,
                            control_dt: dtc,
                            contact_z: env_var("BIPED_CONTACT_Z")
                                .ok()
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0.05),
                            contact_force_n: self.contact_force_n,
                            contact_sense: self.contact_sense as u32,
                            body_weight: self.robot.total_mass * 9.81,
                        },
                    )
                    .expect("gpu feet"),
                );
            }
            let mut sole = vec![0.0f32; NUM_FEET * 3 * n];
            let mut fgh = vec![0.0f32; NUM_FEET * n];
            let mut sfv = vec![0.0f32; NUM_FEET * n];
            let mut pfv = vec![0.0f32; NUM_FEET * n];
            for e in 0..n {
                for i in 0..NUM_FEET {
                    let sl = self.foot_sole_local[e][i];
                    sole[(i * 3) * n + e] = sl.x;
                    sole[(i * 3 + 1) * n + e] = sl.y;
                    sole[(i * 3 + 2) * n + e] = sl.z;
                    let link = self.idx.foot_links[i] as usize;
                    let fpz = &poses[e * cpb + link].translation;
                    fgh[i * n + e] = self
                        .terrain
                        .as_ref()
                        .map_or(0.0, |ter| ter.strip_for(e).height(fpz.x, fpz.y));
                    sfv[i * n + e] = self.sensed_force[e][i];
                    pfv[i * n + e] = self.prev_sensed_force[e][i];
                }
            }
            let hpf: Vec<u32> = (0..n).map(|e| self.has_prev_force[e] as u32).collect();
            let poses_t3 = self.state.body_poses();
            let gf = self
                .gpu_feet
                .as_mut()
                .unwrap()
                .compute(
                    &self.gpu,
                    poses_t3,
                    zealot_gpu_obs::FeetInputs {
                        sole_local: &sole,
                        prev_force: &pfv,
                        ground_h: &fgh,
                        sensed_force: &sfv,
                        have_prev: &hpp,
                        have_prev_force: &hpf,
                    },
                )
                .await
                .expect("gpu feet compute");
            // BIPED_FUSE_BENCH: measure the dispatch shape directly instead of
            // projecting it. Same three state kernels, same inputs, timed as
            // (a) one submit + one readback each -- the shape every binding was
            // built with -- versus (b) one shared encoder, one submit, then the
            // readbacks. The delta is the per-sync-point cost, which is the
            // whole question of whether the GPU step can beat the host block.
            if env_var("BIPED_FUSE_BENCH").is_ok() {
                let reps = 20u32;
                let t_un = std::time::Instant::now();
                for _ in 0..reps {
                    let pt = self.state.body_poses();
                    let _ = self.gpu_joints.as_mut().unwrap().compute(&self.gpu, pt, &hp).await;
                    let pt = self.state.body_poses();
                    let _ = self.gpu_base.as_mut().unwrap().compute(&self.gpu, pt, &hpp, &gh).await;
                }
                let d_un = t_un.elapsed().as_secs_f64() / reps as f64;

                let t_f = std::time::Instant::now();
                for _ in 0..reps {
                    let mut e = self.gpu.begin_encoding();
                    {
                        let pt = self.state.body_poses();
                        self.gpu_joints
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hp)
                            .expect("enc joints");
                    }
                    {
                        let pt = self.state.body_poses();
                        self.gpu_base
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hpp, &gh)
                            .expect("enc base");
                    }
                    self.gpu.submit(e).expect("submit fused");
                    let _ = self.gpu_joints.as_ref().unwrap().read(&self.gpu).await;
                    let _ = self.gpu_base.as_ref().unwrap().read(&self.gpu).await;
                }
                let d_f = t_f.elapsed().as_secs_f64() / reps as f64;

                // Third shape: fused encode, one submit, and only ONE readback.
                // The delta against `fused` (two readbacks) isolates the cost of
                // a readback from the cost of a dispatch, which decides whether
                // consolidating outputs into one buffer is worth building.
                let t_1r = std::time::Instant::now();
                for _ in 0..reps {
                    let mut e = self.gpu.begin_encoding();
                    {
                        let pt = self.state.body_poses();
                        self.gpu_joints
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hp)
                            .expect("enc joints");
                    }
                    {
                        let pt = self.state.body_poses();
                        self.gpu_base
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hpp, &gh)
                            .expect("enc base");
                    }
                    self.gpu.submit(e).expect("submit fused");
                    let _ = self.gpu_base.as_ref().unwrap().read(&self.gpu).await;
                }
                let d_1r = t_1r.elapsed().as_secs_f64() / reps as f64;

                // And the dispatch floor: encode + submit, NO readback at all.
                let t_0r = std::time::Instant::now();
                for _ in 0..reps {
                    let mut e = self.gpu.begin_encoding();
                    {
                        let pt = self.state.body_poses();
                        self.gpu_joints
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hp)
                            .expect("enc joints");
                    }
                    {
                        let pt = self.state.body_poses();
                        self.gpu_base
                            .as_mut()
                            .unwrap()
                            .encode(&self.gpu, &mut e, pt, &hpp, &gh)
                            .expect("enc base");
                    }
                    self.gpu.submit(e).expect("submit fused");
                }
                self.gpu.synchronize().expect("sync");
                let d_0r = t_0r.elapsed().as_secs_f64() / reps as f64;

                eprintln!(
                    "[fuse_bench] 2 kernels: unfused={:.3} fused2r={:.3} fused1r={:.3} dispatch_only={:.3} ms",
                    d_un * 1e3,
                    d_f * 1e3,
                    d_1r * 1e3,
                    d_0r * 1e3,
                );
            }

            const FEET_FIELDS: [&str; 11] = [
                "contact", "first", "air", "height", "planar", "tilt", "yaw", "x", "y", "vz",
                "dF",
            ];
            let mut wf = [0.0f32; 11];
            for e in 0..n {
                for i in 0..NUM_FEET {
                    for k in 0..11 {
                        let d = (gf[(i * 11 + k) * n + e] - computed[e].feet_dbg[i * 11 + k]).abs();
                        if d > wf[k] {
                            wf[k] = d;
                        }
                    }
                }
            }
            // The history is device-owned now, so check it hasn't drifted from
            // the host's copy — a divergence here corrupts every gait term
            // downstream and would otherwise be invisible.
            let (dev_air, dev_ltd) = self
                .gpu_feet
                .as_ref()
                .unwrap()
                .read_history(&self.gpu)
                .await
                .expect("read feet history");
            let (mut wair, mut wltd) = (0.0f32, 0.0f32);
            for e in 0..n {
                let want_ltd = if computed[e].td_foot >= 0 {
                    computed[e].td_foot as f32
                } else {
                    self.last_td_foot[e] as f32
                };
                wltd = wltd.max((dev_ltd[e] - want_ltd).abs());
                for i in 0..NUM_FEET {
                    wair = wair.max((dev_air[i * n + e] - computed[e].new_air[i]).abs());
                }
            }
            eprintln!("[verify_feet_hist] air_time={wair:.2e} last_td={wltd:.2e}");

            let mut msg = String::from("[verify_feet]");
            for k in 0..11 {
                msg.push_str(&format!(" {}={:.2e}", FEET_FIELDS[k], wf[k]));
            }
            eprintln!("{msg}");

            // ---- observation assembly (actor + critic) ----
            // The last host-side consumer of `state`. The step cue is NOT
            // derived here: it needs the terrain patch structure and the
            // per-env RNG stream, so the host still probes and the kernel takes
            // both cues as input.
            if self.gpu_observe.is_none() {
                let defaults: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].default_pos).collect();
                self.gpu_observe = Some(
                    zealot_gpu_obs::GpuObserve::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS as u32,
                        OBS_DIM,
                        CRITIC_OBS_DIM,
                        &defaults,
                    )
                    .expect("gpu observe"),
                );
            }
            let mut cmd_b = vec![0.0f32; 4 * n];
            let mut ph_b = vec![0.0f32; n];
            let mut cue_b = vec![0.0f32; 10 * n];
            for e in 0..n {
                cmd_b[e] = self.cmd[e].vx;
                cmd_b[n + e] = self.cmd[e].vy;
                cmd_b[2 * n + e] = self.cmd[e].yaw_rate;
                ph_b[e] = self.gait_phase[e];
                for k in 0..10 {
                    cue_b[k * n + e] = computed[e].cue_obs[k];
                }
            }
            let (qt, qdt) = {
                let j = self.gpu_joints.as_ref().unwrap();
                (j.q_tensor(), j.qd_tensor())
            };
            let bt = self.gpu_base.as_ref().unwrap().out_tensor();
            let mut obs_enc = self.gpu.begin_encoding();
            self.gpu_observe
                .as_mut()
                .unwrap()
                                .encode(&self.gpu, &mut obs_enc, 0, &la, &cmd_b, &ph_b, &cue_b, qt, qdt, bt)
                .expect("gpu observe encode");
            self.gpu.submit(obs_enc).expect("gpu observe submit");
            let (gobs, gcobs) = self
                .gpu_observe
                .as_ref()
                .unwrap()
                .read_back(&self.gpu)
                .await
                .expect("gpu observe readback");
            let (mut wo, mut wc) = (0.0f32, 0.0f32);
            let (mut wo_i, mut wc_i) = (0usize, 0usize);
            for e in 0..n {
                for r in 0..OBS_DIM {
                    let d = (gobs[r * n + e] - computed[e].obs[r]).abs();
                    if d > wo {
                        wo = d;
                        wo_i = r;
                    }
                }
                for r in 0..CRITIC_OBS_DIM {
                    let d = (gcobs[r * n + e] - computed[e].critic_obs[r]).abs();
                    if d > wc {
                        wc = d;
                        wc_i = r;
                    }
                }
            }
            // Occupancy guard: slots 48-52 (actor cue) and the critic's clean
            // cue are zero whenever no env sees a step edge, and a comparison
            // of zeros proves nothing. Report how many envs actually carry a
            // live cue so a clean diff can be trusted.
            let live_n = (0..n).filter(|&e| computed[e].cue_obs[4] > 0.5).count();
            let live_c = (0..n).filter(|&e| computed[e].cue_obs[9] > 0.5).count();
            eprintln!(
                "[verify_obs] actor={wo:.2e} (slot {wo_i}) critic={wc:.2e} (slot {wc_i}) cue_live={live_n}/{n} clean_live={live_c}/{n}"
            );

            // ---- self-contained per-foot reward terms ----
            if self.gpu_feet_terms.is_none() {
                self.gpu_feet_terms =
                    Some(zealot_gpu_obs::GpuRewardFeetTerms::new(&self.gpu, n).expect("feet terms"));
            }
            let wq2 = &self.task.weights;
            let fp = zealot_obs_shaders::RewardFeetParams {
                n_envs: n as u32,
                dt: dtc,
                w_flight: wq2.flight,
                w_foot_slip: wq2.foot_slip,
                w_force_rate: wq2.force_rate,
                force_rate_deadband: wq2.force_rate_deadband,
                w_foot_orientation: wq2.foot_orientation,
                w_feet_yaw_mean: wq2.feet_yaw_mean,
                w_feet_yaw_diff: wq2.feet_yaw_diff,
                w_feet_distance: wq2.feet_distance,
                feet_distance_ref: wq2.feet_distance_ref,
                w_touchdown_vz: wq2.touchdown_vz,
                touchdown_vz_h: wq2.touchdown_vz_h,
                touchdown_vz_ok: wq2.touchdown_vz_ok,
                pad0: 0,
                pad1: 0,
            };
            let ft = self
                .gpu_feet_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, fp, &gf, &gb)
                .await
                .expect("gpu feet terms compute");
            // comps order: flight 13, foot_slip 15, force_rate 28,
            // foot_orientation 17, feet_yaw_mean 18, feet_yaw_diff 27,
            // feet_distance 19, touchdown_vz 30.
            const FEET_TERMS: [(usize, usize); 8] = [
                (13, 0), (15, 1), (28, 2), (17, 3), (18, 4), (27, 5), (19, 6), (30, 7),
            ];
            let mut wft = [0.0f32; 8];
            for (ti, (comp, row)) in FEET_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (ft[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wft[ti] {
                        wft[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_feet_terms] flight={:.2e} slip={:.2e} dF={:.2e} orient={:.2e} yaw_mean={:.2e} yaw_diff={:.2e} dist={:.2e} td_vz={:.2e}",
                wft[0], wft[1], wft[2], wft[3], wft[4], wft[5], wft[6], wft[7]
            );

            // ---- gated gait terms ----
            if self.gpu_gait_terms.is_none() {
                self.gpu_gait_terms =
                    Some(zealot_gpu_obs::GpuRewardGaitTerms::new(&self.gpu, n).expect("gait terms"));
            }
            let wq3 = &self.task.weights;
            let gp = zealot_obs_shaders::RewardGaitParams {
                n_envs: n as u32,
                dt: dtc,
                w_air_time: wq3.air_time,
                w_single_support: wq3.single_support,
                w_stand_planted: wq3.stand_planted,
                w_foot_clearance: wq3.foot_clearance,
                foot_clearance_target: wq3.foot_clearance_target,
                w_gait_clock: wq3.gait_clock,
                gait_swing_ratio: wq3.gait_swing_ratio,
                max_swing_s: 0.45,
                foot_rest_h: 0.035,
                step_clear_margin: 0.05,
                step_relax_dist: self.task.step_relax_dist,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            // aux: phase, progress, cmd_speed_full, cue_height, stepping — the
            // per-env scalars the gates read.
            let mut aux = vec![0.0f32; 5 * n];
            for e in 0..n {
                let c = &self.cmd[e];
                let sp2 = c.vx * c.vx + c.vy * c.vy;
                let bq = &computed[e].base_dbg;
                // body-frame planar velocity, as the host's `progress` uses.
                let vb = zealot_env::math::quat_rotate_inv(
                    [bq[0], bq[1], bq[2], bq[3]],
                    [bq[4], bq[5], bq[6]],
                );
                aux[e] = self.gait_phase[e];
                aux[n + e] = if sp2 > 1e-6 {
                    ((vb[0] * c.vx + vb[1] * c.vy) / sp2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                aux[2 * n + e] = c.speed();
                aux[3 * n + e] = computed[e].cue_h_dbg;
                aux[4 * n + e] = computed[e].stepping_dbg;
            }
            let gt = self
                .gpu_gait_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, gp, &gf, &aux)
                .await
                .expect("gpu gait terms compute");
            const GAIT_TERMS: [(usize, usize); 5] =
                [(12, 0), (14, 1), (26, 2), (16, 3), (25, 4)];
            let mut wgt = [0.0f32; 5];
            for (ti, (comp, row)) in GAIT_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (gt[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wgt[ti] {
                        wgt[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_gait_terms] air_time={:.2e} single_support={:.2e} stand_planted={:.2e} clearance={:.2e} gait_clock={:.2e}",
                wgt[0], wgt[1], wgt[2], wgt[3], wgt[4]
            );

            // ---- self_coll / chest_ang_vel / termination ----
            if self.gpu_misc_terms.is_none() {
                let pa: Vec<u32> =
                    self.idx.self_collision_pairs.iter().map(|&(a, _)| a as u32).collect();
                let pb: Vec<u32> =
                    self.idx.self_collision_pairs.iter().map(|&(_, b)| b as u32).collect();
                self.gpu_misc_terms = Some(
                    zealot_gpu_obs::GpuRewardMiscTerms::new(&self.gpu, n, &pa, &pb)
                        .expect("misc terms"),
                );
            }
            let chest_link = self.idx.chest_link as usize;
            let mut pchest = vec![0.0f32; 4 * n];
            let mut fellv = vec![0u32; n];
            for e in 0..n {
                let pq = self.prev_body_poses[e * cpb + chest_link].rotation;
                pchest[e] = pq.x;
                pchest[n + e] = pq.y;
                pchest[2 * n + e] = pq.z;
                pchest[3 * n + e] = pq.w;
                fellv[e] = computed[e].fell as u32;
            }
            let mp = zealot_obs_shaders::RewardMiscParams {
                n_envs: n as u32,
                colliders_per_env: cpb as u32,
                n_pairs: self.idx.self_collision_pairs.len() as u32,
                dt: dtc,
                sc_margin: self.knobs.sc_margin,
                sc_weight: self.knobs.sc_weight,
                chest_link: self.idx.chest_link,
                chest_w: self.knobs.chest_w,
                w_termination: self.task.weights.termination,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let poses_t4 = self.state.body_poses();
            let mt = self
                .gpu_misc_terms
                .as_mut()
                .unwrap()
                .compute(&self.gpu, mp, poses_t4, &pchest, &hpp, &fellv)
                .await
                .expect("gpu misc terms compute");
            const MISC_TERMS: [(usize, usize); 3] = [(22, 0), (31, 1), (23, 2)];
            let mut wm = [0.0f32; 3];
            for (ti, (comp, row)) in MISC_TERMS.iter().enumerate() {
                for e in 0..n {
                    let d = (mt[row * n + e] - computed[e].comps[*comp]).abs();
                    if d > wm[ti] {
                        wm[ti] = d;
                    }
                }
            }
            eprintln!(
                "[verify_misc_terms] self_coll={:.2e} chest_ang_vel={:.2e} termination={:.2e}",
                wm[0], wm[1], wm[2]
            );
        }

        // ---- GPU reward consumption (BIPED_GPU_REWARD=1) ----
        // The parallel block above skipped every reward-term computation;
        // compute all seven device term groups here — state kernels + term
        // kernels in ONE encoder and ONE submit (the shape `BIPED_FUSE_BENCH`
        // measured) — then read the term matrices back and fill `comps`.
        // Host state assembly, obs, the step cue, termination detection and
        // the feet bookkeeping above are unchanged; the device values were
        // verified element-wise against the host terms (BIPED_VERIFY_REWARD).
        if self.use_gpu_reward {
            let t = Instant::now();
            let n = self.n;
            let jn = NUM_JOINTS * n;
            let bps = (self.state.body_poses().len() as usize / n) as u32;
            let cpb = self.idx.colliders_per_batch as usize;
            let dtc = self.task.control_dt();

            // -- lazy init: identical constructors to the verify path --
            if self.gpu_reward.is_none() {
                self.gpu_reward = Some(
                    zealot_gpu_obs::GpuRewardTerms::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &self.task.hip_yawroll_idx(),
                    )
                    .expect("gpu reward terms"),
                );
            }
            if self.gpu_joints.is_none() {
                let rest: Vec<glamx::Vec4> = (0..NUM_JOINTS)
                    .map(|k| {
                        let r = self.idx.actuated_rest_quat[k];
                        glamx::Vec4::new(r.x, r.y, r.z, r.w)
                    })
                    .collect();
                let children: Vec<u32> =
                    (0..NUM_JOINTS).map(|k| self.idx.actuated[k].0).collect();
                self.gpu_joints = Some(
                    zealot_gpu_obs::GpuJointState::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &self.idx.actuated_parent_links,
                        &children,
                        &rest,
                        bps,
                        dtc,
                    )
                    .expect("gpu joint state"),
                );
            }
            if self.gpu_joint_terms.is_none() {
                let dpos: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].default_pos).collect();
                let lo: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].pos_limit.0 * 0.9).collect();
                let hi: Vec<f32> =
                    (0..NUM_JOINTS).map(|k| self.robot.joints[k].pos_limit.1 * 0.9).collect();
                self.gpu_joint_terms = Some(
                    zealot_gpu_obs::GpuRewardJointTerms::new(
                        &self.gpu,
                        n,
                        NUM_JOINTS,
                        &dpos,
                        &lo,
                        &hi,
                        &self.task.hip_yawroll_idx(),
                        &(0..NUM_JOINTS)
                            .map(|k| self.robot.mirror[k] as u32)
                            .collect::<Vec<_>>(),
                        &(0..NUM_JOINTS)
                            .map(|k| self.robot.mirror_sign[k])
                            .collect::<Vec<_>>(),
                    )
                    .expect("gpu joint terms"),
                );
            }
            if self.gpu_torque_terms.is_none() {
                let w_torques = env_f32("BIPED_W_TORQUES").unwrap_or(1e-4);
                let w_ankle_t = env_f32("BIPED_W_ANKLE_TORQUES").unwrap_or(1.5e-3);
                let w_ankle_roll = env_f32("BIPED_W_ANKLE_ROLL_TORQUES").unwrap_or(0.0);
                let w_knee = self.knobs.w_knee_torques;
                let js = &self.task.robot.joints;
                let kp: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].kp).collect();
                let kd: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].kd).collect();
                let eff: Vec<f32> = (0..NUM_JOINTS).map(|k| js[k].effort_limit).collect();
                let wl = vec![w_torques; NUM_JOINTS];
                let wa: Vec<f32> = (0..NUM_JOINTS)
                    .map(|k| {
                        let nm = &js[k].name;
                        if nm.contains("ankle") {
                            let roll = nm.contains("ankle_roll") || nm.contains("anklex");
                            w_ankle_t + if roll { w_ankle_roll } else { 0.0 }
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let wk: Vec<f32> = (0..NUM_JOINTS)
                    .map(|k| if js[k].name.contains("knee") { w_knee } else { 0.0 })
                    .collect();
                self.gpu_torque_terms = Some(
                    zealot_gpu_obs::GpuRewardTorqueTerms::new(
                        &self.gpu, n, NUM_JOINTS, &kp, &kd, &eff, &wl, &wa, &wk,
                    )
                    .expect("gpu torque terms"),
                );
            }
            if self.gpu_base.is_none() {
                self.gpu_base = Some(
                    zealot_gpu_obs::GpuBaseState::new(
                        &self.gpu,
                        n,
                        bps,
                        self.idx.torso_link,
                        dtc,
                    )
                    .expect("gpu base state"),
                );
            }
            if self.gpu_feet.is_none() {
                let ff = self.robot.foot_forward_local;
                self.gpu_feet = Some(
                    zealot_gpu_obs::GpuFeetState::new(
                        &self.gpu,
                        n,
                        NUM_FEET,
                        &(0..NUM_FEET).map(|i| self.idx.foot_links[i]).collect::<Vec<_>>(),
                        &[ff[0], ff[1], ff[2]],
                        zealot_obs_shaders::FeetStateParams {
                            n_envs: n as u32,
                            colliders_per_env: cpb as u32,
                            torso_link: self.idx.torso_link,
                            control_dt: dtc,
                            contact_z: env_var("BIPED_CONTACT_Z")
                                .ok()
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0.05),
                            contact_force_n: self.contact_force_n,
                            contact_sense: self.contact_sense as u32,
                            body_weight: self.robot.total_mass * 9.81,
                        },
                    )
                    .expect("gpu feet"),
                );
            }
            if self.gpu_base_terms.is_none() {
                self.gpu_base_terms =
                    Some(zealot_gpu_obs::GpuRewardBaseTerms::new(&self.gpu, n).expect("base terms"));
            }
            if self.gpu_feet_terms.is_none() {
                self.gpu_feet_terms =
                    Some(zealot_gpu_obs::GpuRewardFeetTerms::new(&self.gpu, n).expect("feet terms"));
            }
            if self.gpu_gait_terms.is_none() {
                self.gpu_gait_terms =
                    Some(zealot_gpu_obs::GpuRewardGaitTerms::new(&self.gpu, n).expect("gait terms"));
            }
            if self.gpu_misc_terms.is_none() {
                let pa: Vec<u32> =
                    self.idx.self_collision_pairs.iter().map(|&(a, _)| a as u32).collect();
                let pb: Vec<u32> =
                    self.idx.self_collision_pairs.iter().map(|&(_, b)| b as u32).collect();
                self.gpu_misc_terms = Some(
                    zealot_gpu_obs::GpuRewardMiscTerms::new(&self.gpu, n, &pa, &pb)
                        .expect("misc terms"),
                );
            }

            // -- host-staged inputs (same layouts as the verify path) --
            let mut la = vec![0.0f32; jn];
            let mut pa = vec![0.0f32; jn];
            let mut p2a = vec![0.0f32; jn];
            for e in 0..n {
                for k in 0..NUM_JOINTS {
                    la[k * n + e] = self.last_action[e][k];
                    pa[k * n + e] = self.prev_action[e][k];
                    p2a[k * n + e] = self.prev2_action[e][k];
                }
            }
            let hp: Vec<u32> = (0..n).map(|e| self.has_prev_joint_pos[e] as u32).collect();
            let hpp: Vec<u32> = (0..n).map(|e| self.has_prev_pose[e] as u32).collect();
            let gh: Vec<f32> = (0..n)
                .map(|e| {
                    let tp = &poses[e * bps as usize + self.idx.torso_link as usize];
                    self.terrain
                        .as_ref()
                        .map_or(0.0, |ter| ter.strip_for(e).height(tp.translation.x, tp.translation.y))
                })
                .collect();
            let mut sole = vec![0.0f32; NUM_FEET * 3 * n];
            let mut fgh = vec![0.0f32; NUM_FEET * n];
            let mut sfv = vec![0.0f32; NUM_FEET * n];
            let mut pfv = vec![0.0f32; NUM_FEET * n];
            for e in 0..n {
                for i in 0..NUM_FEET {
                    let sl = self.foot_sole_local[e][i];
                    sole[(i * 3) * n + e] = sl.x;
                    sole[(i * 3 + 1) * n + e] = sl.y;
                    sole[(i * 3 + 2) * n + e] = sl.z;
                    let link = self.idx.foot_links[i] as usize;
                    let fpz = &poses[e * cpb + link].translation;
                    fgh[i * n + e] = self
                        .terrain
                        .as_ref()
                        .map_or(0.0, |ter| ter.strip_for(e).height(fpz.x, fpz.y));
                    sfv[i * n + e] = self.sensed_force[e][i];
                    pfv[i * n + e] = self.prev_sensed_force[e][i];
                }
            }
            let hpf: Vec<u32> = (0..n).map(|e| self.has_prev_force[e] as u32).collect();
            let w = &self.task.weights;
            let (w_ar, w_hip, w_arr) =
                (w.action_rate, w.action_rate_hipz_hipx, w.action_rate_rate);
            let (wp, wl, wv, wb) = (w.pose, w.dof_pos_limits, w.dof_vel, w.bilateral_symmetry);
            let gate = self.task.sym_yaw_gate;
            let yaws: Vec<f32> = (0..n).map(|e| self.cmd[e].yaw_rate).collect();
            let (tw, atw, pw) =
                (self.torque_scale, self.knobs.ankle_torque_w, self.knobs.power_w);
            let tr = self.targets_row.clone();
            let tk = &self.task;
            let bp = zealot_obs_shaders::RewardBaseParams {
                n_envs: n as u32,
                dt: dtc,
                w_track_lin: tk.weights.track_lin_vel,
                w_forward_progress: tk.weights.forward_progress,
                w_track_ang: tk.weights.track_ang_vel,
                w_upright: tk.weights.upright,
                w_base_height: tk.weights.base_height,
                w_body_ang_vel: tk.weights.body_ang_vel,
                w_lin_vel_z: tk.weights.lin_vel_z,
                std_lin: tk.stds.lin_vel,
                std_ang: tk.stds.ang_vel,
                std_base_h: tk.stds.base_height,
                std_upright: tk.stds.upright,
                step_std_base_h: tk.step_std_base_h,
                step_std_upright: tk.step_std_upright,
                step_relax_dist: tk.step_relax_dist,
                h_target_stand: tk.weights.base_height_target_stand,
                h_target_walk: tk.weights.base_height_target,
                pad0: 0,
                pad1: 0,
            };
            let fp = zealot_obs_shaders::RewardFeetParams {
                n_envs: n as u32,
                dt: dtc,
                w_flight: w.flight,
                w_foot_slip: w.foot_slip,
                w_force_rate: w.force_rate,
                force_rate_deadband: w.force_rate_deadband,
                w_foot_orientation: w.foot_orientation,
                w_feet_yaw_mean: w.feet_yaw_mean,
                w_feet_yaw_diff: w.feet_yaw_diff,
                w_feet_distance: w.feet_distance,
                feet_distance_ref: w.feet_distance_ref,
                w_touchdown_vz: w.touchdown_vz,
                touchdown_vz_h: w.touchdown_vz_h,
                touchdown_vz_ok: w.touchdown_vz_ok,
                pad0: 0,
                pad1: 0,
            };
            let gp = zealot_obs_shaders::RewardGaitParams {
                n_envs: n as u32,
                dt: dtc,
                w_air_time: w.air_time,
                w_single_support: w.single_support,
                w_stand_planted: w.stand_planted,
                w_foot_clearance: w.foot_clearance,
                foot_clearance_target: w.foot_clearance_target,
                w_gait_clock: w.gait_clock,
                gait_swing_ratio: w.gait_swing_ratio,
                max_swing_s: 0.45,
                foot_rest_h: 0.035,
                step_clear_margin: 0.05,
                step_relax_dist: self.task.step_relax_dist,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let mp = zealot_obs_shaders::RewardMiscParams {
                n_envs: n as u32,
                colliders_per_env: cpb as u32,
                n_pairs: self.idx.self_collision_pairs.len() as u32,
                dt: dtc,
                sc_margin: self.knobs.sc_margin,
                sc_weight: self.knobs.sc_weight,
                chest_link: self.idx.chest_link,
                chest_w: self.knobs.chest_w,
                w_termination: self.task.weights.termination,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            let mut cmdb = vec![0.0f32; 4 * n];
            let mut cueb = vec![0.0f32; 4 * n];
            let mut aux = vec![0.0f32; 5 * n];
            for e in 0..n {
                let c = &self.cmd[e];
                cmdb[e] = c.vx;
                cmdb[n + e] = c.vy;
                cmdb[2 * n + e] = c.yaw_rate;
                cmdb[3 * n + e] = c.speed();
                let cd = computed[e].cue_dbg;
                cueb[e] = cd[0];
                cueb[n + e] = cd[1];
                cueb[2 * n + e] = cd[2];
                cueb[3 * n + e] = cd[3];
                let sp2 = c.vx * c.vx + c.vy * c.vy;
                let bq = &computed[e].base_dbg;
                let vb = zealot_env::math::quat_rotate_inv(
                    [bq[0], bq[1], bq[2], bq[3]],
                    [bq[4], bq[5], bq[6]],
                );
                aux[e] = self.gait_phase[e];
                aux[n + e] = if sp2 > 1e-6 {
                    ((vb[0] * c.vx + vb[1] * c.vy) / sp2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                aux[2 * n + e] = c.speed();
                aux[3 * n + e] = computed[e].cue_h_dbg;
                aux[4 * n + e] = computed[e].stepping_dbg;
            }
            let chest_link = self.idx.chest_link as usize;
            let mut pchest = vec![0.0f32; 4 * n];
            let mut fellv = vec![0u32; n];
            for e in 0..n {
                let pq = self.prev_body_poses[e * cpb + chest_link].rotation;
                pchest[e] = pq.x;
                pchest[n + e] = pq.y;
                pchest[2 * n + e] = pq.z;
                pchest[3 * n + e] = pq.w;
                fellv[e] = computed[e].fell as u32;
            }

            // -- one fused encoder, one submit --
            {
                let mut enc = self.gpu.begin_encoding();
                {
                    let pt = self.state.body_poses();
                    self.gpu_joints
                        .as_mut()
                        .unwrap()
                        .encode(&self.gpu, &mut enc, pt, &hp)
                        .expect("enc joints");
                }
                {
                    let pt = self.state.body_poses();
                    self.gpu_base
                        .as_mut()
                        .unwrap()
                        .encode(&self.gpu, &mut enc, pt, &hpp, &gh)
                        .expect("enc base");
                }
                {
                    let pt = self.state.body_poses();
                    self.gpu_feet
                        .as_mut()
                        .unwrap()
                        .encode(
                            &self.gpu,
                            &mut enc,
                            pt,
                            zealot_gpu_obs::FeetInputs {
                                sole_local: &sole,
                                prev_force: &pfv,
                                ground_h: &fgh,
                                sensed_force: &sfv,
                                have_prev: &hpp,
                                have_prev_force: &hpf,
                            },
                        )
                        .expect("enc feet");
                }
                self.gpu_reward
                    .as_mut()
                    .unwrap()
                    .encode(&self.gpu, &mut enc, &la, &pa, &p2a, dtc, w_ar, w_hip, w_arr)
                    .expect("enc action terms");
                {
                    let joints_ref = self.gpu_joints.as_ref().unwrap();
                    self.gpu_joint_terms
                        .as_mut()
                        .unwrap()
                        .encode(&self.gpu, &mut enc, joints_ref, dtc, wp, wl, wv, wb, gate, &yaws)
                        .expect("enc joint terms");
                    self.gpu_torque_terms
                        .as_mut()
                        .unwrap()
                        .encode(&self.gpu, &mut enc, joints_ref, &tr, dtc, tw, atw, pw)
                        .expect("enc torque terms");
                }
                {
                    let base_t = self.gpu_base.as_ref().unwrap().out_tensor();
                    let feet_t = self.gpu_feet.as_ref().unwrap().out_tensor();
                    self.gpu_base_terms
                        .as_mut()
                        .unwrap()
                        .encode_dev(&self.gpu, &mut enc, bp, base_t, &cmdb, &cueb)
                        .expect("enc base terms");
                    self.gpu_feet_terms
                        .as_mut()
                        .unwrap()
                        .encode_dev(&self.gpu, &mut enc, fp, feet_t, base_t)
                        .expect("enc feet terms");
                    self.gpu_gait_terms
                        .as_mut()
                        .unwrap()
                        .encode_dev(&self.gpu, &mut enc, gp, feet_t, &aux)
                        .expect("enc gait terms");
                }
                {
                    let pt = self.state.body_poses();
                    self.gpu_misc_terms
                        .as_mut()
                        .unwrap()
                        .encode(&self.gpu, &mut enc, mp, pt, &pchest, &hpp, &fellv)
                        .expect("enc misc terms");
                }
                self.gpu.submit(enc).expect("gpu reward submit");
            }

            // -- term-matrix readbacks (the only D2H the rewards need) --
            let rt = self.gpu_reward.as_ref().unwrap().read(&self.gpu).await.expect("rd action");
            let jt = self.gpu_joint_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd joint");
            let tt = self.gpu_torque_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd torque");
            let btm = self.gpu_base_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd base");
            let ftm = self.gpu_feet_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd feet");
            let gtm = self.gpu_gait_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd gait");
            let mtm = self.gpu_misc_terms.as_ref().unwrap().read(&self.gpu).await.expect("rd misc");

            // -- scatter into comps by the verified mappings; total = Σ comps --
            const ACT_T: [(usize, usize); 3] = [(6, 0), (7, 1), (29, 2)];
            const JNT_T: [(usize, usize); 4] = [(4, 0), (10, 1), (11, 2), (5, 3)];
            const TRQ_T: [(usize, usize); 3] = [(20, 0), (21, 1), (24, 2)];
            const BAS_T: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (3, 3), (8, 4), (9, 5)];
            const FEE_T: [(usize, usize); 8] =
                [(13, 0), (15, 1), (28, 2), (17, 3), (18, 4), (27, 5), (19, 6), (30, 7)];
            const GAI_T: [(usize, usize); 5] = [(12, 0), (14, 1), (26, 2), (16, 3), (25, 4)];
            const MIS_T: [(usize, usize); 3] = [(22, 0), (31, 1), (23, 2)];
            for e in 0..n {
                let c = &mut computed[e];
                for &(comp, row) in &ACT_T {
                    c.comps[comp] = rt[row * n + e];
                }
                for &(comp, row) in &JNT_T {
                    c.comps[comp] = jt[row * n + e];
                }
                for &(comp, row) in &TRQ_T {
                    c.comps[comp] = tt[row * n + e];
                }
                for &(comp, row) in &BAS_T {
                    c.comps[comp] = btm[row * n + e];
                }
                for &(comp, row) in &FEE_T {
                    c.comps[comp] = ftm[row * n + e];
                }
                for &(comp, row) in &GAI_T {
                    c.comps[comp] = gtm[row * n + e];
                }
                for &(comp, row) in &MIS_T {
                    c.comps[comp] = mtm[row * n + e];
                }
                c.reward = c.comps.iter().sum();
            }
            self.timings.par_compute_ns += t.elapsed().as_nanos() as u64;
        }

        // (5) Serial commit: per-env mutable state + StepOut assembly.
        let t = Instant::now();
        let cpb = self.idx.colliders_per_batch as usize;
        // Observation noise (sensor DR): uniform additive noise on the ACTOR obs
        // only (the critic keeps a clean privileged obs — asymmetric PPO). Models
        // encoder quantization / IMU noise so the policy can't overfit to
        // pixel-perfect proprioception. Amplitudes mirror Isaac Lab's UniformNoise
        // for proprioceptive humanoid obs; BIPED_OBS_NOISE scales them (0 = off).
        let obs_noise: f32 = env_var("BIPED_OBS_NOISE")
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
                // base_ang_vel / gyro [45..48]: ±0.2 rad/s. This block predates
                // the gyro (added with the 48-dim frame in v22) and stopped at
                // projected_gravity, so every gyro-era policy trained on a
                // PERFECT IMU while every other channel was noised -- the one
                // proprioceptive input with no sim-to-real margin. Measured on
                // v24: 0.01 rad/s of gyro noise moves the action by 0.016,
                // ~0.8 N-m of knee torque ripple at action_scale 0.25.
                // 0.2 completes Isaac Lab's UniformNoise set the amplitudes
                // above are taken from (joint_pos 0.01 / joint_vel 1.5 /
                // ang_vel 0.2 / gravity 0.05).
                if OBS_DIM >= 48 {
                    for v in &mut c.obs[45..48] {
                        *v += rng.range(-0.2, 0.2) * obs_noise;
                    }
                }
            }
            // (obs history is stacked in one parallel pass after this loop —
            // it must see the final NOISED frame, and per-env ring buffers
            // are disjoint, so batching after the serial mutations is
            // bit-identical to pushing here. Serial it cost ~12 ms/step.)
            self.air_time[e] = c.new_air;
            if c.td_foot >= 0 {
                self.last_td_foot[e] = c.td_foot;
            }
            // Gait clock: fully derived from the command, matching the
            // deploy contract. Standing command (< 0.1 m/s) freezes the phase
            // (the obs clock stops asking for swings the gated reward no
            // longer pays); a move command advances it with a cadence that
            // scales with commanded speed — 0.8 s/cycle at 0.1 m/s down to
            // 0.55 s at the full 0.5 m/s (stride rate carries part of the
            // speed, as in biological gait). Deterministic from command
            // history: any deploy stack reproduces it without an estimator.
            // Same magnitude the task's standing predicate uses (INCLUDING
            // yaw rate) — a turn-in-place command must tick the clock, or the
            // gait reward scores against a frozen phase.
            let cmd_speed = self.cmd[e].speed();
            if cmd_speed >= 0.1 {
                // The cap must track the command range: with BIPED_VX raised to
                // 0.8 but the cap left at 0.5, every command from 0.5 to 0.8
                // gets an IDENTICAL cadence, so the policy can only go 60%
                // faster by lengthening its stride. Raise both together.
                // NOTE: this mapping is part of the observation contract -- the
                // sim2sim harnesses and the LeRobot controller hardcode it too.
                let t = ((cmd_speed.min(self.gait_speed_cap)) - 0.1) / 0.4;
                let period = (GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * t)
                    .max(GAIT_PERIOD_MIN);
                self.gait_phase[e] =
                    (self.gait_phase[e] + self.task.control_dt() / period).fract();
            }
            self.prev_joint_pos[e] = c.new_joint_pos;
            self.has_prev_joint_pos[e] = true;
            // Snapshot poses for this env into prev_body_poses for the next
            // step's finite-diff base / foot velocities.
            let env_base = e * cpb;
            self.prev_body_poses[env_base..env_base + cpb]
                .copy_from_slice(&poses[env_base..env_base + cpb]);
            self.has_prev_pose[e] = true;
            self.prev2_action[e] = self.prev_action[e];
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
        // Obs history, batched: push each env's final (noised) frame and
        // replace it with the stacked window — in PARALLEL over the disjoint
        // per-env ring buffers (the serial in-loop version cost ~12 ms/step
        // at 4096 envs). Runs before any caller-side resets, exactly where
        // the serial pushes sat, so semantics are bit-identical.
        if let Some(hist) = &mut self.obs_hist {
            hist.env_views()
                .into_par_iter()
                .zip(outs.par_iter_mut())
                .for_each(|(mut view, o)| view.push_stacked_replace(&mut o.obs));
        }
        self.timings.serial_commit_ns += t.elapsed().as_nanos() as u64;
        self.timings.steps += 1;
        // Per-step so the stance-phase path/drift accumulation is real (each call
        // does 2 readbacks — fine for a short diagnostic run, not for training).
        if env_var("BIPED_DEBUG_CONTACT").is_ok() {
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
    /// Reset one env. Thin wrapper over [`Self::reset_envs`] — prefer the
    /// batched form in the rollout loop, where the per-dispatch overhead of a
    /// reset dwarfs the few kilobytes it moves.
    pub async fn reset_env(&mut self, env: usize) -> (Vec<f32>, Vec<f32>) {
        let mut out = self.reset_envs(&[env]).await;
        out.pop().expect("reset_envs returns one entry per env")
    }

    /// Reset many envs with ONE multibody scatter dispatch for the whole set.
    ///
    /// Split into three phases so the GPU work batches: per-env host draws +
    /// snapshot preparation, a single batched scatter, then per-env host
    /// bookkeeping. Every env draws from its own RNG stream, so batching does
    /// not perturb per-env determinism.
    pub async fn reset_envs(&mut self, envs: &[usize]) -> Vec<(Vec<f32>, Vec<f32>)> {
        if envs.is_empty() {
            return Vec::new();
        }
        // Max envs per scatter dispatch. Default = the whole set (one dispatch
        // per rollout step); `BIPED_RESET_BATCH=1` forces the old
        // one-dispatch-per-env cadence, which is the A/B for isolating batching
        // bugs from everything else.
        let chunk = env_var("BIPED_RESET_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|c| *c > 0)
            .unwrap_or(usize::MAX);

        // Phase split for `[prof-reset]` (BIPED_RESET_PROF=1) — prepare is host
        // RNG + spawn placement, scatter is staging build + dispatch, finish is
        // per-env bookkeeping + the post-reset obs.
        let prof = env_var("BIPED_RESET_PROF").is_ok();
        let (mut t_prep, mut t_scatter, mut t_finish) = (0u128, 0u128, 0u128);

        let mut out = Vec::with_capacity(envs.len());
        for group in envs.chunks(chunk.min(envs.len().max(1))) {
            let t0 = std::time::Instant::now();
            let mut ts = Vec::with_capacity(group.len());
            let mut plans = Vec::with_capacity(group.len());
            for &e in group {
                let (t, offset, vels) = self.reset_prepare(e);
                ts.push(t);
                plans.push((offset, vels));
            }
            if prof {
                t_prep += t0.elapsed().as_micros();
            }
            let t1 = std::time::Instant::now();
            // Borrow the templates directly — the spawn teleport is applied by
            // the scatter kernel and the velocity override lands in the staging
            // build, so nothing is cloned per reset.
            let dpb = self.state.multibodies_mut().dofs_per_batch_count() as usize;
            let mut meta = Vec::with_capacity(group.len());
            let mut offs = Vec::with_capacity(group.len());
            // Flat [env x dofs_per_batch]; zeros = the template build-time
            // (at rest) velocities, matching the old per-env None fallback.
            let mut dof_vels = vec![0.0f32; group.len() * dpb];
            for (i, ((e, (offset, vels)), t)) in
                group.iter().zip(plans.iter()).zip(ts.iter()).enumerate()
            {
                meta.push((*e as u32, *t as u32));
                offs.push(*offset);
                if let Some(v) = vels.as_deref() {
                    let m = v.len().min(dpb);
                    dof_vels[i * dpb..i * dpb + m].copy_from_slice(&v[..m]);
                }
            }
            self.state
                .reset_envs_from_templates(&self.gpu, &meta, &offs, &dof_vels);
            // The feet history is device-owned, so a reset must clear it there
            // too — otherwise air time keeps accumulating across the teleport
            // and every gait term reads a stale swing.
            if let Some(gf) = self.gpu_feet.as_mut() {
                let ids: Vec<u32> = group.iter().map(|&e| e as u32).collect();
                let mut seed = vec![0.0f32; NUM_FEET * self.n];
                for e in 0..self.n {
                    for i in 0..NUM_FEET {
                        seed[i * self.n + e] = self.sensed_force[e][i];
                    }
                }
                gf.reset(&self.gpu, &ids, &seed).await.expect("gpu feet reset");
            }
            if prof {
                t_scatter += t1.elapsed().as_micros();
            }
            let t2 = std::time::Instant::now();
            for (i, &e) in group.iter().enumerate() {
                out.push(self.reset_finish(e, ts[i]).await);
            }
            if prof {
                t_finish += t2.elapsed().as_micros();
            }
        }
        if prof && !envs.is_empty() {
            eprintln!(
                "[prof-reset] n={} prepare={:.1}ms scatter={:.1}ms finish={:.1}ms",
                envs.len(),
                t_prep as f64 / 1000.0,
                t_scatter as f64 / 1000.0,
                t_finish as f64 / 1000.0,
            );
        }
        out
    }

    /// Host half of a reset: the RNG draws + spawn placement, returning the
    /// template index, the spawn offset and any velocity override. No GPU work,
    /// and no snapshot clone — the kernel consumes the template in place.
    fn reset_prepare(&mut self, env: usize) -> (usize, NexusVector, Option<Vec<f32>>) {
        // Pick a template via this env's RNG so reset choices are deterministic
        // for a given seed.
        let r = self.rng[env].range(0.0, 1.0);
        let t = ((r * self.templates.len() as f32) as usize).min(self.templates.len() - 1);
        // AGILE reset-velocity randomization: the fresh env starts in motion
        // rather than at rest. Layout per env: [0..3) root lin, [3..6) root
        // ang, [6..dpb) joint velocities.
        //
        // Drawn BEFORE the reset so it can ride along in the reset's own
        // staging upload. The alternative — reset, then write `dof_state` —
        // costs one 4-byte H2D copy PER DOF (the buffer is batch-interleaved,
        // dof d of env e at d·n + e, so an env's velocities are strided), which
        // measured ~65 µs of the ~90 µs per reset and 42% of a cold iteration.
        // Draw order on `self.rng[env]` is unchanged (the template pick above
        // is the only earlier draw; `terrain_spawn_offset` uses the terrain's
        // own per-env stream), so this stays bit-identical to the old path.
        let vels: Option<Vec<f32>> = if self.reset_vel {
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
            Some(v)
        } else {
            None
        };
        // Teleport to the env's current difficulty patch (level was already
        // updated by the curriculum when the episode ended).
        let off = if self.terrain.is_some() {
            Some(self.terrain_spawn_offset(env, t))
        } else {
            None
        };
        (t, off.unwrap_or(NexusVector::ZERO), vels)
    }

    /// Host half of a reset that runs AFTER the scatter: per-env bookkeeping and
    /// the post-reset observation.
    async fn reset_finish(&mut self, env: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        // Mirror the template's sole-normal so update_feet's tilt makes sense.
        // Cached per-template (constant) — NO per-reset rapier-scene rebuild.
        self.foot_sole_local[env] = self.template_foot_sole[t];

        // Reset host state.
        self.cmd[env] = eval_cmd_override().unwrap_or_else(|| self.sampler.sample(&mut self.rng[env]));
        self.arm_reset(env); // respawn holds home — kill playback instantly
        self.arm_resample(env); // then re-roll against the fresh command
        self.step_count[env] = 0;
        self.resample_at[env] = self
            .sampler
            .resample_steps(&mut self.rng[env], self.task.control_dt());
        self.last_action[env] = [0.0; NUM_JOINTS];
        self.prev_action[env] = [0.0; NUM_JOINTS];
        self.prev2_action[env] = [0.0; NUM_JOINTS];
        self.air_time[env] = [0.0; NUM_FEET];
        // Force-sensed contact: seed the new episode as planted (half body
        // weight per foot) — the spawn pose stands on both soles, and the
        // real sensed value arrives with the first step's readback.
        self.sensed_force[env] = [0.5 * self.robot.total_mass * 9.81; NUM_FEET];
        self.prev_sensed_force[env] = self.sensed_force[env];
        self.has_prev_force[env] = false; // no ΔF across the reset teleport
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
            if env_var("BIPED_VERIFY_RESET").is_ok() {
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
        // TERRAIN-AWARE. The snapshot is the as-built pose at the ORIGIN, and
        // the terrain strip starts 8 m away at x = STRIP_X0, so a plain
        // snapshot reset silently undoes the on-terrain teleport that `new()`
        // performed -- the rollout then walks on flat ground while every log
        // line still says "terrain ENABLED". That produced an eval where the
        // step-cue A/B returned byte-identical numbers for cue on and cue off
        // because the robot never met a step at all.
        if self.terrain.is_some() {
            let off = self.terrain_spawn_offset(e, 0);
            self.state
                .reset_env_from_snapshot_offset(&self.gpu, e as u32, &self.template_snapshots[0], off);
        } else {
            self.state
                .reset_env_from_snapshot(&self.gpu, e as u32, &self.template_snapshots[0]);
        }
        self.foot_sole_local[e] = self.idx.foot_sole_local;
        self.cmd[e] = VelocityCommand::default();
        self.arm_reset(e); // eval reset: deterministic home arms, no playback
        // BIPED_EVAL_ARM_CLIP=<idx>: pin clip <idx> from t=0 for THIS eval
        // rollout (deterministic — no RNG), so renders/evals can show the
        // upper-body playback that the training-side arm_reset suppresses.
        if let Some(am) = &self.arm_motion {
            if let Some(c) = std::env::var("BIPED_EVAL_ARM_CLIP")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                let n_held = self.idx.held.len();
                self.arm_clip[e] = c % am.clips.len() as u32;
                self.arm_time[e] = 0.0;
                self.arm_active[e] = true;
                // Fade in from home (arm_reset above just staged it).
                self.arm_from[e * n_held..(e + 1) * n_held]
                    .copy_from_slice(&self.arm_staged[e * n_held..(e + 1) * n_held]);
                self.arm_blend[e] = 0.0;
            }
        }
        self.step_count[e] = 0;
        // Pin the resample so the command stays where the caller pins it.
        self.resample_at[e] = u32::MAX;
        self.last_action[e] = [0.0; NUM_JOINTS];
        self.prev_action[e] = [0.0; NUM_JOINTS];
        self.prev2_action[e] = [0.0; NUM_JOINTS];
        self.air_time[e] = [0.0; NUM_FEET];
        self.sensed_force[e] = [0.5 * self.robot.total_mass * 9.81; NUM_FEET];
        self.prev_sensed_force[e] = self.sensed_force[e];
        self.has_prev_force[e] = false; // no ΔF across the reset teleport
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
    /// Per-link world rotations (x, y, z, w), same ordering as
    /// [`Self::body_positions_for`]. Needed by the native replay renderer to
    /// place link MESHES -- positions alone can only place spheres.
    pub fn body_rotations_for(&self, e: usize, poses: &[NexusPose]) -> Vec<[f32; 4]> {
        let cpb = self.idx.colliders_per_batch as usize;
        let base = e * cpb;
        (0..self.idx.mjcf_to_link.len())
            .map(|i| {
                let r = poses[base + i].rotation;
                [r.x, r.y, r.z, r.w]
            })
            .collect()
    }

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
    /*
     * GPU-resident rollout accessors (browser demo's GPU-obs path): expose
     * the physics buffers + a physics-only step so an external obs-assembly
     * kernel + GPU policy can close the control loop without any per-step
     * host readback.
     */

    /// The khal backend the physics runs on.
    pub fn gpu_backend(&self) -> &khal::backend::GpuBackend {
        &self.gpu
    }

    /// Absolute assembly-dof index of each policy joint (root 6 DOFs first).
    pub fn actuated_assembly_dofs(&self) -> [u32; NUM_JOINTS] {
        self.idx.joint_dof_offset
    }

    /// Multibody link id of each policy joint's child link.
    pub fn actuated_link_ids(&self) -> Vec<u32> {
        self.idx.actuated.iter().map(|(l, _)| *l).collect()
    }

    /// GPU buffers the obs-assembly kernel reads, from one borrow:
    /// (generalized coordinates, per-link SoA workspace).
    pub fn resident_buffers(
        &mut self,
    ) -> (
        &vortx::tensor::Tensor<f32>,
        &vortx::tensor::Tensor<nexus3d::rbd::glamx::Vec4>,
    ) {
        let mb = self.state.multibodies_mut();
        // Two disjoint field borrows through one &mut — split via raw parts.
        let dv: *const vortx::tensor::Tensor<f32> = mb.dof_values();
        let ws = mb.links_workspace_buffer();
        // SAFETY: both point into `mb`'s distinct fields; neither aliases the
        // other, and the returned lifetimes are tied to &mut self.
        (unsafe { &*dv }, ws)
    }

    /// The `body_poses` buffer `snapshot()` reads. Exposed so a renderer can
    /// pipeline that readback itself (start the copy at the end of one frame,
    /// take the result at the start of the next) instead of fencing on the
    /// blocking `snapshot()`.
    pub fn body_poses_buffer(&self) -> &khal::backend::GpuBuffer<NexusPose> {
        self.state.body_poses().buffer()
    }

    /// Scatter policy PD targets (row-major [12 × n], radians) from a GPU
    /// buffer straight into the motor constraints — the GPU analog of the
    /// per-env `stage_motor_position` + `flush_links_static`.
    pub fn scatter_targets_gpu(&mut self, targets: &vortx::tensor::Tensor<f32>) {
        let links = self.actuated_link_ids();
        self.state
            .multibodies_mut()
            .scatter_motor_targets_gpu(&self.gpu, targets, &links, JointAxis::AngZ as u32)
            .expect("scatter_motor_targets_gpu");
    }

    /// Encode the target scatter into an existing encoder (single-submit
    /// control steps).
    pub fn encode_scatter_targets(
        &mut self,
        enc: &mut <KhalGpuBackend as Backend>::Encoder,
        targets: &vortx::tensor::Tensor<f32>,
    ) {
        let links = self.actuated_link_ids();
        // Split borrow: the encoder call needs &self.gpu and &mut multibodies.
        let gpu: *const KhalGpuBackend = &self.gpu;
        self.state
            .multibodies_mut()
            .encode_scatter_motor_targets(
                unsafe { &*gpu },
                enc,
                targets,
                &links,
                JointAxis::AngZ as u32,
            )
            .expect("encode_scatter_motor_targets");
    }

    /// Upload this step's actuated targets and scatter them into the motor
    /// constraints with a kernel.
    ///
    /// Replaces the per-step `flush_links_static`, which re-uploaded the ENTIRE
    /// `links_static` mirror (`links_per_batch × n` structs, each carrying a
    /// full `GenericJoint` + mass properties — tens of MB) just to change
    /// `NUM_JOINTS` floats per env. This uploads exactly those floats.
    fn flush_motor_targets(&mut self) {
        let n = self.n;
        if self.motor_targets_gpu.is_none() {
            self.motor_targets_gpu = Some(
                vortx::tensor::Tensor::matrix(
                    &self.gpu,
                    NUM_JOINTS as u32,
                    n as u32,
                    &self.targets_row,
                    khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_DST,
                )
                .expect("motor target buffer"),
            );
        } else {
            let buf = self.motor_targets_gpu.as_mut().unwrap();
            self.gpu
                .write_buffer(buf.buffer_mut(), 0, &self.targets_row)
                .expect("motor target upload");
        }
        let targets = self.motor_targets_gpu.take().expect("motor targets");
        self.scatter_targets_gpu(&targets);
        self.motor_targets_gpu = Some(targets);
    }

    /// Advance physics one control step (`decimation` substeps) WITHOUT any
    /// observation/reward readback — motor targets must already be staged
    /// (e.g. via [`Self::scatter_targets_gpu`]).
    pub fn step_physics_only(&mut self) {
        for _ in 0..self.task.decimation {
            let _ = self.pipeline.step(&self.gpu, &mut self.state, None);
        }
        self.global_step += 1;
    }

    /// [`Self::step_physics_only`], recorded into a caller-owned encoder and
    /// not submitted: the resident demo folds all `decimation` substeps AND
    /// the obs/policy/scatter work into ONE queue submission. Each submit is
    /// a wasm→browser crossing (~1 ms of main-thread time in Chrome); at 50
    /// control steps × ~10 submits/step they were the demo's frame budget.
    pub fn step_physics_encoded(&mut self, enc: &mut <KhalGpuBackend as khal::backend::Backend>::Encoder) {
        for _ in 0..self.task.decimation {
            let _ = self
                .pipeline
                .step_encoded(&self.gpu, &mut self.state, None, enc);
        }
        self.global_step += 1;
    }

    /// [`Self::step_physics_only`] with per-pass GPU timestamps recorded into
    /// `ts` — the `?prof=1` breakdown. One profiled step at a time: the
    /// caller gates on `ts.is_idle()` and reads back non-blockingly.
    pub fn step_physics_profiled(&mut self, ts: &mut khal::backend::GpuTimestamps) {
        for _ in 0..self.task.decimation {
            let _ = self.pipeline.step(&self.gpu, &mut self.state, Some(ts));
        }
        self.global_step += 1;
    }

    /// One submit per SUBSTEP (each fully merged internally). The middle
    /// ground between `step_physics_only` (per-PHASE submits, ~10 crossings a
    /// control step) and `step_physics_encoded` (everything in one submit,
    /// which frees the CPU but stalls the GPU for the whole encode — measured
    /// as higher fps and LOWER sim throughput). Substep-granular submission
    /// keeps the GPU chewing substep k while the CPU encodes k+1.
    pub fn step_physics_substep_submits(&mut self) {
        for _ in 0..self.task.decimation {
            let mut enc = self.gpu.begin_encoding();
            let _ = self
                .pipeline
                .step_encoded(&self.gpu, &mut self.state, None, &mut enc);
            let _ = self.gpu.submit(enc);
        }
        self.global_step += 1;
    }

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
        let cbuf = self
            .state
            .multibodies_mut()
            .dbg_contact_constraints()
            .buffer();
        // SAFETY: MultibodyContactConstraint is Pod (plain f32/u32 fields);
        // zeroed is a valid bit pattern. Debug-only readback scratch.
        let mut v: Vec<NexusMbContact> = (0..cbuf.len())
            .map(|_| unsafe { std::mem::zeroed() })
            .collect();
        self.gpu
            .slow_read_buffer(cbuf, &mut v)
            .await
            .expect("mb contact constraints readback");
        (Vec::new(), v)
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
    pub async fn dbg_links_static(&mut self) -> Vec<nexus3d::rbd::shaders::dynamics::MultibodyLinkStatic> {
        let buf = self.state.multibodies_mut().dbg_links_static().buffer();
        let mut v: Vec<nexus3d::rbd::shaders::dynamics::MultibodyLinkStatic> =
            (0..buf.len() as usize).map(|_| unsafe { std::mem::zeroed() }).collect();
        self.gpu.slow_read_buffer(buf, &mut v).await.expect("links static readback");
        v
    }

    pub async fn dbg_body_jacobians(&mut self) -> Vec<f32> {
        let buf = self.state.multibodies_mut().dbg_body_jacobians().buffer();
        let mut v = vec![0f32; buf.len() as usize];
        self.gpu
            .slow_read_buffer(buf, &mut v)
            .await
            .expect("body jacobians readback");
        v
    }

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
    // In the browser (wasm), asking for desktop-sized buffer limits can exceed
    // the adapter's caps and fail device creation outright. The web demo runs
    // a single env, so 256 MB is plenty; the shader-side limits stay.
    #[cfg(target_arch = "wasm32")]
    const MAX_BUFFER: u64 = 256 * 1024 * 1024;
    #[cfg(not(target_arch = "wasm32"))]
    const MAX_BUFFER: u64 = 1_200_000_000;
    let limits = wgpu::Limits {
        max_buffer_size: MAX_BUFFER,
        max_storage_buffer_binding_size: MAX_BUFFER,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    // Clamp each requested limit to the adapter's (browsers reject requests
    // above them — e.g. Chrome caps maxStorageBuffersPerShaderStage at 10).
    #[cfg(target_arch = "wasm32")]
    let limits = {
        let instance = wgpu::Instance::default();
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => {
                let a = adapter.limits();
                wgpu::Limits {
                    max_buffer_size: limits.max_buffer_size.min(a.max_buffer_size),
                    max_storage_buffer_binding_size: limits
                        .max_storage_buffer_binding_size
                        .min(a.max_storage_buffer_binding_size),
                    max_storage_buffers_per_shader_stage: limits
                        .max_storage_buffers_per_shader_stage
                        .min(a.max_storage_buffers_per_shader_stage),
                    max_compute_workgroup_storage_size: limits
                        .max_compute_workgroup_storage_size
                        .min(a.max_compute_workgroup_storage_size),
                    ..limits
                }
            }
            Err(_) => limits,
        }
    };
    // TIMESTAMP_QUERY when the adapter offers it: per-pass GPU timings for
    // the `?prof=1` breakdown (Chrome exposes it; harmless where absent).
    let feats = {
        #[cfg(target_arch = "wasm32")]
        {
            let mut f = wgpu::Features::default();
            f |= wgpu::Features::TIMESTAMP_QUERY;
            f
        }
        #[cfg(not(target_arch = "wasm32"))]
        wgpu::Features::default()
    };
    let mut bk = match KhalGpuBackend::auto(feats, limits.clone()).await {
        Ok(bk) => bk,
        // Adapter without timestamps: fall back rather than fail to start.
        Err(_) => KhalGpuBackend::auto(wgpu::Features::default(), limits)
            .await
            .expect("init GPU backend"),
    };
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

/// `BIPED_EVAL_CMD="vx,vy,yaw"`: pin every sampled velocity command to a fixed
/// value (benchmark/eval runs; overrides the command curriculum).
fn eval_cmd_override() -> Option<VelocityCommand> {
    static CMD: std::sync::OnceLock<Option<VelocityCommand>> = std::sync::OnceLock::new();
    *CMD.get_or_init(|| {
        let v = env_var("BIPED_EVAL_CMD").ok()?;
        let p: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (p.len() == 3).then(|| VelocityCommand { vx: p[0], vy: p[1], yaw_rate: p[2] })
    })
}

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
    if env_var("BIPED_AGILE_DR").map_or(true, |v| v != "0") {
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
            foot_shape_id: sample_foot_shape(rng),
        };
    }
    // BIPED_SPAWN_DR scales the initial-pose tilt/height randomization (default
    // 1.0). Set to 0.0 to start every episode upright at nominal height — used to
    // test whether aggressive spawn DR is what's preventing the policy from
    // getting a learning gradient (the rng draws are still consumed, so dynamics
    // DR and determinism are unchanged).
    let sdr: f32 = env_var("BIPED_SPAWN_DR")
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
    let friction = match env_var("BIPED_FRICTION")
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
    let mdr: f32 = env_var("BIPED_MASS_DR")
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
        contact_natural_frequency: match env_var("BIPED_CONTACT_FREQ")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(f) => {
                let _ = rng.range(10.0, 50.0);
                f
            }
            None => rng.range(10.0, 50.0),
        },
        contact_damping_ratio: match env_var("BIPED_CONTACT_DAMP")
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
            let hw: f32 = env_var("BIPED_ASYM_DR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.15);
            let mut a = [1.0f32; NUM_JOINTS];
            for v in a.iter_mut() {
                *v = 1.0 + rng.range(-hw, hw);
            }
            a
        },
        foot_shape_id: sample_foot_shape(rng),
    }
}

/// Per-template foot-shape draw: `BIPED_FOOT_SHAPE=dr` → 50/50 box/capsule
/// (structural-error DR — see the DrParams field doc); any other value → 0
/// (the global env default, no draw consumed so existing streams are
/// unchanged in non-dr mode... draw IS consumed in dr mode only).
fn sample_foot_shape(rng: &mut Lcg) -> u8 {
    if env_var("BIPED_FOOT_SHAPE").as_deref() == Ok("dr") {
        if rng.range(0.0, 1.0) < 0.5 { 1 } else { 2 }
    } else {
        0
    }
}
