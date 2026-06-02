//! Feasibility gate on **nexus GPU physics**, built from the curated MJCF model.
//! Same scene `biped_smoke_cpu`/`biped_mjcf` validate on rapier CPU: floating-base
//! multibody, MJCF inertials, revolute PD motors (the `set_motor_position`
//! patch), foot contact boxes, Z-up.
//!
//! ## Root cause of the original pinned-torso bug
//!
//! Initial runs of this example reproduced dimforge/nexus-rustgpu#1 — torso
//! pinned at SPAWN_Z forever. With the dynamic-root fix (`70cbaef`) already in
//! `origin/cuda`, that pinning *should* be gone, but it wasn't. Tracing the
//! GPU buffers (see the diagnostic readbacks below) localized it to
//! `gen_accelerations[2] == 0` on every step → `gravity_and_lu` was skipping
//! gravity assembly for the root, gated by `if inv_mass_x != 0.0`. Reading
//! `links_mprops` confirmed it: `inv_mass = (0, 0, 0)` for *every* link.
//!
//! That zero comes from **rapier**, not nexus. `RigidBody::mass_properties()`
//! returns the body's `local_mprops` field, which is populated by
//! `recompute_mass_properties_from_colliders` — normally called automatically
//! during rapier's own step pipeline. Since this example hands the rapier scene
//! directly to nexus *without* stepping rapier first, `local_mprops` stays at
//! `MassProperties::default()` (zero mass / zero inertia) and nexus reads the
//! zero through to its mass-matrix and gravity-force kernels.
//!
//! **Fix:** call `rb.recompute_mass_properties_from_colliders(&colliders)` on
//! every body before constructing `GpuPhysicsState::from_rapier`. Done below.
//!
//! Run: `cargo run --release --example biped_nexus --features biped_gpu -- [steps]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose as NexusPose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;
use roxmltree::Node;
use std::collections::HashMap;
use zealot_env::robots::LeRobotBipedal;

const DT: f32 = 1.0 / 200.0;
const SOLVER_ITERS: u32 = 16;
const SPAWN_Z: f32 = 0.72;

// --- MJCF parsing (focused reader for this model) ---------------------------

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
fn av(node: &Node, attr: &str, default: Vec3) -> Vec3 {
    node.attribute(attr).map_or(default, |s| {
        let f = floats(s);
        Vec3::new(f[0], f[1], f[2])
    })
}
fn aq(node: &Node) -> Rotation {
    node.attribute("quat").map_or(Rotation::IDENTITY, |s| {
        let f = floats(s); // MuJoCo: w x y z
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
                com = av(&c, "pos", Vec3::ZERO);
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
            local_pos: av(node, "pos", Vec3::ZERO),
            local_quat: aq(node),
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

fn parse_mjcf(xml: &str) -> Vec<MjBody> {
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

// --- scene build ------------------------------------------------------------

struct Scene {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    /// nexus link id (collider iteration order) of the torso and feet.
    torso_id: u32,
    foot_ids: Vec<u32>,
    /// `(nexus link id, name)` of each actuated joint's child.
    actuated: Vec<(u32, String)>,
}

fn gain(robot: &LeRobotBipedal, name: &str) -> (f32, f32, f32) {
    robot
        .joints
        .iter()
        .find(|j| j.name == name)
        .map(|j| (j.kp, j.kd, j.effort_limit))
        .unwrap_or((50.0, 1.0, 20.0))
}

fn build_scene(mjcf: &[MjBody], robot: &LeRobotBipedal) -> Scene {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();

    // FK world poses (root lifted to SPAWN_Z).
    let mut world = Vec::with_capacity(mjcf.len());
    for b in mjcf {
        let w = match b.parent {
            None => Pose::from_parts(Vec3::new(0.0, 0.0, SPAWN_Z), Rotation::IDENTITY),
            Some(p) => world[p] * Pose::from_parts(b.local_pos, b.local_quat),
        };
        world.push(w);
    }

    // Bodies, each with exactly ONE collider (nexus requires it): a foot box for
    // feet, an inert tiny box otherwise (so every link gets a mass/pose entry).
    let mut handles = Vec::with_capacity(mjcf.len());
    let mut torso_handle = RigidBodyHandle::invalid();
    let mut foot_handles = Vec::new();
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
            // Inert placeholder (collides with nothing) — just to register the link.
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.01, 0.01, 0.01)
                    .density(0.0)
                    .collision_groups(InteractionGroups::none()),
                h,
                &mut bodies,
            );
        } else {
            // Foot: one box covering the union AABB of the foot's capsules.
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
            foot_handles.push(h);
        }
    }

    // Revolute multibody joints (free child-local AngZ) with build-time PD motors;
    // `set_motor_position` then updates targets at runtime on the GPU.
    let locked = JointAxesMask::LIN_X
        | JointAxesMask::LIN_Y
        | JointAxesMask::LIN_Z
        | JointAxesMask::ANG_X
        | JointAxesMask::ANG_Y;
    let mut actuated_h: Vec<(RigidBodyHandle, String)> = Vec::new();
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some(jname)) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let (kp, kd, effort) = gain(robot, jname);
        let mut joint = GenericJointBuilder::new(locked)
            .local_frame1(Pose::from_parts(b.local_pos, b.local_quat))
            .local_frame2(Pose::IDENTITY)
            .build();
        joint.set_motor_model(JointAxis::AngZ, MotorModel::ForceBased);
        joint.set_motor_position(JointAxis::AngZ, 0.0, kp, kd);
        joint.set_motor_max_force(JointAxis::AngZ, effort);
        multibody.insert(handles[parent], handles[i], joint, true);
        actuated_h.push((handles[i], jname.clone()));
    }

    // Ground (Z-up), top at z = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );

    // nexus indexes bodies by their collider's position in `colliders.iter()`.
    let mut nexus_id = HashMap::new();
    for (idx, (_, co)) in colliders.iter().enumerate() {
        if let Some(p) = co.parent() {
            nexus_id.insert(p, idx as u32);
        }
    }
    let id = |h: RigidBodyHandle| *nexus_id.get(&h).expect("nexus id");

    Scene {
        torso_id: id(torso_handle),
        foot_ids: foot_handles.iter().map(|&h| id(h)).collect(),
        actuated: actuated_h.into_iter().map(|(h, n)| (id(h), n)).collect(),
        bodies,
        colliders,
        impulse,
        multibody,
    }
}

// --- GPU boilerplate --------------------------------------------------------

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

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<NexusPose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("poses")
}

async fn read_dofs(gpu: &KhalGpuBackend, state: &mut GpuPhysicsState) -> Vec<f32> {
    gpu.slow_read_vec(state.multibodies_mut().dof_values().buffer())
        .await
        .expect("dofs")
}

/// Diagnostic snapshot of link 0's workspace state on the GPU. While debugging
/// dimforge/nexus-rustgpu#1: lets us see whether the integrator actually writes
/// to `ws.coords` for the dynamic root vs whether everything in the pipeline
/// stays at zero. `MultibodyLinkWorkspace` doesn't implement `Default`, but
/// it's `Copy + bytemuck::Zeroable`, so an `unsafe` zero-init is safe.
async fn read_ws_link0(
    gpu: &KhalGpuBackend,
    state: &mut GpuPhysicsState,
) -> nexus3d::rbd::shaders::dynamics::MultibodyLinkWorkspace {
    use nexus3d::rbd::shaders::dynamics::MultibodyLinkWorkspace;
    let mut out: [MultibodyLinkWorkspace; 1] = [unsafe { std::mem::zeroed() }];
    gpu.slow_read_buffer(state.multibodies_mut().links_workspace().buffer(), &mut out)
        .await
        .expect("links_workspace");
    out[0]
}

async fn read_dof_state(gpu: &KhalGpuBackend, state: &mut GpuPhysicsState) -> Vec<f32> {
    gpu.slow_read_vec(state.multibodies_mut().dof_state().buffer())
        .await
        .expect("dof_state")
}

async fn read_accels(gpu: &KhalGpuBackend, state: &mut GpuPhysicsState) -> Vec<f32> {
    gpu.slow_read_vec(state.multibodies_mut().gen_accelerations().buffer())
        .await
        .expect("gen_accelerations")
}

/// Read the local mass properties for link 0 — checks whether the dynamic root's
/// inv_mass is silently zero (which would make `gravity_and_lu` skip the gravity
/// force assembly for that link).
async fn read_mprops_link0(
    gpu: &KhalGpuBackend,
    state: &mut GpuPhysicsState,
) -> nexus3d::rbd::shaders::dynamics::LocalMassProperties {
    use nexus3d::rbd::shaders::dynamics::LocalMassProperties;
    let mut out: [LocalMassProperties; 1] = [unsafe { std::mem::zeroed() }];
    gpu.slow_read_buffer(state.multibodies_mut().links_mprops().buffer(), &mut out)
        .await
        .expect("links_mprops");
    out[0]
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let xml = {
        let home = std::env::var("HOME").unwrap_or_default();
        std::fs::read_to_string(format!(
            "{home}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml"
        ))
        .expect("read mjcf")
    };
    let mjcf = parse_mjcf(&xml);
    let robot = LeRobotBipedal::new();

    pollster::block_on(async {
        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
        let mut scene = build_scene(&mjcf, &robot);
        // rapier's `local_mprops` is populated by its step pipeline — since we
        // hand the scene to nexus without stepping rapier first, we have to
        // recompute it ourselves. Otherwise every rb's `mass_properties()`
        // returns 0, nexus's `links_mprops` reads zero, and `gravity_and_lu`'s
        // `if inv_mass_x != 0.0` guard skips force assembly forever. Root cause
        // for dimforge/nexus-rustgpu#1 in our setup.
        {
            let colliders = scene.colliders.clone();
            for (_, rb) in scene.bodies.iter_mut() {
                rb.recompute_mass_properties_from_colliders(&colliders);
            }
        }
        println!(
            "scene: {} bodies, {} colliders; torso id {}, feet {:?}, {} actuated joints",
            scene.bodies.len(),
            scene.colliders.len(),
            scene.torso_id,
            scene.foot_ids,
            scene.actuated.len(),
        );

        let mut sp = GpuSimParams::default();
        sp.dt = DT;
        sp.num_solver_iterations = SOLVER_ITERS;
        let envs = vec![(
            &scene.bodies,
            &scene.colliders,
            &scene.impulse,
            &scene.multibody,
            &sp,
        )];
        // Sanity check: every dynamic body now has non-zero local_mprops mass
        // (the recompute call above does the work rapier's own step would do).
        let total_mass: f32 = scene
            .bodies
            .iter()
            .filter(|(_, b)| b.is_dynamic())
            .map(|(_, b)| b.mass_properties().local_mprops.mass())
            .sum();
        println!("total robot mass = {total_mass:.3} kg (should match the URDF spec)");
        assert!(
            total_mass > 1.0,
            "rapier local_mprops still zero — recompute didn't fire"
        );
        let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);
        state.multibodies_mut().set_gravity(&gpu, [0.0, 0.0, -9.81]);

        let p0 = read_poses(&gpu, &state).await;
        let z = |p: &[NexusPose], i: u32| {
            p.get(i as usize)
                .map(|x| x.translation.z)
                .unwrap_or(f32::NAN)
        };
        println!(
            "initial: torso_z={:.3}  feet_z={:?}",
            z(&p0, scene.torso_id),
            scene
                .foot_ids
                .iter()
                .map(|&i| (z(&p0, i) * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
        );

        // Initial workspace + DOF buffer snapshot (link 0 = dynamic root torso).
        let ws0_init = read_ws_link0(&gpu, &mut state).await;
        let dof_state_init = read_dof_state(&gpu, &mut state).await;
        let dof_values_init = read_dofs(&gpu, &mut state).await;
        let mp0 = read_mprops_link0(&gpu, &mut state).await;
        println!(
            "init link0: coords[0..6]={:?}  joint_rot={:?}  local_to_world.t={:?}",
            &ws0_init.coords[..6],
            ws0_init.joint_rot,
            ws0_init.local_to_world.translation,
        );
        println!(
            "init dof_state[0..6]={:?}  dof_values[0..6]={:?}",
            &dof_state_init[..6.min(dof_state_init.len())],
            &dof_values_init[..6.min(dof_values_init.len())],
        );
        println!(
            "init link0 mprops: inv_mass={:?}  inv_principal_inertia={:?}  com={:?}",
            mp0.inv_mass, mp0.inv_principal_inertia, mp0.com,
        );

        // Second half: command a −0.3 rad crouch on every joint, to check whether
        // the *joints* respond (isolating any freeze to the free base).
        let crouch_at = steps / 2;
        println!(
            "\n{:>5}  {:>8}  {:>8}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}",
            "step", "t(s)", "torso_z", "min_foot_z", "max|dq|", "ws.coord2", "dof_v[2]", "acc[2]"
        );
        for step in 0..steps {
            let target = if step < crouch_at { 0.0 } else { -0.3 };
            {
                let mm = state.multibodies_mut();
                for (lid, _) in &scene.actuated {
                    let _ = mm.set_motor_position(&gpu, 0, *lid, JointAxis::AngZ, target);
                }
            }
            let _ = pipeline.step(&gpu, &mut state, None).await;
            gpu.synchronize().expect("sync");
            pipeline.auto_resize_buffers(&gpu, &mut state).await;

            if step < 3 || step % 25 == 0 || step == steps - 1 || step == crouch_at {
                let p = read_poses(&gpu, &state).await;
                let tz = z(&p, scene.torso_id);
                let fz = scene
                    .foot_ids
                    .iter()
                    .map(|&i| z(&p, i))
                    .fold(f32::INFINITY, f32::min);
                let dofs = read_dofs(&gpu, &mut state).await;
                let max_dq = dofs
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let ws0 = read_ws_link0(&gpu, &mut state).await;
                let dof_state = read_dof_state(&gpu, &mut state).await;
                let accels = read_accels(&gpu, &mut state).await;
                let nan = p.iter().any(|x| !x.translation.z.is_finite());
                let tag = if step == crouch_at {
                    "  <- crouch -0.3"
                } else {
                    ""
                };
                println!(
                    "{:>5}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}  {:>9.4}  {:>9.4}  {:>9.4}{}{}",
                    step,
                    step as f32 * DT,
                    tz,
                    fz,
                    max_dq,
                    ws0.coords[2],
                    dof_state.get(2).copied().unwrap_or(f32::NAN),
                    accels.get(2).copied().unwrap_or(f32::NAN),
                    if nan { "  NON-FINITE" } else { "" },
                    tag
                );
                if nan {
                    println!("\nFAILED: non-finite state.");
                    return;
                }
            }
        }
        println!(
            "\nVerdict: torso_z should drop off {SPAWN_Z:.2} on step 1 under gravity\n\
             and oscillate as the passively-unstable biped falls. ws.coord2 / dof_v[2]\n\
             / acc[2] should all become non-zero by step 1. If torso_z stays pinned,\n\
             the rapier `recompute_mass_properties_from_colliders` call above didn't\n\
             actually populate `local_mprops` — check that first."
        );
    });
}
