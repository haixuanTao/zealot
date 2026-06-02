//! Stage 2 — the feasibility gate: does the LeRobot bipedal *stand* on nexus's
//! GPU contact solver while PD-holding its default pose?
//!
//! This is the first time the real robot meets the real physics engine. It:
//! 1. loads the bipedal URDF (`rapier3d-urdf`) as a **floating-base** multibody,
//!    converting the 92 STL collision meshes to cheap **Obb** (cuboid) proxies and
//!    rotating the Z-up URDF into nexus's Y-up world;
//! 2. configures a **force-based PD position motor** on each of the 12 leg joints
//!    using the `LeRobotBipedal` gains;
//! 3. drops it onto a cuboid ground, uploads to the GPU, and steps physics while
//!    holding the default (all-zero) pose via the new
//!    `set_motor_position` patch;
//! 4. reads back the link poses + joint coordinates and reports whether the robot
//!    holds height (stands) or collapses / blows up (NaN).
//!
//! A short final phase commands a uniform crouch target to confirm the runtime
//! `set_motor_position` setter actually drives the joints.
//!
//! Run: `cargo run --release --example biped_smoke --features biped -- [spawn_height] [steps]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};
use std::collections::HashMap;
use zealot_env::robots::LeRobotBipedal;

const DT: f32 = 1.0 / 200.0;
const SOLVER_ITERS: u32 = 16;

/// PD gain + torque cap for one of the 12 actuated joints, looked up by name from
/// the robot spec. Returns `None` for non-actuated joints (base, fixed frames).
fn gain_for(robot: &LeRobotBipedal, name: &str) -> Option<(f32, f32, f32)> {
    robot
        .joints
        .iter()
        .find(|j| j.name == name)
        .map(|j| (j.kp, j.kd, j.effort_limit))
}

/// nexus requires exactly one collider per body and doesn't support `Compound`
/// shapes, but each URDF link carries many mesh (→Obb) colliders. Collapse each
/// link's colliders into a single body-local axis-aligned `Cuboid` covering their
/// union AABB. Density 0 so the URDF inertials (already applied) stay authoritative.
fn collapse_link_colliders(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    link_bodies: &[RigidBodyHandle],
) {
    let mut islands = IslandManager::new();
    for &body in link_bodies {
        let handles: Vec<ColliderHandle> = bodies[body].colliders().to_vec();
        // Union AABB of this link's mesh colliders (in the body frame). Empty links
        // (massless frames) get a tiny placeholder so every body has exactly one
        // collider — nexus requires that and indexes bodies by their collider.
        let mut aabb = Aabb::new_invalid();
        for h in &handles {
            let c = &colliders[*h];
            let pose = c.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
            let a = c.shared_shape().compute_aabb(&pose);
            aabb.take_point(a.mins);
            aabb.take_point(a.maxs);
        }
        for h in handles {
            colliders.remove(h, &mut islands, bodies, false);
        }
        let (he, center) = if aabb.mins.x.is_finite() {
            (aabb.half_extents(), aabb.center())
        } else {
            (Vec3::splat(1e-3), Vec3::ZERO)
        };
        let he = Vec3::new(he.x.max(1e-3), he.y.max(1e-3), he.z.max(1e-3));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(he.x, he.y, he.z)
                .position(Pose::from_parts(center, Rotation::IDENTITY))
                .density(0.0)
                .friction(1.0),
            body,
            bodies,
        );
    }
}

/// Build the rapier-handle → nexus-id map exactly as nexus's `from_rapier` does:
/// the nexus id of a body is the position of its (single) collider in
/// `colliders.iter()`. Used to address joints (`set_motor_position`) and to index
/// the pose readback. Must be called on the *final* collider set.
fn nexus_body_ids(colliders: &ColliderSet) -> HashMap<RigidBodyHandle, u32> {
    let mut map = HashMap::new();
    for (idx, (_, co)) in colliders.iter().enumerate() {
        if let Some(parent) = co.parent() {
            map.insert(parent, idx as u32);
        }
    }
    map
}

/// Build the scene: floating-base bipedal (PD motors configured) + cuboid ground.
/// Returns the rapier sets and, for each actuated joint, `(link_id, name)` so the
/// runtime motor setter can address it.
/// What `build_scene` hands back: the rapier sets, the actuated `(link_id, name)`
/// map, and the body indices of the base (torso) and the two feet for tracking.
struct Scene {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    actuated: Vec<(u32, String)>,
    base_idx: u32,
    foot_idx: Vec<u32>,
}

fn build_scene(robot: &LeRobotBipedal, spawn_height: f32) -> Scene {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    // The URDF references meshes as `package://assets/<file>.stl`; strip the
    // scheme and point the loader at the sibling `assets/` directory.
    let urdf_path = robot.urdf_path();
    let urdf_dir = urdf_path.parent().expect("urdf dir");
    let assets_dir = urdf_dir.join("assets");
    let urdf_str = std::fs::read_to_string(&urdf_path)
        .unwrap_or_else(|e| panic!("read urdf {}: {e}", urdf_path.display()))
        .replace("package://assets/", "");

    let options = UrdfLoaderOptions {
        create_colliders_from_collision_shapes: true,
        create_colliders_from_visual_shapes: false,
        apply_imported_mass_props: true,
        // Floating base: keep the root's 6 free DOFs (we want it to fall/balance,
        // not be pinned in the air).
        make_roots_fixed: false,
        // Cheap convex proxies for physics instead of raw triangle meshes.
        mesh_converter: Some(MeshConverter::Obb),
        // URDF is Z-up; nexus is Y-up. Rotate −90° about X and lift to spawn height.
        shift: Pose::from_parts(
            Vec3::new(0.0, spawn_height, 0.0),
            Rotation::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        ),
        ..UrdfLoaderOptions::default()
    };

    let (mut urdf_robot, urdf) = UrdfRobot::from_str(&urdf_str, options, &assets_dir)
        .unwrap_or_else(|e| panic!("parse urdf: {e}"));

    // Configure a force-based PD position motor (target 0) on each actuated joint's
    // free angular axis (rapier-urdf maps every revolute DOF to AngX).
    for (i, uj) in urdf_robot.joints.iter_mut().enumerate() {
        let name = &urdf.joints[i].name;
        if let Some((kp, kd, max_force)) = gain_for(robot, name) {
            uj.joint
                .set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
            uj.joint.set_motor_position(JointAxis::AngX, 0.0, kp, kd);
            uj.joint.set_motor_max_force(JointAxis::AngX, max_force);
        }
    }

    let handles = urdf_robot.insert_using_multibody_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody_joints,
        UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
    );

    // Capture the rapier body handles for each actuated joint's child, the base
    // (parent of a hip-yaw joint), and the feet (children of the ankle-roll
    // joints). These are converted to nexus ids below, once colliders are final.
    let mut actuated_h: Vec<(RigidBodyHandle, String)> = Vec::new();
    let mut base_h = handles
        .links
        .first()
        .map(|l| l.body)
        .unwrap_or_else(|| bodies.iter().next().unwrap().0);
    let mut foot_h: Vec<RigidBodyHandle> = Vec::new();
    for (i, jh) in handles.joints.iter().enumerate() {
        let name = urdf.joints[i].name.clone();
        if gain_for(robot, &name).is_some() {
            if name.starts_with("hipz") {
                base_h = jh.link1;
            }
            if name.starts_with("anklex") {
                foot_h.push(jh.link2);
            }
            actuated_h.push((jh.link2, name));
        }
    }

    // One nexus-friendly collider per link.
    let link_bodies: Vec<RigidBodyHandle> = handles.links.iter().map(|l| l.body).collect();
    collapse_link_colliders(&mut bodies, &mut colliders, &link_bodies);

    // Ground: a large cuboid whose top surface sits at y = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 0.5, 50.0).friction(1.0),
        ground,
        &mut bodies,
    );

    // Resolve nexus ids (= collider iteration order) now that colliders are final.
    let ids = nexus_body_ids(&colliders);
    let id_of = |h: RigidBodyHandle| *ids.get(&h).expect("body has a nexus id");
    let actuated: Vec<(u32, String)> = actuated_h
        .into_iter()
        .map(|(h, name)| (id_of(h), name))
        .collect();

    Scene {
        bodies,
        colliders,
        impulse: impulse_joints,
        multibody: multibody_joints,
        actuated,
        base_idx: id_of(base_h),
        foot_idx: foot_h.into_iter().map(id_of).collect(),
    }
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

/// Per-rigid-body world poses (one per body, base-frame), indexed by body id.
async fn read_body_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.body_poses().buffer())
        .await
        .expect("body_poses")
}

/// Read the multibody generalized coordinates (joint angles, possibly + base DOFs).
async fn read_dofs(gpu: &KhalGpuBackend, state: &mut GpuPhysicsState) -> Vec<f32> {
    gpu.slow_read_vec(state.multibodies_mut().dof_values().buffer())
        .await
        .expect("dof_values")
}

fn main() {
    let spawn_height: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.55);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    pollster::block_on(async {
        let robot = LeRobotBipedal::new();
        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);

        let scene = build_scene(&robot, spawn_height);
        println!(
            "scene: {} bodies, {} colliders, {} actuated joints; base body [{}]; feet {:?}",
            scene.bodies.len(),
            scene.colliders.len(),
            scene.actuated.len(),
            scene.base_idx,
            scene.foot_idx,
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
        let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);
        state.multibodies_mut().set_gravity(&gpu, [0.0, -9.81, 0.0]);

        // --- initial geometry / DOF diagnostics ---
        let p0 = read_body_poses(&gpu, &state).await;
        let body_y = |i: u32| {
            p0.get(i as usize)
                .map(|p| p.translation.y)
                .unwrap_or(f32::NAN)
        };
        let lowest = p0
            .iter()
            .map(|p| p.translation.y)
            .fold(f32::INFINITY, f32::min);
        let highest = p0
            .iter()
            .map(|p| p.translation.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let foot_y: Vec<f32> = scene.foot_idx.iter().map(|&i| body_y(i)).collect();
        let dofs0 = read_dofs(&gpu, &mut state).await;
        println!(
            "initial: base_y={:.3}  feet_y={:?}  lowest_body_y={:.3}  highest_body_y={:.3}",
            body_y(scene.base_idx),
            foot_y,
            lowest,
            highest
        );
        println!(
            "dof_values: len={} (expected 12 leg DOFs{}), values={:?}",
            dofs0.len(),
            if dofs0.len() > 12 { " + base" } else { "" },
            dofs0
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        if lowest > 0.02 {
            println!(
                "NOTE: lowest body is {lowest:.3} m above ground — robot will free-fall {lowest:.3} m \
                 before contact. Lower --spawn_height by ~{lowest:.2}."
            );
        }

        let crouch_at = steps * 3 / 4;
        println!(
            "\n{:>5}  {:>8}  {:>8}  {:>8}  {:>9}  {:>9}",
            "step", "t(s)", "base_y", "foot_y", "max|dq|", "|basevel|?"
        );
        for step in 0..steps {
            // Hold the default pose; in the final quarter, command a −0.3 rad crouch on
            // every actuated joint to exercise the runtime PD setter.
            let target = if step < crouch_at { 0.0 } else { -0.3 };
            {
                let mm = state.multibodies_mut();
                for (lid, _) in &scene.actuated {
                    let _ = mm.set_motor_position(&gpu, 0, *lid, JointAxis::AngX, target);
                }
            }

            let _ = pipeline.step(&gpu, &mut state, None).await;
            gpu.synchronize().expect("sync");
            pipeline.auto_resize_buffers(&gpu, &mut state).await;

            if step % 25 == 0 || step == steps - 1 || step == crouch_at {
                let poses = read_body_poses(&gpu, &state).await;
                let by = poses
                    .get(scene.base_idx as usize)
                    .map(|p| p.translation.y)
                    .unwrap_or(f32::NAN);
                let fy = scene
                    .foot_idx
                    .iter()
                    .map(|&i| {
                        poses
                            .get(i as usize)
                            .map(|p| p.translation.y)
                            .unwrap_or(f32::NAN)
                    })
                    .fold(f32::INFINITY, f32::min);
                let dofs = read_dofs(&gpu, &mut state).await;
                let max_dq = dofs
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let any_nan = poses.iter().any(|p| !p.translation.y.is_finite())
                    || dofs.iter().any(|v| !v.is_finite());
                let tag = if step == crouch_at {
                    "  <- crouch -0.3"
                } else {
                    ""
                };
                println!(
                    "{:>5}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}{}{}",
                    step,
                    step as f32 * DT,
                    by,
                    fy,
                    max_dq,
                    if any_nan { "   <-- NON-FINITE" } else { "" },
                    tag
                );
                if any_nan {
                    println!("\nFAILED: physics produced non-finite state (exploded).");
                    return;
                }
            }
        }

        let dofs = read_dofs(&gpu, &mut state).await;
        println!(
            "\nfinal dof_values = {:?}",
            dofs.iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        println!(
            "Verdict: read base_y stability through the hold phase (steps < {crouch_at}). If it\n\
             settles near a constant height and feet stay ~0, the biped stands under PD on nexus.\n\
             The crouch phase should drive max|dq| toward 0.3, confirming set_motor_position."
        );
    });
}
