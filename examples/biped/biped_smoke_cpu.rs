//! Stage 2 (CPU) — the feasibility gate on rapier's mature CPU multibody:
//! does the LeRobot bipedal *stand* while PD-holding its default pose?
//!
//! After the nexus GPU path turned out not to simulate a floating-base
//! articulation stably yet, we run the env on rapier's CPU `PhysicsPipeline`
//! (still all-Rust; the same engine nexus wraps). This loads the bipedal URDF as
//! a floating-base multibody with Obb collision proxies, configures a force-based
//! PD position motor on each of the 12 leg joints, drops it on a ground cuboid,
//! and reports the torso height + uprightness over time.
//!
//! Run: `cargo run --release --example biped_smoke_cpu --features cpu -- [spawn_h] [steps]`

use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};
use zealot_env::robots::{LeRobotBipedal, RobotSpec};

const DT: f32 = 1.0 / 200.0;
const SOLVER_ITERS: usize = 8;

/// PD gain + torque cap for one of the 12 actuated joints, by name.
fn gain_for(robot: &RobotSpec, name: &str) -> Option<(f32, f32, f32)> {
    robot
        .joints
        .iter()
        .find(|j| j.name == name)
        .map(|j| (j.kp, j.kd, j.effort_limit))
}

struct Scene {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    torso: RigidBodyHandle,
    feet: Vec<RigidBodyHandle>,
    /// Every link body with its URDF name, for the spawn-geometry dump.
    links: Vec<(RigidBodyHandle, String)>,
}

/// Lowest world-space point (Z, the up-axis) of a body's colliders.
fn body_lowest_z(bodies: &RigidBodySet, colliders: &ColliderSet, h: RigidBodyHandle) -> f32 {
    let mut lo = f32::INFINITY;
    for &ch in bodies[h].colliders() {
        let aabb = colliders[ch].compute_aabb();
        lo = lo.min(aabb.mins.z);
    }
    lo
}

fn build_scene(robot: &RobotSpec, spawn_height: f32) -> Scene {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();

    let urdf_path = robot.urdf_path();
    let assets_dir = urdf_path.parent().expect("urdf dir").join("assets");
    let urdf_str = std::fs::read_to_string(&urdf_path)
        .unwrap_or_else(|e| panic!("read urdf {}: {e}", urdf_path.display()))
        .replace("package://assets/", "");

    let options = UrdfLoaderOptions {
        create_colliders_from_collision_shapes: true,
        create_colliders_from_visual_shapes: false,
        apply_imported_mass_props: true,
        make_roots_fixed: false, // floating base
        mesh_converter: Some(MeshConverter::Obb),
        // Match the MuJoCo model exactly: Z-up, identity base orientation, torso
        // lifted to `spawn_height` (the working model uses z = 0.72). No rotation —
        // the zero-joint pose is already the standing pose in Z-up.
        shift: Pose::from_parts(Vec3::new(0.0, 0.0, spawn_height), Rotation::IDENTITY),
        ..UrdfLoaderOptions::default()
    };

    let (mut urdf_robot, urdf) = UrdfRobot::from_str(&urdf_str, options, &assets_dir)
        .unwrap_or_else(|e| panic!("parse urdf: {e}"));

    for (i, uj) in urdf_robot.joints.iter_mut().enumerate() {
        let name = &urdf.joints[i].name;
        if let Some((kp, kd, max_force)) = gain_for(robot, name) {
            // Stiffen constraint motors so the rapier multibody can actually STAND
            // (it lacks nexus's explicit-PD). BIPED_KP_SCALE (default 1).
            let ks: f32 = std::env::var("BIPED_KP_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let (kp, kd, max_force) = (kp * ks, kd * ks.sqrt(), max_force * ks);
            // Constraint-based ForceBased under-realizes kp → robot buckles (the
            // nexus saga). For the rapier contact-drift reference we need it to
            // STAND, so default to AccelerationBased (holds); BIPED_FORCE_MOTOR=1
            // reverts. (Either way it's rapier's contact solver under test.)
            let model = if std::env::var("BIPED_FORCE_MOTOR").is_ok() {
                MotorModel::ForceBased
            } else {
                MotorModel::AccelerationBased
            };
            uj.joint.set_motor_model(JointAxis::AngX, model);
            uj.joint.set_motor_position(JointAxis::AngX, 0.0, kp, kd);
            uj.joint.set_motor_max_force(JointAxis::AngX, max_force);
        }
    }

    let handles = urdf_robot.insert_using_multibody_joints(
        &mut bodies,
        &mut colliders,
        &mut multibody,
        UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
    );

    let mut torso = handles.links[0].body;
    let mut feet = Vec::new();
    for (i, jh) in handles.joints.iter().enumerate() {
        let name = &urdf.joints[i].name;
        if name.starts_with("hipz") {
            torso = jh.link1;
        }
        if name.starts_with("anklex") {
            feet.push(jh.link2);
        }
    }

    // rapier-urdf inserts multibody links *without* running forward kinematics, so
    // every link body starts stacked at the root (a folded heap that the solver
    // then explodes). Run FK from the root's spawn pose so the legs extend into the
    // neutral standing configuration before we simulate.
    if let Some(link_id) = multibody.rigid_body_link(torso).copied() {
        if let Some(mb) = multibody.get_multibody_mut(link_id.multibody) {
            mb.forward_kinematics(&bodies, true);
            mb.update_rigid_bodies(&mut bodies, true);
        }
    }

    // Name each link body (handles.links is parallel to urdf.links).
    let links: Vec<(RigidBodyHandle, String)> = handles
        .links
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                l.body,
                urdf.links
                    .get(i)
                    .map(|u| u.name.clone())
                    .unwrap_or_default(),
            )
        })
        .collect();

    // Ground plane (Z-up): a big cuboid in XY whose top surface sits at z = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );

    Scene {
        bodies,
        colliders,
        impulse,
        multibody,
        torso,
        feet,
        links,
    }
}

fn main() {
    let spawn_height: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.72); // MuJoCo model's standing torso height (Z-up)
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let robot = RobotSpec::from_env();
    let mut scene = build_scene(&robot, spawn_height);

    // Z-up convention (matching MuJoCo / the real robot): up = +Z.
    let body_z = |bodies: &RigidBodySet, h: RigidBodyHandle| bodies[h].translation().z;
    let upright = |bodies: &RigidBodySet, h: RigidBodyHandle| {
        // Torso local +Z rotated into the world, dotted with world up (+Z). 1 = upright.
        bodies[h].rotation().mul_vec3(Vec3::Z).z
    };

    println!(
        "scene: {} bodies, {} colliders, {} feet; spawn {spawn_height} m",
        scene.bodies.len(),
        scene.colliders.len(),
        scene.feet.len()
    );

    // Spawn geometry: per-link body Z and lowest collider point, to see the
    // zero-pose configuration (standing? feet at the bottom?).
    println!("\nspawn geometry (link body_z / lowest collider z):");
    let mut rows: Vec<(String, f32, f32)> = scene
        .links
        .iter()
        .map(|(h, name)| {
            (
                name.clone(),
                body_z(&scene.bodies, *h),
                body_lowest_z(&scene.bodies, &scene.colliders, *h),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.2.total_cmp(&b.2));
    for (name, bz, lo) in &rows {
        println!("  {:<26} body_z={:>7.3}  lowest={:>7.3}", name, bz, lo);
    }
    let global_lowest = rows.iter().map(|r| r.2).fold(f32::INFINITY, f32::min);
    println!("global lowest collider point: {global_lowest:.3} m (want ~0 for feet on ground)");
    let foot_z = |bodies: &RigidBodySet| {
        scene
            .feet
            .iter()
            .map(|&h| body_z(bodies, h))
            .fold(f32::INFINITY, f32::min)
    };
    println!(
        "initial: torso_z={:.3}  min_foot_z={:.3}  upright={:.3}",
        body_z(&scene.bodies, scene.torso),
        foot_z(&scene.bodies),
        upright(&scene.bodies, scene.torso),
    );

    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut ip = IntegrationParameters::default();
    ip.dt = DT;
    ip.num_solver_iterations = std::env::var("BIPED_SOLVER_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SOLVER_ITERS);
    println!("num_solver_iterations = {}", ip.num_solver_iterations);
    let mut pipeline = PhysicsPipeline::new();
    // RAPIER reference for the foot-slip/drift-∝-substeps bug: track each foot's
    // horizontal travel while planted (low). If rapier ALSO drifts ∝ substeps,
    // nexus faithfully ports a rapier TGS characteristic; if not, nexus diverged.
    let foot_xy = |bodies: &RigidBodySet, h: RigidBodyHandle| {
        let t = bodies[h].translation();
        (t.x, t.y)
    };
    let mut prev_xy: Vec<(f32, f32)> = scene
        .feet
        .iter()
        .map(|&h| foot_xy(&scene.bodies, h))
        .collect();
    let mut path_len: Vec<f32> = vec![0.0; scene.feet.len()];
    let mut planted_steps: Vec<u32> = vec![0; scene.feet.len()];
    let mut islands = IslandManager::new();
    let mut broad_phase = BroadPhaseBvh::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut ccd = CCDSolver::new();

    println!(
        "\n{:>5}  {:>8}  {:>8}  {:>8}  {:>9}",
        "step", "t(s)", "torso_z", "foot_z", "upright"
    );
    for step in 0..steps {
        pipeline.step(
            gravity,
            &ip,
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut scene.bodies,
            &mut scene.colliders,
            &mut scene.impulse,
            &mut scene.multibody,
            &mut ccd,
            &(),
            &(),
        );

        // Per-foot horizontal drift while planted (foot body z < 0.10 ≈ on ground).
        for (i, &h) in scene.feet.iter().enumerate() {
            let (x, y) = foot_xy(&scene.bodies, h);
            if scene.bodies[h].translation().z < 0.10 {
                let (px, py) = prev_xy[i];
                path_len[i] += ((x - px).powi(2) + (y - py).powi(2)).sqrt();
                planted_steps[i] += 1;
            }
            prev_xy[i] = (x, y);
        }

        if step % 50 == 0 || step == steps - 1 {
            let tz = body_z(&scene.bodies, scene.torso);
            let fz = foot_z(&scene.bodies);
            let up = upright(&scene.bodies, scene.torso);
            let bad = !tz.is_finite() || !up.is_finite();
            println!(
                "{:>5}  {:>8.3}  {:>8.3}  {:>8.3}  {:>9.3}{}",
                step,
                step as f32 * DT,
                tz,
                fz,
                up,
                if bad { "  <-- NON-FINITE" } else { "" }
            );
            if bad {
                println!("\nFAILED: non-finite state.");
                return;
            }
        }
    }

    // Foot-drift report: mean planted-foot horizontal speed (m/s). If this grows
    // with num_solver_iterations like nexus does, the bug is a rapier-TGS trait.
    println!("\n[rapier foot drift] planted-foot mean horizontal speed:");
    for (i, &h) in scene.feet.iter().enumerate() {
        let secs = planted_steps[i] as f32 * DT;
        let speed = if secs > 1e-3 { path_len[i] / secs } else { 0.0 };
        println!(
            "  foot{i}: path={:.1}cm  planted={:.2}s  drift_rate={speed:.3} m/s",
            path_len[i] * 100.0,
            secs,
        );
        let _ = h;
    }

    let tz = body_z(&scene.bodies, scene.torso);
    let up = upright(&scene.bodies, scene.torso);
    println!(
        "\nFinal torso_z={tz:.3} (spawn {spawn_height:.3}), upright={up:.3}.\n\
         Stands if torso settles near a constant height with upright≈1. A collapse\n\
         shows torso_z dropping and upright falling toward 0."
    );
}
