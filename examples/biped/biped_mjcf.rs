//! Stage 2 (take 2) — build the LeRobot bipedal in rapier from the **curated
//! MuJoCo model** (`to_real_robot/RL_policy/robot.xml`, the model the deployed
//! policy actually uses), not the raw URDF.
//!
//! Why: the URDF's zero-joint pose doesn't load into a standing stance in rapier
//! (rapier-urdf's conversion mangles it). The MJCF is curated: each body carries
//! its `pos`/`quat` relative to its parent (so the zero-pose IS standing, Z-up,
//! torso at 0.72 m), proper inertials, and clean foot collision capsules. It has
//! no `<actuator>` (mjlab adds those), so we drive the joints with the
//! `LeRobotBipedal` PD gains.
//!
//! This is a focused reader for *this* model's structure (a single kinematic
//! tree of hinge joints about each child's local Z), not a general MJCF loader.
//!
//! Run: `cargo run --release --example biped_mjcf --features cpu -- [steps]`

use rapier3d::prelude::*;
use roxmltree::Node;
use zealot_env::robots::{LeRobotBipedal, RobotSpec};

const DT: f32 = 1.0 / 200.0;
const SOLVER_ITERS: usize = 8;

/// One body parsed from the MJCF tree.
struct MjBody {
    name: String,
    parent: Option<usize>,
    local_pos: Vec3,
    local_quat: Rotation,
    /// Hinge joint name + axis (child-local), or `None` for the root / fixed bodies.
    joint: Option<(String, Vec3)>,
    is_free: bool,
    com: Vec3,
    mass: f32,
    inertia_diag: Vec3,
    /// Collision capsules `(a, b, radius)` in the body frame.
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
/// MuJoCo quaternions are scalar-first `w x y z`.
fn quat_wxyz(node: &Node, default: Rotation) -> Rotation {
    match node.attribute("quat") {
        Some(s) => {
            let f = floats(s);
            Rotation::from_xyzw(f[1], f[2], f[3], f[0]).normalize()
        }
        None => default,
    }
}

fn parse_body(node: &Node, parent: Option<usize>, out: &mut Vec<MjBody>) {
    let name = node.attribute("name").unwrap_or("").to_string();
    let mut joint = None;
    let mut is_free = false;
    let (mut com, mut mass, mut inertia_diag) = (Vec3::ZERO, 0.0, Vec3::splat(1e-4));
    let mut capsules = Vec::new();

    for child in node.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "freejoint" => is_free = true,
            "joint" => {
                let jn = child.attribute("name").unwrap_or("").to_string();
                joint = Some((jn, vec3(&child, "axis", Vec3::Z)));
            }
            "inertial" => {
                com = vec3(&child, "pos", Vec3::ZERO);
                mass = child
                    .attribute("mass")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                if let Some(s) = child.attribute("fullinertia") {
                    let f = floats(s); // Ixx Iyy Izz Ixy Ixz Iyz
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

    // Only keep the root and jointed bodies (skip fixed visual-only children like
    // `torso_mesh`, which would just be a rigid appendage with no actuation).
    let idx = out.len();
    let keep = parent.is_none() || joint.is_some() || is_free;
    if keep {
        out.push(MjBody {
            name,
            parent,
            local_pos: vec3(node, "pos", Vec3::ZERO),
            local_quat: quat_wxyz(node, Rotation::IDENTITY),
            joint,
            is_free,
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

fn gain_for(robot: &RobotSpec, name: &str) -> (f32, f32, f32) {
    robot
        .joints
        .iter()
        .find(|j| j.name == name)
        .map(|j| (j.kp, j.kd, j.effort_limit))
        .unwrap_or((50.0, 1.0, 20.0))
}

struct Built {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    handles: Vec<RigidBodyHandle>,
    torso: usize,
    feet: Vec<usize>,
}

fn build(mjcf: &[MjBody], robot: &RobotSpec, spawn_z: f32) -> Built {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();

    // Forward kinematics: world pose of each body (root lifted to spawn_z).
    let mut world: Vec<Pose> = Vec::with_capacity(mjcf.len());
    for (i, b) in mjcf.iter().enumerate() {
        let w = match b.parent {
            None => Pose::from_parts(Vec3::new(0.0, 0.0, spawn_z), Rotation::IDENTITY),
            Some(p) => world[p] * Pose::from_parts(b.local_pos, b.local_quat),
        };
        world.push(w);
        debug_assert!(i + 1 == world.len());
    }

    // Rigid bodies with MJCF mass properties.
    let mut handles = Vec::with_capacity(mjcf.len());
    let mut feet = Vec::new();
    let mut torso = 0;
    for (i, b) in mjcf.iter().enumerate() {
        if b.parent.is_none() {
            torso = i;
        }
        let rb = RigidBodyBuilder::dynamic()
            .position(world[i])
            .additional_mass_properties(MassProperties::new(
                b.com,
                b.mass.max(1e-3),
                b.inertia_diag,
            ))
            .build();
        let h = bodies.insert(rb);
        handles.push(h);

        // Foot collision capsules (the only collisions in the model).
        if !b.capsules.is_empty() {
            feet.push(i);
            for (a, c, r) in &b.capsules {
                colliders.insert_with_parent(
                    ColliderBuilder::new(SharedShape::capsule(*a, *c, *r))
                        .density(0.0)
                        .friction(1.0),
                    h,
                    &mut bodies,
                );
            }
        }
    }

    // Revolute multibody joints about each child's local Z, with PD motors.
    let locked = JointAxesMask::LIN_X
        | JointAxesMask::LIN_Y
        | JointAxesMask::LIN_Z
        | JointAxesMask::ANG_X
        | JointAxesMask::ANG_Y;
    for (i, b) in mjcf.iter().enumerate() {
        let (Some(parent), Some((jname, _axis))) = (b.parent, b.joint.as_ref()) else {
            continue;
        };
        let (kp, kd, effort) = gain_for(robot, jname);
        let mut joint = GenericJointBuilder::new(locked)
            .local_frame1(Pose::from_parts(b.local_pos, b.local_quat))
            .local_frame2(Pose::IDENTITY)
            .build();
        joint.set_motor_model(JointAxis::AngZ, MotorModel::ForceBased);
        joint.set_motor_position(JointAxis::AngZ, 0.0, kp, kd);
        joint.set_motor_max_force(JointAxis::AngZ, effort);
        multibody.insert(handles[parent], handles[i], joint, true);
    }

    // Re-run FK from the root through the multibody so coords-0 poses are exact.
    if let Some(link_id) = multibody.rigid_body_link(handles[torso]).copied() {
        if let Some(mb) = multibody.get_multibody_mut(link_id.multibody) {
            mb.forward_kinematics(&bodies, true);
            mb.update_rigid_bodies(&mut bodies, true);
        }
    }

    // Ground plane (Z-up), top at z = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );

    Built {
        bodies,
        colliders,
        impulse,
        multibody,
        handles,
        torso,
        feet,
    }
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let home = std::env::var("HOME").unwrap_or_default();
    let mjcf_path =
        format!("{home}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml");
    let xml =
        std::fs::read_to_string(&mjcf_path).unwrap_or_else(|e| panic!("read {mjcf_path}: {e}"));

    let mjcf = parse_mjcf(&xml);
    let robot = RobotSpec::from_env();
    println!("parsed {} bodies from MJCF:", mjcf.len());
    for b in &mjcf {
        println!(
            "  {:<24} parent={:?} joint={:?} mass={:.3} caps={}",
            b.name,
            b.parent,
            b.joint.as_ref().map(|(n, _)| n.as_str()),
            b.mass,
            b.capsules.len()
        );
    }

    let spawn_z = 0.72;
    let mut s = build(&mjcf, &robot, spawn_z);
    let bz = |bodies: &RigidBodySet, i: usize| bodies[s.handles[i]].translation().z;
    let upright =
        |bodies: &RigidBodySet, i: usize| bodies[s.handles[i]].rotation().mul_vec3(Vec3::Z).z;

    // Symmetry + stance check at spawn (post-FK).
    println!("\npost-FK stance (body_z):");
    for (i, b) in mjcf.iter().enumerate() {
        println!("  {:<24} z={:>7.3}", b.name, bz(&s.bodies, i));
    }
    println!(
        "feet z: {:?}  (should be ~equal & lowest)",
        s.feet
            .iter()
            .map(|&i| (bz(&s.bodies, i) * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );

    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut ip = IntegrationParameters::default();
    ip.dt = DT;
    ip.num_solver_iterations = SOLVER_ITERS;
    let mut pipeline = PhysicsPipeline::new();
    let mut islands = IslandManager::new();
    let mut bp = BroadPhaseBvh::new();
    let mut np = NarrowPhase::new();
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
            &mut bp,
            &mut np,
            &mut s.bodies,
            &mut s.colliders,
            &mut s.impulse,
            &mut s.multibody,
            &mut ccd,
            &(),
            &(),
        );
        if step % 50 == 0 || step == steps - 1 {
            let tz = bz(&s.bodies, s.torso);
            let fz = s
                .feet
                .iter()
                .map(|&i| bz(&s.bodies, i))
                .fold(f32::INFINITY, f32::min);
            let up = upright(&s.bodies, s.torso);
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
    let tz = bz(&s.bodies, s.torso);
    let up = upright(&s.bodies, s.torso);
    println!(
        "\nFinal torso_z={tz:.3} (spawn {spawn_z:.2}), upright={up:.3}.\n\
         Stands if torso_z settles near constant with upright≈1."
    );
}
