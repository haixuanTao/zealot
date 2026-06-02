//! Minimal physics-only repro showing rapier's contact/friction lets a foot-shaped
//! box rest on its edge against gravity (when it should tip flat).
//!
//! No robot, no MJCF, no policy — just a single dynamic rigid body (box matching the
//! LeRobot bipedal foot's AABB), a ground plane, and `IntegrationParameters` identical
//! to the ones we train against in `BipedEnv` (dt = 1/200, num_solver_iterations = 8,
//! friction = 1.0 on both foot and ground).
//!
//! Run: `cargo run --release --example foot_tip_stability --features cpu`
//!
//! Backs the discussion in dimforge/rapier#934.

use rapier3d::prelude::*;
use std::fmt::Write as _;

/// Foot box half-extents (m). The actual MJCF foot capsules (at local Y = −0.02,
/// X ∈ ±0.032, Z ∈ [−0.10, +0.005], capsule radius 0.015) form a sole in the foot
/// link's local X–Z plane, with the sole-normal along **local Y** (the thin axis):
///   X half-ext: 0.032 + 0.015 = 0.047   (long axis of the sole)
///   Y half-ext: 0.015                   (thin = sole-normal, vertical at spawn)
///   Z half-ext: (0.10 + 0.005)/2 + 0.015 ≈ 0.0675   (wide axis of the sole)
///
/// In the actual robot the foot link is rotated so its local +Y maps to world +Z
/// at spawn (sole flat with thin axis vertical). To keep this standalone repro's
/// "identity rotation = sole flat on ground" convention, we PERMUTE the half-extents
/// so the thin axis is along local **Z** (and local +Z = sole-normal). That matches
/// the orientation the box has in world frame at the robot's spawn pose.
const HE: (f32, f32, f32) = (0.047, 0.0675, 0.015);

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
}

/// Fresh world with the same ground + parameters `BipedEnv` uses.
fn build_world() -> World {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    // Ground: fixed cuboid 100×100×1 m, top surface at z=0, friction 1.0 — exactly
    // matching biped_env.rs:293–299.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );
    World {
        bodies,
        colliders,
        impulse: ImpulseJointSet::new(),
        multibody: MultibodyJointSet::new(),
        islands: IslandManager::new(),
        bp: BroadPhaseBvh::new(),
        np: NarrowPhase::new(),
        ccd: CCDSolver::new(),
        pipeline: PhysicsPipeline::new(),
    }
}

/// Spawn the dynamic foot box at `(0, 0, height)` with the given orientation.
/// Density 1000 kg/m³ gives the box ~0.3 kg, close to the actual foot link mass.
fn spawn_foot(w: &mut World, tilt: Rotation, height: f32) -> RigidBodyHandle {
    let pose = Pose::from_parts(Vec3::new(0.0, 0.0, height), tilt);
    let body = w.bodies.insert(RigidBodyBuilder::dynamic().pose(pose));
    w.colliders.insert_with_parent(
        ColliderBuilder::cuboid(HE.0, HE.1, HE.2)
            .friction(1.0)
            .density(1000.0),
        body,
        &mut w.bodies,
    );
    body
}

/// Sole-tilt (rad) — angle of the box's local +Z (sole-normal at spawn) from world
/// +Z. Same convention as `BipedEnv::foot_tilts` so numbers are directly comparable.
fn sole_tilt(w: &World, h: RigidBodyHandle) -> f32 {
    let n = w.bodies[h].rotation() * Vec3::Z;
    n.z.abs().clamp(0.0, 1.0).acos()
}

/// One pose record per simulation step (used by the JSON dump for the renderer).
#[derive(Clone, Copy)]
struct PoseFrame {
    pos: Vec3,
    rot: Rotation,
}

/// Lowest world-Z reached by any of the 8 box corners after applying `rot` to
/// the half-extents — used to set the spawn body-z so the box just-touches the
/// ground (no free fall, no perturbation from impact).
fn lowest_corner_z(rot: Rotation, he: (f32, f32, f32)) -> f32 {
    let (a, b, c) = he;
    let mut lo = f32::INFINITY;
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let v = rot * Vec3::new(sx * a, sy * b, sz * c);
                if v.z < lo {
                    lo = v.z;
                }
            }
        }
    }
    lo
}

/// Run one scenario: build a fresh world, spawn at the initial tilt with the
/// lowest corner placed exactly on the ground (no drop perturbation for the
/// knife-edge tests), step physics for 5 s, report tilt + a verdict.
/// Returns the per-step pose trajectory so it can be replayed in the renderer.
fn run(name: &str, tilt: Rotation) -> Vec<PoseFrame> {
    let mut w = build_world();
    // Drop test: spawn 5 cm ABOVE the just-touching height so the box hits the
    // ground with real impact. Idealised knife-edge balance is a strict unstable
    // equilibrium that any engine preserves at perfect symmetry — a drop adds the
    // physical perturbation real life supplies, which an honest contact model
    // should let tip the box off the unstable angle.
    let spawn_z = -lowest_corner_z(tilt, HE) + 0.05;
    let foot = spawn_foot(&mut w, tilt, spawn_z);

    let mut ip = IntegrationParameters::default();
    ip.dt = 1.0 / 200.0;
    ip.num_solver_iterations = 8;
    let gravity = Vec3::new(0.0, 0.0, -9.81);

    let total_steps = 1000; // 5 s at dt=0.005
    let probe_every = 100; //  0.5 s cadence

    println!(
        "\n[{name}]  initial tilt = {:.1}°  spawn body-z = {:.4} m (5 cm drop above just-touching)",
        sole_tilt(&w, foot).to_degrees(),
        spawn_z
    );
    println!("   t (s)   tilt (deg)   z (m)     |omega| (rad/s)");
    let mut tipped_at: Option<f32> = None;
    let mut frames: Vec<PoseFrame> = Vec::with_capacity(total_steps + 1);
    frames.push(PoseFrame {
        pos: w.bodies[foot].translation(),
        rot: *w.bodies[foot].rotation(),
    });
    for s in 0..total_steps {
        w.pipeline.step(
            gravity,
            &ip,
            &mut w.islands,
            &mut w.bp,
            &mut w.np,
            &mut w.bodies,
            &mut w.colliders,
            &mut w.impulse,
            &mut w.multibody,
            &mut w.ccd,
            &(),
            &(),
        );
        let t = (s + 1) as f32 * ip.dt;
        let tilt_deg = sole_tilt(&w, foot).to_degrees();
        frames.push(PoseFrame {
            pos: w.bodies[foot].translation(),
            rot: *w.bodies[foot].rotation(),
        });
        if (s + 1) % probe_every == 0 {
            let z = w.bodies[foot].translation().z;
            let av = w.bodies[foot].angvel();
            let omega = (av.x * av.x + av.y * av.y + av.z * av.z).sqrt();
            println!("  {t:5.2}    {tilt_deg:6.2}      {z:+.4}    {omega:.4}");
        }
        if tipped_at.is_none() && tilt_deg < 10.0 {
            tipped_at = Some(t);
        }
    }

    let final_tilt = sole_tilt(&w, foot).to_degrees();
    match tipped_at {
        Some(t) => println!(
            "VERDICT: settled flat (tilt < 10°) at t = {t:.2} s — final tilt {final_tilt:.2}°"
        ),
        None => println!(
            "VERDICT: came to rest at {final_tilt:.2}° after 5 s. Check the rendering \
             to see whether that's an actual edge or a small-face stable rest."
        ),
    }
    frames
}

/// Serialize all scenarios' pose trajectories to JSON for the renderer.
fn dump_json(path: &str, dt: f32, he: (f32, f32, f32), scenarios: &[(&str, Vec<PoseFrame>)]) {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = write!(s, "  \"dt\": {dt},\n");
    let _ = write!(s, "  \"half_extents\": [{},{},{}],\n", he.0, he.1, he.2);
    s.push_str("  \"scenarios\": {\n");
    for (i, (name, frames)) in scenarios.iter().enumerate() {
        let _ = write!(s, "    \"{name}\": [\n");
        for (j, f) in frames.iter().enumerate() {
            let comma = if j + 1 < frames.len() { "," } else { "" };
            let _ = write!(
                s,
                "      [{:.5},{:.5},{:.5},{:.6},{:.6},{:.6},{:.6}]{comma}\n",
                f.pos.x, f.pos.y, f.pos.z, f.rot.x, f.rot.y, f.rot.z, f.rot.w
            );
        }
        let comma = if i + 1 < scenarios.len() { "," } else { "" };
        let _ = write!(s, "    ]{comma}\n");
    }
    s.push_str("  }\n}\n");
    std::fs::write(path, &s).expect("write json");
    println!("\nwrote pose trajectories → {path}");
}

fn main() {
    println!("Minimal repro: foot-box edge-balance under rapier contact (dimforge/rapier#934)");
    println!(
        "  foot half-extents: ({:.3}, {:.3}, {:.3}) m   friction: 1.0 (foot & ground)",
        HE.0, HE.1, HE.2
    );
    println!("  dt = 1/200 s   num_solver_iterations = 8   gravity = -9.81 z");

    // Scenarios — each rebuilds a fresh world and runs 5 s of physics.
    // The knife-edge angles for this box (a=0.047, b=0.0675, c=0.015) — i.e. the
    // angles at which the CoM is *exactly* above the contact set, so an honest
    // contact model must let gravity tip the box one way or the other:
    //
    //   • on the 9.4-cm edge  (X-parallel body edge): X-rot by atan(b/c) ≈ 77.5°
    //   • on the 13.5-cm edge (Y-parallel body edge): Y-rot by atan(a/c) ≈ 72.3°
    //   • on a single corner (one tip): rotation that aligns the body diagonal
    //                                   (a, -b, -c)/|·| with world -Z, so the CoM
    //                                   sits *exactly* over that one corner.
    //
    // Anything stable for 5 s in these configurations is a contact-fidelity artifact.
    let theta_94 = (0.0675_f32 / 0.015).atan(); // atan(b/c) ≈ 77.47°
    let phi_135 = (0.047_f32 / 0.015).atan(); // atan(a/c) ≈ 72.34°
    let diag = Vec3::new(0.047, -0.0675, -0.015).normalize();
    let one_tip_rot = Rotation::from_rotation_arc(diag, -Vec3::Z);

    let scenarios = vec![
        ("flat_baseline", run("flat_baseline", Rotation::IDENTITY)),
        (
            "knife_edge_94mm",
            run(
                "knife_edge_94mm",
                Rotation::from_axis_angle(Vec3::X, theta_94),
            ),
        ),
        (
            "knife_edge_135mm",
            run(
                "knife_edge_135mm",
                Rotation::from_axis_angle(Vec3::Y, phi_135),
            ),
        ),
        ("one_tip_corner", run("one_tip_corner", one_tip_rot)),
    ];

    dump_json("/tmp/foot_tip_poses.json", 1.0 / 200.0, HE, &scenarios);
    println!(
        "render with: python3 examples/biped/render_foot_tip.py /tmp/foot_tip_poses.json /tmp/foot_tip.mp4"
    );

    // --- Knife-edge sweep: which rapier settings actually tip the 94 mm artifact?
    println!("\n================  KNIFE-EDGE TUNABLE SWEEP  ================");
    println!("Re-running knife_edge_94mm with each tweaked rapier setting in turn.");
    println!("Goal: find a quick fix that makes the box correctly tip flat.\n");
    let tilt = Rotation::from_axis_angle(Vec3::X, theta_94);
    sweep_knife_edge("baseline (defaults)", tilt, |_ip, _b, _c| {});
    sweep_knife_edge("FrictionModel::Coulomb", tilt, |ip, _b, _c| {
        ip.friction_model = FrictionModel::Coulomb;
    });
    sweep_knife_edge("contact_softness nf=10 Hz (softer)", tilt, |ip, _b, _c| {
        ip.contact_softness.natural_frequency = 10.0;
    });
    sweep_knife_edge(
        "contact_softness nf=5 Hz (very soft)",
        tilt,
        |ip, _b, _c| {
            ip.contact_softness.natural_frequency = 5.0;
        },
    );
    sweep_knife_edge("num_solver_iterations=32", tilt, |ip, _b, _c| {
        ip.num_solver_iterations = 32;
    });
    sweep_knife_edge("num_solver_iterations=64", tilt, |ip, _b, _c| {
        ip.num_solver_iterations = 64;
    });
    sweep_knife_edge("dt = 1/1000 s (5× smaller)", tilt, |ip, _b, _c| {
        ip.dt = 1.0 / 1000.0;
    });
    sweep_knife_edge("friction = 0.1", tilt, |_ip, _b, c| {
        c.set_friction(0.1);
    });
    sweep_knife_edge("body angular_damping = 0.1", tilt, |_ip, b, _c| {
        b.set_angular_damping(0.1);
    });
    sweep_knife_edge(
        "ALL: Coulomb + soft nf=10 + iter=32 + dt=1/500",
        tilt,
        |ip, _b, _c| {
            ip.friction_model = FrictionModel::Coulomb;
            ip.contact_softness.natural_frequency = 10.0;
            ip.num_solver_iterations = 32;
            ip.dt = 1.0 / 500.0;
        },
    );
}

/// Settings-sweep variant of `run`: tweak `IntegrationParameters` / body / collider
/// just before stepping, and report only the verdict (tipped vs held).
fn sweep_knife_edge(
    label: &str,
    tilt: Rotation,
    tweak: impl Fn(&mut IntegrationParameters, &mut RigidBody, &mut Collider),
) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, -0.5)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 50.0, 0.5).friction(1.0),
        ground,
        &mut bodies,
    );
    let spawn_z = -lowest_corner_z(tilt, HE) + 0.001;
    let pose = Pose::from_parts(Vec3::new(0.0, 0.0, spawn_z), tilt);
    let foot_h = bodies.insert(RigidBodyBuilder::dynamic().pose(pose));
    let foot_col_h = colliders.insert_with_parent(
        ColliderBuilder::cuboid(HE.0, HE.1, HE.2)
            .friction(1.0)
            .density(1000.0),
        foot_h,
        &mut bodies,
    );

    let mut ip = IntegrationParameters::default();
    ip.dt = 1.0 / 200.0;
    ip.num_solver_iterations = 8;
    tweak(&mut ip, &mut bodies[foot_h], &mut colliders[foot_col_h]);

    let mut impulse = ImpulseJointSet::new();
    let mut multibody = MultibodyJointSet::new();
    let mut islands = IslandManager::new();
    let mut bp = BroadPhaseBvh::new();
    let mut np = NarrowPhase::new();
    let mut ccd = CCDSolver::new();
    let mut pipeline = PhysicsPipeline::new();
    let gravity = Vec3::new(0.0, 0.0, -9.81);

    let total_t = 5.0_f32;
    let total_steps = (total_t / ip.dt).round() as usize;
    let mut tipped_at: Option<f32> = None;
    for s in 0..total_steps {
        pipeline.step(
            gravity,
            &ip,
            &mut islands,
            &mut bp,
            &mut np,
            &mut bodies,
            &mut colliders,
            &mut impulse,
            &mut multibody,
            &mut ccd,
            &(),
            &(),
        );
        let tilt_deg = sole_tilt_of(&bodies[foot_h]).to_degrees();
        if tipped_at.is_none() && tilt_deg < 10.0 {
            tipped_at = Some((s + 1) as f32 * ip.dt);
        }
    }
    let r = *bodies[foot_h].rotation();
    let av = bodies[foot_h].angvel();
    let omega = (av.x * av.x + av.y * av.y + av.z * av.z).sqrt();
    let final_tilt = sole_tilt_of(&bodies[foot_h]).to_degrees();
    let verdict = match tipped_at {
        Some(t) => format!("TIPPED at t={t:.2}s  final={final_tilt:5.2}°"),
        None => format!("HELD     final={final_tilt:5.2}°  |ω|={omega:.4}"),
    };
    let _ = r;
    println!("  {label:48} → {verdict}");
}

/// Sole-tilt from the foot body alone (no World struct needed).
fn sole_tilt_of(body: &RigidBody) -> f32 {
    let n = body.rotation() * Vec3::Z;
    n.z.abs().clamp(0.0, 1.0).acos()
}
