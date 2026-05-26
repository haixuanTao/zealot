//! Controlled inverted pendulum on the nexus GPU rbd pipeline.
//!
//! A single rod on a revolute joint (axis Z), starting ~30° off vertical. A PD
//! controller commands the joint *velocity* motor each step to drive the rod
//! toward upright and hold it there. We measure the fraction of steps the rod
//! spends "upside down" (balanced near vertical) — the classic inverted-pendulum
//! score — and contrast it with an unactuated baseline that just falls.
//!
//! Run: `cargo run --release --example inverted_pendulum --features pendulum`

use std::f32::consts::PI;

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

// --- scene ---
const ROD_LEN: f32 = 2.0; // pivot-to-tip, pole lies along the body's local +X
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 300; // 5 s at 1/60
const EPISODES: usize = 8; // randomized initial conditions per run
const MAX_TILT: f32 = 50.0 * PI / 180.0; // initial tilt sampled from ±this (from vertical)

/// Tiny seeded LCG so the randomized starts are reproducible per seed (no dep).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) // [0,1)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

// --- controller (velocity-PD on the joint angle) ---
// Gravity torque on this rod is ~1.5 Nm, so the motor only needs a modest force
// budget; an over-large max_force flings the (light) rod and the loop diverges.
const KP: f32 = 5.0;
const KD: f32 = 0.5;
const VMAX: f32 = 8.0; // rad/s clamp on the commanded joint velocity
const MOTOR_DAMPING: f32 = 50.0; // stiff velocity tracking so the motor follows cmd
const MOTOR_MAX_FORCE: f32 = 50.0; // Nm the motor may exert (gravity torque ~1.5 Nm)
// LINK_ID and MOTOR_SIGN are taken from argv (defaults below) so we can probe
// the right link index / motor polarity without recompiling:
//   inverted_pendulum [link_id] [motor_sign]
const DEFAULT_LINK_ID: u32 = 1; // dynamic link (root = 0); commands land here
const DEFAULT_MOTOR_SIGN: f32 = 1.0; // +AngX velocity increases θ (toward upright)

const UPRIGHT: f32 = PI / 2.0; // pole pointing +Y, measured from +X axis
const BALANCED_TOL: f32 = 25.0 * PI / 180.0; // "upside down" = within this of vertical

/// Build the single-rod pendulum. With `motor`, the revolute joint carries a
/// velocity motor (damping + max force) so the runtime PD has authority.
fn build(motor: bool, tilt: f32) -> (RigidBodySet, ColliderSet, MultibodyJointSet, GpuSimParams) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    // Fixed pivot at the origin.
    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);

    // Rod: pole along local +X. Rotate by `theta0` about Z so it starts tilted,
    // and place the COM so the rod's inner end sits on the pivot.
    let theta0 = UPRIGHT - tilt; // tilt>0 leans toward +X, tilt<0 toward -X
    let com = Vec3::new(ROD_LEN * theta0.cos(), ROD_LEN * theta0.sin(), 0.0);
    let rod = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(com)
            .rotation(Vec3::new(0.0, 0.0, theta0)),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(ROD_LEN * 0.5, 0.1, 0.1).density(1.0),
        rod,
        &mut bodies,
    );

    let mut joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::ZERO) // on the pivot
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0)); // rod's inner end
    if motor {
        joint = joint
            .motor_velocity(0.0, MOTOR_DAMPING)
            .motor_max_force(MOTOR_MAX_FORCE);
    }
    multibody_joints.insert(root, rod, joint.build(), true);

    let mut sim_params = GpuSimParams::default();
    sim_params.dt = DT;
    sim_params.num_solver_iterations = 8;

    (bodies, colliders, multibody_joints, sim_params)
}

async fn webgpu_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("init WebGPU");
    webgpu.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(webgpu)
}

/// Pole angle from the +X axis (UPRIGHT = +90°), from the rod COM world pose.
fn pole_angle(poses: &[Pose]) -> f32 {
    let com = poses.last().expect("rod pose").translation;
    com.y.atan2(com.x)
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("read poses")
}

/// Run one episode from a given initial `tilt` (rad, from vertical). Returns the
/// fraction of steps the rod spent "upside down" (within BALANCED_TOL of upright).
async fn run(label: &str, controlled: bool, link_id: u32, motor_sign: f32, tilt: f32) -> f32 {
    let (bodies, colliders, multibody_joints, sim_params) = build(controlled, tilt);
    let impulse_joints = ImpulseJointSet::new();
    let gpu = webgpu_backend().await;
    let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
    let envs = vec![(
        &bodies,
        &colliders,
        &impulse_joints,
        &multibody_joints,
        &sim_params,
    )];
    let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);

    let mut theta = pole_angle(&read_poses(&gpu, &state).await);
    let mut prev_theta = theta;
    let mut balanced_steps = 0usize;

    for _step in 1..=STEPS {
        if controlled {
            // Velocity-PD toward upright; θ̇ from the last step's finite difference.
            let err = UPRIGHT - theta;
            let theta_dot = (theta - prev_theta) / DT;
            let cmd = (KP * err - KD * theta_dot).clamp(-VMAX, VMAX) * motor_sign;
            let _ =
                state
                    .multibodies_mut()
                    .set_motor_velocity(&gpu, 0, link_id, JointAxis::AngX, cmd);
        }

        let _ = pipeline.step(&gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(&gpu, &mut state).await;

        prev_theta = theta;
        theta = pole_angle(&read_poses(&gpu, &state).await);

        if (theta - UPRIGHT).abs() < BALANCED_TOL {
            balanced_steps += 1;
        }
    }
    let frac = balanced_steps as f32 / STEPS as f32;
    let start_deg = (UPRIGHT - tilt).to_degrees();
    println!(
        "  [{label}] start θ={start_deg:>5.1}° (tilt {:>+5.1}°) → final {:>6.1}°, upright {:>3.0}% ({:.2}s)",
        tilt.to_degrees(),
        theta.to_degrees(),
        frac * 100.0,
        frac * STEPS as f32 * DT,
    );
    frac
}

/// Run one episode and record the rod's full pose [x,y,z,qx,qy,qz,qw] at every
/// step (incl. start), for 3D rendering. Mirrors `run`'s control loop.
async fn record(controlled: bool, link_id: u32, motor_sign: f32, tilt: f32) -> Vec<[f32; 7]> {
    let (bodies, colliders, multibody_joints, sim_params) = build(controlled, tilt);
    let impulse_joints = ImpulseJointSet::new();
    let gpu = webgpu_backend().await;
    let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
    let envs = vec![(
        &bodies,
        &colliders,
        &impulse_joints,
        &multibody_joints,
        &sim_params,
    )];
    let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);

    let pose7 = |poses: &[Pose]| {
        let p = poses.last().expect("rod");
        let (t, q) = (p.translation, p.rotation);
        [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    };
    let mut traj = Vec::with_capacity(STEPS + 1);
    let p = read_poses(&gpu, &state).await;
    traj.push(pose7(&p));
    let mut theta = p
        .last()
        .unwrap()
        .translation
        .y
        .atan2(p.last().unwrap().translation.x);
    let mut prev_theta = theta;

    for _ in 1..=STEPS {
        if controlled {
            let err = UPRIGHT - theta;
            let theta_dot = (theta - prev_theta) / DT;
            let cmd = (KP * err - KD * theta_dot).clamp(-VMAX, VMAX) * motor_sign;
            let _ =
                state
                    .multibodies_mut()
                    .set_motor_velocity(&gpu, 0, link_id, JointAxis::AngX, cmd);
        }
        let _ = pipeline.step(&gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(&gpu, &mut state).await;
        let p = read_poses(&gpu, &state).await;
        traj.push(pose7(&p));
        prev_theta = theta;
        theta = p
            .last()
            .unwrap()
            .translation
            .y
            .atan2(p.last().unwrap().translation.x);
    }
    traj
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Recording mode: `inverted_pendulum record [tilt_deg] [out.csv]` writes the
    // controlled and baseline rod trajectories (same start) for rendering.
    if args.get(1).map(String::as_str) == Some("record") {
        let tilt = args
            .get(2)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(45.0)
            * PI
            / 180.0;
        let out = args
            .get(3)
            .cloned()
            .unwrap_or_else(|| "/tmp/pendulum_traj.csv".to_string());
        pollster::block_on(async {
            let ctrl = record(true, DEFAULT_LINK_ID, DEFAULT_MOTOR_SIGN, tilt).await;
            let base = record(false, DEFAULT_LINK_ID, DEFAULT_MOTOR_SIGN, tilt).await;
            // full 6-DOF pose for both rods: translation + quaternion
            let mut s = String::from("step,cx,cy,cz,cqx,cqy,cqz,cqw,bx,by,bz,bqx,bqy,bqz,bqw\n");
            for i in 0..ctrl.len() {
                let (c, b) = (ctrl[i], base[i]);
                s.push_str(&format!(
                    "{i},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}\n",
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6],
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                ));
            }
            std::fs::write(&out, s).expect("write csv");
            println!("wrote {} steps to {out}", ctrl.len());
        });
        return;
    }

    let link_id = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LINK_ID);
    let motor_sign = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MOTOR_SIGN);
    // Seed: argv[3] if given (reproducible), else wall-clock nanos (fresh each run).
    let seed = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            | 1
    });
    println!(
        "(link_id={link_id}, motor_sign={motor_sign:+}, seed={seed}, {EPISODES} randomized episodes)"
    );

    pollster::block_on(async {
        let mut rng = Lcg(seed);
        let mut ctrl_scores = Vec::new();
        let mut base_scores = Vec::new();

        for ep in 0..EPISODES {
            // Random initial tilt in ±MAX_TILT (both lean directions), per episode.
            let tilt = rng.range(-MAX_TILT, MAX_TILT);
            let c = run(&format!("ep{ep} ctrl"), true, link_id, motor_sign, tilt).await;
            let b = run(&format!("ep{ep} base"), false, link_id, motor_sign, tilt).await;
            ctrl_scores.push(c);
            base_scores.push(b);
        }

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let min = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
        let solved = ctrl_scores.iter().filter(|&&f| f > 0.8).count();
        println!(
            "\nrandomized inverted pendulum ({EPISODES} eps, tilt ±{:.0}°):\n  \
             controlled  mean {:>3.0}%  worst {:>3.0}%  (≥80% upright in {}/{} eps)\n  \
             baseline    mean {:>3.0}%  worst {:>3.0}%",
            MAX_TILT.to_degrees(),
            mean(&ctrl_scores) * 100.0,
            min(&ctrl_scores) * 100.0,
            solved,
            EPISODES,
            mean(&base_scores) * 100.0,
            min(&base_scores) * 100.0,
        );
    });
}
