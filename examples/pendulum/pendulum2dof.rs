//! 2-DOF inverted pendulum (spherical / ball joint) on the nexus GPU rbd pipeline.
//!
//! A rod on a ball joint can tip in *any* direction (2 tilt DOF + a free, inert
//! twist). The pole direction `u` (unit, from pivot to rod COM) should be held at
//! +Y (up). Near upright, a no-twist angular velocity ω=(ωx,0,ωz) moves the pole
//! by du ≈ (−ωz, 0, ωx), so a per-axis PD on the horizontal pole offset balances
//! it. We score the fraction of steps within TOL of vertical, and record the full
//! pose for 3D rendering.
//!
//! Run: `cargo run --release --example pendulum2dof --features pendulum -- [out.csv]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const ROD_LEN: f32 = 2.0;
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 360;
const LINK_ID: u32 = 1;

// PD on the pole's horizontal offset (u.x, u.z). KP/KD in rad/s per unit offset.
const KP: f32 = 9.0;
const KD: f32 = 1.2;
const VMAX: f32 = 8.0;
const MOTOR_DAMPING: f32 = 50.0;
const MOTOR_MAX_FORCE: f32 = 60.0;
const SIGN_X: f32 = 1.0; // flip after a calibration run if it diverges
const SIGN_Z: f32 = 1.0;

const TOL: f32 = 25.0 * std::f32::consts::PI / 180.0; // "upright" = u within this of +Y

/// Initial pole direction: tilted ~30° off vertical, in a diagonal (x AND z).
fn start_dir() -> Vec3 {
    Vec3::new(0.40, 1.0, 0.30).normalize()
}

fn build(controlled: bool) -> (RigidBodySet, ColliderSet, MultibodyJointSet, GpuSimParams) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb = MultibodyJointSet::new();

    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);

    // Rod: pole along local +X; orient local +X onto the (tilted) start direction.
    let u = start_dir();
    let rot = Quat::from_rotation_arc(Vec3::X, u);
    let rod = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(u * ROD_LEN)
            .rotation(rot.to_scaled_axis()),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(ROD_LEN * 0.5, 0.12, 0.12).density(1.0),
        rod,
        &mut bodies,
    );

    // Ball joint: all linear locked, all 3 angular free (2 tilt DOF + free twist).
    let mut joint = SphericalJointBuilder::new()
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0));
    if controlled {
        for axis in [JointAxis::AngX, JointAxis::AngZ] {
            joint = joint
                .motor_velocity(axis, 0.0, MOTOR_DAMPING)
                .motor_max_force(axis, MOTOR_MAX_FORCE);
        }
    }
    mb.insert(root, rod, joint.build(), true);

    let mut sp = GpuSimParams::default();
    sp.dt = DT;
    sp.num_solver_iterations = 8;
    (bodies, colliders, mb, sp)
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
        .expect("webgpu");
    webgpu.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(webgpu)
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("poses")
}

/// pole direction (unit) from the rod COM (pivot at origin).
fn pole_dir(poses: &[Pose]) -> Vec3 {
    let t = poses.last().expect("rod").translation;
    Vec3::new(t.x, t.y, t.z).normalize()
}

async fn run(label: &str, controlled: bool, out_csv: Option<&str>) -> f32 {
    let (bodies, colliders, mb, sp) = build(controlled);
    let impulse = ImpulseJointSet::new();
    let gpu = webgpu_backend().await;
    let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
    let envs = vec![(&bodies, &colliders, &impulse, &mb, &sp)];
    let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);

    let mut u = pole_dir(&read_poses(&gpu, &state).await);
    let mut prev = u;
    let mut balanced = 0usize;
    let mut rows = String::from("step,x,y,z,qx,qy,qz,qw\n");
    let push = |rows: &mut String, step: usize, p: &Pose| {
        let (t, q) = (p.translation, p.rotation);
        rows.push_str(&format!(
            "{step},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}\n",
            t.x, t.y, t.z, q.x, q.y, q.z, q.w
        ));
    };
    {
        let p = read_poses(&gpu, &state).await;
        push(&mut rows, 0, p.last().unwrap());
    }

    for step in 1..=STEPS {
        if controlled {
            // du/dt of the horizontal pole offset for damping
            let dxz = ((u.x - prev.x) / DT, (u.z - prev.z) / DT);
            // ω = (ωx, 0, ωz): du ≈ (−ωz, 0, ωx) ⇒ drive offset to 0
            let wx = (-KP * u.z - KD * dxz.1).clamp(-VMAX, VMAX) * SIGN_X;
            let wz = (KP * u.x + KD * dxz.0).clamp(-VMAX, VMAX) * SIGN_Z;
            let m = state.multibodies_mut();
            let _ = m.set_motor_velocity(&gpu, 0, LINK_ID, JointAxis::AngX, wx);
            let _ = m.set_motor_velocity(&gpu, 0, LINK_ID, JointAxis::AngZ, wz);
        }
        let _ = pipeline.step(&gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(&gpu, &mut state).await;
        let p = read_poses(&gpu, &state).await;
        push(&mut rows, step, p.last().unwrap());
        prev = u;
        u = pole_dir(&p);
        if u.y.acos().abs() < TOL {
            balanced += 1;
        }
        if step % 60 == 0 || step == STEPS {
            let tilt = u.y.clamp(-1.0, 1.0).acos().to_degrees();
            println!(
                "  [{label}] step {step:>3}: tilt-from-up = {tilt:>5.1}°  u=({:.2},{:.2},{:.2})",
                u.x, u.y, u.z
            );
        }
    }
    if let Some(path) = out_csv {
        std::fs::write(path, &rows).expect("write csv");
    }
    let frac = balanced as f32 / STEPS as f32;
    println!(
        "[{label}] upright {balanced}/{STEPS} = {:.0}% ({:.2}s)",
        frac * 100.0,
        frac * STEPS as f32 * DT
    );
    frac
}

fn main() {
    let csv = std::env::args().nth(1);
    pollster::block_on(async {
        let c = run("controlled", true, csv.as_deref()).await;
        let b = run("baseline", false, None).await;
        println!(
            "\n2-DOF inverted pendulum — controlled {:.0}% vs baseline {:.0}%",
            c * 100.0,
            b * 100.0
        );
    });
}
