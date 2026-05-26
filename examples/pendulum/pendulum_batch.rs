//! Batched, randomized 2-DOF inverted pendulums on the nexus GPU rbd pipeline —
//! the vectorized-environment substrate an RL setup needs.
//!
//! N independent ball-joint pendulums, each with a randomly sampled initial tilt
//! (direction + magnitude), are packed into one batched `GpuPhysicsState` via
//! `from_rapier` and stepped together on the GPU. Each step: one pose readback
//! for the whole batch, then a per-env two-axis velocity-PD writes that env's
//! motor targets (the `batch` arg). Reports the score distribution + throughput.
//!
//! Run: `cargo run --release --example pendulum_batch --features pendulum -- [N] [seed]`

use std::time::Instant;

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const ROD_LEN: f32 = 2.0;
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 240;
const LINK_ID: u32 = 1;

const KP: f32 = 9.0;
const KD: f32 = 1.2;
const VMAX: f32 = 8.0;
const MOTOR_DAMPING: f32 = 50.0;
const MOTOR_MAX_FORCE: f32 = 60.0;
const TOL: f32 = 25.0 * std::f32::consts::PI / 180.0;

const TILT_MIN: f32 = 12.0 * std::f32::consts::PI / 180.0;
const TILT_MAX: f32 = 40.0 * std::f32::consts::PI / 180.0;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
}

/// One env: ball-joint pendulum with a random initial tilt. Returns the sets +
/// the initial pole direction (for reporting).
fn build_env(rng: &mut Lcg) -> (RigidBodySet, ColliderSet, MultibodyJointSet, Vec3) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb = MultibodyJointSet::new();

    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);

    // random tilt: azimuth in [0,2π), polar tilt in [TILT_MIN, TILT_MAX] from +Y
    let az = rng.range(0.0, std::f32::consts::TAU);
    let tilt = rng.range(TILT_MIN, TILT_MAX);
    let u = Vec3::new(tilt.sin() * az.cos(), tilt.cos(), tilt.sin() * az.sin()).normalize();

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

    let mut joint = SphericalJointBuilder::new()
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0));
    for axis in [JointAxis::AngX, JointAxis::AngZ] {
        joint = joint
            .motor_velocity(axis, 0.0, MOTOR_DAMPING)
            .motor_max_force(axis, MOTOR_MAX_FORCE);
    }
    mb.insert(root, rod, joint.build(), true);

    (bodies, colliders, mb, u)
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

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    pollster::block_on(async {
        let mut rng = Lcg(seed | 1);
        let mut all_b = Vec::with_capacity(n);
        let mut all_c = Vec::with_capacity(n);
        let mut all_mb = Vec::with_capacity(n);
        let mut starts = Vec::with_capacity(n);
        for _ in 0..n {
            let (b, c, mb, u) = build_env(&mut rng);
            all_b.push(b);
            all_c.push(c);
            all_mb.push(mb);
            starts.push(u);
        }
        let impulse = ImpulseJointSet::new();
        let mut sp = GpuSimParams::default();
        sp.dt = DT;
        sp.num_solver_iterations = 8;

        let envs: Vec<_> = (0..n)
            .map(|i| (&all_b[i], &all_c[i], &impulse, &all_mb[i], &sp))
            .collect();

        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
        let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);
        let stride = state.num_colliders_per_batch() as usize;
        println!("batched 2-DOF pendulums: N={n}, stride={stride} colliders/env, {STEPS} steps");

        let pole = |poses: &[Pose], b: usize| {
            let t = poses[b * stride + 1].translation; // rod = collider 1
            Vec3::new(t.x, t.y, t.z).normalize()
        };

        let p = read_poses(&gpu, &state).await;
        let mut u: Vec<Vec3> = (0..n).map(|b| pole(&p, b)).collect();
        let mut prev = u.clone();
        let mut upright = vec![0usize; n];

        let t0 = Instant::now();
        for _ in 1..=STEPS {
            {
                let m = state.multibodies_mut();
                for b in 0..n {
                    let du = ((u[b].x - prev[b].x) / DT, (u[b].z - prev[b].z) / DT);
                    let wx = (-KP * u[b].z - KD * du.1).clamp(-VMAX, VMAX);
                    let wz = (KP * u[b].x + KD * du.0).clamp(-VMAX, VMAX);
                    let _ = m.set_motor_velocity(&gpu, b as u32, LINK_ID, JointAxis::AngX, wx);
                    let _ = m.set_motor_velocity(&gpu, b as u32, LINK_ID, JointAxis::AngZ, wz);
                }
            }
            let _ = pipeline.step(&gpu, &mut state, None).await;
            gpu.synchronize().expect("sync");
            pipeline.auto_resize_buffers(&gpu, &mut state).await;
            let p = read_poses(&gpu, &state).await;
            prev.copy_from_slice(&u);
            for b in 0..n {
                u[b] = pole(&p, b);
                if u[b].y.clamp(-1.0, 1.0).acos() < TOL {
                    upright[b] += 1;
                }
            }
        }
        let wall = t0.elapsed().as_secs_f64();

        // score distribution
        let fr: Vec<f32> = upright.iter().map(|&c| c as f32 / STEPS as f32).collect();
        let mean = fr.iter().sum::<f32>() / n as f32;
        let mn = fr.iter().cloned().fold(f32::INFINITY, f32::min);
        let solved = fr.iter().filter(|&&f| f > 0.8).count();
        let start_tilts: Vec<f32> = starts.iter().map(|u| u.y.acos().to_degrees()).collect();
        let max_tilt = start_tilts.iter().cloned().fold(0.0, f32::max);

        println!(
            "\nrandom start tilts {:.0}–{:.0}° from vertical",
            start_tilts.iter().cloned().fold(f32::INFINITY, f32::min),
            max_tilt
        );
        println!(
            "balanced (≥80% upright): {solved}/{n} envs   mean uptime {:.0}%   worst {:.0}%",
            mean * 100.0,
            mn * 100.0
        );
        println!(
            "throughput: {n} envs × {STEPS} steps in {:.2}s  =  {:.0} env-steps/s  ({:.2} ms/step for the whole batch)",
            wall,
            n as f64 * STEPS as f64 / wall,
            1000.0 * wall / STEPS as f64
        );
    });
}
