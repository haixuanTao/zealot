//! Verifies nexus's new per-env reset (`GpuPhysicsState::reset_env_from`).
//!
//! A correct reset must leave the reset environment behaving EXACTLY like a
//! freshly built one. Test: batch B runs 40 steps, then env0 is reset to config
//! S; reference batch R has env0 = config S from the start. After the reset, B's
//! env0 trajectory must match R's env0 step-for-step (and B's other envs must be
//! untouched). Passive pendulums (no motor) → fully deterministic.
//!
//! Run: `cargo run --release --example pendulum_reset --features pendulum`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const ROD_LEN: f32 = 2.0;
const DT: f32 = 1.0 / 60.0;
const NENV: usize = 4;
const PRE: usize = 40; // steps before reset
const POST: usize = 40; // steps compared after reset

// config S (the reset target): a fixed tilt direction.
const S_TILT: f32 = 28.0 * std::f32::consts::PI / 180.0;
const S_AZ: f32 = 0.7;

fn dir(tilt: f32, az: f32) -> Vec3 {
    Vec3::new(tilt.sin() * az.cos(), tilt.cos(), tilt.sin() * az.sin()).normalize()
}

fn build_env(u: Vec3) -> (RigidBodySet, ColliderSet, MultibodyJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb = MultibodyJointSet::new();
    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);
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
    // passive: ball joint, no motor → deterministic free swing
    let joint = SphericalJointBuilder::new()
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0));
    mb.insert(root, rod, joint.build(), true);
    (bodies, colliders, mb)
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

async fn read(gpu: &KhalGpuBackend, s: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(s.poses().buffer()).await.expect("poses")
}

fn build_state<'a>(
    gpu: &KhalGpuBackend,
    dirs: &[Vec3],
    keep: &'a mut (
        Vec<RigidBodySet>,
        Vec<ColliderSet>,
        Vec<MultibodyJointSet>,
        ImpulseJointSet,
        nexus3d::rbd::dynamics::GpuSimParams,
    ),
) -> GpuPhysicsState {
    for u in dirs {
        let (b, c, m) = build_env(*u);
        keep.0.push(b);
        keep.1.push(c);
        keep.2.push(m);
    }
    keep.4.dt = DT;
    keep.4.num_solver_iterations = 8;
    let envs: Vec<_> = (0..dirs.len())
        .map(|i| (&keep.0[i], &keep.1[i], &keep.3, &keep.2[i], &keep.4))
        .collect();
    GpuPhysicsState::from_rapier(gpu, &envs)
}

fn main() {
    pollster::block_on(async {
        let gpu = webgpu_backend().await;
        let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
        let us = dir(S_TILT, S_AZ);
        let pole = |p: &[Pose], e: usize| {
            let t = p[e * 2 + 1].translation;
            Vec3::new(t.x, t.y, t.z).normalize()
        };

        // Reference: 4-env batch, env0 = config S from the start.
        let mut rk = (
            vec![],
            vec![],
            vec![],
            ImpulseJointSet::new(),
            Default::default(),
        );
        let ref_dirs = [us, dir(0.5, 1.0), dir(0.4, 2.0), dir(0.6, 3.0)];
        let mut rstate = build_state(&gpu, &ref_dirs, &mut rk);
        let mut refs = Vec::new();
        for _ in 0..POST {
            let _ = pipeline.step(&gpu, &mut rstate, None).await;
            gpu.synchronize().unwrap();
            pipeline.auto_resize_buffers(&gpu, &mut rstate).await;
            refs.push(pole(&read(&gpu, &rstate).await, 0));
        }

        // Test batch: different env0 start; run PRE steps, then reset env0 -> S.
        let mut bk = (
            vec![],
            vec![],
            vec![],
            ImpulseJointSet::new(),
            Default::default(),
        );
        let b_dirs = [dir(0.9, 4.5), dir(0.5, 1.0), dir(0.4, 2.0), dir(0.6, 3.0)];
        let mut bstate = build_state(&gpu, &b_dirs, &mut bk);
        for _ in 0..PRE {
            let _ = pipeline.step(&gpu, &mut bstate, None).await;
            gpu.synchronize().unwrap();
            pipeline.auto_resize_buffers(&gpu, &mut bstate).await;
        }
        let env1_before = pole(&read(&gpu, &bstate).await, 1);

        // build a single-env state for config S, reset env0 from it
        let mut sk = (
            vec![],
            vec![],
            vec![],
            ImpulseJointSet::new(),
            Default::default(),
        );
        let sstate = build_state(&gpu, &[us], &mut sk);
        bstate.reset_env_from(&gpu, 0, &sstate).await;

        let after = read(&gpu, &bstate).await;
        let env0_reset = pole(&after, 0);
        let env1_after = pole(&after, 1);
        println!(
            "env0 target dir   = ({:+.3},{:+.3},{:+.3})",
            us.x, us.y, us.z
        );
        println!(
            "env0 after reset  = ({:+.3},{:+.3},{:+.3})  (should match target)",
            env0_reset.x, env0_reset.y, env0_reset.z
        );
        println!(
            "env1 reset-induced change = {:.5}  (should be ~0)",
            (env1_after - env1_before).length()
        );

        // step both and compare env0 trajectories
        let mut max_err: f32 = 0.0;
        for k in 0..POST {
            let _ = pipeline.step(&gpu, &mut bstate, None).await;
            gpu.synchronize().unwrap();
            pipeline.auto_resize_buffers(&gpu, &mut bstate).await;
            let b0 = pole(&read(&gpu, &bstate).await, 0);
            max_err = max_err.max((b0 - refs[k]).length());
        }
        println!("\nmax |reset-env0 − fresh-env0| over {POST} steps = {max_err:.5}");
        println!(
            "{}",
            if max_err < 1e-2 {
                "RESET CONSISTENT ✓ (reset env evolves identically to a fresh one)"
            } else {
                "INCONSISTENT ✗ (missing state buffer in reset)"
            }
        );
    });
}
