//! Headless smoke test for the rbd + from_rapier GPU path.
//!
//! Builds the 20-link revolute-joint multibody pendulum from
//! `bench_multibody_pendulum3`, runs it on the WebGPU backend with no graphics,
//! and prints the tip link's pose over time so we can see it actually swing.

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

/// Build one 20-link pendulum environment (same scene as the testbed bench).
fn build_pendulum(
    num_links: usize,
    num_substeps: u32,
) -> (
    RigidBodySet,
    ColliderSet,
    ImpulseJointSet,
    MultibodyJointSet,
    GpuSimParams,
) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    let rad = 0.4;
    let link_len = 2.0;

    let root_body = RigidBodyBuilder::fixed();
    let mut parent_handle = bodies.insert(root_body);
    let root_collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(root_collider, parent_handle, &mut bodies);

    for i in 0..num_links {
        let x = (i as f32 + 1.0) * link_len;
        let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, 0.0, 0.0));
        let handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(link_len * 0.5, rad, rad);
        colliders.insert_with_parent(collider, handle, &mut bodies);

        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            .build();
        multibody_joints.insert(parent_handle, handle, joint, true);

        parent_handle = handle;
    }

    let mut sim_params = GpuSimParams::default();
    sim_params.num_solver_iterations = num_substeps;

    (
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        sim_params,
    )
}

/// Initialise the WebGPU backend with the same limits the pendulum scene needs.
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
        .expect("Failed to initialize WebGPU backend");
    webgpu.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(webgpu)
}

async fn run() {
    let num_links = 20;
    let num_substeps = 4;
    let steps = 120;

    println!("Headless pendulum — {num_links} links, {num_substeps} substeps, {steps} GPU steps");

    let (bodies, colliders, impulse_joints, multibody_joints, sim_params) =
        build_pendulum(num_links, num_substeps);

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

    // The last pose in the buffer is the tip link — watch it fall under gravity.
    let read_tip = |poses: &[Pose]| *poses.last().expect("at least one pose");

    let poses0: Vec<Pose> = gpu
        .slow_read_vec(state.poses().buffer())
        .await
        .expect("read poses");
    println!("  step   0: tip {:?}", read_tip(&poses0));

    for step in 1..=steps {
        let _stats = pipeline.step(&gpu, &mut state, None).await;
        gpu.synchronize().expect("gpu sync");
        pipeline.auto_resize_buffers(&gpu, &mut state).await;

        if step % 30 == 0 || step == steps {
            let poses: Vec<Pose> = gpu
                .slow_read_vec(state.poses().buffer())
                .await
                .expect("read poses");
            println!("  step {step:>3}: tip {:?}", read_tip(&poses));
        }
    }

    println!(
        "done — {} poses/step, sim advanced {steps} steps on GPU",
        poses0.len()
    );
}

fn main() {
    pollster::block_on(run());
}
