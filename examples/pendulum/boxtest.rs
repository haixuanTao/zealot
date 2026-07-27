//! Minimal statics reproduction: ONE tall rigid box (10x10x80 cm) resting on a
//! fixed ground. No joints, no motors, no multibody. If this can't sit still,
//! the statics defect is in the basic rigid-body contact solve; if it can, the
//! biped instability is specific to the articulated path.
//!
//! GPU (nexus/WebGPU):  cargo run --release --example boxtest --features pendulum
//! CPU (pure rapier):   BOXTEST_CPU=1 cargo run --release --example boxtest --features pendulum

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::RbdSimParams;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{RbdCapacities, RbdPipeline, RbdState};
use rapier3d::prelude::*;

const HX: f32 = 0.05; // half extents: 10 x 10 x 80 cm tall box
const HZ: f32 = 0.4;
const DT: f32 = 0.005;
const SUBSTEPS: u32 = 8;
const STEPS: usize = 400; // 2 s

fn build_scene() -> (
    RigidBodySet,
    ColliderSet,
    ImpulseJointSet,
    MultibodyJointSet,
    RbdSimParams,
    RigidBodyHandle,
) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Ground: top surface at z = 0.
    // Y-UP scene: nexus free-body gravity is hardcoded Y-down in the solver
    // shader (solver.rs: `Vector::Y * -9.81`; only MULTIBODY gravity is
    // configurable). Ground top at y = 0.
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 0.5, 50.0).friction(1.0),
        ground,
        &mut bodies,
    );

    // The tall box, base exactly on the floor.
    let bx = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(0.0, HZ, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(HX, HZ, HX).friction(1.0).density(500.0),
        bx,
        &mut bodies,
    );

    let mut sim_params = RbdSimParams::default();
    sim_params.dt = DT;
    sim_params.num_solver_iterations = SUBSTEPS;

    (
        bodies,
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
        sim_params,
        bx,
    )
}

fn tilt_deg(q: [f32; 4]) -> f32 {
    let (x, z) = (q[0], q[2]);
    let cy = 1.0 - 2.0 * (x * x + z * z);
    cy.clamp(-1.0, 1.0).acos().to_degrees()
}

fn report(step: usize, t: [f32; 3], q: [f32; 4]) {
    println!(
        "  step {step:>3}  pos ({:+.4} {:+.4} {:.4})  tilt {:6.2} deg",
        t[0],
        t[1],
        t[2],
        tilt_deg(q)
    );
}

async fn run_gpu() {
    let (bodies, colliders, ij, mj, sim_params, _bx) = build_scene();
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
    let gpu = KhalGpuBackend::WebGpu(webgpu);
    let mut pipeline = RbdPipeline::new(&gpu).unwrap();
    let envs = vec![(&bodies, &colliders, &ij, &mj, &sim_params)];
    let mut state = RbdState::from_rapier(
        &gpu,
        &envs,
        RbdCapacities {
            batches: 1,
            ..Default::default()
        },
    );

    println!(
        "GPU nexus: tall box {:.0}x{:.0}x{:.0} cm, dt {DT}, {SUBSTEPS} substeps",
        HX * 200.0,
        HX * 200.0,
        HZ * 200.0
    );
    for step in 0..=STEPS {
        if step > 0 {
            let _ = pipeline.step(&gpu, &mut state, None);
            gpu.synchronize().expect("sync");
        }
        if step % 40 == 0 {
            let mut poses = vec![Pose::default(); state.body_poses().len() as usize];
            gpu.slow_read_buffer(state.body_poses().buffer(), &mut poses)
                .await
                .expect("read poses");
            let p = poses[1]; // 0 = ground, 1 = box
            let t = [p.translation.x, p.translation.y, p.translation.z];
            let q = [p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w];
            report(step, t, q);
        }
    }
}

fn run_cpu() {
    let (mut bodies, mut colliders, mut ij, mut mj, _sp, bx) = build_scene();
    let mut ip = IntegrationParameters::default();
    ip.dt = DT;
    ip.num_solver_iterations = SUBSTEPS as usize;
    let mut pipeline = PhysicsPipeline::new();
    let mut islands = IslandManager::new();
    let mut bp = BroadPhaseBvh::new();
    let mut np = NarrowPhase::new();
    let mut ccd = CCDSolver::new();
    println!("CPU rapier: tall box, dt {DT}, {SUBSTEPS} substeps");
    for step in 0..=STEPS {
        if step > 0 {
            pipeline.step(
                Vec3::new(0.0, -9.81, 0.0),
                &ip,
                &mut islands,
                &mut bp,
                &mut np,
                &mut bodies,
                &mut colliders,
                &mut ij,
                &mut mj,
                &mut ccd,
                &(),
                &(),
            );
        }
        if step % 40 == 0 {
            let b = &bodies[bx];
            let tr = b.translation();
            let q = b.rotation();
            report(step, [tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w]);
        }
    }
}

fn main() {
    if std::env::var("BOXTEST_CPU").is_ok() {
        run_cpu();
    } else {
        pollster::block_on(run_gpu());
    }
}
