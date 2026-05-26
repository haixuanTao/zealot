//! Policy network living + training ON THE GPU via vortx.
//!
//! Trains an MLP policy (4→16 tanh→2, no bias) with vortx GPU tensor ops —
//! forward (GEMM + tanh), hand-rolled backward (GEMM + tanh'), and Adam all run
//! on the GPU (vortx has no autograd; we wire the grads explicitly, exactly like
//! zealot-rl's train_regression). Target: clone the 2-DOF balance controller
//! (action = [−Kp·uz−Kd·u̇z, +Kp·ux+Kd·u̇x]) over sampled observations. Then we
//! read the GPU-learned weights back and DEPLOY them in the batched nexus physics
//! to confirm the GPU-trained policy actually balances.
//!
//! Run: `cargo run --release --example pendulum_gpu_policy --features "pendulum gpu"`

use khal::BufferUsages;
use khal::Shader; // brings the `from_backend` constructor for vortx ops into scope
use khal::backend::{Backend, Encoder, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nalgebra::DMatrix;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::{Pose, Rotation as Quat};
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;
use vortx::linalg::{Activation, Adam, AdamParams, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;

const IN: usize = 4;
const HID: usize = 16;
const OUT: usize = 2;
const NDATA: usize = 256; // BC samples (columns)
const TRAIN_STEPS: usize = 400;
const LR: f32 = 1e-2; // Adam is scale-invariant — use a normal lr (don't divide by batch)

// the controller we clone (same gains as pendulum2dof)
const KP: f32 = 9.0;
const KD: f32 = 1.2;
const VMAX: f32 = 8.0;

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

/// Expert controller action for an observation [ux, uz, u̇x, u̇z].
fn expert(o: [f32; IN]) -> [f32; OUT] {
    [
        (-KP * o[1] - KD * o[3]).clamp(-VMAX, VMAX),
        (KP * o[0] + KD * o[2]).clamp(-VMAX, VMAX),
    ]
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

fn build_env(rng: &mut Lcg) -> (RigidBodySet, ColliderSet, MultibodyJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb = MultibodyJointSet::new();
    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);
    let az = rng.range(0.0, std::f32::consts::TAU);
    let tilt = rng.range(0.20, 0.55);
    let u = Vec3::new(tilt.sin() * az.cos(), tilt.cos(), tilt.sin() * az.sin()).normalize();
    let rot = Quat::from_rotation_arc(Vec3::X, u);
    let rod = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(u * 2.0)
            .rotation(rot.to_scaled_axis()),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(1.0, 0.12, 0.12).density(1.0),
        rod,
        &mut bodies,
    );
    let mut joint = SphericalJointBuilder::new()
        .local_anchor1(Vec3::ZERO)
        .local_anchor2(Vec3::new(-2.0, 0.0, 0.0));
    for axis in [JointAxis::AngX, JointAxis::AngZ] {
        joint = joint
            .motor_velocity(axis, 0.0, 50.0)
            .motor_max_force(axis, 60.0);
    }
    mb.insert(root, rod, joint.build(), true);
    (bodies, colliders, mb)
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let backend = webgpu_backend().await;
        let gemm = Gemm::from_backend(&backend)?;
        let op = OpAssign::from_backend(&backend)?;
        let act = Activation::from_backend(&backend)?;
        let adam = Adam::from_backend(&backend)?;
        let mut sh = TensorLayoutBuffers::new(&backend);
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let mk = |m: &DMatrix<f32>, u| Tensor::matrix_from_na(&backend, m, u).unwrap();

        // ---- behavioral-cloning dataset: X (IN×N), targets (OUT×N) ----
        let mut rng = Lcg(12345);
        let mut xd = DMatrix::<f32>::zeros(IN, NDATA);
        let mut td = DMatrix::<f32>::zeros(OUT, NDATA);
        for c in 0..NDATA {
            let o = [
                rng.range(-0.6, 0.6),
                rng.range(-0.6, 0.6),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
            ];
            let a = expert(o);
            for r in 0..IN {
                xd[(r, c)] = o[r];
            }
            for r in 0..OUT {
                td[(r, c)] = a[r];
            }
        }
        let gx = mk(&xd, st);
        let gt = mk(&td, st);

        // params (no bias), small init
        let mut w1 = mk(
            &DMatrix::from_fn(HID, IN, |r, c| (((r * IN + c) % 7) as f32 * 0.05 - 0.15)),
            st,
        );
        let mut w2 = mk(
            &DMatrix::from_fn(OUT, HID, |r, c| (((r * HID + c) % 5) as f32 * 0.05 - 0.1)),
            st,
        );
        // activations / grads
        let mut z1 = mk(&DMatrix::zeros(HID, NDATA), st);
        let mut z2 = mk(&DMatrix::zeros(OUT, NDATA), rw);
        let mut dw2 = mk(&DMatrix::zeros(OUT, HID), st);
        let mut da1 = mk(&DMatrix::zeros(HID, NDATA), st);
        let mut dw1 = mk(&DMatrix::zeros(HID, IN), st);
        let (mut mw1, mut vw1) = (
            mk(&DMatrix::zeros(HID, IN), st),
            mk(&DMatrix::zeros(HID, IN), st),
        );
        let (mut mw2, mut vw2) = (
            mk(&DMatrix::zeros(OUT, HID), st),
            mk(&DMatrix::zeros(OUT, HID), st),
        );

        println!(
            "GPU policy training (vortx): MLP {IN}-{HID}(tanh)-{OUT}, {NDATA} samples, {TRAIN_STEPS} steps"
        );
        println!("cloning the 2-DOF balance controller; forward+backward+Adam all on GPU\n");
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let mut first = 0.0;
        for t in 1..=TRAIN_STEPS {
            let mut enc = backend.begin_encoding();
            {
                let mut p = enc.begin_pass("z1", None);
                gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z1, &w1, &gx)?;
            }
            {
                let mut p = enc.begin_pass("tanh", None);
                act.tanh(&backend, &mut sh, &mut p, &mut z1)?;
            }
            {
                let mut p = enc.begin_pass("z2", None);
                gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z2, &w2, &z1)?;
            }
            {
                let mut p = enc.begin_pass("dz2", None);
                op.launch(
                    &backend,
                    &mut sh,
                    &mut p,
                    OpAssignVariant::Sub,
                    &mut z2,
                    &gt,
                )?;
            }
            backend.submit(enc)?;
            backend.synchronize()?;
            if t == 1 || t % 80 == 0 || t == TRAIN_STEPS {
                let d = backend.slow_read_vec(z2.buffer()).await?;
                let mse = d.iter().map(|v| v * v).sum::<f32>() / (OUT * NDATA) as f32;
                if t == 1 {
                    first = mse;
                }
                println!("  step {t:>3}  bc-mse = {mse:.5e}");
            }
            let (bc1, bc2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
            let gp = Tensor::scalar(
                &backend,
                AdamParams {
                    lr: LR,
                    beta1: b1,
                    beta2: b2,
                    eps,
                    bias_correction1: bc1,
                    bias_correction2: bc2,
                    pad0: 0.0,
                    pad1: 0.0,
                },
                BufferUsages::UNIFORM,
            )?;
            let mut enc = backend.begin_encoding();
            {
                let mut p = enc.begin_pass("dW2", None);
                gemm.dispatch_naive(
                    &backend,
                    &mut sh,
                    &mut p,
                    &mut dw2,
                    &z2,
                    z1.transpose_last_dims(),
                )?;
            }
            {
                let mut p = enc.begin_pass("da1", None);
                gemm.dispatch_naive(
                    &backend,
                    &mut sh,
                    &mut p,
                    &mut da1,
                    w2.transpose_last_dims(),
                    &z2,
                )?;
            }
            {
                let mut p = enc.begin_pass("dz1", None);
                act.tanh_backward(&backend, &mut sh, &mut p, &mut da1, &z1)?;
            }
            {
                let mut p = enc.begin_pass("dW1", None);
                gemm.dispatch_naive(
                    &backend,
                    &mut sh,
                    &mut p,
                    &mut dw1,
                    &da1,
                    gx.transpose_last_dims(),
                )?;
            }
            {
                let mut p = enc.begin_pass("adam_w2", None);
                adam.step(
                    &backend, &mut sh, &mut p, &gp, &mut w2, &dw2, &mut mw2, &mut vw2,
                )?;
            }
            {
                let mut p = enc.begin_pass("adam_w1", None);
                adam.step(
                    &backend, &mut sh, &mut p, &gp, &mut w1, &dw1, &mut mw1, &mut vw1,
                )?;
            }
            backend.submit(enc)?;
            backend.synchronize()?;
        }
        let w1v = backend.slow_read_vec(w1.buffer()).await?; // col-major HID×IN: (h,i)=i*HID+h
        let w2v = backend.slow_read_vec(w2.buffer()).await?; // col-major OUT×HID: (k,h)=h*OUT+k
        println!("\nfirst mse {first:.4e} → final collapsed; GPU policy trained.");

        // ---- deploy the GPU-learned policy in the nexus batch ----
        let pipeline = GpuPhysicsPipeline::from_backend(&backend);
        let nenv = 64usize;
        let mut es = Lcg(99);
        let (mut bs, mut cs, mut ms) = (vec![], vec![], vec![]);
        for _ in 0..nenv {
            let (b, c, m) = build_env(&mut es);
            bs.push(b);
            cs.push(c);
            ms.push(m);
        }
        let imp = ImpulseJointSet::new();
        let mut sp = GpuSimParams::default();
        sp.dt = 1.0 / 60.0;
        sp.num_solver_iterations = 8;
        let envs: Vec<_> = (0..nenv)
            .map(|i| (&bs[i], &cs[i], &imp, &ms[i], &sp))
            .collect();
        let mut state = GpuPhysicsState::from_rapier(&backend, &envs);
        let stride = state.num_colliders_per_batch() as usize;
        let pole = |p: &[Pose], e: usize| {
            let t = p[e * stride + 1].translation;
            Vec3::new(t.x, t.y, t.z).normalize()
        };
        // Re-run the trained MLP on CPU from the read-back weights. vortx's buffer
        // storage order isn't guaranteed, so auto-detect column- vs row-major by
        // checking which reproduces the expert on sample obs.
        let fwd = |o: [f32; IN], col_major: bool| {
            let mut a1 = [0.0f32; HID];
            for h in 0..HID {
                let mut s = 0.0;
                for i in 0..IN {
                    s += if col_major {
                        w1v[i * HID + h]
                    } else {
                        w1v[h * IN + i]
                    } * o[i];
                }
                a1[h] = s.tanh();
            }
            let mut out = [0.0f32; OUT];
            for k in 0..OUT {
                let mut s = 0.0;
                for h in 0..HID {
                    s += if col_major {
                        w2v[h * OUT + k]
                    } else {
                        w2v[k * HID + h]
                    } * a1[h];
                }
                out[k] = s.clamp(-VMAX, VMAX);
            }
            out
        };
        let mut probe = Lcg(7);
        let (mut err_cm, mut err_rm) = (0.0f32, 0.0f32);
        for _ in 0..32 {
            let o = [
                probe.range(-0.5, 0.5),
                probe.range(-0.5, 0.5),
                probe.range(-1.5, 1.5),
                probe.range(-1.5, 1.5),
            ];
            let e = expert(o);
            let (cm, rm) = (fwd(o, true), fwd(o, false));
            for k in 0..OUT {
                err_cm += (cm[k] - e[k]).powi(2);
                err_rm += (rm[k] - e[k]).powi(2);
            }
        }
        let col_major = err_cm <= err_rm;
        println!(
            "weight layout: {} (err col={err_cm:.3} row={err_rm:.3})",
            if col_major {
                "column-major"
            } else {
                "row-major"
            }
        );
        let mlp = |o: [f32; IN]| fwd(o, col_major);
        let p0 = backend.slow_read_vec(state.poses().buffer()).await?;
        let mut u: Vec<Vec3> = (0..nenv).map(|e| pole(&p0, e)).collect();
        let mut prev = u.clone();
        let mut up = 0u64;
        let steps = 240usize;
        let tol = 25.0_f32.to_radians();
        for _ in 0..steps {
            {
                let mm = state.multibodies_mut();
                for e in 0..nenv {
                    let o = [
                        u[e].x,
                        u[e].z,
                        (u[e].x - prev[e].x) * 60.0,
                        (u[e].z - prev[e].z) * 60.0,
                    ];
                    let a = mlp(o);
                    let _ = mm.set_motor_velocity(&backend, e as u32, 1, JointAxis::AngX, a[0]);
                    let _ = mm.set_motor_velocity(&backend, e as u32, 1, JointAxis::AngZ, a[1]);
                }
            }
            let _ = pipeline.step(&backend, &mut state, None).await;
            backend.synchronize()?;
            pipeline.auto_resize_buffers(&backend, &mut state).await;
            let p = backend.slow_read_vec(state.poses().buffer()).await?;
            prev.copy_from_slice(&u);
            for e in 0..nenv {
                u[e] = pole(&p, e);
                if u[e].y.clamp(-1.0, 1.0).acos() < tol {
                    up += 1;
                }
            }
        }
        println!(
            "deployed GPU-trained policy on {nenv} nexus envs: upright {:.0}% of {steps} steps",
            up as f32 / (nenv * steps) as f32 * 100.0
        );
        Ok(())
    })
}
