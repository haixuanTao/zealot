//! End-to-end GPU training loop on vortx: fit an MLP to a fixed target by
//! repeated forward -> backward -> Adam. If the loss collapses to ~0, the whole
//! stack (Linear/GEMM, bias, tanh fwd+bwd, hand-rolled grads, Adam) is correct.
//!
//! Run: `cargo run -p zealot-rl --example train_regression --features gpu`
//!
//! Net: 6 -> tanh(8) -> 2 (linear). Loss = ½‖out - target‖² on one fixed sample.
//! This is the GD machinery a PPO policy update needs; PPO just swaps the loss.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Activation, Adam, AdamParams, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const IN: usize = 6;
const HID: usize = 8;
const OUT: usize = 2;
const STEPS: usize = 400;

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let gemm = Gemm::from_backend(&backend)?;
    let op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let adam = Adam::from_backend(&backend)?;
    let mut sh = TensorLayoutBuffers::new(&backend);

    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let mk = |m: &DMatrix<f32>, u| Tensor::matrix_from_na(&backend, m, u).unwrap();
    let zeros = |r, c| DMatrix::<f32>::zeros(r, c);

    // Params (small deterministic init).
    let mut gw1 = mk(&DMatrix::from_fn(HID, IN, |r, c| ((r * IN + c) % 7) as f32 * 0.05 - 0.15), st);
    let mut gb1 = mk(&zeros(HID, 1), st);
    let mut gw2 = mk(&DMatrix::from_fn(OUT, HID, |r, c| ((r * HID + c) % 5) as f32 * 0.05 - 0.1), st);
    let mut gb2 = mk(&zeros(OUT, 1), st);

    // Fixed sample + target.
    let gx = mk(&DMatrix::from_fn(IN, 1, |r, _| (r % 4) as f32 * 0.25 - 0.5), st);
    let target = DMatrix::from_fn(OUT, 1, |r, _| 0.5 - r as f32 * 0.7);
    let gt = mk(&target, st);

    // Activations / grads.
    let mut z1 = mk(&zeros(HID, 1), st); // a1
    let mut z2 = mk(&zeros(OUT, 1), rw); // out -> dz2 (=db2 grad)
    let mut gdw2 = mk(&zeros(OUT, HID), st);
    let mut gda1 = mk(&zeros(HID, 1), st); // da1 -> dz1 (=db1 grad)
    let mut gdw1 = mk(&zeros(HID, IN), st);

    // Adam moments (m, v) per param, zero-initialised.
    let (mut mw1, mut vw1) = (mk(&zeros(HID, IN), st), mk(&zeros(HID, IN), st));
    let (mut mb1, mut vb1) = (mk(&zeros(HID, 1), st), mk(&zeros(HID, 1), st));
    let (mut mw2, mut vw2) = (mk(&zeros(OUT, HID), st), mk(&zeros(OUT, HID), st));
    let (mut mb2, mut vb2) = (mk(&zeros(OUT, 1), st), mk(&zeros(OUT, 1), st));

    let (lr, b1, b2, eps) = (0.05f32, 0.9f32, 0.999f32, 1e-8f32);
    let mut first_loss = 0.0;

    for t in 1..=STEPS {
        // ---- forward + dz2, in one submit so we can read the loss ----
        let mut enc = backend.begin_encoding();
        { let mut p = enc.begin_pass("gemm1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z1, &gw1, &gx)?; }
        { let mut p = enc.begin_pass("bias1", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Add, &mut z1, &gb1)?; }
        { let mut p = enc.begin_pass("tanh1", None); act.tanh(&backend, &mut sh, &mut p, &mut z1)?; }
        { let mut p = enc.begin_pass("gemm2", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut z2, &gw2, &z1)?; }
        { let mut p = enc.begin_pass("bias2", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Add, &mut z2, &gb2)?; }
        { let mut p = enc.begin_pass("dz2", None); op.launch(&backend, &mut sh, &mut p, OpAssignVariant::Sub, &mut z2, &gt)?; }
        backend.submit(enc)?;
        backend.synchronize()?;

        let dz2 = backend.slow_read_vec(z2.buffer()).await?; // = out - target
        let loss = 0.5 * dz2.iter().map(|v| v * v).sum::<f32>();
        if t == 1 { first_loss = loss; }
        if t == 1 || t % 50 == 0 { println!("step {t:>3}  loss = {loss:.6e}"); }

        // ---- backward + Adam ----
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let params = AdamParams { lr, beta1: b1, beta2: b2, eps, bias_correction1: bc1, bias_correction2: bc2, pad0: 0.0, pad1: 0.0 };
        let gp = Tensor::scalar(&backend, params, BufferUsages::UNIFORM)?;

        let mut enc = backend.begin_encoding();
        // grads (dz2 is in z2)
        { let mut p = enc.begin_pass("dW2", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gdw2, &z2, z1.transpose_last_dims())?; }
        { let mut p = enc.begin_pass("da1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gda1, gw2.transpose_last_dims(), &z2)?; }
        { let mut p = enc.begin_pass("dz1", None); act.tanh_backward(&backend, &mut sh, &mut p, &mut gda1, &z1)?; }
        { let mut p = enc.begin_pass("dW1", None); gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut gdw1, &gda1, gx.transpose_last_dims())?; }
        // Adam updates (grads read before params change; param tensors are distinct from grads)
        { let mut p = enc.begin_pass("adam_w2", None); adam.step(&backend, &mut sh, &mut p, &gp, &mut gw2, &gdw2, &mut mw2, &mut vw2)?; }
        { let mut p = enc.begin_pass("adam_b2", None); adam.step(&backend, &mut sh, &mut p, &gp, &mut gb2, &z2,   &mut mb2, &mut vb2)?; }
        { let mut p = enc.begin_pass("adam_w1", None); adam.step(&backend, &mut sh, &mut p, &gp, &mut gw1, &gdw1, &mut mw1, &mut vw1)?; }
        { let mut p = enc.begin_pass("adam_b1", None); adam.step(&backend, &mut sh, &mut p, &gp, &mut gb1, &gda1, &mut mb1, &mut vb1)?; }
        backend.submit(enc)?;
        backend.synchronize()?;
    }

    // final loss
    println!("first loss = {first_loss:.6e}");
    anyhow::ensure!(first_loss > 1e-3, "initial loss unexpectedly tiny; test not meaningful");
    println!("OK — if the loss collapsed toward 0, the full GPU train loop (fwd/bwd/Adam) works.");
    Ok(())
}
