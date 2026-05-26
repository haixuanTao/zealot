//! GPU MLP forward pass on vortx, verified against an nalgebra CPU reference.
//!
//! Run: `cargo run -p zealot-rl --example mlp_forward --features gpu`
//!
//! A tiny policy network: 6 inputs -> hidden 8 (tanh) -> 2 outputs (linear).
//! Forward = Linear (vortx GEMM) + bias (vortx OpAssign add) + tanh (our custom
//! vortx kernel). Verifying this end-to-end also verifies the tanh kernel, since
//! the CPU reference applies `f32::tanh`.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Activation, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const IN: usize = 6;
const HID: usize = 8;
const OUT: usize = 2;

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);

    let gemm = Gemm::from_backend(&backend)?;
    let op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let mut shapes = TensorLayoutBuffers::new(&backend);

    // Deterministic weights/inputs (single sample, batch = 1).
    let w1 = DMatrix::<f32>::from_fn(HID, IN, |r, c| ((r * IN + c) % 7) as f32 * 0.1 - 0.3);
    let b1 = DMatrix::<f32>::from_fn(HID, 1, |r, _| (r % 3) as f32 * 0.1 - 0.1);
    let w2 = DMatrix::<f32>::from_fn(OUT, HID, |r, c| ((r * HID + c) % 5) as f32 * 0.1 - 0.2);
    let b2 = DMatrix::<f32>::from_fn(OUT, 1, |r, _| r as f32 * 0.05);
    let x = DMatrix::<f32>::from_fn(IN, 1, |r, _| (r % 4) as f32 * 0.25 - 0.5);

    // CPU reference.
    let z1c = &w1 * &x + &b1;
    let a1c = z1c.map(|v| v.tanh());
    let z2c = &w2 * &a1c + &b2;

    // GPU tensors.
    let gw1 = Tensor::matrix_from_na(&backend, &w1, BufferUsages::STORAGE)?;
    let gb1 = Tensor::matrix_from_na(&backend, &b1, BufferUsages::STORAGE)?;
    let gw2 = Tensor::matrix_from_na(&backend, &w2, BufferUsages::STORAGE)?;
    let gb2 = Tensor::matrix_from_na(&backend, &b2, BufferUsages::STORAGE)?;
    let gx = Tensor::matrix_from_na(&backend, &x, BufferUsages::STORAGE)?;
    let mut z1 = Tensor::matrix_from_na(&backend, &DMatrix::<f32>::zeros(HID, 1), BufferUsages::STORAGE)?;
    let mut z2 = Tensor::matrix_from_na(
        &backend,
        &DMatrix::<f32>::zeros(OUT, 1),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;

    // Forward: each dependent op in its own pass so wgpu inserts the needed barriers.
    let mut encoder = backend.begin_encoding();
    {
        let mut pass = encoder.begin_pass("gemm1", None);
        gemm.dispatch_naive(&backend, &mut shapes, &mut pass, &mut z1, &gw1, &gx)?; // z1 = W1·x
    }
    {
        let mut pass = encoder.begin_pass("bias1", None);
        op.launch(&backend, &mut shapes, &mut pass, OpAssignVariant::Add, &mut z1, &gb1)?; // z1 += b1
    }
    {
        let mut pass = encoder.begin_pass("tanh1", None);
        act.tanh(&backend, &mut shapes, &mut pass, &mut z1)?; // z1 = tanh(z1)
    }
    {
        let mut pass = encoder.begin_pass("gemm2", None);
        gemm.dispatch_naive(&backend, &mut shapes, &mut pass, &mut z2, &gw2, &z1)?; // z2 = W2·a1
    }
    {
        let mut pass = encoder.begin_pass("bias2", None);
        op.launch(&backend, &mut shapes, &mut pass, OpAssignVariant::Add, &mut z2, &gb2)?; // z2 += b2
    }
    backend.submit(encoder)?;
    backend.synchronize()?;

    let z2_gpu = backend.slow_read_vec(z2.buffer()).await?;

    let mut max_err = 0f32;
    for r in 0..OUT {
        max_err = max_err.max((z2_gpu[r] - z2c[(r, 0)]).abs());
    }

    println!("MLP forward ({IN}->{HID}->{OUT})");
    println!("  gpu out = {:?}", &z2_gpu[..OUT]);
    println!("  cpu out = {:?}", (0..OUT).map(|r| z2c[(r, 0)]).collect::<Vec<_>>());
    println!("  max|gpu - cpu| = {max_err:.3e}");
    anyhow::ensure!(max_err < 1e-4, "MLP GPU output diverged from CPU reference");
    println!("OK — GPU MLP (Linear + bias + tanh) matches CPU. tanh kernel verified.");
    Ok(())
}
