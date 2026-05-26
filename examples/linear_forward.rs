//! GPU smoke test: a linear-layer forward `Y = W · X` on vortx, verified against CPU.
//!
//! Run: `cargo run -p zealot-rl --example linear_forward --features gpu`
//! (requires the cargo-gpu rust-gpu toolchain, since it builds vortx's shaders).
//!
//! This proves the `zealot-rl` ↔ vortx integration end to end — tensor allocation,
//! GEMM dispatch, and readback — which is the core of an MLP linear layer. Adding
//! the nonlinearity (tanh) is the next step and needs a small custom rust-gpu kernel,
//! because vortx 0.1 has no activation ops.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::Gemm;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);

    let gemm = Gemm::from_backend(&backend)?;
    let mut shapes = TensorLayoutBuffers::new(&backend);

    // A linear layer applied over a batch: W is (out x in), X is (in x batch),
    // Y = W · X is (out x batch). Square here (out = in = batch = N) to match the
    // shape vortx's GEMM is known-good on; non-square comes once this is verified.
    let n = 16usize;
    // Deterministic fill (avoids nalgebra's `rand` feature; keeps the test reproducible).
    let w_cpu = DMatrix::<f32>::from_fn(n, n, |r, c| ((r * 3 + c * 7) % 11) as f32 * 0.1 - 0.5);
    let x_cpu = DMatrix::<f32>::from_fn(n, n, |r, c| ((r * 5 + c * 2) % 9) as f32 * 0.1 - 0.4);
    let y_ref = &w_cpu * &x_cpu; // CPU reference

    let w = Tensor::matrix_from_na(&backend, &w_cpu, BufferUsages::STORAGE)?;
    let x = Tensor::matrix_from_na(&backend, &x_cpu, BufferUsages::STORAGE)?;
    let mut y = Tensor::matrix_from_na(
        &backend,
        &DMatrix::<f32>::zeros(n, n),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;

    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("linear-forward", None);
    gemm.dispatch_naive(&backend, &mut shapes, &mut pass, &mut y, &w, &x)?;
    drop(pass);
    backend.submit(encoder)?;
    backend.synchronize()?;

    // vortx tensors are row-major; nalgebra DMatrix is column-major.
    let y_gpu = backend.slow_read_vec(y.buffer()).await?;
    let mut max_err = 0f32;
    for r in 0..n {
        for c in 0..n {
            let diff = (y_gpu[r * n + c] - y_ref[(r, c)]).abs();
            max_err = max_err.max(diff);
        }
    }

    println!("Y = W·X  (n={n})   max|gpu - cpu| = {max_err:.3e}");
    anyhow::ensure!(max_err < 1e-3, "GPU result diverged from CPU reference");
    println!("OK — vortx GEMM matches CPU. zealot-rl ↔ vortx verified end-to-end.");
    Ok(())
}
