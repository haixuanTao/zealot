//! GPU ELU forward + backward on vortx, verified against the CPU reference that
//! `zealot-rl` uses (`net.rs::elu` / `elu_grad_from_act`).
//!
//! Run: `cargo run --release --example elu_check --features gpu`
//!
//! ELU(alpha=1): `f(x) = x if x > 0 else exp(x) - 1`. The backward is expressed
//! purely from the cached post-activation `y = elu(x)`: `f'(x) = 1 if y > 0 else
//! y + 1` (since `exp(x) = elu(x) + 1` for `x <= 0`, and `y > 0 <=> x > 0`). This
//! is exactly the CPU formulation, so a match here verifies both new kernels.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::Activation;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const ROWS: usize = 7; // hidden width
const COLS: usize = 5; // batch (columns) — exercises the >1 batch path too

// CPU references — identical to zealot-rl/src/net.rs.
fn elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}
fn elu_grad_from_act(a: f32) -> f32 {
    if a > 0.0 { 1.0 } else { a + 1.0 }
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let act = Activation::from_backend(&backend)?;
    let mut shapes = TensorLayoutBuffers::new(&backend);
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    // Pre-activations spanning negatives, zero, and positives so both ELU branches
    // (and the y>0 split in the backward) are exercised.
    let x = DMatrix::<f32>::from_fn(ROWS, COLS, |r, c| {
        (r as f32 - 3.0) * 0.7 + (c as f32 - 2.0) * 0.3
    });
    // An arbitrary upstream gradient to multiply through the backward.
    let g_in = DMatrix::<f32>::from_fn(ROWS, COLS, |r, c| 0.5 + (r * COLS + c) as f32 * 0.05);

    // ---- forward: a = elu(x), in place ----
    let mut a = Tensor::matrix_from_na(&backend, &x, rw)?;
    {
        let mut enc = backend.begin_encoding();
        {
            let mut p = enc.begin_pass("elu", None);
            act.elu(&backend, &mut shapes, &mut p, &mut a)?;
        }
        backend.submit(enc)?;
        backend.synchronize()?;
    }
    let y_gpu = backend.slow_read_vec(a.buffer()).await?;

    // ---- backward: g *= elu'(y), in place (uses the cached forward output a) ----
    let mut g = Tensor::matrix_from_na(&backend, &g_in, rw)?;
    {
        let mut enc = backend.begin_encoding();
        {
            let mut p = enc.begin_pass("elu_backward", None);
            act.elu_backward(&backend, &mut shapes, &mut p, &mut g, &a)?;
        }
        backend.submit(enc)?;
        backend.synchronize()?;
    }
    let g_gpu = backend.slow_read_vec(g.buffer()).await?;

    // ---- compare against CPU ----
    // `matrix_from_na` lays the buffer out ROW-major (logical [r,c] -> r*COLS + c),
    // not nalgebra's column-major `as_slice()`. Build the reference in the same
    // row-major order so the positional comparison is apples-to-apples. (This is
    // the same buffer-layout gotcha pendulum_gpu_policy.rs:302 auto-detects.)
    let mut y_cpu = Vec::with_capacity(ROWS * COLS);
    let mut g_cpu = Vec::with_capacity(ROWS * COLS);
    for r in 0..ROWS {
        for c in 0..COLS {
            let y = elu(x[(r, c)]);
            y_cpu.push(y);
            g_cpu.push(g_in[(r, c)] * elu_grad_from_act(y));
        }
    }

    let fwd_err = y_gpu
        .iter()
        .zip(&y_cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let bwd_err = g_gpu
        .iter()
        .zip(&g_cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    println!("ELU check ({ROWS}x{COLS})");
    println!("  forward  max|gpu - cpu| = {fwd_err:.3e}");
    println!("  backward max|gpu - cpu| = {bwd_err:.3e}");
    anyhow::ensure!(fwd_err < 1e-5, "gpu_elu diverged from CPU elu");
    anyhow::ensure!(
        bwd_err < 1e-5,
        "gpu_elu_backward diverged from CPU elu_grad_from_act"
    );
    println!("OK — gpu_elu + gpu_elu_backward match the zealot-rl CPU reference.");

    // ---- vec4 ELU (4 f32/thread, 128-bit loads); buffer len must be % 4 ----
    let xv = DMatrix::<f32>::from_fn(8, 16, |r, c| {
        (r as f32 - 4.0) * 0.5 + (c as f32 - 8.0) * 0.2
    });
    let mut av = Tensor::matrix_from_na(&backend, &xv, rw)?;
    {
        let mut enc = backend.begin_encoding();
        {
            let mut p = enc.begin_pass("elu_vec4", None);
            act.elu_vec4(&backend, &mut shapes, &mut p, &mut av)?;
        }
        backend.submit(enc)?;
        backend.synchronize()?;
    }
    let yv_gpu = backend.slow_read_vec(av.buffer()).await?;
    let yv_cpu: Vec<f32> = {
        let mut v = Vec::with_capacity(8 * 16);
        for r in 0..8 {
            for c in 0..16 {
                v.push(elu(xv[(r, c)]));
            }
        }
        v
    };
    let v4_err = yv_gpu
        .iter()
        .zip(&yv_cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("  vec4 ELU max|gpu - cpu| = {v4_err:.3e}");
    anyhow::ensure!(v4_err < 1e-5, "gpu_elu_vec4 diverged from CPU elu");
    println!("OK — gpu_elu_vec4 (vec4, 128-bit) matches CPU too.");
    Ok(())
}
