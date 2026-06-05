//! Verify `dispatch_tiled_vec4` (128-bit vec4 global loads) matches the scalar
//! `dispatch_tiled` for a tile-aligned, contiguous GEMM.
//! Run: `cargo run --release --example gemm_vec4_check --features gpu`

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::Gemm;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const M: usize = 128; // %64
const K: usize = 256; // %16
const N: usize = 128; // %64

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let bk = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let g = Gemm::from_backend(&bk)?;
    let mut sh = TensorLayoutBuffers::new(&bk);
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    let a = DMatrix::<f32>::from_fn(M, K, |r, c| ((r * K + c) % 13) as f32 * 0.1 - 0.6);
    let b = DMatrix::<f32>::from_fn(K, N, |r, c| ((r * N + c) % 7) as f32 * 0.1 - 0.3);
    let at = Tensor::matrix_from_na(&bk, &a, st)?;
    let bt = Tensor::matrix_from_na(&bk, &b, st)?;
    let mut c_scalar = Tensor::matrix_from_na(&bk, &DMatrix::<f32>::zeros(M, N), rw)?;
    let mut c_vec4 = Tensor::matrix_from_na(&bk, &DMatrix::<f32>::zeros(M, N), rw)?;

    let mut enc = bk.begin_encoding();
    {
        let mut p = enc.begin_pass("scalar", None);
        g.dispatch_tiled(&bk, &mut sh, &mut p, &mut c_scalar, &at, &bt)?;
    }
    {
        let mut p = enc.begin_pass("vec4", None);
        g.dispatch_tiled_vec4(&bk, &mut sh, &mut p, &mut c_vec4, &at, &bt)?;
    }
    bk.submit(enc)?;
    bk.synchronize()?;

    let cs = bk.slow_read_vec(c_scalar.buffer()).await?;
    let cv = bk.slow_read_vec(c_vec4.buffer()).await?;
    let err = cs
        .iter()
        .zip(&cv)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("GEMM {M}x{K}x{N}  tiled vs tiled_vec4  max err = {err:.3e}");
    anyhow::ensure!(err < 1e-4, "vec4 GEMM diverged from scalar tiled");
    println!("OK — gemm_tiled_vec4 (vec4 global loads) matches gemm_tiled.");
    Ok(())
}
