//! Fused GPU action-sampler (`gpu_sample_targets`) verified against the CPU math.
//!
//! Run (WebGPU):     `cargo run --release --example sample_check --features gpu`
//! Run (native CUDA): `BIPED_CUDA=1 cargo run --release --example sample_check --features "gpu cuda_backend"`
//!
//! The kernel draws per-(env,joint) Gaussian noise from a counter RNG, then
//! computes `action = mean + exp(log_std)·noise` and
//! `target = default_pos + action_scale·action`, all GPU-resident. Three checks:
//!   1. Statistical — the noise readback is ~N(0,1) (mean≈0, std≈1).
//!   2. Arithmetic  — reconstruct action/target on the CPU from the kernel's own
//!      noise readback; must match bit-closely (transcendentals aside).
//!   3. Determinism — same seed reproduces the exact same noise.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend};
use nalgebra::DMatrix;
use vortx::linalg::{Sample, SampleParams};
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const A: usize = 12; // action dim (NUM_JOINTS)
const N: usize = 4096; // envs
const SEED: u32 = 0x1234_5678;

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

fn mk(backend: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(backend, m, u).unwrap()
}

async fn make_backend() -> GpuBackend {
    GpuBackend::auto(Features::default(), Limits::default())
        .await
        .expect("init GPU backend")
}

/// Run the kernel once for `seed`; returns (action, target, noise) row-major [A x N].
async fn run(
    backend: &GpuBackend,
    sampler: &Sample,
    mean: &DMatrix<f32>,
    log_std: &[f32; A],
    default_pos: &[f32; A],
    action_scale: &[f32; A],
    seed: u32,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let mean_t = mk(backend, mean, st);
    let log_std_t = mk(backend, &DMatrix::from_fn(A, 1, |r, _| log_std[r]), st);
    let default_t = mk(backend, &DMatrix::from_fn(A, 1, |r, _| default_pos[r]), st);
    let scale_t = mk(backend, &DMatrix::from_fn(A, 1, |r, _| action_scale[r]), st);
    let mut action_t = mk(backend, &DMatrix::<f32>::zeros(A, N), rw);
    let mut target_t = mk(backend, &DMatrix::<f32>::zeros(A, N), rw);
    let mut noise_t = mk(backend, &DMatrix::<f32>::zeros(A, N), rw);

    let params = Tensor::scalar(
        backend,
        SampleParams {
            action_dim: A as u32,
            num_envs: N as u32,
            seed,
            pad0: 0,
        },
        BufferUsages::UNIFORM,
    )?;

    let mut enc = backend.begin_encoding();
    {
        let mut p = enc.begin_pass("sample", None);
        sampler.sample_targets(
            &mut p,
            &params,
            &mean_t,
            &log_std_t,
            &default_t,
            &scale_t,
            &mut action_t,
            &mut target_t,
            &mut noise_t,
        )?;
    }
    backend.submit(enc)?;
    backend.synchronize()?;

    Ok((
        backend.slow_read_vec(action_t.buffer()).await?,
        backend.slow_read_vec(target_t.buffer()).await?,
        backend.slow_read_vec(noise_t.buffer()).await?,
    ))
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mut rng = Lcg(0xC0FFEE);
    let mut log_std = [0f32; A];
    let mut default_pos = [0f32; A];
    let mut action_scale = [0f32; A];
    for k in 0..A {
        log_std[k] = rng.range(-1.5, 0.3);
        default_pos[k] = rng.range(-0.5, 0.5);
        action_scale[k] = rng.range(0.2, 0.6);
    }
    let mean = DMatrix::from_fn(A, N, |_, _| rng.range(-1.0, 1.0));

    let backend = make_backend().await;
    let sampler = Sample::from_backend(&backend)?;

    let (action_g, target_g, noise_g) =
        run(&backend, &sampler, &mean, &log_std, &default_pos, &action_scale, SEED).await?;

    // ---- 1. statistical: noise ~ N(0,1) ----
    let total = (A * N) as f64;
    let sum: f64 = noise_g.iter().map(|&z| z as f64).sum();
    let mu = sum / total;
    let var: f64 = noise_g.iter().map(|&z| (z as f64 - mu).powi(2)).sum::<f64>() / total;
    let sd = var.sqrt();
    println!("noise stats over {} draws: mean={mu:+.4} std={sd:.4}", A * N);
    anyhow::ensure!(mu.abs() < 0.02, "noise mean off zero: {mu}");
    anyhow::ensure!((sd - 1.0).abs() < 0.03, "noise std off one: {sd}");

    // ---- 2. arithmetic: reconstruct from the kernel's own noise ----
    let (mut e_act, mut e_tgt) = (0f32, 0f32);
    for e in 0..N {
        for k in 0..A {
            let idx = k * N + e;
            let act = mean[(k, e)] + log_std[k].exp() * noise_g[idx];
            let tgt = default_pos[k] + action_scale[k] * act;
            e_act = e_act.max((action_g[idx] - act).abs());
            e_tgt = e_tgt.max((target_g[idx] - tgt).abs());
        }
    }
    println!("arithmetic max|gpu-cpu|: action={e_act:.3e}  target={e_tgt:.3e}");
    anyhow::ensure!(e_act < 1e-5 && e_tgt < 1e-5, "fused arithmetic diverged");

    // ---- 3. determinism: same seed -> identical noise ----
    let (_, _, noise_g2) =
        run(&backend, &sampler, &mean, &log_std, &default_pos, &action_scale, SEED).await?;
    let max_d = noise_g
        .iter()
        .zip(&noise_g2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("determinism max|run1-run2| = {max_d:.3e}");
    anyhow::ensure!(max_d == 0.0, "RNG not deterministic across runs");

    // Different seed must move the noise.
    let (_, _, noise_g3) = run(
        &backend, &sampler, &mean, &log_std, &default_pos, &action_scale, SEED.wrapping_add(1),
    )
    .await?;
    let moved = noise_g.iter().zip(&noise_g3).filter(|(a, b)| a != b).count();
    println!("fresh-seed changed {moved}/{} draws", A * N);
    anyhow::ensure!(moved > A * N / 2, "fresh seed barely changed the noise");

    println!("OK — gpu_sample_targets matches the CPU sample+target math and is a good N(0,1).");
    Ok(())
}
