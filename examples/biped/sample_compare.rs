//! GPU Gaussian action sampler vs CPU reference — the "pin & compare" check for
//! the GPU-resident rollout (Stage 2). Verifies that vortx's `gpu_sample_gaussian`
//! produces the same actions/log-probs as the host `cpu_sample` reference, using
//! the shared counter-based RNG.
//!
//! Run (native CUDA):
//!   export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
//!   export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
//!   BIPED_CUDA=1 cargo run --release --example sample_compare \
//!       --features "gpu biped_gpu cuda_backend"
//! Run (WebGPU): drop BIPED_CUDA and the cubin env vars.
//!
//! pin_mode 1 (mean) and 2 (uniform) are bit-exact (err 0.0); pin_mode 0
//! (Gaussian/Box-Muller) may differ in the last ULP between host std and device
//! libdevice transcendentals — reported separately.

use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use vortx::linalg::{cpu_sample, SampleParams, Sampler};
use vortx::tensor::Tensor;

const N: usize = 2048; // envs
const ADIM: usize = 12; // action dims

async fn make_backend() -> GpuBackend {
    GpuBackend::auto(wgpu::Features::default(), wgpu::Limits::default())
        .await
        .expect("init GPU backend")
}

async fn run_mode(
    backend: &GpuBackend,
    sampler: &Sampler,
    means: &[f32],
    log_std: &[f32],
    seed: u32,
    step: u32,
    pin_mode: u32,
) -> (f32, f32) {
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let uni = BufferUsages::UNIFORM;

    let params = SampleParams {
        num_envs: N as u32,
        action_dim: ADIM as u32,
        seed,
        step,
        pin_mode,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let params_t = Tensor::scalar(backend, params, uni).expect("params");
    let means_t = Tensor::vector(backend, means, st).expect("means");
    let logstd_t = Tensor::vector(backend, log_std, st).expect("log_std");
    let mut actions_t = Tensor::vector(backend, &vec![0f32; N * ADIM], rw).expect("actions");
    let mut logp_t = Tensor::vector(backend, &vec![0f32; N], rw).expect("logp");

    {
        let mut enc = backend.begin_encoding();
        {
            let mut p = enc.begin_pass("sample", None);
            sampler
                .sample(
                    backend,
                    &mut p,
                    N as u32,
                    &params_t,
                    &means_t,
                    &logstd_t,
                    &mut actions_t,
                    &mut logp_t,
                )
                .expect("sample dispatch");
        }
        backend.submit(enc).expect("submit");
        backend.synchronize().expect("sync");
    }

    let a_gpu = backend.slow_read_vec(actions_t.buffer()).await.expect("read actions");
    let lp_gpu = backend.slow_read_vec(logp_t.buffer()).await.expect("read logp");

    let mut a_cpu = vec![0f32; N * ADIM];
    let mut lp_cpu = vec![0f32; N];
    cpu_sample(&params, means, log_std, &mut a_cpu, &mut lp_cpu);

    let a_err = a_gpu
        .iter()
        .zip(&a_cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0f32, f32::max);
    let lp_err = lp_gpu
        .iter()
        .zip(&lp_cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0f32, f32::max);
    (a_err, lp_err)
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = make_backend().await;
    let sampler = Sampler::from_backend(&backend)?;

    // Deterministic but non-trivial means and (negative) log-stds.
    let means: Vec<f32> = (0..N * ADIM)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.05)
        .collect();
    let log_std: Vec<f32> = (0..ADIM).map(|d| -1.0 + 0.1 * d as f32).collect();

    let seed = 0x1234_5678;
    let step = 3;

    println!("\nGPU sampler vs CPU reference  (N={N} envs, action_dim={ADIM})");
    println!("  seed={seed:#x} step={step}\n");
    for (mode, name) in [(1u32, "MEAN   "), (2, "UNIFORM"), (0, "GAUSSIAN")] {
        let (a_err, lp_err) = run_mode(&backend, &sampler, &means, &log_std, seed, step, mode).await;
        // logp is the pure-RNG check (no exp/transcendental): logp_err == 0 proves
        // the counter-based noise `z` is bit-identical host<->device. The action
        // path additionally applies `std = exp(log_std)`, whose last ULP can differ
        // between host std and device libdevice — so a ~1 ULP (<1e-6) action error
        // with logp_err == 0 is the expected, correct outcome, NOT a logic bug.
        let one_ulp = 2e-6;
        let verdict = if a_err == 0.0 && lp_err == 0.0 {
            "bit-exact ✓".to_string()
        } else if lp_err == 0.0 && a_err < one_ulp {
            "RNG bit-exact; action ≈ (exp ULP) ✓".to_string()
        } else if mode == 0 && lp_err < 1e-4 && a_err < 1e-4 {
            "≈ (Box-Muller transcendental ULP)".to_string()
        } else {
            format!("MISMATCH ✗ (a={a_err:.2e} lp={lp_err:.2e})")
        };
        println!("  pin_mode={mode} {name}  action err {a_err:.3e}   logp err {lp_err:.3e}   {verdict}");
    }
    println!();
    Ok(())
}
