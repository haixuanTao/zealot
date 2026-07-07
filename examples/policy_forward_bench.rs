//! Benchmark: CPU vs GPU **batched policy forward** at biped scale.
//!
//! Run: `cargo run --release --example policy_forward_bench --features gpu`
//!
//! This is the Stage-A spike for the vortx GPU-policy port. The biped trainer's
//! rollout does `for e in 0..N { actor.forward(); critic.forward() }` — N serial
//! CPU MLP passes per timestep (the rollout bottleneck). Here we time that CPU
//! loop against a single batched GPU forward (one GEMM-stack over `[obs x N]`)
//! using vortx's GEMM + ELU (the kernels just added) + bias-add, and verify the
//! GPU output matches the CPU reference before reporting the speedup.
//!
//! Net shapes match the deployed biped (velocity_flat): actor [43,256,256,128,12],
//! critic [49,512,256,128,1], ELU hidden / linear output.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::{Activation, Gemm, OpAssign, OpAssignVariant};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};
use zealot_rl::net::Mlp;
use zealot_rl::rng::Lcg;

const N: usize = 4096; // envs (batch columns) — the deployed training batch
const STEPS: usize = 50; // timed forward passes (rollout does T=32/iter)

const ACTOR: [usize; 5] = [43, 256, 256, 128, 12];
const CRITIC: [usize; 5] = [49, 512, 256, 128, 1];

/// One net's GPU-resident parameters: per-layer weight `[out x in]` and bias
/// pre-broadcast to `[out x N]` (so the bias-add is a proven same-shape op).
struct GpuNet {
    w: Vec<Tensor<f32>>,
    b: Vec<Tensor<f32>>,
    /// Activation buffers, `a[0]` = input `[in x N]`, `a[l]` = layer-l output.
    a: Vec<Tensor<f32>>,
    dims: Vec<usize>,
}

impl GpuNet {
    fn new(backend: &GpuBackend, net: &Mlp) -> anyhow::Result<Self> {
        let dims = net.dims.clone();
        let layers = net.w.len();
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let mut w = Vec::with_capacity(layers);
        let mut b = Vec::with_capacity(layers);
        for l in 0..layers {
            let (out, inp) = (dims[l + 1], dims[l]);
            // net.w[l] is row-major [out x in]; mirror mlp_forward's from_fn recipe.
            let wm = DMatrix::from_fn(out, inp, |r, c| net.w[l][r * inp + c]);
            w.push(Tensor::matrix_from_na(backend, &wm, st)?);
            // bias broadcast across all N columns.
            let bm = DMatrix::from_fn(out, N, |r, _| net.b[l][r]);
            b.push(Tensor::matrix_from_na(backend, &bm, st)?);
        }
        // a[0] input, a[1..=layers] outputs; all need COPY_SRC for final readback.
        let mut a = Vec::with_capacity(layers + 1);
        for l in 0..=layers {
            a.push(Tensor::matrix_from_na(
                backend,
                &DMatrix::<f32>::zeros(dims[l], N),
                rw,
            )?);
        }
        Ok(Self { w, b, a, dims })
    }

    /// Upload the input matrix `[in x N]` into `a[0]`.
    fn set_input(&mut self, backend: &GpuBackend, x: &DMatrix<f32>) -> anyhow::Result<()> {
        self.a[0] =
            Tensor::matrix_from_na(backend, x, BufferUsages::STORAGE | BufferUsages::COPY_SRC)?;
        Ok(())
    }

    /// Encode the full forward (GEMM -> bias -> ELU per hidden layer, linear out).
    fn encode(
        &mut self,
        backend: &GpuBackend,
        gemm: &Gemm,
        op: &OpAssign,
        act: &Activation,
        shapes: &mut TensorLayoutBuffers,
        enc: &mut <GpuBackend as Backend>::Encoder,
    ) -> anyhow::Result<()> {
        let layers = self.w.len();
        for l in 0..layers {
            let (left, right) = self.a.split_at_mut(l + 1);
            let a_in = &left[l];
            let a_out = &mut right[0];
            {
                let mut p = enc.begin_pass("gemm", None);
                gemm.dispatch_tiled(backend, shapes, &mut p, &mut *a_out, &self.w[l], a_in)?;
            }
            {
                let mut p = enc.begin_pass("bias", None);
                op.launch(
                    backend,
                    shapes,
                    &mut p,
                    OpAssignVariant::Add,
                    &mut *a_out,
                    &self.b[l],
                )?;
            }
            if l < layers - 1 {
                let mut p = enc.begin_pass("elu", None);
                act.elu(backend, shapes, &mut p, &mut *a_out)?;
            }
        }
        let _ = self.dims;
        Ok(())
    }

    fn output(&self) -> &Tensor<f32> {
        self.a.last().unwrap()
    }
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mut rng = Lcg::new(1);
    let actor = Mlp::new(&ACTOR, 0.01, &mut rng);
    let critic = Mlp::new(&CRITIC, 1.0, &mut rng);

    // Random per-env inputs (already "normalized" — we compare like-for-like).
    let mut obs = vec![[0f32; ACTOR[0]]; N];
    let mut cobs = vec![[0f32; CRITIC[0]]; N];
    for e in 0..N {
        for v in obs[e].iter_mut() {
            *v = rng.gauss();
        }
        for v in cobs[e].iter_mut() {
            *v = rng.gauss();
        }
    }

    // ---------------- CPU: the serial per-env loop the rollout does ----------------
    let mut sink = 0f32;
    // warmup
    for e in 0..N {
        sink += actor.forward(&obs[e]).output()[0];
        sink += critic.forward(&cobs[e]).output()[0];
    }
    let t0 = Instant::now();
    for _ in 0..STEPS {
        for e in 0..N {
            sink += actor.forward(&obs[e]).output()[0];
            sink += critic.forward(&cobs[e]).output()[0];
        }
    }
    let cpu = t0.elapsed();

    // ---------------- GPU: one batched forward per net per step --------------------
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let gemm = Gemm::from_backend(&backend)?;
    let op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let mut shapes = TensorLayoutBuffers::new(&backend);

    let mut g_actor = GpuNet::new(&backend, &actor)?;
    let mut g_critic = GpuNet::new(&backend, &critic)?;

    // Input matrices [in x N] (column e = env e).
    let obs_m = DMatrix::from_fn(ACTOR[0], N, |r, c| obs[c][r]);
    let cobs_m = DMatrix::from_fn(CRITIC[0], N, |r, c| cobs[c][r]);
    g_actor.set_input(&backend, &obs_m)?;
    g_critic.set_input(&backend, &cobs_m)?;

    let run_once = |backend: &GpuBackend,
                    shapes: &mut TensorLayoutBuffers,
                    ga: &mut GpuNet,
                    gc: &mut GpuNet|
     -> anyhow::Result<()> {
        let mut enc = backend.begin_encoding();
        ga.encode(backend, &gemm, &op, &act, shapes, &mut enc)?;
        gc.encode(backend, &gemm, &op, &act, shapes, &mut enc)?;
        backend.submit(enc)?;
        backend.synchronize()?;
        Ok(())
    };

    // warmup + correctness check
    run_once(&backend, &mut shapes, &mut g_actor, &mut g_critic)?;
    let a_gpu = backend.slow_read_vec(g_actor.output().buffer()).await?; // [12 x N] row-major
    let c_gpu = backend.slow_read_vec(g_critic.output().buffer()).await?; // [1 x N]
    // CPU reference for a few envs; GPU buffer is row-major [out x N] -> idx r*N + c.
    let mut max_err = 0f32;
    for &e in &[0usize, 1, 7, 100, N - 1] {
        let am = actor.forward(&obs[e]);
        let ao = am.output();
        for r in 0..ACTOR[4] {
            max_err = max_err.max((a_gpu[r * N + e] - ao[r]).abs());
        }
        let cv = critic.forward(&cobs[e]).output()[0];
        max_err = max_err.max((c_gpu[e] - cv).abs());
    }

    let t1 = Instant::now();
    for _ in 0..STEPS {
        run_once(&backend, &mut shapes, &mut g_actor, &mut g_critic)?;
    }
    let gpu = t1.elapsed();

    // R1 measurement: same forward, but reading BOTH outputs back to CPU each
    // step (what the rollout actually does). delta vs `gpu` = the per-step
    // readback cost we'd remove by going GPU-resident.
    let t2 = Instant::now();
    for _ in 0..STEPS {
        run_once(&backend, &mut shapes, &mut g_actor, &mut g_critic)?;
        let _a = backend.slow_read_vec(g_actor.output().buffer()).await?;
        let _c = backend.slow_read_vec(g_critic.output().buffer()).await?;
        sink += _a[0] + _c[0];
    }
    let gpu_rb = t2.elapsed();

    let cpu_per = cpu.as_secs_f64() / STEPS as f64 * 1e3;
    let gpu_per = gpu.as_secs_f64() / STEPS as f64 * 1e3;
    let gpu_rb_per = gpu_rb.as_secs_f64() / STEPS as f64 * 1e3;
    println!("policy forward bench — N={N} envs, {STEPS} steps");
    println!("  actor {ACTOR:?}  critic {CRITIC:?}  (ELU hidden, linear out)");
    println!("  GPU vs CPU max|out| err = {max_err:.3e}");
    anyhow::ensure!(max_err < 2e-3, "GPU forward diverged from CPU reference");
    println!(
        "  CPU serial loop : {cpu_per:8.3} ms/step  ({:.2} us/env)",
        cpu_per * 1e3 / N as f64
    );
    println!("  GPU batched     : {gpu_per:8.3} ms/step  (no readback)");
    println!(
        "  GPU + readback  : {gpu_rb_per:8.3} ms/step  (R1 = {:.3} ms/step)",
        gpu_rb_per - gpu_per
    );
    println!("  speedup         : {:.1}x", cpu_per / gpu_per);
    println!("  (sink={sink:.3})");
    Ok(())
}
