//! Stage-2 measurement: does a zero-host-touch rollout loop scale past N≈2000?
//!
//! Isolates the mechanism that caused the host-bound ceiling — the per-step
//! policy-forward → means-readback → (host sample) round-trip — and compares it
//! against the device-resident version (policy forward + on-GPU sampler, no
//! per-step sync). Same GPU work per step; the only difference is the sync /
//! readback pattern.
//!
//!   SYNC      (host-bound): per step  encode forward → submit → SYNC → readback
//!                           means → host-sample. One GPU↔host round-trip/step.
//!   RESIDENT  (zero touch): per step  encode forward + GPU sampler → submit
//!                           (no sync). One sync + one readback after all T steps.
//!
//! Reports env-steps/second = N*T / wall for both, swept over N. The thesis: SYNC
//! plateaus (host-bound), RESIDENT keeps scaling with N until the GPU saturates.
//!
//! Run (native CUDA):
//!   export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
//!   export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
//!   BIPED_CUDA=1 cargo run --release --example resident_rollout_bench \
//!       --features "gpu biped_gpu cuda_backend"

use khal::backend::{Backend, Encoder, GpuBackend};
use khal::{BufferUsages, Shader};
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::{Activation, Gemm, OpAssign, OpAssignVariant, SampleParams, Sampler};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;

const OBS: usize = 45;
const ACT: usize = 12;
const HIDDEN: [usize; 3] = [256, 256, 128];
const T_STEPS: usize = 32;
const SWEEP: [usize; 5] = [512, 1024, 2048, 4096, 8192];

fn matrix(b: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(b, m, u).unwrap()
}

/// A representative actor stack: [OBS, 256, 256, 128, ACT], GEMM→bias→ELU
/// (linear output), batched over `n` envs. Mirrors `gpu_policy::GpuNet::encode`.
struct Net {
    w: Vec<Tensor<f32>>,
    b: Vec<Tensor<f32>>,
    a: Vec<Tensor<f32>>,
    dims: Vec<usize>,
}
impl Net {
    fn new(backend: &GpuBackend, n: usize) -> Self {
        let dims = {
            let mut d = vec![OBS];
            d.extend_from_slice(&HIDDEN);
            d.push(ACT);
            d
        };
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        // Deterministic pseudo-random weights (a tiny LCG — value irrelevant to timing).
        let mut seed = 0x2545_f491u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        };
        let layers = dims.len() - 1;
        let mut w = Vec::new();
        let mut b = Vec::new();
        for l in 0..layers {
            let (out, inp) = (dims[l + 1], dims[l]);
            let scale = (2.0 / inp as f32).sqrt();
            w.push(matrix(backend, &DMatrix::from_fn(out, inp, |_, _| rnd() * scale), st));
            b.push(matrix(backend, &DMatrix::from_fn(out, n, |_, _| 0.0), st));
        }
        // Persistent activation buffers; a[0] is the (fixed placeholder) obs input.
        let a: Vec<_> = (0..=layers)
            .map(|l| matrix(backend, &DMatrix::from_fn(dims[l], n, |r, c| ((r + c) as f32).sin() * 0.1), rw))
            .collect();
        Self { w, b, a, dims }
    }

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
                op.launch(backend, shapes, &mut p, OpAssignVariant::Add, &mut *a_out, &self.b[l])?;
            }
            if l < layers - 1 {
                let mut p = enc.begin_pass("elu", None);
                act.elu(backend, shapes, &mut p, &mut *a_out)?;
            }
        }
        Ok(())
    }

    fn output(&self) -> &Tensor<f32> {
        self.a.last().unwrap()
    }
    fn _dims(&self) -> &[usize] {
        &self.dims
    }
}

async fn make_backend() -> GpuBackend {
    GpuBackend::auto(wgpu::Features::default(), wgpu::Limits::default())
        .await
        .expect("init GPU backend")
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let backend = make_backend().await;
    let gemm = Gemm::from_backend(&backend)?;
    let op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let sampler = Sampler::from_backend(&backend)?;
    let log_std: Vec<f32> = (0..ACT).map(|d| -1.0 + 0.05 * d as f32).collect();

    println!("\nResident-rollout vs host-bound rollout  (policy [{OBS},256,256,128,{ACT}] + sampler, T={T_STEPS})");
    println!(
        "{:>7} | {:>14} | {:>14} | {:>7}",
        "N", "SYNC k env/s", "RESIDENT k env/s", "speedup"
    );
    println!("{}", "-".repeat(54));

    for &n in &SWEEP {
        let mut shapes = TensorLayoutBuffers::new(&backend);
        let mut net = Net::new(&backend, n);
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let uni = BufferUsages::UNIFORM;
        let logstd_t = Tensor::vector(&backend, &log_std, st)?;
        let mut actions_t = Tensor::vector(&backend, &vec![0f32; n * ACT], rw)?;
        let mut logp_t = Tensor::vector(&backend, &vec![0f32; n], rw)?;

        // ---- warmup (JIT, buffer alloc) ----
        for _ in 0..3 {
            let mut enc = backend.begin_encoding();
            net.encode(&backend, &gemm, &op, &act, &mut shapes, &mut enc)?;
            backend.submit(enc)?;
        }
        backend.synchronize()?;

        // ---------- SYNC (host-bound): sync + readback means each step ----------
        let mut host_actions = vec![0f32; n * ACT];
        let t0 = Instant::now();
        for _ in 0..T_STEPS {
            let mut enc = backend.begin_encoding();
            net.encode(&backend, &gemm, &op, &act, &mut shapes, &mut enc)?;
            backend.submit(enc)?;
            backend.synchronize()?;
            let means = backend.slow_read_vec(net.output().buffer()).await?;
            // emulate the host Gaussian sample (the work the current bench does on CPU)
            for (i, m) in means.iter().enumerate() {
                let z = ((i as f32) * 0.0001).sin();
                host_actions[i % (n * ACT)] = m + 0.3 * z;
            }
        }
        let sync_ms = t0.elapsed().as_secs_f64() * 1e3;
        let sync_eps = (n * T_STEPS) as f64 / (sync_ms / 1e3) / 1e3;

        // ---------- RESIDENT (zero host touch): GPU sampler, sync once ----------
        let params = SampleParams {
            num_envs: n as u32,
            action_dim: ACT as u32,
            seed: 0x1234_5678,
            step: 0,
            pin_mode: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_t = Tensor::scalar(&backend, params, uni)?;
        let t0 = Instant::now();
        for _ in 0..T_STEPS {
            let mut enc = backend.begin_encoding();
            net.encode(&backend, &gemm, &op, &act, &mut shapes, &mut enc)?;
            {
                let mut p = enc.begin_pass("sample", None);
                sampler.sample(
                    &backend,
                    &mut p,
                    n as u32,
                    &params_t,
                    net.output(),
                    &logstd_t,
                    &mut actions_t,
                    &mut logp_t,
                )?;
            }
            backend.submit(enc)?; // NO per-step sync
        }
        backend.synchronize()?; // single sync
        let _ = backend.slow_read_vec(actions_t.buffer()).await?; // single readback
        let res_ms = t0.elapsed().as_secs_f64() * 1e3;
        let res_eps = (n * T_STEPS) as f64 / (res_ms / 1e3) / 1e3;

        println!(
            "{:>7} | {:>14.1} | {:>14.1} | {:>6.2}x",
            n,
            sync_eps,
            res_eps,
            res_eps / sync_eps
        );
    }
    println!();
    Ok(())
}
