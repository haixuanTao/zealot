//! Multi-layer, batched, ELU GPU backward — Stage-B backbone for the PPO update.
//!
//! Run: `cargo run --release --example mlp_backward_batch --features "gpu biped_gpu"`
//!
//! Generalizes `mlp_backward.rs` (1 hidden layer, tanh, batch=1) to the real
//! actor shape [43,256,256,128,12], ELU hidden, batch N, given an arbitrary
//! upstream output gradient `g_out [out x N]`. Computes every `dW_l` / `db_l` on
//! the GPU and checks them against the CPU `zealot_rl::net::Mlp::backward`
//! summed over the batch. Establishes the gradient path Stage B's PPO loss
//! kernels feed into.
//!
//! Per layer l (forward cache a_l), with delta_l = dL/dz_l:
//!   dW_l = delta_l · a_{l-1}ᵀ        (GEMM; sums over the batch automatically)
//!   db_l = delta_l · 1_N             (GEMM with a ones vector = row-sum)
//!   da_{l-1} = W_lᵀ · delta_l        (GEMM)
//!   delta_{l-1} = da_{l-1} ⊙ elu'(a_{l-1})   (elu_backward, hidden only)

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Activation, Gemm, OpAssign};
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};
use zealot_rl::net::{Mlp, MlpGrad};
use zealot_rl::rng::Lcg;

const DIMS: [usize; 5] = [43, 256, 256, 128, 12];
const N: usize = 256; // batch (smaller than 4096 — this is a correctness check)

fn mk(backend: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(backend, m, u).unwrap()
}

/// max abs error between a row-major GPU readback `[r x c]` and a CPU matrix.
fn err(gpu: &[f32], cpu: &DMatrix<f32>) -> f32 {
    let (nr, nc) = (cpu.nrows(), cpu.ncols());
    let mut e = 0f32;
    for r in 0..nr {
        for c in 0..nc {
            e = e.max((gpu[r * nc + c] - cpu[(r, c)]).abs());
        }
    }
    e
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let mut rng = Lcg::new(3);
    let net = Mlp::new(&DIMS, 0.01, &mut rng);
    let layers = net.w.len();

    // Random batch input [in x N] and an arbitrary output gradient [out x N].
    let x: Vec<[f32; DIMS[0]]> = (0..N)
        .map(|_| std::array::from_fn(|_| rng.gauss()))
        .collect();
    let gout: Vec<[f32; DIMS[4]]> = (0..N)
        .map(|_| std::array::from_fn(|_| rng.gauss() * 0.1))
        .collect();

    // ---- CPU reference: accumulate Mlp::backward over the batch ----
    let mut g_cpu = MlpGrad::zero(&net);
    for e in 0..N {
        let act = net.forward(&x[e]);
        net.backward(&act, &gout[e], &mut g_cpu);
    }

    // ---- GPU ----
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let gemm = Gemm::from_backend(&backend)?;
    let _op = OpAssign::from_backend(&backend)?;
    let act = Activation::from_backend(&backend)?;
    let mut sh = TensorLayoutBuffers::new(&backend);
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    // Weights / biases (bias broadcast to [out x N] for the forward add).
    let w: Vec<Tensor<f32>> = (0..layers)
        .map(|l| {
            let (out, inp) = (DIMS[l + 1], DIMS[l]);
            mk(
                &backend,
                &DMatrix::from_fn(out, inp, |r, c| net.w[l][r * inp + c]),
                st,
            )
        })
        .collect();
    let bfull: Vec<Tensor<f32>> = (0..layers)
        .map(|l| {
            mk(
                &backend,
                &DMatrix::from_fn(DIMS[l + 1], N, |r, _| net.b[l][r]),
                st,
            )
        })
        .collect();
    let ones = mk(&backend, &DMatrix::<f32>::from_element(N, 1, 1.0), st); // [N x 1]

    // Activation buffers a[0..=layers], a[0] = input.
    let xm = DMatrix::from_fn(DIMS[0], N, |r, c| x[c][r]);
    let mut a: Vec<Tensor<f32>> = vec![mk(&backend, &xm, rw)];
    for l in 1..=layers {
        a.push(mk(&backend, &DMatrix::<f32>::zeros(DIMS[l], N), rw));
    }
    // delta[l] = dL/dz_l buffers (same shape as a[l+1]); reuse separate buffers.
    let mut delta: Vec<Tensor<f32>> = (0..layers)
        .map(|l| mk(&backend, &DMatrix::<f32>::zeros(DIMS[l + 1], N), rw))
        .collect();
    let mut dw: Vec<Tensor<f32>> = (0..layers)
        .map(|l| mk(&backend, &DMatrix::<f32>::zeros(DIMS[l + 1], DIMS[l]), rw))
        .collect();
    let mut db: Vec<Tensor<f32>> = (0..layers)
        .map(|l| mk(&backend, &DMatrix::<f32>::zeros(DIMS[l + 1], 1), rw))
        .collect();
    let goutm = DMatrix::from_fn(DIMS[4], N, |r, c| gout[c][r]);
    let g_out_t = mk(&backend, &goutm, st);

    let mut enc = backend.begin_encoding();
    // ---- forward (cache activations) ----
    for l in 0..layers {
        let (left, right) = a.split_at_mut(l + 1);
        let (ain, aout) = (&left[l], &mut right[0]);
        {
            let mut p = enc.begin_pass("gemm", None);
            gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut *aout, &w[l], ain)?;
        }
        {
            let mut p = enc.begin_pass("bias", None);
            _op.launch(
                &backend,
                &mut sh,
                &mut p,
                vortx::linalg::OpAssignVariant::Add,
                &mut *aout,
                &bfull[l],
            )?;
        }
        if l < layers - 1 {
            let mut p = enc.begin_pass("elu", None);
            act.elu(&backend, &mut sh, &mut p, &mut *aout)?;
        }
    }
    // ---- backward ----
    // Seed top delta = g_out (output layer linear, so delta_top = g_out).
    {
        let mut p = enc.begin_pass("seed", None);
        _op.launch(
            &backend,
            &mut sh,
            &mut p,
            vortx::linalg::OpAssignVariant::Add,
            &mut delta[layers - 1],
            &g_out_t,
        )?;
    }
    for l in (0..layers).rev() {
        // dW_l = delta_l · a_lᵀ   (a[l] is the layer input)
        {
            let mut p = enc.begin_pass("dW", None);
            gemm.dispatch_naive(
                &backend,
                &mut sh,
                &mut p,
                &mut dw[l],
                &delta[l],
                a[l].transpose_last_dims(),
            )?;
        }
        // db_l = delta_l · ones_N
        {
            let mut p = enc.begin_pass("db", None);
            gemm.dispatch_naive(&backend, &mut sh, &mut p, &mut db[l], &delta[l], &ones)?;
        }
        if l > 0 {
            // da_{l-1} = W_lᵀ · delta_l  -> into delta[l-1]
            {
                let (left, right) = delta.split_at_mut(l);
                let dprev = &mut left[l - 1];
                let dcur = &right[0];
                let mut p = enc.begin_pass("da", None);
                gemm.dispatch_naive(
                    &backend,
                    &mut sh,
                    &mut p,
                    dprev,
                    w[l].transpose_last_dims(),
                    dcur,
                )?;
            }
            // delta_{l-1} = da_{l-1} ⊙ elu'(a_{l-1})   (a[l] is the hidden output)
            {
                let mut p = enc.begin_pass("elu_bwd", None);
                act.elu_backward(&backend, &mut sh, &mut p, &mut delta[l - 1], &a[l])?;
            }
        }
    }
    backend.submit(enc)?;
    backend.synchronize()?;

    // ---- compare ----
    let mut worst = 0f32;
    for l in 0..layers {
        let (out, inp) = (DIMS[l + 1], DIMS[l]);
        let dwg = backend.slow_read_vec(dw[l].buffer()).await?;
        let dbg = backend.slow_read_vec(db[l].buffer()).await?;
        let dwc = DMatrix::from_fn(out, inp, |r, c| g_cpu.w[l][r * inp + c]);
        let dbc = DMatrix::from_fn(out, 1, |r, _| g_cpu.b[l][r]);
        let ew = err(&dwg, &dwc);
        let eb = err(&dbg, &dbc);
        println!("  layer {l}: dW err {ew:.3e}   db err {eb:.3e}");
        worst = worst.max(ew).max(eb);
    }
    println!("worst gradient error (gpu vs cpu Mlp::backward) = {worst:.3e}");
    anyhow::ensure!(worst < 2e-3, "GPU multi-layer backward diverged from CPU");
    println!("OK — batched multi-layer ELU GPU backward matches CPU. Stage-B backbone verified.");
    Ok(())
}
