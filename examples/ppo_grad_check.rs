//! PPO loss-gradient kernels verified against the CPU `minibatch_step` math.
//!
//! Run: `cargo run --release --example ppo_grad_check --features gpu`
//!
//! Checks `gpu_ppo_actor_grad` (clipped-surrogate g_mean + log_std contribution)
//! and `gpu_ppo_value_grad` (clipped value-loss dv) against a scalar CPU
//! reference transcribed from zealot-rl/src/ppo.rs `minibatch_step`. Inputs are
//! randomized so `logp_old` spreads the importance ratio across BOTH the clipped
//! and unclipped branches.

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Ppo, PpoActorParams, PpoValueParams};
use vortx::tensor::Tensor;
use wgpu::{Features, Limits};

const A: usize = 12; // action dim
const M: usize = 64; // minibatch columns
const CLIP: f32 = 0.2;
const ENT: f32 = 0.005;
const VALUE_COEF: f32 = 0.5;

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

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let log_sqrt_2pi = (0.5_f64 * (2.0 * std::f64::consts::PI).ln()) as f32;
    let scale = 1.0 / M as f32;
    let mut rng = Lcg(424242);

    // Random per-sample data.
    let mut mean = vec![[0f32; A]; M];
    let mut action = vec![[0f32; A]; M];
    let mut log_std = [0f32; A];
    let mut adv = [0f32; M];
    let mut logp_old = [0f32; M];
    let (mut v_pred, mut value_old, mut ret) = ([0f32; M], [0f32; M], [0f32; M]);
    for k in 0..A {
        log_std[k] = rng.range(-1.0, 0.2);
    }
    for m in 0..M {
        for k in 0..A {
            mean[m][k] = rng.range(-0.5, 0.5);
            action[m][k] = mean[m][k] + rng.range(-0.6, 0.6);
        }
        // true logp at (action, mean); offset it so ratios span the clip range.
        let mut logp = 0.0f32;
        for k in 0..A {
            let std = log_std[k].exp();
            let d = (action[m][k] - mean[m][k]) / std;
            logp += -0.5 * d * d - log_std[k] - log_sqrt_2pi;
        }
        logp_old[m] = logp + rng.range(-0.4, 0.4);
        adv[m] = rng.range(-2.0, 2.0);
        v_pred[m] = rng.range(-1.0, 1.0);
        value_old[m] = v_pred[m] + rng.range(-0.5, 0.5);
        ret[m] = rng.range(-1.0, 1.0);
    }

    // ---- CPU reference (transcribed from minibatch_step) ----
    let mut g_mean_c = vec![[0f32; A]; M];
    let mut g_logstd_c = vec![[0f32; A]; M];
    let mut g_v_c = [0f32; M];
    for m in 0..M {
        let mut logp = 0.0f32;
        for k in 0..A {
            let std = log_std[k].exp();
            let d = (action[m][k] - mean[m][k]) / std;
            logp += -0.5 * d * d - log_std[k] - log_sqrt_2pi;
        }
        let ratio = (logp - logp_old[m]).exp();
        let a = adv[m];
        let clipped = (a >= 0.0 && ratio > 1.0 + CLIP) || (a < 0.0 && ratio < 1.0 - CLIP);
        for k in 0..A {
            let inv_var = (-2.0 * log_std[k]).exp();
            if !clipped {
                let d = action[m][k] - mean[m][k];
                g_mean_c[m][k] = -(a * ratio * d * inv_var) * scale;
                let dls = a * ratio * (d * d * inv_var - 1.0);
                g_logstd_c[m][k] = -dls * scale;
            }
            g_logstd_c[m][k] += -ENT * scale;
        }
        let v = v_pred[m];
        let v_clipped = value_old[m] + (v - value_old[m]).clamp(-CLIP, CLIP);
        let l_un = (v - ret[m]).powi(2);
        let l_cl = (v_clipped - ret[m]).powi(2);
        let dv = if l_cl > l_un {
            2.0 * (v_clipped - ret[m])
        } else {
            2.0 * (v - ret[m])
        };
        g_v_c[m] = VALUE_COEF * dv * scale;
    }

    // ---- GPU ----
    let backend = GpuBackend::WebGpu(WebGpu::new(Features::default(), Limits::default()).await?);
    let ppo = Ppo::from_backend(&backend)?;
    let st = BufferUsages::STORAGE;
    let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    let mean_t = mk(&backend, &DMatrix::from_fn(A, M, |r, c| mean[c][r]), st);
    let action_t = mk(&backend, &DMatrix::from_fn(A, M, |r, c| action[c][r]), st);
    let log_std_t = mk(&backend, &DMatrix::from_fn(A, 1, |r, _| log_std[r]), st);
    let adv_t = mk(&backend, &DMatrix::from_fn(1, M, |_, c| adv[c]), st);
    let logp_old_t = mk(&backend, &DMatrix::from_fn(1, M, |_, c| logp_old[c]), st);
    let mut g_mean_t = mk(&backend, &DMatrix::<f32>::zeros(A, M), rw);
    let mut g_logstd_t = mk(&backend, &DMatrix::<f32>::zeros(A, M), rw);

    let v_pred_t = mk(&backend, &DMatrix::from_fn(1, M, |_, c| v_pred[c]), st);
    let value_old_t = mk(&backend, &DMatrix::from_fn(1, M, |_, c| value_old[c]), st);
    let ret_t = mk(&backend, &DMatrix::from_fn(1, M, |_, c| ret[c]), st);
    let mut g_v_t = mk(&backend, &DMatrix::<f32>::zeros(1, M), rw);

    let ap = Tensor::scalar(
        &backend,
        PpoActorParams {
            clip: CLIP,
            entropy_coef: ENT,
            scale,
            log_sqrt_2pi,
            action_dim: A as u32,
            num_cols: M as u32,
            pad0: 0,
            pad1: 0,
        },
        BufferUsages::UNIFORM,
    )?;
    let vp = Tensor::scalar(
        &backend,
        PpoValueParams {
            clip: CLIP,
            value_coef: VALUE_COEF,
            scale,
            num_cols: M as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        },
        BufferUsages::UNIFORM,
    )?;

    let mut enc = backend.begin_encoding();
    {
        let mut p = enc.begin_pass("actor_grad", None);
        ppo.actor_grad(
            &mut p,
            &ap,
            &mean_t,
            &action_t,
            &log_std_t,
            &adv_t,
            &logp_old_t,
            &mut g_mean_t,
            &mut g_logstd_t,
        )?;
    }
    {
        let mut p = enc.begin_pass("value_grad", None);
        ppo.value_grad(&mut p, &vp, &v_pred_t, &value_old_t, &ret_t, &mut g_v_t)?;
    }
    backend.submit(enc)?;
    backend.synchronize()?;

    let g_mean_g = backend.slow_read_vec(g_mean_t.buffer()).await?; // [A x M] row-major
    let g_logstd_g = backend.slow_read_vec(g_logstd_t.buffer()).await?;
    let g_v_g = backend.slow_read_vec(g_v_t.buffer()).await?;

    let (mut e_mean, mut e_ls, mut e_v) = (0f32, 0f32, 0f32);
    let mut clipped_count = 0;
    for m in 0..M {
        // count clipped samples (where every g_mean is exactly 0 from the clip branch)
        if g_mean_g[m] == 0.0 && g_mean_g[A.saturating_sub(1) * M + m] == 0.0 {
            clipped_count += 1;
        }
        for k in 0..A {
            e_mean = e_mean.max((g_mean_g[k * M + m] - g_mean_c[m][k]).abs());
            e_ls = e_ls.max((g_logstd_g[k * M + m] - g_logstd_c[m][k]).abs());
        }
        e_v = e_v.max((g_v_g[m] - g_v_c[m]).abs());
    }
    println!("PPO grad check (A={A}, M={M}, ~{clipped_count} clipped samples)");
    println!("  g_mean   max|gpu-cpu| = {e_mean:.3e}");
    println!("  g_logstd max|gpu-cpu| = {e_ls:.3e}");
    println!("  g_value  max|gpu-cpu| = {e_v:.3e}");
    let worst = e_mean.max(e_ls).max(e_v);
    anyhow::ensure!(
        worst < 1e-5,
        "PPO gradient kernels diverged from CPU (worst {worst:.3e})"
    );
    println!("OK — PPO loss-gradient kernels match the CPU minibatch_step reference.");
    Ok(())
}
