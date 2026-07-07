//! Numerical parity check: the GPU PPO output-gradient kernels (vortx
//! `Ppo::actor_grad` / `value_grad`, the PPO-specific part of the GPU trainer's
//! update) vs an independent scalar-CPU reference of the SAME clipped-surrogate
//! / clipped-value formulas used by `zealot-rl`'s `minibatch_step`.
//!
//! This isolates the novel code (the per-sample output gradients) from the
//! generic GEMM / ELU backward backbone and Adam, which are vortx-tested. A
//! random minibatch is fed to both paths; inputs are deliberately spread so
//! both the clipped and unclipped surrogate branches AND both value-clip
//! branches are exercised. Passes when every element agrees to f32 tolerance.
//!
//! Run:
//!   cargo run --release --example ppo_grad_parity --features "gpu biped_gpu"

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::{Ppo, PpoActorParams, PpoValueParams};
use vortx::tensor::Tensor;
use zealot_rl::rng::Lcg;

const A: usize = 12; // action dim (NUM_JOINTS)
const M: usize = 64; // minibatch columns
const CLIP: f32 = 0.2;
const ENTROPY: f32 = 0.01;
const VALUE_COEF: f32 = 1.0;
const LOG_SQRT_2PI: f32 = 0.918_938_5;

fn mk(b: &GpuBackend, m: &DMatrix<f32>, u: BufferUsages) -> Tensor<f32> {
    Tensor::matrix_from_na(b, m, u).unwrap()
}

async fn backend() -> GpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        ..Default::default()
    };
    let mut w = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("webgpu");
    w.force_buffer_copy_src = true;
    GpuBackend::WebGpu(w)
}

fn main() {
    pollster::block_on(async {
        let bk = backend().await;
        let mut rng = Lcg::new(0xC0FFEE_u64);

        // ---- synthetic minibatch (row-major [A x M] for the matrices) ----
        let mut log_std = [0.0f32; A];
        for v in log_std.iter_mut() {
            *v = -1.0 + rng.unit(); // log_std in [-1, 0] → std 0.37 .. 1.0
        }
        let mut mean = vec![0.0f32; A * M]; // [k*M + m]
        let mut action = vec![0.0f32; A * M];
        for m in 0..M {
            for k in 0..A {
                let mu = rng.gauss() * 0.5;
                let std = log_std[k].exp();
                mean[k * M + m] = mu;
                action[k * M + m] = mu + std * rng.gauss();
            }
        }
        // logp under the current params; logp_old is a perturbation of it so the
        // importance ratio straddles 1 ± clip (hits both surrogate branches).
        let mut logp_old = vec![0.0f32; M];
        let mut adv = vec![0.0f32; M];
        for m in 0..M {
            let mut logp = 0.0f32;
            for k in 0..A {
                let std = log_std[k].exp();
                let d = (action[k * M + m] - mean[k * M + m]) / std;
                logp += -0.5 * d * d - log_std[k] - LOG_SQRT_2PI;
            }
            logp_old[m] = logp + (-0.6 + 1.2 * rng.unit()); // ratio in ~[0.55, 1.82]
            adv[m] = rng.gauss(); // both signs
        }
        // Value inputs: v_pred straddles value_old ± clip (hits both value branches).
        let mut value_old = vec![0.0f32; M];
        let mut v_pred = vec![0.0f32; M];
        let mut ret = vec![0.0f32; M];
        for m in 0..M {
            value_old[m] = rng.gauss();
            v_pred[m] = value_old[m] + (-0.5 + rng.unit());
            ret[m] = value_old[m] + rng.gauss() * 0.5;
        }
        let scale = 1.0 / M as f32;

        // ---- CPU reference (scalar Rust; same formula as minibatch_step) ----
        let mut cpu_gmean = vec![0.0f32; A * M];
        let mut cpu_glogstd = vec![0.0f32; A * M];
        let mut cpu_gv = vec![0.0f32; M];
        let mut n_clip = 0usize;
        let (mut n_vclip_hi, mut n_vclip_lo) = (0usize, 0usize);
        for m in 0..M {
            let mut logp = 0.0f32;
            for k in 0..A {
                let std = log_std[k].exp();
                let d = (action[k * M + m] - mean[k * M + m]) / std;
                logp += -0.5 * d * d - log_std[k] - LOG_SQRT_2PI;
            }
            let ratio = (logp - logp_old[m]).exp();
            let a = adv[m];
            let clipped = (a >= 0.0 && ratio > 1.0 + CLIP) || (a < 0.0 && ratio < 1.0 - CLIP);
            if clipped {
                n_clip += 1;
            }
            for k in 0..A {
                let inv_var = (-2.0 * log_std[k]).exp();
                if clipped {
                    cpu_gmean[k * M + m] = 0.0;
                    cpu_glogstd[k * M + m] = -ENTROPY * scale;
                } else {
                    let d = action[k * M + m] - mean[k * M + m];
                    cpu_gmean[k * M + m] = -(a * ratio * d * inv_var) * scale;
                    let dls = a * ratio * (d * d * inv_var - 1.0);
                    cpu_glogstd[k * M + m] = (-dls - ENTROPY) * scale;
                }
            }
            // clipped value loss
            let (v, vo, r) = (v_pred[m], value_old[m], ret[m]);
            let diff = v - vo;
            let clamped = diff.clamp(-CLIP, CLIP);
            let v_clipped = vo + clamped;
            let l_un = (v - r) * (v - r);
            let l_cl = (v_clipped - r) * (v_clipped - r);
            let dv = if l_cl > l_un {
                if diff > CLIP {
                    n_vclip_hi += 1;
                } else if diff < -CLIP {
                    n_vclip_lo += 1;
                }
                2.0 * (v_clipped - r)
            } else {
                2.0 * (v - r)
            };
            cpu_gv[m] = VALUE_COEF * dv * scale;
        }

        // ---- GPU path (vortx kernels) ----
        let st = BufferUsages::STORAGE;
        let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let mean_t = mk(&bk, &DMatrix::from_fn(A, M, |r, c| mean[r * M + c]), st);
        let action_t = mk(&bk, &DMatrix::from_fn(A, M, |r, c| action[r * M + c]), st);
        let logstd_t = mk(&bk, &DMatrix::from_fn(A, 1, |r, _| log_std[r]), st);
        let adv_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| adv[c]), st);
        let lpo_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| logp_old[c]), st);
        let vpred_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| v_pred[c]), st);
        let vold_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| value_old[c]), st);
        let ret_t = mk(&bk, &DMatrix::from_fn(1, M, |_, c| ret[c]), st);
        let mut gmean_t = mk(&bk, &DMatrix::<f32>::zeros(A, M), rw);
        let mut glogstd_t = mk(&bk, &DMatrix::<f32>::zeros(A, M), rw);
        let mut gv_t = mk(&bk, &DMatrix::<f32>::zeros(1, M), rw);

        let ppo = Ppo::from_backend(&bk).unwrap();
        let ap = Tensor::scalar(
            &bk,
            PpoActorParams {
                clip: CLIP,
                entropy_coef: ENTROPY,
                scale,
                log_sqrt_2pi: LOG_SQRT_2PI,
                action_dim: A as u32,
                num_cols: M as u32,
                pad0: 0,
                pad1: 0,
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();
        let vp = Tensor::scalar(
            &bk,
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
        )
        .unwrap();

        let mut enc = bk.begin_encoding();
        {
            let mut p = enc.begin_pass("ag", None);
            ppo.actor_grad(
                &mut p,
                &ap,
                &mean_t,
                &action_t,
                &logstd_t,
                &adv_t,
                &lpo_t,
                &mut gmean_t,
                &mut glogstd_t,
            )
            .unwrap();
        }
        {
            let mut p = enc.begin_pass("vg", None);
            ppo.value_grad(&mut p, &vp, &vpred_t, &vold_t, &ret_t, &mut gv_t)
                .unwrap();
        }
        bk.submit(enc).unwrap();
        bk.synchronize().unwrap();

        let gpu_gmean = bk.slow_read_vec(gmean_t.buffer()).await.unwrap();
        let gpu_glogstd = bk.slow_read_vec(glogstd_t.buffer()).await.unwrap();
        let gpu_gv = bk.slow_read_vec(gv_t.buffer()).await.unwrap();

        // ---- compare ----
        let maxdiff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let d_mean = maxdiff(&cpu_gmean, &gpu_gmean[..A * M]);
        let d_logstd = maxdiff(&cpu_glogstd, &gpu_glogstd[..A * M]);
        let d_v = maxdiff(&cpu_gv, &gpu_gv[..M]);

        println!("minibatch: A={A} M={M}");
        println!(
            "branch coverage: surrogate clipped={n_clip}/{M}, value-clip hi={n_vclip_hi} lo={n_vclip_lo}"
        );
        println!("max |Δ| g_mean   = {d_mean:.3e}");
        println!("max |Δ| g_logstd = {d_logstd:.3e}");
        println!("max |Δ| g_value  = {d_v:.3e}");

        let tol = 1e-4f32;
        let ok = d_mean < tol && d_logstd < tol && d_v < tol;
        // Sanity: branches actually exercised (else the test is vacuous).
        let covered = n_clip > 0 && n_clip < M && (n_vclip_hi + n_vclip_lo) > 0;
        if ok && covered {
            println!("\nPASS — GPU PPO grad kernels match the CPU reference (tol {tol:.0e}).");
        } else {
            println!(
                "\nFAIL — ok={ok} covered={covered} (tol {tol:.0e}). The GPU update diverges from CPU PPO."
            );
            std::process::exit(1);
        }
    });
}
