//! Microbench: env-major vs SoA (env-fastest) layout for the multibody
//! contact-constraint back-solve (the `finalize` kernel — 680 µs/call, 18% of
//! physics at 4096 envs). Same sparse LᵀDL math, same parallel work, only the
//! memory layout + thread→work mapping differ. See
//! `nexus src_rbd_shaders/dynamics/multibody/layout_bench.rs`.
//!
//! Run: BIPED_CUDA=1 cargo run --release --example layout_solve_bench \
//!        --features "gpu biped_gpu cuda_backend"

use khal::BufferUsages;
use khal::Shader as _;
use khal::backend::{Backend, Encoder as _, GpuBackend as KhalGpuBackend};
use nexus3d::rbd::dynamics::LayoutBenchKernels as BenchKernels;
use vortx::tensor::Tensor;

const N: u32 = 35; // G1 29-DOF + 6 floating base
const C: u32 = 24; // ~8 contact points × 3 constraints
const E: u32 = 4096;
const NO_PARENT: u32 = u32::MAX;

fn main() {
    pollster::block_on(async {
        let bk = {
            use khal::backend::Cuda;
            KhalGpuBackend::Cuda(Cuda::new(0).expect("cuda"))
        };
        let k = BenchKernels::from_backend(&bk).expect("kernels");

        // Depth-~8 chains (G1-ish tree): parent = i-1 within blocks of 8.
        let parents: Vec<u32> = (0..N)
            .map(|i| if i % 8 == 0 { NO_PARENT } else { i - 1 })
            .collect();

        // Synthetic factors: unit-ish D, small L entries on ancestor chains
        // (values don't affect timing; chains are what get walked).
        let mut rng = 0x12345678u64;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 0.1
        };
        let nn = (N * N) as usize;
        let mut mat_env = vec![0.0f32; nn * E as usize];
        for env in 0..E as usize {
            for i in 0..N as usize {
                mat_env[env * nn + i * N as usize + i] = 1.0 + next();
                let mut j = parents[i];
                while j != NO_PARENT {
                    mat_env[env * nn + i * N as usize + j as usize] = next();
                    j = parents[j as usize];
                }
            }
        }
        let vec_len = (E * C * N) as usize;
        let jac_env: Vec<f32> = (0..vec_len).map(|_| next()).collect();

        // SoA mirrors: mat[(i*n+j)*E+env], jac[(s*n+i)*E+env].
        let mut mat_soa = vec![0.0f32; nn * E as usize];
        for env in 0..E as usize {
            for ij in 0..nn {
                mat_soa[ij * E as usize + env] = mat_env[env * nn + ij];
            }
        }
        let mut jac_soa = vec![0.0f32; vec_len];
        for env in 0..E as usize {
            for s in 0..C as usize {
                for i in 0..N as usize {
                    jac_soa[(s * N as usize + i) * E as usize + env] =
                        jac_env[(env * C as usize + s) * N as usize + i];
                }
            }
        }

        let st = BufferUsages::STORAGE;
        let un = BufferUsages::UNIFORM;
        let t_mat_env = Tensor::vector(&bk, &mat_env, st).unwrap();
        let t_mat_soa = Tensor::vector(&bk, &mat_soa, st).unwrap();
        let t_jac_env = Tensor::vector(&bk, &jac_env, st).unwrap();
        let t_jac_soa = Tensor::vector(&bk, &jac_soa, st).unwrap();
        let mut t_cols_a = Tensor::vector(&bk, &vec![0.0f32; vec_len], st).unwrap();
        let mut t_cols_b = Tensor::vector(&bk, &vec![0.0f32; vec_len], st).unwrap();
        let mut t_out_a = Tensor::vector(&bk, &vec![0.0f32; (E * C) as usize], st).unwrap();
        let mut t_out_b = Tensor::vector(&bk, &vec![0.0f32; (E * C) as usize], st).unwrap();
        let t_parents = Tensor::vector(&bk, &parents, st).unwrap();
        let t_n = Tensor::scalar(&bk, N, un).unwrap();
        let t_e = Tensor::scalar(&bk, E, un).unwrap();
        let t_c = Tensor::scalar(&bk, C, un).unwrap();

        let reps = 50u32;
        let mut run = |which: u32| -> f64 {
            // warmup 3 + timed reps
            for phase in 0..2 {
                let iters = if phase == 0 { 3 } else { reps };
                if phase == 1 {
                    bk.synchronize().unwrap();
                }
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let mut enc = bk.begin_encoding();
                    let mut pass = enc.begin_pass("bench", None);
                    if which == 0 {
                        k.env_major
                            .call(
                                &mut pass,
                                [32, E, 1],
                                &t_mat_env,
                                &t_jac_env,
                                &mut t_cols_a,
                                &mut t_out_a,
                                &t_parents,
                                &t_n,
                                &t_c,
                            )
                            .unwrap();
                    } else {
                        k.soa
                            .call(
                                &mut pass,
                                [E * C, 1, 1],
                                &t_mat_soa,
                                &t_jac_soa,
                                &mut t_cols_b,
                                &mut t_out_b,
                                &t_parents,
                                &t_n,
                                &t_e,
                                &t_c,
                            )
                            .unwrap();
                    }
                    drop(pass);
                    bk.submit(enc).unwrap();
                }
                bk.synchronize().unwrap();
                if phase == 1 {
                    return t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
                }
            }
            unreachable!()
        };

        let us_a = run(0);
        let us_b = run(1);

        // Correctness cross-check: inv_lhs must match between layouts.
        let out_a: Vec<f32> = bk.slow_read_vec(t_out_a.buffer()).await.unwrap();
        let out_b: Vec<f32> = bk.slow_read_vec(t_out_b.buffer()).await.unwrap();
        let mut max_rel = 0.0f32;
        for env in 0..E as usize {
            for s in 0..C as usize {
                let a = out_a[env * C as usize + s];
                let b = out_b[s * E as usize + env];
                let rel = (a - b).abs() / a.abs().max(1e-6);
                max_rel = max_rel.max(rel);
            }
        }

        println!("finalize-equivalent back-solve, E={E} C={C} N={N} (depth~8 chains)");
        println!("  env-major (production mimic) : {us_a:9.1} us/call");
        println!("  SoA env-fastest (hypothesis) : {us_b:9.1} us/call   speedup {:.2}x", us_a / us_b);
        println!("  cross-check max rel err      : {max_rel:.2e}  (must be ~1e-6: identical math)");
    });
}
