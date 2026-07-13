//! vortx GEMM benchmark on the biped PPO-update shapes — the vortx side of the
//! cutile-rs comparison (see `tools/` notes / the cutile `zealot_shapes` bench).
//!
//! Times every distinct GEMM in one `biped_train_gpu` minibatch step
//! (mb = 12288 columns): actor [45,256,256,128,12] and critic
//! [51,512,256,128,1] forward, dgrad and wgrad, using the REAL (unpadded)
//! shapes and the same encode-once-submit-once pattern the trainer uses.
//!
//! Run: `BIPED_CUDA=1 cargo run --release --example gemm_shapes_bench \
//!       --features "gpu cuda_backend"`

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend as KhalGpuBackend};
use khal::re_exports::wgpu;
use nalgebra::DMatrix;
use std::time::Instant;
use vortx::linalg::Gemm;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;

// (label, M, N, K, occurrences per minibatch step)
const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("actor fwd l0", 256, 12288, 45, 1),
    ("fwd/dgrad 256x256", 256, 12288, 256, 2),
    ("fwd 128<-256", 128, 12288, 256, 2),
    ("fwd out 12<-128", 12, 12288, 128, 2),
    ("critic fwd l0", 512, 12288, 51, 1),
    ("critic fwd l1", 256, 12288, 512, 1),
    ("dgrad 128<-12", 128, 12288, 12, 2),
    ("dgrad 256<-128", 256, 12288, 128, 2),
    ("critic dgrad l1", 512, 12288, 256, 1),
    ("wgrad 12x128", 12, 128, 12288, 2),
    ("wgrad 128x256", 128, 256, 12288, 2),
    ("wgrad 256x256", 256, 256, 12288, 1),
    ("wgrad 256x45", 256, 45, 12288, 1),
    ("wgrad 256x512", 256, 512, 12288, 1),
    ("wgrad 512x51", 512, 51, 12288, 1),
];

fn main() {
    pollster::block_on(async {
        let limits = wgpu::Limits {
            max_buffer_size: 1_200_000_000,
            max_storage_buffer_binding_size: 1_200_000_000,
            ..Default::default()
        };
        let bk = KhalGpuBackend::auto(wgpu::Features::default(), limits)
            .await
            .expect("backend");
        let gemm = Gemm::from_backend(&bk).expect("gemm");
        let mut sh = TensorLayoutBuffers::new(&bk);
        let st = BufferUsages::STORAGE;
        let iters = 50usize;
        let mut step_total_s = 0.0f64;
        println!(
            "{:<18} {:>5}x{:>5}x{:>5}   ms/call  TFLOPS  x",
            "shape", "M", "N", "K"
        );
        for &(label, m, n, k, mult) in SHAPES {
            let a = Tensor::matrix_from_na(&bk, &DMatrix::<f32>::zeros(m, k), st).unwrap();
            let b = Tensor::matrix_from_na(&bk, &DMatrix::<f32>::zeros(k, n), st).unwrap();
            let mut c = Tensor::matrix_from_na(&bk, &DMatrix::<f32>::zeros(m, n), st).unwrap();
            // Warmup.
            let mut enc = bk.begin_encoding();
            for _ in 0..3 {
                let mut p = enc.begin_pass("g", None);
                gemm.dispatch(&bk, &mut sh, &mut p, &mut c, &a, &b).unwrap();
            }
            bk.submit(enc).unwrap();
            bk.synchronize().unwrap();
            // Timed: encode all iters into one submission (trainer pattern).
            let start = Instant::now();
            let mut enc = bk.begin_encoding();
            for _ in 0..iters {
                let mut p = enc.begin_pass("g", None);
                gemm.dispatch(&bk, &mut sh, &mut p, &mut c, &a, &b).unwrap();
            }
            bk.submit(enc).unwrap();
            bk.synchronize().unwrap();
            let s = start.elapsed().as_secs_f64() / iters as f64;
            let tflops = 2.0 * (m * n * k) as f64 / s / 1e12;
            step_total_s += s * mult as f64;
            println!(
                "{label:<18} {m:>5}x{n:>5}x{k:>5}  {:>8.3}  {:>6.2}  {mult}",
                s * 1e3,
                tflops
            );
        }
        println!(
            "\nGEMMs per minibatch step: {:.3} ms -> projected GEMM time per PPO update (x20): {:.3} s",
            step_total_s * 1e3,
            step_total_s * 20.0
        );
    });
}
