//! Isolation repro for the CUDA Contiguous column-gather mismatch seen in the
//! PPO update (`f_obs.columns(off,nb)` -> contiguous). Builds a known matrix,
//! gathers a column range, and checks the result against the expected columns
//! on whichever backend is selected (BIPED_CUDA=1 -> native CUDA, else WebGPU).
//!
//! Run: BIPED_CUDA=1 cargo run --release --example cuda_gather_check --features "gpu cuda_backend"

use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use nalgebra::DMatrix;
use vortx::linalg::Contiguous;
use vortx::shapes::TensorLayoutBuffers;
use vortx::tensor::Tensor;

async fn make_backend() -> KhalGpuBackend {
    #[cfg(feature = "cuda_backend")]
    {
        if std::env::var("BIPED_CUDA").as_deref() == Ok("1") {
            use khal::backend::Cuda;
            eprintln!("backend = native CUDA");
            return KhalGpuBackend::Cuda(Cuda::new(0).expect("cuda"));
        }
    }
    eprintln!("backend = WebGPU");
    KhalGpuBackend::WebGpu(
        WebGpu::new(wgpu::Features::default(), wgpu::Limits::default())
            .await
            .expect("webgpu"),
    )
}

fn main() {
    pollster::block_on(async {
        let bk = make_backend().await;
        let (rows, total, off, nb) = (64usize, 8usize, 5u32, 3u32);
        // f[r,c] = r*100 + c   (column-major DMatrix)
        let f = DMatrix::from_fn(rows, total, |r, c| (r * 100 + c) as f32);
        let f_t = Tensor::matrix_from_na(&bk, &f, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .unwrap();
        let mut out = Tensor::matrix_from_na(
            &bk,
            &DMatrix::<f32>::zeros(rows, nb as usize),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let cont = Contiguous::from_backend(&bk).unwrap();
        let mut sh = TensorLayoutBuffers::new(&bk);
        let mut enc = bk.begin_encoding();
        {
            use khal::backend::Encoder;
            let mut p = enc.begin_pass("gather", None);
            cont.launch(&bk, &mut sh, &mut p, &mut out, f_t.columns(off, nb), None)
                .unwrap();
        }
        bk.submit(enc).unwrap();
        bk.synchronize().unwrap();

        let got = bk.slow_read_vec(out.buffer()).await.unwrap();
        // out is [rows x nb], row-major index r*nb + c
        let mut maxerr = 0f32;
        println!("col_off={off} nb={nb}  (expect out[r,c] = r*100 + (off+c))");
        for r in (0..rows).step_by(21) {
            let mut line = String::new();
            for c in 0..nb as usize {
                let g = got[r * nb as usize + c];
                let exp = (r * 100) as f32 + (off as usize + c) as f32;
                maxerr = maxerr.max((g - exp).abs());
                line.push_str(&format!("{g:7.0}(exp {exp:.0})  "));
            }
            println!("  r{r}: {line}");
        }
        println!("maxerr = {maxerr}");
    });
}
