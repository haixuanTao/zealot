//! Standalone parity probe for the two NEW device kernels the GPU-obs
//! consumption switch builds on: `gpu_obs_stack` (history stacking) and
//! `gpu_welford` (normalizer statistics). Each is driven with random inputs
//! and compared element-for-element against the host reference
//! (`ObsHistory` / `PendingNorm::push`).
//!
//!   cargo run --release --bin obs_kernels_probe --features "gpu biped_gpu"

use khal::BufferUsages;
use khal::backend::Backend;
use vortx::tensor::Tensor;
use zealot_env::obs_history::ObsHistory;
use zealot_env::rng::Lcg;
use zealot_gpu_obs::{GpuObsStack, GpuWelford};
use zealot_rl::ppo::PendingNorm;

#[path = "../biped/biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "../biped/biped_env.rs"]
mod biped_env;
#[path = "../biped/cutile_gemm.rs"]
mod cutile_gemm;
#[path = "../biped/gpu_policy.rs"]
mod gpu_policy;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let bk = biped_env_nexus::make_backend().await;
    let (n, f, h, steps) = (257usize, 53usize, 5usize, 7usize);
    let s = f * h;
    let mut rng = Lcg::new(0xC0FFEE);
    let st = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    // ---------------- gpu_obs_stack vs ObsHistory ----------------
    let mut hist = ObsHistory::new(n, h, f);
    let mut stack = GpuObsStack::new(&bk, n, f as u32, h as u32).unwrap();
    let mut prev = Tensor::<f32>::vector_uninit(&bk, (s * n) as u32, st).unwrap();
    let mut out = Tensor::<f32>::vector_uninit(&bk, (s * n) as u32, st).unwrap();
    let mut frame_t = Tensor::<f32>::vector_uninit(&bk, (f * n) as u32, st).unwrap();
    let mut max_stack = 0.0f32;
    for step in 0..steps {
        // random per-env frames; a few random resets after the first step
        let mut frames = vec![0.0f32; f * n];
        for v in frames.iter_mut() {
            *v = rng.range(-2.0, 2.0);
        }
        let mut fresh = vec![0u32; n];
        for (e, fr) in fresh.iter_mut().enumerate() {
            if step == 0 || rng.range(0.0, 1.0) < 0.07 {
                *fr = 1;
            } else {
                let _ = e;
            }
        }
        // host reference
        let mut host_stacked = vec![0.0f32; s * n]; // [dim x n] to match device
        for e in 0..n {
            let fr: Vec<f32> = (0..f).map(|k| frames[k * n + e]).collect();
            let stacked = if fresh[e] != 0 {
                hist.reset_stacked(e, &fr)
            } else {
                hist.push_stacked(e, &fr)
            };
            for (d, v) in stacked.iter().enumerate() {
                host_stacked[d * n + e] = *v;
            }
        }
        // device
        bk.write_buffer(frame_t.buffer_mut(), 0, &frames).unwrap();
        stack.set_fresh(&bk, &fresh).unwrap();
        let mut enc = bk.begin_encoding();
        stack.encode(&mut enc, &frame_t, &prev, &mut out).unwrap();
        bk.submit(enc).unwrap();
        bk.synchronize().unwrap();
        let dev: Vec<f32> = bk.slow_read_vec(out.buffer()).await.unwrap();
        let md = dev
            .iter()
            .zip(&host_stacked)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_stack = max_stack.max(md);
        // out becomes prev for the next step
        std::mem::swap(&mut prev, &mut out);
    }
    println!("[probe_stack] {steps} steps, maxdiff={max_stack:.3e}");

    // ---------------- gpu_welford vs PendingNorm ----------------
    let dim = s;
    let mut wf = GpuWelford::new(&bk, dim, n).unwrap();
    let mut pn = PendingNorm::default();
    let mut data_t = Tensor::<f32>::vector_uninit(&bk, (dim * n) as u32, st).unwrap();
    for _ in 0..steps {
        let mut data = vec![0.0f32; dim * n];
        for v in data.iter_mut() {
            *v = rng.range(-3.0, 3.0);
        }
        // host: push per env in index order (the trainer's loop order)
        for e in 0..n {
            let x: Vec<f32> = (0..dim).map(|d| data[d * n + e]).collect();
            pn.push(&x);
        }
        bk.write_buffer(data_t.buffer_mut(), 0, &data).unwrap();
        let mut enc = bk.begin_encoding();
        wf.encode(&mut enc, &data_t).unwrap();
        bk.submit(enc).unwrap();
    }
    bk.synchronize().unwrap();
    let (count, mean, m2) = wf.take_moments(&bk).await.unwrap();
    let (hc, hmean, hm2) = pn.moments();
    let md_mean = mean
        .iter()
        .zip(hmean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let md_m2 = m2
        .iter()
        .zip(hm2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "[probe_welford] count dev={count} host={hc} | mean maxdiff={md_mean:.3e} m2 maxdiff={md_m2:.3e}"
    );
}
