//! Stage-2 measurement, physics half: does capturing the nexus decimation loop
//! into a CUDA graph and replaying it (one `cuGraphLaunch`/step, zero host
//! encode/submit/sync between the ~decimation×N dispatches) beat the current
//! per-step `gpu.synchronize()` pattern? This is where the host-bound ceiling
//! actually lives (the policy+sampler half does not — see resident_rollout_bench).
//!
//! Run (native CUDA, fixed-grid required for capture-safety):
//!   export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
//!   export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
//!   BIPED_CUDA=1 cargo run --release --example physics_capture_bench \
//!       --features "gpu biped_gpu cuda_backend"

#[path = "biped_env.rs"]
mod biped_env;
#[path = "biped_env_nexus.rs"]
mod biped_env_nexus;
#[path = "gpu_policy.rs"]
mod gpu_policy;

use biped_env_nexus::{default_mjcf_path, BipedNexusBatchEnv};

const T_STEPS: usize = 32;
const SWEEP: [usize; 5] = [512, 1024, 2048, 4096, 8192];

#[async_std::main]
async fn main() {
    // Fixed-grid dispatch is required so the decimation loop has no indirect
    // host readbacks that would break stream capture.
    unsafe {
        std::env::set_var("BIPED_FIXED_GRID", "1");
    }
    let xml = std::fs::read_to_string(default_mjcf_path()).expect("mjcf");

    println!("\nPhysics decimation loop: per-step sync vs CUDA-graph replay  (T={T_STEPS} control steps)");
    println!(
        "{:>7} | {:>12} | {:>12} | {:>9} | {:>9} | {:>7}",
        "N", "sync ms", "graph ms", "sync k/s", "graph k/s", "speedup"
    );
    println!("{}", "-".repeat(72));

    for &n in &SWEEP {
        let mut env = BipedNexusBatchEnv::new(&xml, n, 32, 0xC0FFEE).await;
        let (sync_ms, graph_ms) = env.bench_physics_modes(T_STEPS).await;
        let sync_eps = (n * T_STEPS) as f64 / (sync_ms / 1e3) / 1e3;
        match graph_ms {
            Some(g_ms) => {
                let g_eps = (n * T_STEPS) as f64 / (g_ms / 1e3) / 1e3;
                println!(
                    "{:>7} | {:>12.1} | {:>12.1} | {:>9.1} | {:>9.1} | {:>6.2}x",
                    n, sync_ms, g_ms, sync_eps, g_eps, sync_ms / g_ms
                );
            }
            None => {
                println!(
                    "{:>7} | {:>12.1} | {:>12} | {:>9.1} | {:>9} | {:>7}",
                    n, sync_ms, "n/a", sync_eps, "n/a", "(cpu/wgpu)"
                );
            }
        }
    }
    println!();
}
