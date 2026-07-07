//! # zealot — GPU-resident RL for the LeRobot bipedal
//!
//! All-Rust PPO training for the LeRobot bipedal on dimforge's GPU physics
//! stack (nexus rigid-body sim + vortx tensors + khal backends). The whole
//! loop — physics, policy forward, PPO update — runs on the GPU; this crate
//! hosts the runnable examples under `examples/` (there is no library API).
//!
//! ## Pick your platform
//!
//! | you have | guide | backend |
//! |---|---|---|
//! | an NVIDIA box (RTX 5090 / sm_120) | [`guides::train_on_5090`] | native CUDA (cuda-oxide) |
//! | a Mac (Apple Silicon) | [`guides::train_on_macos`] | WebGPU → Metal |
//!
//! Both guides produce the same trainer, checkpoints, and (verified) the
//! same physics to ~1e-4 — pick by hardware, not by feature set.
//!
//! ## Quick start (either platform)
//!
//! ```bash
//! #                                        iters  num_envs  checkpoint
//! cargo run --release --example biped_train_gpu \
//!     --features "gpu biped_gpu" --        2000   1024      walking_policy.safetensors
//! ```
//!
//! The backend auto-selects (CUDA on sm_120, otherwise WebGPU). See your
//! platform guide for the required sibling checkouts and one-time setup —
//! **macOS needs the vendored-naga patch clone** or the sim silently breaks.

// Doc-only modules: each guide is its own rustdoc page (sidebar-navigable),
// pulled verbatim from docs/*.md via include_str! so the markdown stays the
// single source of truth. Edit the markdown, not this file.
/// Platform setup + training guides (one page per platform).
pub mod guides {
    /// Training on an RTX 5090 box with the native CUDA backend.
    pub mod train_on_5090 {
        #![doc = include_str!("../docs/train-on-5090.md")]
    }
    /// Training on macOS (Apple Silicon / Metal) — includes the mandatory
    /// naga patch and verification goldens.
    pub mod train_on_macos {
        #![doc = include_str!("../docs/train-on-macos.md")]
    }
}
