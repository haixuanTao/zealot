//! Learning tier: policy network, autodiff, and the PPO training loop.
//!
//! The learning stack is intentionally undecided (see project notes):
//! - **burn** (wgpu backend) — batteries-included autodiff + optimizers, or
//! - **vortx** + hand-rolled MLP backprop — a single dimforge/WebGPU stack
//!   shared with nexus (zero readback, browser-capable; better dora story).
//!
//! Modules:
//! - `net`  — MLP (ELU, multi-layer), hand-written backprop, Adam, grad clip.
//! - `ppo`  — diagonal-Gaussian actor-critic, GAE(λ), clipped PPO update with
//!            adaptive-KL LR + entropy bonus.
//! - `rng`  — a small deterministic LCG for init / exploration.
//!
//! These are the CPU reference implementation (a port of the `pendulum_ppo`
//! math); a `burn`/GPU backend can later sit behind the same [`ppo::ActorCritic`]
//! surface.

pub mod net;
pub mod ppo;
pub mod rng;

pub use net::{Adam, Mlp, MlpGrad};
pub use ppo::{ActorCritic, PpoConfig, PpoStats, Sample, gae};

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
