//! Learning tier: policy network, autodiff, and the PPO training loop.
//!
//! The learning stack is intentionally undecided (see project notes):
//! - **burn** (wgpu backend) — batteries-included autodiff + optimizers, or
//! - **vortx** + hand-rolled MLP backprop — a single dimforge/WebGPU stack
//!   shared with nexus (zero readback, browser-capable; better dora story).
//!
//! Planned modules:
//! - `policy`  — MLP policy + value network
//! - `ppo`     — PPO update (GAE + clipped surrogate objective)
//! - `rollout` — rollout buffer collected from `zealot-env`

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
