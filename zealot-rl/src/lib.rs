//! Learning tier: policy network, autodiff, and the PPO training loop.
//!
//! These are the **CPU reference implementations**. The production trainer
//! runs the same math as GPU kernels on vortx (the dimforge tensor stack
//! nexus shares), and every GPU kernel is verified against this crate to
//! float epsilon — see `examples/` (`ppo_grad_check`, `elu_check`,
//! `policy_forward_bench`) in the
//! [zealot repo](https://github.com/haixuanTao/zealot). Start at the crate
//! front page of
//! [`zealot_env`](https://haixuantao.github.io/zealot/doc/zealot_env/) for
//! the getting-started walkthrough.
//!
//! Modules:
//! - `net`  — MLP (ELU, multi-layer), hand-written backprop, Adam, grad clip.
//! - `ppo`  — diagonal-Gaussian actor-critic, GAE(λ), clipped PPO update with
//!            adaptive-KL LR + entropy bonus.
//! - `rng`  — a small deterministic LCG for init / exploration.
//! - `trainstate` — the `<ckpt>.train` sidecar: Adam moments, global step and
//!            curriculum progress, i.e. the state a *resumed* run needs but
//!            [`ppo::ActorCritic`] deliberately does not carry.
//!
//! These are the CPU reference implementation (a port of the `pendulum_ppo`
//! math); a `burn`/GPU backend can later sit behind the same [`ppo::ActorCritic`]
//! surface.

pub mod net;
pub mod ppo;
pub mod rng;
pub mod trainstate;

pub use net::{Adam, Mlp, MlpGrad};
pub use ppo::{ActorCritic, PpoConfig, PpoStats, Sample, gae};
pub use trainstate::{MomentSet, TrainState};

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
