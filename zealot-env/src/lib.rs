//! Vectorized RL environment + MDP layer over the nexus GPU physics engine.
//!
//! This is zealot's "Isaac Lab tier": it wraps nexus's batched
//! `GpuPhysicsPipeline` into a gym-style vectorized environment and provides
//! the MDP managers. nexus already supports the primitives this needs
//! (per-env reset today; motor-target writes and velocity readback after a
//! ~10-line nexus patch).
//!
//! Modules:
//! - `config` — the [`EnvConfig`] task interface (the MDP contract). **Implemented.**
//!
//! Planned (built incrementally):
//! - `env`     — vectorized runtime: `reset()`, `step(actions) -> (obs, reward, done)`,
//!               driving nexus's batched pipeline across all parallel environments.
//! - `command` — command sampling (e.g. target base velocities).

pub mod config;

pub use config::{BodyState, EnvConfig};

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
