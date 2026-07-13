//! Vectorized RL environment + MDP layer over the nexus GPU physics engine.
//!
//! This is zealot's "Isaac Lab tier": it wraps nexus's batched
//! `GpuPhysicsPipeline` into a gym-style vectorized environment and provides
//! the MDP managers. nexus already supports the primitives this needs
//! (per-env reset today; motor-target writes and velocity readback after a
//! ~10-line nexus patch).
//!
//! Modules:
//! - `config` — the generic [`EnvConfig`] task interface (the MDP contract).
//! - `math`   — dependency-free vec/quat helpers used by the MDP.
//! - `rng`    — a small deterministic LCG for command / domain sampling.
//! - `robots` — robot asset specs (pure data): LeRobot bipedal, Unitree G1,
//!               Unitree H2 Plus (select with `BIPED_ROBOT`).
//! - `tasks`  — concrete task MDPs; currently flat velocity tracking.
//!
//! Planned (built incrementally):
//! - `env`     — vectorized runtime: `reset()`, `step(actions) -> (obs, reward, done)`,
//!               driving nexus's batched pipeline across all parallel environments.

pub mod config;
pub mod math;
pub mod rng;
pub mod robots;
pub mod tasks;

pub use config::{BodyState, EnvConfig};
pub use robots::{JointSpec, LeRobotBipedal, RobotSpec, NUM_JOINTS};
pub use tasks::VelocityFlatTask;

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
