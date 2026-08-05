//! Vectorized RL environment + MDP layer over the nexus GPU physics engine.
//!
//! # Getting started
//!
//! **Watch it walk first — nothing to install:** the
//! [live demo](https://haixuantao.github.io/zealot/) is this training
//! environment compiled to WebAssembly, stepping nexus GPU physics in your
//! browser. Tap the ground and the robot walks there; switch engine tabs to
//! step the same checkpoint through rapier.js and MuJoCo; load any published
//! policy from the
//! [Hugging Face repo](https://huggingface.co/haixuantao/zealot-g1-locomotion)
//! by handle or URL.
//!
//! **Train the humanoid** (needs the
//! [`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu) toolchain — see the
//! [development guide](https://github.com/haixuanTao/zealot/blob/master/docs/development.md)):
//!
//! ```sh
//! BIPED_ROBOT=g1_29dof_agile BIPED_CUTILE_GEMM=1 BIPED_TERRAIN=1 \
//!   cargo run --release --example biped_train_gpu \
//!   --features "gpu biped_gpu cutile" -- 50000 4096 my_policy.safetensors
//! ```
//!
//! **Watch *your* policy walk:** upload the checkpoint to any Hugging Face
//! repo and open
//! `https://haixuantao.github.io/zealot/?ckpt=your-name/your-repo`.
//!
//! More: [getting started](https://github.com/haixuanTao/zealot/blob/master/docs/getting-started.md) ·
//! [how it's built and why](https://github.com/haixuanTao/zealot/blob/master/docs/explanation.md) ·
//! [benchmarks](https://github.com/haixuanTao/zealot/blob/master/docs/benchmarks.md)
//!
//! # Architecture
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
//! - `obs_history` — per-env observation-frame stacking (`BIPED_OBS_HISTORY`).
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
pub mod motion;
pub mod obs_history;
pub mod rng;
pub mod robots;
pub mod tasks;
pub mod terrain;

pub use config::{BodyState, EnvConfig};
pub use obs_history::ObsHistory;
pub use robots::{JointSpec, LeRobotBipedal, RobotSpec, NUM_JOINTS};
pub use tasks::VelocityFlatTask;

/// Crate version — used to sanity-check that the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
