//! Robot asset specifications — the analogue of AGILE's `assets/robots/*.py`.
//!
//! A robot spec is **pure data**: the kinematic/actuation facts a task needs that
//! don't belong to any single MDP — joint names and ordering, default ("home")
//! pose, per-joint PD gains and effort limits, action scales, and where the URDF
//! lives. It is deliberately free of nexus / GPU types so it builds and unit-tests
//! without the `cargo-gpu` toolchain; the env loop (Isaac Lab "manager" tier)
//! consumes it once physics is wired.

pub mod lerobot_bipedal;

pub use lerobot_bipedal::{JointSpec, LeRobotBipedal};
