//! Task definitions — the analogue of AGILE's `rl_env/tasks/*`.
//!
//! A task is the MDP: how a robot's physics state becomes observations, how a
//! policy action becomes joint targets, the reward, and the termination rule.
//! Each task is one file; the per-term functions (frame transforms, tracking
//! kernels) live in the shared [`crate::math`] helpers, mirroring AGILE's
//! "self-contained task config + shared MDP library" split.
//!
//! Tasks here operate on plain state structs ([`crate::tasks::velocity_flat::RobotState`]),
//! not nexus types, so the whole MDP is unit-testable on the CPU. The vectorized
//! env loop (a later stage) fills those structs from GPU readback.

pub mod velocity_flat;

pub use velocity_flat::{
    BaseState, CommandSampler, RewardBreakdown, RewardWeights, RobotState, VelocityCommand,
    VelocityFlatTask,
};
