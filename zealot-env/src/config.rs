//! The environment interface — zealot's MDP / "manager" layer (the Isaac Lab tier).
//!
//! This is *layer 3* of the stack (see the README): the
//! observation / action / reward / termination contract that turns a physics
//! scene into an RL environment. It is the analogue of an Isaac Lab task config
//! or a Gym `Env`, written in Rust.
//!
//! It is deliberately **decoupled from nexus** for now so the crate builds with
//! zero external dependencies (nexus needs `cargo-gpu`). The two coupling points
//! become concrete once `nexus3d` is wired in (see this crate's `Cargo.toml`):
//! - [`EnvConfig::Scene`] becomes rapier's `(RigidBodySet, ColliderSet, ImpulseJointSet)`.
//! - [`BodyState`] is filled from nexus's `Pose` / `Velocity` GPU readback.

/// Read-only physics state of one rigid body, read back from the GPU each step.
///
/// Placeholder shape: these fields map directly onto nexus `Pose` (position +
/// orientation) and `Velocity` (linear + angular). Observations and rewards are
/// computed from a per-environment slice of these.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    /// World-space position.
    pub position: [f32; 3],
    /// World-space orientation as a quaternion `(x, y, z, w)`.
    pub orientation: [f32; 4],
    /// Linear velocity.
    pub linvel: [f32; 3],
    /// Angular velocity.
    pub angvel: [f32; 3],
}

/// A task definition: implement once per `(robot, task)` pair.
///
/// The same `EnvConfig` is shared across all parallel environments. Only
/// [`build_scene`](EnvConfig::build_scene) runs per-instance (at construction /
/// reset); [`observe`](EnvConfig::observe), [`act`](EnvConfig::act),
/// [`reward`](EnvConfig::reward) and [`terminated`](EnvConfig::terminated) run
/// every step over the GPU readback.
///
/// Note: this single trait bundles the whole MDP for the first cut (like a Gym
/// `Env`). It can later decompose into separate observation/reward/termination
/// "manager terms" the way Isaac Lab does, if that proves useful.
pub trait EnvConfig {
    /// CPU-built scene for one environment instance, uploaded to nexus via
    /// `GpuPhysicsState::from_rapier`.
    ///
    /// Abstract for now so tasks don't depend on nexus; becomes
    /// `(RigidBodySet, ColliderSet, ImpulseJointSet)` once nexus is wired.
    type Scene;

    /// Length of the flat observation vector, per environment.
    fn num_obs(&self) -> usize;

    /// Length of the flat action vector, per environment.
    ///
    /// Typically the number of actuated joint DOFs (one motor target each).
    fn num_actions(&self) -> usize;

    /// Layer 1 → 2: construct one environment instance's physics scene.
    fn build_scene(&self) -> Self::Scene;

    /// Assemble the observation vector for one environment from its body states.
    ///
    /// `obs.len() == self.num_obs()`.
    fn observe(&self, bodies: &[BodyState], obs: &mut [f32]);

    /// Map a policy action to joint-motor targets written back into the sim.
    ///
    /// `action.len() == self.num_actions()`; `motor_targets` is the per-env slice
    /// of the nexus joint buffer to fill (e.g. target velocities or positions).
    fn act(&self, action: &[f32], motor_targets: &mut [f32]);

    /// Per-step scalar reward for one environment.
    fn reward(&self, bodies: &[BodyState]) -> f32;

    /// Whether this environment's episode should terminate (e.g. the robot fell).
    fn terminated(&self, bodies: &[BodyState]) -> bool;
}
