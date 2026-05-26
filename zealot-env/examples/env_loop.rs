//! Skeleton of zealot's first milestone: the vectorized environment step loop.
//!
//! Run with: `cargo run -p zealot-env --example env_loop`
//!
//! This is an outline, not a working demo yet. The real implementation fills in
//! the nexus calls once we (1) apply the ~10-line nexus patch (writable joint
//! motors + readable velocities) and (2) pick the learning stack for the policy.

fn main() {
    println!("zealot-env {} — env loop skeleton", zealot_env::version());

    // TODO: build a batched scene (rapier3d -> nexus `GpuPhysicsState::from_rapier`).
    // TODO: reset(): write per-env pose/vel sub-ranges via `write_buffer` at the batch offset.
    //
    // for _step in 0..STEPS {
    //     TODO: write joint-motor targets into the nexus joint buffer  (actions)
    //     TODO: pipeline.step(...).await                               (advance the GPU sim)
    //     TODO: read poses + velocities back                           (observations)
    //     TODO: compute per-env reward + termination, reset done envs
    // }

    println!("(no physics yet — see TODOs; this is the env-loop scaffold)");
}
