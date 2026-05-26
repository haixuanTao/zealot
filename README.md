# zealot

Reinforcement-learning locomotion on top of [nexus](https://github.com/dimforge/nexus),
dimforge's cross-platform GPU physics engine — aiming to be, roughly, "the
WBC-AGILE of nexus" but all-Rust and WebGPU-native.

## Workspace layout

| Crate | Role | Analogy |
| --- | --- | --- |
| `zealot-env` | Vectorized environment + MDP layer over nexus's batched `GpuPhysicsPipeline` (observations, actions, rewards, terminations, per-env reset). | Isaac Lab tier |
| `zealot-rl` | Policy network, autodiff, PPO. | rsl_rl tier |

nexus itself provides the GPU physics + parallel environments (the Isaac Sim tier).

## Describing environments

There are three distinct layers, often conflated:

1. **Asset / scene description** (bodies, joints, collision, mass) — URDF / MJCF / USD / SDF.
2. **Engine model** — nexus consumes **rapier** types via `GpuPhysicsState::from_rapier`;
   it has no file-format loader of its own.
3. **RL env contract (the MDP)** — observation / action / reward / reset; this is *code*,
   not a file. In zealot it's the [`EnvConfig`] trait in `zealot-env` (`config.rs`).

Because nexus is rapier-backed, the asset path is always `format → rapier → from_rapier → GPU`,
so the format choice reduces to "what can produce rapier types." Decisions:

- **First toy walker:** build the scene programmatically in Rust (like the nexus examples) —
  faster than authoring/parsing a file for ~5 bodies.
- **Real robots (G1/T1):** load **URDF** via [`rapier3d-urdf`](https://crates.io/crates/rapier3d-urdf)
  (dimforge, rapier 0.32 — matches nexus). MJCF has no mature pure-Rust loader; USD is Omniverse-bound.
- **The MDP** stays Rust code (`EnvConfig`), separate from the asset.

There is no CLI binary — the project is demonstrated through runnable **examples**
(as nexus does). Examples live alongside the crate they exercise, e.g.
`cargo run -p zealot-env --example env_loop`.

## Status

Scaffold only. `env_loop` is an outline of the first milestone (the vectorized
env step loop); it has no physics yet. Two things gate the first real example:

1. A ~10-line nexus patch: make joint-motor targets writable + velocities readable.
2. The learning-stack decision for `zealot-rl`: **burn** (fast, batteries-included)
   vs **vortx + hand-rolled backprop** (unified dimforge/WebGPU stack, stronger
   browser/dora story). Both are confirmed viable on nexus.

## Building

`zealot-env` will depend on `nexus3d`, whose Rust-GPU shaders require
[`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu):

```sh
cargo install cargo-gpu
```
