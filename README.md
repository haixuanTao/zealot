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
in the workspace-root `examples/` directory. They touch the GPU, so each is gated
behind a feature (off by default; needs the [`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu)
toolchain). Two groups:

- **Tensor primitives** (`--features gpu`): vortx forward/backward/Adam demos —
  `linear_forward`, `mlp_forward`, `mlp_backward`, `train_regression`.
  e.g. `cargo run --example train_regression --features gpu`.
- **Pendulum** (`--features pendulum`, in `examples/pendulum/`): a GPU rigid-body
  inverted pendulum and RL trainers on it, from `pendulum_smoke` (a physics smoke
  test) up through `pendulum_pg` (REINFORCE), `pendulum_ppo` (PPO swing-up), and
  `pendulum_gpu_policy` (GPU-resident vortx policy, also needs `gpu`).
  e.g. `cargo run --release --example pendulum_ppo --features pendulum`.
  See [`examples/pendulum/README.md`](examples/pendulum/README.md) for a full
  getting-started guide (prerequisites, the example progression, video rendering).

## Status

The `zealot-env` / `zealot-rl` crates are still scaffolds (env step loop has no
physics yet). The working physics + RL currently lives in the **pendulum
examples**, ported verbatim from nexus-rl's `pendulum_headless`: they drive
`nexus3d` directly rather than going through `zealot-env`/`zealot-rl`. Folding
that logic into the crates is the next step, gated on:

1. A ~10-line nexus patch: make joint-motor targets writable + velocities readable.
2. The learning-stack decision for `zealot-rl`: **burn** (fast, batteries-included)
   vs **vortx + hand-rolled backprop** (unified dimforge/WebGPU stack, stronger
   browser/dora story). Both are confirmed viable on nexus — `pendulum_gpu_policy`
   already trains a vortx policy on-GPU.

## Benchmark — rapier CPU vs nexus GPU

`examples/biped/biped_fps.rs` steps the LeRobot bipedal on both engines under
identical control settings (50 Hz control, physics decimation 4, neutral
actions) and reports wall-clock throughput. Same MJCF model, same MDP, same N
envs; only the physics engine differs — CPU runs independent rapier multibodies
stepped in parallel via rayon, GPU runs one batched `GpuPhysicsState` advanced
per dispatch.

Numbers are **env-control-steps / second** (multiply by 4 for raw physics
sub-steps/s). Bold = winner of that row. Measured on a M-series mac + WebGPU
and on a 24-core x86 box with an RTX 5090 (wgpu Vulkan).

| N envs | mac CPU (rapier) | mac GPU (nexus, WebGPU) | linux CPU (24-core rapier) | linux GPU (nexus, RTX 5090) |
|-------:|------------------:|-------------------------:|---------------------------:|----------------------------:|
| 32     | **6.2 k**         | 503                      | **13.9 k**                 | 758                         |
| 128    | **12.5 k**        | 2.0 k                    | **17.8 k**                 | 3.0 k                       |
| 512    | **11.2 k**        | 5.9 k                    | **23.0 k**                 | 11.3 k                      |
| 2 048  | **12.0 k**        | 10.0 k                   | 24.7 k                     | **34.7 k**                  |
| 4 096  | **12.0 k**        | 11.7 k                   | 24.9 k                     | **41.0 k**                  |
| 8 192  | 9.5 k             | **11.8 k**               | 25.0 k                     | **49.2 k**                  |
| 16 384 | —                 | —                        | 25.1 k                     | **56.9 k**                  |
| 32 768 | —                 | —                        | 25.2 k                     | **59.6 k**                  |

| machine                          | peak CPU | peak GPU | GPU > CPU at N ≈ | best GPU/CPU ratio |
|----------------------------------|---------:|---------:|-----------------:|-------------------:|
| mac (M-series + WebGPU)          | 12.5 k   | 11.8 k   | ~6 000           | 1.24×              |
| linux (24-core + RTX 5090)       | 25.2 k   | 59.6 k   | ~1 800           | 2.36×              |

What this says in practice:

- On the mac, CPU rapier wins almost everywhere; nexus only edges ahead past
  N ≈ 8 000, and that's because per-step WebGPU readback (`slow_read_buffer`
  on `links_workspace` + `body_poses`) dominates the host loop.
- On the 5090 box, nexus crosses CPU at ~2 000 envs and runs ~2× CPU at
  N = 8 192. GPU utilisation sits at 86–98% during the bench — real GPU
  compute is the bottleneck there, not host plumbing.
- The CPU plateau (~25 k env/s on 24 cores, ~12 k on the mac) is core-count
  bound. The GPU plateau is bounded by the per-env Rust post-step loop
  (`update_feet` + `task.observe` + the `Vec<f32>` obs/critic_obs allocs);
  moving obs/reward into a compute shader would push it further.

Reproduce:

```sh
cargo run --release --example biped_fps --features "cpu biped_gpu" -- <num_envs> <control_steps>
```

### Cross-engine reference — Isaac Lab + PhysX 5

To ground-truth "how fast *should* GPU physics be for this robot?", we ran the
same LeRobot bipedal velocity-tracking task through NVIDIA's [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)
(Isaac Lab v2.3.2 + Isaac Sim 5.1 + PhysX 5). Identical robot (`Velocity-LeRobot-NoArms-v0`),
identical 50 Hz control / 200 Hz physics / decimation 4, same RTX 5090. The
"Isaac Lab" column below is the rsl_rl `collection` time (physics only, no
policy update), which is directly comparable to the `biped_fps` env-ctrl/s.

| N envs | zealot rapier CPU (24-core) | zealot nexus GPU (5090) | Isaac Lab + PhysX 5 (5090) | PhysX vs nexus |
|-------:|----------------------------:|------------------------:|---------------------------:|---------------:|
| 2 048  | 24.7 k                      | 34.7 k                  | **73.8 k**                 | 2.1×           |
| 4 096  | 24.9 k                      | 41.0 k                  | **135.4 k**                | 3.3×           |
| 8 192  | 25.0 k                      | 49.2 k                  | **229.7 k**                | 4.7×           |
| 16 384 | 25.1 k                      | 56.9 k                  | **331.8 k**                | 5.8×           |
| 32 768 | 25.2 k                      | 59.6 k                  | **430.5 k**                | **7.2×**       |

(`zealot nexus GPU` column reflects the post-Tier-1 host-path rework
documented in the next sub-section; pre-rework numbers are listed there as
a before/after.)

So the answer to "shouldn't GPU be way more than 2× CPU?" is *yes, and it is*
— a mature GPU physics engine (PhysX 5) on this exact robot hits **17× the
24-core CPU rapier** at N = 32 768, and scales nearly linearly with N. The
zealot/nexus GPU path leaves roughly **7× on the table** vs PhysX 5, and that
gap grows with N because nexus plateaus around 60 k env/s while PhysX keeps
climbing.

The gap isn't in the GPU physics kernels themselves — it's in the surrounding
plumbing: per-step host readback over wgpu→Vulkan, the per-env Rust post-step
loop, and (for the biggest gap at large N) the fact that PhysX's broad-phase
+ solver are tuned for thousands of identical instanced articulations while
nexus is still in early-stage bring-up. PhysX is also a decade-old
NVIDIA-tuned engine; nexus is research-grade open source. None of this is a
nexus knock — it's the right reference point for where there's headroom.

### After Tier 1 host-side optimizations

Three host-side fixes to `BipedNexusBatchEnv` to claw back some of the
PhysX gap without touching nexus itself:

1. **Drop the `slow_read_buffer(links_workspace)` from the hot path.** The
   `body_poses` readback alone now feeds the entire post-step loop — joint
   angles derive from parent⇄child relative rotation
   (`q_child = q_parent · rest_quat · R_z(θ)`), base + foot lin/ang
   velocities finite-diff against the previous step's poses. Kills ~13 MB of
   per-step CPU↔GPU traffic at N = 8 192.
2. **rayon-parallel post-step compute.** The per-env loop (`compute_feet`,
   `read_state`, `task.observe`, `task.observe_critic`) now runs over
   `into_par_iter().with_min_len(64)`; per-env mutable state commits stay
   serial.
3. **Drop `gpu.synchronize()` from hot path; throttle
   `auto_resize_buffers`.** The next `slow_read_buffer` syncs implicitly;
   `auto_resize_buffers` only fires every 32 control steps (it's a no-op
   once contact buffers settle after warmup).

Effect on champagne (RTX 5090, 24-core, Vulkan):

| N envs | nexus before | nexus after | Δ | new vs PhysX |
|-------:|-------------:|------------:|-----:|-------------:|
| 2 048  | 33.8 k       | 34.7 k      | +3%  | 2.1× slower  |
| 4 096  | 39.2 k       | 41.0 k      | +5%  | 3.3× slower  |
| 8 192  | 47.0 k       | 49.2 k      | +5%  | 4.7× slower  |
| 16 384 | 51.6 k       | 56.9 k      | +10% | 5.8× slower  |
| 32 768 | 53.1 k       | 59.6 k      | **+12%** | 7.2× slower |

So real but modest (5–12% at large N, ~flat below N = 2 000). The bench
already showed GPU utilisation at 86–98% before these changes, so on the
5090 box the engine itself was the gate — host fixes shave ~8 ms of CPU
time per ~170 ms step at N = 8 192, but can't crack the GPU-compute ceiling.
Mac numbers move a tick (the WebGPU plumbing on Metal is slower in absolute
terms but GPU compute is still the gate at the N values where it matters).

A fourth Tier 1 idea — preallocating obs/critic slabs to drop the per-env
`vec![0.0; OBS_DIM]` calls — was deliberately skipped: making it truly
zero-alloc requires changing `StepOut` to borrow from env-owned buffers,
but the trainer interleaves `env.reset_env(e)` (which needs `&mut env`)
into the consumption loop, which conflicts with an outstanding `&env`
borrow. The alloc itself is only ~50 floats per env per step (small
compared to ~150–200 GPU dispatches per step), so not worth the API
churn alone — revisit if/when we tackle Tier 2 (GPU-side obs/reward) and
the contract is being changed anyway.

**Where the next 2–3× lives** (Tier 2, not yet implemented):

- Move obs + reward computation into a nexus compute shader. The current
  bench reads back the full `body_poses` (~14 poses × N envs) every step
  just to derive ~50 obs floats per env on host. A GPU-side compaction
  would cut readback by ~10× and let the host post-step loop disappear.
- Switch the nexus3d backend from wgpu→Vulkan to native CUDA (nexus
  supports both). At ~150–200 dispatches per control step, command-encoder
  overhead per dispatch (~µs each on Vulkan) is real CPU time the CUDA
  path doesn't pay.
- Lower `SOLVER_ITERS` (currently 8) — PhysX hits the same biped stability
  with TGS at 4–6 iters. Halving solver work would ~halve GPU time per
  step.

## Building

`zealot-env` will depend on `nexus3d`, whose Rust-GPU shaders require
[`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu):

```sh
cargo install cargo-gpu
```

## Development

Versioned git hooks in `.githooks/` enforce formatting, warnings, and tests.
Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

- **pre-commit** runs `cargo fmt --check` (workspace members only) and
  `cargo check --workspace --all-targets` with `RUSTFLAGS="-D warnings"`, so any
  warning fails the commit.
- **pre-push** runs `cargo test --workspace` (also with `-D warnings`) so the
  test suite only gates pushes, keeping individual commits fast.

The `gpu` feature is intentionally left off everywhere — its checks need the
`cargo-gpu` toolchain. To run the same checks by hand:

```sh
cargo fmt -p zealot -p zealot-env -p zealot-rl
RUSTFLAGS="-D warnings" cargo check --workspace --all-targets
RUSTFLAGS="-D warnings" cargo test --workspace
```
