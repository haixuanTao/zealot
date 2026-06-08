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
- **GPU policy port** (the vortx PPO actor/critic — verified piece by piece
  against the CPU `zealot-rl` reference, each to float epsilon): `elu_check`
  (ELU fwd/bwd kernels), `mlp_backward_batch` (multi-layer batched ELU backward,
  `--features "gpu biped_gpu"`), `ppo_grad_check` (clipped-surrogate + value-loss
  gradient kernels), and `policy_forward_bench` (CPU vs GPU batched forward,
  `--features "gpu biped_gpu"`). The batched forward is wired into
  `biped_render_nexus` so the rollout runs the policy on the same backend as the
  physics.
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

## Quick start — reproduce the GPU benchmark (agent-runnable)

Measures full training-iteration throughput (CPU vs GPU, env-control-steps/sec)
on an RTX 5090 Linux box. Assumes the sibling-repo layout (`zealot`, `vortx`,
`nexus3d`/`khal` under the same parent dir, e.g. `~/Documents/work/`).

**One-time setup** (the cargo-gpu version + PATH are the two known gotchas):

```sh
export PATH=$HOME/.cargo/bin:$PATH                          # cargo-gpu subprocess won't find it otherwise
cargo install cargo-gpu --version 0.10.0-alpha.1 --force    # plain `cargo install cargo-gpu` gets 0.1.0 = wrong API
( cd ../vortx/vortx-shaders && cargo-gpu install --auto-install-rust-toolchain )   # installs the rust-gpu nightly into the cache
```

**Run the benchmark** (args: `<num_envs> <T-steps> <epochs> <minibatches>`):

```sh
export PATH=$HOME/.cargo/bin:$PATH
cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- 8192 32 5 16
```

Expected output on a 5090 (numbers are `num_envs·T / iteration_time` =
collection + learning, the same unit as the tables below):

```
full iteration — 8192 envs, T=32, 5e x 16mb
  FULL CPU (rapier + CPU MLP + CPU update) :  72511 ms =   3.6 k env/s
  FULL GPU (nexus + vortx + GPU update)    :  11916 ms =  22.0 k env/s
```

Sweep N by re-running with `2048` / `4096` / `8192`. For **rollout-only**
throughput (no PPO update, matches the rollout table below): `cargo run --release
--example rollout_e2e_bench --features "gpu biped_gpu" -- 8192`. First build is
slow (shader compile); the toolchain is cached afterward (~16 s rebuilds).

## Benchmark — full training iteration (rollout + PPO update)

A full training step is a **rollout** plus a **PPO update**. `examples/biped/iter_e2e_bench.rs` times one whole iteration each
way — **full CPU** = rapier rollout (CPU MLP + rayon physics) + CPU PPO update;
**full GPU** = nexus rollout (vortx policy) + GPU PPO update (the Stage-B path
verified in `gpu_update_check`) — over a T=32 rollout and a 5-epoch × 16-minibatch
update. Throughput is `N·T / iteration_time`, same env-control-steps/s unit, and
the **Isaac** column is PhysX 5's total `Computation` (collection + learning):

The GPU update is **GPU-resident** (the batch is normalized + uploaded once, each
minibatch is an on-GPU column gather), uses a **tiled GEMM**, and the GEMM inner
loop is **vec4** (4-wide FMA) — all verified bit-exact against the CPU update
(`gpu_update_check`):

| N envs | mac CPU | mac GPU | linux CPU (24-core) | linux GPU (RTX 5090, WebGPU) | linux GPU (RTX 5090, native CUDA) | Isaac/PhysX 5 (5090) |
|-------:|--------:|--------:|--------------------:|-----------------------------:|----------------------------------:|---------------------:|
| 512    | 2.0 k   | 3.8 k   | 2.1 k               | 6.4 k                        | **12.0 k**                        | —                    |
| 1 024  | 2.0 k   | 5.3 k   | 2.3 k               | 10.6 k                       | **19.1 k**                        | —                    |
| 2 048  | 2.1 k   | 6.5 k   | 2.7 k               | 15.5 k                       | **26.2 k**                        | 67 k                 |
| 4 096  | 2.1 k   | 7.7 k   | 3.1 k               | 19.9 k                       | **30.1 k**                        | 126 k                |
| 8 192  | 2.1 k   | 8.0 k   | 3.6 k               | 23.1 k                       | **32.6 k**                        | 201 k                |

The **native-CUDA** column is the *same* Rust stack compiled to PTX via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Rust→PTX, LLVM 21) — no WebGPU
at all, both the vortx tensor ops and the `nexus3d` physics, straight from the
*verbatim* `#[spirv]` shader source, bit-exact vs WebGPU (boxes-physics pose
fingerprint identical; biped obs-gather err 0.0). It's **~1.4× over WebGPU at
N = 8 192 and up to ~1.9× on the small batches** (where WebGPU's per-dispatch
overhead bites hardest): the lever is the GEMM-heavy PPO update (~3× via
`cuLaunchKernel` — no per-dispatch bind groups, higher GEMM throughput), diluted
by the heavier articulated-multibody physics. Getting there took ~12 general
cuda-oxide codegen fixes plus two khal↔cuda-oxide ABI fixes (push element-count not
byte-length for slice kernel args; pass a shader's `&0` offset by value to dodge a
null-deref that DCE'd a whole kernel).

Full GPU beats full CPU by ~**6.5×** on the 5090 (~3.5× on the mac). The
optimizations (GPU-resident batch + tiled GEMM + vec4) lifted the 5090 WebGPU
iteration from 12.7 k → **23.1 k env/s** at N = 8 192. **The Isaac gap is now ~6–9×
(was ~16×).** Two hardware notes worth recording. (1) The vec4 **inner-loop FMA** is a ~12%
win on the 5090 but **flat on the mac** — Metal auto-vectorizes the inner loop;
rust-gpu → SPIR-V → NVIDIA does not, so the explicit `Vec4` FMA matters there.
(2) vec4 **global loads** (a `gemm_tiled_vec4` with 128-bit loads, verified
bit-exact) add **0%** — because a *tiled* GEMM already amortizes global memory
(each element is loaded once into shared memory and reused), so it isn't
global-bandwidth-bound; the win was compute, not bandwidth. The remaining gap is
the rollout (~5× off PhysX) plus the update vs Isaac's fused-CUDA learning step
(~0.09 s) — the next lever is fewer dispatches, not vec4. None of it is new math.

Reproduce:

```sh
# WebGPU
cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- <num_envs> 32 5 16
# native CUDA (needs the cuda-oxide toolchain + embedded cubins)
BIPED_CUDA=1 cargo run --release --example iter_e2e_bench --features "gpu biped_gpu cuda_backend" -- <num_envs> 32 5 16
```

## Benchmark — full CPU vs full GPU rollout

`examples/biped/rollout_e2e_bench.rs` runs the full rollout control step —
**policy forward** (an action per env) **+ physics step** — and reports
wall-clock throughput. **Full CPU** = rapier multibody physics (rayon) + the CPU
MLP policy (serial per-env `actor.mean`/`critic.value`). **Full GPU** = one
batched nexus `GpuPhysicsState` + the vortx GPU policy (batched GEMM forward).
Same MJCF model, same MDP, same N envs, same net shapes (actor
`[43,256,256,128,12]`, critic `[49,512,256,128,1]`). No PPO update — this is
rollout throughput, like `biped_fps`.

Numbers are **env-control-steps / second**. Bold = winner of each CPU/GPU pair.
Measured on a M-series mac + WebGPU and a 24-core x86 box with an RTX 5090. The
**Isaac** column is NVIDIA Isaac Lab + PhysX 5 on the same 5090, same task
(`Velocity-LeRobot-NoArms-v0`), measured the same way — rsl_rl *collection*
throughput (`num_envs · 24 / collection_time`, i.e. physics + policy inference,
no learning step) — as the production reference for "how fast this *should* go".

| N envs | mac CPU (rapier + MLP) | mac GPU (nexus + vortx) | linux CPU (24-core) | linux GPU (RTX 5090) | Isaac/PhysX 5 (5090) |
|-------:|-----------------------:|-------------------------:|--------------------:|---------------------:|---------------------:|
| 32     | **3.7 k**              | 0.5 k                    | **5.0 k**           | 0.7 k                | —                    |
| 128    | **4.1 k**              | 1.7 k                    | **6.0 k**           | 2.4 k                | —                    |
| 512    | 4.1 k                  | **4.7 k**                | 6.4 k               | **8.7 k**            | —                    |
| 1 024  | 4.1 k                  | **6.5 k**                | 6.5 k               | **15.9 k**           | —                    |
| 2 048  | 4.0 k                  | **8.6 k**                | 6.5 k               | **27.0 k**           | 73.8 k               |
| 4 096  | 3.9 k                  | **10.4 k**               | 6.6 k               | **35.1 k**           | 139 k                |
| 8 192  | 3.9 k                  | **10.8 k**               | 6.5 k               | **44.5 k**           | 220 k                |

| machine                          | peak CPU | peak GPU | GPU > CPU at N ≈ | best GPU/CPU ratio |
|----------------------------------|---------:|---------:|-----------------:|-------------------:|
| mac (M-series + WebGPU)          | 4.1 k    | 10.8 k   | ~500             | 2.77×              |
| linux (24-core + RTX 5090)       | 6.6 k    | 44.5 k   | ~450             | 6.80×              |

What this says in practice:

- Below N ≈ 450–500 the full-CPU path wins on both machines: rapier physics is
  cheap at small batch and the GPU path pays a fixed per-step physics + policy
  readback (~70 ms on the mac, ~50 ms on the 5090) that doesn't amortize yet.
- Past the crossover the full-GPU path pulls ahead — ~2.8× by N = 8 192 on the
  mac, **~6.8× on the 5090** (44.5 k vs 6.5 k env/s). The full-CPU throughput is
  flat (~4 k mac, ~6.5 k 24-core) — bottlenecked by the **serial** per-env CPU
  MLP forward, not the physics.
- The PPO *update* still runs on the CPU and isn't in this measurement — moving
  it to the GPU (Stage B) is what makes a full training iteration scale, not just
  the rollout.
- Versus the **Isaac/PhysX 5** reference, nexus+vortx is ~2.7× behind at N = 2 048
  and ~5× at N = 8 192 (44.5 k vs 220 k). That gap is the all-Rust/WebGPU tax —
  PhysX is the ceiling to chase, and it scales harder with batch (the nexus GPU
  curve is flattening past 4 k while PhysX keeps climbing).

Reproduce:

```sh
cargo run --release --example rollout_e2e_bench --features "gpu biped_gpu" -- <num_envs> <steps>
```

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
