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
  physics. See the policy-forward benchmark below.
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

**Sibling repos.** A standalone `zealot` clone will not build — `nexus3d` is a
path dependency and the `[patch.crates-io]` table redirects `khal`/`vortx`/
`parry3d`/`rapier3d` to local dimforge forks. Clone these next to `zealot/`
(all under the same parent dir, e.g. `~/Documents/work/`), each on the branch
last verified to build together:

| Repo | Clone | Branch |
| --- | --- | --- |
| nexus-cuda | `gh repo clone haixuanTao/nexus-cuda` | `master` |
| khal | `gh repo clone haixuanTao/khal` | `feat/cuda-oxide-backend` |
| vortx | `gh repo clone haixuanTao/vortx` | `feat/gpu-policy-shaders` |
| parry | `gh repo clone haixuanTao/parry` | `spirv-compat` |
| rapier | `gh repo clone haixuanTao/rapier` | `world` |

The MJCF model is loaded from a fixed path (`biped_env_nexus.rs`); clone the
model repo into `~/Documents/work/` too:

```sh
gh repo clone haixuanTao/lerobot-humanoid-design -- --branch main   # provides to_real_robot/RL_policy/robot.xml
```

**One-time setup** (the cargo-gpu version + PATH are the two known gotchas):

```sh
export PATH=$HOME/.cargo/bin:$PATH                          # cargo-gpu subprocess won't find it otherwise
cargo install cargo-gpu --version 0.10.0-alpha.1 --force    # plain `cargo install cargo-gpu` gets 0.1.0 = wrong API
( cd ../vortx/vortx-shaders && cargo-gpu install --auto-install-rust-toolchain )   # installs the rust-gpu nightly into the cache
```

**Build fixes for a fresh checkout (mid-2026).** `glam 0.33.1` was published
after these forks were last built and breaks the shader toolchain
(`spirv-std 0.10.0-alpha.1` can't compile against glam ≥ 0.33 under
`default-features = false` — 112 errors about missing `UVec3`/`UVec4`). A clean
clone of the branches above needs the fixes below so the `cargo gpu build`
sub-invocations resolve glam < 0.33 and the cross-fork feature graph lines up.

Three are **committed source edits** (already in the branches above if you pull
after this writing; listed here for provenance):

1. **khal** — `crates/khal-std/Cargo.toml`: add `glam = { version = "=0.32.1",
   default-features = false }` beside the `spirv-std` dependency (caps glam for
   the nexus shader builds, mirroring the existing pin in `vortx-shaders`).
2. **vortx** — `vortx-shaders/Cargo.toml`: add `cuda-oxide = ["khal-std/cuda-oxide"]`
   (nexus-cuda's shader crates reference `vortx-shaders/cuda-oxide`, which must
   exist even though the WebGPU benchmark never activates it).
3. **vortx** — `Cargo.toml`: uncomment the `[patch.crates-io]` block so vortx's
   own shader sub-build resolves `khal-std` to the local fork (the published
   `khal-std 0.1.1` lacks the `cuda-oxide` feature).

One is a **post-clone step**, because `vortx/Cargo.lock` is git-ignored — a fresh
clone re-resolves and `spirv-std`'s open `glam >=0.30.8` grabs 0.33.1 again. Pin
it back down once per clone:

```sh
cd ../vortx
cargo update -p glam@0.33.1 --precise 0.32.1   # spirv-std drops to glam 0.30.10
# then edit Cargo.lock: in the [[package]] block for spirv-std, change its
# `"glam 0.30.10"` dependency line to `"glam 0.32.1"` so it matches glamx
# (the proven config; both at 0.32.1, as nexus-cuda's committed lock has them).
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

Sweep N by re-running with `2048` / `4096` / `8192`. First build is slow
(shader compile); the toolchain is cached afterward (~16 s rebuilds).

### Reproduced on an RTX 4090 Laptop GPU (16 GB)

Independently reproduced on a laptop — **RTX 4090 Laptop GPU** (16 GB, 150 W
cap; ~mobile, not the desktop 4090) + a 32-thread mobile CPU — following the
quick start above (including the four mid-2026 build fixes). Every full-GPU
iteration passed the bench's built-in correctness guard (the GPU minibatch
gather matched the CPU-normalized batch bit-exact, `err 0.0e0`).

Its GPU throughput is folded straight into the full-training-iteration table
below as the **RTX 4090 laptop** column, so 4090 and 5090 sit side by side. The
laptop's own full-CPU baseline — needed to read its speedups, since it differs
from champagne's 24-core — is **0.86 k → 1.45 k env/s** (N = 512 → 8 192),
giving full-iteration speedups of **7.5× → 12×**.

Reading the laptop numbers:

- **GPU ceiling ~0.7× the 5090.** Peak GPU iteration 16.2 k vs 23.1 k — in line
  with a 16 GB / 150 W mobile part vs a full desktop 5090. N = 8 192 fits
  comfortably in 16 GB; larger N untested here (the desktop went to 32 768 on
  32 GB).
- **CPU baseline is lower than champagne's 24-core**, so the laptop's *speedup*
  multipliers read higher than the 5090 box's even though its absolute GPU
  throughput is lower. The CPU update + serial-MLP rollout are the laptop's
  bottleneck, not core count, so the slower mobile single-thread drags the CPU
  column down and inflates the ratio — compare absolute GPU env/s, not the
  speedup column, across machines.

## Benchmark — full training iteration (CPU vs GPU)

A full training **iteration** = one **rollout** (act in the sim to collect a
T-step batch of experience) **+** one **PPO update** (learn from that batch).
`examples/biped/iter_e2e_bench.rs` times one whole iteration each way —
**full CPU** = rapier rollout (CPU MLP + rayon physics) + CPU PPO update;
**full GPU** = nexus rollout (vortx policy) + GPU PPO update (the Stage-B path
verified in `gpu_update_check`) — over a T=32 rollout and a 5-epoch ×
16-minibatch update. Same MJCF model, same MDP, same N envs, same net shapes
(actor `[43,256,256,128,12]`, critic `[49,512,256,128,1]`). Throughput is
`N·T / iteration_time` in **env-control-steps/second**; the **Isaac** column is
NVIDIA Isaac Lab + PhysX 5 on the same 5090 (same task
`Velocity-LeRobot-NoArms-v0`), total `Computation` (collection + learning) — the
production reference for how fast this *should* go.

The **RTX 4090 laptop** column is the independent laptop reproduction (16 GB,
150 W mobile part), placed next to the 5090 for a direct GPU-to-GPU read.

The GPU update is **GPU-resident** (the batch is normalized + uploaded once, each
minibatch is an on-GPU column gather), uses a **tiled GEMM**, and the GEMM inner
loop is **vec4** (4-wide FMA) — all verified bit-exact against the CPU update
(`gpu_update_check`):

| N envs | mac CPU | mac GPU | linux CPU (24-core) | linux GPU (RTX 5090) | linux GPU (RTX 4090 laptop) | Isaac/PhysX 5 (5090) |
|-------:|--------:|--------:|--------------------:|---------------------:|----------------------------:|---------------------:|
| 512    | 2.0 k   | **3.8 k** | 2.1 k             | **6.4 k**            | 6.4 k                       | —                    |
| 1 024  | 2.0 k   | **5.3 k** | 2.3 k             | **10.6 k**           | 9.8 k                       | —                    |
| 2 048  | 2.1 k   | **6.5 k** | 2.7 k             | **15.5 k**           | 11.5 k                      | 67 k                 |
| 4 096  | 2.1 k   | **7.7 k** | 3.1 k             | **19.9 k**           | 14.7 k                      | 126 k                |
| 8 192  | 2.1 k   | **8.0 k** | 3.6 k             | **23.1 k**           | 16.2 k                      | 201 k                |

Full GPU beats full CPU by ~**6.5×** on the 5090 (~3.5× on the mac). The
optimizations (GPU-resident batch + tiled GEMM + vec4) lifted the 5090 iteration
from 12.7 k → **23.1 k env/s** at N = 8 192. **The Isaac gap is now ~6–9× (was
~16×).** Two hardware notes worth recording. (1) The vec4 **inner-loop FMA** is a ~12%
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
cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- <num_envs> 32 5 16
```

### Benchmark — full training iteration, WebGPU vs hybrid native-CUDA (cuda-oxide), RTX 5090

The whole stack runs on **khal** (dimforge's GPU backend abstraction), default
backend **WebGPU/Vulkan** (rust-gpu → SPIR-V). khal also has a **CUDA** backend,
but its shader→PTX compiler (`cargo-cuda` → `rustc_codegen_nvvm`) is wedged on an
unbuildable LLVM 7. We unblocked it by compiling vortx's kernels with
**[cuda-oxide](https://github.com/NVlabs/cuda-oxide)** (Rust → PTX, LLVM 21) and
loading the result through khal's `cudarc` backend — so the *same* vortx ops run
on **native CUDA**. The **entire `vortx-shaders` crate now compiles wholesale to
one cubin** — all 23 `#[spirv]` kernels (GEMM, op-assign, activations, reductions,
Adam, PPO grads, contiguous/repeat) straight from the *verbatim* shader source,
via a `khal_std` cuda-oxide backend feature plus a handful of cuda-oxide codegen
fixes (`&mut [T]` slice writes, `link_llvm_intrinsics` barriers/sregs, shared
memory, `ConstantIndex`). Every class is bit-exact on the RTX 5090: op-assign /
Adam / shared-mem reductions = 0.0, libdevice activations = 5.96e-8, tiled GEMM =
1.9e-7. (`exp`/ELU libdevice kernels are linked to a self-contained cubin via
libNVVM + nvJitLink and loaded through a patched khal loader.)

The honest end-to-end question is the **full PPO iteration** (T=32 rollout +
5 epochs × 16 minibatches of update), per env. **Physics is `nexus3d`, itself a
WebGPU engine — it cannot move to CUDA** — so the most that moves is the
**policy forward** (in the rollout) and the **PPO update** (GEMM-heavy backprop).
This "hybrid" keeps physics on WebGPU and runs the learnable compute on
cuda-oxide. Composed from measured components on the RTX 5090:

| N envs | all-WebGPU | hybrid (policy fwd + update on cuda-oxide) | speedup |
|-------:|-----------:|-------------------------------------------:|--------:|
| 4 096  | 20.1k env/s | **28.9k env/s**                           | **1.44×** |
| 8 192  | 23.1k env/s | **35.1k env/s**                           | **1.52×** |

Where the time goes (N = 8 192, ms / iteration):

| stage | all-WebGPU | hybrid | note |
|-------|-----------:|-------:|------|
| rollout (physics + policy forward) | 5 891 | 5 764 | physics 5 746 (immovable) + forward 144 → **11** |
| PPO update (GEMM backprop)         | 5 457 | **1 715** | **3.2×** — the real lever |
| **full iteration**                 | **11 348** | **7 479** | **1.52×** |

So the headline is **the update, not the forward**. The policy forward *is*
5.5–13× faster on cuda-oxide in isolation (0.34 vs 4.5 ms/step at N = 8 192;
14.4 TFLOPS vs ~1.1 — WebGPU is dispatch-bound, creating a bind group per
dispatch, which `cuLaunchKernel` avoids), but the rollout is **physics-bound**:
the forward is ~2% of it, so 13× there buys almost nothing end-to-end. The
update is half the iteration and ~all GEMM, so its **3.2×** is what lifts the
whole iteration to **1.5×**.

Caveats, stated plainly: the physics half stays WebGPU (it must); the cuda-oxide
update measures the dominant GEMM backprop (forward + the two backward GEMMs +
ELU/ELU-bwd + weight step) and omits the cheap PPO-specific gradient/reduction
kernels, so its speedup is a mild over-estimate; the per-step Vulkan↔CUDA obs
transfer is estimated negligible (~6 ms/iter; the deployed sampler already
round-trips means to the CPU). This is a research track; the all-WebGPU column
is the shipping pipeline.

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
