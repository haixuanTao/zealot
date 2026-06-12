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
  FULL CPU (rapier + CPU MLP + CPU update) :  68319 ms =   3.8 k env/s
  FULL GPU (nexus + vortx + GPU update)    :   7242 ms =  36.2 k env/s
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

> **Solver config (2026-06):** position-iterations = **4** + **explicit Coriolis**,
> matching Isaac Lab / PhysX 5. The **linux CPU**, **linux GPU (WebGPU)** and
> **linux GPU (native CUDA)** columns were all re-measured at this config — the
> native-CUDA biped now runs end-to-end (the contact-path codegen crash is fixed,
> nexus-cuda `29fac1e`), gather bit-exact (err 0.0).
>
> **Dispatch optimizations (2026-06):** the **native-CUDA** column now also
> includes a set of dispatch-overhead reductions, all bit-identical physics:
> (1) **CUDA-graph capture** of the physics rollout (`BIPED_CAPTURE=1`) — the
> radix-sort capture wall was a `ThreadCount`-vs-`Grid` one-liner; replay = one
> `cuGraphLaunch`, encode 80 ms→0.5 ms; (2) **fixed-grid dispatch** — now the
> **default on CUDA** (was `BIPED_FIXED_GRID=1`): replaces CUDA indirect dispatch
> (which does a full GPU drain — `synchronize()`+device→host read — per launch);
> measured as the *entire* host-encode cost, `physics_encode` **1114 ms→25 ms** at
> N=8192, leaving the step cleanly GPU-compute-bound. WebGPU keeps indirect (it's
> native there); `BIPED_FIXED_GRID=0/1` overrides; (3) **`max_colors` 8→4**;
> (4) **color-counter fusion** (~80 fewer `inc_color`/`reset_color` launches/step).
>
> **Per-env parallelism (2026-06):** profiling (a `KHAL_CUDA_PROFILE` per-kernel
> timer — Nsight is too old for Blackwell) found ~50% of GPU time in **three
> `threads(1)` "dispatch-arg" kernels** that each *serially scanned all N batches
> in one thread* to compute an indirect grid size (an O(N) anti-scaling term) —
> replaced with single-workgroup parallel reductions (**155× per call**,
> bit-identical). Then **`gpu_mb_finalize_contact_constraints`** (the per-contact
> LU back-solve) was made block-per-articulation (`threads(32)`, lane-split over
> the independent back-solves): **~11.6× on the kernel, +11% e2e @ N=8192**. The
> dominant kernel `gpu_mb_init_solve_joint_with_bias` (~37%) was then split into a
> serial metadata-discovery walk → a **parallel back-solve phase** (the back-solves
> are what's expensive; gating them inside the serial walk doesn't parallelize —
> SIMT lockstep runs one active lane per slot — so they're hoisted into a separate
> finalize-style `s += lanes` loop): **~4.5× on the kernel** (37%→12%). The same
> two patterns were then applied to the rest of the `threads(1)` multibody set:
> the **joint + contact PGS sweeps** (`remove_solve_joint`, `solve_contact`,
> `remove_solve_contact`) are now cooperative `threads(32)` — Gauss-Seidel across
> constraints stays serial, but each constraint's `J·v` reduction + DOF-apply is
> split across lanes with a `delta` broadcast + per-constraint barrier (joint PGS
> **~3.4×**; contact PGS ~1.3×, limited by reduction-barrier overhead at the
> biped's small contact count), and **`gpu_mb_integrate`** is lane-split over
> links (~1.9×). All bit-identical. These per-env wins lifted the native-CUDA
> column to **63.6 k @ N=8 192 / 61.0 k @ N=2 048** (~+41–45% over the
> dispatch-only column). The remaining gap to Isaac at large N is the
> fused-megakernel / per-articulation-FK gap (the solver lacks cooperative-launch /
> grid-sync — only workgroup barriers), and the now-largest single GPU kernel is
> the PPO `gemm_tiled` (~22%), not physics.
> The **†** column (**mac**) is
> the previous `position-iters = 8` setting and awaits re-measurement on that
> hardware. The **Isaac** column is PhysX's own solver and is config-independent.

| N envs | mac CPU† | mac GPU† | linux CPU (24-core) | linux GPU (RTX 5090, WebGPU) | linux GPU (RTX 5090, native CUDA) | Isaac/PhysX 5 (5090) |
|-------:|--------:|--------:|--------------------:|-----------------------------:|----------------------------------:|---------------------:|
| 512    | 2.0 k   | 3.8 k   | 2.2 k               | 17.2 k                       | **34.7 k**                        | —                    |
| 1 024  | 2.0 k   | 5.3 k   | 2.5 k               | 24.9 k                       | **49.7 k**                        | —                    |
| 2 048  | 2.1 k   | 6.5 k   | 2.9 k               | 31.8 k                       | **61.0 k**                        | 67 k                 |
| 4 096  | 2.1 k   | 7.7 k   | 3.4 k               | 34.7 k                       | **63.0 k**                        | 126 k                |
| 8 192  | 2.1 k   | 8.0 k   | 3.8 k               | 36.2 k                       | **63.6 k**                        | 201 k                |

The **native-CUDA** column is the *same* Rust stack compiled to PTX via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Rust→PTX, LLVM 21) — no WebGPU
at all, both the vortx tensor ops and the `nexus3d` physics, straight from the
*verbatim* `#[spirv]` shader source, bit-exact vs WebGPU (boxes-physics pose
fingerprint identical; full biped iteration gather err 0.0 — the contact-path
codegen crash is fixed, nexus-cuda `29fac1e`). It's now **~1.8–2× over WebGPU
across the sweep** (up to ~2× on the smaller batches; was ~1.4× before the
2026-06 per-env-parallelism work). Two levers compound: (1) the GEMM-heavy PPO
update (~3× via `cuLaunchKernel` — no per-dispatch bind groups, higher GEMM
throughput), and (2) the per-env optimizations land harder on CUDA — CUDA-graph
capture (effective only on CUDA) lets the cooperative-kernel physics wins show,
and fixed-grid dispatch is CUDA-only (WebGPU keeps its native indirect dispatch).
Getting there took ~12 general cuda-oxide codegen fixes plus two khal↔cuda-oxide
ABI fixes (push element-count not byte-length for slice kernel args; pass a
shader's `&0` offset by value to dodge a null-deref that DCE'd a whole kernel).

Full GPU beats full CPU by ~**16.7×** on the 5090 (native CUDA; ~9.5× on WebGPU,
~3.8× on the mac). The optimizations (GPU-resident batch + tiled GEMM + vec4)
lifted the 5090 WebGPU iteration from 12.7 k → 23.1 k env/s at N = 8 192, and the
Isaac-matching solver config (position-iters 8 → 4 + explicit Coriolis) lifted it
further to **29.8 k** (WebGPU) / **41.5 k** (native CUDA). The dispatch +
per-env-parallelism optimizations above are mostly **backend-shared** (the
cooperative kernels live in the verbatim `#[spirv]` source, so they compile for
both SPIR-V and PTX), so they lifted **both** columns again: native CUDA to
**63.6 k @ N=8 192 / 61.0 k @ N=2 048** (34.7 k @ N=512) and WebGPU to **36.2 k @
N=8 192 / 31.8 k @ N=2 048** (17.2 k @ N=512, +22–47% across the sweep; the
fixed-grid default is CUDA-only, the rest is shared). **The Isaac gap (native
CUDA) is now ~3.2× at N = 8 192 (was 4.8×) and ~1.1× at N = 2 048 (was 1.9×)** —
the dispatch and per-env wins close it substantially, but the large-N gap is the
fused-solver (megakernel) / per-articulation-FK architecture, not dispatches. Two hardware notes worth recording. (1) The vec4 **inner-loop FMA** is a ~12%
win on the 5090 but **flat on the mac** — Metal auto-vectorizes the inner loop;
rust-gpu → SPIR-V → NVIDIA does not, so the explicit `Vec4` FMA matters there.
(2) vec4 **global loads** (a `gemm_tiled_vec4` with 128-bit loads, verified
bit-exact) add **0%** — because a *tiled* GEMM already amortizes global memory
(each element is loaded once into shared memory and reused), so it isn't
global-bandwidth-bound; the win was compute, not bandwidth. The remaining gap is
the rollout (~2.3× off PhysX) plus the update vs Isaac's fused-CUDA learning step
(~0.09 s) — the next lever is fewer dispatches, not vec4. None of it is new math.

> **Next levers (2026-06, in priority order).** After the dispatch + per-env-parallelism
> wins above, the GPU profile (via `KHAL_CUDA_PROFILE=1`) has shifted — physics is no
> longer the single dominant cost:
> 1. **`gemm_tiled` (PPO policy/value matmuls) is now the top GPU kernel (~22%).** The
>    next lever is the *learning* side: higher GEMM throughput and/or fusing the PPO
>    update, not more physics work.
> 2. **Block-per-articulation resident-state substep megakernel** — fuse FK → mass-matrix
>    → LU → constraint solve → integrate into one workgroup-scoped kernel that loads the
>    articulation's `M`/jacobians/velocities into **shared memory once**, killing both
>    the ~10–15 inter-kernel launches/substep and the repeated global round-trips of `M`.
>    This is workgroup-scoped (only workgroup barriers — *not* the grid-sync the stack
>    lacks), so it's feasible, and it's what closes most of the remaining large-N Isaac gap.
> 3. ~~Remaining `threads(1)` multibody kernels~~ **(mostly done)** — the joint + contact
>    PGS sweeps and `gpu_mb_integrate` are now cooperative `threads(32)` (see the per-env
>    note above); `gpu_mb_gravity_and_lu` / `compute_dynamics_pre` turned out to be
>    *already* cooperative (the heavy mass-matrix/ABA/LU work, not a `threads(1)` problem).
>    The one remaining `threads(1)` builder is **`gpu_mb_init_contact_constraints`** (~6 %):
>    its per-constraint jacobian fill needs the walk's per-contact geometry, so the
>    init_solve-style phase-split needs that geometry stored/recomputed — left as a TODO.
> 4. **Stage `M` in shared memory** for the finalize/init back-solves (each lane currently
>    re-reads `M` from global) — folds into lever 2.

Reproduce:

```sh
# WebGPU
cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- <num_envs> 32 5 16
# native CUDA (needs the cuda-oxide toolchain + embedded cubins; fixed-grid is the CUDA default)
BIPED_CUDA=1 cargo run --release --example iter_e2e_bench --features "gpu biped_gpu cuda_backend" -- <num_envs> 32 5 16
# with CUDA-graph capture of the rollout (the updated native-CUDA column):
BIPED_CAPTURE=1 BIPED_CUDA=1 cargo run --release --example iter_e2e_bench --features "gpu biped_gpu cuda_backend" -- <num_envs> 32 5 16
```

## End-to-end training (GPU PPO)

The iteration benchmark above times *one* step end-to-end (rollout + update) for
**throughput**. `examples/biped/biped_train_gpu.rs` runs the **full training
loop** to a learned policy with the same GPU-resident machinery — rollout forward
(`GpuPolicy`) **and** the PPO update (`GpuMlp` forward/backward/Adam + vortx `Ppo`
actor/value grads) on the GPU, with weights + Adam moments **persisted across
iterations** (the benchmark discards them; the trainer also advances Adam
bias-correction with a global step) and a velocity curriculum. Net +
hyperparameters mirror WBC-AGILE's T1 velocity policy (actor
`[obs,256,256,128,12]`, critic `[cobs,512,256,128,1]`, `init_noise_std=1.0`,
entropy 0.005, clip 0.2).

```sh
BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- <iters> <num_envs> <ckpt.safetensors>
```

Logs `iter / curr / step_rew / falls / torso_z / sec` per 10 iters and
checkpoints to safetensors (auto-resumes). **~9 s/iter @ N=2048** — ≈5× the
CPU-policy reference trainer `biped_train_nexus` (which runs the policy + PPO on
the CPU). This is the *training* counterpart to the throughput benchmark, and the
basis for a WBC-AGILE training A/B (matched config, matched iteration count).

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

Same config note as the iteration table: **linux** columns re-measured at
position-iters = 4 + explicit Coriolis; **†** columns are the previous
`position-iters = 8` config; **Isaac** is config-independent.

| N envs | mac CPU (rapier + MLP)† | mac GPU (nexus + vortx)† | linux CPU (24-core) | linux GPU (RTX 5090) | Isaac/PhysX 5 (5090) |
|-------:|-----------------------:|-------------------------:|--------------------:|---------------------:|---------------------:|
| 32     | **3.7 k**              | 0.5 k                    | **5.6 k**           | 1.6 k                | —                    |
| 128    | **4.1 k**              | 1.7 k                    | **6.9 k**           | 5.6 k                | —                    |
| 512    | 4.1 k                  | **4.7 k**                | 7.3 k               | **19.6 k**           | —                    |
| 1 024  | 4.1 k                  | **6.5 k**                | 7.5 k               | **34.5 k**           | —                    |
| 2 048  | 4.0 k                  | **8.6 k**                | 7.6 k               | **57.1 k**           | 73.8 k               |
| 4 096  | 3.9 k                  | **10.4 k**               | 7.5 k               | **76.1 k**           | 139 k                |
| 8 192  | 3.9 k                  | **10.8 k**               | 7.5 k               | **95.7 k**           | 220 k                |

> ⚠️ The **GPU rollout** numbers above predate the 2026-06 per-env-parallelism
> work and have **not** been refreshed: a re-measure showed much higher values
> (WebGPU ~238 k @ N=8 192) but they're confounded by `rollout_e2e_bench`'s
> per-step policy-forward + obs readback + CPU sampling — wall-clock there is
> dominated by backend-dependent readback/sync, not physics (native CUDA even
> measures *slower* than WebGPU at large N because of per-step `synchronize` +
> `clone_dtoh`). Treat this column as a *rough* physics+inference reference, not a
> clean rollout-throughput comparison. The iteration table above (full train step)
> is the reliable, refreshed measurement.

| machine                          | peak CPU | peak GPU | GPU > CPU at N ≈ | best GPU/CPU ratio |
|----------------------------------|---------:|---------:|-----------------:|-------------------:|
| mac (M-series + WebGPU)†         | 4.1 k    | 10.8 k   | ~500             | 2.77×              |
| linux (24-core + RTX 5090)       | 7.6 k    | 95.7 k   | ~200             | 12.8×              |

What this says in practice:

- Below the crossover (N ≈ 500 on the mac, **N ≈ 200 on the 5090**) the full-CPU
  path wins: rapier physics is cheap at small batch and the GPU path pays a fixed
  per-step physics + policy cost (~70 ms on the mac, **~20 ms on the 5090** at the
  new solver config) that doesn't amortize yet.
- Past the crossover the full-GPU path pulls ahead — ~2.8× by N = 8 192 on the
  mac, **~12.8× on the 5090** (95.7 k vs 7.5 k env/s). The full-CPU throughput is
  flat (~4 k mac, ~7.5 k 24-core) — bottlenecked by the **serial** per-env CPU
  MLP forward, not the physics.
- The PPO *update* still runs on the CPU and isn't in this measurement — moving
  it to the GPU (Stage B) is what makes a full training iteration scale, not just
  the rollout.
- Versus the **Isaac/PhysX 5** reference, nexus+vortx is now ~1.3× behind at
  N = 2 048 and ~2.3× at N = 8 192 (95.7 k vs 220 k) — closed from ~2.7× / ~5×
  once the solver config matched Isaac's (position-iters 4 + explicit Coriolis).
  PhysX is still the ceiling and scales harder with batch (its curve keeps
  climbing past 4 k while nexus softens).

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
