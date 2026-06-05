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

### Benchmark — full training iteration (rollout + PPO update)

The rollout table above is only half a training step; the other half is the
**PPO update**. `examples/biped/iter_e2e_bench.rs` times one whole iteration each
way — **full CPU** = rapier rollout (CPU MLP + rayon physics) + CPU PPO update;
**full GPU** = nexus rollout (vortx policy) + GPU PPO update (the Stage-B path
verified in `gpu_update_check`) — over a T=32 rollout and a 5-epoch × 16-minibatch
update. Throughput is `N·T / iteration_time`, same env-control-steps/s unit, and
the **Isaac** column is PhysX 5's total `Computation` (collection + learning):

The GPU update is **GPU-resident** (the batch is normalized + uploaded once, each
minibatch is an on-GPU column gather), uses a **tiled GEMM**, and the GEMM inner
loop is **vec4** (4-wide FMA) — all verified bit-exact against the CPU update
(`gpu_update_check`):

| N envs | mac CPU | mac GPU | linux CPU (24-core) | linux GPU (RTX 5090) | Isaac/PhysX 5 (5090) |
|-------:|--------:|--------:|--------------------:|---------------------:|---------------------:|
| 512    | 2.0 k   | **3.8 k** | 2.1 k             | **6.4 k**            | —                    |
| 1 024  | 2.0 k   | **5.3 k** | 2.3 k             | **10.6 k**           | —                    |
| 2 048  | 2.1 k   | **6.5 k** | 2.7 k             | **15.5 k**           | 67 k                 |
| 4 096  | 2.1 k   | **7.7 k** | 3.1 k             | **19.9 k**           | 126 k                |
| 8 192  | 2.1 k   | **8.0 k** | 3.6 k             | **23.1 k**           | 201 k                |

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

### Benchmark — policy forward, CPU vs GPU (vortx)

The bench above measures *physics*. The other half of the rollout is the
**policy forward**: the original trainer ran the actor and critic as
`for e in 0..N { actor.mean(); critic.value() }` — N serial CPU MLP passes. The
vortx GPU policy replaces that with one batched GEMM-stack per net (GEMM → bias
→ ELU, linear output) on the same backend as the physics. Net shapes are the
deployed biped (actor `[43,256,256,128,12]`, critic `[49,512,256,128,1]`).

Two measurements (mac M-series, WebGPU, naïve GEMM). `policy_forward_bench`
times *just the compute*, tensors resident. `rollout_bench` times the **real
rollout path** as `biped_render_nexus` runs it — including the per-step GPU→CPU
`slow_read_vec` readback of means/values for CPU sampling — so it's the honest
end-to-end-of-the-forward number:

| N envs | CPU serial loop | GPU compute only | GPU + per-step readback (real) |
|-------:|----------------:|-----------------:|-------------------------------:|
| 1 024  | 183 ms/step     | —                | 11 ms/step — **16×**           |
| 4 096  | 727 ms/step     | 23 ms/step (32×) | 28 ms/step — **26×**           |

So the readback is real but modest — it knocks the isolated 32× down to ~26× at
deployed scale, **not** a regression. GPU output matches the CPU net to 4e-7.

Caveat on *training* throughput: this speeds up the **rollout**, but a full PPO
iteration is currently dominated by the **CPU update** (`ActorCritic::update`
over ~131 k samples/iter — unchanged by this work), so end-to-end iters/s won't
move ~26× until that update also moves to the GPU. The update path is built and
unit-verified but not yet wired (every new kernel matches the CPU `zealot-rl`
reference to ~1e-6 — ELU `elu_check`, batched backward `mlp_backward_batch`, PPO
gradients `ppo_grad_check`). The 5090 figures are TBD.

Reproduce:

```sh
cargo run --release --example policy_forward_bench --features "gpu biped_gpu"
cargo run --release --example rollout_bench --features "gpu biped_gpu" -- 4096 32
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
