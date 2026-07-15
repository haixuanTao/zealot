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
- **Real robots (G1/H2):** ship as **MJCF** assets in `assets/robots/`, parsed by the
  in-repo subset loader (`examples/biped/biped_mjcf.rs`) and generated from Unitree's
  official models by `tools/convert_unitree_biped.py`. Select at runtime with
  `BIPED_ROBOT=lerobot|g1|g1_29dof|h2plus` — 12-DOF legs-only variants, plus a
  full-body 29-joint G1 whose wrists are welded to fit the solver's 32-DOF cap.
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

The working core is the **biped stack** in `examples/biped/`: batched nexus GPU
envs (`biped_env_nexus.rs`), the GPU-resident PPO trainer (`biped_train_gpu.rs`,
see below), and multi-robot assets selected with `BIPED_ROBOT=lerobot` (default)
| `g1` | `g1_29dof` | `h2plus` (robot table in `zealot-env/src/robots/`). The
learning-stack decision landed on **vortx + hand-rolled backprop**, with the hot
PPO GEMMs since moved to **cuTile tf32 tensor cores** (`BIPED_CUTILE_GEMM=1`,
kernels bit-checked in `examples/biped/cutile_gemm.rs`). The pendulum examples
remain the gentle introduction; `zealot-rl` carries the CPU reference
implementations the GPU path is verified against.

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
  FULL CPU (rapier + CPU MLP + CPU update) :  68545 ms =   3.8 k env/s
  FULL GPU (nexus + vortx + GPU update)    :  11256 ms =  23.3 k env/s
```

(That's the WebGPU path; the native-CUDA + cuTile build reaches **99.5 k** at the
same N — see the table + reproduce block below.)

Sweep N by re-running with `2048` / `4096` / `8192`. For **rollout-only**
throughput (no PPO update, matches the rollout table below): `cargo run --release
--example rollout_e2e_bench --features "gpu biped_gpu" -- 8192`. First build is
slow (shader compile); the toolchain is cached afterward (~16 s rebuilds).

## Benchmark — full training iteration (rollout + PPO update)

A full training step is a **rollout** plus a **PPO update**. `examples/biped/iter_e2e_bench.rs` times one whole iteration each
way — **full CPU** = rapier rollout (CPU MLP + rayon physics) + CPU PPO update;
**full GPU** = nexus rollout (vortx policy) + GPU PPO update (the Stage-B path
verified in `gpu_update_check`) — over a T=32 rollout and a 5-epoch × 16-minibatch
update. Throughput is `N·T / iteration_time`, same env-control-steps/s unit. The
**Isaac** column is not a synthetic benchmark: it's NVIDIA's
[WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) training pipeline
(Isaac-Lab/PhysX-5-based) running **its own LeRobot no-arms velocity task**
(`velocity_lerobot_no_arms`, rsl_rl runner) on the same 5090 — the reported
number is rsl_rl's total `Computation` (collection + learning), the same unit.
Being Isaac-Lab-based, WBC-AGILE's throughput ≈ the Isaac Lab engine ceiling, so
this column doubles as both the production-pipeline proxy and the PhysX
reference:

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
> hardware. The **Isaac** column is PhysX's own solver (via WBC-AGILE's task, see
> above) and is config-independent.
>
> **Table refresh (2026-07, LeRobot biped, position-iters 4):** the **linux CPU /
> WebGPU / native CUDA** columns below were re-measured on the current stack.
> `iter_e2e_bench` now runs the same **cuTile tf32** update + rollout forward as
> the trainer (build with `--features cutile`, run with `BIPED_CUTILE_GEMM=1`;
> cuTile's one-time per-shape JIT — ~9 s update, ~2 s forward — is warmed
> *outside* the timers, since a training run pays it once per process). Native
> CUDA jumped to **79.8 k @ N=2 048 / 99.5 k @ N=8 192** (+31%/+56% over the
> 2026-06 column) and **no longer plateaus at large N** — the old flat line was
> the update, not the physics. That puts the Isaac gap at **~2.0× @ N=8 192**
> (was 3.2×) and **ahead of the table's Isaac number at N=2 048**. Two
> regressions were found and are recorded rather than hidden: (1) the **WebGPU
> column dropped** (36.2 k → 23.3 k @ N=8 192) — cuTile is CUDA-only so WebGPU
> still rides the vortx GEMM update, which got slower somewhere in the
> cuTile/nvptx refactor window (root cause unbisected; the CUDA-side vortx
> fallback shows the same disease ~15× worse per call); (2) **CUDA-graph replay
> is currently 4–7× slower than eager** on this driver state (`BIPED_CAPTURE=1`
> / the update's default capture path; measured directly, un-profiled, and
> invisible under nsys) — the bench grew a `BIPED_UPD_GRAPH=0` escape hatch and
> the CUDA column above is eager.
>
> **G1 cross-sim check (2026-07):** a strictly-sequential, same-hour run on the same
> 5090, every sim driving the Unitree G1 (full training env-steps/s, `biped_train_gpu`):
> zealot 12-DOF **61 k / 71 k / 82 k** @ N=2 048/4 096/8 192 vs Isaac Lab full-body G1
> (stock `Isaac-Velocity-Flat-G1-v0` + rsl_rl — not the WBC-AGILE task used in the
> LeRobot table above)
> 72 k / 115 k / 180 k, MJX full-body 76.5 k / 89 k / 97.7 k, and Genesis 12-DOF
> 342 k / 622 k / 963 k. The Genesis headline is not iteration-equivalent — it
> integrates 2×10 ms strides per 20 ms control step vs zealot's 4×5 ms substeps
> (each with 8 TGS iterations), so per integration step it's roughly engine parity.
> Isaac's lead, by contrast, is real: PhysX 5 runs the *same* TGS budget on the G1
> (4 steps × 8 position iterations, plus velocity iterations). At 2 048 envs zealot
> is ~0.85× of Isaac; the gap opens with batch (2.2× at 8 192, where zealot
> plateaus) — that large-N slope is the megakernel lever below.

| N envs | mac CPU† | mac GPU† | linux CPU (24-core) | linux GPU (RTX 5090, WebGPU) | linux GPU (RTX 5090, native CUDA + cuTile) | Isaac/PhysX 5 (5090) |
|-------:|--------:|--------:|--------------------:|-----------------------------:|----------------------------------:|---------------------:|
| 512    | 2.0 k   | 3.8 k   | 2.2 k               | 17.0 k                       | **40.7 k**                        | —                    |
| 1 024  | 2.0 k   | 5.3 k   | 2.5 k               | 20.7 k                       | **62.1 k**                        | —                    |
| 2 048  | 2.1 k   | 6.5 k   | 2.9 k               | 22.5 k                       | **79.8 k**                        | 67 k                 |
| 4 096  | 2.1 k   | 7.7 k   | 3.2 k               | 22.8 k                       | **91.4 k**                        | 126 k                |
| 8 192  | 2.1 k   | 8.0 k   | 3.7 k               | 23.3 k                       | **99.5 k**                        | 201 k                |

The **native-CUDA** column is the *same* Rust stack compiled to PTX via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Rust→PTX, LLVM 21) — no WebGPU
at all, both the vortx tensor ops and the `nexus3d` physics, straight from the
*verbatim* `#[spirv]` shader source, bit-exact vs WebGPU (boxes-physics pose
fingerprint identical; full biped iteration gather err 0.0 — the contact-path
codegen crash is fixed, nexus-cuda `29fac1e`). With the 2026-07 cuTile update
it's now **~2.4–4.3× over WebGPU across the sweep**. Three levers compound:
(1) the PPO update and rollout forward on **cuTile tf32 tensor cores**
(CUDA-only), (2) lower dispatch cost via `cuLaunchKernel` — no per-dispatch
bind groups, and (3) fixed-grid dispatch is CUDA-only (WebGPU keeps its native
indirect dispatch).
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
CUDA) is now ~2.0× at N = 8 192 (was 4.8×, then 3.2×) and zealot is ~1.2×
AHEAD at N = 2 048** after the 2026-07 cuTile refresh — what remains at large N
is the fused-solver (megakernel) / per-articulation-FK architecture, not
dispatches or the update. Two hardware notes worth recording. (1) The vec4 **inner-loop FMA** is a ~12%
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
> 1. ~~`gemm_tiled` (PPO policy/value matmuls), the top GPU kernel (~22%)~~ **(done,
>    2026-07)** — the PPO update's GEMMs now run on **cuTile tf32 tensor cores**
>    (`BIPED_CUTILE_GEMM=1`, `cutile` feature: ~90× the vortx GEMM on the PPO shapes,
>    split-K for the weight grads, fused bias+ELU forward, row-sum bias grads —
>    update 0.23 s → 0.06 s @ N=2 048). The rollout's policy forward rides the same
>    tf32 path. Same month, the *host* side of the iteration was fixed too: GAE /
>    bootstrap / mirror-aug / normalize were 8 192 serial CPU value-net forwards —
>    now rayon-parallel, bit-identical, 0.57 s → 0.04 s @ N=8 192.
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
>
> **DOF scaling — a negative result worth recording (2026-07).** On full-body robots
> the solver's **dense n×n LU** is the wall: the factor is O(n³), the per-substep
> limit/contact back-solves are O(n²) each (`init_solve_joint_with_bias` alone was 28%
> of GPU at 31 nv), the full-body G1 runs at only ~0.65× the 12-DOF rate, and
> `MAX_MB_DOFS = 32` hard-caps what fits (hence the welded wrists). The textbook fix —
> a MuJoCo-style **tree-sparse LᵀDL** (factor O(n·depth²), every solve O(n·depth)) —
> was built, shipped shader-only, and benchmarked: it wins single-env but **loses
> 10–15% at training batch scale** and leaves the DOF penalty unchanged. Cause: the
> sparse factor/solves run serially on one lane while the dense LU spreads its larger
> work across all 32 warp lanes — at 18–31 nv that's *more serial steps per warp*
> despite far fewer FLOPs, and batch scale is throughput-bound
> ([dimforge/nexus#15](https://github.com/dimforge/nexus/pull/15), closed with the
> write-up). The salvage is an **env-per-lane layout** (one warp = 32 envs, the serial
> sparse code becomes the per-lane inner loop, MJX/Isaac-style) — which folds into the
> megakernel work in lever 2 and is what eventually dissolves the 32-DOF cap.

Reproduce:

```sh
# WebGPU
cargo run --release --example iter_e2e_bench --features "gpu biped_gpu" -- <num_envs> 32 5 16
# native CUDA + cuTile tf32 (the current native-CUDA column; needs the cuda-oxide
# toolchain + embedded cubins; fixed-grid is the CUDA default)
BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 cargo run --release --example iter_e2e_bench \
    --features "gpu biped_gpu cuda_backend cutile" -- <num_envs> 32 5 16
# BIPED_CAPTURE=1 (rollout CUDA-graph capture) and the update's default graph path
# are currently pathological on this driver (replay 4–7× slower than eager, see the
# note above); BIPED_UPD_GRAPH=0 forces the eager update without per-launch syncs.
```

## Benchmark — G1 vs the production pipeline (WBC-AGILE, 2026-07)

The LeRobot table above compares against WBC-AGILE's *lightest* task. This one
compares the **Unitree G1** against the pipeline NVIDIA actually ships for it:
[WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)'s `Velocity-G1-History-v0`
— full 29-DOF G1, delayed DC-motor actuation model, 5-step observation history,
contact sensors, the full manager/reward stack, PhysX TGS at 8 position + 4
velocity iterations, 200 Hz physics / 50 Hz control (the same 4×5 ms structure
zealot uses). Same 5090, same hour, strictly sequential; WBC-AGILE numbers are
steady-state rsl_rl `Computation` (collection + learning); zealot is
`iter_e2e_bench` (native CUDA + cuTile, T=32, 5e×16mb) with `BIPED_ROBOT=g1`
(12-DOF legs-only, wrists/waist fused):

| N envs | zealot 12-DOF (iters 4) | zealot 12-DOF (iters 8) | zealot full-body, 31 nv (iters 8) | WBC-AGILE full-body, 35 nv |
|-------:|------------------------:|------------------------:|----------------------------------:|---------------------------:|
| 2 048  | 70.4 k                  | 51.1 k                  | **35.4 k**                        | 20.6 k                     |
| 4 096  | 81.3 k                  | 60.4 k                  | **42.9 k**                        | 32.3 k                     |
| 8 192  | 89.5 k                  | 68.5 k                  | **56.1 k**                        | 47.4 k                     |

The bold column is the par-to-par comparison — full body vs full body at the
same TGS budget: zealot is **1.7× / 1.3× / 1.2× ahead**. Residual asymmetries,
one per side: zealot's full-body model welds the wrists (31 nv vs AGILE's true
29-joint 35 nv, the `MAX_MB_DOFS = 32` cap) and its task layer is lighter — no
actuator-delay model or observation history (both hold the upper body with PD
and actuate the legs). Against WBC-AGILE: its own task overhead dominates its
profile — the *stock* `Isaac-Velocity-Flat-G1-v0` on the same box does
72 k / 115 k / 180 k, so the production stack runs **~3.5–3.8× below its
engine's ceiling**, and a GPU-utilization probe during its steady iterations
shows why: 75% "utilization" at only **156 W of the 600 W budget** — the
launch-bound signature of many tiny manager-framework kernels with Python gaps,
not extra simulation compute. A production-grade task costs PhysX most of its
headline throughput, while zealot's task layer (rewards/obs on rayon, cuTile
policy) keeps ~all of it — that, not the engine, is the actual competitive gap.

Reproduce (zealot side; WBC-AGILE side is `scripts/train.py
--task Velocity-G1-History-v0 --headless --num_envs N --max_iterations 15` from
its repo):

```sh
BIPED_ROBOT=g1 BIPED_SOLVER_ITERS=8 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    cargo run --release --example iter_e2e_bench \
    --features "gpu biped_gpu cuda_backend cutile" -- <num_envs> 32 5 16
```

## End-to-end training (GPU PPO) — **default trainer**

The iteration benchmark above times *one* step end-to-end (rollout + update) for
**throughput**. `examples/biped/biped_train_gpu.rs` is the **default end-to-end
trainer**: it runs the **full training loop** to a learned policy with the same
GPU-resident machinery — rollout forward (`GpuPolicy`) **and** the PPO update
(`GpuMlp` forward/backward/Adam + vortx `Ppo` actor/value grads) on the GPU, with
weights + Adam moments **persisted across iterations** (the benchmark discards
them; the trainer also advances Adam bias-correction with a global step). It also
carries the full feature set — stand-before-walk + torque curricula,
time-limit bootstrapping, adaptive-KL LR, log_std re-flooring, L/R mirror
augmentation, and per-component reward logging. Net + hyperparameters mirror
WBC-AGILE's T1 velocity policy (actor `[obs,256,256,128,12]`, critic
`[cobs,512,256,128,1]`, `init_noise_std=1.0`, entropy 0.01, clip 0.2).

```sh
# The fast path: native CUDA + cuTile tf32 GEMMs. Build WITH the `cutile` feature —
# without it BIPED_CUTILE_GEMM silently no-ops and the update falls back ~20× slower.
BIPED_ROBOT=g1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend cutile" -- <iters> <num_envs> <ckpt.safetensors>
```

Logs `iter / curr / step_rew / falls / torso_z / lr / kl / sec` per 10 iters
(plus a `[prof]` per-phase split and a `[rb]` per-component reward line) and
checkpoints to safetensors. **Auto-resumes** if the checkpoint file exists
(default `/tmp/biped_policy_gpu.safetensors`) — delete it for a fresh run, and
*always* before benchmarking or comparing iter-0 stats. As of 2026-07 (cuTile
tf32 update path + rayon-parallel GAE/bootstrap + one-launch small-batch sort)
the trainer sustains **~0.8 s/iter @ N=2 048** on the G1 12-DOF — **61 k / 71 k
/ 82 k env-steps/s @ N=2 048/4 096/8 192** — ≈8–10× the legacy CPU-policy
trainer `biped_train_nexus` (policy + PPO on the CPU), which is kept only as a
reference/fallback. `BIPED_GRAPH=1` (CUDA-graph capture of the training rollout)
is opt-in and under investigation: nsys shows replay is GPU-perfect (~7 ms/step
of launch bubbles eliminated) yet un-profiled wall-clock currently regresses. This is the
*training* counterpart to the throughput benchmark, and the basis for a WBC-AGILE
training A/B (matched config, matched iteration count).

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
**Isaac** column is NVIDIA's WBC-AGILE pipeline (Isaac Lab + PhysX 5) on the same
5090, running its LeRobot no-arms velocity task (`velocity_lerobot_no_arms`),
measured the same way — rsl_rl *collection* throughput
(`num_envs · 24 / collection_time`, i.e. physics + policy inference, no learning
step) — as the production reference for "how fast this *should* go".

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
