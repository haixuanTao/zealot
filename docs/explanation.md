# Explanation — how zealot is built, and why

The short version: **every layer of the robot-learning stack — physics,
model, training loop, deployment — is Rust compiled to whatever GPU is
available.** The rest of this page unpacks what that buys.

## 100% Rust simulator, on any GPU

Physics is [nexus](https://github.com/dimforge/nexus), dimforge's GPU
multiphysics engine: the whole solver is compute shaders written in Rust via
[Rust-GPU](https://rust-gpu.github.io/). The same source runs through WebGPU
in a browser and through CUDA or Metal natively — no Python, no CUDA C, no
per-backend rewrite. That is why the [live demo](https://haixuantao.github.io/zealot/)
can be the *actual training environment* rather than a canned animation.

## The learning half is Rust too

`zealot-rl` is the rsl_rl tier rewritten in Rust: model definition, autodiff,
PPO, GAE, Adam — not a Python front-end over a C++ core. It runs on
[vortx](https://github.com/dimforge/vortx) and
[khal](https://github.com/dimforge/khal), the same portable GPU layer nexus is
built on, so the learning half is no more platform-bound than the physics
half. The CPU implementations in `zealot-rl` are the *reference*: every GPU
kernel is verified against them to float epsilon.

## One source, three compilers

The kernels are written once and compiled three ways — Rust-GPU → SPIR-V for
WebGPU/Vulkan, [cuda-oxide](https://github.com/NVlabs/cuda-oxide) → PTX for
native CUDA, and naga → MSL for Metal — with no second implementation to keep
in sync. On an RTX 5090 the native-CUDA path runs 2.4–4.3× faster than WebGPU
while staying bit-exact against it. The hot PPO GEMMs additionally use
cuTile tf32 tensor cores.

## Workspace layout

| Crate | Role | Analogy |
| --- | --- | --- |
| `zealot-env` | Vectorized environment + MDP layer over nexus's batched `GpuPhysicsPipeline` (observations, actions, rewards, terminations, per-env reset). | Isaac Lab tier |
| `zealot-rl` | Policy network, autodiff, PPO. | rsl_rl tier |
| `zealot-obs-shaders` / `zealot-gpu-obs` | GPU observation assembly + action commit (the kernels that keep the browser demo's control loop GPU-resident). | — |
| `website/` | The [live demo site](https://haixuantao.github.io/zealot/) (not a crate; Vite + React + the wasm demo builds). | — |

nexus itself provides the GPU physics + parallel environments (the Isaac Sim tier).

## Three layers people conflate

1. **Asset / scene description** — bodies, joints, collision, mass. Ships as
   MJCF in `assets/robots/`, generated from Unitree's official models and
   parsed by an in-repo subset loader. `BIPED_ROBOT=` selects at runtime.
2. **Engine model** — nexus consumes rapier types; the asset path is always
   `MJCF → rapier → GPU`.
3. **The MDP** — observations, rewards, terminations, resets. This is *code*
   in `zealot-env`, not a config file: reward terms with their gating logic
   are the actual research surface, and they version with the code.

## The GPU-resident control loop

In both training and the browser demo, a control step never crosses to the
CPU: observation assembly, the policy GEMMs, and PD-target scatter are GPU
kernels (`zealot-obs-shaders`, `zealot-gpu-obs`) chained with the physics
substeps. The CPU's only steady-state job is deciding how many steps to run
and reading back poses for rendering — pipelined, never fenced.

## What's different about the physics (vs MuJoCo, PhysX, Genesis)

This section is derived from the solver source; file pointers are to the
nexus tree (`src_rbd/`, `src_rbd_shaders/`).

### Dynamics: generalized coordinates, CRBA + LU, per substep

Each robot is a reduced-coordinate multibody. Per 5 ms substep, on the GPU
(`dynamics/multibody/compute_dynamics_pre.rs`,
`gravity_and_lu.rs` — one workgroup per (robot, environment)):

1. forward kinematics and body Jacobians from the joint coordinates;
2. velocity propagation down the tree;
3. a **CRBA mass matrix** (chain-bounded, with optional implicit-Coriolis
   augmentation), armature and `damping·dt` folded onto the diagonal;
4. gravity + actuation assembled into the generalized force vector;
5. a **dense LU factor and solve** in workgroup shared memory → generalized
   accelerations.

This is the same algorithmic family as MuJoCo (CRB + factorization in
generalized coordinates — MuJoCo factors sparse L᾿DL, nexus a dense tile,
bounded at 64 DOFs/robot). The difference is where it runs: batched as GPU
kernels across thousands of environments, from one Rust source, on WebGPU /
Metal / CUDA. PhysX 5's reduced-coordinate articulations are the closest
GPU analog; MJX gets to GPUs by XLA-compiling MuJoCo with its own
constraint-model restrictions; most other GPU RL engines approximate
articulations as rigid bodies with stiff joints.

### Constraints: Soft-TGS (rapier's algorithm)

`dynamics/solver.rs` states it directly: "Uses the Soft-TGS algorithm (as in
Rapier)". Substepping with warmstarting; graph-colored Gauss–Seidel sweeps
(parallel within a color, fused single-dispatch variant for small batches);
a bias pass and a no-bias stabilization pass per substep. Contacts are
**soft**, parameterized by natural frequency and damping ratio
(`ConstraintSoftness` in `sim_params.rs`; training uses 240 Hz/ζ=1) with
CFM compliance, speculative contact handling, a max-corrective-velocity
clamp, and **box friction** (independent per-tangent ±μ·N clamp — the
circular-cone clamp is noted in-source as a future refinement; a
friction-cone coupling attempt was tried and reverted after testing).
In the training configuration (`NEXUS_SUBSTEP_REFRESH=1`) joint and contact
constraints re-linearize from live poses every substep, and the
dynamics chain above reruns with them.

### Actuation: hardware-shaped, no gain inflation

The training env uses the **force-based motor model**: τ = kp·(q* − q) −
kd·q̇, clamped to the joint's real torque limit, computed *inside the
dynamics kernel* each substep (`gravity_and_lu.rs::apply_force_based_pd`) —
optionally reading a delayed target through a per-env actuator-delay ring.
Armature (reflected rotor inertia), implicit joint damping, and Coulomb
friction-loss come from Unitree's official model values (see
`zealot-env/src/robots/unitree_g1.rs`, which documents each choice against
the MJX playground's).

The gains are Unitree's real ones, and that is load-bearing: at the real
ankle gains a G1 **cannot passively stand** — they are far below the m·g·L
gravity stiffness, and MuJoCo reproduces the same result. Stability at 5 ms
timesteps comes from the per-substep refresh + soft-contact structure
above, not from the quiet gain multiplication common in RL configs. What
trains is what deploys.

### Throughput, with the caveats attached

On one RTX 5090 at the same TGS budget (4 substeps × 8 position
iterations), zealot runs ~0.85× of Isaac/PhysX 5 at 2048 envs, the gap
opening at larger batches (the large-N megakernel lever is tracked in
[benchmarks.md](benchmarks.md)). Genesis posts larger headline numbers by
integrating 2×10 ms strides per control step against zealot's 4×5 ms
substeps — per integration step it is roughly engine parity. Environments
are batch-interleaved SoA buffers (batch-minor, Genesis-style layout) so
one dispatch covers every environment; adding robots adds no dispatches.

## A policy is only trustworthy if it survives a different solver

The same checkpoint runs sim2sim on rapier.js and on the official MuJoCo
WebAssembly build — the reference engine — walking bit-identical terrain (the
demo's JS terrain generator is a bit-faithful port of the Rust one, verified
to 2e-10). Where the engines agree, believe the policy; where they diverge,
you have found either an engine artifact being exploited or a genuine
robustness gap. The same discipline extends natively to Genesis and Isaac
harnesses, and to the real G1 via the deployment stack in `deploy/`.
