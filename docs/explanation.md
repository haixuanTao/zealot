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

GPU RL simulators usually buy throughput by weakening the dynamics —
rigid-body soups with stiff joints, frozen mass matrices, inflated actuator
gains. nexus keeps the dynamics and batches them:

- **Featherstone on the GPU, refreshed every substep.** The articulated-body
  mass matrix is rebuilt and LU-factored *per 5 ms substep*, per environment,
  as GPU kernels; joint and contact constraints re-linearize on the same
  cadence, with TGS iterations on top. This is the fidelity class of CPU
  MuJoCo — but batched across thousands of environments on whatever GPU is
  present.
- **Real gains are non-negotiable.** The training env runs Unitree's actual
  PD gains and torque limits. At the real ankle gains a G1 cannot passively
  stand — the gains are far below the m·g·L gravity stiffness, and MuJoCo
  reproduces the same result — so stability at 5 ms timesteps comes from
  solver structure (implicit PD through the constraint solver, per-substep
  refresh, contact stiffness tuned at 240 Hz), not from quietly multiplying
  the gains the way many RL configs do. What trains is what deploys.
- **Throughput with the caveats attached.** On one RTX 5090 with the same
  TGS budget (4 substeps × 8 position iterations), zealot runs ~0.85× of
  Isaac/PhysX 5 at 2048 envs, with the gap opening at larger batches;
  Genesis posts bigger headline numbers by integrating 2×10 ms strides per
  control step vs zealot's 4×5 ms substeps — per integration step it is
  roughly engine parity. Full methodology: [benchmarks.md](benchmarks.md).
- **And it runs where the others don't.** PhysX is CUDA; MJX is XLA; the
  same nexus solver runs in a browser tab, on Metal, and on CUDA from one
  Rust source — with the CUDA and WebGPU builds verified bit-exact against
  each other.

## A policy is only trustworthy if it survives a different solver

The same checkpoint runs sim2sim on rapier.js and on the official MuJoCo
WebAssembly build — the reference engine — walking bit-identical terrain (the
demo's JS terrain generator is a bit-faithful port of the Rust one, verified
to 2e-10). Where the engines agree, believe the policy; where they diverge,
you have found either an engine artifact being exploited or a genuine
robustness gap. The same discipline extends natively to Genesis and Isaac
harnesses, and to the real G1 via the deployment stack in `deploy/`.
