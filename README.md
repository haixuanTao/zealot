# zealot

A full whole-body-control training stack for humanoid robots — environment,
PPO trainer, and deployment — written entirely in Rust, on top of
[nexus](https://github.com/dimforge/nexus), dimforge's cross-platform GPU
physics engine.

[![The web demo: Unitree G1s walking the training terrain in the browser](docs/img/web-demo.png)](https://haixuantao.github.io/zealot/)

## Why it's built this way

**Every layer — physics, model, training loop, deployment — is Rust,
compiled to whatever GPU is available.**

- **The simulator is Rust on any GPU.** The nexus solver is compute shaders
  written via [Rust-GPU](https://rust-gpu.github.io/); the same source runs
  through WebGPU in a browser and through CUDA or Metal natively. That is why
  the demo above can be the *actual training environment*, not an animation.
- **The learning half too.** `zealot-rl` is the rsl_rl tier rewritten in
  Rust — model, autodiff, PPO, GAE, Adam — on the same portable GPU layer
  ([vortx](https://github.com/dimforge/vortx) /
  [khal](https://github.com/dimforge/khal)) the physics uses. CPU reference
  implementations verify every GPU kernel to float epsilon.
- **One source, three compilers.** Rust-GPU → SPIR-V,
  [cuda-oxide](https://github.com/NVlabs/cuda-oxide) → PTX, naga → MSL — no
  second implementation to keep in sync. The native-CUDA path (with cuTile
  tf32 GEMMs) runs 2.4–4.3× faster than WebGPU while staying bit-exact.
- **The control loop never leaves the GPU.** Observation assembly, policy
  GEMMs, PD-target scatter, and physics substeps are one chained GPU
  workload, in training and in the browser alike.
- **Policies must survive a different solver.** The same checkpoint runs
  sim2sim on rapier.js, MuJoCo (wasm and native), Genesis, and Isaac — and on
  the physical G1 via `deploy/`.

## What's different about the physics

Read from the solver source, not the marketing (details + file pointers in
[docs/explanation.md](docs/explanation.md)):

- **Generalized-coordinate dynamics on the GPU.** Each robot is a reduced-
  coordinate multibody: per 5 ms substep, GPU kernels run forward kinematics,
  body Jacobians, velocity propagation, a CRBA mass matrix, then a dense LU
  factor + solve in workgroup shared memory — the same algorithmic family as
  MuJoCo's CRB + factorization, batched across thousands of environments.
  Most GPU RL engines don't do articulated dynamics this way.
- **Soft-TGS constraint solver** (rapier's algorithm): substepping,
  warmstarting, graph-colored Gauss–Seidel, soft contacts parameterized by
  natural frequency/damping ratio (240 Hz/ζ=1 in training), speculative
  contacts, box friction. Contact/joint constraints re-linearize from live
  poses every substep.
- **Actuators modeled like hardware.** PD torque (τ = kp·Δq − kd·v, clamped
  to the real Unitree torque limits) is computed inside the dynamics kernel
  each substep, with optional per-env actuator delay; armature, implicit
  joint damping, and Coulomb friction-loss come from the official Unitree
  model values. At the real ankle gains a G1 cannot passively stand (MuJoCo
  reproduces this) — no gain inflation anywhere.
- **Honest speed, MuJoCo as referee.** ~0.85× Isaac/PhysX 5 at 2048 envs on
  the same TGS budget and GPU ([tables](docs/benchmarks.md), including why
  Genesis's headline numbers aren't iteration-equivalent) — and the same
  checkpoint steps through official MuJoCo on bit-identical terrain, in the
  browser and natively, as the standing cross-check.

## Getting started

Three steps — full walkthrough in
[docs/getting-started.md](docs/getting-started.md):

1. **Watch it walk** — the [live demo](https://haixuantao.github.io/zealot/)
   is the training env in your browser. Tap the ground, switch engines, load
   any published checkpoint.
2. **Train the humanoid** (needs
   [`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu); see
   [development.md](docs/development.md)):

   ```sh
   # RTX-class GPU: ~71 k env-steps/s at N=4096 on a 5090
   BIPED_ROBOT=g1_29dof_agile BIPED_CUTILE_GEMM=1 BIPED_TERRAIN=1 \
     cargo run --release --example biped_train_gpu \
     --features "gpu biped_gpu cutile" -- 50000 4096 my_policy.safetensors
   ```

3. **Watch *your* policy walk** — upload the checkpoint to Hugging Face and
   open `https://haixuantao.github.io/zealot/?ckpt=your-name/your-repo`.

## Documentation

[Diátaxis](https://diataxis.fr) set, hub at
[**haixuantao.github.io/zealot/doc**](https://haixuantao.github.io/zealot/doc/):

| | |
| --- | --- |
| **Getting started** | [docs/getting-started.md](docs/getting-started.md) — demo → training → your checkpoint |
| **How-to guides** | [building & development](docs/development.md) · [reproducing the benchmarks](docs/benchmarks.md) · [the demo site](website/README.md) |
| **Reference** | hosted rustdoc: [`zealot_env`](https://haixuantao.github.io/zealot/doc/zealot_env/) · [`zealot_rl`](https://haixuantao.github.io/zealot/doc/zealot_rl/) |
| **Explanation** | [docs/explanation.md](docs/explanation.md) — how it's built and why |

## Benchmarks

Full methodology and tables: [docs/benchmarks.md](docs/benchmarks.md). The
short version (same RTX 5090, sequential same-hour runs, Unitree G1): full
training iterations at 61 k / 71 k / 82 k env-steps/s for N = 2048/4096/8192 —
≈0.85× Isaac Lab/PhysX 5 at 2048 envs, the gap opening at large N — with the
WebGPU build currently ~4× behind native CUDA at scale.
