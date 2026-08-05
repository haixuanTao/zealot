# zealot

A full whole-body-control training stack for humanoid robots — environment,
PPO trainer, and deployment — written entirely in Rust, on top of
[nexus](https://github.com/dimforge/nexus), dimforge's cross-platform GPU
physics engine. The engine core is portable WebGPU; training performance work
targets the native-CUDA (cuda-oxide) fast path.

## Live demo — [haixuantao.github.io/zealot](https://haixuantao.github.io/zealot/)

[![The web demo: Unitree G1s walking the training terrain in the browser](docs/img/web-demo.png)](https://haixuantao.github.io/zealot/)

The **training environment itself, compiled to wasm**: the same nexus GPU
physics the trainer runs (real Unitree PD gains, 5 ms substeps + per-substep
refresh, 50 Hz control) executing as WebGPU compute shaders in the browser,
with the whole control loop GPU-resident — observation assembly, the policy
GEMMs, and PD-target scatter never leave the GPU (`zealot-obs-shaders` +
`examples/biped/gpu_policy.rs`).

- **Three engines, one policy, one terrain** — the nexus (WebGPU) tab plus two
  sim2sim tabs stepping the identical checkpoint through
  [rapier.js](https://rapier.rs) and the official MuJoCo wasm build, on a
  bit-faithful JS port of the training terrain generator. Browser-grade
  sim2sim, no install.
- **Run any published checkpoint** — the Policy picker lists
  [`haixuantao/zealot-g1-locomotion`](https://huggingface.co/haixuantao/zealot-g1-locomotion),
  and pasting any Hugging Face handle (`owner/repo`), model-page URL, or direct
  `.safetensors` link loads that policy instead. Obs layout (45- vs 48-dim
  frames) and the matching gait clock are detected from the checkpoint, so
  pre- and post-gyro policies both walk.
- **Interactive** — terrain difficulty / roughness / slope and robot-count
  sliders; drive the robot with arrows/WASD or a gamepad; the URL carries the
  configuration, so a link like
  [`?ckpt=haixuantao/zealot-g1-locomotion&n=1`](https://haixuantao.github.io/zealot/?ckpt=haixuantao/zealot-g1-locomotion&n=1)
  is a shareable pointer at a specific policy.
- **Browser support** — the WebGPU tab needs Chrome (or another Chromium);
  Safari, Firefox and iOS automatically open on the CPU engines instead.
- Diagnostics for the curious: `?prof=1` prints a per-kernel GPU-time
  breakdown to the console; the HUD reports sim speed, pose-fence time and
  GPU-boundary counters.

The demo sources are `examples/biped/g1_web_demo.rs` (shared implementation)
behind the `g1_terrain_web` / `g1_web` examples; the site lives in
[`website/`](website/) (Vite + React, deployed to GitHub Pages —
see `website/README.md`).

## Workspace layout

| Crate | Role | Analogy |
| --- | --- | --- |
| `zealot-env` | Vectorized environment + MDP layer over nexus's batched `GpuPhysicsPipeline` (observations, actions, rewards, terminations, per-env reset). | Isaac Lab tier |
| `zealot-rl` | Policy network, autodiff, PPO. | rsl_rl tier |
| `zealot-obs-shaders` / `zealot-gpu-obs` | GPU observation assembly + action commit (the kernels that keep the browser demo's control loop GPU-resident). | — |
| `website/` | The [live demo site](https://haixuantao.github.io/zealot/) (not a crate; Vite + React + the wasm demo builds). | — |

nexus itself provides the GPU physics + parallel environments (the Isaac Sim tier).

## The stack in one paragraph

Scenes ship as MJCF assets in `assets/robots/` (generated from Unitree's
official models, parsed by the in-repo subset loader), selected at runtime
with `BIPED_ROBOT=lerobot|g1|g1_29dof_agile|h2plus`. The MDP — observations,
rewards, terminations — is Rust code in `zealot-env`, not a config file. The
working core is the biped stack in `examples/biped/`: batched nexus GPU envs
(`biped_env_nexus.rs`) and the GPU-resident PPO trainer (`biped_train_gpu.rs`),
with the hot PPO GEMMs on cuTile tf32 tensor cores (`BIPED_CUTILE_GEMM=1`).
`zealot-rl` carries the CPU reference implementations every GPU kernel is
verified against, and `examples/pendulum/` is the gentle introduction
([guide](examples/pendulum/README.md)).

## Train

```sh
# native CUDA + cuTile (RTX 5090): ~71 k env-steps/s at N=4096 (12-DOF; full
# training iteration = rollout + PPO update)
BIPED_ROBOT=g1_29dof_agile BIPED_CUTILE_GEMM=1 BIPED_TERRAIN=1 \
  cargo run --release --example biped_train_gpu --features "gpu biped_gpu cutile" -- 50000 4096 out.safetensors
```

Checkpoints publish to
[huggingface.co/haixuantao/zealot-g1-locomotion](https://huggingface.co/haixuantao/zealot-g1-locomotion)
and are directly loadable in the live demo. Sim2sim harnesses (MuJoCo,
Genesis, Isaac) live in `scripts/` and `examples/biped/`.

## Documentation

Organized as a [Diátaxis](https://diataxis.fr) set, hub at
[**haixuantao.github.io/zealot/doc**](https://haixuantao.github.io/zealot/doc/):

- **[Getting started](docs/getting-started.md)** — the [live demo](https://haixuantao.github.io/zealot/)
  is step one (nothing to install), then pendulum → humanoid training → your
  checkpoint back in the demo.
- **How-to guides** — [building & development](docs/development.md),
  [reproducing the benchmarks](docs/benchmarks.md), [the demo site](website/README.md).
- **Reference** — hosted rustdoc for
  [`zealot_env`](https://haixuantao.github.io/zealot/doc/zealot_env/) and
  [`zealot_rl`](https://haixuantao.github.io/zealot/doc/zealot_rl/), rebuilt on
  every site deploy.
- **[Explanation](docs/explanation.md)** — how the stack is built and why:
  100% Rust on any GPU, one source / three compilers, the GPU-resident loop,
  sim2sim as a trust discipline.

## Benchmarks

Full methodology, tables, and repro commands: [docs/benchmarks.md](docs/benchmarks.md).
The short version (same RTX 5090, sequential same-hour runs, Unitree G1):
full training iterations at 61 k / 71 k / 82 k env-steps/s for
N = 2048/4096/8192 — ≈0.85× Isaac Lab/PhysX 5 at 2048 envs, with the gap
opening at large N (the megakernel lever in the notes) — and the WebGPU
build currently ~4× behind native CUDA at scale (open vortx-GEMM
regression).

## Building & development

Toolchain setup (cargo-gpu, the native-CUDA cubin chain) and the repo's
hook/test conventions: [docs/development.md](docs/development.md).
