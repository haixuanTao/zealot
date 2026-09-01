# Ablation — could PyTorch replace the custom PPO update?

**Measured on this box:** RTX 5090 (sm_120), native CUDA + cuTile tf32 build,
G1 29-DOF, N=4096, T=24, 5 epochs × 4 minibatches (mirror augmentation doubles
the count to 8), actor `395→256→256→128→12`, critic `90→512→256→128→1`,
196 608 samples per iteration, 40 optimizer steps. torch 2.11.0+cu128.

Reproduce with `python bench/torch_ppo_update.py --transfer`.

## What is actually replaceable

The rollout is nexus GPU physics — torch cannot replace it, and it is **85 %**
of the iteration. What torch can replace is the *update*: policy/value forward,
PPO clipped loss, backward, Adam. So this ablation swaps that one phase and
holds everything else fixed.

## The update, head to head

| update implementation | ms / iteration | ms / optimizer step |
| --- | ---: | ---: |
| vortx GEMM (no `cutile` feature) | 2 360 | 59.0 |
| **zealot cuTile tf32** (current default) | **150** | 3.75 |
| torch, fp32 (tf32 off) | 105 | 2.64 |
| **torch, tf32** | **66.6** | 1.67 |
| torch, tf32 + `torch.compile` | 69.9 | 1.75 |

torch's tf32 update is **2.3× faster** than the cuTile path at the same shapes
and the same arithmetic class. `torch.compile` does not help — at these sizes
the update is launch-bound, not compute-bound, and compilation adds overhead.

## But end-to-end it barely moves

The iteration is rollout-dominated, so a 2.3× win on the update is a small win
overall:

| | current | torch update |
| --- | ---: | ---: |
| Rollout (nexus physics) | 850 ms | 850 ms |
| GAE + reset | 20 ms | 20 ms |
| Update | 190 ms | ~107 ms |
| Host round-trip (batch out/in) | — | 15 ms |
| **Iteration** | **1 000 ms** | **~892 ms** |
| env-steps/s | 98 k | ~110 k |

So: **~2.3× on the phase, ~1.12× on the trainer.** The 15 ms is measured — 391 MB
each way over pinned memory — and is the price of leaving the Rust/khal device
buffers to reach torch. Zero-copy interop (dlpack / CUDA IPC) would remove it
and get to ~877 ms, i.e. ~1.14×.

## What this ablation does *not* prove

- It times the update on synthetic tensors of the production shape, not a ported
  trainer. A real port has to move the batch out of the nexus env (measured
  above) or share device memory with it.
- Numerical parity is not checked. The Rust update is verified bit-exact against
  a scalar-CPU reference (`gpu_update_check`); the torch path here is not held to
  that, and tf32 accumulation differs between the two.
- It says nothing about the rest of the stack. The single-source-of-truth
  argument for the Rust update — one kernel language for physics *and* learning,
  same portable GPU layer, WebGPU/Metal builds from the same source — is not a
  throughput argument, and this measurement does not touch it.

## Where the time would actually be better spent

From the nsys trace of the same iteration: physics is 595 ms of the 747 ms of
GPU-busy time, and `gpu_mb_compute_dynamics_without_coriolis_pre` alone is
173 ms across 96 launches (1.8 ms each). Halving the update saves ~90 ms;
halving that one physics kernel saves ~86 ms — and the solver's dense-LU cost is
the documented O(n³) wall the repo already flags for full-body robots.
