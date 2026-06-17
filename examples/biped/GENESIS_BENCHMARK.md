# Genesis vs nexus — biped benchmark

Head-to-head of **Genesis** (genesis-world 1.1.2) against this project's **nexus**
GPU physics (and CPU rapier) on the *same* LeRobot biped, same sim config, same
4090 laptop. Two axes: full-rollout **throughput** and cross-engine **fidelity**.

Setup is identical to the Rust env (see `biped_genesis_common.py`): MJCF model,
physics dt 0.005, decimation 4 (50 Hz control), per-joint PD (kp/kd) + effort cap
+ armature, 12 leg DOFs. Full rollout = policy MLP `[43,256,256,128,12]` forward
→ Gaussian sample → PD control → 4× physics step → 43-dim obs → reward, per step.

## Throughput — env-control-steps/s (full rollout, RTX 4090 laptop)

| N envs | CPU rapier | nexus GPU | **Genesis GPU** | Genesis / nexus |
|-------:|-----------:|----------:|----------------:|----------------:|
|   512  |    4.4 k   |   15.9 k  |   **67.2 k**    |     4.2×        |
|  1024  |    4.4 k   |   17.7 k  |  **121.0 k**    |     6.8×        |
|  2048  |    4.4 k   |   22.5 k  |  **214.5 k**    |     9.5×        |
|  4096  |    4.4 k   |   26.1 k  |  **372.4 k**    |    14.3×        |
|  8192  |    4.4 k   |   27.5 k  |  **568.7 k**    |    20.7×        |

(nexus = `rollout_e2e_bench` native CUDA; Genesis = `bench_genesis_sweep.sh`.)

**Read:** nexus plateaus ~16–28 k env-ctrl/s — it's **host/dispatch-bound** (per-step
obs/reward readback + many small physics dispatches; see `biped_fps.rs` per-phase
split). Genesis scales near-linearly with N (GPU not saturated until high N),
reaching **569 k** at N=8192. Genesis is **4–21× faster**, gap widening with batch
size. This is the regime Genesis is built for (tens of thousands of envs, fully
on-GPU rollout).

## Throughput — FULL training iteration (rollout + PPO update)

One iteration = T=32 rollout + GAE + 5 epochs × 16 minibatches of PPO
backprop+Adam. nexus = `iter_e2e_bench` (native CUDA, GPU update via vortx);
Genesis = `bench_genesis_train.py` (matched config: same nets, clip 0.2,
γ0.99/λ0.95, lr 1e-3). env-control-steps/s = N·T / iter_time.

| N envs | nexus full-iter | **Genesis full-iter** | Genesis / nexus |
|-------:|----------------:|----------------------:|----------------:|
|   512  |     8.8 k       |    **46.0 k**         |     5.2×        |
|  1024  |     9.8 k       |    **82.2 k**         |     8.4×        |
|  2048  |    11.1 k       |   **151.8 k**         |    13.7×        |
|  4096  |    12.0 k       |   **267.8 k**         |    22.4×        |
|  8192  |    12.1 k       |   **381.6 k**         |    31.6×        |

**Read (and a correction):** I expected the engine-agnostic PPO update to *narrow*
the rollout gap. It does the opposite — the **full-iteration gap is wider** (5–32×
vs the rollout's 4–21×). The PPO update costs nexus proportionally *more*: adding
it drops nexus to ~44% of its rollout throughput at N=8192 (27.5→12.1 k) but
Genesis only to ~67% (569→382 k). So nexus's GPU PPO-update path (vortx) is itself
a bottleneck relative to plain PyTorch autograd+Adam — not just the rollout.
nexus full-iter plateaus hard at ~12 k env-ctrl/s; Genesis keeps scaling to 382 k.

## Fidelity — replay nexus joint targets through each engine

Replay a nexus policy's joint-target stream through Genesis / MuJoCo with matched
PD gains; measure when the torso drops <0.40 m and base XY divergence vs nexus.
(`cross_engine_eval_genesis.py` / `cross_engine_eval.py`; trajectory from
`biped_render_nexus`.)

| engine | torso fell at | XY div @1.0s | XY div @end |
|--------|--------------:|-------------:|------------:|
| nexus (ground truth) | survives at 0.44 m crouch | 0 | 0 |
| **Genesis**          | 0.38 s | 36.2 cm | 43.7 cm |
| MuJoCo (reference)   | 0.54 s | 69.5 cm | 70.9 cm |

**Read:** Genesis behaves like a faithful reference engine — it tracks the nexus
targets *closer* than MuJoCo does (36 cm vs 70 cm), and both reference engines
**reject** this policy (it's sim-specific — exploits rapier/nexus contact
behaviour, exactly what `sim2sim_xval.py` / `cross_engine_eval.py` are designed to
catch). So Genesis isn't "fast because it simulates a different/unstable robot";
it simulates the same robot at fidelity comparable to MuJoCo. The throughput
comparison above is therefore apples-to-apples.

> Caveat: the fidelity replay used a degenerate (crouch) nexus policy — the only
> kind currently available. A truly-walking policy would be a cleaner fidelity
> probe; the conclusion (Genesis ≈ MuJoCo as a reference) holds regardless.

## Reproduce

```bash
# Genesis (venv with genesis-world + CUDA torch):
examples/biped/bench_genesis_sweep.sh                          # throughput sweep
~/genesis-venv/bin/python examples/biped/cross_engine_eval_genesis.py /tmp/biped_rollout.json  # fidelity

# nexus side (native CUDA):
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
for N in 512 1024 2048 4096 8192; do BIPED_CUDA=1 \
  ./target/release/examples/rollout_e2e_bench $N 32; done
# trajectory dump for fidelity:
BIPED_CUDA=1 ./target/release/examples/biped_render_nexus 0 400 /tmp/biped_rollout.json <ckpt.safetensors>
```

## Caveats
- Contact models differ across engines; some trajectory divergence is expected —
  MuJoCo is the reference bar, not bit-exactness.
- Genesis sim_options: dt 0.005, 4 physics steps/control, RigidOptions iterations 8.
  Solver settings only approximately match across engines.
- nexus number is the **rollout** (no PPO update); a full training iter is lower.
- Genesis FPS here is the *rollout* path (its own obs/reward in torch); a real
  Genesis training loop (rsl_rl) adds the PPO update, as it would for nexus too.
