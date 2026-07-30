# SONIC-style whole-body motion tracking

This example keeps zealot's existing nexus G1 physics and CPU PPO implementation,
but replaces the locomotion command/task with a motion-conditioned whole-body
controller. The actor observes ten proprioceptive history frames and ten future
reference frames, then emits absolute position targets for the 25 live joints in
the solver-compatible G1 model.

It reads raw BONES-SEED G1 CSV files directly. Translation columns are converted
from centimetres to metres, Euler rotations and joints from degrees to radians,
the 29 source joints are mapped by name to the 25 live joints, and 120 Hz motion
is resampled to the 50 Hz control loop.

```bash
# Validate one CSV or recursively inspect a motion directory (no GPU required).
cargo run --example sonic_wbc --features biped_gpu -- \
  inspect-motion /path/to/BONES-SEED/G1

# Train with batched nexus physics and save a safetensors checkpoint.
BIPED_ROBOT=g1_29dof_agile cargo run --release \
  --example sonic_wbc --features biped_gpu -- \
  train /path/to/BONES-SEED/G1 100 32 /tmp/sonic_wbc.safetensors

# Real trainer: reuse the locomotion stack's GPU policy and GPU PPO path.
BIPED_ROBOT=g1_29dof_agile SONIC_MAX_MOTIONS=4096 \
  cargo run --release --example sonic_train_gpu --features "gpu biped_gpu" -- \
  /path/to/BONES-SEED/G1 5000 4096 /tmp/sonic_gpu.safetensors

# Deterministic evaluation and a skeleton rollout compatible with render_biped.py.
BIPED_ROBOT=g1_29dof_agile cargo run --release \
  --example sonic_wbc --features biped_gpu -- \
  eval /path/to/motion.csv /tmp/sonic_wbc.safetensors /tmp/sonic.json
python3 examples/biped/render_biped.py /tmp/sonic.json /tmp/sonic.mp4
```

`SONIC_MAX_MOTIONS` limits recursive training ingestion (default 64 for the
CPU debug trainer and 4096 for `sonic_train_gpu`). When the corpus is larger,
clips are sampled uniformly from the sorted corpus rather than taken from one
alphabetical/capture-date prefix. On WebGPU, the GPU trainer increases its PPO
minibatch count automatically at high environment counts to stay within the
per-dispatch limit; native CUDA keeps the locomotion pipeline's four
minibatches.

On a provisioned NVIDIA host, build with `cuda_backend` and select cuda-oxide:

```bash
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=/path/to/nexus_rbd_shaders3d.cubin \
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=/path/to/vortx_shaders.cubin \
  cargo build --release --example sonic_train_gpu \
  --features "gpu biped_gpu cuda_backend"

KHAL_BACKEND=cuda BIPED_ROBOT=g1_29dof_agile SONIC_MAX_MOTIONS=4096 \
  target/release/examples/sonic_train_gpu \
  /path/to/BONES-SEED/G1 5000 4096 /tmp/sonic_gpu.safetensors
```

## Multi-GPU training with NCCL

The optional `nccl` feature adds synchronous data parallelism without moving
the trainer into Python. It runs one `sonic_train_gpu` process per GPU. Every
rank owns an independent Nexus environment batch and PPO minibatch; after each
backward pass NCCL averages the actor, critic, and `log_std` gradients directly
in their CUDA buffers before the existing GPU Adam passes run. Advantage
moments, adaptive-KL decisions, and deferred observation-normalizer statistics
are also reduced so the replicas cannot silently diverge.

`num-envs` is **per GPU**. For example, 8 GPUs with `4096` environments produce
`8 * 4096 * 24 = 786432` rollout samples per iteration. This is data
parallelism, not model parallelism: every GPU holds a full copy of the small
actor and critic. The physics CUDA graph remains rank-local; NCCL collectives
are issued on the same CUDA stream between backpropagation and Adam and are not
captured into that graph.

Build once with NCCL enabled:

```bash
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=/path/to/nexus_rbd_shaders3d.cubin \
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=/path/to/vortx_shaders.cubin \
  cargo build --release --example sonic_train_gpu \
  --features "gpu biped_gpu cuda_backend nccl"
```

cudarc loads `libnccl.so.2` dynamically. If NCCL came from a Python wheel,
expose that wheel's library directory at runtime:

```bash
export LD_LIBRARY_PATH=/path/to/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export BIPED_ROBOT=g1_29dof_agile
export SONIC_MAX_MOTIONS=4096
GPUS_PER_NODE=8 examples/sonic/launch_nccl.sh \
  /path/to/BONES-SEED/G1 5000 4096 /shared/sonic_gpu.safetensors
```

For multiple nodes, run the same command on each node with common
`NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT`, and a distinct
zero-based `NODE_RANK`. `MASTER_ADDR:MASTER_PORT` must be reachable from every
node. The motion corpus and checkpoint must contain identical data on every
rank (normally via a shared filesystem). Standard NCCL controls such as
`NCCL_SOCKET_IFNAME` and `NCCL_DEBUG=INFO` remain available.

The launcher translates the conventional `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, and `MASTER_PORT` variables into one local GPU per process.
Those variables can also be supplied by a scheduler instead of using the
included launcher.

This is a focused reproduction of SONIC's core reference-tracking idea, not a
checkpoint-compatible implementation of its universal motion-token architecture.
`sonic_wbc` remains a small CPU-PPO debugging path. `sonic_train_gpu` keeps
policy inference, the actor/critic update, and Adam state on the GPU while
reusing the batched Nexus physics and checkpoint format from locomotion.
