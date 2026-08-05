#!/bin/bash
# Launch biped_train_gpu with the full production feature set (native CUDA +
# cuTile tf32 GEMMs). Usage: scripts/train.sh [iters] [envs] [ckpt.safetensors]
#
# Requirements on the box (see docs/train-on-5090.md):
#   ~/nexus_ptx/            sm_120 cubins (rebuild: scripts/build_cubins.sh)
#   ~/cuda-13-shim/         CUDA 13.2+ headers (BUILD-time, cutile feature)
#   ~/cuda-13.3-tile/bin/   tileiras + ptxas 13.3 (RUNTIME JIT, cutile)
set -eo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/cuda-13.3-tile/bin:$HOME/.cargo/bin:$PATH"
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D="$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin"
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS="$HOME/nexus_ptx/vortx_shaders.cubin"
export CUDA_TOOLKIT_PATH="$HOME/cuda-13-shim"
export KHAL_BACKEND=cuda
ITERS="${1:-50000}"
ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
echo "[launch] $(date) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
cargo run --release --example biped_train_gpu --features "gpu biped_gpu cutile" \
    -- "$ITERS" "$ENVS" "$CKPT"
echo "[done] exit=$? $(date)"
