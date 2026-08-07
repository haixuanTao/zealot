#!/bin/bash
# Launch biped_train_gpu. The full production config is the CODE DEFAULT —
# robot g1_29dof_agile, terrain curriculum, AGILE DR, pushes, motor delay 0-4,
# obs history 5, mirror aug, contact sensing, grad clip, all reward weights.
# Every knob is still overridable via its BIPED_* env var.
#
# Backend is auto-detected: NVIDIA GPU -> native CUDA + cuTile tf32 GEMMs
# (needs ~/nexus_ptx cubins + ~/cuda-13-shim + ~/cuda-13.3-tile, see
# docs/train-on-5090.md); otherwise WebGPU (Metal on macOS, Vulkan on Linux —
# see docs/train-on-macos.md for the one-time toolchain setup).
#
# Usage: scripts/train.sh [iters] [envs] [ckpt.safetensors]
set -eo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
if command -v nvidia-smi > /dev/null 2>&1; then
    export PATH="$HOME/cuda-13.3-tile/bin:$PATH"
    export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D="$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin"
    export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS="$HOME/nexus_ptx/vortx_shaders.cubin"
    export CUDA_TOOLKIT_PATH="$HOME/cuda-13-shim"
    export KHAL_BACKEND=cuda
    export NEXUS_SMALL_SORT=1
    FEATURES="gpu biped_gpu cutile"
else
    export KHAL_BACKEND=webgpu
    FEATURES="gpu biped_gpu"
fi
# Arm-motion playback dataset (machine-local path; unset = arms hold still).
[ -d "$HOME/sonic-motions" ] && export BIPED_ARM_MOTION="$HOME/sonic-motions"
ITERS="${1:-50000}"
ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
mkdir -p "$(dirname "$CKPT")"
echo "[launch] $(date) backend=$KHAL_BACKEND branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
cargo run --release --bin biped_train_gpu --features "$FEATURES" \
    -- "$ITERS" "$ENVS" "$CKPT"
echo "[done] exit=$? $(date)"
