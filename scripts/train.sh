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
# Arm-motion playback dataset (machine-local path). Missing dataset is a HARD
# ERROR — a box without ~/sonic-motions used to silently train with frozen
# arms. To intentionally train without it: BIPED_ARM_MOTION=off scripts/train.sh
if [ "$BIPED_ARM_MOTION" = "off" ]; then
    unset BIPED_ARM_MOTION
    echo "[launch] arm-motion playback explicitly disabled (BIPED_ARM_MOTION=off)"
elif [ -z "$BIPED_ARM_MOTION" ]; then
    if [ -d "$HOME/sonic-motions" ]; then
        export BIPED_ARM_MOTION="$HOME/sonic-motions"
    else
        echo "ERROR: ~/sonic-motions not found — this box would train WITHOUT the" >&2
        echo "AMASS upper-body disturbance. rsync the dataset here, point" >&2
        echo "BIPED_ARM_MOTION at a clip dir, or opt out with BIPED_ARM_MOTION=off." >&2
        echo "See docs/arm-motion-dataset.md for how to get the data." >&2
        exit 1
    fi
fi
ITERS="${1:-50000}"
ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
mkdir -p "$(dirname "$CKPT")"
echo "[launch] $(date) backend=$KHAL_BACKEND branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
cargo run --release --bin biped_train_gpu --features "$FEATURES" \
    -- "$ITERS" "$ENVS" "$CKPT"
echo "[done] exit=$? $(date)"
