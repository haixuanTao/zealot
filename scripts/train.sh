#!/bin/bash
# Launch biped_train_gpu. The full production config is now the CODE DEFAULT —
# robot g1_29dof_agile, terrain curriculum, AGILE DR, pushes, motor delay 0-4,
# obs history 5, mirror aug, contact sensing, all production reward weights.
# Every knob is still overridable via its BIPED_* env var; a bare run equals
# the old 50-variable invocation (verified: identical startup config echo).
#
# Usage: scripts/train.sh [iters] [envs] [ckpt.safetensors]
#
# Box requirements (see docs/train-on-5090.md):
#   ~/nexus_ptx/            sm_120 cubins (rebuild: scripts/build_cubins.sh)
#   ~/cuda-13-shim/         CUDA 13.2+ headers (build-time, cutile)
#   ~/cuda-13.3-tile/bin/   tileiras + ptxas 13.3 (runtime JIT, cutile)
set -eo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/cuda-13.3-tile/bin:$HOME/.cargo/bin:$PATH"
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D="$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin"
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS="$HOME/nexus_ptx/vortx_shaders.cubin"
export CUDA_TOOLKIT_PATH="$HOME/cuda-13-shim"
export KHAL_BACKEND=cuda
export NEXUS_SMALL_SORT=1
# Arm-motion playback dataset (machine-local path; unset = arms hold still).
[ -d "$HOME/sonic-motions" ] && export BIPED_ARM_MOTION="$HOME/sonic-motions"
ITERS="${1:-50000}"
ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
echo "[launch] $(date) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
cargo run --release --bin biped_train_gpu --features "gpu biped_gpu cutile" \
    -- "$ITERS" "$ENVS" "$CKPT"
echo "[done] exit=$? $(date)"
