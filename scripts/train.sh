#!/bin/bash
# Convenience launcher — equivalent to the per-method `cargo run` documented
# in the README (production config is the code default; cubin/toolkit paths
# come from .cargo/config.toml + ~/.cargo/config.toml [env]):
#   NVIDIA GPU:  cargo run --release --bin biped_train_gpu --features "gpu biped_gpu cutile"
#   otherwise:   cargo run --release --bin biped_train_gpu --features "gpu biped_gpu"
# Adds: backend auto-pick, arm-motion dataset check, dated checkpoint name.
set -eo pipefail
cd "$(dirname "$0")/.."
if command -v nvidia-smi > /dev/null 2>&1; then FEATURES="gpu biped_gpu cutile"; else FEATURES="gpu biped_gpu"; fi
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
ITERS="${1:-50000}"; ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
mkdir -p "$(dirname "$CKPT")"
echo "[launch] $(date) features=$FEATURES branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
exec cargo run --release --bin biped_train_gpu --features "$FEATURES" -- "$ITERS" "$ENVS" "$CKPT"
