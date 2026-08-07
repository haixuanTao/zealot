#!/bin/bash
# Convenience launcher — equivalent to the per-method `cargo run` documented
# in the README (production config is the code default; cubin/toolkit paths
# come from .cargo/config.toml + ~/.cargo/config.toml [env]):
#   NVIDIA GPU:  cargo run --release --bin biped_train_gpu --features "gpu biped_gpu cutile"
#   otherwise:   cargo run --release --bin biped_train_gpu --features "gpu biped_gpu"
# Adds: backend auto-pick, arm-motion dataset autodetect, dated checkpoint name.
set -eo pipefail
cd "$(dirname "$0")/.."
if command -v nvidia-smi > /dev/null 2>&1; then FEATURES="gpu biped_gpu cutile"; else FEATURES="gpu biped_gpu"; fi
[ -d "$HOME/sonic-motions" ] && export BIPED_ARM_MOTION="$HOME/sonic-motions"
ITERS="${1:-50000}"; ENVS="${2:-4096}"
CKPT="${3:-$HOME/overnight/biped_$(date +%Y%m%d_%H%M).safetensors}"
mkdir -p "$(dirname "$CKPT")"
echo "[launch] $(date) features=$FEATURES branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
exec cargo run --release --bin biped_train_gpu --features "$FEATURES" -- "$ITERS" "$ENVS" "$CKPT"
