#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: launch_nccl.sh <motion-directory> [iterations] [envs-per-gpu] [checkpoint]" >&2
  exit 2
fi

sonic_binary="${SONIC_BINARY:-target/release/examples/sonic_train_gpu}"
gpus_per_node="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
node_count="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
world_size=$((gpus_per_node * node_count))

if [[ ! -x "$sonic_binary" ]]; then
  echo "SONIC binary is not executable: $sonic_binary" >&2
  echo "Build it with: cargo build --release --example sonic_train_gpu --features 'gpu biped_gpu cuda_backend nccl'" >&2
  exit 2
fi

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for ((local_rank = 0; local_rank < gpus_per_node; local_rank++)); do
  rank=$((node_rank * gpus_per_node + local_rank))
  RANK="$rank" \
  LOCAL_RANK="$local_rank" \
  WORLD_SIZE="$world_size" \
  MASTER_ADDR="$master_addr" \
  MASTER_PORT="$master_port" \
  KHAL_BACKEND=cuda \
    "$sonic_binary" "$@" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
    cleanup
  fi
done
exit "$status"
