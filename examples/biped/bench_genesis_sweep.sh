#!/bin/bash
# Genesis full-rollout throughput sweep (one fresh process per N — Genesis can't
# cleanly re-init/rebuild in-process). Pair with the nexus side:
#   rollout_e2e_bench (native CUDA) over the same N for the head-to-head table.
#
# Usage: examples/biped/bench_genesis_sweep.sh
set -eo pipefail
VENV=${GENESIS_VENV:-$HOME/genesis-venv}
HERE="$(cd "$(dirname "$0")" && pwd)"
T=${T:-32}
echo "=== Genesis full-rollout throughput (RTX 4090, T=$T) ==="
for N in 512 1024 2048 4096 8192; do
  "$VENV/bin/python" "$HERE/bench_genesis_rollout.py" "$N" "$T" 2>/dev/null | grep GENESIS_RESULT
done
