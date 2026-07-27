#!/bin/bash
# Remaining published bench scenarios (complements perf_bench.sh):
#  - CUDA 12-DOF G1 columns (solver-iters 8 and 4)
#  - CUDA 29-DOF +realism FLAT (no terrain) column
#  - rollout-only CPU vs GPU (rollout_e2e_bench)
#  - true-training throughput (biped_train_gpu, BIPED_ROBOT=g1)
#  - nexus-side bench_batch_sweep3 (boxes + chain)
# Usage: scripts/perf_bench2.sh <label>
set -eo pipefail
LABEL=${1:?label}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/perf/$LABEL"
mkdir -p "$OUT"
PTX_DIR=$HOME/rt_build/nexus_ptx_unified
export CUDA_TOOLKIT_PATH=$HOME/miniconda3/targets/x86_64-linux
export PATH=$HOME/cuda-13.3-tile/bin:$HOME/.cargo/bin:$PATH

idle_check() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" -eq 0 ] || { echo "FATAL: GPU busy ($n procs)"; exit 1; }
}
echo "=== GPU idle (before) ==="; idle_check

CUDA_BIN=$ROOT/target-cuda/release/examples
WG_BIN=$ROOT/target/release/examples

echo "=== build: trainer + rollout bench ==="
cd "$ROOT"
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$PTX_DIR/nexus_rbd_shaders3d.cubin \
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$PTX_DIR/vortx_shaders.cubin \
CARGO_TARGET_DIR=$ROOT/target-cuda \
cargo build --release --example biped_train_gpu --example iter_e2e_bench --features "gpu biped_gpu cuda_backend cutile" 2>&1 | tail -1
cargo build --release --example rollout_e2e_bench --features "gpu biped_gpu" 2>&1 | tail -1

iter_rows () { # tag env...
  local tag=$1; shift
  for n in 2048 4096 8192; do
    echo "--- $tag N=$n ---" | tee -a "$OUT/raw2.log"
    env "$@" "$CUDA_BIN/iter_e2e_bench" $n 32 5 16 2>&1 | tee -a "$OUT/raw2.log" | grep -E "FULL GPU" | tail -1
  done
}

echo "=== A: CUDA 12-DOF G1, solver-iters 8 ===" | tee -a "$OUT/raw2.log"
iter_rows cuda-12dof-it8 KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_ROBOT=unitree_g1 BIPED_SOLVER_ITERS=8

echo "=== B: CUDA 12-DOF G1, solver-iters 4 ===" | tee -a "$OUT/raw2.log"
iter_rows cuda-12dof-it4 KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_ROBOT=unitree_g1 BIPED_SOLVER_ITERS=4

echo "=== C: CUDA 29-DOF +realism FLAT (no terrain) ===" | tee -a "$OUT/raw2.log"
iter_rows cuda-29dof-flat KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_ROBOT=g1_29dof_agile \
  BIPED_SOLVER_ITERS=8 BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_CONTACT_REDUCE=1 BIPED_CONTACT_CAP=128

echo "=== D: rollout-only CPU vs GPU (rollout_e2e_bench, 256 steps) ===" | tee -a "$OUT/raw2.log"
for n in 2048 4096 8192; do
  echo "--- rollout N=$n ---" | tee -a "$OUT/raw2.log"
  KHAL_BACKEND=webgpu "$WG_BIN/rollout_e2e_bench" $n 256 2>&1 | tee -a "$OUT/raw2.log" | grep -iE "env/s|rollout|CPU|GPU" | tail -4
done

echo "=== E: true-training throughput (biped_train_gpu, BIPED_ROBOT=g1, 40 iters) ===" | tee -a "$OUT/raw2.log"
for n in 2048 4096 8192; do
  echo "--- train N=$n ---" | tee -a "$OUT/raw2.log"
  rm -f /tmp/pb_train.safetensors*
  env KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 BIPED_UPD_GRAPH=0 BIPED_ROBOT=g1 NEXUS_SMALL_SORT=1 NEXUS_FIXED_GRID=1 \
    "$CUDA_BIN/biped_train_gpu" 40 $n /tmp/pb_train.safetensors > /tmp/pb_train.log 2>&1 || true
  grep "\[prof\]" /tmp/pb_train.log | tail -5 | tee -a "$OUT/raw2.log" | tail -1
  python3 - "$n" <<'PYEOF' | tee -a "$OUT/raw2.log"
import re, sys
n = int(sys.argv[1])
rolls, upds = [], []
for line in open("/tmp/pb_train.log"):
    m = re.search(r"roll=([0-9.]+)s .*upd=([0-9.]+)s", line)
    if m:
        rolls.append(float(m.group(1))); upds.append(float(m.group(2)))
tail = slice(-10, None)
r = sum(rolls[tail])/len(rolls[tail]); u = sum(upds[tail])/len(upds[tail])
steps = 24 * n
print(f"train N={n}: {r+u:.2f}s/iter -> {steps/(r+u)/1000:.1f} k env-steps/s (roll {r:.2f} upd {u:.2f})")
PYEOF
done

echo "=== F: nexus bench_batch_sweep3 (boxes + chain, WebGPU) ===" | tee -a "$OUT/raw2.log"
cd $HOME/rt_build/nexus-unified
CARGO_TARGET_DIR=$HOME/rt_build/nexus-unified/target cargo build --release -p nexus_examples_3d --example bench_batch_sweep3 2>&1 | tail -1
for cfg in "boxes 3" "chain 6"; do
  set -- $cfg
  echo "--- nexus sweep $1 size=$2 (NEXUS_BENCH_ONLY_MAX @1024) ---" | tee -a "$OUT/raw2.log"
  NEXUS_BENCH_ONLY_MAX=1 $HOME/rt_build/nexus-unified/target/release/examples/bench_batch_sweep3 $1 $2 1024 2>&1 | tee -a "$OUT/raw2.log" | tail -4
done

echo "=== GPU idle (after) ==="; idle_check
echo "BENCH2_DONE — $OUT/raw2.log"
