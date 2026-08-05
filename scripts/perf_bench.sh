#!/bin/bash
# Recompute the zealot perf tables (iter_e2e_bench) on the current stack
# (cubins from scripts/build_cubins.sh), with the production
# physics config (d4 + 4 substeps + NEXUS_SUBSTEP_REFRESH). WebGPU + native
# CUDA(+cuTile) legs, N = 2048/4096/8192.
#
# Usage: scripts/perf_bench.sh <label>   (writes bench/perf/<label>/)
set -eo pipefail
LABEL=${1:?label}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/perf/$LABEL"
mkdir -p "$OUT"
PTX_DIR=$HOME/nexus_ptx
export CUDA_TOOLKIT_PATH=$HOME/cuda-13-shim

# Cubins are built separately: scripts/build_cubins.sh (host-interception + O3).
[ -s "$PTX_DIR/nexus_rbd_shaders3d.cubin" ] || { echo "FATAL: run scripts/build_cubins.sh first"; exit 1; }

idle_check() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" -eq 0 ] || { echo "FATAL: GPU busy ($n procs) — refusing to bench"; exit 1; }
}

echo "=== GPU idle check (before) ==="; idle_check

echo "=== STAGE 4: bench binaries ==="
cd "$ROOT"
cargo build --release --example iter_e2e_bench --features "gpu biped_gpu" 2>&1 | tail -1
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$PTX_DIR/nexus_rbd_shaders3d.cubin \
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$PTX_DIR/vortx_shaders.cubin \
CARGO_TARGET_DIR=$ROOT/target-cuda \
cargo build --release --example iter_e2e_bench --features "gpu biped_gpu cuda_backend cutile" 2>&1 | tail -1

export PATH=$HOME/cuda-13.3-tile/bin:$PATH   # tileiras + ptxas 13.3 for the cuTile runtime JIT

run_bench () { # tag binary extra-env...
  local tag=$1 bin=$2; shift 2
  for n in 2048 4096 8192; do
    echo "--- $tag N=$n ---" | tee -a "$OUT/raw.log"
    env "$@" "$bin" $n 32 5 16 2>&1 | tee -a "$OUT/raw.log" | grep -E "env/s|FULL|rollout|update" | tail -6
  done
}

echo "=== STAGE 5: WebGPU legacy 12-DOF (README parity) ===" | tee -a "$OUT/raw.log"
run_bench webgpu-12dof "$ROOT/target/release/examples/iter_e2e_bench" KHAL_BACKEND=webgpu

echo "=== STAGE 6: CUDA+cuTile bold column (AGILE-parity realism+terrain) ===" | tee -a "$OUT/raw.log"
run_bench cuda-bold "$ROOT/target-cuda/release/examples/iter_e2e_bench" \
  KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_ROBOT=g1_29dof_agile BIPED_SOLVER_ITERS=8 \
  BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_TERRAIN=1 BIPED_CONTACT_REDUCE=1 BIPED_CONTACT_CAP=128

echo "=== STAGE 7: CUDA+cuTile PRODUCTION config (d4+it4+refresh, stiff contacts) ===" | tee -a "$OUT/raw.log"
run_bench cuda-production "$ROOT/target-cuda/release/examples/iter_e2e_bench" \
  KHAL_BACKEND=cuda BIPED_CUTILE_GEMM=1 BIPED_ROBOT=g1_29dof_agile BIPED_SOLVER_ITERS=4 \
  BIPED_DECIMATION=4 NEXUS_SUBSTEP_REFRESH=1 BIPED_CONTACT_NF=240 BIPED_CONTACT_DR=1 \
  BIPED_MAX_CORR_VEL=0.2 BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_TERRAIN=1 \
  BIPED_CONTACT_REDUCE=1 BIPED_CONTACT_CAP=128

echo "=== GPU idle check (after) ==="; idle_check
echo "BENCH_DONE — raw output in $OUT/raw.log"
