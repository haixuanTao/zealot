#!/bin/bash
# Recompute the zealot perf tables (iter_e2e_bench) on the current stack:
# unified NVlabs cuda-oxide toolchain, solver-fix branch, and the production
# physics config (d4 + 4 substeps + NEXUS_SUBSTEP_REFRESH). WebGPU + native
# CUDA(+cuTile) legs, N = 2048/4096/8192.
#
# Usage: scripts/perf_bench.sh <label>   (writes bench/perf/<label>/)
set -eo pipefail
LABEL=${1:?label}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/perf/$LABEL"
mkdir -p "$OUT"
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=$HOME/rt_build/bench-venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc
PTXAS=$HOME/miniconda3/bin/ptxas
BACKEND_DIR=$HOME/rt_build/cuda-oxide-upstreaming
BACKEND=$BACKEND_DIR/target/release/librustc_codegen_cuda.so
PTX_DIR=$HOME/rt_build/nexus_ptx_unified
export CUDA_TOOLKIT_PATH=$HOME/miniconda3/targets/x86_64-linux

idle_check() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" -eq 0 ] || { echo "FATAL: GPU busy ($n procs) — refusing to bench"; exit 1; }
}

echo "=== GPU idle check (before) ==="; idle_check

echo "=== STAGE 1: cuda-oxide backend (unified bridge) ==="
(cd $BACKEND_DIR/crates/rustc-codegen-cuda && CARGO_TARGET_DIR=$BACKEND_DIR/target RUSTFLAGS="-L $HOME/rt_build/linkshim" cargo build --release) 2>&1 | tail -1
[ -s "$BACKEND" ] || { echo "FATAL: backend missing"; exit 1; }

echo "=== STAGE 2: cubins (host-target interception) ==="
mkdir -p $PTX_DIR
gen () { # ws pkg features name srcdir tgt
  cd "$1"; rm -f $PTX_DIR/$4.ll
  find "$5" -name "*.rs" -exec touch {} +
  CUDA_OXIDE_DEVICE_ARCH=sm_120 CUDA_OXIDE_PTX_DIR=$PTX_DIR CARGO_TARGET_DIR="$6" CARGO_INCREMENTAL=0 \
  RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release --no-default-features --features "$3" 2>&1 | tail -1
  [ -s $PTX_DIR/$4.ll ] || { echo "FATAL: $4.ll missing"; exit 1; }
  grep -q ") #0 {" $PTX_DIR/$4.ll || { echo "FATAL: $4.ll lacks convergent attrs"; exit 1; }
  echo "$4.ll ok: $(grep -c "^define" $PTX_DIR/$4.ll) defines"
}
gen $HOME/rt_build/nexus-unified nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d $HOME/rt_build/nexus-unified/src_rbd_shaders $HOME/rt_build/nexus-unified/target
gen $HOME/rt_build/vortx-unified vortx-shaders "cuda-oxide" vortx_shaders $HOME/rt_build/vortx-unified/vortx-shaders/src $HOME/rt_build/vortx-unified/target-unified

echo "=== STAGE 3: O3 lowering ==="
for name in nexus_rbd_shaders3d vortx_shaders; do
  $TOOL/llvm-as $PTX_DIR/$name.ll -o /tmp/pb.bc
  $TOOL/llvm-link /tmp/pb.bc $LIBDEV -o /tmp/pb_linked.bc
  $TOOL/opt -passes="internalize,globaldce,default<O3>" /tmp/pb_linked.bc -o /tmp/pb_pruned.bc
  $TOOL/llc -mcpu=sm_120 -O3 -fp-contract=fast /tmp/pb_pruned.bc -o /tmp/pb.ptx
  $PTXAS -arch=sm_120 -O3 /tmp/pb.ptx -o $PTX_DIR/$name.cubin
  echo "$name.cubin built"
done

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
