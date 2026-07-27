#!/bin/bash
# Unified cuda-oxide cubin chain — NVlabs upstream backend, host-target
# interception (no nvptx64 device mode, no compiler fork).
# Replaces /tmp/full_cubin_chain.sh (fork device-target chain).
set -eo pipefail
export PATH=$HOME/.cargo/bin:$PATH
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=$HOME/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=$HOME/cuda-13.3-tile/bin/ptxas
BACKEND_DIR=$HOME/cuda-oxide-upstreaming            # NVlabs main + local/unified-bridge
BACKEND=$BACKEND_DIR/target/release/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx_unified
export CUDA_OXIDE_DEVICE_ARCH=sm_120

echo "=== STAGE 1: build upstream backend ==="
(cd $BACKEND_DIR/crates/rustc-codegen-cuda && CARGO_TARGET_DIR=$BACKEND_DIR/target cargo build --release) 2>&1 | tail -1

gen () { # $1 ws, $2 pkg, $3 features, $4 name, $5 srcdir, $6 target-dir
  cd "$1"; rm -f $CUDA_OXIDE_PTX_DIR/$4.ll
  find "$5" -name "*.rs" -exec touch {} +
  CARGO_TARGET_DIR="$6" CARGO_INCREMENTAL=0 \
  RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release --no-default-features --features "$3" 2>&1 | tail -1
  [ -s $CUDA_OXIDE_PTX_DIR/$4.ll ] || { echo "FATAL: $4.ll missing"; exit 1; }
  grep -q ") #0 {" $CUDA_OXIDE_PTX_DIR/$4.ll || { echo "FATAL: $4.ll lacks convergent attrs"; exit 1; }
  echo "$4.ll ok: $(grep -c "^define" $CUDA_OXIDE_PTX_DIR/$4.ll) defines"
}
echo "=== STAGE 2: regen .ll (unified host-target interception) ==="
gen ~/Documents/work/nexus-unified nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d ~/Documents/work/nexus-unified/src_rbd_shaders ~/Documents/work/nexus-unified/target
gen ~/Documents/work/vortx-unified vortx-shaders "cuda-oxide" vortx_shaders ~/Documents/work/vortx-unified/vortx-shaders/src ~/Documents/work/vortx-unified/target-unified

echo "=== STAGE 3: full-O3 lowering ==="
for name in nexus_rbd_shaders3d vortx_shaders; do
  LL=$CUDA_OXIDE_PTX_DIR/$name.ll
  $TOOL/llvm-as $LL -o /tmp/u.bc
  $TOOL/llvm-link /tmp/u.bc $LIBDEV -o /tmp/u_linked.bc
  $TOOL/opt -passes="internalize,globaldce,default<O3>" /tmp/u_linked.bc -o /tmp/u_pruned.bc
  $TOOL/llc -mcpu=sm_120 -O3 -fp-contract=fast /tmp/u_pruned.bc -o /tmp/u.ptx
  $PTXAS -arch=sm_120 -O3 /tmp/u.ptx -o $CUDA_OXIDE_PTX_DIR/$name.cubin
  echo "$name.cubin O3 built"
done

echo "=== STAGE 4: rebuild trainer ==="
cd ~/Documents/work/zealot-pr4
touch examples/biped/biped_train_gpu.rs
CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin \
CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin \
CUDA_TOOLKIT_PATH=$HOME/cuda-13-shim \
cargo build --release --example biped_train_gpu --features "gpu biped_gpu cuda_backend cutile" 2>&1 | tail -1
echo CHAIN_DONE
