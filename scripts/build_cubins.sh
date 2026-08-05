#!/bin/bash
# Rebuild the sm_120 cubins for the production stack (nexus-unified +
# vortx-unified + khal-unified) into ~/nexus_ptx.
#
# Toolchain (see docs/train-on-5090.md):
#   TOOL    llvm bins from the pinned nightly toolchain
#   LIBDEV  libdevice from the nvcc wheel (~/nvvm-wheel)
#   PTXAS   ptxas 13.3 (~/cuda-13.3-tile — system ptxas has no sm_120)
#   BACKEND rustc_codegen_cuda .so — built from NVlabs/cuda-oxide at the rev
#           khal-std pins (see its Cargo.toml `cuda-device` git dep). The
#           old local clone ~/cuda-oxide-src was removed 2026-08-05; clone
#           upstream at that rev and `cargo build -p rustc-codegen-cuda`.
set -eo pipefail
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=$HOME/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=$HOME/cuda-13.3-tile/bin/ptxas
BACKEND="${CUDA_OXIDE_BACKEND:?set CUDA_OXIDE_BACKEND to librustc_codegen_cuda.so (build from NVlabs/cuda-oxide at khal-std's pinned rev)}"
export CUDA_OXIDE_PTX_DIR="${PTX_OUT:-$HOME/nexus_ptx}"
mkdir -p "$CUDA_OXIDE_PTX_DIR"
export PATH=$HOME/.cargo/bin:$PATH

build_one () { # $1 workspace dir, $2 package, $3 features, $4 ll name
  cd "$1"
  cargo clean -p "$2" 2>/dev/null || true
  CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release \
    --no-default-features --features "$3" \
    --target nvptx64-nvidia-cuda -Z build-std=core
  LL=$CUDA_OXIDE_PTX_DIR/$4.ll
  echo "ll: $(grep -c '^define' $LL) defines"
  $TOOL/llvm-as $LL -o /tmp/u.bc
  $TOOL/llvm-link /tmp/u.bc $LIBDEV -o /tmp/u_linked.bc
  $TOOL/opt -passes="internalize,globaldce" /tmp/u_linked.bc -o /tmp/u_pruned.bc
  $TOOL/llc -mcpu=sm_120 -O3 /tmp/u_pruned.bc -o /tmp/u.ptx
  $PTXAS -arch=sm_120 -O3 /tmp/u.ptx -o $CUDA_OXIDE_PTX_DIR/$4.cubin
  ls -la $CUDA_OXIDE_PTX_DIR/$4.cubin
}

build_one "$HOME/Documents/work/nexus-unified" nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d
build_one "$HOME/Documents/work/vortx-unified" vortx-shaders "cuda-oxide" vortx_shaders
echo DONE
