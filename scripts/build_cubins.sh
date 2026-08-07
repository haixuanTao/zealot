#!/bin/bash
# Rebuild the sm_120 cubins for the production stack (nexus + vortx-unified +
# khal-unified) into ~/nexus_ptx, then rebuild the trainer against them.
# Unified successor of the old build_cubins.sh (device-target) and
# full_unified_chain.sh (host-target interception) — this is the
# host-interception chain with full-O3 lowering and fp-contract=fast.
#
# Codegen backend: built automatically from NVlabs/cuda-oxide at BACKEND_REV
# (cached under ~/.cache/zealot/cuda-oxide), default = the khal-std-pinned
# upstream rev. PAIRING MATTERS: host-target interception (this script)
# works with the upstream pin; the old device-target route needed the
# a5f4062f reconcile-translators merge instead (upstream pin fails its
# atomics lowering there). Override with CUDA_OXIDE_BACKEND / BACKEND_REV.
set -eo pipefail
export PATH=$HOME/.cargo/bin:$PATH
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=$HOME/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=$HOME/cuda-13.3-tile/bin/ptxas
BACKEND_REV="${BACKEND_REV:-62472763}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_OXIDE_PTX_DIR="${PTX_OUT:-$ROOT/cubins}"
export CUDA_OXIDE_DEVICE_ARCH=sm_120
WORK=$HOME/Documents/work
mkdir -p "$CUDA_OXIDE_PTX_DIR"

if [ -z "$CUDA_OXIDE_BACKEND" ]; then
  BDIR=$HOME/.cache/zealot/cuda-oxide
  if [ ! -d "$BDIR" ]; then
    git clone --filter=blob:none https://github.com/NVlabs/cuda-oxide.git "$BDIR"
  fi
  git -C "$BDIR" fetch -q origin "$BACKEND_REV" && git -C "$BDIR" checkout -q "$BACKEND_REV"
  echo "=== STAGE 1: build codegen backend @ ${BACKEND_REV:0:9} ==="
  (cd "$BDIR/crates/rustc-codegen-cuda" && CARGO_TARGET_DIR="$BDIR/target" cargo +nightly-2026-04-03 build --release) 2>&1 | tail -1
  CUDA_OXIDE_BACKEND=$BDIR/target/release/librustc_codegen_cuda.so
fi
[ -f "$CUDA_OXIDE_BACKEND" ] || { echo "FATAL: backend .so not found: $CUDA_OXIDE_BACKEND"; exit 1; }

gen () { # $1 ws, $2 pkg, $3 features, $4 ll name, $5 srcdir
  cd "$1"; rm -f $CUDA_OXIDE_PTX_DIR/$4.ll
  find "$5" -name "*.rs" -exec touch {} +
  CARGO_INCREMENTAL=0 \
  RUSTFLAGS="-Z codegen-backend=$CUDA_OXIDE_BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release --no-default-features --features "$3" 2>&1 | tail -1
  [ -s $CUDA_OXIDE_PTX_DIR/$4.ll ] || { echo "FATAL: $4.ll missing"; exit 1; }
  grep -q ") #0 {" $CUDA_OXIDE_PTX_DIR/$4.ll || { echo "FATAL: $4.ll lacks convergent attrs"; exit 1; }
  echo "$4.ll ok: $(grep -c "^define" $CUDA_OXIDE_PTX_DIR/$4.ll) defines"
}
echo "=== STAGE 2: regen .ll (host-target interception) ==="
gen $WORK/nexus nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d $WORK/nexus/src_rbd_shaders
gen $WORK/vortx-unified vortx-shaders "cuda-oxide" vortx_shaders $WORK/vortx-unified/vortx-shaders/src

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

echo "=== STAGE 4: rebuild trainer (forced re-embed) ==="
# Cubin paths + toolkit come from .cargo/config.toml [env] (repo) and
# ~/.cargo/config.toml [env] (machine) — see docs/development.md.
cd $WORK/zealot
cargo clean -p nexus_rbd_shaders3d -p vortx-shaders 2>/dev/null || true
cargo build --release --bin biped_train_gpu --features "gpu biped_gpu cutile" 2>&1 | tail -1
echo CHAIN_DONE
