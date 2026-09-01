//! cuTile (NVlabs cutile-rs) tf32 tensor-core GEMMs for the PPO update —
//! the `BIPED_CUTILE_GEMM=1` fast path.
//!
//! ## Why
//!
//! The update is GEMM-compute-bound in the vortx kernels (measured 2026-07-13,
//! 5090, 2048 envs): 1.40 s of GEMM per update, 63% of it the wgrad shapes
//! (K = mb = 12288, tiny M×N) which get only a handful of CTAs without
//! split-K. The cuTile tf32 kernels below measured ~0.015 s on the same shape
//! set (17–47 TFLOPS) — see `examples/gemm_shapes_bench.rs` (vortx side) and
//! the scratch bench in the cutile-rs checkout.
//!
//! ## How the interop works
//!
//! khal's CUDA backend and cuTile's `cuda-core` BOTH retain the device's
//! primary context, so device pointers are shared and no copies are needed:
//!
//! - `cuda_core::Device::borrow_raw` / `Stream::borrow_raw` wrap khal's
//!   cudarc context and stream — cuTile kernels launch **on khal's own
//!   stream**, so ordering with khal passes is by host issue order, no extra
//!   synchronization. The caller must SUBMIT any khal encoder before a cuTile
//!   launch (see `EncCursor` in the trainer) so the stream order matches.
//! - `cutile::Tensor::from_raw_parts` wraps khal buffer pointers zero-copy
//!   (vortx tensors are row-major; transposed operands are stride-swapped
//!   views). The wrappers are cached and NEVER dropped — cuTile's
//!   `DeviceBuffer::drop` would `cuMemFree` khal's memory — which is why
//!   [`CutileGemm::init`] leaks the adapter (`Box::leak`).
//!
//! ## Kernels
//!
//! Plain tiled GEMM and a split-K variant (for the wgrad shapes), tf32 inputs
//! via `convert_tile` with f32 accumulate — same numerics class as PyTorch's
//! default `allow_tf32`. CHECKED accesses (out-of-bounds loads zero-pad,
//! stores mask), accumulators zero-initialised, and ceil-div K loops: ragged
//! dims (45, 51, 12, 1 and any mb) need no padding. A numeric self-test
//! against a CPU reference runs at init.
//!
//! ## Machine setup (see memory / Cargo.toml)
//!
//! Build: `CUDA_TOOLKIT_PATH=~/cuda-13-shim` (CUDA 13.2+ headers).
//! Runtime JIT: `CUTILE_TILEIRAS_PATH=~/cuda-13.3-tile/bin/tileiras` and that
//! bin dir FIRST on PATH (tileiras execs `ptxas`; the system 12.0 ptxas dies
//! on sm_120a). `init` fills those in if unset.

#![allow(dead_code)]

#[cfg(feature = "cutile")]
pub use real::CutileGemm;

use khal::backend::{Backend, Encoder as _, GpuBackend};

/// Encoder cursor: khal command recording that can be SPLIT at any point so a
/// cuTile kernel can be launched directly on the (shared) CUDA stream between
/// khal passes. `pass()` lazily opens an encoder; `flush()` submits whatever is
/// recorded (no synchronize — same stream keeps ordering by issue order). With
/// the cuTile path off, a whole phase still records into one encoder and
/// behaves exactly like the pre-cuTile code.
pub struct EncCursor<'b> {
    bk: &'b GpuBackend,
    enc: Option<<GpuBackend as Backend>::Encoder>,
}
impl<'b> EncCursor<'b> {
    pub fn new(bk: &'b GpuBackend) -> Self {
        Self { bk, enc: None }
    }
    /// Wrap an existing encoder so GEMM passes share the caller's command
    /// buffer (single-submit control steps in the browser demo).
    pub fn from_encoder(bk: &'b GpuBackend, enc: <GpuBackend as Backend>::Encoder) -> Self {
        Self { bk, enc: Some(enc) }
    }
    /// Hand the (possibly created) encoder back to the caller WITHOUT
    /// submitting — the inverse of [`Self::from_encoder`].
    pub fn into_encoder(mut self) -> Option<<GpuBackend as Backend>::Encoder> {
        self.enc.take()
    }
    pub fn pass(&mut self, name: &str) -> khal::backend::GpuPass {
        if self.enc.is_none() {
            self.enc = Some(self.bk.begin_encoding());
        }
        self.enc.as_mut().unwrap().begin_pass(name, None)
    }
    /// Submit pending khal work to the stream (required before a cuTile launch).
    pub fn flush(&mut self) {
        if let Some(e) = self.enc.take() {
            self.bk.submit(e).unwrap();
        }
    }
}

/// Stub when the `cutile` feature is off: `init` always yields `None`, so the
/// trainer's vortx path is untouched.
#[cfg(not(feature = "cutile"))]
pub struct CutileGemm;

#[cfg(not(feature = "cutile"))]
#[allow(clippy::too_many_arguments)]
impl CutileGemm {
    pub async fn init(_bk: &GpuBackend) -> Option<&'static CutileGemm> {
        if std::env::var("BIPED_CUTILE_GEMM").map_or(true, |v| v != "0") {
            eprintln!(
                "[cutile] BIPED_CUTILE_GEMM=1 but zealot was built without --features cutile; \
                 using the vortx GEMM path"
            );
        }
        None
    }
    pub fn gemm(
        &self,
        _out: &vortx::tensor::Tensor<f32>,
        _lhs: &vortx::tensor::Tensor<f32>,
        _lhs_t: bool,
        _rhs: &vortx::tensor::Tensor<f32>,
        _rhs_t: bool,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> anyhow::Result<()> {
        unreachable!("stub CutileGemm is never constructed")
    }
    pub fn gemm_bias_act(
        &self,
        _out: &vortx::tensor::Tensor<f32>,
        _lhs: &vortx::tensor::Tensor<f32>,
        _rhs: &vortx::tensor::Tensor<f32>,
        _m: usize,
        _n: usize,
        _k: usize,
        _bias: &vortx::tensor::Tensor<f32>,
        _bias_row_stride: usize,
        _elu: bool,
    ) -> anyhow::Result<()> {
        unreachable!("stub CutileGemm is never constructed")
    }
    pub fn elu_backward(
        &self,
        _g: &vortx::tensor::Tensor<f32>,
        _y: &vortx::tensor::Tensor<f32>,
        _m: usize,
        _n: usize,
    ) -> anyhow::Result<()> {
        unreachable!("stub CutileGemm is never constructed")
    }
    pub fn row_sum(
        &self,
        _out: &vortx::tensor::Tensor<f32>,
        _x: &vortx::tensor::Tensor<f32>,
        _m: usize,
        _n: usize,
    ) -> anyhow::Result<()> {
        unreachable!("stub CutileGemm is never constructed")
    }
}

#[cfg(feature = "cutile")]
mod real {
    use cuda_async::device_operation::DeviceOp;
    use cutile::api;
    use cutile::prelude::IntoPartition;
    use cutile::tensor::Tensor as CtTensor;
    use cutile::tile_kernel::{PartitionOp, TileKernel};
    use khal::Shader;
    use khal::backend::{Backend, GpuBackend, GpuBuffer};
    use nalgebra::DMatrix;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[cutile::module]
    mod kernels {
        use cutile::core::*;

        /// Tiled GEMM `z = x·y`, tf32 tensor cores, f32 accumulate. Checked
        /// accesses (OOB loads zero-pad, stores mask) + ceil-div K loop, so no
        /// dimension needs to be a tile multiple. Overwrites `z`.
        /// Codegen hints. These are keyed BY ARCHITECTURE — the compiler selects
        /// the matching set and other targets fall back to defaults, so this is
        /// not an sm_120 lock-in. (This module is the CUDA fast path already;
        /// the portable SPIR-V/Metal shaders are separate source and untouched.)
        /// `occupancy` and `num_cta_in_cga` are scheduling hints; both are pure
        /// codegen and cannot change results. Re-tune per GPU like TUNED_TILES.
        #[cutile::entry(optimization_hints = (
            sm_120 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_100 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_90 = (num_cta_in_cga = 1,),
        ))]
        unsafe fn gemm_tf32<const BM: i32, const BN: i32, const BK: i32>(
            z: &mut Tensor<f32, { [BM, BN] }>,
            x: &Tensor<f32, { [-1, -1] }>,
            y: &Tensor<f32, { [-1, -1] }>,
            k: i32,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut acc: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
            let kt = (k + BK - 1) / BK;
            for i in 0i32..kt {
                let tile_x: Tile<f32, { [BM, BK] }> = part_x.load([pid.0, i]);
                let tile_y: Tile<f32, { [BK, BN] }> = part_y.load([i, pid.1]);
                let tx: Tile<tf32, { [BM, BK] }> = convert_tile(tile_x);
                let ty: Tile<tf32, { [BK, BN] }> = convert_tile(tile_y);
                acc = mma(tx, ty, acc);
            }
            z.store(acc);
        }

        /// Split-K partial GEMM: chunk `s = pid.0 / blocks_m` accumulates its
        /// K-range into `z_parts` (shape `[S·blocks_m·BM, N]`). NOTE checked
        /// partition access TRAPS on an out-of-range BLOCK index (only
        /// within-tile ragged edges zero-pad), so the tail chunk's k-tile
        /// range is clamped to `ktiles_total` explicitly.
        /// Codegen hints. These are keyed BY ARCHITECTURE — the compiler selects
        /// the matching set and other targets fall back to defaults, so this is
        /// not an sm_120 lock-in. (This module is the CUDA fast path already;
        /// the portable SPIR-V/Metal shaders are separate source and untouched.)
        /// `occupancy` and `num_cta_in_cga` are scheduling hints; both are pure
        /// codegen and cannot change results. Re-tune per GPU like TUNED_TILES.
        #[cutile::entry(optimization_hints = (
            sm_120 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_100 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_90 = (num_cta_in_cga = 1,),
        ))]
        unsafe fn gemm_splitk_tf32<const BM: i32, const BN: i32, const BK: i32>(
            z_parts: &mut Tensor<f32, { [BM, BN] }>,
            x: &Tensor<f32, { [-1, -1] }>,
            y: &Tensor<f32, { [-1, -1] }>,
            blocks_m: i32,
            ktiles_per_chunk: i32,
            ktiles_total: i32,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let s = pid.0 / blocks_m;
            let mb = pid.0 % blocks_m;
            let mut acc: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
            let lo = s * ktiles_per_chunk;
            let mut hi = lo + ktiles_per_chunk;
            if hi > ktiles_total {
                hi = ktiles_total;
            }
            for kt in lo..hi {
                let tile_x: Tile<f32, { [BM, BK] }> = part_x.load([mb, kt]);
                let tile_y: Tile<f32, { [BK, BN] }> = part_y.load([kt, pid.1]);
                let tx: Tile<tf32, { [BM, BK] }> = convert_tile(tile_x);
                let ty: Tile<tf32, { [BK, BN] }> = convert_tile(tile_y);
                acc = mma(tx, ty, acc);
            }
            z_parts.store(acc);
        }

        /// Tiled GEMM with fused epilogue: `z = x·y + bias` (bias is a column
        /// vector broadcast over N), optionally followed by ELU (alpha = 1:
        /// `v > 0 ? v : exp(v) − 1`, matching vortx `gpu_elu`). Replaces the
        /// forward pass's gemm + bias-broadcast-GEMV + add + ELU pass chain.
        /// Codegen hints. These are keyed BY ARCHITECTURE — the compiler selects
        /// the matching set and other targets fall back to defaults, so this is
        /// not an sm_120 lock-in. (This module is the CUDA fast path already;
        /// the portable SPIR-V/Metal shaders are separate source and untouched.)
        /// `occupancy` and `num_cta_in_cga` are scheduling hints; both are pure
        /// codegen and cannot change results. Re-tune per GPU like TUNED_TILES.
        #[cutile::entry(optimization_hints = (
            sm_120 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_100 = (occupancy = 2, num_cta_in_cga = 2,),
            sm_90 = (num_cta_in_cga = 1,),
        ))]
        unsafe fn gemm_bias_act_tf32<const BM: i32, const BN: i32, const BK: i32>(
            z: &mut Tensor<f32, { [BM, BN] }>,
            x: &Tensor<f32, { [-1, -1] }>,
            y: &Tensor<f32, { [-1, -1] }>,
            bias: &Tensor<f32, { [-1, -1] }>,
            k: i32,
            apply_elu: i32,
        ) {
            let part_x = x.partition(const_shape![BM, BK]);
            let part_y = y.partition(const_shape![BK, BN]);
            let part_b = bias.partition(const_shape![BM, 1]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut acc: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
            let kt = (k + BK - 1) / BK;
            for i in 0i32..kt {
                let tile_x: Tile<f32, { [BM, BK] }> = part_x.load([pid.0, i]);
                let tile_y: Tile<f32, { [BK, BN] }> = part_y.load([i, pid.1]);
                let tx: Tile<tf32, { [BM, BK] }> = convert_tile(tile_x);
                let ty: Tile<tf32, { [BK, BN] }> = convert_tile(tile_y);
                acc = mma(tx, ty, acc);
            }
            let bt: Tile<f32, { [BM, 1] }> = part_b.load([pid.0, 0]);
            acc = acc + bt.broadcast(const_shape![BM, BN]);
            if apply_elu != 0 {
                let zero: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
                let one: Tile<f32, { [BM, BN] }> = 1.0f32.broadcast(const_shape![BM, BN]);
                let mask = cmpf(acc, zero, predicate::GreaterThan, cmp_ordering::Ordered);
                let em1 = exp(acc) - one;
                acc = select(mask, acc, em1);
            }
            z.store(acc);
        }

        /// In-place ELU backward (alpha = 1): `g *= (y > 0 ? 1 : y + 1)` where
        /// `y` is the cached POST-activation — matching vortx `gpu_elu_backward`.
        #[cutile::entry()]
        unsafe fn elu_backward_ct<const BM: i32, const BN: i32>(
            g: &mut Tensor<f32, { [BM, BN] }>,
            y: &Tensor<f32, { [-1, -1] }>,
        ) {
            let part_y = y.partition(const_shape![BM, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let gt: Tile<f32, { [BM, BN] }> = g.load();
            let yt: Tile<f32, { [BM, BN] }> = part_y.load([pid.0, pid.1]);
            let zero: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
            let one: Tile<f32, { [BM, BN] }> = 1.0f32.broadcast(const_shape![BM, BN]);
            let mask = cmpf(yt, zero, predicate::GreaterThan, cmp_ordering::Ordered);
            let deriv = select(mask, one, yt + one);
            g.store(gt * deriv);
        }

        /// Row sums: `out[r] = Σ_c x[r, c]` — the bias gradient (replaces the
        /// vortx `delta · ones` GEMV, which ran ~100x below memory bandwidth).
        #[cutile::entry()]
        unsafe fn row_sum_ct<const BM: i32, const BN: i32>(
            out: &mut Tensor<f32, { [BM] }>,
            x: &Tensor<f32, { [-1, -1] }>,
            n: i32,
        ) {
            let part = x.partition(const_shape![BM, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            // Seed the accumulator from the first column-tile's reduction (both
            // addition operands then come from reduce_sum — the AST compiler
            // resolves the const-generic result shape inconsistently between
            // broadcast and reduce_sum otherwise).
            let t0: Tile<f32, { [BM, BN] }> = part.load([pid.0, 0]);
            let mut acc: Tile<f32, { [BM] }> = reduce_sum(t0, 1i32);
            let nt = (n + BN - 1) / BN;
            for j in 1i32..nt {
                let t: Tile<f32, { [BM, BN] }> = part.load([pid.0, j]);
                let s: Tile<f32, { [BM] }> = reduce_sum(t, 1i32);
                acc = acc + s;
            }
            out.store(acc);
        }

        /// Bias + optional ELU as a SEPARATE pass, for the cuBLAS path.
        ///
        /// Fusing this into our own GEMM epilogue saves an elementwise pass but
        /// costs far more in matmul throughput: measured, the fused kernel runs
        /// the forward GEMMs in 23.3 ms where cuBLAS does them in ~10, against
        /// only ~4 ms saved on elementwise. So when cuBLAS is driving, take the
        /// unfused route.
        #[cutile::entry()]
        unsafe fn bias_act_ct<const BM: i32, const BN: i32>(
            z: &mut Tensor<f32, { [BM, BN] }>,
            bias: &Tensor<f32, { [-1, -1] }>,
            apply_elu: i32,
        ) {
            let part_b = bias.partition(const_shape![BM, 1]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut acc: Tile<f32, { [BM, BN] }> = z.load();
            let bt: Tile<f32, { [BM, 1] }> = part_b.load([pid.0, 0]);
            acc = acc + bt.broadcast(const_shape![BM, BN]);
            if apply_elu != 0 {
                let zero: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
                let one: Tile<f32, { [BM, BN] }> = 1.0f32.broadcast(const_shape![BM, BN]);
                let mask = cmpf(acc, zero, predicate::GreaterThan, cmp_ordering::Ordered);
                let em1 = exp(acc) - one;
                acc = select(mask, acc, em1);
            }
            z.store(acc);
        }

        /// Split row-sums — pass 1. Block `p` handles row-tile `p % row_tiles`
        /// and column-chunk `p / row_tiles`, summing `tps` column-tiles into
        /// `parts[chunk·row_tiles + row_tile]`. The 1-D grid is
        /// `row_tiles · splits`, so `parts` is a rank-1 `[splits·m]` buffer laid
        /// out chunk-major; pass 2 views it as `[m, splits]` (strides `[1, m]`)
        /// and re-uses `row_sum_ct`.
        ///
        /// Why: the single-pass kernel parallelises over `m` only — and `m` here
        /// is a bias dimension (<=512), so a 25 MB reduction ran on 2-4 CTAs at
        /// ~12% of memory bandwidth. `splits` must divide the column-tile count
        /// exactly (the launcher picks a divisor), so no bounds check is needed.
        #[cutile::entry()]
        unsafe fn row_sum_split_ct<const BM: i32, const BN: i32>(
            parts: &mut Tensor<f32, { [BM] }>,
            x: &Tensor<f32, { [-1, -1] }>,
            row_tiles: i32,
            tps: i32,
        ) {
            let part = x.partition(const_shape![BM, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let rt = pid.0 % row_tiles;
            let j0 = (pid.0 / row_tiles) * tps;
            let t0: Tile<f32, { [BM, BN] }> = part.load([rt, j0]);
            let mut acc: Tile<f32, { [BM] }> = reduce_sum(t0, 1i32);
            for j in 1i32..tps {
                let t: Tile<f32, { [BM, BN] }> = part.load([rt, j0 + j]);
                let s: Tile<f32, { [BM] }> = reduce_sum(t, 1i32);
                acc = acc + s;
            }
            parts.store(acc);
        }

        /// Sum the split-K partials: `out[mb, nb] = Σ_s parts[s·blocks_m + mb, nb]`.
        /// Overwrites `out`.
        #[cutile::entry()]
        unsafe fn reduce_splitk<const BM: i32, const BN: i32>(
            out: &mut Tensor<f32, { [BM, BN] }>,
            parts: &Tensor<f32, { [-1, -1] }>,
            blocks_m: i32,
            s_count: i32,
        ) {
            let part = parts.partition(const_shape![BM, BN]);
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut acc: Tile<f32, { [BM, BN] }> = 0.0f32.broadcast(const_shape![BM, BN]);
            for s in 0i32..s_count {
                let t: Tile<f32, { [BM, BN] }> = part.load([s * blocks_m + pid.0, pid.1]);
                acc = acc + t;
            }
            out.store(acc);
        }
    }
    use kernels::*;

    /// Split-K only kicks in below this many CTAs from the tile grid alone.
    const SPLITK_MIN_CTAS: usize = 96;
    /// ...and then splits only far enough to reach this many CTAs (2 per SM on
    /// an RTX 5090). `BIPED_CUTILE_SPLITK_LOG=1` prints the decision per shape.
    const SPLITK_TARGET_CTAS: usize = 340;
    /// Never leave a split with fewer K-tiles than this — a split that does
    /// almost no work still pays a full prologue and another merge input.
    const SPLITK_MIN_KTILES_PER_SPLIT: usize = 8;

    /// Offline-tuned GEMM tiles: `(kind, m, n, k, bm, bn, bk)`.
    ///
    /// Measured on an RTX 5090 (sm_120) with `BIPED_CUTILE_TUNE=1`; `kind` 0 is
    /// `gemm`, 1 is `gemm_bias_act`. Shapes not listed fall back to the
    /// analytical rule. Re-tune per GPU — these are hardware-specific.
    /// `BIPED_CUTILE_TUNED=0` ignores the table.
    #[rustfmt::skip]
    const TUNED_TILES: [(u8, usize, usize, usize, usize, usize, usize); 33] = [
        (0, 1, 128, 24576, 16, 64, 64), // 0.034 ms
        (0, 12, 128, 24576, 16, 64, 64), // 0.034 ms
        (0, 12, 300, 45, 16, 128, 64), // 0.008 ms
        (0, 128, 24576, 1, 128, 64, 16), // 0.011 ms
        (0, 128, 24576, 12, 128, 64, 16), // 0.012 ms
        (0, 128, 256, 24576, 64, 64, 64), // 0.082 ms
        (0, 256, 24576, 128, 64, 128, 64), // 0.048 ms
        (0, 256, 24576, 256, 64, 128, 64), // 0.078 ms
        (0, 256, 256, 24576, 128, 128, 64), // 0.125 ms
        (0, 256, 395, 24576, 64, 64, 64), // 0.224 ms
        (0, 256, 45, 300, 16, 64, 64), // 0.012 ms
        (0, 256, 512, 24576, 128, 128, 64), // 0.202 ms
        (0, 45, 300, 256, 16, 128, 64), // 0.011 ms
        (0, 512, 12288, 51, 64, 128, 64), // 0.031 ms
        (0, 512, 24576, 256, 128, 128, 64), // 0.137 ms
        (0, 512, 90, 24576, 64, 64, 64), // 0.139 ms
        (0, 64, 96, 3072, 16, 128, 64), // 0.022 ms
        (0, 64, 96, 4096, 16, 128, 64), // 0.024 ms
        (1, 1, 24576, 128, 16, 256, 64), // 0.010 ms
        (1, 1, 4096, 128, 16, 128, 64), // 0.009 ms
        (1, 100, 300, 45, 64, 64, 64), // 0.009 ms
        (1, 12, 24576, 128, 16, 256, 64), // 0.012 ms
        (1, 12, 4096, 128, 16, 64, 64), // 0.009 ms
        (1, 128, 24576, 256, 64, 64, 64), // 0.039 ms
        (1, 128, 4096, 256, 64, 64, 64), // 0.010 ms
        (1, 256, 24576, 256, 64, 64, 64), // 0.068 ms
        (1, 256, 24576, 395, 64, 64, 64), // 0.132 ms
        (1, 256, 24576, 512, 64, 64, 64), // 0.109 ms
        (1, 256, 4096, 256, 64, 64, 64), // 0.019 ms
        (1, 256, 4096, 395, 64, 64, 64), // 0.033 ms
        (1, 256, 4096, 512, 64, 64, 64), // 0.027 ms
        (1, 512, 24576, 90, 64, 64, 64), // 0.087 ms
        (1, 512, 4096, 90, 64, 64, 64), // 0.022 ms
    ];

    /// Smallest tile size in {16, 32, 64, 128} covering `dim` (checked kernels
    /// handle the ceil-grid remainder).
    fn tile_for(dim: usize, max: usize) -> usize {
        for c in [16usize, 32, 64, 128, 256] {
            if c >= dim || c == max {
                return c.min(max);
            }
        }
        max
    }

    /// `BIPED_CUTILE_TILE=bm,bn,bk` overrides the GEMM tile shape (each capped
    /// by the dimension it covers). The default heuristic picks the smallest
    /// covering tile, which pins every large GEMM to 128x128x64; cuBLAS chooses
    /// per shape from a tuned set (128x256, 256x128, ...), which is most of its
    /// edge on these skinny shapes. Set to sweep candidates without a rebuild.
    fn tile_override() -> Option<(usize, usize, usize)> {
        static OVERRIDE: std::sync::OnceLock<Option<(usize, usize, usize)>> =
            std::sync::OnceLock::new();
        *OVERRIDE.get_or_init(|| {
            let v = std::env::var("BIPED_CUTILE_TILE").ok()?;
            let mut it = v.split(',').map(|x| x.trim().parse::<usize>());
            match (it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?) {
                (bm, bn, bk) => {
                    eprintln!("[cutile] tile override {bm}x{bn}x{bk}");
                    Some((bm, bn, bk))
                }
            }
        })
    }

    /// (device_ptr, shape[2], strides[2]) — element strides.
    type ViewKey = (u64, [i32; 2], [i32; 2]);

    pub struct CutileGemm {
        stream: Arc<cuda_core::Stream>,
        // Keep the borrowed device alive as long as the (leaked) adapter.
        _device: Arc<cuda_core::Device>,
        device_id: usize,
        /// Zero-copy input views over khal buffers, keyed by
        /// (device_ptr, rows, cols, transposed). Never dropped (leaked adapter):
        /// dropping would cuMemFree khal's memory.
        inputs: RefCell<HashMap<ViewKey, Arc<CtTensor<f32>>>>,
        /// Zero-copy OUTPUT views (taken out / re-inserted around each launch,
        /// since the launcher takes the output tensor by value).
        outputs: RefCell<HashMap<ViewKey, CtTensor<f32>>>,
        /// cuTile-owned split-K partial buffers, keyed by (padded rows, cols).
        parts: RefCell<HashMap<(usize, usize), CtTensor<f32>>>,
        /// BIPED_CUBLAS_GEMM=1: reference cuBLAS path for the plain GEMMs, to
        /// measure the cuTile kernels against inside the real trainer.
        cublas: Option<crate::cublas::Cublas>,
        /// BIPED_CUTILE_TUNE only: (kind, m, n, k) -> measured winner.
        tuned: RefCell<HashMap<(u8, usize, usize, usize), (usize, usize, usize)>>,
    }

    impl CutileGemm {
        /// Build the adapter if `BIPED_CUTILE_GEMM=1` and the backend is CUDA.
        /// Runs a numeric self-test (vs a CPU reference, through the real
        /// khal-buffer interop path) before returning. Leaked: see module docs.
        pub async fn init(bk: &GpuBackend) -> Option<&'static CutileGemm> {
            if std::env::var("BIPED_CUTILE_GEMM").is_ok_and(|v| v == "0") {
                return None;
            }
            let Some(cuda) = bk.as_cuda() else {
                eprintln!("[cutile] BIPED_CUTILE_GEMM=1 needs the CUDA backend (BIPED_CUDA=1)");
                return None;
            };
            // JIT toolchain defaults (machine-local): tileiras 13.3 + its ptxas
            // first on PATH — the system CUDA 12.0 ptxas can't do sm_120a.
            let home = std::env::var("HOME").unwrap_or_default();
            let tile_bin = format!("{home}/cuda-13.3-tile/bin");
            if std::env::var("CUTILE_TILEIRAS_PATH").is_err() {
                // SAFETY: single-threaded init, before any JIT compile.
                unsafe { std::env::set_var("CUTILE_TILEIRAS_PATH", format!("{tile_bin}/tileiras")) };
            }
            let path = std::env::var("PATH").unwrap_or_default();
            if !path.starts_with(&tile_bin) {
                unsafe { std::env::set_var("PATH", format!("{tile_bin}:{path}")) };
            }
            let ctx = cuda.context();
            // SAFETY: khal's context/stream are primary-context handles that
            // outlive the leaked adapter; cuTile only borrows them.
            let device = unsafe {
                cuda_core::Device::borrow_raw(
                    ctx.cu_ctx() as *mut std::ffi::c_void,
                    ctx.cu_device(),
                    ctx.ordinal(),
                )
            };
            let stream = unsafe {
                cuda_core::Stream::borrow_raw(
                    cuda.stream().cu_stream() as *mut std::ffi::c_void,
                    &device,
                )
            };
            let cu_stream_raw = stream.cu_stream() as *mut std::ffi::c_void;
            let me: &'static CutileGemm = Box::leak(Box::new(CutileGemm {
                stream,
                device_id: ctx.ordinal(),
                _device: device,
                inputs: RefCell::new(HashMap::new()),
                outputs: RefCell::new(HashMap::new()),
                parts: RefCell::new(HashMap::new()),
                cublas: if std::env::var("BIPED_CUBLAS_GEMM").is_ok_and(|v| v != "0") {
                    // `stream` is moved into the struct below; take the raw
                    // handle first so cuBLAS shares the same in-order stream.
                    match crate::cublas::Cublas::new(cu_stream_raw) {
                        Ok(c) => {
                            eprintln!("[cublas] reference GEMM path ENABLED (plain gemm only)");
                            Some(c)
                        }
                        Err(e) => {
                            eprintln!("[cublas] unavailable ({e}); using cuTile GEMM");
                            None
                        }
                    }
                } else {
                    None
                },
                tuned: RefCell::new(HashMap::new()),
            }));
            match me.self_test(bk).await {
                Ok(worst) => {
                    println!(
                        "[cutile] tf32 GEMM path ENABLED (self-test worst rel err {worst:.2e})"
                    );
                    Some(me)
                }
                Err(e) => {
                    eprintln!("[cutile] self-test FAILED ({e}); falling back to vortx GEMM");
                    None
                }
            }
        }

        /// Wrap a khal buffer as a cuTile tensor view. `rows`/`cols` are the
        /// LOGICAL gemm-operand dims; `transposed` means the underlying vortx
        /// tensor is the (cols × rows) row-major matrix and we view its
        /// transpose via swapped strides.
        fn view(
            &self,
            t: &vortx::tensor::Tensor<f32>,
            rows: usize,
            cols: usize,
            transposed: bool,
        ) -> Arc<CtTensor<f32>> {
            let (shape, strides) = if transposed {
                // Base allocation is row-major (cols × rows); its transpose is
                // (rows × cols) with element strides (1, rows).
                ([rows as i32, cols as i32], [1i32, rows as i32])
            } else {
                ([rows as i32, cols as i32], [cols as i32, 1i32])
            };
            self.view_strided(buf_ptr(t.buffer()), shape, strides)
        }

        /// Arbitrary-stride view — e.g. a bias column vector inside a
        /// pre-broadcast [out × n] buffer: shape [m, 1], strides [n, 1].
        fn view_strided(&self, ptr: u64, shape: [i32; 2], strides: [i32; 2]) -> Arc<CtTensor<f32>> {
            let key = (ptr, shape, strides);
            if let Some(v) = self.inputs.borrow().get(&key) {
                return v.clone();
            }
            let v = Arc::new(self.raw_view(ptr, shape, strides));
            self.inputs.borrow_mut().insert(key, v.clone());
            v
        }

        fn raw_view(&self, ptr: u64, shape: [i32; 2], strides: [i32; 2]) -> CtTensor<f32> {
            // cuTile asserts logical bytes == storage bytes at construction, so
            // declare the LOGICAL size even for sparse (strided-column) views —
            // the strided reads stay inside the real (larger) khal allocation.
            let len_bytes = shape[0] as usize * shape[1] as usize * 4;
            // SAFETY: ptr is a live khal allocation whose extent covers every
            // strided element of the view; the view is cached in the leaked
            // adapter and never dropped.
            unsafe {
                CtTensor::from_raw_parts(ptr, len_bytes, self.device_id, shape.to_vec(), strides.to_vec())
            }
        }

        /// `out(m×n) = lhs(m×k) · rhs(k×n)`, all operands khal/vortx f32
        /// tensors (row-major). `lhs_t`/`rhs_t`: the passed tensor is the
        /// transposed base (e.g. wgrad's `aᵀ`), viewed via strides. The caller
        /// must have SUBMITTED all pending khal work touching these buffers
        /// (same stream ⇒ ordering by issue).
        /// Offline tile tuning — `BIPED_CUTILE_TUNE=1`.
        ///
        /// Times every candidate on the real buffers and prints a table row per
        /// shape, ready to paste into `TUNED_TILES`. NOT for production runs:
        /// it measures inside iteration 0 (cold caches, JIT firing mid-timing,
        /// physics queued on the same in-order stream) and it re-runs each GEMM
        /// several times — safe only because `gemm_splitk` STORES its partials
        /// rather than accumulating them. Re-run per GPU; the winners are
        /// hardware-specific.
        const TUNE_CANDIDATES: [(usize, usize, usize); 8] = [
            (128, 128, 64),
            (128, 64, 64),
            (64, 128, 64),
            (64, 64, 64),
            (16, 128, 64),
            (16, 256, 64),
            (128, 256, 64),
            (256, 128, 64),
        ];

        fn tune_tiles<F>(&self, kind: u8, m: usize, n: usize, k: usize, run: F) -> (usize, usize, usize)
        where
            F: Fn((usize, usize, usize)) -> anyhow::Result<()>,
        {
            let key = (kind, m, n, k);
            if let Some(t) = self.tuned.borrow().get(&key) {
                return *t;
            }
            let mut cands: Vec<(usize, usize, usize)> = Self::TUNE_CANDIDATES
                .iter()
                .map(|&(a, b, c)| (tile_for(m, a), tile_for(n, b), tile_for(k, c)))
                .collect();
            cands.sort_unstable();
            cands.dedup();
            let sync = || api::zeros::<f32>(&[1]).sync_on(&self.stream).is_ok();
            let mut best = (self.analytic_tiles(m, n, k), f64::INFINITY);
            for t in cands {
                if run(t).is_err() || !sync() {
                    continue;
                }
                let t0 = std::time::Instant::now();
                for _ in 0..5 {
                    let _ = run(t);
                }
                if !sync() {
                    continue;
                }
                let dt = t0.elapsed().as_secs_f64() / 5.0;
                if dt < best.1 {
                    best = (t, dt);
                }
            }
            let (bm, bn, bk) = best.0;
            eprintln!(
                "[cutile-tune]     ({kind}, {m}, {n}, {k}, {bm}, {bn}, {bk}), // {:.3} ms",
                best.1 * 1e3
            );
            self.tuned.borrow_mut().insert(key, best.0);
            best.0
        }

        /// Analytical tile choice for one GEMM shape.
        ///
        /// The old heuristic took the smallest covering tile from
        /// {16,32,64,128}, which collapses to 128x128x64 for nearly every call
        /// — one tile shape for ~10 very different GEMMs. Forcing a single
        /// wider tile is worse, not better (128x256 globally: update 0.11 ->
        /// 0.23 s), because the update mixes wide-N shapes (n=256/512, the
        /// hidden layers) with degenerate ones (n=12 actor output, n=1 critic
        /// head) — a 256-wide tile wastes 95% of its columns on the latter.
        ///
        /// So pick per shape, from the shape alone. Two constraints decide it:
        ///
        /// 1. **Fill the GPU.** A `BM x BN` tile emits `ceil(m/BM)*ceil(n/BN)`
        ///    CTAs; below ~2 per SM the tail dominates. Shrink the tile until
        ///    the grid covers `2 * SM_COUNT`, preferring to shrink whichever
        ///    dimension still has slack.
        /// 2. **Do not pay for padding.** Never choose a tile wider than the
        ///    dimension it covers, rounded up to the next power of two — that
        ///    is what makes 128x256 catastrophic at n=12.
        ///
        /// `BK` trades pipeline depth against register pressure: deep-K shapes
        /// (the weight grads, k = batch) want the larger step, shallow ones the
        /// smaller. No timing, no JIT fan-out, deterministic across runs.
        fn analytic_tiles(&self, m: usize, n: usize, k: usize) -> (usize, usize, usize) {
            if let Some((tm, tn, tk)) = tile_override() {
                return (
                    tile_for(m, tm.min(256)),
                    tile_for(n, tn.min(256)),
                    tile_for(k, tk.min(128)),
                );
            }
            if std::env::var("BIPED_CUTILE_ANALYTIC").is_ok_and(|v| v == "0") {
                return (tile_for(m, 128), tile_for(n, 128), tile_for(k, 64));
            }
            // Deep-K shapes (the weight grads, k = batch) get their occupancy
            // from the split-K path below, which only triggers while
            // `blocks_m * blocks_n < 96`. Shrinking their tiles to "fill the
            // GPU" would push the grid past that and silently turn split-K OFF
            // — slower, not faster. Leave that regime exactly as tuned.
            if k >= 1024 {
                return (tile_for(m, 128), tile_for(n, 128), tile_for(k, 64));
            }
            // Constraint 2: cap each tile side at the next power of two >= dim
            // (and at the kernel's supported maximum).
            let cap = |d: usize, hi: usize| -> usize {
                let mut c = 16usize;
                while c < d && c < hi {
                    c *= 2;
                }
                c.min(hi)
            };
            let mut bm = cap(m, 256);
            let mut bn = cap(n, 256);
            // Constraint 1: shrink the larger side until the grid fills the GPU.
            const SM: usize = 170; // RTX 5090 (sm_120)
            let target = 2 * SM;
            while bm.max(bn) > 16 && m.div_ceil(bm) * n.div_ceil(bn) < target {
                if bm >= bn && bm > 16 {
                    bm /= 2;
                } else if bn > 16 {
                    bn /= 2;
                } else {
                    break;
                }
            }
            // K step: deep reductions amortise a larger step; shallow ones do
            // not, and an oversized BK just pads.
            let bk = if k >= 1024 { cap(k, 128) } else { cap(k, 64) };
            (bm, bn, bk)
        }

        /// Tile for one GEMM shape: the offline-tuned table when the shape is in
        /// it, else the analytical rule. The table is what cuBLAS effectively
        /// ships — fitted per architecture, not derived — and it recovers the
        /// last factor of two in BN that shape alone does not determine (the
        /// choice depends on K-depth interacting with occupancy: 256x24576x395
        /// wants 128x64 while 256x24576x256 wants 128x128).
        fn pick_tiles(&self, kind: u8, m: usize, n: usize, k: usize) -> (usize, usize, usize) {
            if tile_override().is_some() {
                return self.analytic_tiles(m, n, k);
            }
            if !std::env::var("BIPED_CUTILE_TUNED").is_ok_and(|v| v == "0") {
                if let Some(&(_, _, _, _, bm, bn, bk)) = TUNED_TILES
                    .iter()
                    .find(|&&(kd, tm, tn, tk, _, _, _)| kd == kind && tm == m && tn == n && tk == k)
                {
                    return (bm, bn, bk);
                }
            }
            self.analytic_tiles(m, n, k)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn gemm(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            lhs: &vortx::tensor::Tensor<f32>,
            lhs_t: bool,
            rhs: &vortx::tensor::Tensor<f32>,
            rhs_t: bool,
            m: usize,
            n: usize,
            k: usize,
        ) -> anyhow::Result<()> {
            if let Some(cb) = &self.cublas {
                // Same stream, so ordering against the cuTile launches holds.
                return cb.gemm(
                    buf_ptr(out.buffer()),
                    buf_ptr(lhs.buffer()),
                    lhs_t,
                    buf_ptr(rhs.buffer()),
                    rhs_t,
                    m,
                    n,
                    k,
                );
            }
            let tiles = if std::env::var("BIPED_CUTILE_TUNE").is_ok() {
                self.tune_tiles(0, m, n, k, |t| {
                    self.gemm_inner(out, lhs, lhs_t, rhs, rhs_t, m, n, k, t)
                })
            } else {
                self.pick_tiles(0, m, n, k)
            };
            self.gemm_inner(out, lhs, lhs_t, rhs, rhs_t, m, n, k, tiles)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn gemm_bias_act(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            lhs: &vortx::tensor::Tensor<f32>,
            rhs: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
            k: usize,
            bias: &vortx::tensor::Tensor<f32>,
            bias_row_stride: usize,
            elu: bool,
        ) -> anyhow::Result<()> {
            if let Some(cb) = &self.cublas {
                cb.gemm(
                    buf_ptr(out.buffer()),
                    buf_ptr(lhs.buffer()),
                    false,
                    buf_ptr(rhs.buffer()),
                    false,
                    m,
                    n,
                    k,
                )?;
                return self.bias_act(out, m, n, bias, bias_row_stride, elu);
            }
            let tiles = if std::env::var("BIPED_CUTILE_TUNE").is_ok() {
                self.tune_tiles(1, m, n, k, |t| {
                    self.gemm_bias_act_inner(out, lhs, rhs, m, n, k, bias, bias_row_stride, elu, t)
                })
            } else {
                self.pick_tiles(1, m, n, k)
            };
            self.gemm_bias_act_inner(out, lhs, rhs, m, n, k, bias, bias_row_stride, elu, tiles)
        }

        #[allow(clippy::too_many_arguments)]
        fn gemm_inner(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            lhs: &vortx::tensor::Tensor<f32>,
            lhs_t: bool,
            rhs: &vortx::tensor::Tensor<f32>,
            rhs_t: bool,
            m: usize,
            n: usize,
            k: usize,
            tiles: (usize, usize, usize),
        ) -> anyhow::Result<()> {
            let (bm, bn, bk) = tiles;
            let x = self.view(lhs, m, k, lhs_t);
            let y = self.view(rhs, k, n, rhs_t);
            let out_ptr = buf_ptr(out.buffer());
            let out_key = (out_ptr, [m as i32, n as i32], [n as i32, 1i32]);
            let out_t = self
                .outputs
                .borrow_mut()
                .remove(&out_key)
                .unwrap_or_else(|| self.raw_view(out_ptr, [m as i32, n as i32], [n as i32, 1i32]));

            let blocks_m = m.div_ceil(bm);
            let blocks_n = n.div_ceil(bn);
            let ktiles = k.div_ceil(bk);
            // Split-K for deep-K, small-output shapes (the wgrads): without it
            // they run on blocks_m·blocks_n CTAs and leave the GPU idle.
            //
            // The split count is sized to the deficit, not taken from a fixed
            // list. The old code picked the largest s <= ktiles — always 32,
            // since ktiles is 384 at batch 24576 — regardless of how many CTAs
            // the tile already produced. With the tuned 64x64 tile the base
            // grid is ~28 CTAs, so s=32 launched ~900 CTAs on 170 SMs (5x
            // oversubscribed) AND made `reduce_splitk` sum 32 partials per
            // output element. Both halves of that are wasted: the merge is
            // linear in s, and its 7.4 ms plus the split GEMM's 29.6 ms were
            // 43% of the whole update.
            //
            // So: split only as far as it takes to fill the machine, keeping
            // enough K per split that each CTA still amortises its prologue.
            let s_count = if k >= 1024 && blocks_m * blocks_n < SPLITK_MIN_CTAS {
                let base = (blocks_m * blocks_n).max(1);
                let want = SPLITK_TARGET_CTAS.div_ceil(base);
                // Round down to a power of two: the merge indexes partials as
                // `s * blocks_m + mb`, and uneven splits leave a ragged tail.
                let mut s = 1usize;
                while s * 2 <= want.min(ktiles / SPLITK_MIN_KTILES_PER_SPLIT.max(1)) {
                    s *= 2;
                }
                // BIPED_CUTILE_SPLITS=n forces the count, for sweeping the
                // GEMM-parallelism vs merge-cost trade directly.
                match std::env::var("BIPED_CUTILE_SPLITS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    Some(forced) => forced.clamp(1, ktiles.max(1)),
                    None => s.clamp(1, ktiles.max(1)),
                }
            } else {
                1
            };
            if std::env::var("BIPED_CUTILE_SPLITK_LOG").is_ok() && s_count > 1 {
                eprintln!(
                    "[cutile] splitk {m}x{n}x{k} tile {bm}x{bn}x{bk} base={} splits={s_count} -> {} ctas",
                    blocks_m * blocks_n,
                    blocks_m * blocks_n * s_count
                );
            }

            let g = vec![bm.to_string(), bn.to_string(), bk.to_string()];
            let stored = if s_count > 1 {
                let kpc = ktiles.div_ceil(s_count);
                let mp = blocks_m * bm;
                let pkey = (s_count * mp, n);
                let parts_t = match self.parts.borrow_mut().remove(&pkey) {
                    Some(p) => p,
                    None => api::zeros::<f32>(&[s_count * mp, n])
                        .sync_on(&self.stream)
                        .map_err(anyhow_err)?,
                };
                let (parts_back, _, _, _, _, _) = unsafe {
                    gemm_splitk_tf32(
                        parts_t.partition([bm, bn]),
                        x,
                        y,
                        blocks_m as i32,
                        kpc as i32,
                        ktiles as i32,
                    )
                    .generics(g)
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
                };
                let parts_t = Arc::new(parts_back.unpartition());
                let (out_back, parts_t, _, _) = unsafe {
                    reduce_splitk(
                        out_t.partition([bm, bn]),
                        parts_t,
                        blocks_m as i32,
                        s_count as i32,
                    )
                    .generics(vec![bm.to_string(), bn.to_string()])
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
                };
                self.parts.borrow_mut().insert(
                    pkey,
                    Arc::try_unwrap(parts_t)
                        .map_err(|_| anyhow::anyhow!("split-K parts still shared"))?,
                );
                out_back
            } else {
                let (out_back, _, _, _) = unsafe {
                    gemm_tf32(out_t.partition([bm, bn]), x, y, k as i32)
                        .generics(g)
                        .async_on(&self.stream)
                        .map_err(anyhow_err)?
                };
                out_back
            };
            self.outputs
                .borrow_mut()
                .insert(out_key, stored.unpartition());
            Ok(())
        }

        /// Fused forward layer: `out(m×n) = act(lhs(m×k)·rhs(k×n) + bias)`,
        /// where `bias` is a column vector read with `bias_row_stride` (1 for a
        /// dense [m×1] tensor; n for column 0 of a pre-broadcast [m×n] one) and
        /// `act` is ELU when `elu` (hidden layers) else identity. Replaces the
        /// vortx gemm + bias-GEMV + add + ELU pass chain in one launch.
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
        fn gemm_bias_act_inner(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            lhs: &vortx::tensor::Tensor<f32>,
            rhs: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
            k: usize,
            bias: &vortx::tensor::Tensor<f32>,
            bias_row_stride: usize,
            elu: bool,
            tiles: (usize, usize, usize),
        ) -> anyhow::Result<()> {
            let (bm, bn, bk) = tiles;
            let x = self.view(lhs, m, k, false);
            let y = self.view(rhs, k, n, false);
            let b = self.view_strided(
                buf_ptr(bias.buffer()),
                [m as i32, 1],
                [bias_row_stride as i32, 1],
            );
            let out_ptr = buf_ptr(out.buffer());
            let out_key = (out_ptr, [m as i32, n as i32], [n as i32, 1i32]);
            let out_t = self
                .outputs
                .borrow_mut()
                .remove(&out_key)
                .unwrap_or_else(|| self.raw_view(out_ptr, [m as i32, n as i32], [n as i32, 1i32]));
            let g = vec![bm.to_string(), bn.to_string(), bk.to_string()];
            let (out_back, _, _, _, _, _) = unsafe {
                gemm_bias_act_tf32(
                    out_t.partition([bm, bn]),
                    x,
                    y,
                    b,
                    k as i32,
                    if elu { 1i32 } else { 0i32 },
                )
                .generics(g)
                .async_on(&self.stream)
                .map_err(anyhow_err)?
            };
            self.outputs
                .borrow_mut()
                .insert(out_key, out_back.unpartition());
            Ok(())
        }

        /// In-place ELU backward over `g(m×n)`: `g *= (y > 0 ? 1 : y + 1)`,
        /// `y` the cached post-activation (matches vortx `gpu_elu_backward`).
        pub fn elu_backward(
            &self,
            g: &vortx::tensor::Tensor<f32>,
            y: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
        ) -> anyhow::Result<()> {
            let bm = tile_for(m, 128);
            let bn = tile_for(n, 128);
            let yv = self.view(y, m, n, false);
            let g_ptr = buf_ptr(g.buffer());
            let g_key = (g_ptr, [m as i32, n as i32], [n as i32, 1i32]);
            let g_t = self
                .outputs
                .borrow_mut()
                .remove(&g_key)
                .unwrap_or_else(|| self.raw_view(g_ptr, [m as i32, n as i32], [n as i32, 1i32]));
            let (g_back, _) = unsafe {
                elu_backward_ct(g_t.partition([bm, bn]), yv)
                    .generics(vec![bm.to_string(), bn.to_string()])
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
            };
            self.outputs
                .borrow_mut()
                .insert(g_key, g_back.unpartition());
            Ok(())
        }

        /// Two-pass row-sum: `splits` chunks of column-tiles reduced in
        /// parallel, then merged. Same result as the single-pass kernel up to
        /// summation order (fp32 addition is not associative).
        #[allow(clippy::too_many_arguments)]
        fn row_sum_split(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            x: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
            bm: usize,
            bn: usize,
            row_tiles: usize,
            nt: usize,
            splits: usize,
        ) -> anyhow::Result<()> {
            let xv = self.view(x, m, n, false);
            // Pass 1 -> parts[splits * m], chunk-major.
            let pkey = (splits * row_tiles * bm, 1usize);
            let parts_t = match self.parts.borrow_mut().remove(&pkey) {
                Some(p) => p,
                None => api::zeros::<f32>(&[pkey.0])
                    .sync_on(&self.stream)
                    .map_err(anyhow_err)?,
            };
            let tps = nt / splits;
            let (parts_back, _, _, _) = unsafe {
                row_sum_split_ct(
                    parts_t.partition([bm]),
                    xv,
                    row_tiles as i32,
                    tps as i32,
                )
                .generics(vec![bm.to_string(), bn.to_string()])
                .async_on(&self.stream)
                .map_err(anyhow_err)?
            };
            let parts_t = parts_back.unpartition();
            let parts_ptr = parts_t.device_pointer().cu_deviceptr();
            self.parts.borrow_mut().insert(pkey, parts_t);

            // Pass 2: view the chunk-major partials as [m, splits] (strides
            // [1, m']) and re-use the single-pass kernel over `splits` columns.
            let mp = row_tiles * bm;
            // MUST go through the cache: a bare `raw_view` is dropped at the end
            // of this call and its Tensor frees the pointer it does not own
            // ("Free async failed: invalid argument"). Cached views are leaked.
            let pv = self.view_strided(parts_ptr, [m as i32, splits as i32], [1i32, mp as i32]);
            let bn2 = tile_for(splits, 128);
            let out_ptr = buf_ptr(out.buffer());
            let out_key = (out_ptr, [m as i32, 0], [1i32, 0]);
            let out_t = match self.outputs.borrow_mut().remove(&out_key) {
                Some(t) => t,
                // SAFETY: same invariants as raw_view, rank-1.
                None => unsafe {
                    CtTensor::from_raw_parts(out_ptr, m * 4, self.device_id, vec![m as i32], vec![1])
                },
            };
            let (out_back, _, _) = unsafe {
                row_sum_ct(out_t.partition([bm]), pv, splits as i32)
                    .generics(vec![bm.to_string(), bn2.to_string()])
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
            };
            self.outputs
                .borrow_mut()
                .insert(out_key, out_back.unpartition());
            Ok(())
        }

        /// Apply bias (+ ELU) in place — the cuBLAS path's epilogue.
        fn bias_act(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
            bias: &vortx::tensor::Tensor<f32>,
            bias_row_stride: usize,
            elu: bool,
        ) -> anyhow::Result<()> {
            let (bm, bn) = (tile_for(m, 128), tile_for(n, 128));
            let b = self.view_strided(
                buf_ptr(bias.buffer()),
                [m as i32, 1],
                [bias_row_stride as i32, 1],
            );
            let out_ptr = buf_ptr(out.buffer());
            let out_key = (out_ptr, [m as i32, n as i32], [n as i32, 1i32]);
            let out_t = self
                .outputs
                .borrow_mut()
                .remove(&out_key)
                .unwrap_or_else(|| self.raw_view(out_ptr, [m as i32, n as i32], [n as i32, 1i32]));
            let (out_back, _, _) = unsafe {
                bias_act_ct(out_t.partition([bm, bn]), b, i32::from(elu))
                    .generics(vec![bm.to_string(), bn.to_string()])
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
            };
            self.outputs
                .borrow_mut()
                .insert(out_key, out_back.unpartition());
            Ok(())
        }

        /// Bias gradient: `out(m×1) = row_sums(x(m×n))` in one launch.
        pub fn row_sum(
            &self,
            out: &vortx::tensor::Tensor<f32>,
            x: &vortx::tensor::Tensor<f32>,
            m: usize,
            n: usize,
        ) -> anyhow::Result<()> {
            let bm = tile_for(m, 128);
            let bn = tile_for(n, 128);
            // Parallelise over the REDUCED dimension too. `m` is a bias width
            // (<=512) so the single-pass grid is 2-4 CTAs; split the column
            // tiles across `splits` chunks (a divisor of the tile count, so the
            // kernel needs no tail check) and merge with a second, tiny pass.
            let nt = n.div_ceil(bn);
            let row_tiles = m.div_ceil(bm);
            let splits = if std::env::var("BIPED_CUTILE_ROWSUM_SPLIT").is_ok_and(|v| v == "0") {
                1
            } else {
                (2..=64)
                    .rev()
                    .find(|s| nt % s == 0 && row_tiles * s <= 512 && nt / s >= 2)
                    .unwrap_or(1)
            };
            if splits > 1 {
                return self.row_sum_split(out, x, m, n, bm, bn, row_tiles, nt, splits);
            }
            let xv = self.view(x, m, n, false);
            let out_ptr = buf_ptr(out.buffer());
            // Rank-1 view; keyed with zeroed second slots to stay distinct
            // from any 2-D view of the same buffer.
            let out_key = (out_ptr, [m as i32, 0], [1i32, 0]);
            let out_t = match self.outputs.borrow_mut().remove(&out_key) {
                Some(t) => t,
                // SAFETY: same invariants as raw_view, rank-1.
                None => unsafe {
                    CtTensor::from_raw_parts(out_ptr, m * 4, self.device_id, vec![m as i32], vec![1])
                },
            };
            let (out_back, _, _) = unsafe {
                row_sum_ct(out_t.partition([bm]), xv, n as i32)
                    .generics(vec![bm.to_string(), bn.to_string()])
                    .async_on(&self.stream)
                    .map_err(anyhow_err)?
            };
            self.outputs
                .borrow_mut()
                .insert(out_key, out_back.unpartition());
            Ok(())
        }

        /// Numeric self-test through the REAL interop path (khal buffers,
        /// strided transposes, ragged dims, split-K): compares against a CPU
        /// reference. Returns the worst relative error (tf32 tolerance).
        async fn self_test(&self, bk: &GpuBackend) -> anyhow::Result<f64> {
            use khal::BufferUsages;
            let rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
            // (m, n, k, lhs_t, rhs_t) — ragged dims + a split-K trigger.
            let cases = [
                (12usize, 300usize, 45usize, false, false),
                (45, 300, 256, true, false),  // dgrad-style: Wᵀ · delta
                (256, 45, 300, false, true),  // wgrad-style: delta · aᵀ
                (64, 96, 4096, false, true), // split-K path
                // Split-K with ktiles NOT divisible by S (48 tiles / S=32):
                // exercises the tail-chunk clamp (checked block access traps).
                (64, 96, 3072, false, true),
                (512, 12288, 51, false, false),
            ];
            let mut worst = 0.0f64;
            for (ci, &(m, n, k, lt, rt)) in cases.iter().enumerate() {
                let f = |r: usize, c: usize, seed: usize| {
                    DMatrix::<f32>::from_fn(r, c, |i, j| {
                        let h = (i * 31 + j * 17 + seed * 101) % 97;
                        (h as f32) / 48.5 - 1.0
                    })
                };
                let lhs_m = f(m, k, ci);
                let rhs_m = f(k, n, ci + 7);
                let refr = &lhs_m * &rhs_m;
                // Store bases the way the trainer does: transposed operands are
                // the (k×m)/(n×k) base matrices viewed via strides.
                let lhs_base = if lt { lhs_m.transpose() } else { lhs_m.clone() };
                let rhs_base = if rt { rhs_m.transpose() } else { rhs_m.clone() };
                let gl = vortx::tensor::Tensor::matrix_from_na(bk, &lhs_base, rw)?;
                let gr = vortx::tensor::Tensor::matrix_from_na(bk, &rhs_base, rw)?;
                let go = vortx::tensor::Tensor::matrix_from_na(bk, &DMatrix::<f32>::from_element(m, n, 7.7), rw)?;
                self.gemm(&go, &gl, lt, &gr, rt, m, n, k)?;
                bk.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let got = bk
                    .slow_read_vec(go.buffer())
                    .await
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let scale = refr.amax().max(1e-6);
                let mut err = 0.0f64;
                for r in 0..m {
                    for c in 0..n {
                        let d = (got[r * n + c] - refr[(r, c)]).abs() as f64;
                        err = err.max(d / scale as f64);
                    }
                }
                if err > 5e-2 {
                    anyhow::bail!(
                        "case {ci} (m={m} n={n} k={k} lt={lt} rt={rt}): rel err {err:.3e}"
                    );
                }
                worst = worst.max(err);
            }

            // Fused bias+ELU forward and the ELU backward, on ragged dims and a
            // STRIDED bias column (row stride > 1, the GpuPolicy layout).
            {
                let elu = |v: f32| if v > 0.0 { v } else { v.exp() - 1.0 };
                let (m, n, k) = (100usize, 300usize, 45usize);
                let f = |r: usize, c: usize, seed: usize| {
                    DMatrix::<f32>::from_fn(r, c, |i, j| {
                        let h = (i * 29 + j * 13 + seed * 89) % 83;
                        (h as f32) / 41.5 - 1.0
                    })
                };
                let (lhs_m, rhs_m) = (f(m, k, 1), f(k, n, 2));
                // Bias stored pre-broadcast [m × 8]: use column 0 with stride 8.
                let bias_b = f(m, 8, 3);
                let z = &lhs_m * &rhs_m;
                let refr = DMatrix::<f32>::from_fn(m, n, |r, c| elu(z[(r, c)] + bias_b[(r, 0)]));
                let gl = vortx::tensor::Tensor::matrix_from_na(bk, &lhs_m, rw)?;
                let gr = vortx::tensor::Tensor::matrix_from_na(bk, &rhs_m, rw)?;
                let gb = vortx::tensor::Tensor::matrix_from_na(bk, &bias_b, rw)?;
                let go = vortx::tensor::Tensor::matrix_from_na(
                    bk,
                    &DMatrix::<f32>::from_element(m, n, 7.7),
                    rw,
                )?;
                self.gemm_bias_act(&go, &gl, &gr, m, n, k, &gb, 8, true)?;
                // ELU backward: g *= (y > 0 ? 1 : y + 1) with y = the fused output.
                let g_m = f(m, n, 4);
                let gg = vortx::tensor::Tensor::matrix_from_na(bk, &g_m, rw)?;
                self.elu_backward(&gg, &go, m, n)?;
                bk.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let got = bk
                    .slow_read_vec(go.buffer())
                    .await
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let got_g = bk
                    .slow_read_vec(gg.buffer())
                    .await
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let scale = refr.amax().max(1e-6);
                let mut err = 0.0f64;
                let mut err_g = 0.0f64;
                for r in 0..m {
                    for c in 0..n {
                        let y = refr[(r, c)];
                        err = err.max((got[r * n + c] - y).abs() as f64 / scale as f64);
                        let gref = g_m[(r, c)] * if y > 0.0 { 1.0 } else { y + 1.0 };
                        err_g = err_g.max((got_g[r * n + c] - gref).abs() as f64);
                    }
                }
                if err > 5e-2 || err_g > 5e-2 {
                    anyhow::bail!("fused bias+elu: rel err {err:.3e}, backward err {err_g:.3e}");
                }
                worst = worst.max(err).max(err_g);
                // Row sums (bias gradient) — pure f32 adds, tight tolerance.
                let gs = vortx::tensor::Tensor::matrix_from_na(
                    bk,
                    &DMatrix::<f32>::from_element(m, 1, 7.7),
                    rw,
                )?;
                self.row_sum(&gs, &gg, m, n)?;
                bk.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let got_s = bk
                    .slow_read_vec(gs.buffer())
                    .await
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let mut err_s = 0.0f64;
                for r in 0..m {
                    let mut refv = 0.0f64;
                    for c in 0..n {
                        refv += got_g[r * n + c] as f64;
                    }
                    err_s = err_s.max((got_s[r] as f64 - refv).abs() / refv.abs().max(1.0));
                }
                if err_s > 1e-4 {
                    anyhow::bail!("row_sum err {err_s:.3e}");
                }
                worst = worst.max(err_s);
            }
            Ok(worst)
        }
    }

    fn buf_ptr(b: &GpuBuffer<f32>) -> u64 {
        match b {
            GpuBuffer::Cuda(cb) => cb.device_ptr_raw(),
            _ => panic!("cutile gemm needs CUDA khal buffers"),
        }
    }

    fn anyhow_err<E: std::fmt::Debug>(e: E) -> anyhow::Error {
        anyhow::anyhow!("{e:?}")
    }
}
