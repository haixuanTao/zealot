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
        if std::env::var("BIPED_CUTILE_GEMM").is_ok_and(|v| v == "1") {
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
        #[cutile::entry()]
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
        #[cutile::entry()]
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
        #[cutile::entry()]
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

    /// Smallest tile size in {16, 32, 64, 128} covering `dim` (checked kernels
    /// handle the ceil-grid remainder).
    fn tile_for(dim: usize, max: usize) -> usize {
        for c in [16usize, 32, 64, 128] {
            if c >= dim || c == max {
                return c.min(max);
            }
        }
        max
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
    }

    impl CutileGemm {
        /// Build the adapter if `BIPED_CUTILE_GEMM=1` and the backend is CUDA.
        /// Runs a numeric self-test (vs a CPU reference, through the real
        /// khal-buffer interop path) before returning. Leaked: see module docs.
        pub async fn init(bk: &GpuBackend) -> Option<&'static CutileGemm> {
            if !std::env::var("BIPED_CUTILE_GEMM").is_ok_and(|v| v == "1") {
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
            let me: &'static CutileGemm = Box::leak(Box::new(CutileGemm {
                stream,
                device_id: ctx.ordinal(),
                _device: device,
                inputs: RefCell::new(HashMap::new()),
                outputs: RefCell::new(HashMap::new()),
                parts: RefCell::new(HashMap::new()),
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
            let bm = tile_for(m, 128);
            let bn = tile_for(n, 128);
            let bk = tile_for(k, 64);
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
            let s_count = if k >= 1024 && blocks_m * blocks_n < 96 {
                [32usize, 16, 8, 4, 2, 1]
                    .into_iter()
                    .find(|&s| ktiles >= s)
                    .unwrap_or(1)
            } else {
                1
            };

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
            let bm = tile_for(m, 128);
            let bn = tile_for(n, 128);
            let bk = tile_for(k, 64);
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
