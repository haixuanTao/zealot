//! Minimal cuBLAS FFI — a reference GEMM to measure the cuTile kernels against.
//!
//! Neither cuTile nor cuda-oxide ships host-side cuBLAS bindings (cuda-oxide
//! references only device-side cuBLASDx), so this declares the five entry points
//! we need and links `libcublas` directly.
//!
//! Why it exists: on the update's shapes cuBLAS reaches ~68 TFLOP/s while the
//! cuTile kernels reach ~33. This path makes that comparison runnable *inside
//! the trainer* rather than in a standalone benchmark — enable with
//! `BIPED_CUBLAS_GEMM=1`. It is a measurement tool, not the default: adopting
//! cuBLAS for the update would give up the single-source property that lets the
//! same kernels compile to SPIR-V and Metal for the browser/mac builds.
//!
//! Layout note: our tensors are ROW-major, cuBLAS is COLUMN-major. A row-major
//! `C(m×n) = A(m×k)·B(k×n)` is the column-major `Cᵀ(n×m) = Bᵀ·Aᵀ`, so we pass
//! B first, A second, and swap m/n — no transposes and no copies.

use std::ffi::c_void;

/// Opaque `CUstream` — kept as a raw pointer so this needs no extra binding
/// crate; `cuda_core::Stream::cu_stream()` hands one over directly.
pub type CuStream = *mut c_void;

pub type CublasHandle = *mut c_void;

// cublasStatus_t == 0 on success.
type Status = i32;

// cublasOperation_t
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
// cudaDataType
const CUDA_R_32F: i32 = 0;
// cublasComputeType_t — tf32 tensor cores with f32 accumulate, matching what
// the cuTile kernels do (`convert_tile` to tf32, f32 accumulator).
const CUBLAS_COMPUTE_32F_FAST_TF32: i32 = 77;
// cublasGemmAlgo_t
const CUBLAS_GEMM_DEFAULT: i32 = -1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut CublasHandle) -> Status;
    fn cublasDestroy_v2(handle: CublasHandle) -> Status;
    fn cublasSetStream_v2(handle: CublasHandle, stream: CuStream) -> Status;
    #[allow(clippy::too_many_arguments)]
    fn cublasGemmEx(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const c_void,
        atype: i32,
        lda: i32,
        b: *const c_void,
        btype: i32,
        ldb: i32,
        beta: *const f32,
        c: *mut c_void,
        ctype: i32,
        ldc: i32,
        compute_type: i32,
        algo: i32,
    ) -> Status;
}

/// Owns the cuBLAS handle, bound to the caller's stream so ordering matches the
/// cuTile launches (same in-order stream ⇒ no extra synchronisation).
pub struct Cublas {
    handle: CublasHandle,
}

// SAFETY: the handle is used only from the thread that owns the adapter, and is
// bound to a single stream.
unsafe impl Send for Cublas {}

impl Cublas {
    pub fn new(stream: CuStream) -> anyhow::Result<Self> {
        let mut handle: CublasHandle = std::ptr::null_mut();
        let st = unsafe { cublasCreate_v2(&mut handle) };
        if st != 0 {
            anyhow::bail!("cublasCreate failed: {st}");
        }
        let st = unsafe { cublasSetStream_v2(handle, stream) };
        if st != 0 {
            anyhow::bail!("cublasSetStream failed: {st}");
        }
        Ok(Self { handle })
    }

    /// Row-major `out(m×n) = lhs(m×k) · rhs(k×n)`, tf32 tensor cores.
    ///
    /// `lhs_t`/`rhs_t` mean the operand is stored transposed (the passed buffer
    /// is the (k×m) / (n×k) row-major matrix), matching `CutileGemm::gemm`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        out: u64,
        lhs: u64,
        lhs_t: bool,
        rhs: u64,
        rhs_t: bool,
        m: usize,
        n: usize,
        k: usize,
    ) -> anyhow::Result<()> {
        let (alpha, beta) = (1.0f32, 0.0f32);
        // Column-major view: compute Cᵀ = Bᵀ·Aᵀ by passing (B, A) and (n, m).
        // Leading dimensions are the ROW-major row lengths of each operand.
        let (op_b, ldb) = if rhs_t {
            (CUBLAS_OP_T, k as i32)
        } else {
            (CUBLAS_OP_N, n as i32)
        };
        let (op_a, lda) = if lhs_t {
            (CUBLAS_OP_T, m as i32)
        } else {
            (CUBLAS_OP_N, k as i32)
        };
        let st = unsafe {
            cublasGemmEx(
                self.handle,
                op_b,
                op_a,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                rhs as *const c_void,
                CUDA_R_32F,
                ldb,
                lhs as *const c_void,
                CUDA_R_32F,
                lda,
                &beta,
                out as *mut c_void,
                CUDA_R_32F,
                n as i32,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != 0 {
            anyhow::bail!("cublasGemmEx({m}x{n}x{k}) failed: {st}");
        }
        Ok(())
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.handle) };
    }
}
