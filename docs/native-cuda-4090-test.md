# Testing the native-CUDA biped perf wins on a 4090

Goal: reproduce / re-measure the native-CUDA optimizations (dispatch reductions,
fixed-grid default, cooperative multibody kernels) on an **RTX 4090** and confirm
they're **bit-identical** (gather err 0.0). Numbers were developed on a 5090; the
4090 has fewer SMs (~128 vs ~170) and less bandwidth (~1 TB/s vs ~1.7), so expect
lower *absolute* env/s but the *relative* wins and bit-exactness should hold.

## Branches / getting the code

- `zealot` → branch **`feat/native-cuda-e2e-bench`** (pushed to origin).
- nexus physics = the **`nexus-rl`** repo (`git@github.com:haixuanTao/nexus-rl.git`).
  The changes are pushed there as branch **`feat/per-env-parallelism`**, based off
  **`cuda-oxide`** (the local dir may be named `nexus-cuda`, but the live remote is
  `nexus-rl`; the old `nexus-cuda` GitHub remote is archived). On the box:
  ```sh
  cd <nexus-rl checkout>          # the dir cloned from haixuanTao/nexus-rl
  git fetch origin                # or whatever remote points at nexus-rl
  git checkout feat/per-env-parallelism
  ```
  Note it's based off `cuda-oxide`, **not** the default `cuda-build` (which has
  extra motor-staging work) — so this branch doesn't include those. If you need
  both, rebase onto `cuda-build` and resolve the `solver.rs` / `multibody/mod.rs`
  overlap.
  - Fallback (if the box's nexus checkout is the *archived* `nexus-cuda` at base
    `29fac1e`, not `nexus-rl`): apply the bundled patches instead —
    `git am ../zealot/docs/nexus-cuda-patches/*.patch`.

`khal` and `vortx` are **local dimforge forks, not under git** — the box must
already have them as sibling dirs (same parent, e.g. `~/Documents/work/`). One
small khal edit is needed — see "khal caveat" below.

## Prereqs on the box

The native-CUDA path compiles the `#[spirv]` shaders to PTX via **cuda-oxide**
(rust-gpu → LLVM → ptxas), so it needs that toolchain. The build scripts in
`build_cuda/` are `$HOME`-relative on `nexus-rl` (a "portable across boxes"
commit already landed there), but they still assume the cuda-oxide toolchain +
the NVVM/ptxas/libdevice wheels live at the expected `$HOME` locations (LLVM
tools from a nightly rustup toolchain, `ptxas`, `libdevice.10.bc`,
`librustc_codegen_cuda.so`, libNVVM). **Verify the vars at the top of each script
match this box** before running.

## ⚠️ The #1 gotcha: arch is `sm_120` (5090) → use `sm_89` (4090)

The cubin build scripts hardcode Blackwell `sm_120`. For Ada (4090) change every
`sm_120` to `sm_89`:

- `nexus-cuda/build_cuda/build_nexus_cubin.sh`: `llc -mcpu=sm_120` and
  `ptxas -arch=sm_120`
- `nexus-cuda/build_cuda/build_vortx_cubin_only.sh`: the `make_cubin` invocation
  passes `sm_120` (and uses libNVVM `CUDA_OXIDE_UNROLL_LOOPS`) — pass `sm_89`.

(If a kernel uses an sm_120-only feature it'll fail at ptxas; none should here —
it's plain FP32 SIMT.)

## khal caveat (un-versioned)

The bench calls `khal::backend::cuda::dump_kernel_profile()` — a local profiling
helper added to `khal/crates/khal/src/backend/cuda.rs` that is **not in the
dimforge fork** and can't be pushed (khal isn't a git repo). Two options:

- **Simplest (skip profiling):** in `zealot/examples/biped/iter_e2e_bench.rs`,
  delete the two lines
  ```rust
  #[cfg(feature = "cuda_backend")]
  khal::backend::cuda::dump_kernel_profile();
  ```
  The bench then builds + runs fine; you just lose `KHAL_CUDA_PROFILE`.
- **Keep profiling:** apply the patch in Appendix A to the box's `khal`.

## Build

The two cubins must be embedded into the host binary via **per-crate** env vars
(the generic `CUDA_OXIDE_SHADERS_PTX` would inject one cubin into both crates →
`gemm_naive not found`). Cubins land in `$HOME/nexus_ptx/`.

```sh
cd nexus-cuda
# 1) build the nexus rbd shader cubin (rebuilds from the modified shader source)
bash build_cuda/build_nexus_cubin.sh          # after the sm_89 edit; ends "EMBED HASH MATCH OK"
# 2) build the vortx (policy/gemm) shader cubin
bash build_cuda/build_vortx_cubin_only.sh     # after the sm_89 edit

# 3) build the bench, embedding BOTH cubins
cd ../zealot
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
touch ../nexus-cuda/crates/nexus_rbd3d/build.rs   # force re-embed of the fresh cubin
BIPED_CUDA=1 cargo build --release --example iter_e2e_bench --features "gpu biped_gpu cuda_backend"
```

If you change a shader, re-run step 1 (or 2) then step 3. Plain host changes only
need step 3.

## Run + what to check

Args are `<num_envs> <T-steps> <epochs> <minibatches>`. Principal config is
`<N> 32 5 16`. Fixed-grid is the **default on CUDA** now (no flag needed);
`BIPED_CAPTURE=1` adds CUDA-graph capture of the rollout.

```sh
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin

# main number (capture path = the headline column)
BIPED_CAPTURE=1 BIPED_CUDA=1 ./target/release/examples/iter_e2e_bench 8192 32 5 16
# sweep N: 512 1024 2048 4096 8192
```

Check in the output:
- **`(gather verified, err 0.000e0)`** — this is the bit-exactness check vs the CPU
  reference. **If err is non-zero on the 4090, that's the interesting finding**
  (a real arch-dependent bug); flag it.
- `FULL GPU ... = NN k env/s` — the throughput.

Optional diagnostics:
- `KHAL_CUDA_PROFILE=1 BIPED_CUDA=1 ... 8192 4 1 2` → per-kernel GPU time table
  (serialized; use ratios, not absolutes). Skips the capture path automatically.
- `BIPED_IDLE_DBG=1 BIPED_CAPTURE=1 BIPED_CUDA=1 ... 8192 16 1 4` → splits the
  rollout into `physics_encode` / `gpu_wait(compute)` / readback / cpu. On CUDA
  the default should show `physics_encode ≈ 25 ms` (fixed-grid working) and
  `gpu_wait` dominating. If `physics_encode` is ~1000+ ms, fixed-grid isn't
  active (check `BIPED_FIXED_GRID` isn't set to 0).
- `BIPED_FIXED_GRID=0` forces the old indirect path (the regression baseline);
  `=1` forces fixed-grid.

## 5090 reference (principal config, capture, bit-identical)

| N | before (this work) | after |
|---|---|---|
| 512 | 24.1k | 34.7k |
| 1024 | 34.3k | 49.7k |
| 2048 | 42.1k | 59.2k → 61.0k |
| 4096 | 44.3k | 63.0k |
| 8192 | 45.0k | 63.6k |

"before" = the committed baseline before the per-env-parallelism work; "after" =
current branches. The 4090 absolutes will be lower; what matters is (a) the wins
reproduce as a *ratio* and (b) `gather err 0.0` holds. Quick before/after on the
box without rebuilding the old cubin: compare `BIPED_FIXED_GRID=0` (closer to the
old indirect path) vs default, and eyeball that the cooperative kernels show up
small in `KHAL_CUDA_PROFILE` (`gpu_mb_*` no longer dominating).

## What changed (so you know what you're testing)

- 3 `threads(1)` dispatch-arg kernels (serial N-batch scan) → parallel reductions.
- Fixed-grid dispatch is the CUDA default (kills the indirect host round-trip).
- Cooperative `threads(32)` multibody kernels: `finalize_contact`,
  `init_solve_joint` (serial-walk → parallel back-solve), joint+contact PGS
  (cooperative Gauss-Seidel apply), `gpu_mb_integrate`.
- All bit-identical; remaining levers (init_contact, the block-per-articulation
  megakernel, tensor-core GEMM) are noted in `README.md` "Next levers".

---

## Appendix A — the khal profiling patch (optional)

Add to `khal/crates/khal/src/backend/cuda.rs`. Near the top (after the
`use std::sync::{...}` line, make it `{Arc, Mutex, OnceLock}`):

```rust
static KERNEL_PROFILE: OnceLock<Mutex<HashMap<String, (u64, u128)>>> = OnceLock::new();
fn kernel_profile() -> &'static Mutex<HashMap<String, (u64, u128)>> {
    KERNEL_PROFILE.get_or_init(|| Mutex::new(HashMap::new()))
}
/// Print accumulated per-kernel timings (sorted desc) and clear. No-op if unused.
pub fn dump_kernel_profile() {
    let mut map = kernel_profile().lock().unwrap();
    if map.is_empty() { return; }
    let mut rows: Vec<_> = map.iter().map(|(k,(c,ns))| (k.clone(),*c,*ns)).collect();
    rows.sort_by(|a,b| b.2.cmp(&a.2));
    let total: u128 = rows.iter().map(|r| r.2).sum();
    eprintln!("\n=== KHAL_CUDA_PROFILE: {} kernels, {:.3} ms total (serialized) ===",
              rows.len(), total as f64/1e6);
    for (n,c,ns) in &rows {
        eprintln!("{:>9.3} ms  {:>6}x  {:>9.2} us  {:>6.2}%  {}",
            *ns as f64/1e6, c, (*ns as f64/ *c as f64)/1e3, *ns as f64/total as f64*100.0, n);
    }
    map.clear();
}
```

In `fn launch(...)`, wrap the `builder.launch(cfg)` call:

```rust
let prof_start = if std::env::var_os("KHAL_CUDA_PROFILE").is_some() {
    self.stream.synchronize()?;
    Some(std::time::Instant::now())
} else { None };
unsafe { builder.launch(cfg)?; }
if let Some(t0) = prof_start {
    self.stream.synchronize()?;
    let ns = t0.elapsed().as_nanos();
    let mut map = kernel_profile().lock().unwrap();
    let e = map.entry(self.function.name.clone()).or_insert((0, 0));
    e.0 += 1; e.1 += ns;
}
```
