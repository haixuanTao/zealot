# Native-CUDA (cuda-oxide) build of zealot on the vast.ai 5060 box — build log

Live log of standing up the native-CUDA backend (`BIPED_CUDA=1`) for zealot on a
fresh vast.ai box, so it survives if the attempt stalls. Recipe source:
`nexus-cuda/build_cuda/SETUP.md` + `README.md` + `build_nexus_cubin.sh`.

## Target box
- `ssh -p 16199 root@ssh1.vast.ai` (vast.ai, container, root, passwordless-sudo limited)
- GPU: **RTX 5060, sm_120 Blackwell, 8 GB**, driver 580.95.05
- System CUDA: **13.0** (nvcc V13.0.88) — NOT used for the build; we download CUDA 12.9 wheels userspace (system CUDA only needs the driver for runtime).
- 12 cores, 125 GB RAM, **32 GB disk** (the tight constraint — monitor `df -h /`).
- Fresh: no rust/repos at start. GitHub SSH from the box: NOT authorized (clone public via https; private forks come via rsync from Mac).

## Why CUDA-native here
Mac is Metal-only and its WebGpu contact path is broken (see `metal-contact-bug-proposal.md`). The 5090 boxes (baguette/champagne) run native CUDA but were **unreachable** this session (`172.18.128.x:22` timeout), so no cubin shortcut — full toolchain rebuild on the box.

## Recipe (from nexus-cuda/build_cuda/SETUP.md)
Repos as siblings under `~/Documents/work/` + `~/cuda-oxide-src`:
- haixuanTao/cuda-oxide @ `feat/nexus3d-vortx-native-cuda` → `~/cuda-oxide-src` (PUBLIC; clone on box)
- khal @ feat/cuda-oxide-backend, vortx @ feat/gpu-policy-shaders, nexus-cuda, parry, rapier, zealot → rsync from Mac (latest state, incl. uncommitted fixes)

Toolchain: nightly-2026-04-03 + rust-src/rustc-dev/llvm-tools; LLVM 21.1.0 at `~/llvm21`; CUDA 12.9 wheels (`nvidia-cuda-nvcc-cu12==12.9.86` → libnvvm/libdevice/ptxas, `nvidia-nvjitlink-cu12==12.9.86`).

Build: backend `.so` (cuda-oxide/crates/rustc-codegen-cuda) → cubins via `build_nexus_cubin.sh` (llc+ptxas, libdevice linked) and the vortx cubin script → embed via `CUDA_OXIDE_SHADERS_PTX_*` → build zealot `--features "gpu biped_gpu cuda_backend"` → run `BIPED_CUDA=1`.

CRITICAL flags (in the scripts): `-Zmir-enable-passes=-JumpThreading` (barrier-deadlock fix, REQUIRED), `-Zalways-encode-mir`, clean build, `--features "cuda-oxide dim3 unsafe_remove_boundchecks" --target nvptx64-nvidia-cuda -Z build-std=core`.

## Progress

### 2026-06-24
- Box probed: 5060 / CUDA 13.0 / 32 GB disk / fresh. ✅
- rustup + **nightly-2026-04-03 installed** (rustc 1.96.0-nightly 55e86c996). ✅
- cuda-oxide fork branch confirmed PUBLIC/cloneable via https (ref f0f9494). ✅
- Tasks + this log created.
- **Repos ON BOX** ✅: rsync'd zealot/khal/vortx/nexus-cuda/parry/rapier to `~/Documents/work/` (note: macOS rsync 2.6.9 rejects `--info=progress2` — drop it). cuda-oxide cloned to `~/cuda-oxide-src`. Verified patched files present (khal-derive/spirv_bindgen/cuda.rs, cuda-oxide shader features) AND uncommitted Mac work carried over (bug#1 `unwrap_or(true)`, `dbg_mb_contacts`). `git config --global --add safe.directory '*'` to silence post-rsync ownership warnings.
- Toolchain bootstrap (detached `~/bootstrap.sh` → `~/bootstrap.log`): rust components ✅, cuda-oxide clone ✅, **LLVM 21 extracting** (disk 9.7G/32G), CUDA 12.9 wheels queued.
- Toolchain bootstrap **DONE** ✅: LLVM 21.1.0 at `~/llvm21`; CUDA 12.9 wheels extracted (`~/nvvm-wheel/.../{libnvvm.so,libdevice.10.bc,bin/ptxas}`, `~/nvjit-wheel/.../libnvJitLink.so.12`). `~/llvm21/bin` has llvm-as/llvm-link/opt/llc (LLVM 21, matches cuda-oxide IR).

### ⛔ BLOCKER — cuda-oxide backend `.so` does NOT build against its pinned pliron
- `cd ~/cuda-oxide-src/crates/rustc-codegen-cuda && cargo +nightly-2026-04-03 build` → **7 errors, all in `crates/mir-lower/src/convert/ops/cast.rs` (~L517–522)**:
  - `E0433: cannot find module or crate dialect_llvm` (stale import path)
  - `E0308: AddrSpaceCastOp::new(ctx, val, d_as)` passes a `u32` (address space) where pliron `af1b7d0`'s `CastOpInterface::new(ctx, operand, res_type: Ptr<TypeObj>)` now wants a result **type ptr**.
- **NOT a version float / not the tip's fault:** pliron is rev-pinned to `af1b7d0` identically at the tip `f0f9494` AND at the "compile & run end-to-end" commit `52d5765` (only diff between them = the `barrier_div` example). So `52d5765` fails identically.
- **Root cause:** the pushed fork's `mir-lower` is **stale vs the pinned pliron** — the cast-op API migration (u32 addr-space → `Ptr<TypeObj>` result type, + the `dialect_llvm`→? import) was a working-tree fix on **baguette that was never pushed**. The fork compiles the *shaders* (commit msg) using an already-built backend `.so`; the *backend source* as pushed doesn't rebuild.
- Localized to one file's cast lowering, BUT: fixing it blind = pliron-API archaeology, and once it passes there may be further unpushed fixes downstream (the build stopped at the first failing crate).

### Resolution options
1. **Baguette (most reliable):** copy baguette's prebuilt `~/cuda-oxide-src/.../target/debug/librustc_codegen_cuda.so` (skips building the backend entirely — the cubins only need the `.so` + the wheels, both then present), OR rsync baguette's *complete* working `~/cuda-oxide-src`. Needs baguette reachable.
2. **Patch `cast.rs`** to the new pliron API (construct the pointer result type for AddrSpaceCast; fix `dialect_llvm` path). Risky — compiler internals, possible cascade of further unpushed fixes.
3. **Pin pliron older** to the rev `mir-lower` was written against. Risky — Cargo.toml warns "MUST stay on af1b7d0"; other crates may need af1b7d0.
- **Recommendation:** option 1. The blocker is unpushed baguette state, so baguette is the clean source of truth (its prebuilt `.so` is the shortcut).

### FIX APPLIED (option 2) — patched `cast.rs` to the pinned pliron API
User chose to patch. The 7 errors were 2 trivial migrations the rest of `mir-lower` already had; `cast.rs` was just missed in the rename/API bump. Backup: `/tmp/cast.rs.bak` on box.
1. **`dialect_llvm` → `llvm_export`** (6 sites): the LLVM-dialect crate was renamed to `llvm_export`; the file itself already uses `llvm_export::types::StructType` at L375/451. `perl -pi -e 's/dialect_llvm::types::/llvm_export::types::/g'`.
2. **`AddrSpaceCastOp::new` result-type** (2 sites, L400 & L522): pliron's `CastOpInterface::new(ctx, operand, res_type: Ptr<TypeObj>)` now takes a result type, not the dest address space `u32`. Proven pattern at `common.rs:93` / `cast.rs:569`: build it with `PointerType::get(ctx, d_as).into()`. Inserted `let cast_ty = llvm_export::types::PointerType::get(ctx, d_as).into();` before each call and passed `cast_ty`. (`common.rs:21: use llvm_export::types as llvm_types;` confirms the module path.)
- Verified: 0 `dialect_llvm` left, 2 `cast_ty` builds, 0 bad calls. → `mir-lower` compiles. ✅

### FIX 2 — `mir-importer` `float_math.rs` was corrupted (dropped merge hunk), RECOVERED from git
- After `mir-lower` passed, `mir-importer` failed: "unclosed delimiter" in `translator/terminator/intrinsics/float_math.rs` (50 open / 47 close braces). `from_fast_intrinsic_path` ended abruptly after the `frem_fast` arm, and `is_libm_path` (called from `mod.rs:1176` + 2 sites in-file) was **undefined**.
- Root cause: the wholesale commit `52d5765` made a **pure deletion** of 83 lines from this file (a botched merge), removing `_ => None,`+closing braces, the entire `placeholder_callee` fn, and `is_libm_path`. Confirmed via `git diff 097274c HEAD -- <file>` = **add=0 del=83**.
- Recovery (no inference): `git show 097274c:<file> > <file>` — the parent version is brace-balanced (53/53), 721 lines, `is_libm_path` present. Backup `/tmp/float_math.corrupt.bak`.
- **Proactive scan** of all files `52d5765` touched in mir-importer/mir-lower/llvm-export: only `float_math.rs` was a pure deletion (add=0); the rest (`mod.rs` +158/-34, `types.rs` +16/-46, `lib.rs` +26/-0, `cast.rs` +95/-11) are intentional changes. So no other pure-deletion corruptions expected.
- Rebuilding. NOTE: the pushed fork is in a partially-broken state; each fix is verifiable (compiles / git-recovered).

### FIX 3 — `mir-importer/types.rs`: 2 `get_with_full_layout` calls missing `abi_align`
`dialect_mir::MirStructType::get_with_full_layout` gained an 8th param `abi_align: u64`; 2 placeholder-struct callers (types.rs ~L1022/1033, `vec![],vec![],vec![],vec![], 0,`) still passed 7. Added `0` (the wrapper `get_with_layout` itself passes `0,0`). Backup `/tmp/types.rs.bak`.

### FIX 4 — `rustc_codegen_cuda/collector.rs`: dangling `has_invalid_chars`
`compute_export_name` no longer computes `has_invalid_chars` (logic now "always export mangled symbol"), but `52d5765` left `let _ = (has_invalid_chars, has_generic_args);` referencing it. Changed to `let _ = has_generic_args;`. Backup `/tmp/collector.rs.bak`.

### ✅ BACKEND `.so` BUILT (EXIT 0)
`librustc_codegen_cuda.so` (298 MB) at `~/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/`. Four small fixes total (2 pliron-API misses, 1 git-recovered dropped hunk, 1 dead-var) — all in unpushed-baguette-state territory, none deep. **Patch-forward (option 2) worked.**

### Cubins building
Agnostic script `~/build_cubins.sh` (all `$HOME`-relative, sm_120; addresses the "make it agnostic" ask): builds `nexus_rbd_shaders3d` + `vortx_shaders` via cuda-oxide → `.ll` → llvm-as/llvm-link(libdevice)/opt/llc(sm_120)/ptxas → `~/nexus_ptx/*.cubin`. Uses `~/llvm21` tools (LLVM 21, matches the IR) + wheel libdevice/ptxas. Launched (PID 5184).

### FIX 5 — `khal-std` cuda-device dep (Mac keeps it commented; build box needs it)
First cubin run failed: `khal-std` wouldn't compile for nvptx (`E0432`, `cuda_device` crate missing) → both shader builds cascaded (no `.ll`). Cause: the Mac's `khal/crates/khal-std/Cargo.toml` keeps the `cuda-device` path-dep **commented out** (line 64) so the Mac stays host-buildable; the build box must uncomment it. `~/cuda-oxide-src/crates/cuda-device` exists. Fixes (backup `/tmp/khalstd_cargo.bak`):
- L41: `cuda-oxide = ["dep:num-traits"]` → `cuda-oxide = ["dep:num-traits", "dep:cuda-device"]`
- L64: uncomment → `cuda-device = { path = "/root/cuda-oxide-src/crates/cuda-device", optional = true }`
Verified: `khal-std` builds for `--target nvptx64 --features cuda-oxide` (Finished, only libm warnings). Re-ran cubin build.
> If reproducing on another box: same uncomment, with that box's absolute cuda-oxide path. (This is the one piece SETUP.md glosses as "added on the build box".)

### FIX 6 — `khal-std` cuda-oxide `Float` trait missing `cos`/`sin`
Next cubin run: `vortx-shaders` failed `E0599: no method named cos found for f32` at `linalg/sample.rs:97` (Box-Muller Gaussian sampler). khal-std's cuda-oxide `Float` trait (`num_traits.rs`, a minimal libdevice-backed trait) implemented exp/ln/sqrt/powf/floor/ceil/abs/max/min/atan but **not cos/sin**. Added them following the same `core::intrinsics::$fn` pattern (→ libdevice `__nv_cosf/__nv_sinf`): extended the trait decl, the `float_impl!` macro (sig + 2 impls), and both f32/f64 invocations (`cosf32,sinf32` / `cosf64,sinf64`). 5 string edits, each verified unique. Backup `/tmp/num_traits.rs.bak`. Re-ran cubin build.
- Tally so far: 6 small fixes (4 backend pliron/dead-var + git-recovery, 2 khal-std cuda-oxide-feature gaps). All "unpushed baguette state". Patch-forward holding up.

### ⛔ BIG WALL — nexus_rbd_shaders3d: 82 errors (the cuda-oxide port isn't on this branch)
After fix 6, vortx got past `cos`; nexus then surfaced **82 errors** + a backend codegen gap. Breakdown:
- vortx: 1 backend codegen gap — `mir.construct_struct` ZST `TryFromIntError` (the `step_by spec_next` ZST-with-0-operands issue; my notes say this was FIXED on baguette in the backend's `rvalue.rs translate_zero_sized_constant_value`, but that fix is NOT in the pushed cuda-oxide fork). Needs a backend `.so` patch + rebuild.
- nexus: 82 errors (61 E0308, 15 E0599, 3 E0782, 3 E0277) + the same `step_by` PTX-verification gap. Concentrated in the **hardest parallel kernels**: `utils/radix_sort/*` (56), `broad_phase/lbvh.rs` (12), `utils/prefix_sum.rs` (6), `dynamics/multibody/{jacobian,mass_matrix,lu,...}`. Error types (`SmemBuf<_>: MaybeIndexUnchecked not satisfied`, `.read()` on `&f32`/`&mut f32`, if/else type mismatch) = the cuda-oxide shared-memory adaptations are MISSING.
- **Root cause:** the 2 recent contact commits touched **0** shader files (`git log 7214111~1..ef501d3 -- src_rbd_shaders` = empty), so it's NOT the new work. It's that the **full cuda-oxide nexus port (radix_sort/prefix_sum/lbvh/multibody shared-mem genericity) was never migrated to the `nexus-rl feat/per-env-parallelism` branch** the box uses — only a partial (step-23) migration happened per the notes. The complete port lived on baguette / the archived `nexus-cuda` repo.
- This is exactly the "volume grind" the notes flagged: parallel radix sort, prefix scan, BVH build, LU factorization — the gnarliest GPU kernels. Reconstructing 82 fixes by inference ≠ the 6 small fixes so far; multi-hour, real risk of subtle-but-compiling bugs, plus 2 backend codegen gaps needing `.so` rebuilds.

### Status: toolchain + backend = DONE (reusable). nexus shaders = blocked on the unmigrated port.
Options: (1) baguette — rsync its complete cuda-oxide-ported nexus shaders (reliable; needs baguette reachable). (2) Patch-forward the 82 (+backend gaps) — multi-hour, uncertain. (3) Stop at the milestone — the hard, reusable toolchain+backend is proven & documented; the shader port to this branch is a separate large task.

### ★ RESOLUTION — the working cuda-oxide port lives on dedicated BRANCHES (no hand-patching)
The 82-error grind was avoidable: the verified port is a coherent **branch set** on the haixuanTao forks (the `feat/per-env-parallelism` etc. branches the box had simply diverged from it):
- **nexus → `haixuanTao/nexus-rl` `cuda-oxide-curated`** (efc9ba8, 2026-06-08; "Merge portable-cuda-build-rl"; ships `build_cuda/detect_env.sh`). This is the snapshot my notes say ran full e2e CUDA training (~2× WebGPU) on baguette.
- **khal → `haixuanTao/khal` `fix/cuda-oxide-arbitrary-element-shaders`** (6007d03, 2026-06-16). Adds the generic `pub struct SmemBuf<T: Copy, const N: usize>` + `impl<T,const N> MaybeIndexUnchecked<T> for SmemBuf<T,N>` — exactly the `SmemBuf<u32>` support the E0277s needed. (The box's khal `fix/cuda-slice-arg-element-count` had only `SmemBuf<N>` = f32.)
- **vortx → `haixuanTao/vortx` `feat/gpu-policy-shaders`** (33bf742, 2026-06-08).
- cuda-oxide backend `.so`: keep the one already built (fork tip + my 6 fixes).
Coherence: nexus/vortx shader SOURCE doesn't reference `SmemBuf` directly — khal-derive's generator emits it — so khal drives the `SmemBuf` signature; pairing arbitrary-element khal with the June-8 nexus/vortx is consistent.
Method: `git worktree` checkouts on the Mac (no disturbance to working trees) → rsync to box (`--delete`, exclude .git/target) replacing nexus-cuda/khal/vortx. Trade-off: June-era physics snapshot (lacks the newest contact-solver tweaks), but it's the version proven to RUN on CUDA = the goal. Keeping latest zealot; will align only if its nexus3d/khal host-API needs it.
- Tradeoff noted for later: forward-port the June-23 contact-solver fixes onto cuda-oxide-curated once running, or forward-port the cuda-oxide shader fixes onto per-env-parallelism.

### Coherence correction — khal must be `feat/cuda-oxide-backend`, NOT arbitrary-element
First attempt with arbitrary-element khal (June 16) failed Cargo resolution: curated nexus's `nexus_fem3d` needs `khal/metal`, but arbitrary-element dropped that feature. The SmemBuf<u32> errors that pushed me to arbitrary-element came from the *newer* (per-env-parallelism) nexus — the **curated nexus uses no `SmemBuf<u32>`** (grep empty), so it needs only f32 `SmemBuf<N>`. Switched khal → **`feat/cuda-oxide-backend`** (1e1e661, has `metal` feature + f32 `SmemBuf<N>`) per the curated SETUP.md. Coherent June-8 set:
- nexus `cuda-oxide-curated` + khal `feat/cuda-oxide-backend` + vortx `feat/gpu-policy-shaders` + cuda-oxide fork(+my 6 backend fixes).
- Re-applied cuda-device dep fix. (feat/cuda-oxide-backend lacks cos/sin; will re-add only if the June-8 vortx needs it.)
- LESSON: don't mix branch eras — use the exact set the curated `build_cuda/SETUP.md` prescribes.

### FIX 7 — khal version skew (0.1 vs 0.2): bump consumers to `0.2`
`khal/metal` still failed: the resolver pulled **registry khal 0.1.1** (no `metal`) because the local khal is **0.2.0** while curated nexus/vortx/zealot require `khal = "0.1"` — `0.2.0` is outside `^0.1`, so the `[patch.crates-io]` couldn't override. And the cuda-oxide khal support only exists at **0.2.0** (added 2026-06-16, after the May-29 `0.2.0` bump — there is NO `0.1.x` khal with cuda-oxide). So the curated nexus's `0.1` requirement is simply stale. Fix: bumped `khal/khal-std/khal-derive = "0.1"→"0.2"` and `khal-builder = "0.1.1"→"0.2.0"` in nexus-cuda + vortx + zealot `Cargo.toml`, kept local khal at its real `0.2.0`, cleared stale `Cargo.lock`s. Now the local `0.2.0` khal (with `metal` + cuda-oxide) satisfies via the patch. Backups `/tmp/{nexus-cuda,vortx,zealot}_cargo.bak`.
- This (and FIX 5/6) is the cost of the pushed branches not being a version-coherent set; baguette's local working trees were coherent, the forks drifted.

### FIX 8 — glam math backend; FIX 9 — glamx 0.2 vs 0.3 alignment
- **glam:** `glam = "=0.32.1"` in khal-std had `default-features=false` and NO math backend → `compile_error! "You must specify a math backend"`. Added `features=["libm","scalar-math"]` (scalar-math also preempts the documented x86-SSE2 union-cast issue). `/tmp/khalstd_glam.bak`.
- **glamx:** June-8 nexus/vortx declare glamx `0.2`, June-16 cuda-oxide khal uses glamx `0.3` → `UVec3`(0.2) vs `UVec3`(0.3) = ~16 E0308 in vortx-shaders. Bumped all 6 nexus/vortx glamx `0.2→0.3` (+ added `u32,i32,f64` features to the shader crates to match khal-std). Cleared locks.
- These are pure branch-era version skew (June-8 shaders vs June-16 khal). Grinding them per the user's call; each is a version bump, not logic.

### ⛔⛔ CONCLUSIVE WALL — no coherent cubin build exists among the PUSHED branches
Tested ~6 branch combinations; all fail on version coherence, tracing to ONE coupling:
- The **generic `SmemBuf<T,N>`** (arbitrary-element khal — the only thing that fixes the nexus `SmemBuf<u32>` errors) is **hard-coupled to glamx 0.3** (khal-std needs glamx features `u32/i32/f64`, which **glamx 0.2 lacks** — proven: `khal-std depends on glamx with feature f64 but glamx 0.2 does not have that feature`).
- Every nexus/vortx **shader** branch (June-8 curated AND latest per-env-parallelism) uses **glamx 0.2**, and **bumping them to 0.3 breaks the shaders** (16→35 errors).
- So: generic-SmemBuf ⟺ glamx 0.3; shaders ⟺ glamx 0.2; mutually exclusive. No pushed khal pairs cuda-oxide + generic-SmemBuf + glamx 0.2; no pushed nexus/vortx pairs glamx 0.3 + working shaders.
Combinations tried: (curated nexus + feat/cuda-oxide-backend khal), (curated + arbitrary-element khal), (latest nexus/vortx + arbitrary-element khal @0.3), (latest + arbitrary-element @0.2). Each fails differently (metal feature / glamx u32 type mismatch / glamx feature-missing / khal version skew).
**Root cause:** the working build was baguette's LOCAL trees + its `Cargo.lock` (cuda-oxide khal at glamx 0.2 with the generic-SmemBuf cherry-picked — a state never pushed in combinable form). The forks drifted apart afterward.

### Conclusion + handoff
- **DONE & reusable:** the box has the full toolchain (nightly + LLVM21 + CUDA 12.9 wheels) and the **cuda-oxide backend `.so`** (8 source fixes, all documented above). This is the hard 80%.
- **Blocked:** the cubins need a version-coherent nexus+khal+vortx set, which only baguette has (its `Cargo.lock` + working trees). **Unblock = `rsync -a baguette:~/{Documents/work/{nexus-cuda,khal,vortx},nexus_ptx} <box>`** — its prebuilt `~/nexus_ptx/*.cubin` (sm_120, same as the 5060) would even skip the cubin build entirely.
- Grinding pushed-branch versions is **proven non-convergent** — stop until baguette is reachable.
- Box backups of every edit: `/tmp/*.bak`. Box is scratch; re-rsync from baguette will overwrite cleanly.

## Known landmines (from notes — watch for these)
- 32 GB disk: LLVM21 (~5G) + toolchain + 6 repos + target dirs (~15G) + wheels (~2G) ≈ tight. Monitor.
- CUDA 13 system vs 12.9 wheels: wheels are userspace, should isolate; driver supports sm_120.
- `build_nexus_cubin.sh` has **hardcoded /home/baguette paths + a nightly-2025-08-04 llvm-tools path** — must edit for the box (use ~/llvm21 tools or matching llvm-tools; the .ll is LLVM-21 IR so the assembler must be LLVM 21).
- step-1 barrier deadlock if `-JumpThreading` flag missing; step-23 contact ILLEGAL_ADDRESS was fixed (29fac1e).
- `make_cubin` tool is baguette-only; `build_nexus_cubin.sh` avoids it (llc+ptxas path) — prefer that.
