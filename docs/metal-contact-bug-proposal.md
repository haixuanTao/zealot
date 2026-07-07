# Proposal: fix the WebGpu/Metal nexus contact bugs (so the biped sim works off-CUDA)

> ## ✅ RESOLVED (2026-07-06)
>
> **Root cause: a naga 29 MSL-backend loop miscompile — not the solver, not
> dispatch ordering.** naga's MSL writer hoists a loop's `continuing` block to
> the top of the next iteration and *re-evaluates* the `break_if` condition
> expression there — after the continuing block has already advanced the loop
> phis. rust-gpu `while` loops therefore exit one body-execution early on
> Metal, dropping the final iteration's stores. In the multibody solve
> kernels the per-lane `J·v` loops (`while i < ndofs { …; i += 32 }`, exactly
> one iteration per lane) executed **zero** times, so `J·v = 0` → zero
> impulses from BOTH the contact PGS sweep and the joint/PD sweep, while
> gravity still integrated per TGS iteration (hence −2·g·dt after one step,
> and `BIPED_SOLVER_ITERS=16` → 16× worse). The robot free-fell ~8 mm/step,
> then the accumulated-penetration bias launched it → bounce → NaN.
>
> **Diagnosis trail** (all with `BIPED_DECIMATION=1`, seed `0xC0FFEE`,
> `BIPED_SPAWN_DR=0`, CUDA on the vast 5060 box as golden):
> spawn poses matched to 1–2 ulp → narrow-phase dist matched (−5e-7) →
> Jᵀ / M⁻¹·Jᵀ / inv_lhs matched to ~1e-3 → but impulse was 0.1698 (CUDA) vs
> 0.0000 (Metal). An in-kernel lane-liveness stamp (each lane writes lane+1
> to shared memory, lane 0 sums) read 496 = Σ1..31 — the summation loop's
> last iteration was gone; a *direct* read of the "missing" slot returned the
> value. Translating the same SPIR-V with naga to WGSL (correct: `break if`
> uses the baked body value) vs MSL (broken: re-evaluates after phi update)
> pinpointed the writer bug.
>
> **Fix: `../naga-fixed`** (vendored naga 29.0.4, applied via
> `[patch.crates-io]` in zealot's Cargo.toml): the MSL writer now snapshots
> the `break_if` condition value into a pre-declared bool at every `continue`
> site / body fall-through and tests the snapshot in the hoisted gate
> (`src/back/msl/writer.rs`, `Statement::Loop` + `Statement::Continue`).
> Upstream-worthy (gfx-rs/wgpu).
>
> **Verified on Metal:** contact_probe step-0 impulse 0.1697 vs CUDA 0.1698;
> dof_state matches CUDA to ~1e-4; torso 0.718/0.716/0.715 identical to
> CUDA; broad-phase pairs stable [2,2,2,2] (the "flicker" was fallout of the
> wrong dynamics, not a broad-phase bug). `passive_stand` now behaves
> IDENTICALLY to CUDA (both fall over at ~step 40 — that's the model's real
> zero-action behavior, not a backend bug).
>
> **Bug #1 (indirect dispatch → 0 workgroups) was very likely the same
> miscompile** (the indirect-args kernels' loops), as were the LBVH
> "pass-ordering" symptoms. All khal/nexus-side workarounds from those
> misdiagnoses were reverted after the naga fix; the fixed-grid dispatch
> default stays (CUDA needs it regardless). Upstream fix:
> https://github.com/gfx-rs/wgpu/pull/9815 — the local `../naga-fixed`
> `[patch.crates-io]` vendor can be dropped once it ships in a wgpu release.

**Audience:** an agent on a 5090 box (baguette/champagne) with a working CUDA nexus build.
**Why you:** the bug is a WebGpu-vs-CUDA divergence; CUDA is the known-good reference. You can generate golden intermediates I can't from a Mac.

---

## TL;DR

On the WebGpu/Metal backend the batched nexus biped physics has **no working floor** — every body free-falls. Two independent bugs:

1. **Indirect dispatch launches 0 workgroups on WebGpu** → narrow-phase + contact solver never run → no contacts. **FIXED** (default WebGpu to fixed-grid dispatch). Details below for context.
2. **With contacts running (fixed grid), the contact solve is numerically unstable on WebGpu** → the robot is launched upward and the sim NaNs within ~60 steps. **OPEN — this is the task.**

Bug #2 is subtle: contact generation, constraint setup, and the solve-kernel structure all look *correct* on inspection. It needs a **CUDA-vs-WebGpu intermediate diff** to localize. Prime suspect: the **articulated-body M⁻¹ columns** (`contact_constraint_columns`) coming out of the multibody LU solve.

---

## Repos / layout

- `zealot` (this repo): biped env + examples. Backend chosen at runtime: WebGpu default; `BIPED_CUDA=1` + `--features cuda_backend` → native CUDA.
- `nexus-cuda` (sibling, `../nexus-cuda`): the physics. Host = `src_rbd/`, shaders = `src_rbd_shaders/` (rust-gpu → spirv for WebGpu, → ptx/cubin for CUDA). The shader source is path-included by `crates/nexus_rbd*_shaders3d`; **editing `src_rbd_shaders/*` does NOT trigger a rebuild unless you `touch src_rbd_shaders/lib.rs`** (cargo doesn't track path-included files).

Deterministic seed `0xC0FFEE` → the spawn is identical across backends, so per-step intermediates are directly comparable CUDA vs WebGpu.

---

## Bug #1 (fixed, for context)

`narrow_phase` + contact-solver kernels are dispatched **indirectly** (workgroup count read from a GPU buffer derived from the broad-phase pair count). On WebGpu/Metal that indirect dispatch runs **0 workgroups** — the kernels never execute. Broad-phase (direct dispatch) works, so pairs are found but no contacts are generated → free-fall.

`nexus_rbd3d` gates this via `crate::dispatch_grid(indirect, fixed)` (`src_rbd/lib.rs`), defaulting fixed-grid ON for CUDA, OFF for WebGpu (the code *assumed* "WebGpu indirect is native/cheap" — false for Metal).

**Fix applied** (`src_rbd/lib.rs`, `set_fixed_dispatch_grid_default`): default fixed-grid `true` for both backends (`unwrap_or(is_cuda)` → `unwrap_or(true)`). CUDA unchanged. Verified: contacts now generate on WebGpu by default.

> The *real* root cause is khal's WebGpu indirect dispatch launching 0 workgroups. If you'd rather fix it there (to keep the indirect fast-path on WebGpu), that's the cleaner fix — but the fixed-grid default is correct and sufficient.

---

## Bug #2 (the task): contact-solve instability on WebGpu

### Symptom
With fixed-grid dispatch, contacts run, but `passive_stand` (zero action, no noise, no reset = pure PD hold) diverges:

```
step  0: torso ~0.72   (spawn, fine)
step 20: torso scatters min −28 .. max +20
step 60: NaN
```
The robot is **launched upward** (0.72 → 0.94 → 1.29 in the first control steps), then explodes. **More solver iterations make it dramatically worse** (`BIPED_SOLVER_ITERS=16` → min −971 vs −181) — i.e. the iteration *diverges* rather than converging.

### What is already RULED OUT (verified on Metal via instrumentation)
- **Contact generation is correct.** World-space normal ≈ `(0,0,−1)`; solver force = `−normal` = `+Z` (up, correct). Penetration shallow (`dist0 ≈ −0.004`).
- **Per-constraint values are sane.** Read back `MultibodyContactConstraint`: `inv_lhs = 1/(J·M⁻¹·Jᵀ)` ≈ 0.6–3.9 (normal slots 3.46, 3.93; tangents 1.2, 0.64), `impulse ≈ 0.17` N·s (right order to cancel gravity), `lin_jac` correct, `free_body_im = 0` (ground static), `rhs = 0` at rest.
- **The PGS solve kernel is structurally correct.** `gpu_mb_solve_contact_constraints` / `solve_contact_constraints_par` (`src_rbd_shaders/dynamics/multibody/contact_constraints.rs:678`): one workgroup per multibody, contacts solved **sequentially** (`for s in 0..count`), 32-lane tree reduction with **explicit `workgroup_memory_barrier_with_group_sync()`** after every step (lines 722/729/773/788). No missing-barrier / warp-sync assumption. Static ground ⇒ no cross-workgroup `solver_vels` race.

### Remaining hypothesis (prime suspect)
The values that are **NOT yet compared to CUDA** are the **articulated-body M⁻¹ outputs**:
- `contact_constraint_jacs` (Jᵀ rows)
- `contact_constraint_columns` (**M⁻¹·Jᵀ columns** — used at `contact_constraints.rs:782`, `dof_state[v] += delta * col`)

`inv_lhs` is a single scalar derived from these and looks plausible, but the **full M⁻¹ column vector can be wrong while the scalar still looks reasonable**, and a wrong column propagates a wrong velocity delta to every DOF each sweep → divergence that worsens with iterations. The M⁻¹ columns come from the multibody **LU / articulated-inertia** kernels (`src_rbd_shaders/dynamics/multibody/{mass_matrix,lu,gravity_and_lu,jacobian}.rs`) — exactly the area that needed ~10 fixes during the CUDA port (FMA/libdevice, scalarize-accumulator, padding-index). A spirv/Metal analogue there is the leading candidate.

Secondary candidates if the columns match: the `dof_state` velocity integration, or a double-application of contacts (rigid `TwoBodyConstraint` path AND multibody path both consuming the same contact).

---

## Plan: CUDA-reference diff

The deterministic spawn makes this a clean apples-to-apples diff. Goal: find the **first** intermediate that differs between CUDA (golden) and WebGpu, walking the pipeline in order.

### Step 0 — get the tooling
The Mac working tree (branch `feat/native-cuda-e2e-bench`) has uncommitted diagnostic tooling + the bug-#1 fix. Either sync that branch, or recreate (small):
- `zealot/examples/biped/contact_probe.rs` — builds the env, steps with zero action, reads back broad-phase pairs, narrow-phase manifolds, and multibody contact constraints; prints normals (local+world), `inv_lhs/rhs/impulse/lin_jac`.
- `zealot/examples/biped/biped_env_nexus.rs` — added `BipedNexusBatchEnv::dbg_contacts()`, `dbg_collision_pairs()`, `dbg_mb_contacts()` (read `state.dbg_*` / `state.multibodies_mut().dbg_contact_constraint*()` via `slow_read_buffer`).
- `nexus-cuda/src_rbd/dynamics/multibody.rs` — added `GpuMultibodySet::dbg_contact_constraints()` / `dbg_contact_constraint_count()` (read-only).
- `nexus-cuda/src_rbd/lib.rs` — the bug-#1 fix.
- `zealot/Cargo.toml` — `[[example]]` entries for `contact_probe` (+ `ppo_grad_parity`).

### Step 1 — does the bug even reproduce on the box's WebGpu (Vulkan)?
The 5090's WebGpu backend is Vulkan, not Metal — the instability may be **Metal-only**. Run on the box:
```
cargo run --release --example contact_probe --features "gpu biped_gpu" -- 4        # WebGpu/Vulkan
BIPED_CUDA=1 cargo run --release --example contact_probe --features "gpu biped_gpu cuda_backend" -- 4   # CUDA
```
- If Vulkan-WebGpu **also** diverges → you have a same-machine WebGpu-vs-CUDA diff. Best case.
- If Vulkan-WebGpu is **stable** → bug #2 is Metal-specific; use the **Metal baseline values below** as the "broken" side and CUDA as golden.

### Step 2 — expose and diff the M⁻¹ outputs (prime suspect)
Add a readback for the suspect buffers (mirror the existing `dbg_contact_constraints`):
```rust
// GpuMultibodySet (nexus-cuda/src_rbd/dynamics/multibody.rs)
pub fn dbg_contact_constraint_jacs(&self) -> &Tensor<f32> { &self.contact_constraint_jacs }
pub fn dbg_contact_constraint_columns(&self) -> &Tensor<f32> { &self.contact_constraint_columns }
```
Expose via a `BipedNexusBatchEnv` method (like `dbg_mb_contacts`) and print, for the first active foot-ground constraint, the full Jᵀ row and M⁻¹·Jᵀ column (length = ndofs).

Run CUDA and WebGpu for the same `0xC0FFEE` spawn, **step 0 only** (before divergence pollutes state), and diff:
1. **Narrow-phase manifold** (normal, `dist`, points) — rule out a geometry difference first.
2. **`contact_constraint_jacs`** (Jᵀ) — should be pure geometry; differences ⇒ jacobian kernel.
3. **`contact_constraint_columns`** (M⁻¹·Jᵀ) — **the suspect.** Differences here ⇒ articulated-inertia / LU bug (`mass_matrix.rs` / `lu.rs` / `gravity_and_lu.rs`). This is the most likely first divergence.
4. **`dof_state` velocities** before/after one PGS sweep — if columns match but this diverges, the bug is in the solve/integrate step.

### Step 3 — fix at the first divergence
Most likely outcome: the M⁻¹ columns differ → audit the multibody LU/mass-matrix spirv kernels for the same class of issue the CUDA port hit (accumulator scalarization, FMA/precision, struct-padding index). Patch, `touch src_rbd_shaders/lib.rs`, rebuild, re-diff until the columns match CUDA.

### Step 4 — verify
```
BIPED_SPAWN_DR=0 cargo run --release --example passive_stand --features "gpu biped_gpu" -- 256 300
```
Success = `torso` holds ~0.70, `fell_frac → 0` on WebGpu (matching CUDA). Then run a short `biped_train_gpu` on WebGpu and confirm it no longer collapses for lack of a floor.

---

## Metal baseline (the "broken" side), captured 2026-06-20

`contact_probe`, `BIPED_FIXED_GRID=1`, seed `0xC0FFEE`, 4 envs, step 0 (foot-ground, ground collider idx 13):

```
narrow-phase: world normal ≈ (0,0,−1)  force=+Z  dist0 ≈ −0.004 .. −0.030
MultibodyContactConstraint (link 6, kind=1 normal): inv_lhs=3.456  rhs=0.000  impulse=0.172  free_im=0.000  lin_jac=(0,0,−1)
                            (link 6, kind=1 normal): inv_lhs=3.927  rhs=0.000  impulse=0.197  lin_jac=(0,0,−1)
                            (link 6, kind=2 tangent): inv_lhs=1.208  impulse=0.000  lin_jac=(0,−1,0)
                            (link 6, kind=2 tangent): inv_lhs=0.637  impulse=0.000  lin_jac=(1,0,0)
passive_stand (zero action): torso 0.72 → 0.94 → 1.29 (launched up) → NaN by ~step 60
                             BIPED_SOLVER_ITERS=16 → far worse (min −971); knobs (CONTACT_NF/DR, IMPLICIT_CORIOLIS) don't help
```
`contact_constraint_columns` (M⁻¹·Jᵀ) was **not** captured (needs the Step-2 readback) — get the CUDA golden for it and compare; that's the likely smoking gun.

---

## Key file references
- Dispatch gate / bug-#1 fix: `nexus-cuda/src_rbd/lib.rs` (`dispatch_grid`, `set_fixed_dispatch_grid_default`)
- Narrow-phase dispatch (indirect): `nexus-cuda/src_rbd/broad_phase/narrow_phase.rs:45-92`
- Contact solve kernel: `nexus-cuda/src_rbd_shaders/dynamics/multibody/contact_constraints.rs:678` (`solve_contact_constraints_par`), `:796` / `:887` (entry points)
- Constraint init / M⁻¹ columns: `gpu_mb_init_contact_constraints` (same file, `:152`) + `mass_matrix.rs` / `lu.rs` / `gravity_and_lu.rs` / `jacobian.rs`
- `MultibodyContactConstraint` struct: `nexus-cuda/src_rbd_shaders/dynamics/multibody/types.rs:208` (`inv_lhs`, `rhs`, `impulse`, `lin_jac`, `ii_ang_jac`, ...)
- Probe + readbacks: `zealot/examples/biped/contact_probe.rs`, `zealot/examples/biped/biped_env_nexus.rs` (`dbg_*`)

## Done when
WebGpu `passive_stand` holds a stand (matches CUDA), and the fix is either in the LU/mass-matrix spirv kernel (preferred) or, if it turns out to be khal indirect-dispatch + a separate solver issue, documented as such. Bonus: fix khal's WebGpu indirect dispatch so the fixed-grid default in `lib.rs` can be reverted.
