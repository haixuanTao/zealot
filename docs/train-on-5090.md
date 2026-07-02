# Training the biped on an RTX 5090 (native CUDA)

How to stand up a fresh Blackwell box (RTX 5090 `sm_120`, or the 5060 vast box)
and run the GPU-resident PPO biped trainer end-to-end on the **native-CUDA**
backend — physics *and* the PPO update on the GPU, no PhysX / no Isaac.

This is the all-Rust path: nexus physics + vortx PPO shaders compiled Rust→PTX via
[cuda-oxide](https://github.com/haixuanTao/cuda-oxide) and loaded as `sm_120`
cubins. The Mac is Metal-only and its contact path is broken — **train on CUDA
boxes only** (`baguette` / `champagne` = the two 5090s; the vast 5060 also works).

## Conventions

Every path below is derived from two variables — set them once and the rest of
the guide is copy-paste:

```bash
export WORK="$HOME/work"       # parent dir for the repos (pick anything)
export PTX="$HOME/nexus_ptx"   # where the compiled cubins land
```

> **The repos must be siblings under `$WORK`.** zealot resolves `khal`, `vortx`,
> and `nexus3d` by *relative* path (`../nexus-cuda`, …), so the directory layout
> is load-bearing — keep them side by side. `$WORK` itself can be anywhere.

---

## TL;DR (box already provisioned)

If the toolchain, wheels, and cubins already exist on the box (the usual case on
`baguette`/`champagne`), you only need three things: fresh cubins, the two env
vars, and the run command.

```bash
cd "$WORK/nexus-cuda"
bash build_cuda/build_nexus_cubin.sh        # -> $PTX/nexus_rbd_shaders3d.cubin
bash build_cuda/build_vortx_cubin_llc.sh    # -> $PTX/vortx_shaders.cubin

cd "$WORK/zealot"
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D="$PTX/nexus_rbd_shaders3d.cubin"
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS="$PTX/vortx_shaders.cubin"

# Native CUDA is auto-selected on sm_120 — no flag needed.
# iters  num_envs  checkpoint
cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- 2000 4096 "$HOME/biped_convex.safetensors"
```

Run it inside `tmux` so it survives the SSH session. Skip to
[Run training](#6-run-training) for the knobs.

---

## 0. Box requirements

- **GPU:** Blackwell `sm_120` (RTX 5090 = 32 GB; 5060 = 8 GB — trains fine, just
  fewer envs). Driver must support `sm_120` (≥ 570-series).
- **Disk:** the tight constraint. LLVM 21 (~5 G) + toolchain + sibling repos +
  target dirs (~15 G) + CUDA wheels (~2 G) ≈ 25 G. The vast 5060 has only 32 G —
  watch `df -h /`.
- System CUDA version does **not** matter for the build: we download CUDA 12.9
  wheels into userspace and only rely on the driver at runtime.

---

## 1. Repos (sibling layout)

Clone the forks at these branches, all as siblings under `$WORK`.

| GitHub repo | branch | clone into |
|-------------|--------|------------|
| `haixuanTao/zealot` | `feat/native-cuda-e2e-bench` | `$WORK/zealot` |
| `haixuanTao/nexus-rl` **(private)** | `feat/per-env-parallelism` | `$WORK/nexus-cuda` |
| `haixuanTao/khal` | `fix/cuda-slice-arg-element-count` | `$WORK/khal` |
| `haixuanTao/vortx` | `fix/reduce-generic-followup` | `$WORK/vortx` |
| `haixuanTao/cuda-oxide` | `feat/nexus3d-vortx-native-cuda` | `$HOME/cuda-oxide-src` |

> **Gotchas that bite a fresh clone:**
> - **Repo name ≠ dir name for nexus:** the physics repo is `haixuanTao/nexus-rl`
>   but zealot resolves it by the relative path `../nexus-cuda`, so clone it into
>   a dir literally named `nexus-cuda` (`git clone …/nexus-rl.git nexus-cuda`).
> - **`nexus-rl` is private** — a new machine needs GitHub auth (`gh auth login`
>   / an authorized SSH key) to clone it. `zealot`/`khal`/`vortx`/`cuda-oxide`
>   are public over https.
> - `cuda-oxide` lives **outside `$WORK`** at `$HOME/cuda-oxide-src` (the build
>   scripts reference that path); adjust if you keep it elsewhere.

**Fastest path — copy a built environment from an existing 5090:**

```bash
rsync -a baguette:'~/{cuda-oxide-src,make_cubin,nvvm-wheel,nvjit-wheel,llvm21}' "$HOME/"
rsync -a baguette:'~/work/{nexus-cuda,khal,vortx,zealot}' "$WORK/"
```

This skips sections 2–4 entirely. Prefer it when a 5090 is reachable. (Adjust the
remote source paths to wherever baguette keeps its checkout.)

---

## 2. Toolchain

```bash
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev llvm-tools --toolchain nightly-2026-04-03
```

- `rust-src` — needed for `-Zbuild-std=core` (shaders target `nvptx64` bare).
- `rustc-dev` + `llvm-tools` — the cuda-oxide codegen backend links rustc
  internals and uses `llvm-as`/`llvm-link`/`opt`/`llc` from the toolchain.
- **LLVM 21.1.0** at `$HOME/llvm21` (the backend links it; scripts reference
  `$HOME/llvm21/bin/llc` as `CUDA_OXIDE_LLC`).

---

## 3. CUDA 12.9 wheels (userspace)

The box's system CUDA is usually too old. Download 12.9 into userspace — it ships
`libnvvm`, `libdevice`, and an `sm_120`-capable `ptxas`:

```bash
pip download nvidia-cuda-nvcc-cu12==12.9.86   # -> nvvm/lib64/libnvvm.so
                                              #    nvvm/libdevice/libdevice.10.bc
                                              #    bin/ptxas
pip download nvidia-nvjitlink-cu12==12.9.86   # -> libnvJitLink.so.12
```

Extract each wheel and point the build scripts at the resulting paths (see the
`LIBDEV`, `PTXAS`, `LIBNVVM_PATH`, `LIBNVJITLINK_PATH` vars in
`$WORK/nexus-cuda/build_cuda/*.sh`). For `sm_120` SASS inspection you also want
the 12.9 `cuobjdump` / `nvdisasm` redist tarballs.

---

## 4. Build the cuda-oxide backend `.so`

```bash
cd "$HOME/cuda-oxide-src/crates/rustc-codegen-cuda"
cargo +nightly-2026-04-03 build          # -> target/debug/librustc_codegen_cuda.so
```

Build it from its own directory — it is **not** a workspace member.

---

## 5. Build the cubins

Two cubins get compiled Rust→PTX and embedded into the zealot host:

- **`nexus_rbd_shaders3d.cubin`** — rigid-body / contact physics (rapier-derived).
- **`vortx_shaders.cubin`** — the PPO GEMM/Adam/actor-value gradient shaders.

```bash
cd "$WORK/nexus-cuda"
bash build_cuda/build_nexus_cubin.sh       # nexus physics -> $PTX/nexus_rbd_shaders3d.cubin
bash build_cuda/build_vortx_cubin_llc.sh   # vortx PPO     -> $PTX/vortx_shaders.cubin
```

> **Edit the paths in the scripts first.** They still contain absolute
> `/home/baguette/...` paths and a stale `nightly-2025-08-04` llvm-tools path.
> Point `TOOL`/`LIBDEV`/`PTXAS`/`BACKEND` at *this* box's toolchain, wheels, and
> the `.so` from section 4, and `CUDA_OXIDE_PTX_DIR` at `$PTX`. The `.ll` is
> LLVM-21 IR, so the assembler must be LLVM 21.

### Critical flags (already baked into the scripts — do not drop them)

- **`-Zmir-enable-passes=-JumpThreading`** — *REQUIRED*. JumpThreading correlates
  repeated `if lane==0 {..}` conditions and duplicates the intervening
  `workgroup_barrier()` across the if-arms → asymmetric per-lane `bar.sync`
  arrivals → **CTA deadlock at the first barrier (the step-1 hang)**.
- **`-Zalways-encode-mir`** — makes external-crate (`parry3d`/`rapier3d`) MIR
  collectable.
- Clean build (`cargo clean -p <shader-crate>`) before each cubin build.
- `--no-default-features --features "cuda-oxide dim3 unsafe_remove_boundchecks"
  --target nvptx64-nvidia-cuda -Z build-std=core`.

`build_nexus_cubin.sh` asserts the device-side symbol hash matches the hash
embedded in the host binary (`EMBED HASH MATCH OK`). If you see
`EMBED MISMATCH`, the host is linking a stale cubin — `touch
crates/nexus_rbd3d/build.rs` and rebuild.

> **Shortcut — download prebuilt `sm_120` cubins** instead of building (only valid
> on Blackwell, and only for the matching source commit — see the release notes):
> ```bash
> mkdir -p "$PTX" && base=https://github.com/haixuanTao/zealot/releases/download/cubins-sm120-20260624
> curl -L "$base/nexus_rbd_shaders3d.cubin" -o "$PTX/nexus_rbd_shaders3d.cubin"
> curl -L "$base/vortx_shaders.cubin"       -o "$PTX/vortx_shaders.cubin"
> ```

---

## 6. Run training

```bash
cd "$WORK/zealot"
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D="$PTX/nexus_rbd_shaders3d.cubin"
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS="$PTX/vortx_shaders.cubin"

cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- <iters> <num_envs> <checkpoint>
```

> **Backend selection.** Built with `cuda_backend`, the trainer **auto-selects
> native CUDA on `sm_120`** (Blackwell) and falls back to WebGPU otherwise — no
> flag needed. Force it with `KHAL_BACKEND=cuda` or `KHAL_BACKEND=webgpu`.
> (`BIPED_CUDA=1` still works as a deprecated alias.)

**Positional args** (after `--`):

| pos | arg | default | notes |
|-----|-----|---------|-------|
| 1 | `iters` | — | PPO iterations |
| 2 | `num_envs` | — | parallel envs. 5090 → 4096–8192; 5060 → ~1024–2048 |
| 3 | `checkpoint` | `/tmp/biped_policy_gpu.safetensors` | resumes if it exists, else fresh init; saved here periodically |

Everything runs on the GPU except action sampling, GAE, and reward/obs (host).
Fixed hyperparameters (match WBC-AGILE T1 velocity policy): horizon `T=24`, 5
epochs, 4 minibatches, `lr=1e-3` (adaptive-KL scheduled), clip 0.2, entropy 0.01,
γ 0.99, λ 0.95, velocity curriculum 0→1 over the first 40 % of iters.

### Useful env knobs (read at startup by `biped_train_gpu.rs`)

| env var | effect |
|---------|--------|
| `KHAL_BACKEND=cuda`\|`webgpu` | force the backend; unset auto-selects native CUDA on sm_120 (`BIPED_CUDA=1` = deprecated alias) |
| `BIPED_STAND_FRAC=<f>` | fraction of the curriculum spent standing before the command ramps |
| `BIPED_RAMP_END=<f>` | curriculum point where the command reaches full speed |
| `BIPED_TORQUE_MAX=<nm>` | motor torque clamp |
| `BIPED_MAX_CSCALE=<s>` | cap the sampled command magnitude (default 1.0) |

Always launch under `tmux` so training survives disconnects:

```bash
tmux new -s train
cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- 2000 4096 "$HOME/biped_convex.safetensors"
# detach: Ctrl-b d   |   reattach: tmux attach -t train
```

---

## 7. Monitor & retrieve checkpoints

- Per-iter reward-component breakdown prints to stdout (`REWARD_COMP_NAMES`).
- TensorBoard is served on the vast box at port `16006` (logdir `/workspace`);
  forward it with `-L` on your SSH command if you want the UI.
- Checkpoints are `.safetensors` at the path you passed. Pull them to the Mac:
  ```bash
  rsync -avP -e 'ssh -p 16199' root@ssh1.vast.ai:'~/biped_convex*.safetensors' "$WORK/zealot/"
  ```
- Back up the cubins too (`$PTX/*.cubin`) — they cost a full toolchain rebuild to
  regenerate (or grab them from the release linked in section 5).

---

## Sanity checks / gotchas

- **Trainer correctness is proven.** `ppo_grad_parity` shows the GPU PPO update
  matches CPU to 1e-7. If a run fails to learn, it is env/reward/curriculum, not
  the optimizer — don't re-hunt trainer bugs.
- **KL "runaway"** on a fresh launch is almost always a launch-config bug
  (curriculum reset / wrong `num_envs`), not the trainer. Check the adaptive-KL
  LR is actually adapting.
- **`glam`/`spirv-std` drift** breaks the GPU build periodically — pin `glam` to
  `0.32.1` if `spirv-std` rejects `0.33`. Use `fix/*` sibling branches, not
  `upstream/*`.
- **5060 vs 5090:** identical code path; just scale `num_envs` down (8 GB vs
  32 GB) and expect proportionally lower throughput.
