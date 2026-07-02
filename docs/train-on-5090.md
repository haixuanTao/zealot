# Training the biped on an RTX 5090 (native CUDA)

How to stand up a fresh Blackwell box (RTX 5090 `sm_120`, or the 5060 vast box)
and run the GPU-resident PPO biped trainer end-to-end on the **native-CUDA**
backend — physics *and* the PPO update on the GPU, no PhysX / no Isaac.

This is the all-Rust path: nexus physics + vortx PPO shaders compiled Rust→PTX via
[cuda-oxide](https://github.com/haixuanTao/cuda-oxide) and loaded as `sm_120`
cubins. The Mac is Metal-only and its contact path is broken — **train on CUDA
boxes only** (`baguette` / `champagne` = the two 5090s; the vast 5060 also works).

---

## TL;DR (box already provisioned)

If the toolchain, wheels, and cubins already exist on the box (the usual case on
`baguette`/`champagne`), you only need three things: fresh cubins, the two env
vars, and the run command.

```bash
cd ~/Documents/work/nexus-cuda
bash build_cuda/build_nexus_cubin.sh        # -> ~/nexus_ptx/nexus_rbd_shaders3d.cubin
bash build_cuda/build_vortx_cubin_llc.sh    # -> ~/nexus_ptx/vortx_shaders.cubin

cd ~/Documents/work/zealot
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin

# iters  num_envs  checkpoint
BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- 2000 4096 ~/biped_convex.safetensors
```

Run it inside `tmux` so it survives the SSH session. Skip to
[Run training](#5-run-training) for the knobs.

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

Clone the forks at these branches. `nexus-cuda`, `khal`, `vortx` live under
`~/Documents/work/`; `cuda-oxide` lives at `~/cuda-oxide-src`.

| repo | branch | path |
|------|--------|------|
| `haixuanTao/zealot` | `feat/native-cuda-e2e-bench` | `~/Documents/work/zealot` |
| `haixuanTao/nexus-cuda` | `master` | `~/Documents/work/nexus-cuda` |
| `haixuanTao/khal` | `feat/cuda-oxide-backend` | `~/Documents/work/khal` |
| `haixuanTao/vortx` | `feat/gpu-policy-shaders` | `~/Documents/work/vortx` |
| `haixuanTao/cuda-oxide` | `feat/nexus3d-vortx-native-cuda` | `~/cuda-oxide-src` |

> zealot resolves `khal`/`vortx`/`nexus3d` as **path siblings** — the directory
> layout matters. `cuda-oxide` is public (clone over https); the others may need
> `gh auth login` or an rsync from the Mac if the box has no GitHub access.

**Fastest path — copy a built environment from an existing 5090:**

```bash
rsync -a baguette:~/{cuda-oxide-src,make_cubin,nvvm-wheel,nvjit-wheel,llvm21} ~/
rsync -a baguette:~/Documents/work/{nexus-cuda,khal,vortx,zealot} ~/Documents/work/
```

This skips sections 2–4 entirely. Prefer it when a 5090 is reachable.

---

## 2. Toolchain

```bash
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev llvm-tools --toolchain nightly-2026-04-03
```

- `rust-src` — needed for `-Zbuild-std=core` (shaders target `nvptx64` bare).
- `rustc-dev` + `llvm-tools` — the cuda-oxide codegen backend links rustc
  internals and uses `llvm-as`/`llvm-link`/`opt`/`llc` from the toolchain.
- **LLVM 21.1.0** at `~/llvm21` (the backend links it; scripts reference
  `~/llvm21/bin/llc` as `CUDA_OXIDE_LLC`).

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
`nexus-cuda/build_cuda/*.sh`). For `sm_120` SASS inspection you also want the
12.9 `cuobjdump` / `nvdisasm` redist tarballs.

---

## 4. Build the cuda-oxide backend `.so`

```bash
cd ~/cuda-oxide-src/crates/rustc-codegen-cuda
cargo +nightly-2026-04-03 build          # -> target/debug/librustc_codegen_cuda.so
```

Build it from its own directory — it is **not** a workspace member.

---

## 5. Build the cubins

Two cubins get compiled Rust→PTX and embedded into the zealot host:

- **`nexus_rbd_shaders3d.cubin`** — rigid-body / contact physics (rapier-derived).
- **`vortx_shaders.cubin`** — the PPO GEMM/Adam/actor-value gradient shaders.

```bash
cd ~/Documents/work/nexus-cuda
bash build_cuda/build_nexus_cubin.sh       # nexus physics -> ~/nexus_ptx/nexus_rbd_shaders3d.cubin
bash build_cuda/build_vortx_cubin_llc.sh   # vortx PPO     -> ~/nexus_ptx/vortx_shaders.cubin
```

> **Edit the hardcoded paths first.** The scripts contain absolute
> `/home/baguette/...` paths and a stale `nightly-2025-08-04` llvm-tools path.
> Point `TOOL`/`LIBDEV`/`PTXAS`/`BACKEND` at *this* box's toolchain, wheels, and
> the `.so` from section 4. The `.ll` is LLVM-21 IR, so the assembler must be
> LLVM 21.

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

---

## 6. Run training

```bash
cd ~/Documents/work/zealot
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin

BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- <iters> <num_envs> <checkpoint>
```

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
| `BIPED_CUDA=1` | select the native-CUDA backend (required here) |
| `BIPED_MIRROR_AUG=1` | append the L/R mirror of every transition (free 2× data, symmetry) |
| `BIPED_MIRROR_LOSS=<w>` | symmetry loss term, weight `w` (0 = off) |
| `BIPED_MIRROR_NET=1` | project actor+critic onto the symmetric subspace (NET method) |
| `BIPED_STAND_FRAC=<f>` | fraction of the curriculum spent standing before the command ramps |
| `BIPED_RAMP_END=<f>` | curriculum point where the command reaches full speed |
| `BIPED_TORQUE_MAX=<nm>` | motor torque clamp |
| `BIPED_MAX_CSCALE=<s>` | cap the sampled command magnitude (default 1.0) |

Always launch under `tmux` so training survives disconnects:

```bash
tmux new -s train
BIPED_CUDA=1 cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu cuda_backend" -- 2000 4096 ~/biped_convex.safetensors
# detach: Ctrl-b d   |   reattach: tmux attach -t train
```

---

## 7. Monitor & retrieve checkpoints

- Per-iter reward-component breakdown prints to stdout (`REWARD_COMP_NAMES`).
- TensorBoard is served on the vast box at port `16006` (logdir `/workspace`);
  forward it with `-L` on your SSH command if you want the UI.
- Checkpoints are `.safetensors` at the path you passed. Pull them to the Mac:
  ```bash
  rsync -avP -e 'ssh -p 16199' root@ssh1.vast.ai:'~/biped_convex*.safetensors' \
      ~/Documents/work/zealot/
  ```
- Back up the cubins too (`~/nexus_ptx/*.cubin`) — they cost a full toolchain
  rebuild to regenerate.

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
```

