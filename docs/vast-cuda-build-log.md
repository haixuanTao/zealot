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
- **NEXT:** components (rust-src/rustc-dev/llvm-tools), clone cuda-oxide, LLVM21, CUDA 12.9 wheels, rsync repos.

## Known landmines (from notes — watch for these)
- 32 GB disk: LLVM21 (~5G) + toolchain + 6 repos + target dirs (~15G) + wheels (~2G) ≈ tight. Monitor.
- CUDA 13 system vs 12.9 wheels: wheels are userspace, should isolate; driver supports sm_120.
- `build_nexus_cubin.sh` has **hardcoded /home/baguette paths + a nightly-2025-08-04 llvm-tools path** — must edit for the box (use ~/llvm21 tools or matching llvm-tools; the .ll is LLVM-21 IR so the assembler must be LLVM 21).
- step-1 barrier deadlock if `-JumpThreading` flag missing; step-23 contact ILLEGAL_ADDRESS was fixed (29fac1e).
- `make_cubin` tool is baguette-only; `build_nexus_cubin.sh` avoids it (llc+ptxas path) — prefer that.
