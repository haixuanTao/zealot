# cuTile on a Vast box: `CUDA_HOME` silently breaks `tileiras`

Findings from standing zealot up on a Vast RTX 5090 (`sm_120`) instance,
2026-08-11. Not committed — kept here because no GitHub credentials were
available on the box.

## Symptom

Every cuTile JIT compile fails, with no useful diagnostic:

```
tileiras failed (exit exit status: 5) for gpu sm_120:
stderr: error: failed to compile Tile IR program
stdout:
```

It is invariant across `--opt-level 0..3` and across every `--gpu-name`
(`sm_89`/`sm_90`/`sm_100`/`sm_120`/`sm_121`), and it reproduces on
**cutile-rs's own test suite** — `cargo test -p cutile-ir --test
bytecode_validate` fails 8/8, including `simple_arithmetic`. So it is neither
a codegen bug nor anything specific to zealot's GEMM kernel.

## Root cause

`tileiras` resolves `libnvvm` through **`CUDA_HOME` / `CUDA_PATH`** — not from
its own location, and not via `LD_LIBRARY_PATH` (it `dlopen`s an absolute
path, so `LD_LIBRARY_PATH` has no effect). Confirmed with:

```sh
strace -f -e trace=openat tileiras --gpu-name sm_120 -o out.cubin in.bc \
  | grep nvvm
# openat(AT_FDCWD, "/usr/local/cuda/nvvm/lib64/libnvvm.so", ...) = 3
strings $(which tileiras) | grep -E "CUDA_HOME|CUDA_PATH|nvvm/lib64"
```

The Vast base image exports `CUDA_HOME=/usr/local/cuda`, which is **CUDA
13.0**. So a 13.3 `tileiras` frontend loads a **13.0 `libnvvm`**, and every
program fails to compile. cuTile requires CUDA >= 13.2.

cutile-rs's `flake.nix` documents the same failure mode (same **exit status
5**) for a different reason — a `symlinkJoin` farm placing `nvvm/` outside
tileiras's dereferenced `/proc/self/exe` path.

## Fix

Point `CUDA_HOME`/`CUDA_PATH` at a CUDA >= 13.2 tree for the trainer process:

```sh
export CUDA_HOME=/workspace/cuda-13.3
export CUDA_PATH=/workspace/cuda-13.3
export CUTILE_TILEIRAS_PATH=/workspace/cuda-13.3/bin/tileiras
export PATH=/workspace/cuda-13.3/bin:$PATH
```

Result: `bytecode_validate` goes 0/8 -> 8/8, and the trainer prints
`[cutile] tf32 GEMM path ENABLED (self-test worst rel err 1.13e-3)`.

## Building the 13.3 shim (box ships 13.0)

Six redist components — the same set cutile-rs's `flake.nix` uses. **`libnvvm`
is a separate component**; omitting it is easy to miss, and `cuda_nvcc` does
*not* contain `nvvm/`.

```sh
B=https://developer.download.nvidia.com/compute/cuda/redist
# cuda_crt cuda_nvcc libnvvm cuda_cudart libcurand cuda_tileiras
# extract all into ONE real directory tree (not a symlink farm)
```

Versions used: `cuda_tileiras` 13.3.36, `cuda_nvcc` 13.3.33, `libnvvm`
13.3.33, `cuda_cudart` 13.3.29, `libcurand` 10.4.3.29, `cuda_crt` 13.3.33.
13.3.36 is the newest public tileiras (13.3.0 and 13.3.1 manifests carry the
same build) and it is the version cutile-rs pins — **no version gap exists**.
13.2's tileiras is not an option: it rejects `sm_120` outright
(`invalid GPU architecture: 120`).

## Measured effect (29-DOF G1 AGILE, N=4096, RTX 5090)

cuTile removes host-side encode from the PPO update:

| | update total | of which encode |
|---|---|---|
| vortx GEMM path | 3.13 s | 2.35 s |
| cuTile tf32 | 0.89 s | 0.14 s |

Iteration time 8.7 s -> ~6.6 s. The remaining floor is GPU physics
(`gpuwait` ~121 ms/step x 24 steps ~= 2.9 s/iter) plus env resets, which spike
early in training while the fall rate is high.

## Unrelated note on benchmarking

CUDA graph capture (`BIPED_GRAPH`, default on) only engages after
`GRAPH_CAPTURE_AT = 64` warmup steps. At `T=24` steps/iter that is mid-iter-2,
so any benchmark of <= 3 iterations measures the eager dispatch path. Capture
shows up as `pipe` dropping ~2.1 -> ~0.0 ms/step in `[prof]`.
