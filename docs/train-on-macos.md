# Training the biped on macOS (Apple Silicon / Metal)

macOS is a first-class training target since 2026-07-06: at 1024 envs an
M-series Mac matches an RTX 5060 wall-clock (~4.1 s/iter, ~6.1k samples/s).
Before that date the sim was broken on Metal by a shader-compiler bug — the
robot free-fell through the floor and nothing could learn. The fix is a
patched naga; everything below assumes it (and the repo already wires it in).

## TL;DR (siblings already cloned)

```bash
cd ~/Documents/work/zealot
# WebGPU/Metal is the auto-selected backend on a Mac — no flag needed.
#                                       iters  num_envs  checkpoint
cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu" --       2000   1024      walking_policy.safetensors
```

Resuming from an existing checkpoint automatically skips the stand-phase
curriculum and fast-ramps the velocity command (see "curriculum" below).

## 1. Repos (sibling layout)

Same sibling layout as the 5090 guide, plus one extra clone:

```text
work/
├── zealot          # this repo
├── nexus-cuda      # dimforge nexus fork, branch feat/per-env-parallelism
├── khal            # branch fix/cuda-slice-arg-element-count
├── vortx           # branch fix/reduce-generic-followup
├── rapier          # branch world
├── parry           # branch spirv-compat
└── naga-fixed      # github.com/haixuanTao/naga-fixed  ← macOS-critical
```

`naga-fixed` is a vendored naga 29.0.4 carrying the MSL `break_if` fix.
zealot's `[patch.crates-io]` already points at it; you only need the clone.

## 2. The naga bug (why the patch is not optional)

naga 29's MSL backend re-evaluates a loop's `break if` condition *after* the
hoisted `continuing` block has advanced the loop variables, so every
rust-gpu `while` loop exits one body-execution early on Metal
([gfx-rs/wgpu#4558], fix in [gfx-rs/wgpu#9815]). The multibody solver's
per-lane `J·v` reductions are one-iteration-per-lane loops → they run
**zero** times → zero contact/PD impulses → the biped free-falls, then the
accumulated-penetration bias launches it into NaN. `BIPED_SOLVER_ITERS=16`
making it *16× worse* was this bug (gravity integrates per TGS iteration
with nothing cancelling it), not a tuning problem.

CUDA and Vulkan never go through naga (PTX / SPIR-V passthrough), which is
why only macOS was broken. Drop the patch once wgpu ships the fix — and
re-run the verification below when you do.

If you ever see `[[patch.unused]] naga` in `Cargo.lock`, the patch is being
ignored: run `cargo update -p naga`.

[gfx-rs/wgpu#4558]: https://github.com/gfx-rs/wgpu/issues/4558
[gfx-rs/wgpu#9815]: https://github.com/gfx-rs/wgpu/pull/9815

## 3. Toolchain

Only the rust-gpu shader toolchain (cargo-gpu) from the 5090 guide §2 —
no CUDA wheels, no cubins, no `.so`. The SPIR-V shaders build on first
`cargo build` and are translated to MSL by (patched) naga at runtime.

After editing shader source, remember cargo doesn't track path-included
files:

```bash
touch ../nexus-cuda/src_rbd_shaders/lib.rs   # force SPIR-V rebuild
```

## 4. Verify the sim before training

```bash
BIPED_SPAWN_DR=0 cargo run --release --example contact_probe \
    --features "gpu biped_gpu" -- 3
```

Healthy (matches the CUDA golden bit-for-bit at these print precisions):

```text
step 0: torso_z=0.718   step 1: torso_z=0.716   step 2: torso_z=0.715
```

and with `BIPED_DECIMATION=1`, the step-0 normal-contact impulse is
`1.697e-1` (CUDA: `1.698e-1`). Broken naga instead shows `impulse=0.000e0`,
a dof_state of pure free-fall (all ≈0 except base-z = −n·g·dt), and torso
launching upward within a few steps.

## 5. Run training

```bash
# Long runs: use nohup — session-managed background jobs may get reaped.
nohup cargo run --release --example biped_train_gpu \
    --features "gpu biped_gpu" -- 2000 1024 walking_policy.safetensors \
    > walk_train.log 2>&1 &
```

- Checkpoints save every 50 iters; the trainer auto-resumes if the
  checkpoint file exists.
- **Curriculum is resume-aware**: fresh runs do stand-before-walk
  (stand until 30%, full command by 70%); warm starts skip the stand phase
  and re-ramp the command over the first 20%. Override with
  `BIPED_STAND_FRAC` / `BIPED_RAMP_END`.
- Gait-quality defaults (tuned for the real robot's fragile ~11 N·m
  ankles): `BIPED_ANKLE_TORQUE_W=4`, `BIPED_POWER_W=4e-3`,
  `BIPED_MAX_CSCALE=0.4` (±0.2 m/s deliberate gait). All overridable.
- Expected learning curve at 1024 envs: reward −0.36 → positive by
  ~iter 250, falls 1400 → single digits once walking stabilizes.

## 6. Render a rollout video

```bash
cargo run --release --example biped_render_nexus \
    --features "gpu biped_gpu" -- 0 500 /tmp/rollout.json walking_policy.safetensors
MUJOCO_GL=cgl python3 examples/biped/render_biped_mujoco.py \
    /tmp/rollout.json /tmp/walk.mp4
```

The MuJoCo renderer needs `mujoco`, `trimesh`, `numpy` (pip) and `ffmpeg`
on PATH, plus `BIPED_ROBOT_XML` / `BIPED_ASSETS` pointing at the robot MJCF
and its STL directory if they are not at the defaults.

## Performance reference (measured 2026-07-06)

| 1024 envs, ~24.6k samples/iter | M-series Mac (Metal) | RTX 5060 (CUDA) |
|---|---|---|
| wall clock / iter              | 4.0–4.2 s | 4.0 s |
| physics GPU wait / step        | ~96–105 ms | ~52 ms |
| CPU encode+launch / step       | ~6.5 ms | ~48 ms |
| PPO update                     | 0.87 s | 1.20 s |

The 5060's GPU is ~2× faster on raw physics but pays ~7× the per-dispatch
CPU overhead; at this batch size they tie. Training trajectories match the
CUDA reference run-for-run (rewards, KL, fall rates).

## Sanity checks / gotchas

- **Robot falls through floor / launches / NaN** → unpatched naga
  (§2: clone `naga-fixed`, `cargo update -p naga`).
- **`[[patch.unused]] naga` in Cargo.lock** → `cargo update -p naga`.
- **Shader edits don't take effect** → `touch src_rbd_shaders/lib.rs` (§3).
- **Long background runs die silently** → launch with `nohup`, not a
  terminal-session job.
- **passive_stand falls over at ~step 40** → expected on every backend
  (zero-action PD hold is not a stable stand); it's only a bug if it falls
  *through* the floor.
