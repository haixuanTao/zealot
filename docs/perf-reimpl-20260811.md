# Reimplementation of the v29-era perf chain (from the dead session's commits)

Vast RTX 5090 (sm_120), driver 580.95.05. 2026-08-11. Baseline: zealot
`ab42c46` (= origin/master) + siblings, cuda-oxide native CUDA, cuTile tf32,
CUDA graph capture — all on. Production 29-DOF G1 AGILE config, N=4096.

The original work lived only on "baguette" (`~/Documents/work/{zealot,
vortx-unified}`, ~30 unpushed commits, session_01VX6hvTjCVR22fwKdx8eBzS,
dead). Reimplemented from the recovered commit list + transcript excerpts.
Their trajectory: 6.2 → 1.0 s/iter. Ours: **6.6 → 1.5 s/iter (4.4×)**,
every step A/B-validated with matching iter-0 stats and, for the device
staging, exact element-wise parity.

Final [prof] @4096: roll=1.20 (reset=0.15) gae=0.10 upd=0.21; per-step
flush=0.1 ms (= baguette), forward 0.01 s/iter, sample 0.06, post 0.21.

Final round (after the transfer-trace appendix was written):
- da0d345 clone removal: samples carry obs only under BIPED_VERIFY_STAGE;
  mirror_sample skips empty; post-step obs MOVED out of `outs` not cloned.
  sample 0.12→0.06, gae 0.15→0.10.
- Delay-state device path: `gpu_mb_delay_state_update` (nexus kernel #335)
  copies prev-target lanes from the PRE-scatter motor tensor (which IS
  delay_prev_targets) + writes tick/k; only k_eff [n] (16 KB) crosses PCIe,
  replacing the 573 KB + 143k-element host fill. First attempt REGRESSED
  (flush 3.2→5.9): per-call from_backend + tensor allocs cost more than the
  saved upload — fixed by caching shader bundles + constant tensors in
  MultibodySet (also applied to the target scatter). flush 3.2→0.1 ms/step.
- Remaining, machine-bound: reward block 4.1 ms/step (their CPU: 1.9),
  commit 4.6 (their 1.3; 580 vs 595 driver), readback 1.4, gpuwait 22.8
  (≈ theirs).

Remaining gap to their 1.0 s/iter, fully accounted:
- step 0.84 vs 0.71 — per-step commit 3.9 / flush(delay-state) 3.2 /
  reward-block 3.5 ms vs their 1.3 / 0.1 / 1.9 (machine + config, possibly
  driver 580 vs 595 on submit overhead)
- post 0.19 (gc/gcc clones — their `da0d345` removed; kept here for the
  verify/probe paths), sample 0.12, norm 0.10 (raw-arena upload),
  reset 0.14 (fall-rate: the true "warm policy" effect, → ~0.01 once it
  walks), gae 0.15.

## Levers, in the order applied (all A/B'd at N=4096, 12 iters)

| Their commit | Reimplementation | Measured here |
|---|---|---|
| `5aa1d00` (half) substep refresh OFF by default | `NEXUS_SUBSTEP_REFRESH=0` env (knob existed; not made default in code) | gpuwait 121 → 22.9 ms/step; iter 6.6 → 5.3 s |
| `b16c9f4`+`ca56430` motor scatter, flush-first ordering | zealot: `flush_arms_and_scatter_targets()` — [12×n] host vec, one upload, `scatter_targets_gpu`; links mirror flush only when arm playback stages held targets, and BEFORE the scatter | stage 0.7→0.2, links flush gone; iter 5.3 → 5.2 s. Remaining `flush` 3.1 ms/step = motor-DELAY state upload, untouched |
| `09d734f`+`1643f52`+`cb8dddb`+`5aa1d00` (half) batched GPU resets | NEW nexus kernels `gpu_mb_env_reset_batch` + `gpu_env_reset_bodies` (env_reset.rs): templates GPU-resident (publish once), teleport offset applied in-kernel (WS_LTW+1 / WS_LTP+1 / WS_COORDS quads + body-pose mask, per `GpuMultibodySnapshot::translated`), AGILE reset velocities written in-kernel. Host: `RbdState::publish_reset_templates` / `reset_envs_from_templates`; zealot `reset_envs(&[env])` batch API; trainer batches done envs. RNG draw order per env preserved exactly | reset 2.49 → 0.12 s/iter; iter 5.2 → 2.8 s. Iter-0 stats match sequential path (−0.2716, falls 14308 vs 14303/14310) |
| `8378c87`+`08646f3` host staging fusion | `stage_norm_flat`: fused normalize+clamp(+sperm) into flat row-major, rayon row-blocks; kills per-sample normalize allocs + TWO transposes (`from_fn` + `matrix_from_na`) | upd stage 0.74 → 0.26 s; iter 2.8 → 2.4 s |
| `0b2cf01` sperm tables | Already at HEAD (`obs_sperm`/`critic_sperm`) BUT stale at 45 dims vs 53-dim v28 frames — fixed to match `mirror_frame`/`mirror_critic` (45/47 ang-vel pseudovector, 50 edge_sin, critic tail). Also fixes latent `symmetrize_ac` bug at HEAD | correctness fix |
| `a91b`+`bdcd096`+`da0d345` device PPO batch | vortx kernel `gpu_ppo_stage_batch` (vortx-shaders/linalg/ppo.rs) + `Ppo::stage_batch` host op: step-blocked raw arenas uploaded once per rollout step; update stages [dim×total] halves ([mirrored; originals]) with sperm/identity tables, normalizer affine + ±5 clamp in-kernel; mirror-loss tensor = same staging, halves swapped. Persistent outputs (`matrix_uninit`, their trap #1 avoided). `BIPED_VERIFY_STAGE=1` parity harness (their trap #2) | **verify: obs/critic maxdiff exactly 0.000e0**; perf A/B pending |
| `8326313`+`bdcd096` (rollout half) rollout-forward device staging | `gpu_ppo_stage_batch` gained `step_select` (stage one raw-arena step); per rollout step it normalizes straight into `GpuPolicy::actor_input_mut`/`critic_input_mut` (frozen affine, identity perm) and `forward_prestaged` skips the host normalize + 2 transposes + upload. Host fallback when `BIPED_NORM_FREEZE=0` (stats evolve within the rollout) | forward 0.56 → **0.02 s/iter**; iter 2.2 → **1.7 s** |
| `8bf5119` StepKnobs | SKIPPED — priced first (their later lesson): ~20 per-step single-threaded getenvs ≈ 500/iter ≈ <1 ms/iter. User also flagged unvalidated | not worth it here |
| 16 `reward:` commits (6a92741…fc65018) | SKIPPED per their own transcript verdict: verified but inert; host reward() costs ~0.1 ms/step | — |

## Not carried over / open

- Their `5aa1d00` also made refresh-off the CODE default; here it's still the
  env var (`NEXUS_SUBSTEP_REFRESH=0` must be exported). Decide before long runs.
  NOTE: physics changes — fall counts roughly double early on. They trained
  v29 with it; do the same or A/B learning curves.
- Motor-delay state upload (~3.1 ms/step) — not in their list, untouched.
- Their remaining endgame was physics kernels (see memory note
  `multibody-cooperative-rewrite`): `gpu_mb_init_contact_constraints` still
  threads(1), block-per-articulation resident-M megakernel — the levers behind
  their 45→63k env-steps/s. Not reimplemented.
- `gae` phase 0.20-0.23 s (incl. host `mirror_sample` clones); their da0d345
  removed sample obs storage entirely — kept here for the verify/probe paths.

## Run command (all of it)

```sh
cd /workspace/zealot
export CUDA_HOME=/workspace/cuda-13.3 CUDA_PATH=/workspace/cuda-13.3 \
  CUTILE_TILEIRAS_PATH=/workspace/cuda-13.3/bin/tileiras \
  PATH=/workspace/cuda-13.3/bin:$PATH NEXUS_SMALL_SORT=1 NEXUS_SUBSTEP_REFRESH=0
./target/release/biped_train_gpu 50000 4096 /workspace/g1_wbc.safetensors
```

Rebuild chain after shader edits: `/workspace/build_cubins_local.sh`
(nexus 334 kernels, vortx 73). See also `/workspace/cutile-cuda-home-fix.md`
(tileiras resolves libnvvm via CUDA_HOME — the exit-5 fix).

## Appendix: every host↔device transfer removed, added, or kept

N=4096, T=24, mirror_aug on (total = 196,608 batch columns). Sizes marked ≈
are derived from struct layouts (MultibodyLinkStatic ≈ 400 B: 6 u32 +
GenericJoint [2 poses + 6 limits + 6 motors] + LocalMassProperties) and
cross-check against measured [prof] columns.

### Removed / replaced

| # | Transfer | Before | After | Validated by |
|---|---|---|---|---|
| 1 | Motor targets H2D | full `links_static` mirror ≈ **50 MB/step** (31 links × 4096 × ≈400 B) ≈ 1.2 GB/iter — re-sent whole to change 12 floats/env | `[12×n]` f32 = **196 KB/step** + scatter uniforms ≈ 4.7 MB/iter (~250×) | flush 4.5 → 3.1 ms/step (residue = delay state, separate) |
| 2 | Per-reset staging H2D | PER RESET (≤ ~700/step early): ws 7.4 KB + links 12.4 KB + dofs 0.3 KB + poses 1 KB + vels 1.1 KB ≈ 22 KB, each with its own submit, plus the host snapshot clone | templates resident (one-time ≈ 22 KB × n_templates at build); per STEP one compact upload: 32 B meta/offset + 140 B dof_vels per reset ≈ 86 KB/step worst case, ONE submit | reset 2.49 → 0.12 s/iter |
| 3 | Reset velocities H2D | **~35 separate 4-byte writes per reset** (batch-interleaved layout forced strided writes) — ~20k PCIe transactions/step at early fall rates | folded into #2's dof_vels array, written by the batch kernel | same |
| 4 | Update batch H2D | f_obs 199 MB + f_cobs 44 MB (+ f_obs_mir 199 MB if mirror-loss) ≈ **243 MB/iter**, uploaded ON the update critical path, each preceded by two full host transposes | **0** — built on device from the raw arenas; only the affine vectors ≈ 2.6 KB + uniforms | upd stage 0.74 → 0.06 s |
| 5 | Rollout forward inputs H2D | normalized [265×n] 4.34 MB + [59×n] 0.97 MB = **5.3 MB/step** ≈ 127 MB/iter | **0** — `step_select` staging writes policy inputs on device | forward 0.56 → 0.02 s |
| 6 | Raw arenas H2D (ADDED) | — | 5.3 MB/step ≈ **127 MB/iter** (one copy of the obs, raw) — replaces BOTH #4 and #5's payloads, and lands during the rollout where it overlaps `gpuwait` (pipe = 0.0) | — |

**Net H2D ≈ 1.55 GB/iter → ≈ 0.14 GB/iter (~11×), and the tiny-transfer
storm (#3) is gone entirely.** Just as important: what remains was moved off
the update critical path into the rollout, where it hides behind physics.

### Still crossing (the remaining floor, per step)

| Transfer | Dir | Size | Why it stays |
|---|---|---|---|
| `slurp_poses` | D2H | ≈ 4 MB | host still derives joint/feet/base state + obs + reward from poses (readback 1.4 ms/step). The dead session's GPU-obs port targeted exactly this; their corrected verdict: the state DERIVATION (~1.7 ms of the per-env block) is the payable part, rewards are not |
| contact / sensed force | D2H | small | force-based foot contact obs |
| action means + values | D2H | 196 + 16 KB | host samples actions (RNG) and runs GAE |
| motor-delay state | H2D | 573 KB | per-step tick/k/prev-targets (flush ≈ 3.2 ms incl. the 143k-element host fill loop) — untouched by the recovered commit list |
| gc/gcc obs clones | none (host RAM) | ≈ 5 MB/step memcpy | kept for verify/probe paths; their `da0d345` removed it (~0.19 s/iter here) |
