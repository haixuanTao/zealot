# Bit-exact training runs (determinism)

**Status (2026-08-31): zealot G1 training is bit-exact run to run** on the
same binary + GPU + config. Two identical launches produce byte-identical
checkpoints (same SHA-256) and digit-identical training stats.

## What it took

Seeding was never the problem — the trainer (`Lcg::new(7)`), the env
(`0xC0FFEE`), and every derived RNG were already fixed, and nothing reads
the clock into the sim. The divergence came from **nexus's GPU narrow
phase**: contacts (and trimesh pfm pairs) are appended through
`atomic_add_u32` cursors, so the contact *buffer order* depended on warp
scheduling. Two order-sensitive consumers amplified that into float
divergence within ~25 steps of first ground contact:

- the per-multibody contact loop (jacobians accumulate in buffer order);
- the greedy contact reduce (merges each pair's manifolds in buffer order).

The fix (nexus `691f3e1`, on `main` and on the `fix/gpu-run-determinism`
branch that `../nexus-train` pins): after the narrow phase, contacts are
stable-sorted to (collider pair, trimesh-leaf feature id) order — a key
that is unique per contact record, so the order is fully canonical — and
the contact reduce runs after the sort. Details and scope live in nexus's
`MIGRATION_GOLDENS.md` ("Run-to-run determinism with contacts").

## Knobs

- `NEXUS_DETERMINISTIC=1|0` — default **on** for native builds (Metal and
  CUDA), **off** on wasm (the browser demo keeps the fast path). Turning it
  off buys back ~3% step time and gives up reproducibility.
- Bit-exactness holds only for the **same binary on the same GPU/driver**
  with the same env-var config (`BIPED_*`, `BIPED_OBS_HISTORY`,
  `BIPED_ARM_MOTION` dataset, env count, iteration count). Rebuilding can
  legally shift last bits (FMA contraction, pipeline caches) — that is
  compiler territory, not fixable in nexus.

## Verifying (the two-run hash test)

```sh
# physics only (fails fast if the engine regressed):
KHAL_BACKEND=webgpu ./target/release/zero_action_probe 256 600 > /tmp/a.log
KHAL_BACKEND=webgpu ./target/release/zero_action_probe 256 600 > /tmp/b.log
diff /tmp/a.log /tmp/b.log   # must be identical

# full trainer:
./target/release/biped_train_gpu 21 256 /tmp/a.safetensors
./target/release/biped_train_gpu 21 256 /tmp/b.safetensors
shasum -a 256 /tmp/{a,b}.safetensors   # hashes must match
```

Verified on Mac Metal at 256 and 4096 envs (2026-08-31). The CUDA boxes run
the same code path but have not been re-verified — run the recipe above
once on champagne/baguette before relying on it there.

Remaining caveat inside nexus: free-rigid-body scenes (not our robot
scenes, which are `rb_contacts_inert`) still have a nondeterministic
constraint-CSR fill and coloring partition — see the nexus doc.
