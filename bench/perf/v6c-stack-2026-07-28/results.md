# zealot perf recompute — current stack (2026-07-28)

Machine: champagne, RTX 5090 (published tables were measured on a different
RTX 5090 box; WBC-AGILE reference numbers are from that box and are not
re-measured here). Stack: solver-fix branch (all 8 fixes; the substep-count
fix means "8 solver iters" now truly runs 8 substeps — the published numbers
unknowingly ran 4), unified NVlabs cuda-oxide toolchain, full-O3 cubins.
Bench: `iter_e2e_bench <N> 32 5 16` (T=32, 5 epochs x 16 minibatches — the
rsl_rl-comparable heavy schedule).

## FULL iteration, k env-steps/s

| N | WebGPU 12-DOF | CUDA+cuTile bold (29-DOF +realism +terrain, it8) | CUDA+cuTile PRODUCTION (d4+it4+refresh, NF240/DR1) |
|---:|---:|---:|---:|
| 2048 | 39.9 | 60.6 | 38.5 |
| 4096 | 42.2 | 67.3 | 40.2 |
| 8192 | 42.3 | 69.0 | 41.6 |

Published July table (same bench, pre-solver-fix, other box): bold column
25.4 / 30.7 / 33.6; WebGPU-8192 iteration 23.1. WBC-AGILE (terrain, 35 nv,
other box): 20.6 / 32.3 / 47.4.

Read: (1) the bold column is ~2x the published one despite now running twice
the substeps — the accumulated perf merges plus the unified-toolchain O3
cubins are real; vs WBC-AGILE's published curve that is 2.9x / 2.1x / 1.5x.
(2) The PRODUCTION physics (the standing-capable stiff-contact config v7
trains on) costs ~1.65x vs the bold config and still lands at 1.9x / 1.2x /
0.9x of WBC-AGILE on the standardized schedule. (3) CPU rapier reference:
2.8 / 3.2 / 3.8 k env/s.

Caveats: cross-box comparison vs published/AGILE rows pending a rerun on the
original box after the current training run frees it; production row measured
at the bench schedule, not the production update.
