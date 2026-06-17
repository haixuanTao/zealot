#!/usr/bin/env python3
"""Genesis full-rollout throughput bench (one N per process; gs.init is global).

Per control step: policy MLP forward -> Gaussian sample -> PD control -> 4x
physics step -> obs -> reward. Reports env-ctrl/s + sim/s + stability, matching
the metric in biped_fps.rs / iter_e2e_bench.rs so the tables line up.

Run one point:  ~/genesis-venv/bin/python bench_genesis_rollout.py [N] [T]
Sweep:          bench_genesis_sweep.sh   (invokes this per N, since each needs a
                fresh process — Genesis can't cleanly re-init/rebuild in-process)
"""
import sys
import time

import torch

import biped_genesis_common as C
from genesis_biped_env import GenesisBipedEnv


def mlp(dims, dev):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(dev).eval()


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    WARMUP = 8

    env = GenesisBipedEnv(N, on_gpu=True)
    dev = env.device
    # Actor matches the trained net architecture [43,256,256,128,12] (FLOPs parity).
    actor = mlp([C.OBS_DIM, *C.HIDDEN, C.ACT_DIM], dev)
    log_std = torch.zeros(C.ACT_DIM, device=dev)

    obs = env.get_obs()
    for _ in range(WARMUP):
        with torch.no_grad():
            mean = actor(obs)
            action = mean + torch.randn_like(mean) * log_std.exp()
        obs, rew, z = env.step(action)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(T):
        with torch.no_grad():
            mean = actor(obs)
            action = mean + torch.randn_like(mean) * log_std.exp()
        obs, rew, z = env.step(action)
    torch.cuda.synchronize()
    wall = time.time() - t0

    env_ctrl = N * T / wall
    nan = int(torch.isnan(z).any().item())
    zmean = z.mean().item()
    zlo, zhi = z.min().item(), z.max().item()
    print(
        f"GENESIS_RESULT N={N} T={T} wall_s={wall:.3f} "
        f"env_ctrl_per_s={env_ctrl:.1f} sim_per_s={env_ctrl * C.DECIMATION:.1f} "
        f"torso_z_mean={zmean:.3f} torso_z_range=[{zlo:.3f},{zhi:.3f}] nan={nan}"
    )


if __name__ == "__main__":
    main()
